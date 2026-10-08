# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed cheap dispatch rules, fresh backgrounds, and capped dense workspace.

Policies are fixed before collecting this experiment's measurements. The primary
rule is shallow; broad and exponential are exploratory comparisons, not fitted
classifiers. All workspace caps exclude common input/model/output allocations,
CUDA modules and compiler-managed local memory. No production API is changed.
"""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import cupy as cp
import numpy as np
import treelite
import xgboost as xgb

from benchmark_optimizations import PeakPool, timed
from optimization import Model, Paths, Patterns, Traversal

POLICIES = ["traversal", "shallow", "broad", "exponential"]


def choose(depth, rows, references, policy, budget=64 * 1024**2):
    """O(1) metadata-only decision; no pattern construction or input inspection."""
    if policy == "traversal" or rows < 1000 or depth > 8:
        return "traversal"
    eligible = (
        depth <= 3 and references >= 100
        if policy == "shallow"
        else references >= 100
        if policy == "broad"
        else references >= max(10, 2**depth)
    )
    # Conservative one-leaf estimate, including the constant-leaf sparse fallback.
    # The allocator limit below enforces the cap if this estimate is insufficient.
    leaf_bytes = 128 * (rows + references) + 8 * rows * (depth + 1)
    leaf_bytes += 8 * (depth + 1) * 2**depth + 12 * depth + 4096
    return "dense64" if eligible and leaf_bytes <= budget else "traversal"


class CappedPool(PeakPool):
    def __init__(self, budget):
        super().__init__()
        self.pool.set_limit(size=budget)
        self.reserved_peak = 0

    def allocate(self, size):
        pointer = super().allocate(size)
        self.reserved_peak = max(self.reserved_peak, self.pool.total_bytes())
        return pointer


def median(record):
    return statistics.median(s["wall_seconds"] for s in record["samples"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    source = Path(__file__).parent
    digest = hashlib.sha256(
        b"".join(
            (source / f).read_bytes()
            for f in [
                "policy_experiment.py",
                "optimization.py",
                "optimization_kernels.cu",
                "pattern_kernels.cu",
            ]
        )
    ).hexdigest()
    result = dict(
        source_sha256=digest,
        seed=20261008,
        repeats=args.repeats,
        policy_specification="shallow: D<=3,N>=1000,B>=100; broad: D<=8,N>=1000,B>=100; exponential: D<=8,N>=1000,B>=max(10,2**D)",
        primary_policy="shallow",
        records=[],
        complete=False,
        background="seeded permutation of cached foreground pool; fresh empirical background, possible duplicate values and foreground overlap",
        workspace="dedicated CuPy pool limit; common inputs, model, output, CUDA modules and compiler local memory excluded",
    )
    if args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["source_sha256"] != digest:
            raise ValueError(
                "existing results were produced by different sources"
            )
        result = previous
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    done = {r["case"] for r in result["records"]}
    for dataset in ["adult", "cal_housing", "covtype", "fashion_mnist"]:
        data = np.load(args.cache / f"{dataset}-inputs.npz")["x"]
        indices = np.random.default_rng(20261008).permutation(len(data))
        for size in ["small", "med", "large"]:
            name = f"{dataset}-{size}"
            configurations = [
                (n, b) for n in [100, 10000] for b in [10, 100, 1000]
            ] + [(1000, 100)]
            if size == "large":
                configurations = [(10000, 1000)]
            prefix = size == "large"
            names = [
                f"{name}-{'prefix2' if prefix else 'full'}-n{n}-b{b}"
                for n, b in configurations
            ]
            if all(c in done for c in names):
                continue
            booster = xgb.Booster(model_file=args.cache / f"{name}.ubj")
            if prefix:
                booster = booster[:2]
            tm = treelite.frontend.from_xgboost(booster)
            m = Model(tm)
            started = time.perf_counter()
            paths = Paths(m)
            path_setup = time.perf_counter() - started
            host_path_bytes = sum(
                a.nbytes
                for a in vars(paths).values()
                if isinstance(a, np.ndarray)
            )
            for (n, b), case in zip(configurations, names):
                if case in done:
                    continue
                print("START", case, "depth", m.depth, flush=True)
                x = cp.asarray(data[:n])
                r = cp.asarray(data[indices[:b]])
                out = cp.empty((n, m.groups, x.shape[1] + 1), dtype=cp.float64)
                pools, backends = {}, {}
                direct = Traversal(m, "lookup_reduce")
                direct.prepare(x, r)
                expected = cp.asnumpy(direct.compute(x, r, out))
                margins = treelite.gtil.predict(
                    tm, data[:n], pred_margin=True, nthread=8
                )[:, 0, :]
                np.testing.assert_allclose(
                    expected.sum(-1), margins, atol=1e-4, rtol=1e-4
                )
                records = {}
                for variant, budget in [
                    ("traversal", 64),
                    ("dense64", 64),
                    ("dense256", 256),
                ]:
                    pools[variant] = CappedPool(budget * 1024**2)
                    with cp.cuda.using_allocator(pools[variant].allocate):
                        backend = (
                            Traversal(m, "lookup_reduce")
                            if variant == "traversal"
                            else Patterns(
                                m,
                                paths,
                                "pattern_dense",
                                tile=64,
                                memory_limit=budget * 1024**2,
                            )
                        )
                    backends[variant] = backend
                    records[variant] = dict(
                        samples=[], max_error=0.0, fallbacks=0
                    )

                def run(variant):
                    backend = backends[variant]
                    with cp.cuda.using_allocator(pools[variant].allocate):
                        try:
                            if variant == "traversal":
                                backend.prepare(x, r)
                                backend.compute(x, r, out)
                            else:
                                backend.compute(x, r, out, streamed=True)
                        except cp.cuda.memory.OutOfMemoryError:
                            # The direct backend uses common preallocated output and no input cache.
                            records[variant]["fallbacks"] += 1
                            direct.compute(x, r, out)
                    return out

                for variant in records:
                    run(variant)
                    cp.cuda.Stream.null.synchronize()
                    np.testing.assert_allclose(
                        cp.asnumpy(out), expected, atol=1e-5, rtol=1e-5
                    )
                rng = np.random.default_rng(20261008 + n + b)
                for repeat in range(args.repeats):
                    for variant in rng.permutation(list(records)):
                        _, sample = timed(lambda: run(variant))
                        actual = cp.asnumpy(out)
                        np.testing.assert_allclose(
                            actual, expected, atol=1e-5, rtol=1e-5
                        )
                        records[variant]["max_error"] = max(
                            records[variant]["max_error"],
                            float(np.max(np.abs(actual - expected))),
                        )
                        records[variant]["samples"].append(sample)
                for variant, pool in pools.items():
                    records[variant].update(
                        peak_live_bytes=pool.peak,
                        peak_reserved_bytes=pool.reserved_peak,
                    )
                    assert (
                        pool.reserved_peak
                        <= (256 if variant == "dense256" else 64) * 1024**2
                    )
                choices = {p: choose(m.depth, n, b, p) for p in POLICIES}
                # Measure only metadata dispatch: no GPU operation and no pattern inspection.
                start = time.perf_counter()
                for _ in range(10000):
                    choose(m.depth, n, b, "shallow")
                decision_ns = (time.perf_counter() - start) * 1e5
                result["records"].append(
                    dict(
                        case=case,
                        model=name,
                        prefix=prefix,
                        trees=tm.num_tree,
                        depth=m.depth,
                        rows=n,
                        background=b,
                        leaves=len(paths.widths),
                        path_setup_seconds=path_setup,
                        host_path_bytes=host_path_bytes,
                        shared_output_bytes=out.nbytes,
                        decisions=choices,
                        decision_ns=decision_ns,
                        variants=records,
                        holdout=(
                            size in ["small", "med"]
                            and name != "fashion_mnist-small"
                        ),
                    )
                )
                save()
                print(
                    "DONE",
                    case,
                    {v: round(median(rec), 6) for v, rec in records.items()},
                    choices,
                    flush=True,
                )
                backends.clear()
                backend = direct = out = actual = expected = x = r = None
                gc.collect()
                for pool in pools.values():
                    pool.pool.free_all_blocks()
                cp.get_default_memory_pool().free_all_blocks()
            m = tm = paths = booster = None
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
    result["complete"] = True
    save()
    print("COMPLETE", len(result["records"]), flush=True)


if __name__ == "__main__":
    main()
