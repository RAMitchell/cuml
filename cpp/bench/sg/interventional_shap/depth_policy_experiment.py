# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test the existing batch guards with the depth ceiling raised to six."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cupy as cp
import numpy as np
import treelite
import xgboost as xgb

import repository_workload as workload
from benchmark_optimizations import timed
from optimization import Model, Paths, Patterns, Traversal
from policy_experiment import CappedPool, median


BUDGET = 64 * 1024**2


def choose(depth, rows, references):
    if (
        depth > 6
        or rows < 1000
        or references < 100
        or rows * references < 1_000_000
    ):
        return "traversal"
    estimate = 128 * (rows + references) + 8 * rows * (depth + 1)
    estimate += 8 * (depth + 1) * 2**depth + 12 * depth + 4096
    return "dense" if estimate <= BUDGET else "traversal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    datasets = ["adult", "cal_housing", "covtype", "fashion_mnist"]
    if args.prepare:
        workload.SIZES.update(
            {"depth4": (10, 4), "depth5": (10, 5), "depth6": (100, 6)}
        )
        workload.prepare(
            SimpleNamespace(
                cache=args.cache,
                datasets=datasets,
                sizes=["depth4", "depth5", "depth6"],
                rows=10000,
                background=100,
            )
        )
    folder = Path(__file__).parent
    files = [
        "depth_policy_experiment.py",
        "repository_workload.py",
        "policy_experiment.py",
        "benchmark_optimizations.py",
        "optimization.py",
        "optimization_kernels.cu",
        "pattern_kernels.cu",
    ]
    digest = hashlib.sha256(
        b"".join((folder / f).read_bytes() for f in files)
    ).hexdigest()
    result = dict(
        source_sha256=digest,
        seed=20261010,
        repeats=3,
        complete=False,
        records=[],
        policy="D<=6,N>=1000,B>=100,N*B>=1e6,one-leaf estimate<=64MiB",
    )
    if args.output.exists():
        result = json.loads(args.output.read_text())
        assert result["source_sha256"] == digest
    done = {r["case"] for r in result["records"]}
    for dataset in datasets:
        data = np.load(args.cache / f"{dataset}-inputs.npz")["x"]
        rng = np.random.default_rng(20261010)
        xi, ri = rng.permutation(len(data)), rng.permutation(len(data))
        for depth, rounds in [(4, 10), (5, 10), (6, 10), (6, 100)]:
            model_file = args.cache / f"{dataset}-depth{depth}.ubj"
            booster = xgb.Booster(model_file=model_file)[:rounds]
            tm = treelite.frontend.from_xgboost(booster)
            m = Model(tm)
            start = time.perf_counter()
            paths = Paths(m)
            setup = time.perf_counter() - start
            for n, b in [
                (1000, 100),
                (1000, 1000),
                (10000, 100),
                (10000, 1000),
            ]:
                case = f"{dataset}-d{depth}-r{rounds}-n{n}-b{b}"
                if case in done:
                    continue
                print("START", case, flush=True)
                x, r = cp.asarray(data[xi[:n]]), cp.asarray(data[ri[:b]])
                out = cp.empty((n, m.groups, x.shape[1] + 1), dtype=cp.float64)
                fallback = Traversal(m, "lookup_reduce")
                fallback.prepare(x, r)
                expected = cp.asnumpy(fallback.compute(x, r, out))
                prediction = treelite.gtil.predict(
                    tm, data[xi[:n]], pred_margin=True, nthread=8
                )[:, 0, :]
                np.testing.assert_allclose(
                    expected.sum(-1), prediction, atol=1e-4, rtol=1e-4
                )
                pools = {v: CappedPool(BUDGET) for v in ["traversal", "dense"]}
                backends = {}
                for v in pools:
                    with cp.cuda.using_allocator(pools[v].allocate):
                        backends[v] = (
                            Traversal(m, "lookup_reduce")
                            if v == "traversal"
                            else Patterns(
                                m, paths, "pattern_dense", memory_limit=BUDGET
                            )
                        )
                records = {
                    v: dict(samples=[], max_error=0.0, fallbacks=0)
                    for v in ["traversal", "dense", "policy"]
                }

                def run(v):
                    selected = choose(m.depth, n, b) if v == "policy" else v
                    with cp.cuda.using_allocator(pools[selected].allocate):
                        try:
                            if selected == "dense":
                                backends[selected].compute(
                                    x, r, out, streamed=True
                                )
                            else:
                                backends[selected].prepare(x, r)
                                backends[selected].compute(x, r, out)
                        except cp.cuda.memory.OutOfMemoryError:
                            records[v]["fallbacks"] += 1
                            fallback.compute(x, r, out)
                    return out

                for v in records:
                    run(v)
                    np.testing.assert_allclose(
                        cp.asnumpy(out), expected, atol=1e-5, rtol=1e-5
                    )
                order = np.random.default_rng(
                    20261010 + n + b + depth + rounds
                )
                for _ in range(3):
                    for v in order.permutation(list(records)):
                        _, sample = timed(lambda: run(v))
                        actual = cp.asnumpy(out)
                        np.testing.assert_allclose(
                            actual, expected, atol=1e-5, rtol=1e-5
                        )
                        records[v]["max_error"] = max(
                            records[v]["max_error"],
                            float(np.max(np.abs(actual - expected))),
                        )
                        records[v]["samples"].append(sample)
                selected = choose(m.depth, n, b)
                for v in records:
                    pool = pools[selected if v == "policy" else v]
                    assert pool.reserved_peak <= BUDGET
                    records[v].update(
                        peak_live_bytes=pool.peak,
                        peak_reserved_bytes=pool.reserved_peak,
                    )
                result["records"].append(
                    dict(
                        case=case,
                        dataset=dataset,
                        requested_depth=depth,
                        depth=m.depth,
                        rounds=rounds,
                        trees=tm.num_tree,
                        leaves=len(paths.widths),
                        rows=n,
                        background=b,
                        selected=selected,
                        path_setup_seconds=setup,
                        shared_output_bytes=out.nbytes,
                        model_sha256=hashlib.sha256(
                            model_file.read_bytes()
                        ).hexdigest(),
                        variants=records,
                    )
                )
                args.output.write_text(json.dumps(result, indent=2) + "\n")
                print(
                    "DONE",
                    case,
                    {v: round(median(rec), 6) for v, rec in records.items()},
                    selected,
                    flush=True,
                )
                backends.clear()
                fallback = out = actual = expected = x = r = None
                gc.collect()
                for pool in pools.values():
                    pool.pool.free_all_blocks()
                cp.get_default_memory_pool().free_all_blocks()
            m = tm = paths = booster = None
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE", len(result["records"]), flush=True)


if __name__ == "__main__":
    main()
