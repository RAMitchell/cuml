# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fresh confirmation of a conservative rule after the first policy experiment.

Fixed before this run: depth<=3, N>=1000, B>=100, N*B>=1_000_000, and
one-leaf memory guard. This is a second-stage confirmation, not a claim that the
first-stage shallow rule succeeded. Includes tenfold duplicated shallow forests
as a size stress test; those forests are not independently trained models.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
import treelite
import xgboost as xgb

from benchmark_optimizations import timed
from optimization import Model, Paths, Patterns, Traversal
from policy_experiment import CappedPool, median


def choose_conservative(depth, rows, references, budget=64 * 1024**2):
    if (
        depth > 3
        or rows < 1000
        or references < 100
        or rows * references < 1_000_000
    ):
        return "traversal"
    estimate = 128 * (rows + references) + 8 * rows * (depth + 1)
    estimate += 8 * (depth + 1) * 2**depth + 12 * depth + 4096
    return "dense" if estimate <= budget else "traversal"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folder = Path(__file__).parent
    digest = hashlib.sha256(
        b"".join(
            (folder / f).read_bytes()
            for f in [
                "policy_confirmation.py",
                "policy_experiment.py",
                "optimization.py",
                "optimization_kernels.cu",
                "pattern_kernels.cu",
            ]
        )
    ).hexdigest()
    result = dict(
        source_sha256=digest,
        seed=20261009,
        repeats=3,
        complete=False,
        records=[],
    )
    if args.output.exists():
        result = json.loads(args.output.read_text())
        assert result["source_sha256"] == digest
    done = {r["case"] for r in result["records"]}
    for dataset in ["adult", "cal_housing", "covtype", "fashion_mnist"]:
        data = np.load(args.cache / f"{dataset}-inputs.npz")["x"]
        rng = np.random.default_rng(20261009)
        xi = rng.choice(len(data), 20000, replace=True)
        ri = rng.permutation(len(data))
        base = treelite.frontend.from_xgboost(
            xgb.Booster(model_file=args.cache / f"{dataset}-small.ubj")
        )
        for copies in [1, 10]:
            tm = (
                base
                if copies == 1
                else treelite.Model.concatenate([base] * copies)
            )
            m = Model(tm)
            start = time.perf_counter()
            paths = Paths(m)
            path_seconds = time.perf_counter() - start
            cases = (
                [
                    (100, 10000),
                    (1000, 1000),
                    (2000, 512),
                    (5000, 256),
                    (20000, 100),
                    (5000, 100),
                    (1000, 100),
                ]
                if copies == 1
                else [(10000, 100)]
            )
            for n, b in cases:
                case = f"{dataset}-small-copies{copies}-n{n}-b{b}"
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
                pools = {
                    v: CappedPool(64 * 1024**2) for v in ["traversal", "dense"]
                }
                backends = {}
                for v in pools:
                    with cp.cuda.using_allocator(pools[v].allocate):
                        backends[v] = (
                            Traversal(m, "lookup_reduce")
                            if v == "traversal"
                            else Patterns(
                                m,
                                paths,
                                "pattern_dense",
                                memory_limit=64 * 1024**2,
                            )
                        )
                records = {
                    v: dict(samples=[], max_error=0.0, fallbacks=0)
                    for v in ["traversal", "dense", "policy"]
                }

                def run(v):
                    chosen = (
                        choose_conservative(m.depth, n, b)
                        if v == "policy"
                        else v
                    )
                    with cp.cuda.using_allocator(pools[chosen].allocate):
                        try:
                            if chosen == "dense":
                                backends[chosen].compute(
                                    x, r, out, streamed=True
                                )
                            else:
                                backends[chosen].prepare(x, r)
                                backends[chosen].compute(x, r, out)
                        except cp.cuda.memory.OutOfMemoryError:
                            records[v]["fallbacks"] += 1
                            fallback.compute(x, r, out)
                    return out

                for v in records:
                    run(v)
                    np.testing.assert_allclose(
                        cp.asnumpy(out), expected, atol=1e-5, rtol=1e-5
                    )
                order = np.random.default_rng(20261009 + n + b)
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
                selected = choose_conservative(m.depth, n, b)
                for v in records:
                    pool = pools[selected if v == "policy" else v]
                    assert pool.reserved_peak <= 64 * 1024**2
                    records[v].update(
                        peak_live_bytes=pool.peak,
                        peak_reserved_bytes=pool.reserved_peak,
                    )
                # An intentionally impossible budget must fail safely and leave a correct fallback.
                forced = False
                if n == 1000 and b == 100 and copies == 1:
                    tiny = CappedPool(512)
                    try:
                        with cp.cuda.using_allocator(tiny.allocate):
                            impossible = Patterns(m, paths, "pattern_dense")
                            impossible.compute(x, r, out, streamed=True)
                    except cp.cuda.memory.OutOfMemoryError:
                        forced = True
                        fallback.compute(x, r, out)
                        np.testing.assert_allclose(
                            cp.asnumpy(out), expected, atol=1e-5, rtol=1e-5
                        )
                    assert forced
                    assert (
                        choose_conservative(m.depth, 10000, 100, budget=512)
                        == "traversal"
                    )
                    tiny.pool.free_all_blocks()
                result["records"].append(
                    dict(
                        case=case,
                        dataset=dataset,
                        copies=copies,
                        rows=n,
                        background=b,
                        depth=m.depth,
                        trees=tm.num_tree,
                        selected=selected,
                        path_setup_seconds=path_seconds,
                        shared_output_bytes=out.nbytes,
                        forced_low_budget_fallback=forced,
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
            m = paths = tm = None
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("COMPLETE", len(result["records"]), flush=True)


if __name__ == "__main__":
    main()
