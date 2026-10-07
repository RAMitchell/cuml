# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full twelve-model confirmation of arithmetic/reduction/cache winners."""

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

from benchmark_optimizations import PeakPool
from experiment import Backend, load_library
from optimization import Model, Traversal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lib = load_library(args.library)
    variants = [
        "original",
        "lookup_global",
        "lookup_reduce",
        "lookup_cache_reduce",
    ]
    source_hash = hashlib.sha256(
        b"".join(
            Path(__file__).with_name(f).read_bytes()
            for f in ["optimization.py", "optimization_kernels.cu"]
        )
    ).hexdigest()
    for dataset in ["adult", "cal_housing", "covtype", "fashion_mnist"]:
        data = np.load(args.cache / f"{dataset}-inputs.npz")
        x, r = cp.asarray(data["x"]), cp.asarray(data["background"])
        for size in ["small", "med", "large"]:
            name = f"{dataset}-{size}"
            target = args.output_dir / f"{name}.json"
            if target.exists():
                previous = json.loads(target.read_text())
                if (
                    previous.get("complete")
                    and previous["repeats"] == args.repeats
                    and previous["source_sha256"] == source_hash
                ):
                    print("SKIP", name, flush=True)
                    continue
            model = treelite.frontend.from_xgboost(
                xgb.Booster(model_file=args.cache / f"{name}.ubj")
            )
            start = time.perf_counter()
            m = Model(model)
            cp.cuda.Stream.null.synchronize()
            setup = time.perf_counter() - start
            original = Backend(lib, model, 1)
            out = cp.empty(
                (len(x), m.groups, x.shape[1] + 1), dtype=cp.float64
            )
            print(name, "validate original", flush=True)
            expected = cp.asnumpy(original.compute(x, r, out))
            records = {
                v: dict(
                    variant=v,
                    samples=[],
                    max_error=0.0,
                    peak_workspace_bytes=0,
                )
                for v in variants
            }
            pools = {v: PeakPool() for v in variants if v != "original"}
            backends = {"original": original}
            for v in variants[1:]:
                print(name, "validate", v, flush=True)
                with cp.cuda.using_allocator(pools[v].allocate):
                    backends[v] = Traversal(m, v)
                    backends[v].prepare(x, r)
                    actual = cp.asnumpy(backends[v].compute(x, r, out))
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=1e-5,
                        rtol=1e-5,
                        err_msg=f"{name}/{v}",
                    )
            rng = np.random.default_rng(471)
            for repeat in range(args.repeats):
                for v in rng.permutation(variants):
                    b = backends[v]
                    allocator = (
                        pools[v].allocate
                        if v != "original"
                        else cp.get_default_memory_pool().malloc
                    )
                    with cp.cuda.using_allocator(allocator):
                        cp.cuda.Stream.null.synchronize()
                        start_event, compute_event, end_event = (
                            cp.cuda.Event(),
                            cp.cuda.Event(),
                            cp.cuda.Event(),
                        )
                        wall = time.perf_counter()
                        start_event.record()
                        if v != "original":
                            b.prepare(x, r)
                        compute_event.record()
                        b.compute(x, r, out)
                        end_event.record()
                        end_event.synchronize()
                        sample = dict(
                            wall_seconds=time.perf_counter() - wall,
                            total_gpu_seconds=cp.cuda.get_elapsed_time(
                                start_event, end_event
                            )
                            / 1000,
                            compute_gpu_seconds=cp.cuda.get_elapsed_time(
                                compute_event, end_event
                            )
                            / 1000,
                            preparation_gpu_seconds=cp.cuda.get_elapsed_time(
                                start_event, compute_event
                            )
                            / 1000,
                        )
                        actual = cp.asnumpy(out)
                        error = float(np.max(np.abs(actual - expected)))
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            atol=1e-5,
                            rtol=1e-5,
                            err_msg=f"{name}/{v}/{repeat}",
                        )
                        records[v]["max_error"] = max(
                            records[v]["max_error"], error
                        )
                        records[v]["samples"].append(sample)
                        if v != "original":
                            records[v]["peak_workspace_bytes"] = pools[v].peak
                        print(
                            name,
                            v,
                            repeat + 1,
                            round(sample["wall_seconds"], 6),
                            flush=True,
                        )
                    target.write_text(
                        json.dumps(
                            dict(
                                complete=False,
                                model=name,
                                rows=len(x),
                                background=len(r),
                                repeats=args.repeats,
                                model_setup_seconds=setup,
                                source_sha256=source_hash,
                                results=list(records.values()),
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
            result = json.loads(target.read_text())
            result["complete"] = True
            target.write_text(json.dumps(result, indent=2) + "\n")
            original.close()
            backends.clear()
            b = None
            out = None
            actual = None
            expected = None
            m = model = None
            gc.collect()
            for pool in pools.values():
                pool.pool.free_all_blocks()
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()
