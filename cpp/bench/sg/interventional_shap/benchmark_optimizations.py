# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Randomized optimization screening with explicit preparation and memory costs."""

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

from experiment import Backend, load_library
from optimization import (
    Model,
    Paths,
    Patterns,
    Traversal,
    VARIANTS,
    PATTERN_VARIANTS,
)


class PeakPool:
    def __init__(self):
        self.pool = cp.cuda.MemoryPool()
        self.peak = 0

    def allocate(self, size):
        pointer = self.pool.malloc(size)
        self.peak = max(self.peak, self.pool.used_bytes())
        return pointer


def timed(fn):
    cp.cuda.get_current_stream().synchronize()
    begin, end = cp.cuda.Event(), cp.cuda.Event()
    start = time.perf_counter()
    begin.record()
    value = fn()
    end.record()
    end.synchronize()
    return value, dict(
        wall_seconds=time.perf_counter() - start,
        gpu_seconds=cp.cuda.get_elapsed_time(begin, end) / 1000,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "fashion_mnist-small",
            "fashion_mnist-large",
            "cal_housing-large",
            "adult-large",
        ],
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["original"] + list(VARIANTS) + PATTERN_VARIANTS,
    )
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--background", type=int, default=100)
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="0 retains all boosting rounds; screening uses a prefix",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--memory-mb", type=int, default=256)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.rows, args.background, args.repeats, args.tile, args.memory_mb
        )
        <= 0
    ):
        parser.error("dimensions must be positive")
    lib = load_library(args.library)
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_hash = hashlib.sha256(
        b"".join(
            Path(__file__).with_name(f).read_bytes()
            for f in [
                "optimization.py",
                "optimization_kernels.cu",
                "pattern_kernels.cu",
            ]
        )
    ).hexdigest()
    for stem in args.models:
        name = stem.rsplit("-", 1)[0]
        data = np.load(args.cache / f"{name}-inputs.npz")
        hx = data["x"][: args.rows]
        hr = data["background"][: args.background]
        if len(hr) < args.background:
            from repository_workload import load_data

            _, numeric, _, _ = load_data(name)
            indices = np.random.RandomState(433).choice(
                len(numeric), args.background, replace=False
            )
            hr = numeric[indices]
        assert len(hx) == args.rows and len(hr) == args.background
        booster = xgb.Booster(model_file=args.cache / f"{stem}.ubj")
        if args.rounds:
            booster = booster[: min(args.rounds, booster.num_boosted_rounds())]
        model = treelite.frontend.from_xgboost(booster)
        start = time.perf_counter()
        m = Model(model)
        cp.cuda.Stream.null.synchronize()
        model_setup = time.perf_counter() - start
        paths = None
        path_setup = 0
        if any(v in PATTERN_VARIANTS for v in args.variants):
            start = time.perf_counter()
            paths = Paths(m)
            path_setup = time.perf_counter() - start
        x = cp.asarray(hx)
        r = cp.asarray(hr)
        reference_backend = Backend(lib, model, 1)
        reference = cp.asnumpy(reference_backend.compute(x, r))
        reference_backend.close()
        expected = treelite.gtil.predict(
            model, hx, pred_margin=True, nthread=8
        )[:, 0, :]
        np.testing.assert_allclose(
            reference.sum(-1), expected, atol=1e-4, rtol=1e-4
        )
        print(
            "MODEL",
            stem,
            "trees",
            model.num_tree,
            "rows",
            len(x),
            "background",
            len(r),
            flush=True,
        )
        case = {
            v: dict(
                model=stem,
                variant=v,
                trees=model.num_tree,
                rows=len(x),
                background=len(r),
                rounds=booster.num_boosted_rounds(),
                depth=m.depth,
                tile=args.tile,
                memory_limit_mb=args.memory_mb,
                model_setup_seconds=model_setup,
                path_setup_seconds=path_setup,
                model_bytes=m.nbytes,
                source_sha256=source_hash,
                cold=[],
                warm=[],
                preparation=[],
                peak_workspace_bytes=0,
                max_error=0.0,
            )
            for v in args.variants
        }
        rng = np.random.default_rng(890)
        for repeat in range(args.repeats):
            for variant in rng.permutation(args.variants):
                record = case[variant]
                if record.get("status") == "unsupported":
                    continue
                pool = PeakPool()
                with cp.cuda.using_allocator(pool.allocate):
                    out = cp.empty(reference.shape, dtype=cp.float64)
                    b = None
                    streamed = False
                    start = time.perf_counter()
                    if variant == "original":
                        b = Backend(lib, model, 1)
                    elif variant in PATTERN_VARIANTS:
                        b = Patterns(
                            m,
                            paths,
                            variant,
                            tile=args.tile,
                            memory_limit=args.memory_mb * 1024**2,
                        )
                    else:
                        b = Traversal(m, variant, profile=args.profile)
                    cp.cuda.Stream.null.synchronize()
                    record.setdefault("construction_seconds", []).append(
                        time.perf_counter() - start
                    )
                    if variant != "original":
                        try:
                            _, prep = timed(lambda: b.prepare(x, r))
                            record["preparation"].append(prep)
                        except MemoryError:
                            streamed = True
                            record["cache_mode"] = "streamed_memory_limit"
                            b.tiles = []
                            gc.collect()
                        except ValueError as error:
                            record["status"] = "unsupported"
                            record["reason"] = str(error)
                            b = out = None
                            continue

                    def compute():
                        return (
                            b.compute(x, r, out, streamed=streamed)
                            if isinstance(b, Patterns)
                            else b.compute(x, r, out)
                        )

                    actual, _ = timed(compute)
                    error = float(
                        np.max(np.abs(cp.asnumpy(actual) - reference))
                    )
                    record["max_error"] = max(record["max_error"], error)
                    np.testing.assert_allclose(
                        cp.asnumpy(actual),
                        reference,
                        atol=1e-5,
                        rtol=1e-5,
                        err_msg=f"{stem}/{variant}",
                    )

                    def cold():
                        if variant != "original" and not streamed:
                            b.prepare(x, r)
                        return compute()

                    _, sample = timed(cold)
                    record["cold"].append(sample)
                    if not streamed:
                        _, sample = timed(compute)
                        record["warm"].append(sample)
                    record["peak_workspace_bytes"] = max(
                        record["peak_workspace_bytes"], pool.peak
                    )
                    record["cache_bytes"] = getattr(b, "cache_bytes", 0)
                    if isinstance(b, Patterns):
                        record["choices"] = b.choices
                        record["max_tile_bytes"] = b.max_tile_bytes
                    if isinstance(b, Traversal):
                        record["kernel_attributes"] = b.attributes
                        if args.profile:
                            record["instrumented_counts"] = cp.asnumpy(
                                b.counters
                            ).tolist()
                    if variant == "original":
                        b.close()
                    record["status"] = "passed"
                    b = out = actual = None
                gc.collect()
                pool.pool.free_all_blocks()
                print(
                    stem,
                    variant,
                    repeat + 1,
                    "cold",
                    round(record["cold"][-1]["wall_seconds"], 6),
                    "warm",
                    round(record["warm"][-1]["wall_seconds"], 6)
                    if record["warm"]
                    else None,
                    flush=True,
                )
                args.output.write_text(
                    json.dumps(
                        dict(
                            protocol=vars(args)
                            | {
                                "cache": str(args.cache),
                                "output": str(args.output),
                            },
                            results=records + list(case.values()),
                        ),
                        indent=2,
                    )
                    + "\n"
                )
        records.extend(case.values())
        model = m = booster = x = r = paths = None
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
    args.output.write_text(
        json.dumps(
            dict(
                protocol=vars(args)
                | {"cache": str(args.cache), "output": str(args.output)},
                results=records,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
