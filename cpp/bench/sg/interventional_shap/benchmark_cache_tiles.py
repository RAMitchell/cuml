# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Foreground tiling ablation: retain background bits, rebuild each foreground tile."""

import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
import treelite
import xgboost as xgb

from benchmark_optimizations import PeakPool, timed
from experiment import Backend, load_library
from optimization import Model, Traversal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    lib = load_library(args.library)
    for name in ["fashion_mnist", "cal_housing", "adult"]:
        booster = xgb.Booster(model_file=args.cache / f"{name}-large.ubj")[:2]
        model = treelite.frontend.from_xgboost(booster)
        m = Model(model)
        data = np.load(args.cache / f"{name}-inputs.npz")
        x = cp.asarray(data["x"])
        r = cp.asarray(data["background"])
        old = Backend(lib, model, 1)
        expected = cp.asnumpy(old.compute(x, r))
        old.close()
        for variant in [
            "cache_foreground",
            "cache_both",
            "lookup_cache_reduce",
        ]:
            for tile in [32, 256, 2048, 10000]:
                pool = PeakPool()
                with cp.cuda.using_allocator(pool.allocate):
                    b = Traversal(m, variant)
                    out = cp.empty(expected.shape, dtype=cp.float64)
                    _, prep = timed(
                        lambda: setattr(b, "pr", b.bits(r))
                        if b.settings.get("CACHE_R")
                        else None
                    )

                    def compute():
                        for begin in range(0, len(x), tile):
                            xx = x[begin : begin + tile]
                            b.px = b.bits(xx)
                            b.compute(xx, r, out[begin : begin + tile])
                        return out

                    actual, _ = timed(compute)
                    np.testing.assert_allclose(
                        cp.asnumpy(actual), expected, atol=1e-5, rtol=1e-5
                    )
                    samples = [timed(compute)[1] for _ in range(5)]
                    records.append(
                        dict(
                            model=name + "-large",
                            rounds=2,
                            rows=len(x),
                            background=len(r),
                            variant=variant,
                            tile=tile,
                            background_preparation=prep,
                            samples=samples,
                            peak_workspace_bytes=pool.peak,
                        )
                    )
                    print(
                        name,
                        variant,
                        tile,
                        np.median([s["wall_seconds"] for s in samples]),
                        flush=True,
                    )
                    b = out = actual = None
                pool.pool.free_all_blocks()
        args.output.write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
