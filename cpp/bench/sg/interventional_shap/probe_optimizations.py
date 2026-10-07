# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight accuracy and instrumented traversal work; not performance timings."""

import argparse
import json
import math
from decimal import Decimal, localcontext
from pathlib import Path

import cupy as cp
import numpy as np
import treelite
import xgboost as xgb

from optimization import Model, Traversal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = []
    work = []
    t, w = np.polynomial.legendre.leggauss(8)
    t = (t + 1) / 2
    w /= 2
    for depth in [16, 32, 64, 128]:
        max_recurrence = max_lookup = max_quad = 0.0
        for n in range(1, depth + 1):
            for a in range(1, n + 1):
                b = n - a
                with localcontext() as ctx:
                    ctx.prec = 100
                    exact = Decimal(1) / (
                        Decimal(a) * Decimal(math.comb(n, a))
                    )
                    recurrence = 1.0
                    for i in range(1, min(a, b) + 1):
                        recurrence *= i / (max(a, b) + i)
                    recurrence /= a
                    lookup = (1 / math.comb(n, a)) * (1 / a)
                    quad = float(w @ (t ** (a - 1) * (1 - t) ** b))
                    max_recurrence = max(
                        max_recurrence,
                        float(abs(Decimal(recurrence) / exact - 1)),
                    )
                    max_lookup = max(
                        max_lookup, float(abs(Decimal(lookup) / exact - 1))
                    )
                    max_quad = max(
                        max_quad, float(abs(Decimal(quad) / exact - 1))
                    )
        weights.append(
            dict(
                depth=depth,
                recurrence_max_relative_error=max_recurrence,
                lookup_max_relative_error=max_lookup,
                fixed_quad8_max_relative_error=max_quad,
            )
        )
    for name in ["fashion_mnist", "cal_housing", "adult"]:
        model = treelite.frontend.from_xgboost(
            xgb.Booster(model_file=args.cache / f"{name}-large.ubj")[:2]
        )
        m = Model(model)
        data = np.load(args.cache / f"{name}-inputs.npz")
        x = cp.asarray(data["x"][:128])
        r = cp.asarray(data["background"])
        for variant in [
            "control",
            "lookup_global",
            "local_reduce",
            "lookup_reduce",
            "lookup_cache_reduce",
        ]:
            b = Traversal(m, variant, profile=True)
            b.prepare(x, r)
            b.compute(x, r)
            cp.cuda.Stream.null.synchronize()
            counts = cp.asnumpy(b.counters)
            work.append(
                dict(
                    model=name + "-large",
                    rounds=2,
                    rows=len(x),
                    background=len(r),
                    variant=variant,
                    node_visits=int(counts[0]),
                    leaf_visits=int(counts[1]),
                    atomic_updates=int(counts[2]),
                    kernel_attributes=b.attributes,
                )
            )
            print(work[-1], flush=True)
    args.output.write_text(
        json.dumps(dict(weights=weights, work=work), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
