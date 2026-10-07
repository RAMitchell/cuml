# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive CPU Shapley oracle for the Adult-large legacy discrepancy."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import treelite
import xgboost as xgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument(
        "--new-output", type=Path, help="Optional saved new-backend outputs"
    )
    parser.add_argument(
        "--library", help="Compute the new output if --new-output is omitted"
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = np.load(args.cache / "adult-inputs.npz")
    x, background = data["x"], data["background"]
    legacy = np.load(
        args.cache / f"adult-large-reference-{len(x)}-{len(background)}.npy"
    )
    model = treelite.frontend.from_xgboost(
        xgb.Booster(model_file=args.cache / "adult-large.ubj")
    )
    margins = treelite.gtil.predict(model, x, pred_margin=True, nthread=8)[
        :, 0, :
    ]
    row = int(np.argmax(np.abs(legacy.sum(axis=-1) - margins)))
    if args.new_output:
        new_values = np.load(args.new_output)[row, 0]
    else:
        if not args.library:
            parser.error("Supply --library or --new-output")
        import cupy as cp
        from experiment import Backend, load_library

        backend = Backend(load_library(args.library), model, 1)
        try:
            new_values = cp.asnumpy(
                backend.compute(
                    cp.asarray(x[row : row + 1]), cp.asarray(background)
                )
            )[0, 0]
        finally:
            backend.close()
    features = x.shape[1]
    masks = np.arange(1 << features, dtype=np.uint32)
    selected = (masks[:, None] & (1 << np.arange(features))) != 0
    hybrids = np.where(
        selected[:, None, :], x[row][None, None, :], background[None, :, :]
    )
    flat = np.ascontiguousarray(hybrids.reshape(-1, features))
    # Bitwise row deduplication only reduces prediction work. The inverse restores
    # every coalition/reference pair, including duplicate background weights.
    keys = flat.view(
        np.dtype((np.void, flat.dtype.itemsize * features))
    ).ravel()
    _, indices, inverse = np.unique(
        keys, return_index=True, return_inverse=True
    )
    print(
        f"row={row}, coalitions={len(masks)}, hybrids={len(flat)}, unique={len(indices)}",
        flush=True,
    )
    predictions = treelite.gtil.predict(
        model, flat[indices], pred_margin=True, nthread=8
    )[:, 0, 0]
    values = (
        predictions[inverse]
        .reshape(len(masks), len(background))
        .mean(axis=1, dtype=np.float64)
    )
    exact = np.zeros(features + 1)
    exact[-1] = values[0]
    counts = np.array([int(mask).bit_count() for mask in masks])
    for feature in range(features):
        without = masks[(masks & (1 << feature)) == 0]
        weights = np.array(
            [
                1 / (features * math.comb(features - 1, int(n)))
                for n in counts[without]
            ]
        )
        exact[feature] = np.sum(
            weights * (values[without | (1 << feature)] - values[without])
        )
    record = dict(
        row=row,
        coalitions=len(masks),
        background=len(background),
        unique_hybrid_rows=len(indices),
        oracle="exhaustive coalitions / CPU Treelite",
        reference=exact.tolist(),
        legacy=legacy[row, 0].tolist(),
        new=new_values.tolist(),
        max_legacy_error=float(np.max(np.abs(legacy[row, 0] - exact))),
        max_new_error=float(np.max(np.abs(new_values - exact))),
    )
    np.testing.assert_allclose(new_values, exact, atol=1e-5, rtol=1e-5)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in record.items()
                if key not in ("reference", "legacy", "new")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
