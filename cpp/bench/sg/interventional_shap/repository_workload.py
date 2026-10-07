# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the GPUTreeShap repository's real-data model grid with interventional SHAP."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import treelite
import xgboost as xgb
from sklearn import datasets

from experiment import Backend, load_library

SOURCE = "b24645352ea07292621bee0ac8b327867b46093c"
SIZES = {"small": (10, 3), "med": (100, 8), "large": (1000, 16)}


def load_data(name):
    if name == "adult":
        x, y = datasets.fetch_openml(data_id=1590, return_X_y=True)
        y = np.asarray(y != "<=50K", dtype=np.int32)
        objective = "binary:logistic"
    elif name == "fashion_mnist":
        x, y = datasets.fetch_openml(data_id=40996, return_X_y=True)
        y = np.asarray(y, dtype=np.int32)
        objective = "multi:softmax"
    elif name == "covtype":
        x, y = datasets.fetch_covtype(return_X_y=True)
        # Preserve the repository's labels 1..7 and resulting eight output groups.
        objective = "multi:softmax"
    else:
        x, y = datasets.fetch_california_housing(return_X_y=True)
        objective = "reg:squarederror"
    if isinstance(x, pd.DataFrame):
        numeric = x.copy()
        for col in x:
            if isinstance(x[col].dtype, pd.CategoricalDtype):
                numeric[col] = x[col].cat.codes.replace(-1, np.nan)
        numeric = numeric.to_numpy(dtype=np.float32)
    else:
        x = np.asarray(x, dtype=np.float32)
        numeric = x
    return x, np.ascontiguousarray(numeric), np.asarray(y), objective


def model_stats(model):
    nodes = leaves = leaf_depth_sum = max_depth = 0
    for ti in range(model.num_tree):
        accessor = model.get_tree_accessor(ti)
        left = accessor.get_field("cleft")
        right = accessor.get_field("cright")
        nodes += len(left)
        pending = [(0, 0)]
        while pending:
            nid, depth = pending.pop()
            if left[nid] == -1:
                leaves += 1
                leaf_depth_sum += depth
                max_depth = max(max_depth, depth)
            else:
                pending.append((int(left[nid]), depth + 1))
                pending.append((int(right[nid]), depth + 1))
    return dict(
        trees=model.num_tree,
        nodes=nodes,
        leaves=leaves,
        max_depth=max_depth,
        average_leaf_depth=leaf_depth_sum / leaves,
    )


def prepare(args):
    for name in args.datasets:
        print(f"LOAD {name}", flush=True)
        x, numeric, y, objective = load_data(name)
        foreground = np.random.RandomState(432).randint(
            0, len(numeric), args.rows
        )
        background = np.random.RandomState(433).choice(
            len(numeric), args.background, replace=False
        )
        np.savez(
            args.cache / f"{name}-inputs.npz",
            x=numeric[foreground],
            background=numeric[background],
            foreground_indices=foreground,
            background_indices=background,
        )
        dtrain = None
        for size in args.sizes:
            stem = f"{name}-{size}"
            path = args.cache / f"{stem}.ubj"
            if path.exists() and (args.cache / f"{stem}.json").exists():
                print(f"CACHED {stem}", flush=True)
                continue
            if dtrain is None:
                feature_types = (
                    [
                        "c"
                        if isinstance(x[col].dtype, pd.CategoricalDtype)
                        else "q"
                        for col in x
                    ]
                    if isinstance(x, pd.DataFrame)
                    else None
                )
                # Explicit numeric category codes preserve split semantics without the
                # XGBoost 3.2 string encoder that Treelite 4.7 cannot import.
                dtrain = xgb.QuantileDMatrix(
                    numeric,
                    y,
                    enable_categorical=True,
                    feature_types=feature_types,
                    nthread=16,
                )
            rounds, depth = SIZES[size]
            params = dict(
                tree_method="hist",
                device="cuda",
                max_depth=depth,
                eta=0.01,
                objective=objective,
                seed=0,
                nthread=16,
                base_score=0.5,
            )
            if objective == "multi:softmax":
                params["num_class"] = int(y.max()) + 1
            print(f"TRAIN {stem} {params}", flush=True)
            start = time.perf_counter()
            booster = xgb.train(params, dtrain, num_boost_round=rounds)
            seconds = time.perf_counter() - start
            booster.save_model(path)
            model = treelite.frontend.from_xgboost(booster)
            record = dict(
                name=stem,
                source_revision=SOURCE,
                xgboost=xgb.__version__,
                parameters=params,
                rounds=rounds,
                training_seconds=seconds,
                dataset_rows=len(numeric),
                features=numeric.shape[1],
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                **model_stats(model),
            )
            (args.cache / f"{stem}.json").write_text(
                json.dumps(record, indent=2) + "\n"
            )
            print("READY " + json.dumps(record), flush=True)
            del model, booster
            gc.collect()
        del dtrain, x, numeric, y
        gc.collect()


def run(args):
    lib = load_library(args.library)
    for name in args.datasets:
        data = np.load(args.cache / f"{name}-inputs.npz")
        host_x = data["x"][: args.rows]
        host_r = data["background"][: args.background]
        assert len(host_x) == args.rows and len(host_r) == args.background
        x, r = cp.asarray(host_x), cp.asarray(host_r)
        for size in args.sizes:
            stem = f"{name}-{size}"
            booster = xgb.Booster(model_file=args.cache / f"{stem}.ubj")
            model = treelite.frontend.from_xgboost(booster)
            metadata = json.loads((args.cache / f"{stem}.json").read_text())
            # All predictions are raw margins, as required for additive tree SHAP.
            margins = treelite.gtil.predict(
                model, host_x, pred_margin=True, nthread=16
            )[:, 0, :]
            bias = treelite.gtil.predict(
                model, host_r, pred_margin=True, nthread=16
            )[:, 0, :].mean(axis=0, dtype=np.float64)
            reference_path = (
                args.cache
                / f"{stem}-reference-{args.rows}-{args.background}.npy"
            )
            legacy_record = args.output / f"{stem}-b{args.background}-v3.json"
            legacy_valid = (
                json.loads(legacy_record.read_text()).get(
                    "prediction_checks_passed", True
                )
                if legacy_record.exists()
                else False
            )
            for variant in args.variants:
                result_path = (
                    args.output / f"{stem}-b{args.background}-v{variant}.json"
                )
                if result_path.exists():
                    previous = json.loads(result_path.read_text())
                    expected = (
                        args.rows,
                        args.background,
                        args.batch,
                        args.repeats,
                        metadata["model_sha256"],
                    )
                    observed = (
                        previous["rows"],
                        previous["background"],
                        previous["batch_rows"],
                        len(previous["samples_seconds"]),
                        previous["model"]["model_sha256"],
                    )
                    if observed != expected:
                        raise ValueError(
                            f"Cached result has a different protocol: {result_path}"
                        )
                    print(f"SKIP {result_path.name}", flush=True)
                    continue
                print(f"PREPARE {stem} variant={variant}", flush=True)
                start = time.perf_counter()
                backend = Backend(lib, model, variant)
                cp.cuda.Stream.null.synchronize()
                setup = time.perf_counter() - start
                groups = backend.groups
                out = cp.empty(
                    (min(args.batch, args.rows), groups, x.shape[1] + 1),
                    dtype=cp.float64,
                )
                max_error = max_additivity = max_bias = 0.0
                prediction_checks_passed = True
                # Validate every foreground row, saving legacy outputs for subsequent backends.
                reference = (
                    np.load(reference_path, mmap_mode="r")
                    if reference_path.exists()
                    else None
                )
                target = None
                if variant == 3 and reference is None:
                    target = np.lib.format.open_memmap(
                        reference_path,
                        mode="w+",
                        dtype=np.float64,
                        shape=(args.rows, groups, x.shape[1] + 1),
                    )
                validation_start = time.perf_counter()
                for begin in range(0, args.rows, args.batch):
                    end = min(begin + args.batch, args.rows)
                    result = backend.compute(
                        x[begin:end], r, out[: end - begin]
                    )
                    actual = cp.asnumpy(result)
                    if target is not None:
                        target[begin:end] = actual
                    if reference is not None:
                        expected = reference[begin:end]
                        max_error = max(
                            max_error, float(np.max(np.abs(actual - expected)))
                        )
                        if legacy_valid:
                            np.testing.assert_allclose(
                                actual, expected, atol=1e-5, rtol=1e-5
                            )
                    max_additivity = max(
                        max_additivity,
                        float(
                            np.max(
                                np.abs(
                                    actual.sum(axis=-1) - margins[begin:end]
                                )
                            )
                        ),
                    )
                    max_bias = max(
                        max_bias, float(np.max(np.abs(actual[..., -1] - bias)))
                    )
                    prediction_checks_passed &= bool(
                        np.isfinite(actual).all()
                        and np.allclose(
                            actual.sum(axis=-1),
                            margins[begin:end],
                            atol=1e-4,
                            rtol=1e-4,
                        )
                        and np.allclose(
                            actual[..., -1], bias, atol=1e-4, rtol=1e-4
                        )
                    )
                    if variant in (1, 4):
                        assert prediction_checks_passed, (
                            "New traversal failed prediction checks"
                        )
                if variant == 3:
                    legacy_valid = prediction_checks_passed
                if not prediction_checks_passed:
                    print(
                        f"LEGACY VALIDATION FAILURE {stem} v{variant}: additivity={max_additivity}, bias={max_bias}",
                        flush=True,
                    )
                if target is not None:
                    target.flush()
                    del target
                del reference
                print(
                    f"VALIDATED {stem} v{variant} in {time.perf_counter() - validation_start:.2f}s",
                    flush=True,
                )
                samples = []
                wall_samples = []
                for repeat in range(args.repeats):
                    elapsed = 0.0
                    wall_start = time.perf_counter()
                    for begin in range(0, args.rows, args.batch):
                        end = min(begin + args.batch, args.rows)
                        start_event, end_event = (
                            cp.cuda.Event(),
                            cp.cuda.Event(),
                        )
                        start_event.record()
                        backend.compute(x[begin:end], r, out[: end - begin])
                        end_event.record()
                        end_event.synchronize()
                        elapsed += (
                            cp.cuda.get_elapsed_time(start_event, end_event)
                            / 1000
                        )
                    samples.append(elapsed)
                    wall_samples.append(time.perf_counter() - wall_start)
                    print(
                        f"TIME {stem} v{variant} {repeat + 1}/{args.repeats}: {elapsed:.6f}s",
                        flush=True,
                    )
                record = dict(
                    model=metadata,
                    rows=args.rows,
                    background=args.background,
                    batch_rows=args.batch,
                    variant=variant,
                    setup_seconds=setup,
                    samples_seconds=samples,
                    wall_samples_seconds=wall_samples,
                    median_seconds=float(np.median(samples)),
                    mean_seconds=float(np.mean(samples)),
                    std_seconds=float(np.std(samples)),
                    max_legacy_difference=max_error,
                    max_additivity_error=max_additivity,
                    max_bias_error=max_bias,
                    prediction_checks_passed=prediction_checks_passed,
                    legacy_reference_valid=legacy_valid,
                    legacy_compared=variant != 3 and reference_path.exists(),
                )
                result_path.write_text(json.dumps(record, indent=2) + "\n")
                backend.close()
                del out, backend
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
            del model, booster
            gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["adult", "cal_housing", "covtype", "fashion_mnist"],
    )
    parser.add_argument("--sizes", nargs="+", default=list(SIZES))
    parser.add_argument(
        "--variants", nargs="+", type=int, default=[3, 1, 4, 0]
    )
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--background", type=int, default=100)
    parser.add_argument("--batch", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(args.rows, args.background, args.batch, args.repeats) < 1:
        parser.error("rows, background, batch, and repeats must be positive")
    if not args.prepare and not args.library:
        parser.error("--library is required for timing")
    args.cache.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
