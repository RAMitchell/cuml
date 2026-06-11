#!/usr/bin/env python
"""Run a focused deep RandomForest sweep on cover type.

This reuses one dataset split across cases and warms cuML once before measured
runs. It is intentionally narrower than a full benchmark matrix: the goal is to
explain deep-tree behavior around max_depth and max_features.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from profile_covtype_rf import load_and_split, run_cuml, run_sklearn


def _parse_max_features(value: str) -> str:
    return value


def _optional_int(value: str) -> int | None:
    if value == "None":
        return None
    return int(value)


def _warmup_cuml(X_train: np.ndarray, y_train: np.ndarray, args: argparse.Namespace) -> None:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier

    rows = min(args.warmup_rows, X_train.shape[0])
    model = RandomForestClassifier(
        n_estimators=4,
        max_depth=4,
        max_features="sqrt",
        n_bins=args.n_bins,
        n_streams=args.n_streams,
        random_state=args.random_state,
    )
    X_warm = cp.asarray(np.asarray(X_train[:rows], dtype=np.float32, order="F"), order="F")
    y_warm = cp.asarray(y_train[:rows].astype(np.int32, copy=False))
    model.fit(X_warm, y_warm)
    cp.cuda.get_current_stream().synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sklearn", "cuml", "both"], default="both")
    parser.add_argument("--rows", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--depth", type=_optional_int, action="append", default=[])
    parser.add_argument("--feature", type=_parse_max_features, action="append", default=[])
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--n-streams", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--warmup-rows", type=int, default=10_000)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/covtype_rf_deep_cases.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depths = args.depth or [5, 10, 15, 20, 25, 30, None]
    features = args.feature or ["sqrt", "0.25", "0.5", "1.0"]

    X_train, X_test, y_train, y_test, phases = load_and_split(args)

    if args.backend in {"cuml", "both"}:
        start = time.perf_counter()
        _warmup_cuml(X_train, y_train, args)
        print(f"cuml_warmup_seconds={time.perf_counter() - start:.6f}", flush=True)

    cases = []
    seen = set()
    for depth in depths:
        cases.append((depth, "sqrt"))
    for feature in features:
        cases.append((30, feature))

    rows = []
    for depth, feature in cases:
        key = (depth, feature)
        if key in seen:
            continue
        seen.add(key)

        args.max_depth = depth
        args.max_features = feature
        if args.backend in {"sklearn", "both"}:
            row = asdict(run_sklearn(X_train, X_test, y_train, y_test, args, phases))
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if args.backend in {"cuml", "both"}:
            row = asdict(run_cuml(X_train, X_test, y_train, y_test, args, phases))
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"results": rows}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
