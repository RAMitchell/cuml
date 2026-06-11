#!/usr/bin/env python
"""Inspect fitted model size for deep cover type RF cases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier

from profile_covtype_rf import _jsonable_max_features, _optional_int, load_and_split


def _forest_stats(model) -> dict[str, int | float]:
    trees = model.estimators_
    node_counts = [int(tree.tree_.node_count) for tree in trees]
    leaves = [int(tree.tree_.n_leaves) for tree in trees]
    depths = [int(tree.tree_.max_depth) for tree in trees]
    return {
        "n_trees": len(trees),
        "total_nodes": int(sum(node_counts)),
        "mean_nodes": float(np.mean(node_counts)),
        "max_nodes": int(max(node_counts)),
        "total_leaves": int(sum(leaves)),
        "mean_leaves": float(np.mean(leaves)),
        "max_depth_seen": int(max(depths)),
        "mean_depth_seen": float(np.mean(depths)),
    }


def run_sklearn(X_train, y_train, args) -> dict[str, object]:
    max_features = _jsonable_max_features(args.max_features)
    model = SkRandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=max_features,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    start = time.perf_counter()
    model.fit(np.asarray(X_train, dtype=np.float32, order="C"), y_train.astype(np.int32, copy=False))
    fit_seconds = time.perf_counter() - start
    return {
        "backend": "sklearn",
        "fit_seconds": fit_seconds,
        "export_seconds": 0.0,
        "max_depth": args.max_depth,
        "max_features": max_features,
        **_forest_stats(model),
    }


def run_cuml(X_train, y_train, args) -> dict[str, object]:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as CuRandomForestClassifier

    max_features = _jsonable_max_features(args.max_features)
    X_dev = cp.asarray(np.asarray(X_train, dtype=np.float32, order="F"), order="F")
    y_dev = cp.asarray(y_train.astype(np.int32, copy=False))
    model = CuRandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=max_features,
        n_bins=args.n_bins,
        n_streams=args.n_streams,
        random_state=args.random_state,
    )
    start = time.perf_counter()
    model.fit(X_dev, y_dev)
    cp.cuda.get_current_stream().synchronize()
    fit_seconds = time.perf_counter() - start

    start = time.perf_counter()
    sk_model = model.as_sklearn()
    export_seconds = time.perf_counter() - start

    return {
        "backend": "cuml",
        "fit_seconds": fit_seconds,
        "export_seconds": export_seconds,
        "treelite_bytes": len(model._treelite_model_bytes),
        "max_depth": args.max_depth,
        "max_features": max_features,
        "n_bins": args.n_bins,
        "n_streams": args.n_streams,
        **_forest_stats(sk_model),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sklearn", "cuml", "both"], default="both")
    parser.add_argument("--rows", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=_optional_int, default=30)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--n-streams", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/covtype_rf_model_inspect.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    X_train, _, y_train, _, _ = load_and_split(args)
    rows = []
    if args.backend in {"sklearn", "both"}:
        row = run_sklearn(X_train, y_train, args)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    if args.backend in {"cuml", "both"}:
        row = run_cuml(X_train, y_train, args)
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
