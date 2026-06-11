#!/usr/bin/env python
"""Phase timing for RandomForestClassifier on the Forest Cover Type dataset.

The regular benchmark runner reports a single fit time. This script keeps the
accuracy-oriented cover type setup but splits the run into phases so we can see
whether time is going to data preparation, host/device transfer, fitting,
prediction, or scoring.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.ensemble import RandomForestClassifier as SkRandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


@dataclass
class TimedResult:
    backend: str
    n_train: int
    n_test: int
    n_features: int
    n_estimators: int
    max_depth: int | None
    max_features: str | float | int | None
    n_bins: int | None
    n_streams: int | None
    random_state: int
    load_seconds: float
    split_seconds: float
    convert_seconds: float
    fit_seconds: float
    predict_seconds: float
    score_seconds: float
    accuracy: float


@contextmanager
def timed(phases: dict[str, float], name: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    phases[name] = time.perf_counter() - start


def _jsonable_max_features(value: str) -> str | float | int | None:
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _optional_int(value: str) -> int | None:
    if value == "None":
        return None
    return int(value)


def _sync() -> None:
    try:
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()
    except Exception:
        pass


def _as_numpy(x: Any) -> np.ndarray:
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def load_and_split(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    phases: dict[str, float] = {}
    with timed(phases, "load"):
        X, y = fetch_covtype(return_X_y=True, download_if_missing=not args.no_download)

    with timed(phases, "split"):
        y = y.astype(np.int32, copy=False) - 1
        if args.rows and args.rows < X.shape[0]:
            rng = np.random.default_rng(args.random_state)
            idx = rng.choice(X.shape[0], size=args.rows, replace=False)
            X = X[idx]
            y = y[idx]
        X = X.astype(np.float32, copy=False)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
    return X_train, X_test, y_train, y_test, phases


def run_sklearn(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    phases: dict[str, float],
) -> TimedResult:
    local = dict(phases)
    max_features = _jsonable_max_features(args.max_features)
    with timed(local, "convert"):
        X_train_local = np.asarray(X_train, dtype=np.float32, order="C")
        X_test_local = np.asarray(X_test, dtype=np.float32, order="C")
        y_train_local = np.asarray(y_train, dtype=np.int32)
        y_test_local = np.asarray(y_test, dtype=np.int32)

    model = SkRandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=max_features,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    with timed(local, "fit"):
        model.fit(X_train_local, y_train_local)
    with timed(local, "predict"):
        pred = model.predict(X_test_local)
    with timed(local, "score"):
        acc = float(accuracy_score(y_test_local, pred))

    return TimedResult(
        backend="sklearn",
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        n_features=X_train.shape[1],
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=max_features,
        n_bins=None,
        n_streams=None,
        random_state=args.random_state,
        load_seconds=local["load"],
        split_seconds=local["split"],
        convert_seconds=local["convert"],
        fit_seconds=local["fit"],
        predict_seconds=local["predict"],
        score_seconds=local["score"],
        accuracy=acc,
    )


def run_cuml(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    args: argparse.Namespace,
    phases: dict[str, float],
) -> TimedResult:
    import cupy as cp
    from cuml.ensemble import RandomForestClassifier as CuRandomForestClassifier

    local = dict(phases)
    max_features = _jsonable_max_features(args.max_features)
    with timed(local, "convert"):
        X_train_local = cp.asarray(np.asarray(X_train, dtype=np.float32, order="F"), order="F")
        X_test_local = cp.asarray(np.asarray(X_test, dtype=np.float32, order="C"), order="C")
        y_train_local = cp.asarray(y_train.astype(np.int32, copy=False))
        y_test_local = y_test.astype(np.int32, copy=False)
        _sync()

    model = CuRandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=max_features,
        n_bins=args.n_bins,
        n_streams=args.n_streams,
        random_state=args.random_state,
    )
    with timed(local, "fit"):
        model.fit(X_train_local, y_train_local)
        _sync()
    with timed(local, "predict"):
        pred = model.predict(X_test_local)
        _sync()
    with timed(local, "score"):
        pred_np = _as_numpy(pred).reshape(-1)
        acc = float(accuracy_score(y_test_local, pred_np))

    return TimedResult(
        backend="cuml",
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        n_features=X_train.shape[1],
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        max_features=max_features,
        n_bins=args.n_bins,
        n_streams=args.n_streams,
        random_state=args.random_state,
        load_seconds=local["load"],
        split_seconds=local["split"],
        convert_seconds=local["convert"],
        fit_seconds=local["fit"],
        predict_seconds=local["predict"],
        score_seconds=local["score"],
        accuracy=acc,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sklearn", "cuml", "both"], default="both")
    parser.add_argument("--rows", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=_optional_int, default=5)
    parser.add_argument("--max-features", default="1.0")
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--n-streams", type=int, default=4)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("/tmp/covtype_rf_profile.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for repeat in range(args.repeats):
        X_train, X_test, y_train, y_test, phases = load_and_split(args)
        if args.backend in {"sklearn", "both"}:
            result = run_sklearn(X_train, X_test, y_train, y_test, args, phases)
            row = asdict(result) | {"repeat": repeat}
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if args.backend in {"cuml", "both"}:
            result = run_cuml(X_train, X_test, y_train, y_test, args, phases)
            row = asdict(result) | {"repeat": repeat}
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
