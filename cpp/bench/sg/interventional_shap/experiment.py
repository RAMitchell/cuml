# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate pair traversal against exhaustive coalitions, then benchmark variants."""

import argparse
import ctypes
import itertools
import json
import math
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np
import treelite
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class Backend:
    def __init__(self, library, model, variant):
        self.lib = library
        self.groups = int(
            model.get_header_accessor().get_field("num_class")[0]
        )
        self.handle = library.iv_create(model.handle, variant)
        if not self.handle:
            raise RuntimeError(library.iv_error().decode())

    def compute(self, x, background, out=None, stream=None):
        if out is None:
            out = cp.empty(
                (len(x), self.groups, x.shape[1] + 1), dtype=cp.float64
            )
        stream = stream or cp.cuda.Stream.null
        result = self.lib.iv_compute(
            self.handle,
            x.data.ptr,
            len(x),
            x.shape[1],
            background.data.ptr,
            len(background),
            out.data.ptr,
            x.dtype.itemsize,
            stream.ptr,
        )
        if result:
            raise RuntimeError(self.lib.iv_error().decode())
        return out

    def close(self):
        if self.handle:
            self.lib.iv_destroy(self.handle)
            self.handle = None


def load_library(path):
    lib = ctypes.CDLL(str(Path(path).resolve()))
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    lib.iv_error.restype = ctypes.c_char_p
    lib.iv_create.argtypes = [ptr, ctypes.c_int]
    lib.iv_create.restype = ptr
    lib.iv_destroy.argtypes = [ptr]
    lib.iv_compute.argtypes = [
        ptr,
        ptr,
        size,
        size,
        ptr,
        size,
        ptr,
        ctypes.c_int,
        ptr,
    ]
    lib.iv_compute.restype = ctypes.c_int
    return lib


def predict(model, x):
    return treelite.gtil.predict(
        model, np.ascontiguousarray(x), pred_margin=True, nthread=1
    )[:, 0, :]


def exhaustive(model, x, background):
    """Independent Shapley definition: all coalitions, uniform empirical background."""
    features = x.shape[1]
    values = []
    for mask in range(1 << features):
        use_x = np.array([bool(mask & (1 << j)) for j in range(features)])
        hybrids = np.where(use_x, x[:, None, :], background[None, :, :])
        values.append(
            predict(model, hybrids.reshape(-1, features))
            .reshape(len(x), len(background), -1)
            .mean(axis=1)
        )
    result = np.zeros((len(x), values[0].shape[1], features + 1))
    result[..., -1] = values[0]
    for j in range(features):
        for mask in range(1 << features):
            if mask & (1 << j):
                continue
            count = mask.bit_count()
            weight = 1 / (features * math.comb(features - 1, count))
            result[..., j] += weight * (values[mask | (1 << j)] - values[mask])
    return result


def custom_model(nodes, features, dtype="float64", bias=0.25, cover=True):
    from treelite.model_builder import (
        Metadata,
        ModelBuilder,
        PostProcessorFunc,
        TreeAnnotation,
    )

    builder = ModelBuilder(
        threshold_type=dtype,
        leaf_output_type=dtype,
        metadata=Metadata(features, "kRegressor", False, 1, [1], (1, 1)),
        tree_annotation=TreeAnnotation(1, [0], [0]),
        postprocessor=PostProcessorFunc("identity"),
        base_scores=[float(bias)],
    )
    builder.start_tree()
    for nid, node in enumerate(nodes):
        builder.start_node(nid)
        if "leaf" in node:
            builder.leaf(node["leaf"])
        elif "cats" in node:
            builder.categorical_test(
                feature_id=node["feature"],
                default_left=node.get("default", False),
                category_list=node["cats"],
                category_list_right_child=node.get("right", False),
                left_child_key=node["left"],
                right_child_key=node["right_child"],
            )
        else:
            builder.numerical_test(
                node["feature"],
                node["threshold"],
                default_left=node.get("default", True),
                opname=node.get("op", "<"),
                left_child_key=node["left"],
                right_child_key=node["right_child"],
            )
        if cover:
            builder.sum_hess(float(node.get("count", 10)))
        builder.end_node()
    builder.end_tree()
    return builder.commit()


def validate(lib, include_legacy=True, check_callback=None):
    records = []

    def check(name, model, x, background, baseline=True, expected=None):
        if check_callback is not None:
            check_callback(name, model, x, background, expected)
            return
        reference = (
            exhaustive(model, x, background) if expected is None else expected
        )
        for variant in (
            [0, 1, 2, 3, 4] if baseline and include_legacy else [1, 2, 4]
        ):
            backend = Backend(lib, model, variant)
            try:
                # Upload before entering the independent stream; no implicit default-stream dependency.
                dx, dr = cp.asarray(x), cp.asarray(background)
                cp.cuda.Stream.null.synchronize()
                stream = (
                    cp.cuda.Stream(non_blocking=True)
                    if variant in (1, 2, 4)
                    else cp.cuda.Stream.null
                )
                with stream:
                    actual = backend.compute(dx, dr, stream=stream)
                stream.synchronize()
                actual = cp.asnumpy(actual)
                np.testing.assert_allclose(
                    actual, reference, atol=2e-6, rtol=2e-6
                )
                np.testing.assert_allclose(
                    actual.sum(axis=-1),
                    predict(model, x),
                    atol=3e-6,
                    rtol=3e-6,
                )
                records.append(
                    {
                        "case": name,
                        "variant": variant,
                        "max_error": float(np.max(np.abs(actual - reference))),
                    }
                )
                if variant in (1, 2, 4):
                    empty = backend.compute(dx[:0], dr)
                    assert empty.size == 0
                    try:
                        backend.compute(dx, dr[:0])
                    except RuntimeError as error:
                        assert "nonempty" in str(error)
                    else:
                        raise AssertionError("empty background accepted")
            finally:
                backend.close()

    for dtype, seed, classification in itertools.product(
        [np.float32, np.float64], range(3), [False, True]
    ):
        rng = np.random.default_rng(seed)
        train = rng.normal(size=(100, 5)).astype(dtype)
        target = (
            rng.integers(0, 3, 100) if classification else rng.normal(size=100)
        )
        cls = (
            RandomForestClassifier if classification else RandomForestRegressor
        )
        forest = cls(n_estimators=4, max_depth=5, random_state=seed).fit(
            train, target
        )
        model = treelite.sklearn.import_model(forest)
        x, background = train[:5].copy(), train[5:12].copy()
        check(
            f"forest-{dtype.__name__}-{seed}-{classification}",
            model,
            x,
            background,
        )
        check(
            f"identical-{dtype.__name__}-{seed}-{classification}",
            model,
            x[:1],
            x[:1],
        )
        # Repeating background rows must not change their empirical distribution.
        check(
            f"duplicate-background-{dtype.__name__}-{seed}-{classification}",
            model,
            x[:2],
            np.repeat(background[:2], 3, axis=0),
        )

    nodes = [
        dict(
            feature=0,
            threshold=0.5,
            left=1,
            right_child=4,
            default=False,
            op="<=",
        ),
        dict(feature=0, threshold=-0.5, left=2, right_child=3, default=True),
        dict(leaf=-2.0),
        dict(leaf=3.0),
        dict(feature=1, threshold=0.0, left=5, right_child=6),
        dict(leaf=7.0),
        dict(leaf=-4.0),
    ]
    for dtype in [np.float32, np.float64]:
        model = custom_model(nodes, 3)
        x = np.array(
            [[np.nan, 1, 0], [-1, 0, 2], [0.5, -1, 3], [np.inf, np.nan, 1]],
            dtype=dtype,
        )
        r = np.array(
            [[0.25, 0, 1], [-np.inf, 1, 0], [np.nan, -1, 0]], dtype=dtype
        )
        check(
            f"missing-repeated-boundary-{dtype.__name__}",
            model,
            x,
            r,
            baseline=False,
        )
        stump = custom_model([dict(leaf=2.5)], 3)
        check(f"constant-{dtype.__name__}", stump, x, r)
        categorical = custom_model(
            [
                dict(
                    feature=0,
                    cats=[1, 3],
                    left=1,
                    right_child=2,
                    right=True,
                    default=True,
                ),
                dict(leaf=-3.0),
                dict(
                    feature=0, cats=[3], left=3, right_child=4, default=False
                ),
                dict(leaf=2.0),
                dict(leaf=7.0),
            ],
            2,
        )
        # Unknown and negative categories exercise new traversal's membership bounds.
        cx = np.array([[1, 0], [3, 1], [2, 0], [np.nan, 1]], dtype=dtype)
        cr = np.array([[3, 0], [1, 1], [0, 1]], dtype=dtype)
        check(f"categorical-{dtype.__name__}", categorical, cx, cr)
        check(
            f"categorical-unseen-{dtype.__name__}",
            categorical,
            np.array([[1000, 0], [-1, 0], [np.inf, 0]], dtype=dtype),
            cr,
            baseline=False,
        )

    # Conjunctions check all stack buckets, exactness beyond fixed quadrature,
    # and independence from training-cover statistics.
    for depth in [8, 9, 16, 17, 32, 33, 40, 64, 65, 128]:
        chain = []
        for i in range(depth):
            chain.append(
                dict(
                    feature=i,
                    threshold=0.5,
                    left=2 * i + 1,
                    right_child=2 * i + 2,
                )
            )
            chain.append(dict(leaf=0.0))
        chain.append(dict(leaf=1.0))
        model = custom_model(chain, depth, bias=0, cover=False)
        x, r = np.ones((1, depth)), np.zeros((1, depth))
        expected = np.full((1, 1, depth + 1), 1 / depth)
        expected[..., -1] = 0
        check(
            f"depth{depth}-conjunction",
            model,
            x,
            r,
            baseline=False,
            expected=expected,
        )

    # Scalar-leaf multiclass boosting complements the vector-leaf RF cases.
    import xgboost

    rng = np.random.default_rng(81)
    train = rng.normal(size=(100, 4)).astype(np.float32)
    booster = xgboost.train(
        dict(
            max_depth=3,
            objective="multi:softprob",
            num_class=3,
            nthread=1,
            seed=81,
            base_score=0.5,
        ),
        xgboost.DMatrix(train, label=rng.integers(0, 3, len(train))),
        num_boost_round=3,
    )
    model = treelite.frontend.from_xgboost(booster)
    check("xgboost-scalar-multiclass", model, train[:4], train[4:9])
    if check_callback is None:
        print(
            f"Validated {len(records)} backend/case combinations", flush=True
        )
    return records


def benchmark(lib, repeats, seeds):
    results = []
    cases = [
        (64, 16, 10, 3, 8),
        (256, 32, 50, 6, 16),
        (256, 128, 50, 6, 16),
        (128, 32, 30, 10, 32),
        (1024, 8, 20, 4, 8),
        (16, 256, 10, 8, 16),
    ]
    workloads = [("forest", case) for case in cases]
    workloads += [
        ("all-disagree", (64, 32, 1, 8, 8)),
        ("all-identical", (64, 32, 1, 8, 8)),
    ]
    for seed, (
        workload,
        (rows, refs, trees, depth, features),
    ) in itertools.product(range(seeds), workloads):
        rng = np.random.default_rng(seed)
        train = rng.normal(size=(2048, features)).astype(np.float32)
        y = (
            train[:, 0] * train[:, 1]
            + np.sin(train[:, 2])
            + rng.normal(size=len(train)) * 0.1
        )
        forest = RandomForestRegressor(
            n_estimators=trees, max_depth=depth, random_state=seed
        ).fit(train, y)
        model = treelite.sklearn.import_model(forest)
        x = cp.asarray(rng.normal(size=(rows, features)).astype(np.float32))
        background = cp.asarray(train[:refs])
        if workload != "forest":
            nodes = []
            for nid in range((1 << depth) - 1):
                level = (nid + 1).bit_length() - 1
                nodes.append(
                    dict(
                        feature=level,
                        threshold=0.5,
                        left=2 * nid + 1,
                        right_child=2 * nid + 2,
                    )
                )
            nodes += [dict(leaf=float(v)) for v in rng.normal(size=1 << depth)]
            model = custom_model(nodes, features, bias=0)
            x = cp.ones((rows, features), dtype=cp.float32)
            background = (cp.zeros if workload == "all-disagree" else cp.ones)(
                (refs, features), dtype=cp.float32
            )
        backends, preparation = [], []
        for variant in range(5):
            start = time.perf_counter()
            backends.append(Backend(lib, model, variant))
            cp.cuda.Stream.null.synchronize()
            preparation.append((time.perf_counter() - start) * 1000)
        outputs = [b.compute(x, background) for b in backends]
        cp.cuda.Stream.null.synchronize()
        for output in outputs[1:]:
            np.testing.assert_allclose(
                cp.asnumpy(output),
                cp.asnumpy(outputs[0]),
                atol=3e-6,
                rtol=3e-6,
            )
        for _ in range(2):
            for b, out in zip(backends, outputs):
                b.compute(x, background, out)
        cp.cuda.Stream.null.synchronize()
        reference_outputs = [cp.asnumpy(out) for out in outputs]
        numerical_spread = [0.0 for _ in backends]
        samples = [[] for _ in backends]
        wall_samples = [[] for _ in backends]
        for _ in range(repeats):
            for variant in rng.permutation(5):
                start, end = cp.cuda.Event(), cp.cuda.Event()
                wall = time.perf_counter()
                start.record()
                backends[variant].compute(x, background, outputs[variant])
                end.record()
                end.synchronize()
                wall_samples[variant].append(
                    (time.perf_counter() - wall) * 1000
                )
                samples[variant].append(cp.cuda.get_elapsed_time(start, end))
                numerical_spread[variant] = max(
                    numerical_spread[variant],
                    float(
                        np.max(
                            np.abs(
                                cp.asnumpy(outputs[variant])
                                - reference_outputs[variant]
                            )
                        )
                    ),
                )
        case = dict(
            workload=workload,
            seed=seed,
            rows=rows,
            background=refs,
            trees=trees,
            depth=depth,
            features=features,
        )
        for variant in range(5):
            results.append(
                dict(
                    **case,
                    variant=variant,
                    preparation_ms=preparation[variant],
                    median_ms=float(np.median(samples[variant])),
                    max_repeat_difference=numerical_spread[variant],
                    p10_ms=float(np.percentile(samples[variant], 10)),
                    p90_ms=float(np.percentile(samples[variant], 90)),
                    std_ms=float(np.std(samples[variant])),
                    wall_median_ms=float(np.median(wall_samples[variant])),
                    samples_ms=samples[variant],
                )
            )
            backends[variant].close()
        print(
            case,
            "ms:",
            [round(float(np.median(s)), 3) for s in samples],
            flush=True,
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--new-only",
        action="store_true",
        help="Validate only new kernels (for sanitizers)",
    )
    args = parser.parse_args()
    lib = load_library(args.library)
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    output = dict(
        environment=dict(
            gpu=device["name"].decode(),
            python=platform.python_version(),
            cupy=cp.__version__,
            treelite=treelite.__version__,
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
            cuda_driver=cp.cuda.runtime.driverGetVersion(),
        ),
        validation=validate(lib, include_legacy=not args.new_only),
    )
    if not args.validate_only:
        output["benchmarks"] = benchmark(lib, args.repeats, args.seeds)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
