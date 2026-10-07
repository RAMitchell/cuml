# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assemble measured optimization results; refuses an incomplete experiment grid."""

import argparse
import csv
import gzip
import json
import statistics
from pathlib import Path


def med(samples, key="wall_seconds"):
    return statistics.median(s[key] for s in samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checks-log-dir", type=Path)
    parser.add_argument("--hardware-profile-note", default="not collected")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    def read(name):
        return json.loads((args.build / name).read_text())

    screen = read("optimization-screen.json")
    full = read("optimization-full-screen.json")
    pattern_confirmation = read("optimization-pattern-confirmation.json")
    validation = read("optimization-validation.json")
    probes = read("optimization-probes.json")
    if args.checks_log_dir is not None:
        sanitizer = {"hardware_profile": args.hardware_profile_note}
        for tool in ["memcheck", "racecheck"]:
            checks = read(f"optimization-{tool}.json")
            log = (
                args.checks_log_dir / f"optimization-{tool}.log"
            ).read_text()
            summaries = [
                line.strip("= ")
                for line in log.splitlines()
                if "SUMMARY:" in line
            ]
            expected = (
                "ERROR SUMMARY: 0 errors"
                if tool == "memcheck"
                else "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
            )
            assert summaries and summaries[-1] == expected, (
                f"{tool} did not pass"
            )
            sanitizer[tool] = dict(
                summary=summaries[-1],
                passed=sum(r["status"] == "passed" for r in checks),
                unsupported=sum(r["status"] == "unsupported" for r in checks),
            )
        (args.build / "optimization-sanitizers.json").write_text(
            json.dumps(sanitizer, indent=2) + "\n"
        )
    else:
        sanitizer = read("optimization-sanitizers.json")
    cache_tiles = read("optimization-cache-tiles.json")
    matrix = {
        p.stem: json.loads(p.read_text())
        for p in sorted((args.build / "optimization-matrix").glob("*.json"))
    }
    confirmation = {
        p.stem: json.loads(p.read_text())
        for p in sorted(
            (args.build / "optimization-confirmation").glob("*.json")
        )
    }
    assert len(screen["results"]) == 80
    assert len(full["results"]) == 60
    assert len(pattern_confirmation["results"]) == 2
    assert all(len(r["cold"]) == 5 for r in pattern_confirmation["results"])
    assert len(matrix) == 15
    for name, d in matrix.items():
        assert len(d["results"]) == len(d["protocol"]["models"]) * len(
            d["protocol"]["variants"]
        )
        assert all(len(r["cold"]) == 3 for r in d["results"]), name
    assert len(confirmation) == 12 and all(
        d["complete"] for d in confirmation.values()
    )
    assert len(cache_tiles) == 36
    assert all(
        len(r["samples"]) == 5
        for d in confirmation.values()
        for r in d["results"]
    )
    combined = dict(
        screen=screen,
        full_screen=full,
        pattern_confirmation=pattern_confirmation,
        matrix=matrix,
        cache_tiles=cache_tiles,
        confirmation=confirmation,
        validation=validation,
        probes=probes,
        sanitizer=sanitizer,
    )
    raw = json.dumps(combined, separators=(",", ":")).encode()
    (out / "optimization_results.json.gz").write_bytes(
        gzip.compress(raw, mtime=0)
    )
    with (out / "optimization_summary.csv").open("w") as f:
        fields = [
            "stage",
            "model",
            "variant",
            "trees",
            "rows",
            "background",
            "tile",
            "cold_median_seconds",
            "warm_median_seconds",
            "cold_cv_percent",
            "peak_workspace_bytes",
            "max_error",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for stage, data in [
            ("screen", screen),
            ("full_screen", full),
            ("pattern_confirmation", pattern_confirmation),
            *matrix.items(),
        ]:
            for r in data["results"]:
                values = [s["wall_seconds"] for s in r["cold"]]
                writer.writerow(
                    dict(
                        stage=stage,
                        **{k: r.get(k, "") for k in fields[1:7]},
                        cold_median_seconds=med(r["cold"]),
                        warm_median_seconds=med(r["warm"])
                        if r["warm"]
                        else "",
                        cold_cv_percent=100
                        * statistics.pstdev(values)
                        / statistics.mean(values),
                        peak_workspace_bytes=r["peak_workspace_bytes"],
                        max_error=r["max_error"],
                    )
                )
    text = """# Interventional SHAP optimization results

All fifteen requested experiment families were executed. The strongest general
candidates combine exact coefficient lookup, local contribution reduction, and
optionally packed decision caches. Quadrature and pattern transforms remain
separate measured alternatives; no public TreeExplainer dispatch was changed.

The [experiment specification](OPTIMIZATION_EXPERIMENTS.md) maps all variants to
the requested list and documents algorithms, prototype limits, and reproduction.
Full raw measurements are in [optimization_results.json.gz](optimization_results.json.gz),
with a flat [screening/scaling summary](optimization_summary.csv).

## Full twelve-model confirmation

10,000 foreground rows, 100 background rows, complete cached models from the
previous repository workload. Seconds are medians of five randomized repetitions.
New-batch total wall time includes rebuilding input decision caches, zeroing,
computation and final bias handling. Model preparation and compilation are excluded.
The original is the frozen, unoptimized experimental backend, not legacy GPUTreeShap.

| Model | Original | Lookup | Lookup + reduction | Lookup + caches + reduction | Best speedup |
|---|---:|---:|---:|---:|---:|
"""
    for dataset in ["adult", "cal_housing", "covtype", "fashion_mnist"]:
        for size in ["small", "med", "large"]:
            name = f"{dataset}-{size}"
            d = confirmation[name]
            by = {r["variant"]: r for r in d["results"]}
            times = [
                med(by[v]["samples"])
                for v in [
                    "original",
                    "lookup_global",
                    "lookup_reduce",
                    "lookup_cache_reduce",
                ]
            ]
            text += (
                f"| {name} | "
                + " | ".join(f"{v:.6f}" for v in times)
                + f" | {times[0] / min(times[1:]):.2f}x |\n"
            )
    text += "\n| Model | Best variant | Timing CV | Peak workspace MiB | Max absolute error |\n|---|---|---:|---:|---:|\n"
    for name, d in confirmation.items():
        best = min(
            (r for r in d["results"] if r["variant"] != "original"),
            key=lambda r: med(r["samples"]),
        )
        values = [sample["wall_seconds"] for sample in best["samples"]]
        cv = 100 * statistics.pstdev(values) / statistics.mean(values)
        text += f"| {name} | {best['variant']} | {cv:.2f}% | {best['peak_workspace_bytes'] / 2**20:.2f} | {best['max_error']:.3g} |\n"
    text += """
“Best” selects the smallest median among these three optimized candidates; other
methods were not confirmed across this entire twelve-model grid. Close differences should not be
interpreted as a reliable dispatch rule from five repetitions. Timing CV is the
standard deviation divided by the mean of repeated wall times.

The raw confirmation records also separate CUDA-event preparation and compute
time. Peak optimization workspace excludes the shared input/output/model buffers.
The earlier legacy comparison is retained in [REPOSITORY_RESULTS.md](REPOSITORY_RESULTS.md);
its Adult-large legacy correctness failure remains unresolved and is not used as
an oracle for these optimizations.

## Every traversal variant on complete diagnostic models

128 rows and 100 references; median new-batch wall milliseconds from three
randomized repetitions. These timings are direct measurements of complete models,
not extrapolations from the two-round screen.

| Variant | Fashion small | Fashion large | California large | Adult large |
|---|---:|---:|---:|---:|
"""
    models = [
        "fashion_mnist-small",
        "fashion_mnist-large",
        "cal_housing-large",
        "adult-large",
    ]
    by = {(r["model"], r["variant"]): r for r in full["results"]}
    for variant in full["protocol"]["variants"]:
        text += (
            f"| {variant} | "
            + " | ".join(
                f"{1000 * med(by[name, variant]['cold']):.3f}"
                for name in models
            )
            + " |\n"
        )
    text += """
Global, shared, and constant coefficient storage perform similarly here. Shared
or constant placement offers no consistent large-model advantage over global
loads. Updating coefficients incrementally is slower than lookup in these tests.
Threshold binning incurs substantial batch preparation overhead, especially with
784 input features. Both exact quadrature variants are slower than lookup on
these complete models.

## Pattern reuse and background scaling

All variants were also run on the same first two boosting rounds of each diagnostic
model at every combination of 1/100/10,000 foreground rows and 1/10/100/1000
references, with three randomized repetitions. This keeps the full scaling matrix
tractable without presenting prefixes as complete-model timings.

Below: Fashion-MNIST-large's two-round prefix, 10,000 rows; total new-batch wall
seconds include streamed preparation where a persistent cache exceeded 256 MB.

| Variant | 1 reference | 10 references | 100 references | 1000 references |
|---|---:|---:|---:|---:|
"""
    selected = [
        "original",
        "lookup_reduce",
        "lookup_cache_reduce",
        "pattern_background",
        "pattern_foreground",
        "pattern_sparse",
        "pattern_dense",
        "pattern_adaptive",
    ]
    for v in selected:
        cells = []
        for nr in [1, 10, 100, 1000]:
            row = next(
                r
                for r in matrix[f"scaling-n10000-b{nr}"]["results"]
                if r["model"] == "fashion_mnist-large" and r["variant"] == v
            )
            cells.append(f"{med(row['cold']):.6f}")
        text += f"| {v} | " + " | ".join(cells) + " |\n"
    text += """
The dense transform is competitive in the large-background regime, while its
exponential path-width cost and preparation overhead make it unsuitable as an
unconditional replacement. The tested adaptive heuristic does not reliably pick
the fastest alternative; its decision overhead is included in these numbers.

Prepared-input warm lookups are recorded separately in the raw data. They reuse
both the model and the same foreground/background values; they are not the cost
of explaining a new batch. Pattern generation, sorting, contribution construction,
and host tile scheduling are included in new-batch costs. Streamed timings exclude
the initial failed attempt to build a persistent cache; the measured mode has
already been selected. Screening peak workspace includes that attempt. These CuPy-orchestrated
prototypes do not establish a hardware-independent lower bound for Woodelf.

Leaf-tile sweeps (8/64/256) and foreground decision-tile sweeps
(32/256/2048/10,000) are included in the raw archive. The adaptive policy was tested
as specified, including cases where its heuristic chose poorly; it was not tuned
to hide slow cases.

## Numerical accuracy and traversal work

The original exhaustive-coalition and analytic depth-boundary suite exercises
regression/multiclass models, float32/float64 inputs, repeated features, categorical
and missing values, comparison boundaries, and independent CUDA streams.
"""
    passed = sum(r["status"] == "passed" for r in validation)
    unsupported = [r for r in validation if r["status"] != "passed"]
    text += f"\nThere were {passed} passing backend/fixture checks and {len(unsupported)} explicit unsupported cases. "
    text += """Sparse pattern prototypes stop above 64 distinct features; dense transforms stop
above 20. The other traversal variants pass through depth 128. Unsupported cases
are listed in the raw archive; they are not counted as successful executions.
Every measured real-data variant was checked against the frozen exact backend.

The coefficient-only CPU sweep compares against 100-digit rational arithmetic:

| Maximum depth | Recurrence max relative error | Lookup max relative error | Fixed-eight-point quadrature max relative error |
|---|---:|---:|---:|
"""
    for r in probes["weights"]:
        text += f"| {r['depth']} | {r['recurrence_max_relative_error']:.3g} | {r['lookup_max_relative_error']:.3g} | {r['fixed_quad8_max_relative_error']:.3g} |\n"
    text += """
These are errors in individual coefficients, not model-output relative errors.
Timed quadrature uses a depth-dependent exact polynomial order, not fixed eight
points. Lookup is a speed optimization here; the original recurrence was already
accurate in double precision.

Instrumented kernels count work in a separate run, excluded from timings:

| Model prefix | Variant | Node visits | Leaf visits | Global atomic updates |
|---|---|---:|---:|---:|
"""
    for r in probes["work"]:
        text += f"| {r['model']} | {r['variant']} | {r['node_visits']:,} | {r['leaf_visits']:,} | {r['atomic_updates']:,} |\n"
    text += "\nGPU memory and shared-memory race checks:\n\n"
    for tool in ["memcheck", "racecheck"]:
        result = sanitizer[tool]
        text += f"- {tool}: {result['summary']}; {result['passed']} passing numerical checks.\n"
    text += f"\nHardware profiling: {sanitizer['hardware_profile']}.\n"
    text += """
Compiler register counts and per-thread local-memory sizes accompany the raw
records. Local reduction increases local state while reducing output atomics;
lookup changes arithmetic without changing the visited tree regions.

## Scope

These are measurements on one RTX PRO 6000 Blackwell GPU, CUDA 12.9, CuPy 14.1.1,
Treelite 4.7.0 and XGBoost 3.2.0. Calls were run sequentially on physical GPU 1.
Repeated-call variability is recorded; background and retraining seeds were not
varied. No implementation was promoted into the public API. Production integration
still needs model/input cache invalidation, memory-budget policy, stream-aware
resource ownership, and additional hardware coverage.
"""
    pc = {r["variant"]: r for r in pattern_confirmation["results"]}
    assert len(pc) == 2 and all(len(r["cold"]) == 5 for r in pc.values())
    text += (
        "\n## Small-model pattern follow-up\n\nA separate five-repetition confirmation of dense pattern reuse on the complete "
        "Fashion-MNIST-small model (10,000 rows, 100 references) measured "
        f"{med(pc['original']['cold']):.6f} seconds for the original and "
        f"{med(pc['pattern_dense']['cold']):.6f} seconds for dense patterns, including "
        "input preparation. This follow-up was prompted by the small-model scaling result; "
        "it is separate from the four-way confirmation above.\n"
    )
    text += """
## Experiment decisions

| Experiment | What the measurements support |
|---|---|
| 1. Coefficient lookup | Useful arithmetic optimization, especially with reduction; plain lookup can lose on large batches. |
| 2. Table placement | Global/shared/constant are close in the complete-model 128-row screen; no clear universal placement winner. |
| 3. Incremental weights | Slower than lookup in the diagnostic screen; deprioritize this formulation. |
| 4. Background decision cache | Modest standalone gains; evaluate preparation and memory before enabling it. |
| 5. Foreground decision cache | Useful in some combinations; full-model storage can cost multiple GiB. Larger row tiles reduce launch overhead. |
| 6. Threshold bins | Significant preparation cost for many features; little reason to prefer this prototype for a fresh small batch. |
| 7. Background pattern frequencies | Helps the large-background prefix workload, but dense transforms can be faster. |
| 8. Foreground pattern reuse | Same-input reuse can be cheap; rebuilding for new foreground rows must be included. |
| 9. Sparse pattern pairs | Exact alternative with sorting/construction overhead; not a universal replacement. |
| 10. Dense transforms | Strong large-background results on the tested prefixes; exponential width cost still requires a fallback. |
| 11. Adaptive selection | Current heuristic often loses to an explicit method choice; do not promote it as-is. |
| 12. Leaf quadrature | Exact-order version is slower in the complete-model screen; fixed eight points is not exact beyond depth 16. |
| 13. Subtree quadrature | Improves on leaf quadrature here, but still trails lookup in the complete-model screen. |
| 14. Local reduction | Cuts atomic updates substantially; increases local state and is not always faster without considering batch size. |
| 15. Combined variants | Worth carrying forward, with cache-free fallbacks and model/batch-aware selection. |
"""
    (out / "OPTIMIZATION_RESULTS.md").write_text(text)
    print("Wrote report and complete compressed measurement archive")


if __name__ == "__main__":
    main()
