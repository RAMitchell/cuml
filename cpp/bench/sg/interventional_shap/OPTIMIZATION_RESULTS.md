# Interventional SHAP optimization results

All fifteen requested experiment families were executed. The strongest general
candidates combine exact coefficient lookup, local contribution reduction, and
optionally packed decision caches. Quadrature and pattern transforms remain
separate measured alternatives; no public TreeExplainer dispatch was changed.

Dense pattern reuse also wins on the complete Fashion-MNIST-small model: 10.22x
faster than the original, including input preparation.

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
| adult-small | 0.004307 | 0.002120 | 0.001944 | 0.001377 | 3.13x |
| adult-med | 0.294657 | 0.139970 | 0.122204 | 0.112658 | 2.62x |
| adult-large | 4.655908 | 2.252966 | 2.702687 | 2.177048 | 2.14x |
| cal_housing-small | 0.002981 | 0.001440 | 0.001292 | 0.000928 | 3.21x |
| cal_housing-med | 0.255812 | 0.121948 | 0.104259 | 0.107923 | 2.45x |
| cal_housing-large | 7.841380 | 3.837771 | 3.697651 | 4.373256 | 2.12x |
| covtype-small | 0.020713 | 0.010009 | 0.009275 | 0.006365 | 3.25x |
| covtype-med | 0.898475 | 0.420054 | 0.429386 | 0.419380 | 2.14x |
| covtype-large | 19.252449 | 9.277963 | 10.676888 | 12.290769 | 2.08x |
| fashion_mnist-small | 0.053070 | 0.023211 | 0.022094 | 0.016604 | 3.20x |
| fashion_mnist-med | 3.735664 | 2.049723 | 1.898459 | 1.821254 | 2.05x |
| fashion_mnist-large | 160.768088 | 180.460155 | 77.801591 | 83.643376 | 2.07x |

| Model | Best variant | Timing CV | Peak workspace MiB | Max absolute error |
|---|---|---:|---:|---:|
| adult-large | lookup_cache_reduce | 0.16% | 1068.586 | 2.11e-12 |
| adult-med | lookup_cache_reduce | 0.14% | 27.332 | 3.66e-13 |
| adult-small | lookup_cache_reduce | 2.47% | 0.184 | 8.27e-15 |
| cal_housing-large | lookup_reduce | 0.11% | 0.002 | 4.45e-13 |
| cal_housing-med | lookup_reduce | 0.07% | 0.002 | 8.69e-14 |
| cal_housing-small | lookup_cache_reduce | 0.29% | 0.184 | 4e-15 |
| covtype-large | lookup_global | 0.06% | 0.002 | 2.05e-13 |
| covtype-med | lookup_cache_reduce | 0.05% | 275.762 | 1.5e-13 |
| covtype-small | lookup_cache_reduce | 0.13% | 1.260 | 5.5e-15 |
| fashion_mnist-large | lookup_reduce | 0.13% | 0.002 | 4.87e-13 |
| fashion_mnist-med | lookup_cache_reduce | 0.07% | 348.197 | 1.09e-13 |
| fashion_mnist-small | lookup_cache_reduce | 0.18% | 1.816 | 7.49e-15 |

“Best” selects the smallest median among these three optimized candidates; other
methods were not confirmed across this entire twelve-model grid. Close differences should not be
interpreted as a reliable dispatch rule from five repetitions. Timing CV is the
standard deviation divided by the mean of repeated wall times.

The raw confirmation records also separate CUDA-event preparation and compute
time. Peak optimization workspace excludes the shared input/output/model buffers.
The earlier legacy comparison is retained in [REPOSITORY_RESULTS.md](REPOSITORY_RESULTS.md);
its Adult-large legacy correctness failure remains unresolved and is not used as
an oracle for these optimizations.


## Small-model pattern follow-up

A separate five-repetition confirmation of dense pattern reuse on the complete Fashion-MNIST-small model (10,000 rows, 100 references) measured 0.052990 seconds for the original and 0.005186 seconds for dense patterns, including input preparation. This follow-up was prompted by the small-model scaling result; it is separate from the four-way confirmation above.

## Every traversal variant on complete diagnostic models

128 rows and 100 references; median new-batch wall milliseconds from three
randomized repetitions. These timings are direct measurements of complete models,
not extrapolations from the two-round screen.

| Variant | Fashion small | Fashion large | California large | Adult large |
|---|---:|---:|---:|---:|
| original | 0.737 | 1243.217 | 120.714 | 72.913 |
| control | 0.705 | 1408.877 | 137.534 | 78.197 |
| lookup_global | 0.438 | 561.751 | 63.905 | 38.304 |
| lookup_shared | 0.433 | 562.570 | 64.005 | 38.327 |
| lookup_constant | 0.433 | 565.936 | 65.800 | 38.424 |
| incremental | 0.914 | 1499.638 | 166.567 | 92.198 |
| cache_background | 0.704 | 1366.100 | 131.611 | 72.743 |
| cache_foreground | 0.705 | 1367.371 | 131.600 | 72.723 |
| cache_both | 0.675 | 1327.380 | 125.612 | 67.241 |
| threshold_bins | 65.908 | 1397.966 | 122.461 | 73.346 |
| quadrature_leaf | 0.974 | 4482.529 | 582.315 | 313.943 |
| quadrature_subtree | 0.925 | 2160.056 | 290.542 | 155.434 |
| local_reduce | 0.677 | 1225.734 | 126.572 | 71.844 |
| lookup_reduce | 0.413 | 474.051 | 54.255 | 32.758 |
| lookup_cache_reduce | 0.376 | 459.909 | 50.408 | 22.225 |

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
| original | 0.008660 | 0.045448 | 0.360978 | 2.651215 |
| lookup_reduce | 0.005051 | 0.019200 | 0.134275 | 1.055965 |
| lookup_cache_reduce | 0.005674 | 0.016273 | 0.109590 | 1.025159 |
| pattern_background | 0.051934 | 0.083305 | 0.289983 | 1.090435 |
| pattern_foreground | 0.347141 | 0.361678 | 0.454842 | 1.232276 |
| pattern_sparse | 0.347341 | 0.363784 | 0.424107 | 0.629306 |
| pattern_dense | 0.355851 | 0.363241 | 0.375606 | 0.384910 |
| pattern_adaptive | 0.010115 | 0.683578 | 0.581900 | 0.637762 |

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

There were 1031 passing backend/fixture checks and 14 explicit unsupported cases. Sparse pattern prototypes stop above 64 distinct features; dense transforms stop
above 20. The other traversal variants pass through depth 128. Unsupported cases
are listed in the raw archive; they are not counted as successful executions.
Every measured real-data variant was checked against the frozen exact backend.

The coefficient-only CPU sweep compares against 100-digit rational arithmetic:

| Maximum depth | Recurrence max relative error | Lookup max relative error | Fixed-eight-point quadrature max relative error |
|---|---:|---:|---:|
| 16 | 3.48e-16 | 2.14e-16 | 4.44e-15 |
| 32 | 3.48e-16 | 2.14e-16 | 0.0217 |
| 64 | 1.23e-15 | 2.38e-16 | 0.228 |
| 128 | 1.45e-15 | 2.55e-16 | 0.668 |

These are errors in individual coefficients, not model-output relative errors.
Timed quadrature uses a depth-dependent exact polynomial order, not fixed eight
points. Lookup is a speed optimization here; the original recurrence was already
accurate in double precision.

Instrumented kernels count work in a separate run, excluded from timings:

| Model prefix | Variant | Node visits | Leaf visits | Global atomic updates |
|---|---|---:|---:|---:|
| fashion_mnist-large | control | 23,143,722 | 6,216,210 | 42,073,181 |
| fashion_mnist-large | lookup_global | 23,143,722 | 6,216,210 | 42,073,181 |
| fashion_mnist-large | local_reduce | 23,143,722 | 6,216,210 | 6,216,197 |
| fashion_mnist-large | lookup_reduce | 23,143,722 | 6,216,210 | 6,216,197 |
| fashion_mnist-large | lookup_cache_reduce | 23,143,722 | 6,216,210 | 6,216,197 |
| cal_housing-large | control | 1,867,666 | 344,937 | 1,482,093 |
| cal_housing-large | lookup_global | 1,867,666 | 344,937 | 1,482,093 |
| cal_housing-large | local_reduce | 1,867,666 | 344,937 | 344,937 |
| cal_housing-large | lookup_reduce | 1,867,666 | 344,937 | 344,937 |
| cal_housing-large | lookup_cache_reduce | 1,867,666 | 344,937 | 344,937 |
| adult-large | control | 1,811,386 | 341,963 | 1,537,432 |
| adult-large | lookup_global | 1,811,386 | 341,963 | 1,537,432 |
| adult-large | local_reduce | 1,811,386 | 341,963 | 341,878 |
| adult-large | lookup_reduce | 1,811,386 | 341,963 | 341,878 |
| adult-large | lookup_cache_reduce | 1,811,386 | 341,963 | 341,878 |

GPU memory and shared-memory race checks:

- memcheck: ERROR SUMMARY: 0 errors; 1031 passing numerical checks.
- racecheck: RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings); 1031 passing numerical checks.

Hardware profiling: not collected: installed Nsight Compute 2022.4.1 lists Ada/Hopper and older chips but no Blackwell support; software operation counts and compiler resource attributes were collected instead.

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
| 10. Dense transforms | Strong complete-small-model and large-background prefix results; exponential width cost still requires a fallback. |
| 11. Adaptive selection | Current heuristic often loses to an explicit method choice; do not promote it as-is. |
| 12. Leaf quadrature | Exact-order version is slower in the complete-model screen; fixed eight points is not exact beyond depth 16. |
| 13. Subtree quadrature | Improves on leaf quadrature here, but still trails lookup in the complete-model screen. |
| 14. Local reduction | Cuts atomic updates substantially; increases local state and is not always faster without considering batch size. |
| 15. Combined variants | Worth carrying forward, with cache-free fallbacks and model/batch-aware selection. |
