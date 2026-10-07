# Interventional SHAP optimization experiments

These are standalone research kernels, not public TreeExplainer changes.
`optimization.py` compiles CUDA sources with CuPy/NVRTC and preserves the frozen
C++ backend from `568f23525` as `original`. `control` is the same analytic traversal
expressed in this experimental harness, so both baselines are measured.

## Experiment mapping

| Requested experiment | Executable variants / driver |
|---|---|
| 1. Exact coefficient lookup | `lookup_global` |
| 2. Table placement | `lookup_global`, `lookup_shared`, `lookup_constant` |
| 3. Incremental weights | `incremental` |
| 4. Background split decisions | `cache_background` |
| 5. Foreground split decisions | `cache_foreground`, `cache_both`; `benchmark_cache_tiles.py` |
| 6. Exact threshold bins | `threshold_bins` |
| 7. Background pattern frequencies | `pattern_background` |
| 8. Foreground pattern reuse | `pattern_foreground` |
| 9. Sparse observed pattern pairs | `pattern_sparse` |
| 10. Dense Woodelf-HD transform | `pattern_dense` |
| 11. Adaptive dispatch | `pattern_adaptive` |
| 12. Leaf quadrature | `quadrature_leaf`; fixed-eight-point accuracy probe |
| 13. Quadrature subtree accumulation | `quadrature_subtree` |
| 14. Local contribution reduction | `local_reduce` |
| 15. Combined variants | `lookup_reduce`, `lookup_cache_reduce` |

## Algorithms and limits

Coefficient tables hold the symmetric inverse binomial plus reciprocal counts.
They are generated once per depth specialization and never modified by a kernel.
The largest depth-128 tables fit in constant/shared memory. The incremental
variant saves the parent inverse binomial and updates it by the appropriate
count ratio at each new assignment. All variants accumulate in double precision.

Decision caches pack 32 row outcomes per word for each node. They include missing
and categorical routing. Threshold bins use both lower and upper insertion ranks
so strict and non-strict tests remain distinct; NaNs use a separate sentinel and
categorical tests retain the original membership calculation. Binning preparation
is included in the measured new-batch cost.

The subtree quadrature kernel assigns lanes to abscissae and returns conditional
subtree expectations. On completion of a feature's first divergent split it adds
`reach_probability * (foreground_subtree - background_subtree)`, integrates across
abscissae, and then returns the mixed subtree value. Previously assigned features
follow their selected source. It uses `ceil(actual_depth / 2)` Gauss-Legendre
points, with multiple groups when more than 32 points are required. Fixed eight
points are studied only as an explicitly approximate numerical probe.

Local reduction keeps a contribution accumulator for each active feature
assignment and flushes it when the assignment is popped. Ancestors accumulate
across both subtrees. This reduces atomic updates at the cost of more local state.

Pattern variants build one bit per distinct path feature, ANDing all conditions
on repeated features. Sparse variants group observed masks with frequencies and
reject incompatible foreground/background masks before computing contributions.
Foreground grouping is scoped to a particular input batch; its preparation must
be paid again for new rows. Background frequencies preserve their joint empirical
distribution. Cache reuse is valid only for unchanged model and input values.

The dense variant implements the Woodelf-HD structured multiplication as a
superset zeta transform of background frequencies, multiplication by reversed
frequencies and the feature's anti-diagonal coefficients, and a subset zeta
transform. It processes equally sized paths in tiles. Its mathematical basis is
[Woodelf-HD, Sections 3–5](https://arxiv.org/html/2604.10569v1#S3);
the authors' [reference implementation](https://github.com/ron-wettenstein/woodelf/blob/e3a528e0309882042e885078913a3d7852203ddd/woodelf/core/path_to_s_vectors/archive/woodelfhd_paper_version_p2s.py)
was consulted to verify the transform. The CUDA implementation here is separate.
Pattern caching follows [Woodelf](https://arxiv.org/html/2511.09376v1), while
quadrature follows the integral/subtree formulation in
[Quadrature-TreeSHAP](https://arxiv.org/html/2605.04497v1). Exact threshold-region
reuse is also motivated by [Woodelf++](https://arxiv.org/html/2605.14578v1).

Pattern masks currently support up to 64 distinct features per path. Dense
transforms explicitly reject more than 20 distinct features. These are prototype
limits, reported as unsupported in the depth tests, not silent approximations.
The other traversal variants support the existing depth-128 range. Adaptive
selection uses direct traversal for at most eight foreground rows or four
references, then compares sparse observed-pattern work against dense transform
work per tile. It is an experimental heuristic, not a fitted production policy.

## Measurement protocol

- Initial screening: the first two boosting rounds of Fashion-MNIST small/large,
  California Housing large, and Adult large; 128 rows, 100 references, five
  repetitions. Every variant is run on exactly the same sliced model.
- Complete-model screening: all traversal variants on the same four complete
  models, 128 rows and 100 references, three repetitions.
- Scaling: every variant on all four two-round prefixes, with the Cartesian
  product of 1/100/10,000 foreground rows and 1/10/100/1000 references, three
  repetitions. These are measured prefixes, never extrapolated full-model times.
- Pattern leaf tiles: 8/64/256 leaves with 1000 rows and 100 references.
- Foreground decision-cache tiles: 32/256/2048/10,000 rows, retaining background
  decisions, with 10,000 total rows and 100 references.
- Follow-up: dense patterns versus the original on the complete Fashion-MNIST
  small model, 10,000 rows and 100 references, five repetitions.
- Final confirmation: complete twelve-model grid, 10,000 foreground rows and
  100 references, five repetitions of the original and selected lookup/reduction
  variants. Preparation is included in the total and also measured separately.

Variant order is randomized independently in each timing round of screening,
scaling, and confirmation. The separate tile ablations use fixed variant/tile
order. Inputs and model seeds remain fixed; repeated-call variance is not
background-sampling or retraining variance. Model and path construction are
recorded separately. JIT compilation is excluded from execution timing.

Screening records cold/new-batch preparation plus execution, and execution with
already prepared inputs. “Cold” means rebuilding input-dependent caches; hardware
caches are not flushed. Inputs remain fixed across timing repetitions. Pattern caches exceeding the configured 256 MB retained
cache budget are released and measured in streaming mode; no cached execution
number is assigned to them. Timings then measure the established streaming mode;
they exclude the initial failed attempt to retain all tiles. Screening peak
workspace includes that initial attempt, and `cache_bytes` for streamed records
is the attempted retained size at the cutoff, not an active persistent cache.
This estimate sums array entries conservatively and can count aliased arrays more
than once; allocator peaks reflect actual allocations.
The budget limits retained caches, not total device
memory: output arrays and temporary buffers are additional. A dedicated CuPy
allocator tracks peak live workspace allocations, including temporary arrays and
output during screening. It excludes driver/module allocations, allocations
made outside CuPy (including C++ model storage), and compiler local memory. Final confirmation reports optimization workspace separately from its
shared output and model allocations. Compiler register/local-memory attributes
are included rather than treating allocator measurements as total GPU footprint.

## Reproduction

Build the original library as in [README.md](README.md), activate the same CUDA
Python environment, and select an otherwise idle GPU. The cached datasets/models
come from `repository_workload.py --prepare`; no model is retrained by these
optimization runners.

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"
python cpp/bench/sg/interventional_shap/validate_optimizations.py \
  --output cpp/build/interventional/optimization-validation.json
python cpp/bench/sg/interventional_shap/benchmark_optimizations.py \
  --cache cpp/build/interventional/workload \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output cpp/build/interventional/optimization-screen.json
python cpp/bench/sg/interventional_shap/run_optimization_matrix.py \
  --stage scaling --cache cpp/build/interventional/workload \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output-dir cpp/build/interventional/optimization-matrix
python cpp/bench/sg/interventional_shap/run_optimization_matrix.py \
  --stage tiles --cache cpp/build/interventional/workload \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output-dir cpp/build/interventional/optimization-matrix
python cpp/bench/sg/interventional_shap/benchmark_cache_tiles.py \
  --cache cpp/build/interventional/workload \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output cpp/build/interventional/optimization-cache-tiles.json
python cpp/bench/sg/interventional_shap/probe_optimizations.py \
  --cache cpp/build/interventional/workload \
  --output cpp/build/interventional/optimization-probes.json
python cpp/bench/sg/interventional_shap/benchmark_optimizations.py \
  --cache cpp/build/interventional/workload \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output cpp/build/interventional/optimization-pattern-confirmation.json \
  --models fashion_mnist-small --variants original pattern_dense \
  --rounds 0 --rows 10000 --background 100 --repeats 5
python cpp/bench/sg/interventional_shap/confirm_optimizations.py \
  --cache cpp/build/interventional/workload \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output-dir cpp/build/interventional/optimization-confirmation
```

Capture sanitizer logs and per-case outputs for report generation:

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 \
  python cpp/bench/sg/interventional_shap/validate_optimizations.py \
  --output cpp/build/interventional/optimization-memcheck.json \
  > cpp/build/interventional/optimization-memcheck.log 2>&1
compute-sanitizer --tool racecheck --error-exitcode 99 \
  python cpp/bench/sg/interventional_shap/validate_optimizations.py \
  --output cpp/build/interventional/optimization-racecheck.json \
  > cpp/build/interventional/optimization-racecheck.log 2>&1
```

For full-model screening, add `--rounds 0 --repeats 3` to the screening command
and select the traversal variants with `--variants`. Run `compute-sanitizer`
memcheck and racecheck separately around `validate_optimizations.py`. Unsupported
pattern depths remain explicitly recorded in its JSON output.

Generate the combined report after all runs and sanitizer summaries are present:

```bash
python cpp/bench/sg/interventional_shap/report_optimizations.py \
  --build cpp/build/interventional \
  --output-dir cpp/bench/sg/interventional_shap \
  --checks-log-dir cpp/build/interventional
```
