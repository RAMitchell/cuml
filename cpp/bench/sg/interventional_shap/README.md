# Interventional Tree SHAP traversal experiment

This branch develops the interventional part of the proposed GPU Tree SHAP replacement.
The new implementation is in `cpp/src/explainer/interventional_shap.cuh`. It reads
Treelite trees directly and has no GPUTreeShap dependency. It is an experimental
backend exercised by this standalone harness; the public TreeExplainer dispatch
has not changed. Ordinary SHAP, standard interactions, and Taylor interactions
remain subsequent migration work.

Base: cuML `1911c12b3b9536cf4f3f6390cca06d2732a75785` (upstream main when started).
The motivation is XGBoost's in-tree QuadratureTreeSHAP at
[`98eebd02169136ecf1abe0c37c180a412e34b686`](https://github.com/dmlc/xgboost/tree/98eebd02169136ecf1abe0c37c180a412e34b686/src/predictor/interpretability).
This interventional implementation is new code, not a port of an existing XGBoost
interventional backend. It shares the tree-traversal approach and polynomial
interpretation, but integrates the interventional polynomial analytically.

## Algorithm

For a foreground row x and background row r, a coalition chooses each feature's
value from either x or r. The empirical-background Shapley value is the average
of these fixed-reference games. Linearity permits processing each tree and each
row/reference pair independently.

At a split:

- If x and r take the same child, only that child is reachable.
- If they differ and this feature already has a source assignment on the current
  path, follow that source. A repeated split never creates another player.
- Otherwise visit the foreground child, then the background child, recording one
  distinct feature assignment in a depth-first traversal stack.

At a reachable leaf, let a be the number of features forced to foreground and b
those forced to background. Other features are dummy players for this leaf. Its
coalition polynomial is `leaf_value * t^a * (1-t)^b`. A foreground feature's
contribution is

```
leaf_value * integral_0^1 t^(a-1) * (1-t)^b dt
  = leaf_value / (a * binomial(a+b, a)).
```

A background feature contributes
`-leaf_value / (b * binomial(a+b, a))`. No division by zero occurs: a foreground
feature implies a > 0, and likewise for b. The baseline is the leaf reached with
all assignments set to background (a = 0). All terms are averaged over background
rows, then ensemble averaging and the global base score are applied.

The reciprocal binomial coefficient is computed using a short multiplicative
recurrence in double precision, avoiding factorial overflow. Unlike fixed
8-point quadrature, this formulation introduces no depth-dependent quadrature
approximation. Floating-point accumulation still has normal rounding error.
Training cover statistics are unnecessary.

## GPU variants and comparison

| ID | Backend | Work assigned to a GPU thread/group |
|---|---|---|
| 0 | Current GPUTreeShap API | Current cuML paths; preprocessing on each call |
| 1 | Thread pair | One thread per tree/foreground/reference triple; private stack |
| 2 | Warp pair | 32 lanes per triple; shared stack, parallel leaf updates |
| 3 | Cached GPUTreeShap | Same legacy kernel, with path preprocessing moved to setup |
| 4 | Subwarp pair | Eight lanes per triple; four independent groups per warp |

Stack capacities are selected from 8, 16, 32, 64, and 128 using the ensemble's
maximum tree depth. All variants use double output accumulation. Both float32
and float64 inputs are exercised. The new variants accept an explicit CUDA
stream, check launch errors, and perform no dynamic scratch allocations during computation.
Model preparation currently uses synchronous Thrust device-vector uploads.

The legacy backends are compiled from this checkout's production adapter and
its exact pinned GPUTreeShap revision. Backend 3 is essential to distinguish
traversal gains from avoiding repeated preprocessing. All timings include output
zeroing and identical final ensemble averaging/base-score handling. Model
preparation is measured separately; training, input uploads, and host output
copies are excluded from compute timing. Legacy and new output buffers are
preallocated. CUDA-event and synchronized wall-clock timings are both recorded.

The synthetic experiment uses three dataset/model seeds, eleven timing samples per backend, randomized
backend order each round, and warmup calls before measurement. Results include
median, p10, p90, standard deviation, raw samples, and maximum repeated-output
difference. Forest workloads vary foreground/background sizes, features, depth,
and tree count. Fully disagreeing pairs in a complete tree deliberately remove
pruning opportunities; identical pairs test the opposite extreme.

Measured results and sanitizer outcomes are in [RESULTS.md](RESULTS.md), with
raw measurements in [results.json](results.json). The separate
[GPUTreeShap repository workload report](REPOSITORY_RESULTS.md) covers all four
real datasets and three model sizes with 10,000 foreground rows and 100
background rows. Its five-repeat, fixed-order protocol and correctness findings
are documented separately; the synthetic speedups do not apply uniformly.

## Reproduce

Use a CUDA environment with cuML's C++ dependencies (RAFT, RMM, Treelite and
CCCL), plus Python packages CuPy, NumPy, scikit-learn, Treelite and XGBoost. The
C++ and Python Treelite versions must match. The harness does not install or
replace libcuml in the environment.

From the repository root, in the activated environment:

```bash
cmake -S cpp/bench/sg/interventional_shap -B cpp/build/interventional -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build cpp/build/interventional -j 2
CUDA_VISIBLE_DEVICES=1 python cpp/bench/sg/interventional_shap/experiment.py \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output results.json --repeats 11 --seeds 3
```

Select the architecture and device for your machine. For conda CUDA layouts,
CuPy may also require `CUDA_PATH=$CONDA_PREFIX/targets/x86_64-linux`. An existing
copy of the pinned legacy source can be supplied to CMake with
`-DFETCHCONTENT_SOURCE_DIR_GPUTREESHAP=/absolute/path/to/gputreeshap`.

Correctness only:

```bash
python cpp/bench/sg/interventional_shap/experiment.py \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output validation.json --validate-only
```

GPU checking (run each tool separately on an idle device):

```bash
compute-sanitizer --tool memcheck --error-exitcode 99 python \
  cpp/bench/sg/interventional_shap/experiment.py \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output memcheck.json --validate-only --new-only
compute-sanitizer --tool racecheck --error-exitcode 99 python \
  cpp/bench/sg/interventional_shap/experiment.py \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output racecheck.json --validate-only --new-only
```

## Validation and limits

The independent oracle enumerates all feature coalitions and evaluates hybrid
rows with CPU Treelite prediction, rather than using another SHAP implementation.
Tests include regression, vector-leaf multiclass forests, scalar-leaf multiclass
boosting, repeated features, duplicate references, identical rows, unused
features, NaNs, infinities, threshold equality, mixed comparison operators,
categorical membership and both category directions. Depth-boundary conjunctions
through depth 128 use an analytic oracle and omit training cover. Empty foreground
input, empty background rejection, and non-default streams are exercised.

The legacy backend disagrees with the independent oracle on the combined
mixed-operator/repeated-feature boundary fixture. Out-of-range categorical values
also exceed its path-bitfield representation's safe range. These fixtures test
only the new backends. Deep conjunctions above the legacy path-width limit are
also reference-only tests. Ordinary random-forest and categorical fixtures compare
both legacy baselines and every new variant.

This is not yet a public-backend replacement. Remaining work includes:

- Integrating model preparation with TreeExplainer's model lifetime and removing
  the training-cover requirement from its interventional construction path.
- More hardware and real-model coverage, and a dispatch policy if regressions
  appear. This experiment does not claim a universal speedup.
- Better output reduction for highly overlapping pairs: atomic contention and
  full-tree traversal remain expensive when foreground/reference decisions differ
  everywhere. In the worst case all leaves are reachable.
- Depths above 128, target-specific base scores, and multi-target regression.
  The harness retains the current adapter's single-target/global-base-score scope.
- Stream-aware RMM model allocation and per-tree depth buckets if preparation
  cost or mixed-depth ensembles warrant them.

The legacy harness requires training-cover metadata; handcrafted test fixtures use
`ModelBuilder.sum_hess`. Treelite 4.7.0's Python `data_count` method in the tested
environment incorrectly calls the gain setter, so it cannot create that metadata.

Optimization experiments, including quadrature and decision-pattern reuse, are
described in [OPTIMIZATION_EXPERIMENTS.md](OPTIMIZATION_EXPERIMENTS.md). See
[OPTIMIZATION_RESULTS.md](OPTIMIZATION_RESULTS.md) for the complete measured results.

## Cheap dispatch follow-up

[Policy experiments and results](POLICY_RESULTS.md) test metadata-only switching
between bounded streamed dense processing and low-workspace traversal across 92
cases. Raw repeats are in `policy_results.json.gz`, with a flat
`policy_summary.csv` for inspection.

[Depth-six follow-up](DEPTH_POLICY_RESULTS.md) extends the same batch guards to
newly trained depth-four, depth-five and depth-six models, including 100-round
depth-six ensembles.
