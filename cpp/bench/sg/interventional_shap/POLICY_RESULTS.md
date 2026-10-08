# Cheap interventional SHAP dispatch experiments

Follow-up: [depth-six experiments](DEPTH_POLICY_RESULTS.md) extend the validated
candidate depth ceiling while retaining the batch-size and memory guards below.

Measured 2026-10-08 on physical GPU 1, NVIDIA RTX PRO 6000 Blackwell (96 GB),
using the existing standalone CuPy prototypes. No production API or CUDA kernel
changed. Another observed compute process was on physical GPU 0.

## Finding

A conservative metadata-only rule is a useful candidate on this hardware:

```text
Use streamed dense Woodelf-style processing if:
    maximum tree depth <= 3
    foreground rows >= 1,000
    background rows >= 100
    foreground rows * background rows >= 1,000,000
    estimated one-leaf workspace fits the configured budget
Otherwise use lookup + local-reduction traversal, without decision caches.
Enforce a 64 MiB allocator cap on extra workspace; fall back on allocation failure.
```

The decision needs no observed-pattern counting, sorting, workload sampling or
trial execution. Depth is model metadata; the other inputs are batch dimensions.
This is deliberately conservative, not an optimal crossover model. It misses
real wins on medium/deep models and on smaller foreground batches.

## Exploration: 60 cases

Complete small and medium models for Adult, California Housing, Covtype and
Fashion-MNIST: N in {100, 10,000}, B in {10, 100, 1,000}, plus N=1,000/B=100.
Four additional large-model cases use only the first two boosting rounds at
N=10,000/B=1,000; these are explicitly not full large-model measurements.
Backgrounds are seeded permutations of the cached foreground pool (seed
20261008), with possible duplicate values and overlap with explained rows.

Three initial rules were fixed before this run. All require N>=1,000 and the
one-leaf memory guard. Their choices below select measured 64 MiB backend times
offline; they do not include an actual dispatch call.

| Initial rule | Geometric mean speedup vs traversal, all 60 | Worst slowdown |
|---|---:|---:|
| D<=3, B>=100 | 1.36x | 2.21x |
| D<=8, B>=100 | 1.57x | 7.32x |
| D<=8, B>=max(10, 2^D) | 1.57x | 2.70x |

In particular, N=1,000/B=100 loses even on shallow models. Depth alone is not
sufficient. The 49 new dense configurations on complete models other than the
previously measured Fashion-MNIST-small give the same worst slowdowns; these
are configuration holdouts, not unseen datasets or independently trained models.

## Confirmation: 32 cases

After exploration, the conservative rule above was fixed before a separate run.
Fresh foreground indices were sampled with replacement and fresh background
indices permuted from the same cached pool, seed 20261009. Each of the four
complete small models was tested at (N,B): (100,10000), (1000,1000), (2000,512),
(5000,256), (20000,100), (5000,100), and (1000,100). Four further tests duplicate
each forest tenfold at (10000,100), stressing tree count at unchanged depth.
Those duplicated forests are artificial stress tests, not independent training.

Unlike exploration, confirmation times the actual policy call as well as each
backend. All 20 cases selecting dense processing won: **3.41–10.31x**, with a
6.04x geometric mean against lookup+reduction traversal. Across all 32 cases,
the policy's geometric mean speedup is 3.07x.

The 12 cases retaining traversal have a 0.9945x geometric mean speedup. The worst
observed regression is **7.26%**, Covtype at N=1,000/B=100: direct samples
0.899/0.916/0.910 ms versus policy samples 0.901/1.059/0.976 ms. Both execute the
same backend; three repeats cannot distinguish scheduling variation from small
wrapper effects. This is not a zero-regression guarantee. The first-stage
metadata decision alone took 42–188 ns; full confirmation timings include the
Python dispatch wrapper.

The rule intentionally retains traversal at N=100/B=10,000 and N=5,000/B=100,
even though dense wins on these confirmation inputs. A broader rule would need
further independent confirmation; we do not fit one retrospectively here.

## How much Woodelf wins on small models

Full small models, N=10,000/B=100, fresh backgrounds from exploration. Wall-time
medians include rebuilding batch-dependent masks and contributions. Both
algorithms use the same input and output buffers.

| Model | Improved traversal (ms) | Streamed dense, 64 MiB cap (ms) | Speedup |
|---|---:|---:|---:|
| adult-small | 1.526 | 0.402 | 3.80x |
| cal_housing-small | 1.282 | 0.382 | 3.36x |
| covtype-small | 9.267 | 2.243 | 4.13x |
| fashion_mnist-small | 21.860 | 4.101 | 5.33x |

These compare against **lookup plus local reduction**, not the original slower
traversal. The earlier Fashion-MNIST-small result was 53.0 ms to 5.19 ms: 10.22x
against the original traversal, approximately 4.3x against lookup+reduction and
3.2x against lookup+reduction+decision caches. Those earlier measurements used
the earlier background/run; do not mix them into the paired table above.

## Memory and correctness

Dedicated CuPy pools enforce 64 MiB and 256 MiB caps on allocated/reserved extra
workspace in exploration. Confirmation uses 64 MiB. These caps **exclude common
input, model and output storage, CUDA modules and compiler-managed local memory**;
they are not limits on total GPU usage. The output alone can exceed a gigabyte.
Model path extraction is one-time CPU setup outside batch timings (0.33–20.89 ms
in confirmation), recorded in raw results. There are no persistent input caches
in the streamed dense measurements.

Confirmation's selected dense workspace peaks at **44.00 MiB**; selected traversal
uses **2 KiB** of explicit pool allocations. No ordinary confirmation case falls
back. Four intentional 512-byte-budget tests reject dense allocation and verify
correct traversal fallback; the metadata guard also rejects that tiny budget.

Exploration exposed the existing tile estimator's limitation: Covtype-medium at
N=10,000 with B=10,100,1000 exceeds the 64 MiB cap, in its constant-path handling.
The allocator rejects the request and traversal recomputes the complete output.
Each case records four fallbacks (warmup plus three samples), all correct. The
attempt adds overhead; **dense64 timings in these cases are attempt-plus-fallback,
not successful dense timings**. The 256 MiB runs succeed. A hard allocator limit
is necessary; an estimated footprint alone does not guarantee bounded memory.
The conservative rule rejects these medium models before attempting dense.

All 92 cases passed prediction-additivity checks against Treelite (1e-4 absolute
and relative tolerance). All three variants, each warmup and each of three timed
repeats, matched the improved traversal (1e-5 absolute and relative tolerance).
The maximum recorded absolute attribution difference was 3.49e-13. This checks
agreement and additivity, not a new independent exhaustive SHAP oracle. Existing
kernel correctness/sanitizer coverage is described in OPTIMIZATION_RESULTS.md;
those unchanged kernels were not re-sanitized for this dispatch-only experiment.

## Reproduction and limits

Use the cached models/data created by repository_workload.py and the same
cuml_dev environment as the prior optimization experiments:

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_PATH=/home/rorym/miniforge3/envs/cuml_dev/targets/x86_64-linux
python cpp/bench/sg/interventional_shap/policy_experiment.py \
  --cache cpp/build-interventional/workload \
  --output cpp/build-interventional/policy-experiment.json
python cpp/bench/sg/interventional_shap/policy_confirmation.py \
  --cache cpp/build-interventional/workload \
  --output cpp/build-interventional/policy-confirmation.json
```

Both scripts randomize backend order for three timed repeats, validate every
result, record wall/GPU times and source hashes, and resume completed cases.
`policy_results.json.gz` contains both complete raw runs. `policy_summary.csv`
contains medians, ranges, allocation peaks, errors and fallback counts.

This is evidence for a cheap conservative dispatch candidate on one GPU and four
model families. It does not establish universal thresholds, independent training
seed robustness, or performance on complete deep large models. The policy is
implemented in the standalone benchmark, not installed into TreeExplainer.
If only one backend is desired, lookup+reduction without decision caches remains
the predictable low-workspace choice; dense processing is an optional bounded
acceleration for the validated region.
