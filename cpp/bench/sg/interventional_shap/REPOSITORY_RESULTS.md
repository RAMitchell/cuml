# GPUTreeShap repository workload: interventional before/after

This experiment adapts the [GPUTreeShap benchmark recipe](https://github.com/rapidsai/gputreeshap/blob/b24645352ea07292621bee0ac8b327867b46093c/benchmark/benchmark.py)
to interventional SHAP. The repository's published tables measure ordinary SHAP;
these are newly measured interventional results, not comparisons against those
historical timings. The experimental backend remains separate from public
TreeExplainer dispatch.

## Protocol

- Four datasets: Adult, California Housing, Covtype and Fashion-MNIST.
- Small/medium/large: 10/100/1000 boosting rounds, maximum depth 3/8/16,
  learning rate 0.01, GPU histogram training. Multiclass models have one tree
  per class per round; the original Covtype labels 1..7 imply eight groups.
- Newly trained XGBoost 3.2.0 models, seed 0, explicit `base_score=0.5` to retain
  the historical scalar intercept. These are not bit-identical historical models.
  Adult uses numeric category codes and categorical feature types, retaining
  native categorical splits while avoiding the string encoder unsupported by
  Treelite 4.7.0. Missing category codes become NaN.
- 10,000 foreground rows, sampled with replacement using `RandomState(432)` as
  in the source recipe. 100 background rows sampled without replacement with
  `RandomState(433)`. The same rows are used for every size/backend per dataset.
- RTX PRO 6000 Blackwell Workstation Edition (96 GB), physical GPU 1;
  CUDA 12.9.86, driver 580.173.02, Python 3.14.5, CuPy 14.1.1,
  Treelite 4.7.0. No competing GPU compute job on this device; it also hosts
  the desktop. Another GPU was used by an unrelated job.
- A complete validation call warms each backend, followed by five timed calls.
  Each call processes all 10,000 foreground rows and 100 references without
  batching. CUDA events measure compute; synchronized wall times are also saved.
  Output allocation, input upload, model preparation, training and validation
  copies are excluded. Output zeroing and common final averaging/bias handling
  are included. The current API repeats its path preprocessing inside each call;
  the cached legacy backend moves that work to setup.
- Backends run sequentially in fixed order: cached legacy (3), thread (1),
  eight-lane subgroup (4), current API (0), keeping only one backend resident.
  This controls peak memory but does not remove order/thermal effects. An
  independent CPU-only oracle ran during part of the GPU experiment. Close
  differences should not be interpreted as robust wins without randomized reruns.
- One trained model and one background sample per configuration. Timing standard
  deviations describe repeated calls, not uncertainty across retraining seeds,
  background samples, hardware, or SHAP estimators. This algorithm is exact for
  the chosen empirical background, apart from floating-point error.

## Results

Seconds per complete call; median of five measurements. Speedups compare the
new one-thread-per-pair variant to each legacy baseline. `*` marks a legacy
correctness failure, detailed below.

| Dataset/model | Current API | Cached legacy | New thread | New 8-lane | Speedup vs API | Speedup vs cached |
|---|---:|---:|---:|---:|---:|---:|
| adult-small | 0.006795 | 0.006391 | 0.004295 | 0.020888 | 1.58x | 1.49x |
| adult-med | 0.300958 | 0.297276 | 0.295249 | 0.945935 | 1.02x | 1.01x |
| adult-large* | 13.801362 | 13.603476 | 4.658907 | 14.471989 | 2.96x | 2.92x |
| cal_housing-small | 0.008896 | 0.008630 | 0.002972 | 0.019736 | 2.99x | 2.90x |
| cal_housing-med | 0.449152 | 0.444808 | 0.256651 | 0.855641 | 1.75x | 1.73x |
| cal_housing-large | 79.428469 | 77.968578 | 7.846825 | 25.117029 | 10.12x | 9.94x |
| covtype-small | 0.019974 | 0.019232 | 0.020758 | 0.092961 | 0.96x | 0.93x |
| covtype-med | 2.679631 | 2.657044 | 0.899310 | 2.717714 | 2.98x | 2.95x |
| covtype-large | 191.384125 | 187.474719 | 19.239980 | 57.212430 | 9.95x | 9.74x |
| fashion_mnist-small | 0.026889 | 0.024868 | 0.053352 | 0.209430 | 0.50x | 0.47x |
| fashion_mnist-med | 4.777838 | 4.745305 | 3.764520 | 13.075954 | 1.27x | 1.26x |
| fashion_mnist-large | 122.512617 | 121.058445 | 161.249297 | 260.543453 | 0.76x | 0.75x |

The thread variant is not a universal replacement: Fashion-MNIST-small and
Fashion-MNIST-large regress, while California Housing-large and Covtype-large
improve substantially. Adult-medium is effectively tied. The eight-lane variant
should be judged independently; allocating more lanes per pair has not generally
helped this grid. These findings motivate profiling the Fashion-MNIST cases and
retaining a fallback/dispatch option before integrating into the public API.

### Timing variability

Population standard deviation divided by mean, as a percentage, across the five
CUDA-event measurements. Raw event and wall samples plus separate setup times
are in [repository_results.json](repository_results.json).

| Dataset/model | API CV (%) | Cached CV (%) | Thread CV (%) | 8-lane CV (%) |
|---|---:|---:|---:|---:|
| adult-small | 1.023 | 0.625 | 0.460 | 0.068 |
| adult-med | 0.028 | 0.013 | 0.040 | 0.063 |
| adult-large | 0.017 | 0.146 | 0.012 | 0.037 |
| cal_housing-small | 0.443 | 0.264 | 0.326 | 0.107 |
| cal_housing-med | 0.043 | 0.008 | 0.028 | 0.157 |
| cal_housing-large | 0.029 | 0.004 | 0.013 | 0.096 |
| covtype-small | 0.948 | 0.257 | 0.130 | 0.151 |
| covtype-med | 0.010 | 0.007 | 0.063 | 0.155 |
| covtype-large | 0.026 | 0.007 | 0.038 | 0.042 |
| fashion_mnist-small | 0.611 | 0.479 | 0.189 | 0.053 |
| fashion_mnist-med | 0.025 | 0.029 | 0.030 | 0.034 |
| fashion_mnist-large | 0.036 | 0.049 | 0.051 | 0.116 |

### Actual model sizes

| Dataset/model | Features | Trees | Leaves | Maximum depth | Mean leaf depth |
|---|---:|---:|---:|---:|---:|
| adult-small | 14 | 10 | 80 | 3 | 3.000 |
| adult-med | 14 | 100 | 11,350 | 8 | 7.464 |
| adult-large | 14 | 1,000 | 442,334 | 16 | 12.956 |
| cal_housing-small | 8 | 10 | 80 | 3 | 3.000 |
| cal_housing-med | 8 | 100 | 21,691 | 8 | 7.863 |
| cal_housing-large | 8 | 1,000 | 3,382,173 | 16 | 13.992 |
| covtype-small | 54 | 80 | 560 | 3 | 2.929 |
| covtype-med | 54 | 800 | 114,420 | 8 | 7.705 |
| covtype-large | 54 | 8,000 | 6,658,214 | 16 | 13.683 |
| fashion_mnist-small | 784 | 100 | 800 | 3 | 3.000 |
| fashion_mnist-med | 784 | 1,000 | 144,470 | 8 | 7.527 |
| fashion_mnist-large | 784 | 10,000 | 2,956,547 | 16 | 11.446 |

## Validation summary

| Dataset/model | Legacy prediction checks | New thread prediction checks | New thread max legacy difference | New thread max additivity error |
|---|---|---|---:|---:|
| adult-small | Pass | Pass | 3.61e-08 | 1.4e-08 |
| adult-med | Pass | Pass | 2.31e-07 | 7.24e-07 |
| adult-large | FAIL | Pass | 0.0338 | 1.37e-05 |
| cal_housing-small | Pass | Pass | 5.52e-09 | 5.4e-08 |
| cal_housing-med | Pass | Pass | 2.42e-07 | 1.06e-06 |
| cal_housing-large | Pass | Pass | 4.27e-07 | 1.03e-05 |
| covtype-small | Pass | Pass | 4.03e-08 | 4.7e-08 |
| covtype-med | Pass | Pass | 3.73e-07 | 9.75e-07 |
| covtype-large | Pass | Pass | 7.87e-07 | 1.25e-05 |
| fashion_mnist-small | Pass | Pass | 5.94e-08 | 7.26e-08 |
| fashion_mnist-med | Pass | Pass | 2.2e-07 | 9.09e-07 |
| fashion_mnist-large | Pass | Pass | 3.46e-07 | 1.16e-05 |

## Correctness

Every foreground row is checked for finite output, raw-margin additivity and
background-mean bias against CPU Treelite predictions (absolute and relative
tolerance `1e-4`). Where legacy predictions pass those checks, new feature values
are also compared to cached legacy values (`1e-5` absolute and relative).
Both new variants must pass these checks before a timing record is written.

Adult-large fails the legacy checks: maximum additivity error is about 0.12355
and baseline error about 0.000405. We retain these timings and flag them instead
of using a known-invalid legacy output as the correctness oracle. The root cause
has not been isolated; this is a finding for this model, not a claim that all
categorical models fail.

For the row with the worst legacy additivity error (index 3593), an independent
CPU oracle enumerates all 16,384 coalitions of the 14 features over all 100
references. Bitwise deduplication reduces prediction work from 1,638,400 hybrid
rows to 47,222 unique rows while preserving the weights. Maximum error against
this exhaustive Shapley calculation is 0.024238 for legacy and 8.83e-7 for the new
thread variant, including the bias column. Full vectors are stored in the JSON.
This oracle validates individual feature attributions, beyond additivity alone.
The exhaustive real-model oracle covers this one row; the earlier synthetic
suite supplies broader exhaustive and analytic edge-case coverage.

## Reproduce

Build the standalone library as described in [README.md](README.md). Dataset
fetches require network access on the first run. Models, data and full legacy
output arrays are cached locally; they are not committed. Allow substantial disk
and device memory for the large models and multiclass outputs.

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_PATH="$CONDA_PREFIX/targets/x86_64-linux"
python cpp/bench/sg/interventional_shap/repository_workload.py \
  --prepare --cache cpp/build/interventional/workload \
  --output cpp/build/interventional/workload-results
python cpp/bench/sg/interventional_shap/repository_workload.py \
  --cache cpp/build/interventional/workload \
  --output cpp/build/interventional/workload-results \
  --library cpp/build/interventional/libinterventional_shap_experiment.so
python cpp/bench/sg/interventional_shap/validate_adult_large.py \
  --cache cpp/build/interventional/workload \
  --library cpp/build/interventional/libinterventional_shap_experiment.so \
  --output cpp/build/interventional/workload-results/adult-large-oracle.json
```

The timing command checkpoints each completed backend/model. It can resume the
same cached experiment, rejecting mismatched dimensions, repetitions or model
hashes. Use fresh cache/output directories for a new experiment or changed code;
resume is intended for the same binary and inputs. Model SHA-256 hashes and
training parameters accompany each result.
