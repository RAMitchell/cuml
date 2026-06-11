# Random Forest Deep Cover Type Performance Notes

Date: 2026-06-11

Branch: `codex/rf-deep-perf-notes`

## Setup

- Dataset: sklearn Forest Cover Type, full cached dataset
- Shape: 581,012 rows, 54 features, 7 classes
- Split: stratified 80/20, random_state=42
- Train/test rows: 464,809 / 116,203
- Inputs: fp32 features, int32 labels remapped to 0-based classes
- sklearn: 1.8.0, `RandomForestClassifier(n_jobs=-1)`
- cuML: 26.08.00, `RandomForestClassifier(n_bins=128, n_streams=4)`
- GPU: Tesla V100-SXM2-32GB
- CPU: dual Intel Xeon E5-2698 v4, 80 logical CPUs

## Scripts Added

- `rf_quantile_experiments/profile_covtype_rf.py`
  - Phase timing for one RF configuration.
  - Separates load, split, conversion/transfer, fit, predict, and scoring.
- `rf_quantile_experiments/profile_covtype_deep_cases.py`
  - Reuses one cover type split and warms cuML before a deep-case sweep.
- `rf_quantile_experiments/inspect_covtype_deep_model.py`
  - Fits a case and exports to sklearn-compatible trees for node/leaf counts.

Primary result artifacts were written to `/tmp`:

- `/tmp/covtype_rf_deep_cases_full.json`
- `/tmp/covtype_rf_model_inspect_d30_sqrt.json`
- `/tmp/covtype_rf_model_inspect_d30_allfeat.json`
- `/tmp/covtype_rf_d30_sqrt_nsys.nsys-rep`
- `/tmp/covtype_rf_d30_allfeat_nsys.nsys-rep`

## Deep `max_features=sqrt` Sweep

All cases use 100 trees.

| max_depth | sklearn fit | cuML fit | sklearn predict | cuML predict | sklearn acc | cuML acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 3.23s | 1.52s | 0.193s | 0.009s | 0.6834 | 0.6879 |
| 10 | 4.14s | 2.06s | 0.207s | 0.008s | 0.7520 | 0.7507 |
| 15 | 5.25s | 3.29s | 0.219s | 0.010s | 0.8270 | 0.8272 |
| 20 | 5.89s | 5.65s | 0.233s | 0.013s | 0.8891 | 0.8926 |
| 25 | 6.64s | 8.91s | 0.301s | 0.020s | 0.9295 | 0.9301 |
| 30 | 6.83s | 12.28s | 0.309s | 0.021s | 0.9460 | 0.9455 |
| None | 6.53s | 14.11s | 0.304s | 0.023s | 0.9533 | 0.9523 |

Takeaway: the crossover is around depth 20-25. cuML wins up to about depth 20, then keeps scaling upward while sklearn flattens.

## Depth 30 Feature-Sampling Sweep

All cases use 100 trees and `max_depth=30`.

| max_features | sklearn fit | cuML fit | sklearn predict | cuML predict | sklearn acc | cuML acc |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sqrt | 6.83s | 12.28s | 0.309s | 0.021s | 0.9460 | 0.9455 |
| 0.25 | 8.85s | 9.71s | 0.301s | 0.017s | 0.9592 | 0.9560 |
| 0.5 | 13.25s | 10.38s | 0.348s | 0.018s | 0.9659 | 0.9644 |
| 1.0 | 21.85s | 13.99s | 0.244s | 0.016s | 0.9656 | 0.9641 |

Takeaway: cuML loses badly only in the very deep `sqrt` regime. With `max_features >= 0.5`, cuML is faster again.

## Model Size Inspection

These counts come from exporting the fitted models to sklearn-compatible trees after fit. The export time itself is inspection-only and is not included in normal inference paths.

| case | backend | total nodes | mean nodes/tree | total leaves | mean leaves/tree | Treelite bytes |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| depth 30, sqrt | sklearn | 7,035,806 | 70,358 | 3,517,953 | 35,180 | n/a |
| depth 30, sqrt | cuML | 7,859,162 | 78,592 | 3,929,631 | 39,296 | 620,894,285 |
| depth 30, all features | sklearn | 3,740,404 | 37,404 | 1,870,252 | 18,703 | n/a |
| depth 30, all features | cuML | 4,676,902 | 46,769 | 2,338,501 | 23,385 | 369,495,745 |

Takeaway: deep `sqrt` grows much larger forests than all-features. cuML also grows somewhat larger forests than sklearn for the matched nominal settings.

## Nsight Systems Summary

Depth 30, `max_features=sqrt`, cuML:

- Python-level fit time: 12.93s under Nsight
- GPU kernel time share:
  - `nodeSplitKernel`: 4.79s, 59.6%
  - `computeSplitKernel`: 2.41s, 30.0%
  - feature-sampling DeviceFor kernel: 0.72s, 9.0%
- CUDA API:
  - 21,062 `cudaStreamSynchronize` calls, 3.75s API time
  - 66,244 `cudaLaunchKernel` calls
  - 72,787 `cudaMemcpyAsync` calls

Depth 30, `max_features=1.0`, cuML:

- Python-level fit time: 15.13s under Nsight
- GPU kernel time share:
  - `computeSplitKernel`: 10.26s, 66.8%
  - `nodeSplitKernel`: 4.71s, 30.7%
  - feature-sampling-labeled DeviceFor kernel: 0.30s, 1.9%
- CUDA API:
  - 3,110 `cudaStreamSynchronize` calls, 5.14s API time
  - 27,220 `cudaLaunchKernel` calls
  - 18,427 `cudaMemcpyAsync` calls

Interpretation:

- `computeSplitKernel` evaluates candidate splits. It builds histograms over rows for a node/feature, converts to CDFs, scores gains, and writes the best split.
- `nodeSplitKernel` applies the chosen split. It partitions/reorders `row_ids` into left/right child ranges.
- `nodeSplitKernel` dominates in the deep `sqrt` case because the forest has many more active nodes and leaves. Per-node partitions get smaller with depth, but total partition work is closer to the sum of row counts over all internal nodes. Rows that reach depth 25-30 participate in many partition operations.
- The deep `sqrt` case also creates many small kernel launches and synchronizations, which is an unfavorable shape for GPU utilization.

## Other Parameter Checks

`n_streams=8` did not help in quick one-off runs:

- depth 30, `sqrt`, `n_streams=8`: 12.33s fit
- depth 30, all features, `n_streams=8`: 15.44s fit

`n_bins` was not the main lever for depth 30 `sqrt`:

| n_bins | cuML fit | cuML acc |
| ---: | ---: | ---: |
| 32 | 14.69s | 0.9413 |
| 64 | 13.60s | 0.9449 |
| 128 | 12.28s | 0.9455 |
| 256 | 13.33s | 0.9457 |

## Current Hypothesis

The problematic shape is not data transfer, prediction, scoring, or global quantile construction. It is deep tree growth with small feature samples:

1. `sqrt` reduces the features scored per node, which helps `computeSplitKernel`.
2. But it also grows larger/deeper forests on cover type, creating many more row partitions.
3. `nodeSplitKernel` becomes dominant because row partitioning is repeated across many internal nodes.
4. Late levels produce many small nodes, so launch/sync overhead and low per-block work become increasingly important.
5. sklearn's CPU implementation appears to flatten better in this deep `sqrt` regime, while cuML remains much faster for prediction and for higher `max_features` training.

## Out-of-Place Partition Idea

Current `nodeSplitKernel` does an in-place partition with one CTA per active node. That gives poor work distribution for large nodes: the root partition is effectively serialized through one 128-thread CTA looping across the whole node.

A prototype out-of-place design can use one thread per active sample by reusing the same `WorkloadInfo` layout used by `computeSplitKernel`:

- Build `n_blocks_per_node = ceil(node.count / TPB)` in `updateWorkloadInfo`.
- Launch partition kernels with `grid.x = sum(n_blocks_per_node)`.
- Each CTA handles one node and one tile within that node.
- Each thread reads one active `row_id` and its split feature value.

Important correctness detail: a double-buffer pointer swap after every batch is not automatically safe. Inactive leaf ranges from previous batches still need to be present in whichever buffer `dataset.row_ids` points to when leaf values are computed. The clean prototype path is therefore to avoid persistent double-buffering and use the temporary array only as staging:

1. Copy valid active batch ranges from `dataset.row_ids` into `tmp_row_ids`.
2. Compact positions of left-range misfits and right-range misfits into staging position arrays.
3. Swap paired misfits in `tmp_row_ids`.
4. Copy valid active ranges back: `dataset.row_ids[i] = tmp_row_ids[i]`.
5. Keep `dataset.row_ids` as the authoritative row-id array for later split computation and leaf prediction.

This intentionally mirrors the current in-place swap semantics instead of doing a pure predicate scatter. That matters when the split threshold has ties or quantile/bin counts do not exactly match the raw `<= split.quesval` count. Unpaired misfits stay in their original side, as they do in the current kernel.

Memory overhead in the current prototype is:

- one `IdxT * n_sampled_rows` staging row-id buffer
- two `IdxT * n_sampled_rows` misfit-position buffers
- two `IdxT * max_batch_size` counters

For cover type this is still modest per concurrently trained tree/stream.

Expected tradeoffs:

- Pros:
  - Much better occupancy for root and near-root partitions.
  - Work scales as active samples instead of active nodes.
  - More natural fit for large-node partitions, where current one-CTA-per-node design is weakest.
- Cons:
  - More global memory traffic: read source row id, read split feature, write temp row id, then copy temp row id back.
  - Misfit pairing order becomes atomic-dependent when there are more misfits on one side than the other.
  - Late tiny-node levels still have at least one CTA per node unless we add a more complex packed segmented layout.
  - Need counter and misfit-position workspace plus extra kernels per batch.

Open measurement question: whether `nodeSplitKernel` time is dominated by early/medium large-node under-parallelism or by the sheer number of late tiny-node launches. The out-of-place path should help the first case much more than the second.

## Out-of-Place Partition Prototype

Implemented a prototype in:

- `cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels_impl.cuh`
- `cpp/src/decisiontree/batched-levelalgo/kernels/builder_kernels.cuh`
- `cpp/src/decisiontree/batched-levelalgo/builder.cuh`

The final prototype uses four partition-stage kernels:

1. `nodeSplitCopySourceKernel`
2. `nodeSplitMisfitKernel`
3. `nodeSplitSwapMisfitsKernel`
4. `nodeSplitCopyBackKernel`

Build validation:

- `ninja -C cpp/build-ninja-gcc12 CMakeFiles/cuml_objs.dir/src/decisiontree/batched-levelalgo/kernels/gini-float.cu.o`
- `ninja -C cpp/build-ninja-gcc12 CMakeFiles/cuml_objs.dir/src/decisiontree/batched-levelalgo/kernels/mse-float.cu.o`
- `ninja -C cpp/build-ninja-gcc12 cuml`

Important benchmarking caveat: the conda-installed `cuml` library and the rebuilt local `libcuml.so` are not apples-to-apples in this workspace. The installed library still gives the previous depth-30 `sqrt` result:

- installed cuML: 12.93s fit, 0.9455 accuracy

The rebuilt local library routed through the original in-place partition path gave:

- local in-place path: 4.99s fit, 0.8675 accuracy

Therefore the prototype should be compared against the rebuilt local in-place path, not the installed package.

Local A/B result on cover type, depth 30, `max_features=sqrt`, 100 trees:

| path | fit | predict | accuracy |
| --- | ---: | ---: | ---: |
| local in-place partition | 4.99s | 0.0136s | 0.8675 |
| local out-of-place swap prototype | 3.30s | 0.0127s | 0.8675 |

Prototype speedup against the rebuilt local in-place path: ~1.51x fit-time speedup.

Artifacts:

- `/tmp/covtype_rf_local_inplace_d30_sqrt_check.json`
- `/tmp/covtype_rf_partition_proto_swap_d30_sqrt.json`
- `/tmp/covtype_rf_partition_proto_swap_d30_sqrt_nsys.nsys-rep`
- `/tmp/covtype_rf_partition_proto_swap_d30_sqrt_nsys_run.json`

Nsight Systems kernel summary for the out-of-place prototype:

| kernel group | total GPU time | launches | share |
| --- | ---: | ---: | ---: |
| `computeSplitKernel` | 0.971s | 3,000 | 69.4% |
| `nodeSplitMisfitKernel` | 0.0857s | 3,000 | 6.1% |
| `nodeSplitSwapMisfitsKernel` | 0.0359s | 3,000 | 2.6% |
| `nodeSplitCopySourceKernel` | 0.0293s | 3,000 | 2.1% |
| `nodeSplitCopyBackKernel` | 0.0273s | 3,000 | 1.9% |
| feature sampling | 0.0848s | 3,000 | 6.1% |
| leaf kernel | 0.127s | 100 | 9.1% |

Combined partition-stage GPU time is about 0.178s in the prototype profile. The original installed-library profile had `nodeSplitKernel` at 4.79s, but because the rebuilt local library has different model behavior and timing, this should not be interpreted as a direct 27x improvement. The clean local comparison is the 4.99s to 3.30s fit-time improvement above.

## 2026-06-11 Blackwell Local-Build Bottleneck Follow-up

Environment:

- GPU: NVIDIA RTX PRO 6000 Blackwell Workstation Edition
- Build: isolated local `libcuml.so` from `/home/rorym/cuml-builds/codex-rf-deep-perf-notes/cpp-release-ninja`
- Runtime: `cuml_dev` Python with `LD_PRELOAD` pointing at the isolated `libcuml.so`
- Dataset: cached sklearn cover type, full rows, stratified 80/20 split
- Case unless noted: 100 trees, `max_depth=30`, `n_bins=128`, `n_streams=4`

Profiling limitation: Nsight Systems collected a `.qdstrm`, but this install is missing the importer needed for stats. Nsight Compute is blocked by `ERR_NVGPUCTRPERM`, so this section relies on timing sweeps and the earlier V100 Nsight summary.

### Current Local Timing Matrix

Depth sweep with `max_features=sqrt`:

| max_depth | fit | accuracy |
| ---: | ---: | ---: |
| 5 | 0.747s | 0.6745 |
| 10 | 0.549s | 0.7383 |
| 15 | 0.622s | 0.7851 |
| 20 | 0.950s | 0.8292 |
| 25 | 1.322s | 0.8545 |
| 30 | 1.600s | 0.8654 |
| None | 1.785s | 0.8692 |

Depth 30 feature sweep:

| max_features | fit | accuracy |
| ---: | ---: | ---: |
| sqrt | 1.600s | 0.8654 |
| 0.25 | 4.377s | 0.9504 |
| 0.5 | 5.767s | 0.9645 |
| 1.0 | 7.533s | 0.9638 |

`n_bins` sweep at depth 30:

| n_bins | sqrt fit | all-feature fit | sqrt accuracy | all-feature accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 1.468s | 7.759s | 0.8493 | 0.9555 |
| 64 | 1.570s | 7.398s | 0.8607 | 0.9615 |
| 128 | 1.600s | 7.533s | 0.8654 | 0.9638 |
| 256 | 1.824s | 9.052s | 0.8688 | 0.9641 |

`n_streams` sweep at depth 30, `n_bins=128`:

| n_streams | sqrt fit | all-feature fit |
| ---: | ---: | ---: |
| 1 | 1.914s | 8.196s |
| 4 | 1.600s | 7.533s |
| 8 | 1.633s | 7.575s |

Tree-count scaling at depth 30, `n_bins=128`, `n_streams=4`:

| n_estimators | sqrt fit | all-feature fit |
| ---: | ---: | ---: |
| 10 | 0.151s | 0.748s |
| 50 | 0.823s | 3.784s |
| 100 | 1.600s | 7.533s |

### Negative Experiment: Larger Column Batches

Temporarily changing `Builder::n_blks_for_cols` from 10 to 32 did not improve the all-feature case:

| n_blks_for_cols | sqrt fit | all-feature fit |
| ---: | ---: | ---: |
| 10 | 1.600s | 7.533s |
| 32 | 1.797s | 7.675s |

This rules out the simplest launch-count explanation for the all-feature slowdown. Fewer column-batch launches are apparently offset by larger `grid.y`, more simultaneously live histogram workspace, scheduling effects, or reduced cache locality.

### Bottleneck Conclusion

The out-of-place partition prototype moved the dominant cost away from row partitioning. The current bottleneck is split evaluation, especially `computeSplitKernel` histogram construction and gain evaluation across node-feature pairs.

Evidence:

- Runtime scales nearly linearly with `n_estimators`, so the cost is inside per-tree growth rather than one-time quantile construction or Python overhead.
- Runtime scales strongly with `max_features`: at depth 30, all-features is about 4.7x slower than `sqrt` on the same 54-feature dataset.
- Runtime moves only modestly with `n_bins` from 32 to 128, so bin count is not the primary lever for this dataset; the expensive part is the repeated row/feature pass itself.
- `n_streams` from 1 to 4 helps modestly, but 8 does not improve further, so stream-level tree concurrency is not the main remaining gap.
- The previous Nsight summary for the prototype already showed `computeSplitKernel` at about 69% of GPU kernel time and all partition-stage kernels combined at about 13%.

Important caveat: this local build still has the previously noted accuracy mismatch versus the installed cuML package for the deep `sqrt` case. These timing conclusions should be used for bottleneck direction, but any production change needs an apples-to-apples correctness baseline first.

### Improvement Candidates

1. Make split evaluation reuse row metadata across multiple features.
   Current `computeSplitKernel` maps one node tile and one feature to a CTA. For all-features, each feature rereads the same row-id range and labels. A multi-feature CTA or warp-group design could stage row ids and labels once, then update histograms for a small group of features. This trades shared memory/register pressure for less repeated row metadata traffic and fewer independent CTAs.

2. Add a specialized small-node split path.
   Deep levels create many small nodes. The current path still launches CTAs per node-feature and pays fixed work for tiny ranges. A packed-small-node kernel could group several tiny node-feature histograms into one CTA or warp block, improving tail utilization and reducing per-node overhead.

3. Keep the out-of-place partition path, but fuse its four stages only if partition reappears in profiles.
   Current timing points to split evaluation, not partition. Fusing copy/misfit/swap/copy-back may still help `sqrt`, but it is second-order until `computeSplitKernel` is improved.

4. Avoid a blanket increase to `n_blks_for_cols`.
   The `10 -> 32` experiment was neutral to negative. Any column-batch tuning should be adaptive and benchmarked against sampled feature count, classes, bins, large-node count, and workspace pressure.

5. Add optional stage timing instrumentation.
   Since Nsight counters are restricted, a low-overhead `CUML_RF_STAGE_TIMING=1` path using CUDA events around feature sampling, compute split, partition, and leaf prediction would make future experiments less dependent on external profiler permissions.
