# Interventional SHAP experiment results — 2026-10-07

The thread-per-pair implementation was the fastest new variant on the sampled
random forests: **6.1–30.1× faster than the cached legacy backend**, across 18
forest configurations/seeds. The eight-lane variant was better on the complete
tree/all-features-differ workload (approximately 1.39× faster than cached legacy).
The full-warp variant was slower than the cached legacy backend on that stress
case. This supports retaining variants for research rather than making an
unconditional public-backend switch.

Device: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, physical GPU 1;
CUDA 12.9, driver 580.173.02; Python 3.14.5, CuPy 14.1.1, Treelite 4.7.0.
GPU clocks were not locked. GPU 1 had no competing compute job, but hosts the
desktop; GPU 0 was running an unrelated workload. These are single-machine,
synthetic-workload measurements, not a general performance guarantee.

## Timings

Milliseconds below are the median of each seed's eleven-sample median. The last
column reports the range of seed-specific cached-legacy/thread speedups. Model
preparation is excluded here and recorded separately in `results.json`.
See `README.md` for timing boundaries and the algorithms.

| Workload | X rows | Background | Trees | Max depth | Features | Legacy API | Legacy cached | Thread | 8 lanes | 32 lanes | Cached/thread range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| forest | 64 | 16 | 10 | 3 | 8 | 1.178 | 0.915 | 0.032 | 0.037 | 0.051 | 24.2–30.1× |
| forest | 256 | 32 | 50 | 6 | 16 | 3.578 | 2.501 | 0.263 | 0.840 | 1.928 | 7.4–11.9× |
| forest | 256 | 128 | 50 | 6 | 16 | 10.608 | 9.536 | 0.881 | 3.640 | 7.518 | 8.5–11.6× |
| forest | 128 | 32 | 30 | 10 | 32 | 3.712 | 2.278 | 0.288 | 0.646 | 1.159 | 6.1–8.6× |
| forest | 1024 | 8 | 20 | 4 | 8 | 1.182 | 0.729 | 0.083 | 0.268 | 0.592 | 8.1–10.7× |
| forest | 16 | 256 | 10 | 8 | 16 | 3.618 | 2.814 | 0.122 | 0.184 | 0.307 | 23.0–26.9× |
| all-disagree | 64 | 32 | 1 | 8 | 8 | 1.422 | 0.935 | 0.840 | 0.674 | 1.318 | 1.1–1.1× |
| all-identical | 64 | 32 | 1 | 8 | 8 | 1.422 | 0.931 | 0.022 | 0.022 | 0.024 | 36.8–47.0× |

Raw CUDA-event samples, p10/p90, standard deviation, synchronized wall-clock
medians, setup times, source checksums, and output repeatability are committed in
`results.json`. The largest observed repeated-output change across all timed
backends was 1.71e-14; all timed outputs agreed with the legacy reference within
3e-6 absolute/relative tolerance. The event timing includes legacy host-side
preprocessing gaps in backend 0; backend 3 removes that preprocessing from compute.

## Correctness and GPU checking

- **247 backend/case comparisons passed**, including exhaustive-coalition
  references and analytic deep-conjunction references. Maximum absolute error
  across all checked backends was 2.98e-7. Tolerance: 2e-6 absolute/relative for
  per-feature values and 3e-6 for additivity.
- **165 new-backend/case comparisons passed under CUDA memcheck: 0 errors.**
- **165 passed under CUDA racecheck: 0 hazards, 0 errors, 0 warnings.**
- Standalone release configure/build passed with CUDA architecture 120.
- Repository-pinned Ruff checks/formatting and clang-format were applied.

The unsupported legacy edge cases described in `README.md` are excluded from
legacy comparisons, not from the independent checks of the new algorithms.
The public TreeExplainer was not rerouted, so this is validation of the standalone
research backend rather than an installed cuML Python-package integration test.

## Commands used

From this branch's worktree, using the existing `cuml_dev` environment:

```bash
conda run -n cuml_dev cmake -S cpp/bench/sg/interventional_shap \
  -B cpp/build-interventional -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_PREFIX_PATH=/home/rorym/miniforge3/envs/cuml_dev \
  -DFETCHCONTENT_SOURCE_DIR_GPUTREESHAP=/home/rorym/cuml-builds/codex-enh-rf-idxt-lower-bound-cleanup/cpp-release/_deps/gputreeshap-src
conda run -n cuml_dev cmake --build cpp/build-interventional -j 2
export CUDA_VISIBLE_DEVICES=1
export CUDA_PATH=/home/rorym/miniforge3/envs/cuml_dev/targets/x86_64-linux
/home/rorym/miniforge3/envs/cuml_dev/bin/python \
  cpp/bench/sg/interventional_shap/experiment.py \
  --library cpp/build-interventional/libinterventional_shap_experiment.so \
  --output /tmp/iv-final-results.json --repeats 11 --seeds 3
```

Sanitizer commands used the same interpreter, library, and CUDA environment, with
`compute-sanitizer --tool memcheck --error-exitcode 99` or
`compute-sanitizer --tool racecheck --error-exitcode 99` before the Python command
and `--validate-only --new-only` instead of the timing options. Neither libcuml
nor the shared development environment was rebuilt/installed.
