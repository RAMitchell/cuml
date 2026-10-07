/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Standalone research harness. Compile the production adapter into this library
// so the baseline uses exactly the same cuML revision as the experiment.
#include "../../../src/explainer/interventional_shap.cuh"
#include "../../../src/explainer/tree_shap.cu"

#include <string>

namespace iv = ML::Explainer::interventional;
template <typename Condition>
struct Cached {
  thrust::device_vector<gpu_treeshap::PathElement<Condition>> paths;
  thrust::device_vector<std::size_t> bins;
};
struct Experiment {
  std::variant<Cached<SplitCondition<float>>, Cached<SplitCondition<double>>> cached;
  ML::Explainer::TreePathHandle legacy;
  iv::Model model;
  int backend;
  std::size_t groups, trees;
  bool average;
  double bias;
};
thread_local std::string last_error;

extern "C" char const* iv_error() { return last_error.c_str(); }
extern "C" void* iv_create(void* handle, int backend)
{
  try {
    auto const& model = *static_cast<treelite::Model*>(handle);
    auto e            = std::make_unique<Experiment>();
    e->backend        = backend;
    e->groups         = model.num_class[0];
    e->average        = model.average_tree_output;
    e->bias           = model.base_scores[0];
    e->trees          = std::visit([](auto const& p) { return p.trees.size(); }, model.variant_);
    if (backend == 0 || backend == 3) {
      e->legacy = ML::Explainer::extract_path_info(handle);
      if (backend == 3) {
        std::visit(
          [&](auto const& info) {
            using Path      = typename std::decay_t<decltype(info->path_segments)>::value_type;
            using Condition = typename Path::split_type;
            auto& cache     = e->cached.emplace<Cached<Condition>>();
            auto paths      = info->path_segments;
            gpu_treeshap::detail::PreprocessPaths<thrust::device_allocator<int>, Condition>(
              &paths, &cache.paths, &cache.bins);
          },
          e->legacy);
      }
    } else {
      e->model = std::visit([&](auto const& p) { return iv::Prepare(model, p); }, model.variant_);
    }
    return e.release();
  } catch (std::exception const& e) {
    last_error = e.what();
    return nullptr;
  }
}
extern "C" void iv_destroy(void* handle) { delete static_cast<Experiment*>(handle); }

template <typename T>
void Compute(Experiment const& e,
             T const* x,
             std::size_t rows,
             std::size_t cols,
             T const* r,
             std::size_t refs,
             double* out,
             cudaStream_t stream)
{
  auto size = rows * e.groups * (cols + 1);
  RAFT_CUDA_TRY(cudaMemsetAsync(out, 0, size * sizeof(double), stream));
  if (e.backend == 0) {
    std::visit(
      [&](auto const& info) {
        gpu_treeshap::GPUTreeShapInterventional(DenseDatasetWrapper<T>(x, rows, cols),
                                                DenseDatasetWrapper<T>(r, refs, cols),
                                                info->path_segments.begin(),
                                                info->path_segments.end(),
                                                e.groups,
                                                thrust::device_pointer_cast(out),
                                                thrust::device_pointer_cast(out) + size);
      },
      e.legacy);
  } else if (e.backend == 3) {
    std::visit(
      [&](auto const& cache) {
        gpu_treeshap::detail::ComputeShapInterventional(DenseDatasetWrapper<T>(x, rows, cols),
                                                        DenseDatasetWrapper<T>(r, refs, cols),
                                                        cache.bins,
                                                        cache.paths,
                                                        e.groups,
                                                        out);
      },
      e.cached);
  } else {
    iv::Compute(e.model,
                x,
                rows,
                cols,
                r,
                refs,
                e.groups,
                out,
                e.backend == 1   ? iv::Variant::thread_pair
                : e.backend == 4 ? iv::Variant::subwarp_pair
                                 : iv::Variant::warp_pair,
                rmm::cuda_stream_view(stream));
  }
  // Identical postprocessing for all three backends, included in timing.
  double divisor = e.average ? e.trees : 1;
  double bias    = e.bias;
  thrust::for_each(thrust::cuda::par.on(stream),
                   thrust::make_counting_iterator(std::size_t{0}),
                   thrust::make_counting_iterator(size),
                   [=] __device__(std::size_t i) {
                     out[i] = out[i] / divisor + (i % (cols + 1) == cols ? bias : 0);
                   });
}
extern "C" int iv_compute(void* handle,
                          void const* x,
                          std::size_t rows,
                          std::size_t cols,
                          void const* r,
                          std::size_t refs,
                          double* out,
                          int dtype,
                          void* stream)
{
  try {
    auto const& e = *static_cast<Experiment*>(handle);
    RAFT_EXPECTS((e.backend != 0 && e.backend != 3) || stream == nullptr,
                 "Legacy backends require the default stream");
    if (dtype == 4) {
      Compute(e,
              static_cast<float const*>(x),
              rows,
              cols,
              static_cast<float const*>(r),
              refs,
              out,
              static_cast<cudaStream_t>(stream));
    } else {
      Compute(e,
              static_cast<double const*>(x),
              rows,
              cols,
              static_cast<double const*>(r),
              refs,
              out,
              static_cast<cudaStream_t>(stream));
    }
    RAFT_CUDA_TRY(cudaGetLastError());
    return 0;
  } catch (std::exception const& e) {
    last_error = e.what();
    return -1;
  }
}
