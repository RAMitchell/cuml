/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuml/common/checked_arithmetic.hpp>

#include <raft/core/error.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/device_vector.h>

#include <treelite/tree.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace ML::Explainer::interventional {

// A compact tree representation, independent of GPUTreeShap and training cover.
struct Node {
  int left{-1}, right{-1}, feature{-1};
  double value{};  // threshold at a split, prediction at a leaf
  std::size_t category_begin{}, category_end{};
  bool default_left{}, less_equal{}, categorical{}, categories_go_right{};
};
struct Tree {
  std::size_t begin{};
  int group{};
};
struct Model {
  thrust::device_vector<Node> nodes;
  thrust::device_vector<Tree> trees;
  thrust::device_vector<std::uint32_t> categories;
  int max_depth{};
};

template <typename ThresholdT, typename LeafT>
Model Prepare(treelite::Model const& model, treelite::ModelPreset<ThresholdT, LeafT> const& preset)
{
  RAFT_EXPECTS(model.num_target == 1, "Interventional SHAP requires a single target");
  std::vector<Node> nodes;
  std::vector<Tree> trees;
  std::vector<std::uint32_t> categories;
  int max_depth    = 0;
  bool vector_leaf = model.leaf_vector_shape[1] > 1;
  int groups       = model.num_class[0];
  RAFT_EXPECTS(groups > 0, "Invalid output group count");
  for (std::size_t ti = 0; ti < preset.trees.size(); ++ti) {
    auto const& tree = preset.trees[ti];
    std::vector<std::pair<int, int>> pending{{0, 0}};
    while (!pending.empty()) {
      auto [nid, depth] = pending.back();
      pending.pop_back();
      max_depth = std::max(max_depth, depth);
      if (!tree.IsLeaf(nid)) {
        pending.emplace_back(tree.LeftChild(nid), depth + 1);
        pending.emplace_back(tree.RightChild(nid), depth + 1);
      }
    }
    for (int group = 0; group < (vector_leaf ? groups : 1); ++group) {
      trees.push_back({nodes.size(), vector_leaf ? group : static_cast<int>(ti % groups)});
      for (int nid = 0; nid < tree.num_nodes; ++nid) {
        Node n;
        if (tree.IsLeaf(nid)) {
          if (vector_leaf) {
            auto leaf = tree.LeafVector(nid);
            RAFT_EXPECTS(leaf.size() == static_cast<std::size_t>(groups), "Invalid leaf shape");
            n.value = leaf[group];
          } else {
            n.value = tree.LeafValue(nid);
          }
        } else {
          n.left         = tree.LeftChild(nid);
          n.right        = tree.RightChild(nid);
          n.feature      = tree.SplitIndex(nid);
          n.default_left = tree.DefaultChild(nid) == n.left;
          n.categorical  = tree.NodeType(nid) == treelite::TreeNodeType::kCategoricalTestNode;
          if (n.categorical) {
            auto cats = tree.CategoryList(nid);
            std::sort(cats.begin(), cats.end());
            n.category_begin = categories.size();
            categories.insert(categories.end(), cats.begin(), cats.end());
            n.category_end        = categories.size();
            n.categories_go_right = tree.CategoryListRightChild(nid);
          } else {
            RAFT_EXPECTS(tree.ComparisonOp(nid) == treelite::Operator::kLT ||
                           tree.ComparisonOp(nid) == treelite::Operator::kLE,
                         "Unsupported split comparison");
            n.less_equal = tree.ComparisonOp(nid) == treelite::Operator::kLE;
            n.value      = tree.Threshold(nid);
          }
        }
        nodes.push_back(n);
      }
    }
  }
  return {thrust::device_vector<Node>(nodes),
          thrust::device_vector<Tree>(trees),
          thrust::device_vector<std::uint32_t>(categories),
          max_depth};
}

template <typename DataT>
__device__ bool GoesLeft(Node const& node, DataT value, std::uint32_t const* categories)
{
  if (isnan(value)) { return node.default_left; }
  if (!node.categorical) { return node.less_equal ? value <= node.value : value < node.value; }
  bool member = false;
  // Guard conversion before casting, including infinity and negative category codes.
  if (value >= 0 && static_cast<double>(value) <= std::numeric_limits<std::uint32_t>::max()) {
    auto cat = static_cast<std::uint32_t>(value);
    auto lo  = node.category_begin;
    auto hi  = node.category_end;
    while (lo < hi) {
      auto mid = lo + (hi - lo) / 2;
      if (categories[mid] < cat) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    member = lo < node.category_end && categories[lo] == cat;
  }
  return member != node.categories_go_right;
}

// For a leaf requiring a foreground-only and b background-only feature choices,
// its coalition polynomial is t^a (1-t)^b. Integrating its derivative yields
// +1/[a*C(a+b,a)] or -1/[b*C(a+b,a)]. Features with identical decisions are dummies.
__device__ inline double InverseBinomial(int a, int b)
{
  int small = min(a, b), large = max(a, b);
  double result = 1;
  for (int i = 1; i <= small; ++i) {
    result *= static_cast<double>(i) / (large + i);
  }
  return result;
}

template <int Depth>
struct State {
  int features[Depth];
  int background_child[Depth];
  bool foreground[Depth];
};

template <int Width>
__device__ void Sync()
{
  if constexpr (Width > 1) {
    constexpr unsigned members = 0xffffffffu >> (32 - Width);
    auto shift                 = (threadIdx.x % 32 / Width) * Width;
    __syncwarp(members << shift);
  }
}

template <int Width, int Depth, typename DataT>
__device__ void Traverse(Node const* nodes,
                         std::uint32_t const* categories,
                         DataT const* x,
                         DataT const* r,
                         std::size_t cols,
                         double scale,
                         double* phi,
                         State<Depth>& state,
                         int lane)
{
  int nid = 0, size = 0, positive = 0, negative = 0;
  while (true) {
    auto node = nodes[nid];
    if (node.feature >= 0) {
      bool x_left = GoesLeft(node, x[node.feature], categories);
      bool r_left = GoesLeft(node, r[node.feature], categories);
      if (x_left == r_left) {
        nid = x_left ? node.left : node.right;
        continue;
      }
      int previous = -1;
      for (int i = 0; i < size; ++i) {
        if (state.features[i] == node.feature) {
          previous = i;
          break;
        }
      }
      if (previous >= 0) {
        bool left = state.foreground[previous] ? x_left : r_left;
        nid       = left ? node.left : node.right;
        continue;
      }
      // Only distinct features on which x and r disagree need traversal state.
      if (lane == 0) {
        state.features[size]         = node.feature;
        state.background_child[size] = r_left ? node.left : node.right;
        state.foreground[size]       = true;
      }
      Sync<Width>();
      ++size;
      ++positive;
      nid = x_left ? node.left : node.right;
      continue;
    }

    double weight = node.value * scale * InverseBinomial(positive, negative);
    for (int i = lane; i < size; i += Width) {
      double contribution = state.foreground[i] ? weight / positive : -weight / negative;
      if (contribution != 0) { atomicAdd(phi + state.features[i], contribution); }
    }
    if (lane == 0 && positive == 0) { atomicAdd(phi + cols, node.value * scale); }
    Sync<Width>();

    // Finish the foreground subtree before visiting its background sibling.
    while (size > 0 && !state.foreground[size - 1]) {
      --size;
      --negative;
    }
    if (size == 0) { break; }
    nid = state.background_child[size - 1];
    Sync<Width>();
    if (lane == 0) { state.foreground[size - 1] = false; }
    Sync<Width>();
    --positive;
    ++negative;
  }
}

enum class Variant { thread_pair, warp_pair, subwarp_pair };

template <int Width, int Depth, typename DataT>
__global__ void PairKernel(Node const* nodes,
                           Tree const* trees,
                           std::uint32_t const* categories,
                           std::size_t tree_count,
                           DataT const* x,
                           std::size_t rows,
                           std::size_t cols,
                           DataT const* background,
                           std::size_t background_rows,
                           std::size_t groups,
                           double* phi)
{
  constexpr int threads = 128;
  // Threads have private stacks; warps cooperate on one shared stack.
  __shared__ State<Depth> shared[Width > 1 ? threads / Width : 1];
  State<Width == 1 ? Depth : 1> local;
  int lane    = threadIdx.x % Width;
  auto task   = (static_cast<std::size_t>(blockIdx.x) * threads + threadIdx.x) / Width;
  auto stride = static_cast<std::size_t>(gridDim.x) * threads / Width;
  auto tasks  = tree_count * rows * background_rows;
  for (; task < tasks; task += stride) {
    auto ri   = task % background_rows;
    auto row  = (task / background_rows) % rows;
    auto tree = trees[task / background_rows / rows];
    auto out  = phi + (row * groups + tree.group) * (cols + 1);
    if constexpr (Width == 1) {
      Traverse<Width>(nodes + tree.begin,
                      categories,
                      x + row * cols,
                      background + ri * cols,
                      cols,
                      1.0 / background_rows,
                      out,
                      local,
                      lane);
    } else {
      Traverse<Width>(nodes + tree.begin,
                      categories,
                      x + row * cols,
                      background + ri * cols,
                      cols,
                      1.0 / background_rows,
                      out,
                      shared[threadIdx.x / Width],
                      lane);
    }
    Sync<Width>();
  }
}

template <int Depth, typename DataT>
void Launch(Model const& model,
            DataT const* x,
            std::size_t rows,
            std::size_t cols,
            DataT const* background,
            std::size_t background_rows,
            std::size_t groups,
            double* phi,
            Variant variant,
            rmm::cuda_stream_view stream)
{
  auto tasks  = ML::checked_mul<std::size_t>(ML::checked_mul<std::size_t>(model.trees.size(), rows),
                                            background_rows);
  auto width  = variant == Variant::thread_pair ? 1 : variant == Variant::subwarp_pair ? 8 : 32;
  auto blocks = static_cast<unsigned>(
    std::min<std::size_t>(tasks / (128 / width) + (tasks % (128 / width) != 0), 65535));
  auto nodes  = thrust::raw_pointer_cast(model.nodes.data());
  auto trees  = thrust::raw_pointer_cast(model.trees.data());
  auto cats   = thrust::raw_pointer_cast(model.categories.data());
  auto launch = [&]<int Width>() {
    PairKernel<Width, Depth><<<blocks, 128, 0, stream.value()>>>(nodes,
                                                                 trees,
                                                                 cats,
                                                                 model.trees.size(),
                                                                 x,
                                                                 rows,
                                                                 cols,
                                                                 background,
                                                                 background_rows,
                                                                 groups,
                                                                 phi);
  };
  if (width == 1) {
    launch.template operator()<1>();
  } else if (width == 8) {
    launch.template operator()<8>();
  } else {
    launch.template operator()<32>();
  }
  RAFT_CUDA_TRY(cudaGetLastError());
}

// Output must be zeroed on the same stream before calling. Retains double accumulation
// for either input dtype, and leaves model averaging/base scores to the caller.
template <typename DataT>
void Compute(Model const& model,
             DataT const* x,
             std::size_t rows,
             std::size_t cols,
             DataT const* background,
             std::size_t background_rows,
             std::size_t groups,
             double* phi,
             Variant variant,
             rmm::cuda_stream_view stream)
{
  RAFT_EXPECTS(background_rows > 0, "Interventional SHAP requires nonempty background data");
  RAFT_EXPECTS(model.max_depth <= 128, "Interventional traversal currently supports depth <= 128");
  RAFT_EXPECTS(groups > 0, "Invalid output group count");
  // Validate products used by the device's indexing before launching.
  ML::checked_mul<std::size_t>(rows, cols);
  ML::checked_mul<std::size_t>(background_rows, cols);
  ML::checked_mul<std::size_t>(ML::checked_mul<std::size_t>(rows, groups),
                               ML::checked_add<std::size_t>(cols, 1));
  if (rows == 0 || model.trees.empty()) { return; }
  if (model.max_depth <= 8) {
    Launch<8>(model, x, rows, cols, background, background_rows, groups, phi, variant, stream);
  } else if (model.max_depth <= 16) {
    Launch<16>(model, x, rows, cols, background, background_rows, groups, phi, variant, stream);
  } else if (model.max_depth <= 32) {
    Launch<32>(model, x, rows, cols, background, background_rows, groups, phi, variant, stream);
  } else if (model.max_depth <= 64) {
    Launch<64>(model, x, rows, cols, background, background_rows, groups, phi, variant, stream);
  } else {
    Launch<128>(model, x, rows, cols, background, background_rows, groups, phi, variant, stream);
  }
}
}  // namespace ML::Explainer::interventional
