/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <algorithm>
#include <common/grid_sync.cuh>
#include <cub/cub.cuh>
#include <limits>
#include <raft/cuda_utils.cuh>
#include <vector>
#include "input.cuh"
#include "node.cuh"
#include "split.cuh"

namespace ML {
namespace DecisionTree {

struct IntBin {
  int x;

  DI static void IncrementHistogram(IntBin* hist, int nbins, int b, int label) {
    auto offset = label * nbins + b;
    IntBin::AtomicAdd(hist + offset, {1});
  }
  DI static void AtomicAdd(IntBin* address, IntBin val) {
    atomicAdd(&address->x, val.x);
  }
  DI IntBin& operator+=(const IntBin& b) {
    x += b.x;
    return *this;
  }
  DI IntBin operator+(IntBin b) const {
    b += *this;
    return b;
  }
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class GiniObjectiveFunction {
 public:
  using DataT = DataT_;
  using LabelT = LabelT_;
  using IdxT = IdxT_;
  IdxT nclasses;
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  using BinT = IntBin;
  GiniObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease,
                        IdxT min_samples_leaf)
    : nclasses(nclasses),
      min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}

  DI IdxT NumClasses() const { return nclasses; }
  DI Split<DataT, IdxT> Gain(BinT* scdf_labels, DataT* sbins, IdxT col,
                             IdxT len, IdxT nbins) {
    Split<DataT, IdxT> sp;
    constexpr DataT One = DataT(1.0);
    DataT invlen = One / len;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      int nLeft = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeft += scdf_labels[nbins * j + i].x;
      }
      auto nRight = len - nLeft;
      auto gain = DataT(0.0);
      // if there aren't enough samples in this split, don't bother!
      if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
        gain = -std::numeric_limits<DataT>::max();
      } else {
        auto invLeft = One / nLeft;
        auto invRight = One / nRight;
        for (IdxT j = 0; j < nclasses; ++j) {
          int val_i = 0;
          auto lval_i = scdf_labels[nbins * j + i].x;
          auto lval = DataT(lval_i);
          gain += lval * invLeft * lval * invlen;

          val_i += lval_i;
          auto total_sum = scdf_labels[nbins * j + nbins - 1].x;
          auto rval_i = total_sum - lval_i;
          auto rval = DataT(rval_i);
          gain += rval * invRight * rval * invlen;

          val_i += rval_i;
          auto val = DataT(val_i) * invlen;
          gain -= val * val;
        }
      }
      // if the gain is not "enough", don't bother!
      if (gain <= min_impurity_decrease) {
        gain = -std::numeric_limits<DataT>::max();
      }
      sp.update({sbins[i], col, gain, nLeft});
    }
    return sp;
  }
  static DI LabelT LeafPrediction(BinT* shist, int nclasses) {
    int class_idx = 0;
    int count = 0;
    for (int i = 0; i < nclasses; i++) {
      auto current_count = shist[i].x;
      if (current_count > count) {
        class_idx = i;
        count = current_count;
      }
    }
    return class_idx;
  }
  template <class... Types>
  static void PostprocessTree(Types... args) {}
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class EntropyObjectiveFunction {
 public:
  using DataT = DataT_;
  using LabelT = LabelT_;
  using IdxT = IdxT_;
  IdxT nclasses;
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  using BinT = IntBin;
  EntropyObjectiveFunction(DataT nclasses, IdxT min_impurity_decrease,
                           IdxT min_samples_leaf)
    : nclasses(nclasses),
      min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}
  DI IdxT NumClasses() const { return nclasses; }
  DI Split<DataT, IdxT> Gain(BinT* scdf_labels, DataT* sbins, IdxT col,
                             IdxT len, IdxT nbins) {
    Split<DataT, IdxT> sp;
    constexpr DataT One = DataT(1.0);
    DataT invlen = One / len;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      int nLeft = 0;
      for (IdxT j = 0; j < nclasses; ++j) {
        nLeft += scdf_labels[nbins * j + i].x;
      }
      auto nRight = len - nLeft;
      auto gain = DataT(0.0);
      // if there aren't enough samples in this split, don't bother!
      if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
        gain = -std::numeric_limits<DataT>::max();
      } else {
        auto invLeft = One / nLeft;
        auto invRight = One / nRight;
        for (IdxT j = 0; j < nclasses; ++j) {
          int val_i = 0;
          auto lval_i = scdf_labels[nbins * j + i].x;
          if (lval_i != 0) {
            auto lval = DataT(lval_i);
            gain += raft::myLog(lval * invLeft) / raft::myLog(DataT(2)) * lval *
                    invlen;
          }

          val_i += lval_i;
          auto total_sum = scdf_labels[2 * nbins * j + nbins - 1].x;
          auto rval_i = total_sum - lval_i;
          if (rval_i != 0) {
            auto rval = DataT(rval_i);
            gain += raft::myLog(rval * invRight) / raft::myLog(DataT(2)) *
                    rval * invlen;
          }

          val_i += rval_i;
          if (val_i != 0) {
            auto val = DataT(val_i) * invlen;
            gain -= val * raft::myLog(val) / raft::myLog(DataT(2));
          }
        }
      }
      // if the gain is not "enough", don't bother!
      if (gain <= min_impurity_decrease) {
        gain = -std::numeric_limits<DataT>::max();
      }
      sp.update({sbins[i], col, gain, nLeft});
    }
    return sp;
  }
  static DI LabelT LeafPrediction(BinT* shist, int nclasses) {
    // Same as Gini
    return GiniObjectiveFunction<DataT, LabelT, IdxT>::LeafPrediction(shist,
                                                                      nclasses);
  }
  template <class... Types>
  static void PostprocessTree(Types... args) {}
};

template <typename DataT_, typename LabelT_, typename IdxT_>
class MSEObjectiveFunction {
 public:
  using DataT = DataT_;
  using LabelT = LabelT_;
  using IdxT = IdxT_;

 private:
  DataT min_impurity_decrease;
  IdxT min_samples_leaf;

 public:
  struct MSEBin {
    double label_sum;
    int count;

    DI static void IncrementHistogram(MSEBin* hist, int nbins, int b,
                                      double label) {
      MSEBin::AtomicAdd(hist + b, {label, 1});
    }
    DI static void AtomicAdd(MSEBin* address, MSEBin val) {
      atomicAdd(&address->label_sum, val.label_sum);
      atomicAdd(&address->count, val.count);
    }
    DI MSEBin& operator+=(const MSEBin& b) {
      label_sum += b.label_sum;
      count += b.count;
      return *this;
    }
    DI MSEBin operator+(MSEBin b) const {
      b += *this;
      return b;
    }
  };
  using BinT = MSEBin;
  HDI MSEObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease,
                           IdxT min_samples_leaf)
    : min_impurity_decrease(min_impurity_decrease),
      min_samples_leaf(min_samples_leaf) {}
  DI IdxT NumClasses() const { return 1; }
  DI Split<DataT, IdxT> Gain(BinT* shist, DataT* sbins, IdxT col, IdxT len,
                             IdxT nbins) {
    Split<DataT, IdxT> sp;
    auto invlen = DataT(1.0) / len;
    for (IdxT i = threadIdx.x; i < nbins; i += blockDim.x) {
      auto nLeft = shist[i].count;
      auto nRight = len - nLeft;
      DataT gain;
      // if there aren't enough samples in this split, don't bother!
      if (nLeft < min_samples_leaf || nRight < min_samples_leaf) {
        gain = -std::numeric_limits<DataT>::max();
      } else {
        auto label_sum = shist[nbins - 1].label_sum;
        DataT parent_obj = -label_sum * label_sum / len;
        DataT left_obj = -(shist[i].label_sum * shist[i].label_sum) / nLeft;
        DataT right_label_sum = shist[i].label_sum - label_sum;
        DataT right_obj = -(right_label_sum * right_label_sum) / nRight;
        gain = parent_obj - (left_obj + right_obj);
        gain *= invlen;
      }
      // if the gain is not "enough", don't bother!
      if (gain <= min_impurity_decrease) {
        gain = -std::numeric_limits<DataT>::max();
      }
      sp.update({sbins[i], col, gain, nLeft});
    }
    return sp;
  }

  static DI LabelT LeafPrediction(BinT* shist, int nclasses) {
    return shist[0].label_sum / shist[0].count;
  }
  template <class... Types>
  static void PostprocessTree(Types... args) {}
};

template <typename T>
void print(T x) {
  for (auto x_i : x) {
    std::cout << x_i << " ";
  }
  std::cout << std::endl;
}

// Use MSE split gain, postprocess tree
template <typename DataT_, typename LabelT_, typename IdxT_>
class MAEObjectiveFunction
  : public MSEObjectiveFunction<DataT_, LabelT_, IdxT_> {
 public:
  using DataT = DataT_;
  using LabelT = LabelT_;
  using IdxT = IdxT_;

  DI MAEObjectiveFunction(IdxT nclasses, DataT min_impurity_decrease,
                          IdxT min_samples_leaf)
    : MSEObjectiveFunction<DataT, LabelT, IdxT>(nclasses, min_impurity_decrease,
                                                min_samples_leaf) {}
  static void PostprocessTree(std::vector<Node<DataT, LabelT, IdxT>>& h_nodes,
                              const Input<DataT, LabelT, IdxT> input) {
    // Get the leaf nodes
    using NodeTuple = thrust::tuple<Node<DataT, LabelT, IdxT>, int>;
    std::vector<NodeTuple> h_leaves;
    h_leaves.reserve(h_nodes.size());
    for (auto i = 0; i < h_nodes.size(); i++) {
      auto n = h_nodes[i];
      if (n.isLeaf()) {
        h_leaves.emplace_back(NodeTuple{n, i});
      }
    }
    thrust::device_vector<NodeTuple> leaves(h_leaves);
    thrust::sort(leaves.begin(), leaves.end(),
                 [] __device__(const NodeTuple& a, const NodeTuple& b) -> bool {
                   return thrust::get<0>(a).start < thrust::get<0>(b).start;
                 });
    thrust::device_vector<LabelT> labels(input.nSampledRows);
    thrust::transform(
      thrust::device, input.rowids, input.rowids + input.nSampledRows,
      labels.begin(),
      [=] __device__(IdxT rowid) { return input.labels[rowid]; });
    thrust::device_vector<IdxT> node_id(input.nSampledRows);
    auto counting = thrust::make_counting_iterator(0ll);
    auto leaf_boundaries = thrust::make_transform_iterator(
      leaves.begin(),
      [=] __device__(const NodeTuple& n) { return thrust::get<0>(n).start; });
    size_t n_leaves = leaves.size();
    // Binary search to find leaf each row belongs to
    thrust::transform(thrust::device, counting, counting + input.nSampledRows,
                      node_id.begin(), [=] __device__(IdxT idx) {
                        return thrust::upper_bound(thrust::seq, leaf_boundaries,
                                                   leaf_boundaries + n_leaves,
                                                   idx) -
                               leaf_boundaries - 1;
                      });
    auto zip = thrust::make_zip_iterator(
      thrust::make_tuple(labels.begin(), node_id.begin()));
    thrust::sort(zip, zip + labels.size(),
                 [] __device__(const thrust::tuple<LabelT, IdxT>& a,
                               const thrust::tuple<LabelT, IdxT>& b) {
                   if (thrust::get<1>(a) < thrust::get<1>(b)) return true;
                   if (thrust::get<1>(b) < thrust::get<1>(a)) return false;
                   if (thrust::get<0>(a) < thrust::get<0>(b)) return true;
                   return false;
                 });
    // Grab the median element for each leaf
    auto d_leaves = leaves.data().get();
    auto d_labels = labels.data().get();
    thrust::for_each_n(counting, n_leaves, [=] __device__(size_t idx) {
      auto& n = thrust::get<0>(d_leaves[idx]);
      // If number of elements is odd, get the middle element
      // If even, take the midpoint
      size_t median_idx_high = n.start + n.count / 2;
      size_t median_idx_low = n.start + std::max(n.count - 1, IdxT(0)) / 2;
      n.info.prediction =
        (d_labels[median_idx_low] + d_labels[median_idx_high]) / 2;
    });
    thrust::copy(leaves.begin(), leaves.end(), h_leaves.begin());
    for (const auto& nt : h_leaves) {
      auto idx = thrust::get<1>(nt);
      auto pred = thrust::get<0>(nt).info.prediction;
      h_nodes[idx].info.prediction = pred;
    }
  }
};

}  // namespace DecisionTree
}  // namespace ML
