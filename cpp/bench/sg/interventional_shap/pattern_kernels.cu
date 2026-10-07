/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Prefix supplies Node, decision(), invchoose(), and generated constants.
extern "C" __global__ void patterns(const Node* nodes,
                                    const unsigned* cats,
                                    const int* path,
                                    const int* slot,
                                    const int* lengths,
                                    const int* widths,
                                    int nl,
                                    int depth,
                                    const DATA* x,
                                    int nx,
                                    int nf,
                                    unsigned long long* out)
{
  for (long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x; task < (long long)nl * nx;
       task += (long long)gridDim.x * blockDim.x) {
    int leaf = task / nx, row = task % nx, k = widths[leaf];
    unsigned long long mask = k == 64 ? ~0ull : ((1ull << k) - 1);
    for (int j = 0; j < lengths[leaf]; ++j) {
      int edge = path[leaf * depth + j], nid = edge >= 0 ? edge : -edge - 1;
      auto n  = nodes[nid];
      bool ok = decision(n, x[(long long)row * nf + n.feature], cats) == (edge >= 0);
      if (!ok) mask &= ~(1ull << slot[leaf * depth + j]);
    }
    out[task] = mask;
  }
}
// Sorted row patterns -> unique keys and frequencies. Unused slots have zero count.
extern "C" __global__ void compress(const unsigned long long* sorted,
                                    int nl,
                                    int nr,
                                    unsigned long long* keys,
                                    int* counts,
                                    int* sizes)
{
  for (int leaf = blockIdx.x * blockDim.x + threadIdx.x; leaf < nl;
       leaf += gridDim.x * blockDim.x) {
    int size = 0;
    for (int j = 0; j < nr; ++j) {
      auto v = sorted[(long long)leaf * nr + j];
      if (j == 0 || v != sorted[(long long)leaf * nr + j - 1]) {
        keys[(long long)leaf * nr + size]   = v;
        counts[(long long)leaf * nr + size] = 1;
        ++size;
      } else
        ++counts[(long long)leaf * nr + size - 1];
    }
    sizes[leaf] = size;
  }
}
extern "C" __global__ void sparse_values(const unsigned long long* x,
                                         int nx,
                                         const int* xsizes,
                                         const unsigned long long* r,
                                         int nr,
                                         const int* counts,
                                         const int* rsizes,
                                         const int* widths,
                                         const double* leaves,
                                         int nl,
                                         int depth,
                                         double* values,
                                         int references)
{
  for (long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x; task < (long long)nl * nx;
       task += (long long)gridDim.x * blockDim.x) {
    int leaf = task / nx, xi = task % nx, k = widths[leaf];
    if (xi >= xsizes[leaf]) continue;
    double sum[65];
    for (int j = 0; j <= k; ++j)
      sum[j] = 0;
    auto u   = x[task];
    auto all = k == 64 ? ~0ull : ((1ull << k) - 1);
    for (int ri = 0; ri < rsizes[leaf]; ++ri) {
      auto v = r[(long long)leaf * nr + ri];
      if ((u | v) != all) continue;
      auto positive = u & ~v, negative = v & ~u;
      int a = __popcll(positive), b = __popcll(negative);
      double base = leaves[leaf] * counts[(long long)leaf * nr + ri] / references;
      double z = base * invchoose(a, b), p = a ? z / a : 0, n = b ? -z / b : 0;
      for (int j = 0; j < k; ++j)
        sum[j] += ((positive >> j) & 1) ? p : ((negative >> j) & 1) ? n : 0;
      if (a == 0) sum[k] += base;
    }
    for (int j = 0; j <= k; ++j)
      values[task * (depth + 1) + j] = sum[j];
  }
}
extern "C" __global__ void scatter_sparse(const unsigned long long* x,
                                          int nx,
                                          const unsigned long long* keys,
                                          const int* sizes,
                                          const double* values,
                                          const int* features,
                                          const int* widths,
                                          const int* groups,
                                          int nl,
                                          int depth,
                                          int nf,
                                          int ng,
                                          double* out,
                                          int grouped)
{
  for (long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x; task < (long long)nl * nx;
       task += (long long)gridDim.x * blockDim.x) {
    int leaf = task / nx, row = task % nx, k = widths[leaf], index = row;
    if (grouped) {
      auto u = x[task];
      int lo = 0, hi = sizes[leaf];
      while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (keys[(long long)leaf * nx + mid] < u)
          lo = mid + 1;
        else
          hi = mid;
      }
      index = lo;
    }
    const double* v = values + ((long long)leaf * nx + index) * (depth + 1);
    double* phi     = out + ((long long)row * ng + groups[leaf]) * (nf + 1);
    for (int j = 0; j < k; ++j)
      if (v[j]) atomicAdd(phi + features[leaf * depth + j], v[j]);
    if (v[k]) atomicAdd(phi + nf, v[k]);
  }
}
extern "C" __global__ void histogram(
  const unsigned long long* r, int nr, int nl, int size, double* f)
{
  for (long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x; task < (long long)nl * nr;
       task += (long long)gridDim.x * blockDim.x)
    atomicAdd(f + (task / nr) * size + r[task], 1.0 / nr);
}
// Superset zeta and subset zeta. Each stage has disjoint source/destination entries.
extern "C" __global__ void zeta(double* f, long long total, int size, int bit, int subset)
{
  for (long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x; task < total;
       task += (long long)gridDim.x * blockDim.x) {
    int mask = task % size;
    if (subset ? ((mask & bit) != 0) : ((mask & bit) == 0))
      f[task] += f[subset ? task - bit : task + bit];
  }
}
extern "C" __global__ void diagonal(
  const double* f, const double* leaf_values, int nl, int k, int size, double* s)
{
  for (long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       task < (long long)nl * k * size;
       task += (long long)gridDim.x * blockDim.x) {
    int u = task % size, j = (task / size) % k, leaf = task / size / k;
    int a = __popc(u), b = k - a;
    double z      = invchoose(a, b);
    double weight = (u & (1u << j)) ? z / a : -z / b;
    s[task]       = leaf_values[leaf] * weight * f[(long long)leaf * size + (size - 1 - u)];
  }
}
extern "C" __global__ void scatter_dense(const unsigned long long* x,
                                         int nx,
                                         const double* s,
                                         const double* f,
                                         const double* leaf_values,
                                         const int* features,
                                         const int* groups,
                                         int nl,
                                         int k,
                                         int depth,
                                         int size,
                                         int nf,
                                         int ng,
                                         double* out)
{
  for (long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x; task < (long long)nl * nx;
       task += (long long)gridDim.x * blockDim.x) {
    int leaf = task / nx, row = task % nx;
    auto u      = x[task];
    double* phi = out + ((long long)row * ng + groups[leaf]) * (nf + 1);
    for (int j = 0; j < k; ++j) {
      double v = s[((long long)leaf * k + j) * size + u];
      if (v) atomicAdd(phi + features[leaf * depth + j], v);
    }
    double bias = leaf_values[leaf] * f[(long long)leaf * size + size - 1];
    if (bias) atomicAdd(phi + nf, bias);
  }
}
