/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
// Standalone NVRTC experiments; all dimensions and storage are checked by Python.
struct Node {
  int left, right, feature;
  double value;
  unsigned long long begin, end;
  bool missing, le, categorical, right_categories;
};
// Generated immutable coefficient/quadrature arrays are prepended by the runner.
__device__ bool decision(Node n, double v, const unsigned* cats)
{
  if (isnan(v)) return n.missing;
  if (!n.categorical) return n.le ? v <= n.value : v < n.value;
  bool member = false;
  if (v >= 0 && v <= 4294967295.0) {
    unsigned c = (unsigned)v;
    auto lo = n.begin, hi = n.end;
    while (lo < hi) {
      auto mid = lo + (hi - lo) / 2;
      if (cats[mid] < c)
        lo = mid + 1;
      else
        hi = mid;
    }
    member = lo < n.end && cats[lo] == c;
  }
  return member != n.right_categories;
}
__device__ double invchoose(int a, int b)
{
  double v  = 1;
  int small = min(a, b), large = max(a, b);
  for (int j = 1; j <= small; ++j)
    v *= double(j) / (large + j);
  return v;
}
__device__ int coeffindex(int a, int b)
{
  int n = a + b;
  return (n + 1) * (n + 1) / 4 + min(a, b);
}
extern "C" __global__ void pack_decisions(const Node* nodes,
                                          long long nn,
                                          const unsigned* cats,
                                          const DATA* x,
                                          long long nr,
                                          int nf,
                                          unsigned* out)
{
  long long warp  = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  int lane        = threadIdx.x % 32;
  long long words = (nr + 31) / 32;
  long long total = nn * words;
  for (; warp < total; warp += (long long)gridDim.x * blockDim.x / 32) {
    long long nid = warp / words, row = (warp % words) * 32 + lane;
    auto n        = nodes[nid];
    bool bit      = row < nr && n.feature >= 0 && decision(n, x[row * nf + n.feature], cats);
    unsigned bits = __ballot_sync(0xffffffff, bit);
    if (lane == 0) out[warp] = bits;
  }
}
__device__ bool cached(const unsigned* p, long long nid, long long row, long long nr)
{
  return (p[nid * ((nr + 31) / 32) + row / 32] >> (row % 32)) & 1;
}
__device__ bool binned(Node n,
                       int nid,
                       const DATA* data,
                       const unsigned* bins,
                       const unsigned* ranks,
                       long long row,
                       int nf,
                       const unsigned* cats)
{
  if (n.categorical) return decision(n, data[row * nf + n.feature], cats);
  unsigned code = bins[(row * nf + n.feature) * 2 + (n.le ? 0 : 1)];
  return code == 0xffffffff ? n.missing : code <= ranks[nid];
}
extern "C" __global__ void pairs(const Node* nodes,
                                 const long long* roots,
                                 const int* groups,
                                 int nt,
                                 const unsigned* cats,
                                 const DATA* x,
                                 long long nx,
                                 int nf,
                                 const DATA* r,
                                 long long nr,
                                 int ng,
                                 double* out,
                                 const double* table,
                                 const double* recip,
                                 const unsigned* px,
                                 const unsigned* pr,
                                 const unsigned* bx,
                                 const unsigned* br,
                                 const unsigned* ranks,
                                 unsigned long long* counters)
{
  __shared__ double shared_table[TABLE_SIZE + DEPTH + 1];
#if WEIGHTS == 2
  for (int i = threadIdx.x; i < TABLE_SIZE; i += blockDim.x)
    shared_table[i] = table[i];
  for (int i = threadIdx.x; i <= DEPTH; i += blockDim.x)
    shared_table[TABLE_SIZE + i] = recip[i];
  __syncthreads();
#endif
  long long task  = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  long long total = (long long)nt * nx * nr;
  for (; task < total; task += (long long)gridDim.x * blockDim.x) {
    long long ri = task % nr, row = (task / nr) % nx;
    int ti      = task / nr / nx;
    double* phi = out + (row * ng + groups[ti]) * (nf + 1);
    int fs[DEPTH], siblings[DEPTH];
    bool fg[DEPTH];
#if WEIGHTS == 4
    double saved[DEPTH];
    double running = 1;
#endif
#if REDUCE
    double sums[DEPTH];
#endif
    int size = 0, a = 0, b = 0, nid = roots[ti];
    unsigned long long visits = 0, leaves = 0, updates = 0;
    while (true) {
      auto n = nodes[nid];
      ++visits;
      if (n.feature >= 0) {
        bool xl, rl;
#if CACHE_X
        xl = cached(px, nid, row, nx);
#elif BINS
        xl = binned(n, nid, x, bx, ranks, row, nf, cats);
#else
        xl = decision(n, x[row * nf + n.feature], cats);
#endif
#if CACHE_R
        rl = cached(pr, nid, ri, nr);
#elif BINS
        rl = binned(n, nid, r, br, ranks, ri, nf, cats);
#else
        rl = decision(n, r[ri * nf + n.feature], cats);
#endif
        if (xl == rl) {
          nid = xl ? n.left : n.right;
          continue;
        }
        int prev = -1;
        for (int j = 0; j < size; ++j)
          if (fs[j] == n.feature) {
            prev = j;
            break;
          }
        if (prev >= 0) {
          nid = (fg[prev] ? xl : rl) ? n.left : n.right;
          continue;
        }
        fs[size]       = n.feature;
        siblings[size] = rl ? n.left : n.right;
        fg[size]       = true;
#if WEIGHTS == 4
        saved[size] = running;
        running *= double(a + 1) / (a + b + 1);
#endif
#if REDUCE
        sums[size] = 0;
#endif
        ++size;
        ++a;
        nid = xl ? n.left : n.right;
        continue;
      }
      ++leaves;
      double pos = 0, neg = 0;
#if WEIGHTS == 5
      for (int q = 0; q < NQUAD; ++q) {
        if (a) pos += qw[q] * pow(qt[q], a - 1) * pow(1 - qt[q], b);
        if (b) neg -= qw[q] * pow(qt[q], a) * pow(1 - qt[q], b - 1);
      }
#elif WEIGHTS == 1
      {
        double v = table[coeffindex(a, b)];
        pos      = v * recip[a];
        neg      = -v * recip[b];
      }
#elif WEIGHTS == 2
      {
        double v = shared_table[coeffindex(a, b)];
        pos      = v * shared_table[TABLE_SIZE + a];
        neg      = -v * shared_table[TABLE_SIZE + b];
      }
#elif WEIGHTS == 3
      {
        double v = ctable[coeffindex(a, b)];
        pos      = v * crecip[a];
        neg      = -v * crecip[b];
      }
#elif WEIGHTS == 4
      pos = a ? running / a : 0;
      neg = b ? -running / b : 0;
#else
      {
        double v = invchoose(a, b);
        pos      = a ? v / a : 0;
        neg      = b ? -v / b : 0;
      }
#endif
      double scale = n.value / nr;
      for (int j = 0; j < size; ++j) {
        double v = scale * (fg[j] ? pos : neg);
#if REDUCE
        sums[j] += v;
#else
        if (v != 0) {
          atomicAdd(phi + fs[j], v);
          ++updates;
        }
#endif
      }
      if (a == 0) {
        atomicAdd(phi + nf, scale);
        ++updates;
      }
      while (size > 0 && !fg[size - 1]) {
#if REDUCE
        if (sums[size - 1] != 0) {
          atomicAdd(phi + fs[size - 1], sums[size - 1]);
          ++updates;
        }
#endif
#if WEIGHTS == 4
        running = saved[size - 1];
#endif
        --size;
        --b;
      }
      if (size == 0) break;
      nid          = siblings[size - 1];
      fg[size - 1] = false;
      --a;
      ++b;
#if WEIGHTS == 4
      running = saved[size - 1] * double(b) / (a + b);
#endif
    }
#if PROFILE
    atomicAdd(counters, visits);
    atomicAdd(counters + 1, leaves);
    atomicAdd(counters + 2, updates);
#endif
  }
}
// Each lane owns one quadrature abscissa. All lanes execute the same traversal.
// Return conditional subtree expectations; update a feature on completion of
// its first divergent split using reach_probability * (foreground - background).
extern "C" __global__ void subtree(const Node* nodes,
                                   const long long* roots,
                                   const int* groups,
                                   int nt,
                                   const unsigned* cats,
                                   const DATA* x,
                                   long long nx,
                                   int nf,
                                   const DATA* r,
                                   long long nr,
                                   int ng,
                                   double* out)
{
  int lane             = threadIdx.x % QWIDTH;
  long long task       = ((long long)blockIdx.x * blockDim.x + threadIdx.x) / QWIDTH;
  constexpr int chunks = (NQUAD + QWIDTH - 1) / QWIDTH;
  long long total      = (long long)nt * nx * nr * chunks;
  for (; task < total; task += (long long)gridDim.x * blockDim.x / QWIDTH) {
    int chunk = task % chunks, q = chunk * QWIDTH + lane;
    long long pair = task / chunks;
    long long ri = pair % nr, row = (pair / nr) % nx;
    int ti      = pair / nr / nx;
    double* phi = out + (row * ng + groups[ti]) * (nf + 1);
    int fs[DEPTH], siblings[DEPTH];
    bool fg[DEPTH];
    double reach[DEPTH], first[DEPTH];
    int size = 0, nid = roots[ti];
    double probability = 1, result = 0;
    double t = q < NQUAD ? qt[q] : 0.5, w = q < NQUAD ? qw[q] : 0;
    while (true) {
      auto n = nodes[nid];
      if (n.feature >= 0) {
        bool xl = decision(n, x[row * nf + n.feature], cats),
             rl = decision(n, r[ri * nf + n.feature], cats);
        if (xl == rl) {
          nid = xl ? n.left : n.right;
          continue;
        }
        int prev = -1;
        for (int j = 0; j < size; ++j)
          if (fs[j] == n.feature) {
            prev = j;
            break;
          }
        if (prev >= 0) {
          nid = (fg[prev] ? xl : rl) ? n.left : n.right;
          continue;
        }
        fs[size]       = n.feature;
        siblings[size] = rl ? n.left : n.right;
        fg[size]       = true;
        reach[size]    = probability;
        probability *= t;
        ++size;
        nid = xl ? n.left : n.right;
        continue;
      }
      result = n.value;
      while (size > 0 && !fg[size - 1]) {
        int j    = size - 1;
        double v = w * reach[j] * (first[j] - result) / nr;
        for (int offset = QWIDTH / 2; offset > 0; offset /= 2)
          v += __shfl_down_sync(__activemask(), v, offset, QWIDTH);
        if (lane == 0 && v != 0) atomicAdd(phi + fs[j], v);
        result      = t * first[j] + (1 - t) * result;
        probability = reach[j];
        --size;
      }
      if (size == 0) break;
      first[size - 1] = result;
      fg[size - 1]    = false;
      probability     = reach[size - 1] * (1 - t);
      nid             = siblings[size - 1];
    }
    if (lane == 0 && chunk == 0) {
      nid = roots[ti];
      while (nodes[nid].feature >= 0) {
        auto n = nodes[nid];
        nid    = decision(n, r[ri * nf + n.feature], cats) ? n.left : n.right;
      }
      atomicAdd(phi + nf, nodes[nid].value / nr);
    }
  }
}
extern "C" __global__ void finish(
  double* out, long long size, int nf, const double* bias, int ng, double divisor)
{
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < size;
       i += (long long)gridDim.x * blockDim.x)
    out[i] = out[i] / divisor + (i % (nf + 1) == nf ? bias[(i / (nf + 1)) % ng] : 0);
}
