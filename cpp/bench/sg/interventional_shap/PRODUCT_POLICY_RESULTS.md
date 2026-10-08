# Simplifying the interventional SHAP batch rule

Measured 2026-10-08 on physical GPU 1, NVIDIA RTX PRO 6000 Blackwell. This tests
removing the separate foreground and background minimums from the depth-six
candidate. No kernels or production API were changed.

## Conclusion

**Remove the foreground minimum, but retain the background minimum.** All 24
newly admitted cases (100 or 200 foreground rows, 10,000 background rows) win,
by 2.49–18.64x against improved traversal.

The candidate is now:

```text
D <= 6 and B >= 100 and N * B >= 1,000,000
    -> streamed dense Woodelf-style processing, subject to memory guard/cap
otherwise
    -> lookup plus local-reduction traversal without decision caches
```

The one-leaf memory estimate and hard 64 MiB allocator limit are unchanged.
This remains a candidate, not a zero-regression promise. One newly exercised
shape hits an existing memory-estimation limitation and falls back, adding
latency; the previous policy would also select dense there (details below).

**Product alone does not suffice.** Equal-product batches have different
crossovers. Without any allocation failure, the worst product-only slowdown is
1.41x. Including failed attempts and correct
fallback, the worst slowdown is 5.08x.

## Equal-product example

California Housing, depth six, complete 100-round model. Every row below has
N*B=1,000,000. All dense attempts in this table succeed without allocation
fallback. Timings are wall-time medians and include the product-policy wrapper.

| Foreground N | Background B | Improved traversal (ms) | Woodelf (ms) | Speedup |
|---|---:|---:|---:|---:|
| 100 | 10,000 | 57.34 | 14.35 | 4.00x |
| 1,000 | 1,000 | 54.12 | 14.10 | 3.84x |
| 10,000 | 100 | 55.04 | 14.21 | 3.87x |
| 20,000 | 50 | 56.93 | 18.00 | 3.16x |
| 100,000 | 10 | 60.49 | 85.20 | 0.71x |

The product estimates traversal pair work. Dense processing still has work that
scales with foreground rows and benefits from aggregating many background rows.
Equal products therefore do not imply equal relative costs. This is an
interpretation of the algorithm and timings, not a hardware-counter analysis.

## Policy comparison

All three policies were specified before this run. Each keeps depth<=6, the
one-leaf estimate and the hard 64 MiB cap. The actual product-policy wrapper was
timed; the other two policies are offline selections of those same measured
backend timings (traversal when rejected). Their extra comparison instructions
were not timed as separate wrappers. Geometric means include all 108 cases and
assume exactly baseline time for rejected cases; they are selection estimates.

| Rule | Selects dense | Selected cases faster | Geometric mean speedup | Worst slowdown |
|---|---:|---:|---:|---:|
| Product only | 96 | 83 | 2.92x | 5.08x |
| Product + B>=100 | 60 | 59 | 2.57x | 1.08x |
| Previous: product + B>=100 + N>=1000 | 36 | 35 | 1.67x | 1.08x |

Cases newly enabled by removing N>=1,000, across depths three/six and 10/100
rounds, at N=100 or 200 and B=10,000:

| Dataset | Newly enabled cases | Speedup range |
|---|---:|---:|
| adult | 6 | 5.44–12.26x |
| cal_housing | 6 | 4.00–12.72x |
| covtype | 6 | 2.49–6.45x |
| fashion_mnist | 6 | 5.89–18.64x |

The simplified rule still intentionally misses wins with B<100 and with products
below one million. We did not tune a lower background threshold after observing
this run or claim the chosen threshold is optimal.

## Memory caveat, including a limitation of the previous rule

8 product-selected cases encounter allocation fallback. Their timings
include attempted dense computation followed by complete traversal recomputation;
they are not timings of successful Woodelf processing. The hard cap protects
workspace but cannot prevent this extra latency. The existing constant-path
handling can exceed the tile estimator's predicted footprint, as already
observed in POLICY_RESULTS.md. The tiling implementation is unchanged.

The background-guarded candidate retains the following measured regression:

- covtype-d6-r100-n20000-b100: 556.74 ms traversal versus 600.29 ms attempted dense plus fallback (1.078x slower); 4 fallback events including warmup.

The previous N>=1,000 rule makes the same selection in that case. Thus removing
the foreground guard adds no losing selections in this grid, while neither rule
is universally regression-free. Fixing the temporary-allocation estimate or
avoiding repeated failed attempts is separate implementation work.

Peak reserved pool workspace across all runs is 58.65 MiB, below 64 MiB.
This cap excludes common model/input/output storage, CUDA modules and compiler
local memory. At 100,000 foreground rows, Fashion-MNIST output alone occupies
6.28 billion bytes; those common output costs are not hidden inside the workspace
claim. Allocation fallback only addresses extra dense workspace, not inability
to allocate common inputs or output.

## Protocol and validation

108 cases: four datasets, three model configurations (complete depth-three
10-round model, first 10 rounds of depth-six training, complete depth-six
100-round model), nine batch shapes each:

- Product one million: (100,10000), (1000,1000), (10000,100), (20000,50), (100000,10).
- Product two million: (200,10000), (20000,100), (100000,20).
- Below threshold: (5000,100).

Models are unchanged from the previous experiments, not independent training
seeds. Foreground rows are sampled with replacement from the cached 10,000-row
pool; backgrounds are a separate permutation of that pool, seed 20261011. Both
may overlap and contain duplicate values. Model hashes and source digest are
recorded. A separate compute process observed during the run was on GPU 0.

Traversal, dense and the product-policy wrapper each have a warmup and three
randomized-order timed repetitions. Dense preparation is rebuilt every call.
One-time CPU path setup and common model/input/output allocation are excluded
from batch timing. CUDA event and wall-clock samples are preserved.

All warmup/timed attribution comparisons passed atol=rtol=1e-5 against improved
traversal; maximum recorded absolute difference: 3.19e-12. All-value attribution comparisons execute on the GPU outside timing, avoiding
large CPU temporary arrays. Every reference
passed raw-margin prediction additivity against Treelite at atol=rtol=1e-4. All
pool cap assertions passed, including cases with correct fallback. These extend
agreement and additivity tests, not independent exhaustive coalition validation.
The unchanged kernels were not re-sanitized for this experiment.

## Reproduce

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_PATH=/home/rorym/miniforge3/envs/cuml_dev/targets/x86_64-linux
python cpp/bench/sg/interventional_shap/product_policy_experiment.py \
  --cache cpp/build-interventional/depth-policy \
  --original-cache cpp/build-interventional/workload \
  --output cpp/build-interventional/product-policy.json
```

Use the prior cached models/inputs and cuml_dev environment.
`product_policy_results.json.gz` archives the full raw run;
`product_policy_summary.csv` includes each backend's median, min/max, memory,
errors, fallback count and all three policy choices. Threshold evidence remains
limited to this GPU, these model families, these shapes and one training seed.
