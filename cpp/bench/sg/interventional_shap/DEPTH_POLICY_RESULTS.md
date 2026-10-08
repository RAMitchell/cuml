# Extending the cheap dispatch rule to depth six

Follow-up: [product-rule experiments](PRODUCT_POLICY_RESULTS.md) test simplifying
the batch guards and expose a memory-estimation fallback at a wider batch shape.

Measured 2026-10-08, NVIDIA RTX PRO 6000 Blackwell (physical GPU 1), same
standalone prototypes as POLICY_RESULTS.md. No CUDA kernels or production API
changed. The rule was fixed before this experiment: raise the maximum depth
from three to six and retain N>=1,000, B>=100, N*B>=1,000,000, the one-leaf
workspace guard, and a hard 64 MiB pool cap with traversal fallback.

## Result

**All 48 cases selecting Woodelf-style dense processing won**, by
2.18–55.75x against lookup plus local-reduction
traversal without input caches. This includes **all 24 depth-six selected
cases**, winning by 2.18–55.75x.
The measured policy call is included in those timings. These are improvements
against the optimized traversal, not the original unoptimized implementation.

| Maximum depth | Dense-selected cases | Speedup range | Geometric mean |
|---|---:|---:|---:|
| 4 | 12 | 3.21–48.61x | 9.66x |
| 5 | 12 | 2.95–51.94x | 8.40x |
| 6 | 24 | 2.18–55.75x | 8.26x |

At N=10,000 explained rows and B=100 background rows, depth six:

| Dataset | Traversal, 10 rounds (ms) | Policy/Woodelf, 10 rounds (ms) | Speedup | Traversal, 100 rounds (ms) | Policy/Woodelf, 100 rounds (ms) | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| adult | 6.74 | 1.50 | 4.49x | 62.53 | 12.08 | 5.18x |
| cal_housing | 5.56 | 2.16 | 2.58x | 53.46 | 15.40 | 3.47x |
| covtype | 24.33 | 11.19 | 2.18x | 247.29 | 96.42 | 2.56x |
| fashion_mnist | 89.62 | 18.43 | 4.86x | 985.02 | 160.69 | 6.13x |

The 16 cases at N=1,000/B=100 retain traversal. Their worst measured slowdown
is 1.43%, with three timed repeats per
backend. Raw ranges are included in the CSV; these short runs do not establish
a zero-overhead or zero-regression guarantee.

## Workload and protocol

Four original benchmark datasets: Adult, California Housing, Covtype and
Fashion-MNIST. Twelve new models were trained using the existing repository
workload recipe (GPU hist, eta=0.01, seed=0, base_score=0.5), varying only depth
and number of boosting rounds. Depths four and five have 10 rounds; depth six
has 100 rounds, tested both as its first 10 rounds and as the complete 100-round
model. Ten-round prefixes are equivalent to stopping that same training run at
10 rounds; they are not independent model seeds. Multiclass models contain one
tree per output group per round. Every model reached its requested depth.

For each of those 16 model configurations, test (N,B)=(1000,100), (1000,1000),
(10000,100), (10000,1000): **64 cases**. Foreground and background indices are
separate seeded permutations (seed 20261010) of the cached foreground pool;
values can duplicate and the two samples can overlap. This follows the previous
empirical-background protocol and does not claim new independent datasets.

Each case compares traversal, streamed dense processing, and the actual policy
wrapper. Every backend is warmed up, then timed three times in randomized order.
Batch-dependent dense preparation is included each call; there is no persistent
input cache. CPU model path extraction is recorded separately outside timings.
Common GPU input/model/output allocations are also outside timings.

## Memory and correctness

Peak selected extra pool workspace: **59.09 MiB**. Total ordinary allocation
fallbacks: **0**. All pool-reservation assertions passed the 64 MiB limit.
The limit excludes common inputs, outputs, model storage, CUDA modules and
compiler-managed local memory, so it does not cap total GPU usage. Traversal
continues to use only 2 KiB of explicit pool workspace. The four forced-low-budget
fallback checks from POLICY_RESULTS.md remain applicable; no fallback code changed.

Every warmup and timed output matched traversal at atol=rtol=1e-5; the maximum
recorded absolute attribution difference was 5.13e-13. Each traversal reference
also passed prediction additivity against Treelite at atol=rtol=1e-4. This extends
agreement/additivity coverage to the newly trained models, rather than providing
a fresh exhaustive SHAP oracle. Unchanged CUDA kernels were not re-sanitized.

## Interpretation

Depth three was a conservative validated boundary, not an algorithmic limit.
This experiment supports extending the candidate rule to depth six on this
hardware. The dense algorithm handles 2^k decision patterns per leaf, where k
is the number of distinct features on that path. At k=6 there are only 64
patterns. Deeper trees also have more leaves and larger transforms, but those
costs do not erase the benefit at the selected batch sizes here. This explanation
follows the prototype's allocation/loop structure; it is not a hardware-counter
attribution of the measured speedups.

Keep the batch guards and bounded workspace. The prior failures from choosing
based mainly on depth still matter. These results cover one GPU, one training
seed per dataset/depth and four batch shapes, not a universal crossover or all
possible depth-six models. Production dispatch remains unchanged.

## Reproduction

```bash
export CUDA_VISIBLE_DEVICES=1
export CUDA_PATH=/home/rorym/miniforge3/envs/cuml_dev/targets/x86_64-linux
python cpp/bench/sg/interventional_shap/depth_policy_experiment.py --prepare \
  --cache cpp/build-interventional/depth-policy \
  --output cpp/build-interventional/depth-policy.json
```

Use the cuml_dev environment and cached source datasets from the earlier runs.
`--prepare` trains/caches the models in a separate directory; omit it for timing
reruns. `depth_policy_results.json.gz` stores the full measurements, source digest,
model hashes, and all twelve training metadata records. `depth_policy_summary.csv`
contains medians, ranges, errors and memory peaks. Neither models nor build
artifacts are committed.
