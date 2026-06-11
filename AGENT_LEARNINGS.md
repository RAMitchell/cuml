# Agent Learnings and Worktree Strategy

This file is the shared notebook for agents working in this repository. Use it
to capture repo-specific lessons that should outlive one task, and to coordinate
parallel branch/build work without stepping on another agent's checkout,
artifacts, or installed libraries.

This does not replace the focused review guides in `python/agents.md` and
`cpp/agents.md`; those remain the source of truth for language-specific review
posture. This file is for cross-cutting project memory and workflow strategy.

## How to Add Learnings

Add new learnings when an agent discovers something reusable about this repo:
build behavior, test selection, dependency quirks, code ownership boundaries,
review heuristics, or pitfalls that would save the next agent time.

Use this shape and put newest entries first:

```markdown
### YYYY-MM-DD - Area - Short title

- Learned: What changed in our understanding.
- Evidence: File, command, PR, issue, or failing test that supports it.
- Reuse: How the next agent should act differently.
- Confidence: High, medium, or low.
```

Keep entries specific and source-backed. If a learning is later disproved,
append a correction instead of silently deleting the old context.

## Current Shared Learnings

### Review posture is risk-first

- Learned: The existing Python and C++ agent guides ask reviewers to focus on
  critical and high-impact issues, not style nits.
- Evidence: `python/agents.md` and `cpp/agents.md`.
- Reuse: When reviewing or auditing changes, prioritize correctness, API
  compatibility, memory/resource safety, numerical stability, and tests.
- Confidence: High.

### Python changes must protect scikit-learn compatibility

- Learned: Public Python estimators are expected to preserve scikit-learn style
  signatures, fitted-attribute behavior, input handling, and compatibility
  tests.
- Evidence: `python/agents.md`, `python/cuml/tests/test_sklearn_compatibility.py`.
- Reuse: For estimator work, compare parameter names/defaults and fitted
  attributes against scikit-learn, cover cuDF/pandas/NumPy where applicable,
  and add compatibility coverage for new estimators.
- Confidence: High.

### C++/CUDA changes must make stream and memory ownership explicit

- Learned: The C++ layer relies on RAFT/RMM conventions, explicit stream
  handling, and careful row-major/column-major assumptions.
- Evidence: `cpp/agents.md`.
- Reuse: For CUDA work, look for unchecked launches, accidental default-stream
  coupling, resource lifetime bugs, data-layout mismatches, and missing
  numerical edge-case checks.
- Confidence: High.

### Tests should avoid external data

- Learned: Both review guides call out external datasets as a test-quality
  problem.
- Evidence: `python/agents.md` and `cpp/agents.md`.
- Reuse: Prefer synthetic data or bundled fixtures. If a test needs a new data
  file, keep it small and committed with a clear reason.
- Confidence: High.

### Build isolation matters in this repo

- Learned: `build.sh` defaults C++ artifacts to `cpp/build`, Python artifacts
  to `python/cuml/build`, and installs to `$INSTALL_PREFIX`, `$PREFIX`,
  `$CONDA_PREFIX`, or finally `cpp/build/install`.
- Evidence: `build.sh`.
- Reuse: Concurrent branches need separate worktrees, and concurrent C++ build
  variants should set distinct `LIBCUML_BUILD_DIR` and, when installing,
  distinct `INSTALL_PREFIX` values.
- Confidence: High.

### Branch names and targets should follow CONTRIBUTING.md

- Learned: PR branches use `<type>-<name>` with `fea`, `enh`, or `bug`.
  PRs target `main` by default, with release and hotfix branches reserved for
  release-specific work.
- Evidence: `CONTRIBUTING.md`.
- Reuse: Create worktree branches from `origin/main` unless the task explicitly
  targets `release/YY.MM` or a hotfix.
- Confidence: High.

## Worktree Strategy for Concurrent Agents

### Goals

- Keep each active branch in its own checkout.
- Keep build outputs and install prefixes isolated by branch and variant.
- Share the Git object database and compiler cache where useful.
- Make ownership visible so multiple agents can work concurrently.
- Keep the primary checkout available for inspection and emergency fixes.

### Recommended Layout

Use sibling directories outside the main checkout:

```text
/home/rorym/cuml                         # primary checkout
/home/rorym/cuml-worktrees/<branch>      # one checkout per active branch
/home/rorym/cuml-builds/<branch>/<kind>  # optional external C++ build dirs
/home/rorym/cuml-installs/<branch>/<kind> # optional install prefixes
```

For example:

```bash
mkdir -p ../cuml-worktrees ../cuml-builds ../cuml-installs
git fetch origin
git worktree add ../cuml-worktrees/enh-fast-kmeans \
  -b enh-fast-kmeans origin/main
```

For release-targeted work:

```bash
git fetch origin
git worktree add ../cuml-worktrees/bug-rf-release-2602 \
  -b bug-rf-release-2602 origin/release/26.02
```

Before creating or removing a worktree, check the active map:

```bash
git worktree list
```

### Ownership Contract

Each agent should own exactly one worktree for write-heavy work. At task start,
record or announce:

- Branch name.
- Base ref, such as `origin/main` or `origin/release/26.02`.
- Worktree path.
- Build directory and install prefix, if building.
- Intended test scope.

If two agents need the same branch, one should create a new branch from the
other's tip rather than checking out the same branch in two worktrees.

### Build Isolation

For ordinary branch work, the worktree itself isolates the default `cpp/build`
and `python/cuml/build` directories.

For multiple C++ build variants from the same source, set separate
`LIBCUML_BUILD_DIR` values:

```bash
cd ../cuml-worktrees/enh-fast-kmeans
conda activate cuml_dev

BRANCH_SLUG=enh-fast-kmeans
LIBCUML_BUILD_DIR=/home/rorym/cuml-builds/${BRANCH_SLUG}/cpp-release \
INSTALL_PREFIX=/home/rorym/cuml-installs/${BRANCH_SLUG}/release \
PARALLEL_LEVEL=8 \
./build.sh libcuml --ccache

LIBCUML_BUILD_DIR=/home/rorym/cuml-builds/${BRANCH_SLUG}/cpp-debug \
INSTALL_PREFIX=/home/rorym/cuml-installs/${BRANCH_SLUG}/debug \
PARALLEL_LEVEL=8 \
./build.sh libcuml -g --ccache
```

Use `--ccache` for branch switching and debug/release churn. The cache can be
shared; source trees, build directories, and install prefixes should remain
separate.

If using the same conda environment for several branches, remember that the
default install target is usually `$CONDA_PREFIX`. Treat a shared conda prefix
as a single-writer install location: rebuild/reinstall the branch under test
before trusting Python imports or linked libraries. For truly concurrent
runtime testing, use separate conda environments or separate install prefixes
with the runtime environment configured to match the selected prefix.

### Test Selection

Prefer the narrowest meaningful validation for the changed area:

- Python estimator/API work: targeted `pytest` under `python/cuml/tests` plus
  compatibility tests when estimator behavior changes.
- C++/CUDA work: targeted C++ tests from `cpp/build/test`, with multi-GPU tests
  only when the code path requires them.
- Build or packaging work: at least one clean configure/build of the affected
  target, using an isolated build directory.
- Documentation-only work: no build required unless the docs syntax or docs
  generation path changed.

When a full build or GPU test is not available, record what was checked and
what remains unverified.

### Sync and Cleanup

Keep bases fresh before starting substantial work:

```bash
git fetch origin
git status --short
```

Remove completed worktrees only after changes are merged, abandoned, or safely
preserved elsewhere:

```bash
git worktree remove ../cuml-worktrees/enh-fast-kmeans
git worktree prune
rm -rf ../cuml-builds/enh-fast-kmeans ../cuml-installs/enh-fast-kmeans
```

Delete the branch only when no PR, handoff, or follow-up depends on it:

```bash
git branch -d enh-fast-kmeans
```

## Coordination Checklist

Before editing:

- `git status --short` is clean or unrelated changes are understood.
- `git worktree list` shows no conflicting checkout for the branch.
- The branch name follows `CONTRIBUTING.md`.
- Build and install paths are unique if a build will be run.

Before handoff:

- The final status includes branch, worktree path, and changed files.
- Tests/builds are listed with exact commands and outcomes.
- Any reusable discovery is added to this file using the learning template.
- Known risks or unverified paths are explicit.
