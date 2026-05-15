# Swarm coordination — CI fix sweep on elizaOS/llama.cpp

Branch: `eliza/token-trie-sampler` (DO NOT switch branches, DO NOT stash, DO NOT use worktrees).

**⚠ Active churn:** another automation is continually merging `refs/remotes/upstream/pr/*` into this branch. HEAD moves frequently and transient `UU` unmerged paths appear and disappear. Before each commit attempt: `git fetch origin`, `git pull --rebase origin eliza/token-trie-sampler` (if behind), check `git status`. If you encounter a merge-in-progress (`MERGE_HEAD` exists or `git ls-files -u` returns rows), DO NOT abort — wait 30s and recheck, or resolve and commit. If git refuses your commit because of unmerged paths, resolve those first using a sensible "keep both upstream + Eliza diffs" strategy and continue.

## Rules

1. Before editing, **append** an entry under "## Live agents" with your scope and the files you intend to touch.
2. After committing, **append** an entry under "## Completed" with your commit SHA and a one-line summary.
3. If you find another agent has already modified your target, **read their changes first**, understand why, and either build on them or coordinate by appending a `## Conflict` entry. NEVER revert another agent's work without an explicit note here.
4. Commit dirty code as you go (small WIP commits are fine). NO stash, NO branch switching, NO worktrees.
5. Work in this submodule only: `eliza/plugins/plugin-local-inference/native/llama.cpp`. The parent `milady/eliza` repo just tracks the submodule SHA — do not touch parent files.
6. After your commit, push to `origin/eliza/token-trie-sampler` so other CI can pick it up.

## Work breakdown (claim by adding yourself under "Live agents")

- **A. Merge-conflict resolver** — resolve `common/speculative.cpp` and `tools/server/server-context.cpp`. The branch carries (1) M-RoPE reject upstream + (2) token-trie sampler diffs. Keep both Eliza features on top of incoming upstream main. Mark `git add` and commit.
- **B. qjl_quantize_avx2.c empty-TU** — `ggml/src/ggml-cpu/qjl/qjl_quantize_avx2.c` ends with no declarations and trips `-Werror=pedantic` "ISO C forbids an empty translation unit" on the Debug ctest job. Likely guarded by `#if defined(__AVX2__)` — needs a fallback decl when the guard is false.
- **C. ggml-rpc.h GGML_OP_COUNT static_assert** — `ggml/include/ggml-rpc.h:14` asserts `GGML_OP_COUNT == 96` but actual is 100. Update to the current count AND bump `RPC_PROTO_PATCH_VERSION` per the assertion message; check `ggml/include/ggml.h` for the real count.
- **D. ops.cpp unhandled enum warnings** — `ggml/src/ggml-cpu/ops.cpp` `ggml_compute_forward_clamp` switch missing `GGML_TYPE_QJL1_256`, `GGML_TYPE_Q4_POLAR`, `GGML_TYPE_TBQ3_TCQ`. Decide per-type: either implement clamp for them or add `GGML_ABORT("clamp not implemented for ...")` cases so `-Wswitch` is satisfied. Look for other ops in the same file with the same missing-enum pattern and fix uniformly.
- **E. build-cache.yml workflow** — `.github/workflows/build-cache.yml` fails in 0s ("workflow file issue"). Inspect with `gh workflow view`, find the YAML syntax / required-context error, fix.

If you spot a problem outside your slice that's blocking your work, claim it explicitly here before fixing.

## Live agents

<!-- agents append here -->
- **A** — qjl_quantize_avx2.c empty-TU. Files: ggml/src/ggml-cpu/qjl/qjl_quantize_avx2.c
- **C** — ops.cpp unhandled enums GGML_TYPE_QJL1_256/Q4_POLAR/TBQ3_TCQ. Files: ggml/src/ggml-cpu/ops.cpp
- **B** — ggml-rpc GGML_OP_COUNT + RPC_PROTO_PATCH_VERSION. Files: ggml/include/ggml-rpc.h, ggml/src/ggml-rpc/ggml-rpc.cpp (read), ggml/include/ggml.h (read only)
- **D** — build-cache.yml workflow file issue (duplicate job id). Files: .github/workflows/build-cache.yml

## Completed

<!-- agents append here after commit -->
- **A** 858c820ad — fix(qjl): empty-TU guard in qjl_quantize_avx2.c when AVX2 disabled (added typedef stub outside `#if __AVX2__`)

## Conflicts / coordination notes

<!-- agents append here if they need to communicate -->
