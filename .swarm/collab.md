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
- **E** — research-and-fix-or-flag additional CI breaks. Files in scope (NEW, not in A-D lanes): ggml/src/ggml-cpu/qjl/quants-qjl.c (Windows MSVC: `pthread.h` missing — replace `pthread_once` with portable atomic CAS), ggml/src/ggml-cpu/fused-attn-qjl-tbq.c (Linux gcc: `alloca` used without `<alloca.h>` include). Read-only investigation everywhere else.

## Completed

<!-- agents append here after commit -->
- **A** 858c820ad — fix(qjl): empty-TU guard in qjl_quantize_avx2.c when AVX2 disabled (added typedef stub outside `#if __AVX2__`)
- **C** 8921d3cc4 — fix(ops): add QJL1_256/Q4_POLAR/TBQ3_TCQ to the exhaustive abort-group in ggml_compute_forward_clamp (only `-Wswitch`-affected switch in ops.cpp; all others use `default:`). No other arch-specific ops.cpp under ggml/src/ggml-cpu.
- **E** ff0b750bb — fix(qjl): replace `pthread_once` with stdatomic CAS in `quants-qjl.c` (Windows MSVC has no `pthread.h`); add `<alloca.h>`/`<malloc.h>` portability include block to `fused-attn-qjl-tbq.c` (mirroring `ggml.c`'s pattern). Unblocks windows-2022-cuda, windows-latest-hip, windows-latest llvm-arm64, ubuntu-24-webgpu, ubuntu-24-webgpu-wasm builds. Backlog of 8 additional findings appended below.
- **B+D** (orchestrator finish) — B and D produced correct local edits but were blocked by transient merge churn before they could push. Orchestrator committed both:
  - ggml-rpc.h: `static_assert(GGML_OP_COUNT == 96)` → `== 100`, `RPC_PROTO_PATCH_VERSION 0 → 1` per the assertion's instruction.
  - build-cache.yml: removed the duplicate top-level job key `ubuntu-24-openvino-cache` (duplicate keys cause GitHub Actions to fail in 0s as "workflow file issue"). One copy of the job remains.

## Conflicts / coordination notes

<!-- agents append here if they need to communicate -->

## Backlog from agent E

Tracked but NOT fixed in this pass — all blocked behind agents A-D landing their build-break fixes (the runtime tests can't even reach these failures until builds pass). Re-investigate once CI compiles end-to-end.

1. **test-quantize-fns SEGFAULT** — `tests/test-quantize-fns.cpp:140` iterates over all `GGML_TYPE_COUNT` types and calls `qfns_cpu->from_float`. Eliza-custom types (`GGML_TYPE_QJL1_256`, `GGML_TYPE_Q4_POLAR`, `GGML_TYPE_TBQ3_TCQ`) likely have `from_float` set but their kernels depend on lazy state (e.g. QJL projection) or different blck-size assumptions vs the test's `test_size = 32 * 128`. Repros on `macOS-latest-arm64-webgpu` (job 76108849237) and `ubuntu-24-webgpu` (job 76108849409). Suggested fix: skip Eliza-custom types in this test or assert their precondition before calling `from_float`.

2. **test-quantize-perf SEGFAULT** — `tests/test-quantize-perf.cpp`, same iteration pattern, same crash group. Same fix as above.

3. **test-fused-kernels — fused QJL+TBQ attention error 8x over tolerance** — `tests/test-fused-kernels.cpp` "fused QJL+TBQ attention smoke" reports `max_rel = 4.06e-02` against target `5e-3`. Iter 50 is first to fail at n_tokens=154. This is NOT a build break — it is a numerical correctness regression in the fused kernel path (likely `ggml/src/ggml-cpu/fused-attn-qjl-tbq.c` or one of its SIMD variants). Bisect against `0d0cccf63 fused-cpu: add fused QJL-K + TBQ-V attention op (W3-B kernel #1)`. Need to inspect the two-pass online-softmax math (`fused-attn-qjl-tbq.c` lines 200-400).

4. **Server tests — backend-sampling `test_load_split_model` returns 500** — `unit/test_basic.py::test_load_split_model` fails with `llama_decode: failed to decode, ret = -1` / "Invalid input batch". Server returns 500 instead of 200. Eliza-specific cause unclear — could be related to the token-trie sampler or M-RoPE reject changes. Job 76108849082. Run the test locally against the merged HEAD and capture full server log.

5. **Server tests — default `test_chat_completion` token mismatch** — `unit/test_chat_completion.py` expects `"But she couldn't"`, gets `"By wanted touge"`. Deterministic sampling has drifted. Likely the token-trie sampler changed default sampling behavior. Job 76108849091. Either (a) the sampler change broke determinism and needs to be gated behind an opt-in flag, or (b) the test's reference string needs to be updated for the new (correct) sampling path. Triage which is the case before deciding.

6. **windows-latest-hip — ROCm installer flake** — Job 76108849302 mostly fails because ROCm installation reports "ROCm installation not found" after install (lines around `01:56:54`). This is independent of any source change; it's flaky CI infra. After the ROCm error, the build hits my pthread.h fix once installed correctly. No code fix needed — the ROCm step itself is fragile.

7. **Cross-check upstream** — `ggml-org/llama.cpp` does not ship `quants-qjl.c`, `fused-attn-qjl-tbq.c`, `qjl_quantize_avx2.c`, `ops.cpp` QJL/POLAR/TBQ cases, or the `GGML_OP_COUNT == 96` static_assert pinning. All listed failures are Eliza-only divergence introduced by commits `0d0cccf63` (fused QJL/TBQ kernel), `71fdb58d3` (QJL1_256 type traits), and `6f2451f05` (Q4_POLAR tests). No upstream regression to mirror.

8. **Workflows syntactically valid?** — A representative sample (`build.yml`, `build-cache.yml`, `server.yml`, `hip-quality-check.yml`, `build-self-hosted.yml`) all parse — failures on this PR are job-content failures, not workflow YAML errors. The only "workflow file issue" is `.github/workflows/build-cache.yml` which agent D is fixing (duplicate job id `ubuntu-24-openvino-cache`).

