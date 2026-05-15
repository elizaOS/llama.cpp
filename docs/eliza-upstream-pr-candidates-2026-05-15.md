# Eliza upstream llama.cpp PR triage, 2026-05-15

Scope: llama.cpp PRs less than one year old, filtered for inference speed,
memory/OOM behavior, quantization, backend support, Qwen/Eliza-1 model
coverage, batching/cache behavior, server tool calling, and Apple/CUDA/Metal/
Vulkan/SYCL/RISC-V/iOS platforms. GitHub search capped the merged-PR scan at
1000 rows; raw triage artifacts are in `/tmp/llama-pr-inventory/`.

## Landed in this pass

- #20836, `model: codefuse-ai/F2LLM-v2 support`.
- #23083, Vulkan UMA / host-visible buffer optimization. Conflict resolution
  preserved this fork's async transfer queue semaphore path and added the
  host-write barrier on both new and reused compute contexts.
- #22960, Anthropic streaming `tool_use` emits empty `input`.
- #21890, server `--parallel-tool-calling`. Conflict resolution preserved this
  fork's template-capability default and lets request/CLI options enable it.
- #21935, optional `libmtmd` Apple XCFramework build support. Conflict
  resolution preserved the OmniVoice build option and added `LLAMA_BUILD_MTMD`.

## Reviewed but not directly merged

- #20361 / #20391, Metal/CUDA GDN kernels. Current fork already carries a newer
  GDN/GLA implementation with transposed state-layout fixes and CUDA fastdiv/
  warp paths. Direct merge would regress fork code. Keep auditing individual
  kernels/tests, but do not merge the whole PR.
- #22768, RISC-V optimized q1_0 dot. Current fork already has RVV VL128/VL256
  q1_0 dot dispatch plus XTheadVector and custom Q1_0_g32/Q1_0_g128 fallbacks.
  Direct merge would remove fork-specific pieces. Treat as covered, but compare
  against #22754/#22500 for any remaining VLEN dispatch gaps.
- #21489, MTMD fit params include mmproj memory. Useful, but the PR conflicts
  with this fork's newer MTMD media-marker/capability/model-target APIs. Manual
  port only: add memory-estimate APIs on top of the fork API, then adjust server
  fit target accounting.
- #21089, CPU TurboQuant KV cache types. Valuable but not ABI-compatible as-is.
  Upstream uses 256-wide `QK_K` TBQ blocks, `ggml-turboq.c/.h`, rotation/
  projection tables, and `vec_dot_type = GGML_TYPE_Q8_K`. This fork uses
  cache-oriented `QK_TBQ = 32` TBQ3/TBQ4 layouts in `ggml-common.h`, quantizers
  in `ggml-quants.c`, `vec_dot_type = GGML_TYPE_F32`, plus QJL1_256,
  Q4_POLAR, and TBQ3_TCQ cache formats. Porting requires either new enum IDs
  for upstream TBQ_K256 formats or a measured replacement of the fork TBQ32
  path. Do not reuse the same type IDs with a different block layout.

## Should add or port next

### Quantization and conversion

- #21089, CPU TurboQuant KV cache types: dedicated TurboQuant port/evaluation,
  with type-layout separation and adapted vec-dot/dispatch tests.
- #23019, SQ4 semantic quantization: evaluate as a new quant type after current
  custom quant ID map is documented.
- #19941, Q3_PT quantization: evaluate alongside SQ4 and TurboQuant; likely
  conflict area is quant enum/type traits.
- #22836, STQ1_0 ternary quantization with ARM NEON vec_dot: useful for Apple/
  iOS CPU paths if it can coexist with existing custom cache quant types.
- #22671, MXFP6 CPU implementation.
- #20609, MXFP flash attention CPU reference.
- #23062, FP8 model dequant conversion.
- #22780, `--fuse-qkv` during HF-to-GGUF conversion.
- #20628, QKV weight fusion for LLaMA/Qwen2/Qwen3.
- #22633, conversion memory reduction by avoiding expanded dequant scale tensors.
- #15648, streaming dtype casts to reduce peak conversion RAM.
- #15727, reflinks for faster conversion.
- #21507, overlap `llama-quant` compute and write with double buffering.
- #21070, quantization recipes from custom recipe files.
- #15550, auto-select quant types to reach a bpw target at lowest error.
- #15060, configurable neutral imatrix prior.
- #21475, replace hardcoded quantization metadata strings with `LLM_KV`
  constants.

### RISC-V, CPU, and NUMA

- #22754, extend RVV quant vec-dot kernels to higher VLENs. Compare carefully
  with current q1_0 VL128/VL256/XTheadVector coverage and port only missing
  kernels.
- #22500, RVV q1_0 x q8_0 vec dot. Likely overlapped by current fork and
  #22768, but audit for dispatch differences.
- #20723, extend RVV repack GEMM/GEMV to other VLENs. Verify whether already
  included; if not, port the VLEN-generic pieces.
- #18348, RISC-V vec-dot dispatch based on VLEN.
- #18926, RISC-V Vector SSM scan for Mamba-2.
- #19196, q4_0/q8_0 scale optimization using Zvfhmin.
- #17483, RISC-V Zvfh f16 dot implementation.
- #14032, RISC-V build condition hardening for GCC version.
- #17113, broad CPU SIMD/pipeline optimizations across vec/mmq/ops/KV/repack.
- #18495, x86 CPU Q4_K x Q8_K runtime optimization.
- #19707 and #19706, Q5_K/Q6_K block interleaving for x86 SIMD.
- #17642, AVX2/FMA sum-of-squares loop unrolling.
- #17641 and #19657, AVX2 mask/sign optimizations.
- #17648, AVX dot-product loaded-vector reuse.
- #18150, small SIMD-like 4-element CPU optimization.
- #16650, CPU RMSNorm optimization.
- #20596, improve `--n-cpu-moe` TG performance.
- #16000, `--numa mirror` for model weights.
- #14232, `GGML_NUMA_MIGRATE`.
- #13731, page-cache migration via `mbind`.
- #18698, NUMA repack-buffer inference speed.
- #16882, disable NUMA-specific chunking on high-core-count HPC systems.
- #13710, cacheline-reduced structure alignment.
- #22460, use math-core count for automatic `--threads`.
- #19260, use physical cores for thread auto-detect.
- #14532, prefer big cores on AArch64 hybrid Linux.
- #14380, Neoverse-V2 CPU build variant.
- #21958, add SVE for `simd-gemm.h`.
- #19357, int2 quantization and KleidiAI SME2 kernels.
- #15719, Q4_K repack support for AArch64.

### CUDA, HIP, and GPU memory

- #18102, CUDA Delta-Net linear attention for Qwen3-Next.
- #22710, fuse RMSNorm, MUL, and Q8_1 quantize.
- #22522, CUDA Programmatic Dependent Launch for Hopper+.
- #22297, CUDA `POOL_1D`.
- #22207, CUDA legacy-pool LRU eviction and overalloc.
- #22112, CUDA repacking for MXFP4.
- #21673, upper VRAM limit for cached CUDA graphs.
- #21652 and #22571, Q8_1 activation overflow prevention / BF16 activation
  swap.
- #20520, CUDA RMSNorm float4 vectorized load/store template.
- #20831, dynamic MMVQ warp count for narrow matrices.
- #20078, BF16 CUBLAS path and higher-precision FP16 path.
- #19098, CUDA flash-attention rowsum / fp16 softmax offset fix.
- #18538, cache intermediate CUDA tensors.
- #16844, CUDA graphs for Gemma 300M embeddings.
- #16548, CUDA graph plans.
- #16016, deterministic CUDA inference mode.
- #15298, 64-bit CUDA copy routines for large tensors.
- #14639, CUDA non-contiguous unary ops.
- #15805 and #17255, CUDA implicit conv2d/conv3d.
- #17383, CUDA rel-pos/window ops.
- #16457, hipblasLt batched GEMM for CDNA3.
- #17495, HIP RDNA3 WMMA MMF.
- #19834, ROCm gfx950 target.

### Metal, Apple, iOS, and Vulkan

- #16143, Metal fused RMSNorm + MUL + SwiGLU for Qwen3Next.
- #22515, Metal async 2D tensor copy functions.
- #21119, Metal opt-in V skip for negligible attention weights.
- #19600, Metal `mul_mv_ext` for large `n` on non-simdgroup_mm GPUs.
- #18878, Metal floor op.
- #16669, Metal DIAG_MASK_INF / IM2COL_3D / PAD support.
- #16530, Metal LOG op.
- #14570, Metal graph reuse.
- #15262, Apple NPU acceleration example. Large and likely experimental, but
  relevant for iOS/macOS roadmap.
- #21556, MoltenVK AMD eGPU GCN detection fix on macOS.
- #21928, unified Apple SDK support in nix.
- #23083, Vulkan UMA optimization: landed locally.
- #22951, Vulkan Q3_K/Q6_K 32-bit alignment. Draft, but keep tracking.
- #20017, Vulkan sparse OOM fallback and chunked staging fallback.
- #19743, Vulkan TQ1_0/TQ2_0 MUL_MAT support.
- #18493, Vulkan tuning for ARM Mali G720.
- #17485, dynamic VRAM heuristic for low-VRAM Vulkan GPUs.
- #17374, Intel GPU subgroup/block-size default changes.
- #17147, Vulkan q2_K mul_mmq implementation.
- #20451, Vulkan Slang flash-attention shader.
- #20377 and #20376, Vulkan GDN chunked kernel and f16 state.
- #19254, test coverage for Vulkan host-memory buffers.
- #15800, Vulkan mul_mat variant for embedded GPUs.

### SYCL and OpenCL

- #21845, SYCL multi-column MMVQ port from CUDA, claimed about 45% speculative
  decoding speedup on Intel Arc.
- #22066, SYCL Battlemage optimizations.
- #22766, SYCL malloc_shared for UMA/integrated GPUs.
- #22526, optional SYCL USM system allocations.
- #22098, SYCL zero-copy path with cache flushing for Intel UMA.
- #16969, SYCL flash attention.
- #18126 and #18138, Intel cooperative-matrix / FA-row experiments; WIP but
  relevant to Intel GPU coverage.
- #21313, OpenCL flash-attention optimizations.
- #21311, OpenCL Q4_K/Q4_0 SOA improvements and Adreno SIGSEGV fix.
- #22642, OpenCL compiled-kernel binary cache.
- #20811, OpenCL host-buffer free after MoE MXFP4 router reorder.
- #22764, OpenCL command-buffer plan proof of concept.

### Batching, KV cache, memory, and model loading

- #22569 and #17579, paged KV cache / PagedAttention. High-impact but large;
  needs isolation from this fork's custom cache quantization work.
- #21792, optional mmap KV cache.
- #18747, KV cache size limiting and block-tracking infrastructure.
- #16000, NUMA mirror loading.
- #20062, parallel model loading across GPU contexts.
- #19180, hybrid model loading with DirectIO and MMAP.
- #14484, KV-cache-aware layer distribution for multi-GPU inference.
- #18373, model pinning to protect critical models from LRU eviction.
- #21231, server router device-memory margin for dynamic unloading.
- #22284, server router unload/reload deadlock fix.
- #20822 and #20819, router slot state checkpoint save/restore.
- #22826, preserve context checkpoint coverage.
- #17428, checkpoints while processing prompt.
- #19670 and #22400, partial `seq_rm` success for hybrid/GDN memory and
  speculative decoding.
- #22692 and #22691, pshard runtime/planning for streamed weights.
- #20834, load-mode refactor for mlock/mmap/directio.
- #22446, `ggml_graph_overhead_custom` pool-size off-by-one.
- #17640, `ggml-alloc` free-block shift via `memmove`.
- #13764, batch/sbatch/ubatch concept tests.
- #15636, `pad_equal` batch RFC.
- #17342, small-batch throughput improvement.

### Qwen, Eliza-1 model coverage, model bugs

- #18102, CUDA Delta-Net linear attention for Qwen3-Next.
- #16143, Metal fused Qwen3Next path.
- #22661, recurrent-layer linear-attention state corruption fix.
- #23082, Mamba2 d_conv/d_inner fixes.
- #23017, `d_conv=15` for SSM CUDA.
- #21437, Qwen3.5 EAGLE3 draft support.
- #20752, Qwen3 TTS architecture.
- #20009, Qwen3 reranker instruction support.
- #15248, Qwen3 reasoning and `tool_choice=required`.
- #21587, Gemma 4 BPE tokenizer SIGSEGV on long prompts.
- #22325, Gemma 4 prefill parsing fix.
- #21433, Gemma 4 nullable type arrays.
- #14148, Mistral Small 3.1 template with tool calling.
- #15083, Nemotron reasoning/tool parsing.
- #22520, Nemotron 3 Nano Omni / parakeet MTMD support.
- #21729, reranker token_type_ids fix.
- #22576, Jina reranker v3 cross-encoder.
- #20880, Phi-4-Mini-Flash-Reasoning.
- #17687, Phi-3.5 Vision.
- #21045, Falcon OCR.
- #20975, DeepSeekOCR 2.
- #17840, R-4B multimodal.
- #15123, T5Gemma.
- #17956, encoder-decoder support for T5/BART/MADLAD.
- #21733, EXAONE 4.5.
- #21412, Zamba2.
- #21032, nvidia/gpt-oss-puzzle-88B.
- #21963, HY-Embodied.
- #21904, MPNet.
- #21161, RUGPT3XL.
- #18719, VAETKI.
- #17454, LLADA 2.0.
- #17141, Megrez-MoE.
- #13908, Eagle2 draft arch.
- #13799, OPT support.

### Server, tool calling, caching, and API behavior

- #21890 and #22960: landed locally.
- #22660, auto-enable `swa_full` for SWA draft models.
- #22083, disable similarity slot selection with `--cache-idle-slots` and
  `--parallel 1`.
- #21815, reinit speculative ngram state after context shift.
- #22055, save dynamic/static ngram cache file.
- #18039, EAGLE3 speculative decoding.
- #17034, profile-guided speculative decoding.
- #18886 and #22673, MTP API/support.
- #20981, Step3.5 MTP.
- #15225, GLM-style MTP.
- #19833, multiple outputs per sequence.
- #22761, `/infill` prompt placement after FIM_MID.
- #22393, slot prompt similarity getter/setter.
- #22081, always include usage in streaming responses.
- #19572, Anthropic-compatible cache-read usage metric.
- #20872, SSE headers behind reverse proxies.
- #20858, signed free-memory logging to show VRAM spillover.
- #20623, avoid adding `x-anthropic-` system message for Claude Code.
- #20479 and #20088, reasoning API parameter support.
- #19753, XML tool calls with duplicate parameter keys.
- #18353, PEG-parser migration for tool-call parsing. WIP but valuable tests.
- #18044, whitespace around JSON tool calls.
- #21308, multi-turn tool-use integration tests.
- #22437, reranking guard/tests.
- #22336, per-request thinking toggle.
- #22089, graceful degradation for malformed tool-call arguments.
- #21657, WebUI thinking-mode request handling.
- #21174, Responses API / Codex CLI compatibility.
- #21477, null-check context on init failure.
- #19841, chat truncation to keep chat going.
- #19855, default-model preset/fallback logic.
- #19694, V-L embedding model fix.
- #18123, embedding `n_batch == n_ubatch` validation.
- #14728, prompt-processing progress streaming.

### Tests and benchmarks to add around risky ports

- #14833, composite op perf/eval testing in `test-backend-ops`.
- #14139, random-model tests.
- #13764, batch/sbatch/ubatch tests.
- #22495, CPU/GPU roofline profiler examples.
- #21160, cross-backend profiler.
- #15643, llama-bench TTFT/E2E/ITL metrics.
- #21794, accumulated `load_time` fix in llama-bench timings.
- #14811, upload benchmark test results.
- #19691, multi-image and no-image vision API tests.
- #19254, Vulkan host-memory buffer test.

## TurboQuant/RISC-V validation gaps

Current local coverage is good enough to prevent basic regressions but not
enough to validate a type-layout replacement:

- `tests/test-quantize-fns.cpp` covers TBQ3/TBQ4/QJL/Q4_POLAR/TBQ3_TCQ type
  registration and CPU quant/dequant behavior.
- `tests/test-qjl-cache.cpp` covers QJL1_256 round trips and
  `GGML_OP_ATTN_SCORE_QJL`.
- `tests/test-cuda-extra-kernels.cpp` validates QJL/Q4_POLAR/TBQ3_TCQ CUDA
  symbol linkage when CUDA is enabled.
- `tests/test-quantize-perf.cpp` and `tools/llama-bench/quant-bench.cpp` are
  the right benchmark hooks, but need explicit TurboQuant/RISC-V cases and
  golden comparisons before replacing fork layouts.

Before importing #21089 or further RVV PRs, add:

- A quant type-ID/layout map documenting Eliza-only and upstream IDs.
- TBQ32 versus TBQ_K256 accuracy, memory, prefill, decode, and KV-cache
  benchmarks on the same smallest Eliza-1 model.
- RISC-V RVV VLEN128/VLEN256/VLEN-generic dispatch tests, ideally in CI or a
  reproducible cross-runner, because local macOS arm64 cannot execute them.
- Conversion/load/save round-trip tests for every custom cache quant format.
- CUDA/Metal/Vulkan fallback tests confirming unsupported custom quant formats
  fail or route predictably instead of silently selecting the wrong vec-dot.

## Validation run

After landing #20836, #23083, #22960, #21890, and #21935:

- Built targets: `llama-cli`, `llama-server`, `test-tokenizer-0`, `test-chat`,
  `test-chat-peg-parser`, `test-chat-auto-parser`, `test-chat-template`,
  `test-quantize-fns`, `test-backend-ops`.
- Ran targeted CTest selection:
  `test-tokenizer-0|^test-chat$|test-chat-peg-parser|test-chat-auto-parser|test-chat-template|test-quantize-fns|test-backend-ops`.
- Result: 21/21 tests passed.

Model-level Eliza-1 validation still requires the actual smallest GGUF for
each Eliza-1 family and should be run after the next manual quant/backend port.
