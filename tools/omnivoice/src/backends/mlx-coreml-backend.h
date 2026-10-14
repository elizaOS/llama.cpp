#pragma once
/*
 * mlx-coreml-backend.h — Apple-Silicon in-process streaming-LLM backend
 * (Gemma-4 cutover plan M5). One of the alternate `LlmBackendSession` /
 * `LlmBackendFactory` implementations behind the multi-runtime FFI seam
 * defined in `../llm-backend.h` (cutover plan M3).
 *
 * Per native/AGENTS.md §11 ("one managed library, one pipe, no
 * sidecar/subprocess/TCP") this backend is COMPILED INTO libelizainference
 * and exposes the SAME `eliza_inference_llm_stream_*` FFI pull contract —
 * it is the owned backend on Apple Silicon (mac first, iOS later), never a
 * child process. Apple Foundation Models stays an opportunistic out-of-
 * process adapter on the TS side and is NOT registered here.
 *
 * ── Two runtimes, one backend ─────────────────────────────────────────────
 *
 * The same `mlx-coreml` factory can serve a bundle through EITHER of two
 * Apple on-device runtimes, picked at open() time from the artifact present
 * under `<bundle_dir>/text/`:
 *
 *   • MLX  (PRIMARY)   — Apple's array framework for Apple Silicon. We drive
 *                        it through the C API `mlx-c` (ml-explore/mlx-c). The
 *                        text weights are an `mlx` weights dir (safetensors,
 *                        the mlx-lm convention) OR a `*.gguf` MLX reads via
 *                        `mlx_load_gguf`. Decode runs the transformer graph
 *                        on the Metal GPU stream with `mlx_quantized_matmul`
 *                        for the quantized weight banks,
 *                        `mlx_fast_scaled_dot_product_attention` for
 *                        attention, and `mlx_fast_rope` for position. The KV
 *                        cache is a pair of resident `mlx_array`s we append to
 *                        per step (host-side cache handle, GPU-resident data).
 *                        This is the preferred path: it gives us full control
 *                        of the sampler, supports the Gemma SWA/shared-KV
 *                        geometry, and matches mlx-lm's published Gemma graph.
 *
 *   • CoreML (ALTERNATE) — Apple's MLModel runtime, which can place the graph
 *                        on the ANE (Apple Neural Engine) as well as GPU/CPU.
 *                        We load a compiled `*.mlmodelc` / `*.mlpackage`
 *                        decoder and use the iOS-18 / macOS-15 **stateful**
 *                        prediction API (`MLState`) so the KV cache lives
 *                        inside CoreML and is updated in-place across decode
 *                        steps (no per-token KV tensor marshalled across the
 *                        ObjC boundary). CoreML needs Objective-C, which is
 *                        why this whole backend is a `.mm` translation unit.
 *
 *   TRADE-OFF (documented per the task brief): MLX is the primary path
 *   because it is the most flexible (custom sampler, exact Gemma geometry,
 *   speculative-decode-ready) and tracks mlx-lm directly; its decode runs on
 *   the GPU stream, not the ANE. CoreML's stateful MLModel can target the ANE
 *   for lower power on phones, but the decoder graph must be pre-compiled
 *   ahead of time, the sampler/KV layout is fixed by the converted model, and
 *   ANE placement of large attention graphs is fragile across OS revisions.
 *   We prefer MLX on mac/dev; CoreML is the alternate for ANE-bound iOS tiers
 *   once a stateful decoder package is published. open() selects MLX when an
 *   mlx weights dir / gguf is present, else falls back to the CoreML package.
 *
 * ── Build gate ────────────────────────────────────────────────────────────
 *
 * The REAL implementation is gated behind `ELIZA_ENABLE_MLX` (the CMake
 * define for this backend, per the cutover plan: LiteRT → ELIZA_ENABLE_LITERT,
 * MLX/CoreML → ELIZA_ENABLE_MLX) AND `__APPLE__`. When the gate is OFF the
 * translation unit includes NO Apple/MLX SDK headers, so it compiles on a
 * plain Linux host: `available()` returns false, `can_serve()` returns false,
 * and `open()` returns nullptr after setting `*out_error` ("not compiled in").
 * The default Linux build links it as a pure no-op and the selector skips it,
 * keeping the in-tree llama.cpp path.
 *
 * ── API research (cited; symbols verified, not invented) ──────────────────
 *
 *   MLX C API — ml-explore/mlx-c, `mlx/c/` headers, main @ 2026-06 (docs MLX C
 *   0.4.1, https://ml-explore.github.io/mlx-c/). Symbols used by the real path:
 *     - device.h : `mlx_device mlx_device_new_type(mlx_device_type, int)` with
 *                  `typedef enum { MLX_CPU, MLX_GPU } mlx_device_type;`
 *     - stream.h : `mlx_stream mlx_default_gpu_stream_new(void)`,
 *                  `mlx_stream mlx_default_cpu_stream_new(void)`
 *     - io.h     : `int mlx_load_safetensors(mlx_map_string_to_array*,
 *                  mlx_map_string_to_string*, const char* file, mlx_stream)`,
 *                  `int mlx_load_gguf(mlx_io_gguf*, const char* file, mlx_stream)`
 *     - array.h  : `mlx_array mlx_array_new_data(const void*, const int* shape,
 *                  int dim, mlx_dtype)`, `int mlx_array_eval(mlx_array)`,
 *                  `int mlx_array_item_int32(int32_t*, mlx_array)`,
 *                  `const float* mlx_array_data_float32(mlx_array)`,
 *                  `int mlx_array_free(mlx_array)`
 *     - ops.h    : `int mlx_quantized_matmul(mlx_array*, x, w, scales, biases,
 *                  bool transpose, mlx_optional_int group_size,
 *                  mlx_optional_int bits, const char* mode, mlx_stream)`,
 *                  `int mlx_matmul(...)`, `int mlx_softmax_axes(...)`,
 *                  `int mlx_argmax_axis(mlx_array*, a, int axis, bool, stream)`,
 *                  `int mlx_take(mlx_array*, a, indices, stream)`,
 *                  `int mlx_astype(...)`, `int mlx_concatenate(...)`
 *     - fast.h   : `int mlx_fast_scaled_dot_product_attention(mlx_array*, q, k,
 *                  v, float scale, const char* mask_mode, mlx_array mask,
 *                  mlx_array sinks, mlx_stream)`,
 *                  `int mlx_fast_rope(mlx_array*, x, int dims, bool traditional,
 *                  mlx_optional_float base, float scale, int offset,
 *                  mlx_array freqs, mlx_stream)`
 *   Gemma on MLX: ml-explore/mlx-lm (`mlx_lm/models/gemma*.py`) — the reference
 *   for the dense SWA + shared-KV + dual-head-dim graph this backend mirrors.
 *
 *   CoreML stateful KV-cache — Apple Core ML, MLState API, macOS 15 / iOS 18
 *   (WWDC24 "Bring your ML and AI models to Apple silicon"; coremltools
 *   Stateful Models guide, https://apple.github.io/coremltools/docs-guides/
 *   source/stateful-models.html). ObjC symbols used:
 *     - `+ (nullable instancetype)modelWithContentsOfURL:(NSURL*)url
 *        error:(NSError**)error;`  (and the compiled-model `compileModelAtURL:`)
 *     - `- (MLState*)newState;`    (creates zeroed KV state buffers; MLState is
 *        +new/-init UNAVAILABLE — only MLModel vends it)
 *     - `- (nullable id<MLFeatureProvider>)predictionFromFeatures:
 *        (id<MLFeatureProvider>)input usingState:(MLState*)state
 *        error:(NSError**)error;`  (the in-place stateful decode step)
 *   Apple's own "On-Device Llama 3.1 with Core ML" research post documents the
 *   prefill-then-stateful-decode loop this backend's MLX/CoreML paths follow.
 *
 * Every hardware-specific assumption that can only be confirmed on Apple
 * Silicon is marked `DEVICE-VERIFY` in the .mm. This header carries no SDK
 * dependency and is safe to include anywhere.
 */

#include "../llm-backend.h"

/* Free-function accessor returning the singleton `mlx-coreml` factory so the
 * selector (llm-backend-selector.cpp, wired separately) can register it via
 * `llm_backend_register(mlx_coreml_backend_factory())`. Defined in
 * mlx-coreml-backend.mm. Always returns a valid non-null static-lifetime
 * pointer — when the build gate is OFF the returned factory reports
 * available()/can_serve() == false and open() == nullptr ("not compiled in"),
 * so registering it unconditionally is safe. */
LlmBackendFactory * mlx_coreml_backend_factory();
