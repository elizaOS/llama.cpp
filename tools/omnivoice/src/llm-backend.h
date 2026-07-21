#pragma once
/*
 * llm-backend.h — multi-runtime streaming-LLM backend seam (cutover plan M3).
 *
 * The libelizainference streaming-LLM FFI (`eliza_inference_llm_stream_*`) is
 * ONE pipe that can be driven by more than one in-process inference runtime:
 *
 *   - llama.cpp   — the default / reference backend (CPU / CUDA / Vulkan-Mali-
 *                   Adreno / Metal). Always present; the in-tree code path.
 *   - LiteRT-LM   — Android NPU (Tensor / Qualcomm QNN / MediaTek NeuroPilot),
 *                   optionally desktop/iOS GPU. Gated -DELIZA_ENABLE_LITERT.
 *   - CoreML/MLX  — Apple Silicon (mac first, iOS later). Gated -DELIZA_ENABLE_MLX.
 *
 * Per native/AGENTS.md §11 (reinterpreted by the Gemma-4 cutover): "one managed
 * library, one pipe, no sidecar/subprocess/TCP." LiteRT-LM and MLX are
 * EMBEDDABLE in-process C++ libraries linked INTO libelizainference and exposed
 * behind the SAME FFI streaming symbols — never a child process or TCP server.
 * (AICore / Apple Foundation stay opportunistic out-of-process adapters on the
 * TS side, not owned backends — they are NOT registered here.)
 *
 * A backend supplies:
 *   - LlmBackendSession  — the per-generation streaming session, mirroring the
 *                          FFI pull contract (prefill -> next* -> close) 1:1 so
 *                          the FFI functions delegate without translation.
 *   - LlmBackendFactory  — names the runtime, reports availability + bundle fit,
 *                          and opens sessions.
 *
 * `llm_backend_select()` picks a backend at `_open` time from the platform, the
 * bundle contents, the build flags, and the `ELIZA_LLM_BACKEND` env override.
 * When it returns nullptr (and no error) the FFI keeps the in-tree llama.cpp
 * path — so a build with no alternate backend behaves exactly as before.
 */

#include "eliza-inference-ffi.h" /* eliza_llm_stream_config_t, EliInferenceContext fwd */

#include <cstddef>
#include <cstdint>

/* Defined in the FFI translation unit. Opaque to backends — a backend reaches
 * the resident model/bundle through the accessors below, not the struct. */
struct EliInferenceContext;

/* The bundle directory the context was opened against. A backend's open()
 * resolves its own artifact under this root (e.g. `<dir>/text/*.litertlm`,
 * `<dir>/text/*.mlpackage`) — the ONLY supported way to read the bundle path,
 * since the struct itself is opaque here. Returns nullptr when ctx is null.
 * Defined in eliza-inference-ffi.cpp; the pointer is owned by the context and
 * stays valid for the session's lifetime. */
const char * llm_backend_context_bundle_dir(const EliInferenceContext * ctx);

/* ---- Per-generation streaming session ------------------------------------ *
 *
 * Lifetime: created by LlmBackendFactory::open(), destroyed via `delete` on the
 * FFI `_close` path. Every method mirrors the matching FFI entry point so the
 * FFI can `return session->method(...)` with no argument translation. Status
 * conventions are identical to the FFI: >= 0 on success, the negative `ELIZA_*`
 * constants on failure, with `*out_error` heap-allocated for the caller to free.
 */
struct LlmBackendSession {
    virtual ~LlmBackendSession() = default;

    /* Mirrors eliza_inference_llm_stream_prefill. Copies the tokens it needs. */
    virtual int prefill(const int32_t * token_ids, size_t num_tokens,
                        char ** out_error) = 0;

    /* Mirrors eliza_inference_llm_stream_next. Returns 0 (more output), 1 (final
     * step — EOS / cap), or a negative ELIZA_* code (ELIZA_ERR_CANCELLED on
     * cancel). `drafter_*_out` carry per-step speculative stats (0 when the
     * backend has no drafter). */
    virtual int next(int32_t * tokens_out, size_t tokens_cap,
                     size_t * num_tokens_out, char * text_out, size_t text_cap,
                     int32_t * drafter_drafted_out, int32_t * drafter_accepted_out,
                     char ** out_error) = 0;

    /* Mirrors eliza_inference_llm_stream_cancel. Publishes a flag an in-flight
     * next() checks at a step boundary; safe to call from another thread.
     * Returns ELIZA_OK whether or not a pass was running. */
    virtual int cancel() = 0;

    /* Mirrors eliza_inference_llm_stream_reset: clear KV + sampler/counters so
     * the next prefill starts a fresh prompt on the same warm session. */
    virtual int reset() = 0;

    /* Mirrors eliza_inference_llm_stream_reset_keep: keep the first `n_keep`
     * tokens of state resident and drop the rest. Returns the n_keep actually
     * applied (>= 0, may be clamped / 0 on a full-reset fallback). A backend
     * that cannot do prefix reuse MUST fall back to a full reset and return 0 —
     * never an error. */
    virtual int reset_keep(int32_t n_keep) = 0;

    /* Slot KV persistence — optional. Default: not supported. */
    virtual int save_slot(const char * /*filename*/, char ** /*out_error*/) {
        return ELIZA_ERR_INVALID_ARG;
    }
    virtual int restore_slot(const char * /*filename*/, char ** /*out_error*/) {
        return ELIZA_ERR_INVALID_ARG;
    }
};

/* ---- Backend factory (one per linked-in runtime) ------------------------- */
struct LlmBackendFactory {
    virtual ~LlmBackendFactory() = default;

    /* Stable lower-case id: "llama.cpp", "litert-lm", "mlx-coreml". Matched
     * case-insensitively against ELIZA_LLM_BACKEND. */
    virtual const char * name() const = 0;

    /* True only when this backend is compiled in AND its runtime dependencies
     * are present on THIS host (the NPU delegate / Metal device / the linked
     * lib). A scaffold whose build gate is OFF returns false. Cheap — must not
     * load a model. */
    virtual bool available() const = 0;

    /* True when this backend can serve the bundle at `bundle_dir` — i.e. the
     * backend-specific artifact exists (e.g. `text/*.litertlm`, `text/*.mlpackage`).
     * Cheap directory probe, no model load. */
    virtual bool can_serve(const char * bundle_dir) const = 0;

    /* Platform-affinity rank used to order candidates when several can serve the
     * same bundle and no env override is set. Higher wins. The in-tree llama.cpp
     * path is rank 0 (the implicit fallback); an accelerator backend that is the
     * preferred path on this device returns a positive value. */
    virtual int preference_rank() const { return 0; }

    /* Create a streaming session for (ctx, cfg). Returns nullptr + `*out_error`
     * on failure. The returned session is owned by the caller (FFI `_close`
     * deletes it). */
    virtual LlmBackendSession * open(EliInferenceContext * ctx,
                                     const eliza_llm_stream_config_t * cfg,
                                     char ** out_error) = 0;
};

/* ---- Registry + selection ------------------------------------------------ *
 *
 * Backends register their singleton factory (idempotent; the registry does not
 * take ownership — factories are static-lifetime singletons). The FFI
 * translation unit calls llm_backend_register_builtins() once to register every
 * compiled-in backend, then calls llm_backend_select() per `_open`.
 */

/* Register a factory (idempotent by name). Safe to call from static init. */
void llm_backend_register(LlmBackendFactory * factory);

/* Register every backend compiled into THIS build (gated by the -DELIZA_ENABLE_*
 * CMake options). Idempotent; call once at first `_open`. Defined in
 * llm-backend-selector.cpp; the gated backends self-register via their headers. */
void llm_backend_register_builtins();

/* Pick a backend for the bundle at `bundle_dir` with `cfg`. Resolution order:
 *
 *   1. ELIZA_LLM_BACKEND env (exact, case-insensitive backend name) — a HARD
 *      select. "llama.cpp" / "llamacpp" forces the in-tree path (returns
 *      nullptr, no error). Any other name that is not registered+available, or
 *      cannot serve the bundle, is a hard error: returns nullptr AND sets
 *      `*out_error` so the FFI aborts rather than silently using llama.cpp.
 *
 *   2. No env override: among registered backends that are available() AND
 *      can_serve(bundle_dir), pick the highest preference_rank(). If none
 *      qualifies, return nullptr (use the in-tree llama.cpp path).
 *
 * A nullptr return with `*out_error == nullptr` means "use the in-tree llama.cpp
 * path" — NOT an error. A nullptr return with `*out_error != nullptr` is a hard
 * failure the caller must propagate.
 */
LlmBackendFactory * llm_backend_select(const char * bundle_dir,
                                       const eliza_llm_stream_config_t * cfg,
                                       char ** out_error);
