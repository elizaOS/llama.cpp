#pragma once
/*
 * asr-backend.h — per-op backend seam for speech-to-text transcription.
 *
 * A one-shot op (eliza_inference_asr_transcribe) that an accelerator backend can
 * serve when it ships an ASR artifact under `<bundle>/asr/`, while every other
 * op — and ASR itself when no artifact is present — stays on the in-tree ggml
 * path.
 *
 * The factory mirrors the FFI 1:1 and the FFI delegates without translation.
 * Selection reuses the shared eliza_backend::Registry (backend-registry.h):
 * ELIZA_ASR_BACKEND (per-op) then ELIZA_BACKEND (global) hard-select, else the
 * highest preference_rank among available()+can_serve() factories, else nullptr
 * (the ggml ASR path).
 */

#include "eliza-inference-ffi.h" /* EliInferenceContext fwd, ELIZA_* codes */

#include <cstddef>

struct EliInferenceContext;

/* One factory per linked-in ASR runtime (e.g. LiteRT). */
struct AsrBackendFactory {
    virtual ~AsrBackendFactory() = default;

    /* Stable lower-case id, e.g. "litert". Matched case-insensitively against
     * ELIZA_ASR_BACKEND / ELIZA_BACKEND. */
    virtual const char * name() const = 0;

    /* Compiled in AND host deps present (the runtime + a GPU/NPU delegate).
     * Cheap — must not load a model. */
    virtual bool available() const = 0;

    /* The ASR artifact exists under `<bundle_dir>/asr/`. Cheap directory probe,
     * no model load. */
    virtual bool can_serve(const char * bundle_dir) const = 0;

    /* Platform-affinity rank (higher wins; the ggml path is the implicit rank 0).
     * An NPU-served ASR returns a high positive value; a GPU-delegate fallback a
     * lower positive value. */
    virtual int preference_rank() const { return 0; }

    /* Mirrors eliza_inference_asr_transcribe 1:1. Returns the number of bytes
     * written (excluding the terminator) on success, or a negative ELIZA_* code
     * with `*out_error` heap-allocated for the caller to free. */
    virtual int asr_transcribe(EliInferenceContext * ctx, const float * pcm, size_t n_samples,
                               int sample_rate_hz, char * out_text, size_t max_text_bytes,
                               char ** out_error) = 0;
};

/* Register a factory (idempotent by name). */
void asr_backend_register(AsrBackendFactory * factory);

/* Register every ASR backend compiled into THIS build (gated by the
 * -DELIZA_ENABLE_* options). Idempotent; called by asr_backend_select. */
void asr_backend_register_builtins();

/* Pick an ASR backend for the bundle at `bundle_dir`. nullptr + no error
 * => use the in-tree ggml ASR path. nullptr + *out_error => hard failure. */
AsrBackendFactory * asr_backend_select(const char * bundle_dir, char ** out_error);
