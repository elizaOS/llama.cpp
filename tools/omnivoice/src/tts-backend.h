#pragma once
/*
 * tts-backend.h — per-op backend seam for text-to-speech synthesis.
 *
 * A one-shot op (eliza_inference_tts_synthesize) that an accelerator backend can
 * serve when it ships a TTS artifact under `<bundle>/tts/`, while every other
 * op — and TTS itself when no artifact is present — stays on the in-tree
 * OmniVoice/ggml path.
 *
 * The factory mirrors the FFI 1:1 and the FFI delegates without translation.
 * Selection reuses the shared eliza_backend::Registry (backend-registry.h):
 * ELIZA_TTS_BACKEND (per-op) then ELIZA_BACKEND (global) hard-select, else the
 * highest preference_rank among available()+can_serve() factories, else nullptr
 * (the in-tree OmniVoice path).
 */

#include "eliza-inference-ffi.h" /* EliInferenceContext fwd, ELIZA_* codes */

#include <cstddef>

struct EliInferenceContext;

/* One factory per linked-in TTS runtime (e.g. LiteRT). */
struct TtsBackendFactory {
    virtual ~TtsBackendFactory() = default;

    /* Stable lower-case id, e.g. "litert". Matched case-insensitively against
     * ELIZA_TTS_BACKEND / ELIZA_BACKEND. */
    virtual const char * name() const = 0;

    /* Compiled in AND host deps present (the runtime + a GPU/NPU delegate).
     * Cheap — must not load a model. */
    virtual bool available() const = 0;

    /* The TTS artifact exists under `<bundle_dir>/tts/`. Cheap directory probe,
     * no model load. */
    virtual bool can_serve(const char * bundle_dir) const = 0;

    /* Platform-affinity rank (higher wins; the ggml path is the implicit rank 0).
     * An NPU-served TTS returns a high positive value; a GPU-delegate fallback a
     * lower positive value. */
    virtual int preference_rank() const { return 0; }

    /* Mirrors eliza_inference_tts_synthesize 1:1. Returns the number of fp32 PCM
     * samples actually written (>= 0) on success, or a negative ELIZA_* code with
     * `*out_error` heap-allocated for the caller to free. */
    virtual int tts_synthesize(EliInferenceContext * ctx, const char * text, size_t text_len,
                               const char * speaker_preset_id, float * out_pcm,
                               size_t max_samples, char ** out_error) = 0;
};

/* Register a factory (idempotent by name). */
void tts_backend_register(TtsBackendFactory * factory);

/* Register every TTS backend compiled into THIS build (gated by the
 * -DELIZA_ENABLE_* options). Idempotent; called by tts_backend_select. */
void tts_backend_register_builtins();

/* Pick a TTS backend for the bundle at `bundle_dir`. nullptr + no error
 * => use the in-tree OmniVoice path. nullptr + *out_error => hard failure. */
TtsBackendFactory * tts_backend_select(const char * bundle_dir, char ** out_error);
