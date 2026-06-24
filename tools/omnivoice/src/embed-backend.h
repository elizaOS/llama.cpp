#pragma once
/*
 * embed-backend.h — per-op backend seam for pooled text embeddings.
 *
 * The first per-op generalization of the M3 streaming-LLM seam: a one-shot op
 * (eliza_inference_embed) that an accelerator backend can serve when it ships an
 * embedding artifact under `<bundle>/embedding/`, while every other op — and
 * embedding itself when no artifact is present — stays on the in-tree ggml path.
 *
 * Embedding is the natural first LiteRT/NPU target: a static-shape, encoder-only
 * forward with no streaming/KV/sampler, so the factory mirrors the FFI 1:1 and
 * the FFI delegates without translation. Selection reuses the shared
 * eliza_backend::Registry (backend-registry.h): ELIZA_EMBED_BACKEND (per-op) then
 * ELIZA_BACKEND (global) hard-select, else the highest preference_rank among
 * available()+can_serve() factories, else nullptr (the ggml encoder path).
 */

#include "eliza-inference-ffi.h" /* EliInferenceContext fwd, ELIZA_* codes */

#include <cstddef>

struct EliInferenceContext;

/* One factory per linked-in embedding runtime (e.g. LiteRT). */
struct EmbedBackendFactory {
    virtual ~EmbedBackendFactory() = default;

    /* Stable lower-case id, e.g. "litert". Matched case-insensitively against
     * ELIZA_EMBED_BACKEND / ELIZA_BACKEND. */
    virtual const char * name() const = 0;

    /* Compiled in AND host deps present (the LiteRT runtime + a GPU/NPU
     * delegate). Cheap — must not load a model. */
    virtual bool available() const = 0;

    /* The embedding artifact exists under `<bundle_dir>/embedding/`. Cheap
     * directory probe, no model load. */
    virtual bool can_serve(const char * bundle_dir) const = 0;

    /* Platform-affinity rank (higher wins; the ggml path is the implicit rank 0).
     * An NPU-served embedding returns a high positive value; a GPU-delegate
     * fallback a lower positive value. */
    virtual int preference_rank() const { return 0; }

    /* Mirrors eliza_inference_embed 1:1. Returns ELIZA_OK and writes `*out_dim`
     * floats into out_embedding (>= out_capacity required), or a negative ELIZA_*
     * code with `*out_error` heap-allocated for the caller to free. */
    virtual int embed(EliInferenceContext * ctx, const char * text, size_t text_len,
                      int pooling, float * out_embedding, size_t out_capacity,
                      int * out_dim, char ** out_error) = 0;
};

/* Register a factory (idempotent by name). */
void embed_backend_register(EmbedBackendFactory * factory);

/* Register every embedding backend compiled into THIS build (gated by the
 * -DELIZA_ENABLE_* options). Idempotent; called by embed_backend_select. */
void embed_backend_register_builtins();

/* Pick an embedding backend for the bundle at `bundle_dir`. nullptr + no error
 * => use the in-tree ggml encoder path. nullptr + *out_error => hard failure. */
EmbedBackendFactory * embed_backend_select(const char * bundle_dir, char ** out_error);
