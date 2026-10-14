#pragma once
/*
 * eot-backend.h — per-op backend seam for end-of-turn scoring.
 *
 * A one-shot op (eliza_inference_llm_eot_score) that an accelerator backend can
 * serve when it ships an EOT artifact under `<bundle>/eot/`, while every other
 * op — and EOT itself when no artifact is present — stays on the in-tree ggml
 * causal-scoring path.
 *
 * The factory mirrors the FFI 1:1 and the FFI delegates without translation.
 * Selection reuses the shared eliza_backend::Registry (backend-registry.h):
 * ELIZA_EOT_BACKEND (per-op) then ELIZA_BACKEND (global) hard-select, else the
 * highest preference_rank among available()+can_serve() factories, else nullptr
 * (the ggml EOT-scoring path).
 */

#include "eliza-inference-ffi.h" /* EliInferenceContext fwd, ELIZA_* codes */

#include <cstddef>
#include <cstdint>

struct EliInferenceContext;

/* One factory per linked-in EOT runtime (e.g. LiteRT). */
struct EotBackendFactory {
    virtual ~EotBackendFactory() = default;

    /* Stable lower-case id, e.g. "litert". Matched case-insensitively against
     * ELIZA_EOT_BACKEND / ELIZA_BACKEND. */
    virtual const char * name() const = 0;

    /* Compiled in AND host deps present (the runtime + a GPU/NPU delegate).
     * Cheap — must not load a model. */
    virtual bool available() const = 0;

    /* The EOT artifact exists under `<bundle_dir>/eot/`. Cheap directory probe,
     * no model load. */
    virtual bool can_serve(const char * bundle_dir) const = 0;

    /* Platform-affinity rank (higher wins; the ggml path is the implicit rank 0).
     * An NPU-served EOT returns a high positive value; a GPU-delegate fallback a
     * lower positive value. */
    virtual int preference_rank() const { return 0; }

    /* Mirrors eliza_inference_llm_eot_score 1:1. Returns ELIZA_OK and writes the
     * next-token probabilities, or a negative ELIZA_* code with `*out_error`
     * heap-allocated for the caller to free. */
    virtual int eot_score(EliInferenceContext * ctx, const int32_t * token_ids, size_t num_tokens,
                          int32_t target_token_id, float * out_target_prob, int32_t * out_top_token,
                          float * out_top_prob, char ** out_error) = 0;
};

/* Register a factory (idempotent by name). */
void eot_backend_register(EotBackendFactory * factory);

/* Register every EOT backend compiled into THIS build (gated by the
 * -DELIZA_ENABLE_* options). Idempotent; called by eot_backend_select. */
void eot_backend_register_builtins();

/* Pick an EOT backend for the bundle at `bundle_dir`. nullptr + no error
 * => use the in-tree ggml EOT-scoring path. nullptr + *out_error => hard failure. */
EotBackendFactory * eot_backend_select(const char * bundle_dir, char ** out_error);
