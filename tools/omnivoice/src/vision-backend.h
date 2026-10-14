#pragma once
/*
 * vision-backend.h — per-op backend seam for mmproj image description.
 *
 * A one-shot op (eliza_inference_describe_image) that an accelerator backend can
 * serve when it ships a vision artifact under `<bundle>/vision/`, while every
 * other op — and vision itself when no artifact is present — stays on the
 * in-tree ggml mmproj path.
 *
 * The factory mirrors the FFI 1:1 and the FFI delegates without translation.
 * Selection reuses the shared eliza_backend::Registry (backend-registry.h):
 * ELIZA_VISION_BACKEND (per-op) then ELIZA_BACKEND (global) hard-select, else
 * the highest preference_rank among available()+can_serve() factories, else
 * nullptr (the ggml mmproj path).
 */

#include "eliza-inference-ffi.h" /* EliInferenceContext fwd, ELIZA_* codes */

#include <cstddef>

struct EliInferenceContext;

/* One factory per linked-in vision runtime (e.g. LiteRT). */
struct VisionBackendFactory {
    virtual ~VisionBackendFactory() = default;

    /* Stable lower-case id, e.g. "litert". Matched case-insensitively against
     * ELIZA_VISION_BACKEND / ELIZA_BACKEND. */
    virtual const char * name() const = 0;

    /* Compiled in AND host deps present (the runtime + a GPU/NPU delegate).
     * Cheap — must not load a model. */
    virtual bool available() const = 0;

    /* The vision artifact exists under `<bundle_dir>/vision/`. Cheap directory
     * probe, no model load. */
    virtual bool can_serve(const char * bundle_dir) const = 0;

    /* Platform-affinity rank (higher wins; the ggml path is the implicit rank 0).
     * An NPU-served vision returns a high positive value; a GPU-delegate
     * fallback a lower positive value. */
    virtual int preference_rank() const { return 0; }

    /* Mirrors eliza_inference_describe_image 1:1. Returns the number of bytes
     * written (excluding the terminator) on success, or a negative ELIZA_* code
     * with `*out_error` heap-allocated for the caller to free. */
    virtual int describe_image(EliInferenceContext * ctx, const unsigned char * image_bytes,
                               size_t n_bytes, const char * mmproj_path, const char * prompt,
                               char * out_text, size_t max_text_bytes, char ** out_error) = 0;
};

/* Register a factory (idempotent by name). */
void vision_backend_register(VisionBackendFactory * factory);

/* Register every vision backend compiled into THIS build (gated by the
 * -DELIZA_ENABLE_* options). Idempotent; called by vision_backend_select. */
void vision_backend_register_builtins();

/* Pick a vision backend for the bundle at `bundle_dir`. nullptr + no error
 * => use the in-tree ggml mmproj path. nullptr + *out_error => hard failure. */
VisionBackendFactory * vision_backend_select(const char * bundle_dir, char ** out_error);
