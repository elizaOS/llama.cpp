#pragma once
// backend.h: shared GGML backend pair from llama.cpp.
// The merged build no longer owns its own backend init — it accepts
// the pair the host llama_context already constructed and shares it
// for the lifetime of the ov_context.  BackendPair fields are
// BORROWED when sourced via omnivoice_backend_from_llama(): omnivoice
// never calls ggml_backend_free() on borrowed handles.
//
// Two construction paths:
//   backend_init_auto()            — standalone: calls ggml_backend_load_all(),
//                                    scans devices, returns a BackendPairOwned
//                                    whose handles omnivoice allocated and owns.
//                                    Used by ov_init() when no llama_context is
//                                    available.
//   omnivoice_backend_from_llama() — extracts GPU + CPU handles from the
//                                    scheduler owned by an existing llama_context.
//                                    Returns a BackendPairOwned with owns_* = false
//                                    so backend_auto_free() is a no-op.
//                                    C++ only; requires llama-context.h.
//
// backend_sched_new(): unchanged — creates a ggml_backend_sched_t from any
// BackendPair regardless of how it was constructed.

#include "ggml-backend.h"
#include "ov-error.h"

#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// BackendPair: a GPU (or CPU fallback) handle plus a CPU handle.
// Ownership is tracked separately in BackendPairOwned below.
// ---------------------------------------------------------------------------
struct BackendPair {
    ggml_backend_t backend;      // GPU (Metal / CUDA / Vulkan) or CPU when no GPU
    ggml_backend_t cpu_backend;  // always CPU
    bool           has_gpu;
};

// ---------------------------------------------------------------------------
// BackendPairOwned: BackendPair + ownership flags.
// When owns_gpu / owns_cpu are true, backend_auto_free() calls
// ggml_backend_free() on the respective handle.  When false (borrowed from
// a llama_context), backend_auto_free() is a no-op.
// ---------------------------------------------------------------------------
struct BackendPairOwned {
    BackendPair bp       = {};
    bool        owns_gpu = false;
    bool        owns_cpu = false;
};

// ---------------------------------------------------------------------------
// backend_init_auto: standalone backend selection for ov_init().
// Calls ggml_backend_load_all() (idempotent when the merged llama.cpp build
// already called it), then walks registered devices to pick the first
// discrete GPU and the CPU fallback.  Respects GGML_BACKEND env override.
//
// Returns a BackendPairOwned with .bp.backend == NULL on failure.
// ---------------------------------------------------------------------------
static BackendPairOwned backend_init_auto(const char * label) {
    BackendPairOwned result = {};

    ggml_backend_load_all();

    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;

    // GGML_BACKEND env var: force a specific device by name.
    const char * force_name = getenv("GGML_BACKEND");
    if (force_name) {
        ggml_backend_t forced = ggml_backend_init_by_name(force_name, nullptr);
        if (!forced) {
            ov_log(OV_LOG_ERROR, "[Load] %s: GGML_BACKEND=%s not found", label, force_name);
            return result;
        }
        ggml_backend_dev_t fdev  = ggml_backend_get_device(forced);
        enum ggml_backend_dev_type ftype = fdev ? ggml_backend_dev_type(fdev)
                                                 : GGML_BACKEND_DEVICE_TYPE_CPU;
        if (ftype == GGML_BACKEND_DEVICE_TYPE_CPU) {
            cpu = forced;
        } else {
            gpu = forced;
        }
    } else {
        // Walk all registered devices; pick first GPU and first CPU.
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev   = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
            if ((dtype == GGML_BACKEND_DEVICE_TYPE_GPU ||
                 dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) && !gpu) {
                gpu = ggml_backend_dev_init(dev, nullptr);
            } else if (dtype == GGML_BACKEND_DEVICE_TYPE_CPU && !cpu) {
                cpu = ggml_backend_dev_init(dev, nullptr);
            }
            if (gpu && cpu) break;
        }
    }

    // CPU is mandatory.
    if (!cpu) {
        cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!cpu) {
            ov_log(OV_LOG_ERROR, "[Load] %s: failed to init CPU backend", label);
            if (gpu) { ggml_backend_free(gpu); }
            return result;
        }
    }

    result.owns_gpu  = (gpu != nullptr);
    result.owns_cpu  = true;
    result.bp.backend     = gpu ? gpu : cpu;
    result.bp.cpu_backend = cpu;
    result.bp.has_gpu     = (gpu != nullptr);

    ov_log(OV_LOG_INFO, "[Load] %s backend: %s (has_gpu=%d)",
           label, ggml_backend_name(result.bp.backend), (int)result.bp.has_gpu);
    return result;
}

// ---------------------------------------------------------------------------
// backend_auto_free: release backends that omnivoice owns.
// Borrowed handles (owns_* == false) are left untouched.
// ---------------------------------------------------------------------------
static void backend_auto_free(BackendPairOwned bpo) {
    if (bpo.owns_gpu && bpo.bp.has_gpu && bpo.bp.backend) {
        ggml_backend_free(bpo.bp.backend);
    }
    if (bpo.owns_cpu && bpo.bp.cpu_backend &&
        bpo.bp.cpu_backend != bpo.bp.backend) {
        ggml_backend_free(bpo.bp.cpu_backend);
    }
}

// ---------------------------------------------------------------------------
// backend_release: compatibility shim retained so that any existing call
// sites outside omnivoice.cpp compile unchanged.  The actual ownership is
// tracked via BackendPairOwned / backend_auto_free(); this function is a
// no-op because ov_free() now calls backend_auto_free() instead.
// ---------------------------------------------------------------------------
static inline void backend_release(ggml_backend_t /*backend*/, ggml_backend_t /*cpu_backend*/) {
    // no-op: ownership is handled by backend_auto_free(ov_context::owned_bp)
}

// ---------------------------------------------------------------------------
// omnivoice_backend_from_llama: extract the backend pair owned by an
// already-initialised llama_context.  C++ only — requires llama-context.h.
//
// Algorithm:
//   1. Retrieve the scheduler via llama_context::get_sched().
//   2. Walk ggml_backend_sched_get_n_backends() slots.
//   3. Classify each by ggml_backend_dev_type(): pick first GPU and CPU.
//   4. Return a BackendPairOwned with owns_* = false (borrowed).
//
// The returned pair is valid for as long as the llama_context is alive.
// ---------------------------------------------------------------------------
#ifdef __cplusplus

#include "llama-context.h"

inline BackendPairOwned omnivoice_backend_from_llama(llama_context * ctx) {
    BackendPairOwned result = {};

    if (!ctx) {
        ov_log(OV_LOG_ERROR, "[OmniVoice] omnivoice_backend_from_llama: ctx is NULL");
        return result;
    }

    // llama_context::get_sched() is a C++ method defined in llama-context.h.
    // It returns the ggml_backend_sched_t that llama_context owns.
    ggml_backend_sched_t sched = ctx->get_sched();
    if (!sched) {
        ov_log(OV_LOG_ERROR, "[OmniVoice] omnivoice_backend_from_llama: no scheduler on context");
        return result;
    }

    int            n_backends = ggml_backend_sched_get_n_backends(sched);
    ggml_backend_t gpu        = nullptr;
    ggml_backend_t cpu        = nullptr;

    for (int i = 0; i < n_backends; ++i) {
        ggml_backend_t     b    = ggml_backend_sched_get_backend(sched, i);
        ggml_backend_dev_t dev  = ggml_backend_get_device(b);
        if (!dev) continue;
        enum ggml_backend_dev_type dtype = ggml_backend_dev_type(dev);
        if ((dtype == GGML_BACKEND_DEVICE_TYPE_GPU ||
             dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) && !gpu) {
            gpu = b;
        } else if (dtype == GGML_BACKEND_DEVICE_TYPE_CPU && !cpu) {
            cpu = b;
        }
    }

    if (!cpu) {
        ov_log(OV_LOG_ERROR,
               "[OmniVoice] omnivoice_backend_from_llama: no CPU backend in context scheduler");
        return result;
    }

    // Borrowed — omnivoice must NOT free these handles.
    result.owns_gpu       = false;
    result.owns_cpu       = false;
    result.bp.backend     = gpu ? gpu : cpu;
    result.bp.cpu_backend = cpu;
    result.bp.has_gpu     = (gpu != nullptr);

    ov_log(OV_LOG_INFO, "[OmniVoice] sharing llama_context backend: %s (has_gpu=%d)",
           ggml_backend_name(result.bp.backend), (int)result.bp.has_gpu);
    return result;
}

#endif // __cplusplus

// ---------------------------------------------------------------------------
// backend_sched_new: create a ggml_backend_sched_t from a BackendPair.
// max_nodes: graph size hint (4096 for small models, 8192 for large).
// When a GPU is present, use the GPU device's host buffer type for the CPU
// slot — pinned memory reduces copies and lets the scheduler keep more ops
// on GPU.  Unchanged from the pre-patch version.
// ---------------------------------------------------------------------------
static ggml_backend_sched_t backend_sched_new(BackendPair bp, int max_nodes) {
    ggml_backend_t             backends[2] = { bp.backend, bp.cpu_backend };
    ggml_backend_buffer_type_t bufts[2]    = { nullptr, nullptr };
    int                        n           = (bp.backend == bp.cpu_backend) ? 1 : 2;

    bufts[0] = ggml_backend_get_default_buffer_type(bp.backend);
    if (n == 2) {
        ggml_backend_dev_t         gpu_dev   = ggml_backend_get_device(bp.backend);
        ggml_backend_buffer_type_t host_buft = gpu_dev ? ggml_backend_dev_host_buffer_type(gpu_dev) : nullptr;
        bufts[1] = host_buft ? host_buft : ggml_backend_get_default_buffer_type(bp.cpu_backend);
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, bufts, n, max_nodes, false, true);
    if (!sched) {
        ov_log(OV_LOG_ERROR, "[Load] failed to create scheduler");
        return nullptr;
    }
    return sched;
}
