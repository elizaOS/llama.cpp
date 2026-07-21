/*
 * asr-backend-selector.cpp — registry + selection for the per-op ASR backend
 * seam. A thin instantiation of eliza_backend::Registry<AsrBackendFactory>
 * (backend-registry.h) — the resolution logic is shared with every other
 * modality. Inert by default: no -DELIZA_ENABLE_* ASR backend is compiled in
 * (none exists yet), so nothing registers and asr_backend_select() returns
 * nullptr, so eliza_inference_asr_transcribe keeps the in-tree ggml path.
 */

#include "asr-backend.h"
#include "backend-registry.h"

#include <mutex>

namespace {
eliza_backend::Registry<AsrBackendFactory> g_registry;
std::once_flag                             g_builtins_once;
} // namespace

void asr_backend_register(AsrBackendFactory * factory) {
    g_registry.register_factory(factory);
}

void asr_backend_register_builtins() {
    std::call_once(g_builtins_once, []() {
        /* No ASR backend exists yet — the seam stays inert. */
    });
}

AsrBackendFactory * asr_backend_select(const char * bundle_dir, char ** out_error) {
    asr_backend_register_builtins();
    return g_registry.select("ELIZA_ASR_BACKEND", "ELIZA_BACKEND", "asr",
                             bundle_dir, out_error);
}
