/*
 * eot-backend-selector.cpp — registry + selection for the per-op end-of-turn
 * scoring backend seam. A thin instantiation of
 * eliza_backend::Registry<EotBackendFactory> (backend-registry.h) — the
 * resolution logic is shared with every other modality. Inert by default: no
 * -DELIZA_ENABLE_* EOT backend is compiled in (none exists yet), so nothing
 * registers and eot_backend_select() returns nullptr, so
 * eliza_inference_llm_eot_score keeps the in-tree ggml causal-scoring path.
 */

#include "eot-backend.h"
#include "backend-registry.h"

#include <mutex>

namespace {
eliza_backend::Registry<EotBackendFactory> g_registry;
std::once_flag                             g_builtins_once;
} // namespace

void eot_backend_register(EotBackendFactory * factory) {
    g_registry.register_factory(factory);
}

void eot_backend_register_builtins() {
    std::call_once(g_builtins_once, []() {
        /* No EOT backend exists yet — the seam stays inert. */
    });
}

EotBackendFactory * eot_backend_select(const char * bundle_dir, char ** out_error) {
    eot_backend_register_builtins();
    return g_registry.select("ELIZA_EOT_BACKEND", "ELIZA_BACKEND", "eot",
                             bundle_dir, out_error);
}
