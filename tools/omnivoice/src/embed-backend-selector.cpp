/*
 * embed-backend-selector.cpp — registry + selection for the per-op embedding
 * backend seam. A thin instantiation of eliza_backend::Registry<EmbedBackendFactory>
 * (backend-registry.h) — the resolution logic is shared with every other
 * modality. Inert by default: with no -DELIZA_ENABLE_* embedding backend
 * compiled in, nothing registers and embed_backend_select() returns nullptr, so
 * eliza_inference_embed keeps the in-tree ggml encoder path.
 */

#include "embed-backend.h"
#include "backend-registry.h"

#include <mutex>

/* Gated factory accessor — declared only when the backend is compiled in. */
#ifdef ELIZA_ENABLE_LITERT
EmbedBackendFactory * litert_embed_backend_factory();
#endif

namespace {
eliza_backend::Registry<EmbedBackendFactory> g_registry;
std::once_flag                               g_builtins_once;
} // namespace

void embed_backend_register(EmbedBackendFactory * factory) {
    g_registry.register_factory(factory);
}

void embed_backend_register_builtins() {
    std::call_once(g_builtins_once, []() {
#ifdef ELIZA_ENABLE_LITERT
        embed_backend_register(litert_embed_backend_factory());
#endif
    });
}

EmbedBackendFactory * embed_backend_select(const char * bundle_dir, char ** out_error) {
    embed_backend_register_builtins();
    return g_registry.select("ELIZA_EMBED_BACKEND", "ELIZA_BACKEND", "embed",
                             bundle_dir, out_error);
}
