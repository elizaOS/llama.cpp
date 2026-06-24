/*
 * tts-backend-selector.cpp — registry + selection for the per-op TTS backend
 * seam. A thin instantiation of eliza_backend::Registry<TtsBackendFactory>
 * (backend-registry.h) — the resolution logic is shared with every other
 * modality. Inert by default: no -DELIZA_ENABLE_* TTS backend is compiled in
 * (none exists yet), so nothing registers and tts_backend_select() returns
 * nullptr, so eliza_inference_tts_synthesize keeps the in-tree OmniVoice path.
 */

#include "tts-backend.h"
#include "backend-registry.h"

#include <mutex>

namespace {
eliza_backend::Registry<TtsBackendFactory> g_registry;
std::once_flag                             g_builtins_once;
} // namespace

void tts_backend_register(TtsBackendFactory * factory) {
    g_registry.register_factory(factory);
}

void tts_backend_register_builtins() {
    std::call_once(g_builtins_once, []() {
        /* No TTS backend exists yet — the seam stays inert. */
    });
}

TtsBackendFactory * tts_backend_select(const char * bundle_dir, char ** out_error) {
    tts_backend_register_builtins();
    return g_registry.select("ELIZA_TTS_BACKEND", "ELIZA_BACKEND", "tts",
                             bundle_dir, out_error);
}
