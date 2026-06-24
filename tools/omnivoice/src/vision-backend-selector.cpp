/*
 * vision-backend-selector.cpp — registry + selection for the per-op vision
 * backend seam. A thin instantiation of eliza_backend::Registry<VisionBackendFactory>
 * (backend-registry.h) — the resolution logic is shared with every other
 * modality. Inert by default: no -DELIZA_ENABLE_* vision backend is compiled in
 * (none exists yet), so nothing registers and vision_backend_select() returns
 * nullptr, so eliza_inference_describe_image keeps the in-tree ggml mmproj path.
 */

#include "vision-backend.h"
#include "backend-registry.h"

#include <mutex>

namespace {
eliza_backend::Registry<VisionBackendFactory> g_registry;
std::once_flag                                g_builtins_once;
} // namespace

void vision_backend_register(VisionBackendFactory * factory) {
    g_registry.register_factory(factory);
}

void vision_backend_register_builtins() {
    std::call_once(g_builtins_once, []() {
        /* No vision backend exists yet — the seam stays inert. */
    });
}

VisionBackendFactory * vision_backend_select(const char * bundle_dir, char ** out_error) {
    vision_backend_register_builtins();
    return g_registry.select("ELIZA_VISION_BACKEND", "ELIZA_BACKEND", "vision",
                             bundle_dir, out_error);
}
