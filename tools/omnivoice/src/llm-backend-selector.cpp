/*
 * llm-backend-selector.cpp — registry + selection for the multi-runtime
 * streaming-LLM seam (cutover plan M3).
 *
 * On a default build (no -DELIZA_ENABLE_* gate) NO alternate backend is
 * registered, so llm_backend_select() always returns nullptr and the FFI keeps
 * the in-tree llama.cpp path. The seam is therefore inert-by-default: the
 * library behaves exactly as before until an accelerator backend is compiled in.
 */

#include "llm-backend.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

/* Gated backend factory accessors. Declared only when the matching backend is
 * compiled in; register_builtins() calls them under the same gate. Keeping the
 * declarations gated means the default build has no unresolved symbols. */
#ifdef ELIZA_ENABLE_LITERT_LM
LlmBackendFactory * litert_backend_factory();
#endif
#if defined(ELIZA_ENABLE_MLX) && defined(__APPLE__)
LlmBackendFactory * mlx_coreml_backend_factory();
#endif

namespace {

std::mutex                       g_reg_mutex;
std::vector<LlmBackendFactory *> g_factories;
std::once_flag                   g_builtins_once;

/* Heap-allocate an error string with malloc so the caller can release it with
 * eliza_inference_free_string() (which calls free()), matching the FFI contract. */
char * dup_error(const std::string & msg) {
    char * out = (char *) std::malloc(msg.size() + 1);
    if (out) std::memcpy(out, msg.c_str(), msg.size() + 1);
    return out;
}

bool iequals(const char * a, const char * b) {
    if (!a || !b) return false;
    while (*a && *b) {
        if (std::tolower((unsigned char) *a) != std::tolower((unsigned char) *b)) {
            return false;
        }
        ++a;
        ++b;
    }
    return *a == *b;
}

bool is_llamacpp_name(const char * s) {
    return iequals(s, "llama.cpp") || iequals(s, "llamacpp") || iequals(s, "llama");
}

} // namespace

void llm_backend_register(LlmBackendFactory * factory) {
    if (!factory) return;
    std::lock_guard<std::mutex> lock(g_reg_mutex);
    for (LlmBackendFactory * f : g_factories) {
        if (iequals(f->name(), factory->name())) return; /* idempotent by name */
    }
    g_factories.push_back(factory);
}

void llm_backend_register_builtins() {
    std::call_once(g_builtins_once, []() {
#ifdef ELIZA_ENABLE_LITERT_LM
        llm_backend_register(litert_backend_factory());
#endif
#if defined(ELIZA_ENABLE_MLX) && defined(__APPLE__)
        llm_backend_register(mlx_coreml_backend_factory());
#endif
    });
}

LlmBackendFactory * llm_backend_select(const char * bundle_dir,
                                       const eliza_llm_stream_config_t * /*cfg*/,
                                       char ** out_error) {
    llm_backend_register_builtins();

    /* (1) ELIZA_LLM_BACKEND env: a HARD select. */
    const char * forced = std::getenv("ELIZA_LLM_BACKEND");
    if (forced && forced[0] != '\0') {
        if (is_llamacpp_name(forced)) {
            return nullptr; /* force the in-tree path, not an error */
        }
        std::lock_guard<std::mutex> lock(g_reg_mutex);
        for (LlmBackendFactory * f : g_factories) {
            if (!iequals(f->name(), forced)) continue;
            if (!f->available()) {
                if (out_error) {
                    *out_error = dup_error(
                        std::string("[libelizainference] ELIZA_LLM_BACKEND=") + forced +
                        " is not available in this build/host");
                }
                return nullptr;
            }
            if (!f->can_serve(bundle_dir)) {
                if (out_error) {
                    *out_error = dup_error(
                        std::string("[libelizainference] ELIZA_LLM_BACKEND=") + forced +
                        " cannot serve the bundle at " +
                        (bundle_dir ? bundle_dir : "(null)"));
                }
                return nullptr;
            }
            return f;
        }
        if (out_error) {
            *out_error = dup_error(
                std::string("[libelizainference] ELIZA_LLM_BACKEND=") + forced +
                " is not a registered backend");
        }
        return nullptr;
    }

    /* (2) Auto-select: the highest preference_rank among available + can_serve.
     * The in-tree llama.cpp path is the implicit rank-0 fallback, so an
     * accelerator backend only wins when it returns a positive rank AND can
     * serve this bundle. */
    std::lock_guard<std::mutex> lock(g_reg_mutex);
    LlmBackendFactory * best      = nullptr;
    int                 best_rank = 0;
    for (LlmBackendFactory * f : g_factories) {
        if (!f->available()) continue;
        if (!f->can_serve(bundle_dir)) continue;
        const int rank = f->preference_rank();
        if (rank > best_rank) {
            best_rank = rank;
            best      = f;
        }
    }
    return best; /* nullptr => in-tree llama.cpp */
}
