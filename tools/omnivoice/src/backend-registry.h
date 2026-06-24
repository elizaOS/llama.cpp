#pragma once
/*
 * backend-registry.h — generic per-modality backend registry + selection.
 *
 * Factored out of the M3 streaming-LLM seam (llm-backend-selector.cpp) so EVERY
 * on-device modality (embed, asr, tts, vision, vad, wakeword, speaker, diarizer,
 * eot, …) reuses ONE resolution implementation instead of copy-pasting it. A
 * modality declares a small factory interface with the four common probes
 * (name / available / can_serve / preference_rank) plus its own forward method,
 * instantiates `eliza_backend::Registry<ThatFactory>`, and selects with the
 * shared logic below:
 *
 *   1. `ELIZA_<MOD>_BACKEND` env (per-op) → else `ELIZA_BACKEND` (global) — a
 *      HARD select. An in-tree name ("llama.cpp"/"ggml"/"default") forces the
 *      ggml path (returns nullptr, no error). Any other name that is not
 *      registered+available or cannot serve the bundle is a hard error
 *      (nullptr + *out_error).
 *   2. No override: among registered factories that are available() AND
 *      can_serve(bundle_dir), pick the highest preference_rank(). None → nullptr.
 *
 * A nullptr return with *out_error == nullptr means "use the in-tree ggml path"
 * — NOT an error. Inert by default: with no -DELIZA_ENABLE_* backend compiled,
 * nothing registers and select() always returns nullptr, so every op keeps the
 * in-tree path byte-for-byte.
 *
 * Factory type F must expose:
 *   const char * name() const;          // stable lower-case id
 *   bool         available() const;     // compiled-in AND host deps present; cheap
 *   bool         can_serve(const char * bundle_dir) const;  // artifact probe; cheap
 *   int          preference_rank() const;                   // higher wins; ggml == 0
 */

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace eliza_backend {

/* malloc-allocate an error string so the caller frees it with
 * eliza_inference_free_string() (free()), matching the FFI contract. */
inline char * dup_error(const std::string & msg) {
    char * out = (char *) std::malloc(msg.size() + 1);
    if (out) std::memcpy(out, msg.c_str(), msg.size() + 1);
    return out;
}

inline bool iequals(const char * a, const char * b) {
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

/* Names that mean "stay on the in-tree ggml/llama.cpp path". */
inline bool is_intree_name(const char * s) {
    return iequals(s, "llama.cpp") || iequals(s, "llamacpp") || iequals(s, "llama") ||
           iequals(s, "ggml") || iequals(s, "intree") || iequals(s, "default");
}

template <class Factory>
class Registry {
public:
    /* Idempotent by name. Safe from static init. Does not take ownership —
     * factories are static-lifetime singletons. */
    void register_factory(Factory * factory) {
        if (!factory) return;
        std::lock_guard<std::mutex> lock(mu_);
        for (Factory * f : factories_) {
            if (iequals(f->name(), factory->name())) return;
        }
        factories_.push_back(factory);
    }

    /* env_key: the per-op override (e.g. "ELIZA_EMBED_BACKEND"); global_key: the
     * cross-op default (e.g. "ELIZA_BACKEND"); modality: for error text. */
    Factory * select(const char * env_key, const char * global_key,
                     const char * modality, const char * bundle_dir,
                     char ** out_error) {
        const char * forced = env_key ? std::getenv(env_key) : nullptr;
        if (!forced || forced[0] == '\0') {
            forced = global_key ? std::getenv(global_key) : nullptr;
        }
        if (forced && forced[0] != '\0') {
            if (is_intree_name(forced)) {
                return nullptr; /* force in-tree, not an error */
            }
            std::lock_guard<std::mutex> lock(mu_);
            for (Factory * f : factories_) {
                if (!iequals(f->name(), forced)) continue;
                if (!f->available()) {
                    set_err(out_error, modality, forced,
                            "is not available in this build/host");
                    return nullptr;
                }
                if (!f->can_serve(bundle_dir)) {
                    set_err(out_error, modality, forced,
                            std::string("cannot serve the bundle at ") +
                                (bundle_dir ? bundle_dir : "(null)"));
                    return nullptr;
                }
                return f;
            }
            set_err(out_error, modality, forced, "is not a registered backend");
            return nullptr;
        }

        /* Auto-select: highest preference_rank among available + can_serve. The
         * in-tree ggml path is the implicit rank-0 fallback, so an accelerator
         * backend only wins with a positive rank that can serve this bundle. */
        std::lock_guard<std::mutex> lock(mu_);
        Factory * best      = nullptr;
        int       best_rank = 0;
        for (Factory * f : factories_) {
            if (!f->available()) continue;
            if (!f->can_serve(bundle_dir)) continue;
            const int rank = f->preference_rank();
            if (rank > best_rank) {
                best_rank = rank;
                best      = f;
            }
        }
        return best; /* nullptr => in-tree ggml path */
    }

private:
    static void set_err(char ** out_error, const char * modality,
                        const char * name, const std::string & why) {
        if (out_error) {
            *out_error = dup_error(std::string("[libelizainference] ") +
                                   (modality ? modality : "backend") +
                                   " backend override '" + name + "' " + why);
        }
    }

    std::mutex             mu_;
    std::vector<Factory *> factories_;
};

} // namespace eliza_backend
