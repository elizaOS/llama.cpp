/*
 * litert-backend.cpp — LiteRT-LM in-process streaming-LLM backend (M4).
 *
 * See litert-backend.h for the targeted LiteRT-LM C++ API (repo + commit
 * date cited there). The real implementation is gated behind
 * `ELIZA_ENABLE_LITERT`; the default (Linux/desktop) build compiles the stub
 * branch, which links zero LiteRT-LM SDK headers and reports
 * `available() == false` so the selector keeps the in-tree llama.cpp path.
 *
 * Error contract (native/AGENTS.md §3 + §9): never log, never return a
 * defaulted result on failure. Every failure path heap-allocates `*out_error`
 * via litert_set_error() (matching the FFI cpp's eliza_strdup/eliza_set_error
 * style) and returns the negative ELIZA_* code or nullptr.
 */

#include "litert-backend.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#if defined(__has_include)
#  if __has_include(<filesystem>)
#    include <filesystem>
#    define LITERT_HAVE_FILESYSTEM 1
#  endif
#endif

/* ── Heap-allocated error strings (mirror eliza-inference-ffi.cpp) ───────── */
namespace {

char * litert_strdup(const std::string & s) {
    char * out = static_cast<char *>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

void litert_set_error(char ** out_error, const std::string & msg) {
    if (!out_error) return;
    *out_error = litert_strdup(msg);
}

#if defined(LITERT_HAVE_FILESYSTEM)
/* Probe <bundle_dir>/text/ for a *.litertlm artifact. Cheap directory walk,
 * no model load (LlmBackendFactory::can_serve contract). */
std::string find_litertlm_artifact(const char * bundle_dir) {
    if (!bundle_dir || bundle_dir[0] == '\0') return std::string();
    std::error_code ec;
    std::filesystem::path text_dir =
        std::filesystem::path(bundle_dir) / LITERT_BUNDLE_TEXT_SUBDIR;
    if (!std::filesystem::is_directory(text_dir, ec)) return std::string();
    for (std::filesystem::directory_iterator it(text_dir, ec), end;
         !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        if (it->path().extension() == LITERT_ARTIFACT_EXT) {
            return it->path().string();
        }
    }
    return std::string();
}
#else
std::string find_litertlm_artifact(const char *) { return std::string(); }
#endif

}  // namespace

/* ════════════════════════════════════════════════════════════════════════ *
 *  REAL implementation — only when ELIZA_ENABLE_LITERT is defined.
 *  Behind this gate we may include LiteRT-LM SDK headers; outside it we
 *  include NONE so the file builds on a host without the SDK.
 * ════════════════════════════════════════════════════════════════════════ */
#ifdef ELIZA_ENABLE_LITERT

#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

/* LiteRT-LM cross-platform C++ runtime. Paths per the repo's bazel layout
 * (github.com/google-ai-edge/LiteRT-LM, `main`, researched 2026-06-22). */
#include "runtime/engine/engine.h"          // litert::lm::Engine, SessionInterface
#include "runtime/engine/engine_settings.h" // EngineSettings, SessionConfig, ModelAssets
#include "runtime/engine/io_types.h"        // InputData, InputText, Responses

namespace {

using litert::lm::Backend;
using litert::lm::Engine;
using litert::lm::EngineSettings;
using litert::lm::InputData;
using litert::lm::InputText;
using litert::lm::ModelAssets;
using litert::lm::Responses;
using litert::lm::SessionConfig;

/* The Session type the templated Engine hands back (Engine::Session is the
 * public alias EngineT<SessionT> exposes; for Engine it is SessionInterface). */
using Session = Engine::Session;

/* The accelerator the factory resolved at open(), recorded for diagnostics
 * and preference reporting. DEVICE-VERIFY: which rung actually initializes is
 * hardware-dependent and can only be confirmed on an NPU/GPU device. */
enum class ResolvedAccelerator { kNone, kNpu, kGpu, kCpu };

const char * accelerator_name(ResolvedAccelerator a) {
    switch (a) {
        case ResolvedAccelerator::kNpu: return "npu";
        case ResolvedAccelerator::kGpu: return "gpu";
        case ResolvedAccelerator::kCpu: return "cpu";
        default:                        return "none";
    }
}

/* Try to build an Engine for `artifact` on `backend`. Returns the Engine on
 * success; on failure returns nullptr (the ladder falls through to the next
 * rung). The error text is captured so the final rung can surface it. */
std::unique_ptr<Engine> try_engine(const std::string & artifact,
                                   Backend backend,
                                   std::string & last_err) {
    auto model_assets = ModelAssets::Create(artifact);
    if (!model_assets.ok()) {
        last_err = std::string(model_assets.status().message());
        return nullptr;
    }
    auto settings = EngineSettings::CreateDefault(*model_assets, backend);
    if (!settings.ok()) {
        last_err = std::string(settings.status().message());
        return nullptr;
    }
    auto engine = Engine::CreateEngine(*settings);
    if (!engine.ok()) {
        last_err = std::string(engine.status().message());
        return nullptr;
    }
    return std::move(*engine);
}

/* ── Session: mirrors the FFI streaming pull contract 1:1 ────────────────── */
class LiteRtBackendSession final : public LlmBackendSession {
public:
    LiteRtBackendSession(std::unique_ptr<Engine> engine,
                         std::unique_ptr<Session> session,
                         const eliza_llm_stream_config_t & cfg,
                         ResolvedAccelerator accel)
        : engine_(std::move(engine)),
          session_(std::move(session)),
          accel_(accel),
          max_tokens_(cfg.max_tokens > 0 ? cfg.max_tokens : 0) {}

    /* prefill: copy the caller's tokens, detokenize through the engine's
     * tokenizer, and run a LiteRT prefill pass. The FFI hands pre-tokenized
     * ids (text-model vocab); LiteRT-LM's prefill consumes InputData (text),
     * so we round-trip ids → text via the shared tokenizer rather than
     * assuming vocab parity (the .litertlm graph carries its own tokenizer).
     * DEVICE-VERIFY: id/text round-trip fidelity needs a real .litertlm. */
    int prefill(const int32_t * token_ids, size_t num_tokens,
                char ** out_error) override {
        if (!session_) {
            litert_set_error(out_error,
                "[litert-lm] prefill: session is not open");
            return ELIZA_ERR_INVALID_ARG;
        }
        if (cancelled_.load(std::memory_order_acquire)) {
            return ELIZA_ERR_CANCELLED;
        }
        std::vector<int> ids;
        ids.reserve(num_tokens);
        for (size_t i = 0; i < num_tokens; ++i) ids.push_back(token_ids[i]);

        const std::string text = engine_->GetTokenizer().Detokenize(ids);
        std::vector<InputData> contents;
        contents.emplace_back(InputText(std::string(text)));

        absl::Status st = session_->RunPrefill(contents);
        if (!st.ok()) {
            litert_set_error(out_error,
                std::string("[litert-lm] RunPrefill failed: ") +
                std::string(st.message()));
            return ELIZA_ERR_FFI_FAULT;
        }
        prefilled_ = true;
        return ELIZA_OK;
    }

    /* next: one decode step. LiteRT-LM's RunDecode() returns a Responses
     * batch; we emit the newly-produced UTF-8 delta as detokenized text and
     * its token ids. LiteRT-LM has no in-process MTP drafter exposed through
     * this surface, so drafted/accepted are always 0. Returns 1 (final) at
     * EOS or the max-token cap, 0 otherwise. */
    int next(int32_t * tokens_out, size_t tokens_cap, size_t * num_tokens_out,
             char * text_out, size_t text_cap, int32_t * drafter_drafted_out,
             int32_t * drafter_accepted_out, char ** out_error) override {
        if (num_tokens_out) *num_tokens_out = 0;
        if (text_out && text_cap) text_out[0] = '\0';
        if (drafter_drafted_out)  *drafter_drafted_out = 0;
        if (drafter_accepted_out) *drafter_accepted_out = 0;

        if (!session_) {
            litert_set_error(out_error, "[litert-lm] next: session not open");
            return ELIZA_ERR_INVALID_ARG;
        }
        if (!prefilled_) {
            litert_set_error(out_error,
                "[litert-lm] next: prefill must run before next");
            return ELIZA_ERR_INVALID_ARG;
        }
        if (cancelled_.load(std::memory_order_acquire)) {
            return ELIZA_ERR_CANCELLED;
        }

        auto responses = session_->RunDecode();
        if (!responses.ok()) {
            litert_set_error(out_error,
                std::string("[litert-lm] RunDecode failed: ") +
                std::string(responses.status().message()));
            return ELIZA_ERR_FFI_FAULT;
        }

        /* RunDecode yields the running candidate texts; GetTexts()[0] is the
         * cumulative decode for candidate 0. Emit only the suffix produced
         * since the last step so the FFI streams a delta per pull. */
        const std::vector<std::string> & texts = responses->GetTexts();
        std::string cumulative = texts.empty() ? std::string() : texts.front();
        std::string delta = compute_delta(cumulative);
        emitted_chars_ = cumulative.size();

        /* Re-tokenize the delta against the engine tokenizer so the FFI gets
         * committed text-vocab ids (the same round-trip the prefill used). */
        std::vector<int> delta_ids = engine_->GetTokenizer().Tokenize(delta);
        size_t n_emit = delta_ids.size();
        if (n_emit > tokens_cap) n_emit = tokens_cap;
        if (tokens_out) {
            for (size_t i = 0; i < n_emit; ++i) {
                tokens_out[i] = static_cast<int32_t>(delta_ids[i]);
            }
        }
        if (num_tokens_out) *num_tokens_out = n_emit;
        if (text_out && text_cap) {
            const size_t copy = delta.size() < text_cap - 1
                                    ? delta.size()
                                    : text_cap - 1;
            std::memcpy(text_out, delta.data(), copy);
            text_out[copy] = '\0';
        }

        decoded_tokens_ += static_cast<int32_t>(delta_ids.size());
        const bool hit_cap =
            max_tokens_ > 0 && decoded_tokens_ >= max_tokens_;
        /* DEVICE-VERIFY: the precise EOS signal LiteRT-LM exposes per step is
         * runtime-version-dependent. A done decode yields no new delta; treat
         * an empty delta or the token cap as the final step. */
        const bool eos = delta_ids.empty();
        return (hit_cap || eos) ? 1 : 0;
    }

    /* cancel: publish a flag the next decode step observes. Thread-safe. */
    int cancel() override {
        cancelled_.store(true, std::memory_order_release);
        return ELIZA_OK;
    }

    /* reset: drop a fresh Session from the same Engine (clears KV + sampler).
     * Reuses the warm Engine (model weights stay resident) — only the
     * per-generation Session is rebuilt. */
    int reset() override {
        auto cfg = SessionConfig::CreateDefault();
        auto session = engine_->CreateSession(cfg);
        if (!session.ok()) {
            /* reset has no out_error param; a failed rebuild leaves the old
             * session in place and surfaces on the next prefill/next. */
            return ELIZA_ERR_FFI_FAULT;
        }
        session_ = std::move(*session);
        cancelled_.store(false, std::memory_order_release);
        prefilled_ = false;
        decoded_tokens_ = 0;
        emitted_chars_ = 0;
        return ELIZA_OK;
    }

    /* reset_keep: LiteRT-LM's Session does not expose prefix-preserving KV
     * trimming through this surface, so fall back to a full reset and return 0
     * (no prefix kept) — never an error (llm-backend.h contract). */
    int reset_keep(int32_t /*n_keep*/) override {
        reset();
        return 0;
    }

    const char * accelerator() const { return accelerator_name(accel_); }

private:
    /* The suffix of `cumulative` produced since the last emitted step. */
    std::string compute_delta(const std::string & cumulative) const {
        if (cumulative.size() <= emitted_chars_) return std::string();
        return cumulative.substr(emitted_chars_);
    }

    std::unique_ptr<Engine>  engine_;
    std::unique_ptr<Session> session_;
    std::atomic<bool>        cancelled_{false};
    bool                     prefilled_ = false;
    int32_t                  decoded_tokens_ = 0;
    size_t                   emitted_chars_ = 0;
    ResolvedAccelerator      accel_ = ResolvedAccelerator::kNone;
    int32_t                  max_tokens_ = 0;
};

/* ── Factory ─────────────────────────────────────────────────────────────── */
class LiteRtBackendFactory final : public LlmBackendFactory {
public:
    const char * name() const override { return LITERT_BACKEND_NAME; }

    /* available(): compiled in AND an accelerator (NPU or GPU) initializes on
     * THIS host. Cheap — must not load a model. We probe by building a minimal
     * EngineSettings on NPU then GPU with NO model assets; a backend whose
     * delegate is missing fails settings validation. CPU alone does NOT make
     * this backend "available" (CPU is the in-tree llama.cpp path's job).
     * DEVICE-VERIFY: real delegate presence is only knowable on-device. */
    bool available() const override {
        return probe_accelerator() != ResolvedAccelerator::kNone;
    }

    /* can_serve(): a *.litertlm exists under <bundle_dir>/text/. Cheap probe,
     * no caching — open() re-resolves the bundle from the context accessor. */
    bool can_serve(const char * bundle_dir) const override {
        return !find_litertlm_artifact(bundle_dir).empty();
    }

    /* preference_rank(): high on Android NPU (the whole reason this backend
     * exists), modest on a GPU-only fallback, 0 otherwise so llama.cpp wins. */
    int preference_rank() const override {
        switch (probe_accelerator()) {
            case ResolvedAccelerator::kNpu: return 100;
            case ResolvedAccelerator::kGpu: return 20;
            default:                        return 0;
        }
    }

    /* open(): resolve the .litertlm under the cached bundle, then walk the
     * accelerator ladder NPU → GPU → CPU, recording which rung built the
     * Engine. Builds a default Session and returns the streaming session. */
    LlmBackendSession * open(EliInferenceContext * ctx,
                             const eliza_llm_stream_config_t * cfg,
                             char ** out_error) override {
        if (!cfg) {
            litert_set_error(out_error, "[litert-lm] open: cfg is NULL");
            return nullptr;
        }
        const char * bundle_dir = llm_backend_context_bundle_dir(ctx);
        const std::string bundle = bundle_dir ? bundle_dir : std::string();
        std::string artifact = find_litertlm_artifact(bundle.c_str());
        if (artifact.empty()) {
            litert_set_error(out_error,
                std::string("[litert-lm] open: no ") + LITERT_ARTIFACT_EXT +
                " artifact under " + bundle + "/" + LITERT_BUNDLE_TEXT_SUBDIR);
            return nullptr;
        }

        /* Accelerator ladder — NPU first (Qualcomm QNN / MediaTek NeuroPilot /
         * Google Tensor), then GPU (OpenCL/Metal/WebGPU), then CPU (XNNPACK).
         * Each rung's failure text is preserved for the final diagnostic.
         * DEVICE-VERIFY: rung availability is hardware-specific. */
        struct Rung { Backend backend; ResolvedAccelerator accel; };
        const Rung ladder[] = {
            {Backend::NPU, ResolvedAccelerator::kNpu},
            {Backend::GPU, ResolvedAccelerator::kGpu},
            {Backend::CPU, ResolvedAccelerator::kCpu},
        };

        std::unique_ptr<Engine> engine;
        ResolvedAccelerator resolved = ResolvedAccelerator::kNone;
        std::string last_err;
        for (const Rung & rung : ladder) {
            engine = try_engine(artifact, rung.backend, last_err);
            if (engine) {
                resolved = rung.accel;
                break;
            }
        }
        if (!engine) {
            litert_set_error(out_error,
                std::string("[litert-lm] open: no accelerator could build the "
                            "engine (last error: ") + last_err + ")");
            return nullptr;
        }

        auto session_cfg = SessionConfig::CreateDefault();
        auto session = engine->CreateSession(session_cfg);
        if (!session.ok()) {
            litert_set_error(out_error,
                std::string("[litert-lm] open: CreateSession failed on ") +
                accelerator_name(resolved) + ": " +
                std::string(session.status().message()));
            return nullptr;
        }

        return new LiteRtBackendSession(std::move(engine), std::move(*session),
                                        *cfg, resolved);
    }

private:
    /* Build a no-model EngineSettings on NPU then GPU; the first whose
     * delegate validates marks that rung present. Result is memoized so the
     * repeated available()/preference_rank() calls are cheap.
     * DEVICE-VERIFY: settings-only validation is the cheapest honest probe;
     * the true delegate handshake happens at open() on-device. */
    ResolvedAccelerator probe_accelerator() const {
        std::call_once(probe_once_, [this]() {
            auto empty = ModelAssets::Create(std::string());
            if (!empty.ok()) { probed_ = ResolvedAccelerator::kNone; return; }
            if (EngineSettings::CreateDefault(*empty, Backend::NPU).ok()) {
                probed_ = ResolvedAccelerator::kNpu;
            } else if (EngineSettings::CreateDefault(*empty, Backend::GPU).ok()) {
                probed_ = ResolvedAccelerator::kGpu;
            } else {
                probed_ = ResolvedAccelerator::kNone;
            }
        });
        return probed_;
    }

    mutable std::once_flag      probe_once_;
    mutable ResolvedAccelerator probed_ = ResolvedAccelerator::kNone;
};

}  // namespace

LlmBackendFactory * litert_backend_factory() {
    static LiteRtBackendFactory factory;
    return &factory;
}

#else  /* ────────────────────────── STUB (no LiteRT-LM SDK) ──────────────── */

/*
 * Compiled-out stub: zero LiteRT-LM headers, so this builds on any host. The
 * factory links in as a no-op — available() is false, can_serve() is false,
 * preference_rank() is 0, and open() returns nullptr + sets `*out_error`
 * "not compiled in" so the selector cleanly keeps the in-tree llama.cpp path.
 */
namespace {

class LiteRtBackendFactoryStub final : public LlmBackendFactory {
public:
    const char * name() const override { return LITERT_BACKEND_NAME; }
    bool available() const override { return false; }
    bool can_serve(const char * /*bundle_dir*/) const override { return false; }
    int preference_rank() const override { return 0; }

    LlmBackendSession * open(EliInferenceContext * /*ctx*/,
                             const eliza_llm_stream_config_t * /*cfg*/,
                             char ** out_error) override {
        litert_set_error(out_error,
            "[litert-lm] backend not compiled in "
            "(build with -DELIZA_ENABLE_LITERT to enable the LiteRT-LM NPU path)");
        return nullptr;
    }
};

}  // namespace

LlmBackendFactory * litert_backend_factory() {
    static LiteRtBackendFactoryStub factory;
    return &factory;
}

#endif  /* ELIZA_ENABLE_LITERT */
