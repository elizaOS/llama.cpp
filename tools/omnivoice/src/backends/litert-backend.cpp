/*
 * litert-backend.cpp — LiteRT-LM in-process streaming-LLM backend (M4).
 *
 * The real implementation (gated behind `ELIZA_ENABLE_LITERT`) targets the
 * LiteRT-LM STABLE C API (`litert_lm_*`, engine.h) — NOT the C++ runtime
 * (`litert::lm::Engine`). The C surface is ABI-stable, so the fused
 * libelizainference links the prebuilt liblitert-lm.so with no libc++
 * C++-ABI coupling. See litert-backend.h for the targeted symbols.
 *
 * The default (Linux/desktop CI) build compiles the stub branch, which links
 * zero LiteRT-LM SDK headers and reports `available() == false` so the selector
 * keeps the in-tree llama.cpp path.
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

#include <condition_variable>
#include <deque>
#include <memory>
#include <utility>
#include <vector>

/* LiteRT-LM STABLE C API (engine.h). C API only — NO `litert::lm::` C++
 * symbols — so the fused libelizainference carries no libc++ C++-ABI
 * dependency on the prebuilt liblitert-lm.so. The whole surface below maps the
 * llm-backend.h FFI pull contract onto the litert_lm_* functions:
 *
 *   open()    -> litert_lm_engine_settings_create (+ speculative decoding for
 *                MTP) -> litert_lm_engine_create -> litert_lm_engine_create_session
 *   prefill() -> litert_lm_session_run_prefill (ids round-tripped to text via
 *                the engine tokenizer; the .litertlm carries its own tokenizer)
 *   next()    -> one chunk pulled from the async-decode callback queue, emitted
 *                as a text delta + re-tokenized ids
 *   cancel()  -> litert_lm_session_cancel_process
 *   reset()   -> rebuild the session from the warm engine
 */
#include "engine.h"

namespace {

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

/* The backend_str the C API expects for each rung. */
const char * backend_str(ResolvedAccelerator a) {
    switch (a) {
        case ResolvedAccelerator::kNpu: return "npu";
        case ResolvedAccelerator::kGpu: return "gpu";
        default:                        return "cpu";
    }
}

/* RAII for the C engine settings (the only place it lives is open()). */
struct SettingsHandle {
    LiteRtLmEngineSettings * p = nullptr;
    ~SettingsHandle() { if (p) litert_lm_engine_settings_delete(p); }
};

/* Detokenize text-vocab ids → UTF-8 through the engine's own tokenizer. */
std::string detokenize(LiteRtLmEngine * engine, const int32_t * ids,
                       size_t num) {
    if (!engine || num == 0) return std::string();
    std::vector<int> tmp(ids, ids + num);
    LiteRtLmDetokenizeResult * res =
        litert_lm_engine_detokenize(engine, tmp.data(), tmp.size());
    if (!res) return std::string();
    const char * s = litert_lm_detokenize_result_get_string(res);
    std::string out = s ? std::string(s) : std::string();
    litert_lm_detokenize_result_delete(res);
    return out;
}

/* Tokenize UTF-8 → text-vocab ids through the engine's own tokenizer. */
std::vector<int> tokenize(LiteRtLmEngine * engine, const std::string & text) {
    std::vector<int> out;
    if (!engine || text.empty()) return out;
    LiteRtLmTokenizeResult * res = litert_lm_engine_tokenize(engine, text.c_str());
    if (!res) return out;
    size_t n = litert_lm_tokenize_result_get_num_tokens(res);
    const int * toks = litert_lm_tokenize_result_get_tokens(res);
    if (toks) out.assign(toks, toks + n);
    litert_lm_tokenize_result_delete(res);
    return out;
}

/* ── Session: mirrors the FFI streaming pull contract 1:1 ────────────────── *
 *
 * Decode is driven through litert_lm_session_run_decode_async: prefill kicks
 * off the async stream and each next() pops one chunk from a thread-safe
 * queue. This maps the library's push-streaming surface onto the FFI's
 * pull-one-step contract without buffering the whole response. The engine
 * is owned and kept warm so reset() rebuilds only the per-generation session.
 */
class LiteRtBackendSession final : public LlmBackendSession {
public:
    LiteRtBackendSession(LiteRtLmEngine * engine, LiteRtLmSession * session,
                         const eliza_llm_stream_config_t & cfg,
                         ResolvedAccelerator accel)
        : engine_(engine),
          session_(session),
          accel_(accel),
          max_tokens_(cfg.max_tokens > 0 ? cfg.max_tokens : 0) {}

    ~LiteRtBackendSession() override {
        if (session_) litert_lm_session_delete(session_);
        if (engine_)  litert_lm_engine_delete(engine_);
    }

    /* prefill: round-trip the caller's text-vocab ids to text through the
     * engine tokenizer, feed it as a single text input, and start the async
     * decode stream so next() can pull chunks. The FFI hands pre-tokenized ids;
     * the .litertlm graph carries its own tokenizer, so we round-trip rather
     * than assume vocab parity. DEVICE-VERIFY: round-trip fidelity needs a real
     * device .litertlm. */
    int prefill(const int32_t * token_ids, size_t num_tokens,
                char ** out_error) override {
        if (!session_) {
            litert_set_error(out_error, "[litert-lm] prefill: session not open");
            return ELIZA_ERR_INVALID_ARG;
        }
        if (cancelled_.load(std::memory_order_acquire)) {
            return ELIZA_ERR_CANCELLED;
        }

        const std::string text = detokenize(engine_, token_ids, num_tokens);

        LiteRtLmInputData input;
        input.type = kLiteRtLmInputDataTypeText;
        input.data = text.c_str();
        input.size = text.size();

        if (litert_lm_session_run_prefill(session_, &input, 1) != 0) {
            litert_set_error(out_error, "[litert-lm] run_prefill failed");
            return ELIZA_ERR_FFI_FAULT;
        }

        /* Kick off the streaming decode; chunks land on the queue via the
         * static callback below. next() drains them one at a time. */
        reset_stream_state();
        if (litert_lm_session_run_decode_async(session_, &on_stream_chunk,
                                               this) != 0) {
            litert_set_error(out_error,
                "[litert-lm] run_decode_async failed to start");
            return ELIZA_ERR_FFI_FAULT;
        }
        prefilled_ = true;
        return ELIZA_OK;
    }

    /* next: pop one decode chunk from the async stream. Emits the chunk text as
     * a delta and its re-tokenized ids. LiteRT-LM exposes no in-process MTP
     * drafter on this surface, so drafted/accepted are always 0. Returns 1
     * (final) at EOS, stream error, or the token cap, 0 otherwise. */
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

        Chunk chunk;
        {
            std::unique_lock<std::mutex> lock(mu_);
            cv_.wait(lock, [this] { return !queue_.empty(); });
            chunk = std::move(queue_.front());
            queue_.pop_front();
        }

        if (chunk.error) {
            litert_set_error(out_error,
                std::string("[litert-lm] decode stream error: ") + chunk.text);
            return ELIZA_ERR_FFI_FAULT;
        }

        const std::string & delta = chunk.text;
        std::vector<int> delta_ids = tokenize(engine_, delta);
        size_t n_emit = delta_ids.size();
        if (n_emit > tokens_cap) n_emit = tokens_cap;
        if (tokens_out) {
            for (size_t i = 0; i < n_emit; ++i) {
                tokens_out[i] = static_cast<int32_t>(delta_ids[i]);
            }
        }
        if (num_tokens_out) *num_tokens_out = n_emit;
        if (text_out && text_cap && !delta.empty()) {
            const size_t copy =
                delta.size() < text_cap - 1 ? delta.size() : text_cap - 1;
            std::memcpy(text_out, delta.data(), copy);
            text_out[copy] = '\0';
        }

        decoded_tokens_ += static_cast<int32_t>(delta_ids.size());
        const bool hit_cap = max_tokens_ > 0 && decoded_tokens_ >= max_tokens_;
        return (chunk.is_final || hit_cap) ? 1 : 0;
    }

    /* cancel: signal the C runtime to stop the in-flight decode and wake any
     * blocked next(). Thread-safe; idempotent. */
    int cancel() override {
        cancelled_.store(true, std::memory_order_release);
        if (session_) litert_lm_session_cancel_process(session_);
        {
            std::lock_guard<std::mutex> lock(mu_);
            queue_.push_back(Chunk{std::string(), true, false});
        }
        cv_.notify_all();
        return ELIZA_OK;
    }

    /* reset: rebuild the per-generation session from the warm engine (clears
     * KV + sampler). Model weights stay resident. */
    int reset() override {
        if (session_) {
            litert_lm_session_cancel_process(session_);
            litert_lm_session_delete(session_);
            session_ = nullptr;
        }
        LiteRtLmSession * fresh =
            litert_lm_engine_create_session(engine_, nullptr);
        if (!fresh) {
            /* reset has no out_error param; the failure surfaces on the next
             * prefill/next when session_ is null. */
            return ELIZA_ERR_FFI_FAULT;
        }
        session_ = fresh;
        cancelled_.store(false, std::memory_order_release);
        prefilled_ = false;
        decoded_tokens_ = 0;
        reset_stream_state();
        return ELIZA_OK;
    }

    /* reset_keep: the C session surface exposes no prefix-preserving KV trim,
     * so fall back to a full reset and return 0 (no prefix kept) — never an
     * error (llm-backend.h contract). */
    int reset_keep(int32_t /*n_keep*/) override {
        reset();
        return 0;
    }

    const char * accelerator() const { return accelerator_name(accel_); }

private:
    struct Chunk {
        std::string text;
        bool is_final = false;
        bool error = false;
    };

    void reset_stream_state() {
        std::lock_guard<std::mutex> lock(mu_);
        queue_.clear();
    }

    /* C stream callback (litert_lm_session_run_decode_async). Runs on the C
     * runtime's background thread; pushes each chunk onto the queue. The
     * `callback_data` is the owning session. */
    static void on_stream_chunk(void * callback_data, const char * chunk,
                                bool is_final, const char * error_msg) {
        auto * self = static_cast<LiteRtBackendSession *>(callback_data);
        Chunk c;
        if (error_msg) {
            c.error = true;
            c.is_final = true;
            c.text = error_msg;
        } else {
            c.text = chunk ? std::string(chunk) : std::string();
            c.is_final = is_final;
        }
        {
            std::lock_guard<std::mutex> lock(self->mu_);
            self->queue_.push_back(std::move(c));
        }
        self->cv_.notify_one();
    }

    LiteRtLmEngine *  engine_  = nullptr;  /* owned */
    LiteRtLmSession * session_ = nullptr;  /* owned */
    std::atomic<bool> cancelled_{false};
    bool              prefilled_ = false;
    int32_t           decoded_tokens_ = 0;
    ResolvedAccelerator accel_ = ResolvedAccelerator::kNone;
    int32_t           max_tokens_ = 0;

    std::mutex                  mu_;
    std::condition_variable     cv_;
    std::deque<Chunk>           queue_;
};

/* Build a C engine for `artifact` on `accel`. Returns the engine on success;
 * on failure returns nullptr (the ladder falls through to the next rung). */
LiteRtLmEngine * try_engine(const std::string & artifact,
                            ResolvedAccelerator accel) {
    SettingsHandle settings;
    settings.p = litert_lm_engine_settings_create(
        artifact.c_str(), backend_str(accel),
        /*vision_backend=*/nullptr, /*audio_backend=*/nullptr);
    if (!settings.p) return nullptr;

    /* MTP speculative decoding (§3 — mandatory). The .litertlm carries its own
     * drafter; enabling speculative decoding turns it on. */
    litert_lm_engine_settings_set_enable_speculative_decoding(settings.p, true);

    return litert_lm_engine_create(settings.p);
}

/* ── Factory ─────────────────────────────────────────────────────────────── */
class LiteRtBackendFactory final : public LlmBackendFactory {
public:
    const char * name() const override { return LITERT_BACKEND_NAME; }

    /* available(): compiled in AND a *.litertlm-servable accelerator exists.
     * The C API offers no cheap no-model delegate probe, so availability here
     * is "the SDK is linked in" — the real NPU/GPU delegate handshake happens
     * at open() and falls back down the ladder. The selector still only picks
     * this backend when can_serve() finds an artifact, so an available()==true
     * with no .litertlm bundle never displaces the llama.cpp path.
     * DEVICE-VERIFY: true delegate presence is only knowable on-device. */
    bool available() const override { return true; }

    /* can_serve(): a *.litertlm exists under <bundle_dir>/text/. Cheap probe,
     * no caching — open() re-resolves the bundle from the context accessor. */
    bool can_serve(const char * bundle_dir) const override {
        return !find_litertlm_artifact(bundle_dir).empty();
    }

    /* preference_rank(): positive so an NPU/GPU device that ships a .litertlm
     * bundle prefers this backend over llama.cpp. The accelerator the engine
     * actually binds is resolved at open() via the ladder; this rank only
     * orders candidates that can_serve the same bundle.
     * DEVICE-VERIFY: the right rank vs the GPU llama.cpp path is device-tuned. */
    int preference_rank() const override { return 50; }

    /* open(): resolve the .litertlm under the cached bundle, walk the
     * accelerator ladder NPU → GPU → CPU (the C API picks the delegate from
     * backend_str; a missing delegate fails engine_create and we fall to the
     * next rung), then create the streaming session. */
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

        /* NPU first (Qualcomm QNN / MediaTek NeuroPilot / Google Tensor), then
         * GPU (OpenCL/Metal/WebGPU), then CPU (XNNPACK). engine_create returns
         * null when the rung's delegate is unavailable; we fall through.
         * DEVICE-VERIFY: rung availability is hardware-specific. */
        const ResolvedAccelerator ladder[] = {
            ResolvedAccelerator::kNpu,
            ResolvedAccelerator::kGpu,
            ResolvedAccelerator::kCpu,
        };

        LiteRtLmEngine * engine = nullptr;
        ResolvedAccelerator resolved = ResolvedAccelerator::kNone;
        for (ResolvedAccelerator accel : ladder) {
            engine = try_engine(artifact, accel);
            if (engine) {
                resolved = accel;
                break;
            }
        }
        if (!engine) {
            litert_set_error(out_error,
                "[litert-lm] open: no accelerator (npu/gpu/cpu) could build the "
                "engine for the bundle's .litertlm");
            return nullptr;
        }

        LiteRtLmSession * session =
            litert_lm_engine_create_session(engine, nullptr);
        if (!session) {
            litert_lm_engine_delete(engine);
            litert_set_error(out_error,
                std::string("[litert-lm] open: create_session failed on ") +
                accelerator_name(resolved));
            return nullptr;
        }

        return new LiteRtBackendSession(engine, session, *cfg, resolved);
    }
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
