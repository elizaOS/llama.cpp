/* tools/omnivoice/src/eliza-asr-parakeet.inc - Parakeet-TDT (Phonon-2)
 * backend for the eliza_inference_asr_* surface.
 *
 * #included by eliza-inference-ffi.cpp inside `#ifdef ELIZA_ENABLE_PARAKEET`
 * after EliInferenceContext is defined. Everything here runs on
 * tools/parakeet (parakeet_lib): native FastConformer encoder + TDT decoder
 * on the CPU, five-value int8 kernels, no ONNX / MLX / Core ML.
 *
 * Selection (mmap_acquire("asr")):
 *   ELIZA_ASR_BACKEND=parakeet  require a parakeet-tdt GGUF under <bundle>/asr/
 *   ELIZA_ASR_BACKEND=qwen3     never use parakeet (Qwen3-ASR via mtmd)
 *   unset / auto                parakeet when <bundle>/asr/ holds a GGUF with
 *                               general.architecture = "parakeet-tdt",
 *                               otherwise the Qwen3-ASR path, unchanged.
 * Threads: ELIZA_ASR_THREADS, else parakeet's default (fast cores, <= 4).
 *
 * Streaming: windowed re-decode with local agreement (tools/parakeet/src/
 * pk-stream.cpp). eliza_inference_asr_stream_supported() returns 1 while at
 * least one context has the parakeet backend loaded. Token ids are not
 * returned (*io_n_tokens = 0): Parakeet's 8192-piece SentencePiece vocabulary
 * is not the text model's.
 */

#include "parakeet.h"

static std::atomic<int> g_eliza_parakeet_loaded{ 0 };

struct EliParakeetAsr {
    pk_model * model = nullptr;
    std::string path;
    uint64_t generation = 0;  /* bumped on every load, so open streams notice an evict */
};

static uint64_t g_eliza_parakeet_generation = 0;

static int eliza_parakeet_cancel_cb(void * user) {
    auto * ctx = (EliInferenceContext *) user;
    return ctx && ctx->asr_cancel.load(std::memory_order_acquire) ? 1 : 0;
}

static std::string eliza_parakeet_backend_pref() {
    const char * env = std::getenv("ELIZA_ASR_BACKEND");
    std::string v = env ? env : "";
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return v;
}

static std::string eliza_parakeet_find(const std::filesystem::path & bundle_dir) {
    const auto dir = bundle_dir / "asr";
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return "";
    std::vector<std::string> found;
    for (const auto & e : std::filesystem::directory_iterator(dir, ec)) {
        if (!e.is_regular_file(ec) || e.path().extension() != ".gguf") continue;
        if (pk_gguf_is_parakeet(e.path().string().c_str())) found.push_back(e.path().string());
    }
    std::sort(found.begin(), found.end());
    return found.empty() ? "" : found.front();
}

/* Caller holds asr_mutex. Returns 1 when the parakeet backend is (now)
 * loaded, 0 when this bundle / preference does not use it, negative on a
 * load error. */
static int eliza_parakeet_load_locked(EliInferenceContext * ctx, char ** out_error) {
    if (ctx->parakeet && ctx->parakeet->model) return 1;
    const std::string pref = eliza_parakeet_backend_pref();
    if (pref == "qwen3" || pref == "mtmd") return 0;
    const std::string path = eliza_parakeet_find(std::filesystem::path(ctx->bundle_dir));
    if (path.empty()) {
        if (pref == "parakeet") {
            eliza_set_error(out_error, std::string("[libelizainference] ELIZA_ASR_BACKEND=parakeet but no parakeet-tdt GGUF under ") +
                                           (std::filesystem::path(ctx->bundle_dir) / "asr").string());
            return ELIZA_ERR_INVALID_ARG;
        }
        return 0;
    }
    pk_params pp = pk_default_params();
    /* same policy as the Qwen3-ASR path: Android's CPU profile keeps ASR
     * weights in private memory (ELIZA_ASR_USE_MMAP overrides) */
    pp.use_mmap = eliza_asr_use_mmap() ? 1 : 0;
    if (const char * env = std::getenv("ELIZA_ASR_THREADS")) {
        const int n = std::atoi(env);
        if (n > 0) pp.n_threads = n;
    }
    char err[512] = { 0 };
    eliza_asr_debug_log("loading parakeet-tdt ASR");
    pk_model * m = pk_model_load(path.c_str(), pp, err, sizeof(err));
    if (!m) {
        eliza_set_error(out_error, std::string("[libelizainference] parakeet ASR load failed: ") + err);
        return ELIZA_ERR_FFI_FAULT;
    }
    if (!ctx->parakeet) ctx->parakeet = new EliParakeetAsr();
    ctx->parakeet->model = m;
    ctx->parakeet->path = path;
    ctx->parakeet->generation = ++g_eliza_parakeet_generation;
    g_eliza_parakeet_loaded.fetch_add(1);
    eliza_asr_debug_log("parakeet-tdt ASR ready");
    return 1;
}

/* Caller holds asr_mutex. */
static void eliza_parakeet_free_locked(EliInferenceContext * ctx) {
    if (!ctx->parakeet) return;
    if (ctx->parakeet->model) {
        pk_model_free(ctx->parakeet->model);
        g_eliza_parakeet_loaded.fetch_sub(1);
    }
    delete ctx->parakeet;
    ctx->parakeet = nullptr;
}

static bool eliza_parakeet_active_locked(const EliInferenceContext * ctx) {
    return ctx->parakeet && ctx->parakeet->model;
}

/* Caller holds asr_mutex and has cleared asr_cancel. */
static int eliza_parakeet_transcribe_locked(
    EliInferenceContext * ctx,
    const float * pcm,
    size_t n_samples,
    int sample_rate_hz,
    std::string & out_text,
    std::vector<int> * word_start,
    std::vector<int> * word_end,
    char ** out_error) {
    pk_model * m = ctx->parakeet->model;
    std::vector<float> audio = eliza_resample_linear(pcm, n_samples, sample_rate_hz, pk_model_sample_rate(m));
    pk_result r{};
    const int rc = pk_transcribe(m, audio.data(), audio.size(), eliza_parakeet_cancel_cb, ctx, &r);
    if (rc == PK_E_CANCELLED) {
        pk_result_free(&r);
        eliza_set_error(out_error, "[libelizainference] asr_transcribe: cancelled");
        return ELIZA_ERR_CANCELLED;
    }
    if (rc != PK_OK) {
        pk_result_free(&r);
        eliza_set_error(out_error, "[libelizainference] asr_transcribe: parakeet decode failed rc=" + std::to_string(rc));
        return ELIZA_ERR_FFI_FAULT;
    }
    out_text = r.text ? r.text : "";
    if (word_start && word_end) {
        size_t n = r.n_tokens + 1;
        word_start->assign(n, 0);
        word_end->assign(n, 0);
        pk_result_words(m, &r, word_start->data(), word_end->data(), &n);
        word_start->resize(n);
        word_end->resize(n);
    }
    pk_result_free(&r);
    return 0;
}

/* ---- streaming ---------------------------------------------------------- */

struct EliAsrStream {
    EliInferenceContext * ctx = nullptr;
    pk_stream * st = nullptr;
    uint64_t generation = 0;
    int in_rate = 16000;
    /* continuous linear resampler state (in_rate -> 16 kHz) */
    double phase = 0.0;
    float last = 0.0f;
    bool have_last = false;
};

static void eliza_parakeet_stream_resample(EliAsrStream * s, const float * pcm, size_t n, std::vector<float> & out) {
    out.clear();
    if (s->in_rate == 16000) {
        out.assign(pcm, pcm + n);
        return;
    }
    const double step = (double) s->in_rate / 16000.0;
    /* input index space: -1 = s->last, 0..n-1 = pcm */
    double pos = s->phase;
    for (;;) {
        const double ip = std::floor(pos);
        const long i0 = (long) ip;
        if (i0 + 1 >= (long) n) break;
        const float a = i0 < 0 ? (s->have_last ? s->last : pcm[0]) : pcm[i0];
        const float b = pcm[i0 + 1];
        const float t = (float) (pos - ip);
        out.push_back(a + (b - a) * t);
        pos += step;
    }
    if (n > 0) {
        s->last = pcm[n - 1];
        s->have_last = true;
        s->phase = pos - (double) n;  /* relative to the next chunk; >= -1 */
    }
}
