#include "server-context.h"
#include "server-http.h"
#include "server-models.h"

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

// MILADY-OMNIVOICE-AUDIO-SPEECH-ROUTE-V1
#ifdef MILADY_FUSE_OMNIVOICE
#include "omnivoice.h"
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace milady_omnivoice {

// Resolve a config value: prefer the CLI override captured in main(), then
// the env var, then empty.
static std::string g_model_path;
static std::string g_codec_path;

static std::string cli_or_env(const std::string & cli, const char * name) {
    if (!cli.empty()) return cli;
    const char * v = std::getenv(name);
    if (v && v[0] != '\0') return std::string(v);
    return std::string();
}

static std::string resolved_model_path() {
    return cli_or_env(g_model_path, "ELIZA_OMNIVOICE_MODEL");
}
static std::string resolved_codec_path() {
    return cli_or_env(g_codec_path, "ELIZA_OMNIVOICE_CODEC");
}

static std::mutex      g_mu;
static ov_context *    g_ctx = nullptr;   // lazily initialised under g_mu
static std::string     g_init_error;       // sticky: a failed init stays failed until paths change
static std::string     g_init_signature;   // model|codec the live ctx was built from

// Returns the OmniVoice context, initialising it on first use. Returns
// nullptr and sets *err on failure. Caller must hold g_mu.
static ov_context * acquire_locked(std::string & err) {
    const std::string model = resolved_model_path();
    const std::string codec = resolved_codec_path();
    const std::string sig = model + "|" + codec;
    if (g_ctx && g_init_signature == sig) return g_ctx;
    if (g_ctx && g_init_signature != sig) {
        ov_free(g_ctx);
        g_ctx = nullptr;
        g_init_error.clear();
    }
    if (model.empty() || codec.empty()) {
        err = "omnivoice TTS not configured: pass --omnivoice-model and "
              "--omnivoice-codec (or set ELIZA_OMNIVOICE_MODEL / "
              "ELIZA_OMNIVOICE_CODEC) when launching the fused server";
        return nullptr;
    }
    if (!g_init_error.empty() && g_init_signature == sig) {
        err = g_init_error;
        return nullptr;
    }
    ov_init_params ip;
    ov_init_default_params(&ip);
    ip.model_path = model.c_str();
    ip.codec_path = codec.c_str();
    ov_context * ctx = ov_init(&ip);
    if (!ctx) {
        const char * le = ov_last_error();
        g_init_error = std::string("omnivoice ov_init failed: ") + (le ? le : "(no detail)");
        g_init_signature = sig;
        err = g_init_error;
        return nullptr;
    }
    g_ctx = ctx;
    g_init_signature = sig;
    g_init_error.clear();
    return g_ctx;
}

// Build a 16-bit PCM WAV container around f32 mono samples at sample_rate.
static std::string wav16_from_f32(const float * pcm, int n, int sample_rate) {
    auto put_u32 = [](std::string & s, uint32_t v) {
        s.push_back((char)(v & 0xff));
        s.push_back((char)((v >> 8) & 0xff));
        s.push_back((char)((v >> 16) & 0xff));
        s.push_back((char)((v >> 24) & 0xff));
    };
    auto put_u16 = [](std::string & s, uint16_t v) {
        s.push_back((char)(v & 0xff));
        s.push_back((char)((v >> 8) & 0xff));
    };
    const uint16_t channels = 1;
    const uint16_t bits = 16;
    const uint32_t byte_rate = (uint32_t)sample_rate * channels * (bits / 8);
    const uint16_t block_align = channels * (bits / 8);
    const uint32_t data_bytes = (uint32_t)n * (bits / 8);
    std::string out;
    out.reserve(44 + data_bytes);
    out += "RIFF";
    put_u32(out, 36 + data_bytes);
    out += "WAVE";
    out += "fmt ";
    put_u32(out, 16);          // PCM fmt chunk size
    put_u16(out, 1);           // PCM
    put_u16(out, channels);
    put_u32(out, (uint32_t)sample_rate);
    put_u32(out, byte_rate);
    put_u16(out, block_align);
    put_u16(out, bits);
    out += "data";
    put_u32(out, data_bytes);
    for (int i = 0; i < n; ++i) {
        float v = pcm[i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        int32_t s = (int32_t)(v * 32767.0f);
        put_u16(out, (uint16_t)(int16_t)s);
    }
    return out;
}

// Raw little-endian f32 PCM (the runtime's preferred wire form — the JS
// ring buffer is f32 @ 24 kHz, no decode step).
static std::string pcm_f32_le(const float * pcm, int n) {
    std::string out;
    out.resize((size_t)n * sizeof(float));
    std::memcpy(out.data(), pcm, out.size());
    return out;
}

static server_http_res_ptr error_res(int status, const std::string & message) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->content_type = "application/json; charset=utf-8";
    json body = { { "error", { { "message", message }, { "type", "omnivoice_error" } } } };
    res->data = body.dump();
    return res;
}

static int env_int_clamped(const char * name, int fallback, int lo, int hi) {
    const char * v = std::getenv(name);
    if (!v || v[0] == '\0') return fallback;
    char * end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (end == v) return fallback;
    return (int) std::max((long) lo, std::min((long) hi, parsed));
}

static int json_int_clamped(const json & in, const char * name, int fallback, int lo, int hi) {
    if (!in.contains(name)) return fallback;
    try {
        if (in[name].is_number_integer()) {
            const int v = in[name].get<int>();
            return std::max(lo, std::min(hi, v));
        }
        if (in[name].is_string()) {
            const std::string s = in[name].get<std::string>();
            char * end = nullptr;
            long parsed = std::strtol(s.c_str(), &end, 10);
            if (end != s.c_str()) return (int) std::max((long) lo, std::min((long) hi, parsed));
        }
    } catch (...) {
    }
    return fallback;
}

static float json_float_positive(const json & in, const char * name, float fallback) {
    if (!in.contains(name)) return fallback;
    try {
        if (in[name].is_number()) {
            const float v = in[name].get<float>();
            return v > 0.0f ? v : fallback;
        }
        if (in[name].is_string()) {
            const std::string s = in[name].get<std::string>();
            char * end = nullptr;
            float parsed = std::strtof(s.c_str(), &end);
            if (end != s.c_str() && parsed > 0.0f) return parsed;
        }
    } catch (...) {
    }
    return fallback;
}

// handler_t for POST /v1/audio/speech.
static server_http_context::handler_t audio_speech_handler() {
    return [](const server_http_req & req) -> server_http_res_ptr {
        json in;
        try {
            in = req.body.empty() ? json::object() : json::parse(req.body);
        } catch (const std::exception & e) {
            return error_res(400, std::string("invalid JSON body: ") + e.what());
        }
        std::string text;
        if (in.contains("input") && in["input"].is_string()) {
            text = in["input"].get<std::string>();
        } else if (in.contains("text") && in["text"].is_string()) {
            text = in["text"].get<std::string>();
        }
        if (text.empty()) {
            return error_res(400, "missing or empty 'input' field");
        }
        std::string fmt = "wav";
        if (in.contains("response_format") && in["response_format"].is_string()) {
            fmt = in["response_format"].get<std::string>();
        }
        // 'voice' is accepted for OpenAI shape compatibility; the Eliza-1
        // bundle ships one default voice preset, so it is informational only
        // until per-voice presets are wired into omnivoice-core.

        std::string err;
        ov_context * ctx = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            ctx = acquire_locked(err);
        }
        if (!ctx) return error_res(503, err);

        ov_tts_params tp;
        ov_tts_default_params(&tp);
        tp.text = text.c_str();
        int mg_steps = env_int_clamped("ELIZA_OMNIVOICE_MG_NUM_STEP", tp.mg_num_step, 4, 64);
        mg_steps = json_int_clamped(in, "num_step", mg_steps, 4, 64);
        mg_steps = json_int_clamped(in, "num_steps", mg_steps, 4, 64);
        mg_steps = json_int_clamped(in, "steps", mg_steps, 4, 64);
        tp.mg_num_step = mg_steps;
        const float duration_sec = json_float_positive(in, "duration", 0.0f);
        if (duration_sec > 0.0f) {
            const int frames = ov_duration_sec_to_tokens(ctx, duration_sec);
            if (frames > 0) tp.T_override = frames;
        }
        ov_audio audio = {};
        ov_status st;
        {
            // ov_synthesize is not reentrant on one context; serialise.
            std::lock_guard<std::mutex> lk(g_mu);
            st = ov_synthesize(ctx, &tp, &audio);
        }
        if (st != OV_STATUS_OK) {
            const char * le = ov_last_error();
            ov_audio_free(&audio);
            return error_res(500, std::string("ov_synthesize failed (status ") +
                std::to_string((int)st) + "): " + (le ? le : "(no detail)"));
        }
        const int sample_rate = 24000; // omnivoice codec output rate
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        if (fmt == "pcm" || fmt == "f32" || fmt == "raw") {
            res->content_type = "application/octet-stream";
            res->headers["X-Sample-Rate"] = std::to_string(sample_rate);
            res->headers["X-Sample-Format"] = "f32le";
            res->data = pcm_f32_le(audio.samples, audio.n_samples);
        } else {
            res->content_type = "audio/wav";
            res->data = wav16_from_f32(audio.samples, audio.n_samples, sample_rate);
        }
        ov_audio_free(&audio);
        return res;
    };
}

} // namespace milady_omnivoice
#endif // MILADY_FUSE_OMNIVOICE
// end // MILADY-OMNIVOICE-AUDIO-SPEECH-ROUTE-V1

#include <atomic>
#include <clocale>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <windows.h>
#endif

static std::function<void(int)> shutdown_handler;
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // own arguments required by this example
    common_params params;

#ifdef MILADY_FUSE_OMNIVOICE
    // Strip omnivoice-fused-only flags before common_params_parse so the
    // upstream parser doesn't reject them. Values feed the lazily-created
    // OmniVoice context (see the milady_omnivoice namespace above).
    {
        std::vector<char *> filtered;
        filtered.reserve((size_t)argc);
        for (int i = 0; i < argc; ++i) {
            const std::string a = argv[i];
            if ((a == "--omnivoice-model" || a == "--omnivoice-codec") && i + 1 < argc) {
                if (a == "--omnivoice-model") milady_omnivoice::g_model_path = argv[++i];
                else                          milady_omnivoice::g_codec_path = argv[++i];
                continue;
            }
            filtered.push_back(argv[i]);
        }
        static std::vector<char *> s_filtered = filtered;
        argc = (int) s_filtered.size();
        argv = s_filtered.data();
    }
#endif

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    // validate batch size for embeddings
    // embeddings require all tokens to be processed in a single ubatch
    // see https://github.com/ggml-org/llama.cpp/issues/12836
    if (params.embedding && params.n_batch > params.n_ubatch) {
        LOG_WRN("%s: embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", __func__, params.n_batch, params.n_ubatch);
        LOG_WRN("%s: setting n_batch = n_ubatch = %d to avoid assertion failure\n", __func__, params.n_ubatch);
        params.n_batch = params.n_ubatch;
    }

    if (params.n_parallel < 0) {
        LOG_INF("%s: n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n", __func__);

        params.n_parallel = 4;
        params.kv_unified = true;
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    common_init();

    // struct that contains llama context and inference
    server_context ctx_server;

    llama_backend_init();
    llama_numa_init(params.numa);

    LOG_INF("system info: n_threads = %d, n_threads_batch = %d, total_threads = %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads, std::thread::hardware_concurrency());
    LOG_INF("\n");
    LOG_INF("%s\n", common_params_get_system_info(params).c_str());
    LOG_INF("\n");

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        LOG_ERR("%s: failed to initialize HTTP server\n", __func__);
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server);

    bool is_router_server = params.model.path.empty();
    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            LOG_ERR("%s: failed to initialize router models: %s\n", __func__, e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.get_api_show                = models_routes->proxy_get;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props  = models_routes->get_router_props;
        routes.get_models = models_routes->get_router_models;
        ctx_http.post("/models/load",   ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload", ex_wrapper(models_routes->post_router_models_unload));
    }

    ctx_http.get ("/health",              ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",           ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",             ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",               ex_wrapper(routes.get_props));
    ctx_http.post("/props",               ex_wrapper(routes.post_props));
    ctx_http.post("/api/show",            ex_wrapper(routes.get_api_show));
    ctx_http.get ("/models",              ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",           ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/api/tags",            ex_wrapper(routes.get_models)); // ollama specific endpoint. public endpoint (no API key check)
    ctx_http.post("/completion",          ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",         ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",      ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",    ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions", ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/api/chat",            ex_wrapper(routes.post_chat_completions)); // ollama specific endpoint
    ctx_http.post("/v1/responses",        ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",           ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/messages",         ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    ctx_http.post("/infill",              ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",           ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",          ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",       ex_wrapper(routes.post_embeddings_oai));
#ifdef MILADY_FUSE_OMNIVOICE
    // Fused omnivoice TTS — same process as the text/DFlash routes above.
    ctx_http.post("/v1/audio/speech",     ex_wrapper(milady_omnivoice::audio_speech_handler()));
    ctx_http.post("/audio/speech",        ex_wrapper(milady_omnivoice::audio_speech_handler()));
#endif
    ctx_http.post("/rerank",              ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",           ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",        ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",            ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",          ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",      ex_wrapper(routes.post_apply_template));
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",       ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",       ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",               ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",      ex_wrapper(routes.post_slots));

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        LOG_INF("%s: starting router server, no model will be loaded in this process\n", __func__);

        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            if (models_routes.has_value()) {
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            ctx_http.stop();
        };

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            ctx_http.stop();
            ctx_server.terminate();
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            LOG_ERR("%s: exiting due to HTTP server error\n", __func__);
            return 1;
        }

        // load the model
        LOG_INF("%s: loading model\n", __func__);

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            LOG_ERR("%s: exiting due to model loading error\n", __func__);
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        LOG_INF("%s: model loaded\n", __func__);

        shutdown_handler = [&](int) {
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // TODO: refactor in common/console
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = signal_handler;
    sigemptyset (&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
    sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif

    if (is_router_server) {
        LOG_INF("%s: router server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
        LOG_INF("%s: NOTE: router mode is experimental\n", __func__);
        LOG_INF("%s:       it is not recommended to use this mode in untrusted environments\n", __func__);
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        LOG_INF("%s: server is listening on %s\n", __func__, ctx_http.listening_address.c_str());
        LOG_INF("%s: starting the main loop...\n", __func__);

        // optionally, notify router server that this instance is ready
        const char * router_port = std::getenv("LLAMA_SERVER_ROUTER_PORT");
        std::thread monitor_thread;
        if (router_port != nullptr) {
            monitor_thread = server_models::setup_child_server(shutdown_handler);
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join();
        }
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            llama_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
