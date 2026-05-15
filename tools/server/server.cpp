#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

// ELIZA-OMNIVOICE-AUDIO-SPEECH-ROUTE-V1
#ifdef ELIZA_FUSE_OMNIVOICE
#include "omnivoice.h"
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

namespace eliza_omnivoice {

// Resolve a config value: prefer the CLI override captured in main(), then
// the env var, then empty.
static std::string g_model_path;
static std::string g_codec_path;

static std::string resolved_model_path();
static std::string resolved_codec_path();

// LE byte readers shared with the FFI preset parser. Inline so we don't
// depend on the FFI bridge — this route runs in llama-server, not the
// libelizainference dylib.
static uint32_t ov_route_le_u32(const uint8_t * p) {
    return  (uint32_t) p[0]
          | ((uint32_t) p[1] << 8)
          | ((uint32_t) p[2] << 16)
          | ((uint32_t) p[3] << 24);
}
static int32_t ov_route_le_i32(const uint8_t * p) {
    return (int32_t) ov_route_le_u32(p);
}

// Holds a parsed v2 ELZ2 voice preset payload (just the bits the
// synth path consumes). Owns the int32 token storage.
struct route_voice_preset {
    std::string instruct;
    std::string ref_text;
    std::vector<int32_t> ref_audio_tokens;
    int K = 0;
    int ref_T = 0;
    bool empty_payload = true;
};

// voiceId path-safety: lowercase letters/digits/dot-underscore-dash, no parent.
static bool ov_route_safe_voice_id(const std::string & v) {
    if (v.empty()) return false;
    if (v.find("..") != std::string::npos) return false;
    for (char c : v) {
        if (!(std::isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-'))
            return false;
    }
    return true;
}

// Resolve <bundle_dir> from the model path. Model lives at
// <bundle>/tts/omnivoice-base.gguf; we walk two levels up. Falls back
// to the parent dir if the path is non-canonical.
static std::string ov_route_bundle_dir() {
    const std::string model = resolved_model_path();
    if (model.empty()) return std::string();
    // strip filename
    auto pos = model.find_last_of('/');
    if (pos == std::string::npos) return std::string();
    std::string parent = model.substr(0, pos);
    // strip "tts" subdir if present
    auto pos2 = parent.find_last_of('/');
    if (pos2 == std::string::npos) return parent;
    std::string maybe_bundle = parent.substr(0, pos2);
    std::string leaf = parent.substr(pos2 + 1);
    if (leaf == "tts" || leaf == "voice" || leaf == "speech") {
        return maybe_bundle;
    }
    return parent;
}

// Read entire preset file into a buffer. Returns false on missing/IO error.
static bool ov_route_read_file(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long n = std::ftell(f);
    if (n < 0) { std::fclose(f); return false; }
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) n);
    size_t got = std::fread(out.data(), 1, (size_t) n, f);
    std::fclose(f);
    return got == (size_t) n;
}

// Parse v2 ELZ2 voice preset bytes into `out`. v1 files are accepted
// (no payload extracted — empty_payload stays true). Returns false on
// hard format errors with `err` populated; missing/unsafe inputs are
// the caller's job to surface.
static bool ov_route_parse_preset(const std::vector<uint8_t> & bytes,
                                  route_voice_preset & out,
                                  std::string & err) {
    if (bytes.size() < 24) { err = "voice preset truncated header"; return false; }
    const uint8_t * p = bytes.data();
    const size_t len = bytes.size();
    if (ov_route_le_u32(p) != 0x315A4C45u /* 'ELZ1' */) {
        err = "voice preset bad magic"; return false;
    }
    const uint32_t version = ov_route_le_u32(p + 4);
    if (version != 1u && version != 2u) {
        err = "voice preset unsupported version"; return false;
    }
    if (version == 1u) {
        // v1 only carries Kokoro-style embedding + phrase seed. Nothing
        // for the synth path to apply. Keep empty_payload = true.
        return true;
    }
    if (len < 64) { err = "voice preset v2 truncated header"; return false; }
    auto sec_at = [&](size_t off, uint32_t & soff, uint32_t & ssz) {
        soff = ov_route_le_u32(p + off);
        ssz  = ov_route_le_u32(p + off + 4);
    };
    uint32_t ref_tok_off = 0, ref_tok_sz = 0;
    uint32_t ref_txt_off = 0, ref_txt_sz = 0;
    uint32_t instr_off = 0, instr_sz = 0;
    sec_at(24, ref_tok_off, ref_tok_sz);
    sec_at(32, ref_txt_off, ref_txt_sz);
    sec_at(40, instr_off, instr_sz);
    auto in_bounds = [&](uint32_t soff, uint32_t ssz) {
        if (ssz == 0) return true;
        if (soff < 64) return false;
        return (size_t) soff + (size_t) ssz <= len;
    };
    if (!in_bounds(ref_tok_off, ref_tok_sz) ||
        !in_bounds(ref_txt_off, ref_txt_sz) ||
        !in_bounds(instr_off, instr_sz)) {
        err = "voice preset section out of bounds"; return false;
    }
    if (ref_tok_sz > 0) {
        if (ref_tok_sz < 8) { err = "voice preset ref_audio_tokens truncated"; return false; }
        const uint8_t * rt = p + ref_tok_off;
        const uint32_t K = ov_route_le_u32(rt);
        const uint32_t refT = ov_route_le_u32(rt + 4);
        if ((size_t) ref_tok_sz - 8 != (size_t) K * (size_t) refT * 4u) {
            err = "voice preset ref_audio_tokens shape mismatch"; return false;
        }
        out.K = (int) K;
        out.ref_T = (int) refT;
        out.ref_audio_tokens.resize((size_t) K * (size_t) refT);
        for (size_t i = 0; i < out.ref_audio_tokens.size(); ++i) {
            out.ref_audio_tokens[i] = ov_route_le_i32(rt + 8 + i * 4u);
        }
    }
    if (ref_txt_sz > 0) {
        out.ref_text.assign(reinterpret_cast<const char *>(p + ref_txt_off),
                            (size_t) ref_txt_sz);
    }
    if (instr_sz > 0) {
        out.instruct.assign(reinterpret_cast<const char *>(p + instr_off),
                            (size_t) instr_sz);
    }
    out.empty_payload =
        out.instruct.empty() && out.ref_text.empty() && out.ref_audio_tokens.empty();
    return true;
}

// Resolve `voice` to a preset. Returns false (with err set) when the
// id is unsafe or the file is missing/malformed. Returns true with
// empty_payload=true when the voice resolves to "default" or "" —
// in those cases the caller stays on OmniVoice auto-voice.
static bool ov_route_load_voice(const std::string & voice,
                                route_voice_preset & out,
                                std::string & err) {
    if (voice.empty() || voice == "default") return true;
    if (!ov_route_safe_voice_id(voice)) {
        err = "voice id is not a safe single segment: " + voice;
        return false;
    }
    const std::string bundle = ov_route_bundle_dir();
    if (bundle.empty()) {
        err = "voice preset bundle dir not resolvable from --omnivoice-model";
        return false;
    }
    const std::string path = bundle + "/cache/voice-preset-" + voice + ".bin";
    std::vector<uint8_t> bytes;
    if (!ov_route_read_file(path, bytes)) {
        err = "voice preset file not found or unreadable: " + path;
        return false;
    }
    return ov_route_parse_preset(bytes, out, err);
}

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
        // OpenAI-compatible `voice` field. Resolves to
        // <bundle>/cache/voice-preset-<voice>.bin (ELZ2 v2). v1 / empty /
        // missing presets fall through to OmniVoice auto-voice mode.
        // R6 §3.3 / brief §3: load instruct + ref_audio_tokens + ref_text
        // before threading into ov_tts_params.
        std::string voice_id;
        if (in.contains("voice") && in["voice"].is_string()) {
            voice_id = in["voice"].get<std::string>();
        }
        route_voice_preset preset;
        std::string preset_err;
        const bool preset_ok = ov_route_load_voice(voice_id, preset, preset_err);
        if (!preset_ok) {
            // A bad/malformed preset id is a 400, not a silent fall-through.
            return error_res(400, std::string("invalid voice preset: ") + preset_err);
        }
        // `?interactive=0` (or JSON {"interactive": false}) is the
        // explicit non-interactive path. The default route keeps the
        // synchronous-mutex behaviour; interactive turns are expected
        // to use the FFI streaming path (`ttsSynthesizeStream`) where
        // mid-utterance cancellation is supported (R11 / brief §6
        // Path B).
        bool interactive = true;
        if (in.contains("interactive") && in["interactive"].is_boolean()) {
            interactive = in["interactive"].get<bool>();
        }

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
        // Default to OmniVoice's auto-voice path. Preset (if any) overrides
        // params.instruct / ref_audio_tokens / ref_text below.
        tp.instruct = "";
        if (!preset.empty_payload) {
            if (!preset.instruct.empty()) tp.instruct = preset.instruct.c_str();
            if (!preset.ref_audio_tokens.empty() && preset.K > 0 && preset.ref_T > 0) {
                tp.ref_audio_tokens = preset.ref_audio_tokens.data();
                tp.ref_T = preset.ref_T;
            }
            if (!preset.ref_text.empty()) tp.ref_text = preset.ref_text.c_str();
        }
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
        if (interactive) {
            // Interactive turns should route through the FFI streaming
            // path. Return 409 with the diagnostic so the JS layer can
            // pick `OmniVoiceFfiBackend.ttsStream` instead of the HTTP
            // route. The HTTP route stays available for batch jobs that
            // explicitly opt out with `interactive: false`.
            return error_res(409, "interactive=true: use FFI streaming "
                                  "(eliza_inference_tts_synthesize_stream) for "
                                  "mid-utterance cancellation; pass "
                                  "{\"interactive\": false} to use this batch route");
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

} // namespace eliza_omnivoice
#endif // ELIZA_FUSE_OMNIVOICE
// end // ELIZA-OMNIVOICE-AUDIO-SPEECH-ROUTE-V1

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

    common_init();

#ifdef ELIZA_FUSE_OMNIVOICE
    // Strip omnivoice-fused-only flags before common_params_parse so the
    // upstream parser doesn't reject them. Values feed the lazily-created
    // OmniVoice context (see the eliza_omnivoice namespace above).
    {
        std::vector<char *> filtered;
        filtered.reserve((size_t)argc);
        for (int i = 0; i < argc; ++i) {
            const std::string a = argv[i];
            if ((a == "--omnivoice-model" || a == "--omnivoice-codec") && i + 1 < argc) {
                if (a == "--omnivoice-model") eliza_omnivoice::g_model_path = argv[++i];
                else                          eliza_omnivoice::g_codec_path = argv[++i];
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

    llama_backend_init();
    llama_numa_init(params.numa);

    common_params_print_info(params);

    // validate batch size for embeddings
    // embeddings require all tokens to be processed in a single ubatch
    // see https://github.com/ggml-org/llama.cpp/issues/12836
    if (params.embedding && params.n_batch > params.n_ubatch) {
        SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
        SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
        params.n_batch = params.n_ubatch;
    }

    if (params.n_parallel < 0) {
        SRV_INF("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

        params.n_parallel = 4;
        params.kv_unified = true;
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    if (params.model_alias.empty() && !params.model.name.empty()) {
        params.model_alias.insert(params.model.name);
    }

    // struct that contains llama context and inference
    server_context ctx_server;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server);
    server_tools tools;

    bool is_router_server = params.model.path.empty();
    std::optional<server_models_routes> models_routes{};
    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
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
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models)); // public endpoint (no API key check)
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    ctx_http.post("/chat/completions",         ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(routes.post_chat_completions));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
#ifdef ELIZA_FUSE_OMNIVOICE
    // Fused omnivoice TTS — same process as the text/DFlash routes above.
    ctx_http.post("/v1/audio/speech",          ex_wrapper(eliza_omnivoice::audio_speech_handler()));
    ctx_http.post("/audio/speech",             ex_wrapper(eliza_omnivoice::audio_speech_handler()));
#endif
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/v1/messages/count_tokens", ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    if (params.webui_mcp_proxy) {
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "CORS proxy is enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be removed or changed in future versions\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
    }
    // EXPERIMENTAL built-in tools
    if (!params.server_tools.empty()) {
        try {
            tools.setup(params.server_tools);
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        SRV_WRN("%s", "-----------------\n");
        SRV_WRN("%s", "Built-in tools are enabled, do not expose server to untrusted environments\n");
        SRV_WRN("%s", "This feature is EXPERIMENTAL and may be changed in the future\n");
        SRV_WRN("%s", "-----------------\n");
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting router server, no model will be loaded in this process\n");

        clean_up = [&models_routes]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            if (models_routes.has_value()) {
                models_routes->models.unload_all();
            }
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
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
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // load the model
        SRV_INF("%s", "loading model\n");

        if (server_models::is_child_server()) {
            ctx_server.on_sleeping_changed([&](bool sleeping) {
                server_models::notify_router_sleeping_state(sleeping);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            if (ctx_http.thread.joinable()) {
                ctx_http.thread.join();
            }
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

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
        SRV_INF("router server is listening on %s\n", ctx_http.listening_address.c_str());
        SRV_WRN("%s", "NOTE: router mode is experimental\n");
        SRV_WRN("%s", "      it is not recommended to use this mode in untrusted environments\n");
        if (ctx_http.thread.joinable()) {
            ctx_http.thread.join(); // keep the main thread alive
        }

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        SRV_INF("server is listening on %s\n", ctx_http.listening_address.c_str());

        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (server_models::is_child_server()) {
            json model_info = routes.get_model_info();
            monitor_thread = server_models::setup_child_server(shutdown_handler, model_info);
        }

        // TODO(dflash-native-events): after verifier batch completes, emit:
        // { "type": "dflash_event", "native": true, "draft_tokens": [...],
        //   "accept_count": N, "reject_range": [s,e]|null,
        //   "accept_tokens": [...], "timing": { "proposal_ms": X, "verify_ms": Y } }
        // Use send_event(slot_id, json_str) which already exists in this file.
        // Flag: --dflash-emit-events (default on when drafter loaded).
        // The speculative-decode slot loop (server-context.h) is the emission
        // point: hook into the post-verify step where llama_speculative_decode
        // returns the accepted token mask and corrected token.

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
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
