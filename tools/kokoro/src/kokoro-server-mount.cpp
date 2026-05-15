// SPDX-License-Identifier: MIT
//
// kokoro-server-mount.cpp — committed implementation of the Kokoro
// `/v1/audio/speech` HTTP route mounted on `llama-server` when
// LLAMA_BUILD_KOKORO=ON. See kokoro-server-mount.h for the contract.

#ifdef LLAMA_BUILD_KOKORO

#include "kokoro-server-mount.h"
#include "kokoro.h"
#include "server-http.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace eliza_kokoro {

std::string g_model_path;
std::string g_voices_dir;

namespace {

std::mutex g_mu;
kokoro_model_ptr g_model{nullptr, kokoro_model_deleter{}};

server_http_res_ptr error_res(int status, const std::string & msg) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->content_type = "application/json";
    json err = { {"error", msg}, {"status", status} };
    res->data = err.dump();
    // server_http_res::data is a string; we use it for either text/json or
    // binary blobs (audio response below sets content_type to audio/wav and
    // packs raw bytes into the same string).
    return res;
}

// voiceId path-safety: lowercase letters/digits/dot-underscore-dash, no parent.
bool safe_voice_id(const std::string & v) {
    if (v.empty()) return false;
    if (v.find("..") != std::string::npos) return false;
    for (char c : v) {
        if (!(std::isalnum((unsigned char) c) || c == '.' || c == '_' || c == '-'))
            return false;
    }
    return true;
}

std::string resolve_voice_path(const std::string & voice_id) {
    if (!safe_voice_id(voice_id)) return {};
    if (g_voices_dir.empty()) return {};
    std::string p = g_voices_dir;
    if (!p.empty() && p.back() != '/') p.push_back('/');
    p += voice_id;
    p += ".bin";
    return p;
}

// 16-bit little-endian PCM WAV header + samples. 24kHz mono.
std::string wav16_from_f32(const std::vector<float> & samples, int sample_rate) {
    const uint32_t n_samples = (uint32_t) samples.size();
    const uint32_t byte_rate = (uint32_t) sample_rate * 2;
    const uint32_t data_size = n_samples * 2;
    const uint32_t riff_size = 36 + data_size;

    std::string out;
    out.reserve(44 + (size_t) data_size);
    auto put32 = [&out](uint32_t v) {
        char b[4] = { (char) (v & 0xff), (char) ((v >> 8) & 0xff),
                      (char) ((v >> 16) & 0xff), (char) ((v >> 24) & 0xff) };
        out.append(b, 4);
    };
    auto put16 = [&out](uint16_t v) {
        char b[2] = { (char) (v & 0xff), (char) ((v >> 8) & 0xff) };
        out.append(b, 2);
    };
    out.append("RIFF", 4);
    put32(riff_size);
    out.append("WAVE", 4);
    out.append("fmt ", 4);
    put32(16);
    put16(1);                     // PCM
    put16(1);                     // mono
    put32((uint32_t) sample_rate);
    put32(byte_rate);
    put16(2);                     // block align
    put16(16);                    // bits per sample
    out.append("data", 4);
    put32(data_size);
    for (uint32_t i = 0; i < n_samples; ++i) {
        float v = samples[i];
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        const int16_t s = (int16_t) std::lrintf(v * 32767.0f);
        char b[2] = { (char) (s & 0xff), (char) ((s >> 8) & 0xff) };
        out.append(b, 2);
    }
    return out;
}

// Raw little-endian f32 PCM payload (no header).
std::string pcm_f32_le(const std::vector<float> & samples) {
    std::string out;
    out.resize(samples.size() * sizeof(float));
    std::memcpy(out.data(), samples.data(), samples.size() * sizeof(float));
    return out;
}

kokoro_model * acquire_model_locked(std::string & err) {
    if (g_model) return g_model.get();
    if (g_model_path.empty()) {
        err = "kokoro: no model path set (pass --kokoro-model <path-to-gguf>)";
        return nullptr;
    }
    g_model = kokoro_load_model(g_model_path, err);
    return g_model ? g_model.get() : nullptr;
}

} // namespace

bool is_enabled() noexcept {
    return !g_model_path.empty();
}

server_http_context::handler_t audio_speech_handler() {
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

        std::string voice_id;
        if (in.contains("voice") && in["voice"].is_string()) {
            voice_id = in["voice"].get<std::string>();
        }
        if (voice_id.empty()) voice_id = "af_sam";

        std::string fmt = "wav";
        if (in.contains("response_format") && in["response_format"].is_string()) {
            fmt = in["response_format"].get<std::string>();
        }
        float speed = 1.0f;
        if (in.contains("speed") && in["speed"].is_number()) {
            speed = (float) in["speed"].get<double>();
            if (!(speed > 0.05f && speed < 10.0f)) speed = 1.0f;
        }

        const std::string voice_path = resolve_voice_path(voice_id);
        if (voice_path.empty()) {
            return error_res(400, "invalid voice id");
        }

        std::string err;
        kokoro_model * model = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            model = acquire_model_locked(err);
        }
        if (!model) return error_res(503, err);

        kokoro_voice_preset voice;
        const kokoro_status vst = kokoro_load_voice_preset(
            voice_path,
            kokoro_get_hparams(model)->style_dim,
            voice,
            err);
        if (vst != KOKORO_OK) {
            return error_res(400, std::string("voice load failed: ") + err);
        }
        voice.id = voice_id;

        kokoro_audio audio;
        const kokoro_status sst = kokoro_synthesize(model, voice, text, speed, audio, err);
        if (sst != KOKORO_OK) {
            return error_res(500, std::string("synthesize failed: ") + err);
        }
        const int sample_rate = audio.sample_rate;

        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        if (fmt == "pcm" || fmt == "f32" || fmt == "raw") {
            res->content_type = "application/octet-stream";
            res->headers["X-Sample-Rate"] = std::to_string(sample_rate);
            res->headers["X-Sample-Format"] = "f32le";
            res->data = pcm_f32_le(audio.samples);
        } else {
            res->content_type = "audio/wav";
            res->data = wav16_from_f32(audio.samples, sample_rate);
        }
        return res;
    };
}

} // namespace eliza_kokoro

#endif // LLAMA_BUILD_KOKORO
