// SPDX-License-Identifier: MIT
//
// kokoro-tts.cpp — standalone CLI harness for the Kokoro fork inference path.
//
// Usage:
//     kokoro-tts \
//         --model    <path-to-kokoro-v1.0.gguf> \
//         --voice    <path-to-voices/af_sam.bin> \
//         --text     "Hello world." \
//         --output   <out.wav> \
//         [--speed 1.0]
//
// Exits 0 on a non-blank WAV being written; non-zero on any failure. Used
// by the J2 verification step + the `tools/voice-kokoro/` test harness.

#include "kokoro.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "Usage: %s --model <gguf> --voice <bin> --text \"...\" --output <wav> [--speed 1.0]\n",
        argv0);
}

// 16-bit little-endian mono WAV writer.
bool write_wav(const std::string & path, const std::vector<float> & samples, int sample_rate) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout) return false;

    const uint32_t n_samples = (uint32_t) samples.size();
    const uint32_t byte_rate = (uint32_t) sample_rate * 2;
    const uint32_t data_size = n_samples * 2;
    const uint32_t riff_size = 36 + data_size;

    auto put32 = [&fout](uint32_t v) {
        char b[4] = { (char) (v & 0xff), (char) ((v >> 8) & 0xff),
                      (char) ((v >> 16) & 0xff), (char) ((v >> 24) & 0xff) };
        fout.write(b, 4);
    };
    auto put16 = [&fout](uint16_t v) {
        char b[2] = { (char) (v & 0xff), (char) ((v >> 8) & 0xff) };
        fout.write(b, 2);
    };
    fout.write("RIFF", 4);
    put32(riff_size);
    fout.write("WAVE", 4);
    fout.write("fmt ", 4);
    put32(16);
    put16(1);
    put16(1);
    put32((uint32_t) sample_rate);
    put32(byte_rate);
    put16(2);
    put16(16);
    fout.write("data", 4);
    put32(data_size);
    for (uint32_t i = 0; i < n_samples; ++i) {
        float v = samples[i];
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        const int16_t s = (int16_t) std::lrintf(v * 32767.0f);
        char b[2] = { (char) (s & 0xff), (char) ((s >> 8) & 0xff) };
        fout.write(b, 2);
    }
    return (bool) fout;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, voice_path, text, out_path;
    float speed = 1.0f;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--model"  && i + 1 < argc) model_path = argv[++i];
        else if (a == "--voice"  && i + 1 < argc) voice_path = argv[++i];
        else if (a == "--text"   && i + 1 < argc) text       = argv[++i];
        else if (a == "--output" && i + 1 < argc) out_path   = argv[++i];
        else if (a == "--speed"  && i + 1 < argc) speed      = (float) std::atof(argv[++i]);
        else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }

    if (model_path.empty() || voice_path.empty() || text.empty() || out_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    std::string err;
    auto model = eliza_kokoro::kokoro_load_model(model_path, err);
    if (!model) {
        std::fprintf(stderr, "kokoro_load_model failed: %s\n", err.c_str());
        return 1;
    }
    const auto * hp = eliza_kokoro::kokoro_get_hparams(model.get());

    eliza_kokoro::kokoro_voice_preset voice;
    const auto vst = eliza_kokoro::kokoro_load_voice_preset(voice_path, hp->style_dim, voice, err);
    if (vst != eliza_kokoro::KOKORO_OK) {
        std::fprintf(stderr, "kokoro_load_voice_preset failed: %s (status=%s)\n",
                     err.c_str(), eliza_kokoro::kokoro_status_str(vst));
        return 1;
    }
    voice.id = voice_path;

    eliza_kokoro::kokoro_audio audio;
    const auto sst = eliza_kokoro::kokoro_synthesize(model.get(), voice, text, speed, audio, err);
    if (sst != eliza_kokoro::KOKORO_OK) {
        std::fprintf(stderr, "kokoro_synthesize failed: %s (status=%s)\n",
                     err.c_str(), eliza_kokoro::kokoro_status_str(sst));
        return 1;
    }

    if (audio.samples.empty()) {
        std::fprintf(stderr, "kokoro_synthesize returned no samples\n");
        return 1;
    }

    if (!write_wav(out_path, audio.samples, audio.sample_rate)) {
        std::fprintf(stderr, "write_wav failed: %s\n", out_path.c_str());
        return 1;
    }

    // Diagnostic: peak amplitude. A blank WAV would have peak ~0.
    float peak = 0.0f;
    for (float v : audio.samples) {
        const float a = std::fabs(v);
        if (a > peak) peak = a;
    }
    std::fprintf(stdout,
        "kokoro-tts: wrote %s (samples=%zu, rate=%d, peak=%.4f)\n",
        out_path.c_str(),
        audio.samples.size(),
        audio.sample_rate,
        peak);
    return peak > 1e-6f ? 0 : 1;
}
