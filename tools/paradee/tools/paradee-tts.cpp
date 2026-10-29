// SPDX-License-Identifier: MIT
//
// paradee-tts — synthesize misaki phonemes with Paradee-8M to a 24 kHz WAV.
//
//   paradee-tts --model paradee-8m-v1.0-f32.gguf --phonemes "həlˈO wˈɜɹld." --output hello.wav
//               [--speed 1.0] [--voice NAME] [--seed N] [--f32] [--repeat N]
//
// --voice picks a row of a v1.2 file's voice table (default: the file's
// default voice; --voice list prints them). --seed is the v1.1 / v1.2
// head's noise seed.
//
// The input is misaki's phoneme spelling (Kokoro's G2P), not text and not
// eSpeak IPA. --repeat N synthesizes N times and reports the real-time factor
// of the fastest run (thread count: KOKORO_NUM_THREADS; on Apple also
// VECLIB_MAXIMUM_THREADS).

#include "paradee.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(const char * argv0) {
    std::fprintf(stderr,
        "usage: %s --model <gguf> (--phonemes \"...\" | --phonemes-file <utf8.txt>) --output <wav>\n"
        "          [--speed 1.0] [--voice NAME|list] [--seed N] [--f32] [--repeat N]\n", argv0);
}

bool write_wav(const std::string & path, const std::vector<float> & samples, int sample_rate, bool f32) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    const uint32_t n = (uint32_t) samples.size();
    const uint16_t bits = f32 ? 32 : 16, fmt = f32 ? 3 : 1;
    const uint32_t bytes = n * (bits / 8u);
    auto p32 = [&f](uint32_t v) { char b[4] = { (char) v, (char) (v >> 8), (char) (v >> 16), (char) (v >> 24) }; f.write(b, 4); };
    auto p16 = [&f](uint16_t v) { char b[2] = { (char) v, (char) (v >> 8) }; f.write(b, 2); };
    f.write("RIFF", 4); p32(36 + bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); p32(16); p16(fmt); p16(1); p32((uint32_t) sample_rate);
    p32((uint32_t) sample_rate * (bits / 8u)); p16(bits / 8); p16(bits);
    f.write("data", 4); p32(bytes);
    for (uint32_t i = 0; i < n; ++i) {
        if (f32) {
            f.write((const char *) &samples[i], 4);
        } else {
            const float v = std::max(-1.0f, std::min(1.0f, samples[i]));
            const int16_t s = (int16_t) std::lrintf(v * 32767.0f);
            p16((uint16_t) s);
        }
    }
    return (bool) f;
}

} // namespace

int main(int argc, char ** argv) {
    std::string model_path, phonemes, out_path;
    float speed = 1.0f;
    bool f32 = false;
    int repeat = 1;
    std::string voice_name;
    uint64_t seed = eliza_paradee::PARADEE_DEFAULT_NOISE_SEED;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char * { return i + 1 < argc ? argv[++i] : nullptr; };
        const char * v = nullptr;
        if (a == "--model" && (v = next())) model_path = v;
        else if (a == "--phonemes" && (v = next())) phonemes = v;
        else if (a == "--phonemes-file" && (v = next())) {
            std::ifstream in(v, std::ios::binary);
            std::stringstream ss; ss << in.rdbuf(); phonemes = ss.str();
            while (!phonemes.empty() && (phonemes.back() == '\n' || phonemes.back() == '\r')) phonemes.pop_back();
        }
        else if (a == "--output" && (v = next())) out_path = v;
        else if (a == "--speed" && (v = next())) speed = std::strtof(v, nullptr);
        else if (a == "--repeat" && (v = next())) repeat = std::max(1, std::atoi(v));
        else if (a == "--voice" && (v = next())) voice_name = v;
        else if (a == "--seed" && (v = next())) seed = std::strtoull(v, nullptr, 10);
        else if (a == "--f32") f32 = true;
        else { usage(argv[0]); return 2; }
    }
    const bool list_voices = voice_name == "list";
    if (model_path.empty() || (!list_voices && (phonemes.empty() || out_path.empty()))) { usage(argv[0]); return 2; }

    std::string err;
    auto t0 = std::chrono::steady_clock::now();
    auto model = eliza_paradee::paradee_load_model(model_path, err);
    if (!model) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (list_voices) {
        for (int i = 0; i < eliza_paradee::paradee_voice_count(model.get()); ++i) {
            std::printf("%d %s %s%s\n", i, eliza_paradee::paradee_voice_name(model.get(), i),
                        eliza_paradee::paradee_voice_accent(model.get(), i),
                        i == eliza_paradee::paradee_default_voice(model.get()) ? " (default)" : "");
        }
        return 0;
    }
    eliza_paradee::paradee_synth_options opts;
    opts.speed = speed;
    opts.noise_seed = seed;
    if (!voice_name.empty()) {
        opts.voice = eliza_paradee::paradee_find_voice(model.get(), voice_name);
        if (opts.voice < 0) { std::fprintf(stderr, "no voice '%s' (--voice list)\n", voice_name.c_str()); return 1; }
    }

    std::vector<int32_t> ids;
    eliza_paradee::paradee_status st = eliza_paradee::paradee_phonemes_to_ids(model.get(), phonemes, ids, err);
    if (st != eliza_paradee::PARADEE_OK) {
        std::fprintf(stderr, "phonemes rejected (%s): %s\n", eliza_paradee::paradee_status_str(st), err.c_str());
        return 1;
    }
    eliza_paradee::paradee_audio audio;
    std::vector<float> & pcm = audio.samples;
    double best = 1e30;
    for (int r = 0; r < repeat; ++r) {
        const auto t = std::chrono::steady_clock::now();
        st = eliza_paradee::paradee_synthesize(model.get(), ids, opts, audio, err);
        best = std::min(best, std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count());
        if (st != eliza_paradee::PARADEE_OK) {
            std::fprintf(stderr, "synthesis failed (%s): %s\n", eliza_paradee::paradee_status_str(st), err.c_str());
            return 1;
        }
    }
    const int sr = eliza_paradee::paradee_sample_rate(model.get());
    if (!write_wav(out_path, pcm, sr, f32)) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    const double secs = (double) pcm.size() / sr;
    std::printf("ids=%zu samples=%zu audio=%.2fs load=%.0fms synth=%.3fs rtf=%.4f (%.1fx realtime)\n",
                ids.size(), pcm.size(), secs, load_ms, best, best / secs, secs / best);
    return 0;
}
