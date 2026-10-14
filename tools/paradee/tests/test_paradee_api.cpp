// SPDX-License-Identifier: MIT
//
// test-paradee-api — the eliza_paradee C++ API contract on the real model:
// loading, phoneme mapping (pads, unknown symbols, length limit, UTF-8),
// input validation, determinism, speed, and cancellation (before the call,
// between stages and inside the generator loops).
//
// Needs PARADEE_GGUF (environment); exits 77 (CTest SKIP) without it.

#include "paradee.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace eliza_paradee;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++failures; } } while (0)

struct counter { std::atomic<int> polls{0}; int stop_after = 0; };
static bool cancel_after(void * u) {
    counter * c = (counter *) u;
    return ++c->polls > c->stop_after;
}

int main() {
    const char * gguf = std::getenv("PARADEE_GGUF");
    if (!gguf || !*gguf) {
        std::printf("test-paradee-api: PARADEE_GGUF not set, skipping\n");
        return 77;
    }
    std::string err;

    // Loader rejections.
    CHECK(!paradee_load_model("/nonexistent/paradee.gguf", err) && !err.empty(), "missing file accepted");
    auto model = paradee_load_model(gguf, err);
    if (!model) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    CHECK(paradee_sample_rate(model.get()) == 24000, "sample rate %d", paradee_sample_rate(model.get()));
    CHECK(paradee_sample_rate(nullptr) == 0, "null model sample rate");
    const paradee_lock_params * lp = paradee_get_lock_params(model.get());
    CHECK(lp && lp->n_fft == 1024 && lp->hop == 256 && lp->smooth_frames == 33 && lp->f0_hop == 300 &&
          lp->band_lo_hz == 2000.0f && lp->band_hi_hz == 8000.0f && lp->voiced_f0_hz == 60.0f,
          "lock parameters not read from the GGUF");

    // Phonemes -> ids.
    std::vector<int32_t> ids;
    CHECK(paradee_phonemes_to_ids(model.get(), "həlˈO wˈɜɹld.", ids, err) == PARADEE_OK, "%s", err.c_str());
    const std::vector<int32_t> want = { 0, 50, 83, 54, 156, 31, 16, 65, 156, 87, 123, 54, 46, 4, 0 };
    CHECK(ids == want, "hello world ids differ (got %zu ids)", ids.size());
    CHECK(paradee_phonemes_to_ids(model.get(), "hɛlo#", ids, err) == PARADEE_E_UNKNOWN_SYMBOL &&
          err.find("U+0023") != std::string::npos && ids.empty(), "unknown symbol: %s", err.c_str());
    CHECK(paradee_phonemes_to_ids(model.get(), "ɡ", ids, err) == PARADEE_OK && ids.size() == 3 && ids[1] == 92,
          "U+0261 script g");
    CHECK(paradee_phonemes_to_ids(model.get(), "g", ids, err) == PARADEE_E_UNKNOWN_SYMBOL, "ASCII g is not in the vocab");
    CHECK(paradee_phonemes_to_ids(model.get(), "", ids, err) == PARADEE_E_INVALID_ARG, "empty accepted");
    CHECK(paradee_phonemes_to_ids(model.get(), std::string("a\xC3", 2), ids, err) == PARADEE_E_INVALID_ARG,
          "truncated UTF-8 accepted");
    CHECK(paradee_phonemes_to_ids(model.get(), std::string("\xC0\xAF"), ids, err) == PARADEE_E_INVALID_ARG,
          "overlong UTF-8 accepted");
    CHECK(paradee_phonemes_to_ids(nullptr, "a", ids, err) == PARADEE_E_INVALID_ARG, "null model");
    {
        std::string ok510(510, 'a'), long511(511, 'a');
        CHECK(paradee_phonemes_to_ids(model.get(), ok510, ids, err) == PARADEE_OK && ids.size() == 512,
              "510 phonemes must give 512 ids");
        CHECK(paradee_phonemes_to_ids(model.get(), long511, ids, err) == PARADEE_E_TOO_LONG && ids.empty(),
              "511 phonemes accepted");
    }

    // Unpadded, unlimited form for chunkers.
    {
        std::vector<int32_t> sym;
        CHECK(paradee_phonemes_to_symbol_ids(model.get(), "həlˈO wˈɜɹld.", sym, err) == PARADEE_OK &&
              sym == std::vector<int32_t>(want.begin() + 1, want.end() - 1), "symbol ids without pads");
        CHECK(paradee_phonemes_to_symbol_ids(model.get(), std::string(2000, 'a'), sym, err) == PARADEE_OK &&
              sym.size() == 2000, "symbol ids must not be length-limited");
        CHECK(paradee_phonemes_to_symbol_ids(model.get(), "a#", sym, err) == PARADEE_E_UNKNOWN_SYMBOL && sym.empty(),
              "symbol ids must reject unknown symbols");
    }

    // Synthesis input validation.
    std::vector<float> pcm;
    paradee_phonemes_to_ids(model.get(), "həlˈO wˈɜɹld.", ids, err);
    {
        std::vector<int32_t> nopad(ids.begin() + 1, ids.end());
        CHECK(paradee_synthesize_ids(model.get(), nopad.data(), nopad.size(), 1.0f, nullptr, nullptr, pcm, err) ==
              PARADEE_E_INVALID_ARG, "missing leading pad accepted");
        std::vector<int32_t> bad = ids; bad[3] = 178;
        CHECK(paradee_synthesize_ids(model.get(), bad.data(), bad.size(), 1.0f, nullptr, nullptr, pcm, err) ==
              PARADEE_E_INVALID_ARG, "id 178 accepted");
        const int32_t two[2] = { 0, 0 };
        CHECK(paradee_synthesize_ids(model.get(), two, 2, 1.0f, nullptr, nullptr, pcm, err) == PARADEE_E_INVALID_ARG,
              "2 ids accepted");
        std::vector<int32_t> big(513, 43); big.front() = 0; big.back() = 0;
        CHECK(paradee_synthesize_ids(model.get(), big.data(), big.size(), 1.0f, nullptr, nullptr, pcm, err) ==
              PARADEE_E_TOO_LONG, "513 ids accepted");
        CHECK(paradee_synthesize_ids(model.get(), ids.data(), ids.size(), 0.0f, nullptr, nullptr, pcm, err) ==
              PARADEE_E_INVALID_ARG, "speed 0 accepted");
        CHECK(paradee_synthesize_ids(model.get(), ids.data(), ids.size(), NAN, nullptr, nullptr, pcm, err) ==
              PARADEE_E_INVALID_ARG, "speed NaN accepted");
        CHECK(paradee_synthesize_ids(nullptr, ids.data(), ids.size(), 1.0f, nullptr, nullptr, pcm, err) ==
              PARADEE_E_INVALID_ARG, "null model");
    }

    // Synthesis: deterministic, sane level, speed shortens.
    std::vector<float> a, b, fast;
    CHECK(paradee_synthesize_ids(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, a, err) == PARADEE_OK,
          "%s", err.c_str());
    CHECK(paradee_synthesize_ids(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, b, err) == PARADEE_OK,
          "%s", err.c_str());
    CHECK(!a.empty() && a == b, "synthesis is not deterministic");
    CHECK(a.size() % 600 == 0, "length %zu is not a whole number of frames", a.size());
    double rms = 0, peak = 0;
    for (float v : a) { rms += (double) v * v; peak = std::max(peak, (double) std::fabs(v)); }
    rms = std::sqrt(rms / (double) a.size());
    CHECK(rms > 0.01 && rms < 0.3 && peak < 1.5, "level rms %.4f peak %.3f", rms, peak);
    CHECK(paradee_synthesize_ids(model.get(), ids.data(), ids.size(), 1.5f, nullptr, nullptr, fast, err) == PARADEE_OK &&
          fast.size() < a.size(), "speed 1.5 not shorter (%zu vs %zu)", fast.size(), a.size());

    // Convenience form gives the same audio.
    {
        paradee_synth_options opts;
        paradee_audio audio;
        CHECK(paradee_synthesize(model.get(), ids, opts, audio, err) == PARADEE_OK && audio.samples == a &&
              audio.sample_rate == 24000, "paradee_synthesize differs");
    }

    // Cancellation: before the call, after the text side, inside the generator.
    {
        int total_polls = 0;
        counter count; count.stop_after = 1 << 30;
        CHECK(paradee_synthesize_ids(model.get(), ids.data(), ids.size(), 1.0f, cancel_after, &count, pcm, err) == PARADEE_OK,
              "%s", err.c_str());
        total_polls = count.polls.load();
        CHECK(total_polls >= 8, "only %d cancel polls in one synthesis", total_polls);
        for (int stop : { 0, 2, total_polls / 2, total_polls - 2 }) {
            counter c; c.stop_after = stop;
            pcm.assign(10, 1.0f);
            const paradee_status st = paradee_synthesize_ids(model.get(), ids.data(), ids.size(), 1.0f, cancel_after, &c, pcm, err);
            CHECK(st == PARADEE_CANCELLED && pcm.empty(), "cancel after %d polls: status %s, %zu samples",
                  stop, paradee_status_str(st), pcm.size());
        }
    }

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test-paradee-api: OK (%zu samples, rms %.4f)\n", a.size(), rms);
    return 0;
}
