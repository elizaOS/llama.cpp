// SPDX-License-Identifier: MIT
//
// test-paradee-api — the eliza_paradee C++ API contract on the real model:
// loading, phoneme mapping (pads, unknown symbols, length limit, UTF-8),
// input validation, determinism, speed, and cancellation (before the call,
// between stages and inside the generator loops). Runs on a v1.0, v1.1 or
// v1.2 GGUF; for v1.1 / v1.2 also the noise seed and injected noise, and for
// v1.2 the voice table (names, accents, row selection).
//
// Needs PARADEE_GGUF (environment); exits 77 (CTest SKIP) without it.

#include "paradee.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
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
    const bool v11 = paradee_get_head(model.get()) == PARADEE_HEAD_SOURCE_FILTER;   // v1.1 or v1.2
    const std::string version = paradee_version(model.get());
    const bool v12 = version == "1.2";
    CHECK(version == (v12 ? "1.2" : v11 ? "1.1" : "1.0"), "version %s", version.c_str());

    // Voices: one for v1.0 / v1.1, the v1.2 table in model order.
    {
        const int nv = paradee_voice_count(model.get());
        CHECK(paradee_voice_count(nullptr) == 0 && paradee_default_voice(nullptr) == -1 &&
              std::string(paradee_voice_name(nullptr, 0)).empty() && paradee_find_voice(nullptr, "af_heart") == -1,
              "null model voices");
        const bool delibes = std::string(paradee_family(model.get())) == "delibes";
        const char * def_name = delibes ? "em_alex" : "af_heart";
        CHECK(paradee_default_voice(model.get()) == 0 && std::string(paradee_voice_name(model.get(), 0)) == def_name,
              "default voice %d %s", paradee_default_voice(model.get()), paradee_voice_name(model.get(), 0));
        CHECK(std::string(paradee_voice_name(model.get(), nv)).empty() &&
              std::string(paradee_voice_name(model.get(), -1)).empty(), "voice name out of range");
        CHECK(paradee_find_voice(model.get(), def_name) == 0 && paradee_find_voice(model.get(), "nobody") == -1,
              "find voice");
        // Release metadata and the validated envelope.
        float lo = 0, hi = 0;
        CHECK(paradee_validated_speed(model.get(), lo, hi), "validated speed");
        if (delibes) {
            CHECK(std::string(paradee_language(model.get())) == "es" &&
                  std::string(paradee_phoneme_format(model.get())) == "espeak-es" &&
                  paradee_validated_max_ids(model.get()) == 360 && lo == 0.85f && hi == 1.15f,
                  "Delibes metadata: %s %s %d [%g, %g]", paradee_language(model.get()),
                  paradee_phoneme_format(model.get()), paradee_validated_max_ids(model.get()), lo, hi);
        } else {
            CHECK(std::string(paradee_family(model.get())) == "paradee" && std::string(paradee_language(model.get())) == "en" &&
                  std::string(paradee_phoneme_format(model.get())) == "misaki" &&
                  paradee_validated_max_ids(model.get()) == PARADEE_MAX_INPUT_IDS &&
                  lo == PARADEE_MIN_SPEED && hi == PARADEE_MAX_SPEED, "Paradee metadata defaults");
        }
        CHECK(std::string(paradee_family(nullptr)).empty() && paradee_validated_max_ids(nullptr) == 0 &&
              !paradee_validated_speed(nullptr, lo, hi), "null model metadata");
        if (v12) {
            const char * names[] = { "af_heart", "bf_isabella", "am_puck", "bm_fable" };
            const char * accents[] = { "en-us", "en-gb", "en-us", "en-gb" };
            CHECK(nv == 4, "v1.2 has %d voices", nv);
            for (int i = 0; i < 4 && nv == 4; ++i) {
                CHECK(std::string(paradee_voice_name(model.get(), i)) == names[i] &&
                      std::string(paradee_voice_accent(model.get(), i)) == accents[i] &&
                      paradee_find_voice(model.get(), names[i]) == i, "voice %d: %s %s", i,
                      paradee_voice_name(model.get(), i), paradee_voice_accent(model.get(), i));
            }
        } else {
            CHECK(nv == 1 && std::string(paradee_voice_accent(model.get(), 0)).empty(), "v1.0 / v1.1 / Delibes have one voice");
        }
    }
    CHECK(paradee_get_head(nullptr) == PARADEE_HEAD_ISTFTNET_LOCK && std::string(paradee_version(nullptr)).empty(),
          "null model head / version");
    const paradee_lock_params * lp = paradee_get_lock_params(model.get());
    if (v11) {
        CHECK(!lp, "a v1.1 model has no phase lock");
    } else {
        CHECK(lp && lp->n_fft == 1024 && lp->hop == 256 && lp->smooth_frames == 33 && lp->f0_hop == 300 &&
              lp->band_lo_hz == 2000.0f && lp->band_hi_hz == 8000.0f && lp->voiced_f0_hz == 60.0f,
              "lock parameters not read from the GGUF");
    }

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

        // Lenient form (the libelizainference contract): unknown symbols are
        // dropped and counted, empty input is OK, bad UTF-8 is still rejected.
        size_t dropped = 99;
        CHECK(paradee_phonemes_to_symbol_ids_lenient(model.get(), "h#əlˈO g", sym, dropped, err) == PARADEE_OK &&
              dropped == 2 && sym.size() == 6, "lenient drop: %zu ids, %zu dropped", sym.size(), dropped);
        CHECK(paradee_phonemes_to_symbol_ids_lenient(model.get(), "", sym, dropped, err) == PARADEE_OK &&
              sym.empty() && dropped == 0, "lenient empty");
        CHECK(paradee_phonemes_to_symbol_ids_lenient(model.get(), "##", sym, dropped, err) == PARADEE_OK &&
              sym.empty() && dropped == 2, "lenient all dropped");
        CHECK(paradee_phonemes_to_symbol_ids_lenient(model.get(), std::string("a\xC3", 2), sym, dropped, err) ==
              PARADEE_E_INVALID_ARG && sym.empty(), "lenient truncated UTF-8 accepted");
        CHECK(paradee_phonemes_to_symbol_ids_lenient(nullptr, "a", sym, dropped, err) == PARADEE_E_INVALID_ARG,
              "lenient null model");
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
        CHECK(paradee_synthesize_ids(model.get(), ids.data(), ids.size(), PARADEE_MIN_SPEED * 0.5f, nullptr, nullptr,
                                     pcm, err) == PARADEE_E_INVALID_ARG, "speed below the minimum accepted");
        CHECK(paradee_synthesize_ids(model.get(), ids.data(), ids.size(), PARADEE_MAX_SPEED * 2.0f, nullptr, nullptr,
                                     pcm, err) == PARADEE_E_INVALID_ARG, "speed above the maximum accepted");
        CHECK(paradee_synthesize_ids(model.get(), ids.data(), ids.size(), INFINITY, nullptr, nullptr, pcm, err) ==
              PARADEE_E_INVALID_ARG, "speed inf accepted");
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

    // Voice selection: -1 and the default row give the same audio; every
    // other row is a different voice; an index out of range is rejected.
    {
        const int nv = paradee_voice_count(model.get());
        paradee_synth_options opts;
        paradee_audio out;
        opts.voice = paradee_default_voice(model.get());
        CHECK(paradee_synthesize(model.get(), ids, opts, out, err) == PARADEE_OK && out.samples == a,
              "the default voice by index differs from voice -1");
        for (int v = 1; v < nv; ++v) {
            opts.voice = v;
            CHECK(paradee_synthesize(model.get(), ids, opts, out, err) == PARADEE_OK && !out.samples.empty() &&
                  out.samples != a, "voice %d gives af_heart's audio", v);
            paradee_audio again;
            CHECK(paradee_synthesize(model.get(), ids, opts, again, err) == PARADEE_OK && again.samples == out.samples,
                  "voice %d is not deterministic", v);
        }
        for (int bad : { nv, -2 }) {
            opts.voice = bad;
            out.samples.assign(4, 1.0f);
            CHECK(paradee_synthesize(model.get(), ids, opts, out, err) == PARADEE_E_INVALID_ARG && out.samples.empty(),
                  "voice %d accepted", bad);
        }
        paradee_stages st;
        st.voice = nv;
        CHECK(paradee_synthesize_ids_staged(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, pcm, err, &st) ==
              PARADEE_E_INVALID_ARG, "staged voice out of range accepted");
        if (nv > 1) {
            paradee_stages s1;
            s1.voice = 1;
            opts.voice = 1;
            CHECK(paradee_synthesize_ids_staged(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, pcm, err, &s1) ==
                  PARADEE_OK && paradee_synthesize(model.get(), ids, opts, out, err) == PARADEE_OK && pcm == out.samples,
                  "staged voice 1 differs from voice 1");
        }
    }

    // v1.1: the noise comes from the seed (the default one unless the options
    // choose), or from the caller.
    if (v11) {
        paradee_synth_options opts;
        paradee_audio s0, s1, s1b;
        opts.noise_seed = PARADEE_DEFAULT_NOISE_SEED;
        CHECK(paradee_synthesize(model.get(), ids, opts, s0, err) == PARADEE_OK && s0.samples == a, "default seed");
        opts.noise_seed = 1;
        CHECK(paradee_synthesize(model.get(), ids, opts, s1, err) == PARADEE_OK && s1.samples.size() == a.size() &&
              s1.samples != a, "another seed must change the noise");
        CHECK(paradee_synthesize(model.get(), ids, opts, s1b, err) == PARADEE_OK && s1b.samples == s1.samples,
              "the same seed must give the same audio");
        paradee_stages st;
        const std::vector<float> wrong(a.size() + 1, 0.0f);
        st.noise_in = &wrong;
        CHECK(paradee_synthesize_ids_staged(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, pcm, err, &st) ==
              PARADEE_E_INVALID_ARG && pcm.empty(), "noise of the wrong length accepted");
        paradee_stages s2;
        CHECK(paradee_synthesize_ids_staged(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, pcm, err, &s2) ==
              PARADEE_OK && pcm == a && s2.noise.size() == a.size(), "staged synthesis differs");
        paradee_stages s3;
        s3.noise_in = &s2.noise;
        CHECK(paradee_synthesize_ids_staged(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, pcm, err, &s3) ==
              PARADEE_OK && pcm == a, "the seeded noise fed back in must give the same audio");
        const std::vector<float> silent(a.size(), 0.0f);
        paradee_stages s4;
        s4.noise_in = &silent;
        CHECK(paradee_synthesize_ids_staged(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, pcm, err, &s4) ==
              PARADEE_OK && pcm != a, "zero noise must change the audio");
        CHECK(paradee_head_staged(model.get(), s2.trunk.data(), (int) (s2.trunk.size() / (2 * s2.T_frame)) + 1,
                                  2 * s2.T_frame, s2.F0.data(), s2.noise.data(), &s4, err) == PARADEE_E_INVALID_ARG,
              "head with the wrong trunk width accepted");
    } else {
        paradee_stages st;
        const std::vector<float> zeros(128 * 4 * 300, 0.0f);
        CHECK(paradee_head_staged(model.get(), zeros.data(), 128, 4, zeros.data(), zeros.data(), &st, err) ==
              PARADEE_E_INVALID_ARG, "a v1.0 model has no source-filter head");
    }

    // Concurrent calls on one model are serialised and give the same audio.
    {
        std::vector<float> t1, t2;
        paradee_status s1 = PARADEE_E_RUNTIME, s2 = PARADEE_E_RUNTIME;
        std::string e1, e2;
        std::thread th1([&] { s1 = paradee_synthesize_ids(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, t1, e1); });
        std::thread th2([&] { s2 = paradee_synthesize_ids(model.get(), ids.data(), ids.size(), 1.0f, nullptr, nullptr, t2, e2); });
        th1.join();
        th2.join();
        CHECK(s1 == PARADEE_OK && s2 == PARADEE_OK && t1 == a && t2 == a, "concurrent synthesis differs");
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
