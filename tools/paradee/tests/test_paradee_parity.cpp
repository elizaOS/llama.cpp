// SPDX-License-Identifier: MIT
//
// test-paradee-parity — native Paradee against the pinned PyTorch reference
// (tools/dump_paradee_reference.py dump) on the misaki phonemes of the model
// card's samples/sentences.txt (fixtures/sentences.misaki.json).
//
// Gates, per sentence:
//   ids         paradee_phonemes_to_ids == the fixture's ids
//   durations   integer durations identical
//   text side   bert, d, dur, F0, N, asr: max|err| <= 1e-3 * RMS(reference)
//   source      harmonic source from the reference F0: max|err| <= 1e-3 * RMS
//   decoder     generator output from the reference asr / F0 / N / source:
//               max|err| <= 1e-3 * RMS
//   lock        phase lock alone on the reference decoder output and source:
//               max|err| <= 1e-4 against phase_lock.lock() in float64, and no
//               further from the shipped float32 phase_lock.lock() than that
//               float32 evaluation is from float64 (+1e-4)
//   end to end  final waveform SNR >= 35 dB against the reference
//
// Each acoustic stage is fed the reference's inputs, so an error is charged
// to the stage that made it: the harmonic source integrates F0 over seconds,
// and float32 rounding differences in F0 (~1e-6 relative) would otherwise
// show up as source phase differences. The end-to-end check covers the chain.
//
// Needs PARADEE_GGUF and PARADEE_REF_DIR (environment); exits 77 (CTest
// SKIP) without them. PARADEE_PARITY_OUT=<dir> also writes the native
// waveforms (<i>.wav.f32) for `dump_paradee_reference.py report`.

#include "paradee.h"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef PARADEE_FIXTURE_DIR
#error "PARADEE_FIXTURE_DIR must be defined"
#endif

using json = nlohmann::json;
using namespace eliza_paradee;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++failures; } } while (0)

template <typename T>
static std::vector<T> read_raw(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
    std::vector<T> v((size_t) f.tellg() / sizeof(T));
    f.seekg(0);
    f.read((char *) v.data(), (std::streamsize) (v.size() * sizeof(T)));
    return v;
}

static json read_json(const std::string & path) {
    std::ifstream f(path);
    if (!f) { std::fprintf(stderr, "cannot read %s\n", path.c_str()); std::exit(1); }
    std::stringstream ss; ss << f.rdbuf();
    return json::parse(ss.str());
}

struct err_stats { double max_abs = 0, rms_ref = 0, snr_db = 0; bool same_size = true; };

static err_stats compare(const std::vector<float> & ref, const std::vector<float> & got) {
    err_stats s;
    if (ref.size() != got.size() || ref.empty()) { s.same_size = false; return s; }
    double e2 = 0, r2 = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = (double) got[i] - (double) ref[i];
        s.max_abs = std::max(s.max_abs, std::fabs(d));
        e2 += d * d;
        r2 += (double) ref[i] * ref[i];
    }
    s.rms_ref = std::sqrt(r2 / (double) ref.size());
    s.snr_db = e2 > 0 ? 10.0 * std::log10(r2 / e2) : 300.0;
    return s;
}

static void gate_stage(int sent, const char * name, const std::vector<float> & ref, const std::vector<float> & got) {
    const err_stats s = compare(ref, got);
    if (!s.same_size) {
        CHECK(false, "sentence %d %s: size %zu != reference %zu", sent, name, got.size(), ref.size());
        return;
    }
    const double ratio = s.max_abs / std::max(s.rms_ref, 1e-30);
    std::printf("  %-6s max|err| %9.3g  rms %9.4g  ratio %9.3g  snr %6.1f dB\n", name, s.max_abs, s.rms_ref, ratio, s.snr_db);
    CHECK(ratio <= 1e-3, "sentence %d %s: max|err| %.3g > 1e-3 * rms %.4g", sent, name, s.max_abs, s.rms_ref);
}

int main() {
    const char * gguf = std::getenv("PARADEE_GGUF");
    const char * ref_dir_c = std::getenv("PARADEE_REF_DIR");
    if (!gguf || !*gguf || !ref_dir_c || !*ref_dir_c) {
        std::printf("test-paradee-parity: PARADEE_GGUF / PARADEE_REF_DIR not set, skipping\n");
        return 77;
    }
    const std::string ref_dir = ref_dir_c;
    const char * out_dir = std::getenv("PARADEE_PARITY_OUT");

    const json manifest = read_json(ref_dir + "/manifest.json");
    const json fixture = read_json(std::string(PARADEE_FIXTURE_DIR) + "/sentences.misaki.json");
    const float speed = manifest.value("speed", 1.0f);

    std::string err;
    auto model = paradee_load_model(gguf, err);
    if (!model) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }
    const paradee_lock_params lock = *paradee_get_lock_params(model.get());

    CHECK(manifest["sentences"].size() == fixture["sentences"].size(), "reference and fixture sentence counts differ");
    double total_audio = 0, total_secs = 0;
    for (size_t k = 0; k < manifest["sentences"].size(); ++k) {
        const json & m = manifest["sentences"][k];
        const json & fx = fixture["sentences"][k];
        const int sent = m["index"].get<int>();
        const std::string phonemes = fx["phonemes"].get<std::string>();
        auto file = [&](const char * name) { return ref_dir + "/" + m["files"][name]["file"].get<std::string>(); };
        std::printf("sentence %d (%zu bytes of phonemes)\n", sent, phonemes.size());
        CHECK(m["phonemes"].get<std::string>() == phonemes, "sentence %d: reference was dumped from other phonemes", sent);

        // ids
        std::vector<int32_t> ids;
        CHECK(paradee_phonemes_to_ids(model.get(), phonemes, ids, err) == PARADEE_OK, "%s", err.c_str());
        CHECK(ids == fx["ids"].get<std::vector<int32_t>>(), "sentence %d: ids differ from the fixture", sent);
        CHECK(ids == read_raw<int32_t>(file("ids")), "sentence %d: ids differ from the reference", sent);

        // full synthesis
        paradee_stages st;
        std::vector<float> pcm;
        const auto t0 = std::chrono::steady_clock::now();
        const paradee_status rc = paradee_synthesize_ids_staged(model.get(), ids.data(), ids.size(), speed,
                                                                nullptr, nullptr, pcm, err, &st);
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (rc != PARADEE_OK) { CHECK(false, "sentence %d: synthesis failed: %s", sent, err.c_str()); continue; }
        total_secs += secs;
        total_audio += (double) pcm.size() / 24000.0;

        const std::vector<int32_t> ref_dur = read_raw<int32_t>(file("dur_int"));
        CHECK(st.dur_int == ref_dur, "sentence %d: integer durations differ", sent);
        std::printf("  durations identical: %s (%d frames)\n", st.dur_int == ref_dur ? "yes" : "NO", st.T_frame);
        gate_stage(sent, "bert", read_raw<float>(file("bert")), st.bert);
        gate_stage(sent, "d", read_raw<float>(file("d")), st.d);
        gate_stage(sent, "dur", read_raw<float>(file("dur")), st.dur);
        gate_stage(sent, "F0", read_raw<float>(file("F0")), st.F0);
        gate_stage(sent, "N", read_raw<float>(file("N")), st.N);
        gate_stage(sent, "asr", read_raw<float>(file("asr")), st.asr);

        // acoustic half, teacher-forced stage by stage
        const std::vector<float> ref_asr = read_raw<float>(file("asr"));
        const std::vector<float> ref_F0 = read_raw<float>(file("F0"));
        const std::vector<float> ref_N = read_raw<float>(file("N"));
        const std::vector<float> ref_src = read_raw<float>(file("src"));
        const std::vector<float> ref_dec = read_raw<float>(file("dec"));
        const std::vector<float> ref_wav = read_raw<float>(file("wav"));
        const std::vector<float> ref_l32 = read_raw<float>(file("lock_f32"));
        const int T = (int) ref_F0.size() / 2;
        {
            paradee_stages a;
            CHECK(paradee_decode_staged(model.get(), ref_asr.data(), 512, T, ref_F0.data(), ref_N.data(), nullptr,
                                        nullptr, nullptr, &a, err) == PARADEE_OK, "%s", err.c_str());
            gate_stage(sent, "src", ref_src, a.src);
            paradee_stages b;
            CHECK(paradee_decode_staged(model.get(), ref_asr.data(), 512, T, ref_F0.data(), ref_N.data(), &ref_src,
                                        nullptr, nullptr, &b, err) == PARADEE_OK, "%s", err.c_str());
            gate_stage(sent, "dec", ref_dec, b.dec);
        }
        {
            std::vector<float> locked;
            CHECK(paradee_phase_lock(ref_dec.data(), ref_dec.size(), ref_src.data(), ref_src.size(),
                                     ref_F0.data(), ref_F0.size(), lock, locked), "lock rejected its input");
            const err_stats e64 = compare(ref_wav, locked);
            const err_stats e32 = compare(ref_l32, locked);
            const err_stats f32 = compare(ref_wav, ref_l32);
            std::printf("  lock   max|err| %9.3g vs float64 phase_lock (gate 1e-4); %9.3g vs float32 phase_lock "
                        "(float32 itself: %.3g)\n", e64.max_abs, e32.max_abs, f32.max_abs);
            CHECK(e64.same_size && e64.max_abs <= 1e-4, "sentence %d lock: %.3g > 1e-4", sent, e64.max_abs);
            CHECK(e32.same_size && e32.max_abs <= f32.max_abs + 1e-4, "sentence %d lock vs float32: %.3g > %.3g",
                  sent, e32.max_abs, f32.max_abs + 1e-4);
        }

        // end to end
        const err_stats e = compare(ref_wav, pcm);
        std::printf("  end to end: snr %.1f dB, max|err| %.3g, %zu samples, %.3f s synth (%.1fx realtime)\n",
                    e.snr_db, e.max_abs, pcm.size(), secs, (double) pcm.size() / 24000.0 / secs);
        CHECK(e.same_size && e.snr_db >= 35.0, "sentence %d: end-to-end SNR %.1f dB < 35 dB", sent, e.snr_db);

        if (out_dir && *out_dir) {
            std::ofstream f(std::string(out_dir) + "/" + std::to_string(sent) + ".wav.f32", std::ios::binary);
            f.write((const char *) pcm.data(), (std::streamsize) (pcm.size() * sizeof(float)));
        }
    }
    std::printf("total: %.2f s of audio in %.2f s (%.1fx realtime)\n", total_audio, total_secs, total_audio / total_secs);
    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test-paradee-parity: OK\n");
    return 0;
}
