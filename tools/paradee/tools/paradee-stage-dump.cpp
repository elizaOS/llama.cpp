// SPDX-License-Identifier: MIT
//
// paradee-stage-dump — run one synthesis and write every stage as raw
// little-endian arrays, in the layouts tools/dump_paradee_reference.py writes
// for the PyTorch reference:
//
//   paradee-stage-dump <model.gguf> <phonemes.txt> <out-prefix> [speed]
//   paradee-stage-dump --decode <model.gguf> <asr.f32> <F0.f32> <N.f32> <out-prefix>
//
// --decode runs only the decoder, generator and phase lock on given text-side
// outputs (asr [512, T] channel-major, F0 / N [2T]).
//
// writes <prefix>.{bert,d,dur,F0,N,asr,dec,src,wav}.f32, <prefix>.dur_int.i32
// and <prefix>.ids.i32.

#include "paradee.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

template <typename T>
static bool write_raw(const std::string & path, const std::vector<T> & v) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char *) v.data(), (std::streamsize) (v.size() * sizeof(T)));
    return (bool) f;
}

static std::vector<float> read_f32(const char * path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    std::vector<float> v((size_t) f.tellg() / sizeof(float));
    f.seekg(0);
    f.read((char *) v.data(), (std::streamsize) (v.size() * sizeof(float)));
    return v;
}

int main(int argc, char ** argv) {
    if (argc == 7 && std::string(argv[1]) == "--decode") {
        std::string err;
        auto model = eliza_paradee::paradee_load_model(argv[2], err);
        if (!model) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
        const std::vector<float> asr = read_f32(argv[3]), F0 = read_f32(argv[4]), N = read_f32(argv[5]);
        const int T = (int) F0.size() / 2;
        eliza_paradee::paradee_stages st;
        if (eliza_paradee::paradee_decode_staged(model.get(), asr.data(), (int) (asr.size() / (size_t) T), T,
                                                 F0.data(), N.data(), nullptr, nullptr, nullptr, &st, err) != eliza_paradee::PARADEE_OK) {
            std::fprintf(stderr, "decode: %s\n", err.c_str());
            return 1;
        }
        const std::string pre = argv[6];
        return write_raw(pre + ".dec.f32", st.dec) && write_raw(pre + ".src.f32", st.src) &&
               write_raw(pre + ".wav.f32", st.wav) ? 0 : 1;
    }
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <model.gguf> <phonemes.txt> <out-prefix> [speed]\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[2], std::ios::binary);
    std::stringstream ss; ss << in.rdbuf();
    std::string ph = ss.str();
    while (!ph.empty() && (ph.back() == '\n' || ph.back() == '\r')) ph.pop_back();
    const float speed = argc > 4 ? std::strtof(argv[4], nullptr) : 1.0f;
    const std::string pre = argv[3];

    std::string err;
    auto model = eliza_paradee::paradee_load_model(argv[1], err);
    if (!model) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    std::vector<int32_t> ids;
    if (eliza_paradee::paradee_phonemes_to_ids(model.get(), ph, ids, err) != eliza_paradee::PARADEE_OK) {
        std::fprintf(stderr, "phonemes: %s\n", err.c_str());
        return 1;
    }
    eliza_paradee::paradee_stages st;
    std::vector<float> pcm;
    if (eliza_paradee::paradee_synthesize_ids_staged(model.get(), ids.data(), ids.size(), speed,
                                                     nullptr, nullptr, pcm, err, &st) != eliza_paradee::PARADEE_OK) {
        std::fprintf(stderr, "synthesis: %s\n", err.c_str());
        return 1;
    }
    bool ok = write_raw(pre + ".ids.i32", ids) && write_raw(pre + ".bert.f32", st.bert) &&
              write_raw(pre + ".d.f32", st.d) && write_raw(pre + ".dur.f32", st.dur) &&
              write_raw(pre + ".dur_int.i32", st.dur_int) && write_raw(pre + ".F0.f32", st.F0) &&
              write_raw(pre + ".N.f32", st.N) && write_raw(pre + ".asr.f32", st.asr) &&
              write_raw(pre + ".dec.f32", st.dec) && write_raw(pre + ".src.f32", st.src) &&
              write_raw(pre + ".wav.f32", st.wav);
    std::printf("T_phon=%d T_frame=%d samples=%zu\n", st.T_phon, st.T_frame, st.wav.size());
    return ok ? 0 : 1;
}
