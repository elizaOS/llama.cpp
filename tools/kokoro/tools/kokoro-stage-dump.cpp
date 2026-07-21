// SPDX-License-Identifier: MIT
//
// kokoro-stage-dump — validation harness for the Kokoro C++ forward port.
// Loads a GGUF model, reads reference input_ids (text) + ref_s (256 f32 bin),
// runs kokoro_predictor_forward, and dumps pred_dur / F0_pred / N_pred / asr
// as raw little-endian f32/i32 for comparison against the PyTorch reference.
//
// Usage: kokoro-stage-dump <model.gguf> <input_ids.txt> <ref_s.f32> <out-prefix>

#include "kokoro.h"
#include "kokoro-predictor.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int32_t> read_ids(const std::string & p) {
    std::ifstream f(p);
    std::vector<int32_t> v;
    int x;
    while (f >> x) v.push_back(x);
    return v;
}
static std::vector<float> read_f32(const std::string & p) {
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t n = (size_t) f.tellg() / sizeof(float);
    f.seekg(0);
    std::vector<float> v(n);
    f.read((char *) v.data(), (std::streamsize) (n * sizeof(float)));
    return v;
}
template <typename T>
static void write_bin(const std::string & p, const std::vector<T> & v) {
    std::ofstream f(p, std::ios::binary);
    f.write((const char *) v.data(), (std::streamsize) (v.size() * sizeof(T)));
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s <model.gguf> <ids.txt> <ref_s.f32> <out-prefix>\n", argv[0]);
        return 2;
    }
    std::string model_path = argv[1], ids_path = argv[2], refs_path = argv[3], prefix = argv[4];

    std::string err;
    auto model = eliza_kokoro::kokoro_load_model(model_path, err);
    if (!model) { std::fprintf(stderr, "load failed: %s\n", err.c_str()); return 1; }

    std::vector<int32_t> ids = read_ids(ids_path);
    std::vector<float> ref_s = read_f32(refs_path);
    if (ref_s.size() < 256) { std::fprintf(stderr, "ref_s too small: %zu\n", ref_s.size()); return 1; }

    eliza_kokoro::PredictorOut out;
    if (!eliza_kokoro::kokoro_predictor_forward(model.get(), ids, ref_s.data(), 1.0f, out, err)) {
        std::fprintf(stderr, "predictor_forward failed: %s\n", err.c_str());
        return 1;
    }

    std::printf("T_phon=%d T_frame=%d pred_dur_sum=%d F0_len=%zu N_len=%zu asr_len=%zu\n",
                out.T_phon, out.T_frame,
                [&] { int s = 0; for (auto d : out.pred_dur) s += d; return s; }(),
                out.F0_pred.size(), out.N_pred.size(), out.asr.size());

    write_bin(prefix + "_pred_dur.i32", out.pred_dur);
    write_bin(prefix + "_F0.f32", out.F0_pred);
    write_bin(prefix + "_N.f32", out.N_pred);
    write_bin(prefix + "_asr.f32", out.asr);      // [T_frame, 512] row-major (T-major)
    std::printf("wrote %s_{pred_dur.i32,F0.f32,N.f32,asr.f32}\n", prefix.c_str());
    return 0;
}
