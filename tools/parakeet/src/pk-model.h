// tools/parakeet/src/pk-model.h - internal model structures shared by
// parakeet.cpp (load, encoder, decoder) and pk-stream.cpp.
#pragma once

#include "parakeet.h"
#include "pk-kernels.h"
#include "pk-ops.h"
#include "pk-pool.h"

#include <memory>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;

namespace pk {

struct hparams {
    int d_model = 1024, n_layers = 24, n_heads = 8, ff_dim = 4096, conv_kernel = 9;
    int sub_channels = 256, sub_factor = 8, n_mels = 128;
    int sample_rate = 16000, n_fft = 512, win_length = 400, hop_length = 160;
    int pred_hidden = 640, pred_layers = 2, joint_hidden = 640, vocab = 8192, blank_id = 8192, max_symbols = 10;
    float preemph = 0.97f, log_guard = 5.9604644775390625e-8f, norm_eps = 1e-5f, ln_eps = 1e-5f;
    std::vector<int> durations{ 0, 1, 2, 3, 4 };
};

struct layer {
    wmat ff1_w1, ff1_w2, qkv, out, pos, pw1, pw2, ff2_w1, ff2_w2;
    const float * bias_u = nullptr;
    const float * bias_v = nullptr;
    const float * ln_w[5] = {};
    const float * ln_b[5] = {};  // ff1, att, conv, ff2, out
    const float * dw_w = nullptr; // [d][k] (BatchNorm folded)
    std::vector<float> dw_wT;     // [k][d]
    const float * dw_b = nullptr;
};

struct mapped_file {
    void * addr = nullptr;
    size_t size = 0;
    std::vector<uint8_t> heap;  // fallback when mmap is unavailable
    ~mapped_file();
    bool open(const std::string & path, bool use_mmap, std::string & err);
    const uint8_t * data() const { return addr ? (const uint8_t *) addr : heap.data(); }
};

// scratch buffers reused across calls on one model (calls are serialised)
struct workspace {
    std::vector<float> x, xn, h, qkv, att, g, u, v, pos, rpos, encp;
    qact qa, qh, qp;
    std::vector<float> sub_a, sub_b, sub_c;
    std::vector<std::vector<float>> att_tls;  // per-thread attention scratch
};

}  // namespace pk

struct pk_model {
    pk::hparams hp;
    pk::mapped_file file;
    gguf_context * gg = nullptr;
    ggml_context * meta = nullptr;
    std::unique_ptr<pk::pool> pool;
    const pk::kernel_table * kt = nullptr;
    std::string notice;

    std::vector<std::string> pieces;  // with U+2581 already turned into ' '
    std::vector<uint8_t> is_control;  // <...> control pieces are never printed
    std::vector<uint8_t> word_start;  // piece starts a word

    const float * mel_fb = nullptr;   // [n_mels][n_fft/2+1]
    std::vector<int> mel_lo, mel_hi;  // non-zero bin range per mel
    std::vector<float> window;        // [n_fft], hann(win) centred

    std::vector<float> c0_wT, dw1_wT, dw2_wT;  // [9][C]
    const float *c0_b = nullptr, *dw1_b = nullptr, *dw2_b = nullptr;
    pk::wmat sub_pw1, sub_pw2, sub_out, enc_proj;
    std::vector<pk::layer> layers;

    const int8_t * embed_q = nullptr;
    const float * embed_s = nullptr;
    const int8_t * lstm_ih[2] = {};
    const int8_t * lstm_hh[2] = {};
    const float * lstm_ih_s[2] = {};
    const float * lstm_hh_s[2] = {};
    const float * lstm_b[2] = {};
    const int8_t * proj_q = nullptr;
    const float * proj_s = nullptr;
    const float * proj_b = nullptr;
    const int8_t * head_q = nullptr;
    const float * head_s = nullptr;
    const float * head_b = nullptr;
    int head_rows = 0;

    pk::workspace ws;
    double fv5_grid_err = 0.0;
    bool fv5t7 = false;  // weights in `fv5-b5x3t7-v1` (else `fv5-pair16-v1`)

    // relative-position keys W_pos . e(p) per layer for p in [-rc_P, rc_P],
    // row c <-> p = rc_P - c, fp16. They depend only on p, so they are
    // computed once and grown on demand (24 x (2P+1) x 1024 x 2 bytes:
    // 12 MB for 10 s windows, 37 MB for 30 s).
    int rc_P = -1;
    std::vector<std::vector<uint16_t>> rcache;
};

namespace pk {

// log-mel features [n_frames][n_mels] (n_frames = n_samples / hop); returns
// false when the input is too short to normalise (< 2 frames).
bool compute_mel(pk_model * m, const float * pcm, size_t n, std::vector<float> & mel, int & n_frames);
// encoder output projected to the joint width: [T][joint_hidden]. Optionally
// keeps the raw encoder output [T][d_model] in `enc_out`.
int encode(pk_model * m, const float * mel, int n_mel, std::vector<float> & encp, pk_cancel_cb cancel, void * cu,
           std::vector<float> * enc_out = nullptr, std::vector<float> * sub_out = nullptr,
           std::vector<float> * l0_out = nullptr);
// greedy TDT decode over encp [T][joint_hidden]
int tdt_decode(pk_model * m, const float * encp, int T, std::vector<pk_token> & out, pk_cancel_cb cancel, void * cu);
std::string detokenize(const pk_model * m, const pk_token * toks, size_t n);
int subsampled_length(int n_mel);

}  // namespace pk
