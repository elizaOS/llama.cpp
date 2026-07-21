// SPDX-License-Identifier: MIT
// kokoro-decoder-front.h — Decoder.forward up to the generator (validated port, #9588).
#pragma once
#include <cmath>
#include <cstring>
#include <vector>
#include "kokoro-layers.h"   // conv1d_forward, adain1d_forward, convtranspose1d_depthwise_forward, convtranspose1d_out_len

namespace eliza_kokoro {

struct DecAdainResBlk {
    int Cin = 0, Cout = 0, Sdim = 128;
    bool upsample = false;
    bool learned_sc = false;             // dim_in != dim_out
    const float * norm1_fc_w = nullptr;  // [2*Cin, Sdim]
    const float * norm1_fc_b = nullptr;  // [2*Cin]
    const float * norm2_fc_w = nullptr;  // [2*Cout, Sdim]
    const float * norm2_fc_b = nullptr;  // [2*Cout]
    const float * conv1_w    = nullptr;  // [Cout, Cin, 3]
    const float * conv1_b    = nullptr;  // [Cout]
    const float * conv2_w    = nullptr;  // [Cout, Cout, 3]
    const float * conv2_b    = nullptr;  // [Cout]
    const float * conv1x1_w  = nullptr;  // [Cout, Cin, 1] (learned_sc only)
    const float * conv1x1_b  = nullptr;  // [Cout] (null — conv1x1 bias=False)
    const float * pool_w     = nullptr;  // [Cin, 1, 3] (upsample only)
    const float * pool_b     = nullptr;  // [Cin] (upsample only)
};

// AdainResBlk1d (decode-block flavor: leaky_relu 0.2; pool/shortcut;
// out = (residual + shortcut)/sqrt(2)). Output y [Cout, T_out].
inline void dec_adainresblk1d_forward(
        const DecAdainResBlk & w, const float * x, int T_in, const float * s,
        std::vector<float> & y, int & T_out) {
    const int Cin = w.Cin, Cout = w.Cout, Sdim = w.Sdim;

    // residual branch: norm1 -> leaky_relu(0.2) -> [pool] -> conv1 -> norm2 -> leaky_relu -> conv2
    std::vector<float> r(x, x + (size_t)Cin * T_in);
    adain1d_forward(r.data(), Cin, T_in, s, Sdim, w.norm1_fc_w, w.norm1_fc_b);
    for (size_t i = 0; i < r.size(); ++i) if (r[i] < 0) r[i] *= 0.2f;

    int T_pool = T_in;
    if (w.upsample) {
        T_pool = convtranspose1d_out_len(T_in, 3, 2, 1, 1);
        std::vector<float> r2((size_t)Cin * T_pool);
        convtranspose1d_depthwise_forward(r.data(), Cin, T_in, w.pool_w, w.pool_b, 3, 2, 1, 1, r2.data(), T_pool);
        r.swap(r2);
    }
    std::vector<float> r3((size_t)Cout * T_pool);
    conv1d_forward(r.data(), Cin, T_pool, w.conv1_w, w.conv1_b, Cout, 3, 1, 1, 1, r3.data(), T_pool);
    adain1d_forward(r3.data(), Cout, T_pool, s, Sdim, w.norm2_fc_w, w.norm2_fc_b);
    for (size_t i = 0; i < r3.size(); ++i) if (r3[i] < 0) r3[i] *= 0.2f;
    std::vector<float> r4((size_t)Cout * T_pool);
    conv1d_forward(r3.data(), Cout, T_pool, w.conv2_w, w.conv2_b, Cout, 3, 1, 1, 1, r4.data(), T_pool);

    // shortcut branch: [nearest-upsample x2] -> [conv1x1 if learned_sc]
    T_out = T_pool;
    std::vector<float> sc;
    if (w.upsample) {
        const int T_up = T_in * 2;  // == T_pool
        std::vector<float> up((size_t)Cin * T_up);
        for (int c = 0; c < Cin; ++c)
            for (int t = 0; t < T_up; ++t)
                up[(size_t)c * T_up + t] = x[(size_t)c * T_in + (t / 2)];
        if (w.learned_sc) {
            sc.assign((size_t)Cout * T_up, 0.0f);
            conv1d_forward(up.data(), Cin, T_up, w.conv1x1_w, w.conv1x1_b, Cout, 1, 1, 0, 1, sc.data(), T_up);
        } else sc.swap(up);
    } else {
        if (w.learned_sc) {
            sc.assign((size_t)Cout * T_in, 0.0f);
            conv1d_forward(x, Cin, T_in, w.conv1x1_w, w.conv1x1_b, Cout, 1, 1, 0, 1, sc.data(), T_in);
        } else sc.assign(x, x + (size_t)Cin * T_in);
    }

    y.assign((size_t)Cout * T_out, 0.0f);
    const float rsqrt2 = 1.0f / std::sqrt(2.0f);
    for (size_t i = 0; i < y.size(); ++i) y[i] = (r4[i] + sc[i]) * rsqrt2;
}

struct DecoderFrontWeights {
    const float * F0_conv_w = nullptr;  // [1,1,3]
    const float * F0_conv_b = nullptr;  // [1]
    const float * N_conv_w  = nullptr;  // [1,1,3]
    const float * N_conv_b  = nullptr;  // [1]
    const float * asr_res_w = nullptr;  // [64,512,1]
    const float * asr_res_b = nullptr;  // [64]
    DecAdainResBlk encode;              // 514 -> 1024, learned_sc
    DecAdainResBlk decode[4];           // 1090->1024 (x3), 1090->512 upsample
};

// Decoder.forward up to (not including) the generator.
//   asr[512,T_asr] (T_asr=132), F0_curve[2*T_asr], N[2*T_asr], s[128]
// Output: x_out [512, 2*T_asr] (== generator_in_0); also returns the
// stride-2 conv outputs F0_down[T_asr], N_down[T_asr] (caller passes them,
// together with the ORIGINAL F0_curve, into the generator).
inline void decoder_front(
        const DecoderFrontWeights & W,
        const float * asr, int Cin_asr, int T_asr,
        const float * F0_curve, const float * N_in, const float * s,
        std::vector<float> & x_out,
        std::vector<float> & F0_down,
        std::vector<float> & N_down) {
    const int Tc = 2 * T_asr;  // 264

    F0_down.assign(T_asr, 0.0f);
    conv1d_forward(F0_curve, 1, Tc, W.F0_conv_w, W.F0_conv_b, 1, 3, 2, 1, 1, F0_down.data(), T_asr);
    N_down.assign(T_asr, 0.0f);
    conv1d_forward(N_in, 1, Tc, W.N_conv_w, W.N_conv_b, 1, 3, 2, 1, 1, N_down.data(), T_asr);

    // x = cat([asr, F0, N], dim=channels) -> [514, T_asr]
    std::vector<float> xcat((size_t)(Cin_asr + 2) * T_asr);
    std::memcpy(xcat.data(), asr, sizeof(float) * (size_t)Cin_asr * T_asr);
    std::memcpy(xcat.data() + (size_t)Cin_asr * T_asr, F0_down.data(), sizeof(float) * T_asr);
    std::memcpy(xcat.data() + (size_t)(Cin_asr + 1) * T_asr, N_down.data(), sizeof(float) * T_asr);

    std::vector<float> x; int T_x;
    dec_adainresblk1d_forward(W.encode, xcat.data(), T_asr, s, x, T_x);   // encode 514->1024

    std::vector<float> asr_res((size_t)64 * T_asr);                       // asr_res Conv1d k1 512->64
    conv1d_forward(asr, Cin_asr, T_asr, W.asr_res_w, W.asr_res_b, 64, 1, 1, 0, 1, asr_res.data(), T_asr);

    bool res = true;
    for (int b = 0; b < 4; ++b) {
        std::vector<float> blk_in;
        if (res) {  // cat([x, asr_res, F0, N]) -> 1024+64+1+1 = 1090
            const int Cx = (int)(x.size() / T_x);
            const int Cin_blk = Cx + 64 + 1 + 1;
            blk_in.assign((size_t)Cin_blk * T_x, 0.0f);
            std::memcpy(blk_in.data(), x.data(), sizeof(float) * (size_t)Cx * T_x);
            std::memcpy(blk_in.data() + (size_t)Cx * T_x, asr_res.data(), sizeof(float) * 64 * T_x);
            std::memcpy(blk_in.data() + (size_t)(Cx + 64) * T_x, F0_down.data(), sizeof(float) * T_x);
            std::memcpy(blk_in.data() + (size_t)(Cx + 65) * T_x, N_down.data(), sizeof(float) * T_x);
        } else {
            blk_in.assign(x.begin(), x.end());
        }
        std::vector<float> y; int T_y;
        dec_adainresblk1d_forward(W.decode[b], blk_in.data(), T_x, s, y, T_y);
        x.swap(y); T_x = T_y;
        if (W.decode[b].upsample) res = false;  // decode3 upsamples -> res stops
    }
    x_out.swap(x);  // [512, 264]
}

} // namespace eliza_kokoro