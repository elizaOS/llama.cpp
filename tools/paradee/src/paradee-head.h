// SPDX-License-Identifier: MIT
//
// paradee-head.h — Paradee-8M v1.1's frame-rate source-filter head
// (training/scripts/vhead.py SourceFilterHead2, sahilmahendrakar/paradee
// 606e16e1, Apache-2.0). Internal to paradee_lib.
//
// The head replaces v1.0's iSTFTNet generator and phase lock. It never
// upsamples: from the AdaIN trunk's 80 fps features x [in_ch, n] and the F0
// curve [n] it predicts, for every frame of an STFT with n_fft 600 / hop 150
// (2 frames per trunk frame),
//
//   - a harmonic gain, a phase (cos, sin) and a noise gain at G = 80
//     mel-spaced points, linearly interpolated to the 301 bins,
//   - gains of 2*mod pulsing noise sources noise*uv*cos/sin(2 pi m phi),
//   - one pulse shift per frame (0.5 tanh, in pitch periods),
//
// applies them to the STFTs of a 64-harmonic sine bank built from F0 and of
// white noise, and inverts:
//
//   y = iSTFT(E * gh * u * exp(i 2 pi (f / f0) shift) + (Nz + sum_j a_j Nm_j) * gn)
//     + iSTFT_150/75(STFT_150/75(reverse(noise)) * gb)      (3 ms noise bursts)
//
// The ConvNeXt stack is ConvNeXt(dim) x blocks between LayerNorms: depthwise
// conv 7, LayerNorm, Linear(dim, 3 dim), exact GELU, Linear(3 dim, dim),
// per-channel gamma, residual.
//
// The noise is an input. paradee.cpp draws it with paradee_head_noise (a
// seeded Gaussian), the parity test feeds the reference's noise.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct gguf_context;

namespace eliza_paradee {
namespace head {

struct params {
    int in_ch = 128, dim = 160, blocks = 4, kernel = 7, mult = 3;
    int f0_hop = 300;               // samples per trunk frame / F0 value
    int n_fft = 600, hop = 150;     // main STFT
    int K = 64;                     // sine bank harmonics
    int mod = 2;                    // pulsing noise orders
    int G = 80;                     // gain points (mel-spaced)
    int burst_n_fft = 150, burst_hop = 75, Gb = 24;
    float sine_amp = 0.1f, max_harmonic_hz = 11000.0f, voiced_f0_hz = 10.0f, min_f0_hz = 60.0f;
    float gain_clamp = 6.0f, shift_scale = 0.5f;
    int sample_rate = 24000;
};

struct block_weights {
    const float * dw_w = nullptr;    // [dim, 1, kernel]
    const float * dw_b = nullptr;
    const float * norm_w = nullptr;  // [dim]
    const float * norm_b = nullptr;
    const float * pw1_w = nullptr;   // [mult*dim, dim]
    const float * pw1_b = nullptr;
    const float * pw2_w = nullptr;   // [dim, mult*dim]
    const float * pw2_b = nullptr;
    const float * gamma = nullptr;   // [dim]
};

struct weights {
    const float * inp_w = nullptr;   // [dim, in_ch, kernel]
    const float * inp_b = nullptr;
    const float * norm0_w = nullptr, * norm0_b = nullptr;
    std::vector<block_weights> blocks;
    const float * norm1_w = nullptr, * norm1_b = nullptr;
    const float * out_w = nullptr;   // [up * (C*G + 1), dim], C = 4 + 2*mod, up = f0_hop / hop
    const float * out_b = nullptr;
    const float * outb_w = nullptr;  // [(f0_hop / burst_hop) * Gb, dim]
    const float * outb_b = nullptr;
};

// Read the hparams ("paradee.head.*") and tensors ("paradee.head.*") and
// check every shape. Returns false with err set.
bool load(gguf_context * g, ggml_context * ctx, params & p, weights & w, std::string & err);

// Optional intermediate outputs, in the reference dump's layouts.
struct stages {
    std::vector<float> hn;      // [n, dim]   ConvNeXt stack after norm1 (time-major)
    std::vector<float> sine;    // [n*f0_hop] sine bank
    std::vector<float> main;    // [n*f0_hop] the n_fft 600 path
    std::vector<float> burst;   // [n*f0_hop] the noise-burst path
};

using cancel_fn = bool (*)(void *);

// x [in_ch, n] channel-major (the trunk output), f0 [n], noise [n*f0_hop].
// y gets n*f0_hop samples. Returns false with y empty when cancel() returned
// true (cancelled = true) or on bad input (err set).
bool forward(const params & p, const weights & w,
             const float * x, int n, const float * f0, const float * noise,
             std::vector<float> & y, stages * st,
             cancel_fn cancel, void * user, bool & cancelled, std::string & err);

// The ConvNeXt stack alone: hn [dim, n] (channel-major) from x [in_ch, n].
void conv_stack(const params & p, const weights & w, const float * x, int n, std::vector<float> & hn);

// vhead.sine_bank: e [n*f0_hop] from f0 [n].
void sine_bank(const params & p, const float * f0, int n, std::vector<float> & e);

// Seeded standard-normal noise (xoshiro256** seeded by splitmix64, then
// Box-Muller in double). The same seed gives the same samples on every
// platform up to the last bit of libm's log/sin/cos.
void gaussian_noise(uint64_t seed, size_t count, std::vector<float> & out);

} // namespace head
} // namespace eliza_paradee
