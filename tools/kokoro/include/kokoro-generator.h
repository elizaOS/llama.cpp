// SPDX-License-Identifier: MIT
//
// kokoro-generator.h — iSTFTNet Generator.forward (StyleTTS-2 decoder back-end).
//
// The generator turns the decoder body output `x` [Cin, T0], the style
// vector `s` [Sdim], and the (un-downsampled) F0 curve `f0_curve` [T0] into
// `audio` [T0 * prod(upsample_rates) * hop] (24 kHz).
//
// Kokoro-82M: Cin 512, stage channels 256/128, Sdim 128.
// Paradee-8M: Cin 128, stage channels 64/32, Sdim 16.
//
// Weights are raw float pointers (PyTorch row-major, weight_norm-fused):
//   Conv1d weight       [Cout, Cin, K]
//   ConvTranspose1d wt  [Cin,  Cout, K]
//   Linear weight       [out,  in]
//   AdaIN1d fc.weight    [2C,   style_dim]
//   Snake alpha          [C]
// The caller supplies them via GeneratorWeights so the function composes with
// any weight-loading boundary (GGUF tensor lookup, raw .f32 fixtures, …).

#pragma once

#include <vector>

namespace eliza_kokoro {

// One AdaINResBlock1 sub-block (the block has three, sharing the same channel
// count). convs use [Cout=Cin=C, Cin=C, K].
struct GenSubBlockWeights {
    const float * conv1_w = nullptr;     // [C, C, K]
    const float * conv1_b = nullptr;     // [C]
    const float * conv2_w = nullptr;     // [C, C, K]
    const float * conv2_b = nullptr;     // [C]
    const float * adain1_fc_w = nullptr; // [2C, Sdim]
    const float * adain1_fc_b = nullptr; // [2C]
    const float * adain2_fc_w = nullptr; // [2C, Sdim]
    const float * adain2_fc_b = nullptr; // [2C]
    const float * alpha1 = nullptr;      // [C]
    const float * alpha2 = nullptr;      // [C]
    // Trained InstanceNorm affine inside AdaIN1d (students only; null keeps
    // Kokoro-82M's identity affine).
    const float * adain1_norm_w = nullptr;  // [C]
    const float * adain1_norm_b = nullptr;  // [C]
    const float * adain2_norm_w = nullptr;  // [C]
    const float * adain2_norm_b = nullptr;  // [C]
};

struct GenAdaResBlockWeights {
    GenSubBlockWeights sub[3];
};

struct GeneratorWeights {
    // m_source.l_linear: Linear(harmonics -> 1).
    const float * l_linear_w = nullptr;  // [1, harmonics]
    const float * l_linear_b = nullptr;  // [1]

    // ups[0], ups[1]: ConvTranspose1d. weight [Cin, Cout, K], bias [Cout].
    const float * ups_w[2] = { nullptr, nullptr };
    const float * ups_b[2] = { nullptr, nullptr };

    // noise_convs[0], noise_convs[1]: Conv1d. weight [Cout, n_fft+2, K], bias [Cout].
    const float * noise_convs_w[2] = { nullptr, nullptr };
    const float * noise_convs_b[2] = { nullptr, nullptr };

    // noise_res[0] (k=7), noise_res[1] (k=11): AdaINResBlock1.
    GenAdaResBlockWeights noise_res[2];

    // resblocks[0..5]: AdaINResBlock1 (3 kernels per stage, 2 stages).
    GenAdaResBlockWeights resblocks[6];

    // conv_post: Conv1d(ch[1] -> n_fft+2, k=7, pad=3).
    const float * conv_post_w = nullptr;
    const float * conv_post_b = nullptr;
};

// Generator shape and behaviour. The defaults are Kokoro-82M v1.0.
struct GeneratorConfig {
    int Cin = 512;                    // input channels (upsample_initial_channel)
    int ch[2] = { 256, 128 };         // channels after ups[0], ups[1]
    int Sdim = 128;                   // style width
    int up_rates[2] = { 10, 6 };
    int up_kernels[2] = { 20, 12 };
    int rb_kernels[3] = { 3, 7, 11 };
    int dil[3] = { 1, 3, 5 };
    int n_fft = 20;
    int hop = 5;
    int harmonics = 9;                // harmonic_num + 1
    float sine_amp = 0.1f;
    float voiced_threshold = 10.0f;
    float sample_rate = 24000.0f;

    // Source STFT convention. false (Kokoro-82M as shipped): a direct double
    // DFT, whose rounding noise decides the sign of near-zero imaginary parts
    // and therefore the phase of near-empty bins. true (Paradee): imaginary
    // parts and magnitudes below the DFT's rounding floor are set to exactly
    // zero, so the DC/Nyquist bins of a real signal and the empty bins of a
    // constant (unvoiced, noise-free) source get torch.stft's phase (0, or pi
    // for a negative real part) instead of a sign picked by rounding noise.
    bool canonical_source_stft = false;
    // Harmonic source precision. false (Kokoro-82M as shipped): rad/phase
    // pass through float between the SineGen steps. true (Paradee): every
    // SineGen step runs in double and only sine_waves is rounded to float.
    bool precise_source = false;
    bool instnorm_affine = false;     // documents that the AdaIN1d norm_* pointers are set

    // Optional outputs / hooks.
    std::vector<float> * har_source_out = nullptr;   // m_source output [T0 * upsample_scale]
    const std::vector<float> * har_source_in = nullptr;  // parity hook: use this source instead of m_source
    bool (*cancel)(void * user) = nullptr;           // checked between generator stages
    void * cancel_user = nullptr;
};

// Generator.forward with an explicit configuration. Returns false only when
// `cfg.cancel` asked it to stop (audio is then cleared).
bool generator_forward(
        const float * x_in,      // [cfg.Cin, T0] channel-major
        int T0,                  // input time (== 2 * predictor T_frame)
        const float * s,         // [cfg.Sdim]
        const float * f0_curve,  // [T0]
        const GeneratorWeights & w,
        const GeneratorConfig & cfg,
        std::vector<float> & audio);

// Kokoro-82M Generator.forward. audio is resized to T0 * 300 (== 79200 for T0=264).
void kokoro_generator_forward(
        const float * x_in,      // [512, T0] channel-major
        int T0,                  // input time (== 2 * predictor T_frame)
        const float * s,         // [128]
        const float * f0_curve,  // [T0]
        const GeneratorWeights & w,
        std::vector<float> & audio);

} // namespace eliza_kokoro
