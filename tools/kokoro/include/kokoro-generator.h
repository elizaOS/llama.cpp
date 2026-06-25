// SPDX-License-Identifier: MIT
//
// kokoro-generator.h — iSTFTNet Generator.forward (StyleTTS-2 decoder back-end).
//
// The generator turns the decoder body output `x` [512, 264], the style
// vector `s` [128], and the (un-downsampled) F0 curve `f0_curve` [264] into
// `audio` [79200] (24 kHz).
//
// Weights are raw float pointers (PyTorch row-major, weight_norm-fused):
//   Conv1d weight       [Cout, Cin, K]
//   ConvTranspose1d wt  [Cin,  Cout, K]
//   Linear weight       [out,  in]
//   AdaIN1d fc.weight    [2C,   style_dim]   (style_dim = 128)
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
    const float * adain1_fc_w = nullptr; // [2C, 128]
    const float * adain1_fc_b = nullptr; // [2C]
    const float * adain2_fc_w = nullptr; // [2C, 128]
    const float * adain2_fc_b = nullptr; // [2C]
    const float * alpha1 = nullptr;      // [C]
    const float * alpha2 = nullptr;      // [C]
};

struct GenAdaResBlockWeights {
    GenSubBlockWeights sub[3];
};

struct GeneratorWeights {
    // m_source.l_linear: Linear(9 -> 1).
    const float * l_linear_w = nullptr;  // [1, 9]
    const float * l_linear_b = nullptr;  // [1]

    // ups[0], ups[1]: ConvTranspose1d. weight [Cin, Cout, K], bias [Cout].
    const float * ups_w[2] = { nullptr, nullptr };
    const float * ups_b[2] = { nullptr, nullptr };

    // noise_convs[0], noise_convs[1]: Conv1d. weight [Cout, 22, K], bias [Cout].
    const float * noise_convs_w[2] = { nullptr, nullptr };
    const float * noise_convs_b[2] = { nullptr, nullptr };

    // noise_res[0] (k=7), noise_res[1] (k=11): AdaINResBlock1.
    GenAdaResBlockWeights noise_res[2];

    // resblocks[0..5]: AdaINResBlock1 (stage0: k=3,7,11 ch=256; stage1: ch=128).
    GenAdaResBlockWeights resblocks[6];

    // conv_post: Conv1d(128 -> 22, k=7, pad=3). weight [22, 128, 7], bias [22].
    const float * conv_post_w = nullptr;
    const float * conv_post_b = nullptr;
};

// Generator.forward. audio is resized to T0 * 300 (== 79200 for T0=264).
void kokoro_generator_forward(
        const float * x_in,      // [512, T0] channel-major
        int T0,                  // input time (== 2 * predictor T_frame)
        const float * s,         // [128]
        const float * f0_curve,  // [T0]
        const GeneratorWeights & w,
        std::vector<float> & audio);

} // namespace eliza_kokoro
