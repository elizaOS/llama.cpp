// SPDX-License-Identifier: MIT
//
// kokoro-decoder.h — StyleTTS-2 / iSTFTNet decoder: predictor outputs -> 24 kHz audio.
//
// Wires the validated decoder_front (kokoro-decoder-front.h) + Generator
// (kokoro-generator.h) against the model's all-F32 ggml context. Replaces the
// J2-ship placeholder spectrogram in kokoro_synthesize (#9588).
//
// Every channel width is read from the GGUF tensor shapes, so the same code
// runs Kokoro-82M and width-scaled students of it (Paradee-8M).

#pragma once

#include <string>
#include <vector>

struct ggml_context;

namespace eliza_kokoro {

struct kokoro_model;

struct DecoderConfig {
    std::string ns = "kokoro";          // tensor namespace ("<ns>.decoder.*", "<ns>.gen.*")
    bool instnorm_affine = false;       // AdaIN1d uses the trained InstanceNorm affine (normN.norm.*, adainN.j.norm.*)
    bool canonical_source_stft = false; // see GeneratorConfig::canonical_source_stft
    bool precise_source = false;        // see GeneratorConfig::precise_source
    int up_rates[2] = { 10, 6 };        // generator upsample rates (kernels and widths come from the tensors)
    int istft_hop = 5;                  // generator iSTFT hop (n_fft comes from conv_post)
    std::vector<float> * har_source_out = nullptr;  // generator m_source output
    const std::vector<float> * har_source_in = nullptr;  // parity hook: replace m_source's output
    bool (*cancel)(void * user) = nullptr;
    void * cancel_user = nullptr;
    bool cancelled = false;             // out: set when cancel() stopped the forward
};

// Generic decoder over a loaded F32 context.
//   asr_ct : [asr_dim, T_frame] channel-major
//   F0, N  : [2*T_frame]
//   style  : [Sdim] decoder style (Sdim read from the GGUF)
// Returns false on error (err set) or cancellation (cfg.cancelled, err "cancelled").
bool decoder_forward(
        ggml_context * ctx,
        DecoderConfig & cfg,
        const float * asr_ct, int asr_dim, int T_frame,
        const float * F0, const float * N,
        const float * style,
        std::vector<float> & audio,
        std::string & err);

// Run the full decoder. Inputs come from kokoro_predictor_forward:
//   asr_ct   : [512, T_frame] channel-major (transpose of PredictorOut.asr [T,512])
//   F0, N    : [2*T_frame]    (PredictorOut.F0_pred / N_pred — the up-2x curves)
//   ref_s_dec: [128]          decoder-half style (ref_s[:128])
// Output: audio (24 kHz mono), resized to (2*T_frame)*300.
bool kokoro_decoder_forward(
        const kokoro_model * model,
        const float * asr_ct, int T_frame,
        const float * F0, const float * N,
        const float * ref_s_dec,
        std::vector<float> & audio,
        std::string & err);

} // namespace eliza_kokoro
