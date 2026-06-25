// SPDX-License-Identifier: MIT
//
// kokoro-decoder.h — StyleTTS-2 / iSTFTNet decoder: predictor outputs -> 24 kHz audio.
//
// Wires the validated decoder_front (kokoro-decoder-front.h) + Generator
// (kokoro-generator.h) against the model's all-F32 ggml context. Replaces the
// J2-ship placeholder spectrogram in kokoro_synthesize (#9588).

#pragma once

#include <string>
#include <vector>

namespace eliza_kokoro {

struct kokoro_model;

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
