// SPDX-License-Identifier: MIT
//
// kokoro-predictor.h — StyleTTS-2 prosody predictor forward.
//
// Mirrors kokoro/modules.py ProsodyPredictor + DurationEncoder + TextEncoder.
// Consumes the tensors emitted by convert_kokoro_pth_to_gguf.py (the
// `kokoro.bert.*`, `kokoro.bert_encoder.*`, `kokoro.text_encoder.*`,
// `kokoro.predictor.*` namespaces). Pure-CPU scalar forward — produces
// numerically equivalent activations to the upstream Python reference.
//
// Stages (all forwarded by `kokoro_predictor_forward`):
//   1. Albert BERT (12 layers, num_hidden_groups=1) over input phoneme ids.
//   2. bert_encoder Linear 768→512 → d_en [T, 512].
//   3. predictor.text_encoder (DurationEncoder): 3 × (BiLSTM + AdaLayerNorm),
//      conditioned on ref_s[:, 128:] (predictor-half style).
//      Output `d` has shape [T, 512+128=640] (concat of LSTM output + style).
//   4. predictor.lstm BiLSTM over `d` → x [T, 512].
//   5. duration_proj Linear (512→50) → duration logits → sigmoid → sum/speed
//      → per-phoneme float duration. round.clamp(min=1).long() → pred_dur.

#pragma once

#include "kokoro-layers.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace eliza_kokoro {

// Forward declaration — kokoro_model holds the actual tensor pointers.
struct kokoro_model;

// Output of the predictor pre-decoder stage. Populated by
// `kokoro_predictor_forward`. Caller owns the storage.
struct PredictorOut {
    int T_phon = 0;                       // padded input length (BOS+phonemes+EOS)
    int T_frame = 0;                      // total frames after length-regulate (sum of pred_dur)

    std::vector<float> d;                 // [T_phon, 640] = DurationEncoder output (LSTM cat style)
    std::vector<float> x_lstm;            // [T_phon, 512] = predictor.lstm output (forward+reverse concat)
    std::vector<float> duration;          // [T_phon] = sigmoid+sum (pre-round, post-speed)
    std::vector<int32_t> pred_dur;        // [T_phon] = round.clamp(1).long()

    // After length-regulate:
    std::vector<float> en;                // [T_frame, 640] = d transposed and matmul'd with pred_aln_trg
    std::vector<float> t_en;              // [T_phon, 512] = text_encoder forward (separate from BERT branch)
    std::vector<float> asr;               // [T_frame, 512] = t_en @ pred_aln_trg

    // F0 / N predictions (run via predictor.F0Ntrain on `en`).
    std::vector<float> F0_pred;           // [T_frame]
    std::vector<float> N_pred;            // [T_frame]
};

// Run the predictor + text_encoder + length-regulation. `phoneme_ids` is the
// raw phoneme sequence (BOS+phon+EOS); `ref_s` is the 256-dim style vector
// (split internally into predictor + decoder halves).
//
// Returns true on success; on failure, `err_out` is populated and the
// PredictorOut may be partially filled.
bool kokoro_predictor_forward(
        const kokoro_model * model,
        const std::vector<int32_t> & phoneme_ids,
        const float * ref_s /* [256] */,
        float speed,
        PredictorOut & out,
        std::string & err_out);

} // namespace eliza_kokoro
