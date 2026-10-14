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
//   1. Albert BERT (n_layer, num_hidden_groups=1) over input phoneme ids.
//   2. bert_encoder Linear d_model→hid → d_en [T, hid].
//   3. predictor.text_encoder (DurationEncoder): 3 × (BiLSTM + AdaLayerNorm),
//      conditioned on the predictor style.
//      Output `d` has shape [T, hid+sty] (concat of LSTM output + style).
//   4. predictor.lstm BiLSTM over `d` → x [T, hid].
//   5. duration_proj Linear (hid→max_dur) → duration logits → sigmoid → sum/speed
//      → per-phoneme float duration. round.clamp(min=1).long() → pred_dur.
//
// Widths come from the GGUF tensor shapes, so the same code runs Kokoro-82M
// (hid 512, style 128) and width-scaled students of it such as Paradee-8M
// (hid 192, style 32). `predictor_forward` takes a PredictorConfig for the
// student-only parts (tensor namespace, asr projection MLP, InstanceNorm
// affine, torch rounding, cancellation); `kokoro_predictor_forward` is the
// Kokoro-82M entry and keeps its arithmetic unchanged.

#pragma once

#include "kokoro-layers.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;

namespace eliza_kokoro {

// Forward declaration — kokoro_model holds the actual tensor pointers.
struct kokoro_model;

// Output of the predictor pre-decoder stage. Populated by
// `kokoro_predictor_forward`. Caller owns the storage.
struct PredictorOut {
    int T_phon = 0;                       // padded input length (BOS+phonemes+EOS)
    int T_frame = 0;                      // total frames after length-regulate (sum of pred_dur)
    int hid = 0;                          // predictor hidden width (Kokoro 512)
    int sty = 0;                          // predictor style width (Kokoro 128)
    int asr_dim = 0;                      // width of `asr` rows (Kokoro 512 = hid)

    std::vector<float> bert;              // [T_phon, d_model] (only when PredictorConfig::keep_bert)
    std::vector<float> d;                 // [T_phon, hid+sty] = DurationEncoder output (LSTM cat style)
    std::vector<float> x_lstm;            // [T_phon, hid] = predictor.lstm output (forward+reverse concat)
    std::vector<float> duration;          // [T_phon] = sigmoid+sum (pre-round, post-speed)
    std::vector<int32_t> pred_dur;        // [T_phon] = round.clamp(1).long()

    // After length-regulate:
    std::vector<float> en;                // [T_frame, hid+sty] = d transposed and matmul'd with pred_aln_trg
    std::vector<float> t_en;              // [T_phon, hid] = text_encoder forward (separate from BERT branch)
    std::vector<float> asr;               // [T_frame, asr_dim] = (asr_proj(t_en) or t_en) @ pred_aln_trg

    // F0 / N predictions (run via predictor.F0Ntrain on `en`).
    std::vector<float> F0_pred;           // [2*T_frame]
    std::vector<float> N_pred;            // [2*T_frame]

    bool cancelled = false;               // set when PredictorConfig::cancel stopped the forward
};

// Cancellation hook: return true to stop. Checked between stages.
using predictor_cancel_fn = bool (*)(void * user);

struct PredictorConfig {
    // Tensor namespace: "<ns>.bert.*", "<ns>.predictor.*", ... ("kokoro" for
    // Kokoro-82M, "paradee" for Paradee-8M).
    std::string ns = "kokoro";

    // Albert text branch (from the GGUF hparams).
    int text_n_layer = 12;
    int text_n_head  = 12;
    int text_d_model = 768;
    int text_d_ff    = 2048;

    int n_layer     = 3;    // DurationEncoder / TextEncoder depth
    int max_dur     = 50;   // duration_proj width
    int text_kernel = 5;    // TextEncoder conv kernel

    // Predictor style vector, [sty] floats (sty read from the GGUF).
    const float * style = nullptr;

    // Student-only behaviour; all false reproduces Kokoro-82M exactly.
    bool torch_round      = false;  // round half to even like torch.round (Kokoro: std::round)
    bool instnorm_affine  = false;  // AdaIN1d applies the trained InstanceNorm affine (<pfx>.normN.norm.*)
    bool asr_proj_mlp     = false;  // asr = Linear(<ns>.asr_proj.1)(GELU(Linear(<ns>.asr_proj.0)(t_en)))
    bool keep_bert        = false;  // keep the BERT output in PredictorOut::bert (stage dumps)

    predictor_cancel_fn cancel = nullptr;
    void * cancel_user = nullptr;
};

// Generic predictor forward over a loaded F32 ggml context. Returns false on
// error (`err_out` set) or cancellation (`out.cancelled`, err_out "cancelled").
bool predictor_forward(
        ggml_context * ctx,
        const PredictorConfig & cfg,
        const std::vector<int32_t> & phoneme_ids,
        float speed,
        PredictorOut & out,
        std::string & err_out);

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
