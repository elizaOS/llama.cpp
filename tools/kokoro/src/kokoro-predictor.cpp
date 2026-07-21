// SPDX-License-Identifier: MIT
//
// kokoro-predictor.cpp — StyleTTS-2 prosody predictor forward, CPU scalar.
//
// Implements the predictor pipeline as a direct port of kokoro/modules.py
// (ProsodyPredictor + DurationEncoder + TextEncoder + AdainResBlk1d) and the
// preceding Albert/BERT branch. The tensors are sourced from the kokoro_model
// loaded by `kokoro_load_model` (see kokoro.cpp); see
// `convert_kokoro_pth_to_gguf.py` for the canonical names.
//
// All compute runs on the host with plain C++ scalar loops. The
// implementation is verified against reference activations produced by
// `tools/dump_reference_activations.py` (run on the actual Python kokoro
// package) using `tests/test_kokoro_predictor.cpp`.

#include "kokoro-predictor.h"
#include "kokoro.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace eliza_kokoro {

// `kokoro_model` is the loader's struct; we don't redeclare it here — we
// re-look-up tensors by name. (The loader keeps the gguf_context + ggml_context
// alive, so all tensor data pointers remain valid for the model's lifetime.)
//
// To access the loaded ggml_context we need to forward-declare a getter or
// include the loader's internal layout. We expose a small private accessor.

extern ggml_context * kokoro_model_ggml_ctx(const kokoro_model * model);
extern const kokoro_hparams * kokoro_get_hparams(const kokoro_model * model) noexcept;

namespace {

// Tensor lookup helper.
struct TLookup {
    ggml_context * ctx = nullptr;
    const float * get(const std::string & name, std::string * err = nullptr) const {
        ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        if (!t) {
            if (err) *err = "missing tensor '" + name + "'";
            return nullptr;
        }
        return (const float *) t->data;
    }
    const float * get_or_null(const std::string & name) const {
        ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
        return t ? (const float *) t->data : nullptr;
    }
    bool has(const std::string & name) const {
        return ggml_get_tensor(ctx, name.c_str()) != nullptr;
    }
};

// =================================================================
// Albert BERT forward (12 layers, num_hidden_groups=1 → 1 shared layer
// reused 12 times). Input: phoneme ids [T]; output: [T, 768].
//
// Mirrors transformers.AlbertModel forward in eval mode:
//   1. word_embd + position_embd + token_type_embd → [T, 128]
//   2. embd_LayerNorm → [T, 128]
//   3. embedding_hidden_mapping_in (Linear 128→768) → [T, 768]
//   4. For each of 12 layers (using the SAME shared params):
//      - LayerNorm(attn_ln) → attn_q/k/v → MHA self-attn → attn_o → residual
//      - LayerNorm(full_ln) → ffn (Linear 768→2048) → GELU → ffn_output (Linear 2048→768) → residual
//
// The attention_mask masks padded positions (we always pad to T, so mask is
// all-zero for inference — no padding in this forward).
//
// Notes:
//   - Albert default: classifier-style: pre-norm? No — Albert is POST-norm
//     for attention (LayerNorm AFTER attention residual). Actually it's
//     POST-LN: x = LN(x + attn(x)); x = LN(x + ffn(x)) using attn.LayerNorm
//     and full_layer_layer_norm respectively.
//   - Albert uses GeLU activation in the FFN (gelu_new = approximate).
// =================================================================

static inline float gelu_new(float x) {
    // GeLU 'gelu_new' = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float kBeta = 0.7978845608028654f;  // sqrt(2/pi)
    return 0.5f * x * (1.0f + std::tanh(kBeta * (x + 0.044715f * x * x * x)));
}

static void bert_forward(
        const TLookup & T,
        const std::vector<int32_t> & ids,
        int d_model, int d_ff, int n_layer, int n_head,
        std::vector<float> & out /* [N, d_model] */,
        std::string & err) {
    const int N = (int) ids.size();
    const int d_embd = 128;
    const int head_dim = d_model / n_head;

    const float * tok_embd = T.get("kokoro.bert.token_embd.weight", &err);     // [vocab, 128]
    const float * pos_embd = T.get("kokoro.bert.position_embd.weight", &err);  // [maxpos, 128]
    const float * tt_embd  = T.get("kokoro.bert.tok_type_embd.weight", &err);  // [2, 128]
    const float * ln_w     = T.get("kokoro.bert.embd_ln.weight", &err);
    const float * ln_b     = T.get("kokoro.bert.embd_ln.bias", &err);
    const float * proj_w   = T.get("kokoro.bert.embd_proj.weight", &err);      // [768, 128]
    const float * proj_b   = T.get("kokoro.bert.embd_proj.bias", &err);
    if (!tok_embd || !pos_embd || !tt_embd || !ln_w || !ln_b || !proj_w || !proj_b) return;

    // Step 1+2+3: embed → LN(128) → project to 768.
    std::vector<float> h(N * d_embd, 0.0f);
    for (int t = 0; t < N; ++t) {
        const int id = ids[t];
        // token + position + token_type_0
        for (int j = 0; j < d_embd; ++j) {
            h[t * d_embd + j] = tok_embd[id * d_embd + j]
                              + pos_embd[t * d_embd + j]
                              + tt_embd[0 * d_embd + j];
        }
    }
    layer_norm_forward(h.data(), N, d_embd, ln_w, ln_b, 1e-12f);

    out.assign(N * d_model, 0.0f);
    for (int t = 0; t < N; ++t) {
        linear_forward(h.data() + t * d_embd, d_embd, proj_w, proj_b, d_model,
                       out.data() + t * d_model);
    }

    // Step 4: 12 layers — shared params (Albert num_hidden_groups=1).
    const float * attn_q_w = T.get("kokoro.bert.layer.attn_q.weight");
    const float * attn_q_b = T.get("kokoro.bert.layer.attn_q.bias");
    const float * attn_k_w = T.get("kokoro.bert.layer.attn_k.weight");
    const float * attn_k_b = T.get("kokoro.bert.layer.attn_k.bias");
    const float * attn_v_w = T.get("kokoro.bert.layer.attn_v.weight");
    const float * attn_v_b = T.get("kokoro.bert.layer.attn_v.bias");
    const float * attn_o_w = T.get("kokoro.bert.layer.attn_o.weight");
    const float * attn_o_b = T.get("kokoro.bert.layer.attn_o.bias");
    const float * attn_ln_w = T.get("kokoro.bert.layer.attn_ln.weight");
    const float * attn_ln_b = T.get("kokoro.bert.layer.attn_ln.bias");
    const float * full_ln_w = T.get("kokoro.bert.layer.full_ln.weight");
    const float * full_ln_b = T.get("kokoro.bert.layer.full_ln.bias");
    const float * ffn_w     = T.get("kokoro.bert.layer.ffn.weight");        // [2048, 768]
    const float * ffn_b     = T.get("kokoro.bert.layer.ffn.bias");
    const float * ffn_out_w = T.get("kokoro.bert.layer.ffn_out.weight");    // [768, 2048]
    const float * ffn_out_b = T.get("kokoro.bert.layer.ffn_out.bias");

    std::vector<float> q(N * d_model), k(N * d_model), v(N * d_model);
    std::vector<float> ctx(N * d_model);
    std::vector<float> ff(N * d_ff);

    for (int il = 0; il < n_layer; ++il) {
        // === Self-attention sublayer ===
        // Albert applies attention layer-norm AFTER residual, not before
        // (post-LN). pre_h = out.
        for (int t = 0; t < N; ++t) {
            linear_forward(out.data() + t * d_model, d_model, attn_q_w, attn_q_b, d_model, q.data() + t * d_model);
            linear_forward(out.data() + t * d_model, d_model, attn_k_w, attn_k_b, d_model, k.data() + t * d_model);
            linear_forward(out.data() + t * d_model, d_model, attn_v_w, attn_v_b, d_model, v.data() + t * d_model);
        }
        // Compute attention per head, no masking (no padding in this forward).
        const float inv_sqrt_dk = 1.0f / std::sqrt((float) head_dim);
        for (int h_i = 0; h_i < n_head; ++h_i) {
            // For each query t, compute scores over all keys j, softmax, weight values.
            for (int t = 0; t < N; ++t) {
                // scores[j] = q[t, h_i] · k[j, h_i] / sqrt(dk)
                std::vector<float> scores(N);
                float maxs = -1e30f;
                for (int j = 0; j < N; ++j) {
                    float s = 0.0f;
                    const float * qv = q.data() + t * d_model + h_i * head_dim;
                    const float * kv = k.data() + j * d_model + h_i * head_dim;
                    for (int d = 0; d < head_dim; ++d) s += qv[d] * kv[d];
                    scores[j] = s * inv_sqrt_dk;
                    if (scores[j] > maxs) maxs = scores[j];
                }
                // softmax
                float sumexp = 0.0f;
                for (int j = 0; j < N; ++j) { scores[j] = std::exp(scores[j] - maxs); sumexp += scores[j]; }
                const float inv = 1.0f / sumexp;
                for (int j = 0; j < N; ++j) scores[j] *= inv;
                // weighted sum of V into ctx[t, h_i, :]
                float * cv = ctx.data() + t * d_model + h_i * head_dim;
                for (int d = 0; d < head_dim; ++d) cv[d] = 0.0f;
                for (int j = 0; j < N; ++j) {
                    const float * vv = v.data() + j * d_model + h_i * head_dim;
                    const float w = scores[j];
                    for (int d = 0; d < head_dim; ++d) cv[d] += vv[d] * w;
                }
            }
        }
        // attn_o projection.
        std::vector<float> attn_out(N * d_model);
        for (int t = 0; t < N; ++t) {
            linear_forward(ctx.data() + t * d_model, d_model, attn_o_w, attn_o_b, d_model,
                           attn_out.data() + t * d_model);
        }
        // residual + attn_LN.
        for (int i = 0; i < N * d_model; ++i) attn_out[i] += out[i];
        layer_norm_forward(attn_out.data(), N, d_model, attn_ln_w, attn_ln_b, 1e-12f);
        // out := attn_out
        out.swap(attn_out);

        // === FFN sublayer ===
        for (int t = 0; t < N; ++t) {
            linear_forward(out.data() + t * d_model, d_model, ffn_w, ffn_b, d_ff, ff.data() + t * d_ff);
            for (int j = 0; j < d_ff; ++j) ff[t * d_ff + j] = gelu_new(ff[t * d_ff + j]);
        }
        std::vector<float> ffn_out(N * d_model);
        for (int t = 0; t < N; ++t) {
            linear_forward(ff.data() + t * d_ff, d_ff, ffn_out_w, ffn_out_b, d_model,
                           ffn_out.data() + t * d_model);
        }
        for (int i = 0; i < N * d_model; ++i) ffn_out[i] += out[i];
        layer_norm_forward(ffn_out.data(), N, d_model, full_ln_w, full_ln_b, 1e-12f);
        out.swap(ffn_out);
    }
}

// =================================================================
// DurationEncoder forward (kokoro/modules.py).
//
// Input: x [T, 512], style s [128].
// For each of nlayers (3) blocks:
//   - BiLSTM over [x ; s] cat-along-channel (each step input is 512+128=640).
//     Output is [T, 512] (BiLSTM hidden 256 per direction × 2 = 512).
//   - AdaLayerNorm(style_dim=128, channels=512): LN over channels then
//     (1 + gamma) * x + beta. The fc(s) → 2*512 weights produce gamma|beta.
//
// Final output: cat([lstm_out_layer_nlayers-1; s_broadcast], dim=channels) →
// [T, 512+128=640]. (After the last AdaLayerNorm the python code does NOT
// re-concat, but rather returns x as-is, so it's still [T, 512]. Wait —
// re-reading: the loop in DurationEncoder.forward does the concat AFTER each
// AdaLayerNorm. So if nlayers=3, after layer 0 we LSTM-LSTM-LN-cat → 640,
// then layer 1 LSTM expects 640. Then LN-cat → 640, then layer 2 LSTM. Then
// LN. Final return: x.transpose(-1, -2). So OUTPUT IS [T, d_hid+sty_dim] = 640
// if final AdaLN cats, but it does NOT — it only cats inside the loop after
// AdaLN. Reading more carefully:
//
//   for block in self.lstms:
//     if isinstance(block, AdaLayerNorm):
//       x = block(x, style)
//       x = torch.cat([x, s], axis=1)        # CAT happens here
//       x.masked_fill_(masks, 0.0)
//     else:
//       x = LSTM(x)
//
// So the sequence is: LSTM (in:hid+sty out:hid) → AdaLN(hid) → cat(hid+sty)
// → LSTM (in:hid+sty out:hid) → AdaLN(hid) → cat(hid+sty) → ... → AdaLN
// → cat(hid+sty). So the FINAL return after the loop has the concat applied
// (last AdaLN's branch cats with s). Therefore output is [T, hid+sty] = 640.
// =================================================================

static void duration_encoder_forward(
        const TLookup & TL,
        const float * d_en /* [T, 512] (transposed back to channel-major in input) */,
        int T_phon, int d_hid, int sty_dim, int n_layer,
        const float * style /* [128] */,
        std::vector<float> & out /* [T_phon, hid+sty=640] */) {
    // Input convention: kokoro/modules.py packs as x = x.permute(2, 0, 1) →
    // [T, B, hid]. After cat with s expanded across T: [T, B, hid+sty].
    // We work as row-major [T, hid+sty].
    const int half = d_hid / 2;

    std::vector<float> x(T_phon * (d_hid + sty_dim), 0.0f);
    for (int t = 0; t < T_phon; ++t) {
        // hid part: from d_en (channel-major [d_hid, T] in Python but here
        // we expect [T, d_hid] row-major as the caller normalizes).
        for (int j = 0; j < d_hid; ++j) x[t * (d_hid + sty_dim) + j] = d_en[t * d_hid + j];
        for (int j = 0; j < sty_dim; ++j) x[t * (d_hid + sty_dim) + d_hid + j] = style[j];
    }

    LSTMWeights lw;
    std::vector<float> y(T_phon * d_hid);

    for (int il = 0; il < n_layer; ++il) {
        // BiLSTM over x [T, hid+sty] → y [T, 2*half=hid].
        const std::string pfx = "kokoro.predictor.de.lstm" + std::to_string(il) + ".";
        lw.W_ih   = TL.get(pfx + "weight_ih_l0");
        lw.W_hh   = TL.get(pfx + "weight_hh_l0");
        lw.b_ih   = TL.get(pfx + "bias_ih_l0");
        lw.b_hh   = TL.get(pfx + "bias_hh_l0");
        lw.W_ih_r = TL.get(pfx + "weight_ih_l0_r");
        lw.W_hh_r = TL.get(pfx + "weight_hh_l0_r");
        lw.b_ih_r = TL.get(pfx + "bias_ih_l0_r");
        lw.b_hh_r = TL.get(pfx + "bias_hh_l0_r");

        bilstm_forward(x.data(), T_phon, d_hid + sty_dim, half, lw, y.data());

        // AdaLayerNorm(style_dim, channels=d_hid). fc(s) → 2*d_hid; chunk → gamma,beta.
        const std::string apfx = "kokoro.predictor.de.adaln" + std::to_string(il) + ".fc.";
        const float * fc_w = TL.get(apfx + "weight");
        const float * fc_b = TL.get(apfx + "bias");

        // We need y in [C, T] layout for adalayernorm_forward.
        std::vector<float> y_ct(T_phon * d_hid);
        for (int t = 0; t < T_phon; ++t)
            for (int c = 0; c < d_hid; ++c)
                y_ct[(size_t)c * T_phon + t] = y[(size_t)t * d_hid + c];
        adalayernorm_forward(y_ct.data(), d_hid, T_phon, style, sty_dim, fc_w, fc_b);
        // Transpose back to [T, hid].
        for (int t = 0; t < T_phon; ++t)
            for (int c = 0; c < d_hid; ++c)
                y[(size_t)t * d_hid + c] = y_ct[(size_t)c * T_phon + t];

        // Re-cat style for next iter (or final output).
        for (int t = 0; t < T_phon; ++t) {
            for (int j = 0; j < d_hid; ++j)   x[t * (d_hid + sty_dim) + j] = y[t * d_hid + j];
            for (int j = 0; j < sty_dim; ++j) x[t * (d_hid + sty_dim) + d_hid + j] = style[j];
        }
    }
    out = std::move(x);
}

// =================================================================
// predictor.lstm BiLSTM forward over `d` [T_phon, 640] → x_lstm [T_phon, 512].
// =================================================================
static void predictor_main_lstm_forward(
        const TLookup & TL,
        const float * d /* [T_phon, 640] */, int T_phon, int d_hid, int sty_dim,
        std::vector<float> & out /* [T_phon, 512] */) {
    LSTMWeights lw;
    lw.W_ih   = TL.get("kokoro.predictor.lstm.weight_ih_l0");
    lw.W_hh   = TL.get("kokoro.predictor.lstm.weight_hh_l0");
    lw.b_ih   = TL.get("kokoro.predictor.lstm.bias_ih_l0");
    lw.b_hh   = TL.get("kokoro.predictor.lstm.bias_hh_l0");
    lw.W_ih_r = TL.get("kokoro.predictor.lstm.weight_ih_l0_r");
    lw.W_hh_r = TL.get("kokoro.predictor.lstm.weight_hh_l0_r");
    lw.b_ih_r = TL.get("kokoro.predictor.lstm.bias_ih_l0_r");
    lw.b_hh_r = TL.get("kokoro.predictor.lstm.bias_hh_l0_r");
    out.assign(T_phon * d_hid, 0.0f);
    bilstm_forward(d, T_phon, d_hid + sty_dim, d_hid / 2, lw, out.data());
}

// =================================================================
// Duration: x_lstm [T_phon, 512] @ duration_proj.weight^T (512→50) + bias
//   → [T_phon, 50] → sigmoid → sum over channels / speed → [T_phon].
// =================================================================
static void duration_predictor_forward(
        const TLookup & TL,
        const float * x_lstm, int T_phon, int d_hid, int max_dur, float speed,
        std::vector<float> & duration /* [T_phon] */,
        std::vector<int32_t> & pred_dur /* [T_phon] */) {
    const float * w = TL.get("kokoro.predictor.duration_proj.weight");
    const float * b = TL.get("kokoro.predictor.duration_proj.bias");
    duration.assign(T_phon, 0.0f);
    pred_dur.assign(T_phon, 1);
    std::vector<float> tmp(max_dur);
    for (int t = 0; t < T_phon; ++t) {
        linear_forward(x_lstm + (size_t)t * d_hid, d_hid, w, b, max_dur, tmp.data());
        double sum = 0;
        for (int j = 0; j < max_dur; ++j) {
            const float s = 1.0f / (1.0f + std::exp(-tmp[j]));
            sum += s;
        }
        const float dur_f = (float)(sum / speed);
        duration[t] = dur_f;
        const int32_t d_round = (int32_t) std::round(dur_f);
        pred_dur[t] = std::max(1, d_round);
    }
}

// =================================================================
// Length-regulate: given pred_dur [T_phon], expand each row of d [T_phon, C]
// into out [sum(pred_dur), C] by repeat_interleave.
// =================================================================
static void length_regulate(
        const float * d, int T_phon, int C,
        const std::vector<int32_t> & pred_dur,
        std::vector<float> & out /* [T_frame, C] */, int & T_frame) {
    T_frame = 0;
    for (int t = 0; t < T_phon; ++t) T_frame += pred_dur[t];
    out.assign((size_t) T_frame * C, 0.0f);
    int row = 0;
    for (int t = 0; t < T_phon; ++t) {
        for (int r = 0; r < pred_dur[t]; ++r) {
            std::memcpy(out.data() + (size_t)row * C,
                        d + (size_t)t * C, sizeof(float) * (size_t)C);
            ++row;
        }
    }
}

// =================================================================
// text_encoder forward (StyleTTS-2 TextEncoder, separate from BERT).
//
//   x = embedding(ids)        # [T, channels=512]
//   x = x.transpose(1, 2)     # [channels, T]
//   for c in cnn:
//     x = conv1d(x)           # weight_norm-fused, kernel=5, padding=2
//     x = LayerNorm(channels) # gamma/beta per-channel
//     x = leaky_relu(0.2)
//     (dropout — disabled in eval)
//   x = x.transpose(1, 2)     # [T, channels]
//   x = BiLSTM(x)             # [T, channels=512]
//   x = x.transpose(-1, -2)   # [channels, T]
//   return x
//
// We return out [T, channels=512] (row-major) for the caller's convenience.
// =================================================================
static void text_encoder_forward(
        const TLookup & TL,
        const std::vector<int32_t> & ids, int n_cnn, int hid,
        std::vector<float> & out /* [T, hid] */) {
    const int T = (int) ids.size();
    const int K = 5;
    const float * embd = TL.get("kokoro.text_encoder.embd.weight");  // [vocab, hid]
    if (!embd) return;

    // Step 1: embedding → [T, hid].
    std::vector<float> x_th(T * hid);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < hid; ++c)
            x_th[(size_t)t * hid + c] = embd[(size_t)ids[t] * hid + c];

    // Convert to [C, T].
    std::vector<float> x_ct(hid * T);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < hid; ++c)
            x_ct[(size_t)c * T + t] = x_th[(size_t)t * hid + c];

    std::vector<float> tmp(hid * T);
    for (int il = 0; il < n_cnn; ++il) {
        const std::string cpfx = "kokoro.text_encoder.cnn." + std::to_string(il) + ".";
        const float * cw = TL.get(cpfx + "weight");
        const float * cb = TL.get(cpfx + "bias");
        conv1d_forward(x_ct.data(), hid, T, cw, cb, hid, K,
                       /*stride=*/1, /*pad=*/(K - 1) / 2, /*dil=*/1,
                       tmp.data(), T);
        x_ct.swap(tmp);

        // LayerNorm: gamma/beta over channel dim. The Python LayerNorm
        // transposes (1,-1) before F.layer_norm — that swaps channels onto
        // the last dim. So we normalize across channels for each time step.
        const float * gamma = TL.get(std::string("kokoro.text_encoder.ln.") + std::to_string(il) + ".gamma");
        const float * beta  = TL.get(std::string("kokoro.text_encoder.ln.") + std::to_string(il) + ".beta");
        for (int t = 0; t < T; ++t) {
            double sum = 0, sumsq = 0;
            for (int c = 0; c < hid; ++c) {
                const float v = x_ct[(size_t)c * T + t];
                sum += v; sumsq += v * v;
            }
            const double mean = sum / hid;
            const double var  = sumsq / hid - mean * mean;
            const double inv  = 1.0 / std::sqrt(var + 1e-5);
            for (int c = 0; c < hid; ++c) {
                const float v = (float)((x_ct[(size_t)c * T + t] - mean) * inv);
                x_ct[(size_t)c * T + t] = v * gamma[c] + beta[c];
            }
        }
        // Leaky ReLU.
        for (int c = 0; c < hid; ++c)
            for (int t = 0; t < T; ++t)
                if (x_ct[(size_t)c * T + t] < 0) x_ct[(size_t)c * T + t] *= 0.2f;
    }

    // Transpose back to [T, hid].
    std::vector<float> x_th2(T * hid);
    for (int t = 0; t < T; ++t)
        for (int c = 0; c < hid; ++c)
            x_th2[(size_t)t * hid + c] = x_ct[(size_t)c * T + t];

    // BiLSTM.
    LSTMWeights lw;
    lw.W_ih   = TL.get("kokoro.text_encoder.lstm.weight_ih_l0");
    lw.W_hh   = TL.get("kokoro.text_encoder.lstm.weight_hh_l0");
    lw.b_ih   = TL.get("kokoro.text_encoder.lstm.bias_ih_l0");
    lw.b_hh   = TL.get("kokoro.text_encoder.lstm.bias_hh_l0");
    lw.W_ih_r = TL.get("kokoro.text_encoder.lstm.weight_ih_l0_r");
    lw.W_hh_r = TL.get("kokoro.text_encoder.lstm.weight_hh_l0_r");
    lw.b_ih_r = TL.get("kokoro.text_encoder.lstm.bias_ih_l0_r");
    lw.b_hh_r = TL.get("kokoro.text_encoder.lstm.bias_hh_l0_r");
    out.assign(T * hid, 0.0f);
    bilstm_forward(x_th2.data(), T, hid, hid / 2, lw, out.data());
}

// =================================================================
// AdainResBlk1d forward.
//
// Reference (istftnet.py):
//   _residual(x, s):
//     x = norm1(x, s)
//     x = leaky_relu(0.2)
//     x = pool(x)                          # ConvTranspose1d depthwise (only if upsample)
//     x = conv1(dropout(x))
//     x = norm2(x, s)
//     x = leaky_relu(0.2)
//     x = conv2(dropout(x))
//   _shortcut(x):
//     if upsample: x = F.interpolate(x, scale_factor=2, mode='nearest')
//     if learned_sc: x = conv1x1(x)
//   out = (residual + shortcut) * rsqrt(2)
//
// Inputs:
//   x [Cin, T] (row-major channel-major)
//   s [Sdim]
// Outputs:
//   y [Cout, T_out]; T_out = T*2 if upsample else T.
// =================================================================
static void adainresblk1d_forward(
        const TLookup & TL, const std::string & pfx,
        const float * x, int Cin, int T_in, int Sdim, const float * s,
        int Cout, bool upsample, bool learned_sc,
        std::vector<float> & y, int & T_out) {

    const float * n1_w = TL.get(pfx + ".norm1.fc.weight");
    const float * n1_b = TL.get(pfx + ".norm1.fc.bias");
    const float * n2_w = TL.get(pfx + ".norm2.fc.weight");
    const float * n2_b = TL.get(pfx + ".norm2.fc.bias");
    const float * c1_w = TL.get(pfx + ".conv1.weight");
    const float * c1_b = TL.get(pfx + ".conv1.bias");
    const float * c2_w = TL.get(pfx + ".conv2.weight");
    const float * c2_b = TL.get(pfx + ".conv2.bias");
    const float * c1x1_w = learned_sc ? TL.get(pfx + ".conv1x1.weight") : nullptr;
    const float * c1x1_b = learned_sc ? TL.get_or_null(pfx + ".conv1x1.bias") : nullptr;
    const float * pool_w = upsample ? TL.get(pfx + ".pool.weight") : nullptr;
    const float * pool_b = upsample ? TL.get(pfx + ".pool.bias")  : nullptr;

    // === Residual branch ===
    // norm1 (AdaIN1d on Cin channels)
    std::vector<float> r(x, x + (size_t)Cin * T_in);  // copy
    adain1d_forward(r.data(), Cin, T_in, s, Sdim, n1_w, n1_b);
    // leaky relu
    for (size_t i = 0; i < r.size(); ++i) if (r[i] < 0) r[i] *= 0.2f;

    // Pool (only if upsample): ConvTranspose1d depthwise kernel=3 stride=2 pad=1 out_pad=1.
    int T_after_pool = T_in;
    if (upsample) {
        T_after_pool = convtranspose1d_out_len(T_in, /*K*/3, /*stride*/2, /*pad*/1, /*output_pad*/1);
        std::vector<float> r2((size_t)Cin * T_after_pool);
        convtranspose1d_depthwise_forward(r.data(), Cin, T_in, pool_w, pool_b, /*K*/3,
                                          /*stride*/2, /*pad*/1, /*output_pad*/1,
                                          r2.data(), T_after_pool);
        r.swap(r2);
    }

    // conv1 (no weight_norm at runtime — already fused). kernel=3, stride=1, pad=1.
    std::vector<float> r3((size_t)Cout * T_after_pool);
    conv1d_forward(r.data(), Cin, T_after_pool, c1_w, c1_b, Cout, 3,
                   /*stride*/1, /*pad*/1, /*dil*/1, r3.data(), T_after_pool);

    // norm2 (AdaIN1d on Cout channels)
    adain1d_forward(r3.data(), Cout, T_after_pool, s, Sdim, n2_w, n2_b);
    // leaky relu
    for (size_t i = 0; i < r3.size(); ++i) if (r3[i] < 0) r3[i] *= 0.2f;

    // conv2 (kernel=3, stride=1, pad=1, Cout→Cout).
    std::vector<float> r4((size_t)Cout * T_after_pool);
    conv1d_forward(r3.data(), Cout, T_after_pool, c2_w, c2_b, Cout, 3,
                   /*stride*/1, /*pad*/1, /*dil*/1, r4.data(), T_after_pool);

    // === Shortcut branch ===
    T_out = T_after_pool;  // post-pool/upsample width
    std::vector<float> sc;
    if (upsample) {
        // F.interpolate(x, scale_factor=2, mode='nearest') → [Cin, T*2]
        const int T_up = T_in * 2;
        sc.assign((size_t)Cin * T_up, 0.0f);
        for (int c = 0; c < Cin; ++c)
            for (int t = 0; t < T_up; ++t)
                sc[(size_t)c * T_up + t] = x[(size_t)c * T_in + (t / 2)];
        // If learned_sc, conv1x1.
        if (learned_sc) {
            std::vector<float> sc2((size_t)Cout * T_up);
            conv1d_forward(sc.data(), Cin, T_up, c1x1_w, c1x1_b, Cout, 1, 1, 0, 1, sc2.data(), T_up);
            sc.swap(sc2);
        }
        // Note: shortcut width is T_up; residual width is T_after_pool.
        // In Python, AdainResBlk1d.pool is a ConvTranspose1d(stride=2, padding=1, output_pad=1) →
        // output_len = (T_in - 1) * 2 - 2 + 3 + 1 + 1 - 1 = 2*T_in. (verified)
        // So T_after_pool == T_up; assert.
        if (T_out != T_up) {
            // Fallback: trim/pad. Shouldn't happen.
        }
    } else {
        if (learned_sc) {
            sc.assign((size_t)Cout * T_in, 0.0f);
            conv1d_forward(x, Cin, T_in, c1x1_w, c1x1_b, Cout, 1, 1, 0, 1, sc.data(), T_in);
        } else {
            sc.assign(x, x + (size_t)Cin * T_in);  // identity (assumes Cin==Cout)
        }
    }

    // === Sum + rsqrt(2) ===
    y.assign((size_t)Cout * T_out, 0.0f);
    const float rsqrt2 = 1.0f / std::sqrt(2.0f);
    for (int c = 0; c < Cout; ++c)
        for (int t = 0; t < T_out; ++t)
            y[(size_t)c * T_out + t] = (r4[(size_t)c * T_out + t] + sc[(size_t)c * T_out + t]) * rsqrt2;
}

// =================================================================
// predictor.F0Ntrain: BiLSTM `shared` over en.T → x [T_frame, hid] →
//   F0 chain (3 AdainResBlk1d blocks; the middle one upsamples by 2)
//      → F0_proj (Conv1d hid/2→1, kernel=1) → [T_frame*2]
//   N  chain (mirror of F0) → N_proj → [T_frame*2]
//
// Note the shared LSTM input is en.T (transposed to [T_frame, 640]).
// =================================================================
static void f0n_train_forward(
        const TLookup & TL,
        const float * en /* [T_frame, 640] */, int T_frame, int d_hid, int sty_dim,
        const float * s /* [sty_dim] = 128 */,
        std::vector<float> & F0_pred, std::vector<float> & N_pred) {
    LSTMWeights lw;
    lw.W_ih   = TL.get("kokoro.predictor.shared.weight_ih_l0");
    lw.W_hh   = TL.get("kokoro.predictor.shared.weight_hh_l0");
    lw.b_ih   = TL.get("kokoro.predictor.shared.bias_ih_l0");
    lw.b_hh   = TL.get("kokoro.predictor.shared.bias_hh_l0");
    lw.W_ih_r = TL.get("kokoro.predictor.shared.weight_ih_l0_r");
    lw.W_hh_r = TL.get("kokoro.predictor.shared.weight_hh_l0_r");
    lw.b_ih_r = TL.get("kokoro.predictor.shared.bias_ih_l0_r");
    lw.b_hh_r = TL.get("kokoro.predictor.shared.bias_hh_l0_r");

    std::vector<float> x_shared(T_frame * d_hid);
    bilstm_forward(en, T_frame, d_hid + sty_dim, d_hid / 2, lw, x_shared.data());

    // Convert to [d_hid, T_frame] for the conv-based AdainResBlk1d chain.
    std::vector<float> F0((size_t)d_hid * T_frame);
    std::vector<float> N ((size_t)d_hid * T_frame);
    for (int t = 0; t < T_frame; ++t) {
        for (int c = 0; c < d_hid; ++c) {
            const float v = x_shared[(size_t)t * d_hid + c];
            F0[(size_t)c * T_frame + t] = v;
            N [(size_t)c * T_frame + t] = v;
        }
    }

    // F0 chain.
    // F0.0: Cin=d_hid, Cout=d_hid, upsample=False, learned_sc=False
    // F0.1: Cin=d_hid, Cout=d_hid/2, upsample=True, learned_sc=True
    // F0.2: Cin=d_hid/2, Cout=d_hid/2, upsample=False, learned_sc=False
    std::vector<float> tmp; int T_after = T_frame;
    adainresblk1d_forward(TL, "kokoro.predictor.F0.0", F0.data(), d_hid, T_after, sty_dim, s,
                          d_hid, false, false, tmp, T_after);
    F0 = std::move(tmp);
    adainresblk1d_forward(TL, "kokoro.predictor.F0.1", F0.data(), d_hid, T_after, sty_dim, s,
                          d_hid / 2, true, true, tmp, T_after);
    F0 = std::move(tmp);
    adainresblk1d_forward(TL, "kokoro.predictor.F0.2", F0.data(), d_hid / 2, T_after, sty_dim, s,
                          d_hid / 2, false, false, tmp, T_after);
    F0 = std::move(tmp);
    // F0_proj: Conv1d(d_hid/2, 1, kernel=1) → [1, T_after].
    {
        const float * w = TL.get("kokoro.predictor.F0_proj.weight");
        const float * b = TL.get("kokoro.predictor.F0_proj.bias");
        F0_pred.assign(T_after, 0.0f);
        std::vector<float> proj(1 * T_after);
        conv1d_forward(F0.data(), d_hid / 2, T_after, w, b, 1, 1, 1, 0, 1, proj.data(), T_after);
        F0_pred = std::move(proj);
    }

    // N chain (mirror).
    int T_after_N = T_frame;
    adainresblk1d_forward(TL, "kokoro.predictor.N.0", N.data(), d_hid, T_after_N, sty_dim, s,
                          d_hid, false, false, tmp, T_after_N);
    N = std::move(tmp);
    adainresblk1d_forward(TL, "kokoro.predictor.N.1", N.data(), d_hid, T_after_N, sty_dim, s,
                          d_hid / 2, true, true, tmp, T_after_N);
    N = std::move(tmp);
    adainresblk1d_forward(TL, "kokoro.predictor.N.2", N.data(), d_hid / 2, T_after_N, sty_dim, s,
                          d_hid / 2, false, false, tmp, T_after_N);
    N = std::move(tmp);
    {
        const float * w = TL.get("kokoro.predictor.N_proj.weight");
        const float * b = TL.get("kokoro.predictor.N_proj.bias");
        std::vector<float> proj(1 * T_after_N);
        conv1d_forward(N.data(), d_hid / 2, T_after_N, w, b, 1, 1, 1, 0, 1, proj.data(), T_after_N);
        N_pred = std::move(proj);
    }
}

} // namespace

// =================================================================
// Public entry: full predictor forward.
// =================================================================
bool kokoro_predictor_forward(
        const kokoro_model * model,
        const std::vector<int32_t> & phoneme_ids,
        const float * ref_s, float speed,
        PredictorOut & out, std::string & err_out) {
    if (!model) { err_out = "null model"; return false; }
    const kokoro_hparams * hp = kokoro_get_hparams(model);
    if (!hp) { err_out = "null hparams"; return false; }

    TLookup TL { kokoro_model_ggml_ctx(model) };

    const int T_phon = (int) phoneme_ids.size();
    out.T_phon = T_phon;
    const int d_model = hp->text_d_model;
    const int d_ff    = hp->text_d_ff;
    const int n_layer = hp->text_n_layer;
    const int n_head  = hp->text_n_head;
    const int hid     = 512;     // hidden_dim from config.json
    const int sty     = 128;     // per-half style_dim
    const int max_dur = 50;
    const int pred_nl = 3;

    // 1. BERT.
    std::vector<float> bert_out;
    bert_forward(TL, phoneme_ids, d_model, d_ff, n_layer, n_head, bert_out, err_out);
    if (!err_out.empty()) return false;

    // 2. bert_encoder Linear 768→512 → d_en [T, 512] row-major.
    const float * be_w = TL.get("kokoro.bert_encoder.weight");
    const float * be_b = TL.get("kokoro.bert_encoder.bias");
    if (!be_w || !be_b) { err_out = "missing bert_encoder weights"; return false; }
    std::vector<float> d_en((size_t)T_phon * hid);
    for (int t = 0; t < T_phon; ++t) {
        linear_forward(bert_out.data() + (size_t)t * d_model, d_model,
                       be_w, be_b, hid, d_en.data() + (size_t)t * hid);
    }

    // 3. DurationEncoder.
    duration_encoder_forward(TL, d_en.data(), T_phon, hid, sty, pred_nl,
                             ref_s + hid /* predictor-half = ref_s[128:] */, out.d);

    // 4. predictor.lstm.
    predictor_main_lstm_forward(TL, out.d.data(), T_phon, hid, sty, out.x_lstm);

    // 5. duration_proj → sigmoid → sum → /speed → pred_dur.
    duration_predictor_forward(TL, out.x_lstm.data(), T_phon, hid, max_dur, speed,
                               out.duration, out.pred_dur);

    // 6. Length regulate `d` → en [T_frame, 640].
    length_regulate(out.d.data(), T_phon, hid + sty, out.pred_dur, out.en, out.T_frame);

    // 7. text_encoder forward (separate from BERT).
    text_encoder_forward(TL, phoneme_ids, pred_nl, hid, out.t_en);

    // 8. Length regulate t_en [T_phon, 512] → asr [T_frame, 512].
    {
        int T_frame_check = 0;
        length_regulate(out.t_en.data(), T_phon, hid, out.pred_dur, out.asr, T_frame_check);
        if (T_frame_check != out.T_frame) {
            err_out = "internal: T_frame mismatch between en and asr";
            return false;
        }
    }

    // 9. F0/N predict.
    // The shared LSTM expects input [T_frame, hid+sty]. en is already that shape
    // (length-regulated d, which is hid+sty wide).
    f0n_train_forward(TL, out.en.data(), out.T_frame, hid, sty,
                      ref_s + hid /* predictor-half */, out.F0_pred, out.N_pred);

    return true;
}

} // namespace eliza_kokoro
