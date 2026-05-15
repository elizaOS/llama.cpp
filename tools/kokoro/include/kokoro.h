// SPDX-License-Identifier: MIT
//
// kokoro.h — Kokoro-82M (StyleTTS-2 + iSTFTNet) TTS inference, compiled into the
// elizaOS llama.cpp fork as tools/kokoro/. This is the J2 fork-side replacement
// for the ONNX runtime path that historically drove the
// `plugin-local-inference` voice/kokoro backend.
//
// Kokoro is structurally NOT a causal language model. It is a TTS pipeline:
//
//     phonemes (int ids)
//         |--> text_encoder (Albert/BERT-like, 6 transformer layers, 768-dim)
//         |--> predictor_encoder (BERT) + duration / F0 / N predictors
//         |--> style ref_s (256-dim, side-loaded from voices/<id>.bin)
//         |--> decoder (HiFi-GAN-style upsampling + ResBlock + iSTFT)
//         |--> 24kHz PCM
//
// The arch tag `LLM_ARCH_KOKORO` (src/models/kokoro.cpp) exists for the
// K-quant publish pipeline (R8 §3.1). This header sits next to that scaffold
// and exposes a *standalone* inference entry point used by the kokoro-tts CLI
// and the `/v1/audio/speech` route. It does NOT route through llama.cpp's
// generic `llm_graph_context` — TTS does not fit that LM abstraction (no KV
// cache, no autoregressive token gen, no logit head).
//
// Quality note (J2): the from-scratch GGML port does not match the PyTorch /
// ONNX reference numerically. Production keeps the ONNX path during the
// one-release deprecation runway; this fork path is the new default once the
// quality gap closes. See `.swarm/impl/J2-kokoro-port-notes.md`.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eliza_kokoro {

// Kokoro v1.0 model hparams. These are the canonical values from
// hexgrad/Kokoro-82M's `config.json`. The GGUF reader fills them from the
// file metadata; defaults here document the v1.0 shape.
struct kokoro_hparams {
    // BERT text encoder (`bert.albert.encoder.albert_layer_groups.0.albert_layers`).
    int32_t text_n_layer        = 6;       // Albert-style 6 layers
    int32_t text_n_head         = 12;
    int32_t text_d_model        = 768;
    int32_t text_d_ff           = 2048;
    int32_t text_vocab_size     = 178;     // Kokoro phoneme vocab (espeak-ng-like + special tokens)
    int32_t text_max_pos        = 512;

    // Style vector (the `voices/<id>.bin` side-load tensor).
    int32_t style_dim           = 256;

    // Predictor MLP widths (`predictor.duration_proj`, `predictor.F0_proj`,
    // `predictor.N_proj`). These are small 1D-conv / linear heads.
    int32_t predictor_d_hidden  = 512;

    // Decoder (HiFi-GAN-ish upsampling + iSTFT vocoder head).
    int32_t decoder_d_hidden    = 512;
    int32_t decoder_n_upsample  = 5;       // 5 upsampling stages
    int32_t istft_n_fft         = 20;      // iSTFT vocoder n_fft (Kokoro v1.0 = 20)
    int32_t istft_hop_length    = 5;       // iSTFT hop (Kokoro v1.0 = 5)
    int32_t istft_win_length    = 20;      // iSTFT window

    // Output sample rate.
    int32_t sample_rate         = 24000;
};

// Loaded model handle. Opaque; owns the GGUF tensor backing.
struct kokoro_model;
struct kokoro_model_deleter { void operator()(kokoro_model *) const noexcept; };
using kokoro_model_ptr = std::unique_ptr<kokoro_model, kokoro_model_deleter>;

// Voice preset = 256-dim ref_s style vector. The `voices/<id>.bin` files in
// the HF repo are exactly `(n_positions, 1, 256)` fp32 — we expose the flat
// data + the per-position dim so the synthesis path can slice by phoneme
// count (matches kokoro-onnx upstream `voice[len(tokens)]`).
struct kokoro_voice_preset {
    std::string id;
    std::vector<float> data;   // flat fp32 buffer
    int n_positions = 0;       // outer dim
    int style_dim   = 256;     // inner dim (Kokoro v1.0 = 256)
};

// Result of one synthesis call. Owned by the caller.
struct kokoro_audio {
    std::vector<float> samples;
    int sample_rate = 24000;
};

// Status return — see kokoro_status_str() for human-readable labels.
enum kokoro_status {
    KOKORO_OK                = 0,
    KOKORO_E_INVALID_ARG     = 1,
    KOKORO_E_MISSING_TENSOR  = 2,
    KOKORO_E_LOAD_FAIL       = 3,
    KOKORO_E_VOICE_LOAD_FAIL = 4,
    KOKORO_E_OOM             = 5,
    KOKORO_E_NOT_IMPLEMENTED = 6,
    KOKORO_E_RUNTIME         = 7,
};

const char * kokoro_status_str(kokoro_status st) noexcept;

// Load a Kokoro GGUF. Returns nullptr on failure; `err_out` receives a
// diagnostic. The header MUST advertise `general.architecture == "kokoro"` or
// the load is rejected (no silent fallback).
kokoro_model_ptr kokoro_load_model(
    const std::string & gguf_path,
    std::string & err_out) noexcept;

// Load a voice preset (.bin = raw fp32 `(N, 1, style_dim)`). The preset
// itself never feeds through the GGUF — it's a runtime conditioning tensor.
kokoro_status kokoro_load_voice_preset(
    const std::string & bin_path,
    int style_dim,
    kokoro_voice_preset & out,
    std::string & err_out) noexcept;

// Phonemize an input text into Kokoro's int phoneme ids. The implementation
// uses a deterministic ASCII grapheme→phoneme mapping (no espeak-ng
// dependency). This is intentionally lossy vs the upstream phonemizer —
// quality recovery is part of the gap documented in J2-kokoro-port-notes.md.
std::vector<int32_t> kokoro_phonemize(const std::string & text);

// Synthesize a single utterance. `text` is the natural-language input,
// `voice` is the loaded ref_s preset. Output PCM lands in `out`. The
// `speed_mult` parameter scales the predicted durations (1.0 = native rate).
kokoro_status kokoro_synthesize(
    const kokoro_model * model,
    const kokoro_voice_preset & voice,
    const std::string & text,
    float speed_mult,
    kokoro_audio & out,
    std::string & err_out) noexcept;

// Returns the model's audio sample rate (24000 for v1.0).
int kokoro_sample_rate(const kokoro_model * model) noexcept;

// Diagnostic — last GGUF header values used at load time. For the
// `/v1/audio/speech` route to report which arch the live model is.
const kokoro_hparams * kokoro_get_hparams(const kokoro_model * model) noexcept;

} // namespace eliza_kokoro
