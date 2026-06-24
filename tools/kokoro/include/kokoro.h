// SPDX-License-Identifier: MIT
//
// kokoro.h — Kokoro-82M (StyleTTS-2 + iSTFTNet) TTS inference, compiled into
// the elizaOS llama.cpp fork as tools/kokoro/. This is the eliza-facing C++
// surface consumed by the fused `libelizainference` FFI
// (tools/omnivoice/src/eliza-inference-ffi.cpp, the `eliza_inference_kokoro_*`
// symbols) and by the standalone `kokoro-tts` CLI / `/v1/audio/speech` route.
//
// The forward pass behind this surface is the vendored CrispStrobe/CrispASR
// ggml StyleTTS2 implementation (src/kokoro-crispasr.cpp, C API in
// include/kokoro-crispasr.h). This header is a thin adapter facade: it keeps
// the historical `eliza_kokoro::` signatures stable so the FFI and CLI never
// change, while the implementation in src/kokoro-adapter.cpp wraps a single
// CrispASR `kokoro_context` (which fuses model + voice + scheduler buffers).
//
// Kokoro is structurally NOT a causal language model. It is a TTS pipeline:
//
//     phonemes (IPA → vocab ids)
//         |--> text_encoder (custom ALBERT BERT, 768-dim)
//         |--> predictor (duration / F0 / energy)
//         |--> style ref_s (256-dim, indexed by phoneme count from the voice pack)
//         |--> iSTFTNet decoder (HiFi-GAN upsampling + iSTFT)
//         |--> 24 kHz PCM
//
// It does NOT route through llama.cpp's generic `llm_graph_context` — TTS does
// not fit that LM abstraction (no KV cache, no autoregressive token gen, no
// logit head). The synthesis entry takes already-IPA phonemes (the fused FFI
// delivers them pre-phonemized), so the espeak/G2P front-end in CrispASR is
// compiled out — no GPL espeak link dependency in the shipped library.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eliza_kokoro {

// Kokoro v1.0 model hparams. The full topology is read from the GGUF metadata
// by the CrispASR loader; the values surfaced here are the canonical v1.0
// shape, reported via kokoro_get_hparams() for the `/v1/audio/speech` route.
struct kokoro_hparams {
    int32_t style_dim   = 256;   // ref_s style-vector width (Kokoro v1.0 = 256)
    int32_t sample_rate = 24000; // output sample rate
};

// Loaded model handle. Opaque; owns the CrispASR `kokoro_context` (model +
// voice + ggml scheduler buffers).
struct kokoro_model;
struct kokoro_model_deleter { void operator()(kokoro_model *) const noexcept; };
using kokoro_model_ptr = std::unique_ptr<kokoro_model, kokoro_model_deleter>;

// Voice handle. CrispASR fuses the voice pack into the model context, so this
// carries only the resolved voice-pack path; the actual load happens lazily on
// the first synthesize call (the FFI loads model and voice separately and only
// passes both into the synthesis call together). `id` mirrors the path for
// diagnostics, matching the legacy field the CLI set.
struct kokoro_voice_preset {
    std::string id;        // diagnostic label (== path)
    std::string pack_path; // path to the `<voice>.voice.gguf` pack
    int style_dim = 256;   // advisory; the pack metadata is authoritative
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

// Load a Kokoro GGUF (CrispASR layout: `bert.*` / `pred.*` / `dec.gen.*` /
// `plbert.*` + `tokenizer.ggml.tokens`). Returns nullptr on failure; `err_out`
// receives a diagnostic. `general.architecture == "kokoro"` is required.
kokoro_model_ptr kokoro_load_model(
    const std::string & gguf_path,
    std::string & err_out) noexcept;

// Record the voice pack to load (arch="kokoro-voice"). The CrispASR context
// owns the voice, so this only validates the path and stores it; the pack is
// loaded into the model context on the first synthesize call. `style_dim` is
// advisory (the pack metadata is authoritative).
kokoro_status kokoro_load_voice_preset(
    const std::string & pack_path,
    int style_dim,
    kokoro_voice_preset & out,
    std::string & err_out) noexcept;

// Synthesize a single utterance. `phonemes` is an already-IPA phoneme string
// in the model's vocab inventory (the fused FFI delivers pre-phonemized text).
// `speed_mult` scales predicted durations (1.0 = native rate). Output PCM
// lands in `out`.
kokoro_status kokoro_synthesize(
    const kokoro_model * model,
    const kokoro_voice_preset & voice,
    const std::string & phonemes,
    float speed_mult,
    kokoro_audio & out,
    std::string & err_out) noexcept;

// Returns the model's audio sample rate (24000 for v1.0).
int kokoro_sample_rate(const kokoro_model * model) noexcept;

// Diagnostic — hparams used at load time, for the `/v1/audio/speech` route.
const kokoro_hparams * kokoro_get_hparams(const kokoro_model * model) noexcept;

} // namespace eliza_kokoro
