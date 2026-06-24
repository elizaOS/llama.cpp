// SPDX-License-Identifier: MIT
//
// kokoro-adapter.cpp — the `eliza_kokoro::` facade over the vendored
// CrispStrobe/CrispASR Kokoro/StyleTTS2 forward (src/kokoro-crispasr.cpp,
// C API include/kokoro-crispasr.h).
//
// Why an adapter: the fused FFI (tools/omnivoice/src/eliza-inference-ffi.cpp)
// and the kokoro-tts CLI drive Kokoro through the `eliza_kokoro::` C++ API,
// which splits model + voice into two objects and calls
// `kokoro_synthesize(model, voice, phonemes, ...)`. CrispASR instead fuses the
// model, the voice pack, and the ggml scheduler buffers into one
// `kokoro_context`, with `kokoro_load_voice_pack(ctx, path)` mutating that
// context. This file maps the split API onto the fused context without
// changing the FFI / ABI: `kokoro_model` wraps the CrispASR ctx, the voice
// preset carries only the resolved pack path, and the pack is loaded lazily on
// the first synthesize call (the only place that sees both model and voice).
//
// The synthesis entry routes to CrispASR's `kokoro_synthesize_phonemes` (NOT
// `kokoro_synthesize`): the FFI delivers already-IPA phonemes, so the espeak /
// G2P front-end is skipped entirely and never linked (no GPL dependency).

#include "kokoro.h"
#include "kokoro-crispasr.h"

#include <cstdint>
#include <cstring>
#include <string>

namespace eliza_kokoro {

namespace {

// Decide whether `s` is already an IPA / misaki phoneme string (skip the G2P)
// or raw text to be phonemized. Phoneme strings carry codepoints that never
// occur in normal English orthography: the stress marks ˈ (U+02C8) / ˌ (U+02CC),
// the length mark ː (U+02D0), and IPA letters in the U+0250–U+02AF block
// (ə ɹ ʃ ʤ ɛ ɪ ʊ ŋ …). Raw English (even with misaki capitals A/O/W as plain
// letters) contains none of these, so any hit means "already phonemes".
bool looks_like_phonemes(const std::string & s) {
    const unsigned char * p = reinterpret_cast<const unsigned char *>(s.data());
    const size_t n = s.size();
    for (size_t i = 0; i + 1 < n; i++) {
        const unsigned char c0 = p[i];
        const unsigned char c1 = p[i + 1];
        // 2-byte UTF-8: lead 0xC0..0xDF. Decode the codepoint.
        if (c0 >= 0xC0 && c0 <= 0xDF && (c1 & 0xC0) == 0x80) {
            const uint32_t cp = ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)(c1 & 0x3F);
            // IPA Extensions (U+0250–U+02AF) + Spacing Modifier Letters used by
            // the kokoro vocab (ˈ U+02C8, ˌ U+02CC, ː U+02D0, ʰ U+02B0, ʲ U+02B2).
            if ((cp >= 0x0250 && cp <= 0x02AF) || cp == 0x02C8 || cp == 0x02CC ||
                cp == 0x02D0 || cp == 0x02B0 || cp == 0x02B2)
                return true;
        }
    }
    return false;
}

} // namespace

// Opaque model handle: owns the CrispASR context plus the path of the voice
// pack currently loaded into it (so repeated synth calls with the same voice
// don't reload the pack). `loaded_voice` is mutable because synthesis is a
// logically-const operation that may lazily load the pack into the ctx.
struct kokoro_model {
    ::kokoro_context * ctx = nullptr;
    kokoro_hparams     hparams{};
    mutable std::string loaded_voice; // pack path currently armed in ctx
};

void kokoro_model_deleter::operator()(kokoro_model * m) const noexcept {
    if (!m) return;
    if (m->ctx) ::kokoro_free(m->ctx);
    delete m;
}

const char * kokoro_status_str(kokoro_status st) noexcept {
    switch (st) {
        case KOKORO_OK:                return "ok";
        case KOKORO_E_INVALID_ARG:     return "invalid-arg";
        case KOKORO_E_MISSING_TENSOR:  return "missing-tensor";
        case KOKORO_E_LOAD_FAIL:       return "load-fail";
        case KOKORO_E_VOICE_LOAD_FAIL: return "voice-load-fail";
        case KOKORO_E_OOM:             return "oom";
        case KOKORO_E_NOT_IMPLEMENTED: return "not-implemented";
        case KOKORO_E_RUNTIME:         return "runtime-error";
    }
    return "unknown";
}

kokoro_model_ptr kokoro_load_model(
    const std::string & gguf_path,
    std::string & err_out) noexcept {
    ::kokoro_context_params params = ::kokoro_context_default_params();
    ::kokoro_context * ctx = ::kokoro_init_from_file(gguf_path.c_str(), params);
    if (!ctx) {
        err_out = "kokoro_init_from_file failed for " + gguf_path;
        return kokoro_model_ptr{};
    }
    auto * m = new kokoro_model();
    m->ctx = ctx;
    // hparams surfaced for the /v1/audio/speech diagnostic. CrispASR is fixed
    // at 24 kHz / 256-dim style; the GGUF metadata drives the actual topology.
    m->hparams.style_dim   = 256;
    m->hparams.sample_rate = 24000;
    return kokoro_model_ptr{ m };
}

kokoro_status kokoro_load_voice_preset(
    const std::string & pack_path,
    int style_dim,
    kokoro_voice_preset & out,
    std::string & err_out) noexcept {
    if (pack_path.empty()) {
        err_out = "empty voice pack path";
        return KOKORO_E_INVALID_ARG;
    }
    // CrispASR owns the voice inside the model context; defer the actual load
    // to the first synthesize call (which is the only call that has both the
    // model and the voice). Here we just record the path.
    out.id        = pack_path;
    out.pack_path = pack_path;
    out.style_dim = style_dim > 0 ? style_dim : 256;
    return KOKORO_OK;
}

kokoro_status kokoro_synthesize(
    const kokoro_model * model,
    const kokoro_voice_preset & voice,
    const std::string & phonemes,
    float speed_mult,
    kokoro_audio & out,
    std::string & err_out) noexcept {
    if (!model || !model->ctx) {
        err_out = "null model";
        return KOKORO_E_INVALID_ARG;
    }
    if (voice.pack_path.empty()) {
        err_out = "no voice pack armed";
        return KOKORO_E_INVALID_ARG;
    }
    if (phonemes.empty()) {
        err_out = "empty phoneme string";
        return KOKORO_E_INVALID_ARG;
    }

    // Lazily load the voice pack into the ctx on first use / on voice change.
    if (model->loaded_voice != voice.pack_path) {
        if (::kokoro_load_voice_pack(model->ctx, voice.pack_path.c_str()) != 0) {
            err_out = "kokoro_load_voice_pack failed for " + voice.pack_path;
            return KOKORO_E_VOICE_LOAD_FAIL;
        }
        model->loaded_voice = voice.pack_path;
    }

    // speed_mult scales durations (1.0 = native); CrispASR's length_scale is
    // the inverse (>1.0 = slower / longer), so length_scale = 1 / speed.
    const float speed = speed_mult > 0.0f ? speed_mult : 1.0f;
    ::kokoro_set_length_scale(model->ctx, 1.0f / speed);

    // The `phonemes` argument is overloaded: it may be raw English text or an
    // already-IPA/misaki phoneme string. Raw text goes through CrispASR's text
    // entry, which phonemizes internally (built-in LTS+CMUdict G2P → IPA →
    // misaki normalization → vocab ids). The misaki rewrite is mandatory:
    // Kokoro-82M was trained on hexgrad/misaki's phoneme inventory
    // (ʤ/ʧ/A/O/W/I/Y, no ː), not raw espeak/CMU IPA, so a raw-IPA string would
    // tokenize to the wrong embeddings and the model would say the wrong words.
    // A string that already carries IPA stress/length marks or IPA letters is
    // taken verbatim (skips G2P) so pre-phonemized callers still work. Language
    // defaults to en-us in the CrispASR context.
    int n_samples = 0;
    float * pcm = looks_like_phonemes(phonemes)
        ? ::kokoro_synthesize_phonemes(model->ctx, phonemes.c_str(), &n_samples)
        : ::kokoro_synthesize(model->ctx, phonemes.c_str(), &n_samples);
    if (!pcm || n_samples <= 0) {
        if (pcm) ::kokoro_pcm_free(pcm);
        err_out = "kokoro_synthesize produced no audio";
        return KOKORO_E_RUNTIME;
    }

    out.samples.assign(pcm, pcm + n_samples);
    out.sample_rate = 24000;
    ::kokoro_pcm_free(pcm);
    return KOKORO_OK;
}

int kokoro_sample_rate(const kokoro_model * model) noexcept {
    (void) model;
    return 24000;
}

const kokoro_hparams * kokoro_get_hparams(const kokoro_model * model) noexcept {
    return model ? &model->hparams : nullptr;
}

} // namespace eliza_kokoro
