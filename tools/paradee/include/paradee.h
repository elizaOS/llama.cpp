// SPDX-License-Identifier: MIT
//
// paradee.h — Paradee-8M v1.0 text-to-speech, native in the elizaOS llama.cpp
// fork (tools/paradee).
//
// Paradee (sahilmahendrakar/Paradee-8M-v1.0, Apache-2.0) is Kokoro-82M's
// StyleTTS-2 + iSTFTNet code at smaller widths, distilled to the single
// af_heart voice. 8.07M parameters, 24 kHz output. The upstream runtime is an
// ONNX graph; this port runs the PyTorch weights (converted by
// convert_paradee_pt_to_gguf.py) through the generalised tools/kokoro layers
// plus a native implementation of the model's phase-locking filter. No ONNX.
//
//   misaki phonemes (UTF-8)
//     -> paradee_phonemes_to_ids : [0] + vocab ids + [0], at most 512 ids
//     -> text side   : ALBERT (6 x 256) -> duration encoder -> durations
//                      (round(dur / speed), min 1) -> F0 / N curves;
//                      TextEncoder -> asr_proj MLP (192 -> 512)
//     -> decoder     : AdaIN res blocks (256) -> iSTFTNet generator (128 -> 64 -> 32)
//                      with a deterministic harmonic source (no random phase,
//                      no additive noise, as in tools/kokoro)
//     -> phase lock  : in voiced frames, the 2-8 kHz phase follows the
//                      harmonic source (STFT 1024 / hop 256, 33-frame smoothing)
//     -> 24 kHz mono float PCM
//
// Input phonemes must be misaki's spelling (Kokoro's G2P); eSpeak IPA must be
// converted to misaki spelling first.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eliza_paradee {

struct paradee_model;
struct paradee_model_deleter { void operator()(paradee_model *) const noexcept; };
using paradee_model_ptr = std::unique_ptr<paradee_model, paradee_model_deleter>;

// Input contract of Paradee-8M v1.0: at most 512 ids, the pad (0) at each end.
constexpr int PARADEE_MAX_INPUT_IDS = 512;
constexpr int PARADEE_PAD_ID = 0;

// Accepted speed range. Durations scale with 1/speed, so a tiny speed would
// make the frame count (and every decoder buffer) arbitrarily large.
constexpr float PARADEE_MIN_SPEED = 0.25f;
constexpr float PARADEE_MAX_SPEED = 4.0f;

enum paradee_status {
    PARADEE_OK               = 0,
    PARADEE_E_INVALID_ARG    = 1,  // null model, bad ids, bad speed, malformed input
    PARADEE_E_UNKNOWN_SYMBOL = 2,  // a phoneme outside the model vocabulary
    PARADEE_E_TOO_LONG       = 3,  // more than 510 phonemes (512 ids with the pads)
    PARADEE_E_RUNTIME        = 4,  // forward pass failed
    PARADEE_CANCELLED        = 5,  // the cancel callback returned true
    PARADEE_E_OOM            = 6,  // allocation failed
    PARADEE_E_CANCELLED      = PARADEE_CANCELLED,  // alias
};

const char * paradee_status_str(paradee_status st) noexcept;

// Load a Paradee GGUF written by convert_paradee_pt_to_gguf.py. The file must
// declare general.architecture == "paradee"; every hparam the graph reads is
// required. Returns nullptr and sets `err` on failure.
paradee_model_ptr paradee_load_model(const std::string & gguf_path, std::string & err) noexcept;

// 24000 for Paradee-8M v1.0 (0 for a null model).
int paradee_sample_rate(const paradee_model * model) noexcept;

// Map a misaki phoneme string (UTF-8, one vocabulary symbol per code point)
// to model input ids: [0] + ids + [0]. Rejects empty input, invalid UTF-8,
// any symbol outside the vocabulary (PARADEE_E_UNKNOWN_SYMBOL, the code point
// is named in `err`) and more than 510 symbols (PARADEE_E_TOO_LONG). No
// symbol is dropped silently.
paradee_status paradee_phonemes_to_ids(
        const paradee_model * model,
        const std::string & misaki_utf8,
        std::vector<int32_t> & ids,
        std::string & err) noexcept;

using paradee_cancel_fn = bool (*)(void * user);
using paradee_cancel_cb = paradee_cancel_fn;   // alias

// The same mapping without the pads and without the length limit, for
// callers that split long phoneme strings into model-sized chunks: ids are
// the vocabulary ids of every symbol, in order. Rejects empty input, invalid
// UTF-8 and symbols outside the vocabulary exactly like
// paradee_phonemes_to_ids.
paradee_status paradee_phonemes_to_symbol_ids(
        const paradee_model * model,
        const std::string & misaki_utf8,
        std::vector<int32_t> & ids,
        std::string & err) noexcept;

// The same unpadded, unlimited mapping, but symbols outside the vocabulary
// are skipped (and counted in `n_dropped`) instead of rejected, as Kokoro and
// misaki do; empty input (or input that is all dropped) gives no ids and
// PARADEE_OK. Invalid UTF-8 is still rejected. This is the libelizainference
// contract ("characters outside the model vocab are dropped").
paradee_status paradee_phonemes_to_symbol_ids_lenient(
        const paradee_model * model,
        const std::string & misaki_utf8,
        std::vector<int32_t> & ids,
        size_t & n_dropped,
        std::string & err) noexcept;

// Synthesize from input ids (with the pad at each end, 3..512 ids, every id
// in [0, n_token)). `speed` finite and in [PARADEE_MIN_SPEED,
// PARADEE_MAX_SPEED] (1.0 = native rate; larger is faster). `cancel` (optional) is polled between stages and inside the
// generator loops; when it returns true the call returns PARADEE_CANCELLED
// with `pcm` empty. Calls on one model are serialised.
paradee_status paradee_synthesize_ids(
        const paradee_model * model,
        const int32_t * ids, size_t n,
        float speed,
        paradee_cancel_fn cancel, void * user,
        std::vector<float> & pcm,
        std::string & err) noexcept;

// Convenience form of paradee_synthesize_ids for callers that keep options
// and audio in structs.
struct paradee_synth_options {
    float speed = 1.0f;
    paradee_cancel_fn cancel = nullptr;
    void * cancel_user_data = nullptr;
};

struct paradee_audio {
    std::vector<float> samples;
    int sample_rate = 24000;
};

paradee_status paradee_synthesize(
        const paradee_model * model,
        const std::vector<int32_t> & input_ids,
        const paradee_synth_options & opts,
        paradee_audio & out,
        std::string & err) noexcept;

// ---------------------------------------------------------------------------
// Parity / diagnostics. Stage outputs of one synthesis, in the reference
// dump's layouts (see tools/dump_paradee_reference.py).
// ---------------------------------------------------------------------------
struct paradee_stages {
    int T_phon = 0;                 // number of input ids
    int T_frame = 0;                // sum of the integer durations
    int d_model = 0, hid = 0, sty = 0, asr_dim = 0;
    std::vector<float> bert;        // [T_phon, d_model]  ALBERT last hidden state
    std::vector<float> d;           // [T_phon, hid+sty]  duration-encoder output
    std::vector<float> dur;         // [T_phon]           sigmoid-sum durations / speed
    std::vector<int32_t> dur_int;   // [T_phon]           clamp(round(dur), min 1)
    std::vector<float> F0;          // [2*T_frame]
    std::vector<float> N;           // [2*T_frame]
    std::vector<float> asr;         // [asr_dim, T_frame] aligned asr features (channel-major)
    std::vector<float> dec;         // [2*T_frame*300]    generator output before the phase lock
    std::vector<float> src;         // [2*T_frame*300]    harmonic source (m_source output)
    std::vector<float> wav;         // final waveform
};

// paradee_synthesize_ids that also fills `stages` (when non-null).
paradee_status paradee_synthesize_ids_staged(
        const paradee_model * model,
        const int32_t * ids, size_t n,
        float speed,
        paradee_cancel_fn cancel, void * user,
        std::vector<float> & pcm,
        std::string & err,
        paradee_stages * stages) noexcept;

// Decoder + generator + phase lock from given text-side outputs (the parity
// gate feeds the reference's asr / F0 / N here to test the acoustic half on
// its own). asr is [asr_dim, T_frame] channel-major, F0 and N are
// [2*T_frame]. `src` (optional, [2*T_frame*300]) replaces the harmonic
// source the generator would compute. Fills stages->dec / src / wav (stages
// may not be null).
paradee_status paradee_decode_staged(
        const paradee_model * model,
        const float * asr, int asr_dim, int T_frame,
        const float * F0, const float * N,
        const std::vector<float> * src,
        paradee_cancel_fn cancel, void * user,
        paradee_stages * stages,
        std::string & err) noexcept;

// Phase-locking filter parameters (read from the GGUF; defaults are v1.0).
struct paradee_lock_params {
    int n_fft = 1024;
    int hop = 256;
    float band_lo_hz = 2000.0f;
    float band_hi_hz = 8000.0f;
    int smooth_frames = 33;         // L
    float voiced_f0_hz = 60.0f;
    int f0_hop = 300;               // samples per F0 value
    int sample_rate = 24000;
};

// The filter alone (training/scripts/phase_lock.py lock()): s = generator
// output, src = its harmonic source, f0 = the F0 curve at f0_hop samples per
// value. out has min(ns, nsrc) samples. Returns false with out = s when the
// parameters are invalid, and false with out empty when `cancel` stopped it.
// Signals no longer than n_fft/2 pass through unchanged (as torch.stft
// cannot reflect-pad them).
bool paradee_phase_lock(
        const float * s, size_t ns,
        const float * src, size_t nsrc,
        const float * f0, size_t nf0,
        const paradee_lock_params & p,
        std::vector<float> & out,
        paradee_cancel_fn cancel = nullptr, void * user = nullptr);

const paradee_lock_params * paradee_get_lock_params(const paradee_model * model) noexcept;

} // namespace eliza_paradee
