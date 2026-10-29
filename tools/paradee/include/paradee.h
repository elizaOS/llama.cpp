// SPDX-License-Identifier: MIT
//
// paradee.h — Paradee-8M (v1.0, v1.1 and v1.2) text-to-speech, native in the
// elizaOS llama.cpp fork (tools/paradee).
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
// Paradee-8M v1.1 (sahilmahendrakar/Paradee-8M-v1.1) keeps the text side and
// the decoder's AdaIN trunk and replaces the generator and the phase lock
// with a frame-rate source-filter head:
//
//     -> decoder     : AdaIN res blocks (256) -> 128 channels at 80 fps
//     -> head        : ConvNeXt x4 (160) -> per STFT frame (n_fft 600, hop 150)
//                      gains / phase / pulse shift on a 64-harmonic sine bank
//                      from F0, gains on white and pitch-pulsed noise, plus a
//                      3 ms noise-burst path (STFT 150 / 75) -> iSTFT
//
// The GGUF's paradee.decoder.head says which graph to run ("istftnet_lock" or
// "source_filter_v2"; a file without the key is v1.0). The v1.1 head draws
// Gaussian noise: from a fixed seed (PARADEE_DEFAULT_NOISE_SEED unless the
// options say otherwise), so the same input gives the same audio.
//
// Paradee-8M v1.2 (sahilmahendrakar/Paradee-8M-v1.2) is v1.1 with four
// voices: the two learned styles become tables with one row per voice, and
// the voice picks the row. Everything else is shared. A v1.0 / v1.1 file has
// one voice. paradee_voice_count / _name / _accent / paradee_find_voice
// describe the voices; paradee_synth_options::voice (or
// paradee_stages::voice for the staged calls) selects one, and -1 is the
// file's default (paradee.voice, af_heart).
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

// Input contract of Paradee-8M (v1.0 and v1.1): at most 512 ids, the pad (0) at each end.
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

// Which decoder head a loaded model runs (from paradee.decoder.head).
enum paradee_head_kind {
    PARADEE_HEAD_ISTFTNET_LOCK = 0,   // v1.0: iSTFTNet generator + phase-locking filter
    PARADEE_HEAD_SOURCE_FILTER = 1,   // v1.1: vhead.py SourceFilterHead2
};

// Seed of the v1.1 head's noise when the caller does not choose one.
constexpr uint64_t PARADEE_DEFAULT_NOISE_SEED = 0;

// Load a Paradee GGUF written by convert_paradee_pt_to_gguf.py. The file must
// declare general.architecture == "paradee"; every hparam the graph reads is
// required. An all-F32 file is memory-mapped read-only and used in place
// (PARADEE_USE_MMAP=0 reads it into memory instead); other files are read and
// widened to F32. Returns nullptr and sets `err` on failure.
paradee_model_ptr paradee_load_model(const std::string & gguf_path, std::string & err) noexcept;

// 24000 for Paradee-8M (0 for a null model).
int paradee_sample_rate(const paradee_model * model) noexcept;

// The head the model runs (PARADEE_HEAD_ISTFTNET_LOCK for a null model) and
// the release it was converted from ("1.0", "1.1"; "" for a null model).
paradee_head_kind paradee_get_head(const paradee_model * model) noexcept;
const char * paradee_version(const paradee_model * model) noexcept;

// Release metadata. paradee_family: "paradee", or the model family a GGUF
// converted from another release of the same architecture declares
// (paradee.family; "delibes" for Delibes-8M-esES). paradee_language: "en",
// or paradee.language ("es"). paradee_phoneme_format: the phoneme spelling
// the model was trained on, "misaki" (Kokoro's English G2P) or
// paradee.phoneme_format ("espeak-es": misaki espeak.EspeakG2P('es')).
// "" for a null model.
const char * paradee_family(const paradee_model * model) noexcept;
const char * paradee_language(const paradee_model * model) noexcept;
const char * paradee_phoneme_format(const paradee_model * model) noexcept;

// The input envelope the release was validated on: at most this many ids
// per synthesis, pads included (paradee.input.validated_max_ids; 360 for
// Delibes-8M-esES; else the architecture's 512), and this speed range
// (paradee.validated_speed; [0.85, 1.15] for Delibes; else
// [PARADEE_MIN_SPEED, PARADEE_MAX_SPEED]). Advisory: synthesis still
// accepts the architecture's limits, and chunkers should split to
// validated_max_ids. 0 / false for a null model.
int paradee_validated_max_ids(const paradee_model * model) noexcept;
bool paradee_validated_speed(const paradee_model * model, float & lo, float & hi) noexcept;

// Voices: rows of the style tables, in model order. 1 for v1.0 / v1.1, 4 for
// v1.2, 0 for a null model. Names and accents are "" for an index out of
// range; the accent is the misaki dialect the voice was trained on ("en-us",
// "en-gb"), "" when the file does not say. paradee_find_voice returns -1
// for an unknown name. paradee_default_voice is the paradee.voice row (-1
// for a null model).
int paradee_voice_count(const paradee_model * model) noexcept;
int paradee_default_voice(const paradee_model * model) noexcept;
const char * paradee_voice_name(const paradee_model * model, int voice) noexcept;
const char * paradee_voice_accent(const paradee_model * model, int voice) noexcept;
int paradee_find_voice(const paradee_model * model, const std::string & name) noexcept;

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
    uint64_t noise_seed = PARADEE_DEFAULT_NOISE_SEED;   // v1.1 head only
    int voice = -1;   // style row (paradee_find_voice); -1 = paradee_default_voice. Out of range: PARADEE_E_INVALID_ARG
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
    std::vector<float> dec;         // [2*T_frame*300]    generator output before the phase lock (v1.0)
    std::vector<float> src;         // [2*T_frame*300]    harmonic source (m_source output, v1.0)
    std::vector<float> wav;         // final waveform
    // v1.1 (source-filter head) only, n = 2*T_frame:
    std::vector<float> trunk;       // [128, n]           AdaIN trunk output (channel-major)
    std::vector<float> hn;          // [n, 160]           head ConvNeXt stack after norm1
    std::vector<float> sine;        // [n*300]            sine bank (vhead.sine_bank)
    std::vector<float> main_path;   // [n*300]            the n_fft 600 path
    std::vector<float> burst_path;  // [n*300]            the noise-burst path
    std::vector<float> noise;       // [n*300]            the noise the head used
    // Input (v1.1): when set, the head uses this noise ([n*300] samples)
    // instead of drawing it from the seed. The parity test feeds the
    // reference's noise here.
    const std::vector<float> * noise_in = nullptr;
    // Input: the voice (style row) the staged calls use; -1 = the default.
    int voice = -1;
};

// paradee_synthesize_ids that also fills `stages` (when non-null; its
// noise_in, if set, replaces the v1.1 head's noise).
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
// may not be null). For a v1.1 model: trunk + head, filling stages->trunk,
// hn, sine, main_path, burst_path, noise and wav (`src` must be null; the
// noise is stages->noise_in or the default seed's).
paradee_status paradee_decode_staged(
        const paradee_model * model,
        const float * asr, int asr_dim, int T_frame,
        const float * F0, const float * N,
        const std::vector<float> * src,
        paradee_cancel_fn cancel, void * user,
        paradee_stages * stages,
        std::string & err) noexcept;

// v1.1 only: the source-filter head alone from a given trunk output
// (trunk [C, n] channel-major, F0 [n], noise [n*300]). Fills stages->hn,
// sine, main_path, burst_path and wav.
paradee_status paradee_head_staged(
        const paradee_model * model,
        const float * trunk, int channels, int n,
        const float * F0, const float * noise,
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

// nullptr for a null model and for a v1.1 model (it has no phase lock).
const paradee_lock_params * paradee_get_lock_params(const paradee_model * model) noexcept;

} // namespace eliza_paradee
