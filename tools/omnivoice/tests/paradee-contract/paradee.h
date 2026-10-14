// tools/omnivoice/tests/paradee-contract/paradee.h
//
// Test double for the public Paradee header (tools/paradee/include/paradee.h).
// The FFI bridge in src/eliza-inference-ffi.cpp is written against exactly
// these declarations. The FFI unit test (tests/ffi-paradee-asr.cpp) compiles
// the bridge against this header and the scripted fake in fake-paradee.cpp,
// so cancel / race / evict-reload behaviour is tested without model weights.
//
// Keep this file in sync with tools/paradee/include/paradee.h. If the real
// header changes a signature used by the bridge, change this copy too.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace eliza_paradee {

// Model input limit, including the 0 pad token at each end.
constexpr int PARADEE_MAX_INPUT_IDS = 512;
constexpr float PARADEE_MIN_SPEED = 0.25f;
constexpr float PARADEE_MAX_SPEED = 4.0f;

struct paradee_model;
struct paradee_model_deleter { void operator()(paradee_model *) const noexcept; };
using paradee_model_ptr = std::unique_ptr<paradee_model, paradee_model_deleter>;

struct paradee_audio {
    std::vector<float> samples;
    int sample_rate = 24000;
};

enum paradee_status {
    PARADEE_OK               = 0,
    PARADEE_E_INVALID_ARG    = 1,
    PARADEE_E_UNKNOWN_SYMBOL = 2,
    PARADEE_E_TOO_LONG       = 3,
    PARADEE_E_RUNTIME        = 4,
    PARADEE_CANCELLED        = 5,
    PARADEE_E_OOM            = 6,
    PARADEE_E_CANCELLED      = PARADEE_CANCELLED,
};

// Polled by the synthesis graph between stages. Returning true stops the
// forward pass and makes paradee_synthesize return PARADEE_E_CANCELLED.
using paradee_cancel_cb = bool (*)(void * user_data);

struct paradee_synth_options {
    float             speed            = 1.0f;
    paradee_cancel_cb cancel           = nullptr;
    void *            cancel_user_data = nullptr;
};

paradee_model_ptr paradee_load_model(const std::string & gguf_path, std::string & err_out) noexcept;

int paradee_sample_rate(const paradee_model * model) noexcept;

// Map a misaki phoneme string to model vocab ids (no pad tokens added, no
// length limit). Characters outside the vocab are dropped and counted, as
// kokoro/misaki do; empty input gives no ids and PARADEE_OK.
paradee_status paradee_phonemes_to_symbol_ids_lenient(
    const paradee_model * model,
    const std::string & misaki_phonemes,
    std::vector<int32_t> & ids_out,
    size_t & n_dropped,
    std::string & err_out) noexcept;

// Run one forward pass. `input_ids` is the full model input including the
// leading and trailing 0 pad token, at most PARADEE_MAX_INPUT_IDS entries.
paradee_status paradee_synthesize(
    const paradee_model * model,
    const std::vector<int32_t> & input_ids,
    const paradee_synth_options & opts,
    paradee_audio & out,
    std::string & err_out) noexcept;

} // namespace eliza_paradee
