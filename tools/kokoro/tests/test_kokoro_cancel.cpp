// SPDX-License-Identifier: MIT
//
// Kokoro synthesis cancellation. The weight-free checks always run. With
// KOKORO_TEST_MODEL (GGUF) and KOKORO_TEST_VOICE (voice .bin) set, a real
// synthesis is cancelled at each stage boundary in turn.

#include "kokoro.h"
#include "kokoro-decoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

// Returns true from the `cancel_at`-th poll onwards; counts every poll.
struct CancelAfter {
    int cancel_at;
    int polls;
};

bool cancel_after(void * user_data) {
    CancelAfter * state = static_cast<CancelAfter *>(user_data);
    ++state->polls;
    return state->cancel_at > 0 && state->polls >= state->cancel_at;
}

void check_without_weights() {
    using namespace eliza_kokoro;
    check(std::strcmp(kokoro_status_str(KOKORO_E_CANCELLED), "cancelled") == 0,
          "KOKORO_E_CANCELLED has a status string");

    // Argument validation runs before the first poll: a null model is an
    // invalid argument, not a cancellation, and the callback is never called.
    CancelAfter state{1, 0};
    kokoro_voice_preset voice;
    kokoro_audio audio;
    std::string err;
    check(kokoro_synthesize(nullptr, voice, "hi", 1.0f, audio, err,
                            cancel_after, &state) == KOKORO_E_INVALID_ARG,
          "null model is reported as an invalid argument");
    check(kokoro_synthesize_ipa(nullptr, voice, "hi", 1.0f, audio, err,
                                cancel_after, &state) == KOKORO_E_INVALID_ARG,
          "null model is reported as an invalid argument (ipa)");
    check(state.polls == 0, "cancel callback is not polled before validation");

    std::vector<float> pcm;
    check(!kokoro_decoder_forward(nullptr, nullptr, 1, nullptr, nullptr, nullptr,
                                  pcm, err, cancel_after, &state),
          "decoder rejects a null model");
}

int check_with_weights(const char * model_path, const char * voice_path) {
    using namespace eliza_kokoro;
    std::string err;
    kokoro_model_ptr model = kokoro_load_model(model_path, err);
    if (!model) {
        std::fprintf(stderr, "FAIL: load %s: %s\n", model_path, err.c_str());
        return 1;
    }
    const kokoro_hparams * hp = kokoro_get_hparams(model.get());
    kokoro_voice_preset voice;
    if (kokoro_load_voice_preset(voice_path, hp ? hp->style_dim : 256, voice, err) != KOKORO_OK) {
        std::fprintf(stderr, "FAIL: voice %s: %s\n", voice_path, err.c_str());
        return 1;
    }

    // Uncancelled: count the polls a full synthesis makes.
    CancelAfter full{0, 0};
    kokoro_audio audio;
    check(kokoro_synthesize(model.get(), voice, "Hello there.", 1.0f, audio, err,
                            cancel_after, &full) == KOKORO_OK,
          "uncancelled synthesis succeeds");
    check(!audio.samples.empty(), "uncancelled synthesis produces audio");
    check(full.polls >= 4, "synthesis polls at every stage boundary");

    // Cancel at each poll in turn: every one stops the call with no audio.
    for (int at = 1; at <= full.polls; ++at) {
        CancelAfter state{at, 0};
        kokoro_audio cancelled_audio;
        const kokoro_status st = kokoro_synthesize(
            model.get(), voice, "Hello there.", 1.0f, cancelled_audio, err,
            cancel_after, &state);
        check(st == KOKORO_E_CANCELLED, "cancel at a stage boundary returns KOKORO_E_CANCELLED");
        check(cancelled_audio.samples.empty(), "cancelled synthesis returns no audio");
        check(state.polls <= at + 1, "synthesis stops at the poll that cancelled it");
    }
    return 0;
}

} // namespace

int main() {
    check_without_weights();
    const char * model_path = std::getenv("KOKORO_TEST_MODEL");
    const char * voice_path = std::getenv("KOKORO_TEST_VOICE");
    if (model_path && *model_path && voice_path && *voice_path) {
        failures += check_with_weights(model_path, voice_path);
    } else {
        std::printf("test-kokoro-cancel: KOKORO_TEST_MODEL/KOKORO_TEST_VOICE unset; "
                    "skipping the weight-backed stage checks\n");
    }
    return failures == 0 ? 0 : 1;
}
