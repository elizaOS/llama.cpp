// SPDX-License-Identifier: MIT
//
// eliza-espeak-shared.h: the single owner of libespeak-ng's process-global
// state for every component in this tree that calls it (tools/kokoro and
// tools/misaki-g2p).
//
// libespeak-ng keeps its data path, loaded voice and translator in globals
// and is not thread-safe. Two components that each guard it with their own
// mutex and their own "initialised" flag race each other and can leave the
// other one running on a voice it did not select. Here there is one lock and
// one record of what libespeak-ng was initialised with:
//
//   auto lock = eliza_espeak_shared::lock();
//   if (!eliza_espeak_shared::prepare(lock, data_dir, "en-us", err)) ...;
//   espeak_TextToPhonemes(...);           // still under `lock`
//
// prepare() initialises libespeak-ng on first use and re-selects the
// caller's voice whenever another component changed it, so every caller gets
// its own voice regardless of interleaving. Per-call options (phoneme mode,
// tie character) are arguments of espeak_TextToPhonemes and need no reset.
//
// Data path: an empty `data_dir` means "whatever the process already
// initialised with, else libespeak-ng's default" (compiled-in path or
// ESPEAK_DATA_PATH); Kokoro always asks for this. A non-empty, explicitly
// configured path (misaki_g2p_open's espeak_data_dir) wins: when it differs
// from the current one libespeak-ng is re-initialised with it, and later
// default-path callers keep using it.
//
// Header-only on purpose: the state lives in a C++17 inline function's
// static, which the linker folds into one instance per program or shared
// library, so kokoro_lib and misaki_g2p share it without a CMake target
// between them (each stays buildable on its own, with or without the other)
// and without one more static archive for the mobile slices to collect.
// Include it only when the component links libespeak-ng.
#pragma once

#include <espeak-ng/speak_lib.h>

#include <mutex>
#include <string>

namespace eliza_espeak_shared {

struct State {
    std::mutex  mutex;
    bool        initialized = false;
    std::string data_dir;  // "" = libespeak-ng's default path
    std::string voice;     // voice currently selected, "" when unknown
};

inline State & state() {
    static State s;
    return s;
}

// Every libespeak-ng call in the process must be made while holding this.
inline std::unique_lock<std::mutex> lock() {
    return std::unique_lock<std::mutex>(state().mutex);
}

// With `held` from lock(): make libespeak-ng ready for `data_dir` (see the
// data-path rule above) and select `voice`. On failure sets `err`.
inline bool prepare(const std::unique_lock<std::mutex> & held, const std::string & data_dir, const char * voice,
                    std::string & err) {
    State & s = state();
    if (!held.owns_lock() || held.mutex() != &s.mutex) {
        err = "eliza_espeak_shared::prepare called without the shared eSpeak lock";
        return false;
    }
    if (!s.initialized || (!data_dir.empty() && data_dir != s.data_dir)) {
        s.initialized = false;
        s.voice.clear();
        // AUDIO_OUTPUT_SYNCHRONOUS: no audio device is opened.
        const int rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir.empty() ? nullptr : data_dir.c_str(), 0);
        if (rate <= 0) {
            err = "espeak_Initialize failed for data dir '" + data_dir + "'";
            return false;
        }
        s.initialized = true;
        s.data_dir    = data_dir;
    }
    if (s.voice != voice) {
        if (espeak_SetVoiceByName(voice) != EE_OK) {
            s.voice.clear();
            err = std::string("eSpeak NG voice ") + voice + " not found under '" + s.data_dir + "'";
            return false;
        }
        s.voice = voice;
    }
    return true;
}

}  // namespace eliza_espeak_shared
