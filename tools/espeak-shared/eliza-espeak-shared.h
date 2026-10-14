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
// Initialisation never ends the process. espeak_Initialize() calls exit(1)
// when the data cannot be loaded unless espeakINITIALIZE_DONT_EXIT is set,
// and even then reports success through the sample rate. prepare() uses the
// espeak_ng_* calls instead, which return a status.
//
// Path length: libespeak-ng copies the data path into a fixed buffer
// (N_PATH_HOME: 160 bytes on POSIX unless the library was built with a larger
// -DN_PATH_HOME) and silently truncates a longer one, then falls back to its
// compiled-in default directory. prepare() reads the path libespeak-ng kept
// (espeak_Info) and fails when it is not the requested one, so a long path is
// an error naming the limit instead of a quiet load of other data. The Android
// builds of libespeak-ng use a larger N_PATH_HOME; see the eliza
// stage-elizavoice-lib script.
//
// Header-only on purpose: the state lives in a C++17 inline function's
// static, which the linker folds into one instance per program or shared
// library, so kokoro_lib and misaki_g2p share it without a CMake target
// between them (each stays buildable on its own, with or without the other)
// and without one more static archive for the mobile slices to collect.
// Include it only when the component links libespeak-ng.
#pragma once

#include <cstdio>
#include <espeak-ng/espeak_ng.h>
#include <espeak-ng/speak_lib.h>
#include <filesystem>
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

// True when libespeak-ng kept the directory it should use for `data_dir`
// whole after espeak_ng_InitializePath(data_dir): `data_dir`/espeak-ng-data
// when that exists (libespeak-ng prefers it), else `data_dir` itself. A
// truncated `data_dir`/espeak-ng-data makes libespeak-ng fall back to
// `data_dir`, which this also catches.
inline bool path_kept(const std::string & data_dir) {
    const char * kept = nullptr;
    espeak_Info(&kept);
    if (!kept) return false;
    const std::string child = data_dir + "/espeak-ng-data";
    std::error_code   ec;
    return std::string(kept) == (std::filesystem::is_directory(child, ec) ? child : data_dir);
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
        if (!data_dir.empty()) {
            std::error_code ec;
            if (!std::filesystem::is_directory(data_dir, ec)) {
                err = "eSpeak NG data directory not found: '" + data_dir + "'";
                return false;
            }
        }
        espeak_ng_InitializePath(data_dir.empty() ? nullptr : data_dir.c_str());
        if (!data_dir.empty() && !path_kept(data_dir)) {
            err = "eSpeak NG data path is " + std::to_string(data_dir.size()) +
                  " bytes, longer than this libespeak-ng build accepts (its N_PATH_HOME buffer); "
                  "use a shorter path or a libespeak-ng built with a larger -DN_PATH_HOME: '" +
                  data_dir + "'";
            return false;
        }
        espeak_ng_ERROR_CONTEXT context = nullptr;
        espeak_ng_STATUS        status  = espeak_ng_Initialize(&context);
        espeak_ng_ClearErrorContext(&context);
        if (status == ENS_OK) {
            // Synchronous mode: no audio device is opened.
            status = espeak_ng_InitializeOutput(ENOUTPUT_MODE_SYNCHRONOUS, 0, nullptr);
        }
        if (status != ENS_OK) {
            char message[256] = {};
            espeak_ng_GetStatusCodeMessage(status, message, sizeof(message));
            err = "eSpeak NG initialisation failed for data dir '" +
                  (data_dir.empty() ? std::string("<default>") : data_dir) + "': " + message;
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
