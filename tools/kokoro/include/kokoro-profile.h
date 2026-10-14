// SPDX-License-Identifier: MIT
//
// kokoro-profile.h — opt-in wall-clock phase timing for the Kokoro synthesis
// pipeline. Enabled by setting KOKORO_PROFILE=1 in the environment; otherwise
// zero output (the steady_clock reads are negligible next to the phases they
// wrap). Used to attribute RTF between the predictor, decoder front, and the
// iSTFTNet generator stages.

#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace eliza_kokoro {

inline bool kokoro_profile_enabled() {
    static const bool on = [] {
        const char * v = std::getenv("KOKORO_PROFILE");
        return v && *v && *v != '0';
    }();
    return on;
}

inline void kokoro_profile_emit(const char * label, double ms) {
#if defined(__ANDROID__)
    __android_log_print(ANDROID_LOG_INFO, "KokoroProfile", "%s %.1f ms", label, ms);
#else
    std::fprintf(stderr, "[kokoro-profile] %-28s %10.1f ms\n", label, ms);
#endif
}

// Android sends phases to logcat because app-process stderr is not observable;
// other hosts use stderr so CLI benchmarks remain pipe-friendly.
class kokoro_phase_timer {
  public:
    explicit kokoro_phase_timer(const char * label)
        : label_(label), t0_(std::chrono::steady_clock::now()) {}

    kokoro_phase_timer(const kokoro_phase_timer &) = delete;
    kokoro_phase_timer & operator=(const kokoro_phase_timer &) = delete;

    ~kokoro_phase_timer() {
        if (!kokoro_profile_enabled()) return;
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0_).count();
        kokoro_profile_emit(label_, ms);
    }

  private:
    const char * label_;
    std::chrono::steady_clock::time_point t0_;
};

} // namespace eliza_kokoro
