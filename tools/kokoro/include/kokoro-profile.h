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

namespace eliza_kokoro {

inline bool kokoro_profile_enabled() {
    static const bool on = [] {
        const char * v = std::getenv("KOKORO_PROFILE");
        return v && *v && *v != '0';
    }();
    return on;
}

// RAII phase timer: logs "<label> <ms>" to stderr on scope exit when
// KOKORO_PROFILE is set.
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
        std::fprintf(stderr, "[kokoro-profile] %-28s %10.1f ms\n", label_, ms);
    }

  private:
    const char * label_;
    std::chrono::steady_clock::time_point t0_;
};

} // namespace eliza_kokoro
