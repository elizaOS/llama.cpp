// SPDX-License-Identifier: MIT
//
// test_kokoro_istft.cpp — sanity checks for the overlap-add inverse STFT.
//
// The test builds a deterministic magnitude+phase spectrogram, runs istft_hann
// against it, and verifies (a) the output length matches the contract and
// (b) the produced PCM is non-blank (max amplitude > 1e-4).

#include "kokoro-istft.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    using namespace eliza_kokoro;

    const int n_fft      = 20;
    const int hop_length = 5;
    const int win_length = 20;
    const int n_frames   = 96;
    const int F          = n_fft / 2 + 1;

    std::vector<float> mag((size_t) (F * n_frames), 0.0f);
    std::vector<float> phase((size_t) (F * n_frames), 0.0f);

    // Energy in bin 3 across all frames — should produce a sinusoid-ish output.
    for (int t = 0; t < n_frames; ++t) {
        mag[(size_t) (3 * n_frames + t)] = 0.5f;
        phase[(size_t) (3 * n_frames + t)] =
            2.0f * 3.14159265f * (float) t / 10.0f;
    }

    std::vector<float> out;
    istft_hann(mag, phase, n_fft, hop_length, win_length, n_frames, out);

    const int expected_len = (n_frames - 1) * hop_length + win_length;
    assert((int) out.size() == expected_len);

    float peak = 0.0f;
    for (float v : out) {
        const float a = std::fabs(v);
        if (a > peak) peak = a;
    }
    if (!(peak > 1e-4f)) {
        std::fprintf(stderr, "istft peak too low: %.6f\n", peak);
        return 1;
    }

    std::printf("test_kokoro_istft: OK (peak=%.4f, len=%d)\n", peak, expected_len);
    return 0;
}
