// SPDX-License-Identifier: MIT
//
// kokoro-istft.cpp — CPU-side inverse STFT with a Hann window, matching the
// librosa.istft reference (center=False, dtype=float32). See kokoro-istft.h
// for the contract.

#include "kokoro-istft.h"

#include <cmath>
#include <cstddef>
#include <vector>

namespace eliza_kokoro {

namespace {

// MSVC does not define M_PI by default (it's a POSIX/GNU extension behind
// _USE_MATH_DEFINES). Declare a local constant so the Windows-MSVC builds
// compile without depending on the math.h extension. Mirrors the same
// pattern used in ggml/src/ggml-cpu/ops.cpp for the ISTFT op.
static constexpr double K_PI = 3.14159265358979323846;

// Build a periodic Hann window of length N. Matches numpy.hanning's
// "symmetric=False" convention used by librosa.istft.
static std::vector<float> hann_window(int n) {
    std::vector<float> w((size_t) n);
    const double scale = 2.0 * K_PI / (double) n;
    for (int i = 0; i < n; ++i) {
        w[(size_t) i] = (float) (0.5 - 0.5 * std::cos(scale * (double) i));
    }
    return w;
}

// Inverse 1D real-output DFT for one frame. Direct O(N^2) — fine for
// Kokoro's n_fft=20.
//
//   x[t] = sum_{f=0}^{n_fft/2}  re[f] * cos(2*pi*f*t/n_fft) - im[f] * sin(2*pi*f*t/n_fft)
//        + sum_{f=1}^{n_fft/2-1} conj-mirror contribution
//
// Since the input is a hermitian-symmetric spectrum (real iSTFT), we
// reconstruct using the upper half implicitly. Output length = n_fft.
static void irdft_frame(
        const float * re,   // [F]
        const float * im,   // [F]
        int n_fft,
        float * out) {     // [n_fft]
    const int F = n_fft / 2 + 1;
    const double inv_n = 1.0 / (double) n_fft;
    for (int t = 0; t < n_fft; ++t) {
        double acc = re[0];                                  // DC
        if ((n_fft & 1) == 0) {
            // Nyquist bin contributes with cos(pi * t) = (-1)^t
            const double sign = (t & 1) ? -1.0 : 1.0;
            acc += sign * re[F - 1];
        }
        for (int f = 1; f < F - ((n_fft & 1) == 0 ? 1 : 0); ++f) {
            const double angle = 2.0 * K_PI * (double) f * (double) t * inv_n;
            acc += 2.0 * (re[f] * std::cos(angle) - im[f] * std::sin(angle));
        }
        out[t] = (float) (acc * inv_n);
    }
}

} // namespace

void istft_hann(
        const std::vector<float> & mag,
        const std::vector<float> & phase,
        int n_fft,
        int hop_length,
        int win_length,
        int n_frames,
        std::vector<float> & out) {

    const int F = n_fft / 2 + 1;
    const int n_out = (n_frames - 1) * hop_length + win_length;

    out.assign((size_t) n_out, 0.0f);
    std::vector<float> norm((size_t) n_out, 0.0f);

    const std::vector<float> window = hann_window(win_length);
    std::vector<float> re((size_t) F), im((size_t) F), frame((size_t) n_fft);

    for (int t = 0; t < n_frames; ++t) {
        // Pack real/imag from mag,phase polar form (column-major frequency-bin).
        for (int f = 0; f < F; ++f) {
            const float m = mag[(size_t) (f * n_frames + t)];
            const float p = phase[(size_t) (f * n_frames + t)];
            re[(size_t) f] = m * std::cos(p);
            im[(size_t) f] = m * std::sin(p);
        }
        irdft_frame(re.data(), im.data(), n_fft, frame.data());

        const int off = t * hop_length;
        // Apply window + accumulate.
        for (int k = 0; k < win_length; ++k) {
            const int idx = off + k;
            if (idx >= n_out) break;
            const float w = window[(size_t) k];
            out[(size_t) idx]  += frame[(size_t) (k % n_fft)] * w;
            norm[(size_t) idx] += w * w;
        }
    }

    // Normalize by the overlap-add window energy. Skip cells with zero
    // accumulated energy (boundary samples) to avoid division by zero.
    for (int i = 0; i < n_out; ++i) {
        if (norm[(size_t) i] > 1e-8f) {
            out[(size_t) i] /= norm[(size_t) i];
        }
    }
}

} // namespace eliza_kokoro
