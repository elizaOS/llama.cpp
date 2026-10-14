// SPDX-License-Identifier: MIT
//
// paradee-phase-lock.cpp — Paradee's parameter-free phase-locking filter.
//
// Port of training/scripts/phase_lock.py lock() (sahilmahendrakar/paradee,
// Apache-2.0, commit 9c8b4d75), the filter the Paradee-8M v1.0 graph applies
// after the decoder (export_paradee.py Lock, L = 33):
//
//   S = STFT(s), E = STFT(src)          n_fft 1024, hop 256, periodic Hann,
//                                        centre frames with reflect padding
//   voiced frame j  <=>  f0[min(j*hop / f0_hop, len-1)] > 60 Hz
//   R = S * conj(E) / max(|E|, 1e-6)
//   R~ = R smoothed over frames with hann(L+2)[1:-1] / sum (zero-padded)
//   in voiced frames, for bins with 2 kHz <= f < 8 kHz:
//       phase = angle(E) + angle(R~), magnitude = |S|
//   out = iSTFT (window-envelope normalised, length = len(s))
//
// The inverse is applied to the change only (out = s + iSTFT(delta)), as the
// exported graph does, so bins outside the band and unvoiced frames pass
// through untouched. Everything runs in double with a radix-2 FFT; the two
// real forward transforms share one complex FFT.

#include "paradee.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

namespace eliza_paradee {

namespace {

using cd = std::complex<double>;
constexpr double K_PI = 3.14159265358979323846;

// In-place iterative radix-2 FFT (n a power of two). inverse = conjugate
// twiddles, no 1/n scaling.
class Fft {
public:
    explicit Fft(int n) : n_(n), rev_((size_t) n), tw_((size_t) n / 2) {
        int bits = 0;
        while ((1 << bits) < n) ++bits;
        for (int i = 0; i < n; ++i) {
            int r = 0;
            for (int b = 0; b < bits; ++b) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
            rev_[(size_t) i] = r;
        }
        for (int k = 0; k < n / 2; ++k) {
            const double a = -2.0 * K_PI * (double) k / (double) n;
            tw_[(size_t) k] = cd(std::cos(a), std::sin(a));
        }
    }
    void run(std::vector<cd> & a, bool inverse) const {
        const int n = n_;
        for (int i = 0; i < n; ++i) {
            const int r = rev_[(size_t) i];
            if (i < r) std::swap(a[(size_t) i], a[(size_t) r]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            const int half = len >> 1, step = n / len;
            for (int i = 0; i < n; i += len) {
                for (int j = 0; j < half; ++j) {
                    cd w = tw_[(size_t) (j * step)];
                    if (inverse) w = std::conj(w);
                    const cd u = a[(size_t) (i + j)];
                    const cd v = a[(size_t) (i + j + half)] * w;
                    a[(size_t) (i + j)] = u + v;
                    a[(size_t) (i + j + half)] = u - v;
                }
            }
        }
    }
private:
    int n_;
    std::vector<int> rev_;
    std::vector<cd> tw_;
};

// unit(z) = z / |z|, with angle(0) = 0 as torch.angle defines it.
inline cd unit(cd z) {
    const double m = std::abs(z);
    return m > 0.0 ? z / m : cd(1.0, 0.0);
}

} // namespace

bool paradee_phase_lock(
        const float * s, size_t ns,
        const float * src, size_t nsrc,
        const float * f0, size_t nf0,
        const paradee_lock_params & p,
        std::vector<float> & out,
        paradee_cancel_fn cancel, void * user) {
    const size_t k = std::min(ns, nsrc);
    out.assign(s, s + k);
    const int N = p.n_fft, hop = p.hop, L = p.smooth_frames;
    if (!s || !src || !f0 || nf0 == 0 || N < 4 || (N & (N - 1)) != 0 || hop <= 0 ||
        L < 1 || p.f0_hop <= 0 || p.sample_rate <= 0) {
        return false;
    }
    const size_t pad = (size_t) N / 2;
    // torch.stft(center=True) needs reflect padding narrower than the signal.
    if (k <= pad) return true;

    // Band bins: FR = bin * sr / n_fft, lo <= FR < hi.
    const int n_bins = N / 2 + 1;
    int b_lo = n_bins, b_hi = 0;  // [b_lo, b_hi)
    for (int b = 0; b < n_bins; ++b) {
        const double fr = (double) b * (double) p.sample_rate / (double) N;
        if (fr >= p.band_lo_hz && fr < p.band_hi_hz) {
            b_lo = std::min(b_lo, b);
            b_hi = std::max(b_hi, b + 1);
        }
    }
    if (b_lo >= b_hi) return true;
    const int nb = b_hi - b_lo;

    // Reflect-padded signals.
    const size_t np_len = k + 2 * pad;
    std::vector<double> ps(np_len), pe(np_len);
    for (size_t i = 0; i < k; ++i) { ps[pad + i] = s[i]; pe[pad + i] = src[i]; }
    for (size_t i = 0; i < pad; ++i) {
        ps[pad - 1 - i] = s[i + 1];       pe[pad - 1 - i] = src[i + 1];
        ps[pad + k + i] = s[k - 2 - i];   pe[pad + k + i] = src[k - 2 - i];
    }

    // Periodic Hann window (torch.hann_window(N)).
    std::vector<double> w((size_t) N);
    for (int i = 0; i < N; ++i) w[(size_t) i] = 0.5 - 0.5 * std::cos(2.0 * K_PI * (double) i / (double) N);

    const size_t nf = 1 + k / (size_t) hop;
    Fft fft(N);

    // Forward STFTs of the band bins: S and E share one complex FFT of
    // z = s + i*e:  S[b] = (Z[b] + conj(Z[N-b])) / 2,  E[b] = (Z[b] - conj(Z[N-b])) / 2i.
    std::vector<cd> S(nf * (size_t) nb), E(nf * (size_t) nb);
    std::vector<cd> z((size_t) N);
    for (size_t j = 0; j < nf; ++j) {
        if (cancel && (j & 31) == 0 && cancel(user)) { out.clear(); return false; }
        const size_t off = j * (size_t) hop;
        for (int n = 0; n < N; ++n) {
            z[(size_t) n] = cd(ps[off + (size_t) n] * w[(size_t) n], pe[off + (size_t) n] * w[(size_t) n]);
        }
        fft.run(z, false);
        for (int b = b_lo; b < b_hi; ++b) {
            const cd zb = z[(size_t) b];
            const cd zm = std::conj(z[(size_t) ((N - b) % N)]);
            S[j * (size_t) nb + (size_t) (b - b_lo)] = (zb + zm) * 0.5;
            E[j * (size_t) nb + (size_t) (b - b_lo)] = (zb - zm) * cd(0.0, -0.5);
        }
    }

    // Voiced frames.
    std::vector<char> voiced(nf, 0);
    bool any_voiced = false;
    for (size_t j = 0; j < nf; ++j) {
        const size_t fi = std::min((j * (size_t) hop) / (size_t) p.f0_hop, nf0 - 1);
        voiced[j] = f0[fi] > p.voiced_f0_hz;
        any_voiced = any_voiced || voiced[j];
    }
    if (!any_voiced) return true;

    // R = S conj(E) / max(|E|, 1e-6), all frames (the smoothing reads
    // unvoiced neighbours too).
    std::vector<cd> R(nf * (size_t) nb);
    for (size_t i = 0; i < R.size(); ++i) {
        R[i] = S[i] * std::conj(E[i]) / std::max(std::abs(E[i]), 1e-6);
    }

    // Smoothing kernel: hann_window(L + 2) (periodic)[1:-1], normalised.
    std::vector<double> ker((size_t) L);
    double ksum = 0.0;
    for (int m = 0; m < L; ++m) {
        ker[(size_t) m] = 0.5 - 0.5 * std::cos(2.0 * K_PI * (double) (m + 1) / (double) (L + 2));
        ksum += ker[(size_t) m];
    }
    for (double & v : ker) v /= ksum;
    const int half = L / 2;

    // Window envelope sum_j w^2(n - j*hop) over the padded signal.
    std::vector<double> env(np_len, 0.0);
    for (size_t j = 0; j < nf; ++j) {
        const size_t off = j * (size_t) hop;
        for (int n = 0; n < N; ++n) env[off + (size_t) n] += w[(size_t) n] * w[(size_t) n];
    }

    // Inverse STFT of the change in voiced frames, overlap-added.
    std::vector<double> ola(np_len, 0.0);
    std::vector<cd> dz((size_t) N);
    for (size_t j = 0; j < nf; ++j) {
        if (!voiced[j]) continue;
        if (cancel && (j & 31) == 0 && cancel(user)) { out.clear(); return false; }
        std::fill(dz.begin(), dz.end(), cd(0.0, 0.0));
        for (int bi = 0; bi < nb; ++bi) {
            cd sm(0.0, 0.0);
            for (int m = 0; m < L; ++m) {
                const long jj = (long) j + m - half;
                if (jj < 0 || jj >= (long) nf) continue;
                sm += ker[(size_t) m] * R[(size_t) jj * (size_t) nb + (size_t) bi];
            }
            const cd Sv = S[j * (size_t) nb + (size_t) bi];
            const cd Ev = E[j * (size_t) nb + (size_t) bi];
            const cd locked = std::abs(Sv) * unit(Ev) * unit(sm);
            const cd d = locked - Sv;
            const int b = b_lo + bi;
            dz[(size_t) b] += d;
            if (b != 0 && b != N / 2) dz[(size_t) (N - b)] += std::conj(d);
        }
        fft.run(dz, true);
        const size_t off = j * (size_t) hop;
        for (int n = 0; n < N; ++n) {
            ola[off + (size_t) n] += dz[(size_t) n].real() / (double) N * w[(size_t) n];
        }
    }
    for (size_t i = 0; i < k; ++i) {
        const double e = env[pad + i];
        out[i] = (float) ((double) s[i] + (e > 1e-11 ? ola[pad + i] / e : 0.0));
    }
    return true;
}

} // namespace eliza_paradee
