// src/core/fft.h — radix-2 Cooley-Tukey FFT shared by the granite family.
//
// Both `granite_speech.cpp` and `granite_nle.cpp` historically owned a
// byte-identical copy of the same recursive radix-2 FFT plus a
// const-input wrapper that satisfies `core_mel::FftR2C`. This header
// hosts the single shared implementation. Other backends (kokoro,
// mimo_tokenizer) ship their own near-identical copies; those can move
// here too in a follow-up — keeping this lift granite-only avoids
// touching their numerical paths in this commit.
//
// Header-only / `static inline` so each caller inlines the recursion
// and keeps the existing numerical behaviour bit-identical.

#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace core_fft {

// Iterative in-place radix-2 Cooley-Tukey FFT.
//
//   re  : N floats — real parts (mutated in-place)
//   im  : N floats — imaginary parts (mutated in-place, should be zero-
//         initialized for a real-input FFT)
//   N   : must be a power of two
//
// Replaces the previous recursive variant that allocated O(N log N) heap
// per call. This iterative version uses O(1) extra memory (bit-reversal
// is done with swaps). Same algorithm as kaldi_fbank.cpp's fft_radix2.
static inline void fft_radix2_inplace(float* re, float* im, int N) {
    // Bit-reversal permutation.
    for (int i = 1, j = 0; i < N; i++) {
        int bit = N >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    // Butterfly stages.
    for (int len = 2; len <= N; len <<= 1) {
        const float ang = -2.0f * (float)M_PI / (float)len;
        const float wre = std::cos(ang);
        const float wim = std::sin(ang);
        for (int i = 0; i < N; i += len) {
            float cr = 1.0f, ci = 0.0f;
            for (int j = 0; j < len / 2; j++) {
                const float tr = re[i + j + len / 2] * cr - im[i + j + len / 2] * ci;
                const float ti = re[i + j + len / 2] * ci + im[i + j + len / 2] * cr;
                re[i + j + len / 2] = re[i + j] - tr;
                im[i + j + len / 2] = im[i + j] - ti;
                re[i + j] += tr;
                im[i + j] += ti;
                const float nr = cr * wre - ci * wim;
                ci = cr * wim + ci * wre;
                cr = nr;
            }
        }
    }
}

// True N a power of two? (the radix-2 in-place transform above is only
// valid then — for any other N its butterfly stages index past the
// N-element buffer, e.g. N=400 writes indices 400..511 in the len=256
// stage and corrupts the heap.)
static inline bool fft_is_pow2(int N) {
    return N > 0 && (N & (N - 1)) == 0;
}

// Naive O(N^2) real-input DFT → 2*N interleaved (re, im). Correct for any
// N; used as the odd-N base case of the mixed-radix transform below.
static inline void fft_dft_r2c(const float* in, int N, float* out) {
    for (int k = 0; k < N; k++) {
        float re = 0.0f, im = 0.0f;
        for (int n = 0; n < N; n++) {
            const float ang = -2.0f * (float)M_PI * (float)k * (float)n / (float)N;
            re += in[n] * std::cos(ang);
            im += in[n] * std::sin(ang);
        }
        out[2 * k] = re;
        out[2 * k + 1] = im;
    }
}

// Mixed-radix (radix-2 decimation-in-time) FFT of N real samples for
// ARBITRARY N: recurse splitting even/odd while N stays even, then fall
// back to fft_dft_r2c() once the remaining factor is odd. `in` doubles as
// recursion scratch — callers must give it room for >=2*N floats (we pass
// 4*N); `out` likewise needs scratch beyond the 2*N result (we pass 8*N).
// Mirrors the proven recursive FFT in voxtral.cpp.
static inline void fft_mixed_radix_r2c(float* in, int N, float* out) {
    if (N == 1) {
        out[0] = in[0];
        out[1] = 0.0f;
        return;
    }
    const int half = N / 2;
    if (N - half * 2 == 1) { // odd N → direct DFT
        fft_dft_r2c(in, N, out);
        return;
    }
    float* even = in + N;
    for (int i = 0; i < half; i++)
        even[i] = in[2 * i];
    float* ef = out + 2 * N;
    fft_mixed_radix_r2c(even, half, ef);
    float* odd = even;
    for (int i = 0; i < half; i++)
        odd[i] = in[2 * i + 1];
    float* of = ef + N;
    fft_mixed_radix_r2c(odd, half, of);
    for (int k = 0; k < half; k++) {
        const float ang = -2.0f * (float)M_PI * (float)k / (float)N;
        const float re = std::cos(ang), im = std::sin(ang);
        const float reo = of[2 * k], imo = of[2 * k + 1];
        out[2 * k] = ef[2 * k] + re * reo - im * imo;
        out[2 * k + 1] = ef[2 * k + 1] + re * imo + im * reo;
        out[2 * (k + half)] = ef[2 * k] - re * reo + im * imo;
        out[2 * (k + half) + 1] = ef[2 * k + 1] - re * imo - im * reo;
    }
}

// Non-power-of-two fallback shared by the two interleaved wrappers below:
// a correct N-point real DFT (matches the reference transform the callers'
// mel filterbanks were built for) instead of the heap-overrunning radix-2.
static inline void fft_nonpow2_r2c(const float* in, int N, float* out) {
    static thread_local std::vector<float> mr_in;
    static thread_local std::vector<float> mr_out;
    if ((int)mr_in.size() < 4 * N)
        mr_in.assign((size_t)4 * N, 0.0f);
    if ((int)mr_out.size() < 8 * N)
        mr_out.assign((size_t)8 * N, 0.0f);
    std::memcpy(mr_in.data(), in, (size_t)N * sizeof(float));
    fft_mixed_radix_r2c(mr_in.data(), N, mr_out.data());
    std::memcpy(out, mr_out.data(), (size_t)(2 * N) * sizeof(float));
}

// Legacy interleaved-output wrapper: takes N real samples in `in`,
// produces 2*N interleaved (re, im) floats in `out`. Internally
// delegates to the iterative in-place FFT via thread-local scratch.
// Kept for API compatibility with callers that expect the old signature.
static inline void fft_radix2(float* in, int N, float* out) {
    if (!fft_is_pow2(N)) {
        fft_nonpow2_r2c(in, N, out);
        return;
    }
    if (N <= 1) {
        out[0] = in[0];
        out[1] = 0;
        return;
    }
    // Thread-local scratch for the in-place transform.
    static thread_local std::vector<float> scratch_re;
    static thread_local std::vector<float> scratch_im;
    if ((int)scratch_re.size() < N)
        scratch_re.resize((size_t)N);
    if ((int)scratch_im.size() < N)
        scratch_im.resize((size_t)N);
    std::memcpy(scratch_re.data(), in, (size_t)N * sizeof(float));
    std::memset(scratch_im.data(), 0, (size_t)N * sizeof(float));
    fft_radix2_inplace(scratch_re.data(), scratch_im.data(), N);
    // Interleave into output.
    for (int k = 0; k < N; k++) {
        out[2 * k] = scratch_re[(size_t)k];
        out[2 * k + 1] = scratch_im[(size_t)k];
    }
}

// Const-input wrapper matching `core_mel::FftR2C`. Uses thread-local
// scratch buffers so the in-place FFT doesn't disturb the caller's
// input. `fft_radix2_wrapper` is what `core_mel::compute(...)` consumes
// via function pointer.
static inline void fft_radix2_wrapper(const float* in, int N, float* out) {
    if (!fft_is_pow2(N)) {
        fft_nonpow2_r2c(in, N, out);
        return;
    }
    static thread_local std::vector<float> scratch_re;
    static thread_local std::vector<float> scratch_im;
    if ((int)scratch_re.size() < N)
        scratch_re.resize((size_t)N);
    if ((int)scratch_im.size() < N)
        scratch_im.resize((size_t)N);
    std::memcpy(scratch_re.data(), in, (size_t)N * sizeof(float));
    std::memset(scratch_im.data(), 0, (size_t)N * sizeof(float));
    fft_radix2_inplace(scratch_re.data(), scratch_im.data(), N);
    // Interleave into the caller's output buffer.
    for (int k = 0; k < N; k++) {
        out[2 * k] = scratch_re[(size_t)k];
        out[2 * k + 1] = scratch_im[(size_t)k];
    }
}

} // namespace core_fft
