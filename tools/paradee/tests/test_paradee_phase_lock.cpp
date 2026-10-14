// SPDX-License-Identifier: MIT
//
// test-paradee-phase-lock — the native phase-locking filter against
// training/scripts/phase_lock.py lock() on deterministic synthetic input.
//
// fixtures/phase_lock_synthetic.f32 holds two outputs of the reference on the
// inputs built below (dump_paradee_reference.py lock-fixture):
//   [0, n)   phase_lock.lock() evaluated in float64      -> gate: max |err| <= 1e-4
//   [n, 2n)  phase_lock.lock() as shipped (float32)     -> gate: within float32's
//            own distance from the float64 result (+1e-4)
// Also checks pass-through (unvoiced input, short input) and cancellation.

#include "paradee.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifndef PARADEE_FIXTURE_DIR
#error "PARADEE_FIXTURE_DIR must be defined"
#endif

static constexpr int N = 9600;
static constexpr double SR = 24000.0;
static constexpr double PI = 3.14159265358979323846;

static void make_inputs(std::vector<float> & s, std::vector<float> & src, std::vector<float> & f0) {
    const int nf0 = N / 300;
    std::vector<double> f0d((size_t) nf0);
    for (int j = 0; j < nf0; ++j) f0d[(size_t) j] = (j >= 12 && j < 16) ? 0.0 : 110.0 + 110.0 * j / (double) (nf0 - 1);
    s.assign(N, 0.0f); src.assign(N, 0.0f); f0.assign((size_t) nf0, 0.0f);
    for (int j = 0; j < nf0; ++j) f0[(size_t) j] = (float) f0d[(size_t) j];
    double ph = 0.0;
    for (int t = 0; t < N; ++t) {
        const double f = f0d[(size_t) (t / 300)];
        ph += 2 * PI * f / SR;
        const bool voiced = f > 0;
        src[(size_t) t] = (float) (std::tanh(0.8 * std::sin(ph)) * (voiced ? 1.0 : 0.0) + 0.01 * std::sin(0.37 * t));
        double acc = 0.0;
        for (int h = 1; h < 40; ++h) {
            if (f * h >= 11000) break;
            acc += std::sin(h * ph + 0.002 * h * t / 7.0 + 0.5 * h) / h;
        }
        s[(size_t) t] = (float) (0.2 * acc * (voiced ? 1.0 : 0.3) + 0.003 * std::sin(1.3 * t));
    }
}

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++failures; } } while (0)

static double max_abs(const std::vector<float> & a, const float * b, size_t n) {
    double m = 0;
    for (size_t i = 0; i < n; ++i) m = std::max(m, std::fabs((double) a[i] - (double) b[i]));
    return m;
}

static bool cancel_now(void *) { return true; }

int main() {
    std::vector<float> s, src, f0;
    make_inputs(s, src, f0);

    std::vector<float> ref((size_t) 2 * N);
    {
        std::ifstream in(std::string(PARADEE_FIXTURE_DIR) + "/phase_lock_synthetic.f32", std::ios::binary);
        in.read((char *) ref.data(), (std::streamsize) (ref.size() * sizeof(float)));
        if (!in) { std::fprintf(stderr, "cannot read phase_lock_synthetic.f32\n"); return 1; }
    }
    const float * ref64 = ref.data();
    const float * ref32 = ref.data() + N;

    eliza_paradee::paradee_lock_params p;   // v1.0 defaults
    std::vector<float> out;
    CHECK(eliza_paradee::paradee_phase_lock(s.data(), s.size(), src.data(), src.size(), f0.data(), f0.size(), p, out),
          "lock returned false");
    CHECK(out.size() == (size_t) N, "output length %zu", out.size());
    const double e64 = max_abs(out, ref64, N);
    const double e32 = max_abs(out, ref32, N);
    double f32_floor = 0;
    for (int i = 0; i < N; ++i) f32_floor = std::max(f32_floor, std::fabs((double) ref32[i] - ref64[i]));
    double change = 0;
    for (int i = 0; i < N; ++i) change += ((double) ref64[i] - s[(size_t) i]) * ((double) ref64[i] - s[(size_t) i]);
    change = std::sqrt(change / N);
    std::printf("lock vs phase_lock.lock float64: max|err| = %.3g (gate 1e-4)\n", e64);
    std::printf("lock vs phase_lock.lock float32: max|err| = %.3g (float32 vs float64 itself: %.3g)\n", e32, f32_floor);
    std::printf("the filter changes this input by rms %.4g\n", change);
    CHECK(e64 <= 1e-4, "float64 reference error %.3g > 1e-4", e64);
    CHECK(e32 <= f32_floor + 1e-4, "float32 reference error %.3g > %.3g", e32, f32_floor + 1e-4);
    CHECK(change > 1e-3, "fixture does not exercise the filter");

    // Unvoiced everywhere: output == input.
    {
        std::vector<float> zf0(f0.size(), 0.0f), o2;
        eliza_paradee::paradee_phase_lock(s.data(), s.size(), src.data(), src.size(), zf0.data(), zf0.size(), p, o2);
        CHECK(max_abs(o2, s.data(), N) == 0.0, "unvoiced input changed");
    }
    // Shorter than n_fft/2: passes through (torch cannot reflect-pad it).
    {
        std::vector<float> o3;
        CHECK(eliza_paradee::paradee_phase_lock(s.data(), 500, src.data(), 500, f0.data(), f0.size(), p, o3),
              "short input rejected");
        CHECK(o3.size() == 500 && max_abs(o3, s.data(), 500) == 0.0, "short input changed");
    }
    // Lengths differ: output has min(ns, nsrc) samples.
    {
        std::vector<float> o4;
        eliza_paradee::paradee_phase_lock(s.data(), N, src.data(), N - 1000, f0.data(), f0.size(), p, o4);
        CHECK(o4.size() == (size_t) N - 1000, "min length %zu", o4.size());
    }
    // Cancellation empties the output.
    {
        std::vector<float> o5;
        const bool ok = eliza_paradee::paradee_phase_lock(s.data(), s.size(), src.data(), src.size(),
                                                          f0.data(), f0.size(), p, o5, cancel_now, nullptr);
        CHECK(!ok && o5.empty(), "cancel did not stop the filter");
    }
    // Invalid parameters: false, output = input.
    {
        eliza_paradee::paradee_lock_params bad = p;
        bad.n_fft = 1000;
        std::vector<float> o6;
        CHECK(!eliza_paradee::paradee_phase_lock(s.data(), s.size(), src.data(), src.size(), f0.data(), f0.size(), bad, o6),
              "non power-of-two n_fft accepted");
        CHECK(o6.size() == (size_t) N && max_abs(o6, s.data(), N) == 0.0, "rejected call changed the signal");
    }

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test-paradee-phase-lock: OK\n");
    return 0;
}
