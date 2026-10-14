// SPDX-License-Identifier: MIT
//
// test-kokoro-layer-options — the opt-in layer strategies in kokoro-layers.h
// (KokoroLayerOptions: tap_gemm_conv, fast_snake) against the default
// kernels. Kokoro-82M never enables them; Paradee-8M does. Checks that:
//   - Conv1d (stride 1, dilated, strided) and ConvTranspose1d give the same
//     result as the default path within float rounding,
//   - the polynomial sin^2 in fast Snake1D is within 5e-7 of libm,
//   - the options are scoped (restored when the scope ends).

#include "kokoro-layers.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace eliza_kokoro;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); ++failures; } } while (0)

static std::vector<float> randn(size_t n, std::mt19937 & g, float scale) {
    std::normal_distribution<float> d(0.0f, scale);
    std::vector<float> v(n);
    for (auto & x : v) x = d(g);
    return v;
}

static double rel_err(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, (double) std::fabs(a[i] - b[i]));
        den = std::max(den, (double) std::fabs(a[i]));
    }
    return num / std::max(den, 1e-30);
}

static detail::KokoroLayerOptions opts(bool on) {
    detail::KokoroLayerOptions o;
    o.tap_gemm_conv = on;
    o.fast_snake = on;
    return o;
}

static void check_conv(std::mt19937 & g, int Cin, int Cout, int K, int T, int stride, int pad, int dil) {
    const int T_out = (T + 2 * pad - dil * (K - 1) - 1) / stride + 1;
    auto x = randn((size_t) Cin * T, g, 1.0f), W = randn((size_t) Cout * Cin * K, g, 0.1f), b = randn(Cout, g, 0.5f);
    std::vector<float> y0((size_t) Cout * T_out), y1((size_t) Cout * T_out);
    conv1d_forward(x.data(), Cin, T, W.data(), b.data(), Cout, K, stride, pad, dil, y0.data(), T_out);
    {
        detail::KokoroLayerOptionsScope s(opts(true));
        conv1d_forward(x.data(), Cin, T, W.data(), b.data(), Cout, K, stride, pad, dil, y1.data(), T_out);
    }
    const double e = rel_err(y0, y1);
    std::printf("conv1d Cin=%d Cout=%d K=%d T=%d stride=%d dil=%d: rel err %.3g\n", Cin, Cout, K, T, stride, dil, e);
    CHECK(e < 1e-5, "conv1d mismatch %.3g", e);
}

static void check_convtranspose(std::mt19937 & g, int Cin, int Cout, int K, int T, int stride) {
    const int pad = (K - stride) / 2;
    const int T_out = convtranspose1d_out_len(T, K, stride, pad, 0);
    auto x = randn((size_t) Cin * T, g, 1.0f), W = randn((size_t) Cin * Cout * K, g, 0.1f), b = randn(Cout, g, 0.5f);
    std::vector<float> y0((size_t) Cout * T_out), y1((size_t) Cout * T_out);
    convtranspose1d_forward(x.data(), Cin, T, W.data(), b.data(), Cout, K, stride, pad, 0, y0.data(), T_out);
    {
        detail::KokoroLayerOptionsScope s(opts(true));
        convtranspose1d_forward(x.data(), Cin, T, W.data(), b.data(), Cout, K, stride, pad, 0, y1.data(), T_out);
    }
    const double e = rel_err(y0, y1);
    std::printf("convtranspose1d Cin=%d Cout=%d K=%d T=%d stride=%d: rel err %.3g\n", Cin, Cout, K, T, stride, e);
    CHECK(e < 1e-5, "convtranspose1d mismatch %.3g", e);
}

int main() {
    std::mt19937 g(1234);
    check_conv(g, 32, 32, 11, 1001, 1, 5, 1);
    check_conv(g, 32, 32, 7, 997, 1, 15, 5);     // dilated (get_padding(7, 5) = 15)
    check_conv(g, 514, 256, 3, 47, 1, 1, 1);
    check_conv(g, 30, 7, 3, 33, 1, 1, 1);         // ragged row / time tiles
    check_conv(g, 22, 64, 12, 601, 6, 3, 1);      // strided (noise_convs[0])
    check_convtranspose(g, 128, 64, 20, 40, 10);
    check_convtranspose(g, 64, 32, 12, 401, 6);

    // fast sin^2
    double worst = 0;
    for (int i = -200000; i <= 200000; ++i) {
        const float y = (float) i * 1e-3f;
        const double ref = std::sin((double) y) * std::sin((double) y);
        worst = std::max(worst, std::fabs((double) detail::fast_sin2(y) - ref));
    }
    std::printf("fast_sin2 on [-200, 200]: max |err| %.3g\n", worst);
    CHECK(worst < 5e-7, "fast_sin2 error %.3g", worst);

    // Snake with the option on matches libm within that bound.
    {
        auto x = randn(4 * 1000, g, 3.0f), a = randn(4, g, 1.0f);
        std::vector<float> x1 = x;
        snake1d_forward(x.data(), 4, 1000, a.data());
        {
            detail::KokoroLayerOptionsScope s(opts(true));
            snake1d_forward(x1.data(), 4, 1000, a.data());
        }
        // Both paths round the final x + sin^2(a x) / a to float, so each
        // can be up to half an ulp of the output away from the exact value.
        // Subtract that one-ulp allowance, then scale by |a| so the result
        // is in sin^2 units and comparable to the fast_sin2 bound.
        double e = 0;
        for (int c = 0; c < 4; ++c)
            for (int t = 0; t < 1000; ++t) {
                const float r = x[c * 1000 + t];
                const double ulp = (double) std::nextafter(std::fabs(r), INFINITY) - std::fabs(r);
                const double d = std::fabs((double) r - x1[c * 1000 + t]);
                e = std::max(e, std::max(0.0, d - ulp) * std::fabs(a[c]));
            }
        std::printf("snake1d fast vs libm: max excess over output rounding %.3g\n", e);
        CHECK(e < 1e-6, "snake mismatch %.3g", e);
    }

    // Scoped: the options are off again outside the scope.
    {
        detail::KokoroLayerOptionsScope s(opts(true));
        CHECK(detail::kokoro_layer_options().fast_snake, "scope did not set the option");
    }
    CHECK(!detail::kokoro_layer_options().fast_snake && !detail::kokoro_layer_options().tap_gemm_conv,
          "options leaked out of the scope");

    if (failures) { std::fprintf(stderr, "%d failure(s)\n", failures); return 1; }
    std::printf("test-kokoro-layer-options: OK\n");
    return 0;
}
