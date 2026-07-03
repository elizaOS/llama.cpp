// Parity + microbench for the portable-fast (non-Apple) kokoro-layers path.
//
// Compares the threaded + NEON implementations of the four hot primitives
// (conv1d_forward, convtranspose1d_forward, linear_forward, lstm_cell_step)
// against the pure scalar reference on random input — max abs diff must stay
// under 1e-4 — then microbenches scalar vs portable-fast on the real
// iSTFTNet generator shapes (see kokoro-generator.cpp).
//
// Exit code 0 = all parity checks pass.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace kokoro_layer_test {
void ref_linear(const float *, int, const float *, const float *, int, float *);
void ref_conv1d(const float *, int, int, const float *, const float *, int, int, int, int, int, float *, int);
void ref_convtranspose1d(const float *, int, int, const float *, const float *, int, int, int, int, int, float *, int);
void ref_lstm_cell_step(const float *, int, int, const float *, const float *, const float *, const float *, const float *, const float *, float *, float *, float *);
void fast_linear(const float *, int, const float *, const float *, int, float *);
void fast_conv1d(const float *, int, int, const float *, const float *, int, int, int, int, int, float *, int);
void fast_convtranspose1d(const float *, int, int, const float *, const float *, int, int, int, int, int, float *, int);
void fast_lstm_cell_step(const float *, int, int, const float *, const float *, const float *, const float *, const float *, const float *, float *, float *, float *);
bool fast_is_neon();
int  fast_thread_count();
} // namespace kokoro_layer_test

using namespace kokoro_layer_test;

static constexpr float TOL = 1e-4f;
static int g_failures = 0;
static std::mt19937 rng(20260702);

static std::vector<float> randv(size_t n, float lo = -1.0f, float hi = 1.0f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto & x : v) x = d(rng);
    return v;
}

static float max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        const float d = std::fabs(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static void check(const char * name, float diff) {
    const bool ok = diff < TOL && std::isfinite(diff);
    std::printf("%-58s max|Δ| = %.3e  %s\n", name, (double) diff, ok ? "PASS" : "FAIL");
    if (!ok) ++g_failures;
}

static int conv_out_len(int T, int K, int stride, int pad, int dilation) {
    return (T + 2 * pad - dilation * (K - 1) - 1) / stride + 1;
}

static int convtr_out_len(int T, int K, int stride, int pad, int output_pad) {
    return (T - 1) * stride - 2 * pad + (K - 1) + output_pad + 1;
}

static double ms_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

// ---------------------------------------------------------------- parity ---

static void parity_conv1d(int Cin, int Cout, int T, int K, int stride, int pad,
                          int dilation, bool bias) {
    const int T_out = conv_out_len(T, K, stride, pad, dilation);
    if (T_out <= 0) { std::printf("skip conv T_out<=0\n"); return; }
    auto x = randv((size_t) Cin * T);
    auto W = randv((size_t) Cout * Cin * K);
    auto B = randv((size_t) Cout);
    std::vector<float> yr((size_t) Cout * T_out), yf((size_t) Cout * T_out, 1e9f);
    ref_conv1d(x.data(), Cin, T, W.data(), bias ? B.data() : nullptr, Cout, K, stride, pad, dilation, yr.data(), T_out);
    fast_conv1d(x.data(), Cin, T, W.data(), bias ? B.data() : nullptr, Cout, K, stride, pad, dilation, yf.data(), T_out);
    char name[128];
    std::snprintf(name, sizeof name, "conv1d  Cin=%d Cout=%d T=%d K=%d s=%d p=%d d=%d b=%d",
                  Cin, Cout, T, K, stride, pad, dilation, (int) bias);
    check(name, max_abs_diff(yr, yf));
}

static void parity_convtr(int Cin, int Cout, int T, int K, int stride, int pad,
                          int output_pad, bool bias) {
    const int T_out = convtr_out_len(T, K, stride, pad, output_pad);
    if (T_out <= 0) { std::printf("skip convtr T_out<=0\n"); return; }
    auto x = randv((size_t) Cin * T);
    auto W = randv((size_t) Cin * Cout * K);
    auto B = randv((size_t) Cout);
    std::vector<float> yr((size_t) Cout * T_out), yf((size_t) Cout * T_out, 1e9f);
    ref_convtranspose1d(x.data(), Cin, T, W.data(), bias ? B.data() : nullptr, Cout, K, stride, pad, output_pad, yr.data(), T_out);
    fast_convtranspose1d(x.data(), Cin, T, W.data(), bias ? B.data() : nullptr, Cout, K, stride, pad, output_pad, yf.data(), T_out);
    char name[128];
    std::snprintf(name, sizeof name, "convtr  Cin=%d Cout=%d T=%d K=%d s=%d p=%d op=%d b=%d",
                  Cin, Cout, T, K, stride, pad, output_pad, (int) bias);
    check(name, max_abs_diff(yr, yf));
}

static void parity_linear(int in_dim, int out_dim, bool bias) {
    auto x = randv((size_t) in_dim);
    auto W = randv((size_t) out_dim * in_dim);
    auto B = randv((size_t) out_dim);
    std::vector<float> yr((size_t) out_dim), yf((size_t) out_dim, 1e9f);
    ref_linear(x.data(), in_dim, W.data(), bias ? B.data() : nullptr, out_dim, yr.data());
    fast_linear(x.data(), in_dim, W.data(), bias ? B.data() : nullptr, out_dim, yf.data());
    char name[128];
    std::snprintf(name, sizeof name, "linear  in=%d out=%d b=%d", in_dim, out_dim, (int) bias);
    check(name, max_abs_diff(yr, yf));
}

static void parity_lstm(int I, int H, bool bias) {
    auto x = randv((size_t) I);
    auto h0 = randv((size_t) H);
    auto c0 = randv((size_t) H);
    auto Wih = randv((size_t) 4 * H * I, -0.3f, 0.3f);
    auto Whh = randv((size_t) 4 * H * H, -0.3f, 0.3f);
    auto bih = randv((size_t) 4 * H);
    auto bhh = randv((size_t) 4 * H);
    std::vector<float> hr(H), cr(H), hf(H, 1e9f), cf(H, 1e9f), gr(4 * H), gf(4 * H);
    ref_lstm_cell_step(x.data(), I, H, h0.data(), c0.data(),
                       Wih.data(), bias ? bih.data() : nullptr,
                       Whh.data(), bias ? bhh.data() : nullptr,
                       hr.data(), cr.data(), gr.data());
    fast_lstm_cell_step(x.data(), I, H, h0.data(), c0.data(),
                        Wih.data(), bias ? bih.data() : nullptr,
                        Whh.data(), bias ? bhh.data() : nullptr,
                        hf.data(), cf.data(), gf.data());
    char name[128];
    std::snprintf(name, sizeof name, "lstm    I=%d H=%d b=%d (h)", I, H, (int) bias);
    check(name, max_abs_diff(hr, hf));
    std::snprintf(name, sizeof name, "lstm    I=%d H=%d b=%d (c)", I, H, (int) bias);
    check(name, max_abs_diff(cr, cf));
}

// ------------------------------------------------------------- microbench ---

template <typename F>
static double bench_ms(F && fn, int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        const double ms = ms_since(t0);
        if (ms < best) best = ms;
    }
    return best;
}

static void bench_conv1d(const char * label, int Cin, int Cout, int T, int K,
                         int stride, int pad, int dilation) {
    const int T_out = conv_out_len(T, K, stride, pad, dilation);
    auto x = randv((size_t) Cin * T);
    auto W = randv((size_t) Cout * Cin * K);
    auto B = randv((size_t) Cout);
    std::vector<float> y((size_t) Cout * T_out);
    const double ref = bench_ms([&] {
        ref_conv1d(x.data(), Cin, T, W.data(), B.data(), Cout, K, stride, pad, dilation, y.data(), T_out);
    }, 1);
    const double fast = bench_ms([&] {
        fast_conv1d(x.data(), Cin, T, W.data(), B.data(), Cout, K, stride, pad, dilation, y.data(), T_out);
    }, 3);
    std::printf("%-52s scalar %9.1f ms   fast %8.2f ms   %6.1fx\n", label, ref, fast, ref / fast);
}

static void bench_convtr(const char * label, int Cin, int Cout, int T, int K,
                         int stride, int pad) {
    const int T_out = convtr_out_len(T, K, stride, pad, 0);
    auto x = randv((size_t) Cin * T);
    auto W = randv((size_t) Cin * Cout * K);
    auto B = randv((size_t) Cout);
    std::vector<float> y((size_t) Cout * T_out);
    const double ref = bench_ms([&] {
        ref_convtranspose1d(x.data(), Cin, T, W.data(), B.data(), Cout, K, stride, pad, 0, y.data(), T_out);
    }, 1);
    const double fast = bench_ms([&] {
        fast_convtranspose1d(x.data(), Cin, T, W.data(), B.data(), Cout, K, stride, pad, 0, y.data(), T_out);
    }, 3);
    std::printf("%-52s scalar %9.1f ms   fast %8.2f ms   %6.1fx\n", label, ref, fast, ref / fast);
}

static void bench_lstm(const char * label, int I, int H, int steps) {
    auto x = randv((size_t) steps * I);
    auto Wih = randv((size_t) 4 * H * I, -0.3f, 0.3f);
    auto Whh = randv((size_t) 4 * H * H, -0.3f, 0.3f);
    auto bih = randv((size_t) 4 * H);
    auto bhh = randv((size_t) 4 * H);
    std::vector<float> h(H, 0.0f), c(H, 0.0f), hn(H), cn(H), g(4 * H);
    const double ref = bench_ms([&] {
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);
        for (int t = 0; t < steps; ++t) {
            ref_lstm_cell_step(x.data() + (size_t) t * I, I, H, h.data(), c.data(),
                               Wih.data(), bih.data(), Whh.data(), bhh.data(),
                               hn.data(), cn.data(), g.data());
            h.swap(hn); c.swap(cn);
        }
    }, 1);
    const double fast = bench_ms([&] {
        std::fill(h.begin(), h.end(), 0.0f);
        std::fill(c.begin(), c.end(), 0.0f);
        for (int t = 0; t < steps; ++t) {
            fast_lstm_cell_step(x.data() + (size_t) t * I, I, H, h.data(), c.data(),
                                Wih.data(), bih.data(), Whh.data(), bhh.data(),
                                hn.data(), cn.data(), g.data());
            h.swap(hn); c.swap(cn);
        }
    }, 3);
    std::printf("%-52s scalar %9.1f ms   fast %8.2f ms   %6.1fx\n", label, ref, fast, ref / fast);
}

int main() {
    std::printf("portable-fast path: NEON=%d threads=%d\n\n", (int) fast_is_neon(), fast_thread_count());

    std::printf("== parity (fast vs scalar reference, tol %.0e) ==\n", (double) TOL);
    // conv1d — generator resblock shapes (stride 1, dilated) + noise_convs
    // (stride 6 / K 12, and K=1) + conv_post (Cout 22, K 7) + edge cases.
    parity_conv1d(8, 5, 37, 3, 1, 1, 1, true);
    parity_conv1d(16, 16, 100, 7, 1, 15, 5, true);       // get_padding(7,5)=15
    parity_conv1d(16, 16, 100, 11, 1, 25, 5, false);     // get_padding(11,5)=25
    parity_conv1d(22, 32, 264, 12, 6, 3, 1, true);       // noise_convs[0] shape
    parity_conv1d(22, 32, 264, 1, 1, 0, 1, true);        // noise_convs[1] shape
    parity_conv1d(64, 22, 200, 7, 1, 3, 1, true);        // conv_post-like
    parity_conv1d(3, 4, 1, 5, 1, 6, 3, true);            // T=1, taps mostly OOB
    parity_conv1d(1, 1, 9, 3, 2, 0, 1, false);           // stride 2, no pad
    parity_conv1d(256, 256, 128, 7, 1, 9, 3, true);      // threaded-size slice
    // convtranspose1d — generator upsample stages + edges.
    parity_convtr(16, 8, 40, 20, 10, 5, 0, true);        // ups[0]-shaped
    parity_convtr(12, 6, 50, 12, 6, 3, 0, true);         // ups[1]-shaped
    parity_convtr(7, 9, 33, 3, 1, 1, 0, false);          // stride 1 AXPY path
    parity_convtr(5, 5, 1, 4, 2, 0, 1, true);            // T=1 + output_pad
    parity_convtr(512, 256, 16, 20, 10, 5, 0, true);     // threaded-size slice
    // linear — AdaIN fc + predictor proj sizes.
    parity_linear(128, 1024, true);
    parity_linear(640, 512, false);
    parity_linear(3, 2, true);
    // lstm — small (serial) + predictor-sized (threaded).
    parity_lstm(16, 8, true);
    parity_lstm(64, 48, false);
    parity_lstm(512, 512, true);

    std::printf("\n== microbench (real iSTFTNet generator shapes) ==\n");
    bench_conv1d("conv1d 256->256 T=2640 K=7 d=3 (stage0 resblock)", 256, 256, 2640, 7, 1, 9, 3);
    bench_conv1d("conv1d 128->128 T=15841 K=11 d=5 (stage1 resblock)", 128, 128, 15841, 11, 1, 25, 5);
    bench_convtr("convtr 512->256 T=264 K=20 s=10 (ups[0])", 512, 256, 264, 20, 10, 5);
    bench_convtr("convtr 256->128 T=2640 K=12 s=6 (ups[1])", 256, 128, 2640, 12, 6, 3);
    bench_lstm("lstm I=512 H=512 steps=264 (predictor-sized)", 512, 512, 264);

    std::printf("\n%s (%d failures)\n", g_failures == 0 ? "ALL PARITY CHECKS PASSED" : "PARITY FAILURES", g_failures);
    return g_failures == 0 ? 0 : 1;
}
