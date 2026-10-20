// SPDX-License-Identifier: MIT
//
// paradee-head.cpp — Paradee-8M v1.1 source-filter head (see paradee-head.h).
//
// Port of training/scripts/vhead.py SourceFilterHead2.forward and sine_bank
// (sahilmahendrakar/paradee 606e16e1, Apache-2.0). Same operations, same
// order, float32 like the module, with three implementation choices that only
// change rounding:
//
//   - The sine bank's harmonics come from a rotation recurrence in float on
//     exp(i 2 pi cyc) (computed in double), not one sin() per harmonic. The
//     recurrence is closer to the exact value than torch's float32 argument
//     2 pi k cyc (|arg| up to 400 rad, one float32 step ~3e-5 rad).
//   - The pulse shift's exp(i 2 pi (f / f0) shift) steps over the bins with a
//     double recurrence instead of one sincos per bin.
//   - STFTs are mixed-radix FFTs (600 = 4*2*3*5*5, 150 = 2*3*5*5) with two
//     real signals packed into one complex transform. The iSTFT is torch.istft:
//     C2R per frame (imaginary DC / Nyquist ignored), periodic Hann, overlap-
//     add, divided by the summed squared window, centre trimmed.

#include "paradee-head.h"

#include "kokoro-layers.h"
#include "kokoro-profile.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstring>

namespace eliza_paradee {
namespace head {

namespace {

using cf = std::complex<float>;
constexpr double K_PI = 3.14159265358979323846;

// Complex FFT for n with factors 2, 3, 4 and 5: iterative Stockham
// (decimation in frequency, natural order in and out, ping-pong buffers).
// inverse = conj(FFT(conj(x))), no 1/n scaling.
class MixedFft {
public:
    explicit MixedFft(int n) : n_(n) {
        int m = n;
        for (int f : { 4, 2, 3, 5 }) {
            while (m % f == 0) { fac_.push_back(f); m /= f; }
        }
        ok_ = m == 1 && n > 1;
        tw_.resize((size_t) n);
        for (int k = 0; k < n; ++k) {
            const double a = -2.0 * K_PI * (double) k / (double) n;
            tw_[(size_t) k] = cf((float) std::cos(a), (float) std::sin(a));
        }
        work_.resize((size_t) n);
    }
    bool ok() const { return ok_; }
    int size() const { return n_; }
    // out may not alias in. Not thread-safe (one scratch buffer per object).
    void run(const cf * in, cf * out, bool inverse) {
        const size_t N = (size_t) n_;
        if (inverse) for (size_t i = 0; i < N; ++i) work_[i] = std::conj(in[i]);
        else std::copy(in, in + N, work_.begin());
        // Each stage reads x and writes y; the last stage must write `out`.
        cf * bufs[2] = { work_.data(), out };
        int cur = 0;
        if (fac_.size() % 2 == 0) {   // even stage count: start from `out`
            std::copy(work_.begin(), work_.end(), out);
            cur = 1;
        }
        size_t n = N, st = 1;
        for (int p : fac_) {
            stage(p, n, st, bufs[cur], bufs[cur ^ 1]);
            cur ^= 1;
            n /= (size_t) p;
            st *= (size_t) p;
        }
        if (inverse) for (size_t i = 0; i < N; ++i) out[i] = std::conj(out[i]);
    }
private:
    static inline cf mul_mi(cf z) { return cf(z.imag(), -z.real()); }   // -i z
    // One radix-p step: sequence length n, stride st.
    void stage(int p, size_t n, size_t st, const cf * x, cf * y) const {
        const size_t m = n / (size_t) p, tws = (size_t) n_ / n;
        constexpr float h3 = 0.86602540378443865f;                 // sin(2 pi / 3)
        const float c1 = 0.30901699437494742f, c2 = -0.80901699437494742f;   // cos(2 pi / 5), cos(4 pi / 5)
        const float s1 = 0.95105651629515357f, s2 = 0.58778525229247313f;    // sin(2 pi / 5), sin(4 pi / 5)
        for (size_t j = 0; j < m; ++j) {
            const cf * w = tw_.data();
            const size_t wj = j * tws;
            for (size_t q = 0; q < st; ++q) {
                cf * o = y + q + st * (size_t) p * j;
                const cf * a = x + q + st * j;
                const size_t sm = st * m;
                switch (p) {
                    case 2: {
                        const cf a0 = a[0], a1 = a[sm];
                        o[0] = a0 + a1;
                        o[st] = (a0 - a1) * w[wj];
                        break;
                    }
                    case 3: {
                        const cf a0 = a[0], a1 = a[sm], a2 = a[2 * sm];
                        const cf t = a1 + a2, u = a0 - 0.5f * t, v = mul_mi(h3 * (a1 - a2));
                        o[0] = a0 + t;
                        o[st] = (u + v) * w[wj];
                        o[2 * st] = (u - v) * w[2 * wj];
                        break;
                    }
                    case 4: {
                        const cf a0 = a[0], a1 = a[sm], a2 = a[2 * sm], a3 = a[3 * sm];
                        const cf t0 = a0 + a2, t1 = a0 - a2, t2 = a1 + a3, t3 = mul_mi(a1 - a3);
                        o[0] = t0 + t2;
                        o[st] = (t1 + t3) * w[wj];
                        o[2 * st] = (t0 - t2) * w[2 * wj];
                        o[3 * st] = (t1 - t3) * w[3 * wj];
                        break;
                    }
                    default: {   // 5
                        const cf a0 = a[0], a1 = a[sm], a2 = a[2 * sm], a3 = a[3 * sm], a4 = a[4 * sm];
                        const cf t1 = a1 + a4, t2 = a2 + a3, t3 = a1 - a4, t4 = a2 - a3;
                        const cf r1 = a0 + c1 * t1 + c2 * t2, r2 = a0 + c2 * t1 + c1 * t2;
                        const cf i1 = mul_mi(s1 * t3 + s2 * t4), i2 = mul_mi(s2 * t3 - s1 * t4);
                        o[0] = a0 + t1 + t2;
                        o[st] = (r1 + i1) * w[wj];
                        o[2 * st] = (r2 + i2) * w[2 * wj];
                        o[3 * st] = (r2 - i2) * w[3 * wj];
                        o[4 * st] = (r1 - i1) * w[4 * wj];
                        break;
                    }
                }
            }
        }
    }
    int n_;
    bool ok_ = false;
    std::vector<int> fac_;
    std::vector<cf> tw_;
    std::vector<cf> work_;
};

// torch.hann_window(N) (periodic), float32.
std::vector<float> hann(int N) {
    std::vector<float> w((size_t) N);
    for (int j = 0; j < N; ++j) w[(size_t) j] = (float) (0.5 - 0.5 * std::cos(2.0 * K_PI * (double) j / (double) N));
    return w;
}

// Reflect-padded sample j - pad of x (torch.stft center=True, pad_mode="reflect").
inline float reflect_at(const float * x, int L, long j) {
    if (j < 0) j = -j;
    else if (j >= L) j = 2L * (L - 1) - j;
    return x[j];
}

// Frame f of torch.stft(x, N, H, window=hann, center=True) for one or two
// real signals of length L (> N/2): o1 (and o2) get N/2+1 bins. Two signals
// share one complex FFT. z, Z: N-sample scratch.
void stft_frame(MixedFft & fft, const std::vector<float> & w, int H, const float * x1, const float * x2, int L,
                int f, std::vector<cf> & z, std::vector<cf> & Z, cf * o1, cf * o2) {
    const int N = fft.size(), nb = N / 2 + 1, pad = N / 2;
    const long base = (long) f * H - pad;
    if (base >= 0 && base + N <= L) {
        const float * p1 = x1 + base;
        const float * p2 = x2 ? x2 + base : nullptr;
        for (int j = 0; j < N; ++j) z[(size_t) j] = cf(p1[j] * w[(size_t) j], p2 ? p2[j] * w[(size_t) j] : 0.0f);
    } else {
        for (int j = 0; j < N; ++j) {
            const float wj = w[(size_t) j];
            z[(size_t) j] = cf(reflect_at(x1, L, base + j) * wj, x2 ? reflect_at(x2, L, base + j) * wj : 0.0f);
        }
    }
    fft.run(z.data(), Z.data(), false);
    if (!o2) {
        std::copy(Z.begin(), Z.begin() + nb, o1);
        return;
    }
    for (int k = 0; k < nb; ++k) {
        const cf a = Z[(size_t) k], b = std::conj(Z[(size_t) ((N - k) % N)]);
        o1[k] = (a + b) * 0.5f;
        o2[k] = (a - b) * cf(0.0f, -0.5f);
    }
}

// torch.stft of one real signal, all F = L/H + 1 frames: X [F][N/2+1].
void stft(MixedFft & fft, const std::vector<float> & w, int H, const float * x, int L, std::vector<cf> & X) {
    const int N = fft.size(), nb = N / 2 + 1, F = L / H + 1;
    X.assign((size_t) F * nb, cf());
    std::vector<cf> z((size_t) N), Z((size_t) N);
    for (int f = 0; f < F; ++f) stft_frame(fft, w, H, x, nullptr, L, f, z, Z, X.data() + (size_t) f * nb, nullptr);
}

// torch.istft(Y, N, H, window=hann, center=True, length=L), fed frame pairs
// as they are made: C2R per frame (imaginary DC / Nyquist dropped), window,
// overlap-add, divide by the summed squared window, trim the centre padding.
// Two frames share one complex inverse FFT.
class Ola {
public:
    Ola(MixedFft & fft, const std::vector<float> & w, int H, int F)
        : fft_(fft), w_(w), H_(H), N_(fft.size()), nb_(fft.size() / 2 + 1),
          acc_((size_t) fft.size() + (size_t) H * (size_t) (F - 1), 0.0f), env_(acc_.size(), 0.0f),
          z_((size_t) fft.size()), r_((size_t) fft.size()) {}
    // Frames f (A) and f + 1 (B, may be null for the last odd frame).
    void add(const cf * A, const cf * B, int f) {
        for (int k = 0; k < N_; ++k) {
            const cf a = full(A, k);
            z_[(size_t) k] = B ? a + cf(0.0f, 1.0f) * full(B, k) : a;
        }
        fft_.run(z_.data(), r_.data(), true);
        const float inv_n = 1.0f / (float) N_;
        for (int fi = 0; fi < (B ? 2 : 1); ++fi) {
            const size_t off = (size_t) (f + fi) * (size_t) H_;
            for (int j = 0; j < N_; ++j) {
                const float v = (fi == 0 ? r_[(size_t) j].real() : r_[(size_t) j].imag()) * inv_n;
                acc_[off + (size_t) j] += v * w_[(size_t) j];
                env_[off + (size_t) j] += w_[(size_t) j] * w_[(size_t) j];
            }
        }
    }
    void finish(int L, std::vector<float> & y) const {
        const size_t pad = (size_t) N_ / 2;
        y.assign((size_t) L, 0.0f);
        for (int t = 0; t < L; ++t) {
            const float e = env_[(size_t) t + pad];
            y[(size_t) t] = e > 1e-11f ? acc_[(size_t) t + pad] / e : 0.0f;
        }
    }
private:
    cf full(const cf * A, int k) const {   // Hermitian extension
        if (k == 0) return cf(A[0].real(), 0.0f);
        if (2 * k == N_) return cf(A[k].real(), 0.0f);
        return k < nb_ ? A[k] : std::conj(A[N_ - k]);
    }
    MixedFft & fft_;
    const std::vector<float> & w_;
    int H_, N_, nb_;
    std::vector<float> acc_, env_;
    std::vector<cf> z_, r_;
};

void istft(MixedFft & fft, const std::vector<float> & w, int H, const cf * Y, int F, int L, std::vector<float> & y) {
    const size_t nb = (size_t) fft.size() / 2 + 1;
    Ola ola(fft, w, H, F);
    for (int f = 0; f < F; f += 2) ola.add(Y + (size_t) f * nb, f + 1 < F ? Y + (size_t) (f + 1) * nb : nullptr, f);
    ola.finish(L, y);
}

// LayerNorm over channels of a channel-major [C, n] map (nn.LayerNorm(C) on
// x.transpose(1, 2)), eps 1e-5. In place.
void layer_norm_cm(float * x, int C, int n, const float * w, const float * b) {
    std::vector<double> sum((size_t) n, 0.0), sq((size_t) n, 0.0);
    for (int c = 0; c < C; ++c) {
        const float * xc = x + (size_t) c * n;
        for (int t = 0; t < n; ++t) { sum[(size_t) t] += xc[t]; sq[(size_t) t] += (double) xc[t] * xc[t]; }
    }
    std::vector<float> mean((size_t) n), inv((size_t) n);
    for (int t = 0; t < n; ++t) {
        const double m = sum[(size_t) t] / C;
        const double var = std::max(0.0, sq[(size_t) t] / C - m * m);
        mean[(size_t) t] = (float) m;
        inv[(size_t) t] = (float) (1.0 / std::sqrt(var + 1e-5));
    }
    for (int c = 0; c < C; ++c) {
        float * xc = x + (size_t) c * n;
        const float wc = w[c], bc = b[c];
        for (int t = 0; t < n; ++t) xc[t] = (xc[t] - mean[(size_t) t]) * inv[(size_t) t] * wc + bc;
    }
}

// vhead.warp_matrix(nb, G) as (index, weight) pairs: bin b takes
// (1 - w) * g[i] + w * g[i + 1]. float32 as the module computes it.
void warp(int nb, int G, int sr, std::vector<int> & idx, std::vector<float> & wt) {
    idx.resize((size_t) nb);
    wt.resize((size_t) nb);
    const float top = std::log1p((float) sr / 2.0f / 700.0f);
    for (int b = 0; b < nb; ++b) {
        const float f = nb > 1 ? (float) ((double) sr / 2.0 * b / (nb - 1)) : 0.0f;
        const float pos = std::log1p(f / 700.0f) / top * (float) (G - 1);
        const int i = std::min(std::max((int) std::floor(pos), 0), G - 2);
        idx[(size_t) b] = i;
        wt[(size_t) b] = pos - (float) i;
    }
}

// Consecutive phase timings under KOKORO_PROFILE=1 (kokoro-profile.h).
class laps {
public:
    explicit laps(const char * first) : label_(first), t0_(std::chrono::steady_clock::now()) {}
    ~laps() { next(nullptr); }
    void next(const char * label) {
        const auto now = std::chrono::steady_clock::now();
        if (label_ && eliza_kokoro::kokoro_profile_enabled()) {
            eliza_kokoro::kokoro_profile_emit(label_, std::chrono::duration<double, std::milli>(now - t0_).count());
        }
        label_ = label;
        t0_ = now;
    }
private:
    const char * label_;
    std::chrono::steady_clock::time_point t0_;
};

inline float gelu(float x) { return 0.5f * x * (1.0f + std::erf(x * 0.70710678118654752f)); }

bool get_i32(gguf_context * g, const char * key, int & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) { err = std::string("missing GGUF key ") + key; return false; }
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_INT32:  out = gguf_get_val_i32(g, id); return true;
        case GGUF_TYPE_UINT32: out = (int) gguf_get_val_u32(g, id); return true;
        default: err = std::string("GGUF key ") + key + " is not an integer"; return false;
    }
}

bool get_f32(gguf_context * g, const char * key, float & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) { err = std::string("missing GGUF key ") + key; return false; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) { err = std::string("GGUF key ") + key + " is not f32"; return false; }
    out = gguf_get_val_f32(g, id);
    return true;
}

// ggml shape check against the PyTorch shape (ne reversed); 0 = any.
const float * tensor(ggml_context * ctx, const std::string & name, std::initializer_list<int64_t> torch_shape,
                     std::string & err) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t || t->type != GGML_TYPE_F32) { err = "missing F32 tensor " + name; return nullptr; }
    const int nd = (int) torch_shape.size();
    int i = 0;
    for (int64_t d : torch_shape) {
        if (t->ne[nd - 1 - i] != d) { err = "tensor " + name + " has an unexpected shape"; return nullptr; }
        ++i;
    }
    for (int k = nd; k < GGML_MAX_DIMS; ++k) {
        if (t->ne[k] != 1) { err = "tensor " + name + " has an unexpected shape"; return nullptr; }
    }
    return (const float *) t->data;
}

} // namespace

bool load(gguf_context * g, ggml_context * ctx, params & p, weights & w, std::string & err) {
    bool ok =
        get_i32(g, "paradee.head.dim", p.dim, err) &&
        get_i32(g, "paradee.head.blocks", p.blocks, err) &&
        get_i32(g, "paradee.head.kernel", p.kernel, err) &&
        get_i32(g, "paradee.head.mult", p.mult, err) &&
        get_i32(g, "paradee.head.f0_hop", p.f0_hop, err) &&
        get_i32(g, "paradee.head.n_fft", p.n_fft, err) &&
        get_i32(g, "paradee.head.hop", p.hop, err) &&
        get_i32(g, "paradee.head.harmonics", p.K, err) &&
        get_i32(g, "paradee.head.mod", p.mod, err) &&
        get_i32(g, "paradee.head.points", p.G, err) &&
        get_i32(g, "paradee.head.burst_n_fft", p.burst_n_fft, err) &&
        get_i32(g, "paradee.head.burst_hop", p.burst_hop, err) &&
        get_i32(g, "paradee.head.burst_points", p.Gb, err) &&
        get_f32(g, "paradee.head.sine_amp", p.sine_amp, err) &&
        get_f32(g, "paradee.head.max_harmonic_hz", p.max_harmonic_hz, err) &&
        get_f32(g, "paradee.head.voiced_f0_hz", p.voiced_f0_hz, err) &&
        get_f32(g, "paradee.head.min_f0_hz", p.min_f0_hz, err) &&
        get_f32(g, "paradee.head.gain_clamp", p.gain_clamp, err) &&
        get_f32(g, "paradee.head.shift_scale", p.shift_scale, err) &&
        get_i32(g, "paradee.audio.sample_rate", p.sample_rate, err) &&
        get_i32(g, "paradee.decoder.initial_ch", p.in_ch, err);
    if (!ok) return false;
    // Bounds, so a corrupt file cannot ask for absurd buffers, and the
    // structure the graph relies on (whole STFT frames per trunk frame,
    // FFT sizes this file's MixedFft handles).
    const bool sane =
        p.dim > 0 && p.dim <= 4096 && p.blocks >= 0 && p.blocks <= 64 && p.kernel > 0 && p.kernel % 2 == 1 &&
        p.kernel <= 31 && p.mult > 0 && p.mult <= 16 && p.in_ch > 0 && p.in_ch <= 4096 &&
        p.f0_hop > 0 && p.hop > 0 && p.f0_hop % p.hop == 0 && p.burst_hop > 0 && p.f0_hop % p.burst_hop == 0 &&
        p.n_fft >= 2 && p.n_fft % 2 == 0 && p.n_fft <= 8192 && p.burst_n_fft >= 2 && p.burst_n_fft % 2 == 0 &&
        p.burst_n_fft <= 8192 && p.n_fft / 2 < 2 * p.f0_hop && p.burst_n_fft / 2 < 2 * p.f0_hop &&
        p.K > 0 && p.K <= 512 && p.mod >= 0 && p.mod <= 8 && p.G >= 2 && p.G <= 4096 && p.Gb >= 2 && p.Gb <= 4096 &&
        p.sample_rate > 0 && p.sample_rate <= 192000 && std::isfinite(p.sine_amp) && std::isfinite(p.gain_clamp) &&
        std::isfinite(p.shift_scale) && std::isfinite(p.max_harmonic_hz) && std::isfinite(p.voiced_f0_hz) &&
        std::isfinite(p.min_f0_hz) && p.min_f0_hz > 0.0f &&
        MixedFft(p.n_fft).ok() && MixedFft(p.burst_n_fft).ok();
    if (!sane) { err = "unsupported Paradee head hparams"; return false; }

    const int64_t D = p.dim, F = (int64_t) p.mult * p.dim, K = p.kernel;
    const int64_t up = p.f0_hop / p.hop, C = 4 + 2 * (int64_t) p.mod;
    const int64_t R = up * (C * p.G + 1), Rb = (int64_t) (p.f0_hop / p.burst_hop) * p.Gb;
    const std::string h = "paradee.head.";
    w = weights();
    ok = (w.inp_w = tensor(ctx, h + "inp.weight", { D, p.in_ch, K }, err)) &&
         (w.inp_b = tensor(ctx, h + "inp.bias", { D }, err)) &&
         (w.norm0_w = tensor(ctx, h + "norm0.weight", { D }, err)) &&
         (w.norm0_b = tensor(ctx, h + "norm0.bias", { D }, err)) &&
         (w.norm1_w = tensor(ctx, h + "norm1.weight", { D }, err)) &&
         (w.norm1_b = tensor(ctx, h + "norm1.bias", { D }, err)) &&
         (w.out_w = tensor(ctx, h + "out.weight", { R, D }, err)) &&
         (w.out_b = tensor(ctx, h + "out.bias", { R }, err)) &&
         (w.outb_w = tensor(ctx, h + "out_b.weight", { Rb, D }, err)) &&
         (w.outb_b = tensor(ctx, h + "out_b.bias", { Rb }, err));
    if (!ok) return false;
    w.blocks.resize((size_t) p.blocks);
    for (int i = 0; i < p.blocks; ++i) {
        const std::string b = h + "blocks." + std::to_string(i) + ".";
        block_weights & bw = w.blocks[(size_t) i];
        ok = (bw.dw_w = tensor(ctx, b + "dw.weight", { D, 1, K }, err)) &&
             (bw.dw_b = tensor(ctx, b + "dw.bias", { D }, err)) &&
             (bw.norm_w = tensor(ctx, b + "norm.weight", { D }, err)) &&
             (bw.norm_b = tensor(ctx, b + "norm.bias", { D }, err)) &&
             (bw.pw1_w = tensor(ctx, b + "pw1.weight", { F, D }, err)) &&
             (bw.pw1_b = tensor(ctx, b + "pw1.bias", { F }, err)) &&
             (bw.pw2_w = tensor(ctx, b + "pw2.weight", { D, F }, err)) &&
             (bw.pw2_b = tensor(ctx, b + "pw2.bias", { D }, err)) &&
             (bw.gamma = tensor(ctx, b + "gamma", { D }, err));
        if (!ok) return false;
    }
    return true;
}

void conv_stack(const params & p, const weights & w, const float * x, int n, std::vector<float> & hn) {
    using eliza_kokoro::conv1d_forward;
    const int D = p.dim, F = p.mult * p.dim, K = p.kernel;
    hn.assign((size_t) D * n, 0.0f);
    conv1d_forward(x, p.in_ch, n, w.inp_w, w.inp_b, D, K, 1, K / 2, 1, hn.data(), n);
    layer_norm_cm(hn.data(), D, n, w.norm0_w, w.norm0_b);
    std::vector<float> d((size_t) D * n), u((size_t) F * n), v((size_t) D * n);
    for (const block_weights & b : w.blocks) {
        eliza_kokoro::conv1d_grouped_forward(hn.data(), D, n, b.dw_w, b.dw_b, D, K, 1, K / 2, 1, D, d.data(), n);
        layer_norm_cm(d.data(), D, n, b.norm_w, b.norm_b);
        conv1d_forward(d.data(), D, n, b.pw1_w, b.pw1_b, F, 1, 1, 0, 1, u.data(), n);
        for (float & e : u) e = gelu(e);
        conv1d_forward(u.data(), F, n, b.pw2_w, b.pw2_b, D, 1, 1, 0, 1, v.data(), n);
        for (int c = 0; c < D; ++c) {
            const float g = b.gamma[c];
            float * hc = hn.data() + (size_t) c * n;
            const float * vc = v.data() + (size_t) c * n;
            for (int t = 0; t < n; ++t) hc[t] += g * vc[t];
        }
    }
    layer_norm_cm(hn.data(), D, n, w.norm1_w, w.norm1_b);
}

namespace {

// sine_bank plus the fundamental's exp(i 2 pi m cyc) for m = 1..mod (the
// pulsing noise carriers) and the voicing mask, all per sample.
void sine_bank_full(const params & p, const float * f0, int n, std::vector<float> & e,
                    std::vector<float> * carriers, std::vector<float> * uv) {
    const int H = p.f0_hop;
    const size_t L = (size_t) n * H;
    e.assign(L, 0.0f);
    if (carriers) carriers->assign(L * 2 * (size_t) p.mod, 0.0f);   // [2*mod][L]: cos m, sin m
    if (uv) uv->assign(L, 0.0f);
    // c: running phase at each frame (cycles, float64 cumsum, wrapped);
    // d: advance to the next frame's centre.
    std::vector<float> c((size_t) n), d((size_t) n);
    double acc = 0.0;
    for (int j = 0; j < n; ++j) {
        const double r = (double) f0[j] * H / p.sample_rate;
        acc += r;
        c[(size_t) j] = (float) (acc - std::floor(acc));
        if (j > 0) d[(size_t) j - 1] = (float) r;
    }
    d[(size_t) n - 1] = 0.0f;
    std::vector<float> cs((size_t) H), sn((size_t) H), ck((size_t) H), sk((size_t) H), sum((size_t) H);
    for (int j = 0; j < n; ++j) {
        const float f = f0[j];
        const bool voiced = f > p.voiced_f0_hz;
        int kmax = 0;
        if (voiced) while (kmax < p.K && f * (float) (kmax + 1) < p.max_harmonic_hz) ++kmax;
        for (int s = 0; s < H; ++s) {
            const size_t t = (size_t) j * H + (size_t) s;
            const float pf = ((float) t + 0.5f) / (float) H - 0.5f;
            const int i = std::min(std::max((int) std::floor(pf), 0), n - 1);
            const float fr = std::min(std::max(pf - (float) i, 0.0f), 1.0f);
            // torch rounds fr * d and the sum separately; spell both roundings
            // out so no compiler contracts them into one fused multiply-add
            // (the product of two floats is exact in double).
            const float step = (float) ((double) fr * (double) d[(size_t) i]);
            const float cyc = (float) ((double) c[(size_t) i] + (double) step);
            const double a = 2.0 * K_PI * (double) cyc;
            const std::complex<double> z1(std::cos(a), std::sin(a));
            cs[(size_t) s] = (float) z1.real();
            sn[(size_t) s] = (float) z1.imag();
            if (carriers) {
                std::complex<double> zm = z1;   // exp(i 2 pi m cyc)
                for (int m = 1; m <= p.mod; ++m, zm *= z1) {
                    (*carriers)[(size_t) (2 * (m - 1)) * L + t] = (float) zm.real();
                    (*carriers)[(size_t) (2 * (m - 1) + 1) * L + t] = (float) zm.imag();
                }
            }
            if (uv) (*uv)[t] = voiced ? 1.0f : 0.0f;
        }
        if (kmax == 0) continue;
        std::copy(cs.begin(), cs.end(), ck.begin());
        std::copy(sn.begin(), sn.end(), sk.begin());
        std::copy(sn.begin(), sn.end(), sum.begin());
        for (int k = 2; k <= kmax; ++k) {
            for (int s = 0; s < H; ++s) {
                const float c2 = ck[(size_t) s] * cs[(size_t) s] - sk[(size_t) s] * sn[(size_t) s];
                const float s2 = sk[(size_t) s] * cs[(size_t) s] + ck[(size_t) s] * sn[(size_t) s];
                ck[(size_t) s] = c2;
                sk[(size_t) s] = s2;
                sum[(size_t) s] += s2;
            }
        }
        float * ej = e.data() + (size_t) j * H;
        for (int s = 0; s < H; ++s) ej[s] = sum[(size_t) s] * p.sine_amp;
    }
}

} // namespace

void sine_bank(const params & p, const float * f0, int n, std::vector<float> & e) {
    sine_bank_full(p, f0, n, e, nullptr, nullptr);
}

bool forward(const params & p, const weights & w,
             const float * x, int n, const float * f0, const float * noise,
             std::vector<float> & y, stages * st,
             cancel_fn cancel, void * user, bool & cancelled, std::string & err) {
    y.clear();
    cancelled = false;
    if (!x || !f0 || !noise || n < 2) { err = "head: bad input"; return false; }
    auto stop = [&]() { if (cancel && cancel(user)) { cancelled = true; y.clear(); return true; } return false; };
    using eliza_kokoro::conv1d_forward;
    const int L = n * p.f0_hop, up = p.f0_hop / p.hop, T = n * up, nf = T + 1;
    const int nb = p.n_fft / 2 + 1, C = 4 + 2 * p.mod, R = C * p.G + 1;
    const int upb = p.f0_hop / p.burst_hop, Tb = n * upb, nfb = Tb + 1, nbb = p.burst_n_fft / 2 + 1;

    // ConvNeXt stack and the two output projections (channel-major).
    laps lap("head convnext");
    std::vector<float> hn;
    conv_stack(p, w, x, n, hn);
    if (st) {
        st->hn.assign((size_t) n * p.dim, 0.0f);
        for (int c = 0; c < p.dim; ++c)
            for (int t = 0; t < n; ++t) st->hn[(size_t) t * p.dim + c] = hn[(size_t) c * n + t];
    }
    std::vector<float> o((size_t) up * R * n), ob((size_t) upb * p.Gb * n);
    conv1d_forward(hn.data(), p.dim, n, w.out_w, w.out_b, up * R, 1, 1, 0, 1, o.data(), n);
    conv1d_forward(hn.data(), p.dim, n, w.outb_w, w.outb_b, upb * p.Gb, 1, 1, 0, 1, ob.data(), n);
    if (stop()) return false;

    // Sources: sine bank and the pulsing noise (noise * uv * cos/sin(2 pi m phi)).
    lap.next("head sources");
    std::vector<float> e;
    std::vector<std::vector<float>> nm((size_t) 2 * p.mod, std::vector<float>((size_t) L));
    {
        std::vector<float> carriers, uv;
        sine_bank_full(p, f0, n, e, &carriers, &uv);
        for (int j = 0; j < 2 * p.mod; ++j) {
            const float * cj = carriers.data() + (size_t) j * L;
            float * mj = nm[(size_t) j].data();
            for (int t = 0; t < L; ++t) mj[t] = noise[t] * uv[(size_t) t] * cj[t];
        }
    }
    if (st) st->sine = e;
    if (stop()) return false;

    // Main path, one STFT frame at a time (no spectrogram is kept): the
    // spectra of e, the noise and the pulsing noise, the predicted filter,
    // and the inverse STFT's overlap-add.
    lap.next("head main path");
    MixedFft fft(p.n_fft);
    const std::vector<float> win = hann(p.n_fft);
    std::vector<int> wi;
    std::vector<float> ww;
    warp(nb, p.G, p.sample_rate, wi, ww);
    const double bin_hz = (double) p.sample_rate / 2.0 / (nb - 1);
    const int M = 2 * p.mod;
    std::vector<cf> z((size_t) p.n_fft), Z((size_t) p.n_fft);
    std::vector<cf> Ef((size_t) nb), Nf((size_t) nb), Mf((size_t) (M + 1) * nb), Yp((size_t) 2 * nb);
    std::vector<float> gk((size_t) C * nb);
    Ola ola(fft, win, p.hop, nf);
    for (int f0i = 0; f0i < nf; f0i += 2) {
        for (int fi = 0; fi < 2 && f0i + fi < nf; ++fi) {
            const int f = f0i + fi;
            stft_frame(fft, win, p.hop, e.data(), noise, L, f, z, Z, Ef.data(), Nf.data());
            for (int j = 0; j < M; j += 2) {
                stft_frame(fft, win, p.hop, nm[(size_t) j].data(), j + 1 < M ? nm[(size_t) j + 1].data() : nullptr, L,
                           f, z, Z, Mf.data() + (size_t) j * nb, j + 1 < M ? Mf.data() + (size_t) (j + 1) * nb : nullptr);
            }
            const int ff = std::min(f, T - 1), t = ff / up, s = ff % up;
            auto oc = [&](int r) { return o[((size_t) s * R + (size_t) r) * n + (size_t) t]; };
            for (int c = 0; c < C; ++c) {
                float * g = gk.data() + (size_t) c * nb;
                for (int b = 0; b < nb; ++b) {
                    const int i = wi[(size_t) b];
                    const float a = ww[(size_t) b];
                    g[b] = (1.0f - a) * oc(c * p.G + i) + a * oc(c * p.G + i + 1);
                }
            }
            const float shift = p.shift_scale * std::tanh(oc(C * p.G));
            const float f0f = std::max(f0[std::min(f / up, n - 1)], p.min_f0_hz);
            // exp(i 2 pi (b * bin_hz / f0f) shift), stepped over the bins.
            const double step = 2.0 * K_PI * bin_hz / (double) f0f * (double) shift;
            const std::complex<double> rot(std::cos(step), std::sin(step));
            std::complex<double> ph(1.0, 0.0);
            cf * Yf = Yp.data() + (size_t) fi * nb;
            for (int b = 0; b < nb; ++b) {
                const float gh = std::exp(std::min(gk[(size_t) b], p.gain_clamp));
                const float gn = std::exp(std::min(gk[(size_t) 3 * nb + b], p.gain_clamp));
                const float ur = gk[(size_t) nb + b], ui = gk[(size_t) 2 * nb + b];
                const float um = std::max(std::hypot(ur, ui), 1e-4f);
                const cf hh = cf(ur / um, ui / um) * cf((float) ph.real(), (float) ph.imag()) * gh;
                cf ntot = Nf[(size_t) b];
                for (int j = 0; j < M; ++j) ntot += gk[(size_t) (4 + j) * nb + b] * Mf[(size_t) j * nb + b];
                Yf[b] = Ef[(size_t) b] * hh + ntot * gn;
                ph *= rot;
            }
        }
        ola.add(Yp.data(), f0i + 1 < nf ? Yp.data() + nb : nullptr, f0i);
        if ((f0i & 511) == 0 && stop()) return false;
    }
    std::vector<float> main;
    ola.finish(L, main);
    if (stop()) return false;

    // Noise-burst path: STFT 150/75 of the reversed noise, gains at Gb points.
    lap.next("head burst");
    MixedFft fftb(p.burst_n_fft);
    const std::vector<float> winb = hann(p.burst_n_fft);
    std::vector<float> rev(noise, noise + L);
    std::reverse(rev.begin(), rev.end());
    std::vector<cf> Nb;
    stft(fftb, winb, p.burst_hop, rev.data(), L, Nb);
    warp(nbb, p.Gb, p.sample_rate, wi, ww);
    for (int f = 0; f < nfb; ++f) {
        const int ff = std::min(f, Tb - 1), t = ff / upb, s = ff % upb;
        cf * Bf = Nb.data() + (size_t) f * nbb;
        for (int b = 0; b < nbb; ++b) {
            const int i = wi[(size_t) b];
            const float a = ww[(size_t) b];
            const float v = (1.0f - a) * ob[((size_t) s * p.Gb + (size_t) i) * n + (size_t) t] +
                            a * ob[((size_t) s * p.Gb + (size_t) i + 1) * n + (size_t) t];
            Bf[b] *= std::exp(std::min(v, p.gain_clamp));
        }
    }
    std::vector<float> burst;
    istft(fftb, winb, p.burst_hop, Nb.data(), nfb, L, burst);

    y.resize((size_t) L);
    for (int t = 0; t < L; ++t) y[(size_t) t] = main[(size_t) t] + burst[(size_t) t];
    if (st) { st->main = std::move(main); st->burst = std::move(burst); }
    if (stop()) return false;
    return true;
}

void gaussian_noise(uint64_t seed, size_t count, std::vector<float> & out) {
    auto splitmix = [](uint64_t & z) {
        uint64_t r = (z += 0x9e3779b97f4a7c15ull);
        r = (r ^ (r >> 30)) * 0xbf58476d1ce4e5b9ull;
        r = (r ^ (r >> 27)) * 0x94d049bb133111ebull;
        return r ^ (r >> 31);
    };
    uint64_t sm = seed, s[4];
    for (uint64_t & v : s) v = splitmix(sm);
    auto rotl = [](uint64_t v, int k) { return (v << k) | (v >> (64 - k)); };
    auto next = [&]() {
        const uint64_t r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    };
    out.resize(count);
    for (size_t i = 0; i < count; i += 2) {
        const double u1 = (double) ((next() >> 11) + 1) * 0x1.0p-53;   // (0, 1]
        const double u2 = (double) (next() >> 11) * 0x1.0p-53;         // [0, 1)
        const double r = std::sqrt(-2.0 * std::log(u1));
        out[i] = (float) (r * std::cos(2.0 * K_PI * u2));
        if (i + 1 < count) out[i + 1] = (float) (r * std::sin(2.0 * K_PI * u2));
    }
}

} // namespace head
} // namespace eliza_paradee
