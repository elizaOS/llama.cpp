// tools/parakeet/src/pk-ops.cpp
#include "pk-ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define PK_NEON 1
#endif

namespace pk {

// ---------------------------------------------------------------- exp

#if PK_NEON
// Cephes-style expf: range reduction to [-ln2/2, ln2/2], degree-5 polynomial,
// 2^n through the exponent field. Max rel error ~2 ulp on [-87, 88].
static inline float32x4_t exp_ps(float32x4_t x) {
    const float32x4_t hi = vdupq_n_f32(88.3762626647949f);
    const float32x4_t lo = vdupq_n_f32(-87.3365447505531f);
    x = vminq_f32(vmaxq_f32(x, lo), hi);
    const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
    const float32x4_t n = vrndnq_f32(vmulq_f32(x, log2e));
    float32x4_t r = vfmsq_f32(x, n, vdupq_n_f32(0.693359375f));
    r = vfmsq_f32(r, n, vdupq_n_f32(-2.12194440e-4f));
    float32x4_t p = vdupq_n_f32(1.9875691500E-4f);
    p = vfmaq_f32(vdupq_n_f32(1.3981999507E-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(8.3334519073E-3f), p, r);
    p = vfmaq_f32(vdupq_n_f32(4.1665795894E-2f), p, r);
    p = vfmaq_f32(vdupq_n_f32(1.6666665459E-1f), p, r);
    p = vfmaq_f32(vdupq_n_f32(5.0000001201E-1f), p, r);
    const float32x4_t r2 = vmulq_f32(r, r);
    p = vfmaq_f32(vaddq_f32(r, vdupq_n_f32(1.0f)), p, r2);
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

static inline float32x4_t silu_ps(float32x4_t x) {
    const float32x4_t e = exp_ps(vnegq_f32(x));
    return vdivq_f32(x, vaddq_f32(vdupq_n_f32(1.0f), e));
}
#endif

float sigmoidf_(float x) { return 1.0f / (1.0f + std::exp(-x)); }

void exp_inplace(float * x, size_t n) {
    size_t i = 0;
#if PK_NEON
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(x + i, exp_ps(vld1q_f32(x + i)));
    }
#endif
    for (; i < n; ++i) {
        x[i] = std::exp(x[i]);
    }
}

void silu_inplace(float * x, size_t n) {
    size_t i = 0;
#if PK_NEON
    for (; i + 4 <= n; i += 4) {
        vst1q_f32(x + i, silu_ps(vld1q_f32(x + i)));
    }
#endif
    for (; i < n; ++i) {
        x[i] = x[i] / (1.0f + std::exp(-x[i]));
    }
}

// ---------------------------------------------------------- layernorm

void layernorm(const float * x, float * y, int N, int D, const float * g, const float * b, float eps) {
    for (int t = 0; t < N; ++t) {
        const float * xr = x + (size_t) t * D;
        float * yr = y + (size_t) t * D;
        int i = 0;
        float sum = 0.0f;
#if PK_NEON
        float32x4_t s4 = vdupq_n_f32(0.0f);
        for (; i + 4 <= D; i += 4) {
            s4 = vaddq_f32(s4, vld1q_f32(xr + i));
        }
        sum = vaddvq_f32(s4);
#endif
        for (; i < D; ++i) {
            sum += xr[i];
        }
        const float mean = sum / (float) D;
        float var = 0.0f;
        i = 0;
#if PK_NEON
        const float32x4_t m4 = vdupq_n_f32(mean);
        float32x4_t v4 = vdupq_n_f32(0.0f);
        for (; i + 4 <= D; i += 4) {
            const float32x4_t d = vsubq_f32(vld1q_f32(xr + i), m4);
            v4 = vfmaq_f32(v4, d, d);
        }
        var = vaddvq_f32(v4);
#endif
        for (; i < D; ++i) {
            const float d = xr[i] - mean;
            var += d * d;
        }
        const float inv = 1.0f / std::sqrt(var / (float) D + eps);
        i = 0;
#if PK_NEON
        const float32x4_t inv4 = vdupq_n_f32(inv);
        for (; i + 4 <= D; i += 4) {
            const float32x4_t d = vmulq_f32(vsubq_f32(vld1q_f32(xr + i), m4), inv4);
            vst1q_f32(yr + i, vfmaq_f32(vld1q_f32(b + i), d, vld1q_f32(g + i)));
        }
#endif
        for (; i < D; ++i) {
            yr[i] = (xr[i] - mean) * inv * g[i] + b[i];
        }
    }
}

// ------------------------------------------------------ fp32 NT GEMM

void gemm_f32_nt(const float * A, size_t lda, const float * B, size_t ldb, float * C, size_t ldc, int M, int N, int K,
                 float alpha) {
#if PK_NEON
    const int K4 = K & ~3;
    for (int m = 0; m < M; m += 4) {
        const float * a[4];
        for (int i = 0; i < 4; ++i) {
            a[i] = A + (size_t) std::min(m + i, M - 1) * lda;
        }
        for (int n = 0; n < N; n += 4) {
            const float * bp[4];
            for (int j = 0; j < 4; ++j) {
                bp[j] = B + (size_t) std::min(n + j, N - 1) * ldb;
            }
            float32x4_t c00 = vdupq_n_f32(0), c01 = c00, c02 = c00, c03 = c00, c10 = c00, c11 = c00, c12 = c00,
                        c13 = c00, c20 = c00, c21 = c00, c22 = c00, c23 = c00, c30 = c00, c31 = c00, c32 = c00,
                        c33 = c00;
            for (int k = 0; k < K4; k += 4) {
                const float32x4_t a0 = vld1q_f32(a[0] + k), a1 = vld1q_f32(a[1] + k), a2 = vld1q_f32(a[2] + k),
                                  a3 = vld1q_f32(a[3] + k);
                const float32x4_t b0 = vld1q_f32(bp[0] + k), b1 = vld1q_f32(bp[1] + k), b2 = vld1q_f32(bp[2] + k),
                                  b3 = vld1q_f32(bp[3] + k);
                c00 = vfmaq_f32(c00, a0, b0); c01 = vfmaq_f32(c01, a0, b1); c02 = vfmaq_f32(c02, a0, b2); c03 = vfmaq_f32(c03, a0, b3);
                c10 = vfmaq_f32(c10, a1, b0); c11 = vfmaq_f32(c11, a1, b1); c12 = vfmaq_f32(c12, a1, b2); c13 = vfmaq_f32(c13, a1, b3);
                c20 = vfmaq_f32(c20, a2, b0); c21 = vfmaq_f32(c21, a2, b1); c22 = vfmaq_f32(c22, a2, b2); c23 = vfmaq_f32(c23, a2, b3);
                c30 = vfmaq_f32(c30, a3, b0); c31 = vfmaq_f32(c31, a3, b1); c32 = vfmaq_f32(c32, a3, b2); c33 = vfmaq_f32(c33, a3, b3);
            }
            float r[4][4] = {
                { vaddvq_f32(c00), vaddvq_f32(c01), vaddvq_f32(c02), vaddvq_f32(c03) },
                { vaddvq_f32(c10), vaddvq_f32(c11), vaddvq_f32(c12), vaddvq_f32(c13) },
                { vaddvq_f32(c20), vaddvq_f32(c21), vaddvq_f32(c22), vaddvq_f32(c23) },
                { vaddvq_f32(c30), vaddvq_f32(c31), vaddvq_f32(c32), vaddvq_f32(c33) },
            };
            for (int k = K4; k < K; ++k) {
                for (int i = 0; i < 4; ++i) {
                    for (int j = 0; j < 4; ++j) {
                        r[i][j] += a[i][k] * bp[j][k];
                    }
                }
            }
            for (int i = 0; i < 4 && m + i < M; ++i) {
                for (int j = 0; j < 4 && n + j < N; ++j) {
                    C[(size_t) (m + i) * ldc + n + j] = alpha * r[i][j];
                }
            }
        }
    }
#else
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            const float * a = A + (size_t) m * lda;
            const float * b = B + (size_t) n * ldb;
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += a[k] * b[k];
            }
            C[(size_t) m * ldc + n] = alpha * s;
        }
    }
#endif
}

// -------------------------------------------------------- activations

static void quantize_row(const float * x, int K, int8_t * dst, float * scale_out) {
    // dst: first chunk of this token; chunks are 16 bytes apart, 8 bytes each
    float amax = 0.0f;
    int k = 0;
#if PK_NEON
    float32x4_t m4 = vdupq_n_f32(0.0f);
    for (; k + 4 <= K; k += 4) {
        m4 = vmaxq_f32(m4, vabsq_f32(vld1q_f32(x + k)));
    }
    amax = vmaxvq_f32(m4);
#endif
    for (; k < K; ++k) {
        amax = std::max(amax, std::fabs(x[k]));
    }
    if (!(amax > 0.0f) || !std::isfinite(amax)) {
        *scale_out = 0.0f;
        for (int c = 0; c < K / 8; ++c) {
            std::memset(dst + c * 16, 0, 8);
        }
        return;
    }
    const float inv = 127.0f / amax;
    *scale_out = amax / 127.0f;
#if PK_NEON
    const float32x4_t iv = vdupq_n_f32(inv);
    for (int c = 0; c < K / 8; ++c) {
        const int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + 8 * c), iv));
        const int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(x + 8 * c + 4), iv));
        const int16x8_t h = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        vst1_s8(dst + c * 16, vqmovn_s16(h));
    }
#else
    for (int c = 0; c < K / 8; ++c) {
        for (int j = 0; j < 8; ++j) {
            float v = std::nearbyint(x[8 * c + j] * inv);
            v = std::min(127.0f, std::max(-127.0f, v));
            dst[c * 16 + j] = (int8_t) v;
        }
    }
#endif
}

void quantize_acts(pool & P, const float * X, int N, int K, size_t ldx, qact & out) {
    out.N = N;
    out.K = K;
    out.src = X;
    out.ldsrc = ldx;
    out.Npad = (N + 1) & ~1;
    const size_t need = (size_t) out.Npad * K;
    if (out.buf.size() < need) {
        out.buf.resize(need);
    }
    if (out.scale.size() < (size_t) out.Npad) {
        out.scale.resize(out.Npad);
    }
    const size_t ps = out.pair_stride();
    auto body = [&](int b, int e) {
        for (int t = b; t < e; ++t) {
            int8_t * dst = out.buf.data() + (size_t) (t / 2) * ps + (t & 1) * 8;
            if (t < N) {
                quantize_row(X + (size_t) t * ldx, K, dst, &out.scale[t]);
            } else {
                out.scale[t] = 0.0f;
                for (int c = 0; c < K / 8; ++c) {
                    std::memset(dst + c * 16, 0, 8);
                }
            }
        }
    };
    if ((size_t) N * K >= 32768) {
        P.parallel_for(out.Npad, 8, body);
    } else {
        body(0, out.Npad);
    }
}

// ------------------------------------------------------------ weights

void wmat_from_fv5(wmat & w, const uint8_t * codes, const float * lohi, int O, int K) {
    w.O = O;
    w.K = K;
    w.fv5 = codes;
    w.lut.assign((size_t) (O / 2) * 16, 0);
    w.ws.assign(O, 0.0f);
    w.grid_err = 0.0;
    for (int r = 0; r < O; ++r) {
        const double lo = lohi[2 * r], hi = lohi[2 * r + 1];
        int H = 0, L = 0;
        double s = 0.0;
        if (hi > 0.0 && lo >= 0.0 && lo <= hi) {
            const double ratio = lo / hi;
            double best = 1e30;
            for (int h = 127; h >= 64; --h) {
                const int l = (int) std::lround(h * ratio);
                const double err = ratio > 0 ? std::fabs(l - h * ratio) / (h * ratio) : 0.0;
                if (err < best - 1e-12) {
                    best = err;
                    H = h;
                    L = l;
                }
            }
            s = hi / H;
            // error of the lo level relative to the row's largest weight
            w.grid_err = std::max(w.grid_err, std::fabs(L * s - lo) / hi);
        } else if (hi != 0.0 || lo != 0.0) {
            // not seen in Phonon-2; keep both magnitudes on one grid
            const double m = std::max(std::fabs(lo), std::fabs(hi));
            s = m / 127.0;
            H = (int) std::lround(hi / s);
            L = (int) std::lround(lo / s);
        }
        w.ws[r] = (float) s;
        int8_t * lut = w.lut.data() + (size_t) (r / 2) * 16 + (r & 1) * 8;
        lut[0] = (int8_t) -H;
        lut[1] = (int8_t) -L;
        lut[2] = 0;
        lut[3] = (int8_t) L;
        lut[4] = (int8_t) H;
    }
}

void wmat_from_int8(wmat & w, const int8_t * q, const float * scale, int O, int K, const int * perm) {
    w.O = O;
    w.K = K;
    w.fv5 = nullptr;
    w.ws.assign(scale, scale + O);
    w.panels.assign((size_t) O * K, 0);
    const int C = K / 8;
    for (int rb = 0; rb < O / 8; ++rb) {
        int8_t * panel = w.panels.data() + (size_t) rb * 8 * K;
        for (int c = 0; c < C; ++c) {
            for (int p = 0; p < 4; ++p) {
                for (int h = 0; h < 2; ++h) {
                    const int8_t * row = q + (size_t) (rb * 8 + 2 * p + h) * K;
                    for (int j = 0; j < 8; ++j) {
                        const int col = 8 * c + j;
                        panel[((size_t) c * 4 + p) * 16 + h * 8 + j] = row[perm ? perm[col] : col];
                    }
                }
            }
        }
    }
}

// --------------------------------------------------------------- GEMM

static void gemm_f32_debug(pool & P, const kernel_table & kt, const wmat & W, const qact & X, float * Y, size_t ldy,
                           epi op, float alpha) {
    // same weights (decoded int8 levels x row scale), fp32 activations; the
    // source is copied first because Y may alias it
    const int K = W.K;
    std::vector<float> xs((size_t) X.N * K);
    for (int t = 0; t < X.N; ++t) std::memcpy(xs.data() + (size_t) t * K, X.src + (size_t) t * X.ldsrc, sizeof(float) * K);
    P.parallel_for(W.O / 8, 1, [&](int b, int e) {
        std::vector<int8_t> scratch((size_t) 8 * K);
        for (int rb = b; rb < e; ++rb) {
            const int8_t * panel;
            if (W.fv5) {
                kt.decode_fv5(W.fv5 + (size_t) rb * 4 * K, (size_t) K, W.lut.data() + (size_t) rb * 64, K, scratch.data());
                panel = scratch.data();
            } else {
                panel = W.panels.data() + (size_t) rb * 8 * K;
            }
            for (int t = 0; t < X.N; ++t) {
                const float * x = xs.data() + (size_t) t * K;
                for (int r = 0; r < 8; ++r) {
                    double s = 0;
                    for (int k = 0; k < K; ++k) {
                        s += (double) panel[((size_t) (k / 8) * 4 + r / 2) * 16 + (r & 1) * 8 + (k & 7)] * x[k];
                    }
                    const int o = rb * 8 + r;
                    float v = (float) s * W.ws[o] + (W.bias ? W.bias[o] : 0.0f);
                    float & y = Y[(size_t) t * ldy + o];
                    switch (op) {
                        case epi::store: y = v; break;
                        case epi::silu: y = v / (1.0f + std::exp(-v)); break;
                        case epi::relu: y = v > 0 ? v : 0; break;
                        case epi::add: y += alpha * v; break;
                    }
                }
            }
        }
    });
}

void gemm_q8(pool & P, const kernel_table & kt, const wmat & W, const qact & X, float * Y, size_t ldy, epi op,
             float alpha) {
    static const bool dbg = std::getenv("PK_DEBUG_F32_ACTS") != nullptr;
    if (dbg && X.src) {
        gemm_f32_debug(P, kt, W, X, Y, ldy, op, alpha);
        return;
    }
    const int K = W.K;
    const int nb = W.O / 8;
    const int npairs = (X.N + 1) / 2;
    const int ntt = (npairs + 3) / 4;
    const size_t ps = X.pair_stride();
    const int N = X.N;
    // Row blocks are processed in groups of G: the group's panels are decoded
    // once, then every token tile is multiplied against all G panels while it
    // is hot in L1, which cuts the activation traffic from L2 by G.
    int G = 1;
    if (ntt > 1) {
        G = std::max(1, std::min(4, nb / std::max(1, 4 * P.size())));
    }
    const int ng = (nb + G - 1) / G;
    P.parallel_for(ng, 1, [&](int b, int e) {
        thread_local std::vector<int8_t> scratch;
        if (W.fv5 && scratch.size() < (size_t) G * 8 * K) {
            scratch.resize((size_t) G * 8 * K);
        }
        alignas(16) int32_t acc[64];
        const int8_t * panels[4];
        for (int g = b; g < e; ++g) {
            const int rb0 = g * G, rb1 = std::min(nb, rb0 + G);
            for (int rb = rb0; rb < rb1; ++rb) {
                if (W.fv5) {
                    int8_t * dst = scratch.data() + (size_t) (rb - rb0) * 8 * K;
                    kt.decode_fv5(W.fv5 + (size_t) rb * 4 * K, (size_t) K, W.lut.data() + (size_t) rb * 64, K, dst);
                    panels[rb - rb0] = dst;
                } else {
                    panels[rb - rb0] = W.panels.data() + (size_t) rb * 8 * K;
                }
            }
            for (int tt = 0; tt < ntt; ++tt) {
              for (int rb = rb0; rb < rb1; ++rb) {
                const int8_t * panel = panels[rb - rb0];
                const float * ws = W.ws.data() + rb * 8;
                const float * bias = W.bias ? W.bias + rb * 8 : nullptr;
                kt.tile(panel, X.buf.data() + (size_t) tt * 4 * ps, ps, K, std::min(4, npairs - tt * 4), acc);
                const int tn = std::min(8, N - tt * 8);
                for (int t = 0; t < tn; ++t) {
                    const int tok = tt * 8 + t;
                    const float xs = X.scale[tok];
                    float * y = Y + (size_t) tok * ldy + rb * 8;
                    const int32_t * a = acc + t * 8;
#if PK_NEON
                    const float32x4_t xs4 = vdupq_n_f32(xs);
                    float32x4_t v0 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(a)), vld1q_f32(ws)), xs4);
                    float32x4_t v1 = vmulq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(a + 4)), vld1q_f32(ws + 4)), xs4);
                    if (bias) {
                        v0 = vaddq_f32(v0, vld1q_f32(bias));
                        v1 = vaddq_f32(v1, vld1q_f32(bias + 4));
                    }
                    switch (op) {
                        case epi::store:
                            break;
                        case epi::silu:
                            v0 = silu_ps(v0);
                            v1 = silu_ps(v1);
                            break;
                        case epi::relu:
                            v0 = vmaxq_f32(v0, vdupq_n_f32(0.0f));
                            v1 = vmaxq_f32(v1, vdupq_n_f32(0.0f));
                            break;
                        case epi::add: {
                            const float32x4_t al = vdupq_n_f32(alpha);
                            v0 = vfmaq_f32(vld1q_f32(y), v0, al);
                            v1 = vfmaq_f32(vld1q_f32(y + 4), v1, al);
                            break;
                        }
                    }
                    vst1q_f32(y, v0);
                    vst1q_f32(y + 4, v1);
#else
                    for (int r = 0; r < 8; ++r) {
                        float v = (float) a[r] * ws[r] * xs;
                        if (bias) {
                            v += bias[r];
                        }
                        switch (op) {
                            case epi::store: y[r] = v; break;
                            case epi::silu: y[r] = v / (1.0f + std::exp(-v)); break;
                            case epi::relu: y[r] = v > 0.0f ? v : 0.0f; break;
                            case epi::add: y[r] += alpha * v; break;
                        }
                    }
#endif
                }
              }
            }
        }
    });
}

}  // namespace pk
