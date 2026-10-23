// tools/parakeet/src/pk-kernels-dotprod.cpp - ARMv8.2 dotprod tier.
// Compiled with -march=armv8.2-a+dotprod; only called after a runtime check.
#include "pk-kernels.h"

#if PK_HAVE_ARM_EXT
#include "pk-kernels-neon.inc"

namespace pk {
namespace {

// NQ (1 or 2) token pairs at a time: 4 row pairs x NQ token pairs x
// {straight, swapped} accumulators. Straight lanes: r0t0, r0t0, r1t1, r1t1
// (two k halves); swapped (token halves exchanged): r0t1, r0t1, r1t0, r1t0.
template <int NQ>
inline void part_tile(const int8_t * panel, const int8_t * x, size_t x_stride, int C, int32_t * out, int q0) {
    int32x4_t s[4][NQ], w[4][NQ];
    for (int p = 0; p < 4; ++p) {
        for (int q = 0; q < NQ; ++q) {
            s[p][q] = vdupq_n_s32(0);
            w[p][q] = vdupq_n_s32(0);
        }
    }
    for (int c = 0; c < C; ++c) {
        const int8_t * wp = panel + (size_t) c * 64;
        const int8x16_t p0 = vld1q_s8(wp), p1 = vld1q_s8(wp + 16), p2 = vld1q_s8(wp + 32), p3 = vld1q_s8(wp + 48);
        for (int q = 0; q < NQ; ++q) {
            const int8x16_t v = vld1q_s8(x + (size_t) (q0 + q) * x_stride + (size_t) c * 16);
            const int8x16_t u = vextq_s8(v, v, 8);
            s[0][q] = vdotq_s32(s[0][q], p0, v); w[0][q] = vdotq_s32(w[0][q], p0, u);
            s[1][q] = vdotq_s32(s[1][q], p1, v); w[1][q] = vdotq_s32(w[1][q], p1, u);
            s[2][q] = vdotq_s32(s[2][q], p2, v); w[2][q] = vdotq_s32(w[2][q], p2, u);
            s[3][q] = vdotq_s32(s[3][q], p3, v); w[3][q] = vdotq_s32(w[3][q], p3, u);
        }
    }
    for (int p = 0; p < 4; ++p) {
        for (int q = 0; q < NQ; ++q) {
            // [r0t0, r1t1, r0t1, r1t0]
            const int32x4_t r = vpaddq_s32(s[p][q], w[p][q]);
            const int t0 = 2 * (q0 + q);
            out[t0 * 8 + 2 * p] = vgetq_lane_s32(r, 0);
            out[(t0 + 1) * 8 + 2 * p + 1] = vgetq_lane_s32(r, 1);
            out[(t0 + 1) * 8 + 2 * p] = vgetq_lane_s32(r, 2);
            out[t0 * 8 + 2 * p + 1] = vgetq_lane_s32(r, 3);
        }
    }
}

void tile_dotprod(const int8_t * panel, const int8_t * x, size_t x_stride, int K, int np, int32_t * out) {
    const int C = K / 8;
    int q = 0;
    for (; q + 2 <= np; q += 2) part_tile<2>(panel, x, x_stride, C, out, q);
    if (q < np) part_tile<1>(panel, x, x_stride, C, out, q);
}

}  // namespace

const kernel_table & kernels_dotprod() {
    static const kernel_table t = { tier::dotprod, "dotprod", decode_fv5_neon, decode_fv5t7_neon, tile_dotprod, matvec_dotprod };
    return t;
}

}  // namespace pk
#endif
