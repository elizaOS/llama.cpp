// tools/parakeet/src/pk-kernels-i8mm.cpp - ARMv8.6 int8 matrix multiply tier
// (SMMLA). Compiled with -march=armv8.2-a+dotprod+i8mm; only called after a
// runtime check (HWCAP2_I8MM / hw.optional.arm.FEAT_I8MM).
#include "pk-kernels.h"

#if PK_HAVE_ARM_EXT && defined(__ARM_FEATURE_DOTPROD) && defined(__ARM_FEATURE_MATMUL_INT8)
#include "pk-kernels-neon.inc"

namespace pk {
namespace {

template <int NP>
inline void tile_np(const int8_t * panel, const int8_t * x, size_t x_stride, int K, int32_t * out) {
    const int C = K / 8;
    int32x4_t acc[4][NP];
    for (int p = 0; p < 4; ++p) {
        for (int q = 0; q < NP; ++q) {
            acc[p][q] = vdupq_n_s32(0);
        }
    }
    for (int c = 0; c < C; ++c) {
        const int8_t * w = panel + (size_t) c * 64;
        const int8x16_t p0 = vld1q_s8(w), p1 = vld1q_s8(w + 16), p2 = vld1q_s8(w + 32), p3 = vld1q_s8(w + 48);
        for (int q = 0; q < NP; ++q) {
            const int8x16_t v = vld1q_s8(x + (size_t) q * x_stride + (size_t) c * 16);
            acc[0][q] = vmmlaq_s32(acc[0][q], p0, v);
            acc[1][q] = vmmlaq_s32(acc[1][q], p1, v);
            acc[2][q] = vmmlaq_s32(acc[2][q], p2, v);
            acc[3][q] = vmmlaq_s32(acc[3][q], p3, v);
        }
    }
    for (int p = 0; p < 4; ++p) {
        for (int q = 0; q < NP; ++q) {
            // SMMLA result order: [r0t0, r0t1, r1t0, r1t1] (r0 = 2p, t0 = 2q)
            const int32x4_t v = acc[p][q];
            vst1_s32(out + (2 * q) * 8 + 2 * p, vget_low_s32(vuzp1q_s32(v, v)));
            vst1_s32(out + (2 * q + 1) * 8 + 2 * p, vget_low_s32(vuzp2q_s32(v, v)));
        }
    }
}

void tile_i8mm(const int8_t * panel, const int8_t * x, size_t x_stride, int K, int np, int32_t * out) {
    switch (np) {
        case 1: tile_np<1>(panel, x, x_stride, K, out); break;
        case 2: tile_np<2>(panel, x, x_stride, K, out); break;
        case 3: tile_np<3>(panel, x, x_stride, K, out); break;
        default: tile_np<4>(panel, x, x_stride, K, out); break;
    }
}

}  // namespace

const kernel_table & kernels_i8mm() {
    static const kernel_table t = { tier::i8mm, "i8mm", decode_fv5_neon, decode_fv5t7_neon, tile_i8mm, matvec_dotprod };
    return t;
}

}  // namespace pk
#elif PK_HAVE_ARM_EXT
// The build did not pass the per-file -march flag (for example an Xcode iOS,
// tvOS or visionOS build), so this tier falls back to the dotprod table.
namespace pk {
const kernel_table & kernels_i8mm() {
    return kernels_dotprod();
}
}  // namespace pk
#endif
