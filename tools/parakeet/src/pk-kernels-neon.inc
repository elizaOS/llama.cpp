// tools/parakeet/src/pk-kernels-neon.inc - NEON pieces shared by the dotprod
// and i8mm translation units (each is compiled with its own -march).
#include <arm_neon.h>

namespace pk {
namespace {

inline void decode_fv5_neon(const uint8_t * codes, size_t pair_stride, const int8_t * luts, int K, int8_t * panel) {
    const int G = K / 16;
    const uint8x16_t m = vdupq_n_u8(0x0F);
    for (int p = 0; p < 4; ++p) {
        const int8x16_t lut = vld1q_s8(luts + p * 16);
        const uint8_t * src = codes + (size_t) p * pair_stride;
        int8_t * dst = panel + p * 16;
        int g = 0;
        for (; g + 2 <= G; g += 2) {
            const uint8x16_t b0 = vld1q_u8(src + g * 16);
            const uint8x16_t b1 = vld1q_u8(src + g * 16 + 16);
            vst1q_s8(dst + (size_t) (2 * g) * 64, vqtbl1q_s8(lut, vandq_u8(b0, m)));
            vst1q_s8(dst + (size_t) (2 * g + 1) * 64, vqtbl1q_s8(lut, vshrq_n_u8(b0, 4)));
            vst1q_s8(dst + (size_t) (2 * g + 2) * 64, vqtbl1q_s8(lut, vandq_u8(b1, m)));
            vst1q_s8(dst + (size_t) (2 * g + 3) * 64, vqtbl1q_s8(lut, vshrq_n_u8(b1, 4)));
        }
        for (; g < G; ++g) {
            const uint8x16_t b0 = vld1q_u8(src + g * 16);
            vst1q_s8(dst + (size_t) (2 * g) * 64, vqtbl1q_s8(lut, vandq_u8(b0, m)));
            vst1q_s8(dst + (size_t) (2 * g + 1) * 64, vqtbl1q_s8(lut, vshrq_n_u8(b0, 4)));
        }
    }
}

inline void matvec_dotprod(const int8_t * w, int O, int K, const int8_t * x, int32_t * y) {
    int o = 0;
    for (; o + 4 <= O; o += 4) {
        const int8_t * r0 = w + (size_t) o * K;
        const int8_t * r1 = r0 + K;
        const int8_t * r2 = r1 + K;
        const int8_t * r3 = r2 + K;
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
        for (int k = 0; k < K; k += 16) {
            const int8x16_t xv = vld1q_s8(x + k);
            a0 = vdotq_s32(a0, vld1q_s8(r0 + k), xv);
            a1 = vdotq_s32(a1, vld1q_s8(r1 + k), xv);
            a2 = vdotq_s32(a2, vld1q_s8(r2 + k), xv);
            a3 = vdotq_s32(a3, vld1q_s8(r3 + k), xv);
        }
        y[o] = vaddvq_s32(a0);
        y[o + 1] = vaddvq_s32(a1);
        y[o + 2] = vaddvq_s32(a2);
        y[o + 3] = vaddvq_s32(a3);
    }
    for (; o < O; ++o) {
        const int8_t * r0 = w + (size_t) o * K;
        int32x4_t a0 = vdupq_n_s32(0);
        for (int k = 0; k < K; k += 16) {
            a0 = vdotq_s32(a0, vld1q_s8(r0 + k), vld1q_s8(x + k));
        }
        y[o] = vaddvq_s32(a0);
    }
}

}  // namespace
}  // namespace pk
