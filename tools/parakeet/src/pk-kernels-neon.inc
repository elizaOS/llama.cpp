// tools/parakeet/src/pk-kernels-neon.inc - NEON pieces shared by the dotprod
// and i8mm translation units (each is compiled with its own -march).
#include <arm_neon.h>

#include <cstring>

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

// `fv5-b5x3t7-v1`. A code x < 125 splits into base-5 digits with
// q1 = x / 5 = ((x + 1) * 51) >> 8 (exact on 0..124: two UMULLs and a UZP2
// of the high bytes), d0 = x - 5 q1, q2 = q1 / 5 from a 32-byte TBL (q1 < 25)
// and d1 = q1 - 5 q2. Each digit becomes a panel vector with the pair16 LUT
// lookup. A group of 8 code vectors makes 24 panel vectors, a multiple of 4,
// so every group starts at row pair 0.
inline uint8x16_t fv5t7_div5(uint8x16_t x) {
    const uint8x16_t x1 = vaddq_u8(x, vdupq_n_u8(1));
    const uint16x8_t lo = vmull_u8(vget_low_u8(x1), vdup_n_u8(51));
    const uint16x8_t hi = vmull_high_u8(x1, vdupq_n_u8(51));
    return vuzp2q_u8(vreinterpretq_u8_u16(lo), vreinterpretq_u8_u16(hi));
}

template <int P0>
inline void fv5t7_emit(const int8x16_t * lut, uint8x16_t row_hi, const uint8x16x2_t & div5_tab, uint8x16_t x,
                       int8_t * dst) {
    const uint8x16_t five = vdupq_n_u8(5);
    const uint8x16_t q1 = fv5t7_div5(x);
    const uint8x16_t q2 = vqtbl2q_u8(div5_tab, q1);
    const uint8x16_t d0 = vmlsq_u8(x, q1, five);
    const uint8x16_t d1 = vmlsq_u8(q1, q2, five);
    vst1q_s8(dst, vqtbl1q_s8(lut[P0 % 4], vorrq_u8(d0, row_hi)));
    vst1q_s8(dst + 16, vqtbl1q_s8(lut[(P0 + 1) % 4], vorrq_u8(d1, row_hi)));
    vst1q_s8(dst + 32, vqtbl1q_s8(lut[(P0 + 2) % 4], vorrq_u8(q2, row_hi)));
}

inline void decode_fv5t7_neon(const uint8_t * codes, const int8_t * luts, int K, int8_t * panel) {
    const int8x16_t lut[4] = { vld1q_s8(luts), vld1q_s8(luts + 16), vld1q_s8(luts + 32), vld1q_s8(luts + 48) };
    static const uint8_t hi8[16] = { 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8, 8, 8, 8 };
    const uint8x16_t row_hi = vld1q_u8(hi8);
    const uint8x16_t m7 = vdupq_n_u8(0x7F);
    static const uint8_t div5[32] = { 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3,
                                      3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6 };
    const uint8x16x2_t div5_tab = { { vld1q_u8(div5), vld1q_u8(div5 + 16) } };
    const size_t pbytes = (size_t) K * 8;
    const int G = (fv5t7_codes(K) + 7) / 8;
    alignas(16) int8_t stage[384];
    for (int g = 0; g < G; ++g) {
        const uint8_t * s = codes + (size_t) g * 112;
        const uint8x16_t v0 = vld1q_u8(s), v1 = vld1q_u8(s + 16), v2 = vld1q_u8(s + 32), v3 = vld1q_u8(s + 48);
        const uint8x16_t v4 = vld1q_u8(s + 64), v5 = vld1q_u8(s + 80), v6 = vld1q_u8(s + 96);
        // code 7 = the top bits of v0..v6 (bit m from v_m)
        uint8x16_t c7 = vsriq_n_u8(v6, v5, 1);
        c7 = vsriq_n_u8(c7, v4, 2);
        c7 = vsriq_n_u8(c7, v3, 3);
        c7 = vsriq_n_u8(c7, v2, 4);
        c7 = vsriq_n_u8(c7, v1, 5);
        c7 = vsriq_n_u8(c7, v0, 6);
        c7 = vshrq_n_u8(c7, 1);
        const size_t off = (size_t) g * 384;
        const bool full = off + 384 <= pbytes;  // the last group can run past K/2 vectors
        int8_t * dst = full ? panel + off : stage;
        fv5t7_emit<0>(lut, row_hi, div5_tab, vandq_u8(v0, m7), dst);
        fv5t7_emit<3>(lut, row_hi, div5_tab, vandq_u8(v1, m7), dst + 48);
        fv5t7_emit<2>(lut, row_hi, div5_tab, vandq_u8(v2, m7), dst + 96);
        fv5t7_emit<1>(lut, row_hi, div5_tab, vandq_u8(v3, m7), dst + 144);
        fv5t7_emit<0>(lut, row_hi, div5_tab, vandq_u8(v4, m7), dst + 192);
        fv5t7_emit<3>(lut, row_hi, div5_tab, vandq_u8(v5, m7), dst + 240);
        fv5t7_emit<2>(lut, row_hi, div5_tab, vandq_u8(v6, m7), dst + 288);
        fv5t7_emit<1>(lut, row_hi, div5_tab, c7, dst + 336);
        if (!full) {
            std::memcpy(panel + off, stage, pbytes - off);
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
