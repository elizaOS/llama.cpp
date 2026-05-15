/* fused-attn-qjl-tbq-neon.c — NEON inner kernels for fused QJL+TBQ attn.
 *
 * See fused-attn-qjl-tbq.c for algorithm. NEON-specific notes:
 *   - QJL score: same nibble-expand strategy as qjl_score_neon.c
 *   - V mix: tbq3_decode + WHT-uncondition + sign flip + weighted add,
 *     all in 4-lane fp32. The 32-element WHT is laid out as 8 groups of
 *     4-lane vectors; the inner butterfly stage h=1 stays scalar.
 */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include <arm_neon.h>
#include <math.h>
#include <stdalign.h>
#include <stdint.h>
#include <string.h>

#define FUSED_QJL_PROJ_DIM   256
#define FUSED_QJL_PACKED_BYTES 32
#define FUSED_TBQ_BLOCK 32
#define FUSED_QJL_HEAD_DIM 128

static inline float32x4_t expand_signs_nibble(uint8_t b, int bit_offset) {
    uint32_t w0 = 1u << (bit_offset + 0);
    uint32_t w1 = 1u << (bit_offset + 1);
    uint32_t w2 = 1u << (bit_offset + 2);
    uint32_t w3 = 1u << (bit_offset + 3);
    uint32_t warr[4] = {w0, w1, w2, w3};
    uint32x4_t weights = vld1q_u32(warr);
    uint32x4_t bv      = vdupq_n_u32(b);
    uint32x4_t andv    = vandq_u32(bv, weights);
    uint32x4_t mask    = vceqq_u32(andv, weights);
    float32x4_t one  = vdupq_n_f32(1.0f);
    float32x4_t none = vdupq_n_f32(-1.0f);
    return vbslq_f32(mask, one, none);
}

float qjl_score_one_neon(const float * q_sketch, const uint8_t * signs) {
    float32x4_t acc = vdupq_n_f32(0.0f);
    for (int b = 0; b < FUSED_QJL_PACKED_BYTES; b++) {
        uint8_t byte = signs[b];
        float32x4_t s_lo = expand_signs_nibble(byte, 0);
        float32x4_t s_hi = expand_signs_nibble(byte, 4);
        float32x4_t q_lo = vld1q_f32(q_sketch + b * 8);
        float32x4_t q_hi = vld1q_f32(q_sketch + b * 8 + 4);
        acc = vfmaq_f32(acc, s_lo, q_lo);
        acc = vfmaq_f32(acc, s_hi, q_hi);
    }
    return vaddvq_f32(acc);
}

/* TBQ helpers replicated locally. */
static const float k_tbq3_codebook_neon[8] = {
    -2.1519457f, -1.3439093f, -0.7560053f, -0.2450942f,
     0.2450942f,  0.7560053f,  1.3439093f,  2.1519457f,
};

static const float k_tbq_signs_f32_neon[FUSED_TBQ_BLOCK] = {
     1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
};

static inline uint8_t tbq3_get_code_neon(const uint8_t * qs, int idx) {
    const int bit = idx * 3;
    const int byte = bit >> 3;
    const int shift = bit & 7;
    uint32_t bits = qs[byte] >> shift;
    if (shift > 5 && byte + 1 < (int) (FUSED_TBQ_BLOCK * 3 / 8)) {
        bits |= (uint32_t) qs[byte + 1] << (8 - shift);
    }
    return bits & 0x7u;
}

static inline void tbq_hadamard32_neon(float * x) {
    /* h=1 scalar */
    for (int i = 0; i < FUSED_TBQ_BLOCK; i += 2) {
        const float a = x[i];
        const float b = x[i + 1];
        x[i]     = a + b;
        x[i + 1] = a - b;
    }
    /* h=2 scalar (4 elem stride doesn't fit q-vector cleanly) */
    for (int i = 0; i < FUSED_TBQ_BLOCK; i += 4) {
        const float a0 = x[i];
        const float a1 = x[i + 1];
        const float b0 = x[i + 2];
        const float b1 = x[i + 3];
        x[i]     = a0 + b0;
        x[i + 1] = a1 + b1;
        x[i + 2] = a0 - b0;
        x[i + 3] = a1 - b1;
    }
    /* h>=4 vectorized */
    for (int len = 4; len < FUSED_TBQ_BLOCK; len <<= 1) {
        for (int i = 0; i < FUSED_TBQ_BLOCK; i += 2 * len) {
            for (int j = 0; j < len; j += 4) {
                float32x4_t a = vld1q_f32(x + i + j);
                float32x4_t b = vld1q_f32(x + i + j + len);
                vst1q_f32(x + i + j,       vaddq_f32(a, b));
                vst1q_f32(x + i + j + len, vsubq_f32(a, b));
            }
        }
    }
    const float32x4_t vnorm = vdupq_n_f32(0.1767766952966369f);
    for (int i = 0; i < FUSED_TBQ_BLOCK; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vst1q_f32(x + i, vmulq_f32(v, vnorm));
    }
}

static inline void tbq3_decode_block_neon(const uint8_t * qs, float d, float * out) {
    if (d == 0.0f) {
        memset(out, 0, FUSED_TBQ_BLOCK * sizeof(float));
        return;
    }
    for (int i = 0; i < FUSED_TBQ_BLOCK; i++) {
        out[i] = d * k_tbq3_codebook_neon[tbq3_get_code_neon(qs, i)];
    }
    tbq_hadamard32_neon(out);
    for (int i = 0; i < FUSED_TBQ_BLOCK; i += 4) {
        float32x4_t v = vld1q_f32(out + i);
        float32x4_t s = vld1q_f32(k_tbq_signs_f32_neon + i);
        vst1q_f32(out + i, vmulq_f32(v, s));
    }
}

void fused_attn_v_mix_neon(int n_tokens, const float * w,
                           const uint8_t * v_codes, const uint16_t * v_scales,
                           float * out) {
    for (int t = 0; t < n_tokens; t++) {
        const float wt = w[t];
        if (wt == 0.0f) continue;
        const float32x4_t vw = vdupq_n_f32(wt);
        for (int c = 0; c < FUSED_QJL_HEAD_DIM / FUSED_TBQ_BLOCK; c++) {
            const int blk = t * (FUSED_QJL_HEAD_DIM / FUSED_TBQ_BLOCK) + c;
            const uint8_t * codes = v_codes + (size_t) blk * (FUSED_TBQ_BLOCK * 3 / 8);
            const float d = GGML_CPU_FP16_TO_FP32(v_scales[blk]);
            alignas(16) float decoded[FUSED_TBQ_BLOCK];
            tbq3_decode_block_neon(codes, d, decoded);
            float * out_chunk = out + c * FUSED_TBQ_BLOCK;
            for (int i = 0; i < FUSED_TBQ_BLOCK; i += 4) {
                float32x4_t ov = vld1q_f32(out_chunk + i);
                float32x4_t dv = vld1q_f32(decoded + i);
                vst1q_f32(out_chunk + i, vfmaq_f32(ov, vw, dv));
            }
        }
    }
}

#endif /* __ARM_NEON */
