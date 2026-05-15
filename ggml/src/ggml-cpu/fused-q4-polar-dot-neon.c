/* fused-q4-polar-dot-neon.c — NEON fused Q4_POLAR x Q8_0 dot product.
 *
 * See fused-q4-polar-dot.c for the algorithm. NEON-specific notes:
 *   - 32 int8 -> 32 fp32 via vmovl_s8 + vmovl_s16 + vcvtq_f32_s32.
 *     Two 4-fp32 vectors per 8 int8 group.
 *   - WHT128: scalar loops for h<4, vfmaq pairs for h>=4. ARMv8 has 32
 *     128-bit Q registers so register pressure is fine but we keep the
 *     128-fp32 buffer in memory for clarity and let the compiler pin
 *     hot tiles in registers.
 *   - Centroid LUT walk: 16-entry float LUT loaded once into 4 q
 *     registers. We do per-pair scalar lookups since vqtbl4q_u8 doesn't
 *     map cleanly to fp32 LUT lookup; the savings on a 64-byte block
 *     would be small vs. the shuffle setup cost.
 */

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include "polar_centroids.h"

#include <arm_neon.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Stage 1+2: dequantize Q8_0 chunk to fp32 with scale folded in.
 * One Q8_0 block = 32 int8 -> 8 q-vectors of fp32. */
static inline void load_q8_chunk_scaled_neon(const int8_t * qs, float scale, float * dst) {
    float32x4_t vscale = vdupq_n_f32(scale);

    for (int half = 0; half < 2; half++) {
        int8x16_t v8 = vld1q_s8(qs + half * 16);
        /* Sign-extend 16 int8 -> 8 int16 lo + 8 int16 hi. */
        int16x8_t v16_lo = vmovl_s8(vget_low_s8(v8));
        int16x8_t v16_hi = vmovl_s8(vget_high_s8(v8));
        /* Each int16x8 -> two int32x4. */
        int32x4_t v32_a = vmovl_s16(vget_low_s16(v16_lo));
        int32x4_t v32_b = vmovl_s16(vget_high_s16(v16_lo));
        int32x4_t v32_c = vmovl_s16(vget_low_s16(v16_hi));
        int32x4_t v32_d = vmovl_s16(vget_high_s16(v16_hi));

        float32x4_t f_a = vmulq_f32(vcvtq_f32_s32(v32_a), vscale);
        float32x4_t f_b = vmulq_f32(vcvtq_f32_s32(v32_b), vscale);
        float32x4_t f_c = vmulq_f32(vcvtq_f32_s32(v32_c), vscale);
        float32x4_t f_d = vmulq_f32(vcvtq_f32_s32(v32_d), vscale);

        vst1q_f32(dst + half * 16,      f_a);
        vst1q_f32(dst + half * 16 + 4,  f_b);
        vst1q_f32(dst + half * 16 + 8,  f_c);
        vst1q_f32(dst + half * 16 + 12, f_d);
    }
}

/* Stage 3: 7-level in-place WHT on a 128-float buffer.
 * Stages h=1,2 are scalar (compiler vectorises adjacent pairs).
 * Stages h=4..64 are 4-lane vfmaq pairs. */
static inline void wht128_neon(float * x) {
    for (int h = 1; h < 4; h <<= 1) {
        for (int i = 0; i < QK_POLAR; i += (h << 1)) {
            for (int j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int h = 4; h < QK_POLAR; h <<= 1) {
        for (int i = 0; i < QK_POLAR; i += (h << 1)) {
            for (int j = 0; j < h; j += 4) {
                float32x4_t a = vld1q_f32(x + i + j);
                float32x4_t b = vld1q_f32(x + i + j + h);
                vst1q_f32(x + i + j,     vaddq_f32(a, b));
                vst1q_f32(x + i + j + h, vsubq_f32(a, b));
            }
        }
    }
}

/* Stage 4: centroid x yhat dot. */
static inline double centroid_dot_neon(const uint8_t * qs, const float * yhat) {
    static const float centroids[16] = {
        -2.754354807e+00f, -2.093562707e+00f, -1.643041510e+00f, -1.279739752e+00f,
        -9.626409783e-01f, -6.723921169e-01f, -3.978971029e-01f, -1.317577823e-01f,
         1.317577823e-01f,  3.978971029e-01f,  6.723921169e-01f,  9.626409783e-01f,
         1.279739752e+00f,  1.643041510e+00f,  2.093562707e+00f,  2.754354807e+00f,
    };

    float32x4_t acc = vdupq_n_f32(0.0f);

    /* Process 4 bytes (= 8 codes = 8 floats) per iteration. */
    for (int i = 0; i < 64; i += 4) {
        float c[8];
        for (int k = 0; k < 4; k++) {
            const uint8_t b = qs[i + k];
            c[2 * k]     = centroids[b & 0x0Fu];
            c[2 * k + 1] = centroids[(b >> 4) & 0x0Fu];
        }
        float32x4_t cv0 = vld1q_f32(c);
        float32x4_t cv1 = vld1q_f32(c + 4);
        float32x4_t yv0 = vld1q_f32(yhat + 2 * i);
        float32x4_t yv1 = vld1q_f32(yhat + 2 * i + 4);
        acc = vfmaq_f32(acc, cv0, yv0);
        acc = vfmaq_f32(acc, cv1, yv1);
    }

    return (double) vaddvq_f32(acc);
}

/* Public NEON entry point. Forward-declared in fused-q4-polar-dot.c
 * (the dispatcher) but the prototype lives here too so gcc with
 * -Wmissing-prototypes doesn't reject the definition. */
double ggml_vec_dot_q4_polar_q8_0_fused_neon(int nb_polar,
                                             const block_q4_polar * x,
                                             const block_q8_0 * y,
                                             bool use_qjl);

double ggml_vec_dot_q4_polar_q8_0_fused_neon(int nb_polar,
                                             const block_q4_polar * x,
                                             const block_q8_0 * y,
                                             bool use_qjl) {
    const int n_q8_per_polar = QK_POLAR / QK8_0; /* = 4 */
    const float inv_d = 1.0f / (float) QK_POLAR;

    float qjl_signs[QK_POLAR];
    if (use_qjl) polar_qjl_signs(qjl_signs);

    float yhat[QK_POLAR] __attribute__((aligned(16)));
    double acc_total = 0.0;

    for (int b = 0; b < nb_polar; b++) {
        const block_q4_polar * xb = x + b;
        const float l2 = GGML_CPU_FP16_TO_FP32(xb->d);
        if (l2 == 0.0f) continue;

        const float global = l2 * inv_d;

        for (int qb = 0; qb < n_q8_per_polar; qb++) {
            const block_q8_0 * yb = y + b * n_q8_per_polar + qb;
            const float scale = GGML_CPU_FP16_TO_FP32(yb->d) * global;
            load_q8_chunk_scaled_neon(yb->qs, scale, yhat + qb * QK8_0);
        }

        wht128_neon(yhat);

        double local = centroid_dot_neon(xb->qs, yhat);

        if (use_qjl) {
            const uint8_t bit = (uint8_t)(xb->qjl[0] & 1u);
            const float sign  = bit ? 1.0f : -1.0f;
            const float mag   = POLAR_QJL_CORRECTION_MAGNITUDE / sqrtf((float)QK_POLAR);
            float32x4_t ra = vdupq_n_f32(0.0f);
            for (int i = 0; i < QK_POLAR; i += 4) {
                float32x4_t q = vld1q_f32(qjl_signs + i);
                float32x4_t yv = vld1q_f32(yhat + i);
                ra = vfmaq_f32(ra, q, yv);
            }
            local += (double)(sign * mag * vaddvq_f32(ra));
        }

        acc_total += local;
    }

    return acc_total;
}

#endif /* __ARM_NEON */
