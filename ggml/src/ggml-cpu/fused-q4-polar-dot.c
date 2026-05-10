/* fused-q4-polar-dot.c — fused Q4_POLAR x Q8_0 dot product.
 *
 * Today `ggml_vec_dot_q4_polar_q8_0` (quants-polar.c) materialises the
 * dequantized PolarQuant block in a 128-float scratch buffer per block,
 * then sweeps the 4 matching Q8_0 activation blocks. The dequantize
 * pipeline is a 5-step chain:
 *
 *   1. unpack 4-bit codes -> centroid LUT
 *   2. (optional) add QJL residual sign correction along PRNG sign vector
 *   3. inverse Walsh-Hadamard butterfly (in place, log2(128)=7 levels)
 *   4. divide by QK_POLAR (compensation for the un-normalised butterfly)
 *   5. multiply by per-block fp16 L2 norm
 *
 * The fused kernel exploits a single observation: the WHT is an
 * orthogonal linear operator, so for any vectors c, y of length N
 *
 *     <H c, y> = <c, H^T y> = <c, H y>          (H symmetric, self-inverse up to scale)
 *
 * (H_normalised is its own inverse; the un-normalised butterfly used by
 * polar_hadamard_inplace is symmetric. See polar_centroids.h for the
 * butterfly definition.)
 *
 * Strategy:
 *   - Read Q8_0's 4 blocks (4 * 32 = 128 int8 + 4 fp16 scales) into a
 *     fp32 scratch of size 128, with each chunk pre-multiplied by its
 *     scale and the global L2/QK_POLAR factor folded in.
 *   - Apply the WHT to the 128-float scratch ONCE (on the *activation*
 *     side, so it can be vectorised with no per-element control flow).
 *   - Stream centroids directly into a fma accumulator: for each pair of
 *     packed nibbles, look up two centroid values and dot against the
 *     transformed activation pair.
 *   - For QJL residual correction: instead of touching every centroid in
 *     a separate pass, fold the correction `mag * sign * sign_vec[i]`
 *     into the activation buffer BEFORE the WHT (linearity:
 *     <c + r, H y> = <c, H y> + <r, H y>, but pre-WHT it's just one
 *     extra inner-product term).
 *
 * Bit-equivalence: this kernel is mathematically identical to the
 * unfused dequant->dot path, modulo associativity reordering of
 * floating-point sums. The parity test in tests/test-fused-kernels.cpp
 * checks max-relative-error <= 1e-5 across 100 random blocks.
 *
 * Three ISAs in three files:
 *   fused-q4-polar-dot.c            scalar + dispatch entry point
 *   fused-q4-polar-dot-avx2.c       AVX2 inner loop
 *   fused-q4-polar-dot-neon.c       NEON inner loop
 */

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include "polar_centroids.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* SIMD + scalar fused-dot entry points. The scalar version is
 * declared here so the parity test can call it directly without
 * pulling in the full ABI wrapper. SIMD definitions are guarded by
 * __AVX2__ / __ARM_NEON and provide their own external linkage. */
double ggml_vec_dot_q4_polar_q8_0_fused_ref(int nb_polar,
                                            const block_q4_polar * x,
                                            const block_q8_0 * y,
                                            bool use_qjl);
#if defined(__AVX2__)
double ggml_vec_dot_q4_polar_q8_0_fused_avx2(int nb_polar,
                                             const block_q4_polar * x,
                                             const block_q8_0 * y,
                                             bool use_qjl);
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
double ggml_vec_dot_q4_polar_q8_0_fused_neon(int nb_polar,
                                             const block_q4_polar * x,
                                             const block_q8_0 * y,
                                             bool use_qjl);
#endif

/* ---------------- scalar fused reference ---------------- */

/*
 * Scalar fused reference. This is the correctness oracle for the SIMD
 * paths. Mirror of ggml_vec_dot_q4_polar_q8_0 but with the activation-
 * side WHT trick applied so we never materialise the 128-float dequant.
 */
double ggml_vec_dot_q4_polar_q8_0_fused_ref(int nb_polar,
                                            const block_q4_polar * x,
                                            const block_q8_0 * y,
                                            bool use_qjl) {
    const int n_q8_per_polar = QK_POLAR / QK8_0; /* = 4 */

    /* QJL sign vector — built once per call, not per block. */
    float qjl_signs[QK_POLAR];
    if (use_qjl) {
        polar_qjl_signs(qjl_signs);
    }

    const float inv_d = 1.0f / (float)QK_POLAR;
    double acc_total = 0.0;

    /* Per-block fused activation buffer — 128 fp32. Reused across blocks. */
    float yhat[QK_POLAR];

    for (int b = 0; b < nb_polar; b++) {
        const block_q4_polar * xb = x + b;
        const float l2 = GGML_CPU_FP16_TO_FP32(xb->d);

        /* If the per-block norm is zero, dequant emits zeros and the dot
         * contribution is zero. Skip the WHT + LUT walk entirely. */
        if (l2 == 0.0f) continue;

        /* Stage 1: build the activation-side fp32 vector with all
         * post-WHT scaling baked in.
         *
         *   y_raw[i]  = scale_chunk * int8_code      (per-Q8_0-block scale)
         *   yhat = WHT(y_raw) * (l2 * inv_d)
         *
         * We fold `l2 * inv_d` into the per-chunk scale before the WHT
         * since it's a uniform multiplier across the row. */
        const float global = l2 * inv_d;
        for (int qb = 0; qb < n_q8_per_polar; qb++) {
            const block_q8_0 * yb = y + b * n_q8_per_polar + qb;
            const float scale = GGML_CPU_FP16_TO_FP32(yb->d) * global;
            float * dst = yhat + qb * QK8_0;
            for (int i = 0; i < QK8_0; i++) {
                dst[i] = scale * (float)yb->qs[i];
            }
        }

        /* Stage 2: in-place WHT on the activation side. */
        polar_hadamard_inplace(yhat);

        /* Stage 3: stream centroids into the accumulator. We unpack one
         * byte (= two centroid indices) at a time. */
        double local = 0.0;
        for (int i = 0; i < QK_POLAR / 2; i++) {
            const uint8_t byte = xb->qs[i];
            const float clo = POLAR_Q4_CENTROIDS[byte & 0x0Fu];
            const float chi = POLAR_Q4_CENTROIDS[(byte >> 4) & 0x0Fu];
            local += (double)(clo * yhat[2 * i]) + (double)(chi * yhat[2 * i + 1]);
        }

        /* Stage 4: optional QJL residual contribution.
         *
         * The residual adds `sign * mag * qjl_signs[i]` to centroid i
         * BEFORE the inverse WHT. By linearity:
         *
         *   <H(c + r), y_raw> = <c, H y_raw> + <r, H y_raw>
         *
         * but here `yhat = H y_raw * (l2 * inv_d)` already, so:
         *
         *   residual_dot = sign * mag * sum_i qjl_signs[i] * yhat[i]
         */
        if (use_qjl) {
            const uint8_t bit = (uint8_t)(xb->qjl[0] & 1u);
            const float sign  = bit ? 1.0f : -1.0f;
            const float mag   = POLAR_QJL_CORRECTION_MAGNITUDE / sqrtf((float)QK_POLAR);
            float r_dot = 0.0f;
            for (int i = 0; i < QK_POLAR; i++) {
                r_dot += qjl_signs[i] * yhat[i];
            }
            local += (double)(sign * mag * r_dot);
        }

        acc_total += local;
    }

    return acc_total;
}

/* ---------------- dispatch entry point ---------------- */

/*
 * Public fused replacement for ggml_vec_dot_q4_polar_q8_0. Same ABI as
 * the existing dot functions in quants.h: scalar n + s + opaque vx/vy
 * pointers + nrc=1.
 *
 * Picks the best ISA at runtime via __AVX2__ / __ARM_NEON build guards;
 * a future patch can swap to the cpufeats dynamic dispatch table.
 */
void ggml_vec_dot_q4_polar_q8_0_fused(int n, float * GGML_RESTRICT s, size_t bs,
                                      const void * GGML_RESTRICT vx, size_t bx,
                                      const void * GGML_RESTRICT vy, size_t by,
                                      int nrc) {
    GGML_UNUSED(bs); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(nrc);
    GGML_ASSERT(n % QK_POLAR == 0);
    GGML_ASSERT(nrc == 1);

    const block_q4_polar * x = (const block_q4_polar *) vx;
    const block_q8_0     * y = (const block_q8_0 *)     vy;
    const int nb_polar = n / QK_POLAR;
    const bool use_qjl = ggml_q4_polar_get_use_qjl();

    double acc;
#if defined(__AVX2__)
    acc = ggml_vec_dot_q4_polar_q8_0_fused_avx2(nb_polar, x, y, use_qjl);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    acc = ggml_vec_dot_q4_polar_q8_0_fused_neon(nb_polar, x, y, use_qjl);
#else
    acc = ggml_vec_dot_q4_polar_q8_0_fused_ref(nb_polar, x, y, use_qjl);
#endif

    *s = (float) acc;
}
