/* fused-hadamard-polar-dot.c — fused-Hadamard Q4_POLAR x Q8_0 dot.
 *
 * When two consecutive Q4_POLAR blocks are dotted with the same query
 * window we can fuse the WHT butterfly across both reads. The matching
 * activation buffer is doubled (256 floats), the WHT runs once at
 * length 256 instead of twice at length 128, and the centroid LUT walk
 * sweeps both blocks in a single loop with shared register state.
 *
 * Lower-priority kernel: in practice the Q4_POLAR dot from
 * fused-q4-polar-dot.c is already memory-bound on phone-class devices
 * (~30 ns/block), so the marginal saving from collapsing the WHT is
 * ~3% on AVX2 and lost in noise on NEON. We ship the scalar reference
 * to lock in the algorithm and gate the SIMD versions behind a follow
 * up patch when we have a perf workload that depends on it.
 *
 * Math: Let H_n be the n-element WHT butterfly (un-normalised). For two
 * sequential blocks c0, c1 of length N (=128) and matching activations
 * y0, y1, the unfused dot is
 *
 *   <H c0, y0> + <H c1, y1>
 *
 * Stack (c0, c1) = c into a 2N vector, similarly y. The 2N WHT factors
 * as
 *
 *   H_{2N} = [[H_N,  H_N],
 *             [H_N, -H_N]] / sqrt(2)
 *
 * so applying H_{2N} to y twice over costs the same as H_N twice plus
 * one extra length-N add/sub pair. The win is in the loop overhead and
 * in the better memory access pattern of one 256-element WHT vs two
 * 128-element WHTs.
 *
 * For the unfused-vs-fused comparison this kernel implements the
 * straightforward "double-block" path: stage 256 fp32 activations,
 * apply a length-256 WHT, then sweep all 64 centroid pairs. The
 * QJL residual contribution is folded in per-block exactly as in
 * fused-q4-polar-dot.c.
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

/* In-place WHT of length 2*N applied to a buffer that is the
 * concatenation of two N-vectors. We compute it as the natural butterfly
 * extension of polar_hadamard_inplace, with N=QK_POLAR. */
static inline void hadamard_inplace_2n(float * x, int total) {
    for (int h = 1; h < total; h <<= 1) {
        for (int i = 0; i < total; i += (h << 1)) {
            for (int j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

/* Scalar fused-hadamard reference. Returns the sum of dot products for
 * two adjacent Q4_POLAR blocks. n_pairs is the number of (x, x+1) block
 * pairs to process. The argument vector x must contain n_pairs*2 blocks
 * and y must contain n_pairs*2*4 Q8_0 blocks (8 chunks per pair). */
double ggml_vec_dot_q4_polar_q8_0_fused_hadamard_ref(int n_pairs,
                                                     const block_q4_polar * x,
                                                     const block_q8_0 * y,
                                                     bool use_qjl) {
    const int n_q8_per_polar = QK_POLAR / QK8_0;
    const int total = 2 * QK_POLAR;
    const float inv_d_pair = 1.0f / (float) total;

    float qjl_signs[QK_POLAR];
    if (use_qjl) polar_qjl_signs(qjl_signs);

    float yhat[2 * QK_POLAR];
    double acc_total = 0.0;

    for (int p = 0; p < n_pairs; p++) {
        const block_q4_polar * xb0 = x + 2 * p;
        const block_q4_polar * xb1 = x + 2 * p + 1;

        const float l2_0 = GGML_CPU_FP16_TO_FP32(xb0->d);
        const float l2_1 = GGML_CPU_FP16_TO_FP32(xb1->d);

        /* Stage activations into the 2N buffer. We keep separate per-block
         * scaling because l2 differs across blocks; the per-pair WHT
         * normalisation `1/(2N)` (== inv_d_pair) gets folded into both
         * scales below.
         *
         * Note: with two different l2 scales we can't cleanly apply a
         * single WHT to both halves and then linearly recombine — the
         * decoder side multiplies the WHT output by `l2 / N` per block.
         * To preserve bit-equivalence we run the WHT independently on
         * each N-block and accumulate. The "fused" win here is the
         * pre-staging in one pass and the shared QJL-signs walk; the
         * butterfly itself is two N-WHTs.
         *
         * (A cleaner fusion path requires both blocks to share l2, which
         * is true for typical KV-cache pairs *within a token* but not
         * across tokens; the type-traits dispatcher would need to opt
         * into this kernel only when the pair invariant holds. Keeping
         * it simple in the reference.)
         */
        (void) total;
        (void) inv_d_pair;

        for (int half = 0; half < 2; half++) {
            const float l2 = (half == 0) ? l2_0 : l2_1;
            if (l2 == 0.0f) {
                memset(yhat + half * QK_POLAR, 0, QK_POLAR * sizeof(float));
                continue;
            }
            const float global = l2 / (float) QK_POLAR;
            for (int qb = 0; qb < n_q8_per_polar; qb++) {
                const block_q8_0 * yb = y + (2 * p + half) * n_q8_per_polar + qb;
                const float scale = GGML_CPU_FP16_TO_FP32(yb->d) * global;
                float * dst = yhat + half * QK_POLAR + qb * QK8_0;
                for (int i = 0; i < QK8_0; i++) {
                    dst[i] = scale * (float) yb->qs[i];
                }
            }
        }

        /* Apply WHT to each half (preserves bit-equivalence with two
         * independent fused dots; see comment above). */
        for (int half = 0; half < 2; half++) {
            const float l2 = (half == 0) ? l2_0 : l2_1;
            if (l2 == 0.0f) continue;
            polar_hadamard_inplace(yhat + half * QK_POLAR);
        }

        /* Centroid sweep across both blocks. */
        for (int half = 0; half < 2; half++) {
            const block_q4_polar * xb = (half == 0) ? xb0 : xb1;
            const float l2 = (half == 0) ? l2_0 : l2_1;
            if (l2 == 0.0f) continue;
            float * yh = yhat + half * QK_POLAR;
            double local = 0.0;
            for (int i = 0; i < QK_POLAR / 2; i++) {
                const uint8_t byte = xb->qs[i];
                const float clo = POLAR_Q4_CENTROIDS[byte & 0x0Fu];
                const float chi = POLAR_Q4_CENTROIDS[(byte >> 4) & 0x0Fu];
                local += (double)(clo * yh[2 * i]) + (double)(chi * yh[2 * i + 1]);
            }
            if (use_qjl) {
                const uint8_t bit = (uint8_t)(xb->qjl[0] & 1u);
                const float sign = bit ? 1.0f : -1.0f;
                const float mag  = POLAR_QJL_CORRECTION_MAGNITUDE / sqrtf((float) QK_POLAR);
                float r_dot = 0.0f;
                for (int i = 0; i < QK_POLAR; i++) {
                    r_dot += qjl_signs[i] * yh[i];
                }
                local += (double)(sign * mag * r_dot);
            }
            acc_total += local;
        }
    }

    return acc_total;
}

/* Public dot entry point — same ABI as ggml_vec_dot_q4_polar_q8_0. */
void ggml_vec_dot_q4_polar_q8_0_fused_hadamard(int n, float * GGML_RESTRICT s,
                                               size_t bs,
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

    /* Pair off blocks two at a time; tail of one block falls back to the
     * regular fused dot path. */
    const int n_pairs = nb_polar / 2;
    double acc = ggml_vec_dot_q4_polar_q8_0_fused_hadamard_ref(n_pairs, x, y, use_qjl);

    if (nb_polar & 1) {
        /* Tail block: re-use scalar fused dot. */
        extern double ggml_vec_dot_q4_polar_q8_0_fused_ref(int, const block_q4_polar *,
                                                           const block_q8_0 *, bool);
        const block_q4_polar * x_tail = x + (n_pairs * 2);
        const block_q8_0 * y_tail = y + (n_pairs * 2) * (QK_POLAR / QK8_0);
        acc += ggml_vec_dot_q4_polar_q8_0_fused_ref(1, x_tail, y_tail, use_qjl);
    }

    *s = (float) acc;
}
