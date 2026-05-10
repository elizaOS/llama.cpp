/* PolarQuant Q4 shared LUT + WHT helpers.
 *
 * Lloyd-Max optimal scalar quantizer for X ~ N(0, 1), 16 levels.
 * Bit-exact mirror of
 *   packages/training/scripts/quantization/polarquant/polar_quant.py
 *     ::_compute_lloyd_max_centroids(n_levels=16, n_iter=100).
 *
 * The centroid table is auto-generated; do not edit by hand. The
 * Walsh-Hadamard butterfly and QJL sign generator live here as well so
 * both ggml-base (ref kernels in ggml-quants.c) and ggml-cpu
 * (quantize_row_q4_polar, ggml_vec_dot_q4_polar_q8_0 in quants-polar.c)
 * can share a single implementation without a cross-library link edge.
 *
 * Vendored from
 *   packages/native-plugins/polarquant-cpu/.
 */
#ifndef GGML_POLAR_CENTROIDS_H
#define GGML_POLAR_CENTROIDS_H

#include <stdint.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#ifdef __cplusplus
extern "C" {
#endif

#define POLAR_Q4_N_LEVELS 16

/* Centroid values (the 16 reconstruction points for N(0,1)). */
static const float POLAR_Q4_CENTROIDS[POLAR_Q4_N_LEVELS] = {
    -2.754354807e+00f,
    -2.093562707e+00f,
    -1.643041510e+00f,
    -1.279739752e+00f,
    -9.626409783e-01f,
    -6.723921169e-01f,
    -3.978971029e-01f,
    -1.317577823e-01f,
     1.317577823e-01f,
     3.978971029e-01f,
     6.723921169e-01f,
     9.626409783e-01f,
     1.279739752e+00f,
     1.643041510e+00f,
     2.093562707e+00f,
     2.754354807e+00f
};

/* Voronoi cell boundaries (15 internal cuts; the outer two
 * cells extend to +/- infinity). Used by the encoder for
 * O(log N) bucketization instead of an O(N) argmin sweep.
 */
static const float POLAR_Q4_BOUNDARIES[POLAR_Q4_N_LEVELS - 1] = {
    -2.423958757e+00f,
    -1.868302108e+00f,
    -1.461390631e+00f,
    -1.121190365e+00f,
    -8.175165476e-01f,
    -5.351446099e-01f,
    -2.648274426e-01f,
     4.996003611e-16f,
     2.648274426e-01f,
     5.351446099e-01f,
     8.175165476e-01f,
     1.121190365e+00f,
     1.461390631e+00f,
     1.868302108e+00f,
     2.423958757e+00f
};

/* QJL residual config -- mirrors PolarQuant defaults. */
#define POLAR_QJL_CORRECTION_MAGNITUDE 0.5f
#define POLAR_QJL_SEED 42u

/* In-place Walsh-Hadamard butterfly of size QK_POLAR. Self-inverse
 * up to a uniform 1/sqrt(QK_POLAR) factor; we fold that into the
 * 1/QK_POLAR compensation that runs after the dequantize-side butterfly.
 *
 * Inlined here so both ggml-base and ggml-cpu get the same code without
 * a cross-translation-unit dependency.
 */
static inline void polar_hadamard_inplace(float * x) {
    for (int h = 1; h < QK_POLAR; h <<= 1) {
        for (int i = 0; i < QK_POLAR; i += (h << 1)) {
            for (int j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
}

/* Deterministic +/-1 sign vector for the QJL residual. xorshift32 seeded
 * by POLAR_QJL_SEED so encoder and decoder agree byte-for-byte without
 * depending on torch/numpy PRNG ABI.
 */
static inline void polar_qjl_signs(float * out) {
    uint32_t state = POLAR_QJL_SEED;
    if (state == 0u) state = 1u;
    for (int i = 0; i < QK_POLAR; i++) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        out[i] = (state & 1u) ? 1.0f : -1.0f;
    }
}

/* Linear-scan bucketize against POLAR_Q4_BOUNDARIES. Returns 0..15. */
static inline uint8_t polar_q4_bucketize(float v) {
    uint8_t code = 0;
    for (int i = 0; i < POLAR_Q4_N_LEVELS - 1; i++) {
        if (v > POLAR_Q4_BOUNDARIES[i]) {
            code = (uint8_t)(i + 1);
        }
    }
    return code;
}

#ifdef __cplusplus
}
#endif

#endif /* GGML_POLAR_CENTROIDS_H */
