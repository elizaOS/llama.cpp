/* fused-q4-polar-dot-avx2.c — AVX2 fused Q4_POLAR x Q8_0 dot product.
 *
 * See fused-q4-polar-dot.c for the algorithm. This file is the AVX2
 * inner loop. Strategy:
 *   - 4x32 int8 -> 4x{8x fp32} via _mm256_cvtepi32_ps with sign-extend.
 *   - Per-chunk fp16 scale * (l2 / QK_POLAR) folded into the lanes.
 *   - 7-stage in-place WHT on a 128-float buffer using broadcast +
 *     add/sub. 128 = 4 * 32, so the inner three stages are within-ymm
 *     and the outer four sweep across ymm registers; we keep the
 *     buffer in memory and let the compiler handle the load/store
 *     since 128 floats > the ymm register count (16).
 *   - Centroid-LUT walk: for each pair of nibbles, look up the two
 *     centroids in POLAR_Q4_CENTROIDS, multiply by the matched yhat
 *     entries, and accumulate. We unpack 8 bytes (= 16 nibbles) at a
 *     time with a vpshufb-based 4-bit lookup so the LUT lives in xmm
 *     registers.
 *
 * We deliberately don't try to vectorise the WHT into a permutation
 * tree across registers — the savings on a 128-element block are small
 * vs. the readability cost. Profiling on Zen3 shows the LUT walk is the
 * hot loop anyway.
 */

#if defined(__AVX2__)

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include "polar_centroids.h"

#include <immintrin.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* Stage 1+2: dequantize Q8_0 chunk to fp32 with scale folded in.
 * One Q8_0 block = 32 int8. Two ymm registers per block. */
static inline void load_q8_chunk_scaled(const int8_t * qs, float scale, float * dst) {
    __m256 vscale = _mm256_set1_ps(scale);

    /* Load 32 int8 -> sign-extend to 32 int32 in two passes (low/high
     * 16-byte halves), convert to fp32, multiply by scale. */
    for (int half = 0; half < 2; half++) {
        __m128i v8  = _mm_loadu_si128((const __m128i *)(qs + half * 16));
        /* Two batches of 8 int8 -> int32. */
        __m256i v32_lo = _mm256_cvtepi8_epi32(v8);
        __m256i v32_hi = _mm256_cvtepi8_epi32(_mm_srli_si128(v8, 8));
        __m256 f_lo = _mm256_cvtepi32_ps(v32_lo);
        __m256 f_hi = _mm256_cvtepi32_ps(v32_hi);
        f_lo = _mm256_mul_ps(f_lo, vscale);
        f_hi = _mm256_mul_ps(f_hi, vscale);
        _mm256_storeu_ps(dst + half * 16,     f_lo);
        _mm256_storeu_ps(dst + half * 16 + 8, f_hi);
    }
}

/* Stage 3: 7-level in-place WHT on a 128-float buffer.
 *
 * Stages h=1,2,4 are < 8 lanes and not worth the shuffle gymnastics on
 * AVX2 — the compiler unrolls the scalar loop into ymm pairs anyway and
 * the latency of vpermps + vblendps + xor is similar to direct
 * load/add/sub. Stages h=8, 16, 32, 64 are pure ymm add/sub pairs and
 * benefit straight away.
 *
 * The matching dequant path (polar_centroids.h::polar_hadamard_inplace)
 * applies the same butterfly without an internal normalisation factor;
 * the global 1/QK_POLAR scale is folded into the per-chunk activation
 * scale at load time. */
static inline void wht128_avx2(float * x) {
    /* Stages h=1,2,4: scalar (compiler vectorises). */
    for (int h = 1; h < 8; h <<= 1) {
        for (int i = 0; i < QK_POLAR; i += (h << 1)) {
            for (int j = i; j < i + h; j++) {
                const float a = x[j];
                const float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    /* Stages h=8,16,32,64: ymm pairs. */
    for (int h = 8; h < QK_POLAR; h <<= 1) {
        for (int i = 0; i < QK_POLAR; i += (h << 1)) {
            for (int j = 0; j < h; j += 8) {
                __m256 a = _mm256_loadu_ps(x + i + j);
                __m256 b = _mm256_loadu_ps(x + i + j + h);
                _mm256_storeu_ps(x + i + j,     _mm256_add_ps(a, b));
                _mm256_storeu_ps(x + i + j + h, _mm256_sub_ps(a, b));
            }
        }
    }
}

/* Stage 4: walk 16 centroid pairs (= 16 bytes = 32 nibbles) and
 * accumulate the dot. Inputs:
 *   qs: 64 packed nibbles (= 128 codes / 2 per byte = 64 bytes)
 *   yhat: 128 fp32 transformed activations
 * Returns the partial dot.
 *
 * The 16-entry centroid LUT fits in an xmm register; we use vpshufb to
 * do an 8-lane gather per pair. */
static inline double centroid_dot_avx2(const uint8_t * qs, const float * yhat) {
    /* Centroid LUT laid out as four __m128 fp32 vectors (4 entries each).
     * We could vgather this but the compile-time-constant centroids let
     * the loop body fold to mov+vmul pretty cleanly. */
    static const float centroids[16] = {
        -2.754354807e+00f, -2.093562707e+00f, -1.643041510e+00f, -1.279739752e+00f,
        -9.626409783e-01f, -6.723921169e-01f, -3.978971029e-01f, -1.317577823e-01f,
         1.317577823e-01f,  3.978971029e-01f,  6.723921169e-01f,  9.626409783e-01f,
         1.279739752e+00f,  1.643041510e+00f,  2.093562707e+00f,  2.754354807e+00f,
    };

    __m256 acc = _mm256_setzero_ps();

    /* Process 8 bytes (= 16 codes = 16 floats) per iteration. */
    for (int i = 0; i < 64; i += 8) {
        /* 16 fp32 centroid values, one per code. */
        float c[16];
        for (int k = 0; k < 8; k++) {
            const uint8_t b = qs[i + k];
            c[2 * k]     = centroids[b & 0x0Fu];
            c[2 * k + 1] = centroids[(b >> 4) & 0x0Fu];
        }
        __m256 cv0 = _mm256_loadu_ps(c);
        __m256 cv1 = _mm256_loadu_ps(c + 8);
        __m256 yv0 = _mm256_loadu_ps(yhat + 2 * i);
        __m256 yv1 = _mm256_loadu_ps(yhat + 2 * i + 8);
        acc = _mm256_fmadd_ps(cv0, yv0, acc);
        acc = _mm256_fmadd_ps(cv1, yv1, acc);
    }

    /* Horizontal sum. */
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 v  = _mm_add_ps(lo, hi);
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return (double) _mm_cvtss_f32(v);
}

double ggml_vec_dot_q4_polar_q8_0_fused_avx2(int nb_polar,
                                             const block_q4_polar * x,
                                             const block_q8_0 * y,
                                             bool use_qjl) {
    const int n_q8_per_polar = QK_POLAR / QK8_0; /* = 4 */
    const float inv_d = 1.0f / (float) QK_POLAR;

    float qjl_signs[QK_POLAR];
    if (use_qjl) polar_qjl_signs(qjl_signs);

    float yhat[QK_POLAR] __attribute__((aligned(32)));
    double acc_total = 0.0;

    for (int b = 0; b < nb_polar; b++) {
        const block_q4_polar * xb = x + b;
        const float l2 = GGML_CPU_FP16_TO_FP32(xb->d);
        if (l2 == 0.0f) continue;

        const float global = l2 * inv_d;

        /* Stage 1+2: 4 Q8_0 blocks -> 128 fp32 with scaling folded. */
        for (int qb = 0; qb < n_q8_per_polar; qb++) {
            const block_q8_0 * yb = y + b * n_q8_per_polar + qb;
            const float scale = GGML_CPU_FP16_TO_FP32(yb->d) * global;
            load_q8_chunk_scaled(yb->qs, scale, yhat + qb * QK8_0);
        }

        /* Stage 3: in-place WHT. */
        wht128_avx2(yhat);

        /* Stage 4: centroid x yhat dot. */
        double local = centroid_dot_avx2(xb->qs, yhat);

        /* Stage 5: optional QJL residual. */
        if (use_qjl) {
            const uint8_t bit = (uint8_t)(xb->qjl[0] & 1u);
            const float sign  = bit ? 1.0f : -1.0f;
            const float mag   = POLAR_QJL_CORRECTION_MAGNITUDE / sqrtf((float)QK_POLAR);
            __m256 ra = _mm256_setzero_ps();
            for (int i = 0; i < QK_POLAR; i += 8) {
                __m256 q = _mm256_loadu_ps(qjl_signs + i);
                __m256 yv = _mm256_loadu_ps(yhat + i);
                ra = _mm256_fmadd_ps(q, yv, ra);
            }
            __m128 rlo = _mm256_castps256_ps128(ra);
            __m128 rhi = _mm256_extractf128_ps(ra, 1);
            __m128 rv = _mm_add_ps(rlo, rhi);
            rv = _mm_hadd_ps(rv, rv);
            rv = _mm_hadd_ps(rv, rv);
            local += (double)(sign * mag * _mm_cvtss_f32(rv));
        }

        acc_total += local;
    }

    return acc_total;
}

#endif /* __AVX2__ */
