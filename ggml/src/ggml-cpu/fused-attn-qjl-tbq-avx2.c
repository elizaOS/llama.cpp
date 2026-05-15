/* fused-attn-qjl-tbq-avx2.c — AVX2 inner kernels for fused QJL+TBQ attn.
 *
 * Provides:
 *   qjl_score_one_avx2(q_sketch, signs)
 *     -> Σ_j sign_bit_j * q_sketch[j]   (proj_dim=256 bits = 32 sign bytes)
 *   fused_attn_v_mix_avx2(n_tokens, w, v_codes, v_scales, out)
 *     -> out[d] += Σ_t w[t] * dequant_V[t][d]   (V stored as TBQ3_0
 *        blocks, 4 per token, head_dim=128)
 */

#if defined(__AVX2__)

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"
#include "fused-attn-qjl-tbq.h"

#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#define FUSED_QJL_PROJ_DIM   256
#define FUSED_QJL_PACKED_BYTES 32
#define FUSED_TBQ_BLOCK 32
#define FUSED_QJL_HEAD_DIM 128

/* Reused from qjl_score_avx2.c: byte -> 8 fp32 +/-1 lanes. */
static inline __m256 expand_signs_byte(uint8_t b) {
    __m256i v = _mm256_set1_epi32((int) b);
    const __m256i bits = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
    __m256i andv = _mm256_and_si256(v, bits);
    __m256i mask = _mm256_cmpeq_epi32(andv, bits);
    __m256 ones    = _mm256_set1_ps(1.0f);
    __m256 negones = _mm256_set1_ps(-1.0f);
    return _mm256_blendv_ps(negones, ones, _mm256_castsi256_ps(mask));
}

float qjl_score_one_avx2(const float * q_sketch, const uint8_t * signs) {
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < FUSED_QJL_PACKED_BYTES; b++) {
        __m256 sgn = expand_signs_byte(signs[b]);
        __m256 q   = _mm256_loadu_ps(q_sketch + b * 8);
        acc = _mm256_fmadd_ps(sgn, q, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 v  = _mm_add_ps(lo, hi);
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    return _mm_cvtss_f32(v);
}

/* TBQ helpers replicated locally to avoid pulling them out of the
 * static-inline block in ggml-quants.c. */
static const float k_tbq3_codebook_avx2[8] = {
    -2.1519457f, -1.3439093f, -0.7560053f, -0.2450942f,
     0.2450942f,  0.7560053f,  1.3439093f,  2.1519457f,
};

static const int8_t k_tbq_signs_avx2[FUSED_TBQ_BLOCK] = {
     1, -1,  1,  1, -1,  1, -1, -1,
     1,  1, -1,  1, -1, -1,  1, -1,
    -1,  1,  1, -1,  1, -1, -1,  1,
     1, -1,  1, -1, -1,  1, -1,  1,
};

static inline uint8_t tbq3_get_code_avx2(const uint8_t * qs, int idx) {
    const int bit = idx * 3;
    const int byte = bit >> 3;
    const int shift = bit & 7;
    uint32_t bits = qs[byte] >> shift;
    if (shift > 5 && byte + 1 < (int) (FUSED_TBQ_BLOCK * 3 / 8)) {
        bits |= (uint32_t) qs[byte + 1] << (8 - shift);
    }
    return bits & 0x7u;
}

static inline void tbq_hadamard32_avx2(float * x) {
    /* Stages h=1,2: scalar. */
    for (int len = 1; len < 4; len <<= 1) {
        for (int i = 0; i < FUSED_TBQ_BLOCK; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + len];
                x[i + j]       = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    /* Stages h=4,8,16: ymm pairs. */
    for (int len = 4; len < FUSED_TBQ_BLOCK; len <<= 1) {
        for (int i = 0; i < FUSED_TBQ_BLOCK; i += 2 * len) {
            for (int j = 0; j < len; j += 8) {
                if (j + 8 <= len) {
                    __m256 a = _mm256_loadu_ps(x + i + j);
                    __m256 b = _mm256_loadu_ps(x + i + j + len);
                    _mm256_storeu_ps(x + i + j,       _mm256_add_ps(a, b));
                    _mm256_storeu_ps(x + i + j + len, _mm256_sub_ps(a, b));
                } else {
                    /* Tail (len=4): 4 lanes only. Use scalar. */
                    for (int jj = j; jj < len; jj++) {
                        const float a = x[i + jj];
                        const float b = x[i + jj + len];
                        x[i + jj]       = a + b;
                        x[i + jj + len] = a - b;
                    }
                }
            }
        }
    }
    const __m256 vnorm = _mm256_set1_ps(0.1767766952966369f);
    for (int i = 0; i < FUSED_TBQ_BLOCK; i += 8) {
        __m256 v = _mm256_loadu_ps(x + i);
        _mm256_storeu_ps(x + i, _mm256_mul_ps(v, vnorm));
    }
}

static inline void tbq3_decode_block_avx2(const uint8_t * qs, float d, float * out) {
    if (d == 0.0f) {
        memset(out, 0, FUSED_TBQ_BLOCK * sizeof(float));
        return;
    }
    for (int i = 0; i < FUSED_TBQ_BLOCK; i++) {
        out[i] = d * k_tbq3_codebook_avx2[tbq3_get_code_avx2(qs, i)];
    }
    tbq_hadamard32_avx2(out);
    for (int i = 0; i < FUSED_TBQ_BLOCK; i += 8) {
        __m256 v = _mm256_loadu_ps(out + i);
        __m256 s;
        /* Convert 8 int8 signs to 8 fp32. */
        __m128i s8  = _mm_loadl_epi64((const __m128i *)(k_tbq_signs_avx2 + i));
        __m256i s32 = _mm256_cvtepi8_epi32(s8);
        s = _mm256_cvtepi32_ps(s32);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(v, s));
    }
}

void fused_attn_v_mix_avx2(int n_tokens, const float * w,
                           const uint8_t * v_codes, const uint16_t * v_scales,
                           float * out) {
    /* For each token, dequant 4 tbq3_0 blocks and FMA-add into out. */
    for (int t = 0; t < n_tokens; t++) {
        const float wt = w[t];
        if (wt == 0.0f) continue;
        const __m256 vw = _mm256_set1_ps(wt);
        for (int c = 0; c < FUSED_QJL_HEAD_DIM / FUSED_TBQ_BLOCK; c++) {
            const int blk = t * (FUSED_QJL_HEAD_DIM / FUSED_TBQ_BLOCK) + c;
            const uint8_t * codes = v_codes + (size_t) blk * (FUSED_TBQ_BLOCK * 3 / 8);
            const float d = GGML_CPU_FP16_TO_FP32(v_scales[blk]);
            GGML_ALIGN(32) float decoded[FUSED_TBQ_BLOCK];
            tbq3_decode_block_avx2(codes, d, decoded);
            float * out_chunk = out + c * FUSED_TBQ_BLOCK;
            for (int i = 0; i < FUSED_TBQ_BLOCK; i += 8) {
                __m256 ov = _mm256_loadu_ps(out_chunk + i);
                __m256 dv = _mm256_loadu_ps(decoded + i);
                _mm256_storeu_ps(out_chunk + i, _mm256_fmadd_ps(vw, dv, ov));
            }
        }
    }
}

#endif /* __AVX2__ */

/* Avoid ISO C "empty translation unit" pedantic error when __AVX2__ is undefined. */
typedef int fused_attn_qjl_tbq_avx2_iso_c_tu_stub;
