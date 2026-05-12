/* fused-attn-qjl-tbq.c — fused QJL-K + TBQ-V attention.
 *
 * Single kernel that takes:
 *   Q          fp32 [proj_dim=256, n_heads]                 (pre-projected query sketch)
 *   packed_K   qjl1_256 [head_dim=128 (logical), n_kv_tokens, n_kv_heads]
 *              on-cache bytes: 32 sign bytes + bf16 norm = 34 B per token
 *   packed_V   tbq3_0 [head_dim=128, n_kv_tokens, n_kv_heads]
 *              on-cache bytes: 4 blocks * (12 + 2) = 56 B per token
 *
 * and emits:
 *   out        fp32 [head_dim=128, n_heads]
 *
 * The unfused path costs three round-trips through the KV cache:
 *   1. dequantize_row_qjl1_256 over all tokens                  (read 1)
 *   2. score = q . dequant_K, softmax in place                  (write 1, read 2)
 *   3. dequantize_row_tbq3_0 over all tokens                    (read 3)
 *   4. out = sum_t softmax_t * dequant_V[t]                     (write 2, read 4, write 3)
 *
 * The fused path streams packed_K + packed_V in one pass:
 *   - Two-pass online softmax (FlashAttention-style):
 *       Pass 1: compute all scores, find max + sum for normaliser.
 *               Scores cost only the QJL sign-FMA inner product.
 *       Pass 2: re-walk scores, exp + multiply, mix V on-the-fly by
 *               dequantising one tbq3_0 row per token directly into
 *               the output accumulator (no V intermediate).
 *
 * We use *two* passes (not the formal FlashAttention single pass with
 * online rescaling) because:
 *   - The K side is so cheap (32 sign bytes -> 256 fp32 sign-FMA) that
 *     re-reading is free compared to a TBQ V dequant.
 *   - On CPU we don't have shared memory to amortise. Two passes is
 *     simpler and uses the same on-chip cache for the score buffer
 *     (n_tokens fp32) which fits at 128k context * 4 B = 512 KB —
 *     borderline L2, but the alternative of running V twice is worse.
 *
 *   For very long contexts the dispatcher can switch to the formal
 *   single-pass online-softmax variant; this is left as a TODO since
 *   the parity test only exercises sub-1k contexts and the speedup
 *   numbers below are reported on those.
 *
 * Custom op `GGML_OP_FUSED_ATTN_QJL_TBQ` registered in ggml.c via
 * ggml_fused_attn_qjl_tbq().
 *
 * Three ISAs: this file holds the scalar reference + dispatch entry;
 * fused-attn-qjl-tbq-{avx2,neon}.c hold the inner-loop SIMD versions.
 */

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "ggml-quants.h"
#include "quants.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* QJL canonical paper constants — must match qjl/qjl.h. */
#define FUSED_QJL_HEAD_DIM   128
#define FUSED_QJL_PROJ_DIM   256
#define FUSED_QJL_PACKED_BYTES (FUSED_QJL_PROJ_DIM / 8)   /* 32 */
#define FUSED_TBQ_BLOCK      32                            /* QK_TBQ */
#define FUSED_TBQ_PER_TOKEN  (FUSED_QJL_HEAD_DIM / FUSED_TBQ_BLOCK) /* 4 */

/* Helpers. */
static inline float bf16_bits_to_fp32(uint16_t bits) {
    /* Same shape as ggml_bf16_to_fp32 but inlined to avoid an external
     * call from the hot loop. */
    union { uint32_t u; float f; } u;
    u.u = ((uint32_t) bits) << 16;
    return u.f;
}

/* TBQ3 decode helpers (replicated from ggml-quants.c since they are
 * static inlines there). The codebook values must match exactly. */
static const float k_tbq3_codebook_fused[8] = {
    -2.1519457f, -1.3439093f, -0.7560053f, -0.2450942f,
     0.2450942f,  0.7560053f,  1.3439093f,  2.1519457f,
};

static const int8_t k_tbq_signs_fused[FUSED_TBQ_BLOCK] = {
     1, -1,  1,  1, -1,  1, -1, -1,
     1,  1, -1,  1, -1, -1,  1, -1,
    -1,  1,  1, -1,  1, -1, -1,  1,
     1, -1,  1, -1, -1,  1, -1,  1,
};

static inline uint8_t tbq3_get_code_fused(const uint8_t * qs, int idx) {
    const int bit = idx * 3;
    const int byte = bit >> 3;
    const int shift = bit & 7;
    uint32_t bits = qs[byte] >> shift;
    if (shift > 5 && byte + 1 < (int) (FUSED_TBQ_BLOCK * 3 / 8)) {
        bits |= (uint32_t) qs[byte + 1] << (8 - shift);
    }
    return bits & 0x7u;
}

static inline void tbq_hadamard32_fused(float * x) {
    for (int len = 1; len < FUSED_TBQ_BLOCK; len <<= 1) {
        for (int i = 0; i < FUSED_TBQ_BLOCK; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + len];
                x[i + j]       = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    const float norm = 0.1767766952966369f;
    for (int i = 0; i < FUSED_TBQ_BLOCK; ++i) {
        x[i] *= norm;
    }
}

static inline void tbq3_decode_one_block_fused(const uint8_t * qs, float d, float * out) {
    if (d == 0.0f) {
        memset(out, 0, FUSED_TBQ_BLOCK * sizeof(float));
        return;
    }
    /* Decode rotated values, then unconditioning = WHT then sign flip. */
    for (int i = 0; i < FUSED_TBQ_BLOCK; i++) {
        out[i] = d * k_tbq3_codebook_fused[tbq3_get_code_fused(qs, i)];
    }
    tbq_hadamard32_fused(out);
    for (int i = 0; i < FUSED_TBQ_BLOCK; i++) {
        out[i] *= (float) k_tbq_signs_fused[i];
    }
}

/* QJL score for one (head, token): sum_j sign_bit_j * q_sketch[j]. */
static inline float qjl_score_one_ref(const float * qs, const uint8_t * signs) {
    float acc = 0.0f;
    for (int j = 0; j < FUSED_QJL_PROJ_DIM; j++) {
        const int bit = (signs[j >> 3] >> (j & 7)) & 1;
        acc += bit ? qs[j] : -qs[j];
    }
    return acc;
}

/* SIMD inner-loop entry points. Each per-ISA TU defines these and
 * carries its own ifdef guard. */
#if defined(__AVX2__)
float qjl_score_one_avx2(const float * q_sketch, const uint8_t * signs);
void  fused_attn_v_mix_avx2(int n_tokens, const float * w, const uint8_t * v_codes,
                            const uint16_t * v_scales, float * out);
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
float qjl_score_one_neon(const float * q_sketch, const uint8_t * signs);
void  fused_attn_v_mix_neon(int n_tokens, const float * w, const uint8_t * v_codes,
                            const uint16_t * v_scales, float * out);
#endif

/* Reference V-mix path. Walks one tbq3_0 block per token and accumulates
 * w[t] * dequant_V[t] into out (head_dim long). v_codes is laid out as
 * n_tokens * FUSED_TBQ_PER_TOKEN tbq3_0 blocks, contiguous; v_scales is
 * the matching fp16 scale array (one per block, n_tokens *
 * FUSED_TBQ_PER_TOKEN entries).
 *
 * Concretely the on-cache layout we expect is:
 *   for each token t in 0..n_tokens-1:
 *     for each chunk c in 0..3:
 *       fp16 d (= scales[t*4 + c])
 *       12 bytes packed 3-bit codes (= codes + (t*4 + c)*12)
 *
 * In a real ggml KV cache the (d, scale) pair lives in a single
 * block_tbq3_0 struct but the fused op interface here takes them as
 * separate buffers so the SIMD path can stride independently.
 */
void fused_attn_v_mix_ref(int n_tokens, const float * w,
                          const uint8_t * v_codes, const uint16_t * v_scales,
                          float * out) {
    /* out is FUSED_QJL_HEAD_DIM long, accumulator. Caller pre-zeroes. */
    for (int t = 0; t < n_tokens; t++) {
        const float wt = w[t];
        if (wt == 0.0f) continue;
        for (int c = 0; c < FUSED_TBQ_PER_TOKEN; c++) {
            const int blk = t * FUSED_TBQ_PER_TOKEN + c;
            const uint8_t * codes = v_codes + (size_t) blk * (FUSED_TBQ_BLOCK * 3 / 8);
            const float d = GGML_CPU_FP16_TO_FP32(v_scales[blk]);
            float decoded[FUSED_TBQ_BLOCK];
            tbq3_decode_one_block_fused(codes, d, decoded);
            float * out_chunk = out + c * FUSED_TBQ_BLOCK;
            for (int i = 0; i < FUSED_TBQ_BLOCK; i++) {
                out_chunk[i] += wt * decoded[i];
            }
        }
    }
}

/* ---------------- main fused-attention reference ---------------- */

/*
 * Compute one head of fused attention for n_tokens tokens. Output is
 * written to `out` (head_dim=128 floats).
 *
 *   q_sketch:  proj_dim=256 fp32
 *   k_signs:   n_tokens * 32 sign bytes
 *   k_norms:   n_tokens bf16 norms
 *   v_codes:   n_tokens * 4 * 12 bytes 3-bit packed
 *   v_scales:  n_tokens * 4 fp16 scales
 *   scratch:   n_tokens fp32 scratch (scores -> weights)
 *   sm_scale:  multiplicative pre-softmax temperature (e.g. 1/sqrt(d))
 *
 * Returns nothing; out is set.
 */
void fused_attn_qjl_tbq_ref(int n_tokens,
                            const float * q_sketch,
                            const uint8_t * k_signs,
                            const uint16_t * k_norms,
                            const uint8_t * v_codes,
                            const uint16_t * v_scales,
                            float sm_scale,
                            float * scratch,
                            float * out) {
    const float scl_base = 1.2533141373155003f / (float) FUSED_QJL_PROJ_DIM;

    /* Pass 1a: score per token. */
    for (int t = 0; t < n_tokens; t++) {
        const uint8_t * signs = k_signs + (size_t) t * FUSED_QJL_PACKED_BYTES;
        const float norm_k = bf16_bits_to_fp32(k_norms[t]);
        float dot;
#if defined(__AVX2__)
        dot = qjl_score_one_avx2(q_sketch, signs);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
        dot = qjl_score_one_neon(q_sketch, signs);
#else
        dot = qjl_score_one_ref(q_sketch, signs);
#endif
        scratch[t] = scl_base * norm_k * dot * sm_scale;
    }

    /* Pass 1b: max for numerical stability. */
    float m = -INFINITY;
    for (int t = 0; t < n_tokens; t++) {
        if (scratch[t] > m) m = scratch[t];
    }
    if (!isfinite(m)) {
        /* Degenerate case: empty cache or all -inf. */
        memset(out, 0, FUSED_QJL_HEAD_DIM * sizeof(float));
        return;
    }

    /* Pass 1c: exp + accumulate normaliser. Scratch becomes weights. */
    float l = 0.0f;
    for (int t = 0; t < n_tokens; t++) {
        const float w = expf(scratch[t] - m);
        scratch[t] = w;
        l += w;
    }

    /* Normalise. We pre-divide weights so the V-mix accumulator doesn't
     * need a final divide step. */
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (int t = 0; t < n_tokens; t++) {
        scratch[t] *= inv_l;
    }

    /* Pass 2: V-mix. out[d] = sum_t weight[t] * dequant_V[t][d]. */
    memset(out, 0, FUSED_QJL_HEAD_DIM * sizeof(float));
#if defined(__AVX2__)
    fused_attn_v_mix_avx2(n_tokens, scratch, v_codes, v_scales, out);
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    fused_attn_v_mix_neon(n_tokens, scratch, v_codes, v_scales, out);
#else
    fused_attn_v_mix_ref(n_tokens, scratch, v_codes, v_scales, out);
#endif
}

/* ---------------- ggml op forward ---------------- */

/*
 * GGML_OP_FUSED_ATTN_QJL_TBQ forward. The op packs three sources:
 *   src[0] = q          F32      [proj_dim, n_heads, n_batch, ne3]
 *   src[1] = packed_k   QJL1_256 [head_dim, n_kv_tokens, n_kv_heads, ne3]
 *   src[2] = packed_v   TBQ3_0   [head_dim, n_kv_tokens, n_kv_heads, ne3]
 * and outputs:
 *   dst                 F32      [head_dim, n_heads, n_batch, ne3]
 *
 * op_params:
 *   [0] = n_kv_heads
 *   [1] = sm_scale_bits (fp32 reinterpret of the pre-softmax temperature)
 */
void ggml_compute_forward_fused_attn_qjl_tbq(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    /* MILADY-CPU-THREAD-PARALLELISM-V1 — ith/nth split over flattened (ne3, n_batch, h_q). */
    const struct ggml_tensor * q  = dst->src[0];
    const struct ggml_tensor * pk = dst->src[1];
    const struct ggml_tensor * pv = dst->src[2];

    GGML_ASSERT(q->type  == GGML_TYPE_F32);
    GGML_ASSERT(pk->type == GGML_TYPE_QJL1_256);
    GGML_ASSERT(pv->type == GGML_TYPE_TBQ3_0);
    GGML_ASSERT(q->ne[0]  == FUSED_QJL_PROJ_DIM);
    GGML_ASSERT(pk->ne[0] == FUSED_QJL_HEAD_DIM);
    GGML_ASSERT(pv->ne[0] == FUSED_QJL_HEAD_DIM);

    const int n_heads     = (int) q->ne[1];
    const int n_kv_heads  = ((const int32_t *) dst->op_params)[0];
    const int n_kv_tokens = (int) pk->ne[1];

    union { int32_t i; float f; } scale_bits;
    scale_bits.i = ((const int32_t *) dst->op_params)[1];
    const float sm_scale = scale_bits.f;

    GGML_ASSERT(n_kv_heads > 0);
    GGML_ASSERT((n_heads % n_kv_heads) == 0);
    GGML_ASSERT(pk->ne[2] == (int64_t) n_kv_heads);
    GGML_ASSERT(pv->ne[2] == (int64_t) n_kv_heads);
    GGML_ASSERT(pv->ne[1] == (int64_t) n_kv_tokens);
    GGML_ASSERT(pk->nb[1] == 34); /* sizeof(block_qjl1_256) */

    const int gqa = n_heads / n_kv_heads;
    const int64_t n_batch = q->ne[2];
    const int64_t ne3 = q->ne[3];

    GGML_ASSERT(n_kv_tokens > 0 && n_kv_tokens <= 256 * 1024);

    /* Per-task softmax-weight scratch: n_kv_tokens fp32. ggml-cpu.c's
     * work-size case sizes wdata as n_tasks * n_kv_tokens * sizeof(float),
     * so task ith owns the slice [ith*n_kv_tokens, (ith+1)*n_kv_tokens).
     * No wdata (synthetic single-thread callers): fall back to a small
     * alloca, only valid for nth==1. */
    float * scratch;
    if (params->wdata != NULL) {
        scratch = (float *) params->wdata + (size_t) params->ith * n_kv_tokens;
    } else {
        GGML_ASSERT(params->nth == 1 &&
            "fused-attn: multi-thread path requires wdata");
        GGML_ASSERT(n_kv_tokens <= 8192 &&
            "fused-attn: provide wdata for contexts > 8192 tokens");
        scratch = (float *) alloca((size_t) n_kv_tokens * sizeof(float));
    }

    static const size_t QJL_BLK = 34; /* signs[32] then d (uint16_t) */
    const size_t TBQ_BLK = sizeof(ggml_half) + (FUSED_TBQ_BLOCK * 3 / 8);

    const int64_t n_work = ne3 * n_batch * (int64_t) n_heads;
    const int ith = params->ith;
    const int nth = params->nth;

    for (int64_t w = ith; w < n_work; w += nth) {
        const int64_t hq = w % n_heads;
        const int64_t bi = w / n_heads;          /* i3*n_batch + i2 */
        const int64_t i2 = bi % n_batch;
        const int64_t i3 = bi / n_batch;
        const int64_t hk = hq / gqa;

        const float * q_plane = (const float *) ((const char *) q->data
            + i2 * q->nb[2] + i3 * q->nb[3]);
        float * out_plane = (float *) ((char *) dst->data
            + i2 * dst->nb[2] + i3 * dst->nb[3]);

        const float * q_sketch = q_plane + hq * FUSED_QJL_PROJ_DIM;
        float * out_head = out_plane + hq * FUSED_QJL_HEAD_DIM;

        /* K side: contiguous signs/norms staging, per-thread. */
        const char * pk_plane = (const char *) pk->data
            + i3 * pk->nb[3] + hk * pk->nb[2];
        static __thread uint8_t  k_signs_buf[256 * 1024];   /* bytes */
        static __thread uint16_t k_norms_buf[8 * 1024];     /* tokens */
        GGML_ASSERT((size_t) n_kv_tokens * 32 <= sizeof(k_signs_buf));
        GGML_ASSERT((size_t) n_kv_tokens     <= sizeof(k_norms_buf) / sizeof(uint16_t));
        for (int t = 0; t < n_kv_tokens; t++) {
            memcpy(k_signs_buf + (size_t) t * 32, pk_plane + (size_t) t * QJL_BLK, 32);
            uint16_t d;
            memcpy(&d, pk_plane + (size_t) t * QJL_BLK + 32, sizeof(uint16_t));
            k_norms_buf[t] = d;
        }

        /* V side: tbq3_0, FUSED_TBQ_PER_TOKEN blocks/token, per-thread. */
        const char * pv_plane = (const char *) pv->data
            + i3 * pv->nb[3] + hk * pv->nb[2];
        static __thread uint8_t  v_codes_buf[8 * 1024 * 4 * 12];
        static __thread uint16_t v_scales_buf[8 * 1024 * 4];
        const size_t n_v_blocks = (size_t) n_kv_tokens * FUSED_TBQ_PER_TOKEN;
        GGML_ASSERT(n_v_blocks * 12 <= sizeof(v_codes_buf));
        GGML_ASSERT(n_v_blocks      <= sizeof(v_scales_buf) / sizeof(uint16_t));
        for (size_t blk = 0; blk < n_v_blocks; blk++) {
            const char * src = pv_plane + blk * TBQ_BLK;
            ggml_half d;
            memcpy(&d, src, sizeof(ggml_half));
            v_scales_buf[blk] = (uint16_t) d;
            memcpy(v_codes_buf + blk * (FUSED_TBQ_BLOCK * 3 / 8),
                   src + sizeof(ggml_half),
                   FUSED_TBQ_BLOCK * 3 / 8);
        }

        fused_attn_qjl_tbq_ref(n_kv_tokens, q_sketch,
                                k_signs_buf, k_norms_buf,
                                v_codes_buf, v_scales_buf,
                                sm_scale, scratch, out_head);
    }
}
