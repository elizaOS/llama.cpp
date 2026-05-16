/* attn-score-tbq-polar.c — scalar CPU reference for the two custom
 * packed-K attention-score ops that were previously Metal-only:
 *
 *   - GGML_OP_ATTN_SCORE_TBQ    : K is TBQ3_0 / TBQ4_0 / TBQ3_TCQ
 *   - GGML_OP_ATTN_SCORE_POLAR  : K is Q4_POLAR (optional QJL residual)
 *
 * Both ops have the same dataflow as ATTN_SCORE_QJL (see qjl/quants-qjl.c
 * ggml_compute_forward_attn_score_qjl), only the K decoder differs:
 *
 *   q   : F32 [head_dim=128, n_heads,    n_batch, ne3]
 *   pk  : KT  [head_dim=128, n_kv_tokens, n_kv_heads, ne3]
 *   dst : F32 [n_kv_tokens,  n_heads,    n_batch, ne3]
 *
 *   op_params[0]            = n_kv_heads
 *   op_params[1] (POLAR)    = use_qjl   (0/1)
 *   op_params[2] (POLAR)    = q_preht   (0/1)  — query is already H*q
 *
 * For each output cell (i3, i_batch, h_q, t):
 *
 *     scores[t, h_q, i_batch, i3] =
 *         dot( dequant_kT_row(pk[:, t, h_q/gqa, i3]) , q[:, h_q, i_batch, i3] )
 *
 * This file is the CORRECTNESS ORACLE. Performance is irrelevant — the
 * production paths are the Metal kernels (ggml/src/ggml-metal/eliza-shipped/*).
 * The reference exists so test-backend-ops.cpp can compare a backend's
 * output against the CPU result for the same input, the same way every
 * other op is validated.
 *
 * Threading: split over the flattened (ne3, n_batch, h_q) output space
 * exactly like ATTN_SCORE_QJL. Each task owns a disjoint scores row, no
 * shared scratch.
 *
 * Math notes:
 *   - For TBQ types we use the type-traits `to_float` (dequantize_row_*)
 *     to reconstruct the 128-fp32 key vector, then scalar dot against q.
 *   - For Q4_POLAR we inline a per-block decode that takes the per-op
 *     `use_qjl` flag explicitly, instead of using dequantize_row_q4_polar
 *     which reads a global flag (s_q4_polar_use_qjl). The Metal kernel
 *     honours the per-op flag, so the CPU reference must do the same.
 *   - For Q4_POLAR with q_preht=true the query is already H*q. By
 *     orthogonality of H, <H c, y> = <c, H y>, so the natural reference
 *     is the existing `ggml_vec_dot_q4_polar_preht_f32_ref` helper which
 *     does exactly that.
 */

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "quants.h"
#include "simd-mappings.h"

/* PolarQuant helpers — centroids table, QJL signs, Hadamard butterfly. */
#include "polar_centroids.h"
#include "polarquant/polarquant_preht.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>

/* ---------------- Q4_POLAR per-block decode with explicit use_qjl ---------------- */

/* Decode one block_q4_polar into 128 floats. Mirrors
 * dequantize_row_q4_polar in ggml-quants.c but takes use_qjl explicitly
 * instead of reading the global flag. Used by the attn_score_polar
 * reference below, where the flag lives in op_params[1].
 *
 * Bit-for-bit identical to dequantize_row_q4_polar(b=1) when
 * use_qjl == s_q4_polar_use_qjl. The split exists because the op uses
 * a per-call parameter and we do not want to mutate global state. */
static void eliza_dequant_q4_polar_block(
        const block_q4_polar * GGML_RESTRICT src,
        float * GGML_RESTRICT dst,
        bool use_qjl,
        const float * GGML_RESTRICT qjl_signs) {
    const float l2     = GGML_CPU_FP16_TO_FP32(src->d);
    const float inv_d  = 1.0f / (float) QK_POLAR;

    float buf[QK_POLAR];
    for (int i = 0; i < QK_POLAR / 2; i++) {
        const uint8_t byte = src->qs[i];
        const uint8_t lo = (uint8_t)(byte & 0x0Fu);
        const uint8_t hi = (uint8_t)((byte >> 4) & 0x0Fu);
        buf[2 * i]     = POLAR_Q4_CENTROIDS[lo];
        buf[2 * i + 1] = POLAR_Q4_CENTROIDS[hi];
    }

    if (use_qjl) {
        const uint8_t bit = (uint8_t)(src->qjl[0] & 1u);
        const float sign  = bit ? 1.0f : -1.0f;
        const float mag   = POLAR_QJL_CORRECTION_MAGNITUDE / sqrtf((float) QK_POLAR);
        for (int i = 0; i < QK_POLAR; i++) {
            buf[i] += sign * mag * qjl_signs[i];
        }
    }

    polar_hadamard_inplace(buf);
    for (int i = 0; i < QK_POLAR; i++) {
        dst[i] = buf[i] * inv_d * l2;
    }
}

/* ---------------- shared row-driver scaffold ---------------- */

/* Validate the {q, pk, dst} layout shared by both ops and return the
 * per-axis sizes the inner loop needs. Aborts hard on any mismatch so
 * the failure points at the graph builder, not at silent wrong output. */
static void eliza_attn_score_validate(
        const struct ggml_tensor * q,
        const struct ggml_tensor * pk,
        const struct ggml_tensor * dst,
        enum ggml_type expected_pk_type_a,
        enum ggml_type expected_pk_type_b,
        enum ggml_type expected_pk_type_c,
        int  * out_n_heads,
        int  * out_n_kv_heads,
        int  * out_n_kv_tokens,
        int64_t * out_n_batch,
        int64_t * out_ne3) {
    GGML_ASSERT(q->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(pk->type == expected_pk_type_a ||
                pk->type == expected_pk_type_b ||
                pk->type == expected_pk_type_c);
    GGML_ASSERT(q->ne[0]  == 128);
    GGML_ASSERT(pk->ne[0] == 128);

    const int n_heads     = (int) q->ne[1];
    const int n_kv_heads  = ((const int32_t *) dst->op_params)[0];
    const int n_kv_tokens = (int) pk->ne[1];

    GGML_ASSERT(n_kv_heads > 0);
    GGML_ASSERT((n_heads % n_kv_heads) == 0);
    GGML_ASSERT(pk->ne[2] == (int64_t) n_kv_heads);
    GGML_ASSERT(pk->ne[3] == q->ne[3]);
    GGML_ASSERT(dst->ne[0] == (int64_t) n_kv_tokens);
    GGML_ASSERT(dst->ne[1] == (int64_t) n_heads);
    GGML_ASSERT(dst->ne[2] == q->ne[2]);
    GGML_ASSERT(dst->ne[3] == q->ne[3]);

    *out_n_heads     = n_heads;
    *out_n_kv_heads  = n_kv_heads;
    *out_n_kv_tokens = n_kv_tokens;
    *out_n_batch     = q->ne[2];
    *out_ne3         = q->ne[3];
}

/* ---------------- ATTN_SCORE_TBQ forward ---------------- */

void ggml_compute_forward_attn_score_tbq(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct ggml_tensor * q  = dst->src[0];
    const struct ggml_tensor * pk = dst->src[1];

    int n_heads, n_kv_heads, n_kv_tokens;
    int64_t n_batch, ne3;
    eliza_attn_score_validate(q, pk, dst,
        GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_TCQ,
        &n_heads, &n_kv_heads, &n_kv_tokens, &n_batch, &ne3);

    const int gqa = n_heads / n_kv_heads;

    /* Type-traits gives us blck_size + to_float. head_dim=128 must be a
     * whole multiple of blck_size for every supported TBQ type:
     *   TBQ3_0 / TBQ4_0  : blck_size = QK_TBQ = 32  -> 4 blocks per row
     *   TBQ3_TCQ         : blck_size = QK_TBQ3_TCQ = 128 -> 1 block per row
     */
    const struct ggml_type_traits * traits = ggml_get_type_traits(pk->type);
    GGML_ASSERT(traits != NULL);
    GGML_ASSERT(traits->to_float != NULL);
    GGML_ASSERT((128 % traits->blck_size) == 0);

    GGML_ASSERT(pk->nb[1] == ggml_row_size(pk->type, 128));
    GGML_ASSERT(pk->nb[2] == (size_t) n_kv_tokens * pk->nb[1]);

    const size_t pk_row_bytes = pk->nb[1];

    const int64_t n_work = ne3 * n_batch * (int64_t) n_heads;
    const int ith = params->ith;
    const int nth = params->nth;

    for (int64_t w = ith; w < n_work; w += nth) {
        const int64_t hq = w % n_heads;
        const int64_t bi = w / n_heads;          /* i3*n_batch + i2 */
        const int64_t i2 = bi % n_batch;
        const int64_t i3 = bi / n_batch;
        const int64_t hk = hq / gqa;

        const float * q_head = (const float *) ((const char *) q->data
            + i3 * q->nb[3] + i2 * q->nb[2] + hq * q->nb[1]);
        float       * s_head = (float *)       ((char *)       dst->data
            + i3 * dst->nb[3] + i2 * dst->nb[2] + hq * dst->nb[1]);
        const char  * pk_kvh = (const char *) pk->data
            + i3 * pk->nb[3] + hk * pk->nb[2];

        float k_buf[128];
        for (int t = 0; t < n_kv_tokens; t++) {
            const void * pk_row = (const void *) (pk_kvh + (size_t) t * pk_row_bytes);
            traits->to_float(pk_row, k_buf, 128);

            float acc = 0.0f;
            for (int i = 0; i < 128; i++) {
                acc += k_buf[i] * q_head[i];
            }
            s_head[t] = acc;
        }
    }
}

/* ---------------- ATTN_SCORE_POLAR forward ---------------- */

void ggml_compute_forward_attn_score_polar(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    const struct ggml_tensor * q  = dst->src[0];
    const struct ggml_tensor * pk = dst->src[1];

    int n_heads, n_kv_heads, n_kv_tokens;
    int64_t n_batch, ne3;
    eliza_attn_score_validate(q, pk, dst,
        GGML_TYPE_Q4_POLAR, GGML_TYPE_Q4_POLAR, GGML_TYPE_Q4_POLAR,
        &n_heads, &n_kv_heads, &n_kv_tokens, &n_batch, &ne3);

    const int32_t * op_params = (const int32_t *) dst->op_params;
    const bool use_qjl = op_params[1] != 0;
    const bool q_preht = op_params[2] != 0;

    const int gqa = n_heads / n_kv_heads;

    GGML_ASSERT(pk->nb[1] == ggml_row_size(GGML_TYPE_Q4_POLAR, 128));
    GGML_ASSERT(pk->nb[2] == (size_t) n_kv_tokens * pk->nb[1]);

    /* Q4_POLAR has blck_size = QK_POLAR = head_dim = 128, so each cached
     * key vector is exactly one block. */
    GGML_ASSERT(QK_POLAR == 128);
    const size_t pk_row_bytes = pk->nb[1];

    /* QJL sign vector — built once per op, not per (head, token) pair. */
    float qjl_signs[QK_POLAR];
    if (use_qjl && !q_preht) {
        polar_qjl_signs(qjl_signs);
    }

    const int64_t n_work = ne3 * n_batch * (int64_t) n_heads;
    const int ith = params->ith;
    const int nth = params->nth;

    for (int64_t w = ith; w < n_work; w += nth) {
        const int64_t hq = w % n_heads;
        const int64_t bi = w / n_heads;
        const int64_t i2 = bi % n_batch;
        const int64_t i3 = bi / n_batch;
        const int64_t hk = hq / gqa;

        const float * q_head = (const float *) ((const char *) q->data
            + i3 * q->nb[3] + i2 * q->nb[2] + hq * q->nb[1]);
        float       * s_head = (float *)       ((char *)       dst->data
            + i3 * dst->nb[3] + i2 * dst->nb[2] + hq * dst->nb[1]);
        const char  * pk_kvh = (const char *) pk->data
            + i3 * pk->nb[3] + hk * pk->nb[2];

        if (q_preht) {
            /* The query is already H*q. The preht ref helper takes the
             * full strided sequence of blocks in pk_kvh and emits one
             * scalar per block; we feed it the (1 block, 1 score) case
             * per token because pk rows are not contiguous in the
             * single-call expectation (they are, but n=QK_POLAR keeps the
             * loop trivial and matches the helper's existing usage). */
            for (int t = 0; t < n_kv_tokens; t++) {
                const block_q4_polar * pk_row =
                    (const block_q4_polar *) (pk_kvh + (size_t) t * pk_row_bytes);
                float s = 0.0f;
                ggml_vec_dot_q4_polar_preht_f32_ref(QK_POLAR, &s, pk_row, q_head, use_qjl ? 1 : 0);
                s_head[t] = s;
            }
        } else {
            float k_buf[QK_POLAR];
            for (int t = 0; t < n_kv_tokens; t++) {
                const block_q4_polar * pk_row =
                    (const block_q4_polar *) (pk_kvh + (size_t) t * pk_row_bytes);
                eliza_dequant_q4_polar_block(pk_row, k_buf, use_qjl, qjl_signs);

                float acc = 0.0f;
                for (int i = 0; i < QK_POLAR; i++) {
                    acc += k_buf[i] * q_head[i];
                }
                s_head[t] = acc;
            }
        }
    }
}
