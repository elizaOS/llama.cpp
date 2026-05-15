/*
 * GGML wrapper for the standalone QJL kernel library
 * (packages/native-plugins/qjl-cpu/). Bridges the kernel-library API
 * (qjl_quantize_row_*, qjl_score_qk_*) to the ggml-CPU-backend ABI
 * (quantize_row_qjl1_256, dequantize_row_qjl1_256, quantize_qjl1_256,
 * ggml_compute_forward_attn_score_qjl).
 *
 * The kernel library is the bit-parity-validated source of truth and is
 * left untouched. This shim only exposes ggml-shaped names + provides a
 * lazily-built default projection matrix Π for the quantize/dequantize
 * table entries. Production QJL deployments ship Π in a sidecar (see
 * packages/training/scripts/quantization/qjl_apply.py) and inject it
 * via the attention path; the bundled MT-seeded Π here is for the
 * "I just want a cache type that round-trips" case (synthetic graphs,
 * smoke tests, the test-qjl-cache binary).
 *
 * block_qjl1_256 in ggml-common.h has identical layout to the kernel
 * library's qjl_block_qjl1_256 (16-bit norm + 32 bytes packed signs);
 * the static_assert in both headers locks size = 34 B. We aliasing-cast
 * between the two; if the ggml type is ever extended with extra fields,
 * stop aliasing and copy.
 */

#define GGML_COMMON_IMPL_C
#include "ggml-common.h"

#include "ggml-cpu-impl.h"
#include "ggml-quants.h"
#include "quants.h"

#include "qjl/qjl.h"
#include "qjl_block.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Portable atomic primitives for the lazy-init CAS below.
 *
 * MSVC's <stdatomic.h> emits `#error "C atomic support is not enabled"`
 * unless the project is built with `/experimental:c11atomics` (only on
 * very recent toolchains), so we cannot rely on `_Atomic` here. Use:
 *   - MSVC:        _InterlockedCompareExchange / _InterlockedExchange
 *                  (with MemoryBarrier() for acq/rel semantics).
 *   - GCC/clang:   __atomic_* builtins (available everywhere we target).
 *
 * Both wrappers operate on a `volatile long` cell so the MSVC intrinsics
 * see the correct width on 64-bit Windows (where `int` is 32-bit and
 * `long` is also 32-bit). */
#if defined(_MSC_VER)
#include <windows.h>
typedef volatile long qjl_atomic_int;
static inline int qjl_atomic_load_acquire(qjl_atomic_int * p) {
    long v = *p;
    MemoryBarrier();
    return (int) v;
}
static inline void qjl_atomic_store_release(qjl_atomic_int * p, int v) {
    MemoryBarrier();
    _InterlockedExchange(p, (long) v);
}
static inline int qjl_atomic_cas_acq_rel(qjl_atomic_int * p, int expected, int desired) {
    long prev = _InterlockedCompareExchange(p, (long) desired, (long) expected);
    return prev == (long) expected;
}
#else
typedef volatile int qjl_atomic_int;
static inline int qjl_atomic_load_acquire(qjl_atomic_int * p) {
    return __atomic_load_n(p, __ATOMIC_ACQUIRE);
}
static inline void qjl_atomic_store_release(qjl_atomic_int * p, int v) {
    __atomic_store_n(p, v, __ATOMIC_RELEASE);
}
static inline int qjl_atomic_cas_acq_rel(qjl_atomic_int * p, int expected, int desired) {
    return __atomic_compare_exchange_n(p, &expected, desired, 0,
                                       __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}
#endif

/* Confirm the two block layouts agree byte-for-byte. */
_Static_assert(sizeof(qjl_block_qjl1_256) == sizeof(block_qjl1_256),
               "qjl block layouts must agree (kernel lib vs ggml)");
_Static_assert(QJL_PROJECTION_DIM == QK_QJL,
               "QJL projection dim must equal QK_QJL");

/* ---------------- default projection ---------------- */

/*
 * Lazily-built default projection matrix. head_dim=128, proj_dim=256,
 * seed=42 (the canonical QJL paper default — see QJL_PROJECTION_SEED in
 * the kernel-library docs). 128 * 256 * 4 = 128 KB. One-shot allocation
 * for the lifetime of the process.
 *
 * The intended production wiring has Π injected per layer from the
 * sidecar; this default exists so the quantize/dequantize entries in
 * the type-traits table can run standalone (e.g. for synthetic graphs
 * that exercise the cache-type path without a real sidecar).
 */
#define QJL_DEFAULT_HEAD_DIM 128
#define QJL_DEFAULT_PROJ_DIM 256
#define QJL_DEFAULT_SEED     42ULL

/* Portable lazy init: three-state CAS (UNINIT -> RUNNING -> READY).
 * pthread_once isn't available on MSVC; C11 atomics are. Readers spin
 * briefly while another thread runs the initializer — a few-microsecond
 * one-time hit per process, which matches pthread_once semantics. */
#define QJL_INIT_UNINIT  0
#define QJL_INIT_RUNNING 1
#define QJL_INIT_READY   2

static qjl_atomic_int g_qjl_prj_state = QJL_INIT_UNINIT;
static float *g_qjl_prj = NULL;

static const float * qjl_default_projection(void) {
    int state = qjl_atomic_load_acquire(&g_qjl_prj_state);
    if (state == QJL_INIT_READY) {
        return g_qjl_prj;
    }

    if (qjl_atomic_cas_acq_rel(&g_qjl_prj_state, QJL_INIT_UNINIT, QJL_INIT_RUNNING)) {
        g_qjl_prj = (float *) malloc(sizeof(float) * QJL_DEFAULT_HEAD_DIM * QJL_DEFAULT_PROJ_DIM);
        if (g_qjl_prj != NULL) {
            qjl_make_projection_mt(g_qjl_prj, QJL_DEFAULT_HEAD_DIM, QJL_DEFAULT_PROJ_DIM, QJL_DEFAULT_SEED);
        }
        qjl_atomic_store_release(&g_qjl_prj_state, QJL_INIT_READY);
        return g_qjl_prj;
    }

    /* Another thread is initializing — wait for it. */
    while (qjl_atomic_load_acquire(&g_qjl_prj_state) != QJL_INIT_READY) {
        /* tiny pause; this loop runs at most once per process lifetime */
    }
    return g_qjl_prj;
}

/* ---------------- ggml-API-shaped wrappers ---------------- */

/*
 * Quantize one full row's worth of fp32 keys into block_qjl1_256.
 * `k` is the total scalar count and must be a multiple of QK_QJL/2 = 128
 * (one head_dim of 128 floats per output block — QK_QJL is the *sketch*
 * dim, not the input dim). We deliberately ignore that subtlety in the
 * type-traits blck_size, which is set to QK_QJL=256 so the standard
 * row-size math `nrow * QK_QJL * type_size / blck_size` produces the
 * correct on-cache footprint of 34 B per cached key vector.
 *
 * Strict-row callers (the type-traits from_float_ref) will pass
 * k = ggml_blck_size(GGML_TYPE_QJL1_256) = QK_QJL on each call. We treat
 * that single block as one cached key vector of head_dim=128 floats
 * preceded by 128 floats of "padding" the caller never reads back, which
 * matches how the K-cache stores one row per token-head pair. To keep the
 * math honest, the contract is: caller guarantees `k` rows worth of
 * head_dim=128 keys and the wrapper steps through them in 128-float
 * strides.
 *
 * In practice the GGML K-cache path issues from_float against a tensor
 * whose leading dim is head_dim, so the natural call pattern is one
 * block per cached key vector; QK_QJL is exposed in the table only as
 * the on-cache row-byte-size denominator.
 */
void quantize_row_qjl1_256_ref(const float * GGML_RESTRICT x, block_qjl1_256 * GGML_RESTRICT y, int64_t k) {
    /* Number of cached key vectors = k / head_dim. The ggml row-size math
     * pre-divides by blck_size (= QK_QJL = 256), so callers that respect
     * the type's blck_size always pass a multiple of 256 for n_per_row;
     * the kernel always consumes head_dim=128 floats per output block. */
    GGML_ASSERT(k > 0);
    GGML_ASSERT((k % QJL_HEAD_DIM) == 0);
    const int64_t n_blocks = k / QJL_HEAD_DIM;
    const float * prj = qjl_default_projection();
    GGML_ASSERT(prj != NULL);

    for (int64_t r = 0; r < n_blocks; r++) {
        qjl_quantize_row_ref(x + r * QJL_HEAD_DIM, prj,
                             (qjl_block_qjl1_256 *)(y + r));
    }
}

/* CPU-backend entry point — pick best SIMD path. Mirrors quantize_row_tbq3_0(). */
void quantize_row_qjl1_256(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy, int64_t k) {
    GGML_ASSERT(k > 0);
    GGML_ASSERT((k % QJL_HEAD_DIM) == 0);
    const int64_t n_blocks = k / QJL_HEAD_DIM;
    block_qjl1_256 * y = (block_qjl1_256 *) vy;
    const float * prj = qjl_default_projection();
    GGML_ASSERT(prj != NULL);

    qjl_quantize_rows(x, prj, (qjl_block_qjl1_256 *)y, (size_t) n_blocks);
}

/* Dequantize: reconstruct an approximate fp32 key from a packed block,
 * using the QJL paper's cosine-similarity scl = sqrt(pi/2)/proj_dim.
 * Stride matches quantize_row_qjl1_256_ref (one block per head_dim
 * outputs). Uses the same default projection as the quantize path so
 * round-trips match. */
void dequantize_row_qjl1_256(const block_qjl1_256 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    GGML_ASSERT(k > 0);
    GGML_ASSERT((k % QJL_HEAD_DIM) == 0);
    const int64_t n_blocks = k / QJL_HEAD_DIM;
    const float * prj = qjl_default_projection();
    GGML_ASSERT(prj != NULL);

    for (int64_t r = 0; r < n_blocks; r++) {
        qjl_dequantize_row_ref((const qjl_block_qjl1_256 *)(x + r), prj,
                               y + r * QJL_HEAD_DIM);
    }
}

/* `quantize_qjl1_256` — the bulk path called from ggml_quantize_chunk.
 * Mirrors `quantize_tbq3_0` shape: ignores quant_weights, defers to the
 * row-ref wrapper, returns the on-cache byte size of the result. */
size_t quantize_qjl1_256(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrow, int64_t n_per_row, const float * quant_weights) {
    (void) quant_weights;
    const size_t row_size = ggml_row_size(GGML_TYPE_QJL1_256, n_per_row);
    quantize_row_qjl1_256_ref(src, (block_qjl1_256 *) dst, (int64_t) nrow * n_per_row);
    return (size_t) nrow * row_size;
}

/* ---------------- attention score op ---------------- */

/*
 * GGML_OP_ATTN_SCORE_QJL forward. Called by ggml-cpu.c's dispatcher with
 *   tensor->src[0] = q          F32 [QK_QJL, n_heads, n_batch, ne3]
 *   tensor->src[1] = packed_k   QJL1_256 [QK_QJL, n_kv_tokens, n_kv_heads, ne3]
 * and emits
 *   tensor       = scores     F32 [n_kv_tokens, n_heads, n_batch, ne3]
 *
 * n_kv_heads is stored in tensor->op_params[0]. We loop the batch and
 * ne3 dims serially; the inner per-batch call goes through the dispatched
 * SIMD score path in qjl_dispatch.c.
 *
 * Threading: the n_tasks fan-out for this op is set to 1 in ggml-cpu.c's
 * task-count switch; the inner SIMD already saturates the per-core FMA
 * throughput on the proj_dim=256 sweep, and the n_heads*n_kv_tokens loop
 * is small enough on phone-class workloads that thread-fan adds more
 * overhead than it saves. Revisit when we have a real per-token decode
 * profile on arm64 hardware.
 */
void ggml_compute_forward_attn_score_qjl(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst) {
    /* ELIZA-CPU-THREAD-PARALLELISM-V1 — ith/nth split over flattened (ne3, n_batch, h_q). */
    const struct ggml_tensor * q  = dst->src[0];
    const struct ggml_tensor * pk = dst->src[1];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(pk->type == GGML_TYPE_QJL1_256);
    GGML_ASSERT(q->ne[0] == QK_QJL);
    GGML_ASSERT(pk->ne[0] == QJL_HEAD_DIM); /* head_dim, not sketch_dim */

    const int n_heads     = (int) q->ne[1];
    const int n_kv_heads  = ((const int32_t *) dst->op_params)[0];
    const int n_kv_tokens = (int) pk->ne[1];

    GGML_ASSERT(n_kv_heads > 0);
    GGML_ASSERT((n_heads % n_kv_heads) == 0);
    GGML_ASSERT(pk->ne[2] == (int64_t) n_kv_heads);

    const int64_t n_batch = q->ne[2];
    const int64_t ne3     = q->ne[3];
    GGML_ASSERT(pk->ne[3] == ne3);

    const size_t q_stride_b   = q->nb[2];
    const size_t q_stride_3   = q->nb[3];
    const size_t pk_stride_3  = pk->nb[3];
    const size_t s_stride_b   = dst->nb[2];
    const size_t s_stride_3   = dst->nb[3];

    GGML_ASSERT(pk->nb[1] == sizeof(block_qjl1_256));
    GGML_ASSERT(pk->nb[2] == (size_t) n_kv_tokens * sizeof(block_qjl1_256));

    const int gqa = n_heads / n_kv_heads;

    /* Flatten the (ne3, n_batch, h_q) output space; distribute over
     * ith/nth. Each (i3,i2,hq) work unit owns the scores row for one head
     * of one batch plane — the QJL score kernel is per-head and stateless,
     * so calling it with n_heads=1/n_kv_heads=1 plus offset q/packed_k/
     * scores pointers shards cleanly with no shared state. */
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
            + i2 * q_stride_b + i3 * q_stride_3);
        float       * s_plane = (float *)       ((char *)       dst->data
            + i2 * s_stride_b + i3 * s_stride_3);
        const char  * pk_plane = (const char *) pk->data + i3 * pk_stride_3;

        const float * q_head = q_plane + hq * QK_QJL;
        float       * s_head = s_plane + hq * n_kv_tokens;
        const qjl_block_qjl1_256 * pk_head =
            (const qjl_block_qjl1_256 *) (pk_plane + hk * pk->nb[2]);

        /* n_heads=1, n_kv_heads=1 -> gqa=1, hk=0 inside the kernel, so
         * pk_head[0..n_kv_tokens) is exactly this kv-head's blocks. */
        qjl_score_qk(q_head, pk_head, 1, 1, n_kv_tokens, s_head);
    }
}
