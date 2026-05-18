// # ELIZA-KERNEL-PATCH-V1 — copied verbatim from packages/inference/metal/turbo4.metal
// at build time by build-llama-cpp-dflash.mjs. Do not edit in place;
// edit the standalone source and rerun the build.
//
// turbo4 KV cache dequant + Q·K dot product (Metal Shading Language).
//
// Block layout (block_tbq4_0 in ggml-common.h, 18 bytes, alignment 2):
//     half     norm        // block RMS after TurboQuant preconditioning
//     uchar    qs[16]      // 4-bit indices packed like q4_0
//
// Element decode (matches CPU reference dequantize_row_tbq4_0):
//     elem 0..31 within a 32-element block:
//         qb  = qs[elem & 15]
//         idx = elem < 16 ? (qb & 0xF) : (qb >> 4)
//         rotated = TURBO_CENTROIDS_4BIT[idx] * norm
//     then: y = tbq_uncondition_block(rotated)
//         = k_tbq_signs .* H32(rotated)
//
// By the orthogonality of H32 (FWHT normalized by 1/sqrt(32)) and the
// distributivity of pointwise sign multiply,
//     <q, sign .* H32(r)> = <H32(q .* sign), r>
// so we precompute q_t = H32(q .* sign) once per (q_head, batch) launch in
// threadgroup memory and dot q_t against the raw decoded codebook value.
// This is bit-exactly equivalent to the CPU dequant + dot.
//
// Dispatch: one threadgroup per group of `blocks_per_threadgroup` consecutive
// KV tokens for a single (q_head, batch). Threadgroup size = 32 (one Apple
// SIMD-group). Each thread handles 4 of the 128 elements per token.

#include <metal_stdlib>
using namespace metal;

struct block_turbo4_0 {
    half     norm;
    uint8_t  qs[16];
};

constant float TURBO_CENTROIDS_4BIT[16] = {
    -2.7321365f, -2.0685055f, -1.6175243f, -1.2557391f,
    -0.9419147f, -0.6564307f, -0.3878412f, -0.1283243f,
     0.1283243f,  0.3878412f,  0.6564307f,  0.9419147f,
     1.2557391f,  1.6175243f,  2.0685055f,  2.7321365f,
};

// k_tbq_signs[QK_TBQ=32] from ggml/src/ggml-quants.c:59. Reused per block.
constant float TBQ_SIGNS_32[32] = {
     1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
};

// In-place Fast Walsh–Hadamard transform on a 32-element block, with the
// 1/sqrt(32) normalization that makes H32 orthogonal. Mirrors
// tbq_hadamard32 in ggml/src/ggml-quants.c:104.
static inline void tbq_hadamard32_local(thread float * x) {
    for (uint len = 1; len < 32u; len <<= 1) {
        for (uint i = 0; i < 32u; i += 2u * len) {
            for (uint j = 0; j < len; ++j) {
                float a = x[i + j];
                float b = x[i + j + len];
                x[i + j]       = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    const float norm = 0.1767766952966369f;
    for (uint i = 0; i < 32u; ++i) {
        x[i] *= norm;
    }
}

// Precompute q_t[128] = H32(q .* k_tbq_signs) per 32-element block.
// Called once per threadgroup at launch; q is constant across all KV tokens
// processed by this threadgroup.
//
// Distributed across the 32-thread SIMD-group: threads 0..3 each own one of
// the 4 hadamard-32 blocks. Other threads idle through the barrier.
static inline void eliza_tbq_precompute_qt(
        device const float * q_head,
        threadgroup float  * q_t,
        uint tid) {
    if (tid < 4u) {
        thread float buf[32];
        uint base = tid * 32u;
        for (uint i = 0; i < 32u; ++i) {
            buf[i] = q_head[base + i] * TBQ_SIGNS_32[i];
        }
        tbq_hadamard32_local(buf);
        for (uint i = 0; i < 32u; ++i) {
            q_t[base + i] = buf[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

struct turbo_dot_args {
    uint head_dim;          // must be 128
    uint n_kv;
    uint kv_stride_blocks;  // 4 for d=128 (4 blocks per group)
    uint q_head;
    uint head_offset_bytes; // must be a multiple of sizeof(block_turbo4_0) (18)
};

kernel void kernel_turbo4_dot(
        device const float          * q             [[buffer(0)]],
        device const block_turbo4_0 * k_blocks      [[buffer(1)]],
        device       float          * scores        [[buffer(2)]],
        constant     turbo_dot_args & args          [[buffer(3)]],
        uint                          tid           [[thread_position_in_threadgroup]],
        uint                          kv_idx        [[threadgroup_position_in_grid]]) {
    if (kv_idx >= args.n_kv) return;

    threadgroup float q_t[128];
    device const float * q_head = q + args.q_head * args.head_dim;
    eliza_tbq_precompute_qt(q_head, q_t, tid);

    device const block_turbo4_0 * grp =
        (device const block_turbo4_0 *)((device const uchar *)k_blocks + args.head_offset_bytes)
        + kv_idx * args.kv_stride_blocks;
    uint elem0   = tid * 4;
    uint blk_idx = elem0 >> 5;
    uint within0 = elem0 & 31;
    device const block_turbo4_0 & blk = grp[blk_idx];
    float norm = float(blk.norm);

    float4 qtv = float4(q_t[elem0 + 0], q_t[elem0 + 1], q_t[elem0 + 2], q_t[elem0 + 3]);

    uint qb0 = blk.qs[(within0 + 0u) & 15u];
    uint qb1 = blk.qs[(within0 + 1u) & 15u];
    uint qb2 = blk.qs[(within0 + 2u) & 15u];
    uint qb3 = blk.qs[(within0 + 3u) & 15u];
    bool hi = within0 >= 16u;
    uint idx0 = hi ? (qb0 >> 4) : (qb0 & 0xFu);
    uint idx1 = hi ? (qb1 >> 4) : (qb1 & 0xFu);
    uint idx2 = hi ? (qb2 >> 4) : (qb2 & 0xFu);
    uint idx3 = hi ? (qb3 >> 4) : (qb3 & 0xFu);
    float4 kv = float4(
        TURBO_CENTROIDS_4BIT[idx0],
        TURBO_CENTROIDS_4BIT[idx1],
        TURBO_CENTROIDS_4BIT[idx2],
        TURBO_CENTROIDS_4BIT[idx3]) * norm;
    float acc = dot(qtv, kv);

    float sum = simd_sum(acc);
    if (tid == 0) {
        scores[args.q_head * args.n_kv + kv_idx] = sum;
    }
}

// Multi-block-per-dispatch variant. Same math as kernel_turbo4_dot; the
// threadgroup processes `blocks_per_threadgroup` consecutive KV indices in a
// 32-thread serial loop to amortise dispatch launch tax.
struct turbo_dot_multi_args {
    uint head_dim;
    uint n_kv;
    uint kv_stride_blocks;
    uint q_head;
    uint head_offset_bytes;
    uint blocks_per_threadgroup;
};

kernel void kernel_turbo4_dot_multi(
        device const float          * q             [[buffer(0)]],
        device const block_turbo4_0 * k_blocks      [[buffer(1)]],
        device       float          * scores        [[buffer(2)]],
        constant     turbo_dot_multi_args & args    [[buffer(3)]],
        uint                          tid           [[thread_position_in_threadgroup]],
        uint                          tg_idx        [[threadgroup_position_in_grid]]) {
    threadgroup float q_t[128];
    device const float * q_head = q + args.q_head * args.head_dim;
    eliza_tbq_precompute_qt(q_head, q_t, tid);

    uint elem0   = tid * 4;
    uint blk_idx = elem0 >> 5;
    uint within0 = elem0 & 31;
    float4 qtv = float4(q_t[elem0 + 0], q_t[elem0 + 1], q_t[elem0 + 2], q_t[elem0 + 3]);

    uint kv_base = tg_idx * args.blocks_per_threadgroup;
    for (uint b = 0; b < args.blocks_per_threadgroup; ++b) {
        uint kv_idx = kv_base + b;
        if (kv_idx >= args.n_kv) return;

        device const block_turbo4_0 * grp =
            (device const block_turbo4_0 *)((device const uchar *)k_blocks + args.head_offset_bytes)
            + kv_idx * args.kv_stride_blocks;
        device const block_turbo4_0 & blk = grp[blk_idx];
        float norm = float(blk.norm);

        uint qb0 = blk.qs[(within0 + 0u) & 15u];
        uint qb1 = blk.qs[(within0 + 1u) & 15u];
        uint qb2 = blk.qs[(within0 + 2u) & 15u];
        uint qb3 = blk.qs[(within0 + 3u) & 15u];
        bool hi = within0 >= 16u;
        uint idx0 = hi ? (qb0 >> 4) : (qb0 & 0xFu);
        uint idx1 = hi ? (qb1 >> 4) : (qb1 & 0xFu);
        uint idx2 = hi ? (qb2 >> 4) : (qb2 & 0xFu);
        uint idx3 = hi ? (qb3 >> 4) : (qb3 & 0xFu);
        float4 kv = float4(
            TURBO_CENTROIDS_4BIT[idx0],
            TURBO_CENTROIDS_4BIT[idx1],
            TURBO_CENTROIDS_4BIT[idx2],
            TURBO_CENTROIDS_4BIT[idx3]) * norm;
        float acc = dot(qtv, kv);

        float sum = simd_sum(acc);
        if (tid == 0) {
            scores[args.q_head * args.n_kv + kv_idx] = sum;
        }
    }
}
