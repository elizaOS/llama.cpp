// # ELIZA-KERNEL-PATCH-V1 — copied verbatim from packages/inference/metal/turbo3.metal
// at build time by build-llama-cpp-dflash.mjs. Do not edit in place;
// edit the standalone source and rerun the build.
//
// turbo3 KV cache dequant + Q·K dot product (Metal Shading Language).
//
// Block layout (block_tbq3_0 in ggml-common.h, 14 bytes):
//     half     norm                       // per-block scale (corrected)
//     uchar    qs[QK_TBQ*3/8 = 12]        // 32 codes × 3 bits, packed straight
//
// Element decode (matches CPU reference dequantize_row_tbq3_0 +
// ggml/src/ggml-quants.c:66 tbq3_get_code):
//     elem i in 0..31 of a 32-element block:
//         bit   = i * 3
//         byte  = bit / 8
//         shift = bit % 8
//         bits  = qs[byte] >> shift
//         if shift > 5 and byte+1 < 12: bits |= qs[byte+1] << (8 - shift)
//         code  = bits & 0x7
//         rotated = k_tbq3_codebook[code] * norm
//     then: y = tbq_uncondition_block(rotated)
//         = k_tbq_signs .* H32(rotated)
//
// By the orthogonality of H32 (FWHT normalized by 1/sqrt(32)) and the
// distributivity of pointwise sign multiply,
//     <q, sign .* H32(r)> = <H32(q .* sign), r>
// so we precompute q_t = H32(q .* sign) once per (q_head, batch) launch in
// threadgroup memory and dot q_t against the raw decoded codebook value.
// Bit-exactly equivalent to the CPU dequant + dot.
//
// Dispatch: one threadgroup per group of `blocks_per_threadgroup` consecutive
// KV tokens for a single (q_head, batch). Threadgroup size = 32 (one Apple
// SIMD-group). Each thread handles 4 of the 128 elements per token.

#include <metal_stdlib>
using namespace metal;

// Match block_tbq3_0 layout exactly (14 bytes, alignment 2).
struct block_turbo3_0 {
    half     norm;
    uint8_t  qs[12];
};

// k_tbq3_codebook from ggml/src/ggml-quants.c:35. Unlike the legacy fork
// centroids, this set has magnitudes ~11.3× larger; the per-block norm
// absorbs the scale during quantization.
constant float TBQ3_CODEBOOK[8] = {
    -2.1519457f, -1.3439093f, -0.7560053f, -0.2450942f,
     0.2450942f,  0.7560053f,  1.3439093f,  2.1519457f,
};

// k_tbq_signs[QK_TBQ=32] from ggml/src/ggml-quants.c:59. Reused per block.
constant float TBQ_SIGNS_32[32] = {
     1.0f, -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,
     1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,
    -1.0f,  1.0f,  1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f,
     1.0f, -1.0f,  1.0f, -1.0f, -1.0f,  1.0f, -1.0f,  1.0f,
};

// Extract the 3-bit code at element index `i` in 0..31 from the 12-byte
// packed stream stored in device memory. Port of tbq3_get_code in
// ggml-quants.c:66.
static inline uint tbq3_get_code_dev(device const uint8_t * qs, uint i) {
    uint bit   = i * 3u;
    uint byte  = bit >> 3u;
    uint shift = bit & 7u;
    uint bits  = uint(qs[byte]) >> shift;
    if (shift > 5u && byte + 1u < 12u) {
        bits |= uint(qs[byte + 1u]) << (8u - shift);
    }
    return bits & 0x7u;
}

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
    uint head_offset_bytes; // must be a multiple of sizeof(block_turbo3_0) (14)
};

kernel void kernel_turbo3_dot(
        device const float          * q             [[buffer(0)]],
        device const block_turbo3_0 * k_blocks      [[buffer(1)]],
        device       float          * scores        [[buffer(2)]],
        constant     turbo_dot_args & args          [[buffer(3)]],
        uint                          tid           [[thread_position_in_threadgroup]],
        uint                          kv_idx        [[threadgroup_position_in_grid]]) {
    if (kv_idx >= args.n_kv) return;

    threadgroup float q_t[128];
    device const float * q_head = q + args.q_head * args.head_dim;
    eliza_tbq_precompute_qt(q_head, q_t, tid);

    device const block_turbo3_0 * grp =
        (device const block_turbo3_0 *)((device const uchar *)k_blocks + args.head_offset_bytes)
        + kv_idx * args.kv_stride_blocks;
    uint elem0   = tid * 4;
    uint blk_idx = elem0 >> 5;
    uint within0 = elem0 & 31;
    device const block_turbo3_0 & blk = grp[blk_idx];
    float norm = float(blk.norm);

    float4 qtv = float4(q_t[elem0 + 0], q_t[elem0 + 1], q_t[elem0 + 2], q_t[elem0 + 3]);

    uint c0 = tbq3_get_code_dev(blk.qs, within0 + 0u);
    uint c1 = tbq3_get_code_dev(blk.qs, within0 + 1u);
    uint c2 = tbq3_get_code_dev(blk.qs, within0 + 2u);
    uint c3 = tbq3_get_code_dev(blk.qs, within0 + 3u);
    float4 kv = float4(
        TBQ3_CODEBOOK[c0],
        TBQ3_CODEBOOK[c1],
        TBQ3_CODEBOOK[c2],
        TBQ3_CODEBOOK[c3]) * norm;
    float acc = dot(qtv, kv);

    float sum = simd_sum(acc);
    if (tid == 0) {
        scores[args.q_head * args.n_kv + kv_idx] = sum;
    }
}

// Multi-block-per-dispatch variant. Identical math; the threadgroup processes
// `blocks_per_threadgroup` consecutive KV indices serially in a 32-thread loop,
// trading dispatch grid breadth for amortised launch tax.
struct turbo_dot_multi_args {
    uint head_dim;
    uint n_kv;
    uint kv_stride_blocks;
    uint q_head;
    uint head_offset_bytes;
    uint blocks_per_threadgroup;
};

kernel void kernel_turbo3_dot_multi(
        device const float          * q             [[buffer(0)]],
        device const block_turbo3_0 * k_blocks      [[buffer(1)]],
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

        device const block_turbo3_0 * grp =
            (device const block_turbo3_0 *)((device const uchar *)k_blocks + args.head_offset_bytes)
            + kv_idx * args.kv_stride_blocks;
        device const block_turbo3_0 & blk = grp[blk_idx];
        float norm = float(blk.norm);

        uint c0 = tbq3_get_code_dev(blk.qs, within0 + 0u);
        uint c1 = tbq3_get_code_dev(blk.qs, within0 + 1u);
        uint c2 = tbq3_get_code_dev(blk.qs, within0 + 2u);
        uint c3 = tbq3_get_code_dev(blk.qs, within0 + 3u);
        float4 kv = float4(
            TBQ3_CODEBOOK[c0],
            TBQ3_CODEBOOK[c1],
            TBQ3_CODEBOOK[c2],
            TBQ3_CODEBOOK[c3]) * norm;
        float acc = dot(qtv, kv);

        float sum = simd_sum(acc);
        if (tid == 0) {
            scores[args.q_head * args.n_kv + kv_idx] = sum;
        }
    }
}
