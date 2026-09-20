/** Computes graph TBQ3_0/TBQ4_0 attention scores from unrotated queries.
 * These formats use 32-wide signed Hadamard conditioning and differ from the
 * standalone turbo3 fixture format. Each SIMD lane owns one element in each
 * of four blocks; query conditioning is reused across the token group.
 */
#include <metal_stdlib>
using namespace metal;

struct block_score_tbq3 { half d; uchar qs[12]; };
struct block_score_tbq4 { half d; uchar qs[16]; };
struct tbq_score_args {
    uint head_dim;
    uint n_kv;
    uint kv_stride_blocks;
    uint q_head;
    uint head_offset_bytes;
    uint blocks_per_threadgroup;
};

constant float SCORE_TBQ3_CODEBOOK[8] = {
    -2.1519457f, -1.3439093f, -0.7560053f, -0.2450942f,
     0.2450942f,  0.7560053f,  1.3439093f,  2.1519457f,
};
constant float SCORE_TBQ4_CODEBOOK[16] = {
    -2.7321365f, -2.0685055f, -1.6175243f, -1.2557391f,
    -0.9419147f, -0.6564307f, -0.3878412f, -0.1283243f,
     0.1283243f,  0.3878412f,  0.6564307f,  0.9419147f,
     1.2557391f,  1.6175243f,  2.0685055f,  2.7321365f,
};
constant int SCORE_TBQ_SIGNS[32] = {
     1, -1,  1,  1, -1,  1, -1, -1,
     1,  1, -1,  1, -1, -1,  1, -1,
    -1,  1,  1, -1,  1, -1, -1,  1,
     1, -1,  1, -1, -1,  1, -1,  1,
};

inline float score_tbq_decode(device const block_score_tbq3 & block, uint lane) {
    uint bit = 3u * lane;
    uint byte = bit >> 3u;
    uint shift = bit & 7u;
    uint code = uint(block.qs[byte]) >> shift;
    if (shift > 5u && byte + 1u < 12u) {
        code |= uint(block.qs[byte + 1u]) << (8u - shift);
    }
    return float(block.d) * SCORE_TBQ3_CODEBOOK[code & 7u];
}
inline float score_tbq_decode(device const block_score_tbq4 & block, uint lane) {
    uint packed = block.qs[lane & 15u];
    uint code = lane < 16u ? packed & 15u : packed >> 4u;
    return float(block.d) * SCORE_TBQ4_CODEBOOK[code];
}

template<typename Block>
kernel void kernel_attn_score_tbq(
        device const float * q [[buffer(0)]],
        device const Block * packed_k [[buffer(1)]],
        device float * scores [[buffer(2)]],
        constant tbq_score_args & args [[buffer(3)]],
        uint lane [[thread_position_in_threadgroup]],
        uint group [[threadgroup_position_in_grid]]) {
    // H is normalized and symmetric: dot(q, D H k) = dot(H D q, k).
    float4 query;
    for (uint block = 0; block < 4u; ++block) {
        float value = q[args.q_head * args.head_dim + block * 32u + lane] * float(SCORE_TBQ_SIGNS[lane]);
        for (uint stride = 1u; stride < 32u; stride <<= 1u) {
            float other = simd_shuffle_xor(value, stride);
            value = (lane & stride) ? other - value : value + other;
        }
        query[block] = value * 0.1767766952966369f;
    }
    device const Block * head = (device const Block *)
        ((device const uchar *) packed_k + args.head_offset_bytes);
    for (uint offset = 0; offset < args.blocks_per_threadgroup; ++offset) {
        uint token = group * args.blocks_per_threadgroup + offset;
        if (token >= args.n_kv) return;
        float value = 0.0f;
        for (uint block = 0; block < 4u; ++block) {
            value += query[block] * score_tbq_decode(head[token * args.kv_stride_blocks + block], lane);
        }
        float score = simd_sum(value);
        if (lane == 0u) scores[args.q_head * args.n_kv + token] = score;
    }
}

template [[host_name("kernel_attn_score_tbq3_0")]] kernel decltype(kernel_attn_score_tbq<block_score_tbq3>) kernel_attn_score_tbq<block_score_tbq3>;
template [[host_name("kernel_attn_score_tbq4_0")]] kernel decltype(kernel_attn_score_tbq<block_score_tbq4>) kernel_attn_score_tbq<block_score_tbq4>;
