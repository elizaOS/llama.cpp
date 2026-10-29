#include "common.h"

kernel void kernel_rwkv_wkv6_f32(
    device const float * k,
    device const float * v,
    device const float * r,
    device const float * tf,
    device const float * td,
    device const float * state_in,
    device       float * dst,
    constant    uint & B,
    constant    uint & T,
    constant    uint & C,
    constant    uint & H,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]])  {

    const uint head_size = 64; // TODO: support head_size = 128
    const uint batch_id = tgpig.x / H;
    const uint head_id = tgpig.x % H;
    const uint tid = tpitg.x;

    if (batch_id >= B || head_id >= H) {
        return;
    }

    const uint state_size = C * head_size;
    const uint n_seq_tokens = T / B;

    threadgroup float _k[head_size];
    threadgroup float _r[head_size];
    threadgroup float _tf[head_size];
    threadgroup float _td[head_size];

    float state[head_size];

    for (uint i = 0; i < head_size; i++) {
        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size
                          + i * head_size + tid];
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    _tf[tid] = tf[head_id * head_size + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    const uint end_t = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _k[tid] = k[t];
        _r[tid] = r[t];
        _td[tid] = td[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float v_val = v[t];
        float y = 0.0;

        for (uint j = 0; j < head_size; j += 4) {
            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 r_vec = float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            float4 tf_vec = float4(_tf[j], _tf[j+1], _tf[j+2], _tf[j+3]);
            float4 td_vec = float4(_td[j], _td[j+1], _td[j+2], _td[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * v_val;

            float4 temp = tf_vec * kv + s_vec;
            y += dot(r_vec, temp);

            s_vec = s_vec * td_vec + kv;
            state[j]   = s_vec[0];
            state[j+1] = s_vec[1];
            state[j+2] = s_vec[2];
            state[j+3] = s_vec[3];
        }

        dst[t] = y;
    }

    for (uint i = 0; i < head_size; i++) {
        dst[T * C + batch_id * state_size + head_id * head_size * head_size
            + i * head_size + tid] = state[i];
    }
}

kernel void kernel_rwkv_wkv7_f32(
    device const float * r,
    device const float * w,
    device const float * k,
    device const float * v,
    device const float * a,
    device const float * b,
    device const float * state_in,
    device       float * dst,
    constant    uint & B,
    constant    uint & T,
    constant    uint & C,
    constant    uint & H,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]])  {

    const uint head_size = 64; // TODO: support head_size = 128
    const uint batch_id = tgpig.x / H;
    const uint head_id = tgpig.x % H;
    const uint tid = tpitg.x;

    if (batch_id >= B || head_id >= H) {
        return;
    }

    const uint state_size = C * head_size;
    const uint n_seq_tokens = T / B;

    threadgroup float _r[head_size];
    threadgroup float _w[head_size];
    threadgroup float _k[head_size];
    threadgroup float _a[head_size];
    threadgroup float _b[head_size];

    float state[head_size];

    for (uint i = 0; i < head_size; i++) {
        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size
                          + tid * head_size + i];
    }

    const uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    const uint end_t = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _r[tid] = r[t];
        _w[tid] = w[t];
        _k[tid] = k[t];
        _a[tid] = a[t];
        _b[tid] = b[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float v_val = v[t];
        float y = 0.0, sa = 0.0;

        float4 sa_vec(0.0);

        for (uint j = 0; j < head_size; j += 4) {
            float4 a_vec = float4(_a[j], _a[j+1], _a[j+2], _a[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);
            sa_vec += a_vec * s_vec;
        }
        sa = sa_vec[0] + sa_vec[1] + sa_vec[2] + sa_vec[3];

        for (uint j = 0; j < head_size; j += 4) {
            float4 r_vec = float4(_r[j], _r[j+1], _r[j+2], _r[j+3]);
            float4 w_vec = float4(_w[j], _w[j+1], _w[j+2], _w[j+3]);
            float4 k_vec = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 b_vec = float4(_b[j], _b[j+1], _b[j+2], _b[j+3]);
            float4 s_vec = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * v_val;

            s_vec = s_vec * w_vec + kv + sa * b_vec;
            y += dot(s_vec, r_vec);

            state[j]   = s_vec[0];
            state[j+1] = s_vec[1];
            state[j+2] = s_vec[2];
            state[j+3] = s_vec[3];
        }

        dst[t] = y;
    }

    for (uint i = 0; i < head_size; i++) {
        dst[T * C + batch_id * state_size + head_id * head_size * head_size
            + tid * head_size + i] = state[i];
    }
}

// GLA (Gated Linear Attention)
// src[0]=k, src[1]=v, src[2]=q, src[3]=gate, src[4]=state
// dst = [attention_output (C x T), new_state (C x head_size x B)]
template<uint HEAD_SIZE>
kernel void kernel_gla_f32(
    device const float * k,
    device const float * v,
    device const float * q,
    device const float * gate,
    device const float * state_in,
    device       float * dst,
    constant    uint & B,
    constant    uint & T,
    constant    uint & C,
    constant    uint & H,
    constant   float & scale,
    uint3 tgpig[[threadgroup_position_in_grid]],
    uint3 tpitg[[thread_position_in_threadgroup]],
    uint3   ntg[[threads_per_threadgroup]]) {

    const uint head_size = HEAD_SIZE;
    const uint batch_id = tgpig.x / H;
    const uint head_id = tgpig.x % H;
    const uint tid = tpitg.x;

    if (batch_id >= B || head_id >= H) {
        return;
    }

    const uint state_size = C * head_size;
    const uint n_seq_tokens = T / B;

    threadgroup float _k[HEAD_SIZE];
    threadgroup float _q[HEAD_SIZE];
    threadgroup float _gate[HEAD_SIZE];

    float state[HEAD_SIZE];

    // Load initial state
    for (uint i = 0; i < head_size; i++) {
        state[i] = state_in[batch_id * state_size + head_id * head_size * head_size
                          + i * head_size + tid];
    }

    const uint start_t = batch_id * n_seq_tokens * C + head_id * head_size + tid;
    const uint end_t = (batch_id + 1) * n_seq_tokens * C + head_id * head_size + tid;

    for (uint t = start_t; t < end_t; t += C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        _k[tid] = k[t];
        _q[tid] = q[t];
        _gate[tid] = gate[t];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float _v = v[t];
        float y = 0.0f;

        for (uint j = 0; j < head_size; j += 4) {
            float4 k_vec   = float4(_k[j], _k[j+1], _k[j+2], _k[j+3]);
            float4 q_vec   = float4(_q[j], _q[j+1], _q[j+2], _q[j+3]);
            float4 g_vec   = float4(_gate[j], _gate[j+1], _gate[j+2], _gate[j+3]);
            float4 s_vec   = float4(state[j], state[j+1], state[j+2], state[j+3]);

            float4 kv = k_vec * _v;

            // state = state * gate + kv
            s_vec = s_vec * g_vec + kv;

            // y += state * q
            y += dot(q_vec, s_vec);

            state[j]   = s_vec[0];
            state[j+1] = s_vec[1];
            state[j+2] = s_vec[2];
            state[j+3] = s_vec[3];
        }

        dst[t] = y * scale;
    }

    // Write final state
    for (uint i = 0; i < head_size; i++) {
        dst[T * C + batch_id * state_size + head_id * head_size * head_size
            + i * head_size + tid] = state[i];
    }
}

template [[host_name("kernel_gla_f32_hs64")]]  kernel void kernel_gla_f32<64>(
    device const float *, device const float *, device const float *,
    device const float *, device const float *, device float *,
    constant uint &, constant uint &, constant uint &, constant uint &, constant float &,
    uint3, uint3, uint3);

template [[host_name("kernel_gla_f32_hs128")]] kernel void kernel_gla_f32<128>(
    device const float *, device const float *, device const float *,
    device const float *, device const float *, device float *,
    constant uint &, constant uint &, constant uint &, constant uint &, constant float &,
    uint3, uint3, uint3);
