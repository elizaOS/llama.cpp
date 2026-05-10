#pragma once

// QJL (Quantized JL Transform) 1-bit packed-K KV-cache support.
//
// Block layout (block_qjl1_256, ggml-common.h):
//   uint8_t  signs[QK_QJL/8];   // 32 bytes of packed +/-1 signs (LSB-first)
//   uint16_t d;                 // bf16 bit-pattern of ||k||_2
// Total: 34 B per cached key vector.
//
// Conventions match the CPU reference at
// packages/native-plugins/qjl-cpu/src/{qjl_quantize_ref.c,qjl_score_ref.c}
// and the in-fork CPU shim at ggml/src/ggml-cpu/qjl/quants-qjl.c.
//
// The QJL paper's reconstruction uses the unbiased cosine-similarity
// estimator with scale sqrt(pi/2) / proj_dim — see qjl_dequantize_row_ref.
//
// For compile-only validation this header gates on GGML_CUDA_QJL.

#include "common.cuh"

// QJL is gated to keep the CUDA build pristine when the feature is off.
#if defined(GGML_CUDA_QJL)

// QJL_HEAD_DIM matches packages/native-plugins/qjl-cpu/include/qjl/qjl.h.
// The kernels here are specialized for head_dim=128 (Llama 3, Qwen3 sub-9B,
// Qwen2/2.5). 256 is reserved for Qwen3.6-27B / 35B-A3B and not yet wired.
#ifndef QJL_HEAD_DIM
#define QJL_HEAD_DIM 128
#endif

// sqrt(pi/2) = 1.2533141373155003. Locked to match the CPU reference.
#ifndef QJL_SQRT_PI_OVER_2
#define QJL_SQRT_PI_OVER_2 1.2533141373155003f
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Quantize one row of fp32 keys into block_qjl1_256.
//   x : fp32, length nrows * head_dim (128).
//   y : block_qjl1_256, length nrows.
//   prj_d : projection matrix, length head_dim * QK_QJL (= 128 * 256), fp32.
// Throws via GGML_ABORT on launch failure.
void quantize_row_qjl1_256_cuda(
    const float * x_d,
    void * y_d,
    int64_t nrows,
    const float * prj_d,
    cudaStream_t stream);

// Dequantize one row of block_qjl1_256 into fp32 keys (head_dim length each).
void dequantize_row_qjl1_256_cuda(
    const void * x_d,
    float * y_d,
    int64_t nrows,
    const float * prj_d,
    cudaStream_t stream);

// GQA attention-score path. Mirrors qjl_score_qk_ref:
//   score[h_q, t] = ||k_t|| * sqrt(pi/2) / proj_dim *
//                   sum_j sign(packed[t, j]) * q_sketch[h_q, j]
// h_kv = h_q / (n_heads / n_kv_heads) for GQA sharing.
//
// Inputs:
//   q_sketch_d : fp32, [n_heads, QK_QJL]. Pre-projected q.
//   packed_k_d : block_qjl1_256, [n_kv_heads, n_kv_tokens] (token-major).
// Output:
//   scores_d   : fp32, [n_heads, n_kv_tokens].
void attn_score_qjl_cuda(
    const float * q_sketch_d,
    const void  * packed_k_d,
    int n_heads,
    int n_kv_heads,
    int n_kv_tokens,
    float * scores_d,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA_QJL
