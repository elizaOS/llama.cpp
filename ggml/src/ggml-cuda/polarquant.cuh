#pragma once

// PolarQuant Q4 CUDA support.
//
// Block layout (block_q4_polar, ggml-common.h):
//   ggml_half d;                          // per-block fp16 L2 norm
//   uint8_t   qs[QK_POLAR / 2];           // 4-bit centroid indices, 2 per byte (low nibble even)
//   uint8_t   qjl[QJL_RESIDUAL_BYTES];    // optional 1-bit QJL residual sign
// Total 82 bytes per block (= 5.125 bpw with QJL residual on, 4.125 bpw on disk if qjl=0).
//
// Math reference: packages/native-plugins/polarquant-cpu/src/{polar_dequantize_ref.c,
// polar_dot_ref.c, polar_hadamard.c, polar_qjl.c}. Bit parity must hold
// against those — this CUDA port reproduces the same Lloyd-Max centroid
// table, the same iterative 7-stage Walsh-Hadamard butterfly, and the same
// xorshift32 sign sequence used by the QJL residual correction.
//
// Compile-only validation: file body is gated on GGML_CUDA_POLARQUANT.

#include "common.cuh"

#if defined(GGML_CUDA_POLARQUANT)

#ifdef __cplusplus
extern "C" {
#endif

// Dequantize one row of block_q4_polar to fp32.
//   x_d : block_q4_polar, length nrows.
//   y_d : fp32, length nrows * QK_POLAR.
//   use_qjl : non-zero applies the QJL residual correction. Must match
//             encoder. Default per the W3-B Polar slot is "QJL gated to off
//             by runtime flag" — see Q4_POLAR commit 7d0fa96.
void dequantize_row_q4_polar_cuda(
    const void * x_d,
    float * y_d,
    int64_t nrows,
    int use_qjl,
    cudaStream_t stream);

// Q4_POLAR x Q8_0 mul-mat (vector path: produces one fp32 dot product per
// (block_q4_polar row x block_q8_0 row) pair).
//   x_d : block_q4_polar, [nrows_x * (n_per_row / QK_POLAR)]
//   y_d : block_q8_0,     [nrows_y * (n_per_row / QK8_0)]
//   dst : fp32, [nrows_x * nrows_y]
// nrows_x = the number of Q4_POLAR rows we mul-mat. nrows_y = the batch.
// n_per_row = the row length in scalar elements. Must be a multiple of
// QK_POLAR (=128).
void mul_mat_q4_polar_q8_0_cuda(
    const void * x_d,
    const void * y_d,
    float * dst_d,
    int nrows_x,
    int nrows_y,
    int n_per_row,
    int use_qjl,
    cudaStream_t stream);

// get_rows for Q4_POLAR: dequantize selected rows into fp32 / fp16 / bf16
// destination. Mirrors get_rows_cuda_q in getrows.cu, but goes through the
// PolarQuant decoder so the nibble unpack + WHT + L2 rescale is fused in
// one pass (since the dequantize_q4_polar device-side function is too
// heavy to fit the float2-output dequantize_q* signature).
void get_rows_q4_polar_cuda(
    const void * src0_d,
    const int32_t * src1_d,
    void * dst_d,
    int dst_type,                  // ggml_type cast to int (F32=0, F16=1, BF16=30)
    int64_t ne00,
    size_t nb01, size_t nb02, size_t nb03,
    int64_t ne10, int64_t ne11, int64_t ne12,
    size_t nb10, size_t nb11, size_t nb12,
    size_t nb1,  size_t nb2,  size_t nb3,
    int use_qjl,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA_POLARQUANT
