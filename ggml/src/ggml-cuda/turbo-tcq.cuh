#pragma once

// TBQ3_TCQ — TurboQuant trellis-coded 3-bit quantization, decode-only path.
//
// Block layout (block_tbq3_tcq, ggml-common.h):
//   ggml_half d;       // fp16 per-block L2 norm (Viterbi-corrected)
//   uint8_t   qs[49];  // 6 prefix bits (initial state >> 3) followed by
//                      // 128*3 = 390 symbol bits, LSB-first within byte,
//                      // byte-major across qs[].
//   uint8_t   pad;     // alignment to 52 B total.
//
// Decode for symbol t:
//   bit_pos = t * 3
//   read 9 bits starting at bit_pos -> state in [0, 512)
//   x[t]    = codebook[state] * norm
//
// The 9-bit window equals (3 most-recent symbols, MSB = output[t-2..t]).
// Encoder is host-side Viterbi (O(128 * 512) per block). CUDA kernel only
// implements decode + dot.
//
// Math reference: packages/inference/reference/turbo_kernels.c::
// eliza_dequantize_turbo3_tcq_block.
//
// Compile-only validation: file body is gated on GGML_CUDA_TBQ3_TCQ.

#include "common.cuh"

#if defined(GGML_CUDA_TBQ3_TCQ)

#ifdef __cplusplus
extern "C" {
#endif

// Dequantize one row of block_tbq3_tcq to fp32. Each block produces
// QK_TBQ3_TCQ (=128) floats.
void dequantize_row_tbq3_tcq_cuda(
    const void * x_d,
    float * y_d,
    int64_t nrows,
    cudaStream_t stream);

// TBQ3_TCQ x Q8_0 mul-mat (vector path). Produces one fp32 dot product per
// (block_tbq3_tcq row, block_q8_0 row) pair.
//   x_d : block_tbq3_tcq, [nrows_x * (n_per_row / QK_TBQ3_TCQ)]
//   y_d : block_q8_0,     [nrows_y * (n_per_row / QK8_0)]
//   dst : fp32, [nrows_x * nrows_y]
// n_per_row must be a multiple of QK_TBQ3_TCQ (=128).
void mul_mat_tbq3_tcq_q8_0_cuda(
    const void * x_d,
    const void * y_d,
    float * dst_d,
    int nrows_x,
    int nrows_y,
    int n_per_row,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif // GGML_CUDA_TBQ3_TCQ
