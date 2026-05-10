// QJL (Quantized JL Transform) CUDA kernels.
//
// Production translation of the research kernels at
// packages/training/scripts/quantization/qjl/csrc/{qjl_quant_kernel.cu,
// qjl_score_kernel.cu}. The research kernels carry calibration counters,
// outlier hashing, and torch::extension plumbing; this file strips those
// to the in-fork production essentials and dispatches directly on
// block_qjl1_256 (ggml type 46).
//
// Math reference: packages/native-plugins/qjl-cpu/src/qjl_quantize_ref.c
// and qjl_score_ref.c. Parity must hold against those — every bit-pack
// pattern matches the CPU shim at ggml/src/ggml-cpu/qjl/quants-qjl.c.
//
// Compile-only validation: this file is only compiled when GGML_CUDA_QJL
// is defined. The CMake glob in ggml-cuda/CMakeLists.txt picks up the .cu
// unconditionally; the body is gated by the macro so the no-flag build
// still produces an empty object file (no symbols).

#include "ggml.h"
#include "ggml-impl.h"
#include "qjl.cuh"

#if defined(GGML_CUDA_QJL)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

#define QJL_PROJ_DIM   QK_QJL                  // 256
#define QJL_HASH_BYTES (QJL_PROJ_DIM / 8)      // 32
#define QJL_WARP_SIZE  32

// bf16 pattern (uint16) -> fp32. Matches qjl_bf16_to_fp32 in the CPU shim.
static __device__ __forceinline__ float qjl_bf16_to_fp32_dev(uint16_t bits) {
    uint32_t u = ((uint32_t) bits) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// fp32 -> bf16 pattern (round-to-nearest-even, matches qjl_fp32_to_bf16
// in the CPU shim).
static __device__ __forceinline__ uint16_t qjl_fp32_to_bf16_dev(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    // round-to-nearest-even with ties to even.
    uint32_t lsb = (u >> 16) & 1u;
    uint32_t round_bias = 0x7FFFu + lsb;
    uint32_t rounded = u + round_bias;
    return (uint16_t) (rounded >> 16);
}

// ---------------- quantize ----------------
//
// One block per row. Per row we:
//   - compute ||key||_2 (fp32 reduction across the 128 lanes)
//   - compute sketch[j] = sum_i key[i] * prj[i*proj_dim + j], for each of
//     the 256 sketch dims, packing the sign into a 32-byte signs[] buffer.
//
// Block geometry: 256 threads/block, one thread per sketch dim. Each
// thread loads its prj column (head_dim=128 values) and dot-products
// against the shared key. The 256 results are then ballot-reduced into
// 32 packed sign bytes via warp shfl + a single uint32 atomic-or per byte.

__global__ void qjl_quantize_kernel(
    const float * __restrict__ x,
    block_qjl1_256 * __restrict__ y,
    const float * __restrict__ prj,
    int nrows)
{
    const int row = blockIdx.x;
    if (row >= nrows) return;

    const int tid = threadIdx.x;
    const int j   = tid; // sketch dim index, 0..255

    __shared__ float s_key[QJL_HEAD_DIM];

    // Cooperative load of the row's head_dim=128 key into shared.
    if (tid < QJL_HEAD_DIM) {
        s_key[tid] = x[row * QJL_HEAD_DIM + tid];
    }
    __syncthreads();

    // Per-thread sketch[j] = sum_i s_key[i] * prj[i * proj_dim + j].
    float acc = 0.0f;
    if (j < QJL_PROJ_DIM) {
        #pragma unroll
        for (int i = 0; i < QJL_HEAD_DIM; ++i) {
            acc += s_key[i] * prj[i * QJL_PROJ_DIM + j];
        }
    }

    // Pack signs LSB-first. Use a warp-level ballot to combine 32 lanes
    // into a single uint32, then one byte at a time into the output.
    // Layout: 8 warps of 32 threads cover all 256 sketch dims.
    const unsigned active_mask = 0xFFFFFFFFu;
    unsigned bits = __ballot_sync(active_mask, acc > 0.0f);

    // lane 0 of each warp writes 4 bytes of packed signs at offset
    // (warp_id * 4) within signs[].
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    if (lane == 0) {
        // bits[lane k] -> bit k of byte (warp_id*4 + k/8) at position k%8
        // Since bits is already LSB-first by lane, the 4 bytes are
        // (bits & 0xFF), (bits >> 8 & 0xFF), (bits >> 16 & 0xFF), (bits >> 24 & 0xFF).
        y[row].signs[warp_id * 4 + 0] = (uint8_t) ( bits        & 0xFFu);
        y[row].signs[warp_id * 4 + 1] = (uint8_t) ((bits >> 8)  & 0xFFu);
        y[row].signs[warp_id * 4 + 2] = (uint8_t) ((bits >> 16) & 0xFFu);
        y[row].signs[warp_id * 4 + 3] = (uint8_t) ((bits >> 24) & 0xFFu);
    }

    // Compute ||key||_2 with a block-level reduction; lane 0 of warp 0 writes d.
    float ks = 0.0f;
    if (tid < QJL_HEAD_DIM) {
        ks = s_key[tid] * s_key[tid];
    }
    // Warp reduce sum
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        ks += __shfl_xor_sync(active_mask, ks, off);
    }
    __shared__ float s_partial[8];
    if ((tid & 31) == 0 && warp_id < 8) {
        s_partial[warp_id] = ks;
    }
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        // Only the first 4 warps contributed to head_dim=128.
        for (int w = 0; w < 4; ++w) total += s_partial[w];
        y[row].d = qjl_fp32_to_bf16_dev(sqrtf(total));
    }
}

void quantize_row_qjl1_256_cuda(
    const float * x_d,
    void * y_d,
    int64_t nrows,
    const float * prj_d,
    cudaStream_t stream)
{
    GGML_ASSERT(nrows > 0);
    GGML_ASSERT(QJL_HEAD_DIM == 128 && "QJL CUDA path only supports head_dim=128");
    block_qjl1_256 * y = (block_qjl1_256 *) y_d;
    const dim3 grid((unsigned) nrows, 1, 1);
    const dim3 block(QJL_PROJ_DIM, 1, 1);
    qjl_quantize_kernel<<<grid, block, 0, stream>>>(x_d, y, prj_d, (int) nrows);
}

// ---------------- dequantize ----------------
//
// Mirrors qjl_dequantize_row_ref:
//   recon[i] = scl * norm * sum_j sign(blk->signs, j) * prj[i*proj_dim + j]
// with scl = sqrt(pi/2) / proj_dim.
//
// Block geometry: one block per row, 128 threads/block (one per head_dim
// output slot). Each thread runs its 256-term inner product directly. We
// load the signs once into shared.

__global__ void qjl_dequantize_kernel(
    const block_qjl1_256 * __restrict__ x,
    float * __restrict__ y,
    const float * __restrict__ prj,
    int nrows)
{
    const int row = blockIdx.x;
    if (row >= nrows) return;

    const int i = threadIdx.x;
    if (i >= QJL_HEAD_DIM) return;

    __shared__ uint8_t s_signs[QJL_HASH_BYTES];
    if (threadIdx.x < QJL_HASH_BYTES) {
        s_signs[threadIdx.x] = x[row].signs[threadIdx.x];
    }
    __syncthreads();

    const float norm = qjl_bf16_to_fp32_dev(x[row].d);
    const float scl  = QJL_SQRT_PI_OVER_2 / (float) QJL_PROJ_DIM * norm;

    float acc = 0.0f;
    const float * prj_row = prj + i * QJL_PROJ_DIM;
    #pragma unroll 8
    for (int j = 0; j < QJL_PROJ_DIM; ++j) {
        const int bit = (s_signs[j >> 3] >> (j & 7)) & 1;
        acc += bit ? prj_row[j] : -prj_row[j];
    }
    y[row * QJL_HEAD_DIM + i] = scl * acc;
}

void dequantize_row_qjl1_256_cuda(
    const void * x_d,
    float * y_d,
    int64_t nrows,
    const float * prj_d,
    cudaStream_t stream)
{
    GGML_ASSERT(nrows > 0);
    const block_qjl1_256 * x = (const block_qjl1_256 *) x_d;
    const dim3 grid((unsigned) nrows, 1, 1);
    const dim3 block(QJL_HEAD_DIM, 1, 1);
    qjl_dequantize_kernel<<<grid, block, 0, stream>>>(x, y_d, prj_d, (int) nrows);
}

// ---------------- attention score ----------------
//
// score[hq, t] = ||k_t|| * sqrt(pi/2)/proj_dim *
//                sum_j sign(packed[t, j]) * q_sketch[hq, j]
// h_kv = hq / (n_heads / n_kv_heads).
//
// Block geometry: grid (n_heads, n_kv_tokens), block (proj_dim/8 = 32)
// — 32 threads per block reduce 8 sign bytes each, dot-producting against
// q_sketch[hq] in 8-wide chunks. We use warp shfl reduction and one
// scalar write per output.

__global__ void qjl_score_kernel(
    const float * __restrict__ q_sketch,        // [n_heads,    proj_dim]
    const block_qjl1_256 * __restrict__ packed, // [n_kv_heads, n_kv_tokens]
    int n_heads,
    int n_kv_heads,
    int n_kv_tokens,
    float * __restrict__ scores)                // [n_heads,    n_kv_tokens]
{
    const int hq = blockIdx.x;
    const int t  = blockIdx.y;
    if (hq >= n_heads || t >= n_kv_tokens) return;

    const int gqa = n_heads / n_kv_heads;
    const int hk  = hq / gqa;

    const block_qjl1_256 & blk = packed[hk * n_kv_tokens + t];
    const float * qs = q_sketch + hq * QJL_PROJ_DIM;

    // 32 threads, each handles 8 sketch dims (one byte of packed signs).
    const int tid = threadIdx.x;
    const uint8_t byte_signs = blk.signs[tid];
    float acc = 0.0f;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        const int j = tid * 8 + b;
        const float qv = qs[j];
        acc += ((byte_signs >> b) & 1) ? qv : -qv;
    }

    // Warp reduction (32 threads = 1 warp, perfect fit).
    const unsigned mask = 0xFFFFFFFFu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(mask, acc, off);
    }

    if (tid == 0) {
        const float norm = qjl_bf16_to_fp32_dev(blk.d);
        const float scl  = QJL_SQRT_PI_OVER_2 / (float) QJL_PROJ_DIM;
        scores[hq * n_kv_tokens + t] = scl * norm * acc;
    }
}

void attn_score_qjl_cuda(
    const float * q_sketch_d,
    const void  * packed_k_d,
    int n_heads,
    int n_kv_heads,
    int n_kv_tokens,
    float * scores_d,
    cudaStream_t stream)
{
    GGML_ASSERT(n_heads > 0 && n_kv_heads > 0 && n_kv_tokens > 0);
    GGML_ASSERT((n_heads % n_kv_heads) == 0);
    const block_qjl1_256 * packed = (const block_qjl1_256 *) packed_k_d;
    const dim3 grid(n_heads, n_kv_tokens, 1);
    const dim3 block(QJL_HASH_BYTES, 1, 1); // = 32
    qjl_score_kernel<<<grid, block, 0, stream>>>(
        q_sketch_d, packed, n_heads, n_kv_heads, n_kv_tokens, scores_d);
}

#endif // GGML_CUDA_QJL
