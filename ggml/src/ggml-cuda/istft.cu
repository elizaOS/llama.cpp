// SPDX-License-Identifier: MIT
//
// istft.cu — CUDA implementation of GGML_OP_ISTFT.
//
// Strategy:
//   - One CUDA block per output frame (up to 65535 frames).
//   - Each block runs an O(n_fft^2) naive IDFT.  For Kokoro (n_fft=20) this
//     is 20*20=400 muls per frame — trivially fast on any GPU.
//   - Overlap-add is serialized via atomicAdd into the global dst buffer.
//   - Window energy normalization is a second kernel pass.
//
// For large n_fft (e.g. n_fft=1024) a cuFFT path would be preferred; add
// GGML_ISTFT_USE_CUFFT=1 to enable it when cuFFT is available.

#include "istft.cuh"

#include <cmath>
#include <cstdio>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------------
// Kernel 1: hann window build (called once when no window tensor is provided)
// ---------------------------------------------------------------------------
static __global__ void build_hann_kernel(float * win, int win_length) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= win_length) return;
    const double scale = 2.0 * M_PI / (double) win_length;
    win[i] = (float)(0.5 - 0.5 * cos(scale * (double) i));
}

// ---------------------------------------------------------------------------
// Kernel 2: per-frame IDFT + windowed overlap-add
//
// Each block handles one frame t.
// dst_acc accumulates the windowed samples; dst_norm accumulates w^2.
// ---------------------------------------------------------------------------
static __global__ void istft_ola_kernel(
        const float * __restrict__ mag_base,    // [F * T] channel 0
        const float * __restrict__ phase_base,  // [F * T] channel 1
        const float * __restrict__ win,         // [win_length]
        float       * __restrict__ dst_acc,     // [n_out] output accumulator
        float       * __restrict__ dst_norm,    // [n_out] window^2 accumulator
        int T,          // number of frames
        int F,          // n_fft/2+1
        int n_fft,
        int hop_length,
        int win_length,
        int n_out) {

    const int t   = (int) blockIdx.x;   // frame index
    const int tid = (int) threadIdx.x;
    if (t >= T) return;

    // Shared memory: re[F], im[F], frame[n_fft]
    // We bound n_fft to 2048 for this kernel; caller asserts.
    extern __shared__ float shm[];
    float * sh_re    = shm;
    float * sh_im    = sh_re + F;
    float * sh_frame = sh_im + F;

    const double inv_n = 1.0 / (double) n_fft;

    // --- Phase 1: load mag/phase and compute re/im for this frame ---
    for (int f = tid; f < F; f += blockDim.x) {
        const float mag_v   = mag_base  [(int64_t) f * T + t];
        const float phase_v = phase_base[(int64_t) f * T + t];
        sh_re[f] = mag_v * __cosf(phase_v);
        sh_im[f] = mag_v * __sinf(phase_v);
    }
    __syncthreads();

    // --- Phase 2: IDFT for each output sample k in [0, n_fft) ---
    // Each thread handles multiple k values.
    for (int k = tid; k < n_fft; k += blockDim.x) {
        double acc = (double) sh_re[0];  // DC
        if ((n_fft & 1) == 0) {
            const double sign = (k & 1) ? -1.0 : 1.0;
            acc += sign * (double) sh_re[F - 1];  // Nyquist
        }
        const int interior_end = F - ((n_fft & 1) == 0 ? 1 : 0);
        for (int f = 1; f < interior_end; ++f) {
            const double angle = 2.0 * M_PI * (double) f * (double) k * inv_n;
            acc += 2.0 * ((double) sh_re[f] * cos(angle) -
                          (double) sh_im[f] * sin(angle));
        }
        sh_frame[k] = (float)(acc * inv_n);
    }
    __syncthreads();

    // --- Phase 3: windowed overlap-add ---
    const int off = t * hop_length;
    for (int k = tid; k < win_length; k += blockDim.x) {
        const int idx = off + k;
        if (idx >= n_out) break;
        const float w   = win[k];
        const float val = sh_frame[k % n_fft] * w;
        atomicAdd(&dst_acc [idx], val);
        atomicAdd(&dst_norm[idx], w * w);
    }
}

// ---------------------------------------------------------------------------
// Kernel 3: normalize output by window energy
// ---------------------------------------------------------------------------
static __global__ void istft_normalize_kernel(
        float * __restrict__ dst,
        const float * __restrict__ norm,
        int n_out) {
    const int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n_out) return;
    const float n = norm[i];
    if (n > 1e-8f) {
        dst[i] /= n;
    }
}

// ---------------------------------------------------------------------------
// Host-side dispatcher
// ---------------------------------------------------------------------------
void ggml_cuda_op_istft(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // mag_phase [T, F, 2]
    const ggml_tensor * src1 = dst->src[1];  // window [win_length], may be null

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int32_t * op_params = (const int32_t *) dst->op_params;
    const int n_fft      = op_params[0];
    const int hop_length = op_params[1];
    const int win_length = op_params[2];
    const int F          = n_fft / 2 + 1;

    // src0 layout: ne[0]=T (frames), ne[1]=F, ne[2]=2
    const int T = (int) src0->ne[0];
    GGML_ASSERT((int) src0->ne[1] == F);
    GGML_ASSERT((int) src0->ne[2] == 2);

    const int n_out = (T - 1) * hop_length + win_length;
    GGML_ASSERT((int) dst->ne[0] == n_out);

    // n_fft must fit in shared memory for the IDFT kernel.
    GGML_ASSERT(n_fft <= 2048 && "ggml_istft CUDA: n_fft > 2048; use cuFFT path");

    cudaStream_t stream = ctx.stream();

    const float * mag_base   = (const float *) src0->data;
    const float * phase_base = mag_base + (int64_t) F * T;
    float       * out_data   = (float *)       dst->data;

    // Allocate temporary norm buffer and optional window.
    float * d_norm = nullptr;
    float * d_win  = nullptr;

    CUDA_CHECK(cudaMalloc(&d_norm, (size_t) n_out * sizeof(float)));
    CUDA_CHECK(cudaMemsetAsync(d_norm, 0, (size_t) n_out * sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(out_data, 0, (size_t) n_out * sizeof(float), stream));

    if (src1 != nullptr) {
        GGML_ASSERT(src1->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(src1));
        GGML_ASSERT((int) src1->ne[0] == win_length);
        d_win = (float *) src1->data;
    } else {
        // Build periodic Hann window on-device.
        CUDA_CHECK(cudaMalloc(&d_win, (size_t) win_length * sizeof(float)));
        const int bw = (win_length + CUDA_ISTFT_BLOCK_SIZE - 1) / CUDA_ISTFT_BLOCK_SIZE;
        build_hann_kernel<<<bw, CUDA_ISTFT_BLOCK_SIZE, 0, stream>>>(d_win, win_length);
    }

    // Shared memory: 2*F (re + im) + n_fft (frame) floats.
    const size_t shm_bytes = (size_t)(2 * F + n_fft) * sizeof(float);
    const int block_sz = CUDA_ISTFT_BLOCK_SIZE;

    istft_ola_kernel<<<T, block_sz, shm_bytes, stream>>>(
        mag_base, phase_base,
        d_win,
        out_data, d_norm,
        T, F, n_fft, hop_length, win_length, n_out);

    // Normalize.
    const int blocks_n = (n_out + CUDA_ISTFT_BLOCK_SIZE - 1) / CUDA_ISTFT_BLOCK_SIZE;
    istft_normalize_kernel<<<blocks_n, CUDA_ISTFT_BLOCK_SIZE, 0, stream>>>(
        out_data, d_norm, n_out);

    // Free temporaries (device-side; stream-ordered via event in caller).
    CUDA_CHECK(cudaFreeAsync(d_norm, stream));
    if (src1 == nullptr) {
        CUDA_CHECK(cudaFreeAsync(d_win, stream));
    }
}
