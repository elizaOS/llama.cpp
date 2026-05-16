// SPDX-License-Identifier: MIT
//
// istft.cuh — CUDA kernel header for GGML_OP_ISTFT.
//
// Implements an inverse short-time Fourier transform with a Hann window and
// overlap-add synthesis.  Per-frame iDFT is hand-rolled (no cuFFT dependency
// so this compiles on any CUDA toolkit); for large n_fft a cuFFT path can be
// wired in later.
//
// Input src0 (mag_phase): F32, shape ne[0]=T (frames), ne[1]=F, ne[2]=2
//   channel 0 = magnitude, channel 1 = phase (polar form)
//
// Input src1 (window): optional F32 [win_length].  NULL → periodic Hann built
// in the kernel.
//
// Output: F32 [N]  N = (T-1)*hop_length + win_length.

#pragma once
#include "common.cuh"

#define CUDA_ISTFT_BLOCK_SIZE 256

void ggml_cuda_op_istft(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
