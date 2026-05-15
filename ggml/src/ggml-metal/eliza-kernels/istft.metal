// SPDX-License-Identifier: MIT
//
// istft.metal — Metal kernel for GGML_OP_ISTFT.
//
// runtimeStatus = "authored-only"
//
// This kernel was authored for symmetry with the CUDA and Vulkan counterparts
// but cannot be verified without Apple Silicon hardware.  The kernel is
// compiled into the metallib but the Metal backend dispatch is guarded by
// #ifdef GGML_METAL_ISTFT_KERNEL so it does not activate in production builds
// until hardware verification passes.
//
// Algorithm: one threadgroup per output smpl, iterating over all frames
// that overlap that smpl.  For each contributing frame the 1-D iDFT is
// evaluated using the Hermitian symmetry of the real-output STFT.
//
// Input layout (same as Vulkan/CUDA):
//   src0 (mag_phase): F32 [F*T | F*T]  mag first half, phase second half
//   src1 (window):    F32 [win_length]
//   dst:              F32 [n_out]
//
// Params passed via a small constant struct in argument buffer:
//   uint n_fft, hop_length, win_length, T, n_out

#include <metal_stdlib>
using namespace metal;

struct IstftParams {
    uint n_fft;
    uint hop_length;
    uint win_length;
    uint T;
    uint n_out;
};

kernel void kernel_istft_f32(
        device const float * mag_phase   [[ buffer(0) ]],
        device const float * win_data    [[ buffer(1) ]],
        device       float * dst_data    [[ buffer(2) ]],
        constant IstftParams & p         [[ buffer(3) ]],
        uint   gid   [[ thread_position_in_grid ]]) {

    if (gid >= p.n_out) return;

    const uint i = gid;
    const uint F = p.n_fft / 2u + 1u;

    // Frame range overlapping output smpl i.
    const uint t_max = min(i / p.hop_length, p.T - 1u);
    uint t_min = 0u;
    if (i + 1u > p.win_length) {
        t_min = (i + 1u - p.win_length + p.hop_length - 1u) / p.hop_length;
    }

    float acc  = 0.0f;
    float norm = 0.0f;
    const float inv_n  = 1.0f / float(p.n_fft);
    const float two_pi = 2.0f * M_PI_F;

    for (uint t = t_min; t <= t_max; ++t) {
        const uint k = i - t * p.hop_length;
        if (k >= p.win_length) continue;

        const float win_k  = win_data[k];
        const uint  k_mod  = k % p.n_fft;
        norm += win_k * win_k;

        // IDFT output at time-index k_mod for frame t.
        float smpl = 0.0f;

        // DC
        {
            const float mag_v   = mag_phase[0u * p.T + t];
            const float phase_v = mag_phase[F  * p.T + 0u * p.T + t];
            smpl += mag_v * cos(phase_v);
        }

        // Nyquist (n_fft even only)
        if ((p.n_fft & 1u) == 0u) {
            const uint  nyq     = F - 1u;
            const float mag_v   = mag_phase[nyq * p.T + t];
            const float phase_v = mag_phase[F   * p.T + nyq * p.T + t];
            const float re      = mag_v * cos(phase_v);
            const float sign    = ((k_mod & 1u) == 0u) ? 1.0f : -1.0f;
            smpl += sign * re;
        }

        // Interior bins [1, F-1)
        const uint interior_end = F - ((p.n_fft & 1u) == 0u ? 1u : 0u);
        for (uint f = 1u; f < interior_end; ++f) {
            const float mag_v   = mag_phase[f * p.T + t];
            const float phase_v = mag_phase[F * p.T + f * p.T + t];
            const float re      = mag_v * cos(phase_v);
            const float im      = mag_v * sin(phase_v);
            const float angle   = two_pi * float(f) * float(k_mod) * inv_n;
            smpl += 2.0f * (re * cos(angle) - im * sin(angle));
        }

        smpl *= inv_n;
        acc += smpl * win_k;
    }

    dst_data[i] = (norm > 1e-8f) ? (acc / norm) : 0.0f;
}
