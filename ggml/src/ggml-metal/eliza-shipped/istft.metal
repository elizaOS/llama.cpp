// # ELIZA-KERNEL-PATCH-V1 — Metal kernel for GGML_OP_ISTFT.
//
// Inverse short-time Fourier transform with overlap-add synthesis.  One
// threadgroup column per output sample (gid = output index).  Each thread
// computes the windowed IDFT contribution of every frame that overlaps
// `gid` and accumulates it with the window-energy normaliser, matching the
// CPU reference at ggml/src/ggml-cpu/ops.cpp:ggml_compute_forward_istft_f32
// and the Vulkan reference at ggml/src/ggml-vulkan/vulkan-shaders/istft.comp.
//
// Tensor layout (matches ggml_istft contract):
//   src0 (mag_phase): F32 [2, F, T]   ne[0]=2 (mag/phase channel),
//                                     ne[1]=F=n_fft/2+1, ne[2]=T (frames).
//                                     Element [ch,f,t] = data[t*(2*F) + f*2 + ch].
//   src1 (window):    F32 [win_length] — optional, signalled by use_window.
//                                       When zero a periodic Hann window is
//                                       synthesised on-the-fly.
//   dst:              F32 [n_out], n_out = (T-1)*hop_length + win_length.
//
// Params via argument buffer (IstftParams):
//   n_fft, hop_length, win_length, T, n_out, use_window

#include <metal_stdlib>
using namespace metal;

struct IstftParams {
    uint n_fft;
    uint hop_length;
    uint win_length;
    uint T;
    uint n_out;
    uint use_window;
};

kernel void kernel_istft_f32(
        device const float * mag_phase   [[ buffer(0) ]],
        device const float * win_data    [[ buffer(1) ]],
        device       float * dst_data    [[ buffer(2) ]],
        constant IstftParams & p         [[ buffer(3) ]],
        uint   gid                       [[ thread_position_in_grid ]]) {

    if (gid >= p.n_out) return;

    const uint i      = gid;
    const uint F      = p.n_fft / 2u + 1u;
    const uint stride = 2u * F;  // floats per frame in mag_phase

    // Frame range overlapping output sample i.
    const uint t_max = min(i / p.hop_length, p.T - 1u);
    uint t_min = 0u;
    if (i + 1u > p.win_length) {
        t_min = (i + 1u - p.win_length + p.hop_length - 1u) / p.hop_length;
    }

    float acc   = 0.0f;
    float norm  = 0.0f;
    const float inv_n     = 1.0f / float(p.n_fft);
    const float two_pi    = 2.0f * M_PI_F;
    const float hann_step = two_pi / float(p.win_length);

    for (uint t = t_min; t <= t_max; ++t) {
        const uint k = i - t * p.hop_length;
        if (k >= p.win_length) continue;

        const float win_k = (p.use_window != 0u)
            ? win_data[k]
            : (0.5f - 0.5f * cos(hann_step * float(k)));
        norm += win_k * win_k;

        const uint k_mod = k % p.n_fft;
        const uint base  = t * stride;  // start of this frame in mag_phase

        // Pre-extract DC + Nyquist (Hermitian-symmetric input → real output).
        float smpl = 0.0f;
        {
            const float mag_v   = mag_phase[base + 0u * 2u + 0u];
            const float phase_v = mag_phase[base + 0u * 2u + 1u];
            smpl += mag_v * cos(phase_v);
        }
        if ((p.n_fft & 1u) == 0u) {
            const uint  nyq     = F - 1u;
            const float mag_v   = mag_phase[base + nyq * 2u + 0u];
            const float phase_v = mag_phase[base + nyq * 2u + 1u];
            const float re      = mag_v * cos(phase_v);
            const float sign    = ((k_mod & 1u) == 0u) ? 1.0f : -1.0f;
            smpl += sign * re;
        }

        // Interior bins [1, F-1) with Hermitian symmetry (factor 2).
        const uint interior_end = F - ((p.n_fft & 1u) == 0u ? 1u : 0u);
        for (uint f = 1u; f < interior_end; ++f) {
            const float mag_v   = mag_phase[base + f * 2u + 0u];
            const float phase_v = mag_phase[base + f * 2u + 1u];
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
