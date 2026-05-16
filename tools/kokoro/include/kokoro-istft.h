// SPDX-License-Identifier: MIT
//
// kokoro-istft.h — CPU-side inverse STFT for the Kokoro decoder vocoder head.
//
// Kokoro's iSTFTNet vocoder produces a (mag, phase) spectrogram pair at the
// output of the upsampling decoder, then reconstructs the time-domain
// waveform with an inverse STFT (overlap-add). The standard llama.cpp /
// GGML op set does not include `ggml_istft`, so we run the iSTFT on CPU
// after the GGML graph: that's cheap (n_fft=20, hop=5 in Kokoro v1.0 —
// the per-window FFT is a 20-point complex DFT with O(400) muls).
//
// Adding `GGML_OP_ISTFT` to the fork would be the right long-term answer
// (matched Metal / Vulkan kernels for the heavy iSTFT case). For
// Kokoro-82M the post-graph CPU path is the pragmatic short-term answer
// per J2 ship scope.

#pragma once

#include <vector>

namespace eliza_kokoro {

// Inverse short-time Fourier transform with a Hann window.
//
//   mag[F][T] + phase[F][T]  →  x[N]
//
// where F = n_fft/2 + 1, T = number of frames, hop = hop length,
// N = (T - 1) * hop + win_length.
//
// `mag`, `phase` are row-major `F * T` fp32 buffers (frequency-major).
// `out` is resized to the synthesized length.
//
// This is the textbook overlap-add iSTFT with a Hann window matching the
// upstream `librosa.istft` default (`hann`, center=False). For Kokoro v1.0
// (n_fft=20, hop=5), the per-frame DFT is implemented directly (no FFT
// dispatch — the 20-point DFT is faster as a naive O(N^2) for these sizes
// than the bookkeeping of a generic FFT).
void istft_hann(
    const std::vector<float> & mag,
    const std::vector<float> & phase,
    int n_fft,
    int hop_length,
    int win_length,
    int n_frames,
    std::vector<float> & out);

} // namespace eliza_kokoro
