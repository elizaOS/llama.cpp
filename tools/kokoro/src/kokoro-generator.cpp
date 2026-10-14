// SPDX-License-Identifier: MIT
//
// kokoro-generator.cpp — iSTFTNet Generator.forward, CPU scalar.
//
// Direct port of kokoro/istftnet.py `Generator.forward` (StyleTTS-2 +
// iSTFTNet decoder back-end). Implements:
//   - SourceModuleHnNSF / SineGen harmonic source (deterministic: zero
//     initial phase noise, zero additive noise — see note below),
//   - forward STFT (center=True) of the harmonic source,
//   - the two upsample stages (ConvTranspose ups[i] + noise_convs[i] +
//     noise_res[i] (AdaINResBlock1) + 3 resblocks[i*3+j] (AdaINResBlock1,
//     Snake1D activation), with leaky_relu(0.1) and a reflection_pad(1,0)
//     on the final stage),
//   - conv_post (Conv1d 128->22, k7, pad3),
//   - spec = exp(x[:11]), phase = sin(x[11:22]),
//   - inverse STFT (center=True) -> audio.
//
// Widths, rates and the source convention come from GeneratorConfig; the
// defaults (and kokoro_generator_forward) are Kokoro v1.0. Paradee-8M runs the
// same code at Cin 128, stage channels 64/32, style 16.
//
// Config (kokoro v1.0 istftnet):
//   style_dim=128, upsample_initial_channel=512,
//   upsample_rates=[10,6], upsample_kernel_sizes=[20,12],
//   resblock_kernel_sizes=[3,7,11], resblock_dilation_sizes=[[1,3,5]]x3,
//   gen_istft_n_fft=20, gen_istft_hop_size=5,
//   m_source: harmonic_num=8 (dim=9), upsample_scale=300, voiced_threshold=10.
//
// All conv/linear weights are PyTorch row-major and weight_norm-fused:
//   Conv1d weight       [Cout, Cin, K]
//   ConvTranspose1d wt  [Cin,  Cout, K]
//   Linear weight       [out,  in]
//   AdaIN1d fc.weight    [2C,   style_dim]
//   alpha (Snake)        [C]
//
// Determinism note: PyTorch's SineGen seeds an initial random phase
// (`rand_ini`, zeroed for the fundamental) and SourceModuleHnNSF adds
// Gaussian noise. For a reproducible / deterministic on-device renderer we
// drop both (zero phase noise on every harmonic, zero additive noise), so the
// harmonic source and the final audio will not bit-match a seeded PyTorch run
// — they are validated by correlation + structure instead. Every other stage
// (convs, resblocks, ups, conv_post, STFT/iSTFT) is exactly reproduced.
//
// Validated against per-stage reference activations regenerated from the
// real hexgrad/Kokoro-82M v1.0 decoder weights — see kokoro-generator.h and
// the integration notes returned with this work.

#include "kokoro-generator.h"
#include "kokoro-layers.h"
#include "kokoro-profile.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <vector>

#ifdef KOKORO_GEN_DEBUG
#include <fstream>
static void dbg_dump(const char * name, const float * d, size_t n) {
    std::ofstream f(std::string("/tmp/gendbg_") + name + ".f32", std::ios::binary);
    f.write((const char *) d, n * 4);
}
#define DBG(name, d, n) dbg_dump(name, d, n)
#else
#define DBG(name, d, n) do {} while (0)
#endif

namespace eliza_kokoro {

namespace {

static constexpr double K_PI = 3.14159265358979323846;

// ----------------------------------------------------------------------------
// Periodic Hann window of length N: w[i] = 0.5 - 0.5*cos(2*pi*i/N).
// Matches scipy.get_window('hann', N, fftbins=True) / torch.hann_window(N,
// periodic=True).
// ----------------------------------------------------------------------------
static std::vector<float> hann_periodic(int n) {
    std::vector<float> w((size_t) n);
    const double scale = 2.0 * K_PI / (double) n;
    for (int i = 0; i < n; ++i) {
        w[(size_t) i] = (float) (0.5 - 0.5 * std::cos(scale * (double) i));
    }
    return w;
}

// ----------------------------------------------------------------------------
// PyTorch F.interpolate(mode='linear', align_corners=False), 1D.
// in: [C, T_in] (channel-major) -> out: [C, T_out].
// Matches the half-pixel coordinate transform PyTorch uses by default.
// ----------------------------------------------------------------------------
static void interp_linear(const float * x, int C, int T_in, int T_out, float * y) {
    const double scale = (double) T_in / (double) T_out;
    for (int o = 0; o < T_out; ++o) {
        // src = (o + 0.5) * scale - 0.5  (align_corners=False half-pixel)
        double src = ((double) o + 0.5) * scale - 0.5;
        if (src < 0.0) src = 0.0;
        int s0 = (int) std::floor(src);
        int s1 = s0 + 1;
        double frac = src - (double) s0;
        if (s1 > T_in - 1) { s1 = T_in - 1; }
        if (s0 > T_in - 1) { s0 = T_in - 1; }
        for (int c = 0; c < C; ++c) {
            const float a = x[(size_t) c * T_in + s0];
            const float b = x[(size_t) c * T_in + s1];
            y[(size_t) c * T_out + o] = (float) (a + (b - a) * frac);
        }
    }
}

// ----------------------------------------------------------------------------
// Forward STFT, center=True, return polar (magnitude, phase).
// input: signal[N]; n_fft, hop, win (win==n_fft here). Reflect-pad n_fft/2
// each side (PyTorch center=True default pad_mode='reflect'). Periodic Hann
// window. Output: mag[F, n_frames], phase[F, n_frames], F=n_fft/2+1,
// n_frames = N/hop + 1.
// ----------------------------------------------------------------------------
// Canonical source STFT (GeneratorConfig::canonical_source_stft): the same
// centred DFT with a double-precision window and basis tables, then values
// at or below the frame's rounding floor (1e-12 * sum|x*w|) set to exactly 0
// and torch.angle's convention for the phase (pi for a negative real value).
// A float-rounded window would leave ~1e-8 residue in the empty bins of a
// constant frame, above any useful floor, so the window here stays double.
static void stft_center_canonical(const std::vector<float> & padded, int n_fft, int hop, int win,
                                  int n_frames, std::vector<float> & mag, std::vector<float> & phase) {
    const int F = n_fft / 2 + 1;
    std::vector<double> w((size_t) win), cs((size_t) F * win), sn((size_t) F * win);
    for (int k = 0; k < win; ++k) w[(size_t) k] = 0.5 - 0.5 * std::cos(2.0 * K_PI * (double) k / (double) win);
    for (int f = 0; f < F; ++f) {
        for (int k = 0; k < win; ++k) {
            const double ang = 2.0 * K_PI * (double) f * (double) k / (double) n_fft;
            cs[(size_t) f * win + k] = std::cos(ang) * w[(size_t) k];
            sn[(size_t) f * win + k] = -std::sin(ang) * w[(size_t) k];
        }
    }
    std::vector<double> x((size_t) win);
    for (int t = 0; t < n_frames; ++t) {
        const int off = t * hop;
        double floor_abs = 0.0;
        for (int k = 0; k < win; ++k) {
            x[(size_t) k] = (double) padded[(size_t) (off + k)];
            floor_abs += std::fabs(x[(size_t) k] * w[(size_t) k]);
        }
        floor_abs *= 1e-12;
        for (int f = 0; f < F; ++f) {
            double re = 0.0, im = 0.0;
            const double * c = cs.data() + (size_t) f * win;
            const double * s = sn.data() + (size_t) f * win;
            for (int k = 0; k < win; ++k) {
                re += x[(size_t) k] * c[k];
                im += x[(size_t) k] * s[k];
            }
            if (std::fabs(re) <= floor_abs) re = 0.0;
            if (std::fabs(im) <= floor_abs) im = 0.0;
            mag[(size_t) f * n_frames + t] = (float) std::sqrt(re * re + im * im);
            phase[(size_t) f * n_frames + t] =
                (im == 0.0 && re < 0.0) ? (float) K_PI : (float) std::atan2(im, re);
        }
    }
}

static void stft_center(const float * sig, int N, int n_fft, int hop, int win,
                        std::vector<float> & mag, std::vector<float> & phase,
                        int & F, int & n_frames, bool canonical) {
    F = n_fft / 2 + 1;
    const int pad = n_fft / 2;
    const int Np = N + 2 * pad;
    std::vector<float> padded((size_t) Np);
    // reflect pad: padded[pad + i] = sig[i]; left/right reflect (no edge dup).
    for (int i = 0; i < N; ++i) padded[(size_t) (pad + i)] = sig[i];
    for (int i = 0; i < pad; ++i) {
        padded[(size_t) (pad - 1 - i)] = sig[std::min(i + 1, N - 1)];
        padded[(size_t) (pad + N + i)] = sig[std::max(N - 2 - i, 0)];
    }
    n_frames = N / hop + 1;
    const std::vector<float> window = hann_periodic(win);
    mag.assign((size_t) F * n_frames, 0.0f);
    phase.assign((size_t) F * n_frames, 0.0f);
    if (canonical) {
        stft_center_canonical(padded, n_fft, hop, win, n_frames, mag, phase);
        return;
    }
    for (int t = 0; t < n_frames; ++t) {
        const int off = t * hop;
        for (int f = 0; f < F; ++f) {
            double re = 0.0, im = 0.0;
            const double w0 = -2.0 * K_PI * (double) f / (double) n_fft;
            for (int k = 0; k < win; ++k) {
                const double v = (double) padded[(size_t) (off + k)] * (double) window[(size_t) k];
                const double ang = w0 * (double) k;
                re += v * std::cos(ang);
                im += v * std::sin(ang);
            }
            {
                mag[(size_t) f * n_frames + t]   = (float) std::sqrt(re * re + im * im);
                phase[(size_t) f * n_frames + t] = (float) std::atan2(im, re);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Inverse STFT, center=True. magnitude/phase: [F, n_frames] (polar). Periodic
// Hann window, win==n_fft. Overlap-add with window^2 normalization, then trim
// n_fft/2 from each end (the center=True crop). Output length = (n_frames-1)*hop.
// ----------------------------------------------------------------------------
static void istft_center(const float * mag, const float * phase,
                         int n_fft, int hop, int win, int n_frames,
                         std::vector<float> & out) {
    const int F = n_fft / 2 + 1;
    const int pad = n_fft / 2;
    const int n_full = (n_frames - 1) * hop + win;  // before crop
    std::vector<double> acc((size_t) n_full, 0.0);
    std::vector<double> wsum((size_t) n_full, 0.0);
    const std::vector<float> window = hann_periodic(win);

    std::vector<double> frame((size_t) n_fft);
    // Basis tables and per-frame (re, im): the same double values the
    // straightforward triple loop computes, evaluated once instead of n_fft
    // times, so the sums (and their order) are unchanged.
    std::vector<double> cos_t((size_t) F * n_fft), sin_t((size_t) F * n_fft);
    for (int f = 0; f < F; ++f) {
        for (int n = 0; n < n_fft; ++n) {
            const double ang = 2.0 * K_PI * (double) f * (double) n / (double) n_fft;
            cos_t[(size_t) f * n_fft + n] = std::cos(ang);
            sin_t[(size_t) f * n_fft + n] = std::sin(ang);
        }
    }
    std::vector<double> re_f((size_t) F), im_f((size_t) F);
    for (int t = 0; t < n_frames; ++t) {
        for (int f = 0; f < F; ++f) {
            const double m = mag[(size_t) f * n_frames + t];
            const double p = phase[(size_t) f * n_frames + t];
            re_f[(size_t) f] = m * std::cos(p);
            im_f[(size_t) f] = m * std::sin(p);
        }
        // irfft of the hermitian spectrum -> n_fft real samples.
        for (int n = 0; n < n_fft; ++n) {
            double s = 0.0;
            for (int f = 0; f < F; ++f) {
                const double re = re_f[(size_t) f];
                const double im = im_f[(size_t) f];
                double term = re * cos_t[(size_t) f * n_fft + n] - im * sin_t[(size_t) f * n_fft + n];
                if (f != 0 && !(n_fft % 2 == 0 && f == F - 1)) term *= 2.0;
                s += term;
            }
            frame[(size_t) n] = s / (double) n_fft;
        }
        const int off = t * hop;
        for (int k = 0; k < win; ++k) {
            const double w = (double) window[(size_t) k];
            acc[(size_t) (off + k)]  += frame[(size_t) k] * w;
            wsum[(size_t) (off + k)] += w * w;
        }
    }
    for (int i = 0; i < n_full; ++i) {
        if (wsum[(size_t) i] > 1e-11) acc[(size_t) i] /= wsum[(size_t) i];
    }
    const int n_out = n_full - 2 * pad;
    out.assign((size_t) n_out, 0.0f);
    for (int i = 0; i < n_out; ++i) out[(size_t) i] = (float) acc[(size_t) (i + pad)];
}

// ----------------------------------------------------------------------------
// AdaINResBlock1.forward (Snake activation). x[C,T] in place.
// Three sub-blocks: adain1 -> snake(alpha1) -> conv1[dilated] ->
//                   adain2 -> snake(alpha2) -> conv2[dilation 1] -> residual add.
// convs1 dilations = dil[0..2]; convs2 dilation = 1.
// padding = get_padding(K, d) = (K*d - d)/2.
// ----------------------------------------------------------------------------
static int get_padding(int k, int d) { return (k * d - d) / 2; }

// Length after F.interpolate(scale_factor = 1 / upsample_scale).
static int Lds_of(int L, int upsample_scale) { return L / upsample_scale; }

static void adain_resblock1(const GenAdaResBlockWeights & w,
                            float * x, int C, int T, const float * s, int Sdim,
                            int K, const int dil[3]) {
    std::vector<float> xt((size_t) C * T), y1((size_t) C * T), y2((size_t) C * T);
    for (int i = 0; i < 3; ++i) {
        const GenSubBlockWeights & sb = w.sub[i];
        std::memcpy(xt.data(), x, sizeof(float) * (size_t) C * T);
        adain1d_forward(xt.data(), C, T, s, Sdim, sb.adain1_fc_w, sb.adain1_fc_b,
                        sb.adain1_norm_w, sb.adain1_norm_b);
        snake1d_forward(xt.data(), C, T, sb.alpha1);
        const int d1 = dil[i];
        conv1d_forward(xt.data(), C, T, sb.conv1_w, sb.conv1_b, C, K, 1,
                       get_padding(K, d1), d1, y1.data(), T);
        adain1d_forward(y1.data(), C, T, s, Sdim, sb.adain2_fc_w, sb.adain2_fc_b,
                        sb.adain2_norm_w, sb.adain2_norm_b);
        snake1d_forward(y1.data(), C, T, sb.alpha2);
        conv1d_forward(y1.data(), C, T, sb.conv2_w, sb.conv2_b, C, K, 1,
                       get_padding(K, 1), 1, y2.data(), T);
        for (size_t j = 0; j < (size_t) C * T; ++j) x[j] = y2[j] + x[j];
    }
}

} // namespace

// ============================================================================
// generator_forward
// ============================================================================
bool generator_forward(
        const float * x_in,      // [cfg.Cin, T0]
        int T0,                  // input time (== 2 * predictor T_frame)
        const float * s,         // [cfg.Sdim]
        const float * f0_curve,  // [T0]
        const GeneratorWeights & w,
        const GeneratorConfig & cfg,
        std::vector<float> & audio /* out [T0 * upsample_scale] */) {

    const int Sdim   = cfg.Sdim;
    const int up0    = cfg.up_rates[0], up1 = cfg.up_rates[1];    // upsample rates
    const int hop    = cfg.hop;
    const int n_fft  = cfg.n_fft, win = cfg.n_fft;
    const int upsample_scale = up0 * up1 * hop;  // Kokoro: 300
    const int dim = cfg.harmonics;  // harmonic_num + fundamental (Kokoro: 9)
    const float sine_amp = cfg.sine_amp;
    const float voiced_threshold = cfg.voiced_threshold;
    const float sr = cfg.sample_rate;
    const auto cancelled = [&]() {
        if (!cfg.cancel || !cfg.cancel(cfg.cancel_user)) return false;
        audio.clear();
        return true;
    };

    // Phase checkpoints (KOKORO_PROFILE=1): logs the delta since the previous
    // mark, attributing generator time to source/stft, the two upsample
    // stages, and conv_post+iSTFT.
    auto prof_last = std::chrono::steady_clock::now();
    auto prof_mark = [&prof_last](const char * label) {
        if (!kokoro_profile_enabled()) return;
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - prof_last).count();
        kokoro_profile_emit(label, ms);
        prof_last = now;
    };

    // ------------------------------------------------------------------
    // 1. Harmonic source.  f0_up = nearest-upsample(f0_curve) x300.
    // ------------------------------------------------------------------
    const int L = T0 * upsample_scale;  // 79200
    std::vector<float> f0_up((size_t) L);
    for (int t = 0; t < T0; ++t)
        for (int r = 0; r < upsample_scale; ++r)
            f0_up[(size_t) t * upsample_scale + r] = f0_curve[t];

    std::vector<float> sine_waves((size_t) dim * L);
    if (!cfg.precise_source) {
        // fn = f0 * [1..9]; rad = (fn / sr) % 1 ; [dim, L] channel-major.
        std::vector<float> rad((size_t) dim * L);
        for (int h = 0; h < dim; ++h) {
            const float mul = (float) (h + 1);
            for (int t = 0; t < L; ++t) {
                float v = f0_up[(size_t) t] * mul / sr;
                v = v - std::floor(v);  // % 1
                rad[(size_t) h * L + t] = v;
            }
        }
        // Deterministic: rad[:,0,:] += 0 (no random initial phase).
        // _f02sine: downsample rad by 1/upsample_scale (linear), cumsum*2pi,
        // upsample by *upsample_scale (with phase scaled by upsample_scale), sin.
        const int Lds = L / upsample_scale;  // 264
        std::vector<float> rad_ds((size_t) dim * Lds);
        interp_linear(rad.data(), dim, L, Lds, rad_ds.data());
        // cumsum over time per channel, * 2*pi.
        std::vector<float> phase_ds((size_t) dim * Lds);
        for (int h = 0; h < dim; ++h) {
            double cum = 0.0;
            for (int t = 0; t < Lds; ++t) {
                cum += (double) rad_ds[(size_t) h * Lds + t];
                phase_ds[(size_t) h * Lds + t] = (float) (cum * 2.0 * K_PI);
            }
        }
        // F.interpolate(phase * upsample_scale, scale=upsample_scale, linear).
        std::vector<float> phase_scaled((size_t) dim * Lds);
        for (size_t i = 0; i < (size_t) dim * Lds; ++i)
            phase_scaled[i] = phase_ds[i] * (float) upsample_scale;
        std::vector<float> phase_up((size_t) dim * L);
        interp_linear(phase_scaled.data(), dim, Lds, L, phase_up.data());
        // sines = sin(phase) * sine_amp ; voiced mask uv = (f0 > thr).
        // sine_waves = sines * uv  (+ noise = 0). sine_merge = tanh(l_linear(sine_waves)).
        for (int h = 0; h < dim; ++h) {
            for (int t = 0; t < L; ++t) {
                const float uv = (f0_up[(size_t) t] > voiced_threshold) ? 1.0f : 0.0f;
                sine_waves[(size_t) h * L + t] =
                    std::sin(phase_up[(size_t) h * L + t]) * sine_amp * uv;
            }
        }
    } else {
        // The same SineGen._f02sine steps in double, so the phase of the top
        // harmonic (~1e5 rad after a few seconds) keeps sub-microradian
        // precision; values are rounded to float only as sine_waves.
        const double scale_dn = (double) L / (double) Lds_of(L, upsample_scale);
        const int Lds = Lds_of(L, upsample_scale);
        const double scale_up = (double) Lds / (double) L;
        std::vector<double> phase_ds((size_t) Lds);
        for (int h = 0; h < dim; ++h) {
            const double mul = (double) (h + 1);
            auto rad_at = [&](int t) {
                double v = (double) f0_up[(size_t) t] * mul / (double) sr;
                return v - std::floor(v);   // % 1
            };
            double cum = 0.0;
            for (int o = 0; o < Lds; ++o) {
                double src = ((double) o + 0.5) * scale_dn - 0.5;
                if (src < 0.0) src = 0.0;
                int s0 = std::min((int) src, L - 1);
                const int s1 = s0 < L - 1 ? s0 + 1 : s0;
                const double frac = std::min(std::max(src - (double) s0, 0.0), 1.0);
                const double r0 = rad_at(s0), r1 = rad_at(s1);
                cum += r0 * (1.0 - frac) + r1 * frac;
                phase_ds[(size_t) o] = cum * 2.0 * K_PI * (double) upsample_scale;
            }
            for (int t = 0; t < L; ++t) {
                double src = ((double) t + 0.5) * scale_up - 0.5;
                if (src < 0.0) src = 0.0;
                int s0 = std::min((int) src, Lds - 1);
                const int s1 = s0 < Lds - 1 ? s0 + 1 : s0;
                const double frac = std::min(std::max(src - (double) s0, 0.0), 1.0);
                const double ph = phase_ds[(size_t) s0] * (1.0 - frac) + phase_ds[(size_t) s1] * frac;
                const double uv = ((double) f0_up[(size_t) t] > (double) voiced_threshold) ? 1.0 : 0.0;
                sine_waves[(size_t) h * L + t] = (float) (std::sin(ph) * (double) sine_amp * uv);
            }
        }
    }
    // l_linear: Linear(9 -> 1). weight [1, 9], bias [1]. har_source[L].
    std::vector<float> har_source((size_t) L);
    for (int t = 0; t < L; ++t) {
        double acc = w.l_linear_b[0];
        for (int h = 0; h < dim; ++h)
            acc += (double) w.l_linear_w[h] * (double) sine_waves[(size_t) h * L + t];
        har_source[(size_t) t] = std::tanh((float) acc);
    }
    if (cfg.har_source_in && cfg.har_source_in->size() == har_source.size()) {
        har_source = *cfg.har_source_in;
    }
    DBG("har_source", har_source.data(), har_source.size());
    if (cfg.har_source_out) *cfg.har_source_out = har_source;
    if (cancelled()) return false;

    // ------------------------------------------------------------------
    // 2. STFT of har_source -> har = cat(mag, phase) [22, n_frames].
    // ------------------------------------------------------------------
    std::vector<float> hmag, hphase;
    int F = 0, n_frames = 0;
    stft_center(har_source.data(), L, n_fft, hop, win, hmag, hphase, F, n_frames,
                cfg.canonical_source_stft);
    const int Hc = 2 * F;  // 22
    std::vector<float> har((size_t) Hc * n_frames);
    std::memcpy(har.data(), hmag.data(), sizeof(float) * (size_t) F * n_frames);
    std::memcpy(har.data() + (size_t) F * n_frames, hphase.data(),
                sizeof(float) * (size_t) F * n_frames);
    DBG("har", har.data(), har.size());
#ifdef KOKORO_GEN_INJECT_HAR
    {
        std::ifstream hf(KOKORO_GEN_INJECT_HAR, std::ios::binary);
        if (hf) { hf.read((char *) har.data(), sizeof(float) * har.size()); }
    }
#endif

    prof_mark("  gen: source+stft");

    // ------------------------------------------------------------------
    // 3. Upsample stages.
    // ------------------------------------------------------------------
    const int ups_rate[2]   = { up0, up1 };
    const int ups_k[2]      = { cfg.up_kernels[0], cfg.up_kernels[1] };
    const int ch_after[2]   = { cfg.ch[0], cfg.ch[1] };   // upsample_initial_channel >> (i+1)
    const int dil135[3]     = { cfg.dil[0], cfg.dil[1], cfg.dil[2] };

    // current x: [Cin, T0] (copy, mutable).
    int curC = cfg.Cin, curT = T0;
    std::vector<float> x(x_in, x_in + (size_t) curC * curT);

    for (int i = 0; i < 2; ++i) {
        if (cancelled()) return false;
        const int u = ups_rate[i], k = ups_k[i];
        const int outC = ch_after[i];

        // leaky_relu(x, 0.1) (in place).
        leaky_relu(x.data(), curC * curT, 0.1f);

        // x_source = noise_convs[i](har); then noise_res[i](x_source, s).
        // noise_convs[0]: Conv1d(22->256, k=stride_f0*2=12, stride=6, pad=(6+1)/2=3).
        // noise_convs[1]: Conv1d(22->128, k=1, stride=1, pad=0).
        std::vector<float> x_source;
        int xsT = 0;
        if (i == 0) {
            const int stride_f0 = up1;          // prod(upsample_rates[1:]) = 6
            const int nk = stride_f0 * 2;       // 12
            const int npad = (stride_f0 + 1) / 2;  // 3
            xsT = (n_frames + 2 * npad - (nk - 1) - 1) / stride_f0 + 1;
            x_source.assign((size_t) outC * xsT, 0.0f);
            conv1d_forward(har.data(), Hc, n_frames, w.noise_convs_w[i], w.noise_convs_b[i],
                           outC, nk, stride_f0, npad, 1, x_source.data(), xsT);
        } else {
            xsT = n_frames;  // k=1 stride=1 pad=0
            x_source.assign((size_t) outC * xsT, 0.0f);
            conv1d_forward(har.data(), Hc, n_frames, w.noise_convs_w[i], w.noise_convs_b[i],
                           outC, 1, 1, 0, 1, x_source.data(), xsT);
        }
        if (i == 0) DBG("noise_convs0", x_source.data(), x_source.size());
        const int nr_k = (i == 0) ? 7 : 11;
        adain_resblock1(w.noise_res[i], x_source.data(), outC, xsT, s, Sdim, nr_k, dil135);
        if (i == 0) DBG("noise_res0", x_source.data(), x_source.size());
        prof_mark(i == 0 ? "  gen: stage0 noise" : "  gen: stage1 noise");

        // x = ups[i](x). ConvTranspose1d(curC -> outC, k, stride=u, pad=(k-u)/2).
        const int tpad = (k - u) / 2;
        const int upT = convtranspose1d_out_len(curT, k, u, tpad, /*output_pad*/0);
        std::vector<float> xup((size_t) outC * upT);
        convtranspose1d_forward(x.data(), curC, curT, w.ups_w[i], w.ups_b[i], outC, k,
                                u, tpad, 0, xup.data(), upT);

        // Final stage: reflection_pad(1,0) -> T grows by 1 on the left.
        int xT = upT;
        std::vector<float> xpad;
        const float * xptr = xup.data();
        if (i == 1) {
            xT = upT + 1;
            xpad.assign((size_t) outC * xT, 0.0f);
            for (int c = 0; c < outC; ++c) {
                // ReflectionPad1d((1,0)): left pad reflects index 1.
                xpad[(size_t) c * xT + 0] = xup[(size_t) c * upT + 1];
                std::memcpy(xpad.data() + (size_t) c * xT + 1,
                            xup.data() + (size_t) c * upT, sizeof(float) * (size_t) upT);
            }
            xptr = xpad.data();
        }

        // x = x + x_source  (shapes match: xT == xsT).
        std::vector<float> xs((size_t) outC * xT);
        for (size_t j = 0; j < (size_t) outC * xT; ++j) xs[j] = xptr[j] + x_source[j];
        prof_mark(i == 0 ? "  gen: stage0 upsample" : "  gen: stage1 upsample");

        // resblocks: xs_sum = sum_j resblock[i*3+j](x, s) / 3.
        std::vector<float> accum((size_t) outC * xT, 0.0f);
        const int rb_k[3] = { cfg.rb_kernels[0], cfg.rb_kernels[1], cfg.rb_kernels[2] };
        for (int j = 0; j < 3; ++j) {
            if (cancelled()) return false;
            std::vector<float> rb(xs);  // copy current x
            adain_resblock1(w.resblocks[i * 3 + j], rb.data(), outC, xT, s, Sdim,
                            rb_k[j], dil135);
            for (size_t q = 0; q < (size_t) outC * xT; ++q) accum[q] += rb[q];
        }
        x.assign((size_t) outC * xT, 0.0f);
        for (size_t q = 0; q < (size_t) outC * xT; ++q) x[q] = accum[q] / 3.0f;
        curC = outC; curT = xT;
        if (i == 0) DBG("stage0_x", x.data(), x.size());
        if (i == 1) DBG("stage1_x", x.data(), x.size());
        prof_mark(i == 0 ? "  gen: stage0 resblocks" : "  gen: stage1 resblocks");
    }

    // ------------------------------------------------------------------
    // 4. conv_post -> spec/phase -> iSTFT.
    // ------------------------------------------------------------------
    if (cancelled()) return false;
    leaky_relu(x.data(), curC * curT, 0.01f);  // F.leaky_relu (no slope arg) = default 0.01
    const int cpC = n_fft + 2;  // 22
    std::vector<float> cp((size_t) cpC * curT);
    conv1d_forward(x.data(), curC, curT, w.conv_post_w, w.conv_post_b, cpC, 7,
                   1, 3, 1, cp.data(), curT);
    DBG("conv_post", cp.data(), cp.size());

    // spec = exp(cp[:11]); phase = sin(cp[11:22]).
    const int post_F = n_fft / 2 + 1;  // 11
    std::vector<float> spec((size_t) post_F * curT), phase((size_t) post_F * curT);
    for (int f = 0; f < post_F; ++f) {
        for (int t = 0; t < curT; ++t) {
            spec[(size_t) f * curT + t]  = std::exp(cp[(size_t) f * curT + t]);
            phase[(size_t) f * curT + t] = std::sin(cp[(size_t) (post_F + f) * curT + t]);
        }
    }
    istft_center(spec.data(), phase.data(), n_fft, hop, win, curT, audio);
    prof_mark("  gen: post+istft");
    return true;
}

// ============================================================================
// kokoro_generator_forward (Kokoro-82M defaults)
// ============================================================================
void kokoro_generator_forward(
        const float * x_in,      // [512, T0]
        int T0,                  // input time (== 2 * predictor T_frame)
        const float * s,         // [128]
        const float * f0_curve,  // [T0]
        const GeneratorWeights & w,
        std::vector<float> & audio /* out [T0 * 300] */) {
    GeneratorConfig cfg;
    generator_forward(x_in, T0, s, f0_curve, w, cfg, audio);
}

} // namespace eliza_kokoro
