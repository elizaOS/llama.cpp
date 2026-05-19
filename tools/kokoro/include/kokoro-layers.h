// SPDX-License-Identifier: MIT
//
// kokoro-layers.h — small neural-net layer primitives used by the Kokoro
// predictor + decoder forwards. Plain CPU/scalar implementations — no GGML
// graph, no backend dispatch. The forward pass is line-for-line matched to
// the upstream `kokoro` Python package (modules.py + istftnet.py) so the
// trained weights produce numerically equivalent output.
//
// Tensor convention (matches PyTorch Conv1d / Linear):
//   - 1D feature maps: `[C, T]` (channel-major, time-minor).
//   - Linear weight: row-major `[out, in]`; bias `[out]`.
//   - Conv1d weight: `[out, in, k]` (PyTorch default, weight_norm-fused).
//   - LSTM weights: standard PyTorch `weight_ih_l0 [4*H, I]`, `weight_hh_l0 [4*H, H]`,
//     biases `[4*H]` each. Gate ordering: i, f, g, o (PyTorch order).

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace eliza_kokoro {

// =================================================================
// Tensor (lightweight host tensor — view or owned)
// =================================================================
struct Tensor1D {                  // [C]
    int C = 0;
    const float * data = nullptr;
};
struct Tensor2D {                  // [R, C] row-major
    int R = 0;
    int C = 0;
    const float * data = nullptr;  // size = R*C
    inline float get(int r, int c) const { return data[r * C + c]; }
};
struct Tensor3D {                  // [D1, D2, D3] row-major
    int D1 = 0, D2 = 0, D3 = 0;
    const float * data = nullptr;
    inline float get(int i, int j, int k) const { return data[((size_t)i * D2 + j) * D3 + k]; }
};

// =================================================================
// Linear: y = x @ W^T + b. W is [out, in], x is [..., in].
// =================================================================
inline void linear_forward(
        const float * x, int in_dim,
        const float * W, const float * b, int out_dim,
        float * y) {
    for (int i = 0; i < out_dim; ++i) {
        float acc = b ? b[i] : 0.0f;
        const float * w = W + i * in_dim;
        for (int j = 0; j < in_dim; ++j) acc += w[j] * x[j];
        y[i] = acc;
    }
}

// =================================================================
// Conv1d (PyTorch convention).  out = conv(x); x: [Cin, T]; weight: [Cout, Cin, K]; bias: [Cout].
// padding = (K-1)/2 * dilation by default (or specified). stride=1, groups=1.
// Output shape: [Cout, T] (when padding is set as below — "same" padding).
//
// For stride != 1, output T_out = (T + 2*pad - dilation*(K-1) - 1) / stride + 1.
// =================================================================
inline void conv1d_forward(
        const float * x, int Cin, int T,
        const float * W, const float * b, int Cout, int K,
        int stride, int pad, int dilation,
        float * y, int T_out) {
    for (int co = 0; co < Cout; ++co) {
        const float bias_v = b ? b[co] : 0.0f;
        for (int to = 0; to < T_out; ++to) {
            float acc = bias_v;
            const int t_in_origin = to * stride - pad;
            for (int ci = 0; ci < Cin; ++ci) {
                const float * w = W + (((size_t)co) * Cin + ci) * K;
                const float * xi = x + (size_t)ci * T;
                for (int k = 0; k < K; ++k) {
                    const int ti = t_in_origin + k * dilation;
                    if (ti >= 0 && ti < T) acc += w[k] * xi[ti];
                }
            }
            y[(size_t)co * T_out + to] = acc;
        }
    }
}

// =================================================================
// Conv1d with groups support (used by F0_conv / N_conv where groups=1 always,
// and by ConvTranspose1d pool with groups=dim_in for AdainResBlk1d).
// Output T_out matches `same` padding when stride==1.
//
// For grouped conv (groups>1): each group of Cin/groups input channels maps
// to Cout/groups output channels. Weight shape is [Cout, Cin/groups, K].
// =================================================================
inline void conv1d_grouped_forward(
        const float * x, int Cin, int T,
        const float * W, const float * b, int Cout, int K,
        int stride, int pad, int dilation, int groups,
        float * y, int T_out) {
    const int Cin_per_g  = Cin  / groups;
    const int Cout_per_g = Cout / groups;
    for (int g = 0; g < groups; ++g) {
        const int co_base = g * Cout_per_g;
        const int ci_base = g * Cin_per_g;
        for (int co_off = 0; co_off < Cout_per_g; ++co_off) {
            const int co = co_base + co_off;
            const float bias_v = b ? b[co] : 0.0f;
            for (int to = 0; to < T_out; ++to) {
                float acc = bias_v;
                const int t_in_origin = to * stride - pad;
                for (int ci_off = 0; ci_off < Cin_per_g; ++ci_off) {
                    const int ci = ci_base + ci_off;
                    const float * w = W + (((size_t)co) * Cin_per_g + ci_off) * K;
                    const float * xi = x + (size_t)ci * T;
                    for (int k = 0; k < K; ++k) {
                        const int ti = t_in_origin + k * dilation;
                        if (ti >= 0 && ti < T) acc += w[k] * xi[ti];
                    }
                }
                y[(size_t)co * T_out + to] = acc;
            }
        }
    }
}

// =================================================================
// ConvTranspose1d (PyTorch convention).
//   weight: [Cin, Cout/groups, K] (note: in-channel first for transpose).
//   bias: [Cout].
//   T_out = (T - 1) * stride - 2*pad + dilation*(K-1) + output_pad + 1
//   When groups == Cin == Cout (depthwise transpose, used in AdainResBlk1d
//   pool), the conv runs per-channel.
//
// `dilation` defaults to 1 (kokoro generator / AdainResBlk1d both use 1).
// =================================================================
inline int convtranspose1d_out_len(int T_in, int K, int stride, int pad, int output_pad) {
    return (T_in - 1) * stride - 2 * pad + (K - 1) + output_pad + 1;
}

// groups=1 case.
inline void convtranspose1d_forward(
        const float * x, int Cin, int T,
        const float * W, const float * b, int Cout, int K,
        int stride, int pad, int output_pad,
        float * y, int T_out) {
    // output_pad affects T_out (computed by caller) but does not change
    // the per-element formula here.
    (void)output_pad;
    // Zero output, then add contributions.
    std::memset(y, 0, sizeof(float) * (size_t)Cout * (size_t)T_out);
    if (b) {
        for (int co = 0; co < Cout; ++co) {
            float * yc = y + (size_t)co * T_out;
            for (int to = 0; to < T_out; ++to) yc[to] += b[co];
        }
    }
    for (int ci = 0; ci < Cin; ++ci) {
        const float * xi = x + (size_t)ci * T;
        for (int t_in = 0; t_in < T; ++t_in) {
            const float xv = xi[t_in];
            if (xv == 0.0f) continue;
            for (int co = 0; co < Cout; ++co) {
                const float * w = W + (((size_t)ci) * Cout + co) * K;
                float * yc = y + (size_t)co * T_out;
                for (int k = 0; k < K; ++k) {
                    const int to = t_in * stride - pad + k;
                    if (to >= 0 && to < T_out) yc[to] += w[k] * xv;
                }
            }
        }
    }
}

// Depthwise transpose (groups == Cin == Cout). Used by AdainResBlk1d pool.
// weight shape: [Cin, 1, K].
inline void convtranspose1d_depthwise_forward(
        const float * x, int C, int T,
        const float * W, const float * b, int K,
        int stride, int pad, int output_pad,
        float * y, int T_out) {
    // output_pad affects T_out (computed by caller) but does not change
    // the per-element formula here.
    (void)output_pad;
    std::memset(y, 0, sizeof(float) * (size_t)C * (size_t)T_out);
    for (int c = 0; c < C; ++c) {
        const float * xi = x + (size_t)c * T;
        const float * w  = W + (size_t)c * K;  // [1, K] per channel
        float * yc = y + (size_t)c * T_out;
        const float bias_v = b ? b[c] : 0.0f;
        for (int to = 0; to < T_out; ++to) yc[to] = bias_v;
        for (int t_in = 0; t_in < T; ++t_in) {
            const float xv = xi[t_in];
            if (xv == 0.0f) continue;
            for (int k = 0; k < K; ++k) {
                const int to = t_in * stride - pad + k;
                if (to >= 0 && to < T_out) yc[to] += w[k] * xv;
            }
        }
    }
}

// =================================================================
// InstanceNorm1d (affine=True; PyTorch default `track_running_stats=False`).
// Per-channel mean/var across the time dimension. No running stats — pure
// instance norm.
// =================================================================
inline void instance_norm1d_forward(
        float * x, int C, int T, float eps = 1e-5f,
        const float * affine_w = nullptr,  // [C], may be null (= no affine)
        const float * affine_b = nullptr   // [C], may be null
) {
    for (int c = 0; c < C; ++c) {
        float * xc = x + (size_t)c * T;
        double sum = 0, sumsq = 0;
        for (int t = 0; t < T; ++t) { const double v = xc[t]; sum += v; sumsq += v * v; }
        const double mean = sum / T;
        const double var  = sumsq / T - mean * mean;
        const double inv  = 1.0 / std::sqrt(var + (double)eps);
        const float aw = affine_w ? affine_w[c] : 1.0f;
        const float ab = affine_b ? affine_b[c] : 0.0f;
        for (int t = 0; t < T; ++t) {
            xc[t] = (float)((xc[t] - mean) * inv) * aw + ab;
        }
    }
}

// =================================================================
// LayerNorm (PyTorch nn.LayerNorm over last dim).
// In: x: [..., C]; W: [C]; B: [C]; eps.
// Applies per-sample normalization across the last `C` dim.
// =================================================================
inline void layer_norm_forward(
        float * x, int N, int C, const float * W, const float * B, float eps = 1e-5f) {
    for (int n = 0; n < N; ++n) {
        float * xv = x + (size_t)n * C;
        double sum = 0, sumsq = 0;
        for (int i = 0; i < C; ++i) { sum += xv[i]; sumsq += xv[i] * xv[i]; }
        const double mean = sum / C;
        const double var  = sumsq / C - mean * mean;
        const double inv  = 1.0 / std::sqrt(var + (double)eps);
        for (int i = 0; i < C; ++i) {
            xv[i] = (float)((xv[i] - mean) * inv) * (W ? W[i] : 1.0f) + (B ? B[i] : 0.0f);
        }
    }
}

// =================================================================
// LSTM cell forward (single time step).
// Input gates ordering matches PyTorch: i, f, g, o (where g is the cell-state
// proposal, tanh-activated).
//
// W_ih: [4H, I]; W_hh: [4H, H]; b_ih: [4H]; b_hh: [4H].
// =================================================================
inline void lstm_cell_step(
        const float * x, int I, int H,
        const float * h_prev, const float * c_prev,
        const float * W_ih, const float * b_ih,
        const float * W_hh, const float * b_hh,
        float * h_out, float * c_out,
        float * gates_scratch /* [4H] */ ) {
    // gates = W_ih @ x + b_ih + W_hh @ h_prev + b_hh
    for (int g = 0; g < 4 * H; ++g) {
        float acc = (b_ih ? b_ih[g] : 0.0f) + (b_hh ? b_hh[g] : 0.0f);
        const float * w_ih_row = W_ih + (size_t)g * I;
        for (int j = 0; j < I; ++j) acc += w_ih_row[j] * x[j];
        const float * w_hh_row = W_hh + (size_t)g * H;
        for (int j = 0; j < H; ++j) acc += w_hh_row[j] * h_prev[j];
        gates_scratch[g] = acc;
    }
    // Split into i, f, g, o; apply sigmoid/tanh.
    auto sigmoid = [](float v){ return 1.0f / (1.0f + std::exp(-v)); };
    for (int j = 0; j < H; ++j) {
        const float ig = sigmoid(gates_scratch[0 * H + j]);
        const float fg = sigmoid(gates_scratch[1 * H + j]);
        const float gg = std::tanh(gates_scratch[2 * H + j]);
        const float og = sigmoid(gates_scratch[3 * H + j]);
        const float c_new = fg * c_prev[j] + ig * gg;
        c_out[j] = c_new;
        h_out[j] = og * std::tanh(c_new);
    }
}

// =================================================================
// Bidirectional LSTM forward over a sequence x[T, I].
// Returns h_seq[T, 2H] (forward concat reverse, like PyTorch nn.LSTM).
//
// `scratch_hc` must be `4*H*sizeof(float)` for hidden+cell buffers + gates.
// We size internally per-call; this keeps the API simple.
// =================================================================
struct LSTMWeights {
    // Forward direction.
    const float * W_ih = nullptr;   // [4H, I]
    const float * W_hh = nullptr;   // [4H, H]
    const float * b_ih = nullptr;   // [4H]
    const float * b_hh = nullptr;   // [4H]
    // Reverse direction (set by caller if bidirectional).
    const float * W_ih_r = nullptr;
    const float * W_hh_r = nullptr;
    const float * b_ih_r = nullptr;
    const float * b_hh_r = nullptr;
};

inline void bilstm_forward(
        const float * x_seq /* [T, I] */, int T, int I, int H,
        const LSTMWeights & w,
        float * y_seq /* [T, 2H] */) {
    std::vector<float> h_fwd(H, 0.0f), c_fwd(H, 0.0f);
    std::vector<float> h_rev(H, 0.0f), c_rev(H, 0.0f);
    std::vector<float> gates(4 * H, 0.0f);

    // Forward pass.
    for (int t = 0; t < T; ++t) {
        std::vector<float> h_new(H), c_new(H);
        lstm_cell_step(x_seq + (size_t)t * I, I, H,
                       h_fwd.data(), c_fwd.data(),
                       w.W_ih, w.b_ih, w.W_hh, w.b_hh,
                       h_new.data(), c_new.data(), gates.data());
        for (int j = 0; j < H; ++j) y_seq[(size_t)t * 2 * H + j] = h_new[j];
        h_fwd = h_new; c_fwd = c_new;
    }
    // Reverse pass.
    std::fill(h_rev.begin(), h_rev.end(), 0.0f);
    std::fill(c_rev.begin(), c_rev.end(), 0.0f);
    for (int t = T - 1; t >= 0; --t) {
        std::vector<float> h_new(H), c_new(H);
        lstm_cell_step(x_seq + (size_t)t * I, I, H,
                       h_rev.data(), c_rev.data(),
                       w.W_ih_r, w.b_ih_r, w.W_hh_r, w.b_hh_r,
                       h_new.data(), c_new.data(), gates.data());
        for (int j = 0; j < H; ++j) y_seq[(size_t)t * 2 * H + H + j] = h_new[j];
        h_rev = h_new; c_rev = c_new;
    }
}

// =================================================================
// AdaIN1d forward — Instance-Norm with per-sample scale/shift from style.
// Mirrors istftnet.AdaIN1d: gamma, beta = chunk(fc(s), 2, dim=1); out = (1+gamma)*InstanceNorm(x) + beta.
//
// In: x [C, T]; s [Sdim]; fc.weight [2C, Sdim]; fc.bias [2C].
// Out: x is normalized in-place (gamma, beta applied).
// =================================================================
inline void adain1d_forward(
        float * x, int C, int T,
        const float * s, int Sdim,
        const float * fc_W /* [2C, Sdim] */, const float * fc_b /* [2C] */) {
    std::vector<float> h(2 * C);
    linear_forward(s, Sdim, fc_W, fc_b, 2 * C, h.data());
    // gamma = h[0..C], beta = h[C..2C]
    // First: instance-norm x in-place (affine=True, but per-tensor; the
    // AdaIN1d Python code uses InstanceNorm1d(affine=True) but does NOT
    // chain (1+gamma)*affine*(x-mean)/std — instead it computes (1+gamma)*norm(x)+beta
    // where norm is the standard InstanceNorm. The `affine=True` weight/bias
    // are still applied. We bake them into the convert step by treating the
    // norm itself as plain (no learnable affine) — but the upstream code
    // does have learnable InstanceNorm affine params. Since the upstream
    // .pth has no `norm.weight` / `norm.bias` recorded under .norm1/.norm2,
    // PyTorch's InstanceNorm1d(affine=True) defaults to weight=1, bias=0,
    // and they ARE saved when track_running_stats=False... but kokoro's
    // saved state_dict shows no norm.weight key under norm1/norm2 of
    // AdainResBlk1d, only fc.weight/fc.bias. The default-affine
    // weight=1/bias=0 is correct then.
    instance_norm1d_forward(x, C, T, 1e-5f, nullptr, nullptr);
    for (int c = 0; c < C; ++c) {
        const float gamma = h[c];
        const float beta  = h[C + c];
        float * xc = x + (size_t)c * T;
        const float scale = 1.0f + gamma;
        for (int t = 0; t < T; ++t) xc[t] = xc[t] * scale + beta;
    }
}

// =================================================================
// AdaLayerNorm forward (used in DurationEncoder).
// x: [C, T]; s: [Sdim]; fc: [2C, Sdim] + [2C].
// Applies LayerNorm over `C` dim per time step, then (1+gamma)*x + beta
// where gamma, beta = chunk(fc(s), 2, dim=1).
//
// The PyTorch impl transposes to put C on the last dim before F.layer_norm
// then transposes back — we operate in-place on [C, T] by per-time normalization.
// =================================================================
inline void adalayernorm_forward(
        float * x, int C, int T,
        const float * s, int Sdim,
        const float * fc_W /* [2C, Sdim] */, const float * fc_b /* [2C] */) {
    std::vector<float> h(2 * C);
    linear_forward(s, Sdim, fc_W, fc_b, 2 * C, h.data());
    // For each time step, layer-norm over channels.
    for (int t = 0; t < T; ++t) {
        double sum = 0, sumsq = 0;
        for (int c = 0; c < C; ++c) {
            const float v = x[(size_t)c * T + t];
            sum += v; sumsq += v * v;
        }
        const double mean = sum / C;
        const double var  = sumsq / C - mean * mean;
        const double inv  = 1.0 / std::sqrt(var + 1e-5);
        for (int c = 0; c < C; ++c) {
            const float gamma = h[c];
            const float beta  = h[C + c];
            const float xn = (float)((x[(size_t)c * T + t] - mean) * inv);
            x[(size_t)c * T + t] = (1.0f + gamma) * xn + beta;
        }
    }
}

// =================================================================
// AdainResBlk1d ("predictor's residual block").
//
// Python ref (modules.py / istftnet.py):
//   _residual(x, s):
//     x = norm1(x, s)          # AdaIN1d
//     x = leaky_relu(x, 0.2)
//     x = pool(x)              # ConvTranspose1d (groups=dim_in) when upsample
//     x = conv1(dropout(x))
//     x = norm2(x, s)
//     x = leaky_relu(x, 0.2)
//     x = conv2(dropout(x))
//   _shortcut(x):
//     x = upsample(x)          # F.interpolate scale=2 nearest if upsample
//     if learned_sc: x = conv1x1(x)
//   out = (residual + shortcut) * rsqrt(2)
//
// Inputs:
//   x [Cin, T]; s [Sdim]
// Outputs:
//   y [Cout, T_out]; T_out = T * (upsample ? 2 : 1) (set by caller).
// =================================================================
struct AdainResBlk1dWeights {
    int Cin = 0, Cout = 0, Sdim = 0, K = 3;
    const float * norm1_fc_w = nullptr;  // [2 Cin, Sdim]
    const float * norm1_fc_b = nullptr;  // [2 Cin]
    const float * norm2_fc_w = nullptr;  // [2 Cout, Sdim]
    const float * norm2_fc_b = nullptr;  // [2 Cout]
    const float * conv1_w    = nullptr;  // [Cout, Cin, K]
    const float * conv1_b    = nullptr;  // [Cout]
    const float * conv2_w    = nullptr;  // [Cout, Cout, K]
    const float * conv2_b    = nullptr;  // [Cout]
    const float * conv1x1_w  = nullptr;  // [Cout, Cin, 1] (if learned_sc)
    const float * conv1x1_b  = nullptr;  // [Cout]
    const float * pool_w     = nullptr;  // [Cin, 1, 3] (ConvTranspose1d depthwise)
    const float * pool_b     = nullptr;  // [Cin]
    bool upsample = false;
};

// =================================================================
// Snake1D activation: x = x + (1/a) * sin(a*x)^2
// =================================================================
inline void snake1d_forward(float * x, int C, int T, const float * a /* [C] */) {
    for (int c = 0; c < C; ++c) {
        const float av = std::abs(a[c]) > 1e-8f ? a[c] : 1e-8f;
        const float inv_a = 1.0f / av;
        float * xc = x + (size_t)c * T;
        for (int t = 0; t < T; ++t) {
            const float s = std::sin(av * xc[t]);
            xc[t] = xc[t] + inv_a * s * s;
        }
    }
}

inline void leaky_relu(float * x, int N, float slope = 0.2f) {
    for (int i = 0; i < N; ++i) if (x[i] < 0) x[i] *= slope;
}

} // namespace eliza_kokoro
