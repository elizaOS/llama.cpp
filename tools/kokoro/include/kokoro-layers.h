// SPDX-License-Identifier: MIT
//
// Neural-network primitives for the Kokoro predictor and decoder. Production
// convolution layers dispatch through GGML while the portable and scalar
// implementations remain numerical references for parity testing. The forward
// equations match the upstream `kokoro` Python package (modules.py +
// istftnet.py), so backend selection cannot change the trained model contract.
//
// The O(C^2*K*T) hot loops (Conv1d, ConvTranspose1d, Linear, LSTM gate
// matvecs) have three implementations, selected at compile time:
//
//   1. KOKORO_USE_ACCELERATE (Apple): Accelerate BLAS (im2col + sgemm /
//      sgemv, AMX-backed) — same math, same fp32 accumulation, ~2 orders
//      of magnitude faster than the scalar loops on M-series.
//   2. KOKORO_USE_PORTABLE_FAST (everything else): a small internal
//      std::thread pool parallelizes over output channels / gate rows, and
//      the innermost MAC loops run branch-free AXPY/dot kernels — NEON
//      (vfmaq_f32) on aarch64, plain auto-vectorizable loops elsewhere.
//      Thread count: min(hardware_concurrency, 16), override with
//      KOKORO_NUM_THREADS.
//   3. Pure scalar reference: define KOKORO_FORCE_SCALAR to compile only
//      the readable single-threaded loops (the numerical reference the
//      other two paths are validated against). Define KOKORO_NO_ACCELERATE
//      to take path 2 on Apple (used by the layer parity test).
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
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#if defined(KOKORO_USE_GGML_BACKEND)
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#endif

#if defined(__APPLE__) && !defined(KOKORO_NO_ACCELERATE) && !defined(KOKORO_FORCE_SCALAR)
#define KOKORO_USE_ACCELERATE 1
#if !defined(ACCELERATE_NEW_LAPACK)
#define ACCELERATE_NEW_LAPACK
#endif
#include <Accelerate/Accelerate.h>
#elif !defined(KOKORO_FORCE_SCALAR)
#define KOKORO_USE_PORTABLE_FAST 1
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#if defined(__aarch64__) || defined(_M_ARM64)
#define KOKORO_USE_NEON 1
#include <arm_neon.h>
#endif
#endif

namespace eliza_kokoro {

#if defined(KOKORO_USE_GGML_BACKEND)
namespace detail {

struct CachedF16Weight {
    size_t elements = 0;
    std::vector<uint8_t> storage;

    ggml_fp16_t * data() {
        const uintptr_t address = reinterpret_cast<uintptr_t>(storage.data());
        const uintptr_t aligned =
            (address + GGML_MEM_ALIGN - 1) & ~(uintptr_t) (GGML_MEM_ALIGN - 1);
        return reinterpret_cast<ggml_fp16_t *>(aligned);
    }
};

struct KokoroComputeState {
    ggml_backend_t backend = nullptr;
    std::unordered_map<const float *, CachedF16Weight> f16_weights;
};

inline thread_local KokoroComputeState * kokoro_compute_state = nullptr;

class KokoroComputeBackendScope {
public:
    explicit KokoroComputeBackendScope(KokoroComputeState * state)
        : previous_(kokoro_compute_state) {
        kokoro_compute_state = state;
    }

    ~KokoroComputeBackendScope() {
        kokoro_compute_state = previous_;
    }

    KokoroComputeBackendScope(const KokoroComputeBackendScope &) = delete;
    KokoroComputeBackendScope & operator=(const KokoroComputeBackendScope &) = delete;

private:
    KokoroComputeState * previous_;
};

inline bool ggml_compute_graph_and_copy(
        ggml_backend_t backend, ggml_context * ctx, ggml_tensor * output,
        float * destination, size_t elements) {
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, output);
    ggml_gallocr_t allocator = ggml_gallocr_new(
        ggml_backend_get_default_buffer_type(backend));
    if (!allocator) return false;
    const bool allocated = ggml_gallocr_alloc_graph(allocator, graph);
    const enum ggml_status status = allocated
        ? ggml_backend_graph_compute(backend, graph)
        : GGML_STATUS_ALLOC_FAILED;
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(
            output, destination, 0, sizeof(float) * elements);
    }
    ggml_gallocr_free(allocator);
    return status == GGML_STATUS_SUCCESS;
}

inline const ggml_fp16_t * cached_f16_weight(
        KokoroComputeState & state, const float * weight, size_t elements) {
    auto & cached = state.f16_weights[weight];
    if (cached.elements != elements) {
        cached.elements = elements;
        cached.storage.resize(
            elements * sizeof(ggml_fp16_t) + GGML_MEM_ALIGN - 1);
        ggml_fp32_to_fp16_row(weight, cached.data(), elements);
    }
    return cached.data();
}

inline bool ggml_conv1d_f32(
        const float * x, int Cin, int T,
        const float * W, const float * b, int Cout, int K,
        int stride, int pad, int dilation,
        float * y, int T_out) {
    KokoroComputeState * state = kokoro_compute_state;
    if (!state || !state->backend) return false;
    ggml_backend_t backend = state->backend;

    ggml_init_params params = {
        /* mem_size   = */ 2 * 1024 * 1024,
        /* mem_buffer = */ nullptr,
        /* no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * weight =
        ggml_new_tensor_3d(ctx, GGML_TYPE_F16, K, Cin, Cout);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, T, Cin, 1);
    weight->data = const_cast<ggml_fp16_t *>(cached_f16_weight(
        *state, W, (size_t) K * Cin * Cout));
    input->data = const_cast<float *>(x);

    ggml_tensor * output = ggml_conv_1d(
        ctx, weight, input, stride, pad, dilation);
    if (b) {
        ggml_tensor * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, Cout);
        bias->data = const_cast<float *>(b);
        output = ggml_add(ctx, output, bias);
    }

    const bool ok =
        output->ne[0] == T_out &&
        output->ne[1] == Cout &&
        ggml_compute_graph_and_copy(
            backend, ctx, output, y, (size_t) Cout * T_out);
    ggml_free(ctx);
    return ok;
}

inline bool ggml_convtranspose1d_f32(
        const float * x, int Cin, int T,
        const float * W, const float * b, int Cout, int K,
        int stride, int pad, int output_pad,
        float * y, int T_out) {
    KokoroComputeState * state = kokoro_compute_state;
    if (!state || !state->backend) return false;
    ggml_backend_t backend = state->backend;

    ggml_init_params params = {
        /* mem_size   = */ 2 * 1024 * 1024,
        /* mem_buffer = */ nullptr,
        /* no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) return false;

    ggml_tensor * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, Cout, Cin);
    ggml_tensor * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, Cin);
    weight->data = const_cast<float *>(W);
    input->data = const_cast<float *>(x);

    ggml_tensor * output = ggml_conv_transpose_1d(ctx, weight, input, stride, 0, 1);
    if (pad > 0) {
        const int64_t cropped = output->ne[0] - 2 * (int64_t) pad;
        output = ggml_view_2d(
            ctx, output, cropped, output->ne[1], output->nb[1],
            (size_t) pad * output->nb[0]);
        output = ggml_cont(ctx, output);
    }
    if (output_pad > 0) {
        output = ggml_pad(ctx, output, output_pad, 0, 0, 0);
    }
    if (b) {
        ggml_tensor * bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, Cout);
        bias->data = const_cast<float *>(b);
        output = ggml_add(ctx, output, bias);
    }

    const bool ok =
        output->ne[0] == T_out &&
        output->ne[1] == Cout &&
        ggml_compute_graph_and_copy(
            backend, ctx, output, y, (size_t) Cout * T_out);
    ggml_free(ctx);
    return ok;
}

} // namespace detail
#endif

#if defined(KOKORO_USE_PORTABLE_FAST)
namespace detail {

inline int kokoro_thread_count() {
    static const int n = [] {
        if (const char * e = std::getenv("KOKORO_NUM_THREADS")) {
            const int v = std::atoi(e);
            if (v >= 1) return v < 64 ? v : 64;
        }
        const unsigned hc = std::thread::hardware_concurrency();
        const unsigned capped = hc == 0 ? 4u : (hc > 16u ? 16u : hc);
        return (int) capped;
    }();
    return n;
}

// Tiny persistent thread pool. One pool per process (lazy singleton), no
// nested parallelism (callers below are all top-level layer forwards).
// Work items are claimed with an atomic counter; the calling thread
// participates, so a 1-thread pool degrades to the serial loop.
class KokoroThreadPool {
public:
    static KokoroThreadPool & instance() {
        static KokoroThreadPool pool(kokoro_thread_count() - 1);
        return pool;
    }

    // Runs fn(i) for every i in [0, n). Blocks until all items are done.
    void parallel_for(int n, const std::function<void(int)> & fn) {
        if (n <= 0) return;
        if (workers_.empty() || n == 1) {
            for (int i = 0; i < n; ++i) fn(i);
            return;
        }
        {
            std::lock_guard<std::mutex> lock(m_);
            task_ = &fn;
            n_ = n;
            next_.store(0, std::memory_order_relaxed);
            pending_ = (int) workers_.size();
            ++generation_;
        }
        cv_.notify_all();
        run_chunks();
        std::unique_lock<std::mutex> lock(m_);
        done_cv_.wait(lock, [&] { return pending_ == 0; });
        task_ = nullptr;
    }

private:
    explicit KokoroThreadPool(int n_workers) {
        for (int i = 0; i < n_workers; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }
    ~KokoroThreadPool() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
            ++generation_;
        }
        cv_.notify_all();
        for (auto & t : workers_) t.join();
    }

    void run_chunks() {
        const std::function<void(int)> * task = task_;
        const int n = n_;
        for (;;) {
            const int i = next_.fetch_add(1, std::memory_order_relaxed);
            if (i >= n) break;
            (*task)(i);
        }
    }

    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> lock(m_);
                cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
            }
            run_chunks();
            {
                std::lock_guard<std::mutex> lock(m_);
                if (--pending_ == 0) done_cv_.notify_one();
            }
        }
    }

    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_, done_cv_;
    const std::function<void(int)> * task_ = nullptr;
    std::atomic<int> next_{0};
    int n_ = 0;
    int pending_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

// Below this many MACs the pool dispatch overhead outweighs the win.
constexpr size_t KOKORO_PARALLEL_MIN_MACS = 1u << 18;  // 256k

inline void kokoro_parallel_for(size_t total_macs, int n, const std::function<void(int)> & fn) {
    if (total_macs < KOKORO_PARALLEL_MIN_MACS) {
        for (int i = 0; i < n; ++i) fn(i);
        return;
    }
    KokoroThreadPool::instance().parallel_for(n, fn);
}

// dot(a, b, n) — innermost MAC of Linear / LSTM gate rows.
inline float kokoro_dot_f32(const float * a, const float * b, int n) {
#if defined(KOKORO_USE_NEON)
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i),     vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float acc = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; i < n; ++i) acc += a[i] * b[i];
    return acc;
#else
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) acc += a[i] * b[i];
    return acc;
#endif
}

// y[i] += a * x[i] — innermost MAC of the (transposed) conv time loops.
inline void kokoro_axpy_f32(float a, const float * x, float * y, int n) {
#if defined(KOKORO_USE_NEON)
    const float32x4_t av = vdupq_n_f32(a);
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        vst1q_f32(y + i,     vfmaq_f32(vld1q_f32(y + i),     av, vld1q_f32(x + i)));
        vst1q_f32(y + i + 4, vfmaq_f32(vld1q_f32(y + i + 4), av, vld1q_f32(x + i + 4)));
    }
    for (; i < n; ++i) y[i] += a * x[i];
#else
    for (int i = 0; i < n; ++i) y[i] += a * x[i];
#endif
}

} // namespace detail
#endif // KOKORO_USE_PORTABLE_FAST

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
#if defined(KOKORO_USE_ACCELERATE)
    float beta = 0.0f;
    if (b) {
        std::memcpy(y, b, sizeof(float) * (size_t) out_dim);
        beta = 1.0f;
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, out_dim, in_dim,
                1.0f, W, in_dim, x, 1, beta, y, 1);
#elif defined(KOKORO_USE_PORTABLE_FAST)
    detail::kokoro_parallel_for((size_t) out_dim * in_dim, out_dim, [&](int i) {
        y[i] = (b ? b[i] : 0.0f) +
               detail::kokoro_dot_f32(W + (size_t) i * in_dim, x, in_dim);
    });
#else
    for (int i = 0; i < out_dim; ++i) {
        float acc = b ? b[i] : 0.0f;
        const float * w = W + i * in_dim;
        for (int j = 0; j < in_dim; ++j) acc += w[j] * x[j];
        y[i] = acc;
    }
#endif
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
#if defined(KOKORO_USE_GGML_BACKEND)
    if (detail::ggml_conv1d_f32(
            x, Cin, T, W, b, Cout, K, stride, pad, dilation, y, T_out)) {
        return;
    }
#endif
#if defined(KOKORO_USE_ACCELERATE)
    // im2col + one SGEMM. The PyTorch weight layout [Cout, Cin, K] is already
    // a row-major [Cout, Cin*K] GEMM operand; the column matrix is
    // [Cin*K, T_out] with zero padding baked in, so
    //   y[Cout, T_out] = W · cols (+ bias).
    std::vector<float> cols((size_t) Cin * K * T_out, 0.0f);
    for (int ci = 0; ci < Cin; ++ci) {
        const float * xi = x + (size_t) ci * T;
        for (int k = 0; k < K; ++k) {
            float * col = cols.data() + ((size_t) ci * K + k) * T_out;
            const int t_off = k * dilation - pad;  // ti = to*stride + t_off
            // valid `to` range keeping ti within [0, T)
            int to_lo = 0;
            if (t_off < 0) to_lo = (-t_off + stride - 1) / stride;
            int to_hi = (T - 1 - t_off) / stride;  // inclusive
            if (to_hi > T_out - 1) to_hi = T_out - 1;
            if (to_lo > to_hi) continue;
            if (stride == 1) {
                std::memcpy(col + to_lo, xi + (to_lo + t_off),
                            sizeof(float) * (size_t) (to_hi - to_lo + 1));
            } else {
                for (int to = to_lo; to <= to_hi; ++to) {
                    col[to] = xi[to * stride + t_off];
                }
            }
        }
    }
    float beta = 0.0f;
    if (b) {
        for (int co = 0; co < Cout; ++co) {
            float * yc = y + (size_t) co * T_out;
            for (int to = 0; to < T_out; ++to) yc[to] = b[co];
        }
        beta = 1.0f;
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                Cout, T_out, Cin * K,
                1.0f, W, Cin * K, cols.data(), T_out,
                beta, y, T_out);
#elif defined(KOKORO_USE_PORTABLE_FAST)
    // Parallel over output channels (each thread owns disjoint y rows), and
    // per (ci, k) tap a branch-free AXPY over the valid `to` range — the
    // per-element bounds check of the reference loop is hoisted into the
    // range computation (same trick as the Accelerate im2col above).
    const size_t total_macs = (size_t) Cout * T_out * Cin * K;
    detail::kokoro_parallel_for(total_macs, Cout, [&](int co) {
        float * yc = y + (size_t) co * T_out;
        const float bias_v = b ? b[co] : 0.0f;
        for (int to = 0; to < T_out; ++to) yc[to] = bias_v;
        for (int ci = 0; ci < Cin; ++ci) {
            const float * w  = W + (((size_t) co) * Cin + ci) * K;
            const float * xi = x + (size_t) ci * T;
            for (int k = 0; k < K; ++k) {
                const int t_off = k * dilation - pad;  // ti = to*stride + t_off
                int to_lo = 0;
                if (t_off < 0) to_lo = (-t_off + stride - 1) / stride;
                int to_hi = (T - 1 - t_off) / stride;  // inclusive
                if (to_hi > T_out - 1) to_hi = T_out - 1;
                if (to_lo > to_hi) continue;
                if (stride == 1) {
                    detail::kokoro_axpy_f32(w[k], xi + (to_lo + t_off),
                                            yc + to_lo, to_hi - to_lo + 1);
                } else {
                    const float wv = w[k];
                    for (int to = to_lo; to <= to_hi; ++to) {
                        yc[to] += wv * xi[to * stride + t_off];
                    }
                }
            }
        }
    });
#else
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
#endif
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
#if defined(KOKORO_USE_GGML_BACKEND)
    if (detail::ggml_convtranspose1d_f32(
            x, Cin, T, W, b, Cout, K, stride, pad, output_pad, y, T_out)) {
        return;
    }
#endif
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
#if defined(KOKORO_USE_ACCELERATE)
    // One SGEMM + col2im scatter. Viewing W [Cin, Cout, K] as a row-major
    // [Cin, Cout*K] operand:
    //   tmp[Cout*K, T] = W^T · x[Cin, T]
    // then tmp[co*K + k, t_in] scatters into y[co, t_in*stride - pad + k].
    std::vector<float> tmp((size_t) Cout * K * T);
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                Cout * K, T, Cin,
                1.0f, W, Cout * K, x, T,
                0.0f, tmp.data(), T);
    for (int co = 0; co < Cout; ++co) {
        float * yc = y + (size_t) co * T_out;
        for (int k = 0; k < K; ++k) {
            const float * row = tmp.data() + ((size_t) co * K + k) * T;
            const int t_off = k - pad;  // to = t_in*stride + t_off
            for (int t_in = 0; t_in < T; ++t_in) {
                const int to = t_in * stride + t_off;
                if (to >= 0 && to < T_out) yc[to] += row[t_in];
            }
        }
    }
#elif defined(KOKORO_USE_PORTABLE_FAST)
    // Parallel over output channels (each thread scatters into disjoint y
    // rows — no races). Two branch-free AXPY shapes, valid ranges hoisted
    // out of the inner loop:
    //   stride == 1: per (ci, k) tap, AXPY over the contiguous `to` run.
    //   stride  > 1: per (ci, t_in) sample, AXPY over the contiguous K taps
    //                (y[t_in*stride - pad ..] and w[0..K) are both contiguous).
    const size_t total_macs = (size_t) Cin * T * Cout * K;
    detail::kokoro_parallel_for(total_macs, Cout, [&](int co) {
        float * yc = y + (size_t) co * T_out;
        for (int ci = 0; ci < Cin; ++ci) {
            const float * w  = W + (((size_t) ci) * Cout + co) * K;
            const float * xi = x + (size_t) ci * T;
            if (stride == 1) {
                for (int k = 0; k < K; ++k) {
                    const int t_off = k - pad;  // to = t_in + t_off
                    int t_in_lo = t_off < 0 ? -t_off : 0;
                    int t_in_hi = T_out - 1 - t_off;  // inclusive
                    if (t_in_hi > T - 1) t_in_hi = T - 1;
                    if (t_in_lo > t_in_hi) continue;
                    detail::kokoro_axpy_f32(w[k], xi + t_in_lo,
                                            yc + (t_in_lo + t_off),
                                            t_in_hi - t_in_lo + 1);
                }
            } else {
                for (int t_in = 0; t_in < T; ++t_in) {
                    const float xv = xi[t_in];
                    if (xv == 0.0f) continue;
                    const int base = t_in * stride - pad;  // to = base + k
                    int k_lo = base < 0 ? -base : 0;
                    int k_hi = K - 1;  // inclusive
                    if (k_hi > T_out - 1 - base) k_hi = T_out - 1 - base;
                    if (k_lo > k_hi) continue;
                    detail::kokoro_axpy_f32(xv, w + k_lo, yc + (base + k_lo),
                                            k_hi - k_lo + 1);
                }
            }
        }
    });
#else
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
#endif
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
#if defined(KOKORO_USE_ACCELERATE)
    for (int g = 0; g < 4 * H; ++g) {
        gates_scratch[g] = (b_ih ? b_ih[g] : 0.0f) + (b_hh ? b_hh[g] : 0.0f);
    }
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, I,
                1.0f, W_ih, I, x, 1, 1.0f, gates_scratch, 1);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 4 * H, H,
                1.0f, W_hh, H, h_prev, 1, 1.0f, gates_scratch, 1);
#elif defined(KOKORO_USE_PORTABLE_FAST)
    // Parallel over gate rows; each row is two NEON/scalar dots. The MAC
    // threshold keeps small cells (per-step sequential LSTM) off the pool.
    const size_t total_macs = (size_t) 4 * H * (I + H);
    detail::kokoro_parallel_for(total_macs, 4 * H, [&](int g) {
        gates_scratch[g] =
            (b_ih ? b_ih[g] : 0.0f) + (b_hh ? b_hh[g] : 0.0f) +
            detail::kokoro_dot_f32(W_ih + (size_t) g * I, x, I) +
            detail::kokoro_dot_f32(W_hh + (size_t) g * H, h_prev, H);
    });
#else
    for (int g = 0; g < 4 * H; ++g) {
        float acc = (b_ih ? b_ih[g] : 0.0f) + (b_hh ? b_hh[g] : 0.0f);
        const float * w_ih_row = W_ih + (size_t)g * I;
        for (int j = 0; j < I; ++j) acc += w_ih_row[j] * x[j];
        const float * w_hh_row = W_hh + (size_t)g * H;
        for (int j = 0; j < H; ++j) acc += w_hh_row[j] * h_prev[j];
        gates_scratch[g] = acc;
    }
#endif
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
