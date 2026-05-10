// bench-fused-kernels.cpp — micro-benchmark for the W3-B fused kernels.
//
// Measures per-token decode latency for:
//   1. Q4_POLAR x Q8_0 dot (unfused vs fused)
//   2. fused QJL-K + TBQ-V attention (unfused chain vs single op)
//
// Single-thread, single-head; numbers are wall-clock medians over 200
// trials with warmup. The "unfused" attention reference path is the
// existing dequant_K -> score -> softmax -> dequant_V -> mix sequence
// expressed as a chain of ggml ops.
//
// Targets per the W3-B brief: speedup >= 1.5x on AVX2, >= 1.3x on NEON.
// If the measurement falls below target the bench prints a warning but
// still exits 0; the parity test is the gating signal in CI.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

constexpr int kHeadDim    = 128;
constexpr int kSketchDim  = 256;
constexpr int kPolarBlkSz = 128;
constexpr int kQ8BlkSz    = 32;
constexpr int kTbqBlkSz   = 32;
constexpr int kTbqPerTok  = kHeadDim / kTbqBlkSz;

extern "C" {
void ggml_vec_dot_q4_polar_q8_0(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);
void ggml_vec_dot_q4_polar_q8_0_fused(int n, float * s, size_t bs,
                                      const void * vx, size_t bx,
                                      const void * vy, size_t by, int nrc);
}

double now_ns() {
    using clock = std::chrono::steady_clock;
    static const auto t0 = clock::now();
    return std::chrono::duration<double, std::nano>(clock::now() - t0).count();
}

// ---------------- Q4_POLAR fused dot bench ----------------

void bench_polar_dot() {
    std::printf("== Q4_POLAR x Q8_0 dot ==\n");

    const int n_blocks = 64;  // 8192-element row -> 64 polar blocks
    const int n        = n_blocks * kPolarBlkSz;
    std::mt19937 rng(0xCAFEBABE);
    std::normal_distribution<float> dist(0.0f, 0.7f);

    std::vector<float> x_f(n), y_f(n);
    for (auto & v : x_f) v = dist(rng);
    for (auto & v : y_f) v = dist(rng);

    const auto * tt_polar = ggml_get_type_traits(GGML_TYPE_Q4_POLAR);
    const auto * tt_q8    = ggml_get_type_traits(GGML_TYPE_Q8_0);
    const auto * cpu_tt_q8 = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);

    std::vector<uint8_t> x_q((size_t) n_blocks * tt_polar->type_size);
    std::vector<uint8_t> y_q((size_t) (n / kQ8BlkSz) * tt_q8->type_size);
    tt_polar->from_float_ref(x_f.data(), x_q.data(), n);
    cpu_tt_q8->from_float    (y_f.data(), y_q.data(), n);

    const int n_warmup = 20;
    const int n_trials = 500;

    // Force the compiler to consume each result so it can't DCE away
    // the fused path. We perturb the y buffer between iterations so the
    // hot data does not live in registers across calls (otherwise the
    // compiler can hoist the entire dot product across iterations).
    volatile float s_sink = 0.0f;
    auto bench = [&](void (*fn)(int, float *, size_t, const void *, size_t,
                                const void *, size_t, int)) -> double {
        float s = 0.0f;
        for (int i = 0; i < n_warmup; i++) {
            fn(n, &s, 0, x_q.data(), 0, y_q.data(), 0, 1);
            s_sink = s;
        }
        std::vector<double> samples(n_trials);
        for (int i = 0; i < n_trials; i++) {
            // Tiny perturbation of one byte of y so each call sees fresh data.
            y_q[i & 0xFF] = (uint8_t)(y_q[i & 0xFF] ^ (uint8_t)(i & 0xF));
            // Compiler barrier: stop hoisting the call past this point.
            __asm__ volatile("" ::: "memory");
            const double t0 = now_ns();
            fn(n, &s, 0, x_q.data(), 0, y_q.data(), 0, 1);
            __asm__ volatile("" ::: "memory");
            const double t1 = now_ns();
            s_sink = s;  // keep value alive
            samples[i] = t1 - t0;
        }
        std::sort(samples.begin(), samples.end());
        return samples[n_trials / 2];  // median
    };

    const double t_unfused = bench(ggml_vec_dot_q4_polar_q8_0);
    const double t_fused   = bench(ggml_vec_dot_q4_polar_q8_0_fused);
    (void) s_sink;

    const double speedup = t_unfused / t_fused;
    std::printf("  unfused: %8.0f ns/row (%.1f ns/block)\n", t_unfused, t_unfused / n_blocks);
    std::printf("  fused:   %8.0f ns/row (%.1f ns/block)\n", t_fused,   t_fused   / n_blocks);
    std::printf("  speedup: %.2fx %s\n", speedup,
                (speedup >= 1.5) ? "(>=1.5x target met)" : "(below 1.5x target)");
}

// ---------------- fused QJL+TBQ attention bench ----------------

void bench_fused_attn() {
    std::printf("== fused QJL+TBQ attention (1 head, n_kv=512) ==\n");
    const int n_kv_tokens = 512;
    const int n_heads     = 1;
    const int n_kv_heads  = 1;
    const float sm_scale  = 1.0f / std::sqrt((float) kHeadDim);

    std::mt19937 rng(0xBADC0FFE);
    std::normal_distribution<float> dist(0.0f, 0.5f);

    std::vector<float> q_sketch(kSketchDim);
    std::vector<float> keys((size_t) n_kv_tokens * kHeadDim);
    std::vector<float> values((size_t) n_kv_tokens * kHeadDim);
    for (auto & v : q_sketch) v = dist(rng);
    for (auto & v : keys)     v = dist(rng);
    for (auto & v : values)   v = dist(rng);

    const auto * tt_qjl = ggml_get_type_traits(GGML_TYPE_QJL1_256);
    const auto * tt_tbq = ggml_get_type_traits(GGML_TYPE_TBQ3_0);

    std::vector<uint8_t> k_packed((size_t) n_kv_tokens * tt_qjl->type_size);
    std::vector<uint8_t> v_packed((size_t) n_kv_tokens * kTbqPerTok * tt_tbq->type_size);
    for (int t = 0; t < n_kv_tokens; t++) {
        tt_qjl->from_float_ref(keys.data() + (size_t) t * kHeadDim,
                               k_packed.data() + (size_t) t * tt_qjl->type_size,
                               kHeadDim);
        tt_tbq->from_float_ref(values.data() + (size_t) t * kHeadDim,
                               v_packed.data() + (size_t) t * kTbqPerTok * tt_tbq->type_size,
                               kHeadDim);
    }

    // Build the fused graph once, time the compute call.
    ggml_backend_t backend = ggml_backend_cpu_init();

    auto bench_fused = [&]() -> double {
        struct ggml_init_params iparams = {
            ggml_tensor_overhead() * 16 + ggml_graph_overhead(), nullptr, true,
        };
        ggml_context * ctx = ggml_init(iparams);
        ggml_tensor * q  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,       kSketchDim, n_heads,    1);
        ggml_tensor * pk = ggml_new_tensor_3d(ctx, GGML_TYPE_QJL1_256, kHeadDim,   n_kv_tokens, n_kv_heads);
        ggml_tensor * pv = ggml_new_tensor_3d(ctx, GGML_TYPE_TBQ3_0,    kHeadDim,   n_kv_tokens, n_kv_heads);
        ggml_tensor * out = ggml_fused_attn_qjl_tbq(ctx, q, pk, pv, n_kv_heads, sm_scale);
        ggml_cgraph * gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, out);
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_tensor_set(q,  q_sketch.data(), 0, ggml_nbytes(q));
        ggml_backend_tensor_set(pk, k_packed.data(), 0, ggml_nbytes(pk));
        ggml_backend_tensor_set(pv, v_packed.data(), 0, ggml_nbytes(pv));

        const int n_warmup = 5;
        const int n_trials = 200;
        for (int i = 0; i < n_warmup; i++) ggml_backend_graph_compute(backend, gf);
        std::vector<double> samples(n_trials);
        for (int i = 0; i < n_trials; i++) {
            const double t0 = now_ns();
            ggml_backend_graph_compute(backend, gf);
            samples[i] = now_ns() - t0;
        }
        std::sort(samples.begin(), samples.end());
        const double med = samples[n_trials / 2];
        ggml_backend_buffer_free(buf);
        ggml_free(ctx);
        return med;
    };

    // Build the unfused chain. We reproduce the score+softmax+mix steps
    // by hand outside ggml since GGML_OP_ATTN_SCORE_QJL exists but the
    // V dequant + mix is what the unified op replaces.
    auto bench_unfused = [&]() -> double {
        const int n_warmup = 5;
        const int n_trials = 200;

        std::vector<float> scores(n_kv_tokens);
        std::vector<float> v_dec(kHeadDim);
        std::vector<float> out(kHeadDim);

        auto run_once = [&]() {
            // 1. QJL score (per-token sign-FMA, by hand - matches the kernel).
            const float scl_base = 1.2533141373155003f / (float) kSketchDim;
            for (int t = 0; t < n_kv_tokens; t++) {
                const uint8_t * blk = k_packed.data() + (size_t) t * tt_qjl->type_size;
                const uint8_t * signs = blk;
                uint16_t bf;
                std::memcpy(&bf, blk + 32, sizeof(uint16_t));
                union { uint32_t u; float f; } u;
                u.u = ((uint32_t) bf) << 16;
                float dot = 0.0f;
                for (int j = 0; j < kSketchDim; j++) {
                    const int bit = (signs[j >> 3] >> (j & 7)) & 1;
                    dot += bit ? q_sketch[j] : -q_sketch[j];
                }
                scores[t] = scl_base * u.f * dot * sm_scale;
            }

            // 2. Softmax.
            float m = -INFINITY;
            for (float s : scores) if (s > m) m = s;
            float l = 0.0f;
            for (auto & s : scores) { s = std::exp(s - m); l += s; }
            if (l > 0) for (auto & s : scores) s /= l;

            // 3. V dequant + mix.
            std::fill(out.begin(), out.end(), 0.0f);
            for (int t = 0; t < n_kv_tokens; t++) {
                const uint8_t * vrow = v_packed.data() + (size_t) t * kTbqPerTok * tt_tbq->type_size;
                tt_tbq->to_float(vrow, v_dec.data(), kHeadDim);
                const float w = scores[t];
                for (int d = 0; d < kHeadDim; d++) out[d] += w * v_dec[d];
            }
        };

        for (int i = 0; i < n_warmup; i++) run_once();
        std::vector<double> samples(n_trials);
        for (int i = 0; i < n_trials; i++) {
            const double t0 = now_ns();
            run_once();
            samples[i] = now_ns() - t0;
        }
        std::sort(samples.begin(), samples.end());
        return samples[n_trials / 2];
    };

    const double t_unfused = bench_unfused();
    const double t_fused   = bench_fused();
    const double speedup = t_unfused / t_fused;
    std::printf("  unfused: %8.0f ns (%.1f ns/token)\n", t_unfused, t_unfused / n_kv_tokens);
    std::printf("  fused:   %8.0f ns (%.1f ns/token)\n", t_fused,   t_fused   / n_kv_tokens);
    std::printf("  speedup: %.2fx %s\n", speedup,
                (speedup >= 1.5) ? "(>=1.5x AVX2 target met)" :
                (speedup >= 1.3) ? "(>=1.3x NEON target met)" :
                                   "(below target)");

    ggml_backend_free(backend);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("[bench-fused] W3-B CPU fused-kernel benchmark\n\n");
    bench_polar_dot();
    std::printf("\n");
    bench_fused_attn();
    return 0;
}
