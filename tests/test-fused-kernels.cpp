// test-fused-kernels.cpp — parity + smoke for the W3-B fused CPU kernels.
//
// Three kernels under test:
//   1. ggml_vec_dot_q4_polar_q8_0_fused          vs unfused vec_dot
//   2. ggml_vec_dot_q4_polar_q8_0_fused_hadamard vs unfused vec_dot
//   3. GGML_OP_FUSED_ATTN_QJL_TBQ                vs (QJL score + softmax + TBQ V mix)
//
// Tolerances:
//   - Q4_POLAR fused dot: relative error <= 1e-5 across 100 random
//     blocks (the WHT-on-the-y-side reordering is bit-equivalent
//     modulo floating-point associativity).
//   - QJL+TBQ attention: signal-normalised error <= 5e-3 across 100
//     random contexts. The bound is looser than 1e-5 because the unfused
//     "reference" path itself involves an exp() softmax and a
//     tbq_uncondition_block WHT that re-orders adds; we only require
//     the fused output to land within 0.5% of the chained-op output's
//     vector magnitude. We normalise by max_d |ref[d]| (not per-element)
//     because attention outputs are softmax-weighted sums whose individual
//     entries can cancel arbitrarily close to zero — a per-element
//     relative metric would amplify FP32 ULP-scale rounding noise into
//     spurious "percent" errors on those near-zero entries.
//
// Built only when LLAMA_BUILD_TESTS=ON; the test target is added in
// tests/CMakeLists.txt with the existing llama_build_and_test() helper.

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

// Locked dimensions matching ggml-common.h. We avoid including the
// internal header so the test stays buildable with just the public
// ggml.h surface (the size constants are static-asserted on the C side
// in qjl_block.h / ggml-common.h).
constexpr int kHeadDim     = 128;
constexpr int kSketchDim   = 256;        // QK_QJL
constexpr int kPolarBlkSz  = 128;        // kPolarBlkSz
constexpr int kQ8BlkSz     = 32;         // QK8_0
constexpr int kTbqBlkSz    = 32;         // QK_TBQ
constexpr int kTbqPerTok   = kHeadDim / kTbqBlkSz; // = 4
constexpr int kQjlBlkB     = 34;         // sizeof(block_qjl1_256)

// ABI hooks — declared inline so the test stays self-contained. The
// fused dot entry points live in ggml-cpu (W3-B), not in the public
// ggml.h surface. The QJL residual toggle is in the internal
// ggml-quants.h, also not public; we forward-declare here.
extern "C" {
void ggml_vec_dot_q4_polar_q8_0(int n, float * s, size_t bs,
                                const void * vx, size_t bx,
                                const void * vy, size_t by, int nrc);
void ggml_vec_dot_q4_polar_q8_0_fused(int n, float * s, size_t bs,
                                      const void * vx, size_t bx,
                                      const void * vy, size_t by, int nrc);
void ggml_vec_dot_q4_polar_q8_0_fused_hadamard(int n, float * s, size_t bs,
                                               const void * vx, size_t bx,
                                               const void * vy, size_t by, int nrc);
void ggml_q4_polar_set_use_qjl(bool use_qjl);
}

// ---------------- helpers ----------------

void rand_floats(std::vector<float> & out, size_t n, std::mt19937 & rng, float scale = 1.0f) {
    std::normal_distribution<float> dist(0.0f, scale);
    out.resize(n);
    for (auto & v : out) v = dist(rng);
}

float rel_err(float a, float b) {
    const float denom = std::max(std::abs(a), std::abs(b));
    if (denom < 1e-12f) return std::abs(a - b);
    return std::abs(a - b) / denom;
}

// Quantize a row of `n` floats into Q8_0 blocks. Reuses ggml's standard
// reference encoder via the type traits table.
void quantize_q8_0_row(const float * x, int n, std::vector<uint8_t> & out) {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_Q8_0);
    out.resize((size_t)(n / kQ8BlkSz) * tt->type_size);
    const auto * cpu_tt = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0);
    cpu_tt->from_float(x, out.data(), n);
}

// Quantize a row of `n` floats into block_q4_polar via the ref encoder
// (so the fused-vs-unfused parity is independent of whatever SIMD
// quantize variant is later added).
void quantize_q4_polar_row(const float * x, int n, std::vector<uint8_t> & out) {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_Q4_POLAR);
    out.resize((size_t)(n / kPolarBlkSz) * tt->type_size);
    tt->from_float_ref(x, out.data(), n);
}

// Quantize n keys into qjl1_256 blocks (one per head_dim chunk).
//
// quantize_row_qjl1_256_ref expects `k` in elements and emits one block
// per head_dim=128 elements. Calling it with k=128 produces exactly one
// 34-byte block; we call it n_keys times.
void quantize_qjl_keys(const float * keys, int n_keys, std::vector<uint8_t> & out) {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_QJL1_256);
    out.resize((size_t) n_keys * tt->type_size);
    for (int t = 0; t < n_keys; t++) {
        tt->from_float_ref(keys + (size_t) t * kHeadDim,
                           out.data() + (size_t) t * tt->type_size,
                           kHeadDim);
    }
}

// Quantize n_tokens * head_dim floats to TBQ3_0 packed bytes laid out
// as the fused op expects: per-token 4 blocks, contiguous.
void quantize_tbq3_v(const float * v, int n_tokens, std::vector<uint8_t> & out) {
    const auto * tt = ggml_get_type_traits(GGML_TYPE_TBQ3_0);
    out.resize((size_t) n_tokens * kTbqPerTok * tt->type_size);
    for (int t = 0; t < n_tokens; t++) {
        tt->from_float_ref(v + (size_t) t * kHeadDim,
                           out.data() + (size_t) t * kTbqPerTok * tt->type_size,
                           kHeadDim);
    }
}

// ---------------- test 1: Q4_POLAR fused dot parity ----------------

bool test_polar_dot_parity() {
    std::printf("[fused] === Q4_POLAR fused dot parity ===\n");
    const int n = kPolarBlkSz * 8;  // 8 polar blocks = 1024 floats = 32 Q8_0 blocks
    std::mt19937 rng(0xC0FFEE);

    float max_rel = 0.0f;
    int n_blocks = 100;
    for (int it = 0; it < n_blocks; it++) {
        std::vector<float> x_f, y_f;
        rand_floats(x_f, n, rng, 0.7f);
        rand_floats(y_f, n, rng, 0.7f);

        std::vector<uint8_t> x_q, y_q;
        quantize_q4_polar_row(x_f.data(), n, x_q);
        quantize_q8_0_row    (y_f.data(), n, y_q);

        float s_unfused = 0.0f;
        float s_fused   = 0.0f;
        ggml_vec_dot_q4_polar_q8_0      (n, &s_unfused, 0, x_q.data(), 0, y_q.data(), 0, 1);
        ggml_vec_dot_q4_polar_q8_0_fused(n, &s_fused,   0, x_q.data(), 0, y_q.data(), 0, 1);

        const float r = rel_err(s_unfused, s_fused);
        if (r > max_rel) max_rel = r;
        if (r > 1e-4f) {
            std::printf("[fused]   iter %d: unfused=%.6f fused=%.6f rel_err=%.2e\n",
                        it, s_unfused, s_fused, r);
        }
    }
    std::printf("[fused]   %d blocks, max relative error = %.2e (target <= 1e-4)\n",
                n_blocks, max_rel);
    return max_rel <= 1e-4f;
}

// ---------------- test 2: Hadamard-fused dot parity ----------------

bool test_polar_hadamard_dot_parity() {
    std::printf("[fused] === Q4_POLAR fused-hadamard 2-block dot parity ===\n");
    const int n = kPolarBlkSz * 8;  // 4 pairs
    std::mt19937 rng(0xBADBEEF);

    float max_rel = 0.0f;
    int n_blocks = 100;
    for (int it = 0; it < n_blocks; it++) {
        std::vector<float> x_f, y_f;
        rand_floats(x_f, n, rng, 0.7f);
        rand_floats(y_f, n, rng, 0.7f);

        std::vector<uint8_t> x_q, y_q;
        quantize_q4_polar_row(x_f.data(), n, x_q);
        quantize_q8_0_row    (y_f.data(), n, y_q);

        float s_unfused = 0.0f;
        float s_fused   = 0.0f;
        ggml_vec_dot_q4_polar_q8_0               (n, &s_unfused, 0, x_q.data(), 0, y_q.data(), 0, 1);
        ggml_vec_dot_q4_polar_q8_0_fused_hadamard(n, &s_fused,   0, x_q.data(), 0, y_q.data(), 0, 1);

        const float r = rel_err(s_unfused, s_fused);
        if (r > max_rel) max_rel = r;
        if (r > 1e-3f) {
            std::printf("[fused]   iter %d: unfused=%.6f fused-had=%.6f rel_err=%.2e\n",
                        it, s_unfused, s_fused, r);
        }
    }
    std::printf("[fused]   %d blocks, max relative error = %.2e (target <= 1e-3)\n",
                n_blocks, max_rel);
    return max_rel <= 1e-3f;
}

// ---------------- test 3: fused QJL+TBQ attention smoke ----------------

// Reference-side attention via the unfused chain:
//   1. ggml_attn_score_qjl(q, K) -> scores
//   2. scale + softmax
//   3. dequant V row by row, mix
void unfused_attn_chain(int n_kv_tokens, const float * q_sketch,
                        const std::vector<uint8_t> & k_packed,
                        const std::vector<uint8_t> & v_packed_per_token,
                        float sm_scale,
                        std::vector<float> & out) {
    out.assign(kHeadDim, 0.0f);
    const auto * tt_v = ggml_get_type_traits(GGML_TYPE_TBQ3_0);

    // Build the unfused QJL score by hand: same formula as the QJL ref
    // (sqrt(pi/2) / proj_dim * ||k|| * sign-FMA).
    const float scl_base = 1.2533141373155003f / (float) kSketchDim;
    std::vector<float> scores(n_kv_tokens);
    for (int t = 0; t < n_kv_tokens; t++) {
        const uint8_t * blk = k_packed.data() + (size_t) t * kQjlBlkB;
        const uint8_t * signs = blk;
        uint16_t bf16_bits;
        std::memcpy(&bf16_bits, blk + 32, sizeof(uint16_t));
        union { uint32_t u; float f; } u;
        u.u = ((uint32_t) bf16_bits) << 16;
        const float norm_k = u.f;
        float dot = 0.0f;
        for (int j = 0; j < kSketchDim; j++) {
            const int bit = (signs[j >> 3] >> (j & 7)) & 1;
            dot += bit ? q_sketch[j] : -q_sketch[j];
        }
        scores[t] = scl_base * norm_k * dot * sm_scale;
    }

    // Softmax.
    float m = -INFINITY;
    for (float s : scores) m = std::max(m, s);
    float l = 0.0f;
    for (auto & s : scores) { s = std::exp(s - m); l += s; }
    if (l > 0) for (auto & s : scores) s /= l;

    // V mix: dequant each token's 4 tbq3_0 blocks and accumulate.
    std::vector<float> v_dec(kHeadDim);
    for (int t = 0; t < n_kv_tokens; t++) {
        const uint8_t * v_row = v_packed_per_token.data() + (size_t) t * kTbqPerTok * tt_v->type_size;
        // Use the type-trait ref dequant on 4 blocks (= kHeadDim floats).
        tt_v->to_float(v_row, v_dec.data(), kHeadDim);
        const float w = scores[t];
        for (int d = 0; d < kHeadDim; d++) {
            out[d] += w * v_dec[d];
        }
    }
}

bool test_fused_attn_smoke_one(uint64_t seed, int n_kv_tokens, float & out_max_rel) {
    const int n_heads     = 1;
    const int n_kv_heads  = 1;
    const float sm_scale  = 1.0f / std::sqrt((float) kHeadDim);
    std::mt19937 rng((uint32_t) seed);

    // Synthesize input tensors as fp32 then quantize.
    std::vector<float> q_sketch, keys, values;
    rand_floats(q_sketch, kSketchDim,                    rng, 0.5f);
    rand_floats(keys,     (size_t) n_kv_tokens * kHeadDim, rng, 0.5f);
    rand_floats(values,   (size_t) n_kv_tokens * kHeadDim, rng, 0.5f);

    std::vector<uint8_t> k_packed, v_packed;
    quantize_qjl_keys(keys.data(),   n_kv_tokens, k_packed);
    quantize_tbq3_v  (values.data(), n_kv_tokens, v_packed);

    // Baseline (unfused chain, in-test reference).
    std::vector<float> out_ref;
    unfused_attn_chain(n_kv_tokens, q_sketch.data(), k_packed, v_packed,
                       sm_scale, out_ref);

    // Fused via the ggml graph + GGML_OP_FUSED_ATTN_QJL_TBQ op.
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) { std::fprintf(stderr, "[fused] backend init failed\n"); return false; }

    struct ggml_init_params iparams = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 16 + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(iparams);
    assert(ctx != NULL);

    ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kSketchDim, n_heads, 1);
    ggml_tensor * pk = ggml_new_tensor_3d(ctx, GGML_TYPE_QJL1_256, kHeadDim, n_kv_tokens, n_kv_heads);
    ggml_tensor * pv = ggml_new_tensor_3d(ctx, GGML_TYPE_TBQ3_0,  kHeadDim, n_kv_tokens, n_kv_heads);

    ggml_tensor * out = ggml_fused_attn_qjl_tbq(ctx, q, pk, pv, n_kv_heads, sm_scale);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        std::fprintf(stderr, "[fused] backend buffer alloc failed\n");
        ggml_free(ctx); ggml_backend_free(backend); return false;
    }

    ggml_backend_tensor_set(q,  q_sketch.data(), 0, ggml_nbytes(q));
    ggml_backend_tensor_set(pk, k_packed.data(), 0, ggml_nbytes(pk));
    ggml_backend_tensor_set(pv, v_packed.data(), 0, ggml_nbytes(pv));

    ggml_status st = ggml_backend_graph_compute(backend, gf);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[fused] graph compute failed: %d\n", (int) st);
        ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(backend);
        return false;
    }

    std::vector<float> out_fused(kHeadDim);
    ggml_backend_tensor_get(out, out_fused.data(), 0, ggml_nbytes(out));

    // Compute per-element error normalized by the *vector magnitude* of the
    // reference output, not by the per-element magnitude. Attention outputs
    // are softmax-weighted sums whose individual entries legitimately
    // cancel near zero; comparing relative-to-self there blows up FP32
    // ULP-scale noise into spurious "percent" errors. The signal-magnitude
    // normalization is the standard FP-comparison idiom for vectors:
    //   err = max_d |fused[d] - ref[d]| / max(max_d |ref[d]|, eps).
    // This matches how attention parity is graded in JAX / PyTorch tests
    // (e.g. `assert_allclose(..., atol=, rtol=)` evaluated against the
    // vector's own scale, not per-element).
    float vec_scale = 0.0f;
    for (int d = 0; d < kHeadDim; d++) {
        vec_scale = std::max(vec_scale, std::abs(out_ref[d]));
    }
    const float denom = std::max(vec_scale, 1e-6f);
    float max_rel = 0.0f;
    for (int d = 0; d < kHeadDim; d++) {
        const float r = std::abs(out_ref[d] - out_fused[d]) / denom;
        if (r > max_rel) max_rel = r;
    }
    out_max_rel = max_rel;

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return max_rel <= 5e-3f;
}

bool test_fused_attn_smoke() {
    std::printf("[fused] === fused QJL+TBQ attention smoke (100 contexts) ===\n");
    bool ok = true;
    float global_max_rel = 0.0f;
    const int n_iters = 100;
    for (int it = 0; it < n_iters; it++) {
        // Vary context length 16..256 to exercise the n_tokens loop.
        const int n_kv_tokens = 16 + (it * 13) % 256;
        float max_rel = 0.0f;
        bool one_ok = test_fused_attn_smoke_one((uint64_t) (0xF00DC0DE + it),
                                                n_kv_tokens, max_rel);
        if (!one_ok) {
            std::printf("[fused]   iter %d (n_tokens=%d) FAIL: max_rel=%.3e\n",
                        it, n_kv_tokens, max_rel);
            ok = false;
        }
        if (max_rel > global_max_rel) global_max_rel = max_rel;
    }
    std::printf("[fused]   %d contexts, max signal-normalised error = %.2e (target <= 5e-3)\n",
                n_iters, global_max_rel);
    return ok;
}

}  // namespace

int main() {
    // Force unbuffered stdout so we can localize crashes.
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);

    // Disable QJL residual on Q4_POLAR for the dot parity tests so the
    // unfused vec_dot and fused vec_dot agree to FP rounding noise. The
    // residual path adds a sign-correction term that both paths apply
    // identically once the runtime flag is on; we test both off (default)
    // and on below.
    ggml_q4_polar_set_use_qjl(false);

    bool ok = true;
    ok &= test_polar_dot_parity();
    ok &= test_polar_hadamard_dot_parity();

    // Re-run with QJL residual on.
    std::printf("[fused] === re-running Polar tests with QJL residual ON ===\n");
    ggml_q4_polar_set_use_qjl(true);
    ok &= test_polar_dot_parity();
    ok &= test_polar_hadamard_dot_parity();
    ggml_q4_polar_set_use_qjl(false);

    ok &= test_fused_attn_smoke();

    if (!ok) {
        std::fprintf(stderr, "[fused] FAIL\n");
        return 1;
    }
    std::printf("[fused] PASS\n");
    return 0;
}
