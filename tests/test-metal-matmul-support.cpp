/** Checks Apple Silicon Metal matmul admission; --numeric also checks quantized/f16 dispatch against exact scalar products. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <cmath>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cstring>
#include <initializer_list>


// Integer Q4_0 rows include -8 and have scale 1, so the source values are
// exactly representable. Binary-fraction RHS values are exact in f16 too.
// This oracle does not require CPU MUL_MAT to accept a quantized/f16 pair.
static bool check_numeric(ggml_backend_t backend, bool indexed, int n) {
    constexpr int k = 256, m = 16, used = 2, experts = 4;
    ggml_init_params params = { 32*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx);
    ggml_tensor * a = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, k, m, indexed ? experts : 1);
    ggml_tensor * b = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, indexed ? used : n, indexed ? n : 1);
    ggml_tensor * ids = indexed ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, n) : nullptr;
    ggml_tensor * out = indexed ? ggml_mul_mat_id(ctx, a, b, ids) : ggml_mul_mat(ctx, a, b);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    GGML_ASSERT(buffer);
    std::vector<float> av(ggml_nelements(a));
    std::vector<uint8_t> packed(ggml_nbytes(a));
    std::vector<ggml_fp16_t> bv(ggml_nelements(b));
    std::vector<int32_t> choices(indexed ? used*n : 0);
    for (size_t i = 0; i < av.size(); ++i) av[i] = float((i + (i/k)*3) % 16) - 8.0f;
    for (size_t i = 0; i < bv.size(); ++i) bv[i] = ggml_fp32_to_fp16(float(int((i + i/k) % 13) - 6) / 8.0f);
    for (size_t i = 0; i < choices.size(); ++i) choices[i] = int32_t((i + i/used) % experts);
    GGML_ASSERT(ggml_quantize_chunk(GGML_TYPE_Q4_0, av.data(), packed.data(), 0, av.size()/k, k, nullptr) == packed.size());
    ggml_backend_tensor_set(a, packed.data(), 0, packed.size());
    ggml_backend_tensor_set(b, bv.data(), 0, ggml_nbytes(b));
    if (indexed) ggml_backend_tensor_set(ids, choices.data(), 0, ggml_nbytes(ids));
    bool ok = ggml_backend_supports_op(backend, out);
    if (ok) ok = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    float max_error = 0.0f;
    if (ok) {
        std::vector<float> actual(ggml_nelements(out));
        ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
        for (int column = 0; column < n*(indexed ? used : 1); ++column) {
            int expert = indexed ? choices[column] : 0;
            for (int row = 0; row < m; ++row) {
                float expected = 0.0f;
                for (int d = 0; d < k; ++d) {
                    expected += av[(expert*m + row)*k + d] * ggml_fp16_to_fp32(bv[column*k + d]);
                }
                float error = std::fabs(actual[column*m + row] - expected);
                if (!std::isfinite(error) || error > 0.001f) ok = false;
                if (error > max_error) max_error = error;
            }
        }
    }
    std::printf("%s Q4_0/F16 n=%d scalar max error=%g: %s\n",
                indexed ? "MUL_MAT_ID" : "MUL_MAT", n, max_error, ok ? "PASS" : "FAIL");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return ok;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (!dev || std::strncmp(ggml_backend_dev_description(dev), "Apple M", 7) != 0) {
        std::puts("SKIP: Apple Silicon Metal device required");
        return 77;
    }

    if (argc == 2 && std::strcmp(argv[1], "--numeric") == 0) {
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        GGML_ASSERT(backend);
        bool ok = true;
        for (int n : {9, 32}) ok = check_numeric(backend, false, n) && ok;
        for (int n : {32, 33}) ok = check_numeric(backend, true, n) && ok;
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    }
    int checked = 0;
    int failed = 0;
    auto supports = [&](ggml_type a_type, ggml_type b_type, bool indexed, int n, int k, bool transpose) {
        ggml_init_params params = { 16*ggml_tensor_overhead(), nullptr, true };
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx);
        ggml_tensor * a = ggml_new_tensor_3d(ctx, a_type, k, 16, indexed ? 4 : 1);
        const int columns = indexed ? 2 : n;
        ggml_tensor * b = ggml_new_tensor_3d(ctx, b_type, transpose ? columns : k,
                                            transpose ? k : columns, indexed ? n : 1);
        if (transpose) {
            b = ggml_transpose(ctx, b);
        }
        ggml_tensor * op;
        if (indexed) {
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n);
            op = ggml_mul_mat_id(ctx, a, b, ids);
        } else {
            op = ggml_mul_mat(ctx, a, b);
        }
        const bool result = ggml_backend_dev_supports_op(dev, op);
        ggml_free(ctx);
        return result;
    };
    auto check = [&](ggml_type a, ggml_type b, bool indexed, int n, bool expected, int k = 256, bool transpose = false) {
        const bool actual = supports(a, b, indexed, n, k, transpose);
        ++checked;
        if (actual != expected) {
            std::fprintf(stderr, "%s(%s,%s,n=%d,k=%d,transpose=%d): expected support=%d, got=%d\n",
                         indexed ? "MUL_MAT_ID" : "MUL_MAT", ggml_type_name(a), ggml_type_name(b), n, k, transpose, expected, actual);
            ++failed;
        }
    };

    for (ggml_type a : {GGML_TYPE_Q1_0_g32, GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ4_0,
                        GGML_TYPE_QJL1_256, GGML_TYPE_Q4_POLAR, GGML_TYPE_TBQ3_TCQ,
                        GGML_TYPE_TBQ3_K, GGML_TYPE_TBQ4_K}) {
        for (int n : {1, 32}) {
            check(a, GGML_TYPE_F32, false, n, false);
            check(a, GGML_TYPE_F32, true,  n, false);
        }
    }
    for (int n : {1, 32}) {
        check(GGML_TYPE_Q1_0_g128, GGML_TYPE_F32, false, n, true);
        check(GGML_TYPE_Q1_0_g128, GGML_TYPE_F32, true,  n, false);
        check(GGML_TYPE_Q1_0_g128, GGML_TYPE_F16, false, n, false);
        check(GGML_TYPE_Q4_0, GGML_TYPE_F32, false, n, true);
        check(GGML_TYPE_Q4_0, GGML_TYPE_F32, true,  n, true);
        check(GGML_TYPE_Q4_0, GGML_TYPE_F16, false, n, n > 8);
        check(GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  n, n >= 32);
    }
    for (int n : {1, 8, 9, 31, 32}) {
        check(GGML_TYPE_F32, GGML_TYPE_F16, false, n, n > 8);
        check(GGML_TYPE_F16, GGML_TYPE_F16, false, n, true);
        check(GGML_TYPE_F32, GGML_TYPE_F16, true,  n, n >= 32);
        check(GGML_TYPE_F16, GGML_TYPE_F16, true,  n, n >= 32);
    }
    for (ggml_type a : {GGML_TYPE_Q1_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
                        GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0,
                        GGML_TYPE_MXFP4, GGML_TYPE_NVFP4, GGML_TYPE_Q2_K,
                        GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K,
                        GGML_TYPE_Q6_K, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS,
                        GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S, GGML_TYPE_IQ2_S,
                        GGML_TYPE_IQ1_S, GGML_TYPE_IQ1_M, GGML_TYPE_IQ4_NL,
                        GGML_TYPE_IQ4_XS}) {
        for (int n : {8, 9, 31, 32}) {
            check(a, GGML_TYPE_F16, false, n, n > 8);
            check(a, GGML_TYPE_F16, true, n, n >= 32);
            check(a, GGML_TYPE_F16, false, n, false, 256, true);
            check(a, GGML_TYPE_F16, true, n, false, 256, true);
        }
    }
    for (int k : {63, 64}) {
        for (int n : {8, 9, 31, 32}) {
            for (bool transpose : {false, true}) {
                const bool ordinary_mm = k >= 64 && n > 8 && !transpose;
                const bool indexed_mm = k >= 64 && n >= 32 && !transpose;
                check(GGML_TYPE_F32, GGML_TYPE_F16, false, n, ordinary_mm, k, transpose);
                check(GGML_TYPE_F16, GGML_TYPE_F16, false, n, true, k, transpose);
                check(GGML_TYPE_F32, GGML_TYPE_F16, true, n, indexed_mm, k, transpose);
                check(GGML_TYPE_F16, GGML_TYPE_F16, true, n, indexed_mm, k, transpose);
                check(GGML_TYPE_F32, GGML_TYPE_F32, true, n, !transpose, k, transpose);
            }
        }
    }
    // BF16 availability varies across Apple Silicon generations and OS toolchains.
    if (supports(GGML_TYPE_BF16, GGML_TYPE_F32, false, 1, 256, false)) {
        for (int k : {63, 64}) {
            for (int n : {8, 9, 31, 32}) {
                for (bool transpose : {false, true}) {
                    check(GGML_TYPE_BF16, GGML_TYPE_BF16, false, n,
                          !(k >= 64 && n > 8 && !transpose), k, transpose);
                    check(GGML_TYPE_BF16, GGML_TYPE_BF16, true, n, false, k, transpose);
                }
            }
        }
        for (int n : {1, 8, 9, 31, 32}) {
            check(GGML_TYPE_BF16, GGML_TYPE_BF16, false, n, n <= 8);
            check(GGML_TYPE_BF16, GGML_TYPE_BF16, true,  n, false);
            check(GGML_TYPE_BF16, GGML_TYPE_F16, false, n, false);
        }
    }
    std::printf("%d capability checks, %d failures; no GPU numerical execution\n", checked, failed);
    return failed ? 1 : 0;
}
