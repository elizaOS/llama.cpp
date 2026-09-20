/** Exercises actual Apple Silicon Metal matmul admission using real tensor graphs, without dispatching GPU work. */
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <initializer_list>

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("MTL0");
    if (!dev || std::strncmp(ggml_backend_dev_description(dev), "Apple M", 7) != 0) {
        std::puts("SKIP: Apple Silicon Metal device required");
        return 77;
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
        check(GGML_TYPE_Q4_0, GGML_TYPE_F16, false, n, false);
        check(GGML_TYPE_Q4_0, GGML_TYPE_F16, true,  n, false);
    }
    for (int n : {1, 8, 9, 31, 32}) {
        check(GGML_TYPE_F32, GGML_TYPE_F16, false, n, n > 8);
        check(GGML_TYPE_F16, GGML_TYPE_F16, false, n, true);
        check(GGML_TYPE_F32, GGML_TYPE_F16, true,  n, n >= 32);
        check(GGML_TYPE_F16, GGML_TYPE_F16, true,  n, n >= 32);
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
