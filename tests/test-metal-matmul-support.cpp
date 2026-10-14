// Capability checks must distinguish unsupported routing from numerical parity.
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <initializer_list>

int main(int argc, char **) {
    ggml_backend_load_all();
    ggml_backend_dev_t device = ggml_backend_dev_by_name("MTL0");
    if (!device) {
        std::puts("MTL0 unavailable: capability checks skipped");
        return argc > 1 ? 1 : 0;
    }
    const ggml_type types[] = {
        GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q1_0_g32, GGML_TYPE_Q1_0_g128,
        GGML_TYPE_TBQ3_0, GGML_TYPE_TBQ4_0, GGML_TYPE_TBQ3_K, GGML_TYPE_TBQ4_K,
        GGML_TYPE_QJL1_256, GGML_TYPE_Q4_POLAR, GGML_TYPE_TBQ3_TCQ,
    };
    size_t checked = 0;
    for (ggml_type type : types) {
        for (int64_t n : {1, 4, 32}) {
            ggml_init_params params = {16 * ggml_tensor_overhead(), nullptr, true};
            ggml_context * ctx = ggml_init(params);
            GGML_ASSERT(ctx);
            ggml_tensor * a = ggml_new_tensor_2d(ctx, type, 256, 32);
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, n);
            ggml_tensor * ordinary = ggml_mul_mat(ctx, a, b);
            ggml_tensor * experts = ggml_new_tensor_3d(ctx, type, 256, 32, 4);
            ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 256, 2, n);
            ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 2, n);
            ggml_tensor * indexed = ggml_mul_mat_id(ctx, experts, input, ids);
            const bool standard = type == GGML_TYPE_F32 || type == GGML_TYPE_Q4_0;
            const bool ordinary_expected = standard || type == GGML_TYPE_Q1_0_g128;
            GGML_ASSERT(ggml_backend_dev_supports_op(device, ordinary) == ordinary_expected);
            GGML_ASSERT(ggml_backend_dev_supports_op(device, indexed) == standard);
            checked += 2;
            ggml_free(ctx);
        }
    }
    std::printf("MTL0: %zu matmul capability checks passed\n", checked);
    return 0;
}
