/** Exercises real backend readback into subpage heap allocations with guarded partial destinations. */
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdint>
#ifdef __APPLE__
#include <unistd.h>
#endif
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
#ifdef __APPLE__
    const size_t page_size = (size_t) sysconf(_SC_PAGESIZE);
#else
    const size_t page_size = 4096;
#endif
    ggml_backend_load_all();
    size_t tested = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t device = ggml_backend_dev_get(i);
        if (argc > 1 && std::strcmp(argv[1], ggml_backend_dev_name(device)) != 0) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(device, nullptr);
        GGML_ASSERT(backend != nullptr);
        ggml_init_params params = {4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true};
        ggml_context * ctx = ggml_init(params);
        GGML_ASSERT(ctx != nullptr);
        ggml_tensor * tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, page_size / sizeof(float) + 32);
        ggml_tensor * scaled = ggml_scale(ctx, tensor, 2.0f);
        ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, scaled);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        GGML_ASSERT(buffer != nullptr);
        std::vector<float> source(page_size / sizeof(float) + 32);
        for (size_t j = 0; j < source.size(); ++j) {
            source[j] = float(j) / 8.0f;
        }
        ggml_backend_tensor_set(tensor, source.data(), 0, source.size() * sizeof(float));
        for (bool asynchronous : {false, true}) {
            for (size_t count : {size_t(1), size_t(384)}) {
                for (size_t offset : {size_t(0), size_t(17)}) {
                    std::vector<float> output(count + 2, -123.0f);
                    if (asynchronous) {
                        ggml_backend_tensor_get_async(backend, tensor, output.data() + 1,
                                                      offset * sizeof(float), count * sizeof(float));
                        ggml_backend_synchronize(backend);
                    } else {
                        ggml_backend_tensor_get(tensor, output.data() + 1,
                                                offset * sizeof(float), count * sizeof(float));
                    }
                    GGML_ASSERT(output.front() == -123.0f && output.back() == -123.0f);
                    GGML_ASSERT(std::equal(output.begin() + 1, output.end() - 1, source.begin() + offset));
                }
            }
        }
        float untouched = -321.0f;
        ggml_backend_tensor_get(tensor, &untouched, 0, 0);
        ggml_backend_tensor_get_async(backend, tensor, &untouched, 0, 0);
        ggml_backend_synchronize(backend);
        GGML_ASSERT(untouched == -321.0f);

        if (ggml_backend_supports_op(backend, scaled)) {
            // Queue real GPU computation immediately before the readback; no intervening wait.
            GGML_ASSERT(ggml_backend_graph_compute_async(backend, graph) == GGML_STATUS_SUCCESS);
            std::vector<float> computed(386, -123.0f);
            ggml_backend_tensor_get_async(backend, scaled, computed.data() + 1, 17 * sizeof(float), 384 * sizeof(float));
            ggml_backend_synchronize(backend);
            GGML_ASSERT(computed.front() == -123.0f && computed.back() == -123.0f);
            for (size_t j = 0; j < 384; ++j) {
                GGML_ASSERT(computed[j + 1] == 2.0f * source[j + 17]);
            }
        } else {
            std::printf("%s: queued SCALE unsupported; readback checks still run\n", ggml_backend_name(backend));
        }
        // Exercise the aligned no-copy path as well as the subpage staging path.
        std::vector<uint8_t> aligned_storage(page_size * 3, 0xA5);
        uintptr_t address = (uintptr_t) aligned_storage.data() + page_size;
        address -= address % page_size;
        uint8_t * aligned = (uint8_t *) address;
        for (bool asynchronous : {false, true}) {
            std::fill(aligned_storage.begin(), aligned_storage.end(), 0xA5);
            if (asynchronous) {
                ggml_backend_tensor_get_async(backend, tensor, aligned, 0, page_size);
                ggml_backend_synchronize(backend);
            } else {
                ggml_backend_tensor_get(tensor, aligned, 0, page_size);
            }
            GGML_ASSERT(aligned[-1] == 0xA5 && aligned[page_size] == 0xA5);
            GGML_ASSERT(std::memcmp(aligned, source.data(), page_size) == 0);
        }
        std::printf("%s: exact guarded synchronous/asynchronous readback passed\n", ggml_backend_name(backend));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(backend);
        ++tested;
    }
    GGML_ASSERT(tested > 0);
    return 0;
}
