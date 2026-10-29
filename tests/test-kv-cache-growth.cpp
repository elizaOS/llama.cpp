#include "llama-model.h"
#include "llama-kv-cache.h"
#include "llama-graph.h"
#include "ggml-backend.h"

#include <vector>

struct cache_test_model : llama_model {
    cache_test_model() : llama_model(llama_model_default_params()) {
        hparams = {};
        hparams.n_layer_all = 1;
        hparams.n_embd = 32;
        hparams.n_embd_head_k_full = 32;
        hparams.n_embd_head_v_full = 32;
        hparams.n_head_arr.fill(1);
        hparams.n_head_kv_arr.fill(1);
    }
    void load_stats(llama_model_loader &) override {}
    void load_hparams(llama_model_loader &) override {}
    void load_vocab(llama_model_loader &) override {}
    bool load_tensors(llama_model_loader &) override { return false; }
    void load_arch_hparams(llama_model_loader &) override {}
    void load_arch_tensors(llama_model_loader &) override {}
    std::unique_ptr<llm_graph_context> build_arch_graph(const llm_graph_params &) const override { return nullptr; }
};

static void check(bool transposed) {
    cache_test_model model;
    llama_kv_cache cache(model, false, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        transposed, false, false, 1024, 2, 0, LLAMA_SWA_TYPE_NONE,
        nullptr, nullptr, nullptr, nullptr, "", 1024);
    GGML_ASSERT(cache.get_size() == 256);
    std::vector<float> expected(2 * 32 * 256);
    for (size_t i = 0; i < expected.size(); ++i) expected[i] = float(i + 1);
    ggml_tensor * key = cache.get_k_storage(0);
    ggml_backend_tensor_set(key, expected.data(), 0, expected.size() * sizeof(float));
    {
        auto * ctx = ggml_init({4096, nullptr, true});
        llama_kv_cache::slot_info slot = {};
        slot.s1 = 1;
        auto * value = cache.get_v(ctx, 0, 256, slot);
        ggml_backend_tensor_set(value->view_src, expected.data(), 0, expected.size() * sizeof(float));
        ggml_free(ctx);
    }
    GGML_ASSERT(cache.try_resize());
    GGML_ASSERT(cache.get_size() == 512);
    GGML_ASSERT(cache.check_and_clear_resized());
    GGML_ASSERT(!cache.check_and_clear_resized());
    std::vector<float> actual(2 * 32 * 512);
    ggml_backend_tensor_get(cache.get_k_storage(0), actual.data(), 0, actual.size() * sizeof(float));
    for (size_t stream = 0; stream < 2; ++stream)
        for (size_t i = 0; i < 32 * 256; ++i)
            GGML_ASSERT(actual[stream * 32 * 512 + i] == expected[stream * 32 * 256 + i]);
    {
        auto * ctx = ggml_init({4096, nullptr, true});
        llama_kv_cache::slot_info slot = {};
        slot.s1 = 1;
        auto * value = cache.get_v(ctx, 0, 512, slot);
        ggml_backend_tensor_get(value->view_src, actual.data(), 0, actual.size() * sizeof(float));
        for (size_t stream = 0; stream < 2; ++stream)
            for (size_t channel = 0; channel < 32; ++channel)
                for (size_t token = 0; token < 256; ++token) {
                    const size_t before = transposed ? channel * 256 + token : token * 32 + channel;
                    const size_t after  = transposed ? channel * 512 + token : token * 32 + channel;
                    GGML_ASSERT(actual[stream * 32 * 512 + after] == expected[stream * 32 * 256 + before]);
                }
        ggml_free(ctx);
    }
    // Attaching a shared cache must finish growth before borrowing its storage.
    llama_kv_cache shared(model, false, model.hparams, GGML_TYPE_F32, GGML_TYPE_F32,
        transposed, false, false, 1024, 2, 0, LLAMA_SWA_TYPE_NONE,
        &cache, nullptr, nullptr, [](int32_t layer) { return layer; }, "", 1024);
    GGML_ASSERT(cache.get_size() == 1024 && shared.get_size() == 1024);
    GGML_ASSERT(shared.get_k_storage(0) == cache.get_k_storage(0));
    GGML_ASSERT(!cache.try_resize() && !shared.try_resize());
}

int main() {
    ggml_backend_load_all();
    check(false);
    check(true);
}
