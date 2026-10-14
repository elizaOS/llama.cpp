#include "ggml.h"
#include "gguf.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace {
template <typename T> void append(std::vector<uint8_t> & out, T value) {
    const auto * p = reinterpret_cast<const uint8_t *>(&value);
    out.insert(out.end(), p, p + sizeof(value));
}
void string(std::vector<uint8_t> & out, const char * value) {
    append<uint64_t>(out, strlen(value));
    out.insert(out.end(), value, value + strlen(value));
}
void check(uint32_t raw_type, ggml_type expected, uint32_t ftype = 0, const char * layout = nullptr) {
    std::vector<uint8_t> bytes;
    bytes.insert(bytes.end(), {'G', 'G', 'U', 'F'});
    append<uint32_t>(bytes, 3);
    append<uint64_t>(bytes, 1);
    append<uint64_t>(bytes, (ftype ? 1 : 0) + (layout ? 1 : 0));
    if (ftype) {
        string(bytes, "general.file_type");
        append<uint32_t>(bytes, GGUF_TYPE_UINT32);
        append<uint32_t>(bytes, ftype);
    }
    if (layout) {
        string(bytes, "eliza.quantization_type_ids");
        append<uint32_t>(bytes, GGUF_TYPE_STRING);
        string(bytes, layout);
    }
    string(bytes, "weight");
    append<uint32_t>(bytes, 1);
    append<uint64_t>(bytes, 256);
    append<uint32_t>(bytes, raw_type);
    append<uint64_t>(bytes, 0);
    bytes.resize(((bytes.size() + 31) / 32) * 32 + 1024);
    gguf_init_params params = { true, nullptr };
    gguf_context * ctx = gguf_init_from_buffer(bytes.data(), bytes.size(), params);
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(gguf_get_tensor_type(ctx, 0) == expected);
    gguf_free(ctx);
}
}
int main() {
    static_assert(GGML_TYPE_Q2_0 == 42, "upstream GGUF type ID must remain stable");
    check(42, GGML_TYPE_Q2_0);
    check(42, GGML_TYPE_Q2_0, 100, "private-256");
    check(42, GGML_TYPE_Q1_0_g32, 100);
    check(42, GGML_TYPE_Q1_0_g32, 101);
    check(42, GGML_TYPE_Q1_0_g32, 0, "legacy-42");
    for (uint32_t i = 43; i <= 50; ++i) {
        check(i, static_cast<ggml_type>(i + 214));
    }
    for (uint32_t i = 256; i <= 264; ++i) {
        check(i, static_cast<ggml_type>(i));
    }
    ggml_init_params ip = { 1024 * 1024, nullptr, true };
    ggml_context * tensors = ggml_init(ip);
    GGML_ASSERT(tensors);
    ggml_tensor * tensor = ggml_new_tensor_1d(tensors, GGML_TYPE_Q1_0_g32, 256);
    ggml_set_name(tensor, "weight");
    gguf_context * out = gguf_init_empty();
    gguf_add_tensor(out, tensor);
    const int64_t key = gguf_find_key(out, "eliza.quantization_type_ids");
    GGML_ASSERT(key >= 0);
    GGML_ASSERT(strcmp(gguf_get_val_str(out, key), "private-256") == 0);
    std::vector<uint8_t> bytes(gguf_get_meta_size(out) + 1024);
    gguf_get_meta_data(out, bytes.data());
    gguf_init_params rp = { true, nullptr };
    gguf_context * roundtrip = gguf_init_from_buffer(bytes.data(), bytes.size(), rp);
    GGML_ASSERT(roundtrip && gguf_get_tensor_type(roundtrip, 0) == GGML_TYPE_Q1_0_g32);
    gguf_free(roundtrip);
    gguf_free(out);
    ggml_free(tensors);
}
