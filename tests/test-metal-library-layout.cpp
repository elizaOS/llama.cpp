#include "ggml-metal-device.h"

#include <cstdio>

// Both libraries are compiled on the same device. A hardware capability flag
// cannot distinguish their contracts, and a global tensor disable cannot pass.
int main() {
    ggml_metal_device_t dev = ggml_metal_device_init(0);
    if (!dev) {
        std::fprintf(stderr, "No Metal device available\n");
        return 77;
    }

    const char * legacy_source = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void legacy_layout(device uint * out [[buffer(0)]]) { out[0] = 0; }
    )";
    const char * tensor_source = R"(
        #include <metal_stdlib>
        using namespace metal;
        kernel void ggml_metal_tensor_layout_v1(device uint * out [[buffer(0)]]) { out[0] = 1; }
    )";

    ggml_metal_library_t legacy = ggml_metal_library_init_from_source(dev, legacy_source, true);
    ggml_metal_library_t tensor = ggml_metal_library_init_from_source(dev, tensor_source, true);
    const bool passed = legacy && tensor &&
        !ggml_metal_library_has_tensor_layout(legacy) &&
         ggml_metal_library_has_tensor_layout(tensor) &&
        !ggml_metal_library_has_tensor_layout(legacy);

    ggml_metal_library_free(tensor);
    ggml_metal_library_free(legacy);
    ggml_metal_device_free(dev);
    std::fprintf(stderr, "Per-library Metal layout: %s\n", passed ? "PASS" : "FAIL");
    return passed ? 0 : 1;
}
