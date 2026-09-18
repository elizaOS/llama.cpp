/** Exercises real Kokoro loading through the CPU backend registry in static and dynamic builds. */
#include "kokoro.h"
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <string>

int main() {
    const char * path = "kokoro-backend-test.gguf";
    ggml_context * tensors = ggml_init({4096, nullptr, false});
    if (!tensors) return 1;
    ggml_tensor * tensor = ggml_new_tensor_1d(tensors, GGML_TYPE_F32, 1);
    ggml_set_name(tensor, "test.unused");
    *static_cast<float *>(tensor->data) = 1.0f;
    gguf_context * file = gguf_init_empty();
    gguf_set_val_str(file, "general.architecture", "kokoro");
    gguf_add_tensor(file, tensor);
    const bool written = gguf_write_to_file(file, path, false);
    gguf_free(file);
    ggml_free(tensors);
    if (!written) return 1;

    std::string error;
    auto model = eliza_kokoro::kokoro_load_model(path, error);
    std::remove(path);
    // The intentionally incomplete model reaches tensor validation only after
    // backend discovery, initialization, thread setup, and real tensor upload.
    if (model || error.find("BERT token embedding") == std::string::npos) {
        std::fprintf(stderr, "Expected missing model weights after CPU setup, got: %s\n", error.c_str());
        return 1;
    }
    return 0;
}
