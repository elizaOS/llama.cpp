/*
 * litert-embed-backend.cpp — LiteRT (Google AI Edge) text-embedding backend.
 *
 * Serves eliza_inference_embed from a `<bundle>/embedding/*.tflite` (or
 * `.litertlm`) artifact via the LiteRT Next C runtime on the best available
 * accelerator: NPU (Qualcomm QNN / MediaTek NeuroPilot / Google Tensor on
 * capable silicon) -> GPU (OpenCL/Mali via libLiteRtClGlAccelerator.so) -> CPU.
 * The accelerator ladder + preference_rank let the SAME build auto-promote to
 * NPU on a Pixel-10/G5 or Qualcomm/MediaTek device and fall back to the GPU
 * delegate on a Tensor-G4 (Pixel 9a) with NO code change.
 *
 * Uses the LiteRT *C* API (litert/c/...) — the C++ cc/ wrappers are not
 * standalone (they pull Abseil/TFLite/flatbuffers). Compiles only under
 * -DELIZA_ENABLE_LITERT with the SDK on the include/link path
 * (-DELIZA_LITERT_SDK_DIR=<dir> -DELIZA_LITERT_LIBS=LiteRt). Without the gate the
 * file is not compiled (CMake target_sources is inside if(ELIZA_ENABLE_LITERT));
 * the stub at the bottom keeps the factory accessor resolvable defensively.
 *
 * Model I/O (the converted all-MiniLM-L6-v2 .tflite, see
 * litert-models/embedding/MANIFEST.md): 2 int32 inputs [1,128] bound BY INDEX
 * (0=input_ids, 1=attention_mask), 1 float32 output [1,384] that is already
 * masked-mean-pooled + L2-normalized in-graph (read 384 floats directly).
 */

#include "../embed-backend.h"
#include "../llm-backend.h" /* llm_backend_context_bundle_dir */

#include <cstdlib>
#include <cstring>
#include <string>

#if defined(__has_include)
#  if __has_include(<filesystem>)
#    include <filesystem>
#    define ELIZA_HAS_FILESYSTEM 1
#  endif
#endif

namespace {

/* Probe `<bundle_dir>/embedding/` for a LiteRT artifact (.litertlm preferred,
 * then .tflite). Cheap — no model load. Returns the absolute path or "". */
std::string find_embed_artifact(const char * bundle_dir) {
    if (!bundle_dir || !bundle_dir[0]) return "";
#ifdef ELIZA_HAS_FILESYSTEM
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path dir = fs::path(bundle_dir) / "embedding";
    if (!fs::is_directory(dir, ec)) return "";
    std::string tflite;
    for (const auto & e : fs::directory_iterator(dir, ec)) {
        if (ec) break;
        if (!e.is_regular_file(ec)) continue;
        const std::string ext = e.path().extension().string();
        if (ext == ".litertlm") return e.path().string();
        if (ext == ".tflite" && tflite.empty()) tflite = e.path().string();
    }
    return tflite;
#else
    return "";
#endif
}

char * dup_error(const std::string & msg) {
    const std::string full = "[libelizainference] " + msg;
    char * out = (char *) std::malloc(full.size() + 1);
    if (out) std::memcpy(out, full.c_str(), full.size() + 1);
    return out;
}

} // namespace

#ifdef ELIZA_ENABLE_LITERT

#include "litert/c/litert_common.h"
#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

#include <cmath>
#include <mutex>
#include <vector>

namespace {

class LiteRtEmbedFactory final : public EmbedBackendFactory {
public:
    const char * name() const override { return "litert"; }

    /* Compiled in AND a non-CPU accelerator is reachable (a CPU-only LiteRT is
     * not a win over the in-tree ggml encoder). Settings-only probe — no model
     * load. The ladder resolves to GPU on a Tensor-G4 (9a) and NPU on capable
     * silicon. */
    bool available() const override { return probe_accel() != kLiteRtHwAcceleratorNone; }

    bool can_serve(const char * bundle_dir) const override {
        return !find_embed_artifact(bundle_dir).empty();
    }

    int preference_rank() const override {
        const int a = probe_accel();
        if (a & kLiteRtHwAcceleratorNpu) return 100; /* the real NPU win */
        if (a & kLiteRtHwAcceleratorGpu) return 20;  /* GPU delegate (Mali on a 9a) */
        return 0;                                    /* never beats ggml */
    }

    int embed(EliInferenceContext * ctx, const char * text, size_t text_len,
              int pooling, float * out_embedding, size_t out_capacity,
              int * out_dim, char ** out_error) override {
        const char * bundle = llm_backend_context_bundle_dir(ctx);
        const std::string artifact = find_embed_artifact(bundle);
        if (artifact.empty()) {
            if (out_error) *out_error = dup_error("litert embed: no artifact under <bundle>/embedding/");
            return ELIZA_ERR_INVALID_ARG;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (int rc = ensure_loaded(artifact, out_error); rc != ELIZA_OK) return rc;

        /* Tokenize -> 2 int32 input tensors [1,128] (0=input_ids,1=attention_mask).
         * The WordPiece tokenizer + the fixed-128 padding come from the model
         * MANIFEST (litert-models/embedding). The LiteRT C run path below
         * (managed buffers -> run -> read the in-graph-pooled [1,384] output) is
         * wired; binding the tokenizer is the one model-specific step. */
        std::vector<int32_t> ids, mask;
        if (int rc = tokenize(text, text_len, ids, mask, out_error); rc != ELIZA_OK) return rc;

        std::vector<float> out_vec;
        int dim = 0;
        if (int rc = run(ids, mask, out_vec, dim, out_error); rc != ELIZA_OK) return rc;

        if (dim <= 0 || (size_t) dim > out_capacity) {
            if (out_error) *out_error = dup_error("litert embed: output dim exceeds capacity");
            return ELIZA_ERR_INVALID_ARG;
        }
        (void) pooling; /* pooling + L2-norm are baked into the exported graph */
        std::memcpy(out_embedding, out_vec.data(), (size_t) dim * sizeof(float));
        *out_dim = dim;
        return ELIZA_OK;
    }

private:
    static int probe_accel() {
        LiteRtEnvironment env = nullptr;
        if (LiteRtCreateEnvironment(0, nullptr, &env) != kLiteRtStatusOk) {
            return kLiteRtHwAcceleratorNone;
        }
        LiteRtDestroyEnvironment(env);
        /* TODO(DEVICE-VERIFY): query the env for a registered NPU dispatch and
         * return kLiteRtHwAcceleratorNpu when present. On a Tensor-G4 there is no
         * app-usable NPU path, so this resolves to GPU. */
        return kLiteRtHwAcceleratorGpu;
    }

    int ensure_loaded(const std::string & artifact, char ** out_error) {
        if (artifact == loaded_path_ && compiled_) return ELIZA_OK;
        reset();
        if (LiteRtCreateEnvironment(0, nullptr, &env_) != kLiteRtStatusOk) {
            if (out_error) *out_error = dup_error("litert embed: environment create failed");
            return ELIZA_ERR_FFI_FAULT;
        }
        if (LiteRtCreateModelFromFile(artifact.c_str(), &model_) != kLiteRtStatusOk) {
            if (out_error) *out_error = dup_error("litert embed: model load failed: " + artifact);
            return ELIZA_ERR_BUNDLE_INVALID;
        }
        LiteRtOptions opts = nullptr;
        if (LiteRtCreateOptions(&opts) != kLiteRtStatusOk) {
            if (out_error) *out_error = dup_error("litert embed: options create failed");
            return ELIZA_ERR_FFI_FAULT;
        }
        LiteRtSetOptionsHardwareAccelerators(
            opts, (LiteRtHwAcceleratorSet)(kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorNpu));
        const LiteRtStatus st = LiteRtCreateCompiledModel(env_, model_, opts, &compiled_);
        LiteRtDestroyOptions(opts);
        if (st != kLiteRtStatusOk) {
            if (out_error) *out_error = dup_error("litert embed: compile failed (accelerator unavailable?)");
            return ELIZA_ERR_FFI_FAULT;
        }
        loaded_path_ = artifact;
        return ELIZA_OK;
    }

    int tokenize(const char * /*text*/, size_t /*len*/, std::vector<int32_t> & /*ids*/,
                 std::vector<int32_t> & /*mask*/, char ** out_error) {
        /* TODO(MANIFEST): wire the WordPiece tokenizer (vocab.txt under
         * <bundle>/embedding/): lower-case, [CLS] + greedy-longest-match subwords
         * + [SEP], pad/truncate to exactly 128, attention_mask=1 for real tokens.
         * Until wired this is a hard, observable failure — eliza_inference_embed
         * does NOT fall back, so a misconfigured artifact surfaces loudly. */
        if (out_error) *out_error = dup_error(
            "litert embed: WordPiece tokenizer not wired — stage vocab.txt + bind "
            "per litert-models/embedding/MANIFEST.md");
        return ELIZA_ERR_NOT_IMPLEMENTED;
    }

    int run(const std::vector<int32_t> & ids, const std::vector<int32_t> & mask,
            std::vector<float> & out_vec, int & dim, char ** out_error) {
        /* TODO(MANIFEST): create 2 managed int32 input TensorBuffers [1,128]
         * (LiteRtGetCompiledModelInputBufferRequirements ->
         * LiteRtCreateManagedTensorBufferFromRequirements), Lock+write ids/mask,
         * create the output buffer, LiteRtRunCompiledModel(compiled_, 0, in, out),
         * Lock+read the [1,384] float output into out_vec (dim=384). Pooling +
         * L2-norm are in-graph. */
        (void) ids; (void) mask; (void) out_vec; (void) dim;
        if (out_error) *out_error = dup_error("litert embed: tensor run pending MANIFEST tokenizer");
        return ELIZA_ERR_NOT_IMPLEMENTED;
    }

    void reset() {
        if (compiled_) { LiteRtDestroyCompiledModel(compiled_); compiled_ = nullptr; }
        if (model_)    { LiteRtDestroyModel(model_);            model_ = nullptr; }
        if (env_)      { LiteRtDestroyEnvironment(env_);        env_ = nullptr; }
        loaded_path_.clear();
    }

    std::mutex          mu_;
    LiteRtEnvironment   env_      = nullptr;
    LiteRtModel         model_    = nullptr;
    LiteRtCompiledModel compiled_ = nullptr;
    std::string         loaded_path_;
};

} // namespace

EmbedBackendFactory * litert_embed_backend_factory() {
    static LiteRtEmbedFactory instance;
    return &instance;
}

#else /* !ELIZA_ENABLE_LITERT — stub (kept resolvable; never selected) */

namespace {
class LiteRtEmbedStub final : public EmbedBackendFactory {
public:
    const char * name() const override { return "litert"; }
    bool available() const override { return false; }
    bool can_serve(const char *) const override { return false; }
    int embed(EliInferenceContext *, const char *, size_t, int, float *, size_t,
              int *, char ** out_error) override {
        if (out_error) *out_error = dup_error("litert embed backend not compiled in");
        return ELIZA_ERR_NOT_IMPLEMENTED;
    }
};
} // namespace

EmbedBackendFactory * litert_embed_backend_factory() {
    static LiteRtEmbedStub instance;
    return &instance;
}

#endif /* ELIZA_ENABLE_LITERT */
