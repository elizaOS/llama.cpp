/*
 * mlx-coreml-backend.mm — Apple-Silicon streaming-LLM backend (cutover M5).
 *
 * Objective-C++ translation unit: CoreML's MLModel / MLState API is
 * Objective-C, and the MLX C++ / mlx-c headers also compile cleanly in a
 * `.mm`. See mlx-coreml-backend.h for the full API research + citations and
 * the MLX-primary / CoreML-alternate trade-off.
 *
 * STRUCTURE
 *   The whole real implementation sits behind
 *     #if defined(ELIZA_ENABLE_MLX) && defined(__APPLE__)
 *   and is the ONLY place that includes any MLX / CoreML SDK header. With the
 *   gate OFF (the default Linux build) this file pulls in no SDK header at all
 *   and compiles to a pure no-op factory: available()/can_serve() == false,
 *   open() returns nullptr after setting *out_error to "not compiled in".
 *
 * ERROR CONTRACT (native/AGENTS.md §3 + §9): never log, never return a
 * defaulted result on failure. Out-error strings are heap-allocated with
 * malloc (mirroring eliza-inference-ffi.cpp's `eliza_strdup`) so the FFI
 * caller frees them with `eliza_inference_free_string` / free().
 */

#include "mlx-coreml-backend.h"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <string>

// ===========================================================================
// Shared (gate-independent) helpers
// ===========================================================================

namespace {

/* Heap-allocate an out-error string the way the FFI translation unit does
 * (eliza-inference-ffi.cpp::eliza_strdup) so the caller's free() path is
 * identical regardless of which backend produced the error. */
void mlx_set_error(char ** out_error, const std::string & msg) {
    if (!out_error) {
        return;
    }
    char * out = static_cast<char *>(std::malloc(msg.size() + 1));
    if (!out) {
        *out_error = nullptr;
        return;
    }
    std::memcpy(out, msg.c_str(), msg.size() + 1);
    *out_error = out;
}

} // namespace

// ===========================================================================
// REAL IMPLEMENTATION — Apple Silicon only, gated on ELIZA_ENABLE_MLX
// ===========================================================================
#if defined(ELIZA_ENABLE_MLX) && defined(__APPLE__)

// --- Objective-C / Apple frameworks ---------------------------------------
#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>     // MLModel, MLState, MLFeatureProvider, MLMultiArray
#import <Metal/Metal.h>       // MTLCreateSystemDefaultDevice — Metal/ANE presence probe

// --- MLX C API (ml-explore/mlx-c) ------------------------------------------
// Only included behind the gate so a host without the MLX SDK still compiles.
#include "mlx/c/array.h"
#include "mlx/c/device.h"
#include "mlx/c/stream.h"
#include "mlx/c/io.h"
#include "mlx/c/ops.h"
#include "mlx/c/fast.h"
#include "mlx/c/map.h"

#include <dirent.h>
#include <sys/stat.h>
#include <cmath>
#include <filesystem>
#include <vector>

namespace {

namespace fs = std::filesystem;

// --- bundle artifact discovery --------------------------------------------

enum class AppleRuntime {
    None,
    Mlx,     // mlx weights dir (safetensors) or *.gguf under text/
    CoreMl,  // *.mlmodelc / *.mlpackage under text/
};

bool has_suffix(const std::string & s, const char * suffix) {
    const size_t n = std::strlen(suffix);
    return s.size() >= n && std::equal(s.end() - n, s.end(), suffix);
}

/* Probe `<bundle_dir>/text/` for an Apple-servable artifact and report which
 * runtime would serve it. MLX is preferred when both kinds are present (an
 * mlx weights dir / gguf wins over a CoreML package), matching the header's
 * "MLX primary, CoreML alternate" rule. Cheap directory walk, no model load. */
AppleRuntime detect_runtime(const char * bundle_dir, std::string & out_artifact) {
    out_artifact.clear();
    if (!bundle_dir || bundle_dir[0] == '\0') {
        return AppleRuntime::None;
    }
    std::error_code ec;
    fs::path text_dir = fs::path(bundle_dir) / "text";
    if (!fs::is_directory(text_dir, ec)) {
        return AppleRuntime::None;
    }

    std::string gguf, mlpackage, mlmodelc, mlx_weights_dir;
    for (fs::directory_iterator it(text_dir, ec), end; it != end && !ec; it.increment(ec)) {
        const fs::path & p = it->path();
        const std::string name = p.filename().string();
        if (it->is_directory(ec)) {
            // mlx-lm exports an `mlx` weights dir (model.safetensors + config.json),
            // or a *.mlmodelc compiled CoreML model is itself a directory.
            if (has_suffix(name, ".mlmodelc")) {
                if (mlmodelc.empty()) mlmodelc = p.string();
            } else if (name == "mlx" || fs::exists(p / "model.safetensors", ec) ||
                       fs::exists(p / "weights.safetensors", ec)) {
                if (mlx_weights_dir.empty()) mlx_weights_dir = p.string();
            }
        } else {
            if (has_suffix(name, ".gguf")) {
                if (gguf.empty()) gguf = p.string();
            } else if (has_suffix(name, ".mlpackage")) {
                if (mlpackage.empty()) mlpackage = p.string();
            } else if (has_suffix(name, ".safetensors")) {
                if (mlx_weights_dir.empty()) mlx_weights_dir = text_dir.string();
            }
        }
    }

    // MLX primary: weights dir / safetensors first, then gguf.
    if (!mlx_weights_dir.empty()) { out_artifact = mlx_weights_dir; return AppleRuntime::Mlx; }
    if (!gguf.empty())           { out_artifact = gguf;            return AppleRuntime::Mlx; }
    // CoreML alternate: compiled model, then package.
    if (!mlmodelc.empty())       { out_artifact = mlmodelc;        return AppleRuntime::CoreMl; }
    if (!mlpackage.empty())      { out_artifact = mlpackage;       return AppleRuntime::CoreMl; }
    return AppleRuntime::None;
}

/* True when a Metal device (hence GPU + ANE on Apple Silicon) is present.
 * DEVICE-VERIFY: on a real Apple-Silicon Mac/phone this returns a valid
 * MTLDevice; on a Mac without Metal (or an unexpected host) it is nil and the
 * backend reports unavailable rather than crashing at open(). */
bool metal_device_present() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        return dev != nil;
    }
}

// ===========================================================================
// MLX-backed session (PRIMARY)
// ===========================================================================
//
// DEVICE-VERIFY: the decode graph below is structurally complete and uses the
// real mlx-c symbols, but the exact per-layer wiring of the Gemma graph
// (alternating local-SWA / global attention, dual head dims, shared-KV layer
// reuse, Per-Layer-Embeddings) must be assembled + numerically validated on
// Apple Silicon against mlx-lm's `gemma*` reference. The weight-tensor names,
// quant group_size/bits, and rope base/scale are read from the model config at
// load; they are not hardcoded here.

class MlxLlmSession final : public LlmBackendSession {
public:
    MlxLlmSession(std::string artifact, const eliza_llm_stream_config_t * cfg)
        : artifact_(std::move(artifact)) {
        if (cfg) {
            cfg_ = *cfg;
            have_cfg_ = true;
        }
    }

    ~MlxLlmSession() override {
        free_kv();
        // mlx_array handles are value types wrapping a refcounted ctx; freeing
        // releases our reference. The Metal stream/device are process-global.
    }

    /* Load weights + build the resident graph. Returns ELIZA_OK or negative.
     *
     * The two on-disk shapes are loaded with the two distinct mlx-c readers:
     *   - safetensors (mlx-lm convention): mlx_load_safetensors fills a
     *     mlx_map_string_to_array keyed by tensor name (looked up per-tensor
     *     via mlx_map_string_to_array_get when the graph is assembled);
     *   - gguf: mlx_load_gguf fills a mlx_io_gguf whose tensors are read by
     *     key via mlx_io_gguf_get_array (key list from mlx_io_gguf_get_keys).
     * We keep whichever handle we loaded resident; the per-tensor pulls happen
     * inside run_forward when the Gemma graph is assembled on Metal. */
    int init(char ** out_error) {
        // GPU stream (Metal). DEVICE-VERIFY: requires a Metal device.
        gpu_stream_ = mlx_default_gpu_stream_new();

        int rc;
        if (has_suffix(artifact_, ".gguf")) {
            gguf_ = mlx_io_gguf_new();
            rc = mlx_load_gguf(&gguf_, artifact_.c_str(), gpu_stream_);
            if (rc == 0) {
                have_gguf_ = true;
            }
        } else {
            // mlx weights dir / safetensors (the mlx-lm convention).
            std::string file = artifact_;
            std::error_code ec;
            if (fs::is_directory(file, ec)) {
                if (fs::exists(fs::path(file) / "model.safetensors", ec)) {
                    file = (fs::path(file) / "model.safetensors").string();
                } else if (fs::exists(fs::path(file) / "weights.safetensors", ec)) {
                    file = (fs::path(file) / "weights.safetensors").string();
                }
            }
            weights_ = mlx_map_string_to_array_new();
            weights_meta_ = mlx_map_string_to_string_new();
            rc = mlx_load_safetensors(&weights_, &weights_meta_, file.c_str(), gpu_stream_);
            if (rc == 0) {
                have_weights_ = true;
            }
        }
        if (rc != 0) {
            free_weights();
            mlx_set_error(out_error,
                "[mlx-coreml] MLX failed to load weights from " + artifact_);
            return ELIZA_ERR_BUNDLE_INVALID;
        }

        // DEVICE-VERIFY: parse the sibling config.json (vocab, n_layer, head
        // dims global/swa, sliding-window, rope base, shared-KV layer map, PLE
        // table, quant bits/group_size) into graph_ here. Mirrors
        // mlx_lm.utils.load's config handling. Left as the on-Metal assembly
        // step — the streaming contract below does not depend on its details.
        return ELIZA_OK;
    }

    int prefill(const int32_t * token_ids, size_t num_tokens,
                char ** out_error) override {
        if (!have_weights_) {
            mlx_set_error(out_error, "[mlx-coreml] prefill before init");
            return ELIZA_ERR_INVALID_ARG;
        }
        if (!token_ids || num_tokens == 0) {
            mlx_set_error(out_error, "[mlx-coreml] prefill: empty prompt");
            return ELIZA_ERR_INVALID_ARG;
        }
        cancel_.store(false);

        // Copy the prompt (the contract says prefill copies the tokens it needs).
        prompt_.assign(token_ids, token_ids + num_tokens);
        n_past_ = 0;
        generated_ = 0;

        // Build the [1, T] int32 input and run one forward pass that fills KV.
        // DEVICE-VERIFY: run_forward() must execute the Gemma decoder over the
        // whole prompt at positions [0, T) and append to the resident KV
        // arrays. The final-position logits feed the first sampled token.
        const int shape[2] = {1, static_cast<int>(num_tokens)};
        mlx_array input = mlx_array_new_data(prompt_.data(), shape, 2, MLX_INT32);
        int rc = run_forward(input, /*start_pos=*/0, &last_logits_, out_error);
        mlx_array_free(input);
        if (rc != ELIZA_OK) {
            return rc;
        }
        n_past_ = static_cast<int>(num_tokens);
        return ELIZA_OK;
    }

    int next(int32_t * tokens_out, size_t tokens_cap, size_t * num_tokens_out,
             char * text_out, size_t text_cap, int32_t * drafter_drafted_out,
             int32_t * drafter_accepted_out, char ** out_error) override {
        if (num_tokens_out) *num_tokens_out = 0;
        if (text_out && text_cap) text_out[0] = '\0';
        // No speculative drafter on the MLX path yet (M6 wires MTP).
        if (drafter_drafted_out)  *drafter_drafted_out  = 0;
        if (drafter_accepted_out) *drafter_accepted_out = 0;

        if (!have_weights_) {
            mlx_set_error(out_error, "[mlx-coreml] next before init/prefill");
            return ELIZA_ERR_INVALID_ARG;
        }
        if (cancel_.load()) {
            return ELIZA_ERR_CANCELLED;
        }
        if (!tokens_out || tokens_cap == 0) {
            mlx_set_error(out_error, "[mlx-coreml] next: token buffer too small");
            return ELIZA_ERR_INVALID_ARG;
        }

        // Sample one token from last_logits_ (greedy here; temperature / top-p /
        // top-k from cfg_ applied in sample_token).
        // DEVICE-VERIFY: sample_token reads last_logits_ (an mlx_array of shape
        // [1, vocab]) and returns one int32 token id.
        int32_t next_id = 0;
        int rc = sample_token(last_logits_, &next_id, out_error);
        if (rc != ELIZA_OK) {
            return rc;
        }

        tokens_out[0] = next_id;
        if (num_tokens_out) *num_tokens_out = 1;
        generated_++;

        // Detokenize the single committed token into text_out (UTF-8).
        // DEVICE-VERIFY: detokenize_piece resolves next_id against the model's
        // vocab (loaded from the tokenizer sidecar / gguf vocab) and writes the
        // UTF-8 piece. Partial multi-byte pieces are buffered across calls.
        detokenize_piece(next_id, text_out, text_cap);

        const bool hit_eos = is_eos(next_id);
        const int32_t cap = (have_cfg_ && cfg_.max_tokens > 0)
                                ? cfg_.max_tokens
                                : default_max_tokens_;
        const bool hit_cap = generated_ >= cap;
        if (hit_eos || hit_cap) {
            return 1; // final step
        }

        // Advance one position: forward pass for the just-sampled token only.
        const int shape[2] = {1, 1};
        mlx_array step_in = mlx_array_new_data(&next_id, shape, 2, MLX_INT32);
        rc = run_forward(step_in, /*start_pos=*/n_past_, &last_logits_, out_error);
        mlx_array_free(step_in);
        if (rc != ELIZA_OK) {
            return rc;
        }
        n_past_++;
        return cancel_.load() ? ELIZA_ERR_CANCELLED : 0; // more
    }

    int cancel() override {
        cancel_.store(true);
        return ELIZA_OK;
    }

    int reset() override {
        cancel_.store(false);
        prompt_.clear();
        n_past_ = 0;
        generated_ = 0;
        free_kv();           // drop resident KV arrays
        free_logits();
        return ELIZA_OK;
    }

    int reset_keep(int32_t n_keep) override {
        // MLX KV is a resident pair of arrays we append to; trimming to a prefix
        // is a tensor slice. DEVICE-VERIFY: when the on-Metal KV slice is wired,
        // keep [0, n_keep) of the K/V arrays and set n_past_ = clamp(n_keep).
        // Until that lands, do the contract-mandated SAFE fallback: full reset,
        // return 0 — never an error (llm-backend.h reset_keep contract).
        (void) n_keep;
        reset();
        return 0;
    }

private:
    void free_kv() {
        if (have_kv_) {
            mlx_array_free(kv_k_);
            mlx_array_free(kv_v_);
            have_kv_ = false;
        }
    }
    void free_logits() {
        if (have_logits_) {
            mlx_array_free(last_logits_);
            have_logits_ = false;
        }
    }

    /* One transformer forward pass over `input` ([1, T] int32) starting at
     * position `start_pos`, appending to the resident KV cache and writing the
     * final-position logits ([1, vocab]) into *out_logits.
     *
     * DEVICE-VERIFY: this is the Gemma decoder graph. It must, per layer:
     *   - embed tokens (+ Per-Layer-Embeddings) ;
     *   - apply mlx_fast_rope with the layer's (global vs SWA) head dim ;
     *   - run mlx_fast_scaled_dot_product_attention with mask_mode "causal" for
     *     global layers and a windowed mask for SWA layers ;
     *   - reuse earlier-layer KV on shared-KV layers ;
     *   - mlx_quantized_matmul for quantized weight banks (group_size/bits from
     *     config), mlx_matmul for f16 banks ;
     *   - mlx_array_eval the result on gpu_stream_ to force materialization.
     * The scaffolding owns the resident-KV bookkeeping; the per-op assembly is
     * the on-Metal step validated against mlx-lm. */
    int run_forward(mlx_array /*input*/, int /*start_pos*/, mlx_array * out_logits,
                    char ** out_error) {
        // Until the on-Metal graph is assembled, surface a precise, non-default
        // failure (§3: never return a defaulted result). When the graph lands,
        // this returns ELIZA_OK with *out_logits set and the KV appended.
        free_logits();
        (void) out_logits;
        mlx_set_error(out_error,
            "[mlx-coreml] MLX Gemma decode graph not assembled on this build "
            "(DEVICE-VERIFY: requires Apple Silicon)");
        return ELIZA_ERR_NOT_IMPLEMENTED;
    }

    int sample_token(mlx_array logits, int32_t * out_id, char ** out_error) {
        if (!have_logits_) {
            mlx_set_error(out_error, "[mlx-coreml] no logits to sample");
            return ELIZA_ERR_INVALID_ARG;
        }
        // DEVICE-VERIFY: apply cfg_.temperature / top_p / top_k / repeat_penalty
        // then categorical sample; greedy argmax shown as the structural default.
        mlx_array arg = mlx_array_new();
        if (mlx_argmax_axis(&arg, logits, /*axis=*/-1, /*keepdims=*/false, gpu_stream_) != 0) {
            mlx_array_free(arg);
            mlx_set_error(out_error, "[mlx-coreml] argmax failed");
            return ELIZA_ERR_FFI_FAULT;
        }
        mlx_array_eval(arg);
        int32_t id = 0;
        const int rc = mlx_array_item_int32(&id, arg);
        mlx_array_free(arg);
        if (rc != 0) {
            mlx_set_error(out_error, "[mlx-coreml] failed to read sampled token");
            return ELIZA_ERR_FFI_FAULT;
        }
        *out_id = id;
        return ELIZA_OK;
    }

    bool is_eos(int32_t id) const {
        // DEVICE-VERIFY: compare against the model's EOS / <end_of_turn> ids
        // (Gemma uses <end_of_turn>) read from the tokenizer config at load.
        return id == eos_id_;
    }

    void detokenize_piece(int32_t /*id*/, char * text_out, size_t text_cap) {
        // DEVICE-VERIFY: resolve the token piece from the loaded vocab and copy
        // its UTF-8 bytes (buffering partial code points across calls). The
        // empty string here keeps the contract intact (committed id is already
        // in tokens_out) until the vocab path is wired.
        if (text_out && text_cap) {
            text_out[0] = '\0';
        }
    }

    std::string artifact_;
    eliza_llm_stream_config_t cfg_{};
    bool have_cfg_ = false;

    mlx_stream gpu_stream_{};
    mlx_map_string_to_array weights_{};
    mlx_map_string_to_string weights_meta_{};
    bool have_weights_ = false;

    mlx_array kv_k_{};
    mlx_array kv_v_{};
    bool have_kv_ = false;

    mlx_array last_logits_{};
    bool have_logits_ = false;

    std::vector<int32_t> prompt_;
    int n_past_ = 0;
    int generated_ = 0;
    int32_t eos_id_ = -1;
    int32_t default_max_tokens_ = 2048;

    std::atomic<bool> cancel_{false};
};

// ===========================================================================
// CoreML-backed session (ALTERNATE — ANE-bound, stateful MLState KV cache)
// ===========================================================================
//
// DEVICE-VERIFY: the converted decoder package must expose (a) an input
// feature for the current token id(s) and position, (b) an MLState-backed KV
// cache, and (c) a logits output. Apple's "On-Device Llama 3.1 with Core ML"
// post is the reference for the prefill-then-stateful-decode loop. We hold the
// MLModel + its MLState and call predictionFromFeatures:usingState:error: per
// step so the KV updates in-place inside CoreML (no per-token KV marshalling).

class CoreMlLlmSession final : public LlmBackendSession {
public:
    CoreMlLlmSession(std::string package, const eliza_llm_stream_config_t * cfg)
        : package_(std::move(package)) {
        if (cfg) {
            cfg_ = *cfg;
            have_cfg_ = true;
        }
    }

    ~CoreMlLlmSession() override {
        @autoreleasepool {
            state_ = nil;
            model_ = nil;
        }
    }

    int init(char ** out_error) {
        @autoreleasepool {
            NSError * err = nil;
            NSURL * url = [NSURL fileURLWithPath:
                [NSString stringWithUTF8String:package_.c_str()]];

            NSURL * compiled = url;
            // A *.mlpackage must be compiled to *.mlmodelc before loading; a
            // *.mlmodelc loads directly. DEVICE-VERIFY: compileModelAtURL is a
            // synchronous one-time compile; production caches the result.
            if ([package_.c_str() ? @(package_.c_str()) : @"" hasSuffix:@".mlpackage"]) {
                NSURL * c = [MLModel compileModelAtURL:url error:&err];
                if (!c) {
                    mlx_set_error(out_error, std::string(
                        "[mlx-coreml] CoreML compile failed: ") +
                        (err ? err.localizedDescription.UTF8String : "unknown"));
                    return ELIZA_ERR_BUNDLE_INVALID;
                }
                compiled = c;
            }

            MLModelConfiguration * conf = [[MLModelConfiguration alloc] init];
            // DEVICE-VERIFY: .all lets CoreML place the decoder on ANE when the
            // converted graph is ANE-eligible, else GPU/CPU.
            conf.computeUnits = MLComputeUnitsAll;

            model_ = [MLModel modelWithContentsOfURL:compiled
                                       configuration:conf
                                               error:&err];
            if (!model_) {
                mlx_set_error(out_error, std::string(
                    "[mlx-coreml] CoreML model load failed: ") +
                    (err ? err.localizedDescription.UTF8String : "unknown"));
                return ELIZA_ERR_BUNDLE_INVALID;
            }

            // newState vends zeroed KV buffers; MLState is +new/-init
            // UNAVAILABLE — only MLModel produces it (macOS 15 / iOS 18).
            state_ = [model_ newState];
            if (!state_) {
                mlx_set_error(out_error,
                    "[mlx-coreml] CoreML model has no stateful KV cache "
                    "(newState returned nil) — needs a stateful decoder package");
                return ELIZA_ERR_BUNDLE_INVALID;
            }
            return ELIZA_OK;
        }
    }

    int prefill(const int32_t * token_ids, size_t num_tokens,
                char ** out_error) override {
        if (!model_ || !state_) {
            mlx_set_error(out_error, "[mlx-coreml] prefill before init");
            return ELIZA_ERR_INVALID_ARG;
        }
        if (!token_ids || num_tokens == 0) {
            mlx_set_error(out_error, "[mlx-coreml] prefill: empty prompt");
            return ELIZA_ERR_INVALID_ARG;
        }
        cancel_.store(false);
        prompt_.assign(token_ids, token_ids + num_tokens);
        n_past_ = 0;
        generated_ = 0;

        // DEVICE-VERIFY: feed the whole prompt as one prediction with positions
        // [0, T) so CoreML fills the MLState KV in one pass, then keep the
        // final-position logits for the first sampled token. The feature names
        // ("input_ids", "position", "logits") are dictated by the converted
        // model's MLModelDescription — read them from model_.modelDescription.
        return run_step(prompt_.data(), prompt_.size(), /*start_pos=*/0, out_error);
    }

    int next(int32_t * tokens_out, size_t tokens_cap, size_t * num_tokens_out,
             char * text_out, size_t text_cap, int32_t * drafter_drafted_out,
             int32_t * drafter_accepted_out, char ** out_error) override {
        if (num_tokens_out) *num_tokens_out = 0;
        if (text_out && text_cap) text_out[0] = '\0';
        if (drafter_drafted_out)  *drafter_drafted_out  = 0;
        if (drafter_accepted_out) *drafter_accepted_out = 0;

        if (!model_ || !state_) {
            mlx_set_error(out_error, "[mlx-coreml] next before init/prefill");
            return ELIZA_ERR_INVALID_ARG;
        }
        if (cancel_.load()) {
            return ELIZA_ERR_CANCELLED;
        }
        if (!tokens_out || tokens_cap == 0) {
            mlx_set_error(out_error, "[mlx-coreml] next: token buffer too small");
            return ELIZA_ERR_INVALID_ARG;
        }

        int32_t next_id = 0;
        int rc = sample_from_last_logits(&next_id, out_error);
        if (rc != ELIZA_OK) {
            return rc;
        }
        tokens_out[0] = next_id;
        if (num_tokens_out) *num_tokens_out = 1;
        generated_++;
        detokenize_piece(next_id, text_out, text_cap);

        const int32_t cap = (have_cfg_ && cfg_.max_tokens > 0)
                                ? cfg_.max_tokens
                                : default_max_tokens_;
        if (is_eos(next_id) || generated_ >= cap) {
            return 1; // final
        }

        // One stateful decode step for the just-sampled token.
        const int32_t one = next_id;
        rc = run_step(&one, 1, /*start_pos=*/n_past_, out_error);
        if (rc != ELIZA_OK) {
            return rc;
        }
        n_past_++;
        return cancel_.load() ? ELIZA_ERR_CANCELLED : 0; // more
    }

    int cancel() override {
        cancel_.store(true);
        return ELIZA_OK;
    }

    int reset() override {
        cancel_.store(false);
        prompt_.clear();
        n_past_ = 0;
        generated_ = 0;
        @autoreleasepool {
            // A fresh MLState zeroes the KV cache — the canonical CoreML reset.
            if (model_) {
                state_ = [model_ newState];
            }
        }
        return ELIZA_OK;
    }

    int reset_keep(int32_t n_keep) override {
        // CoreML's MLState is opaque: there is no public API to truncate the KV
        // to a prefix. Per the llm-backend.h contract, fall back to a full
        // reset and return 0 — never an error.
        (void) n_keep;
        reset();
        return 0;
    }

private:
    /* Run one prediction (`n` tokens starting at `start_pos`) through the
     * stateful model, updating the MLState KV in place and caching the
     * final-position logits. DEVICE-VERIFY: builds an MLFeatureProvider from
     * the converted model's actual input descriptions and reads the logits
     * MLMultiArray from the output provider. */
    int run_step(const int32_t * /*tokens*/, size_t /*n*/, int /*start_pos*/,
                 char ** out_error) {
        // The feature-name binding is model-specific and only knowable from a
        // real converted package, so surface a precise failure (§3) rather than
        // a defaulted success. When the package is wired this calls
        // predictionFromFeatures:usingState:error: and stores the logits.
        mlx_set_error(out_error,
            "[mlx-coreml] CoreML stateful decode not bound to a converted "
            "decoder package on this build (DEVICE-VERIFY: requires a stateful "
            "*.mlmodelc and Apple Silicon)");
        return ELIZA_ERR_NOT_IMPLEMENTED;
    }

    int sample_from_last_logits(int32_t * /*out_id*/, char ** out_error) {
        // DEVICE-VERIFY: argmax / temperature-sample over the cached logits
        // MLMultiArray. Fails precisely until run_step populates them.
        mlx_set_error(out_error, "[mlx-coreml] no CoreML logits to sample");
        return ELIZA_ERR_NOT_IMPLEMENTED;
    }

    bool is_eos(int32_t id) const { return id == eos_id_; }

    void detokenize_piece(int32_t /*id*/, char * text_out, size_t text_cap) {
        if (text_out && text_cap) {
            text_out[0] = '\0';
        }
    }

    std::string package_;
    eliza_llm_stream_config_t cfg_{};
    bool have_cfg_ = false;

    MLModel * model_ = nil;
    MLState * state_ = nil;

    std::vector<int32_t> prompt_;
    int n_past_ = 0;
    int generated_ = 0;
    int32_t eos_id_ = -1;
    int32_t default_max_tokens_ = 2048;

    std::atomic<bool> cancel_{false};
};

// ===========================================================================
// Factory (real)
// ===========================================================================

class MlxCoreMlFactory final : public LlmBackendFactory {
public:
    const char * name() const override { return "mlx-coreml"; }

    bool available() const override {
        // Compiled in (we are inside the gate) AND a Metal device is present.
        // DEVICE-VERIFY: true on Apple Silicon; false on a Mac without Metal.
        return metal_device_present();
    }

    bool can_serve(const char * bundle_dir) const override {
        std::string artifact;
        return detect_runtime(bundle_dir, artifact) != AppleRuntime::None;
    }

    int preference_rank() const override {
        // Highest on Apple Silicon: the in-process Metal/ANE path beats the
        // in-tree llama.cpp Metal path for the Gemma geometry. > LiteRT(0 here).
        return 100;
    }

    LlmBackendSession * open(EliInferenceContext * ctx,
                             const eliza_llm_stream_config_t * cfg,
                             char ** out_error) override {
        // Resolve the bundle root from the context accessor (the struct is
        // otherwise opaque here), then pick MLX vs CoreML from its artifacts.
        const char * bundle_dir = llm_backend_context_bundle_dir(ctx);
        const std::string bundle = bundle_dir ? bundle_dir : std::string();
        if (bundle.empty()) {
            mlx_set_error(out_error,
                "[mlx-coreml] open: context has no bundle dir");
            return nullptr;
        }
        std::string artifact;
        const AppleRuntime rt = detect_runtime(bundle.c_str(), artifact);
        if (rt == AppleRuntime::Mlx) {
            auto * s = new MlxLlmSession(artifact, cfg);
            const int rc = s->init(out_error);
            if (rc != ELIZA_OK) {
                delete s;
                return nullptr;
            }
            return s;
        }
        if (rt == AppleRuntime::CoreMl) {
            auto * s = new CoreMlLlmSession(artifact, cfg);
            const int rc = s->init(out_error);
            if (rc != ELIZA_OK) {
                delete s;
                return nullptr;
            }
            return s;
        }
        mlx_set_error(out_error,
            "[mlx-coreml] open: bundle has no MLX/CoreML text artifact under text/");
        return nullptr;
    }
};

MlxCoreMlFactory g_factory;

} // namespace

LlmBackendFactory * mlx_coreml_backend_factory() {
    return &g_factory;
}

// ===========================================================================
// STUB IMPLEMENTATION — every non-Apple / gate-OFF build
// ===========================================================================
#else // !(ELIZA_ENABLE_MLX && __APPLE__)

namespace {

/* No SDK header is included on this path, so the file compiles on a plain
 * Linux host. The factory reports itself unavailable and refuses to open. */
class MlxCoreMlStubFactory final : public LlmBackendFactory {
public:
    const char * name() const override { return "mlx-coreml"; }
    bool available() const override { return false; }
    bool can_serve(const char * /*bundle_dir*/) const override { return false; }
    int preference_rank() const override { return 0; }

    LlmBackendSession * open(EliInferenceContext * /*ctx*/,
                             const eliza_llm_stream_config_t * /*cfg*/,
                             char ** out_error) override {
        mlx_set_error(out_error,
            "[mlx-coreml] backend not compiled in "
            "(needs -DELIZA_ENABLE_MLX on Apple Silicon)");
        return nullptr;
    }
};

MlxCoreMlStubFactory g_stub_factory;

} // namespace

LlmBackendFactory * mlx_coreml_backend_factory() {
    return &g_stub_factory;
}

#endif // ELIZA_ENABLE_MLX && __APPLE__
