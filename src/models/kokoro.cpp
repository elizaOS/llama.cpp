/**
 * Kokoro-82M TTS language-model arch for llama.cpp (R8 §3.1).
 *
 * Kokoro-82M is a StyleTTS-2 / iSTFTNet based TTS model. The LM component
 * is a causal transformer (12 layers, 768-dim hidden, 12 heads) with the
 * standard llama-style tensor layout. Registering the arch allows
 * `llama-quantize` to run the K-quant ladder (Q4_K_M / Q5_K_M / Q6_K)
 * over Kokoro GGUF weights without an explicit per-tensor exception table.
 *
 * The `build_arch_graph` implementation is a forward pass stub — the
 * runtime-side inference path drives the ONNX model directly (ONNX
 * `model.onnx` / `model_q4.onnx`). This file exists to satisfy the arch
 * registration required by the K-quant publish pipeline (R8 §3.1, I8
 * follow-up wired in W3-1 close-out).
 *
 * When a native llama.cpp Kokoro inference path is added, implement
 * `build_arch_graph` here and update `llama-graph.cpp` accordingly.
 */

#include "models.h"

void llama_model_kokoro::load_arch_hparams(llama_model_loader & ml) {
    // Kokoro-82M: 12-layer causal transformer, 768-dim hidden, 12 heads.
    // The GGUF metadata uses the standard llama hparam key set.
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps,     false);

    switch (hparams.n_layer) {
        case 12: type = LLM_TYPE_0_1B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_kokoro::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // Token embedding (text/phone vocabulary → 768-dim).
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // Final layer-norm + output projection (logit head over vocab).
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // Transformer blocks (standard Q/K/V/O + FFN Gate/Up/Down layout).
    for (int il = 0; il < n_layer; ++il) {
        layers[il].attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), {n_embd}, 0);

        layers[il].wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", il), {n_embd, n_embd_head_k * n_head},     0);
        layers[il].wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", il), {n_embd, n_embd_head_k * n_head_kv}, 0);
        layers[il].wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", il), {n_embd, n_embd_head_k * n_head_kv}, 0);
        layers[il].wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), {n_embd_head_k * n_head, n_embd},     0);

        layers[il].ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", il), {n_embd}, 0);

        layers[il].ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd,   n_ff}, 0);
        layers[il].ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd,   n_ff}, 0);
        layers[il].ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {n_ff,   n_embd}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_kokoro::build_arch_graph(const llm_graph_params & params) const {
    // Kokoro-82M runtime inference uses the ONNX path (model.onnx /
    // model_q4.onnx). This GGUF arch registration exists for the K-quant
    // quantization pipeline only (R8 §3.1). Calling build_arch_graph at
    // runtime is not expected — throw a clear diagnostic rather than
    // silently returning an empty graph.
    (void)params;
    throw std::runtime_error(
        "LLM_ARCH_KOKORO: native graph inference not yet implemented. "
        "Load the model via the ONNX runtime path (model.onnx / model_q4.onnx). "
        "This arch tag exists to enable the K-quant publish pipeline (R8 §3.1)."
    );
}
