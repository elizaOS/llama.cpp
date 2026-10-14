// llama-eagle3.cpp — stub for EAGLE3 speculative-decoding scaffolding.
//
// EAGLE3 (Extrapolation-aware Algorithm for Generating Likely Encoded
// Tokens, v3) is an upstream draft-model architecture; see PR #18039.
// This file is intentionally a stub: the public API symbols
// `llama_get_eagle3_target_features` and `llama_set_eagle3_g_embeddings`
// exist so callers can link against the eventual EAGLE3 surface, but
// the implementations log a warning and return a failure sentinel.
//
// The real implementation lands once the model loader, hparams,
// layer fields, and graph builder for LLM_ARCH_EAGLE3 are wired in.
// See /tmp/wave6-eagle3-real-journal.md for the full port plan.

#include "llama.h"
#include "llama-impl.h"

#include <cstddef>

struct ggml_tensor * llama_get_eagle3_target_features(struct llama_context * /*ctx*/) {
    LLAMA_LOG_WARN("%s: EAGLE3 not yet implemented (scaffolding only)\n", __func__);
    return nullptr;
}

int32_t llama_set_eagle3_g_embeddings(
        struct llama_context * /*ctx*/,
        const float          * /*embd*/,
        size_t                 /*n_embd*/,
        size_t                 /*n_tokens*/) {
    LLAMA_LOG_WARN("%s: EAGLE3 not yet implemented (scaffolding only)\n", __func__);
    return -1;
}
