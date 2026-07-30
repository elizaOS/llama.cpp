/**
 * Rejects Kokoro GGUFs at the generic causal-model boundary.
 *
 * Kokoro uses the architecture string as a GGUF format discriminator, but its
 * Albert encoder, prosody predictor, and iSTFT vocoder do not implement the
 * llama model contract. Synthesis and model loading therefore belong to the
 * owned tools/kokoro C ABI; accepting the file here would make generic callers
 * infer a nonexistent token-decoder tensor layout.
 */

#include "models.h"

void llama_model_kokoro::load_arch_hparams(llama_model_loader & ml) {
    (void) ml;
    throw std::runtime_error(
        "Kokoro GGUFs are TTS pipelines, not causal language models; "
        "load them through the tools/kokoro synthesis API."
    );
}

void llama_model_kokoro::load_arch_tensors(llama_model_loader &) {
}

// gcc's -Wsuggest-attribute=noreturn flags this method because the only
// control-flow path is an unconditional throw. We can't add [[noreturn]]
// to an overriding virtual that has a non-void return type, so silence
// the suggestion for this single function.
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif
std::unique_ptr<llm_graph_context> llama_model_kokoro::build_arch_graph(const llm_graph_params & params) const {
    // The causal-decoder graph contract cannot represent a predictor/decoder
    // TTS pipeline. Runtime synthesis enters through tools/kokoro's C ABI;
    // fail explicitly if an LM caller selects this architecture.
    (void)params;
    throw std::runtime_error(
        "LLM_ARCH_KOKORO is not a causal decoder architecture; "
        "use the tools/kokoro synthesis API."
    );
}
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
