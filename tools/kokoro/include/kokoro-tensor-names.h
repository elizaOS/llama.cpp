// SPDX-License-Identifier: MIT
//
// kokoro-tensor-names.h — tensor-name compatibility for Kokoro GGUFs.
//
// The published elizaOS Kokoro bundles use the `kokoro.*` namespace emitted by
// tools/kokoro/convert_kokoro_pth_to_gguf.py. Older local/dev GGUFs used short
// unprefixed names. Keep the aliases centralized so desktop and iOS builds load
// the same schema through the same kokoro_lib code path.

#pragma once

namespace eliza_kokoro {

inline constexpr const char * KOKORO_TENSOR_BERT_TOKEN_EMBD[] = {
    "kokoro.bert.token_embd.weight",
    "kokoro.bert.embd.tok.weight",
    "bert.embd.tok.weight",
    "kokoro.token_embd.weight",
    nullptr,
};

inline constexpr const char * KOKORO_TENSOR_BERT_ATTN_Q[] = {
    "kokoro.bert.layer.attn_q.weight",
    "kokoro.bert.attn_q.weight",
    "bert.layer.attn_q.weight",
    "bert.attn_q.weight",
    nullptr,
};

inline constexpr const char * KOKORO_TENSOR_DURATION_PROJ[] = {
    "kokoro.predictor.duration_proj.weight",
    "kokoro.predictor.duration.weight",
    "predictor.duration_proj.weight",
    "pred.duration_proj.weight",
    "pred.duration.weight",
    nullptr,
};

inline constexpr const char * KOKORO_TENSOR_F0_PROJ[] = {
    "kokoro.predictor.F0_proj.weight",
    "predictor.F0_proj.weight",
    "pred.F0_proj.weight",
    nullptr,
};

inline constexpr const char * KOKORO_TENSOR_N_PROJ[] = {
    "kokoro.predictor.N_proj.weight",
    "predictor.N_proj.weight",
    "pred.N_proj.weight",
    nullptr,
};

inline constexpr const char * KOKORO_TENSOR_GEN_CONV_POST[] = {
    "kokoro.gen.conv_post.weight",
    "kokoro.decoder.gen.conv_post.weight",
    "decoder.gen.conv_post.weight",
    "dec.gen.conv_post.weight",
    nullptr,
};

inline const char * kokoro_pick_tensor_name(
        const char * const * aliases,
        bool (*has_tensor)(const char * name, void * user_data),
        void * user_data) {
    if (!aliases || !has_tensor) return nullptr;
    for (const char * const * p = aliases; *p; ++p) {
        if (has_tensor(*p, user_data)) return *p;
    }
    return nullptr;
}

} // namespace eliza_kokoro
