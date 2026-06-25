// SPDX-License-Identifier: MIT
//
// test_kokoro_tensor_names.cpp — regression coverage for issue #9588.
//
// macOS and iOS both link kokoro_lib into the fused libelizainference target.
// If these aliases drift from the published GGUF schema, both platforms can
// export Kokoro symbols yet fail or silently skip the real weights at load time.

#include "kokoro-tensor-names.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>

namespace {

bool has_name(const char * name, void * user_data) {
    const auto * names = static_cast<const std::set<std::string> *>(user_data);
    return names->find(name) != names->end();
}

void expect_pick(
        const char * const * aliases,
        const std::set<std::string> & available,
        const char * expected) {
    const char * actual = eliza_kokoro::kokoro_pick_tensor_name(
        aliases,
        has_name,
        (void *) &available);
    assert(actual != nullptr);
    assert(std::strcmp(actual, expected) == 0);
}

} // namespace

int main() {
    using namespace eliza_kokoro;

    const std::set<std::string> published_schema = {
        "kokoro.bert.token_embd.weight",
        "kokoro.bert.layer.attn_q.weight",
        "kokoro.predictor.duration_proj.weight",
        "kokoro.predictor.F0_proj.weight",
        "kokoro.predictor.N_proj.weight",
        "kokoro.gen.conv_post.weight",
    };

    expect_pick(KOKORO_TENSOR_BERT_TOKEN_EMBD, published_schema, "kokoro.bert.token_embd.weight");
    expect_pick(KOKORO_TENSOR_BERT_ATTN_Q, published_schema, "kokoro.bert.layer.attn_q.weight");
    expect_pick(KOKORO_TENSOR_DURATION_PROJ, published_schema, "kokoro.predictor.duration_proj.weight");
    expect_pick(KOKORO_TENSOR_F0_PROJ, published_schema, "kokoro.predictor.F0_proj.weight");
    expect_pick(KOKORO_TENSOR_N_PROJ, published_schema, "kokoro.predictor.N_proj.weight");
    expect_pick(KOKORO_TENSOR_GEN_CONV_POST, published_schema, "kokoro.gen.conv_post.weight");

    const std::set<std::string> legacy_schema = {
        "bert.embd.tok.weight",
        "bert.layer.attn_q.weight",
        "pred.duration_proj.weight",
        "pred.F0_proj.weight",
        "pred.N_proj.weight",
        "dec.gen.conv_post.weight",
    };

    expect_pick(KOKORO_TENSOR_BERT_TOKEN_EMBD, legacy_schema, "bert.embd.tok.weight");
    expect_pick(KOKORO_TENSOR_BERT_ATTN_Q, legacy_schema, "bert.layer.attn_q.weight");
    expect_pick(KOKORO_TENSOR_DURATION_PROJ, legacy_schema, "pred.duration_proj.weight");
    expect_pick(KOKORO_TENSOR_F0_PROJ, legacy_schema, "pred.F0_proj.weight");
    expect_pick(KOKORO_TENSOR_N_PROJ, legacy_schema, "pred.N_proj.weight");
    expect_pick(KOKORO_TENSOR_GEN_CONV_POST, legacy_schema, "dec.gen.conv_post.weight");

    const std::set<std::string> empty_schema;
    assert(kokoro_pick_tensor_name(KOKORO_TENSOR_BERT_TOKEN_EMBD, has_name, (void *) &empty_schema) == nullptr);

    std::printf("test_kokoro_tensor_names: OK\n");
    return 0;
}
