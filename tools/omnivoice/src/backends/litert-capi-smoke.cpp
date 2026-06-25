/*
 * litert-capi-smoke.cpp — C-API linkage proof for embedding LiteRT-LM into
 * libelizainference.
 *
 * Loads a .litertlm model through the STABLE C API (litert_lm_* — engine.h),
 * prefills a prompt, decodes a response, and prints the text. This is the
 * linkage proof that the prebuilt liblitert-lm.so can be driven from C/C++
 * code with no libc++ C++-ABI dependency (C API only).
 *
 * Build:
 *   g++ -std=c++17 -I<sdk>/include litert-capi-smoke.cpp \
 *       <sdk>/lib/liblitert-lm.so -Wl,-rpath,<sdk>/lib -o /tmp/litert-smoke
 *
 * Usage:
 *   litert-capi-smoke <model.litertlm> ["prompt"]
 */

#include "engine.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    const char * model_path =
        argc > 1 ? argv[1] : "/home/shaw/milady/eliza/.cache/gemma-4-E2B-it.litertlm";
    const char * prompt =
        argc > 2 ? argv[2] : "What is the capital of France?";

    // Keep the library quiet except for true errors (0=VERBOSE..5=FATAL).
    litert_lm_set_min_log_level(4 /* ERROR */);

    // 1. Engine settings — CPU backend, no vision/audio backend for a text
    //    smoke. The C API mirrors the python ctypes binding's create order.
    LiteRtLmEngineSettings * settings = litert_lm_engine_settings_create(
        model_path, "cpu", /*vision_backend=*/nullptr, /*audio_backend=*/nullptr);
    if (!settings) {
        std::fprintf(stderr, "[smoke] engine_settings_create failed for %s\n",
                     model_path);
        return 1;
    }
    // Leave max_num_tokens unset: the model's compiled KV-cache slice sizing is
    // the safe default (overriding it can break the DYNAMIC_UPDATE_SLICE shape
    // on prefill). The CLI's default `run` path passes max_num_tokens=None.

    // 2. Build the engine (loads the model).
    LiteRtLmEngine * engine = litert_lm_engine_create(settings);
    litert_lm_engine_settings_delete(settings);
    if (!engine) {
        std::fprintf(stderr, "[smoke] engine_create failed\n");
        return 1;
    }

    // 3. Session config — bound decode length. We apply the Gemma chat template
    //    by hand below (apply_prompt_template=false) because the raw
    //    prefill/decode session path expects an already-formatted prompt; the
    //    library's own template wrapping lives on the Conversation API.
    LiteRtLmSessionConfig * session_cfg = litert_lm_session_config_create();
    if (session_cfg) {
        litert_lm_session_config_set_max_output_tokens(session_cfg, 64);
        litert_lm_session_config_set_apply_prompt_template(session_cfg, false);
    }

    LiteRtLmSession * session =
        litert_lm_engine_create_session(engine, session_cfg);
    if (session_cfg) litert_lm_session_config_delete(session_cfg);
    if (!session) {
        std::fprintf(stderr, "[smoke] engine_create_session failed\n");
        litert_lm_engine_delete(engine);
        return 1;
    }

    // 4. Prefill the prompt as a single text input. Wrap the user prompt in the
    //    Gemma instruct chat template so an instruct .litertlm answers the
    //    question instead of continuing raw text.
    const std::string templated =
        std::string("<start_of_turn>user\n") + prompt +
        "<end_of_turn>\n<start_of_turn>model\n";

    LiteRtLmInputData input;
    input.type = kLiteRtLmInputDataTypeText;
    input.data = templated.c_str();
    input.size = templated.size();

    if (litert_lm_session_run_prefill(session, &input, 1) != 0) {
        std::fprintf(stderr, "[smoke] run_prefill failed\n");
        litert_lm_session_delete(session);
        litert_lm_engine_delete(engine);
        return 1;
    }

    // 5. Decode the response (blocking; returns the full candidate batch).
    LiteRtLmResponses * responses = litert_lm_session_run_decode(session);
    if (!responses) {
        std::fprintf(stderr, "[smoke] run_decode failed\n");
        litert_lm_session_delete(session);
        litert_lm_engine_delete(engine);
        return 1;
    }

    int num = litert_lm_responses_get_num_candidates(responses);
    int rc = 1;
    if (num > 0) {
        const char * text = litert_lm_responses_get_response_text_at(responses, 0);
        if (text) {
            std::printf("PROMPT:   %s\n", prompt);
            std::printf("RESPONSE: %s\n", text);
            rc = 0;
        }
    }
    if (rc != 0) {
        std::fprintf(stderr, "[smoke] no response text (candidates=%d)\n", num);
    }

    litert_lm_responses_delete(responses);
    litert_lm_session_delete(session);
    litert_lm_engine_delete(engine);
    return rc;
}
