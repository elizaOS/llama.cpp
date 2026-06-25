#pragma once
/*
 * litert-backend.h — LiteRT-LM in-process streaming-LLM backend (cutover plan M4).
 *
 * Implements the M3 backend seam (`llm-backend.h`) on top of Google's
 * LiteRT-LM C++ inference runtime, the in-process path for the Android NPU
 * tier (Qualcomm QNN / MediaTek NeuroPilot / Google Tensor), with an
 * optional desktop/iOS GPU fallback. LiteRT-LM is linked INTO
 * `libelizainference` and exposed behind the same FFI streaming symbols —
 * never a child process or TCP server (native/AGENTS.md §11, gemma4 cutover).
 *
 * The whole real implementation is gated behind the CMake define
 * `ELIZA_ENABLE_LITERT`. When that flag is OFF this header pulls in NO
 * LiteRT-LM SDK headers, so the file compiles on a host without the SDK and
 * the factory links in as a no-op: `available()` is false and `open()`
 * returns nullptr + sets `*out_error` "not compiled in".
 *
 * ── Targeted runtime API (STABLE C API — engine.h) ────────────────────────
 * Repo:    https://github.com/google-ai-edge/LiteRT-LM  (`main`)
 * Docs:    https://developers.google.com/edge/litert-lm/cpp
 *          https://ai.google.dev/edge/litert/next/litert_lm_npu
 * Header:  <ELIZA_LITERT_SDK_DIR>/include/engine.h  (extern "C" — no `litert::lm`)
 *
 * The backend targets the C API ONLY — the ABI-stable `litert_lm_*` functions —
 * so the fused libelizainference carries no libc++ C++-ABI dependency on the
 * prebuilt liblitert-lm.so. Symbols this backend uses (verbatim from engine.h):
 *   - litert_lm_engine_settings_create(model_path, backend_str, vision, audio)
 *     + litert_lm_engine_settings_set_enable_speculative_decoding (MTP)
 *   - litert_lm_engine_create / litert_lm_engine_delete
 *   - litert_lm_engine_create_session / litert_lm_session_delete
 *   - litert_lm_session_run_prefill(session, LiteRtLmInputData[], n)
 *   - litert_lm_session_run_decode_async(session, LiteRtLmStreamCallback, data)
 *     (one chunk per callback; mapped onto the FFI pull contract via a queue)
 *   - litert_lm_session_cancel_process
 *   - litert_lm_engine_tokenize / litert_lm_engine_detokenize (id↔text
 *     round-trip against the .litertlm's own tokenizer)
 *
 * Accelerator ladder (Android NPU first): the factory tries the "npu", then
 * "gpu", then "cpu" backend_str at `open()` (engine_create returns null when a
 * rung's delegate is unavailable) and records which one initialized. Every
 * hardware-gated assumption is tagged `DEVICE-VERIFY` in the .cpp — the
 * accelerator ladder, the .litertlm graph fit, and tok/s can only be confirmed
 * on a real NPU device. The C-API linkage itself is proven by
 * `litert-capi-smoke.cpp` (loads the .litertlm, prefills, decodes "Paris").
 */

#include "../llm-backend.h"

/* Stable id matched case-insensitively against ELIZA_LLM_BACKEND, and the
 * subdir + artifact extension the factory probes under <bundle_dir>/text/. */
#define LITERT_BACKEND_NAME "litert-lm"
#define LITERT_BUNDLE_TEXT_SUBDIR "text"
#define LITERT_ARTIFACT_EXT ".litertlm"

/* Singleton factory accessor. The selector (llm-backend-selector.cpp) calls
 * this from `llm_backend_register_builtins()` to register the backend. The
 * returned pointer is a static-lifetime singleton the registry does not own.
 * Defined unconditionally — a build without ELIZA_ENABLE_LITERT returns a
 * stub factory whose available() is false. */
LlmBackendFactory * litert_backend_factory();
