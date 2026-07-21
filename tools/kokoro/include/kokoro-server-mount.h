// SPDX-License-Identifier: MIT
//
// kokoro-server-mount.h — committed declarations for the Kokoro
// `/v1/audio/speech` HTTP route mounted on `llama-server`.
//
// The route accepts the OpenAI-compatible request body:
//
//   POST /v1/audio/speech
//   {
//     "model":           "kokoro-v1.0",         // ignored, single model per server
//     "input":           "Hello world.",
//     "voice":           "af_sam",                // resolves to voices/af_sam.bin
//     "response_format": "wav" | "pcm",
//     "speed":           1.0                       // optional duration multiplier
//   }
//
// `server.cpp` includes this header inside `#ifdef LLAMA_BUILD_KOKORO` and:
//   1. writes the `--kokoro-model` / `--kokoro-voices-dir` CLI overrides
//      to `eliza_kokoro::g_model_path` / `g_voices_dir` before
//      `common_params_parse`;
//   2. registers the handler returned by `audio_speech_handler()`. When
//      `--kokoro-model` is set, the Kokoro handler takes precedence over
//      OmniVoice for `/v1/audio/speech`.

#pragma once

#ifdef LLAMA_BUILD_KOKORO

#include "server-http.h"

#include <string>

namespace eliza_kokoro {

// CLI overrides captured by `server.cpp` main() before `common_params_parse`.
extern std::string g_model_path;     // path to kokoro-v1.0.gguf
extern std::string g_voices_dir;     // path to voices/<id>.bin directory

// Returns a handler suitable for `server_http_context::post()` registration
// against `POST /v1/audio/speech` (and `POST /audio/speech`).
//
// On the first request the handler lazily loads the GGUF at g_model_path
// and serializes synthesis behind an internal mutex (kokoro_synthesize is
// non-reentrant on one model context).
server_http_context::handler_t audio_speech_handler();

// True when --kokoro-model was passed and the GGUF can be loaded.
bool is_enabled() noexcept;

} // namespace eliza_kokoro

#endif // LLAMA_BUILD_KOKORO
