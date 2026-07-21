// SPDX-License-Identifier: MIT
//
// server-mount.h — committed declarations for the OmniVoice `/v1/audio/speech`
// HTTP route mounted on `llama-server`. The implementation in
// `tools/omnivoice/src/server-mount.cpp` carries the preset-aware request
// parser + handler that used to live in `tools/server/server.cpp` directly
// (and, before W3-3, was injected by
// `packages/app-core/scripts/kernel-patches/server-omnivoice-route.mjs`).
//
// `server.cpp` includes this header inside `#ifdef ELIZA_FUSE_OMNIVOICE`
// and:
//   1. writes the `--omnivoice-model` / `--omnivoice-codec` CLI overrides
//      to `eliza_omnivoice::g_model_path` / `g_codec_path` before
//      `common_params_parse`;
//   2. registers the handler returned by `audio_speech_handler()` against
//      `POST /v1/audio/speech` and `POST /audio/speech`.
//
// This header carries no external types beyond `server_http_context` so it
// stays cheap to include.

#pragma once

#ifdef ELIZA_FUSE_OMNIVOICE

#include "server-http.h"

#include <string>

namespace eliza_omnivoice {

// CLI overrides captured by `server.cpp` main() before `common_params_parse`.
// Promoted to extern so the implementation TU owns the storage.
extern std::string g_model_path;
extern std::string g_codec_path;

// Returns a handler suitable for `server_http_context::post()` registration
// against `POST /v1/audio/speech`.
server_http_context::handler_t audio_speech_handler();

} // namespace eliza_omnivoice

#endif // ELIZA_FUSE_OMNIVOICE
