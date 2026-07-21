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
 * ── Targeted runtime API (researched 2026-06-22) ──────────────────────────
 * Repo:    https://github.com/google-ai-edge/LiteRT-LM  (`main`)
 * Docs:    https://developers.google.com/edge/litert-lm/cpp
 *          https://ai.google.dev/edge/litert/next/litert_lm_npu
 * Namespace: `litert::lm`
 *
 * Symbols this backend targets (verbatim from the headers above):
 *   - runtime/engine/engine.h
 *       using Engine = EngineT<SessionInterface>;
 *       static absl::StatusOr<std::unique_ptr<Engine>>
 *           Engine::CreateEngine(const EngineSettings&);
 *       absl::StatusOr<std::unique_ptr<SessionT>>
 *           EngineT::CreateSession(const SessionConfig&);
 *   - runtime/engine/engine.h  (SessionInterface)
 *       absl::Status        RunPrefill(const std::vector<InputData>&);
 *       absl::StatusOr<Responses> RunDecode();
 *       absl::StatusOr<Responses> RunDecode(const DecodeConfig&);
 *       absl::Status        GenerateContentStream(
 *                               const std::vector<InputData>&,
 *                               absl::AnyInvocable<void(absl::StatusOr<Responses>)>);
 *   - runtime/engine/engine_settings.h
 *       static absl::StatusOr<EngineSettings> EngineSettings::CreateDefault(
 *           ModelAssets, Backend backend = Backend::CPU,
 *           std::optional<Backend> vision_backend  = std::nullopt,
 *           std::optional<Backend> audio_backend   = std::nullopt,
 *           std::optional<Backend> sampler_backend = std::nullopt);
 *       static SessionConfig SessionConfig::CreateDefault();
 *       absl::StatusOr<ModelAssets> ModelAssets::Create(<path>);   // .litertlm
 *   - runtime/engine/io_types.h
 *       using InputData = std::variant<InputText, InputImage, InputAudio, ...>;
 *       class InputText { explicit InputText(std::variant<std::string, TensorBuffer>); };
 *       class Responses  { const std::vector<std::string>& GetTexts() const; };
 *   - runtime/proto/engine.pb.h
 *       enum Backend { ... CPU, GPU, NPU, ... };   // litert::lm::Backend
 *
 * Accelerator ladder (Android NPU first): the factory tries NPU, then GPU,
 * then CPU at `open()` and records which one initialized. Every
 * hardware-gated assumption is tagged `DEVICE-VERIFY` in the .cpp — the
 * accelerator ladder, the .litertlm graph fit, and tok/s can only be
 * confirmed on a real NPU device, which this scaffold does not have.
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
