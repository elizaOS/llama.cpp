#pragma once
/*
 * litert-asr.h — LiteRT-LM in-process ASR (speech-to-text) backend.
 *
 * Transcribes 16 kHz mono fp32 PCM through the Gemma USM audio encoder inside a
 * .litertlm bundle, via the LiteRT-LM STABLE C API (`litert_lm_*`, engine.h).
 * This is the audio sibling of litert-backend.{h,cpp} (the LLM text path): both
 * link the prebuilt liblitert-lm.so by its ABI-stable C surface only — NO
 * `litert::lm::` C++ symbols — so the fused libelizainference carries no libc++
 * C++-ABI coupling.
 *
 * The whole real implementation is gated behind the CMake define
 * `ELIZA_ENABLE_LITERT`. When that flag is OFF this header pulls in NO LiteRT-LM
 * SDK headers, so the file compiles on a host without the SDK and the entry
 * point links in as a no-op: it returns ELIZA_ERR_NOT_IMPLEMENTED and sets
 * `*out_error` "not compiled in".
 *
 * ── Proven audio path (litert-lm CLI, --attachment <wav> --audio-backend cpu) ─
 * The CLI's `litert-lm run <model.litertlm> --attachment jfk.wav
 * --audio-backend cpu --prompt "Transcribe this audio."` produces a perfect
 * transcription. Underneath (litert_lm_cli/commands/run.py + litert_lm
 * engine.py/conversation.py) it executes exactly these C calls:
 *
 *   litert_lm_engine_settings_create(model, "cpu", NULL, "cpu")   // audio="cpu"
 *   litert_lm_engine_create(settings)
 *   litert_lm_conversation_config_create / set_session_config
 *   litert_lm_conversation_create(engine, conv_cfg)
 *   litert_lm_conversation_send_message(conv, msg_json, ctx_json, NULL)
 *     msg_json = {"role":"user","content":[
 *                   {"type":"audio","blob":"<base64 WAV>"},   // AudioBytes
 *                   {"type":"text","text":"Transcribe this audio."}]}
 *   litert_lm_json_response_get_string(resp)  -> {"content":[{"type":"text",
 *                                                 "text":"<transcript>"}],...}
 *
 * The CLI passes the audio as a file `path`; the litert_lm `AudioBytes` content
 * type (_messages.py) shows a raw-bytes `blob` (base64) is equally valid, which
 * is what this backend uses so the PCM-in entry needs no temp file.
 *
 * Error contract (native/AGENTS.md §3 + §9): never log, never return a defaulted
 * result on failure. Every failure path heap-allocates `*out_error` (mirroring
 * the FFI cpp's eliza_strdup/eliza_set_error style) and returns a negative
 * ELIZA_* code.
 */

#include "../../include/eliza-inference-ffi.h" /* EliInferenceContext, ELIZA_* */

#include <cstddef>

/* The fixed prompt the audio encoder path uses to elicit a transcription. Same
 * intent as the proven CLI `--prompt "Transcribe this audio."`. */
#define LITERT_ASR_TRANSCRIBE_PROMPT "Transcribe this audio."

/* Transcribe `n_samples` of 16 kHz mono fp32 PCM through a .litertlm bundle's
 * Gemma USM audio encoder using the LiteRT-LM C API.
 *
 * @param litertlm_path  Absolute path to the .litertlm model file.
 * @param pcm            16 kHz mono fp32 PCM samples (range [-1, 1]).
 * @param n_samples      Number of samples in `pcm`.
 * @param sample_rate_hz Input PCM sample rate (16000 expected; embedded in the
 *                       WAV header so the encoder resamples as needed).
 * @param out_text       Caller buffer; receives a UTF-8 NUL-terminated
 *                       transcript, up to `max_text_bytes - 1` bytes + NUL.
 * @param max_text_bytes Capacity of `out_text` in bytes.
 * @param out_error      On failure, heap-allocates a NUL-terminated message the
 *                       caller must free(). Untouched on success.
 *
 * @return Number of transcript bytes written (excluding the terminator) on
 *         success, or a negative ELIZA_* code on failure. Matches the contract
 *         of `eliza_inference_asr_transcribe` so this slots in behind the same
 *         FFI symbol.
 */
int litert_asr_transcribe(const char * litertlm_path, const float * pcm,
                          size_t n_samples, int sample_rate_hz, char * out_text,
                          size_t max_text_bytes, char ** out_error);

/* ── Warm reusable engine handle ──────────────────────────────────────────── *
 *
 * `litert_asr_transcribe` above builds + tears down the whole LiteRT-LM engine
 * (model + USM audio encoder load) per call, which is fine for the one-shot
 * smoke harness but far too costly for the runtime ASR FFI, where the same
 * .litertlm bundle transcribes many utterances. The engine handle below lets the
 * caller open the engine ONCE and reuse it across transcriptions:
 *
 *   LitertAsrEngine * eng = litert_asr_engine_open(litertlm_path, &err);
 *   litert_asr_engine_transcribe(eng, pcm, n, sr, out, cap, &err);   // repeat
 *   litert_asr_engine_close(eng);
 *
 * Each transcribe opens + tears down only a conversation on the warm engine, so
 * the expensive model/encoder load happens once. The handle is opaque; its
 * concrete type is defined only in the real (ELIZA_ENABLE_LITERT) translation
 * unit. The stub build returns nullptr from open() and sets `*out_error`.
 *
 * Thread-safety: a handle is NOT internally synchronized — the caller serializes
 * transcribe/close (the FFI already holds ctx->asr_mutex around the ASR path).
 */
struct LitertAsrEngine;

/* Open a warm engine for `litertlm_path` (loads the model + USM audio encoder).
 * Returns an owned handle to free with litert_asr_engine_close, or nullptr +
 * heap-allocated `*out_error` on failure / in the stub build. */
LitertAsrEngine * litert_asr_engine_open(const char * litertlm_path,
                                         char ** out_error);

/* Transcribe `n_samples` of 16 kHz mono fp32 PCM on a warm engine. Same
 * argument + return contract as litert_asr_transcribe (bytes written, or a
 * negative ELIZA_* code with heap-allocated `*out_error`). */
int litert_asr_engine_transcribe(LitertAsrEngine * engine, const float * pcm,
                                 size_t n_samples, int sample_rate_hz,
                                 char * out_text, size_t max_text_bytes,
                                 char ** out_error);

/* Free a handle returned by litert_asr_engine_open. NULL-safe. */
void litert_asr_engine_close(LitertAsrEngine * engine);

/* Capability probe: 1 when this build was compiled with ELIZA_ENABLE_LITERT
 * (the LiteRT-LM SDK is linked in), 0 for the stub build. Mirrors the
 * `eliza_inference_asr_stream_supported()` probe style so the loader can choose
 * the LiteRT audio path without calling it and catching
 * ELIZA_ERR_NOT_IMPLEMENTED. */
int litert_asr_supported(void);
