/*
 * litert-asr.cpp — LiteRT-LM in-process ASR (speech-to-text) backend.
 *
 * The real implementation (gated behind `ELIZA_ENABLE_LITERT`) targets the
 * LiteRT-LM STABLE C API (`litert_lm_*`, engine.h) — NOT the C++ runtime. The C
 * surface is ABI-stable, so the fused libelizainference links the prebuilt
 * liblitert-lm.so with no libc++ C++-ABI coupling. See litert-asr.h for the
 * targeted symbols and the proven CLI audio path it mirrors.
 *
 * The default (Linux/desktop CI) build compiles the stub branch, which links
 * zero LiteRT-LM SDK headers and returns ELIZA_ERR_NOT_IMPLEMENTED so the loader
 * keeps the fused Qwen3-ASR path.
 *
 * Error contract (native/AGENTS.md §3 + §9): never log, never return a defaulted
 * result on failure. Every failure path heap-allocates `*out_error` via
 * litert_asr_set_error() (matching the FFI cpp's eliza_strdup/eliza_set_error
 * style) and returns the negative ELIZA_* code.
 */

#include "litert-asr.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

/* ── Heap-allocated error strings (mirror eliza-inference-ffi.cpp) ───────── */
namespace {

char * litert_asr_strdup(const std::string & s) {
    char * out = static_cast<char *>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

void litert_asr_set_error(char ** out_error, const std::string & msg) {
    if (!out_error) return;
    *out_error = litert_asr_strdup(msg);
}

}  // namespace

/* ════════════════════════════════════════════════════════════════════════ *
 *  REAL implementation — only when ELIZA_ENABLE_LITERT is defined.
 *  Behind this gate we may include LiteRT-LM SDK headers; outside it we
 *  include NONE so the file builds on a host without the SDK.
 * ════════════════════════════════════════════════════════════════════════ */
#ifdef ELIZA_ENABLE_LITERT

#include <memory>
#include <vector>

/* LiteRT-LM STABLE C API (engine.h). C API only — no `litert::lm::` C++
 * symbols — so the fused libelizainference carries no libc++ C++-ABI dependency
 * on the prebuilt liblitert-lm.so. */
#include "engine.h"

namespace {

/* ── RAII for the C handles (each lives only inside litert_asr_transcribe) ── */
struct SettingsHandle {
    LiteRtLmEngineSettings * p = nullptr;
    ~SettingsHandle() { if (p) litert_lm_engine_settings_delete(p); }
};
struct EngineHandle {
    LiteRtLmEngine * p = nullptr;
    ~EngineHandle() { if (p) litert_lm_engine_delete(p); }
};
struct SessionConfigHandle {
    LiteRtLmSessionConfig * p = nullptr;
    ~SessionConfigHandle() { if (p) litert_lm_session_config_delete(p); }
};
struct ConvConfigHandle {
    LiteRtLmConversationConfig * p = nullptr;
    ~ConvConfigHandle() { if (p) litert_lm_conversation_config_delete(p); }
};
struct ConversationHandle {
    LiteRtLmConversation * p = nullptr;
    ~ConversationHandle() { if (p) litert_lm_conversation_delete(p); }
};
struct JsonResponseHandle {
    LiteRtLmJsonResponse * p = nullptr;
    ~JsonResponseHandle() { if (p) litert_lm_json_response_delete(p); }
};

/* ── Base64 (RFC 4648) — encode the WAV bytes for the JSON `blob` field ───── */
std::string base64_encode(const uint8_t * data, size_t len) {
    static const char tbl[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((len + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < len; i += 3) {
        const uint32_t n = (uint32_t(data[i]) << 16) |
                           (uint32_t(data[i + 1]) << 8) | uint32_t(data[i + 2]);
        out.push_back(tbl[(n >> 18) & 0x3F]);
        out.push_back(tbl[(n >> 12) & 0x3F]);
        out.push_back(tbl[(n >> 6) & 0x3F]);
        out.push_back(tbl[n & 0x3F]);
    }
    if (i < len) {
        uint32_t n = uint32_t(data[i]) << 16;
        const bool two = (i + 1 < len);
        if (two) n |= uint32_t(data[i + 1]) << 8;
        out.push_back(tbl[(n >> 18) & 0x3F]);
        out.push_back(tbl[(n >> 12) & 0x3F]);
        out.push_back(two ? tbl[(n >> 6) & 0x3F] : '=');
        out.push_back('=');
    }
    return out;
}

/* ── WAV (RIFF/PCM16) builder — fp32 mono PCM → 16-bit little-endian WAV ──── *
 * The LiteRT-LM audio decoder reads the sample rate from the WAV header, so
 * encoding the exact input rate lets the Gemma USM encoder resample internally
 * (the proven CLI feeds a 16 kHz mono PCM16 jfk.wav). */
void put_u32_le(std::vector<uint8_t> & v, uint32_t x) {
    v.push_back(uint8_t(x & 0xFF));
    v.push_back(uint8_t((x >> 8) & 0xFF));
    v.push_back(uint8_t((x >> 16) & 0xFF));
    v.push_back(uint8_t((x >> 24) & 0xFF));
}
void put_u16_le(std::vector<uint8_t> & v, uint16_t x) {
    v.push_back(uint8_t(x & 0xFF));
    v.push_back(uint8_t((x >> 8) & 0xFF));
}
void put_str(std::vector<uint8_t> & v, const char * s) {
    for (; *s; ++s) v.push_back(uint8_t(*s));
}

std::vector<uint8_t> build_wav_pcm16(const float * pcm, size_t n_samples,
                                     int sample_rate_hz) {
    const uint16_t channels = 1;
    const uint16_t bits = 16;
    const uint32_t byte_rate =
        uint32_t(sample_rate_hz) * channels * (bits / 8);
    const uint16_t block_align = channels * (bits / 8);
    const uint32_t data_bytes = uint32_t(n_samples) * (bits / 8);

    std::vector<uint8_t> w;
    w.reserve(44 + data_bytes);
    put_str(w, "RIFF");
    put_u32_le(w, 36 + data_bytes);  /* chunk size = file size - 8 */
    put_str(w, "WAVE");
    put_str(w, "fmt ");
    put_u32_le(w, 16);               /* PCM fmt chunk size */
    put_u16_le(w, 1);                /* audio format = PCM */
    put_u16_le(w, channels);
    put_u32_le(w, uint32_t(sample_rate_hz));
    put_u32_le(w, byte_rate);
    put_u16_le(w, block_align);
    put_u16_le(w, bits);
    put_str(w, "data");
    put_u32_le(w, data_bytes);
    for (size_t i = 0; i < n_samples; ++i) {
        float s = pcm[i];
        if (s > 1.0f) s = 1.0f;
        else if (s < -1.0f) s = -1.0f;
        const int32_t q = int32_t(s * 32767.0f);
        put_u16_le(w, uint16_t(int16_t(q)));
    }
    return w;
}

/* ── Minimal JSON string-literal escaping (for the base64 blob, which is
 * already JSON-safe, and the fixed prompt — defensive against control chars) ── */
std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    static const char hex[] = "0123456789abcdef";
                    out += "\\u00";
                    out.push_back(hex[(c >> 4) & 0xF]);
                    out.push_back(hex[c & 0xF]);
                } else {
                    out.push_back(c);
                }
        }
    }
    return out;
}

/* ── Extract the transcript from the conversation JSON response ───────────── *
 * The response shape (litert_lm conversation.py) is
 *   {"role":"assistant","content":[{"type":"text","text":"<transcript>"}],...}
 * We pull the concatenation of every content[].text. The library owns the
 * tokenizer, so the text arrives already-detokenized UTF-8; we only locate the
 * "text" fields and unescape the JSON string. No general JSON parser is needed
 * for this fixed, library-emitted shape. */
std::string unescape_json_string(const std::string & in, size_t start,
                                 size_t & end_out) {
    /* `start` points at the opening quote of a JSON string. Returns the decoded
     * content and sets end_out to the index just past the closing quote. */
    std::string out;
    size_t i = start + 1;
    for (; i < in.size(); ++i) {
        char c = in[i];
        if (c == '\\' && i + 1 < in.size()) {
            char n = in[++i];
            switch (n) {
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case '/': out.push_back('/');  break;
                case '"': out.push_back('"');  break;
                case '\\': out.push_back('\\'); break;
                case 'u': {
                    if (i + 4 < in.size()) {
                        unsigned code = 0;
                        bool ok = true;
                        for (int k = 1; k <= 4; ++k) {
                            char h = in[i + k];
                            code <<= 4;
                            if (h >= '0' && h <= '9') code |= unsigned(h - '0');
                            else if (h >= 'a' && h <= 'f') code |= unsigned(h - 'a' + 10);
                            else if (h >= 'A' && h <= 'F') code |= unsigned(h - 'A' + 10);
                            else { ok = false; break; }
                        }
                        if (ok) {
                            i += 4;
                            /* Encode the BMP code point as UTF-8. */
                            if (code < 0x80) {
                                out.push_back(char(code));
                            } else if (code < 0x800) {
                                out.push_back(char(0xC0 | (code >> 6)));
                                out.push_back(char(0x80 | (code & 0x3F)));
                            } else {
                                out.push_back(char(0xE0 | (code >> 12)));
                                out.push_back(char(0x80 | ((code >> 6) & 0x3F)));
                                out.push_back(char(0x80 | (code & 0x3F)));
                            }
                        } else {
                            out.push_back('u');
                        }
                    }
                    break;
                }
                default: out.push_back(n); break;
            }
        } else if (c == '"') {
            end_out = i + 1;
            return out;
        } else {
            out.push_back(c);
        }
    }
    end_out = i;
    return out;
}

std::string extract_transcript(const std::string & json) {
    /* Find each `"text"` key followed by `:` and a string value, and
     * concatenate. The library emits only the transcript text in these fields
     * for a transcription turn. */
    std::string out;
    size_t pos = 0;
    const std::string key = "\"text\"";
    while ((pos = json.find(key, pos)) != std::string::npos) {
        size_t i = pos + key.size();
        while (i < json.size() && (json[i] == ' ' || json[i] == ':' ||
                                   json[i] == '\t' || json[i] == '\n'))
            ++i;
        if (i < json.size() && json[i] == '"') {
            size_t end = 0;
            out += unescape_json_string(json, i, end);
            pos = end;
        } else {
            pos = i;
        }
    }
    return out;
}

}  // namespace

int litert_asr_transcribe(const char * litertlm_path, const float * pcm,
                          size_t n_samples, int sample_rate_hz, char * out_text,
                          size_t max_text_bytes, char ** out_error) {
    if (!litertlm_path || litertlm_path[0] == '\0') {
        litert_asr_set_error(out_error,
            "[litert-asr] transcribe: litertlm_path is NULL/empty");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (!pcm || n_samples == 0) {
        litert_asr_set_error(out_error,
            "[litert-asr] transcribe: pcm is NULL or n_samples == 0");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (!out_text || max_text_bytes == 0) {
        litert_asr_set_error(out_error,
            "[litert-asr] transcribe: out_text buffer is NULL/zero");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (sample_rate_hz <= 0) {
        litert_asr_set_error(out_error,
            "[litert-asr] transcribe: sample_rate_hz must be positive");
        return ELIZA_ERR_INVALID_ARG;
    }

    /* Quiet the library except for true errors (0=VERBOSE..4=ERROR..5=FATAL). */
    litert_lm_set_min_log_level(4 /* ERROR */);

    /* 1. Engine settings — CPU text + CPU audio backend. The audio backend is
     *    REQUIRED for an audio attachment (the CLI errors out without it). */
    SettingsHandle settings;
    settings.p = litert_lm_engine_settings_create(
        litertlm_path, /*backend=*/"cpu", /*vision_backend=*/nullptr,
        /*audio_backend=*/"cpu");
    if (!settings.p) {
        litert_asr_set_error(out_error,
            std::string("[litert-asr] engine_settings_create failed for ") +
            litertlm_path);
        return ELIZA_ERR_BUNDLE_INVALID;
    }

    /* 2. Build the engine (loads the model + USM audio encoder). */
    EngineHandle engine;
    engine.p = litert_lm_engine_create(settings.p);
    if (!engine.p) {
        litert_asr_set_error(out_error,
            "[litert-asr] engine_create failed (no CPU audio delegate or "
            "incompatible .litertlm)");
        return ELIZA_ERR_FFI_FAULT;
    }

    /* 3. Conversation config (default session). The conversation API applies the
     *    model's chat template and handles the multimodal content blocks — the
     *    same path the proven CLI uses for --attachment audio. */
    ConvConfigHandle conv_cfg;
    conv_cfg.p = litert_lm_conversation_config_create();
    if (!conv_cfg.p) {
        litert_asr_set_error(out_error,
            "[litert-asr] conversation_config_create failed");
        return ELIZA_ERR_OOM;
    }
    {
        SessionConfigHandle session_cfg;
        session_cfg.p = litert_lm_session_config_create();
        if (session_cfg.p) {
            litert_lm_conversation_config_set_session_config(conv_cfg.p,
                                                             session_cfg.p);
        }
    }

    ConversationHandle conv;
    conv.p = litert_lm_conversation_create(engine.p, conv_cfg.p);
    if (!conv.p) {
        litert_asr_set_error(out_error,
            "[litert-asr] conversation_create failed");
        return ELIZA_ERR_FFI_FAULT;
    }

    /* 4. Build the multimodal user message: audio blob (WAV) + transcribe text.
     *    Mirrors litert_lm AudioBytes ({"type":"audio","blob":"<base64>"}). */
    const std::vector<uint8_t> wav =
        build_wav_pcm16(pcm, n_samples, sample_rate_hz);
    const std::string blob = base64_encode(wav.data(), wav.size());

    std::string msg_json;
    msg_json.reserve(blob.size() + 128);
    msg_json += "{\"role\":\"user\",\"content\":[";
    msg_json += "{\"type\":\"audio\",\"blob\":\"";
    msg_json += blob;  /* base64 is already JSON-safe */
    msg_json += "\"},";
    msg_json += "{\"type\":\"text\",\"text\":\"";
    msg_json += json_escape(LITERT_ASR_TRANSCRIBE_PROMPT);
    msg_json += "\"}]}";

    /* 5. Send (blocking) and read the JSON response. */
    JsonResponseHandle resp;
    resp.p = litert_lm_conversation_send_message(
        conv.p, msg_json.c_str(), /*extra_context=*/"{}",
        /*optional_args=*/nullptr);
    if (!resp.p) {
        litert_asr_set_error(out_error,
            "[litert-asr] conversation_send_message failed (audio decode/"
            "transcription error)");
        return ELIZA_ERR_FFI_FAULT;
    }

    const char * resp_str = litert_lm_json_response_get_string(resp.p);
    if (!resp_str) {
        litert_asr_set_error(out_error,
            "[litert-asr] json_response_get_string returned NULL");
        return ELIZA_ERR_FFI_FAULT;
    }

    const std::string transcript = extract_transcript(resp_str);
    if (transcript.empty()) {
        litert_asr_set_error(out_error,
            std::string("[litert-asr] no transcript text in response: ") +
            resp_str);
        return ELIZA_ERR_FFI_FAULT;
    }

    /* 6. Copy out, NUL-terminate, and report bytes written (matching the
     *    eliza_inference_asr_transcribe contract). */
    const size_t copy = transcript.size() < max_text_bytes - 1
                            ? transcript.size()
                            : max_text_bytes - 1;
    std::memcpy(out_text, transcript.data(), copy);
    out_text[copy] = '\0';
    return static_cast<int>(copy);
}

int litert_asr_supported(void) { return 1; }

#else  /* ────────────────────────── STUB (no LiteRT-LM SDK) ──────────────── */

/*
 * Compiled-out stub: zero LiteRT-LM headers, so this builds on any host. The
 * entry point returns ELIZA_ERR_NOT_IMPLEMENTED + sets `*out_error`
 * "not compiled in" so the loader cleanly keeps the fused Qwen3-ASR path.
 */
int litert_asr_transcribe(const char * /*litertlm_path*/, const float * /*pcm*/,
                          size_t /*n_samples*/, int /*sample_rate_hz*/,
                          char * /*out_text*/, size_t /*max_text_bytes*/,
                          char ** out_error) {
    litert_asr_set_error(out_error,
        "[litert-asr] backend not compiled in "
        "(build with -DELIZA_ENABLE_LITERT to enable the LiteRT-LM ASR path)");
    return ELIZA_ERR_NOT_IMPLEMENTED;
}

int litert_asr_supported(void) { return 0; }

#endif  /* ELIZA_ENABLE_LITERT */
