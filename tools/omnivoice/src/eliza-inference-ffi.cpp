// eliza-inference-ffi.cpp — C ABI bridge for libelizainference.
//
// Vendored into the merged llama.cpp fork by W3-3 (OmniVoice → llama.cpp
// fork literal merge). Before W3-3 this file was generated at build time
// by `packages/app-core/scripts/omnivoice-fuse/prepare.mjs`. The generator
// stays available for one release as a deprecated fallback path; the
// canonical home for the FFI ABI implementation is HERE, inside the fork,
// alongside the rest of the merged OmniVoice tree.
//
// LLAMA_BUILD_OMNIVOICE=ON pulls this file into `libelizainference` via
// `tools/omnivoice/CMakeLists.txt`. The ABI is declared in
// `include/eliza-inference-ffi.h`; the JS / Bun / mobile FFI loaders
// (see `plugins/plugin-local-inference/src/services/voice/ffi-bindings.ts`)
// resolve `eliza_inference_*` symbols from this object.

#include "eliza-inference-ffi.h"
#include "llm-backend.h"
#include "embed-backend.h"
#include "vision-backend.h"
#include "asr-backend.h"
#include "tts-backend.h"
#include "eot-backend.h"
#include "omnivoice.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

// ABI guard: the TS loader (ffi-llm-streaming-abi.ts) marshals
// eliza_llm_stream_config_t by hand-written field offsets, so any reorder /
// insert / type change on the C side silently corrupts every streaming-LLM
// call. Pin the on-the-wire layout (documented "sizeof config = 80" since v8):
// 6×int32 + 5×ptr + 4-byte fields packed to 80 bytes on a 64-bit ABI. Adding a
// field is an ABI bump — update this assert AND the TS marshaller together.
static_assert(
    sizeof(eliza_llm_stream_config_t) == 80,
    "eliza_llm_stream_config_t layout changed — bump ABI + update the TS "
    "marshaller in ffi-llm-streaming-abi.ts, then update this assert.");

/* common/ — the same-file MTP speculative-decode engine wired into the
 * streaming-LLM text path (ABI v8) reuses the DRAFT_MTP implementation in
 * common/speculative.cpp, the host sampler in common/sampling.cpp, and the
 * batch helpers in common/common.cpp. These are the SAME entry points the
 * libllama+eliza-llama-shim desktop path drives, so the fused text path
 * matches it exactly. NOTE: do not pull in httplib.h here — networking stays
 * out of the in-process FFI path. */
#include "common.h"
#include "speculative.h"
#include "sampling.h"

/* Vendored standalone scalar-C voice-classifier forward graphs (no ggml
 * dependency; their own GGUF reader). These back the fused wake-word,
 * speaker-encoder, diarizer, and Silero VAD ABIs so the whole voice
 * pipeline runs through one libelizainference handle instead of separate
 * .so libs. */
extern "C" {
#include "voice-classifiers/wakeword/include/wakeword/wakeword.h"
#include "voice-classifiers/voice_classifier/include/voice_classifier/voice_classifier.h"
#include "voice-classifiers/vad/include/silero_vad/silero_vad.h"
}

#ifdef ELIZA_ENABLE_KOKORO
/* Kokoro-82M TTS (ABI v10) — folded in-process. kokoro_lib is a C++ API
 * (namespace eliza_kokoro) with its own GGUF reader + iSTFT decoder, so it
 * is included OUTSIDE the extern "C" block. The include dir is added by the
 * omnivoice CMakeLists when TARGET kokoro_lib exists. */
#include "kokoro.h"
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
/* MSVC has no POSIX setenv/unsetenv. The only use below pins/restores
 * GGML_BACKEND around ov_init (overwrite is always 1); map both onto
 * _putenv_s (from <cstdlib>), which removes the variable when value is "". */
static inline int setenv(const char *name, const char *value, int overwrite) {
  if (!overwrite && std::getenv(name) != nullptr) {
    return 0;
  }
  return _putenv_s(name, value);
}
static inline int unsetenv(const char *name) { return _putenv_s(name, ""); }
#endif

/* OmniVoice voice-preset payload parsed from
 * <bundle_dir>/cache/voice-preset-<id>.bin (ELZ2 v2 binary format). v1
 * presets (Kokoro-style: embedding + phrase-cache seed only) parse with
 * empty ref_audio_tokens / ref_text / instruct fields so the C side
 * falls back to OmniVoice's auto-voice path on those entries.
 *
 * The format is described in TypeScript at
 * plugins/plugin-local-inference/src/services/voice/voice-preset-format.ts.
 * This C-side parser is a slim, read-only mirror of that contract — it
 * understands both v1 and v2, returns false on any structural error,
 * and never copies large buffers (ref_audio_tokens is a 24 KB span at
 * most for a 30 s reference). */
struct EliVoicePreset {
    std::string voice_id;
    std::string instruct;
    std::string ref_text;
    std::vector<int32_t> ref_audio_tokens;
    int K = 0;
    int ref_T = 0;
    int version = 0;
    bool empty_payload = true;
};

struct EliInferenceContext {
    std::string bundle_dir;
    std::string tts_model_path;
    std::string codec_model_path;
    std::string asr_model_path;
    std::string asr_mmproj_path;
    ov_context * ov = nullptr;
    llama_model * asr_model = nullptr;
    llama_context * asr_lctx = nullptr;
    mtmd_context * asr_mtmd = nullptr;
    llama_sampler * asr_sampler = nullptr;
    int asr_sample_rate = 0;
    int asr_n_batch = 512;
    std::atomic<bool> tts_cancel{false};
    std::mutex tts_mutex;
    std::mutex asr_mutex;
    /* Streaming-LLM text model, loaded lazily on the first
     * eliza_inference_llm_stream_open and shared (read-only weights) across
     * sessions. Each session owns its own llama_context (private KV) over
     * this shared model. Protected by llm_mutex. */
    std::string llm_model_path;
    llama_model * llm_model = nullptr;
    std::mutex llm_mutex;
    /* Dedicated embedding context over the shared text model (ABI v9). Lazily
     * created on the first eliza_inference_embed call. Embeddings require
     * llama_set_embeddings(true) + a pooling type + a non-causal single-ubatch
     * layout, so this is a separate llama_context from the generation /
     * streaming-LLM path. Protected by llm_mutex (it shares the resident
     * text model). */
    llama_context * embed_ctx = nullptr;
    int embed_pooling = -1;   /* the pooling type embed_ctx was built with */
    int embed_n_ctx = 0;      /* the ctx size embed_ctx was built with */
    /* Dedicated CAUSAL scoring context over the shared text model (ABI v11).
     * Lazily created on the first eliza_inference_llm_eot_score call. EOT
     * scoring needs the causal next-token logit distribution at the final
     * position (the fused replacement for the retired node-llama-cpp
     * controlledEvaluate() the EOT classifiers used) — distinct from embed_ctx
     * (non-causal + pooled, exposes no per-token logits) and from the
     * per-session streaming-LLM KV. KV is cleared per call so each score is
     * independent. Protected by llm_mutex. */
    llama_context * eot_ctx = nullptr;
    int eot_n_ctx = 0;        /* the ctx size eot_ctx was built with */
    /* mmproj vision context over the shared text model (ABI v9), keyed by the
     * mmproj path it was initialized from. Lazily created on the first
     * eliza_inference_describe_image call per mmproj_path and reused.
     * Protected by llm_mutex. */
    mtmd_context * vision_mtmd = nullptr;
    std::string vision_mmproj_path;
    /* Parsed voice presets, keyed by voice id. Populated lazily on the
     * first TTS call that mentions the id. The mutex protects the map
     * itself; presets are immutable once inserted. */
    std::mutex preset_mutex;
    std::unordered_map<std::string, EliVoicePreset> presets;
#ifdef ELIZA_ENABLE_KOKORO
    /* Kokoro TTS (ABI v10): the loaded Kokoro model + voice preset, owned by
     * the ctx. Loaded via eliza_inference_kokoro_load, freed in destroy.
     * Protected by kokoro_mutex. */
    eliza_kokoro::kokoro_model_ptr kokoro_model;
    eliza_kokoro::kokoro_voice_preset kokoro_voice;
    bool kokoro_loaded = false;
    /* Paths of the currently-loaded Kokoro GGUF + voice preset, so a repeat
     * load with the same paths is a no-op (idempotent) instead of re-reading
     * the ~80MB GGUF every synth — the resident-host TTS streams many short
     * clauses per reply and must not reload per clause. */
    std::string kokoro_gguf_path;
    std::string kokoro_voice_path;
    std::mutex kokoro_mutex;
#endif
};

/* M3 seam accessor (declared in llm-backend.h): hand a backend's open() the
 * bundle root without exposing the struct. Defined here where the type is
 * complete. */
const char * llm_backend_context_bundle_dir(const EliInferenceContext * ctx) {
    return ctx ? ctx->bundle_dir.c_str() : nullptr;
}

/* ELZ2 magic 'ELZ1' (the ascii bytes 'E','L','Z','1' little-endian).
 * The magic stays 'ELZ1' across format versions — only the version
 * word at offset 4 changes between v1 and v2. */
static constexpr uint32_t ELIZA_VOICE_PRESET_MAGIC = 0x315A4C45u;

static inline uint32_t eliza_le_u32(const uint8_t * p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static inline int32_t eliza_le_i32(const uint8_t * p) {
    return (int32_t) eliza_le_u32(p);
}

/* Validated voice-id allowlist: a single path-safe segment (mirrors the
 * TypeScript voicePresetPath check). Refuses anything containing '/'
 * '..' or characters outside [A-Za-z0-9._-]. */
static bool eliza_is_safe_voice_id(const std::string & id) {
    if (id.empty()) return false;
    if (id.find("..") != std::string::npos) return false;
    for (char c : id) {
        const bool ok =
            (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}

/* Read a whole file into memory. Returns true on success. */
static bool eliza_read_file_bytes(const std::filesystem::path & path,
                                  std::vector<uint8_t> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) return false;
    f.seekg(0, std::ios::end);
    std::streamsize sz = f.tellg();
    if (sz < 0) return false;
    out.resize((size_t) sz);
    if (sz == 0) return true;
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char *>(out.data()), sz);
    return f.good() || f.eof();
}

/* Parse an ELZ2 voice-preset blob (also accepts v1 — v2-only fields
 * stay empty). Returns false on any structural error. */
static bool eliza_parse_voice_preset(const std::vector<uint8_t> & bytes,
                                     EliVoicePreset & out,
                                     std::string & err) {
    if (bytes.size() < 24) {
        err = "voice preset truncated: header < 24 bytes";
        return false;
    }
    const uint8_t * p = bytes.data();
    const size_t len = bytes.size();
    const uint32_t magic = eliza_le_u32(p);
    if (magic != ELIZA_VOICE_PRESET_MAGIC) {
        err = "voice preset bad magic";
        return false;
    }
    const uint32_t version = eliza_le_u32(p + 4);
    if (version != 1u && version != 2u) {
        err = "voice preset unsupported version";
        return false;
    }
    out.version = (int) version;

    /* v1 has a 24-byte header (embedding section + phrase section). v2
     * has a 64-byte header that ADDS ref_audio_tokens / ref_text /
     * instruct / metadata section descriptors. We only care about the
     * v2 sections — the embedding + phrase seed are handled in JS. */
    if (version == 1u) {
        /* v1: nothing the C side needs (embedding is fp32 for Kokoro
         * etc., not OmniVoice). Mark empty and return OK. */
        out.empty_payload = true;
        return true;
    }
    if (len < 64) {
        err = "voice preset v2 truncated header";
        return false;
    }
    auto section_at = [&](size_t hdr_off, uint32_t & off, uint32_t & sz) {
        off = eliza_le_u32(p + hdr_off);
        sz = eliza_le_u32(p + hdr_off + 4);
    };
    uint32_t ref_tok_off = 0, ref_tok_sz = 0;
    uint32_t ref_txt_off = 0, ref_txt_sz = 0;
    uint32_t instr_off = 0, instr_sz = 0;
    section_at(24, ref_tok_off, ref_tok_sz);
    section_at(32, ref_txt_off, ref_txt_sz);
    section_at(40, instr_off, instr_sz);
    /* metadata at +48/+52 — we don't consume it on the C side. */

    auto bounds_ok = [&](uint32_t off, uint32_t sz) {
        if (sz == 0) return true;
        if (off < 64) return false;
        const size_t end = (size_t) off + (size_t) sz;
        return end <= len;
    };
    if (!bounds_ok(ref_tok_off, ref_tok_sz) ||
        !bounds_ok(ref_txt_off, ref_txt_sz) ||
        !bounds_ok(instr_off, instr_sz)) {
        err = "voice preset section out of bounds";
        return false;
    }

    if (ref_tok_sz > 0) {
        if (ref_tok_sz < 8) {
            err = "voice preset ref_audio_tokens truncated";
            return false;
        }
        const uint8_t * rt = p + ref_tok_off;
        const uint32_t K = eliza_le_u32(rt);
        const uint32_t refT = eliza_le_u32(rt + 4);
        const size_t payload = ref_tok_sz - 8;
        if (payload != (size_t) K * (size_t) refT * 4u) {
            err = "voice preset ref_audio_tokens shape mismatch";
            return false;
        }
        out.K = (int) K;
        out.ref_T = (int) refT;
        out.ref_audio_tokens.resize((size_t) K * (size_t) refT);
        for (size_t i = 0; i < out.ref_audio_tokens.size(); ++i) {
            out.ref_audio_tokens[i] = eliza_le_i32(rt + 8 + i * 4u);
        }
    }
    if (ref_txt_sz > 0) {
        out.ref_text.assign(reinterpret_cast<const char *>(p + ref_txt_off),
                            (size_t) ref_txt_sz);
    }
    if (instr_sz > 0) {
        out.instruct.assign(reinterpret_cast<const char *>(p + instr_off),
                            (size_t) instr_sz);
    }
    out.empty_payload =
        out.instruct.empty() && out.ref_text.empty() && out.ref_audio_tokens.empty();
    return true;
}

/* Resolve a preset id to a parsed preset. Returns nullptr if the id is
 * unsafe, the file is missing, or parsing fails — in those cases the
 * caller falls back to OmniVoice auto-voice (params.instruct = "")
 * after logging via out_error.
 *
 * On a cache hit the cached preset is returned without re-reading the
 * file. The preset table is keyed by voice id and lives on the
 * EliInferenceContext, so two different contexts (different bundles)
 * load independent presets.
 *
 * Caller must hold ctx->preset_mutex (or otherwise serialize). */
static const EliVoicePreset * eliza_load_voice_preset_locked(
    EliInferenceContext * ctx,
    const std::string & voice_id,
    std::string & err) {
    auto it = ctx->presets.find(voice_id);
    if (it != ctx->presets.end()) return &it->second;

    if (!eliza_is_safe_voice_id(voice_id)) {
        err = "voice preset id is not a safe single segment: " + voice_id;
        return nullptr;
    }
    std::filesystem::path file =
        std::filesystem::path(ctx->bundle_dir) / "cache" /
        ("voice-preset-" + voice_id + ".bin");
    std::error_code ec;
    if (!std::filesystem::is_regular_file(file, ec)) {
        err = "voice preset file not found: " + file.string();
        return nullptr;
    }
    std::vector<uint8_t> bytes;
    if (!eliza_read_file_bytes(file, bytes)) {
        err = "voice preset file unreadable: " + file.string();
        return nullptr;
    }
    EliVoicePreset preset;
    preset.voice_id = voice_id;
    if (!eliza_parse_voice_preset(bytes, preset, err)) {
        return nullptr;
    }
    auto ins = ctx->presets.emplace(voice_id, std::move(preset));
    return &ins.first->second;
}

/* Apply a resolved preset to an ov_tts_params struct. Sets instruct,
 * ref_audio_tokens, ref_T, ref_text from the preset. Leaves
 * ref_audio_24k / ref_n_samples alone — they are only used by the
 * encode entrypoint, not synthesis. */
static void eliza_apply_preset_to_params(const EliVoicePreset & preset,
                                         ov_tts_params * params) {
    if (!preset.instruct.empty()) {
        params->instruct = preset.instruct.c_str();
    }
    if (!preset.ref_audio_tokens.empty() && preset.K > 0 && preset.ref_T > 0) {
        params->ref_audio_tokens = preset.ref_audio_tokens.data();
        params->ref_T = preset.ref_T;
    }
    if (!preset.ref_text.empty()) {
        params->ref_text = preset.ref_text.c_str();
    }
}

#define ELIZA_STRINGIFY_IMPL(x) #x
#define ELIZA_STRINGIFY(x) ELIZA_STRINGIFY_IMPL(x)

static char * eliza_strdup(const std::string & s) {
    char * out = (char *) std::malloc(s.size() + 1);
    if (!out) return nullptr;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

static void eliza_set_error(char ** out_error, const std::string & msg) {
    if (!out_error) return;
    *out_error = eliza_strdup(msg);
}

static bool eliza_is_region(const char * region_name) {
    return region_name &&
        (std::strcmp(region_name, "tts") == 0 ||
         std::strcmp(region_name, "asr") == 0 ||
         std::strcmp(region_name, "text") == 0 ||
         std::strcmp(region_name, "dflash") == 0 ||
         std::strcmp(region_name, "vad") == 0);
}

static std::vector<std::string> eliza_find_ggufs(const std::filesystem::path & dir) {
    std::vector<std::string> out;
    std::error_code ec;
    if (!std::filesystem::exists(dir, ec) || !std::filesystem::is_directory(dir, ec)) return out;
    for (const auto & entry : std::filesystem::recursive_directory_iterator(dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file(ec)) continue;
        const auto path = entry.path();
        if (path.extension() == ".gguf") out.push_back(path.string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

/* Pick the single GGUF under <bundle_dir>/<subdir>. Returns "" when none
 * found. Used by the VAD, speaker, and diarizer wrappers to resolve their
 * model from the context bundle. */
static std::string eliza_pick_one_gguf(
    const std::filesystem::path & bundle_dir,
    const char * subdir) {
    std::vector<std::string> ggufs = eliza_find_ggufs(bundle_dir / subdir);
    if (ggufs.empty()) return std::string();
    return ggufs[0];
}

static std::string eliza_lower_ascii(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return value;
}

// Kokoro ships its own GGUF under tts/kokoro/ and is loaded through the
// dedicated eliza_inference_kokoro_load path with an explicit file argument.
// eliza_find_ggufs recurses, so without this filter the OmniVoice picker below
// would pull tts/kokoro/kokoro-*.gguf into its candidate list and (sorting
// before "omnivoice-*") mis-select it as the OmniVoice LM — ov_init then aborts
// with "tensor 'llm.embed_tokens.weight' not found" on every tier that bundles
// both backends (0_8b/2b/4b/9b). Drop any candidate under a kokoro/ subdir.
static void eliza_drop_kokoro_ggufs(std::vector<std::string> & ggufs) {
    ggufs.erase(
        std::remove_if(ggufs.begin(), ggufs.end(), [](const std::string & p) {
            return p.find("/kokoro/") != std::string::npos;
        }),
        ggufs.end());
}

static bool eliza_pick_voice_files(
    const std::filesystem::path & bundle_dir,
    std::string & tts_model,
    std::string & codec_model) {
    std::vector<std::string> tts = eliza_find_ggufs(bundle_dir / "tts");
    std::vector<std::string> codec = eliza_find_ggufs(bundle_dir / "codec");
    eliza_drop_kokoro_ggufs(tts);
    eliza_drop_kokoro_ggufs(codec);
    if (tts.empty()) tts = eliza_find_ggufs(bundle_dir / "voice");
    if (tts.empty()) return false;
    tts_model = tts[0];
    if (!codec.empty()) {
        codec_model = codec[0];
    } else if (tts.size() > 1) {
        codec_model = tts[1];
    } else {
        codec_model = tts_model;
    }
    return true;
}

static bool eliza_pick_asr_files(
    const std::filesystem::path & bundle_dir,
    std::string & asr_model,
    std::string & asr_mmproj) {
    const auto asr_dir = bundle_dir / "asr";
    std::error_code ec;
    const auto canonical_model = asr_dir / "eliza-1-asr.gguf";
    const auto canonical_mmproj = asr_dir / "eliza-1-asr-mmproj.gguf";
    if (std::filesystem::is_regular_file(canonical_model, ec) &&
        std::filesystem::is_regular_file(canonical_mmproj, ec)) {
        asr_model = canonical_model.string();
        asr_mmproj = canonical_mmproj.string();
        return true;
    }

    std::vector<std::string> asr = eliza_find_ggufs(asr_dir);
    std::vector<std::string> model_candidates;
    std::vector<std::string> mmproj_candidates;
    for (const auto & candidate : asr) {
        const std::string filename = eliza_lower_ascii(std::filesystem::path(candidate).filename().string());
        if (filename.find("mmproj") != std::string::npos) {
            mmproj_candidates.push_back(candidate);
        } else if (filename.find("tokenizer") == std::string::npos &&
                   filename.find("support") == std::string::npos &&
                   filename.find("vocab") == std::string::npos) {
            model_candidates.push_back(candidate);
        }
    }
    if (asr_model.empty() && model_candidates.size() == 1) {
        asr_model = model_candidates[0];
    }
    if (asr_mmproj.empty() && mmproj_candidates.size() == 1) {
        asr_mmproj = mmproj_candidates[0];
    }
    return !asr_model.empty() && !asr_mmproj.empty();
}

static std::once_flag eliza_llama_backend_once;

static int eliza_thread_count(bool batch) {
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    const unsigned int cap = batch ? 8 : 4;
    return (int) std::max(1u, std::min(hw, cap));
}

static void eliza_apply_tts_env_overrides(ov_tts_params * params) {
    if (!params) return;
    if (const char * env = std::getenv("ELIZA_TTS_MASKGIT_STEPS")) {
        int n = std::atoi(env);
        if (n >= 1 && n <= 64) {
            params->mg_num_step = n;
        }
    }
    if (const char * env = std::getenv("ELIZA_TTS_CHUNK_DURATION_SEC")) {
        char * end = nullptr;
        float v = std::strtof(env, &end);
        if (end != env && v > 0.0f && v <= 120.0f) {
            params->chunk_duration_sec = v;
        }
    }
    if (const char * env = std::getenv("ELIZA_TTS_CHUNK_THRESHOLD_SEC")) {
        char * end = nullptr;
        float v = std::strtof(env, &end);
        if (end != env && v > 0.0f && v <= 120.0f) {
            params->chunk_threshold_sec = v;
        }
    }
}

/* ASR thread budget. The voice-realtime caps in eliza_thread_count() exist
 * so TTS + the DFlash drafter keep cores free during a streaming turn; the
 * ASR Whisper-style audio encoder (the mmproj prefill via
 * mtmd_helper_eval_chunks) and the short greedy text decode are *not*
 * barge-in-sensitive — they run once per utterance, before TTS competes —
 * so they can use more cores. Scale the encoder up to half the box (it is
 * the dominant cost on a ~30 s window) and keep the decode moderate. The
 * ELIZA_ASR_THREADS env var overrides both when set. */
static int eliza_asr_thread_count(bool encoder) {
    if (const char * env = std::getenv("ELIZA_ASR_THREADS")) {
        int n = std::atoi(env);
        if (n > 0) return n;
    }
    unsigned int hw = std::thread::hardware_concurrency();
    if (hw == 0) hw = 4;
    const unsigned int cap = encoder ? std::max(8u, hw / 2u) : std::max(4u, hw / 4u);
    return (int) std::max(1u, std::min(hw, cap));
}

static int eliza_int_env_or_default(const char * name, int fallback) {
    if (const char * env = std::getenv(name)) {
        int n = std::atoi(env);
        if (n > 0) return n;
    }
    return fallback;
}

static bool eliza_bool_env_or_default(const char * name, bool fallback) {
    if (const char * env = std::getenv(name)) {
        std::string value = env;
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return (char) std::tolower(c);
        });
        if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
        if (value == "0" || value == "false" || value == "no" || value == "off") return false;
    }
    return fallback;
}

static void eliza_asr_debug_log(const char * message) {
    if (!eliza_bool_env_or_default("ELIZA_ASR_DEBUG", false)) return;
    std::fprintf(stderr, "[libelizainference][asr] %s\n", message);
    std::fflush(stderr);
}

static bool eliza_running_on_android() {
#if defined(__ANDROID__)
    return true;
#else
    return std::getenv("ANDROID_ROOT") || std::getenv("ANDROID_DATA") || std::getenv("ANDROID_BOOTLOGO");
#endif
}

// Flash-attention selection for the text-LLM context. Flash-attn is ENABLED on
// all platforms (AUTO). The earlier Mali corruption — long prompts (~few-hundred+
// tokens) decoding into degenerate token repetition (" His!!!!") while short
// prompts stayed clean — was the Vulkan split-K FA reduce on ARM Mali, now fixed
// in ggml-vulkan by forcing single-chunk FA on that vendor (the VK_VENDOR_ID_ARM
// split_k gate in ggml_vk_flash_attn). FA itself is correct, so we keep it on
// (fused attention is far faster than the non-FA path for long prefills). The
// override remains for bisecting / pinning a setting per device:
//   ELIZA_LLM_FLASH_ATTN = off|0|false|disabled -> DISABLED
//                        = on|1|true|enabled     -> ENABLED
//                        = auto (default)         -> AUTO (llama.cpp decides)
static enum llama_flash_attn_type eliza_llm_flash_attn_type() {
    if (const char * env = std::getenv("ELIZA_LLM_FLASH_ATTN")) {
        std::string v = env;
        std::transform(v.begin(), v.end(), v.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        if (v == "off" || v == "0" || v == "false" || v == "no" || v == "disabled")
            return LLAMA_FLASH_ATTN_TYPE_DISABLED;
        if (v == "on" || v == "1" || v == "true" || v == "yes" || v == "enabled")
            return LLAMA_FLASH_ATTN_TYPE_ENABLED;
        if (v == "auto")
            return LLAMA_FLASH_ATTN_TYPE_AUTO;
    }
    // FA is DISABLED on Android. The Vulkan FA scalar kernel (flash_attn.comp) is
    // intermittently NON-DETERMINISTIC on Mali (~50-67% " His!!!!" under greedy
    // decode, device-verified) — a race in the kernel CORE that the ARM mitigations
    // (split_k=1 + disable_subgroups in ggml_vk_flash_attn / get_fa_tuning_params_scalar)
    // reduce the surface of but do NOT eliminate: it is NOT split-K and NOT the
    // subgroup reduction. FA-off is perf-NEUTRAL on Mali anyway (no cooperative-matrix
    // support → FA runs the scalar path at the same speed as the non-FA mul_mat path)
    // and is reliably correct (verified 6/6). The real on-device attention
    // optimization is the fused QJL/Polar kernel (elizaOS/eliza#8848), which bypasses
    // flash_attn.comp entirely. Override per-device with ELIZA_LLM_FLASH_ATTN=on.
    return eliza_running_on_android()
        ? LLAMA_FLASH_ATTN_TYPE_DISABLED
        : LLAMA_FLASH_ATTN_TYPE_AUTO;
}

static bool eliza_asr_android_cpu_profile() {
    if (eliza_running_on_android()) {
        return eliza_bool_env_or_default("ELIZA_ASR_ANDROID_CPU_PROFILE", true);
    }
    return false;
}

static bool eliza_asr_use_gpu() {
    return eliza_bool_env_or_default("ELIZA_ASR_USE_GPU", !eliza_asr_android_cpu_profile());
}

static bool eliza_asr_use_mmap() {
    return eliza_bool_env_or_default("ELIZA_ASR_USE_MMAP", !eliza_asr_android_cpu_profile());
}

static bool eliza_asr_use_extra_bufts() {
    return eliza_bool_env_or_default("ELIZA_ASR_USE_EXTRA_BUFTS", !eliza_asr_android_cpu_profile());
}

static int eliza_asr_context_size() {
    return eliza_int_env_or_default("ELIZA_ASR_N_CTX", eliza_asr_android_cpu_profile() ? 4096 : 8192);
}

static int eliza_asr_batch_size() {
    return eliza_int_env_or_default("ELIZA_ASR_N_BATCH", eliza_asr_android_cpu_profile() ? 64 : 512);
}

// OmniVoice TTS backend selection. Default CPU on Android, GPU elsewhere
// (mirrors the ASR Android-CPU profile); override with ELIZA_TTS_USE_GPU=1.
//
// History: the MaskGIT batched LM forward (pipeline_tts_llm_forward_batched)
// used to abort the Android Mali Vulkan driver — its fused pipelines net
// under-request ggml-vulkan descriptor sets and tripped
// GGML_ASSERT(descriptor_set_idx < descriptor_sets.size()) mid-synthesis. That
// crash is now handled by the grow-on-demand safety net in
// ggml_vk_dispatch_pipeline, so GPU TTS runs correctly (device-verified: 45120
// samples / 1.88 s, identical to CPU). CPU remains the Android DEFAULT for two
// reasons, neither of which is the old crash: (1) the iterative MaskGIT denoise
// uses small matrices where Mali GPU dispatch overhead cancels any speedup (GPU
// is not faster than CPU here on a Pixel 9a); (2) in the fused voice pipeline
// the text LM already runs on the GPU, so keeping TTS on the CPU lets the two
// run on separate compute units concurrently instead of contending for the GPU.
// Desktop/discrete-GPU keeps GPU TTS (where it can actually win).
static bool eliza_tts_use_gpu() {
    if (eliza_running_on_android()) {
        return eliza_bool_env_or_default("ELIZA_TTS_USE_GPU", false);
    }
    return eliza_bool_env_or_default("ELIZA_TTS_USE_GPU", true);
}

static std::vector<float> eliza_resample_linear(
    const float * pcm,
    size_t n_samples,
    int source_rate,
    int target_rate) {
    if (source_rate == target_rate || n_samples <= 1) {
        return std::vector<float>(pcm, pcm + n_samples);
    }
    const double scale = (double) target_rate / (double) source_rate;
    const size_t out_samples = std::max<size_t>(1, (size_t) std::llround((double) n_samples * scale));
    std::vector<float> out(out_samples);
    for (size_t i = 0; i < out_samples; ++i) {
        const double src = (double) i * (double) source_rate / (double) target_rate;
        const size_t lo = std::min((size_t) std::floor(src), n_samples - 1);
        const size_t hi = std::min(lo + 1, n_samples - 1);
        const float t = (float) (src - (double) lo);
        out[i] = pcm[lo] * (1.0f - t) + pcm[hi] * t;
    }
    return out;
}

static std::string eliza_llama_token_piece(const llama_vocab * vocab, llama_token token) {
    char small[256];
    int32_t n = llama_token_to_piece(vocab, token, small, (int32_t) sizeof(small), 0, false);
    if (n > 0) return std::string(small, (size_t) n);
    if (n == 0) return "";
    std::string buf((size_t) -n, '\0');
    n = llama_token_to_piece(vocab, token, buf.data(), (int32_t) buf.size(), 0, false);
    if (n > 0) return std::string(buf.data(), (size_t) n);
    return "";
}

/* L2-normalize an embedding vector in place. Mirrors normalizeEmbedding(view,
 * 2) in desktop-llama-adapter.ts (the gte-small convention). A zero-norm
 * vector is left untouched (division by ~0 would produce NaNs). */
static void eliza_l2_normalize(float * vec, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += (double) vec[i] * (double) vec[i];
    const double norm = std::sqrt(sum);
    if (norm <= 1e-12) return;
    const float inv = (float) (1.0 / norm);
    for (int i = 0; i < n; ++i) vec[i] *= inv;
}

static std::string eliza_trim_ascii(std::string value);

static std::string eliza_asr_force_language() {
    const char * env = std::getenv("ELIZA_ASR_FORCE_LANGUAGE");
    if (!env) return "English";
    std::string value = eliza_trim_ascii(env);
    std::string lower = value;
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    if (lower.empty() || lower == "0" || lower == "false" || lower == "none" || lower == "auto") {
        return "";
    }
    return value;
}

static std::string eliza_format_asr_prompt(llama_model * model) {
    (void) model;
    // Mirrors Qwen3-ASR's chat-template structure: empty system context,
    // one user audio turn, and a generation prompt. Appending
    // "language X<asr_text>" follows the upstream text-only forcing path
    // and avoids returning language metadata or role-token chatter.
    std::string prompt = std::string("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n") +
        mtmd_default_marker() +
        "<|im_end|>\n<|im_start|>assistant\n";
    std::string language = eliza_asr_force_language();
    if (!language.empty()) {
        prompt += "language " + language + "<asr_text>";
    }
    return prompt;
}

static std::string eliza_trim_ascii(std::string value) {
    auto is_space = [](unsigned char c) {
        return c == ' ' || c == '\n' || c == '\r' || c == '\t';
    };
    while (!value.empty() && is_space((unsigned char) value.front())) {
        value.erase(value.begin());
    }
    while (!value.empty() && is_space((unsigned char) value.back())) {
        value.pop_back();
    }
    return value;
}

static std::string eliza_clean_asr_transcript(std::string transcript) {
    const std::string asr_marker = "<asr_text>";
    size_t marker = transcript.find(asr_marker);
    if (marker != std::string::npos) {
        transcript = transcript.substr(marker + asr_marker.size());
    }
    const char * sentinels[] = {
        "<|im_start|>",
        "<|im_end|>",
        "<|endoftext|>",
        "<|audio_start|>",
        "<|audio_end|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "</s>",
    };
    for (const char * sentinel : sentinels) {
        size_t pos = transcript.find(sentinel);
        if (pos != std::string::npos) {
            transcript = transcript.substr(0, pos);
        }
    }
    transcript = eliza_trim_ascii(transcript);
    for (const char * sentinel : sentinels) {
        std::string full(sentinel);
        if (full.rfind(transcript, 0) == 0) {
            return "";
        }
    }
    return transcript;
}

static bool eliza_asr_has_text_payload(const std::string & transcript) {
    for (unsigned char c : transcript) {
        if (std::isalnum(c)) return true;
    }
    return false;
}

static int eliza_map_ov_status(ov_status rc) {
    if (rc == OV_STATUS_OK) return ELIZA_OK;
    if (rc == OV_STATUS_OOM) return ELIZA_ERR_OOM;
    if (rc == OV_STATUS_CANCELLED) return ELIZA_ERR_CANCELLED;
    if (rc == OV_STATUS_INVALID_PARAMS || rc == OV_STATUS_INSTRUCT_INVALID) return ELIZA_ERR_INVALID_ARG;
    return ELIZA_ERR_FFI_FAULT;
}

static bool eliza_tts_cancel_requested(void * user_data) {
    EliInferenceContext * ctx = (EliInferenceContext *) user_data;
    return ctx && ctx->tts_cancel.load(std::memory_order_acquire);
}

struct ElizaScopedTtsForward {
    EliInferenceContext * ctx;

    explicit ElizaScopedTtsForward(EliInferenceContext * c) : ctx(c) {
        if (ctx) ctx->tts_cancel.store(false, std::memory_order_release);
    }

    ~ElizaScopedTtsForward() {
        if (ctx) ctx->tts_cancel.store(false, std::memory_order_release);
    }
};

struct ElizaTtsStreamState {
    EliInferenceContext * ctx;
    eliza_tts_chunk_cb on_chunk;
    void * user_data;
    bool callback_cancelled;
};

static bool eliza_tts_stream_chunk(const float * samples, int n_samples, void * user_data) {
    ElizaTtsStreamState * state = (ElizaTtsStreamState *) user_data;
    if (!state || !state->on_chunk) return false;
    if (state->ctx && state->ctx->tts_cancel.load(std::memory_order_acquire)) return false;
    const int rc = state->on_chunk(samples, n_samples < 0 ? 0 : (size_t) n_samples, 0, state->user_data);
    if (rc != 0) {
        state->callback_cancelled = true;
        if (state->ctx) state->ctx->tts_cancel.store(true, std::memory_order_release);
        return false;
    }
    return !(state->ctx && state->ctx->tts_cancel.load(std::memory_order_acquire));
}

static void eliza_free_asr(EliInferenceContext * ctx) {
    if (!ctx) return;
    if (ctx->asr_sampler) {
        llama_sampler_free(ctx->asr_sampler);
        ctx->asr_sampler = nullptr;
    }
    if (ctx->asr_mtmd) {
        mtmd_free(ctx->asr_mtmd);
        ctx->asr_mtmd = nullptr;
    }
    if (ctx->asr_lctx) {
        llama_free(ctx->asr_lctx);
        ctx->asr_lctx = nullptr;
    }
    if (ctx->asr_model) {
        llama_model_free(ctx->asr_model);
        ctx->asr_model = nullptr;
    }
    ctx->asr_sample_rate = 0;
}

static int eliza_load_tts(EliInferenceContext * ctx, char ** out_error) {
    if (!ctx) {
        eliza_set_error(out_error, "[libelizainference] load_tts: ctx is NULL");
        return ELIZA_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> lock(ctx->tts_mutex);
    if (ctx->ov) return ELIZA_OK;
    if (ctx->tts_model_path.empty()) {
        if (!eliza_pick_voice_files(std::filesystem::path(ctx->bundle_dir), ctx->tts_model_path, ctx->codec_model_path)) {
            eliza_set_error(out_error, std::string("[libelizainference] no TTS GGUF found under ") + (std::filesystem::path(ctx->bundle_dir) / "tts").string());
            return ELIZA_ERR_BUNDLE_INVALID;
        }
    }

    ov_init_params params;
    ov_init_default_params(&params);
    params.model_path = ctx->tts_model_path.c_str();
    params.codec_path = ctx->codec_model_path.c_str();
    params.use_fa = true;

    // OmniVoice's backend_init_auto() picks the first GPU unless GGML_BACKEND
    // forces a device by name. When TTS-on-GPU is disabled (default on Android,
    // see eliza_tts_use_gpu) pin the OmniVoice sched to CPU for the duration of
    // ov_init. Restore the prior value so we don't leak the override to other
    // backend inits. ov_init is one-time and runs under tts_mutex, so the
    // global-env window is bounded.
    const bool   force_cpu_tts = !eliza_tts_use_gpu();
    const char * prev_backend  = force_cpu_tts ? getenv("GGML_BACKEND") : nullptr;
    const std::string prev_backend_saved = prev_backend ? std::string(prev_backend) : std::string();
    const bool   had_prev_backend = prev_backend != nullptr;
    if (force_cpu_tts) {
        setenv("GGML_BACKEND", "CPU", 1);
    }
    ctx->ov = ov_init(&params);
    if (force_cpu_tts) {
        if (had_prev_backend) {
            setenv("GGML_BACKEND", prev_backend_saved.c_str(), 1);
        } else {
            unsetenv("GGML_BACKEND");
        }
    }
    if (!ctx->ov) {
        std::string msg = "[libelizainference] ov_init failed: ";
        msg += ov_last_error();
        eliza_set_error(out_error, msg);
        return ELIZA_ERR_FFI_FAULT;
    }
    return ELIZA_OK;
}

static int eliza_load_asr(EliInferenceContext * ctx, char ** out_error) {
    if (!ctx) {
        eliza_set_error(out_error, "[libelizainference] load_asr: ctx is NULL");
        return ELIZA_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> lock(ctx->asr_mutex);
    if (ctx->asr_model && ctx->asr_lctx && ctx->asr_mtmd && ctx->asr_sampler) {
        return ELIZA_OK;
    }
    if (ctx->asr_model_path.empty() || ctx->asr_mmproj_path.empty()) {
        if (!eliza_pick_asr_files(std::filesystem::path(ctx->bundle_dir), ctx->asr_model_path, ctx->asr_mmproj_path)) {
            eliza_set_error(out_error, std::string("[libelizainference] ASR requires both a text GGUF and mmproj GGUF under ") + (std::filesystem::path(ctx->bundle_dir) / "asr").string());
            return ELIZA_ERR_BUNDLE_INVALID;
        }
    }

    std::call_once(eliza_llama_backend_once, []() {
        llama_backend_init();
    });

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = eliza_asr_use_gpu() ? 99 : 0;
    mparams.use_mmap = eliza_asr_use_mmap();
    mparams.use_extra_bufts = eliza_asr_use_extra_bufts();
    eliza_asr_debug_log("loading ASR text model");
    ctx->asr_model = llama_model_load_from_file(ctx->asr_model_path.c_str(), mparams);
    if (!ctx->asr_model) {
        eliza_free_asr(ctx);
        eliza_set_error(out_error, std::string("[libelizainference] failed to load ASR model: ") + ctx->asr_model_path);
        return ELIZA_ERR_BUNDLE_INVALID;
    }
    eliza_asr_debug_log("loaded ASR text model; initializing llama context");

    llama_context_params cparams = llama_context_default_params();
    ctx->asr_n_batch = eliza_asr_batch_size();
    cparams.n_ctx = eliza_asr_context_size();
    cparams.n_batch = (uint32_t) ctx->asr_n_batch;
    cparams.n_ubatch = (uint32_t) ctx->asr_n_batch;
    cparams.n_threads = eliza_asr_thread_count(false);
    cparams.n_threads_batch = eliza_asr_thread_count(true);
    cparams.flash_attn_type = eliza_asr_android_cpu_profile()
        ? LLAMA_FLASH_ATTN_TYPE_DISABLED
        : LLAMA_FLASH_ATTN_TYPE_AUTO;
    ctx->asr_lctx = llama_init_from_model(ctx->asr_model, cparams);
    if (!ctx->asr_lctx) {
        eliza_free_asr(ctx);
        eliza_set_error(out_error, "[libelizainference] failed to initialize ASR llama context");
        return ELIZA_ERR_FFI_FAULT;
    }
    eliza_asr_debug_log("initialized ASR llama context; loading audio mmproj");

    mtmd_context_params aparams = mtmd_context_params_default();
    aparams.use_gpu = eliza_asr_use_gpu();
    aparams.print_timings = false;
    aparams.n_threads = eliza_asr_thread_count(true);
    aparams.flash_attn_type = eliza_asr_android_cpu_profile()
        ? LLAMA_FLASH_ATTN_TYPE_DISABLED
        : LLAMA_FLASH_ATTN_TYPE_AUTO;
    aparams.warmup = !eliza_asr_android_cpu_profile();
    ctx->asr_mtmd = mtmd_init_from_file(ctx->asr_mmproj_path.c_str(), ctx->asr_model, aparams);
    if (!ctx->asr_mtmd) {
        eliza_free_asr(ctx);
        eliza_set_error(out_error, std::string("[libelizainference] failed to load ASR mmproj: ") + ctx->asr_mmproj_path);
        return ELIZA_ERR_BUNDLE_INVALID;
    }
    eliza_asr_debug_log("loaded ASR audio mmproj; initializing sampler");
    if (!mtmd_support_audio(ctx->asr_mtmd)) {
        eliza_free_asr(ctx);
        eliza_set_error(out_error, "[libelizainference] ASR mmproj does not report audio support");
        return ELIZA_ERR_BUNDLE_INVALID;
    }
    ctx->asr_sample_rate = mtmd_get_audio_sample_rate(ctx->asr_mtmd);
    if (ctx->asr_sample_rate <= 0) {
        eliza_free_asr(ctx);
        eliza_set_error(out_error, "[libelizainference] ASR mmproj returned an invalid audio sample rate");
        return ELIZA_ERR_BUNDLE_INVALID;
    }

    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    ctx->asr_sampler = llama_sampler_chain_init(sparams);
    if (!ctx->asr_sampler) {
        eliza_free_asr(ctx);
        eliza_set_error(out_error, "[libelizainference] failed to initialize ASR sampler");
        return ELIZA_ERR_FFI_FAULT;
    }
    llama_sampler_chain_add(ctx->asr_sampler, llama_sampler_init_greedy());
    eliza_asr_debug_log("ASR region ready");
    return ELIZA_OK;
}

/* ---- Streaming LLM (text generation) ------------------------------- *
 *
 * In-process text generation over the bundle's text GGUF. See the ABI
 * declarations in eliza-inference-ffi.h. Each EliLlmStream owns a private
 * llama_context (so KV state never interleaves between sessions) over the
 * shared, lazily-loaded text model.
 */

/* ---- Same-file / separate-drafter MTP speculative-decode engine ----- *
 *
 * Owns the full draft -> verify -> accept loop natively, wrapping
 * common/speculative.cpp's DRAFT_MTP implementation. This is a literal port
 * of the eliza-llama-shim's eliza_mtp_driver.cpp engine
 * (packages/app-core/scripts/desktop-llama-shim/eliza_mtp_driver.cpp) — the
 * SAME engine the libllama path drives — so the fused text path matches it.
 *
 * Two shapes, mirroring tools/server/server-context.cpp:
 *   - same-file MTP: the NextN head lives in the target text GGUF. The
 *     engine creates a second LLAMA_CONTEXT_TYPE_MTP context over the SAME
 *     model; no separate drafter is loaded (`drafter_path` empty).
 *   - separate-drafter MTP: a distinct drafter GGUF is loaded and its
 *     LLAMA_CONTEXT_TYPE_MTP context drafts for the target.
 *
 * Single-sequence (seq_id 0). Contexts that allow partial suffix removal
 * (dense bodies) trim the rejected-draft KV tail directly; FULL/RS bodies
 * (recurrent / hybrid delta-net, e.g. Qwen3.5) roll back via state
 * checkpoints. _TYPE_NO (no rollback at all) is refused at create so the
 * caller falls back to plain decode. */
namespace eliza_mtp {

constexpr llama_state_seq_flags CKPT_FLAGS =
    LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE;

struct Engine {
    llama_model        * model_tgt = nullptr; // borrowed (ctx owns the shared model)
    llama_model        * model_dft = nullptr; // owned iff separate-drafter
    llama_context      * ctx_tgt   = nullptr; // owned by the engine
    llama_context      * ctx_dft   = nullptr; // owned by the engine
    common_speculative * spec      = nullptr; // owned
    common_sampler     * smpl      = nullptr; // owned

    std::vector<llama_token> prompt;
    llama_token              id_last = 0;

    int32_t n_ctx     = 0;
    int32_t draft_min = 1;
    int32_t draft_max = 2;

    /* Per-context rollback capability. `use_ckpt_*` is NOT a static property:
     * an RS (recurrent-state) context rolls back directly via seq_rm when the
     * draft fits inside its n_rs_seq window, and only needs a checkpoint when
     * the draft exceeds it. FULL contexts always need a checkpoint; PART
     * contexts never do. Computed per-step by `use_ckpt()` below, mirroring
     * tools/server/server-context.cpp:2590-2595. */
    common_context_seq_rm_type seq_rm_tgt = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_context_seq_rm_type seq_rm_dft = COMMON_CONTEXT_SEQ_RM_TYPE_NO;
    common_prompt_checkpoint ckpt;

    llama_batch batch{};

    uint64_t st_drafted  = 0;
    uint64_t st_accepted = 0;
};

/* Whether a context with rollback type `t` needs a state checkpoint to
 * retract a draft of `n_draft` tokens, given its `n_rs_seq` window. Mirrors
 * tools/server/server-context.cpp:2590-2595. */
inline bool needs_ckpt(common_context_seq_rm_type t, int32_t n_draft, uint32_t n_rs_seq) {
    return t == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
           (t == COMMON_CONTEXT_SEQ_RM_TYPE_RS && (uint32_t) n_draft > n_rs_seq);
}

inline void rollback_to_committed(
        llama_context *                  ctx,
        bool                             use_ckpt,
        const common_prompt_checkpoint & ckpt,
        bool                             is_tgt,
        int32_t                          n_committed) {
    if (use_ckpt) {
        if (is_tgt) {
            ckpt.load_tgt(ctx, 0, CKPT_FLAGS);
        } else {
            ckpt.load_dft(ctx, 0, CKPT_FLAGS);
        }
        common_context_seq_rm(ctx, 0, ckpt.pos_max + 1, -1);
    } else {
        common_context_seq_rm(ctx, 0, n_committed, -1);
    }
}

static void free_engine(Engine * e) {
    if (!e) return;
    if (e->batch.token != nullptr || e->batch.embd != nullptr) {
        llama_batch_free(e->batch);
    }
    if (e->smpl) common_sampler_free(e->smpl);
    if (e->spec) common_speculative_free(e->spec);
    if (e->ctx_dft) llama_free(e->ctx_dft);
    if (e->ctx_tgt) llama_free(e->ctx_tgt);
    if (e->model_dft) llama_model_free(e->model_dft);
    delete e;
}

/* Reset an MTP engine for warm reuse: clear both KV caches, reset the outer
 * sampler, and zero the committed-token accumulator. The DRAFT_MTP driver's
 * per-sequence accumulators (pending_h / verify_h / i_batch_*) are re-seeded by
 * the next prefill (common_speculative_process per ubatch + common_speculative_begin),
 * so there is nothing else to retract here. `ckpt` is re-snapshotted by step()
 * before every use, so a stale value is never read. This is the close+reopen
 * path minus the context teardown — identical re-prefill cost, no second
 * LLAMA_CONTEXT_TYPE_MTP context re-init per turn. */
static void reset_engine(Engine * e) {
    if (!e) return;
    if (e->ctx_tgt) llama_memory_clear(llama_get_memory(e->ctx_tgt), true);
    if (e->ctx_dft) llama_memory_clear(llama_get_memory(e->ctx_dft), true);
    if (e->smpl) common_sampler_reset(e->smpl);
    e->prompt.clear();
    e->id_last = 0;
}

} // namespace eliza_mtp

struct EliLlmStream {
    EliInferenceContext * ctx = nullptr;
    /* Multi-backend seam (M3): when non-NULL, this session is driven by an
     * alternate in-process runtime (LiteRT-LM / MLX-CoreML) and the llama.cpp
     * fields below (lctx/sampler/mtp) are unused — every FFI streaming entry
     * delegates to `backend` and returns before touching the llama.cpp path. */
    LlmBackendSession * backend = nullptr;
    llama_context * lctx = nullptr;
    llama_sampler * sampler = nullptr;
    int n_past = 0;
    int generated = 0;
    int max_tokens = 0;
    bool eos = false;
    std::atomic<bool> cancel{false};

    /* MTP speculative engine — non-NULL only when mtp_drafter_path is set
     * (separate drafter) or the same-file MTP path is requested. When
     * present, lctx/sampler/n_past are unused (the engine owns ctx_tgt and
     * samples internally); when NULL the plain fixed-KV loop above runs. */
    eliza_mtp::Engine * mtp = nullptr;
    int32_t mtp_first_token = -1; // first token sampled at prefill, emitted first
    std::vector<int32_t> mtp_step_buf; // scratch for one engine step
};

/* Resolve the bundle's text GGUF. Picks the single non-mmproj, non-codec,
 * non-tokenizer .gguf under <bundle>/text. The bundle layout guarantees a
 * single text artifact per tier (AGENTS.md §2). */
static bool eliza_pick_text_file(
    const std::filesystem::path & bundle_dir,
    std::string & text_model) {
    std::vector<std::string> candidates = eliza_find_ggufs(bundle_dir / "text");
    std::vector<std::string> picked;
    for (const std::string & path : candidates) {
        const std::string lower =
            eliza_lower_ascii(std::filesystem::path(path).filename().string());
        if (lower.find("mmproj") != std::string::npos) continue;
        if (lower.find("tokenizer") != std::string::npos) continue;
        if (lower.find("codec") != std::string::npos) continue;
        picked.push_back(path);
    }
    if (picked.empty()) return false;
    text_model = picked[0];
    return true;
}

/* Load the shared text model once. Caller must hold ctx->llm_mutex.
 *
 * `n_gpu_layers` is the per-load GPU-offload count from the session config
 * (ABI v8): -1 selects the default (all layers), 0 forces CPU. The model is
 * loaded once and shared across every session anchored to this ctx, so the
 * FIRST session's `n_gpu_layers` wins; later sessions reuse the resident
 * model. This mirrors desktop-llama-adapter.ts, where gpuLayers is a
 * loadModel() (not per-stream) decision. */
static int eliza_load_llm_model_locked(
    EliInferenceContext * ctx,
    int32_t n_gpu_layers,
    char ** out_error) {
    if (ctx->llm_model) return ELIZA_OK;
    if (ctx->llm_model_path.empty()) {
        if (!eliza_pick_text_file(std::filesystem::path(ctx->bundle_dir),
                                  ctx->llm_model_path)) {
            eliza_set_error(out_error,
                std::string("[libelizainference] no text GGUF found under ") +
                (std::filesystem::path(ctx->bundle_dir) / "text").string());
            return ELIZA_ERR_BUNDLE_INVALID;
        }
    }

    std::call_once(eliza_llama_backend_once, []() {
        llama_backend_init();
    });

    llama_model_params mparams = llama_model_default_params();
    /* -1 = all layers on GPU (99 per llama.cpp convention); 0 = CPU. The
     * env knob remains the fallback when the caller passes -1. */
    mparams.n_gpu_layers =
        n_gpu_layers >= 0
            ? n_gpu_layers
            : (eliza_bool_env_or_default("ELIZA_LLM_USE_GPU", true) ? 99 : 0);
    mparams.use_mmap = true;
    ctx->llm_model =
        llama_model_load_from_file(ctx->llm_model_path.c_str(), mparams);
    if (!ctx->llm_model) {
        eliza_set_error(out_error,
            std::string("[libelizainference] failed to load text model: ") +
            ctx->llm_model_path);
        return ELIZA_ERR_BUNDLE_INVALID;
    }
    return ELIZA_OK;
}

/* Map a KV-cache quant type NAME (e.g. "f16", "q8_0", "qjl1_256",
 * "q4_polar") to the corresponding ggml_type. Mirrors GGML_KV_CACHE_TYPES
 * in desktop-llama-adapter.ts. Returns GGML_TYPE_COUNT and populates
 * out_error when the name is non-empty but unrecognized; returns
 * GGML_TYPE_F16 (the llama.cpp default) when name is NULL/empty. */
static ggml_type eliza_kv_cache_type(const char * name, char ** out_error) {
    if (!name || name[0] == '\0') return GGML_TYPE_F16;
    const std::string n = eliza_lower_ascii(std::string(name));
    static const std::unordered_map<std::string, ggml_type> kKvTypes = {
        {"f32", GGML_TYPE_F32},        {"f16", GGML_TYPE_F16},
        {"q4_0", GGML_TYPE_Q4_0},      {"q4_1", GGML_TYPE_Q4_1},
        {"q5_0", GGML_TYPE_Q5_0},      {"q5_1", GGML_TYPE_Q5_1},
        {"q8_0", GGML_TYPE_Q8_0},      {"q4_k", GGML_TYPE_Q4_K},
        {"q5_k", GGML_TYPE_Q5_K},      {"q6_k", GGML_TYPE_Q6_K},
        {"q8_k", GGML_TYPE_Q8_K},      {"iq4_nl", GGML_TYPE_IQ4_NL},
        {"bf16", GGML_TYPE_BF16},      {"tbq3_0", GGML_TYPE_TBQ3_0},
        {"tbq4_0", GGML_TYPE_TBQ4_0},  {"qjl1_256", GGML_TYPE_QJL1_256},
        {"q4_polar", GGML_TYPE_Q4_POLAR}, {"tbq3_tcq", GGML_TYPE_TBQ3_TCQ},
    };
    const auto it = kKvTypes.find(n);
    if (it == kKvTypes.end()) {
        eliza_set_error(out_error,
            std::string("[libelizainference] llm_stream: unsupported KV cache "
                        "type '") + name + "'");
        return GGML_TYPE_COUNT;
    }
    return it->second;
}

/* Build the per-session sampler chain from cfg. Order (grammar first so it
 * masks the candidate set before any other transform, then penalties,
 * top_k, top_p, temperature, and finally the distribution selector):
 *
 *   grammar? -> penalties? -> top_k? -> top_p? -> temp -> dist
 *
 * A non-empty gbnf grammar installs a grammar sampler keyed at the "root"
 * rule. temperature <= 0 collapses to greedy (argmax) — the dist sampler is
 * skipped and a greedy sampler is appended instead so structured single-
 * value positions decode deterministically. Returns NULL on grammar parse
 * failure (out_error populated). */
static llama_sampler * eliza_build_llm_sampler_chain(
    const llama_model * model,
    const eliza_llm_stream_config_t * cfg,
    char ** out_error) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler * chain = llama_sampler_chain_init(sparams);
    if (!chain) {
        eliza_set_error(out_error,
            "[libelizainference] llm_stream: failed to init sampler chain");
        return nullptr;
    }

    if (cfg->gbnf_grammar && cfg->gbnf_grammar[0] != '\0') {
        llama_sampler * grammar =
            llama_sampler_init_grammar(vocab, cfg->gbnf_grammar, "root");
        if (!grammar) {
            llama_sampler_free(chain);
            eliza_set_error(out_error,
                "[libelizainference] llm_stream: GBNF grammar failed to parse "
                "(llama_sampler_init_grammar returned NULL)");
            return nullptr;
        }
        llama_sampler_chain_add(chain, grammar);
    }

    if (cfg->repeat_penalty != 0.0f && cfg->repeat_penalty != 1.0f) {
        llama_sampler_chain_add(chain,
            llama_sampler_init_penalties(64, cfg->repeat_penalty, 0.0f, 0.0f));
    }

    const bool greedy = cfg->temperature <= 0.0f;
    if (!greedy) {
        if (cfg->top_k > 0) {
            llama_sampler_chain_add(chain, llama_sampler_init_top_k(cfg->top_k));
        }
        if (cfg->top_p > 0.0f && cfg->top_p < 1.0f) {
            llama_sampler_chain_add(chain,
                llama_sampler_init_top_p(cfg->top_p, 1));
        }
        llama_sampler_chain_add(chain,
            llama_sampler_init_temp(cfg->temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }
    return chain;
}

namespace eliza_mtp {

/* Build the MTP speculative engine. `drafter_path` empty -> same-file MTP
 * (NextN head in the target GGUF, ctx_dft over the SAME model). Non-empty ->
 * separate-drafter MTP (load the drafter GGUF, ctx_dft over it). On failure
 * returns nullptr with out_error populated. cparams_tgt is the fully-built
 * target context params (n_ctx / n_batch / threads / flash-attn / KV-quant
 * already applied) so the engine's target context matches the plain path. */
static Engine * create_engine(
    llama_model *                model_tgt,
    const llama_context_params & cparams_tgt,
    const eliza_llm_stream_config_t * cfg,
    const std::string &          drafter_path,
    char **                      out_error) {
    auto * e = new (std::nothrow) Engine();
    if (!e) {
        eliza_set_error(out_error, "[libelizainference] mtp: out of memory");
        return nullptr;
    }
    e->model_tgt = model_tgt;
    e->draft_min = cfg->draft_min > 0 ? cfg->draft_min : 1;
    e->draft_max = cfg->draft_max >= e->draft_min ? cfg->draft_max : e->draft_min;

    /* Target context (engine-owned). Mirrors the plain path's cparams, but
     * sets n_rs_seq = draft_max so a recurrent / hybrid target (Qwen3.5's
     * Gated Delta Net body) can do bounded RS partial rollback of the
     * rejected draft tail. This is what tools/server/server-context.cpp gets
     * via `common_context_params_to_llama` → `need_n_rs_seq()` (= draft.n_max
     * for MTP). Without it the target falls back to FULL/checkpoint rollback,
     * which does not correctly retract multi-token draft tails for larger
     * draft windows — the verifier then rubber-stamps the whole draft and
     * greedy output diverges from the non-speculative path. A non-recurrent
     * model ignores n_rs_seq (clamped to 0 by llama_init). */
    llama_context_params cparams_tgt_mtp = cparams_tgt;
    cparams_tgt_mtp.n_rs_seq = (uint32_t) e->draft_max;
    e->ctx_tgt = llama_init_from_model(model_tgt, cparams_tgt_mtp);
    if (!e->ctx_tgt) {
        eliza_set_error(out_error,
            "[libelizainference] mtp: failed to init target context");
        free_engine(e);
        return nullptr;
    }
    e->n_ctx = (int32_t) llama_n_ctx(e->ctx_tgt);

    e->seq_rm_tgt = common_context_can_seq_rm(e->ctx_tgt);
    if (e->seq_rm_tgt == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
        eliza_set_error(out_error,
            "[libelizainference] mtp: target context cannot retract drafts "
            "(seq_rm_type=NO) — speculative decoding impossible");
        free_engine(e);
        return nullptr;
    }

    /* Draft context. Same-file: MTP context over model_tgt. Separate: load
     * the drafter GGUF and build an MTP context over it. */
    llama_model * model_for_dft = model_tgt;
    if (!drafter_path.empty()) {
        llama_model_params dmp = llama_model_default_params();
        dmp.n_gpu_layers =
            cfg->n_gpu_layers >= 0
                ? cfg->n_gpu_layers
                : (eliza_bool_env_or_default("ELIZA_LLM_USE_GPU", true) ? 99 : 0);
        dmp.use_mmap = true;
        e->model_dft = llama_model_load_from_file(drafter_path.c_str(), dmp);
        if (!e->model_dft) {
            eliza_set_error(out_error,
                std::string("[libelizainference] mtp: failed to load drafter "
                            "model: ") + drafter_path);
            free_engine(e);
            return nullptr;
        }
        model_for_dft = e->model_dft;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = llama_n_ctx(e->ctx_tgt);
    cp.n_batch         = llama_n_batch(e->ctx_tgt);
    cp.n_ubatch        = llama_n_ubatch(e->ctx_tgt);
    cp.n_seq_max       = 1;
    cp.ctx_type        = LLAMA_CONTEXT_TYPE_MTP;
    cp.n_rs_seq        = 0; // draft ctx rolls back via PART/checkpoint, not RS
    cp.n_threads       = cparams_tgt.n_threads;
    cp.n_threads_batch = cparams_tgt.n_threads_batch;

    e->ctx_dft = llama_init_from_model(model_for_dft, cp);
    if (!e->ctx_dft) {
        eliza_set_error(out_error,
            "[libelizainference] mtp: failed to init MTP draft context");
        free_engine(e);
        return nullptr;
    }
    e->seq_rm_dft = common_context_can_seq_rm(e->ctx_dft);
    if (e->seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
        eliza_set_error(out_error,
            "[libelizainference] mtp: draft context cannot retract drafts");
        free_engine(e);
        return nullptr;
    }

    common_params_speculative sp;
    sp.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    sp.draft.ctx_tgt = e->ctx_tgt;
    sp.draft.ctx_dft = e->ctx_dft;
    sp.draft.n_max   = e->draft_max;
    sp.draft.n_min   = e->draft_min;
    if (!drafter_path.empty()) sp.draft.mparams.path = drafter_path;

    try {
        e->spec = common_speculative_init(sp, 1);
    } catch (...) {
        e->spec = nullptr;
    }
    if (!e->spec) {
        eliza_set_error(out_error,
            "[libelizainference] mtp: common_speculative_init failed");
        free_engine(e);
        return nullptr;
    }

    common_params_sampling sparams;
    sparams.seed  = LLAMA_DEFAULT_SEED;
    sparams.temp  = cfg->temperature;
    sparams.top_k = cfg->top_k;
    sparams.top_p = cfg->top_p;
    e->smpl = common_sampler_init(model_tgt, sparams);
    if (!e->smpl) {
        eliza_set_error(out_error,
            "[libelizainference] mtp: common_sampler_init failed");
        free_engine(e);
        return nullptr;
    }

    e->batch = llama_batch_init(1 + e->draft_max, 0, 1);
    return e;
}

/* Prefill the prompt into the target context, seed the speculative state,
 * and sample the first token. Returns ELIZA_OK / negative ELIZA_*. */
static int prefill(Engine * e, const int32_t * tokens, int32_t n_tokens) {
    if (!e || !tokens || n_tokens <= 0) return ELIZA_ERR_INVALID_ARG;
    e->prompt.assign(tokens, tokens + n_tokens);
    llama_set_embeddings(e->ctx_tgt, false);

    /* MTP (DRAFT_MTP) extracts the target's pre-norm embeddings at EVERY
     * prompt position to seed the NextN draft head, so every prefill token
     * must carry an output flag (logits=true) — not just the last. This
     * matches tools/server/server-context.cpp, which sets the per-token
     * output flag to `slot.need_embd()` while filling the prompt batch. A
     * last-only flag triggers `get_embeddings_pre_norm_ith: batch.logits[i]
     * != true` inside common_speculative_process. */
    const int32_t n_batch = (int32_t) llama_n_batch(e->ctx_tgt);
    for (int32_t i = 0; i < n_tokens; i += n_batch) {
        const int32_t cnt = std::min(n_batch, n_tokens - i);
        llama_batch b = llama_batch_init(cnt, 0, 1);
        for (int32_t k = 0; k < cnt; ++k) {
            common_batch_add(b, tokens[i + k], i + k, { 0 }, /* logits= */ true);
        }
        const int rc = llama_decode(e->ctx_tgt, b);
        if (rc != 0) { llama_batch_free(b); return ELIZA_ERR_FFI_FAULT; }
        if (!common_speculative_process(e->spec, b)) {
            llama_batch_free(b);
            return ELIZA_ERR_FFI_FAULT;
        }
        llama_batch_free(b);
    }
    common_speculative_begin(e->spec, 0, e->prompt);
    const llama_token id = common_sampler_sample(e->smpl, e->ctx_tgt, n_tokens - 1);
    common_sampler_accept(e->smpl, id, true);
    e->id_last = id;
    return id;
}

/* Run one speculative step. Writes accepted token ids into `out` (cap >=
 * 1 + draft_max), returns the count (>=1) or negative ELIZA_*. */
static int step(Engine * e, int32_t * out, int32_t cap) {
    if (!e || !out || cap < 1) return ELIZA_ERR_INVALID_ARG;

    const int32_t P    = (int32_t) e->prompt.size();
    const llama_token seed = e->id_last;

    int32_t n_draft_max = e->draft_max;
    const int32_t remaining = e->n_ctx - P;
    if (remaining <= 1) n_draft_max = 0;
    else if (n_draft_max > remaining - 1) n_draft_max = remaining - 1;

    const uint32_t n_rs_tgt = llama_n_rs_seq(e->ctx_tgt);
    const uint32_t n_rs_dft = llama_n_rs_seq(e->ctx_dft);

    /* Snapshot the committed-prefix position before the draft pollutes the
     * draft context's recurrent state. A FULL draft context (no partial
     * rollback) also needs its state captured here. */
    const bool ckpt_dft_full = e->seq_rm_dft == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;
    {
        const llama_pos pos_min = llama_memory_seq_pos_min(llama_get_memory(e->ctx_tgt), 0);
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(e->ctx_tgt), 0);
        e->ckpt.update_pos(P, pos_min, pos_max);
        if (ckpt_dft_full) e->ckpt.update_dft(e->ctx_dft, 0, CKPT_FLAGS);
    }

    std::vector<llama_token> draft;
    auto & dp = common_speculative_get_draft_params(e->spec, 0);
    if (n_draft_max > 0) {
        dp = {
            /* .drafting = */ true,
            /* .n_max    = */ n_draft_max,
            /* .n_past   = */ P,
            /* .id_last  = */ seed,
            /* .prompt   = */ &e->prompt,
            /* .result   = */ &draft,
        };
        common_speculative_draft(e->spec);
    } else {
        dp.drafting = false;
    }
    const int32_t D = (int32_t) draft.size();

    /* Per-step checkpoint decision (mirrors server-context.cpp:2590-2595):
     * FULL contexts always checkpoint; RS contexts checkpoint only when the
     * draft exceeds their n_rs_seq window; PART contexts never do. */
    const bool use_ckpt_tgt = needs_ckpt(e->seq_rm_tgt, D, n_rs_tgt);
    const bool use_ckpt_dft = needs_ckpt(e->seq_rm_dft, D, n_rs_dft);

    /* Restore the draft context to the committed prefix so process() can
     * cleanly re-mirror the verify batch into it. */
    if (ckpt_dft_full) {
        e->ckpt.load_dft(e->ctx_dft, 0, CKPT_FLAGS);
        common_context_seq_rm(e->ctx_dft, 0, e->ckpt.pos_max + 1, -1);
    } else {
        common_context_seq_rm(e->ctx_dft, 0, P, -1);
    }

    /* Snapshot the target AFTER the draft (the draft does not touch ctx_tgt)
     * only when this step's draft will need a checkpoint to retract. */
    if (use_ckpt_tgt && D > 0) {
        e->ckpt.update_tgt(e->ctx_tgt, 0, CKPT_FLAGS);
    }

    common_batch_clear(e->batch);
    std::vector<int> idxs;
    idxs.reserve((size_t) D + 1);
    idxs.push_back(0);
    common_batch_add(e->batch, seed, P, { 0 }, true);
    for (int32_t i = 0; i < D; ++i) {
        idxs.push_back(i + 1);
        common_batch_add(e->batch, draft[i], P + 1 + i, { 0 }, true);
    }

    llama_set_embeddings(e->ctx_tgt, false);
    if (llama_decode(e->ctx_tgt, e->batch) != 0) return ELIZA_ERR_FFI_FAULT;
    if (!common_speculative_process(e->spec, e->batch)) return ELIZA_ERR_FFI_FAULT;
    e->st_drafted += (uint64_t) D;

    std::vector<llama_token> accepted;
    if (D > 0) {
        accepted = common_sampler_sample_and_accept_n(e->smpl, e->ctx_tgt, idxs, draft);
        common_speculative_accept(e->spec, 0, (uint16_t) (accepted.size() - 1));
    } else {
        const llama_token id = common_sampler_sample(e->smpl, e->ctx_tgt, 0);
        common_sampler_accept(e->smpl, id, true);
        accepted.push_back(id);
    }
    const int32_t A = (int32_t) accepted.size();
    e->st_accepted += (uint64_t) (A - 1);

    const int32_t n_rollback = (D + 1) - A;
    const int32_t newP = P + A;
    if (n_rollback == 0) {
        /* full acceptance — nothing to retract */
    } else if (!use_ckpt_tgt && !use_ckpt_dft) {
        /* PART / RS-within-window: trim the rejected-draft tail directly. */
        common_context_seq_rm(e->ctx_tgt, 0, newP, -1);
        common_context_seq_rm(e->ctx_dft, 0, newP, -1);
    } else {
        rollback_to_committed(e->ctx_tgt, use_ckpt_tgt, e->ckpt, /* is_tgt= */ true,  P);
        rollback_to_committed(e->ctx_dft, use_ckpt_dft, e->ckpt, /* is_tgt= */ false, P);
        common_batch_clear(e->batch);
        common_batch_add(e->batch, seed, P, { 0 }, true);
        for (int32_t i = 0; i + 1 < A; ++i) {
            common_batch_add(e->batch, accepted[i], P + 1 + i, { 0 }, true);
        }
        llama_set_embeddings(e->ctx_tgt, false);
        if (llama_decode(e->ctx_tgt, e->batch) != 0) return ELIZA_ERR_FFI_FAULT;
        if (!common_speculative_process(e->spec, e->batch)) return ELIZA_ERR_FFI_FAULT;
    }

    e->prompt.push_back(seed);
    for (int32_t i = 0; i + 1 < A; ++i) e->prompt.push_back(accepted[i]);
    e->id_last = accepted.back();

    int32_t n_out = std::min(A, cap);
    for (int32_t i = 0; i < n_out; ++i) out[i] = accepted[i];
    return n_out;
}

} // namespace eliza_mtp

extern "C" {

const char * eliza_inference_abi_version(void) {
    // Keep this tied to ffi.h so ABI bumps cannot drift between the
    // generated adapter and the TypeScript loader.
    // Keep in lockstep with ELIZA_INFERENCE_ABI_VERSION in
    // packages/app-core/src/services/local-inference/voice/ffi-bindings.ts.
    return ELIZA_STRINGIFY(ELIZA_INFERENCE_ABI_VERSION);
}

EliInferenceContext * eliza_inference_create(
    const char * bundle_dir,
    char ** out_error) {
    if (!bundle_dir || bundle_dir[0] == '\0') {
        eliza_set_error(out_error, "[libelizainference] bundle_dir is required");
        return nullptr;
    }
    std::filesystem::path root(bundle_dir);
    std::error_code ec;
    if (!std::filesystem::exists(root, ec) || !std::filesystem::is_directory(root, ec)) {
        eliza_set_error(out_error, std::string("[libelizainference] bundle_dir does not exist: ") + bundle_dir);
        return nullptr;
    }

    EliInferenceContext * ctx = new (std::nothrow) EliInferenceContext();
    if (!ctx) {
        eliza_set_error(out_error, "[libelizainference] out of memory allocating context");
        return nullptr;
    }
    ctx->bundle_dir = root.string();

    // Metadata-only: heavy voice weights are intentionally loaded by
    // eliza_inference_mmap_acquire("tts") so voice-off does not keep
    // OmniVoice resident.
    return ctx;
}

void eliza_inference_destroy(EliInferenceContext * ctx) {
    if (!ctx) return;
    {
        std::lock_guard<std::mutex> lock(ctx->tts_mutex);
        ov_free(ctx->ov);
        ctx->ov = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(ctx->asr_mutex);
        eliza_free_asr(ctx);
    }
    {
        std::lock_guard<std::mutex> lock(ctx->llm_mutex);
        if (ctx->vision_mtmd) {
            mtmd_free(ctx->vision_mtmd);
            ctx->vision_mtmd = nullptr;
        }
        if (ctx->embed_ctx) {
            llama_free(ctx->embed_ctx);
            ctx->embed_ctx = nullptr;
        }
        if (ctx->eot_ctx) {
            llama_free(ctx->eot_ctx);
            ctx->eot_ctx = nullptr;
        }
        if (ctx->llm_model) {
            llama_model_free(ctx->llm_model);
            ctx->llm_model = nullptr;
        }
    }
#ifdef ELIZA_ENABLE_KOKORO
    {
        std::lock_guard<std::mutex> lock(ctx->kokoro_mutex);
        ctx->kokoro_model.reset();   /* kokoro_model_deleter frees the GGUF */
        ctx->kokoro_loaded = false;
    }
#endif
    delete ctx;
}

/* ===================================================================== *
 * Kokoro TTS (ABI v10) — in-process fold. No local-TCP llama-server route
 * (forbidden on iOS / Google Play). kokoro_lib is a distinct TTS pipeline
 * with its own GGUF reader; loaded once per ctx, owned by the ctx.
 * ===================================================================== */

int eliza_inference_kokoro_supported(void) {
#ifdef ELIZA_ENABLE_KOKORO
    return 1;
#else
    return 0;
#endif
}

int eliza_inference_kokoro_load(
    EliInferenceContext * ctx,
    const char * gguf_path,
    const char * voice_bin_path,
    int style_dim,
    char ** out_error) {
#ifdef ELIZA_ENABLE_KOKORO
    if (!ctx || !gguf_path || !voice_bin_path) {
        eliza_set_error(out_error, "[libelizainference] kokoro_load: null argument");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (style_dim <= 0) style_dim = 256;
    std::lock_guard<std::mutex> lock(ctx->kokoro_mutex);
    /* Idempotent: already loaded with these exact paths → reuse, no reload. */
    if (ctx->kokoro_loaded && ctx->kokoro_model &&
        ctx->kokoro_gguf_path == gguf_path &&
        ctx->kokoro_voice_path == voice_bin_path) {
        return ELIZA_OK;
    }
    std::string err;
    eliza_kokoro::kokoro_model_ptr model =
        eliza_kokoro::kokoro_load_model(gguf_path, err);
    if (!model) {
        eliza_set_error(out_error,
            std::string("[libelizainference] kokoro_load_model failed: ") + err);
        return ELIZA_ERR_BUNDLE_INVALID;
    }
    eliza_kokoro::kokoro_voice_preset voice;
    eliza_kokoro::kokoro_status st =
        eliza_kokoro::kokoro_load_voice_preset(voice_bin_path, style_dim, voice, err);
    if (st != eliza_kokoro::KOKORO_OK) {
        eliza_set_error(out_error,
            std::string("[libelizainference] kokoro_load_voice_preset failed: ") + err);
        return ELIZA_ERR_BUNDLE_INVALID;
    }
    ctx->kokoro_model = std::move(model);
    ctx->kokoro_voice = std::move(voice);
    ctx->kokoro_gguf_path = gguf_path;
    ctx->kokoro_voice_path = voice_bin_path;
    ctx->kokoro_loaded = true;
    return ELIZA_OK;
#else
    (void) ctx; (void) gguf_path; (void) voice_bin_path; (void) style_dim;
    eliza_set_error(out_error,
        "[libelizainference] kokoro not built (ELIZA_ENABLE_KOKORO off)");
    return ELIZA_ERR_NOT_IMPLEMENTED;
#endif
}

int eliza_inference_kokoro_synthesize(
    EliInferenceContext * ctx,
    const char * text,
    size_t text_len,
    float speed,
    float * out_pcm,
    size_t max_samples,
    char ** out_error) {
#ifdef ELIZA_ENABLE_KOKORO
    if (!ctx || !text || !out_pcm) {
        eliza_set_error(out_error, "[libelizainference] kokoro_synthesize: null argument");
        return ELIZA_ERR_INVALID_ARG;
    }
    std::lock_guard<std::mutex> lock(ctx->kokoro_mutex);
    if (!ctx->kokoro_loaded || !ctx->kokoro_model) {
        eliza_set_error(out_error,
            "[libelizainference] kokoro_synthesize: no model loaded (call kokoro_load first)");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (speed <= 0.0f) speed = 1.0f;
    const std::string in(text, text_len ? text_len : std::strlen(text));
    eliza_kokoro::kokoro_audio audio;
    std::string err;
    eliza_kokoro::kokoro_status st = eliza_kokoro::kokoro_synthesize(
        ctx->kokoro_model.get(), ctx->kokoro_voice, in, speed, audio, err);
    if (st != eliza_kokoro::KOKORO_OK) {
        eliza_set_error(out_error,
            std::string("[libelizainference] kokoro_synthesize failed: ") + err);
        return ELIZA_ERR_FFI_FAULT;
    }
    if (audio.samples.size() > max_samples) {
        eliza_set_error(out_error,
            "[libelizainference] kokoro_synthesize: out_pcm too small; required samples=" +
            std::to_string(audio.samples.size()));
        return ELIZA_ERR_INVALID_ARG;
    }
    std::memcpy(out_pcm, audio.samples.data(), audio.samples.size() * sizeof(float));
    return (int) audio.samples.size();
#else
    (void) ctx; (void) text; (void) text_len; (void) speed;
    (void) out_pcm; (void) max_samples;
    eliza_set_error(out_error,
        "[libelizainference] kokoro not built (ELIZA_ENABLE_KOKORO off)");
    return ELIZA_ERR_NOT_IMPLEMENTED;
#endif
}

int eliza_inference_kokoro_sample_rate(EliInferenceContext * ctx) {
#ifdef ELIZA_ENABLE_KOKORO
    if (!ctx) return ELIZA_ERR_INVALID_ARG;
    std::lock_guard<std::mutex> lock(ctx->kokoro_mutex);
    if (!ctx->kokoro_loaded || !ctx->kokoro_model) return ELIZA_ERR_INVALID_ARG;
    return eliza_kokoro::kokoro_sample_rate(ctx->kokoro_model.get());
#else
    (void) ctx;
    return ELIZA_ERR_NOT_IMPLEMENTED;
#endif
}

int eliza_inference_mmap_acquire(
    EliInferenceContext * ctx,
    const char * region_name,
    char ** out_error) {
    if (!ctx) {
        eliza_set_error(out_error, "[libelizainference] mmap_acquire: ctx is NULL");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (!eliza_is_region(region_name)) {
        eliza_set_error(out_error, "[libelizainference] mmap_acquire: invalid region");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (std::strcmp(region_name, "tts") == 0) {
        return eliza_load_tts(ctx, out_error);
    }
    if (std::strcmp(region_name, "asr") == 0) {
        return eliza_load_asr(ctx, out_error);
    }
    return ELIZA_OK;
}

int eliza_inference_mmap_evict(
    EliInferenceContext * ctx,
    const char * region_name,
    char ** out_error) {
    if (!ctx) {
        eliza_set_error(out_error, "[libelizainference] mmap_evict: ctx is NULL");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (!eliza_is_region(region_name)) {
        eliza_set_error(out_error, "[libelizainference] mmap_evict: invalid region");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (std::strcmp(region_name, "tts") == 0) {
        std::lock_guard<std::mutex> lock(ctx->tts_mutex);
        if (ctx->ov) {
            ov_free(ctx->ov);
            ctx->ov = nullptr;
        }
    }
    if (std::strcmp(region_name, "asr") == 0) {
        std::lock_guard<std::mutex> lock(ctx->asr_mutex);
        eliza_free_asr(ctx);
    }
    return ELIZA_OK;
}

int eliza_inference_tts_synthesize(
    EliInferenceContext * ctx,
    const char * text,
    size_t text_len,
    const char * speaker_preset_id,
    float * out_pcm,
    size_t max_samples,
    char ** out_error) {
    if (!ctx || !out_pcm || max_samples == 0) {
        eliza_set_error(out_error, "[libelizainference] tts_synthesize: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (!text || text_len == 0) {
        eliza_set_error(out_error, "[libelizainference] tts_synthesize: text is required");
        return ELIZA_ERR_INVALID_ARG;
    }

    /* Per-op backend seam: a TTS backend (e.g. LiteRT/NPU) serves this when it
     * ships <bundle>/tts/*; otherwise fall through to the in-tree OmniVoice path
     * below. Inert by default (no backend registered). */
    {
        char * be_error = nullptr;
        TtsBackendFactory * be =
            tts_backend_select(llm_backend_context_bundle_dir(ctx), &be_error);
        if (be_error) {
            eliza_set_error(out_error, std::string(be_error));
            std::free(be_error);
            return ELIZA_ERR_BUNDLE_INVALID;
        }
        if (be) {
            return be->tts_synthesize(ctx, text, text_len, speaker_preset_id,
                                      out_pcm, max_samples, out_error);
        }
    }

    std::lock_guard<std::mutex> lock(ctx->tts_mutex);
    if (!ctx->ov) {
        eliza_set_error(out_error, "[libelizainference] tts_synthesize: TTS region is not acquired; call mmap_acquire(\"tts\") after arming voice");
        return ELIZA_ERR_INVALID_ARG;
    }
    ElizaScopedTtsForward forward(ctx);

    std::string text_owned(text, text_len);
    ov_tts_params params;
    ov_tts_default_params(&params);
    eliza_apply_tts_env_overrides(&params);
    params.text = text_owned.c_str();
    /* Default to OmniVoice's auto-voice path. The preset (if any)
     * overwrites params.instruct / ref_audio_tokens / ref_text via
     * eliza_apply_preset_to_params below. */
    params.instruct = "";
    if (speaker_preset_id && speaker_preset_id[0] != '\0') {
        std::string preset_err;
        const EliVoicePreset * preset = nullptr;
        {
            std::lock_guard<std::mutex> preset_lock(ctx->preset_mutex);
            preset = eliza_load_voice_preset_locked(ctx, speaker_preset_id, preset_err);
        }
        if (preset && !preset->empty_payload) {
            eliza_apply_preset_to_params(*preset, &params);
        }
        /* A missing or v1-only preset is not fatal — auto-voice mode
         * still produces audio. The preset_err is only surfaced via
         * out_error when synthesis itself fails. */
    }
    params.cancel = eliza_tts_cancel_requested;
    params.cancel_user_data = ctx;

    ov_audio audio = {};
    ov_status rc = ov_synthesize(ctx->ov, &params, &audio);
    if (rc != OV_STATUS_OK) {
        std::string msg = "[libelizainference] ov_synthesize failed: ";
        msg += ov_last_error();
        ov_audio_free(&audio);
        eliza_set_error(out_error, msg);
        return eliza_map_ov_status(rc);
    }
    if (audio.n_samples < 0 || (size_t) audio.n_samples > max_samples) {
        std::string msg = "[libelizainference] output buffer too small; required samples=" +
            std::to_string(audio.n_samples);
        ov_audio_free(&audio);
        eliza_set_error(out_error, msg);
        return ELIZA_ERR_INVALID_ARG;
    }
    std::memcpy(out_pcm, audio.samples, (size_t) audio.n_samples * sizeof(float));
    int written = audio.n_samples;
    ov_audio_free(&audio);
    return written;
}

/* Shared ASR decode core. Takes the asr_mutex, runs the audio-in/text-out
 * decode, and yields the cleaned transcript. Both eliza_inference_asr_transcribe
 * and eliza_inference_asr_transcribe_timed call this so the decode lives in
 * exactly one place. Returns 0 on success or a negative ELIZA_* code. */
static int eliza_asr_decode_core(
    EliInferenceContext * ctx,
    const float * pcm,
    size_t n_samples,
    int sample_rate_hz,
    size_t max_text_bytes,
    std::string & out_transcript,
    char ** out_error) {
    out_transcript.clear();
    if (n_samples == 0) {
        return 0;
    }

    std::lock_guard<std::mutex> lock(ctx->asr_mutex);
    if (!ctx->asr_model || !ctx->asr_lctx || !ctx->asr_mtmd || !ctx->asr_sampler) {
        eliza_set_error(out_error, "[libelizainference] asr_transcribe: ASR region is not acquired; call mmap_acquire(\"asr\") after arming voice input");
        return ELIZA_ERR_INVALID_ARG;
    }

    std::vector<float> audio = eliza_resample_linear(pcm, n_samples, sample_rate_hz, ctx->asr_sample_rate);
    std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(
        mtmd_bitmap_init_from_audio(audio.size(), audio.data()),
        mtmd_bitmap_free);
    if (!bitmap) {
        eliza_set_error(out_error, "[libelizainference] asr_transcribe: failed to create audio bitmap");
        return ELIZA_ERR_FFI_FAULT;
    }

    std::string prompt = eliza_format_asr_prompt(ctx->asr_model);
    mtmd_input_text text = { prompt.c_str(), true, true };
    const mtmd_bitmap * bitmaps[] = { bitmap.get() };
    std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
        mtmd_input_chunks_init(),
        mtmd_input_chunks_free);
    if (!chunks) {
        eliza_set_error(out_error, "[libelizainference] asr_transcribe: failed to allocate input chunks");
        return ELIZA_ERR_FFI_FAULT;
    }
    int32_t tok_rc = mtmd_tokenize(ctx->asr_mtmd, chunks.get(), &text, bitmaps, 1);
    if (tok_rc != 0) {
        eliza_set_error(out_error, "[libelizainference] asr_transcribe: mtmd_tokenize failed rc=" + std::to_string(tok_rc));
        return ELIZA_ERR_FFI_FAULT;
    }

    llama_memory_clear(llama_get_memory(ctx->asr_lctx), true);
    llama_sampler_reset(ctx->asr_sampler);

    llama_pos n_past = 0;
    int32_t eval_rc = mtmd_helper_eval_chunks(
        ctx->asr_mtmd,
        ctx->asr_lctx,
        chunks.get(),
        n_past,
        0,
        ctx->asr_n_batch,
        true,
        &n_past);
    if (eval_rc != 0) {
        eliza_set_error(out_error, "[libelizainference] asr_transcribe: mtmd_helper_eval_chunks failed rc=" + std::to_string(eval_rc));
        return ELIZA_ERR_FFI_FAULT;
    }

    const llama_vocab * vocab = llama_model_get_vocab(ctx->asr_model);
    std::string transcript;
    transcript.reserve(std::min<size_t>(max_text_bytes, 256));
    const int max_decode_tokens = std::min<int>(
        4096,
        std::max<int>(
            192,
            64 + (int) (((audio.size() + (size_t) ctx->asr_sample_rate - 1) /
                         (size_t) ctx->asr_sample_rate) * 32)));
    bool completed = false;
    for (int i = 0; i < max_decode_tokens; ++i) {
        llama_token token = llama_sampler_sample(ctx->asr_sampler, ctx->asr_lctx, -1);
        if (llama_vocab_is_eog(vocab, token)) {
            completed = true;
            break;
        }
        std::string piece = eliza_llama_token_piece(vocab, token);
        if (!piece.empty()) {
            if (transcript.size() + piece.size() + 1 > max_text_bytes) {
                eliza_set_error(out_error, "[libelizainference] asr_transcribe: output buffer too small");
                return ELIZA_ERR_INVALID_ARG;
            }
            transcript += piece;
            std::string cleaned_partial = eliza_clean_asr_transcript(transcript);
            if (eliza_asr_has_text_payload(cleaned_partial)) {
                // Stop only on a real end-of-turn marker — NOT on the first
                // sentence-final '.'/'?'/'!'. Sentence-final early-stop truncated
                // any multi-sentence utterance after its first clause (e.g.
                // "Hello there. How are you?" -> "Hello there."). Qwen3-ASR emits
                // an EOG / <|im_end|> at the true end of the transcript, which the
                // EOG check above and the sentinels here already catch.
                if (piece.find('\n') != std::string::npos ||
                    transcript.find("<|im_end|>") != std::string::npos ||
                    transcript.find("<|endoftext|>") != std::string::npos ||
                    transcript.find("</s>") != std::string::npos) {
                    completed = true;
                    break;
                }
            }
        }
        llama_sampler_accept(ctx->asr_sampler, token);
        llama_batch batch = llama_batch_get_one(&token, 1);
        int32_t decode_rc = llama_decode(ctx->asr_lctx, batch);
        if (decode_rc != 0) {
            eliza_set_error(out_error, "[libelizainference] asr_transcribe: llama_decode failed rc=" + std::to_string(decode_rc));
            return ELIZA_ERR_FFI_FAULT;
        }
    }
    if (!completed) {
        // Decode hit the token cap without an explicit end marker. The old
        // behaviour hard-failed here, which surfaced as a spurious transcription
        // error on long or unusually-tokenized audio (the cap is sized for
        // typical speech). Return the best-effort transcript instead so the
        // caller gets a usable result; only an EMPTY decode stays an error so a
        // genuine failure is still visible.
        std::string best = eliza_clean_asr_transcript(transcript);
        if (best.empty()) {
            eliza_set_error(out_error,
                "[libelizainference] asr_transcribe: decode produced no text before the token cap");
            return ELIZA_ERR_FFI_FAULT;
        }
        out_transcript = best;
        return 0;
    }
    out_transcript = eliza_clean_asr_transcript(transcript);
    return 0;
}

int eliza_inference_asr_transcribe(
    EliInferenceContext * ctx,
    const float * pcm,
    size_t n_samples,
    int sample_rate_hz,
    char * out_text,
    size_t max_text_bytes,
    char ** out_error) {
    if (!ctx || !pcm || !out_text || max_text_bytes == 0 || sample_rate_hz <= 0) {
        eliza_set_error(out_error, "[libelizainference] asr_transcribe: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }

    /* Per-op backend seam: an ASR backend (e.g. LiteRT/NPU) serves this when it
     * ships <bundle>/asr/*; otherwise fall through to the in-tree ggml path
     * below. Inert by default (no backend registered). */
    {
        char * be_error = nullptr;
        AsrBackendFactory * be =
            asr_backend_select(llm_backend_context_bundle_dir(ctx), &be_error);
        if (be_error) {
            eliza_set_error(out_error, std::string(be_error));
            std::free(be_error);
            return ELIZA_ERR_BUNDLE_INVALID;
        }
        if (be) {
            return be->asr_transcribe(ctx, pcm, n_samples, sample_rate_hz,
                                      out_text, max_text_bytes, out_error);
        }
    }

    std::string transcript;
    int rc = eliza_asr_decode_core(ctx, pcm, n_samples, sample_rate_hz, max_text_bytes, transcript, out_error);
    if (rc < 0) {
        return rc;
    }
    if (transcript.size() + 1 > max_text_bytes) {
        eliza_set_error(out_error, "[libelizainference] asr_transcribe: output buffer too small after transcript normalization");
        return ELIZA_ERR_INVALID_ARG;
    }
    std::memcpy(out_text, transcript.data(), transcript.size());
    out_text[transcript.size()] = '\0';
    return (int) transcript.size();
}

int eliza_inference_asr_timestamps_supported(void) {
    return 1;
}

int eliza_inference_asr_transcribe_timed(
    EliInferenceContext * ctx,
    const float * pcm,
    size_t n_samples,
    int sample_rate_hz,
    char * out_text,
    size_t max_text_bytes,
    int * out_word_start_ms,
    int * out_word_end_ms,
    size_t * io_n_words,
    char ** out_error) {
    if (!ctx || !pcm || !out_text || max_text_bytes == 0 || sample_rate_hz <= 0 ||
        !out_word_start_ms || !out_word_end_ms || !io_n_words) {
        eliza_set_error(out_error, "[libelizainference] asr_transcribe_timed: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    const size_t word_cap = *io_n_words;
    *io_n_words = 0;

    std::string transcript;
    int rc = eliza_asr_decode_core(ctx, pcm, n_samples, sample_rate_hz, max_text_bytes, transcript, out_error);
    if (rc < 0) {
        return rc;
    }
    if (transcript.size() + 1 > max_text_bytes) {
        eliza_set_error(out_error, "[libelizainference] asr_transcribe_timed: output buffer too small after transcript normalization");
        return ELIZA_ERR_INVALID_ARG;
    }
    std::memcpy(out_text, transcript.data(), transcript.size());
    out_text[transcript.size()] = '\0';

    /* Per-word timing: duration-proportional, character-weighted, monotonic over
     * the cleaned transcript words, anchored on the exact input duration. The
     * honest single-model signal — Qwen3-ASR exposes no acoustic frame alignment
     * and flash-attention fuses the cross-attention softmax (see the v12 ABI
     * changelog). Word boundaries are a plain whitespace split, matching the
     * caller's split of out_text. */
    struct WordSpan { size_t begin; size_t end; };
    std::vector<WordSpan> words;
    {
        const size_t n = transcript.size();
        size_t i = 0;
        while (i < n) {
            while (i < n && std::isspace((unsigned char) transcript[i])) ++i;
            if (i >= n) break;
            const size_t begin = i;
            while (i < n && !std::isspace((unsigned char) transcript[i])) ++i;
            words.push_back(WordSpan{ begin, i });
        }
    }
    size_t total_chars = 0;
    for (const auto & w : words) total_chars += (w.end - w.begin);

    const double t_ms = 1000.0 * (double) n_samples / (double) sample_rate_hz;
    size_t written = 0;
    size_t cum = 0;
    for (size_t w = 0; w < words.size() && written < word_cap; ++w) {
        const size_t wlen = words[w].end - words[w].begin;
        const double start = total_chars > 0 ? t_ms * (double) cum / (double) total_chars : 0.0;
        cum += wlen;
        const double end = total_chars > 0 ? t_ms * (double) cum / (double) total_chars : t_ms;
        out_word_start_ms[written] = (int) std::llround(start);
        out_word_end_ms[written] = (int) std::llround(end);
        ++written;
    }
    *io_n_words = written;
    return (int) transcript.size();
}

/* ---- Streaming ASR (ABI v2) ---------------------------------------- *
 *
 * The fused build ships the v1 batch \`eliza_inference_asr_transcribe\`
 * decoder above; the windowed streaming-session decoder is not yet wired
 * (W7). Per packages/inference/AGENTS.md §3 we do NOT fake it — the
 * capability probe returns 0 so EngineVoiceBridge / StreamingTranscriber
 * pick the fused batch ASR adapter instead of opening a session that would
 * only return ELIZA_ERR_NOT_IMPLEMENTED.
 * These symbols exist so the ABI surface is complete and the loader's
 * version check (ffi-bindings.ts expects v3) succeeds.
 */

int eliza_inference_asr_stream_supported(void) {
    return 0;
}

EliAsrStream * eliza_inference_asr_stream_open(
    EliInferenceContext * ctx,
    int sample_rate_hz,
    char ** out_error) {
    (void) ctx;
    (void) sample_rate_hz;
    eliza_set_error(out_error,
        "[libelizainference] streaming ASR session API is not implemented in this build "
        "(eliza_inference_asr_stream_supported() == 0); use the batch transcribe path");
    return nullptr;
}

int eliza_inference_asr_stream_feed(
    EliAsrStream * stream,
    const float * pcm,
    size_t n_samples,
    char ** out_error) {
    (void) stream;
    (void) pcm;
    (void) n_samples;
    eliza_set_error(out_error, "[libelizainference] streaming ASR is not implemented in this build");
    return ELIZA_ERR_NOT_IMPLEMENTED;
}

int eliza_inference_asr_stream_partial(
    EliAsrStream * stream,
    char * out_text,
    size_t max_text_bytes,
    int * out_tokens,
    size_t * io_n_tokens,
    char ** out_error) {
    (void) stream;
    (void) out_text;
    (void) max_text_bytes;
    (void) out_tokens;
    if (io_n_tokens) *io_n_tokens = 0;
    eliza_set_error(out_error, "[libelizainference] streaming ASR is not implemented in this build");
    return ELIZA_ERR_NOT_IMPLEMENTED;
}

int eliza_inference_asr_stream_finish(
    EliAsrStream * stream,
    char * out_text,
    size_t max_text_bytes,
    int * out_tokens,
    size_t * io_n_tokens,
    char ** out_error) {
    (void) stream;
    (void) out_text;
    (void) max_text_bytes;
    (void) out_tokens;
    if (io_n_tokens) *io_n_tokens = 0;
    eliza_set_error(out_error, "[libelizainference] streaming ASR is not implemented in this build");
    return ELIZA_ERR_NOT_IMPLEMENTED;
}

void eliza_inference_asr_stream_close(EliAsrStream * stream) {
    (void) stream;
}

/* ---- Streaming TTS + DFlash verifier callback (ABI v2) ------------- *
 *
 * TTS streaming is backed by OmniVoice's real \`ov_tts_params.on_chunk\`
 * path and cooperative cancel hook. The native DFlash verifier-event
 * callback is still not wired in this generated adapter, so the JS
 * scheduler continues to synthesize verifier events from llama-server SSE
 * deltas until that text-generation path moves in-process.
 */

int eliza_inference_tts_stream_supported(void) {
    return OV_ABI_VERSION >= 2 ? 1 : 0;
}

int eliza_inference_tts_synthesize_stream(
    EliInferenceContext * ctx,
    const char * text,
    size_t text_len,
    const char * speaker_preset_id,
    eliza_tts_chunk_cb on_chunk,
    void * user_data,
    char ** out_error) {
    if (!ctx || !on_chunk) {
        eliza_set_error(out_error, "[libelizainference] tts_synthesize_stream: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (!text || text_len == 0) {
        eliza_set_error(out_error, "[libelizainference] tts_synthesize_stream: text is required");
        return ELIZA_ERR_INVALID_ARG;
    }

    std::lock_guard<std::mutex> lock(ctx->tts_mutex);
    if (!ctx->ov) {
        eliza_set_error(out_error, "[libelizainference] tts_synthesize_stream: TTS region is not acquired; call mmap_acquire(\"tts\") after arming voice");
        return ELIZA_ERR_INVALID_ARG;
    }
    ElizaScopedTtsForward forward(ctx);

    std::string text_owned(text, text_len);
    ov_tts_params params;
    ov_tts_default_params(&params);
    eliza_apply_tts_env_overrides(&params);
    params.text = text_owned.c_str();
    /* Default to OmniVoice's auto-voice path. The preset (if any)
     * overwrites params.instruct / ref_audio_tokens / ref_text via
     * eliza_apply_preset_to_params below. */
    params.instruct = "";
    if (speaker_preset_id && speaker_preset_id[0] != '\0') {
        std::string preset_err;
        const EliVoicePreset * preset = nullptr;
        {
            std::lock_guard<std::mutex> preset_lock(ctx->preset_mutex);
            preset = eliza_load_voice_preset_locked(ctx, speaker_preset_id, preset_err);
        }
        if (preset && !preset->empty_payload) {
            eliza_apply_preset_to_params(*preset, &params);
        }
        /* A missing or v1-only preset is not fatal — auto-voice mode
         * still produces audio. The preset_err is only surfaced via
         * out_error when synthesis itself fails. */
    }
    params.cancel = eliza_tts_cancel_requested;
    params.cancel_user_data = ctx;

    ElizaTtsStreamState state = {
        ctx,
        on_chunk,
        user_data,
        false,
    };
    params.on_chunk = eliza_tts_stream_chunk;
    params.on_chunk_user_data = &state;

    ov_status rc = ov_synthesize(ctx->ov, &params, nullptr);
    const bool cancelled =
        rc == OV_STATUS_CANCELLED ||
        state.callback_cancelled ||
        ctx->tts_cancel.load(std::memory_order_acquire);
    (void) on_chunk(nullptr, 0, 1, user_data);
    if (cancelled) {
        return ELIZA_ERR_CANCELLED;
    }
    if (rc != OV_STATUS_OK) {
        std::string msg = "[libelizainference] ov_synthesize(stream) failed: ";
        msg += ov_last_error();
        eliza_set_error(out_error, msg);
        return eliza_map_ov_status(rc);
    }
    return ELIZA_OK;
}

int eliza_inference_cancel_tts(
    EliInferenceContext * ctx,
    char ** out_error) {
    (void) out_error;
    if (ctx) {
        ctx->tts_cancel.store(true, std::memory_order_release);
    }
    // Cancelling nothing is not an error.
    return ELIZA_OK;
}

int eliza_inference_set_verifier_callback(
    EliInferenceContext * ctx,
    eliza_verifier_cb cb,
    void * user_data,
    char ** out_error) {
    (void) ctx;
    (void) cb;
    (void) user_data;
    eliza_set_error(out_error,
        "[libelizainference] native DFlash verifier callback is not implemented in this build; "
        "the JS scheduler synthesizes verifier events from llama-server streaming deltas");
    return ELIZA_ERR_NOT_IMPLEMENTED;
}

/* ---- OmniVoice reference encode (ABI v4) -------------------------- *
 *
 * Thin wrapper around ov_encode_reference. The TTS region must have
 * been acquired (\`mmap_acquire("tts")\`) before the call. The library
 * malloc-allocates the token buffer; callers release it via
 * eliza_inference_free_tokens.
 */
int eliza_inference_encode_reference(
    EliInferenceContext * ctx,
    const float * pcm,
    size_t n_samples,
    int sample_rate_hz,
    int * out_K,
    int * out_ref_T,
    int ** out_tokens,
    char ** out_error) {
    if (!ctx || !pcm || !out_K || !out_ref_T || !out_tokens) {
        eliza_set_error(out_error, "[libelizainference] encode_reference: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (n_samples == 0) {
        eliza_set_error(out_error, "[libelizainference] encode_reference: n_samples must be > 0");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (sample_rate_hz != 24000) {
        eliza_set_error(out_error,
            "[libelizainference] encode_reference: sample_rate_hz must be 24000 (got " +
            std::to_string(sample_rate_hz) + "); caller is responsible for upstream resample");
        return ELIZA_ERR_INVALID_ARG;
    }

    std::lock_guard<std::mutex> lock(ctx->tts_mutex);
    if (!ctx->ov) {
        eliza_set_error(out_error,
            "[libelizainference] encode_reference: TTS region is not acquired; "
            "call mmap_acquire(\"tts\") after arming voice");
        return ELIZA_ERR_INVALID_ARG;
    }

    int32_t * tokens = nullptr;
    int K = 0;
    int ref_T = 0;
    ov_status rc = ov_encode_reference(ctx->ov, pcm, (int) n_samples,
                                       &tokens, &K, &ref_T);
    if (rc != OV_STATUS_OK) {
        std::string msg = "[libelizainference] ov_encode_reference failed: ";
        msg += ov_last_error();
        eliza_set_error(out_error, msg);
        if (tokens) std::free(tokens);
        return eliza_map_ov_status(rc);
    }
    *out_tokens = tokens;
    *out_K = K;
    *out_ref_T = ref_T;
    return ELIZA_OK;
}

void eliza_inference_free_tokens(int * tokens) {
    if (tokens) std::free(tokens);
}

/* ---- Native VAD (ABI v3 surface, real backend at ABI v7) ----------- *
 *
 * Thin wrapper over the vendored Silero v5 scalar-C forward graph
 * (silero_vad_open/process/reset_state/close). The fused build resolves
 * the Silero GGUF from the context bundle (`<bundle_dir>/vad/` — the
 * single GGUF there, conventionally `silero-vad-v5.gguf`), then streams
 * 512-sample 16 kHz windows through the standalone runtime, emitting one
 * P(speech) ∈ [0, 1] per window. The standalone exposes an in-place
 * `silero_vad_reset_state`, so unlike the wake-word wrapper this maps
 * reset 1:1 (no close+reopen). The JS binding routes VAD through this
 * surface when `eliza_inference_vad_supported() == 1`.
 */

struct EliVad {
    silero_vad_handle handle = nullptr;
};

int eliza_inference_vad_supported(void) {
    return 1;
}

EliVad * eliza_inference_vad_open(
    EliInferenceContext * ctx,
    int sample_rate_hz,
    char ** out_error) {
    if (!ctx) {
        eliza_set_error(out_error,
            "[libelizainference] vad_open: ctx is required");
        return nullptr;
    }
    if (sample_rate_hz != SILERO_VAD_SAMPLE_RATE_HZ) {
        eliza_set_error(out_error,
            std::string("[libelizainference] vad_open: sample_rate_hz must be ") +
            std::to_string(SILERO_VAD_SAMPLE_RATE_HZ));
        return nullptr;
    }

    const std::string path =
        eliza_pick_one_gguf(std::filesystem::path(ctx->bundle_dir), "vad");
    if (path.empty()) {
        eliza_set_error(out_error,
            std::string("[libelizainference] vad_open: no GGUF found under ") +
            (std::filesystem::path(ctx->bundle_dir) / "vad").string() +
            " (expected silero-vad-v5.gguf)");
        return nullptr;
    }

    EliVad * vad = new (std::nothrow) EliVad();
    if (!vad) {
        eliza_set_error(out_error, "[libelizainference] vad_open: out of memory");
        return nullptr;
    }
    int rc = silero_vad_open(path.c_str(), &vad->handle);
    if (rc != 0 || !vad->handle) {
        eliza_set_error(out_error,
            std::string("[libelizainference] vad_open: standalone runtime "
                        "failed to load ") + path +
            " (errno-style rc=" + std::to_string(rc) + ")");
        delete vad;
        return nullptr;
    }
    return vad;
}

int eliza_inference_vad_process(
    EliVad * vad,
    const float * pcm,
    size_t n_samples,
    float * out_probability,
    char ** out_error) {
    if (!vad || !vad->handle || !pcm || !out_probability) {
        eliza_set_error(out_error,
            "[libelizainference] vad_process: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    float prob = 0.0f;
    int rc = silero_vad_process(vad->handle, pcm, n_samples, &prob);
    if (rc != 0) {
        eliza_set_error(out_error,
            std::string("[libelizainference] vad_process: forward failed "
                        "(errno-style rc=") + std::to_string(rc) + ")");
        return rc == -EINVAL ? ELIZA_ERR_INVALID_ARG : ELIZA_ERR_FFI_FAULT;
    }
    *out_probability = prob;
    return ELIZA_OK;
}

int eliza_inference_vad_reset(
    EliVad * vad,
    char ** out_error) {
    if (!vad || !vad->handle) {
        eliza_set_error(out_error,
            "[libelizainference] vad_reset: invalid session");
        return ELIZA_ERR_INVALID_ARG;
    }
    int rc = silero_vad_reset_state(vad->handle);
    if (rc != 0) {
        eliza_set_error(out_error,
            std::string("[libelizainference] vad_reset: reset failed "
                        "(errno-style rc=") + std::to_string(rc) + ")");
        return ELIZA_ERR_FFI_FAULT;
    }
    return ELIZA_OK;
}

void eliza_inference_vad_close(EliVad * vad) {
    if (!vad) return;
    if (vad->handle) silero_vad_close(vad->handle);
    delete vad;
}

/* ---- Native wake-word (ABI v5) ------------------------------------ *
 *
 * Thin wrapper that owns a context-anchored session and drives the
 * vendored openWakeWord scalar-C runtime (wakeword_open/process/close).
 * The fused build resolves the three head GGUFs from the context bundle
 * + the head name. `reset` is implemented as close+reopen of the
 * standalone session (the standalone runtime has no in-place reset; the
 * streaming rings live on its private session struct) — that fully
 * clears mel/embedding streaming state per the ABI contract.
 */

struct EliWakeWord {
    wakeword_handle handle = nullptr;
    /* Retained for reset (close+reopen). */
    std::string melspec_path;
    std::string embedding_path;
    std::string classifier_path;
};

/* Resolve <bundle_dir>/wake/<head>.{melspec,embedding,classifier}.gguf.
 * Returns false (with *out_error) when any of the three is missing. */
static bool eliza_resolve_wakeword_ggufs(
    const std::filesystem::path & bundle_dir,
    const std::string & head,
    std::string & melspec,
    std::string & embedding,
    std::string & classifier,
    char ** out_error) {
    const std::filesystem::path wake_dir = bundle_dir / "wake";
    const std::filesystem::path mel = wake_dir / (head + ".melspec.gguf");
    const std::filesystem::path emb = wake_dir / (head + ".embedding.gguf");
    const std::filesystem::path cls = wake_dir / (head + ".classifier.gguf");
    std::error_code ec;
    if (!std::filesystem::exists(mel, ec) ||
        !std::filesystem::exists(emb, ec) ||
        !std::filesystem::exists(cls, ec)) {
        eliza_set_error(out_error,
            std::string("[libelizainference] wake-word GGUFs not found for head '") +
            head + "' under " + wake_dir.string() +
            " (expected " + head + ".{melspec,embedding,classifier}.gguf)");
        return false;
    }
    melspec = mel.string();
    embedding = emb.string();
    classifier = cls.string();
    return true;
}

int eliza_inference_wakeword_supported(void) {
    return 1;
}

EliWakeWord * eliza_inference_wakeword_open(
    EliInferenceContext * ctx,
    int sample_rate_hz,
    const char * head_name,
    char ** out_error) {
    if (!ctx) {
        eliza_set_error(out_error,
            "[libelizainference] wakeword_open: ctx is required");
        return nullptr;
    }
    if (sample_rate_hz != WAKEWORD_SAMPLE_RATE) {
        eliza_set_error(out_error,
            std::string("[libelizainference] wakeword_open: sample_rate_hz must be ") +
            std::to_string(WAKEWORD_SAMPLE_RATE));
        return nullptr;
    }
    if (!head_name || head_name[0] == '\0') {
        eliza_set_error(out_error,
            "[libelizainference] wakeword_open: head_name is required");
        return nullptr;
    }

    EliWakeWord * wake = new (std::nothrow) EliWakeWord();
    if (!wake) {
        eliza_set_error(out_error, "[libelizainference] wakeword_open: out of memory");
        return nullptr;
    }
    if (!eliza_resolve_wakeword_ggufs(std::filesystem::path(ctx->bundle_dir),
                                      head_name,
                                      wake->melspec_path,
                                      wake->embedding_path,
                                      wake->classifier_path,
                                      out_error)) {
        delete wake;
        return nullptr;
    }

    int rc = wakeword_open(wake->melspec_path.c_str(),
                           wake->embedding_path.c_str(),
                           wake->classifier_path.c_str(),
                           &wake->handle);
    if (rc != 0 || !wake->handle) {
        eliza_set_error(out_error,
            std::string("[libelizainference] wakeword_open: standalone runtime "
                        "failed to load head '") + head_name +
            "' (errno-style rc=" + std::to_string(rc) + ")");
        delete wake;
        return nullptr;
    }
    return wake;
}

int eliza_inference_wakeword_score(
    EliWakeWord * wake,
    const float * pcm,
    size_t n_samples,
    float * out_probability,
    char ** out_error) {
    if (!wake || !wake->handle || !pcm || !out_probability) {
        eliza_set_error(out_error,
            "[libelizainference] wakeword_score: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    float score = 0.0f;
    int rc = wakeword_process(wake->handle, pcm, n_samples, &score);
    if (rc != 0) {
        eliza_set_error(out_error,
            std::string("[libelizainference] wakeword_score: forward failed "
                        "(errno-style rc=") + std::to_string(rc) + ")");
        return ELIZA_ERR_FFI_FAULT;
    }
    *out_probability = score;
    return ELIZA_OK;
}

int eliza_inference_wakeword_reset(
    EliWakeWord * wake,
    char ** out_error) {
    if (!wake) {
        eliza_set_error(out_error,
            "[libelizainference] wakeword_reset: invalid session");
        return ELIZA_ERR_INVALID_ARG;
    }
    /* The standalone runtime has no in-place reset; close+reopen clears
     * every streaming ring (mel state, mel ring, embedding ring, last
     * score) deterministically. */
    if (wake->handle) {
        wakeword_close(wake->handle);
        wake->handle = nullptr;
    }
    int rc = wakeword_open(wake->melspec_path.c_str(),
                           wake->embedding_path.c_str(),
                           wake->classifier_path.c_str(),
                           &wake->handle);
    if (rc != 0 || !wake->handle) {
        eliza_set_error(out_error,
            std::string("[libelizainference] wakeword_reset: reopen failed "
                        "(errno-style rc=") + std::to_string(rc) + ")");
        return ELIZA_ERR_FFI_FAULT;
    }
    return ELIZA_OK;
}

void eliza_inference_wakeword_close(EliWakeWord * wake) {
    if (!wake) return;
    if (wake->handle) wakeword_close(wake->handle);
    delete wake;
}

/* ---- Native speaker encoder (ABI v6) ------------------------------ *
 *
 * Thin wrapper over the vendored WeSpeaker ResNet34-LM scalar-C forward
 * graph (voice_speaker_open/embed/close). Resolves the encoder GGUF
 * from the bundle's speaker dir or an explicit path.
 */

struct EliSpeaker {
    voice_speaker_handle handle = nullptr;
};

int eliza_inference_speaker_supported(void) {
    return 1;
}

EliSpeaker * eliza_inference_speaker_open(
    EliInferenceContext * ctx,
    const char * gguf_path,
    char ** out_error) {
    if (!ctx) {
        eliza_set_error(out_error,
            "[libelizainference] speaker_open: ctx is required");
        return nullptr;
    }
    std::string path;
    if (gguf_path && gguf_path[0] != '\0') {
        path = gguf_path;
    } else {
        path = eliza_pick_one_gguf(std::filesystem::path(ctx->bundle_dir), "speaker");
        if (path.empty()) {
            eliza_set_error(out_error,
                std::string("[libelizainference] speaker_open: no GGUF found under ") +
                (std::filesystem::path(ctx->bundle_dir) / "speaker").string() +
                " and no explicit gguf_path given");
            return nullptr;
        }
    }

    EliSpeaker * speaker = new (std::nothrow) EliSpeaker();
    if (!speaker) {
        eliza_set_error(out_error, "[libelizainference] speaker_open: out of memory");
        return nullptr;
    }
    int rc = voice_speaker_open(path.c_str(), &speaker->handle);
    if (rc != 0 || !speaker->handle) {
        eliza_set_error(out_error,
            std::string("[libelizainference] speaker_open: standalone encoder "
                        "failed to load ") + path +
            " (errno-style rc=" + std::to_string(rc) + ")");
        delete speaker;
        return nullptr;
    }
    return speaker;
}

int eliza_inference_speaker_embed(
    EliSpeaker * speaker,
    const float * pcm,
    size_t n_samples,
    float * out_embedding,
    char ** out_error) {
    if (!speaker || !speaker->handle || !pcm || !out_embedding) {
        eliza_set_error(out_error,
            "[libelizainference] speaker_embed: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    int rc = voice_speaker_embed(speaker->handle, pcm, n_samples, out_embedding);
    if (rc != 0) {
        eliza_set_error(out_error,
            std::string("[libelizainference] speaker_embed: forward failed "
                        "(errno-style rc=") + std::to_string(rc) + ")");
        return rc == -EINVAL ? ELIZA_ERR_INVALID_ARG : ELIZA_ERR_FFI_FAULT;
    }
    return ELIZA_OK;
}

void eliza_inference_speaker_free(float * embedding) {
    (void) embedding; /* caller-owned in _embed; provided for ABI symmetry */
}

void eliza_inference_speaker_close(EliSpeaker * speaker) {
    if (!speaker) return;
    if (speaker->handle) voice_speaker_close(speaker->handle);
    delete speaker;
}

/* ---- Native diarizer (ABI v6) ------------------------------------- *
 *
 * Thin wrapper over the vendored pyannote-segmentation-3.0 scalar-C
 * forward graph (voice_diarizer_open/segment/close). Resolves the
 * diarizer GGUF from the bundle's diariz dir or an explicit path.
 */

struct EliDiariz {
    voice_diarizer_handle handle = nullptr;
};

int eliza_inference_diariz_supported(void) {
    return 1;
}

EliDiariz * eliza_inference_diariz_open(
    EliInferenceContext * ctx,
    const char * gguf_path,
    char ** out_error) {
    if (!ctx) {
        eliza_set_error(out_error,
            "[libelizainference] diariz_open: ctx is required");
        return nullptr;
    }
    std::string path;
    if (gguf_path && gguf_path[0] != '\0') {
        path = gguf_path;
    } else {
        path = eliza_pick_one_gguf(std::filesystem::path(ctx->bundle_dir), "diariz");
        if (path.empty()) {
            eliza_set_error(out_error,
                std::string("[libelizainference] diariz_open: no GGUF found under ") +
                (std::filesystem::path(ctx->bundle_dir) / "diariz").string() +
                " and no explicit gguf_path given");
            return nullptr;
        }
    }

    EliDiariz * diariz = new (std::nothrow) EliDiariz();
    if (!diariz) {
        eliza_set_error(out_error, "[libelizainference] diariz_open: out of memory");
        return nullptr;
    }
    int rc = voice_diarizer_open(path.c_str(), &diariz->handle);
    if (rc != 0 || !diariz->handle) {
        eliza_set_error(out_error,
            std::string("[libelizainference] diariz_open: standalone diarizer "
                        "failed to load ") + path +
            " (errno-style rc=" + std::to_string(rc) + ")");
        delete diariz;
        return nullptr;
    }
    return diariz;
}

int eliza_inference_diariz_segment(
    EliDiariz * diariz,
    const float * pcm,
    size_t n_samples,
    int8_t * out_labels,
    size_t * io_n_labels,
    char ** out_error) {
    if (!diariz || !diariz->handle || !pcm || !out_labels || !io_n_labels) {
        eliza_set_error(out_error,
            "[libelizainference] diariz_segment: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    int rc = voice_diarizer_segment(diariz->handle, pcm, n_samples,
                                    out_labels, io_n_labels);
    if (rc == -ENOSPC) {
        eliza_set_error(out_error,
            std::string("[libelizainference] diariz_segment: labels buffer too "
                        "small; need ") + std::to_string(*io_n_labels) + " frames");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (rc != 0) {
        eliza_set_error(out_error,
            std::string("[libelizainference] diariz_segment: forward failed "
                        "(errno-style rc=") + std::to_string(rc) + ")");
        return rc == -EINVAL ? ELIZA_ERR_INVALID_ARG : ELIZA_ERR_FFI_FAULT;
    }
    return ELIZA_OK;
}

void eliza_inference_diariz_close(EliDiariz * diariz) {
    if (!diariz) return;
    if (diariz->handle) voice_diarizer_close(diariz->handle);
    delete diariz;
}

/* ---- Streaming LLM (text generation, ABI v4) --------------------- *
 *
 * Real in-process forward passes against the bundle's text GGUF. Pull-based
 * surface (open → prefill → next* → close) matching the Bun loader in
 * ffi-bindings.ts. Grammar-constrained sampling forces the structured-reply
 * envelope when cfg->gbnf_grammar is set.
 */

int eliza_inference_llm_stream_supported(void) {
    return 1;
}

/* ABI v8 capability probes: this build wires the same-file / separate-drafter
 * MTP speculative engine (reusing common/speculative.cpp's DRAFT_MTP) and the
 * KV-cache quant pass-through, so both report supported. */
int eliza_inference_llm_mtp_supported(void) {
    return 1;
}
int eliza_inference_llm_kv_quant_supported(void) {
    return 1;
}

EliLlmStream * eliza_inference_llm_stream_open(
    EliInferenceContext * ctx,
    const eliza_llm_stream_config_t * cfg,
    char ** out_error) {
    if (!ctx || !cfg) {
        eliza_set_error(out_error,
            "[libelizainference] llm_stream_open: ctx and cfg are required");
        return nullptr;
    }

    /* Multi-backend seam (M3): an alternate in-process runtime (LiteRT-LM /
     * MLX-CoreML) may serve this bundle. The selector returns nullptr with NO
     * error to keep the in-tree llama.cpp path below; nullptr WITH an error is a
     * hard env-select failure to propagate. */
    {
        char * sel_err = nullptr;
        LlmBackendFactory * factory =
            llm_backend_select(ctx->bundle_dir.c_str(), cfg, &sel_err);
        if (!factory && sel_err) {
            if (out_error) {
                *out_error = sel_err;
            } else {
                eliza_inference_free_string(sel_err);
            }
            return nullptr;
        }
        if (factory) {
            EliLlmStream * bstream = new (std::nothrow) EliLlmStream();
            if (!bstream) {
                eliza_set_error(out_error,
                    "[libelizainference] llm_stream_open: out of memory");
                return nullptr;
            }
            bstream->ctx = ctx;
            bstream->max_tokens = cfg->max_tokens > 0 ? cfg->max_tokens : 0;
            bstream->backend = factory->open(ctx, cfg, out_error);
            if (!bstream->backend) {
                delete bstream;
                return nullptr;
            }
            return bstream;
        }
    }

    llama_model * model = nullptr;
    {
        std::lock_guard<std::mutex> lock(ctx->llm_mutex);
        int rc = eliza_load_llm_model_locked(ctx, cfg->n_gpu_layers, out_error);
        if (rc != ELIZA_OK) return nullptr;
        model = ctx->llm_model;
    }

    EliLlmStream * stream = new (std::nothrow) EliLlmStream();
    if (!stream) {
        eliza_set_error(out_error,
            "[libelizainference] llm_stream_open: out of memory");
        return nullptr;
    }
    stream->ctx = ctx;
    stream->max_tokens = cfg->max_tokens > 0 ? cfg->max_tokens : 0;

    /* Build the target context params once; shared by the plain path and the
     * MTP engine's target context so both decode identically (same n_ctx /
     * batch / threads / flash-attn / KV-quant). */
    llama_context_params cparams = llama_context_default_params();
    const int n_ctx_train = llama_model_n_ctx_train(model);
    int n_ctx = eliza_int_env_or_default("ELIZA_LLM_N_CTX", 8192);
    if (n_ctx_train > 0 && n_ctx > n_ctx_train) n_ctx = n_ctx_train;
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = (uint32_t) eliza_int_env_or_default("ELIZA_LLM_N_BATCH", 512);
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_threads = eliza_thread_count(false);
    cparams.n_threads_batch = eliza_thread_count(true);
    cparams.flash_attn_type = eliza_llm_flash_attn_type();

    /* KV-cache quant (ABI v8): map names -> ggml_type and set type_k/type_v.
     * NULL/empty leaves the f16 default. An unrecognized name is a hard error
     * (no silent fallback). */
    const ggml_type type_k = eliza_kv_cache_type(cfg->cache_type_k, out_error);
    if (type_k == GGML_TYPE_COUNT) { delete stream; return nullptr; }
    const ggml_type type_v = eliza_kv_cache_type(cfg->cache_type_v, out_error);
    if (type_v == GGML_TYPE_COUNT) { delete stream; return nullptr; }
    cparams.type_k = type_k;
    cparams.type_v = type_v;

    /* MTP speculative decoding (ABI v8). Requested when draft bounds are > 0.
     *   - mtp_drafter_path set  -> separate-drafter MTP (hard error on fail).
     *   - mtp_drafter_path NULL -> same-file MTP over the target's NextN head
     *     (falls back to the plain loop when the model has no NextN head,
     *     mirroring desktop-llama-adapter.ts's mtpEngine===0 fallback). */
    const bool mtp_requested = cfg->draft_min > 0 && cfg->draft_max > 0;
    const std::string drafter_path =
        (cfg->mtp_drafter_path && cfg->mtp_drafter_path[0] != '\0')
            ? std::string(cfg->mtp_drafter_path)
            : std::string();
    if (mtp_requested) {
        char * mtp_err = nullptr;
        eliza_mtp::Engine * engine =
            eliza_mtp::create_engine(model, cparams, cfg, drafter_path, &mtp_err);
        if (engine) {
            std::free(mtp_err);
            stream->mtp = engine;
            stream->mtp_step_buf.assign((size_t) cfg->draft_max + 2, 0);
            return stream;
        }
        /* A separate drafter is an explicit config that MUST work — surface
         * the failure. A same-file attempt that fails (no NextN head) falls
         * back to the plain decode loop below. */
        if (!drafter_path.empty()) {
            eliza_set_error(out_error,
                mtp_err
                    ? std::string(mtp_err)
                    : "[libelizainference] llm_stream_open: separate-drafter "
                      "MTP engine failed to initialize");
            std::free(mtp_err);
            delete stream;
            return nullptr;
        }
        std::free(mtp_err);
        /* fall through to the plain non-speculative loop */
    }

    stream->lctx = llama_init_from_model(model, cparams);
    if (!stream->lctx) {
        delete stream;
        eliza_set_error(out_error,
            "[libelizainference] llm_stream_open: failed to init llama context");
        return nullptr;
    }

    stream->sampler = eliza_build_llm_sampler_chain(model, cfg, out_error);
    if (!stream->sampler) {
        llama_free(stream->lctx);
        delete stream;
        return nullptr;
    }

    return stream;
}

int eliza_inference_llm_stream_prefill(
    EliLlmStream * stream,
    const int32_t * token_ids,
    size_t num_tokens,
    char ** out_error) {
    if (stream && stream->backend) {
        return stream->backend->prefill(token_ids, num_tokens, out_error);
    }
    if (!stream || (!stream->lctx && !stream->mtp)) {
        eliza_set_error(out_error,
            "[libelizainference] llm_stream_prefill: invalid session");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (num_tokens == 0) return ELIZA_OK;
    if (!token_ids) {
        eliza_set_error(out_error,
            "[libelizainference] llm_stream_prefill: token_ids is NULL");
        return ELIZA_ERR_INVALID_ARG;
    }

    /* MTP path: the engine prefills the prompt, seeds the speculative state,
     * and samples the first token (emitted first by `_next`). */
    if (stream->mtp) {
        const int rc =
            eliza_mtp::prefill(stream->mtp, token_ids, (int32_t) num_tokens);
        if (rc < 0) {
            eliza_set_error(out_error,
                "[libelizainference] llm_stream_prefill: MTP prefill failed rc=" +
                std::to_string(rc));
            return rc;
        }
        stream->mtp_first_token = rc; /* prefill returns the first token id */
        return ELIZA_OK;
    }

    /* Decode the prompt in n_batch-sized chunks. The final token in the
     * prompt carries logits (it seeds the first sample); intermediate
     * tokens do not, so we only request logits on the last position. */
    std::vector<llama_token> tokens(token_ids, token_ids + num_tokens);
    const int n_batch = (int) llama_n_batch(stream->lctx);
    const size_t total = tokens.size();
    for (size_t off = 0; off < total; off += (size_t) n_batch) {
        if (stream->cancel.load(std::memory_order_acquire)) {
            eliza_set_error(out_error,
                "[libelizainference] llm_stream_prefill: cancelled");
            return ELIZA_ERR_CANCELLED;
        }
        const size_t chunk = std::min((size_t) n_batch, total - off);
        llama_batch batch = llama_batch_get_one(tokens.data() + off, (int32_t) chunk);
        int32_t rc = llama_decode(stream->lctx, batch);
        if (rc != 0) {
            eliza_set_error(out_error,
                "[libelizainference] llm_stream_prefill: llama_decode failed rc=" +
                std::to_string(rc));
            return ELIZA_ERR_FFI_FAULT;
        }
        stream->n_past += (int) chunk;
    }
    return ELIZA_OK;
}

int eliza_inference_llm_stream_next(
    EliLlmStream * stream,
    int32_t * tokens_out,
    size_t tokens_cap,
    size_t * num_tokens_out,
    char * text_out,
    size_t text_cap,
    int32_t * drafter_drafted_out,
    int32_t * drafter_accepted_out,
    char ** out_error) {
    if (num_tokens_out) *num_tokens_out = 0;
    if (drafter_drafted_out) *drafter_drafted_out = 0;
    if (drafter_accepted_out) *drafter_accepted_out = 0;
    if (text_out && text_cap > 0) text_out[0] = '\0';

    if (stream && stream->backend) {
        return stream->backend->next(tokens_out, tokens_cap, num_tokens_out,
                                     text_out, text_cap, drafter_drafted_out,
                                     drafter_accepted_out, out_error);
    }
    if (!stream || (!stream->mtp && (!stream->lctx || !stream->sampler))) {
        eliza_set_error(out_error,
            "[libelizainference] llm_stream_next: invalid session");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (!tokens_out || tokens_cap == 0 || !text_out || text_cap == 0) {
        eliza_set_error(out_error,
            "[libelizainference] llm_stream_next: output buffers are required");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (stream->eos) {
        return 1; /* already finished */
    }

    /* ---- MTP speculative path ------------------------------------- *
     * Emit the prefill's first token, then drive draft->verify->accept steps
     * until the per-call token / text-byte budget is met or EOS is reached.
     * Each engine step commits a multi-token accepted prefix. */
    if (stream->mtp) {
        eliza_mtp::Engine * e = stream->mtp;
        const llama_vocab * vocab = llama_model_get_vocab(e->model_tgt);

        std::string text;
        size_t produced = 0;
        bool final_step = false;

        size_t step_cap = tokens_cap;
        if (stream->max_tokens > 0) {
            const size_t remaining = (size_t) std::max(
                0, stream->max_tokens - stream->generated);
            if (remaining == 0) { stream->eos = true; return 1; }
            step_cap = std::min(step_cap, remaining);
        }

        const uint64_t drafted_before  = e->st_drafted;
        const uint64_t accepted_before = e->st_accepted;

        /* Append a single committed token to the output buffers; returns
         * false when it would overflow the text buffer (defer to next call). */
        auto emit = [&](llama_token token) -> bool {
            if (llama_vocab_is_eog(vocab, token)) {
                stream->eos = true;
                final_step = true;
                return false;
            }
            const std::string piece = eliza_llama_token_piece(vocab, token);
            if (!piece.empty() && text.size() + piece.size() + 1 > text_cap) {
                return false; /* overflow — stop the step, keep token uncommitted */
            }
            tokens_out[produced] = (int32_t) token;
            text += piece;
            produced += 1;
            stream->generated += 1;
            if (stream->max_tokens > 0 && stream->generated >= stream->max_tokens) {
                stream->eos = true;
                final_step = true;
            }
            return true;
        };

        /* First token from prefill. */
        if (stream->mtp_first_token >= 0) {
            const llama_token first = stream->mtp_first_token;
            stream->mtp_first_token = -1;
            emit(first);
        }

        while (!final_step && produced < step_cap) {
            if (stream->cancel.load(std::memory_order_acquire)) {
                eliza_set_error(out_error,
                    "[libelizainference] llm_stream_next: cancelled");
                return ELIZA_ERR_CANCELLED;
            }
            const int n = eliza_mtp::step(
                e, stream->mtp_step_buf.data(),
                (int32_t) stream->mtp_step_buf.size());
            if (n < 0) {
                eliza_set_error(out_error,
                    "[libelizainference] llm_stream_next: MTP step failed rc=" +
                    std::to_string(n));
                return ELIZA_ERR_FFI_FAULT;
            }
            if (n == 0) { final_step = true; break; }
            bool overflowed = false;
            for (int i = 0; i < n; ++i) {
                if (produced >= step_cap) { overflowed = true; break; }
                if (!emit(stream->mtp_step_buf[i])) {
                    /* EOS sets final_step; overflow leaves it false but stops. */
                    overflowed = !final_step;
                    break;
                }
                if (final_step) break;
            }
            if (overflowed) break;
        }

        if (text.size() + 1 > text_cap) text.resize(text_cap - 1);
        std::memcpy(text_out, text.data(), text.size());
        text_out[text.size()] = '\0';
        if (num_tokens_out) *num_tokens_out = produced;
        if (drafter_drafted_out)
            *drafter_drafted_out = (int32_t) (e->st_drafted - drafted_before);
        if (drafter_accepted_out)
            *drafter_accepted_out = (int32_t) (e->st_accepted - accepted_before);
        return final_step ? 1 : 0;
    }

    const llama_model * model = llama_get_model(stream->lctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    std::string text;
    size_t produced = 0;
    /* Per-step token cap: the smaller of the caller's token buffer and the
     * remaining budget toward max_tokens (0 = unbounded). */
    size_t step_cap = tokens_cap;
    if (stream->max_tokens > 0) {
        const size_t remaining = (size_t) std::max(
            0, stream->max_tokens - stream->generated);
        if (remaining == 0) {
            stream->eos = true;
            return 1;
        }
        step_cap = std::min(step_cap, remaining);
    }

    bool final_step = false;
    while (produced < step_cap) {
        if (stream->cancel.load(std::memory_order_acquire)) {
            eliza_set_error(out_error,
                "[libelizainference] llm_stream_next: cancelled");
            return ELIZA_ERR_CANCELLED;
        }

        llama_token token = llama_sampler_sample(stream->sampler, stream->lctx, -1);

        if (llama_vocab_is_eog(vocab, token)) {
            stream->eos = true;
            final_step = true;
            break;
        }

        std::string piece = eliza_llama_token_piece(vocab, token);
        /* Stop appending to the step if the piece would overflow the text
         * buffer; commit what we have and return it as a non-final step so
         * the caller pulls again. */
        if (!piece.empty() && text.size() + piece.size() + 1 > text_cap) {
            break;
        }

        /* NOTE: llama_sampler_sample() already calls llama_sampler_accept()
         * internally, advancing the grammar stack. We must NOT accept again
         * here — a double-accept on the grammar sampler empties the stack and
         * throws "Unexpected empty grammar stack". */
        tokens_out[produced] = (int32_t) token;
        text += piece;
        produced += 1;
        stream->generated += 1;

        /* Feed the accepted token back so the next sample sees it. */
        llama_batch batch = llama_batch_get_one(&token, 1);
        int32_t rc = llama_decode(stream->lctx, batch);
        if (rc != 0) {
            eliza_set_error(out_error,
                "[libelizainference] llm_stream_next: llama_decode failed rc=" +
                std::to_string(rc));
            return ELIZA_ERR_FFI_FAULT;
        }
        stream->n_past += 1;

        if (stream->max_tokens > 0 && stream->generated >= stream->max_tokens) {
            stream->eos = true;
            final_step = true;
            break;
        }
    }

    if (produced > step_cap) produced = step_cap; /* defensive */
    if (text.size() + 1 > text_cap) {
        /* Should not happen given the per-piece guard, but never overrun. */
        text.resize(text_cap - 1);
    }
    std::memcpy(text_out, text.data(), text.size());
    text_out[text.size()] = '\0';
    if (num_tokens_out) *num_tokens_out = produced;

    return final_step ? 1 : 0;
}

int eliza_inference_llm_stream_cancel(EliLlmStream * stream) {
    if (stream && stream->backend) {
        return stream->backend->cancel();
    }
    if (stream) {
        stream->cancel.store(true, std::memory_order_release);
    }
    return ELIZA_OK;
}

int eliza_inference_llm_stream_save_slot(
    EliLlmStream * stream,
    const char * filename,
    char ** out_error) {
    if (stream && stream->backend) {
        return stream->backend->save_slot(filename, out_error);
    }
    (void) stream;
    (void) filename;
    /* v1: cross-launch slot KV persistence is not wired. Return a structured
     * unsupported code (per the optional-symbol contract) rather than
     * faking a save. */
    eliza_set_error(out_error,
        "[libelizainference] llm_stream_save_slot is not implemented in this build");
    return ELIZA_ERR_INVALID_ARG;
}

int eliza_inference_llm_stream_restore_slot(
    EliLlmStream * stream,
    const char * filename,
    char ** out_error) {
    if (stream && stream->backend) {
        return stream->backend->restore_slot(filename, out_error);
    }
    (void) stream;
    (void) filename;
    eliza_set_error(out_error,
        "[libelizainference] llm_stream_restore_slot is not implemented in this build");
    return ELIZA_ERR_INVALID_ARG;
}

int eliza_inference_llm_stream_reset(EliLlmStream * stream) {
    /* Clear the KV cache + sampler + counters so a PERSISTENT stream can be
     * reused for a fresh prompt without re-creating its llama_context. This is
     * the warm-reuse path: keeping one lctx alive (instead of open/close per
     * turn) avoids the per-turn Vulkan pipeline/KV re-init AND the
     * shared-GPU-weights-across-lctx-lifecycle corruption seen when streams are
     * created/destroyed repeatedly. Handles both the plain fixed-KV stream and
     * the MTP speculative engine (which owns its own target/draft KV). */
    if (!stream) return ELIZA_ERR_INVALID_ARG;
    if (stream->backend) return stream->backend->reset();
    if (!stream->mtp && !stream->lctx) return ELIZA_ERR_INVALID_ARG;
    if (stream->mtp) {
        /* MTP stream: clear both the target and draft KV caches, reset the
         * outer sampler, and drop the committed-token accumulator. The next
         * prefill re-seeds the speculative driver (common_speculative_process +
         * common_speculative_begin), so close+reopen of the second
         * LLAMA_CONTEXT_TYPE_MTP context per turn is no longer needed. */
        eliza_mtp::reset_engine(stream->mtp);
        stream->mtp_first_token = -1;
        stream->mtp_step_buf.assign(stream->mtp_step_buf.size(), 0);
    } else {
        llama_memory_clear(llama_get_memory(stream->lctx), true);
        if (stream->sampler) llama_sampler_reset(stream->sampler);
        stream->n_past = 0;
    }
    stream->generated = 0;
    stream->eos = false;
    stream->cancel.store(false);
    return ELIZA_OK;
}

int eliza_inference_llm_stream_reset_keep(EliLlmStream * stream, int32_t n_keep) {
    /* Prefix-preserving reset: KEEP the first n_keep tokens of KV resident and
     * drop everything after, so the next prefill only decodes the per-turn delta
     * (the system + tool-schema prefix is identical turn-to-turn, and on Mali's
     * scalar-matmul prefill the prefix is the dominant per-turn latency). The
     * non-MTP prefill (`llama_batch_get_one`) appends at stream->n_past, so after
     * this the host prefills only tokens[n_keep:] and the prefix is reused.
     *
     * Non-MTP streams only: the MTP engine seeds a speculative draft head from
     * EVERY prompt position during prefill, so partial-prefix reuse there needs
     * separate (riskier) handling — prefix-reuse mode opens the resident stream
     * without MTP, trading MTP's ~1.5x decode for the much larger prefill cut. */
    if (!stream) return ELIZA_ERR_INVALID_ARG;
    if (stream->backend) return stream->backend->reset_keep(n_keep);
    if (stream->mtp || !stream->lctx) return ELIZA_ERR_INVALID_ARG;
    if (n_keep < 0) n_keep = 0;
    if (n_keep > stream->n_past) n_keep = stream->n_past;
    /* Drop KV positions >= n_keep for the stream's sequence (0). */
    if (!llama_memory_seq_rm(llama_get_memory(stream->lctx), 0, n_keep, -1)) {
        /* A partial sequence that can't be trimmed — fall back to a full clear
         * so we never decode against a stale tail. */
        llama_memory_clear(llama_get_memory(stream->lctx), true);
        n_keep = 0;
    }
    if (stream->sampler) llama_sampler_reset(stream->sampler);
    stream->n_past = n_keep;
    stream->generated = 0;
    stream->eos = false;
    stream->cancel.store(false);
    return n_keep;
}

void eliza_inference_llm_stream_close(EliLlmStream * stream) {
    if (!stream) return;
    if (stream->backend) {
        delete stream->backend;
        stream->backend = nullptr;
    }
    if (stream->mtp) {
        eliza_mtp::free_engine(stream->mtp);
        stream->mtp = nullptr;
    }
    if (stream->sampler) {
        llama_sampler_free(stream->sampler);
        stream->sampler = nullptr;
    }
    if (stream->lctx) {
        llama_free(stream->lctx);
        stream->lctx = nullptr;
    }
    delete stream;
}

/* ---- Text embeddings (ABI v9) ------------------------------------- *
 *
 * Pooled sentence embeddings over the shared text model, mirroring
 * desktop-llama-adapter.ts's embed(). The embedding context is non-causal,
 * single-ubatch (sized to n_ctx), and pooled — distinct from the causal
 * generation context the streaming-LLM path builds.
 */

int eliza_inference_embed_supported(void) {
    return 1;
}

/* Build (or reuse) the dedicated embedding context. Caller must hold
 * ctx->llm_mutex and have a resident ctx->llm_model. Rebuilds when the
 * requested pooling type differs from the cached one (a caller that switches
 * pooling mid-session is rare but supported). */
static int eliza_ensure_embed_ctx_locked(
    EliInferenceContext * ctx,
    int pooling,
    char ** out_error) {
    if (ctx->embed_ctx && ctx->embed_pooling == pooling) return ELIZA_OK;
    if (ctx->embed_ctx) {
        llama_free(ctx->embed_ctx);
        ctx->embed_ctx = nullptr;
    }

    const int n_ctx_train = llama_model_n_ctx_train(ctx->llm_model);
    int n_ctx = eliza_int_env_or_default("ELIZA_EMBED_N_CTX", 512);
    if (n_ctx_train > 0 && n_ctx > n_ctx_train) n_ctx = n_ctx_train;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    /* Non-causal encoder layout: the whole sequence is decoded in one ubatch
     * sized to n_ctx (no causal KV streaming). Mirrors the embedding branch in
     * desktop-llama-adapter.ts (nBatch = nUBatch = ctxSize). */
    cparams.n_batch = (uint32_t) n_ctx;
    cparams.n_ubatch = (uint32_t) n_ctx;
    cparams.n_threads = eliza_thread_count(false);
    cparams.n_threads_batch = eliza_thread_count(true);
    cparams.embeddings = true;
    cparams.pooling_type = (enum llama_pooling_type) pooling;

    llama_context * lctx = llama_init_from_model(ctx->llm_model, cparams);
    if (!lctx) {
        eliza_set_error(out_error,
            "[libelizainference] embed: failed to init embedding context");
        return ELIZA_ERR_FFI_FAULT;
    }
    ctx->embed_ctx = lctx;
    ctx->embed_pooling = pooling;
    ctx->embed_n_ctx = n_ctx;
    return ELIZA_OK;
}

int eliza_inference_embed(
    EliInferenceContext * ctx,
    const char * text,
    size_t text_len,
    int pooling,
    float * out_embedding,
    size_t out_capacity,
    int * out_dim,
    char ** out_error) {
    if (!ctx || !text || !out_embedding || !out_dim) {
        eliza_set_error(out_error,
            "[libelizainference] embed: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (pooling == ELIZA_POOLING_NONE) {
        eliza_set_error(out_error,
            "[libelizainference] embed: pooling NONE produces no pooled "
            "vector; pass MEAN (1), CLS (2), or LAST (3)");
        return ELIZA_ERR_INVALID_ARG;
    }
    if (pooling < ELIZA_POOLING_NONE || pooling > ELIZA_POOLING_LAST) {
        eliza_set_error(out_error,
            "[libelizainference] embed: invalid pooling type " +
            std::to_string(pooling));
        return ELIZA_ERR_INVALID_ARG;
    }

    /* Per-op backend seam: an embedding backend (e.g. LiteRT/NPU) serves this
     * when it ships <bundle>/embedding/*; otherwise fall through to the in-tree
     * ggml encoder below. Inert by default (no backend registered). */
    {
        char * be_error = nullptr;
        EmbedBackendFactory * be =
            embed_backend_select(llm_backend_context_bundle_dir(ctx), &be_error);
        if (be_error) {
            eliza_set_error(out_error, std::string(be_error));
            std::free(be_error);
            return ELIZA_ERR_BUNDLE_INVALID;
        }
        if (be) {
            return be->embed(ctx, text, text_len, pooling, out_embedding,
                             out_capacity, out_dim, out_error);
        }
    }

    std::lock_guard<std::mutex> lock(ctx->llm_mutex);
    int rc = eliza_load_llm_model_locked(ctx, /* n_gpu_layers= */ -1, out_error);
    if (rc != ELIZA_OK) return rc;
    rc = eliza_ensure_embed_ctx_locked(ctx, pooling, out_error);
    if (rc != ELIZA_OK) return rc;

    const int n_embd = llama_model_n_embd(ctx->llm_model);
    *out_dim = n_embd;
    if (out_capacity < (size_t) n_embd) {
        eliza_set_error(out_error,
            "[libelizainference] embed: out_capacity " +
            std::to_string(out_capacity) + " < n_embd " +
            std::to_string(n_embd));
        return ELIZA_ERR_INVALID_ARG;
    }

    const llama_vocab * vocab = llama_model_get_vocab(ctx->llm_model);

    /* Tokenize (add_special = BOS, parse_special = false), then truncate to
     * the encoder ctx — a non-causal single-ubatch layout cannot encode
     * input longer than n_ctx. Mirrors embed() in desktop-llama-adapter.ts. */
    int32_t need = llama_tokenize(vocab, text, (int32_t) text_len,
                                  nullptr, 0, true, false);
    int32_t cap = need < 0 ? -need : need;
    if (cap == 0) {
        eliza_set_error(out_error,
            "[libelizainference] embed: empty token sequence");
        return ELIZA_ERR_INVALID_ARG;
    }
    std::vector<llama_token> tokens((size_t) cap);
    int32_t n_tok = llama_tokenize(vocab, text, (int32_t) text_len,
                                   tokens.data(), cap, true, false);
    if (n_tok < 0) {
        eliza_set_error(out_error,
            "[libelizainference] embed: llama_tokenize returned " +
            std::to_string(n_tok));
        return ELIZA_ERR_FFI_FAULT;
    }
    if (n_tok > ctx->embed_n_ctx) n_tok = ctx->embed_n_ctx;
    tokens.resize((size_t) n_tok);

    /* Fresh KV per call so a previous embedding can't bleed into this one. */
    llama_memory_clear(llama_get_memory(ctx->embed_ctx), true);
    llama_set_embeddings(ctx->embed_ctx, true);

    llama_batch batch = llama_batch_get_one(tokens.data(), n_tok);
    const int decode_rc = llama_decode(ctx->embed_ctx, batch);
    if (decode_rc != 0) {
        eliza_set_error(out_error,
            "[libelizainference] embed: llama_decode rc=" +
            std::to_string(decode_rc));
        return ELIZA_ERR_FFI_FAULT;
    }

    const float * emb = llama_get_embeddings_seq(ctx->embed_ctx, 0);
    if (!emb) {
        eliza_set_error(out_error,
            "[libelizainference] embed: llama_get_embeddings_seq returned NULL "
            "(pooling_type must not be NONE)");
        return ELIZA_ERR_FFI_FAULT;
    }
    std::memcpy(out_embedding, emb, (size_t) n_embd * sizeof(float));
    eliza_l2_normalize(out_embedding, n_embd);
    return ELIZA_OK;
}

/* ---- End-of-turn scoring (ABI v11) -------------------------------- *
 *
 * Single causal forward pass over a pre-tokenized context, returning the
 * next-token softmax probability of `target_token_id` (the chat template's
 * end-of-turn marker, e.g. <|im_end|>). This is the fused replacement for the
 * retired node-llama-cpp `controlledEvaluate()` the EOT classifiers depended
 * on: the JS side formats the partial ASR transcript as a user turn, tokenizes
 * it via eliza_inference_tokenize, and reads back P(end-of-turn). Runs on a
 * DEDICATED causal context (logits at the final position), lazily created and
 * reused, KV cleared per call so scores are independent. Does not touch the
 * streaming-LLM generation context or the embedding context.
 */

int eliza_inference_llm_eot_supported(void) {
    return 1;
}

/* Build (or reuse) the dedicated causal scoring context. Caller must hold
 * ctx->llm_mutex and have a resident ctx->llm_model. Causal layout (no
 * embeddings, no pooling) so the next-token logit distribution at the final
 * position is readable via llama_get_logits_ith. */
static int eliza_ensure_eot_ctx_locked(
    EliInferenceContext * ctx,
    char ** out_error) {
    if (ctx->eot_ctx) return ELIZA_OK;

    const int n_ctx_train = llama_model_n_ctx_train(ctx->llm_model);
    int n_ctx = eliza_int_env_or_default("ELIZA_EOT_N_CTX", 512);
    if (n_ctx_train > 0 && n_ctx > n_ctx_train) n_ctx = n_ctx_train;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = (uint32_t) n_ctx;
    cparams.n_ubatch = (uint32_t) n_ctx;
    cparams.n_threads = eliza_thread_count(false);
    cparams.n_threads_batch = eliza_thread_count(true);
    cparams.embeddings = false; /* causal generation layout, per-token logits */

    llama_context * lctx = llama_init_from_model(ctx->llm_model, cparams);
    if (!lctx) {
        eliza_set_error(out_error,
            "[libelizainference] eot: failed to init scoring context");
        return ELIZA_ERR_FFI_FAULT;
    }
    ctx->eot_ctx = lctx;
    ctx->eot_n_ctx = n_ctx;
    return ELIZA_OK;
}

int eliza_inference_llm_eot_score(
    EliInferenceContext * ctx,
    const int32_t * token_ids,
    size_t num_tokens,
    int32_t target_token_id,
    float * out_target_prob,
    int32_t * out_top_token,
    float * out_top_prob,
    char ** out_error) {
    if (out_target_prob) *out_target_prob = 0.0f;
    if (out_top_token) *out_top_token = -1;
    if (out_top_prob) *out_top_prob = 0.0f;

    if (!ctx || !token_ids || num_tokens == 0 || !out_target_prob) {
        eliza_set_error(out_error,
            "[libelizainference] eot: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }

    /* Per-op backend seam: an EOT backend (e.g. LiteRT/NPU) serves this when it
     * ships <bundle>/eot/*; otherwise fall through to the in-tree ggml
     * causal-scoring path below. Inert by default (no backend registered). */
    {
        char * be_error = nullptr;
        EotBackendFactory * be =
            eot_backend_select(llm_backend_context_bundle_dir(ctx), &be_error);
        if (be_error) {
            eliza_set_error(out_error, std::string(be_error));
            std::free(be_error);
            return ELIZA_ERR_BUNDLE_INVALID;
        }
        if (be) {
            return be->eot_score(ctx, token_ids, num_tokens, target_token_id,
                                 out_target_prob, out_top_token, out_top_prob,
                                 out_error);
        }
    }

    std::lock_guard<std::mutex> lock(ctx->llm_mutex);
    int rc = eliza_load_llm_model_locked(ctx, /* n_gpu_layers= */ -1, out_error);
    if (rc != ELIZA_OK) return rc;
    rc = eliza_ensure_eot_ctx_locked(ctx, out_error);
    if (rc != ELIZA_OK) return rc;

    const llama_vocab * vocab = llama_model_get_vocab(ctx->llm_model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (target_token_id < 0 || target_token_id >= n_vocab) {
        eliza_set_error(out_error,
            "[libelizainference] eot: target_token_id " +
            std::to_string(target_token_id) + " out of range [0," +
            std::to_string(n_vocab) + ")");
        return ELIZA_ERR_INVALID_ARG;
    }

    /* Keep the TAIL when the context overflows the scoring ctx — the most
     * recent tokens drive the turn-completion decision. */
    size_t n_tok = num_tokens;
    const int32_t * toks = token_ids;
    if (n_tok > (size_t) ctx->eot_n_ctx) {
        toks = token_ids + (n_tok - (size_t) ctx->eot_n_ctx);
        n_tok = (size_t) ctx->eot_n_ctx;
    }

    /* Fresh KV per call so a previous score can't bleed into this one. */
    llama_memory_clear(llama_get_memory(ctx->eot_ctx), true);
    llama_set_embeddings(ctx->eot_ctx, false);

    std::vector<llama_token> tokens(toks, toks + n_tok);
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) n_tok);
    const int decode_rc = llama_decode(ctx->eot_ctx, batch);
    if (decode_rc != 0) {
        eliza_set_error(out_error,
            "[libelizainference] eot: llama_decode rc=" +
            std::to_string(decode_rc));
        return ELIZA_ERR_FFI_FAULT;
    }

    /* Next-token logits at the final position (llama_batch_get_one enables
     * logits on the last token). Softmax in a numerically-stable pass: read the
     * argmax and the target probability. */
    const float * logits = llama_get_logits_ith(ctx->eot_ctx, -1);
    if (!logits) {
        eliza_set_error(out_error,
            "[libelizainference] eot: llama_get_logits_ith returned NULL");
        return ELIZA_ERR_FFI_FAULT;
    }

    float max_logit = logits[0];
    int32_t top_token = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
            top_token = i;
        }
    }
    double sum_exp = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        sum_exp += std::exp((double) (logits[i] - max_logit));
    }
    if (sum_exp <= 0.0) {
        eliza_set_error(out_error,
            "[libelizainference] eot: degenerate logit distribution");
        return ELIZA_ERR_FFI_FAULT;
    }
    const double target_p =
        std::exp((double) (logits[target_token_id] - max_logit)) / sum_exp;
    /* The argmax carries the max logit, so its unnormalized weight is exp(0)=1. */
    const double top_p = 1.0 / sum_exp;

    *out_target_prob = (float) target_p;
    if (out_top_token) *out_top_token = top_token;
    if (out_top_prob) *out_top_prob = (float) top_p;
    return ELIZA_OK;
}

/* ---- mmproj vision describe (ABI v9) ------------------------------ *
 *
 * Describe an image through the text model's mmproj projector, reusing the
 * mtmd machinery already linked for ASR. Gated on ELIZA_ENABLE_VISION at
 * build time (same flag the libllama+shim path uses). Mirrors
 * desktop-llama-adapter.ts's describeImage().
 */

int eliza_inference_vision_supported(void) {
#if defined(ELIZA_ENABLE_VISION)
    return 1;
#else
    return 0;
#endif
}

#if defined(ELIZA_ENABLE_VISION)

/* Ensure the mmproj vision context exists for `mmproj_path`. Caller must hold
 * ctx->llm_mutex and have a resident ctx->llm_model. Reloads when the path
 * differs from the cached one. */
static int eliza_ensure_vision_mtmd_locked(
    EliInferenceContext * ctx,
    const std::string & mmproj_path,
    char ** out_error) {
    if (ctx->vision_mtmd && ctx->vision_mmproj_path == mmproj_path) {
        return ELIZA_OK;
    }
    if (ctx->vision_mtmd) {
        mtmd_free(ctx->vision_mtmd);
        ctx->vision_mtmd = nullptr;
        ctx->vision_mmproj_path.clear();
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(mmproj_path, ec)) {
        eliza_set_error(out_error,
            "[libelizainference] describe_image: mmproj GGUF not found: " +
            mmproj_path);
        return ELIZA_ERR_BUNDLE_INVALID;
    }
    mtmd_context_params mparams = mtmd_context_params_default();
    mparams.use_gpu = eliza_bool_env_or_default("ELIZA_VISION_USE_GPU", true);
    mparams.print_timings = false;
    mparams.n_threads = eliza_thread_count(true);
    mtmd_context * mctx =
        mtmd_init_from_file(mmproj_path.c_str(), ctx->llm_model, mparams);
    if (!mctx) {
        eliza_set_error(out_error,
            "[libelizainference] describe_image: failed to load mmproj: " +
            mmproj_path);
        return ELIZA_ERR_BUNDLE_INVALID;
    }
    ctx->vision_mtmd = mctx;
    ctx->vision_mmproj_path = mmproj_path;
    return ELIZA_OK;
}

#endif // ELIZA_ENABLE_VISION

int eliza_inference_describe_image(
    EliInferenceContext * ctx,
    const unsigned char * image_bytes,
    size_t n_bytes,
    const char * mmproj_path,
    const char * prompt,
    char * out_text,
    size_t max_text_bytes,
    char ** out_error) {
#if !defined(ELIZA_ENABLE_VISION)
    (void) ctx; (void) image_bytes; (void) n_bytes; (void) mmproj_path;
    (void) prompt; (void) out_text; (void) max_text_bytes;
    eliza_set_error(out_error,
        "[libelizainference] describe_image: this build was compiled without "
        "ELIZA_ENABLE_VISION (eliza_inference_vision_supported() == 0); use the "
        "libllama mtmd path");
    return ELIZA_ERR_NOT_IMPLEMENTED;
#else
    if (!ctx || !image_bytes || n_bytes == 0 || !mmproj_path ||
        mmproj_path[0] == '\0' || !out_text || max_text_bytes == 0) {
        eliza_set_error(out_error,
            "[libelizainference] describe_image: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }

    /* Per-op backend seam: a vision backend (e.g. LiteRT/NPU) serves this when it
     * ships <bundle>/vision/*; otherwise fall through to the in-tree ggml mmproj
     * path below. Inert by default (no backend registered). */
    {
        char * be_error = nullptr;
        VisionBackendFactory * be =
            vision_backend_select(llm_backend_context_bundle_dir(ctx), &be_error);
        if (be_error) {
            eliza_set_error(out_error, std::string(be_error));
            std::free(be_error);
            return ELIZA_ERR_BUNDLE_INVALID;
        }
        if (be) {
            return be->describe_image(ctx, image_bytes, n_bytes, mmproj_path,
                                      prompt, out_text, max_text_bytes, out_error);
        }
    }

    std::lock_guard<std::mutex> lock(ctx->llm_mutex);
    int rc = eliza_load_llm_model_locked(ctx, /* n_gpu_layers= */ -1, out_error);
    if (rc != ELIZA_OK) return rc;
    rc = eliza_ensure_vision_mtmd_locked(ctx, std::string(mmproj_path), out_error);
    if (rc != ELIZA_OK) return rc;

    /* A fresh generation context per describe call (causal, no embeddings).
     * Vision describe is a one-shot per request; a dedicated short-lived ctx
     * keeps KV clean and avoids interleaving with the streaming-LLM sessions. */
    llama_context_params cparams = llama_context_default_params();
    const int n_ctx_train = llama_model_n_ctx_train(ctx->llm_model);
    int n_ctx = eliza_int_env_or_default("ELIZA_VISION_N_CTX", 4096);
    if (n_ctx_train > 0 && n_ctx > n_ctx_train) n_ctx = n_ctx_train;
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = (uint32_t) eliza_int_env_or_default("ELIZA_VISION_N_BATCH", 512);
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_threads = eliza_thread_count(false);
    cparams.n_threads_batch = eliza_thread_count(true);
    cparams.flash_attn_type = eliza_llm_flash_attn_type();
    llama_context * lctx = llama_init_from_model(ctx->llm_model, cparams);
    if (!lctx) {
        eliza_set_error(out_error,
            "[libelizainference] describe_image: failed to init context");
        return ELIZA_ERR_FFI_FAULT;
    }

    llama_sampler * sampler = nullptr;
    mtmd_bitmap * bitmap = nullptr;
    mtmd_input_chunks * chunks = nullptr;
    int result = ELIZA_ERR_FFI_FAULT;
    std::string transcript;

    do {
        const char * marker = mtmd_default_marker();
        std::string user_prompt = prompt && prompt[0] != '\0'
            ? std::string(prompt)
            : std::string("Describe what is in this image.");
        std::string prompt_text =
            (marker && user_prompt.find(marker) != std::string::npos)
                ? user_prompt
                : (std::string(marker ? marker : "<__media__>") + "\n" + user_prompt);

        bitmap = mtmd_helper_bitmap_init_from_buf(
            ctx->vision_mtmd, image_bytes, n_bytes);
        if (!bitmap) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image: image decode failed");
            result = ELIZA_ERR_INVALID_ARG;
            break;
        }
        chunks = mtmd_input_chunks_init();
        if (!chunks) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image: chunks allocation failed");
            break;
        }
        mtmd_input_text text = { prompt_text.c_str(), true, true };
        const mtmd_bitmap * bitmaps[] = { bitmap };
        int32_t tok_rc = mtmd_tokenize(ctx->vision_mtmd, chunks, &text, bitmaps, 1);
        if (tok_rc != 0) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image: mtmd_tokenize rc=" +
                std::to_string(tok_rc));
            break;
        }

        llama_memory_clear(llama_get_memory(lctx), true);
        llama_pos n_past = 0;
        int32_t eval_rc = mtmd_helper_eval_chunks(
            ctx->vision_mtmd, lctx, chunks, n_past, 0,
            (int32_t) cparams.n_batch, true, &n_past);
        if (eval_rc != 0) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image: mtmd_helper_eval_chunks rc=" +
                std::to_string(eval_rc));
            break;
        }

        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        sampler = llama_sampler_chain_init(sparams);
        if (!sampler) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image: failed to init sampler");
            break;
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

        const llama_vocab * vocab = llama_model_get_vocab(ctx->llm_model);
        const int max_tokens = eliza_int_env_or_default("ELIZA_VISION_MAX_TOKENS", 256);
        bool overflow = false;
        for (int i = 0; i < max_tokens; ++i) {
            llama_token token = llama_sampler_sample(sampler, lctx, -1);
            if (llama_vocab_is_eog(vocab, token)) break;
            std::string piece = eliza_llama_token_piece(vocab, token);
            if (!piece.empty()) {
                if (transcript.size() + piece.size() + 1 > max_text_bytes) {
                    overflow = true;
                    break;
                }
                transcript += piece;
            }
            llama_sampler_accept(sampler, token);
            llama_batch batch = llama_batch_get_one(&token, 1);
            int32_t decode_rc = llama_decode(lctx, batch);
            if (decode_rc != 0) {
                eliza_set_error(out_error,
                    "[libelizainference] describe_image: llama_decode rc=" +
                    std::to_string(decode_rc));
                overflow = true; // reuse the error exit
                transcript.clear();
                break;
            }
        }
        if (transcript.empty() && overflow) {
            // decode error path already set out_error
            break;
        }
        if (transcript.size() + 1 > max_text_bytes) {
            transcript.resize(max_text_bytes - 1);
        }
        result = ELIZA_OK;
    } while (false);

    if (sampler) llama_sampler_free(sampler);
    if (chunks) mtmd_input_chunks_free(chunks);
    if (bitmap) mtmd_bitmap_free(bitmap);
    llama_free(lctx);

    if (result != ELIZA_OK) return result;
    std::memcpy(out_text, transcript.data(), transcript.size());
    out_text[transcript.size()] = '\0';
    return (int) transcript.size();
#endif // ELIZA_ENABLE_VISION
}

/* ---- Streaming mmproj vision describe (ABI v13) ------------------- *
 *
 * Token-by-token vision: open primes an EliLlmStream's KV with the image +
 * prompt (the same mtmd prefill as _describe_image), and the caller drives the
 * existing _llm_stream_next loop to pull tokens — so vision streams through the
 * exact same path (and JS FfiStreamingRunner) as chat text. The returned stream
 * carries a greedy sampler + ELIZA_VISION_MAX_TOKENS cap and no MTP engine. */

int eliza_inference_vision_stream_supported(void) {
#if defined(ELIZA_ENABLE_VISION)
    return 1;
#else
    return 0;
#endif
}

EliLlmStream * eliza_inference_describe_image_stream_open(
    EliInferenceContext * ctx,
    const unsigned char * image_bytes,
    size_t n_bytes,
    const char * mmproj_path,
    const char * prompt,
    char ** out_error) {
#if !defined(ELIZA_ENABLE_VISION)
    (void) ctx; (void) image_bytes; (void) n_bytes; (void) mmproj_path;
    (void) prompt;
    eliza_set_error(out_error,
        "[libelizainference] describe_image_stream_open: this build was compiled "
        "without ELIZA_ENABLE_VISION (eliza_inference_vision_stream_supported() == "
        "0); use the buffered _describe_image path");
    return nullptr;
#else
    if (!ctx || !image_bytes || n_bytes == 0 || !mmproj_path ||
        mmproj_path[0] == '\0') {
        eliza_set_error(out_error,
            "[libelizainference] describe_image_stream_open: invalid arguments");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(ctx->llm_mutex);
    int rc = eliza_load_llm_model_locked(ctx, /* n_gpu_layers= */ -1, out_error);
    if (rc != ELIZA_OK) return nullptr;
    rc = eliza_ensure_vision_mtmd_locked(ctx, std::string(mmproj_path), out_error);
    if (rc != ELIZA_OK) return nullptr;

    /* A fresh generation context (causal, no embeddings), owned by the returned
     * stream and freed by eliza_inference_llm_stream_close. Same params as the
     * buffered _describe_image so streamed and buffered describes decode
     * identically. */
    llama_context_params cparams = llama_context_default_params();
    const int n_ctx_train = llama_model_n_ctx_train(ctx->llm_model);
    int n_ctx = eliza_int_env_or_default("ELIZA_VISION_N_CTX", 4096);
    if (n_ctx_train > 0 && n_ctx > n_ctx_train) n_ctx = n_ctx_train;
    cparams.n_ctx = (uint32_t) n_ctx;
    cparams.n_batch = (uint32_t) eliza_int_env_or_default("ELIZA_VISION_N_BATCH", 512);
    cparams.n_ubatch = cparams.n_batch;
    cparams.n_threads = eliza_thread_count(false);
    cparams.n_threads_batch = eliza_thread_count(true);
    cparams.flash_attn_type = eliza_llm_flash_attn_type();
    llama_context * lctx = llama_init_from_model(ctx->llm_model, cparams);
    if (!lctx) {
        eliza_set_error(out_error,
            "[libelizainference] describe_image_stream_open: failed to init context");
        return nullptr;
    }

    llama_sampler * sampler = nullptr;
    mtmd_bitmap * bitmap = nullptr;
    mtmd_input_chunks * chunks = nullptr;
    bool ok = false;
    llama_pos n_past = 0;

    do {
        const char * marker = mtmd_default_marker();
        std::string user_prompt = prompt && prompt[0] != '\0'
            ? std::string(prompt)
            : std::string("Describe what is in this image.");
        std::string prompt_text =
            (marker && user_prompt.find(marker) != std::string::npos)
                ? user_prompt
                : (std::string(marker ? marker : "<__media__>") + "\n" + user_prompt);

        bitmap = mtmd_helper_bitmap_init_from_buf(
            ctx->vision_mtmd, image_bytes, n_bytes);
        if (!bitmap) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image_stream_open: image decode failed");
            break;
        }
        chunks = mtmd_input_chunks_init();
        if (!chunks) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image_stream_open: chunks allocation failed");
            break;
        }
        mtmd_input_text text = { prompt_text.c_str(), true, true };
        const mtmd_bitmap * bitmaps[] = { bitmap };
        int32_t tok_rc = mtmd_tokenize(ctx->vision_mtmd, chunks, &text, bitmaps, 1);
        if (tok_rc != 0) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image_stream_open: mtmd_tokenize rc=" +
                std::to_string(tok_rc));
            break;
        }

        llama_memory_clear(llama_get_memory(lctx), true);
        int32_t eval_rc = mtmd_helper_eval_chunks(
            ctx->vision_mtmd, lctx, chunks, n_past, 0,
            (int32_t) cparams.n_batch, true, &n_past);
        if (eval_rc != 0) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image_stream_open: mtmd_helper_eval_chunks rc=" +
                std::to_string(eval_rc));
            break;
        }

        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        sampler = llama_sampler_chain_init(sparams);
        if (!sampler) {
            eliza_set_error(out_error,
                "[libelizainference] describe_image_stream_open: failed to init sampler");
            break;
        }
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        ok = true;
    } while (false);

    /* The bitmap + chunks are only needed for the prefill eval; the KV now holds
     * the image + prompt, so release them (the lctx + sampler live on in the
     * returned stream). */
    if (chunks) mtmd_input_chunks_free(chunks);
    if (bitmap) mtmd_bitmap_free(bitmap);

    if (!ok) {
        if (sampler) llama_sampler_free(sampler);
        llama_free(lctx);
        return nullptr;
    }

    EliLlmStream * stream = new (std::nothrow) EliLlmStream();
    if (!stream) {
        llama_sampler_free(sampler);
        llama_free(lctx);
        eliza_set_error(out_error,
            "[libelizainference] describe_image_stream_open: out of memory");
        return nullptr;
    }
    stream->ctx = ctx;
    stream->lctx = lctx;
    stream->sampler = sampler;
    stream->n_past = (int) n_past;
    stream->generated = 0;
    stream->max_tokens = eliza_int_env_or_default("ELIZA_VISION_MAX_TOKENS", 256);
    stream->eos = false;
    /* mtp stays null — vision uses the plain fixed-KV decode path in
     * _llm_stream_next, which samples from lctx (logits primed at -1 by
     * mtmd_helper_eval_chunks above). */
    return stream;
#endif // ELIZA_ENABLE_VISION
}

/* ---- Tokenizer (ABI v9) ------------------------------------------- *
 *
 * llama_tokenize / llama_detokenize over the loaded text model's vocab, so the
 * desktop fused runtime stops standing up a libllama tokenizer sidecar.
 */

int eliza_inference_tokenize_supported(void) {
    return 1;
}

int eliza_inference_tokenize(
    EliInferenceContext * ctx,
    const char * text,
    size_t text_len,
    int add_special,
    int parse_special,
    int ** out_tokens,
    size_t * out_n,
    char ** out_error) {
    if (!ctx || !text || !out_tokens || !out_n) {
        eliza_set_error(out_error,
            "[libelizainference] tokenize: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }
    *out_tokens = nullptr;
    *out_n = 0;

    std::lock_guard<std::mutex> lock(ctx->llm_mutex);
    int rc = eliza_load_llm_model_locked(ctx, /* n_gpu_layers= */ -1, out_error);
    if (rc != ELIZA_OK) return rc;

    const llama_vocab * vocab = llama_model_get_vocab(ctx->llm_model);
    int32_t need = llama_tokenize(vocab, text, (int32_t) text_len, nullptr, 0,
                                  add_special != 0, parse_special != 0);
    int32_t cap = need < 0 ? -need : need;
    if (cap == 0) {
        /* Empty token sequence is valid (e.g. empty input). Return an empty,
         * non-NULL buffer so the caller's free path is uniform. */
        int * empty = (int *) std::malloc(1);
        if (!empty) {
            eliza_set_error(out_error, "[libelizainference] tokenize: OOM");
            return ELIZA_ERR_OOM;
        }
        *out_tokens = empty;
        *out_n = 0;
        return ELIZA_OK;
    }
    int * buf = (int *) std::malloc((size_t) cap * sizeof(int));
    if (!buf) {
        eliza_set_error(out_error, "[libelizainference] tokenize: OOM");
        return ELIZA_ERR_OOM;
    }
    static_assert(sizeof(llama_token) == sizeof(int32_t),
                  "llama_token must be int32 for the tokenize ABI");
    int32_t written = llama_tokenize(vocab, text, (int32_t) text_len,
                                     (llama_token *) buf, cap,
                                     add_special != 0, parse_special != 0);
    if (written < 0) {
        std::free(buf);
        eliza_set_error(out_error,
            "[libelizainference] tokenize: llama_tokenize returned " +
            std::to_string(written) + " (buffer too small)");
        return ELIZA_ERR_FFI_FAULT;
    }
    *out_tokens = buf;
    *out_n = (size_t) written;
    return ELIZA_OK;
}

int eliza_inference_detokenize(
    EliInferenceContext * ctx,
    const int * tokens,
    size_t n_tokens,
    int remove_special,
    int unparse_special,
    char * out_text,
    size_t max_text_bytes,
    char ** out_error) {
    if (!ctx || (!tokens && n_tokens > 0) || !out_text || max_text_bytes == 0) {
        eliza_set_error(out_error,
            "[libelizainference] detokenize: invalid arguments");
        return ELIZA_ERR_INVALID_ARG;
    }

    std::lock_guard<std::mutex> lock(ctx->llm_mutex);
    int rc = eliza_load_llm_model_locked(ctx, /* n_gpu_layers= */ -1, out_error);
    if (rc != ELIZA_OK) return rc;

    const llama_vocab * vocab = llama_model_get_vocab(ctx->llm_model);
    int32_t written = llama_detokenize(
        vocab, (const llama_token *) tokens, (int32_t) n_tokens,
        out_text, (int32_t) (max_text_bytes - 1),
        remove_special != 0, unparse_special != 0);
    if (written < 0) {
        eliza_set_error(out_error,
            "[libelizainference] detokenize: output buffer too small (need " +
            std::to_string(-written) + " bytes)");
        return ELIZA_ERR_INVALID_ARG;
    }
    out_text[written] = '\0';
    return (int) written;
}

void eliza_inference_free_string(char * str) {
    std::free(str);
}

} // extern "C"
