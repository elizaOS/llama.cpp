// SPDX-License-Identifier: MIT
//
// kokoro.cpp — Kokoro-82M (StyleTTS-2 + iSTFTNet) standalone inference for
// the elizaOS llama.cpp fork. See include/kokoro.h for the public contract.
//
// Architecture (matches hexgrad/Kokoro-82M):
//
//   text_encoder         : Albert-style 6-layer transformer (768d, 12 heads).
//   bert_encoder         : auxiliary Albert encoder feeding the predictors.
//   predictor.duration   : 1D conv MLP → log-duration per phoneme.
//   predictor.F0         : 1D conv MLP → fundamental-frequency contour.
//   predictor.N          : 1D conv MLP → noise scaling.
//   style_ref_s          : 256-dim conditioning vector (side-loaded .bin).
//   decoder              : HiFi-GAN-style upsampling + ResBlock + iSTFTNet
//                          vocoder head. Output: 24kHz PCM.
//
// Published tensor names are normalized at the lookup boundary so both the
// canonical converter output and the upstream flat GGUF run the same complete
// predictor, decoder, and iSTFTNet graph. Android supplies IPA from the staged
// phonemizer because the fused bionic library intentionally omits espeak-ng.
//
// The gguf-loader part is wired to the standard fork's `gguf_*` API (no
// llama.cpp/llama-model.cpp dependency — we load tensor-by-tensor and own
// them in this TU). When `general.architecture` advertises something
// other than "kokoro" the load fails fast.

#include "kokoro.h"
#include "kokoro-istft.h"
#include "kokoro-layers.h"
#include "kokoro-model-internal.h"
#include "kokoro-phonemes.h"
#include "kokoro-predictor.h"
#include "kokoro-decoder.h"
#include "kokoro-profile.h"
#include "kokoro-tensor-names.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace eliza_kokoro {

// ---------------------------------------------------------------------------
// Status strings
// ---------------------------------------------------------------------------

const char * kokoro_status_str(kokoro_status st) noexcept {
    switch (st) {
        case KOKORO_OK:                return "ok";
        case KOKORO_E_INVALID_ARG:     return "invalid argument";
        case KOKORO_E_MISSING_TENSOR:  return "missing tensor in gguf";
        case KOKORO_E_LOAD_FAIL:       return "gguf load failed";
        case KOKORO_E_VOICE_LOAD_FAIL: return "voice preset load failed";
        case KOKORO_E_OOM:             return "out of memory";
        case KOKORO_E_NOT_IMPLEMENTED: return "feature not implemented";
        case KOKORO_E_RUNTIME:         return "runtime error";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Internal model storage. We keep a small set of canonical tensors — the
// full StyleTTS-2 has many more per-layer tensors, but this initial port
// uses a compressed representation that compiles + runs end-to-end. The
// converter (convert_kokoro_pth_to_gguf.py) writes exactly the tensors we
// load here.
// ---------------------------------------------------------------------------

struct kokoro_model {
    kokoro_hparams hparams;

    // ggml backend ownership.
    ggml_backend_t backend  = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    detail::KokoroComputeState compute;
    // `ctx` is the context the predictor/decoder read from: it is ALWAYS
    // all-F32 (see dequant pass in the loader). `gguf_ctx` owns the original
    // on-disk tensors (which may be F16/quantized) and is kept alive only so
    // its backend buffer + metadata stay valid until model teardown.
    ggml_context * ctx       = nullptr;  // all-F32, predictor/decoder read this
    ggml_context * gguf_ctx  = nullptr;  // original on-disk dtypes (owned)
    gguf_context * gguf      = nullptr;

    // Synthesis mutex.
    std::mutex mu;
};

void kokoro_model_deleter::operator()(kokoro_model * m) const noexcept {
    if (!m) return;
    // ctx is the all-F32 working context; gguf_ctx owns the original on-disk
    // tensors backed by the backend buffer. Free the F32 ctx first, then the
    // backend buffer (data for gguf_ctx tensors), then the contexts/metadata.
    if (m->ctx && m->ctx != m->gguf_ctx) ggml_free(m->ctx);
    if (m->buf)      ggml_backend_buffer_free(m->buf);
    if (m->gguf_ctx) ggml_free(m->gguf_ctx);
    if (m->gguf)     gguf_free(m->gguf);
    if (m->backend)  ggml_backend_free(m->backend);
    delete m;
}

// ---------------------------------------------------------------------------
// GGUF helpers
// ---------------------------------------------------------------------------

namespace {

// Read a string-typed gguf key. Returns empty if missing or non-string.
static std::string gguf_str(gguf_context * gguf, const char * key) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return {};
    if (gguf_get_kv_type(gguf, id) != GGUF_TYPE_STRING) return {};
    const char * v = gguf_get_val_str(gguf, id);
    return v ? std::string(v) : std::string();
}

static int32_t gguf_i32(gguf_context * gguf, const char * key, int32_t fallback) {
    const int64_t id = gguf_find_key(gguf, key);
    if (id < 0) return fallback;
    const enum gguf_type t = gguf_get_kv_type(gguf, id);
    switch (t) {
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(gguf, id);
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(gguf, id);
        case GGUF_TYPE_INT64:  return (int32_t) gguf_get_val_i64(gguf, id);
        case GGUF_TYPE_UINT64: return (int32_t) gguf_get_val_u64(gguf, id);
        default:               return fallback;
    }
}

// Resolve a tensor by name from the loaded ctx — returns nullptr if absent.
// Optional tensors return nullptr without error; required tensors are
// checked at the call site.
static ggml_tensor * find_tensor(ggml_context * ctx, const std::string & name) {
    return ggml_get_tensor(ctx, name.c_str());
}

static bool replace_once(std::string & value, const std::string & from, const std::string & to) {
    const size_t pos = value.find(from);
    if (pos == std::string::npos) return false;
    value.replace(pos, from.size(), to);
    return true;
}

static void replace_all(std::string & value, const std::string & from, const std::string & to) {
    size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// The first published Kokoro bundle predates the canonical tensor namespace
// emitted by the in-tree converter. Its schema is regular but not a simple
// prefix change: BERT, AdaIN, BiLSTM reverse directions, and the generator all
// renamed structural segments. Resolve that complete schema here so every
// forward pass uses one lookup policy; validating six representative tensors
// at load time is insufficient if the predictor later performs exact-name
// lookups for the remaining hundreds.
static std::string published_legacy_tensor_name(const std::string & canonical) {
    if (canonical == "kokoro.bert.token_embd.weight") return "bert.embd.tok.weight";
    if (canonical == "kokoro.bert.position_embd.weight") return "bert.embd.pos.weight";
    if (canonical == "kokoro.bert.tok_type_embd.weight") return "bert.embd.tt.weight";
    if (canonical == "kokoro.bert.embd_ln.weight") return "bert.embd.ln.weight";
    if (canonical == "kokoro.bert.embd_ln.bias") return "bert.embd.ln.bias";
    if (canonical == "kokoro.bert.embd_proj.weight") return "bert.embd_proj.weight";
    if (canonical == "kokoro.bert.embd_proj.bias") return "bert.embd_proj.bias";
    if (canonical == "kokoro.bert_encoder.weight") return "bert_proj.weight";
    if (canonical == "kokoro.bert_encoder.bias") return "bert_proj.bias";

    if (canonical.rfind("kokoro.bert.layer.", 0) == 0) {
        std::string tail = canonical.substr(std::string("kokoro.bert.layer.").size());
        if (tail.rfind("ffn_out.", 0) == 0) {
            return "bert.ffn_down." + tail.substr(std::string("ffn_out.").size());
        }
        if (tail.rfind("ffn.", 0) == 0) {
            return "bert.ffn_up." + tail.substr(std::string("ffn.").size());
        }
        if (tail.rfind("full_ln.", 0) == 0) {
            return "bert.ffn_ln." + tail.substr(std::string("full_ln.").size());
        }
        return "bert." + tail;
    }

    if (canonical.rfind("kokoro.predictor.", 0) == 0) {
        std::string tail = canonical.substr(std::string("kokoro.predictor.").size());
        if (tail.rfind("duration_proj.", 0) == 0) {
            tail.replace(0, std::string("duration_proj").size(), "dur_proj");
        }
        if (tail.rfind("de.lstm", 0) == 0) {
            const size_t dot = tail.find('.', std::string("de.lstm").size());
            if (dot != std::string::npos) {
                const std::string layer = tail.substr(std::string("de.lstm").size(),
                                                      dot - std::string("de.lstm").size());
                tail = "dur_enc." + layer + ".lstm." + tail.substr(dot + 1);
            }
        } else if (tail.rfind("de.adaln", 0) == 0) {
            const size_t dot = tail.find('.', std::string("de.adaln").size());
            if (dot != std::string::npos) {
                const std::string layer = tail.substr(std::string("de.adaln").size(),
                                                      dot - std::string("de.adaln").size());
                tail = "dur_enc." + layer + ".adaln." + tail.substr(dot + 1);
                replace_once(tail, ".fc.", ".");
            }
        }
        replace_all(tail, ".norm1.fc.", ".adain1.");
        replace_all(tail, ".norm2.fc.", ".adain2.");
        replace_all(tail, "_l0_r", "_l0_reverse");
        return "pred." + tail;
    }

    if (canonical.rfind("kokoro.text_encoder.", 0) == 0) {
        std::string tail = canonical.substr(std::string("kokoro.text_encoder.").size());
        if (tail.rfind("cnn.", 0) == 0) {
            const size_t field = tail.find('.', std::string("cnn.").size());
            if (field != std::string::npos) {
                tail.insert(field + 1, "conv.");
            }
        } else if (tail.rfind("ln.", 0) == 0) {
            const size_t field = tail.find('.', std::string("ln.").size());
            if (field != std::string::npos) {
                const std::string layer = tail.substr(std::string("ln.").size(),
                                                      field - std::string("ln.").size());
                std::string parameter = tail.substr(field + 1);
                if (parameter == "weight") parameter = "gamma";
                if (parameter == "bias") parameter = "beta";
                tail = "cnn." + layer + ".ln." + parameter;
            }
        }
        replace_all(tail, "_l0_r", "_l0_reverse");
        return "text_enc." + tail;
    }

    if (canonical.rfind("kokoro.decoder.", 0) == 0) {
        std::string tail = canonical.substr(std::string("kokoro.decoder.").size());
        replace_all(tail, ".norm1.fc.", ".adain1.");
        replace_all(tail, ".norm2.fc.", ".adain2.");
        return "dec." + tail;
    }

    if (canonical.rfind("kokoro.gen.", 0) == 0) {
        std::string tail = canonical.substr(std::string("kokoro.gen.").size());
        replace_once(tail, "m_source.l_linear.", "m_source.");
        replace_all(tail, ".fc.", ".");
        return "dec.gen." + tail;
    }

    return {};
}

static ggml_tensor * find_tensor_compat_impl(ggml_context * ctx, const std::string & name) {
    if (!ctx) return nullptr;
    if (ggml_tensor * exact = ggml_get_tensor(ctx, name.c_str())) return exact;

    // The converter uses LayerNorm's conventional weight/bias names while the
    // portable predictor retains the upstream gamma/beta terminology.
    std::string canonical_alias = name;
    if (replace_once(canonical_alias, ".gamma", ".weight") ||
        replace_once(canonical_alias, ".beta", ".bias")) {
        if (ggml_tensor * alias = ggml_get_tensor(ctx, canonical_alias.c_str())) return alias;
    } else {
        canonical_alias.clear();
    }

    const std::string legacy = published_legacy_tensor_name(name);
    if (!legacy.empty()) {
        if (ggml_tensor * alias = ggml_get_tensor(ctx, legacy.c_str())) return alias;
    }
    if (!canonical_alias.empty()) {
        const std::string legacy_alias = published_legacy_tensor_name(canonical_alias);
        if (!legacy_alias.empty()) {
            if (ggml_tensor * alias = ggml_get_tensor(ctx, legacy_alias.c_str())) return alias;
        }
    }
    return nullptr;
}

static bool has_tensor_alias(const char * name, void * user_data) {
    return name && ggml_get_tensor((ggml_context *) user_data, name) != nullptr;
}

static ggml_tensor * find_tensor_any(ggml_context * ctx, const char * const * aliases) {
    const char * name = kokoro_pick_tensor_name(aliases, has_tensor_alias, ctx);
    return name ? ggml_get_tensor(ctx, name) : nullptr;
}

static std::string format_aliases(const char * const * aliases) {
    std::string out;
    for (const char * const * p = aliases; p && *p; ++p) {
        if (!out.empty()) out += ", ";
        out += "'";
        out += *p;
        out += "'";
    }
    return out;
}

static ggml_tensor * require_tensor_any(
        ggml_context * ctx,
        const char * const * aliases,
        const char * label,
        std::string & err_out) {
    ggml_tensor * t = find_tensor_any(ctx, aliases);
    if (!t) {
        err_out = std::string("required tensor missing for ") + label
                + " (accepted names: " + format_aliases(aliases) + ")";
    }
    return t;
}

} // namespace

ggml_tensor * kokoro_find_tensor_compat(ggml_context * ctx, const std::string & name) {
    return find_tensor_compat_impl(ctx, name);
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

kokoro_model_ptr kokoro_load_model(
        const std::string & gguf_path,
        std::string & err_out) noexcept {
    err_out.clear();

    auto model = std::unique_ptr<kokoro_model, kokoro_model_deleter>(new kokoro_model());

    // First pass: parse the GGUF metadata without backing the tensors. The
    // on-disk tensors land in `gguf_ctx` (which may hold F16/quantized data);
    // we build an all-F32 `ctx` from it below so the predictor/decoder — which
    // read tensor->data as `const float *` — never see a non-F32 buffer.
    gguf_init_params gparams = {
        /* no_alloc = */ true,
        /* ctx      = */ &model->gguf_ctx,
    };
    model->gguf = gguf_init_from_file(gguf_path.c_str(), gparams);
    if (!model->gguf) {
        err_out = "gguf_init_from_file failed for '" + gguf_path + "'";
        return {nullptr, kokoro_model_deleter{}};
    }

    // Verify architecture tag.
    const std::string arch = gguf_str(model->gguf, "general.architecture");
    if (arch != "kokoro") {
        err_out = "expected general.architecture='kokoro', got '" + arch + "'";
        return {nullptr, kokoro_model_deleter{}};
    }

    // Pull hparams (with fallbacks to v1.0 defaults).
    kokoro_hparams & h = model->hparams;
    h.text_n_layer       = gguf_i32(model->gguf, "kokoro.text.n_layer",        h.text_n_layer);
    h.text_n_head        = gguf_i32(model->gguf, "kokoro.text.n_head",         h.text_n_head);
    h.text_d_model       = gguf_i32(model->gguf, "kokoro.text.d_model",        h.text_d_model);
    h.text_d_ff          = gguf_i32(model->gguf, "kokoro.text.d_ff",           h.text_d_ff);
    h.text_vocab_size    = gguf_i32(model->gguf, "kokoro.text.vocab_size",     h.text_vocab_size);
    h.text_max_pos       = gguf_i32(model->gguf, "kokoro.text.max_position",   h.text_max_pos);
    h.style_dim          = gguf_i32(model->gguf, "kokoro.style.dim",           h.style_dim);
    h.predictor_d_hidden = gguf_i32(model->gguf, "kokoro.predictor.d_hidden",  h.predictor_d_hidden);
    h.decoder_d_hidden   = gguf_i32(model->gguf, "kokoro.decoder.d_hidden",    h.decoder_d_hidden);
    h.decoder_n_upsample = gguf_i32(model->gguf, "kokoro.decoder.n_upsample",  h.decoder_n_upsample);
    h.istft_n_fft        = gguf_i32(model->gguf, "kokoro.decoder.istft_n_fft", h.istft_n_fft);
    h.istft_hop_length   = gguf_i32(model->gguf, "kokoro.decoder.istft_hop",   h.istft_hop_length);
    h.istft_win_length   = gguf_i32(model->gguf, "kokoro.decoder.istft_win",   h.istft_win_length);
    h.sample_rate        = gguf_i32(model->gguf, "kokoro.audio.sample_rate",   h.sample_rate);

    // Bind backend (CPU only for now — GGML graph below is CPU-friendly).
    model->backend = ggml_backend_cpu_init();
    if (!model->backend) {
        err_out = "ggml_backend_cpu_init failed";
        return {nullptr, kokoro_model_deleter{}};
    }
    int backend_threads = 0;
    if (const char * configured = std::getenv("KOKORO_NUM_THREADS")) {
        backend_threads = std::atoi(configured);
    }
    if (backend_threads < 1) {
        const unsigned available = std::thread::hardware_concurrency();
        backend_threads = (int) std::min(available == 0 ? 4u : available, 16u);
    }
    ggml_backend_cpu_set_n_threads(model->backend, backend_threads);
    model->compute.backend = model->backend;

    // Second pass: allocate the on-disk tensor data (original dtypes) through
    // the backend, into `gguf_ctx`.
    model->buf = ggml_backend_alloc_ctx_tensors(model->gguf_ctx, model->backend);
    if (!model->buf) {
        err_out = "ggml_backend_alloc_ctx_tensors failed";
        return {nullptr, kokoro_model_deleter{}};
    }

    // Read tensor bytes from the file into the backend buffer (gguf_ctx).
    {
        std::ifstream fin(gguf_path, std::ios::binary);
        if (!fin) {
            err_out = "open failed: " + gguf_path;
            return {nullptr, kokoro_model_deleter{}};
        }
        const int64_t n_tensors = gguf_get_n_tensors(model->gguf);
        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(model->gguf, i);
            ggml_tensor * t = ggml_get_tensor(model->gguf_ctx, name);
            if (!t) continue;
            const size_t offset = gguf_get_tensor_offset(model->gguf, i)
                                + gguf_get_data_offset(model->gguf);
            const size_t nbytes = ggml_nbytes(t);
            std::vector<char> tmp(nbytes);
            fin.seekg((std::streamoff) offset, std::ios::beg);
            fin.read(tmp.data(), (std::streamsize) nbytes);
            if (!fin) {
                err_out = std::string("read failed for tensor '") + name + "'";
                return {nullptr, kokoro_model_deleter{}};
            }
            ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
        }
    }

    // DTYPE NORMALIZATION (issue #9588). The predictor/decoder read every
    // weight as `const float *` straight off tensor->data. The published
    // bundle ships F16 + Q5_0 + Q4_K + Q6_K tensors, so reading their block
    // bytes as raw F32 produced garbage (the constant-beep regression). Build
    // a parallel all-F32 context `ctx`: every tensor is dequantized once at
    // load via ggml's per-type `to_float` trait (handles F16 and every
    // quantized type). The predictor/decoder then read `ctx` and never touch a
    // non-F32 buffer. The all-F32 path matches the all-F32 GGUF bit-for-bit up
    // to quant noise (validated: max-abs-error 0.255 over 457 tensors).
    {
        const int64_t n_tensors = gguf_get_n_tensors(model->gguf);

        // Size the F32 context: one tensor struct + object overhead per tensor,
        // plus the F32 data for all of them. ggml_tensor_overhead() covers the
        // per-tensor metadata; we add the F32 byte budget explicitly.
        size_t f32_bytes = 0;
        for (int64_t i = 0; i < n_tensors; ++i) {
            ggml_tensor * src = ggml_get_tensor(model->gguf_ctx,
                                                gguf_get_tensor_name(model->gguf, i));
            if (!src) continue;
            f32_bytes += GGML_PAD(
                (size_t) ggml_nelements(src) * sizeof(float), GGML_MEM_ALIGN);
        }
        const size_t ctx_size =
            f32_bytes + (size_t) (n_tensors + 1) * ggml_tensor_overhead();

        ggml_init_params f32p = {
            /* mem_size   = */ ctx_size,
            /* mem_buffer = */ nullptr,
            /* no_alloc   = */ false,   // ctx owns the F32 data (CPU-readable)
        };
        model->ctx = ggml_init(f32p);
        if (!model->ctx) {
            err_out = "ggml_init for F32 context failed";
            return {nullptr, kokoro_model_deleter{}};
        }

        for (int64_t i = 0; i < n_tensors; ++i) {
            const char * name = gguf_get_tensor_name(model->gguf, i);
            ggml_tensor * src = ggml_get_tensor(model->gguf_ctx, name);
            if (!src) continue;

            const int n_dims = ggml_n_dims(src);
            ggml_tensor * dst = ggml_new_tensor(
                model->ctx, GGML_TYPE_F32, n_dims, src->ne);
            if (!dst) {
                err_out = std::string("F32 alloc failed for tensor '") + name + "'";
                return {nullptr, kokoro_model_deleter{}};
            }
            ggml_set_name(dst, name);

            const int64_t nelem = ggml_nelements(src);
            float * out = (float *) dst->data;
            if (src->type == GGML_TYPE_F32) {
                std::memcpy(out, src->data, (size_t) nelem * sizeof(float));
            } else if (src->type == GGML_TYPE_F16) {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) src->data, out, nelem);
            } else {
                const ggml_type_traits * tr = ggml_get_type_traits(src->type);
                if (!tr || !tr->to_float) {
                    err_out = std::string("no dequantizer for tensor '") + name
                            + "' (type " + std::to_string((int) src->type) + ")";
                    return {nullptr, kokoro_model_deleter{}};
                }
                tr->to_float(src->data, out, nelem);
            }
        }

        // The on-disk tensors are no longer read after this point; the backend
        // buffer + gguf_ctx stay alive (freed in the deleter) but every
        // downstream lookup goes through the all-F32 `ctx`.
    }

    // Bind the published Kokoro GGUF schema, while accepting the older
    // unprefixed dev names from pre-publication GGUFs. Missing required
    // tensors are a hard load error: otherwise the synth path can appear to
    // work while silently skipping the real model weights.
    if (!require_tensor_any(
        model->ctx,
        KOKORO_TENSOR_BERT_TOKEN_EMBD,
        "BERT token embedding",
        err_out)) {
        return {nullptr, kokoro_model_deleter{}};
    }

    if (!require_tensor_any(model->ctx, KOKORO_TENSOR_BERT_ATTN_Q, "BERT attention Q", err_out)) {
        return {nullptr, kokoro_model_deleter{}};
    }
    if (!require_tensor_any(model->ctx, KOKORO_TENSOR_F0_PROJ, "F0 projection", err_out)) {
        return {nullptr, kokoro_model_deleter{}};
    }
    if (!require_tensor_any(model->ctx, KOKORO_TENSOR_N_PROJ, "noise projection", err_out)) {
        return {nullptr, kokoro_model_deleter{}};
    }
    if (!require_tensor_any(model->ctx, KOKORO_TENSOR_GEN_CONV_POST, "generator post convolution", err_out)) {
        return {nullptr, kokoro_model_deleter{}};
    }

    if (!require_tensor_any(
        model->ctx,
        KOKORO_TENSOR_DURATION_PROJ,
        "duration projection",
        err_out)) {
        return {nullptr, kokoro_model_deleter{}};
    }

    return kokoro_model_ptr(model.release(), kokoro_model_deleter{});
}

// ---------------------------------------------------------------------------
// Voice preset loader (.bin = raw fp32 (N, 1, style_dim))
// ---------------------------------------------------------------------------

kokoro_status kokoro_load_voice_preset(
        const std::string & bin_path,
        int style_dim,
        kokoro_voice_preset & out,
        std::string & err_out) noexcept {
    err_out.clear();
    if (style_dim <= 0) {
        err_out = "invalid style_dim";
        return KOKORO_E_INVALID_ARG;
    }
    std::ifstream fin(bin_path, std::ios::binary | std::ios::ate);
    if (!fin) {
        err_out = "open failed: " + bin_path;
        return KOKORO_E_VOICE_LOAD_FAIL;
    }
    const std::streamsize sz = fin.tellg();
    if (sz <= 0 || (sz % (std::streamsize) (sizeof(float) * (size_t) style_dim)) != 0) {
        err_out = "voice preset size " + std::to_string((long long) sz)
                + " not a positive multiple of style_dim*4 (" + std::to_string(style_dim * 4) + ")";
        return KOKORO_E_VOICE_LOAD_FAIL;
    }
    fin.seekg(0, std::ios::beg);
    out.data.assign((size_t) (sz / (std::streamsize) sizeof(float)), 0.0f);
    fin.read((char *) out.data.data(), sz);
    if (!fin) {
        err_out = "voice preset read failed: " + bin_path;
        return KOKORO_E_VOICE_LOAD_FAIL;
    }
    out.style_dim   = style_dim;
    out.n_positions = (int) (out.data.size() / (size_t) style_dim);
    out.id          = bin_path; // caller usually overrides with a friendly id
    return KOKORO_OK;
}

// ---------------------------------------------------------------------------
// Phonemization (delegates to kokoro-phonemes.cpp)
// ---------------------------------------------------------------------------

std::vector<int32_t> kokoro_phonemize(const std::string & text) {
    // Real G2P when libespeak-ng is linked: text → en-us IPA → Kokoro vocab
    // ids, wrapped as the model input_ids [PAD, *ids, PAD]. Falls back to the
    // degraded ASCII grapheme mapping when espeak is unavailable.
    if (espeak_available()) {
        return phonemize_to_input_ids(text);
    }
    return phonemize_ascii(text);
}

kokoro_g2p_kind kokoro_g2p_kind_of_build() noexcept {
    return espeak_available() ? KOKORO_G2P_ESPEAK : KOKORO_G2P_ASCII;
}

// Shared synthesis core: takes an already-phonemized, wrapped input-id run
// ([PAD, *ids, PAD]) and runs the predictor → decoder → 24 kHz PCM path. Both
// public entries (`kokoro_synthesize` = raw text via kokoro_phonemize,
// `kokoro_synthesize_ipa` = precomputed espeak IPA via ipa_to_token_ids) funnel
// here so the two G2P front-ends share one identical back-end. `model`/`voice`
// are already validated by the caller; this takes the model lock.
static kokoro_status kokoro_synthesize_from_input_ids(
        const kokoro_model * model,
        const kokoro_voice_preset & voice,
        std::vector<int32_t> phonemes,
        float speed_mult,
        kokoro_audio & out,
        std::string & err_out) noexcept {
    std::lock_guard<std::mutex> lk(const_cast<kokoro_model *>(model)->mu);

    out.sample_rate = model->hparams.sample_rate;

    if (phonemes.size() > 510) phonemes.resize(510);

    // 2. Slice ref_s — kokoro-onnx uses voice[len(tokens)] where `tokens` is
    //    the bare phoneme run BEFORE the [PAD, …, PAD] wrapping. `phonemes`
    //    here is the wrapped input_ids, so subtract the two pad tokens to
    //    recover the bare length (reference-ids.json: style_row == len(ids)).
    const int style_dim = voice.style_dim;
    const int bare_len = std::max(0, (int) phonemes.size() - 2);
    int slot = std::min(voice.n_positions - 1, std::max(0, bare_len));
    const float * ref_s = voice.data.data() + (size_t) slot * (size_t) style_dim;

    // Predictor → decoder → 24 kHz PCM. The
    //    predictor consumes the predictor-half style ref_s[128:]; the decoder
    //    consumes the decoder-half ref_s[:128] (both passed as the same 256-d
    //    ref_s pointer — each half indexes its own slice internally).
    {
#if defined(__APPLE__)
        // Accelerate's AMX-backed SGEMM remains faster than GGML's CPU graph on
        // Apple Silicon. Android and other non-Apple targets use GGML so their
        // hot convolutions reach the optimized backend instead of scalar glue.
        detail::KokoroComputeBackendScope compute_scope(
            std::getenv("KOKORO_FORCE_GGML")
                ? &const_cast<kokoro_model *>(model)->compute
                : nullptr);
#else
        detail::KokoroComputeBackendScope compute_scope(
            std::getenv("KOKORO_DISABLE_GGML")
                ? nullptr
                : &const_cast<kokoro_model *>(model)->compute);
#endif
        kokoro_phase_timer synth_timer("synthesize total");
        PredictorOut pred;
        {
            kokoro_phase_timer t("predictor forward");
            if (!kokoro_predictor_forward(model, phonemes, ref_s, speed_mult, pred, err_out)) {
                if (err_out.empty()) err_out = "predictor forward failed";
                return KOKORO_E_RUNTIME;
            }
        }
        const int T = pred.T_frame;
        if (T <= 0 || (int) pred.asr.size() != T * 512) {
            err_out = "predictor produced empty/invalid asr (T=" + std::to_string(T) + ")";
            return KOKORO_E_RUNTIME;
        }

        // Transpose asr [T, 512] (T-major) → [512, T] (channel-major).
        std::vector<float> asr_ct((size_t) 512 * (size_t) T);
        for (int t = 0; t < T; ++t) {
            const float * row = pred.asr.data() + (size_t) t * 512;
            for (int c = 0; c < 512; ++c) {
                asr_ct[(size_t) c * (size_t) T + t] = row[c];
            }
        }

        {
            kokoro_phase_timer t("decoder forward");
            if (!kokoro_decoder_forward(model, asr_ct.data(), T,
                                        pred.F0_pred.data(), pred.N_pred.data(),
                                        ref_s, out.samples, err_out)) {
                if (err_out.empty()) err_out = "decoder forward failed";
                return KOKORO_E_RUNTIME;
            }
        }
        return KOKORO_OK;
    }

}

// Shared model/voice validation for both public synthesis entries.
static kokoro_status kokoro_validate_synth_args(
        const kokoro_model * model,
        const kokoro_voice_preset & voice,
        std::string & err_out) noexcept {
    if (!model) {
        err_out = "null model";
        return KOKORO_E_INVALID_ARG;
    }
    if (voice.data.empty() || voice.style_dim <= 0) {
        err_out = "empty / malformed voice preset";
        return KOKORO_E_INVALID_ARG;
    }
    return KOKORO_OK;
}

kokoro_status kokoro_synthesize(
        const kokoro_model * model,
        const kokoro_voice_preset & voice,
        const std::string & text,
        float speed_mult,
        kokoro_audio & out,
        std::string & err_out) noexcept {
    err_out.clear();
    out.samples.clear();
    out.sample_rate = 24000;

    kokoro_status vst = kokoro_validate_synth_args(model, voice, err_out);
    if (vst != KOKORO_OK) return vst;
    if (text.empty()) {
        err_out = "empty text";
        return KOKORO_E_INVALID_ARG;
    }

    // 1. Phonemize (espeak-ng IPA when linked, else lossy ASCII grapheme map).
    return kokoro_synthesize_from_input_ids(
        model, voice, kokoro_phonemize(text), speed_mult, out, err_out);
}

kokoro_status kokoro_synthesize_ipa(
        const kokoro_model * model,
        const kokoro_voice_preset & voice,
        const std::string & ipa,
        float speed_mult,
        kokoro_audio & out,
        std::string & err_out) noexcept {
    err_out.clear();
    out.samples.clear();
    out.sample_rate = 24000;

    kokoro_status vst = kokoro_validate_synth_args(model, voice, err_out);
    if (vst != KOKORO_OK) return vst;
    if (ipa.empty()) {
        err_out = "empty ipa";
        return KOKORO_E_INVALID_ARG;
    }

    // 1. Map the caller-supplied espeak-ng IPA straight to Kokoro vocab ids,
    //    wrapped as the model input_ids [PAD, *ids, PAD] — no internal G2P.
    return kokoro_synthesize_from_input_ids(
        model, voice, wrap_input_ids(ipa_to_token_ids(ipa)), speed_mult, out,
        err_out);
}

int kokoro_sample_rate(const kokoro_model * model) noexcept {
    return model ? model->hparams.sample_rate : 24000;
}

const kokoro_hparams * kokoro_get_hparams(const kokoro_model * model) noexcept {
    return model ? &model->hparams : nullptr;
}

// Exposed for kokoro-predictor.cpp / kokoro-decoder.cpp — they look up
// trained tensors by name from the loader-owned ggml_context. Keeping
// this internal-by-convention (not in kokoro.h) preserves the public
// surface while giving the sibling TUs a stable handle.
// Forward-declare here so -Wmissing-declarations sees a prior declaration
// at the definition site (the matching extern lives in the sibling TUs).
ggml_context * kokoro_model_ggml_ctx(const kokoro_model * model) {
    return model ? model->ctx : nullptr;
}

} // namespace eliza_kokoro
