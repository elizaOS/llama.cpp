// SPDX-License-Identifier: MIT
//
// paradee.cpp — Paradee-8M v1.0 loader and synthesis (see include/paradee.h).
//
// The text side and decoder are Kokoro's modules at smaller widths, so this
// file only binds the GGUF to the generalised tools/kokoro layers
// (predictor_forward / decoder_forward) with the student-specific options:
//
//   - two learned styles of different sizes (text 32, decoder 16) instead of
//     one 256-d voice row,
//   - the asr_proj GELU MLP (192 -> 512 -> 512) between the text encoder and
//     the decoder,
//   - the trained InstanceNorm affine inside every AdaIN1d,
//   - torch.round (half to even) for the integer durations,
//   - the canonical source STFT (exact zeros for near-empty bins),
//   - the phase-locking filter after the generator.

#include "paradee.h"

#include "kokoro-decoder.h"
#include "kokoro-layers.h"
#include "kokoro-model-internal.h"
#include "kokoro-predictor.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

namespace eliza_paradee {

using eliza_kokoro::DecoderConfig;
using eliza_kokoro::PredictorConfig;
using eliza_kokoro::PredictorOut;

const char * paradee_status_str(paradee_status st) noexcept {
    switch (st) {
        case PARADEE_OK:               return "ok";
        case PARADEE_E_INVALID_ARG:    return "invalid argument";
        case PARADEE_E_UNKNOWN_SYMBOL: return "phoneme outside the vocabulary";
        case PARADEE_E_TOO_LONG:       return "too many phonemes";
        case PARADEE_E_RUNTIME:        return "runtime error";
        case PARADEE_CANCELLED:        return "cancelled";
        case PARADEE_E_OOM:            return "out of memory";
    }
    return "unknown";
}

struct paradee_hparams {
    int text_n_layer = 0, text_n_head = 0, text_d_model = 0, text_d_ff = 0;
    int text_vocab_size = 0, text_max_position = 0;
    int hidden_dim = 0, predictor_n_layer = 0, predictor_max_dur = 0, text_encoder_kernel = 0;
    int style_text_dim = 0, style_decoder_dim = 0, asr_dim = 0;
    int up_rates[2] = { 0, 0 };
    int istft_hop = 0;
    int sample_rate = 0;
    int pad_id = 0, max_ids = 0;
};

struct paradee_model {
    paradee_hparams hp;
    paradee_lock_params lock;
    std::unordered_map<uint32_t, int32_t> vocab;  // code point -> id
    gguf_context * gguf = nullptr;
    ggml_context * file_ctx = nullptr;   // tensors as stored
    ggml_context * ctx = nullptr;        // all-F32 view the layers read
    std::vector<float> style_text, style_dec;
    std::mutex mu;
};

void paradee_model_deleter::operator()(paradee_model * m) const noexcept {
    if (!m) return;
    if (m->ctx && m->ctx != m->file_ctx) ggml_free(m->ctx);
    if (m->file_ctx) ggml_free(m->file_ctx);
    if (m->gguf) gguf_free(m->gguf);
    delete m;
}

namespace {

bool read_i32(gguf_context * g, const char * key, int & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) { err = std::string("missing GGUF key ") + key; return false; }
    switch (gguf_get_kv_type(g, id)) {
        case GGUF_TYPE_INT32:  out = gguf_get_val_i32(g, id); return true;
        case GGUF_TYPE_UINT32: out = (int) gguf_get_val_u32(g, id); return true;
        default: err = std::string("GGUF key ") + key + " is not an integer"; return false;
    }
}

bool read_f32(gguf_context * g, const char * key, float & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) { err = std::string("missing GGUF key ") + key; return false; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_FLOAT32) { err = std::string("GGUF key ") + key + " is not f32"; return false; }
    out = gguf_get_val_f32(g, id);
    return true;
}

bool read_i32_array(gguf_context * g, const char * key, std::vector<int> & out, std::string & err) {
    const int64_t id = gguf_find_key(g, key);
    if (id < 0) { err = std::string("missing GGUF key ") + key; return false; }
    if (gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != GGUF_TYPE_INT32) {
        err = std::string("GGUF key ") + key + " is not an int32 array";
        return false;
    }
    const size_t n = gguf_get_arr_n(g, id);
    const int32_t * data = (const int32_t *) gguf_get_arr_data(g, id);
    out.assign(data, data + n);
    return true;
}

// Decode one UTF-8 code point at s[i]; advances i. Returns false on invalid input.
bool utf8_next(const std::string & s, size_t & i, uint32_t & cp) {
    const unsigned char c = (unsigned char) s[i];
    int len = 0;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
    else return false;
    if (i + (size_t) len > s.size()) return false;
    for (int k = 1; k < len; ++k) {
        const unsigned char cc = (unsigned char) s[i + (size_t) k];
        if ((cc & 0xC0) != 0x80) return false;
        cp = (cp << 6) | (cc & 0x3F);
    }
    // Reject overlong forms, surrogates and out-of-range values.
    if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) ||
        cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
        return false;
    }
    i += (size_t) len;
    return true;
}

bool load_vocab(gguf_context * g, int n_token, std::unordered_map<uint32_t, int32_t> & vocab, std::string & err) {
    const int64_t sid = gguf_find_key(g, "paradee.vocab.symbols");
    std::vector<int> ids;
    if (sid < 0 || gguf_get_kv_type(g, sid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, sid) != GGUF_TYPE_STRING) {
        err = "missing GGUF string array paradee.vocab.symbols";
        return false;
    }
    if (!read_i32_array(g, "paradee.vocab.ids", ids, err)) return false;
    const size_t n = gguf_get_arr_n(g, sid);
    if (n != ids.size() || n == 0) { err = "paradee.vocab.symbols / ids length mismatch"; return false; }
    for (size_t i = 0; i < n; ++i) {
        const std::string sym = gguf_get_arr_str(g, sid, i);
        size_t pos = 0;
        uint32_t cp = 0;
        if (sym.empty() || !utf8_next(sym, pos, cp) || pos != sym.size()) {
            err = "vocabulary symbol " + std::to_string(i) + " is not one code point";
            return false;
        }
        const int id = ids[i];
        if (id <= 0 || id >= n_token) { err = "vocabulary id out of range"; return false; }
        if (!vocab.emplace(cp, id).second) { err = "duplicate vocabulary symbol"; return false; }
    }
    return true;
}

std::string codepoint_label(uint32_t cp) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "U+%04X", (unsigned) cp);
    return buf;
}

} // namespace

paradee_model_ptr paradee_load_model(const std::string & gguf_path, std::string & err) noexcept {
    err.clear();
    try {
        paradee_model_ptr m(new paradee_model());
        gguf_init_params params = { /* no_alloc = */ false, /* ctx = */ &m->file_ctx };
        m->gguf = gguf_init_from_file(gguf_path.c_str(), params);
        if (!m->gguf || !m->file_ctx) {
            err = "gguf_init_from_file failed for '" + gguf_path + "'";
            return nullptr;
        }
        gguf_context * g = m->gguf;
        {
            const int64_t id = gguf_find_key(g, "general.architecture");
            const char * arch = (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_STRING) ? gguf_get_val_str(g, id) : "";
            if (std::string(arch) != "paradee") {
                err = std::string("expected general.architecture='paradee', got '") + arch + "'";
                return nullptr;
            }
        }
        paradee_hparams & h = m->hp;
        std::vector<int> rates;
        bool ok =
            read_i32(g, "paradee.text.n_layer", h.text_n_layer, err) &&
            read_i32(g, "paradee.text.n_head", h.text_n_head, err) &&
            read_i32(g, "paradee.text.d_model", h.text_d_model, err) &&
            read_i32(g, "paradee.text.d_ff", h.text_d_ff, err) &&
            read_i32(g, "paradee.text.vocab_size", h.text_vocab_size, err) &&
            read_i32(g, "paradee.text.max_position", h.text_max_position, err) &&
            read_i32(g, "paradee.hidden_dim", h.hidden_dim, err) &&
            read_i32(g, "paradee.predictor.n_layer", h.predictor_n_layer, err) &&
            read_i32(g, "paradee.predictor.max_dur", h.predictor_max_dur, err) &&
            read_i32(g, "paradee.text_encoder.kernel", h.text_encoder_kernel, err) &&
            read_i32(g, "paradee.style.text_dim", h.style_text_dim, err) &&
            read_i32(g, "paradee.style.decoder_dim", h.style_decoder_dim, err) &&
            read_i32(g, "paradee.asr.dim", h.asr_dim, err) &&
            read_i32_array(g, "paradee.decoder.upsample_rates", rates, err) &&
            read_i32(g, "paradee.decoder.istft_hop", h.istft_hop, err) &&
            read_i32(g, "paradee.audio.sample_rate", h.sample_rate, err) &&
            read_i32(g, "paradee.input.pad_id", h.pad_id, err) &&
            read_i32(g, "paradee.input.max_ids", h.max_ids, err) &&
            read_i32(g, "paradee.lock.n_fft", m->lock.n_fft, err) &&
            read_i32(g, "paradee.lock.hop", m->lock.hop, err) &&
            read_i32(g, "paradee.lock.smooth_frames", m->lock.smooth_frames, err) &&
            read_i32(g, "paradee.lock.f0_hop", m->lock.f0_hop, err) &&
            read_f32(g, "paradee.lock.voiced_f0_hz", m->lock.voiced_f0_hz, err);
        if (!ok) return nullptr;
        int band_lo = 0, band_hi = 0;
        if (!read_i32(g, "paradee.lock.band_lo_hz", band_lo, err) ||
            !read_i32(g, "paradee.lock.band_hi_hz", band_hi, err)) {
            return nullptr;
        }
        m->lock.band_lo_hz = (float) band_lo;
        m->lock.band_hi_hz = (float) band_hi;
        m->lock.sample_rate = h.sample_rate;
        if (rates.size() != 2 || h.pad_id != 0 || h.max_ids < 3 || h.max_ids > h.text_max_position ||
            h.text_n_head <= 0 || h.text_d_model % h.text_n_head != 0 || h.sample_rate <= 0) {
            err = "unsupported Paradee hparams";
            return nullptr;
        }
        h.up_rates[0] = rates[0];
        h.up_rates[1] = rates[1];
        if (!load_vocab(g, h.text_vocab_size, m->vocab, err)) return nullptr;

        // All-F32 working context (the converter's default output is already
        // F32; an F16 file is widened once here).
        bool all_f32 = true;
        const int64_t n_tensors = gguf_get_n_tensors(g);
        size_t f32_bytes = 0;
        for (int64_t i = 0; i < n_tensors; ++i) {
            ggml_tensor * t = ggml_get_tensor(m->file_ctx, gguf_get_tensor_name(g, i));
            if (!t) continue;
            if (t->type != GGML_TYPE_F32) all_f32 = false;
            f32_bytes += GGML_PAD((size_t) ggml_nelements(t) * sizeof(float), GGML_MEM_ALIGN);
        }
        if (all_f32) {
            m->ctx = m->file_ctx;
        } else {
            ggml_init_params p = { f32_bytes + (size_t) (n_tensors + 1) * ggml_tensor_overhead(), nullptr, false };
            m->ctx = ggml_init(p);
            if (!m->ctx) { err = "ggml_init failed"; return nullptr; }
            for (int64_t i = 0; i < n_tensors; ++i) {
                const char * name = gguf_get_tensor_name(g, i);
                ggml_tensor * src = ggml_get_tensor(m->file_ctx, name);
                if (!src) continue;
                ggml_tensor * dst = ggml_new_tensor(m->ctx, GGML_TYPE_F32, ggml_n_dims(src), src->ne);
                ggml_set_name(dst, name);
                const int64_t n = ggml_nelements(src);
                if (src->type == GGML_TYPE_F32) {
                    std::memcpy(dst->data, src->data, (size_t) n * sizeof(float));
                } else if (src->type == GGML_TYPE_F16) {
                    ggml_fp16_to_fp32_row((const ggml_fp16_t *) src->data, (float *) dst->data, n);
                } else {
                    err = std::string("unsupported tensor type for '") + name + "'";
                    return nullptr;
                }
            }
        }

        ggml_tensor * st = ggml_get_tensor(m->ctx, "paradee.style.text");
        ggml_tensor * sd = ggml_get_tensor(m->ctx, "paradee.style.decoder");
        if (!st || !sd || ggml_nelements(st) != h.style_text_dim || ggml_nelements(sd) != h.style_decoder_dim) {
            err = "missing or mis-sized paradee.style.text / paradee.style.decoder";
            return nullptr;
        }
        m->style_text.assign((const float *) st->data, (const float *) st->data + h.style_text_dim);
        m->style_dec.assign((const float *) sd->data, (const float *) sd->data + h.style_decoder_dim);

        const char * required[] = {
            "paradee.bert.token_embd.weight", "paradee.bert.layer.attn_q.weight",
            "paradee.bert_encoder.weight", "paradee.predictor.duration_proj.weight",
            "paradee.predictor.F0_proj.weight", "paradee.text_encoder.embd.weight",
            "paradee.asr_proj.0.weight", "paradee.asr_proj.1.weight",
            "paradee.decoder.encode.conv1.weight", "paradee.gen.conv_post.weight",
        };
        for (const char * name : required) {
            if (!ggml_get_tensor(m->ctx, name)) {
                err = std::string("required tensor missing: ") + name;
                return nullptr;
            }
        }
        ggml_tensor * be = ggml_get_tensor(m->ctx, "paradee.bert_encoder.weight");
        ggml_tensor * ap = ggml_get_tensor(m->ctx, "paradee.asr_proj.0.weight");
        ggml_tensor * tok = ggml_get_tensor(m->ctx, "paradee.bert.token_embd.weight");
        if ((int) be->ne[1] != h.hidden_dim || (int) be->ne[0] != h.text_d_model ||
            (int) ap->ne[1] != h.asr_dim || (int) tok->ne[1] != h.text_vocab_size) {
            err = "tensor shapes do not match the Paradee hparams";
            return nullptr;
        }
        return m;
    } catch (const std::exception & e) {
        err = std::string("paradee_load_model: ") + e.what();
        return nullptr;
    }
}

int paradee_sample_rate(const paradee_model * model) noexcept {
    return model ? model->hp.sample_rate : 0;
}

const paradee_lock_params * paradee_get_lock_params(const paradee_model * model) noexcept {
    return model ? &model->lock : nullptr;
}

namespace {

// Vocabulary ids of every symbol; `max_symbols` = 0 means no limit.
paradee_status map_symbols(
        const paradee_model * model,
        const std::string & misaki_utf8,
        size_t max_symbols,
        bool pads,
        std::vector<int32_t> & ids,
        std::string & err) noexcept {
    err.clear();
    ids.clear();
    if (!model) { err = "null model"; return PARADEE_E_INVALID_ARG; }
    if (misaki_utf8.empty()) { err = "empty phoneme string"; return PARADEE_E_INVALID_ARG; }
    try {
        if (pads) ids.push_back(model->hp.pad_id);
        const size_t lead = ids.size();
        size_t i = 0;
        while (i < misaki_utf8.size()) {
            uint32_t cp = 0;
            const size_t at = i;
            if (!utf8_next(misaki_utf8, i, cp)) {
                err = "invalid UTF-8 at byte " + std::to_string(at);
                ids.clear();
                return PARADEE_E_INVALID_ARG;
            }
            const auto it = model->vocab.find(cp);
            if (it == model->vocab.end()) {
                err = "phoneme " + codepoint_label(cp) + " at byte " + std::to_string(at) +
                      " is not in the Paradee vocabulary";
                ids.clear();
                return PARADEE_E_UNKNOWN_SYMBOL;
            }
            if (max_symbols && ids.size() - lead >= max_symbols) {
                err = "more than " + std::to_string(max_symbols) + " phonemes";
                ids.clear();
                return PARADEE_E_TOO_LONG;
            }
            ids.push_back(it->second);
        }
        if (pads) ids.push_back(model->hp.pad_id);
    } catch (const std::bad_alloc &) {
        err = "out of memory";
        ids.clear();
        return PARADEE_E_OOM;
    } catch (const std::exception & e) {
        err = e.what();
        ids.clear();
        return PARADEE_E_RUNTIME;
    }
    return PARADEE_OK;
}

} // namespace

paradee_status paradee_phonemes_to_ids(
        const paradee_model * model,
        const std::string & misaki_utf8,
        std::vector<int32_t> & ids,
        std::string & err) noexcept {
    if (!model) { ids.clear(); err = "null model"; return PARADEE_E_INVALID_ARG; }
    return map_symbols(model, misaki_utf8, (size_t) model->hp.max_ids - 2, /*pads*/ true, ids, err);
}

paradee_status paradee_phonemes_to_symbol_ids(
        const paradee_model * model,
        const std::string & misaki_utf8,
        std::vector<int32_t> & ids,
        std::string & err) noexcept {
    return map_symbols(model, misaki_utf8, 0, /*pads*/ false, ids, err);
}

namespace {

// Layer strategies Paradee opts into (kokoro-layers.h): per-tap SGEMM
// convolutions and the polynomial Snake activation. PARADEE_EXACT_LAYERS=1
// turns them off (the arithmetic Kokoro-82M uses), for A/B checks.
eliza_kokoro::detail::KokoroLayerOptions fast_layers() {
    eliza_kokoro::detail::KokoroLayerOptions o;
    const char * exact = std::getenv("PARADEE_EXACT_LAYERS");
    const bool fast = !(exact && exact[0] == '1');
    o.tap_gemm_conv = fast;
    o.fast_snake = fast;
    return o;
}

struct cancel_state {
    paradee_cancel_fn fn;
    void * user;
};

bool cancel_trampoline(void * p) {
    const cancel_state * c = (const cancel_state *) p;
    return c && c->fn && c->fn(c->user);
}

} // namespace

namespace {

// Decoder + generator + phase lock. Caller holds model->mu.
paradee_status decode_locked(
        const paradee_model * model,
        const float * asr_ct, int A, int T,
        const float * F0, const float * N,
        paradee_cancel_fn cancel, void * user,
        std::vector<float> & dec, std::vector<float> & src, std::vector<float> & pcm,
        std::string & err, const std::vector<float> * src_in = nullptr) {
    const paradee_hparams & h = model->hp;
    cancel_state cs{ cancel, user };
    DecoderConfig dc;
    dc.ns = "paradee";
    dc.instnorm_affine = true;
    dc.canonical_source_stft = true;
    dc.precise_source = true;
    dc.up_rates[0] = h.up_rates[0];
    dc.up_rates[1] = h.up_rates[1];
    dc.istft_hop = h.istft_hop;
    dc.har_source_out = &src;
    dc.har_source_in = src_in;
    dc.cancel = cancel ? cancel_trampoline : nullptr;
    dc.cancel_user = &cs;
    if (!eliza_kokoro::decoder_forward(model->ctx, dc, asr_ct, A, T, F0, N,
                                       model->style_dec.data(), dec, err)) {
        if (dc.cancelled) { err = "cancelled"; return PARADEE_CANCELLED; }
        if (err.empty()) err = "decoder failed";
        return PARADEE_E_RUNTIME;
    }
    if (!paradee_phase_lock(dec.data(), dec.size(), src.data(), src.size(),
                            F0, (size_t) 2 * (size_t) T, model->lock, pcm, cancel, user)) {
        if (cancel && cancel(user)) { err = "cancelled"; return PARADEE_CANCELLED; }
        err = "phase lock rejected its parameters";
        return PARADEE_E_RUNTIME;
    }
    if (cancel && cancel(user)) { err = "cancelled"; return PARADEE_CANCELLED; }
    return PARADEE_OK;
}

} // namespace

paradee_status paradee_synthesize_ids_staged(
        const paradee_model * model,
        const int32_t * ids, size_t n,
        float speed,
        paradee_cancel_fn cancel, void * user,
        std::vector<float> & pcm,
        std::string & err,
        paradee_stages * stages) noexcept {
    err.clear();
    pcm.clear();
    if (!model) { err = "null model"; return PARADEE_E_INVALID_ARG; }
    const paradee_hparams & h = model->hp;
    if (!ids || n < 3 || n > (size_t) h.max_ids) {
        err = "expected 3.." + std::to_string(h.max_ids) + " input ids, got " + std::to_string(n);
        return n > (size_t) h.max_ids ? PARADEE_E_TOO_LONG : PARADEE_E_INVALID_ARG;
    }
    if (ids[0] != h.pad_id || ids[n - 1] != h.pad_id) {
        err = "input ids must start and end with the pad id 0";
        return PARADEE_E_INVALID_ARG;
    }
    for (size_t i = 0; i < n; ++i) {
        if (ids[i] < 0 || ids[i] >= h.text_vocab_size) {
            err = "input id " + std::to_string(ids[i]) + " out of range";
            return PARADEE_E_INVALID_ARG;
        }
    }
    if (!std::isfinite(speed) || speed <= 0.0f) {
        err = "speed must be finite and > 0";
        return PARADEE_E_INVALID_ARG;
    }

    try {
        std::lock_guard<std::mutex> lk(const_cast<paradee_model *>(model)->mu);
        eliza_kokoro::detail::KokoroLayerOptionsScope layer_scope(fast_layers());
        cancel_state cs{ cancel, user };
        if (cancel && cancel(user)) { err = "cancelled"; return PARADEE_CANCELLED; }

        // 1. Text side.
        PredictorConfig pc;
        pc.ns = "paradee";
        pc.text_n_layer = h.text_n_layer;
        pc.text_n_head = h.text_n_head;
        pc.text_d_model = h.text_d_model;
        pc.text_d_ff = h.text_d_ff;
        pc.n_layer = h.predictor_n_layer;
        pc.max_dur = h.predictor_max_dur;
        pc.text_kernel = h.text_encoder_kernel;
        pc.style = model->style_text.data();
        pc.torch_round = true;
        pc.instnorm_affine = true;
        pc.asr_proj_mlp = true;
        pc.keep_bert = stages != nullptr;
        pc.cancel = cancel ? cancel_trampoline : nullptr;
        pc.cancel_user = &cs;
        PredictorOut pred;
        const std::vector<int32_t> id_vec(ids, ids + n);
        if (!eliza_kokoro::predictor_forward(model->ctx, pc, id_vec, speed, pred, err)) {
            if (pred.cancelled) { pcm.clear(); return PARADEE_CANCELLED; }
            if (err.empty()) err = "text side failed";
            return PARADEE_E_RUNTIME;
        }
        if (pred.sty != h.style_text_dim || pred.asr_dim != h.asr_dim || pred.T_frame <= 0) {
            err = "text side produced unexpected widths";
            return PARADEE_E_RUNTIME;
        }
        const int T = pred.T_frame;
        const int A = pred.asr_dim;
        std::vector<float> asr_ct((size_t) A * (size_t) T);
        for (int t = 0; t < T; ++t) {
            const float * row = pred.asr.data() + (size_t) t * A;
            for (int c = 0; c < A; ++c) asr_ct[(size_t) c * (size_t) T + t] = row[c];
        }
        if (cancel && cancel(user)) { err = "cancelled"; return PARADEE_CANCELLED; }

        // 2. Decoder + generator, 3. phase lock.
        std::vector<float> src, dec;
        const paradee_status dst = decode_locked(model, asr_ct.data(), A, T, pred.F0_pred.data(),
                                                 pred.N_pred.data(), cancel, user, dec, src, pcm, err);
        if (dst != PARADEE_OK) { pcm.clear(); return dst; }

        if (stages) {
            stages->T_phon = pred.T_phon;
            stages->T_frame = T;
            stages->d_model = h.text_d_model;
            stages->hid = pred.hid;
            stages->sty = pred.sty;
            stages->asr_dim = A;
            stages->bert = std::move(pred.bert);
            stages->d = std::move(pred.d);
            stages->dur = std::move(pred.duration);
            stages->dur_int = std::move(pred.pred_dur);
            stages->F0 = std::move(pred.F0_pred);
            stages->N = std::move(pred.N_pred);
            stages->asr = std::move(asr_ct);
            stages->dec = std::move(dec);
            stages->src = std::move(src);
            stages->wav = pcm;
        }
        return PARADEE_OK;
    } catch (const std::bad_alloc &) {
        pcm.clear();
        err = "paradee synthesis: out of memory";
        return PARADEE_E_OOM;
    } catch (const std::exception & e) {
        pcm.clear();
        err = std::string("paradee synthesis: ") + e.what();
        return PARADEE_E_RUNTIME;
    }
}

paradee_status paradee_synthesize_ids(
        const paradee_model * model,
        const int32_t * ids, size_t n,
        float speed,
        paradee_cancel_fn cancel, void * user,
        std::vector<float> & pcm,
        std::string & err) noexcept {
    return paradee_synthesize_ids_staged(model, ids, n, speed, cancel, user, pcm, err, nullptr);
}

} // namespace eliza_paradee

namespace eliza_paradee {

paradee_status paradee_synthesize(
        const paradee_model * model,
        const std::vector<int32_t> & input_ids,
        const paradee_synth_options & opts,
        paradee_audio & out,
        std::string & err) noexcept {
    out.sample_rate = model ? model->hp.sample_rate : 24000;
    return paradee_synthesize_ids_staged(model, input_ids.data(), input_ids.size(), opts.speed,
                                         opts.cancel, opts.cancel_user_data, out.samples, err, nullptr);
}

} // namespace eliza_paradee

namespace eliza_paradee {

paradee_status paradee_decode_staged(
        const paradee_model * model,
        const float * asr, int asr_dim, int T_frame,
        const float * F0, const float * N,
        const std::vector<float> * src,
        paradee_cancel_fn cancel, void * user,
        paradee_stages * stages,
        std::string & err) noexcept {
    err.clear();
    if (!model || !asr || !F0 || !N || !stages || T_frame <= 0 || asr_dim != model->hp.asr_dim) {
        err = "invalid decode arguments";
        return PARADEE_E_INVALID_ARG;
    }
    try {
        std::lock_guard<std::mutex> lk(const_cast<paradee_model *>(model)->mu);
        eliza_kokoro::detail::KokoroLayerOptionsScope layer_scope(fast_layers());
        stages->T_frame = T_frame;
        stages->asr_dim = asr_dim;
        const paradee_hparams & h = model->hp;
        const size_t n_src = (size_t) T_frame * 2 * (size_t) (h.up_rates[0] * h.up_rates[1] * h.istft_hop);
        if (src && src->size() != n_src) {
            err = "src must have " + std::to_string(n_src) + " samples";
            return PARADEE_E_INVALID_ARG;
        }
        return decode_locked(model, asr, asr_dim, T_frame, F0, N, cancel, user,
                             stages->dec, stages->src, stages->wav, err, src);
    } catch (const std::bad_alloc &) {
        err = "out of memory";
        return PARADEE_E_OOM;
    } catch (const std::exception & e) {
        err = e.what();
        return PARADEE_E_RUNTIME;
    }
}

} // namespace eliza_paradee
