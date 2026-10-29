// Scripted implementation of the Paradee / misaki contract headers in this
// directory. Deterministic, weight-free: it lets the FFI unit test drive
// load / synth / cancel / evict paths of src/eliza-inference-ffi.cpp.
//
// Vocab: the Kokoro ids the real model uses for punctuation and space
// (; 1, : 2, , 3, . 4, ! 5, ? 6, — 9, … 10, space 16); 'a'..'z' -> 43..68;
// '#' is out of vocab and dropped; any other code point -> 100.
//
// Model paths name the fake release: "delibes" -> family delibes, language
// es, phoneme format espeak-es, validated 360 ids and speed [0.85, 1.15],
// voice em_alex; "v12" -> Paradee v1.2's four voices (af_heart en-us,
// bf_isabella en-gb, am_puck en-us, bm_fable en-gb); else one voice af_heart.

#include "paradee.h"
#include "misaki-g2p.h"
#include "fake-paradee-control.h"

#include <chrono>
#include <cctype>
#include <thread>

static fake_paradee_state g_state;

fake_paradee_state & fake_paradee() { return g_state; }

void fake_paradee_reset() {
    g_state.model_loads = 0;
    g_state.lexicon_loads = 0;
    g_state.synth_calls = 0;
    g_state.in_flight = false;
    g_state.block_ms = 0;
    g_state.espeak_loads = 0;
    std::lock_guard<std::mutex> lock(g_state.sizes_mutex);
    g_state.input_sizes.clear();
    g_state.voices.clear();
    g_state.speeds.clear();
}

namespace eliza_paradee {

struct paradee_model {
    std::string path;
    bool delibes = false;
    std::vector<std::string> voices, accents;
};

void paradee_model_deleter::operator()(paradee_model * m) const noexcept {
    if (!m) return;
    g_state.live_models.fetch_sub(1);
    delete m;
}

paradee_model_ptr paradee_load_model(const std::string & gguf_path, std::string & err_out) noexcept {
    if (gguf_path.find("missing") != std::string::npos) {
        err_out = "fake: no such file: " + gguf_path;
        return nullptr;
    }
    g_state.model_loads.fetch_add(1);
    g_state.live_models.fetch_add(1);
    paradee_model * m = new paradee_model{ gguf_path, false, {}, {} };
    if (gguf_path.find("delibes") != std::string::npos) {
        m->delibes = true;
        m->voices = { "em_alex" };
        m->accents = { "" };
    } else if (gguf_path.find("v12") != std::string::npos) {
        m->voices = { "af_heart", "bf_isabella", "am_puck", "bm_fable" };
        m->accents = { "en-us", "en-gb", "en-us", "en-gb" };
    } else {
        m->voices = { "af_heart" };
        m->accents = { "" };
    }
    return paradee_model_ptr(m);
}

int paradee_sample_rate(const paradee_model * model) noexcept {
    return model ? 24000 : -1;
}

const char * paradee_family(const paradee_model * m) noexcept { return !m ? "" : m->delibes ? "delibes" : "paradee"; }
const char * paradee_language(const paradee_model * m) noexcept { return !m ? "" : m->delibes ? "es" : "en"; }
const char * paradee_phoneme_format(const paradee_model * m) noexcept {
    return !m ? "" : m->delibes ? "espeak-es" : "misaki";
}
int paradee_validated_max_ids(const paradee_model * m) noexcept { return !m ? 0 : m->delibes ? 360 : 512; }
bool paradee_validated_speed(const paradee_model * m, float & lo, float & hi) noexcept {
    if (!m) return false;
    lo = m->delibes ? 0.85f : PARADEE_MIN_SPEED;
    hi = m->delibes ? 1.15f : PARADEE_MAX_SPEED;
    return true;
}
int paradee_voice_count(const paradee_model * m) noexcept { return m ? (int) m->voices.size() : 0; }
int paradee_default_voice(const paradee_model * m) noexcept { return m ? 0 : -1; }
const char * paradee_voice_name(const paradee_model * m, int v) noexcept {
    return m && v >= 0 && v < (int) m->voices.size() ? m->voices[(size_t) v].c_str() : "";
}
const char * paradee_voice_accent(const paradee_model * m, int v) noexcept {
    return m && v >= 0 && v < (int) m->accents.size() ? m->accents[(size_t) v].c_str() : "";
}
int paradee_find_voice(const paradee_model * m, const std::string & name) noexcept {
    if (!m) return -1;
    for (size_t i = 0; i < m->voices.size(); ++i) if (m->voices[i] == name) return (int) i;
    return -1;
}

paradee_status paradee_phonemes_to_symbol_ids_lenient(
    const paradee_model * model,
    const std::string & s,
    std::vector<int32_t> & ids,
    size_t & n_dropped,
    std::string & err_out) noexcept {
    n_dropped = 0;
    if (!model) { err_out = "fake: null model"; return PARADEE_E_INVALID_ARG; }
    ids.clear();
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        size_t len = 1;
        if (c >= 0xF0) len = 4; else if (c >= 0xE0) len = 3; else if (c >= 0xC0) len = 2;
        const std::string cp = s.substr(i, len);
        i += len;
        if (cp == ";") ids.push_back(1);
        else if (cp == ":") ids.push_back(2);
        else if (cp == ",") ids.push_back(3);
        else if (cp == ".") ids.push_back(4);
        else if (cp == "!") ids.push_back(5);
        else if (cp == "?") ids.push_back(6);
        else if (cp == "\xE2\x80\x94") ids.push_back(9);
        else if (cp == "\xE2\x80\xA6") ids.push_back(10);
        else if (cp == " ") ids.push_back(16);
        else if (cp == "#") { ++n_dropped; continue; }
        else if (len == 1 && c >= 'a' && c <= 'z') ids.push_back(43 + (c - 'a'));
        else ids.push_back(100);
    }
    return PARADEE_OK;
}

paradee_status paradee_synthesize(
    const paradee_model * model,
    const std::vector<int32_t> & input_ids,
    const paradee_synth_options & opts,
    paradee_audio & out,
    std::string & err_out) noexcept {
    if (!model) { err_out = "fake: null model"; return PARADEE_E_INVALID_ARG; }
    if (input_ids.size() < 3 || input_ids.size() > (size_t) PARADEE_MAX_INPUT_IDS ||
        input_ids.front() != 0 || input_ids.back() != 0) {
        err_out = "fake: bad input_ids (size " + std::to_string(input_ids.size()) + ")";
        return PARADEE_E_INVALID_ARG;
    }
    if (!(opts.speed >= PARADEE_MIN_SPEED && opts.speed <= PARADEE_MAX_SPEED)) {
        err_out = "fake: bad speed";
        return PARADEE_E_INVALID_ARG;
    }
    if (opts.voice < -1 || opts.voice >= (int) model->voices.size()) {
        err_out = "fake: voice out of range";
        return PARADEE_E_INVALID_ARG;
    }
    g_state.synth_calls.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(g_state.sizes_mutex);
        g_state.input_sizes.push_back(input_ids.size());
        g_state.voices.push_back(opts.voice);
        g_state.speeds.push_back(opts.speed);
    }
    const int block_ms = g_state.block_ms.load();
    g_state.in_flight = true;
    for (int t = 0; t < block_ms; ++t) {
        if (opts.cancel && opts.cancel(opts.cancel_user_data)) {
            g_state.in_flight = false;
            return PARADEE_E_CANCELLED;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if (opts.cancel && opts.cancel(opts.cancel_user_data)) {
        g_state.in_flight = false;
        return PARADEE_E_CANCELLED;
    }
    out.sample_rate = 24000;
    out.samples.clear();
    for (size_t i = 1; i + 1 < input_ids.size(); ++i) {
        for (int k = 0; k < FAKE_PARADEE_SAMPLES_PER_ID; ++k) {
            out.samples.push_back((float) input_ids[i] / 1000.0f);
        }
    }
    g_state.in_flight = false;
    return PARADEE_OK;
}

} // namespace eliza_paradee

namespace eliza_misaki {

struct misaki_lexicon {
    misaki_fallback_kind kind;
};

void misaki_lexicon_deleter::operator()(misaki_lexicon * l) const noexcept {
    if (!l) return;
    g_state.live_lexicons.fetch_sub(1);
    delete l;
}

misaki_lexicon_ptr misaki_load_lexicon(
    const std::string & lexicon_path,
    const char * espeak_data_dir,
    std::string & err_out) noexcept {
    if (lexicon_path.find("missing") != std::string::npos) {
        err_out = "fake: no such lexicon: " + lexicon_path;
        return nullptr;
    }
    g_state.lexicon_loads.fetch_add(1);
    g_state.live_lexicons.fetch_add(1);
    return misaki_lexicon_ptr(new misaki_lexicon{
        espeak_data_dir ? MISAKI_FALLBACK_ESPEAK : MISAKI_FALLBACK_RULES });
}

misaki_fallback_kind misaki_get_fallback_kind(const misaki_lexicon * lexicon) noexcept {
    return lexicon ? lexicon->kind : MISAKI_FALLBACK_NONE;
}

// "G2P": lowercase ASCII letters and keep punctuation / spaces; digits are
// dropped. Text containing "\x01fail" reports an error.
bool misaki_phonemize(
    const misaki_lexicon * lexicon,
    const std::string & text,
    std::string & out,
    std::string & err_out) noexcept {
    if (!lexicon) { err_out = "fake: null lexicon"; return false; }
    if (text.find("\x01" "fail") != std::string::npos) { err_out = "fake: g2p failure"; return false; }
    out.clear();
    for (unsigned char c : text) {
        if (std::isalpha(c)) out.push_back((char) std::tolower(c));
        else if (std::isdigit(c)) continue;
        else out.push_back((char) c);
    }
    return true;
}

struct espeak_g2p {
    std::string language;
};

void espeak_g2p_deleter::operator()(espeak_g2p * g) const noexcept {
    if (!g) return;
    g_state.live_espeak.fetch_sub(1);
    delete g;
}

espeak_g2p_ptr espeak_g2p_open(const std::string & language, const char * espeak_data_dir,
                               std::string & err_out) noexcept {
    if (language != "es") { err_out = "fake: no eSpeak G2P for " + language; return nullptr; }
    if (!espeak_data_dir || std::string(espeak_data_dir).find("missing") != std::string::npos) {
        err_out = "fake: no eSpeak data";
        return nullptr;
    }
    g_state.espeak_loads.fetch_add(1);
    g_state.live_espeak.fetch_add(1);
    return espeak_g2p_ptr(new espeak_g2p{ language });
}

const char * espeak_g2p_language(const espeak_g2p * g) noexcept { return g ? g->language.c_str() : ""; }

// "G2P": like misaki_phonemize, and '\xC2\xBF' (U+00BF) is dropped as the
// real one drops symbols outside the vocabulary.
bool espeak_g2p_phonemize(const espeak_g2p * g, const std::string & text, std::string & out,
                          std::string & err_out) noexcept {
    if (!g) { err_out = "fake: null espeak g2p"; return false; }
    out.clear();
    for (size_t i = 0; i < text.size(); ++i) {
        const unsigned char c = (unsigned char) text[i];
        if (c == 0xC2 && i + 1 < text.size() && (unsigned char) text[i + 1] == 0xBF) { ++i; continue; }
        if (std::isalpha(c)) out.push_back((char) std::tolower(c));
        else if (std::isdigit(c)) continue;
        else out.push_back((char) c);
    }
    return true;
}

} // namespace eliza_misaki
