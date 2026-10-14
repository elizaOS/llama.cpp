// Scripted implementation of the Paradee / misaki contract headers in this
// directory. Deterministic, weight-free: it lets the FFI unit test drive
// load / synth / cancel / evict paths of src/eliza-inference-ffi.cpp.
//
// Vocab: the Kokoro ids the real model uses for punctuation and space
// (; 1, : 2, , 3, . 4, ! 5, ? 6, — 9, … 10, space 16); 'a'..'z' -> 43..68;
// '#' is out of vocab and dropped; any other code point -> 100.

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
    std::lock_guard<std::mutex> lock(g_state.sizes_mutex);
    g_state.input_sizes.clear();
}

namespace eliza_paradee {

struct paradee_model {
    std::string path;
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
    return paradee_model_ptr(new paradee_model{ gguf_path });
}

int paradee_sample_rate(const paradee_model * model) noexcept {
    return model ? 24000 : -1;
}

paradee_status paradee_phonemes_to_ids(
    const paradee_model * model,
    const std::string & s,
    std::vector<int32_t> & ids,
    std::string & err_out) noexcept {
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
        else if (cp == "#") continue;
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
    if (!(opts.speed > 0.0f)) { err_out = "fake: bad speed"; return PARADEE_E_INVALID_ARG; }
    g_state.synth_calls.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(g_state.sizes_mutex);
        g_state.input_sizes.push_back(input_ids.size());
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

} // namespace eliza_misaki
