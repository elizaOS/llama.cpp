// SPDX-License-Identifier: MIT
//
// kokoro-phonemes.cpp — real G2P for the Kokoro fork path.
//
// text → espeak-ng (en-us IPA) → per-codepoint Kokoro vocab lookup → ids.
// See kokoro-phonemes.h for the contract.
//
// The vocab table is embedded (generated from
// `tts/kokoro/tokenizer.json` model.vocab). Every vocab key is a single
// Unicode codepoint; the IPA string is decoded codepoint-by-codepoint and
// each codepoint is mapped to its id. This reproduces the kokoro reference
// ids exactly (validated against reference-ids.json).
//
// When the build links libespeak-ng (-DKOKORO_USE_ESPEAK), phonemize_ipa()
// runs the real G2P. Otherwise the TS layer (which already runs espeak)
// supplies the IPA and the caller uses ipa_to_token_ids() directly.

#include "kokoro-phonemes.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#if defined(KOKORO_USE_ESPEAK)
#include <espeak-ng/speak_lib.h>
#include <mutex>
#endif

namespace eliza_kokoro {

namespace {

// Embedded Kokoro vocab: codepoint → id, sorted by codepoint for binary search.
// Generated from tokenizer.json model.vocab (115 entries, max id 177).
struct VocabEntry {
    char32_t cp;
    int32_t  id;
};

constexpr VocabEntry kVocab[] = {
    {0x0020u, 16}, {0x0021u, 5}, {0x0022u, 11}, {0x0024u, 0},
    {0x0028u, 12}, {0x0029u, 13}, {0x002Cu, 3}, {0x002Eu, 4},
    {0x003Au, 2}, {0x003Bu, 1}, {0x003Fu, 6}, {0x0041u, 24},
    {0x0049u, 25}, {0x004Fu, 31}, {0x0051u, 33}, {0x0053u, 35},
    {0x0054u, 36}, {0x0057u, 39}, {0x0059u, 41}, {0x0061u, 43},
    {0x0062u, 44}, {0x0063u, 45}, {0x0064u, 46}, {0x0065u, 47},
    {0x0066u, 48}, {0x0068u, 50}, {0x0069u, 51}, {0x006Au, 52},
    {0x006Bu, 53}, {0x006Cu, 54}, {0x006Du, 55}, {0x006Eu, 56},
    {0x006Fu, 57}, {0x0070u, 58}, {0x0071u, 59}, {0x0072u, 60},
    {0x0073u, 61}, {0x0074u, 62}, {0x0075u, 63}, {0x0076u, 64},
    {0x0077u, 65}, {0x0078u, 66}, {0x0079u, 67}, {0x007Au, 68},
    {0x00E6u, 72}, {0x00E7u, 78}, {0x00F0u, 81}, {0x00F8u, 116},
    {0x014Bu, 112}, {0x0153u, 120}, {0x0250u, 70}, {0x0251u, 69},
    {0x0252u, 71}, {0x0254u, 76}, {0x0255u, 77}, {0x0256u, 80},
    {0x0259u, 83}, {0x025Au, 85}, {0x025Bu, 86}, {0x025Cu, 87},
    {0x025Fu, 90}, {0x0261u, 92}, {0x0263u, 139}, {0x0264u, 140},
    {0x0265u, 99}, {0x0268u, 101}, {0x026Au, 102}, {0x026Fu, 110},
    {0x0270u, 111}, {0x0272u, 114}, {0x0273u, 113}, {0x0274u, 115},
    {0x0278u, 118}, {0x0279u, 123}, {0x027Bu, 126}, {0x027Du, 129},
    {0x027Eu, 125}, {0x0281u, 128}, {0x0282u, 130}, {0x0283u, 131},
    {0x0288u, 132}, {0x028Au, 135}, {0x028Bu, 136}, {0x028Cu, 138},
    {0x028Eu, 143}, {0x0292u, 147}, {0x0294u, 148}, {0x029Du, 103},
    {0x02A3u, 18}, {0x02A4u, 82}, {0x02A5u, 19}, {0x02A6u, 20},
    {0x02A7u, 133}, {0x02A8u, 21}, {0x02B0u, 162}, {0x02B2u, 164},
    {0x02C8u, 156}, {0x02CCu, 157}, {0x02D0u, 158}, {0x0303u, 17},
    {0x03B2u, 75}, {0x03B8u, 119}, {0x03C7u, 142}, {0x1D4Au, 42},
    {0x1D5Du, 22}, {0x1D7Bu, 177}, {0x2014u, 9}, {0x201Cu, 14},
    {0x201Du, 15}, {0x2026u, 10}, {0x2192u, 171}, {0x2193u, 169},
    {0x2197u, 172}, {0x2198u, 173}, {0xAB67u, 23},
};

constexpr size_t kVocabSize = sizeof(kVocab) / sizeof(kVocab[0]);
constexpr int32_t kMaxId = 177;

// Decode the next UTF-8 codepoint from s starting at i. Advances i past the
// consumed bytes. Returns (char32_t)-1 on a malformed byte (and advances by 1
// to guarantee forward progress).
char32_t next_codepoint(const std::string & s, size_t & i) noexcept {
    const unsigned char c0 = static_cast<unsigned char>(s[i]);
    if (c0 < 0x80u) {
        i += 1;
        return c0;
    }
    auto cont = [&](size_t k) -> bool {
        return i + k < s.size() &&
               (static_cast<unsigned char>(s[i + k]) & 0xC0u) == 0x80u;
    };
    if ((c0 & 0xE0u) == 0xC0u && cont(1)) {
        const char32_t cp = ((c0 & 0x1Fu) << 6) |
                            (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
        i += 2;
        return cp;
    }
    if ((c0 & 0xF0u) == 0xE0u && cont(1) && cont(2)) {
        const char32_t cp = ((c0 & 0x0Fu) << 12) |
                            ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6) |
                            (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
        i += 3;
        return cp;
    }
    if ((c0 & 0xF8u) == 0xF0u && cont(1) && cont(2) && cont(3)) {
        const char32_t cp = ((c0 & 0x07u) << 18) |
                            ((static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 12) |
                            ((static_cast<unsigned char>(s[i + 2]) & 0x3Fu) << 6) |
                            (static_cast<unsigned char>(s[i + 3]) & 0x3Fu);
        i += 4;
        return cp;
    }
    i += 1;
    return static_cast<char32_t>(-1);
}

#if defined(KOKORO_USE_ESPEAK)

// espeak-ng is process-global and not thread-safe; serialize init + calls.
std::mutex & espeak_mutex() {
    static std::mutex m;
    return m;
}

bool ensure_espeak_init() {
    static bool ok = [] {
        // AUDIO_OUTPUT_SYNCHRONOUS so no audio device is opened.
        const int rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);
        if (rate < 0) {
            return false;
        }
        return espeak_SetVoiceByName("en-us") == EE_OK;
    }();
    return ok;
}

// Run espeak over text, accumulating IPA across all clauses. espeak returns
// one clause per call (stopping at sentence/comma punctuation) and advances
// the text pointer; we join clauses with a single space to match the
// `espeak-ng -q --ipa` binary output the reference was derived from.
std::string espeak_text_to_ipa(const std::string & text) {
    std::lock_guard<std::mutex> lock(espeak_mutex());
    if (!ensure_espeak_init()) {
        return std::string();
    }
    const void * inptr = static_cast<const void *>(text.c_str());
    const int textmode = espeakCHARS_UTF8;
    const int phmode = espeakPHONEMES_IPA; // bits 0-2 = 2 → IPA names
    std::string out;
    while (inptr != nullptr) {
        const char * clause =
            espeak_TextToPhonemes(&inptr, textmode, phmode);
        if (clause == nullptr) {
            break;
        }
        // Trim leading/trailing whitespace espeak may attach to a clause.
        std::string c(clause);
        const size_t b = c.find_first_not_of(" \t\n\r");
        if (b == std::string::npos) {
            continue; // whitespace-only clause
        }
        const size_t e = c.find_last_not_of(" \t\n\r");
        c = c.substr(b, e - b + 1);
        if (!out.empty()) {
            out.push_back(' ');
        }
        out += c;
    }
    return out;
}

#endif // KOKORO_USE_ESPEAK

} // namespace

int32_t kokoro_codepoint_to_id(char32_t cp) noexcept {
    // Binary search over the codepoint-sorted table.
    size_t lo = 0;
    size_t hi = kVocabSize;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (kVocab[mid].cp < cp) {
            lo = mid + 1;
        } else if (kVocab[mid].cp > cp) {
            hi = mid;
        } else {
            return kVocab[mid].id;
        }
    }
    return -1;
}

std::vector<int32_t> ipa_to_token_ids(const std::string & ipa) {
    std::vector<int32_t> ids;
    ids.reserve(ipa.size());
    size_t i = 0;
    while (i < ipa.size()) {
        const char32_t cp = next_codepoint(ipa, i);
        if (cp == static_cast<char32_t>(-1)) {
            continue; // malformed byte, already advanced
        }
        const int32_t id = kokoro_codepoint_to_id(cp);
        if (id >= 0) {
            ids.push_back(id);
        }
        // Unmapped codepoints are dropped (reference behavior).
    }
    return ids;
}

std::vector<int32_t> phonemize_ipa(const std::string & text) {
#if defined(KOKORO_USE_ESPEAK)
    return ipa_to_token_ids(espeak_text_to_ipa(text));
#else
    (void) text;
    return std::vector<int32_t>(); // caller must supply IPA via ipa_to_token_ids
#endif
}

std::vector<int32_t> wrap_input_ids(const std::vector<int32_t> & ids) {
    std::vector<int32_t> out;
    out.reserve(ids.size() + 2);
    out.push_back(KOKORO_PAD_ID);
    // Kokoro's BERT encoder caps the phoneme run at 510 (512 with both pads).
    const size_t cap = std::min<size_t>(ids.size(), 510);
    out.insert(out.end(), ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(cap));
    out.push_back(KOKORO_PAD_ID);
    return out;
}

std::vector<int32_t> phonemize_to_input_ids(const std::string & text) {
    return wrap_input_ids(phonemize_ipa(text));
}

bool espeak_available() noexcept {
#if defined(KOKORO_USE_ESPEAK)
    std::lock_guard<std::mutex> lock(espeak_mutex());
    return ensure_espeak_init();
#else
    return false;
#endif
}

int phoneme_vocab_size() noexcept {
    return kMaxId + 1; // 178
}

// --- Legacy ASCII fallback ---------------------------------------------------
//
// Retained for callers not yet migrated to the espeak path. Maps ASCII letters
// to their direct vocab ids (the lowercase Latin block is in-vocab) and wraps
// with the pad token. This is degraded G2P (graphemes, not phonemes) but emits
// a valid id sequence in the same space as the real path.
std::vector<int32_t> phonemize_ascii(const std::string & text) {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (char ch : text) {
        const char lc =
            (ch >= 'A' && ch <= 'Z') ? static_cast<char>(ch - 'A' + 'a') : ch;
        const int32_t id = kokoro_codepoint_to_id(
            static_cast<char32_t>(static_cast<unsigned char>(lc)));
        if (id >= 0) {
            ids.push_back(id);
        }
        if (ids.size() >= 510) {
            break;
        }
    }
    return wrap_input_ids(ids);
}

} // namespace eliza_kokoro
