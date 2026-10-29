// SPDX-License-Identifier: Apache-2.0
// misaki espeak.EspeakG2P (misaki 0.9.4, version=None): the whole-text eSpeak
// G2P Kokoro's KPipeline uses for languages without a misaki lexicon
// (lang_code 'e' -> EspeakG2P(language='es')). eSpeak NG (GPL-3.0-or-later) is
// called the way phonemizer-fork 3.3.2 calls it for
//   EspeakBackend(language, preserve_punctuation=True, with_stress=True,
//                 tie='^', language_switch='remove-flags')
// then EspeakG2P's own E2M table (shorter than EspeakFallback's: no American
// vowel rewrites). Delibes-8M-esES was trained on exactly this output with
// eSpeak NG 1.52.0.
#include "misaki-internal.h"

#include <exception>
#include <utility>

namespace misaki {
namespace {

// phonemizer language code -> the eSpeak NG voice identifier
// EspeakWrapper.set_voice selects (the first voice listed for that language).
struct LanguageVoice {
    const char * language;
    const char * voice;
    const char * dict;   // the dictionary file the voice loads
    const char * probe;  // a word that must phonemize to phoneme letters
};
constexpr LanguageVoice kVoices[] = {
    {"es", "roa/es", "es_dict", "hola"},  // Spanish (Spain): distinción, ʎ, θ
};

// True when `ps` has a phoneme letter: anything but whitespace, the
// punctuation phonemizer preserves, stress and length marks, ties and '-'.
bool has_phoneme_letter(const u32 & ps) {
    static const u32 kNotLetters = U";:,.!?¡¿—…\"«»“”(){}[]ˈˌː^-";
    for (char32_t c : ps) {
        if (!py_isspace(c) && !contains(kNotLetters, c)) return true;
    }
    return false;
}

}  // namespace

std::unique_ptr<EspeakG2P> EspeakG2P::open(const std::string & language, const std::string & data_dir,
                                           std::string & err) {
    const LanguageVoice * lv = nullptr;
    for (const auto & candidate : kVoices) {
        if (language == candidate.language) lv = &candidate;
    }
    if (!lv) {
        err = "no eSpeak G2P for language '" + language + "' (supported: es)";
        return nullptr;
    }
    std::unique_ptr<EspeakFallback> backend =
        EspeakFallback::open(data_dir, err, lv->voice, /*remove_flags*/ true, lv->dict);
    if (!backend) return nullptr;
    std::unique_ptr<EspeakG2P> g2p(new EspeakG2P(language, std::move(backend)));
    // Probe: a voice that loaded without usable rules phonemizes every word
    // to nothing (only punctuation survives), which would synthesize garbage.
    u32 probed;
    try {
        probed = (*g2p)(from_utf8(lv->probe));
    } catch (const std::exception & e) {
        err = std::string("eSpeak NG probe phonemization of '") + lv->probe + "' failed: " + e.what();
        return nullptr;
    }
    if (!has_phoneme_letter(probed)) {
        err = std::string("eSpeak NG voice ") + lv->voice + " phonemized '" + lv->probe + "' to '" + to_utf8(probed) +
              "' (no phonemes): its " + lv->dict + " data is unusable";
        return nullptr;
    }
    return g2p;
}

u32 EspeakG2P::e2m(const u32 & in) {
    // self.e2m = sorted({...}.items()) for version None: applied in key order.
    static const std::pair<const char32_t *, const char32_t *> kE2M[] = {
        {U"a^ɪ", U"I"}, {U"a^ʊ", U"W"}, {U"d^z", U"ʣ"}, {U"d^ʒ", U"ʤ"}, {U"e^ɪ", U"A"}, {U"o^ʊ", U"O"},
        {U"s^s", U"S"}, {U"t^s", U"ʦ"}, {U"t^ʃ", U"ʧ"}, {U"ɔ^ɪ", U"Y"}, {U"ə^ʊ", U"Q"},
    };
    u32 ps = in;
    for (auto & kv : kE2M) ps = replace_all(ps, kv.first, kv.second);
    ps = replace_all(ps, U"^", U"");  // remaining tie characters
    ps = replace_all(ps, U"-", U"");  // version != '2.0'
    return ps;
}

u32 EspeakG2P::operator()(const u32 & text_in) {
    // Angles to curly quotes, then parentheses to angles (restored at the end).
    u32 text = replace_all(text_in, U"«", U"“");
    text = replace_all(text, U"»", U"”");
    text = replace_all(text, U"(", U"«");
    text = replace_all(text, U")", U"»");
    std::optional<u32> ps = backend_->phonemize(text);
    if (!ps) return u32();
    u32 out = e2m(py_strip(*ps));
    out = replace_all(out, U"«", U"(");
    out = replace_all(out, U"»", U")");
    return out;
}

}  // namespace misaki
