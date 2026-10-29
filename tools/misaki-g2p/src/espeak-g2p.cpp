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

#include <utility>

namespace misaki {
namespace {

// phonemizer language code -> the eSpeak NG voice identifier
// EspeakWrapper.set_voice selects (the first voice listed for that language).
struct LanguageVoice {
    const char * language;
    const char * voice;
};
constexpr LanguageVoice kVoices[] = {
    {"es", "roa/es"},  // Spanish (Spain): distinción, ʎ, θ
};

}  // namespace

std::unique_ptr<EspeakG2P> EspeakG2P::open(const std::string & language, const std::string & data_dir,
                                           std::string & err) {
    const char * voice = nullptr;
    for (const auto & lv : kVoices) {
        if (language == lv.language) voice = lv.voice;
    }
    if (!voice) {
        err = "no eSpeak G2P for language '" + language + "' (supported: es)";
        return nullptr;
    }
    std::unique_ptr<EspeakFallback> backend = EspeakFallback::open(data_dir, err, voice, /*remove_flags*/ true);
    if (!backend) return nullptr;
    return std::unique_ptr<EspeakG2P>(new EspeakG2P(language, std::move(backend)));
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
