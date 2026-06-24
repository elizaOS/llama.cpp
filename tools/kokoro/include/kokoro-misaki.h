// SPDX-License-Identifier: MIT
//
// kokoro-misaki.h — normalize raw espeak/CMU-style IPA into the single-codepoint
// "misaki" symbols that Kokoro's 178-token vocab was trained on.
//
// Why this exists: the built-in G2P (core/g2p_en.h, via phonemize_builtin_en)
// and espeak-ng both emit *standard* IPA — multi-codepoint sequences like
// dʒ, tʃ, oʊ, eɪ, aʊ, aɪ, ɔɪ, and length-marked vowels ɑː, ɔː, uː, iː.
// Kokoro-82M / StyleTTS2, however, was trained on hexgrad/misaki's English
// phoneme set, which spells those as single codepoints: ʤ, ʧ, O, A, W, I, Y,
// and drops the ː length mark on American vowels. kokoro_phonemes_to_ids()
// greedy-matches codepoint-for-codepoint against the embedded vocab and
// silently drops anything it can't find — so feeding it raw "dʒ" tokenises as
// d + ʒ (valid ids, WRONG embeddings), and the model says the wrong sounds with
// no error. This pass closes that gap.
//
// Mapping derived from hexgrad/misaki EN_PHONES and cross-checked codepoint-by-
// codepoint against the reference Python `KPipeline(lang_code='a')` output for a
// battery of phrases (diphthongs, affricates, r-coloured vowels, long vowels).

#pragma once

#include <cstddef>
#include <string>

namespace crispasr {

// Convert raw IPA (espeak / CMU style) into misaki symbols.
//   gb == false → American English (oʊ/əʊ → O, no ː length marks). Default.
//   gb == true  → British English   (oʊ/əʊ → Q).
//
// Rules are applied longest-source-first in a single greedy left-to-right pass,
// so multi-codepoint sequences (dʒ, eɪ, ɜː, …) are always consumed before their
// single-codepoint prefixes (d, e, ɜ, …).
inline std::string ipa_to_misaki(const std::string& ipa, bool gb = false) {
    struct Rule { const char* from; const char* to; };

    // American English. Order matters: 2-codepoint sources precede any rule
    // whose source is a prefix of another.
    static const Rule RULES_US[] = {
        // affricates → misaki ligatures. Only the two English affricates dʒ/tʃ
        // are mapped: misaki spells "cheese"/"judge" with ʧ/ʤ. The ʣ/ʦ/ʥ/ʨ
        // ligatures exist in the vocab but misaki does NOT use them for English
        // ("cats" is kˈæts, "outside" is ˌWtsˈId — t+s, not ʦ), so mapping them
        // here would wrongly fuse cross-morpheme t+s / d+z.
        {"dʒ", "ʤ"}, // U+0064 U+0292 → U+02A4
        {"tʃ", "ʧ"}, // U+0074 U+0283 → U+02A7
        // diphthongs → misaki capitals
        {"aʊ", "W"}, // U+0061 U+028A → W
        {"aɪ", "I"}, // U+0061 U+026A → I
        {"eɪ", "A"}, // U+0065 U+026A → A
        {"oʊ", "O"}, // U+006F U+028A → O
        {"əʊ", "O"}, // U+0259 U+028A → O  (espeak en-us sometimes emits əʊ)
        {"ɔɪ", "Y"}, // U+0254 U+026A → Y
        // r-coloured vowels → misaki spelling (matches KPipeline)
        {"ɜː", "ɜɹ"}, // U+025C U+02D0 → U+025C U+0279
        {"ɝ", "ɜɹ"},  // U+025D        → U+025C U+0279
        {"ɚ", "əɹ"},  // U+025A        → U+0259 U+0279
        // long vowels → drop the length mark (misaki American has no ː)
        {"ɑː", "ɑ"}, // U+0251 U+02D0 → U+0251
        {"ɔː", "ɔ"}, // U+0254 U+02D0 → U+0254
        {"uː", "u"}, // U+0075 U+02D0 → U+0075
        {"iː", "i"}, // U+0069 U+02D0 → U+0069
        {"ɛː", "ɛ"}, // U+025B U+02D0 → U+025B
        {"oː", "o"}, // U+006F U+02D0 → U+006F
    };

    // British English: əʊ/oʊ → Q ("oh"); keep r-colouring rule. The remaining
    // diphthongs/affricates/long-vowel rules are shared with US.
    static const Rule RULES_GB[] = {
        {"dʒ", "ʤ"}, {"tʃ", "ʧ"},
        {"aʊ", "W"}, {"aɪ", "I"}, {"eɪ", "A"}, {"ɔɪ", "Y"},
        {"əʊ", "Q"}, {"oʊ", "Q"},
        {"ɜː", "ɜɹ"}, {"ɝ", "ɜɹ"}, {"ɚ", "əɹ"},
        {"ɑː", "ɑ"}, {"ɔː", "ɔ"}, {"uː", "u"}, {"iː", "i"},
    };

    const Rule* rules = gb ? RULES_GB : RULES_US;
    const size_t n_rules = gb ? (sizeof(RULES_GB) / sizeof(Rule))
                              : (sizeof(RULES_US) / sizeof(Rule));

    std::string out;
    out.reserve(ipa.size());
    size_t i = 0;
    while (i < ipa.size()) {
        bool matched = false;
        for (size_t r = 0; r < n_rules; r++) {
            const size_t flen = std::char_traits<char>::length(rules[r].from);
            if (ipa.compare(i, flen, rules[r].from) == 0) {
                out += rules[r].to;
                i += flen;
                matched = true;
                break;
            }
        }
        if (matched) continue;
        // Drop any stray ː (U+02D0) length mark not consumed by a vowel rule —
        // misaki American has no length marks, and the bare codepoint after a
        // consonant is meaningless to the vocab.
        if (i + 1 < ipa.size() &&
            (unsigned char)ipa[i] == 0xCB && (unsigned char)ipa[i + 1] == 0x90) {
            i += 2;
            continue;
        }
        out += ipa[i++];
    }
    return out;
}

} // namespace crispasr
