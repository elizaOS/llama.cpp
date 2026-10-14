// SPDX-License-Identifier: Apache-2.0
// Internal interfaces of misaki_g2p, a C++ port of misaki 0.9.4's American
// English G2P (hexgrad/misaki, Apache-2.0; see LICENSES/misaki-NOTICE.txt).
// Everything works on UTF-32 strings so that Python's per-code-point string
// semantics (len, slicing, isalpha, lower, ...) carry over directly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace misaki {

using u32 = std::u32string;

// ---------------------------------------------------------------- unicode.cpp
std::string to_utf8(const u32 & s);
u32 from_utf8(const char * s, size_t n);
inline u32 from_utf8(const std::string & s) { return from_utf8(s.data(), s.size()); }
u32 U(const char * utf8);  // literal helper

// Python str semantics (CPython 3.12, Unicode 15.0)
bool py_isalpha(char32_t c);
bool py_isspace(char32_t c);
int py_digit_value(char32_t c);  // str.isdigit() -> int(unicodedata.numeric(c)), else -1
bool py_str_isalpha(const u32 & s);
bool py_str_isdigit(const u32 & s);
u32 py_lower(const u32 & s);
u32 py_upper(const u32 & s);
u32 py_capitalize(const u32 & s);
inline bool py_is_lower(const u32 & s) { return py_lower(s) == s; }  // s == s.lower()
inline bool py_is_upper(const u32 & s) { return py_upper(s) == s; }  // s == s.upper()
u32 py_strip(const u32 & s);
u32 nfkc(const u32 & s);

// regex-module general categories (misaki's subtokenizer)
bool cat_L(char32_t c);
bool cat_Lu(char32_t c);
bool cat_Ll(char32_t c);
bool cat_Nd(char32_t c);

// spaCy char_classes
bool spacy_alpha(char32_t c);
bool spacy_alpha_lower(char32_t c);
bool spacy_alpha_upper(char32_t c);
bool spacy_icon(char32_t c);
bool spacy_suffix_dot_pre(char32_t c);  // lookbehind of the suffix rule for a final '.'

inline bool contains(const u32 & s, char32_t c) { return s.find(c) != u32::npos; }
inline bool contains(const u32 & s, const u32 & t) { return s.find(t) != u32::npos; }
inline bool starts_with(const u32 & s, const u32 & p) { return s.size() >= p.size() && s.compare(0, p.size(), p) == 0; }
inline bool ends_with(const u32 & s, const u32 & p) { return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0; }
u32 replace_all(u32 s, const u32 & from, const u32 & to);

// ---------------------------------------------------------------- lexicon.cpp
enum LexTag : uint8_t {
    TAG_PLAIN = 0, TAG_DEFAULT, TAG_NOUN, TAG_VERB, TAG_ADJ, TAG_ADV, TAG_DT, TAG_NONE, TAG_VBD, TAG_VBN, TAG_VBP,
    TAG_COUNT
};

// One JSON value of us_gold.json / us_silver.json: a plain string, or a dict
// of tag -> string-or-null.
struct LexValue {
    bool found = false;
    bool is_dict = false;
    std::optional<u32> plain;
    std::optional<u32> by_tag[TAG_COUNT];
    bool has_tag[TAG_COUNT] = {};
};

class LexFile {
  public:
    static std::unique_ptr<LexFile> open(const std::string & path, std::string & err);
    ~LexFile();
    LexValue gold(const u32 & key) const;
    LexValue silver(const u32 & key) const;
    bool in_gold(const u32 & key) const;
    bool in_silver(const u32 & key) const;
    uint32_t gold_count() const { return n_gold_; }
    uint32_t silver_count() const { return n_silver_; }

  private:
    LexFile() = default;
    LexValue get(const uint8_t * entries, uint32_t n, const u32 & key) const;
    long find(const uint8_t * entries, uint32_t n, const std::string & key) const;
    const uint8_t * base_ = nullptr;
    size_t size_ = 0;
    bool mapped_ = false;
    std::vector<uint8_t> owned_;
    const uint8_t * gold_ = nullptr;
    const uint8_t * silver_ = nullptr;
    const uint8_t * variants_ = nullptr;
    const uint8_t * strings_ = nullptr;
    uint32_t n_gold_ = 0, n_silver_ = 0, n_var_ = 0, n_str_ = 0;
};

// ---------------------------------------------------------------- tokens
struct MToken {
    u32 text;
    std::string tag;
    u32 whitespace;
    std::optional<u32> phonemes;
    // MToken._ (addict dict; absent keys read as None)
    bool is_head = true;
    std::optional<u32> alias;
    std::optional<double> stress;
    std::optional<u32> currency;
    std::string num_flags;
    bool prespace = false;
    std::optional<int> rating;
};

// ---------------------------------------------------------------- tokenizer.cpp
// spaCy 3.8 English tokenizer: whitespace split, exceptions, prefix/suffix/
// infix rules and URL matching. Returns (text, whitespace) pairs.
struct SpacyToken {
    u32 text;
    bool space_after = false;
    size_t start = 0;  // code point offset into the input
};
std::vector<SpacyToken> spacy_tokenize(const u32 & text);

// misaki en.subtokenize (the SUBTOKEN_REGEX findall)
std::vector<u32> subtokenize(const u32 & word);

// ---------------------------------------------------------------- tagger.cpp
// Penn Treebank tags, approximating en_core_web_sm 3.8.0 closely enough for
// the places misaki reads a tag.
std::vector<std::string> tag_tokens(const std::vector<SpacyToken> & toks, const LexFile * lex);

// ---------------------------------------------------------------- numbers.cpp
// English number words, written for this port (cardinal with "and", ordinal,
// year, decimal) to match the strings misaki got from num2words.
u32 number_cardinal(const std::string & digits);     // digits only, no sign
u32 number_ordinal(const std::string & digits);
u32 number_year(const std::string & digits);
u32 number_float(const std::string & text);          // "12.50", ".5", "3."
bool number_fits(const std::string & digits);        // within the supported range

// ---------------------------------------------------------------- espeak.cpp
class EspeakFallback {
  public:
    // Returns null and sets err when libespeak-ng is not linked or fails to load.
    static std::unique_ptr<EspeakFallback> open(const std::string & data_dir, std::string & err);
    ~EspeakFallback();
    // misaki espeak.EspeakFallback.__call__ on a token text: phonemizer
    // (preserve_punctuation, with_stress, tie='^') then the E2M table.
    std::optional<u32> operator()(const u32 & text);
    // The E2M mapping alone, on phonemizer-style output (tests).
    static u32 e2m(const u32 & espeak_tied_ipa);
  private:
    EspeakFallback() = default;
    u32 phonemize_line(const u32 & text);
};
bool espeak_linked();

// ---------------------------------------------------------------- misaki-g2p.cpp
const u32 & paradee_vocab();
bool in_paradee_vocab(char32_t c);

// ---------------------------------------------------------------- g2p.cpp
struct G2POptions {
    std::u32string unk;  // paradee passes unk=''
};

class G2P {
  public:
    G2P(const LexFile & lex, EspeakFallback * fallback) : lex_(lex), fallback_(fallback) {}
    // misaki en.G2P.__call__: returns the phoneme string and the final tokens.
    u32 operator()(const u32 & text, std::vector<MToken> * tokens_out = nullptr);
    bool has_espeak() const { return fallback_ != nullptr; }
    // number of tokens that went to the OOV fallback in the last call
    int last_fallback_count = 0;
  private:
    const LexFile & lex_;
    EspeakFallback * fallback_;
    std::optional<u32> fallback(const MToken & tk);
};

// misaki en.apply_stress (exposed for the unit tests)
std::optional<u32> apply_stress_for_test(const std::optional<u32> & ps, std::optional<double> stress);

// paradee/tts.py Paradee._chunks: sentence split, phonemize, cut to <= 510.
std::vector<u32> paradee_chunks(G2P & g2p, const u32 & text, size_t max_phonemes = 510);

}  // namespace misaki
