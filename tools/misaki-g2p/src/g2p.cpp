// SPDX-License-Identifier: Apache-2.0
// Port of misaki 0.9.4 en.py (hexgrad/misaki, Apache-2.0): Lexicon lookup,
// stemming, stress, numbers, token merging and G2P.__call__, configured as
// paradee/tts.py uses it: G2P(trf=False, british=False,
// fallback=EspeakFallback(british=False), unk='').
#include "misaki-internal.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace misaki {
namespace {

using Ps = std::optional<u32>;
using Stress = std::optional<double>;

struct Result {
    Ps ps;
    std::optional<int> rating;
};

struct TokenContext {
    std::optional<bool> future_vowel;
    bool future_to = false;
};

const u32 kDiphthongs = U"AIOQWYʤʧ";
const u32 kSubtokenJunks = U"',-._‘’/";
const u32 kPuncts = U";:,.!?—…\"“”";
const u32 kNonQuotePuncts = U";:,.!?—…";
const u32 kConsonants = U"bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const u32 kUsTaus = U"AIOWYiuæɑəɛɪɹʊʌ";
const u32 kVowels = U"AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const char32_t kPrimary = U'ˈ';
const char32_t kSecondary = U'ˌ';

bool is_vowel(char32_t c) { return contains(kVowels, c); }
bool truthy(const Ps & p) { return p && !p->empty(); }
bool is_stress(char32_t c) { return c == kPrimary || c == kSecondary; }

// LEXICON_ORDS: ' - A-Z a-z
bool lexicon_ord(char32_t c) { return c == '\'' || c == '-' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); }
bool all_lexicon_ords(const u32 & w) { return std::all_of(w.begin(), w.end(), lexicon_ord); }

bool is_ascii_digits(const u32 & s) {  // re.match(r'^[0-9]+$')
    if (s.empty()) return false;
    for (char32_t c : s) if (c < '0' || c > '9') return false;
    return true;
}

bool is_punct_tag(const std::string & t) {
    static const std::set<std::string> tags = {".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#", "NFP"};
    return tags.count(t) > 0;
}

int stress_weight(const Ps & ps) {
    if (!ps) return 0;
    int w = 0;
    for (char32_t c : *ps) w += contains(kDiphthongs, c) ? 2 : 1;
    return w;
}

u32 restress(const u32 & ps) {
    // Each stress mark moves to just before the next vowel at or after it.
    std::vector<std::pair<double, char32_t>> ips;
    for (size_t i = 0; i < ps.size(); i++) ips.emplace_back((double) i, ps[i]);
    for (size_t i = 0; i < ps.size(); i++) {
        if (!is_stress(ps[i])) continue;
        size_t j = i;
        while (j < ps.size() && !is_vowel(ps[j])) j++;
        if (j == ps.size()) continue;  // misaki would raise; never reached from apply_stress
        ips[i].first = (double) j - 0.5;
    }
    std::stable_sort(ips.begin(), ips.end());
    u32 out;
    for (auto & p : ips) out += p.second;
    return out;
}

Ps apply_stress(const Ps & ps_in, Stress stress) {
    if (!ps_in) return ps_in;
    const u32 & ps = *ps_in;
    if (!stress) return ps;
    double s = *stress;
    bool has_primary = contains(ps, kPrimary), has_secondary = contains(ps, kSecondary);
    bool has_vowel = std::any_of(ps.begin(), ps.end(), is_vowel);
    if (s < -1) {
        u32 o;
        for (char32_t c : ps) if (!is_stress(c)) o += c;
        return o;
    } else if (s == -1 || ((s == 0 || s == -0.5) && has_primary)) {
        u32 o;
        for (char32_t c : ps) {
            if (c == kSecondary) continue;
            o += c == kPrimary ? kSecondary : c;
        }
        return o;
    } else if ((s == 0 || s == 0.5 || s == 1) && !has_primary && !has_secondary) {
        if (!has_vowel) return ps;
        return restress(u32(1, kSecondary) + ps);
    } else if (s >= 1 && !has_primary && has_secondary) {
        u32 o = ps;
        std::replace(o.begin(), o.end(), kSecondary, kPrimary);
        return o;
    } else if (s > 1 && !has_primary && !has_secondary) {
        if (!has_vowel) return ps;
        return restress(u32(1, kPrimary) + ps);
    }
    return ps;
}

std::string parent_tag(const std::string & tag) {
    if (tag.rfind("VB", 0) == 0) return "VERB";
    if (tag.rfind("NN", 0) == 0) return "NOUN";
    if (tag.rfind("ADV", 0) == 0 || tag.rfind("RB", 0) == 0) return "ADV";
    if (tag.rfind("ADJ", 0) == 0 || tag.rfind("JJ", 0) == 0) return "ADJ";
    return tag;
}

int tag_id(const std::string & t) {
    static const char * names[] = {"PLAIN", "DEFAULT", "NOUN", "VERB", "ADJ", "ADV", "DT", "None", "VBD", "VBN", "VBP"};
    for (int i = 0; i < TAG_COUNT; i++) if (t == names[i]) return i;
    return -1;
}

class Lexicon {
  public:
    explicit Lexicon(const LexFile & f) : f_(f) {}

    bool in_golds(const u32 & w) const { return f_.in_gold(w); }
    bool in_silvers(const u32 & w) const { return f_.in_silver(w); }

    // self.golds[word] for a plain-string entry
    Ps gold_plain(const u32 & w) const {
        LexValue v = f_.gold(w);
        return v.plain;
    }

    Result get_NNP(const u32 & word) const {
        u32 joined;
        for (char32_t c : word) {
            if (!py_isalpha(c)) continue;
            LexValue v = f_.gold(py_upper(u32(1, c)));
            if (!v.found || !v.plain) return {};
            joined += *v.plain;
        }
        Ps ps = apply_stress(joined, 0.0);
        size_t cut = ps->rfind(kSecondary);
        if (cut != u32::npos) (*ps)[cut] = kPrimary;
        return {ps, 3};
    }

    Result get_special_case(const u32 & word, const std::string & tag, Stress stress, const TokenContext & ctx) const {
        if (tag == "ADD" && (word == U"." || word == U"/")) return lookup(word == U"." ? U"dot" : U"slash", "", -0.5, &ctx);
        if (word == U"%") return lookup(U"percent", "", {}, &ctx);
        if (word == U"&") return lookup(U"and", "", {}, &ctx);
        if (word == U"+") return lookup(U"plus", "", {}, &ctx);
        if (word == U"@") return lookup(U"at", "", {}, &ctx);
        {
            // '.' in word.strip('.') and word.replace('.', '').isalpha() and len(max(word.split('.'), key=len)) < 3
            size_t a = 0, b = word.size();
            while (a < b && word[a] == '.') a++;
            while (b > a && word[b - 1] == '.') b--;
            u32 stripped = word.substr(a, b - a);
            u32 nodots;
            for (char32_t c : word) if (c != '.') nodots += c;
            if (contains(stripped, U'.') && py_str_isalpha(nodots)) {
                size_t longest = 0, cur = 0;
                for (char32_t c : word) {
                    if (c == '.') { longest = std::max(longest, cur); cur = 0; }
                    else cur++;
                }
                longest = std::max(longest, cur);
                if (longest < 3) return get_NNP(word);
            }
        }
        if (word == U"a" || word == U"A") return {U(tag == "DT" ? "ɐ" : "ˈA"), 4};
        if (word == U"am" || word == U"Am" || word == U"AM") {
            if (tag.rfind("NN", 0) == 0) return get_NNP(word);
            if (!ctx.future_vowel || word != U"am" || (stress && *stress > 0)) return {gold_plain(U"am"), 4};
            return {U"ɐm", 4};
        }
        if (word == U"an" || word == U"An" || word == U"AN") {
            if (word == U"AN" && tag.rfind("NN", 0) == 0) return get_NNP(word);
            return {U"ɐn", 4};
        }
        if (word == U"I" && tag == "PRP") return {u32(1, kSecondary) + U"I", 4};
        if ((word == U"by" || word == U"By" || word == U"BY") && parent_tag(tag) == "ADV") return {U"bˈI", 4};
        if (word == U"to" || word == U"To" || (word == U"TO" && (tag == "TO" || tag == "IN"))) {
            if (!ctx.future_vowel) return {gold_plain(U"to"), 4};
            return {U(*ctx.future_vowel ? "tʊ" : "tə"), 4};
        }
        if (word == U"in" || word == U"In" || (word == U"IN" && tag != "NNP")) {
            u32 s = (!ctx.future_vowel || tag != "IN") ? u32(1, kPrimary) : u32();
            return {s + U"ɪn", 4};
        }
        if (word == U"the" || word == U"The" || (word == U"THE" && tag == "DT")) {
            return {U(ctx.future_vowel && *ctx.future_vowel ? "ði" : "ðə"), 4};
        }
        if (tag == "IN") {
            // re.match(r'(?i)vs\.?$', word)
            u32 l = py_lower(word);
            if (l == U"vs" || l == U"vs.") return lookup(U"versus", "", {}, &ctx);
        }
        if (word == U"used" || word == U"Used" || word == U"USED") {
            LexValue v = f_.gold(U"used");
            if ((tag == "VBD" || tag == "JJ") && ctx.future_to) return {v.by_tag[TAG_VBD], 4};
            return {v.by_tag[TAG_DEFAULT], 4};
        }
        return {};
    }

    bool is_known(const u32 & word) const {
        if (in_golds(word) || word == U"%" || word == U"&" || word == U"+" || word == U"@" || in_silvers(word)) return true;
        if (!py_str_isalpha(word) || !all_lexicon_ords(word)) return false;
        if (word.size() == 1) return true;
        if (word == py_upper(word) && in_golds(py_lower(word))) return true;
        u32 rest = word.substr(1);
        return rest == py_upper(rest);
    }

    Result lookup(u32 word, std::string tag, Stress stress, const TokenContext * ctx) const {
        std::optional<bool> is_NNP;
        if (word == py_upper(word) && !in_golds(word)) {
            word = py_lower(word);
            is_NNP = tag == "NNP";
        }
        LexValue v = f_.gold(word);
        int rating = 4;
        if (!v.found && !(is_NNP && *is_NNP)) {
            v = f_.silver(word);
            rating = 3;
        }
        Ps ps;
        if (v.found && v.is_dict) {
            int id;
            if (ctx && !ctx->future_vowel && v.has_tag[TAG_NONE]) {
                id = TAG_NONE;
            } else {
                id = tag_id(tag);
                if (id < 0 || !v.has_tag[id]) id = tag_id(parent_tag(tag));
            }
            ps = (id >= 0 && v.has_tag[id]) ? v.by_tag[id] : v.by_tag[TAG_DEFAULT];
        } else if (v.found) {
            ps = v.plain;
        }
        if (!ps || (is_NNP && *is_NNP && !contains(*ps, kPrimary))) {
            // ps, rating = self.get_NNP(word): on failure both become None
            return get_NNP(word);
        }
        return {apply_stress(ps, stress), ps ? std::optional<int>(rating) : std::nullopt};
    }

    u32 _s(const u32 & stem) const {
        char32_t c = stem.back();
        if (contains(U"ptkfθ", c)) return stem + U"s";
        if (contains(U"szʃʒʧʤ", c)) return stem + U"ᵻz";
        return stem + U"z";
    }
    Ps _s(const Ps & stem) const {
        if (!stem || stem->empty()) return {};
        return _s(*stem);
    }

    Result stem_s(const u32 & word, const std::string & tag, Stress stress, const TokenContext * ctx) const {
        if (word.size() < 3 || word.back() != 's') return {};
        u32 stem;
        if (!ends_with(word, U"ss") && is_known(word.substr(0, word.size() - 1))) {
            stem = word.substr(0, word.size() - 1);
        } else if ((ends_with(word, U"'s") || (word.size() > 4 && ends_with(word, U"es") && !ends_with(word, U"ies"))) &&
                   is_known(word.substr(0, word.size() - 2))) {
            stem = word.substr(0, word.size() - 2);
        } else if (word.size() > 4 && ends_with(word, U"ies") && is_known(word.substr(0, word.size() - 3) + U"y")) {
            stem = word.substr(0, word.size() - 3) + U"y";
        } else {
            return {};
        }
        Result r = lookup(stem, tag, stress, ctx);
        return {_s(r.ps), r.rating};
    }

    Ps _ed(const Ps & stem_in) const {
        if (!stem_in || stem_in->empty()) return {};
        const u32 & stem = *stem_in;
        char32_t c = stem.back();
        if (contains(U"pkfθʃsʧ", c)) return stem + U"t";
        if (c == 'd') return stem + U"ᵻd";
        if (c != 't') return stem + U"d";
        if (stem.size() < 2) return stem + U"ɪd";
        if (contains(kUsTaus, stem[stem.size() - 2])) return stem.substr(0, stem.size() - 1) + U"ɾᵻd";
        return stem + U"ᵻd";
    }

    Result stem_ed(const u32 & word, const std::string & tag, Stress stress, const TokenContext * ctx) const {
        if (word.size() < 4 || word.back() != 'd') return {};
        u32 stem;
        if (!ends_with(word, U"dd") && is_known(word.substr(0, word.size() - 1))) {
            stem = word.substr(0, word.size() - 1);
        } else if (word.size() > 4 && ends_with(word, U"ed") && !ends_with(word, U"eed") &&
                   is_known(word.substr(0, word.size() - 2))) {
            stem = word.substr(0, word.size() - 2);
        } else {
            return {};
        }
        Result r = lookup(stem, tag, stress, ctx);
        return {_ed(r.ps), r.rating};
    }

    Ps _ing(const Ps & stem_in) const {
        if (!stem_in || stem_in->empty()) return {};
        const u32 & stem = *stem_in;
        if (stem.size() > 1 && stem.back() == 't' && contains(kUsTaus, stem[stem.size() - 2])) {
            return stem.substr(0, stem.size() - 1) + U"ɾɪŋ";
        }
        return stem + U"ɪŋ";
    }

    static bool double_consonant_ing(const u32 & w) {
        // re.search(r'([bcdgklmnprstvxz])\1ing$|cking$', word)
        if (ends_with(w, U"cking")) return true;
        if (w.size() < 5 || !ends_with(w, U"ing")) return false;
        char32_t a = w[w.size() - 5], b = w[w.size() - 4];
        return a == b && contains(U"bcdgklmnprstvxz", a);
    }

    Result stem_ing(const u32 & word, const std::string & tag, Stress stress, const TokenContext * ctx) const {
        if (word.size() < 5 || !ends_with(word, U"ing")) return {};
        u32 stem;
        u32 base = word.substr(0, word.size() - 3);
        if (word.size() > 5 && is_known(base)) {
            stem = base;
        } else if (is_known(base + U"e")) {
            stem = base + U"e";
        } else if (word.size() > 5 && double_consonant_ing(word) && is_known(word.substr(0, word.size() - 4))) {
            stem = word.substr(0, word.size() - 4);
        } else {
            return {};
        }
        Result r = lookup(stem, tag, stress, ctx);
        return {_ing(r.ps), r.rating};
    }

    Result get_word(u32 word, const std::string & tag, Stress stress, const TokenContext & ctx) const {
        Result sc = get_special_case(word, tag, stress, ctx);
        if (sc.ps) return sc;
        u32 wl = py_lower(word);
        u32 no_apos;
        for (char32_t c : word) if (c != '\'') no_apos += c;
        if (word.size() > 1 && py_str_isalpha(no_apos) && word != wl && (tag != "NNP" || word.size() > 7) &&
            !in_golds(word) && !in_silvers(word) && (word == py_upper(word) || word.substr(1) == py_lower(word.substr(1))) &&
            (in_golds(wl) || in_silvers(wl) || truthy(stem_s(wl, tag, stress, &ctx).ps) ||
             truthy(stem_ed(wl, tag, stress, &ctx).ps) || truthy(stem_ing(wl, tag, stress, &ctx).ps))) {
            word = wl;
        }
        if (is_known(word)) return lookup(word, tag, stress, &ctx);
        if (ends_with(word, U"s'") && is_known(word.substr(0, word.size() - 2) + U"'s")) {
            return lookup(word.substr(0, word.size() - 2) + U"'s", tag, stress, &ctx);
        }
        if (ends_with(word, U"'") && is_known(word.substr(0, word.size() - 1))) {
            return lookup(word.substr(0, word.size() - 1), tag, stress, &ctx);
        }
        Result r = stem_s(word, tag, stress, &ctx);
        if (r.ps) return r;
        r = stem_ed(word, tag, stress, &ctx);
        if (r.ps) return r;
        r = stem_ing(word, tag, stress ? stress : Stress(0.5), &ctx);
        if (r.ps) return r;
        return {};
    }

    static bool is_currency(const u32 & word) {
        size_t dots = std::count(word.begin(), word.end(), U'.');
        if (dots == 0) return true;
        if (dots > 1) return false;
        u32 cents = word.substr(word.find('.') + 1);
        // set(cents) == {0} compares characters with the int 0: never true
        return cents.size() < 3;
    }

    Result get_number(u32 word, const std::optional<u32> & currency, bool is_head, const std::string & num_flags) const {
        // suffix = re.search(r"[a-z']+$", word)
        size_t k = word.size();
        while (k > 0 && ((word[k - 1] >= 'a' && word[k - 1] <= 'z') || word[k - 1] == '\'')) k--;
        std::optional<u32> suffix;
        if (k < word.size()) {
            suffix = word.substr(k);
            word = word.substr(0, k);
        }
        auto is_ordinal_suffix = [&]() {
            return suffix && (*suffix == U"st" || *suffix == U"nd" || *suffix == U"rd" || *suffix == U"th");
        };
        std::vector<Result> result;
        if (!word.empty() && word[0] == '-') {
            result.push_back(lookup(U"minus", "", {}, nullptr));
            word = word.substr(1);
        }
        auto extend_words = [&](const u32 & words, bool first) {
            // re.split(r'[^a-z]+', ...)
            std::vector<u32> splits;
            u32 cur;
            bool in_sep = false;
            for (char32_t c : words) {
                if (c >= 'a' && c <= 'z') {
                    cur += c;
                    in_sep = false;
                } else if (!in_sep) {
                    splits.push_back(cur);
                    cur.clear();
                    in_sep = true;
                }
            }
            splits.push_back(cur);
            for (size_t i = 0; i < splits.size(); i++) {
                const u32 & w = splits[i];
                if (w != U"and" || num_flags.find('&') != std::string::npos) {
                    if (first && i == 0 && splits.size() > 1 && w == U"one" && num_flags.find('a') != std::string::npos) {
                        result.push_back({U"ə", 4});
                    } else {
                        result.push_back(lookup(w, "", w == U"point" ? Stress(-2) : Stress(), nullptr));
                    }
                } else if (w == U"and" && num_flags.find('n') != std::string::npos && !result.empty()) {
                    Result & last = result.back();
                    if (last.ps) last.ps = *last.ps + U"ən";
                }
            }
        };
        auto extend_num = [&](const u32 & num, bool first) {
            std::string d = to_utf8(num);
            extend_words(number_cardinal(d), first);
        };
        const bool digits = is_ascii_digits(word);
        if (digits && is_ordinal_suffix()) {
            extend_words(number_ordinal(to_utf8(word)), true);
        } else if (result.empty() && word.size() == 4 && !(currency && (*currency == U"$" || *currency == U"£" || *currency == U"€")) && digits) {
            extend_words(number_year(to_utf8(word)), true);
        } else if (!is_head && !contains(word, U'.')) {
            u32 num;
            for (char32_t c : word) if (c != ',') num += c;
            if (num.empty()) return {};
            if (num[0] == '0' || num.size() > 3) {
                for (char32_t c : num) extend_num(u32(1, c), false);
            } else if (num.size() == 3 && !ends_with(num, U"00")) {
                extend_num(num.substr(0, 1), true);
                if (num[1] == '0') {
                    result.push_back(lookup(U"O", "", -2.0, nullptr));
                    extend_num(num.substr(2, 1), false);
                } else {
                    extend_num(num.substr(1), false);
                }
            } else {
                extend_num(num, true);
            }
        } else if (std::count(word.begin(), word.end(), U'.') > 1 || !is_head) {
            bool first = true;
            u32 w2;
            for (char32_t c : word) if (c != ',') w2 += c;
            size_t a = 0;
            for (size_t i = 0; i <= w2.size(); i++) {
                if (i < w2.size() && w2[i] != '.') continue;
                u32 num = w2.substr(a, i - a);
                a = i + 1;
                if (num.empty()) {
                } else if (num[0] == '0' || (num.size() != 2 && std::any_of(num.begin() + 1, num.end(), [](char32_t c) { return c != '0'; }))) {
                    for (char32_t c : num) extend_num(u32(1, c), false);
                } else {
                    extend_num(num, first);
                }
                first = false;
            }
        } else if (currency && (*currency == U"$" || *currency == U"£" || *currency == U"€") && is_currency(word)) {
            const char32_t * units[2];
            if (*currency == U"$") { units[0] = U"dollar"; units[1] = U"cent"; }
            else if (*currency == U"£") { units[0] = U"pound"; units[1] = U"pence"; }
            else { units[0] = U"euro"; units[1] = U"cent"; }
            u32 w2;
            for (char32_t c : word) if (c != ',') w2 += c;
            std::vector<std::pair<u32, u32>> pairs;  // (digits, unit)
            size_t dot = w2.find('.');
            pairs.push_back({dot == u32::npos ? w2 : w2.substr(0, dot), units[0]});
            if (dot != u32::npos) pairs.push_back({w2.substr(dot + 1), units[1]});
            auto value_is_zero = [](const u32 & d) { return std::all_of(d.begin(), d.end(), [](char32_t c) { return c == '0'; }); };
            auto value_is_one = [](const u32 & d) {
                size_t i = 0;
                while (i + 1 < d.size() && d[i] == '0') i++;
                return d.substr(i) == U"1";
            };
            if (pairs.size() > 1) {
                if (value_is_zero(pairs[1].first)) pairs.resize(1);
                else if (value_is_zero(pairs[0].first)) pairs.erase(pairs.begin());
            }
            for (size_t i = 0; i < pairs.size(); i++) {
                if (i > 0) result.push_back(lookup(U"and", "", {}, nullptr));
                u32 d = pairs[i].first.empty() ? U"0" : pairs[i].first;
                extend_num(d, i == 0);
                const u32 & unit = pairs[i].second;
                if (!value_is_one(d) && unit != U"pence") result.push_back(stem_s(unit + U"s", "", {}, nullptr));
                else result.push_back(lookup(unit, "", {}, nullptr));
            }
        } else {
            u32 words;
            if (digits) {
                if (!number_fits(to_utf8(word))) return {};
                words = number_cardinal(to_utf8(word));
            } else if (!contains(word, U'.')) {
                u32 w2;
                for (char32_t c : word) if (c != ',') w2 += c;
                if (!is_ascii_digits(w2) || !number_fits(to_utf8(w2))) return {};
                words = is_ordinal_suffix() ? number_ordinal(to_utf8(w2)) : number_cardinal(to_utf8(w2));
            } else {
                u32 w2;
                for (char32_t c : word) if (c != ',') w2 += c;
                if (w2[0] == '.') {
                    words = U"point";
                    for (size_t i = 1; i < w2.size(); i++) words += U" " + number_cardinal(to_utf8(u32(1, w2[i])));
                } else {
                    words = number_float(to_utf8(w2));
                }
            }
            extend_words(words, true);
        }
        if (result.empty()) return {};
        u32 joined;
        int rating = 99;
        for (size_t i = 0; i < result.size(); i++) {
            if (!result[i].ps) return {};  // misaki would raise on a None in ' '.join
            if (i) joined += U" ";
            joined += *result[i].ps;
            rating = std::min(rating, result[i].rating.value_or(0));
        }
        if (suffix && (*suffix == U"s" || *suffix == U"'s")) return {_s(joined), rating};
        if (suffix && (*suffix == U"ed" || *suffix == U"'d")) return {_ed(Ps(joined)), rating};
        if (suffix && *suffix == U"ing") return {_ing(Ps(joined)), rating};
        return {joined, rating};
    }

    Ps append_currency(const Ps & ps, const std::optional<u32> & currency) const {
        if (!currency || currency->empty()) return ps;
        const char32_t * unit = nullptr;
        if (*currency == U"$") unit = U"dollars";
        else if (*currency == U"£") unit = U"pounds";
        else if (*currency == U"€") unit = U"euros";
        if (!unit) return ps;
        Ps c = stem_s(unit, "", {}, nullptr).ps;
        return c ? Ps(*ps + U" " + *c) : ps;
    }

    static bool is_number(const u32 & word_in, bool is_head) {
        if (!std::any_of(word_in.begin(), word_in.end(), [](char32_t c) { return c >= '0' && c <= '9'; })) return false;
        u32 word = word_in;
        for (const char32_t * s : {U"ing", U"'d", U"ed", U"'s", U"st", U"nd", U"rd", U"th", U"s"}) {
            if (ends_with(word, s)) {
                word = word.substr(0, word.size() - u32(s).size());
                break;
            }
        }
        for (size_t i = 0; i < word.size(); i++) {
            char32_t c = word[i];
            if (!((c >= '0' && c <= '9') || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))) return false;
        }
        return true;
    }

    Result operator()(const MToken & tk, const TokenContext & ctx) const {
        u32 word = tk.alias ? *tk.alias : tk.text;
        word = replace_all(replace_all(word, U"‘", U"'"), U"’", U"'");
        word = nfkc(word);
        u32 w2;
        for (char32_t c : word) {
            int d = py_digit_value(c);
            // numeric_if_needed: every isdigit() character has an integral value
            if (d >= 0) w2 += U(std::to_string(d).c_str());
            else w2 += c;
        }
        word = w2;
        Stress stress;
        if (word != py_lower(word)) stress = (word == py_upper(word)) ? 2.0 : 0.5;
        Result r = get_word(word, tk.tag, stress, ctx);
        if (r.ps) return {apply_stress(append_currency(r.ps, tk.currency), tk.stress), r.rating};
        if (is_number(word, tk.is_head)) {
            Result n = get_number(word, tk.currency, tk.is_head, tk.num_flags);
            return {apply_stress(n.ps, tk.stress), n.rating};
        }
        return {};
    }

  private:
    const LexFile & f_;
};

MToken merge_tokens(const std::vector<MToken> & tokens, const std::optional<u32> & unk) {
    std::set<double> stresses;
    for (auto & t : tokens) if (t.stress) stresses.insert(*t.stress);
    std::optional<u32> currency;
    for (auto & t : tokens) if (t.currency && (!currency || *t.currency > *currency)) currency = t.currency;
    bool any_none_rating = false;
    int min_rating = 1 << 30;
    for (auto & t : tokens) {
        if (!t.rating) any_none_rating = true;
        else min_rating = std::min(min_rating, *t.rating);
    }
    MToken out;
    std::optional<u32> ps;
    if (unk) {
        u32 p;
        for (auto & tk : tokens) {
            if (tk.prespace && !p.empty() && !py_isspace(p.back()) && tk.phonemes && !tk.phonemes->empty()) p += U" ";
            p += tk.phonemes ? *tk.phonemes : *unk;
        }
        ps = p;
    }
    for (size_t i = 0; i + 1 < tokens.size(); i++) out.text += tokens[i].text + tokens[i].whitespace;
    out.text += tokens.back().text;
    int best = -1;
    for (auto & tk : tokens) {
        int score = 0;
        for (char32_t c : tk.text) score += (u32(1, c) == py_lower(u32(1, c))) ? 1 : 2;
        if (score > best) {
            best = score;
            out.tag = tk.tag;
        }
    }
    out.whitespace = tokens.back().whitespace;
    out.phonemes = ps;
    out.is_head = tokens[0].is_head;
    out.alias.reset();
    if (stresses.size() == 1) out.stress = *stresses.begin();
    out.currency = currency;
    std::set<char> flags;
    for (auto & t : tokens) for (char c : t.num_flags) flags.insert(c);
    for (char c : flags) out.num_flags += c;
    out.prespace = tokens[0].prespace;
    if (!any_none_rating) out.rating = min_rating;
    return out;
}

TokenContext token_context(const TokenContext & ctx, const Ps & ps, const MToken & token) {
    TokenContext out;
    out.future_vowel = ctx.future_vowel;
    if (ps && !ps->empty()) {
        for (char32_t c : *ps) {
            if (contains(kVowels, c) || contains(kConsonants, c) || contains(kNonQuotePuncts, c)) {
                if (contains(kNonQuotePuncts, c)) out.future_vowel.reset();
                else out.future_vowel = contains(kVowels, c);
                break;
            }
        }
    }
    out.future_to = token.text == U"to" || token.text == U"To" || (token.text == U"TO" && (token.tag == "TO" || token.tag == "IN"));
    return out;
}

void resolve_tokens(std::vector<MToken> & tokens) {
    u32 text;
    for (size_t i = 0; i + 1 < tokens.size(); i++) text += tokens[i].text + tokens[i].whitespace;
    text += tokens.back().text;
    bool prespace = contains(text, U' ') || contains(text, U'/');
    if (!prespace) {
        std::set<int> kinds;
        for (char32_t c : text) {
            if (contains(kSubtokenJunks, c)) continue;
            kinds.insert(py_isalpha(c) ? 0 : ((c >= '0' && c <= '9') ? 1 : 2));
        }
        prespace = kinds.size() > 1;
    }
    for (size_t i = 0; i < tokens.size(); i++) {
        MToken & tk = tokens[i];
        if (!tk.phonemes) {
            if (i == tokens.size() - 1 && tk.text.size() == 1 && contains(kNonQuotePuncts, tk.text[0])) {
                tk.phonemes = tk.text;
                tk.rating = 3;
            } else if (std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { return contains(kSubtokenJunks, c); })) {
                tk.phonemes = u32();
                tk.rating = 3;
            }
        } else if (i > 0) {
            tk.prespace = prespace;
        }
    }
    if (prespace) return;
    struct Idx { bool primary; int weight; size_t i; };
    std::vector<Idx> indices;
    for (size_t i = 0; i < tokens.size(); i++) {
        if (tokens[i].phonemes && !tokens[i].phonemes->empty()) {
            indices.push_back({contains(*tokens[i].phonemes, kPrimary), stress_weight(tokens[i].phonemes), i});
        }
    }
    if (indices.size() == 2 && tokens[indices[0].i].text.size() == 1) {
        size_t i = indices[1].i;
        tokens[i].phonemes = apply_stress(tokens[i].phonemes, -0.5);
        return;
    }
    size_t n_primary = 0;
    for (auto & x : indices) n_primary += x.primary;
    if (indices.size() < 2 || n_primary <= (indices.size() + 1) / 2) return;
    std::sort(indices.begin(), indices.end(), [](const Idx & a, const Idx & b) {
        if (a.primary != b.primary) return a.primary < b.primary;
        if (a.weight != b.weight) return a.weight < b.weight;
        return a.i < b.i;
    });
    indices.resize(indices.size() / 2);
    for (auto & x : indices) tokens[x.i].phonemes = apply_stress(tokens[x.i].phonemes, -0.5);
}

// misaki G2P.preprocess: markdown-style [text](feature) links.
struct Preprocessed {
    u32 text;
    std::vector<u32> tokens;
    std::vector<std::pair<size_t, std::string>> features;  // token index -> raw feature
};

std::vector<u32> py_split(const u32 & s) {
    std::vector<u32> out;
    u32 cur;
    for (char32_t c : s) {
        if (py_isspace(c)) {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

Preprocessed preprocess(const u32 & text_in) {
    Preprocessed p;
    size_t a = 0;
    while (a < text_in.size() && py_isspace(text_in[a])) a++;
    u32 text = text_in.substr(a);
    size_t last_end = 0;
    size_t pos = 0;
    // LINK_REGEX = r'\[([^\]]+)\]\(([^\)]*)\)'
    while (pos < text.size()) {
        size_t lb = text.find('[', pos);
        if (lb == u32::npos) break;
        size_t rb = lb + 1;
        while (rb < text.size() && text[rb] != ']') rb++;
        if (rb >= text.size() || rb == lb + 1 || rb + 1 >= text.size() || text[rb + 1] != '(') {
            pos = lb + 1;
            continue;
        }
        size_t rp = rb + 2;
        while (rp < text.size() && text[rp] != ')') rp++;
        if (rp >= text.size()) {
            pos = lb + 1;
            continue;
        }
        u32 g1 = text.substr(lb + 1, rb - lb - 1);
        u32 f = text.substr(rb + 2, rp - rb - 2);
        p.text += text.substr(last_end, lb - last_end);
        for (auto & t : py_split(text.substr(last_end, lb - last_end))) p.tokens.push_back(t);
        std::string feat;
        u32 body = f;
        if (!f.empty() && (f[0] == '-' || f[0] == '+')) body = f.substr(1);
        if (is_ascii_digits(body)) {
            feat = "i" + to_utf8(f);
        } else if (f == U"0.5" || f == U"+0.5") {
            feat = "f0.5";
        } else if (f == U"-0.5") {
            feat = "f-0.5";
        } else if (f.size() > 1 && f[0] == '/' && f.back() == '/') {
            u32 v = f.substr(1);
            while (!v.empty() && v.back() == '/') v.pop_back();
            while (!v.empty() && v.front() == '/') v.erase(v.begin());  // later .lstrip('/')
            feat = "s/" + to_utf8(v);
        } else if (f.size() > 1 && f[0] == '#' && f.back() == '#') {
            u32 v = f.substr(1);
            while (!v.empty() && v.back() == '#') v.pop_back();
            while (!v.empty() && v.front() == '#') v.erase(v.begin());
            feat = "s#" + to_utf8(v);
        }
        if (!feat.empty()) p.features.emplace_back(p.tokens.size(), feat);
        p.text += g1;
        p.tokens.push_back(g1);
        last_end = rp + 1;
        pos = rp + 1;
    }
    if (last_end < text.size()) {
        p.text += text.substr(last_end);
        for (auto & t : py_split(text.substr(last_end))) p.tokens.push_back(t);
    }
    return p;
}

// A misaki "word": one token, or a run of subtokens with no space between.
struct Word {
    bool is_list = false;
    std::vector<MToken> toks;
};

}  // namespace

std::optional<u32> apply_stress_for_test(const std::optional<u32> & ps, std::optional<double> stress) {
    return apply_stress(ps, stress);
}

std::optional<u32> G2P::fallback(const MToken & tk) {
    last_fallback_count++;
    std::optional<u32> ps;
    if (fallback_) {
        ps = (*fallback_)(tk.text);
    } else {
        // No eSpeak: read the word's letters, like misaki's get_NNP.
        Result r = Lexicon(lex_).get_NNP(tk.text);
        if (r.ps && !r.ps->empty()) ps = r.ps;
    }
    if (ps) {
        // paradee maps phonemes to ids with `if c in vocab`; drop the rare
        // eSpeak symbols outside the vocabulary here so the string has the
        // same ids and every symbol is valid.
        u32 kept;
        for (char32_t c : *ps) if (in_paradee_vocab(c)) kept += c;
        ps = kept;
    }
    return ps;
}

u32 G2P::operator()(const u32 & input, std::vector<MToken> * tokens_out) {
    Lexicon lexicon(lex_);
    last_fallback_count = 0;
    const std::optional<u32> unk = u32();  // paradee: unk=''
    Preprocessed pre = preprocess(input);
    std::vector<SpacyToken> st = spacy_tokenize(pre.text);
    std::vector<std::string> tags = tag_tokens(st, &lex_);
    std::vector<MToken> tokens;
    for (size_t i = 0; i < st.size(); i++) {
        MToken t;
        t.text = st[i].text;
        t.tag = tags[i];
        t.whitespace = st[i].space_after ? U" " : U"";
        tokens.push_back(std::move(t));
    }
    if (!pre.features.empty()) {
        // spacy.training.Alignment.from_strings(tokens, spacy tokens): align
        // on characters, whitespace removed, lowercased.
        std::vector<size_t> xchar, ychar;
        u32 xs, ys;
        for (size_t k = 0; k < pre.tokens.size(); k++) {
            for (char32_t c : pre.tokens[k]) if (!py_isspace(c)) { xs += c; xchar.push_back(k); }
        }
        for (size_t k = 0; k < tokens.size(); k++) {
            for (char32_t c : tokens[k].text) if (!py_isspace(c)) { ys += c; ychar.push_back(k); }
        }
        std::vector<size_t> flat;  // align.y2x.data
        if (py_lower(xs) == py_lower(ys)) {
            for (size_t y = 0; y < tokens.size(); y++) {
                std::set<size_t> xsset;
                for (size_t c = 0; c < ychar.size(); c++) if (ychar[c] == y) xsset.insert(xchar[c]);
                for (size_t x : xsset) flat.push_back(x);
            }
        }
        for (auto & fv : pre.features) {
            size_t k = fv.first;
            const std::string & v = fv.second;
            size_t i = 0;
            for (size_t pos = 0; pos < flat.size(); pos++) {
                if (flat[pos] != k) continue;
                size_t j = pos;
                size_t this_i = i++;
                if (j >= tokens.size()) continue;
                if (v[0] == 'i') tokens[j].stress = std::stod(v.substr(1));
                else if (v[0] == 'f') tokens[j].stress = std::stod(v.substr(1));
                else if (v.rfind("s/", 0) == 0) {
                    tokens[j].is_head = this_i == 0;
                    tokens[j].phonemes = this_i == 0 ? U(v.substr(2).c_str()) : u32();
                    tokens[j].rating = 5;
                } else if (v.rfind("s#", 0) == 0) {
                    tokens[j].num_flags = v.substr(2);
                }
            }
        }
    }
    // fold_left
    {
        std::vector<MToken> result;
        for (auto & tk : tokens) {
            if (!result.empty() && !tk.is_head) {
                MToken prev = result.back();
                result.pop_back();
                result.push_back(merge_tokens({prev, tk}, unk));
            } else {
                result.push_back(tk);
            }
        }
        tokens.swap(result);
    }
    // retokenize
    std::vector<Word> words;
    {
        std::optional<u32> currency;
        for (size_t i = 0; i < tokens.size(); i++) {
            const MToken & token = tokens[i];
            std::vector<MToken> tks;
            if (!token.alias && !token.phonemes) {
                for (const u32 & t : subtokenize(token.text)) {
                    MToken s;
                    s.text = t;
                    s.tag = token.tag;
                    s.whitespace = U"";
                    s.phonemes = token.phonemes;
                    s.is_head = true;
                    s.num_flags = token.num_flags;
                    s.stress = token.stress;
                    s.prespace = false;
                    tks.push_back(std::move(s));
                }
            } else {
                tks.push_back(token);
            }
            if (tks.empty()) continue;  // subtokenize found nothing (misaki would raise on tks[-1])
            tks.back().whitespace = token.whitespace;
            for (size_t j = 0; j < tks.size(); j++) {
                MToken & tk = tks[j];
                if (tk.alias || tk.phonemes) {
                } else if (tk.tag == "$" && (tk.text == U"$" || tk.text == U"£" || tk.text == U"€")) {
                    currency = tk.text;
                    tk.phonemes = u32();
                    tk.rating = 4;
                } else if (tk.tag == ":" && (tk.text == U"-" || tk.text == U"–")) {
                    tk.phonemes = U"—";
                    tk.rating = 3;
                } else if (is_punct_tag(tk.tag) &&
                           !std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) {
                               u32 l = py_lower(u32(1, c));
                               return l.size() == 1 && l[0] >= 97 && l[0] <= 122;
                           })) {
                    if (tk.tag == "-LRB-") tk.phonemes = U"(";
                    else if (tk.tag == "-RRB-") tk.phonemes = U")";
                    else if (tk.tag == "``") tk.phonemes = U"“";
                    else if (tk.tag == "\"\"" || tk.tag == "''") tk.phonemes = U"”";
                    else {
                        u32 p;
                        for (char32_t c : tk.text) if (contains(kPuncts, c)) p += c;
                        tk.phonemes = p;
                    }
                    tk.rating = 4;
                } else if (currency) {
                    if (tk.tag != "CD") {
                        currency.reset();
                    } else if (j + 1 == tks.size() && (i + 1 == tokens.size() || tokens[i + 1].tag != "CD")) {
                        tk.currency = currency;
                    }
                } else if (j > 0 && j + 1 < tks.size() && tk.text == U"2") {
                    u32 pair;
                    pair += tks[j - 1].text.back();
                    pair += tks[j + 1].text.front();
                    if (py_str_isalpha(pair)) tk.alias = U"to";
                }
                if (tk.alias || tk.phonemes) {
                    words.push_back({false, {tk}});
                } else if (!words.empty() && words.back().is_list && words.back().toks.back().whitespace.empty()) {
                    tk.is_head = false;
                    words.back().toks.push_back(tk);
                } else if (!tk.whitespace.empty()) {
                    words.push_back({false, {tk}});
                } else {
                    words.push_back({true, {tk}});
                }
            }
        }
        // misaki's list-word search is cubic in the number of subtokens; a
        // run of more than 64 (never seen in real text) is cut into pieces so
        // hostile input cannot stall the caller.
        constexpr size_t kMaxRun = 64;
        std::vector<Word> capped;
        for (auto & w : words) {
            if (!w.is_list || w.toks.size() <= kMaxRun) {
                capped.push_back(std::move(w));
                continue;
            }
            for (size_t a = 0; a < w.toks.size(); a += kMaxRun) {
                Word part;
                part.is_list = true;
                part.toks.assign(w.toks.begin() + a, w.toks.begin() + std::min(w.toks.size(), a + kMaxRun));
                capped.push_back(std::move(part));
            }
        }
        words.swap(capped);
        for (auto & w : words) if (w.is_list && w.toks.size() == 1) w.is_list = false;
    }
    TokenContext ctx;
    for (size_t wi = words.size(); wi-- > 0;) {
        Word & w = words[wi];
        if (!w.is_list) {
            MToken & tk = w.toks[0];
            if (!tk.phonemes) {
                Result r = lexicon(tk, ctx);
                tk.phonemes = r.ps;
                tk.rating = r.rating;
            }
            if (!tk.phonemes) {
                tk.phonemes = fallback(tk);
                tk.rating = tk.phonemes ? std::optional<int>(2) : std::nullopt;
            }
            ctx = token_context(ctx, tk.phonemes, tk);
            continue;
        }
        size_t left = 0, right = w.toks.size();
        bool should_fallback = false;
        while (left < right) {
            bool any_set = false;
            for (size_t k = left; k < right; k++) if (w.toks[k].alias || w.toks[k].phonemes) any_set = true;
            Result r;
            MToken merged;
            if (!any_set) {
                std::vector<MToken> part(w.toks.begin() + left, w.toks.begin() + right);
                merged = merge_tokens(part, std::nullopt);
                r = lexicon(merged, ctx);
            }
            if (r.ps) {
                w.toks[left].phonemes = r.ps;
                w.toks[left].rating = r.rating;
                for (size_t k = left + 1; k < right; k++) w.toks[k].phonemes = u32();
                ctx = token_context(ctx, r.ps, merged);
                right = left;
                left = 0;
            } else if (left + 1 < right) {
                left++;
            } else {
                right--;
                MToken & tk = w.toks[right];
                if (!tk.phonemes) {
                    if (std::all_of(tk.text.begin(), tk.text.end(), [](char32_t c) { return contains(kSubtokenJunks, c); })) {
                        tk.phonemes = u32();
                        tk.rating = 3;
                    } else {
                        should_fallback = true;
                        break;
                    }
                }
                left = 0;
            }
        }
        if (should_fallback) {
            MToken merged = merge_tokens(w.toks, std::nullopt);
            w.toks[0].phonemes = fallback(merged);
            w.toks[0].rating = w.toks[0].phonemes ? std::optional<int>(2) : std::nullopt;
            for (size_t k = 1; k < w.toks.size(); k++) {
                w.toks[k].phonemes = u32();
                w.toks[k].rating = w.toks[0].rating;
            }
        } else {
            resolve_tokens(w.toks);
        }
    }
    std::vector<MToken> final_tokens;
    for (auto & w : words) {
        if (w.is_list) final_tokens.push_back(merge_tokens(w.toks, unk));
        else final_tokens.push_back(w.toks[0]);
    }
    u32 out;
    for (auto & tk : final_tokens) {
        if (tk.phonemes) {
            for (char32_t & c : *tk.phonemes) {
                if (c == U'ɾ') c = 'T';
                else if (c == U'ʔ') c = 't';
            }
        }
        out += (tk.phonemes ? *tk.phonemes : *unk) + tk.whitespace;
    }
    if (tokens_out) *tokens_out = std::move(final_tokens);
    return out;
}

std::vector<u32> paradee_chunks(G2P & g2p, const u32 & text, size_t max_phonemes) {
    // re.split(r"(?<=[.!?…])\s+|\n+", text.strip())
    u32 t = py_strip(text);
    std::vector<u32> sentences;
    u32 cur;
    size_t i = 0;
    while (i < t.size()) {
        char32_t c = t[i];
        bool after_end = i > 0 && contains(U".!?…", t[i - 1]);
        if (after_end && py_isspace(c)) {
            size_t j = i;
            while (j < t.size() && py_isspace(t[j])) j++;
            sentences.push_back(cur);
            cur.clear();
            i = j;
            continue;
        }
        if (c == '\n') {
            size_t j = i;
            while (j < t.size() && t[j] == '\n') j++;
            sentences.push_back(cur);
            cur.clear();
            i = j;
            continue;
        }
        cur += c;
        i++;
    }
    sentences.push_back(cur);
    std::vector<u32> chunks;
    for (const u32 & s : sentences) {
        if (py_strip(s).empty()) continue;
        u32 ps = g2p(s);
        while (ps.size() > max_phonemes) {
            size_t cut = ps.rfind(U' ', max_phonemes - 1);
            // ps.rfind(" ", 0, MAX) searches ps[0:MAX]; cut > 0 or MAX
            if (cut == u32::npos || cut == 0) cut = max_phonemes;
            chunks.push_back(ps.substr(0, cut));
            ps = ps.substr(cut);
            size_t k = 0;
            while (k < ps.size() && py_isspace(ps[k])) k++;
            ps = ps.substr(k);
        }
        if (!ps.empty()) chunks.push_back(ps);
    }
    return chunks;
}

}  // namespace misaki
