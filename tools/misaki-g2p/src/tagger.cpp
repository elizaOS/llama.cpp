// SPDX-License-Identifier: Apache-2.0
// A small rule-based Penn Treebank tagger standing in for spaCy's
// en_core_web_sm tagger. misaki reads the tag in a few places only:
//   - punctuation, currency ($), dashes (:) and numbers (CD) in retokenize
//   - a/an/am/I/by/in/to/the/that/used/vs special cases
//   - NNP for all-caps words (spelled out unless the lexicon stresses them)
//   - noun / verb / adjective / adverb / past-tense variants of about 790
//     homographs (record, present, read, use, close, live, ...)
// so the rules aim at those decisions, not at full tagging accuracy.
#include "misaki-internal.h"

#include <unordered_map>
#include <unordered_set>

namespace misaki {
namespace {

const std::unordered_map<std::string, const char *> & closed_class() {
    static const std::unordered_map<std::string, const char *> m = [] {
        std::unordered_map<std::string, const char *> t;
        auto add = [&](const char * tag, std::initializer_list<const char *> words) {
            for (const char * w : words) t[w] = tag;
        };
        add("DT", {"the", "a", "an", "this", "these", "those", "every", "each", "no", "another", "some", "any",
                   "all", "both", "either", "neither", "half", "an", "th'"});
        add("PDT", {"such"});
        add("PRP", {"i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them", "myself", "yourself",
                    "himself", "herself", "itself", "ourselves", "yourselves", "themselves", "'em", "one's"});
        add("PRP$", {"my", "your", "his", "its", "our", "their", "her"});
        add("IN", {"of", "in", "on", "at", "by", "for", "with", "from", "into", "onto", "about", "as", "than",
                   "after", "before", "during", "since", "until", "till", "through", "throughout", "over", "under",
                   "between", "among", "amongst", "against", "without", "within", "upon", "toward", "towards",
                   "across", "behind", "beyond", "despite", "like", "near", "per", "via", "if", "because",
                   "although", "though", "while", "whilst", "whether", "unless", "around", "along", "alongside",
                   "beside", "besides", "below", "beneath", "above", "inside", "outside", "underneath", "amid",
                   "amidst", "except", "unlike", "versus", "vs", "vs.", "whereas", "onto", "atop", "aboard",
                   "past", "o'er", "whereby", "albeit", "lest"});
        add("TO", {"to"});
        add("CC", {"and", "or", "but", "nor", "&", "plus", "yet", "either", "neither"});
        add("MD", {"can", "could", "will", "would", "shall", "should", "may", "might", "must", "'ll", "wo", "ca",
                   "cannot", "'d", "ought"});
        add("VB", {"be"});
        add("VBZ", {"is", "has", "does", "'s"});
        add("VBP", {"am", "are", "'re", "'m", "do", "have", "'ve"});
        add("VBD", {"was", "were", "did", "had"});
        add("VBN", {"been"});
        add("VBG", {"being", "having", "doing"});
        add("RB", {"not", "n't", "never", "also", "very", "often", "always", "just", "only", "then", "now", "here",
                   "too", "still", "already", "soon", "even", "again", "ever", "however", "almost", "quite",
                   "rather", "perhaps", "maybe", "away", "back", "later", "together", "once", "twice", "thus",
                   "therefore", "instead", "nearly", "else", "sometimes", "usually", "today", "tonight",
                   "tomorrow", "yesterday", "anyway", "otherwise", "indeed", "somewhat", "abroad", "ago",
                   "afterwards", "afterward", "meanwhile", "nevertheless", "nonetheless", "hence", "elsewhere",
                   "everywhere", "anywhere", "somewhere", "nowhere", "seldom", "rarely", "forth", "further",
                   "furthermore", "moreover", "much", "more", "most", "less", "least", "well", "so"});
        add("WDT", {"which", "whatever", "whichever"});
        add("WP", {"who", "whom", "what", "whoever", "whomever"});
        add("WP$", {"whose"});
        add("WRB", {"when", "where", "why", "how", "whenever", "wherever", "whereupon"});
        add("EX", {"there"});
        add("UH", {"oh", "yes", "hello", "hi", "hey", "okay", "ok", "please", "wow", "goodbye", "thanks",
                   "uh", "um", "ah", "yeah", "bye"});
        add("RP", {"up", "out", "off", "down"});
        add("POS", {"'", "’"});
        add("JJ", {"other", "many", "few", "several", "own", "same", "next", "last", "first", "new", "old",
                   "good", "great", "large", "small", "big", "little", "long", "high", "early", "late", "young",
                   "able", "major", "main", "various", "certain", "different", "important", "possible", "local",
                   "national", "public", "political", "social", "real", "sure", "free", "full", "special",
                   "general", "whole", "previous", "former", "original", "final", "second", "third",
                   "current", "single", "white", "black", "red", "green", "blue", "dark", "rare", "hard", "easy",
                   "fine", "quick", "brown", "lazy", "smooth", "warm", "sunny", "quiet", "ready", "due", "loud",
                   "best", "better", "worse", "worst"});
        add("NN", {"%"});
        return t;
    }();
    return m;
}

bool has_alpha(const u32 & s) {
    for (char32_t c : s) if (py_isalpha(c)) return true;
    return false;
}
bool has_digit(const u32 & s) {
    for (char32_t c : s) if (c >= '0' && c <= '9') return true;
    return false;
}
bool word_like(const u32 & t) { return has_alpha(t) || has_digit(t); }

bool is_currency_symbol(const u32 & t) {
    static const std::unordered_set<std::u32string> s = {U"$", U"£", U"€", U"¥", U"₹", U"₽", U"₩", U"฿", U"US$",
                                                         U"C$", U"A$", U"₱", U"₪", U"₫", U"₴", U"₦"};
    return s.count(t) > 0;
}

bool is_capitalized(const u32 & t) {
    return !t.empty() && py_isalpha(t[0]) && py_lower(t.substr(0, 1)) != t.substr(0, 1);
}

bool is_all_caps(const u32 & t) {
    int letters = 0;
    for (char32_t c : t) {
        if (py_isalpha(c)) {
            letters++;
            if (py_lower(u32(1, c)) == u32(1, c)) return false;
        }
    }
    return letters >= 2;
}

bool is_quote(const u32 & t) { return t == U"\"" || t == U"“" || t == U"”" || t == U"'" || t == U"‘" || t == U"’" || t == U"`" || t == U"``" || t == U"''"; }

std::string punct_tag(const u32 & t) {
    if (t == U"." || t == U"!" || t == U"?" || t == U"?!" || t == U"!?" || t == U"!!" || t == U"??") return ".";
    if (t == U",") return ",";
    if (t == U"(" || t == U"[" || t == U"{") return "-LRB-";
    if (t == U")" || t == U"]" || t == U"}") return "-RRB-";
    if (t == U"“" || t == U"``" || t == U"‘" || t == U"`" || t == U"«") return "``";
    if (t == U"”" || t == U"''" || t == U"»" || t == U"’") return "''";
    if (t == U"…") return "NFP";
    if (t == U"%") return "NN";
    if (t == U"&") return "CC";
    if (t == U"#") return "NN";
    if (t == U"/" || t == U"=" || t == U"+") return "SYM";
    if (t == U"@") return "IN";
    if (t == U"*" || t == U"~" || t == U"•" || t == U"·") return "NFP";
    if (is_currency_symbol(t)) return "$";
    bool dashes = !t.empty();
    for (char32_t c : t) {
        if (!(c == ';' || c == ':' || c == '-' || c == U'—' || c == U'–' || c == '.' || c == U'…')) dashes = false;
    }
    if (dashes) return ":";  // ; : - -- — – ... and runs of them
    return "SYM";            // °, ©, ™, emoji and other symbols are read as words
}

struct Tok {
    u32 text;
    u32 lower;
    std::string tag;
    bool space_after = false;
    bool space_before = false;
    bool sent_start = false;
    bool closed = false;  // tag came from the closed-class list
};

bool is_verb_tag(const std::string & t) { return t.rfind("VB", 0) == 0 || t == "MD"; }
bool is_noun_tag(const std::string & t) { return t.rfind("NN", 0) == 0; }

bool subject_pronoun(const u32 & l) {
    return l == U"i" || l == U"you" || l == U"we" || l == U"they" || l == U"he" || l == U"she" || l == U"it";
}
bool third_person(const u32 & l) { return l == U"he" || l == U"she" || l == U"it"; }
bool is_be(const u32 & l) {
    return l == U"be" || l == U"is" || l == U"are" || l == U"was" || l == U"were" || l == U"been" || l == U"being" ||
           l == U"am" || l == U"'s" || l == U"'re" || l == U"'m";
}
bool is_have(const u32 & l) { return l == U"have" || l == U"has" || l == U"had" || l == U"'ve" || l == U"having"; }
bool is_do(const u32 & l) { return l == U"do" || l == U"does" || l == U"did"; }

std::string guess_open_class(const u32 & l) {
    if (ends_with(l, U"ly") && l.size() > 4) return "RB";
    if (ends_with(l, U"ing") && l.size() > 4) return "VBG";
    if (ends_with(l, U"ed") && l.size() > 3) return "VBN";
    for (const char32_t * s : {U"ous", U"ful", U"ive", U"able", U"ible", U"ic", U"less", U"ish", U"ary", U"al", U"ant",
                               U"ent", U"est"}) {
        if (ends_with(l, s) && l.size() > u32(s).size() + 2) return "JJ";
    }
    if (ends_with(l, U"s") && !ends_with(l, U"ss") && !ends_with(l, U"us") && !ends_with(l, U"is") && l.size() > 3) {
        return "NNS";
    }
    return "NN";
}

// Which readings the lexicon distinguishes for a homograph: from the word's
// own gold entry, or the entry of the stem misaki falls back to.
struct Prior {
    bool any = false;
    bool verb_variant = false;  // DEFAULT is the noun/adjective reading
    bool noun_variant = false;  // DEFAULT is the verb reading
    bool adj_variant = false;
    bool adv_variant = false;
};

Prior homograph_prior(const LexFile * lex, const u32 & l) {
    Prior p;
    if (!lex) return p;
    auto from = [&](const LexValue & v) {
        if (!v.found || !v.is_dict) return false;
        p.verb_variant = v.has_tag[TAG_VERB] || v.has_tag[TAG_VBD] || v.has_tag[TAG_VBN] || v.has_tag[TAG_VBP];
        p.noun_variant = v.has_tag[TAG_NOUN];
        p.adj_variant = v.has_tag[TAG_ADJ];
        p.adv_variant = v.has_tag[TAG_ADV];
        p.any = p.verb_variant || p.noun_variant || p.adj_variant || p.adv_variant;
        return p.any;
    };
    LexValue own = lex->gold(l);
    if (own.found) {
        from(own);
        return p;
    }
    if (lex->in_silver(l)) return p;
    for (const char32_t * suf : {U"s", U"es", U"ed", U"d", U"ing"}) {
        if (ends_with(l, suf) && l.size() > u32(suf).size() + 2) {
            u32 stem = l.substr(0, l.size() - u32(suf).size());
            if (from(lex->gold(stem))) return p;
            if (u32(suf) == U"ing" && from(lex->gold(stem + U"e"))) return p;
            if (u32(suf) == U"es" && ends_with(stem, U"i") && from(lex->gold(stem.substr(0, stem.size() - 1) + U"y"))) return p;
        }
    }
    return p;
}

}  // namespace

std::vector<std::string> tag_tokens(const std::vector<SpacyToken> & in, const LexFile * lex) {
    const auto & cc = closed_class();
    const size_t N = in.size();
    std::vector<Tok> t(N);
    for (size_t i = 0; i < N; i++) {
        t[i].text = in[i].text;
        t[i].lower = py_lower(in[i].text);
        t[i].space_after = in[i].space_after;
        t[i].space_before = i == 0 || in[i - 1].space_after;
    }
    for (size_t i = 0; i < N; i++) {
        if (i == 0) { t[i].sent_start = true; continue; }
        const u32 & p = t[i - 1].text;
        if (p == U"." || p == U"!" || p == U"?" || p == U":" || p == U";" || p == U"“" || p == U"(" || p == U"\"")
            t[i].sent_start = !(p == U"\"" && t[i - 1].space_after && i >= 2 && word_like(t[i - 2].text));
    }

    // pass 1: lexical tags
    int dq = 0;  // straight double quotes seen so far
    for (size_t i = 0; i < N; i++) {
        Tok & k = t[i];
        const u32 & w = k.text;
        if (!word_like(w)) {
            if (w == U"\"") {
                // spaCy reads straight quotes mostly by parity, with a few
                // context habits learned from WikiText-style spacing
                bool odd = (dq % 2) == 0;
                dq++;
                const Tok * pv = i > 0 ? &t[i - 1] : nullptr;
                const Tok * nx = i + 1 < N ? &t[i + 1] : nullptr;
                bool more_quotes = false;
                for (size_t j = i + 1; j < N; j++) if (t[j].text == U"\"") more_quotes = true;
                if (pv && !k.space_before && (k.space_after || !nx || !word_like(nx->text))) {
                    k.tag = "''";  // glued to the word before it
                } else if (nx && k.space_before && !k.space_after && word_like(nx->text)) {
                    k.tag = "``";  // glued to the word after it
                } else if (!more_quotes && odd && pv && (!nx || nx->text == U".")) {
                    k.tag = "''";  // a lone quote closing the sentence
                } else if (odd) {
                    k.tag = (pv && pv->text == U",") ? "''" : "``";
                } else if (pv && nx && word_like(nx->text) &&
                           (pv->tag == "IN" || pv->tag == "DT" || pv->tag == "TO" || pv->tag == "CC" || pv->tag == "," ||
                            pv->tag == "VBD" || pv->tag == "VBZ" || pv->tag == "VBP" || pv->tag == "VB")) {
                    k.tag = "``";  // follows a function word: opens a quote
                } else {
                    k.tag = (nx && nx->text == U"," && pv && is_capitalized(pv->text)) ? "``" : "''";
                }
            } else if ((w == U"'" || w == U"‘" || w == U"’") && !k.space_after && i + 1 < N && k.space_before &&
                       (t[i + 1].text == U"ve" || t[i + 1].text == U"m" || t[i + 1].text == U"re" || t[i + 1].text == U"ll" ||
                        t[i + 1].text == U"d" || t[i + 1].text == U"s" ||
                        (t[i + 1].text == U"t" && i > 0 && t[i - 1].lower != U"doesn"))) {
                k.tag = "XX";  // glued to the next word (WikiText " 've", "'Yeah"): resolved below
            } else if (w.size() > 1 && std::all_of(w.begin(), w.end(), [](char32_t c) { return c == '-'; })) {
                k.tag = ":";
            } else if (w == U"-" && !k.space_before && !k.space_after) {
                k.tag = "HYPH";
            } else if (w == U"-" && !(t.back().text == U"." || t.back().text == U"!" || t.back().text == U"?" ||
                                      t.back().text == U"\"" || t.back().text == U")" || t.back().text == U"'")) {
                k.tag = "HYPH";  // spaced dash in a heading or list item (no final punctuation)
            } else if (w == U":" && i > 0 && !k.space_before && has_digit(t[i - 1].text) && i + 1 < N &&
                       has_digit(t[i + 1].text) && !has_alpha(t[i + 1].text)) {
                k.tag = "SYM";  // 22: 00, 3:2
            } else if ((w == U"'" || w == U"’") && i > 0 && !k.space_before && ends_with(t[i - 1].lower, U"s") &&
                       word_like(t[i - 1].text)) {
                k.tag = "POS";  // plural possessive: "the Smiths' house"
            } else if (w == U"'" && i + 2 < N && t[i + 1].lower == U"n" && t[i + 2].text == U"'" && !k.space_after) {
                k.tag = "CC";  // rock 'n' roll
                t[i + 1].tag = "CC";
                t[i + 2].tag = "CC";
                t[i + 1].closed = t[i + 2].closed = true;
                i += 2;
            } else if (w == U"'" || w == U"‘" || w == U"`") {
                k.tag = (k.space_before || i == 0) ? "``" : "''";
            } else {
                k.tag = punct_tag(w);
            }
            continue;
        }
        if (has_digit(w) && !has_alpha(w)) {
            k.tag = "CD";
            continue;
        }
        if (has_digit(w)) {
            k.tag = (ends_with(k.lower, U"s") && !ends_with(k.lower, U"'s")) ? "NNS" : "NN";
            if (ends_with(k.lower, U"st") || ends_with(k.lower, U"nd") || ends_with(k.lower, U"rd") || ends_with(k.lower, U"th"))
                k.tag = "JJ";
            continue;
        }
        std::string l8 = to_utf8(k.lower);
        auto it = cc.find(l8);
        bool cap = is_capitalized(w);
        if (is_all_caps(w)) {
            bool caps_run = (i > 0 && is_all_caps(t[i - 1].text)) || (i + 1 < N && is_all_caps(t[i + 1].text));
            static const std::unordered_set<std::string> caps_function = {
                "of", "the", "and", "in", "on", "at", "to", "for", "by", "with", "from", "or", "as", "is", "are", "a", "an"};
            if (it != cc.end() && ((caps_run && caps_function.count(l8)) || l8 == "to")) {
                k.tag = it->second;
                k.closed = true;
            } else {
                k.tag = "NNP";
            }
            continue;
        }
        if (it != cc.end() && (!cap || k.sent_start || w == U"I")) {
            k.tag = it->second;
            k.closed = true;
            continue;
        }
        if (it != cc.end() && cap) {
            static const std::unordered_set<std::string> namey = {"may", "will", "can", "march", "mark", "bill",
                                                                  "rose", "grant", "art", "frank", "jack", "will"};
            k.tag = namey.count(l8) ? "NNP" : it->second;
            k.closed = !namey.count(l8);
            continue;
        }
        if (cap) {
            bool common = k.sent_start && lex && (lex->in_gold(k.lower) || lex->in_silver(k.lower)) &&
                          !(i + 1 < N && is_capitalized(t[i + 1].text) && !is_all_caps(t[i + 1].text));
            k.tag = common ? guess_open_class(k.lower) : "NNP";
            continue;
        }
        k.tag = guess_open_class(k.lower);
    }
    // glued apostrophes take the tag of the word they lead
    for (size_t i = 0; i + 1 < N; i++) if (t[i].tag == "XX") t[i].tag = t[i + 1].tag;
    // number words after a number: "$ 66 million"
    for (size_t i = 1; i < N; i++) {
        const u32 & l = t[i].lower;
        if (t[i - 1].tag == "CD" && (l == U"hundred" || l == U"thousand" || l == U"million" || l == U"billion" || l == U"trillion"))
            t[i].tag = "CD";
    }

    auto P = [&](size_t i, size_t back) -> const Tok * { return i >= back ? &t[i - back] : nullptr; };
    auto Nx = [&](size_t i, size_t fwd) -> const Tok * { return i + fwd < N ? &t[i + fwd] : nullptr; };
    auto ends_clause = [&](const Tok * n) { return !n || !word_like(n->text); };
    // "record a memo and then read it": the tag of the verb this one is coordinated with
    auto coordinated_verb = [&](size_t i) -> std::string {
        size_t j = i;
        if (j >= 1 && t[j - 1].tag == "RB") j--;
        if (j < 1 || t[j - 1].tag != "CC") return "";
        for (size_t b = j - 1; b-- > 0 && j - b <= 8;) {
            const std::string & g = t[b].tag;
            if (g == "VB" || g == "VBD" || g == "VBP" || g == "VBZ") return g;
            if (!word_like(t[b].text) && t[b].text != U"'s") return "";
        }
        return "";
    };

    // pass 2: verbs after modals, to, do-support and subjects
    for (size_t i = 0; i < N; i++) {
        Tok & k = t[i];
        if (k.closed || !word_like(k.text) || k.tag == "NNP" || k.tag == "CD") continue;
        const Tok * p = P(i, 1);
        const Tok * n = Nx(i, 1);
        if (!p) continue;
        const u32 & pl = p->lower;
        const u32 & l = k.lower;
        bool after_not = (pl == U"not" || pl == U"n't" || pl == U"never") && i >= 2 &&
                         (t[i - 2].tag == "MD" || is_do(t[i - 2].lower) || t[i - 2].tag == "TO");
        bool after_adv = p->tag == "RB" && i >= 2 && (t[i - 2].tag == "MD" || t[i - 2].tag == "TO");
        if (p->tag == "MD" || p->tag == "TO" || after_not || after_adv || is_do(pl) || pl == U"please") {
            if (!(k.tag == "NNS" && p->tag != "MD" && p->tag != "TO") && k.tag != "RB" && k.tag != "VBG") k.tag = "VB";
        } else if (p->tag == "PRP" && subject_pronoun(pl)) {
            if (ends_with(l, U"ed")) k.tag = "VBD";
            else if (k.tag == "NNS" && third_person(pl)) k.tag = "VBZ";
            else if (k.tag != "RB" && k.tag != "VBG" && k.tag != "JJ") k.tag = third_person(pl) ? "VBD" : "VBP";
        } else if (k.tag == "VBN" && (is_noun_tag(p->tag) || p->tag == "PRP") && !is_be(pl) && !is_have(pl)) {
            k.tag = "VBD";
        }
        if (k.tag == "VBN" && p->tag == "DT" && n && (n->tag == "NN" || n->tag == "NNS")) k.tag = "JJ";
        if (k.tag == "VBG" && (p->tag == "DT" || p->tag == "PRP$" || p->tag == "JJ")) k.tag = "NN";
        if (k.tag == "NN" && ends_with(l, U"er") &&
            (is_be(pl) || pl == U"much" || pl == U"even" || pl == U"far" || pl == U"still" || pl == U"no" || pl == U"any"))
            k.tag = "JJR";
        (void) ends_clause;
    }

    // pass 3: homographs the lexicon splits by part of speech
    for (size_t i = 0; i < N; i++) {
        Tok & k = t[i];
        if (k.closed || !word_like(k.text) || !has_alpha(k.text) || k.tag == "NNP" || k.tag == "NNPS") continue;
        if (is_capitalized(k.text) && !k.sent_start) continue;
        const u32 & l = k.lower;
        Prior pr = homograph_prior(lex, l);
        if (!pr.any) continue;
        const Tok * p = P(i, 1);
        const Tok * pp = P(i, 2);
        const Tok * n = Nx(i, 1);
        const u32 pl = p ? p->lower : u32();
        bool plural_s = ends_with(l, U"s") && !ends_with(l, U"ss");
        bool ed = ends_with(l, U"ed");
        bool ing = ends_with(l, U"ing");
        if (ed || ing) continue;  // participles keep their pass-2 tags
        // evidence
        bool det_before = p && (p->tag == "DT" || p->tag == "PRP$" || p->tag == "POS" || p->tag == "CD" ||
                                p->tag == "JJ" || p->tag == "JJS" || p->tag == "JJR" || p->tag == "PDT" ||
                                (p->tag == "IN" && pl != U"that") || p->tag == "WP$" || p->tag == "``");
        bool of_after = n && n->lower == U"of";
        bool modal_before = false;
        {
            // MD / TO / do, possibly across adverbs: "could no longer conduct"
            for (size_t b = 1; b <= 3 && b <= i; b++) {
                const Tok & q = t[i - b];
                if (q.tag == "MD" || q.tag == "TO" || is_do(q.lower) || (b == 1 && q.lower == U"please")) {
                    modal_before = true;
                    break;
                }
                bool adv = q.tag == "RB" || q.lower == U"no" || q.lower == U"longer" || q.lower == U"not" ||
                           q.lower == U"n't" || q.lower == U"never" || q.lower == U"also" || q.lower == U"still";
                if (!adv) break;
            }
        }
        bool obj_before = p && (pl == U"me" || pl == U"him" || pl == U"us" || pl == U"them" ||
                                ((pl == U"'s" || pl == U"’s") && pp && pp->lower == U"let"));
        bool subj_before = p && ((p->tag == "PRP" && subject_pronoun(pl)) || p->tag == "WDT" || p->tag == "WP" ||
                                 (pl == U"that" && pp && is_noun_tag(pp->tag)));
        bool np_before = false;  // a singular noun phrase opened by a determiner: "the static friction"
        if (p && is_noun_tag(p->tag) && p->tag != "NNS" && p->tag != "NNPS") {
            size_t j = i - 1;
            while (j > 0 && (t[j - 1].tag == "JJ" || t[j - 1].tag == "NN" || t[j - 1].tag == "NNP" || t[j - 1].tag == "VBN")) j--;
            np_before = j > 0 && (t[j - 1].tag == "DT" || t[j - 1].tag == "PRP$" || t[j - 1].tag == "POS");
        }
        bool noun_subj_before = p && (np_before || p->tag == "-RRB-" ||
                                      ((p->tag == "NNP" || p->tag == "NNPS") && !(i >= 2 && t[i - 2].tag == "IN")));
        bool clause_end_strict = !n || n->text == U"." || n->text == U"!" || n->text == U"?" || n->text == U";";
        bool imperative = k.sent_start && n && (n->tag == "DT" || n->tag == "PRP$" || (n->tag == "PRP" && n->lower != U"it" && n->lower != U"i"));
        bool verb;
        if (plural_s) {
            bool next_ok = clause_end_strict || n->tag == "DT" || n->tag == "PRP$" || n->tag == "PRP" || n->tag == "CD" ||
                           n->tag == "TO" || n->tag == "RB" || n->lower == U"that" || n->tag == "NNP" || n->tag == "CC" ||
                           (n->tag == "IN" && n->lower != U"of" && (np_before || (p && p->sent_start)));
            bool plural_subj = p && p->tag == "NNS" && pp && (pp->tag == "DT" || pp->tag == "PRP$") && ends_clause(n);
            bool vbz_ctx = (p && p->tag == "PRP" && third_person(pl)) || (p && (p->tag == "WDT" || p->tag == "WP")) ||
                           (p && p->tag == "RB" && pp && (is_noun_tag(pp->tag) || pp->tag == "-RRB-")) ||
                           (p && p->tag == "CC" && n && (n->tag == "DT" || n->tag == "PRP$")) ||
                           (p && (is_noun_tag(p->tag) && p->tag != "NNS") && n && (n->tag == "DT" || n->tag == "PRP$")) ||
                           ((noun_subj_before || plural_subj) && !of_after && next_ok);
            if (pr.noun_variant && !pr.verb_variant) verb = !det_before;  // DEFAULT reads as a verb
            else verb = vbz_ctx && !det_before;
            k.tag = verb ? "VBZ" : "NNS";
            continue;
        }
        if (pr.noun_variant && !pr.verb_variant) {
            // DEFAULT is the verb reading (use, close, combine, convert, ...)
            bool noun_ctx = det_before || of_after || (p && (p->tag == "NN" || p->tag == "NNP"));
            verb = !noun_ctx || modal_before || subj_before;
            if (!verb) k.tag = "NN";
            else if (k.tag == "NN" || k.tag == "NNS") k.tag = modal_before ? "VB" : "VBP";
            continue;
        }
        if (pr.verb_variant && pr.noun_variant) {
            // DEFAULT is the adjective (close): noun only as "the close (of)"
            verb = modal_before || subj_before || imperative || obj_before;
            if (verb) k.tag = (p && (p->tag == "MD" || p->tag == "TO")) ? "VB" : "VBP";
            else if (p && (p->tag == "DT" || p->tag == "PRP$") && (of_after || ends_clause(n))) k.tag = "NN";
            else k.tag = "JJ";
            continue;
        }
        if (pr.verb_variant) {
            bool adv_subj = p && p->tag == "RB" && pp && (pp->tag == "NNS" || (pp->tag == "PRP" && subject_pronoun(pp->lower)));
            verb = modal_before || subj_before || imperative || obj_before || adv_subj;
            if (verb) k.tag = (modal_before || obj_before || imperative) ? "VB" : "VBP";
            else if (k.tag == "VB" || k.tag == "VBP") k.tag = "NN";
            continue;
        }
        if (pr.adj_variant) {
            bool degree = p && (is_be(pl) || pl == U"very" || pl == U"more" || pl == U"most" || pl == U"so" || pl == U"too");
            bool before_noun = n && (n->tag == "NN" || n->tag == "NNS" || n->tag == "JJ" || n->tag == "NNP") && l != U"bass" &&
                               !(p && (p->tag == "HYPH" || p->tag == "CD"));
            if (degree || before_noun) k.tag = "JJ";
            else if (k.tag == "JJ") k.tag = "NN";
            continue;
        }
        if (pr.adv_variant) {
            bool attributive = p && (p->tag == "DT" || p->tag == "PRP$" || p->tag == "POS");
            k.tag = (!attributive && (ends_clause(n) || (n && is_verb_tag(n->tag)))) ? "RB" : "JJ";
            continue;
        }
    }

    // pass 4: function words whose reading depends on the tag
    for (size_t i = 0; i < N; i++) {
        Tok & k = t[i];
        const Tok * p = P(i, 1);
        const Tok * n = Nx(i, 1);
        // look past a quote for the next word
        const Tok * nw = n;
        if (nw && is_quote(nw->text)) nw = Nx(i, 2);
        const u32 & l = k.lower;
        if (l == U"'s" || l == U"’s") {
            if (p && (is_noun_tag(p->tag) || p->tag == "PRP" || p->tag == "CD")) {
                bool pron = p->tag == "PRP" || p->lower == U"that" || p->lower == U"there" || p->lower == U"what" ||
                            p->lower == U"who" || p->lower == U"here";
                bool verbish = n && (n->tag == "VBN" || n->tag == "VBG" || n->tag == "DT" || n->tag == "RB" ||
                                     n->tag == "IN" || n->tag == "TO");
                k.tag = pron || verbish ? "VBZ" : "POS";
            }
            continue;
        }
        if (l == U"that" && !is_all_caps(k.text)) {
            static const std::unordered_set<std::u32string> time_nouns = {
                U"day", U"night", U"morning", U"evening", U"afternoon", U"year", U"week", U"month", U"time", U"season",
                U"summer", U"winter", U"spring", U"autumn", U"moment", U"point", U"era", U"decade", U"century", U"period"};
            // verbs and adjectives that take a "that" clause
            static const std::unordered_set<std::u32string> clause_takers = {
                U"so", U"such", U"now", U"except", U"given", U"provided", U"said", U"say", U"says", U"believe",
                U"believed", U"think", U"thought", U"know", U"knew", U"found", U"noted", U"stated", U"reported",
                U"claimed", U"suggest", U"suggested", U"argued", U"felt", U"feel", U"show", U"showed", U"shows",
                U"revealed", U"announced", U"realized", U"realised", U"decided", U"means", U"meant", U"ensure",
                U"hoped", U"wrote", U"told", U"added", U"explained", U"indicated", U"concluded", U"concludes",
                U"confirmed", U"insisted", U"recalled", U"discovered", U"learned", U"learnt", U"noticed", U"notices",
                U"observed", U"feared", U"assumed", U"agreed", U"admitted", U"maintained", U"predicted", U"warned",
                U"promised", U"proved", U"remembered", U"understood", U"determined", U"mentioned", U"recommended",
                U"commented", U"estimated", U"testified", U"requested", U"demanded", U"suggests", U"states", U"notes",
                U"argues", U"believes", U"claims", U"reports", U"thinks", U"knows", U"seems", U"seemed", U"denied",
                U"denies", U"stressed", U"emphasized", U"emphasised", U"pleased", U"aware", U"sure", U"clear",
                U"likely", U"possible", U"evident", U"apparent", U"true", U"fact", U"ruled", U"held", U"alleged"};
            const Tok * n2 = nw ? Nx(i, (size_t) (nw - n) + 2) : nullptr;
            bool prep_before = p && (p->tag == "IN" || p->tag == "TO") && !clause_takers.count(p->lower);
            bool dt;
            if (nw && (nw->lower == U"of" || time_nouns.count(nw->lower))) {
                dt = true;  // "that of", "that year"
            } else if (p && is_noun_tag(p->tag) && nw && (nw->lower == U"'s" || is_verb_tag(nw->tag))) {
                dt = false;  // relative: "a style that's", "the house that stood"
            } else if (n && (n->text == U":" || n->text == U";")) {
                dt = false;  // "concludes that:"
            } else if (n && n->text == U",") {
                dt = prep_before || !(p && (p->tag == "IN" || is_verb_tag(p->tag) || p->tag == "," || p->tag == "CC" || p->tag == "JJ"));
            } else if (!nw || !word_like(nw->text)) {
                dt = !(n && is_quote(n->text)) && !(p && p->tag == ",");
            } else if (nw->tag == "NNS" || nw->tag == "NNPS") {
                dt = false;
            } else if (nw->tag == "RB" && ends_clause(n2)) {
                dt = true;  // "say that again"
            } else if (p && is_be(p->lower) && (nw->tag == "JJ" || nw->tag == "VBN" || nw->tag == "RB" || nw->tag == "NN")) {
                dt = true;  // "was that devastated", "is that quality"
            } else if ((nw->lower == U"is" || nw->lower == U"was" || nw->lower == U"'s") && (!p || p->tag == "CC" || p->tag == ",")) {
                dt = true;  // "and that is"
            } else if (prep_before && (nw->tag == "IN" || nw->tag == "VBN" || nw->tag == "RB")) {
                dt = true;  // "like that for years", "on that used by"
            } else if (k.sent_start) {
                dt = !(nw->tag == "PRP" || nw->tag == "DT");
            } else if (nw->tag == "PRP" || nw->tag == "DT" || nw->tag == "PRP$" || nw->tag == "EX" || nw->tag == "NNP" ||
                       nw->tag == "MD" || is_verb_tag(nw->tag) || nw->tag == "IN" || nw->tag == "TO" || nw->tag == "RB" ||
                       nw->tag == "CD" || nw->tag == "WRB" || nw->tag == "``" || nw->tag == "''") {
                dt = nw->lower == U"'s";
            } else if (p && (is_noun_tag(p->tag) || p->tag == "," || p->tag == "JJ")) {
                dt = false;
            } else if (p && (p->tag == "IN" || is_verb_tag(p->tag) || p->tag == "CC" || p->tag == "TO")) {
                dt = !clause_takers.count(p->lower);
            } else {
                dt = false;
            }
            k.tag = dt ? "DT" : (p && is_noun_tag(p->tag) ? "WDT" : "IN");
        } else if (l == U"in" && !is_all_caps(k.text)) {
            const Tok * n2 = Nx(i, 2);
            if (!nw || !word_like(nw->text)) {
                bool final = !n || n->text == U"." || n->text == U"!" || n->text == U"?" || n->text == U"," || n->text == U";";
                k.tag = (final && !(p && p->tag == "CD")) ? "RP" : "IN";  // "plugged in." but "(16 in)" is inches
            } else if (nw != n) {
                k.tag = "IN";  // in " quoted words "
            } else if (n->tag == "CC" || (n->tag == "RB" && (ends_clause(n2) || (n2 && (n2->tag == "CC" || n2->tag == "IN"))))) {
                k.tag = "RB";
            } else if (n->lower == U"as" && p && is_verb_tag(p->tag)) {
                k.tag = "RP";  // "sworn in as"
            } else {
                k.tag = "IN";
            }
        } else if (l == U"by" && !is_all_caps(k.text)) {
            k.tag = (n && word_like(n->text)) ? "IN" : "RB";
        } else if (k.text == U"a") {
            k.tag = (nw && (word_like(nw->text) || is_currency_symbol(nw->text))) ? "DT" : "NN";
        } else if (k.text == U"A") {
            bool letter = p && (p->tag == "NNP" || p->tag == "NN") && !k.sent_start &&
                          (!n || !word_like(n->text) || !is_capitalized(n->text) || n->tag == "CC");
            if (!n || !word_like(n->text)) letter = true;
            k.tag = letter ? "NNP" : "DT";
        } else if (k.text == U"I") {
            // Roman numeral: "World War I", "Division I National", "the I Scouting Group"
            bool roman = p && (p->tag == "NNP" || p->tag == "DT") &&
                         (!n || !word_like(n->text) || n->tag == "CC" || n->tag == "IN" || (n->tag == "NNP" && is_capitalized(n->text)));
            if (p && p->tag == "NNP" && p->sent_start && !(n && n->tag == "NNP")) roman = false;
            k.tag = roman ? "NNP" : "PRP";
        } else if (l == U"used") {
            const Tok * pb = p;
            for (size_t b = 2; pb && pb->tag == "RB" && b <= 3; b++) pb = P(i, b);  // "is commonly used"
            if (n && n->lower == U"to") k.tag = (pb && is_be(pb->lower)) ? "VBN" : "VBD";
            else if (p && (is_be(p->lower) || is_have(p->lower))) k.tag = "VBN";
            else if (p && (is_noun_tag(p->tag) || p->tag == "PRP")) k.tag = "VBD";
            else k.tag = "VBN";
        } else if (l == U"read" || l == U"wound" || l == U"reread") {
            const Tok * pp = P(i, 2);
            if (p && (p->tag == "MD" || p->tag == "TO" || is_do(p->lower) || p->lower == U"please" ||
                      ((p->lower == U"not" || p->lower == U"n't") && pp && (pp->tag == "MD" || is_do(pp->lower)))))
                k.tag = "VB";
            else if (p && (p->lower == U"me" || p->lower == U"him" || p->lower == U"us" || p->lower == U"them" ||
                           ((p->lower == U"'s" || p->lower == U"’s") && pp && pp->lower == U"let")))
                k.tag = "VB";
            else if (!coordinated_verb(i).empty()) k.tag = coordinated_verb(i);
            else if (p && (is_have(p->lower) || is_be(p->lower))) k.tag = "VBN";
            else if (p && (p->tag == "DT" || p->tag == "PRP$" || p->tag == "JJ" || p->tag == "IN")) k.tag = "NN";
            else if (k.sent_start) k.tag = "VB";
            else k.tag = "VBD";
        } else if (l == U"there" && k.closed) {
            k.tag = (n && (is_be(n->lower) || n->tag == "MD" || n->lower == U"seems" || n->lower == U"appears")) ? "EX" : "RB";
        } else if (l == U"her" && k.closed) {
            k.tag = (n && word_like(n->text) && (n->tag == "NN" || n->tag == "NNS" || n->tag == "JJ" || n->tag == "NNP" ||
                                                 n->tag == "VBG" || n->tag == "CD" || n->tag == "JJR" || n->tag == "JJS"))
                        ? "PRP$"
                        : "PRP";
        }
    }
    std::vector<std::string> out;
    out.reserve(N);
    for (auto & k : t) out.push_back(k.tag);
    return out;
}

}  // namespace misaki
