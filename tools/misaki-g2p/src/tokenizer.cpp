// SPDX-License-Identifier: Apache-2.0
// Port of the spaCy 3.8 English tokenizer (spacy/tokenizer.pyx with the
// en prefix/suffix/infix rules, URL_MATCH and tokenizer exceptions; spaCy is
// MIT, LICENSES/spaCy-MIT.txt) plus misaki's subtokenize regex.
#include "misaki-internal.h"

#include <algorithm>
#include <map>
#include <mutex>
#include <set>
#include <unordered_map>

namespace misaki {
namespace {

#include "spacy-en-rules.inc"

struct Rules {
    std::unordered_map<u32, std::vector<u32>> specials;
    // specials that the affix rules would split, matched as token sequences
    // afterwards (Tokenizer._special_matcher / _apply_special_cases)
    std::map<std::vector<u32>, u32> phrase;
    size_t max_phrase = 0;
};

const u32 kConcatQuotes = U"'\"”“`‘´’‚,„»«「」『』（）〔〕【】《》〈〉〈〉⟦⟧";

// LIST_PUNCT, LIST_ELLIPSES, LIST_QUOTES, LIST_CURRENCY in rule order (the
// ellipsis rule \.\.+ is handled by the matchers below).
const std::vector<u32> & list_punct() {
    static const std::vector<u32> v = {U"…", U"……", U",", U":", U";", U"!", U"?", U"¿", U"؟", U"¡",
                                       U"(", U")", U"[", U"]", U"{", U"}", U"<", U">", U"_", U"#", U"*", U"&", U"。",
                                       U"？", U"！", U"，", U"、", U"；", U"：", U"～", U"·",
                                       U"।", U"،", U"۔", U"؛", U"٪"};
    return v;
}
const std::vector<u32> & list_quotes() {
    static const std::vector<u32> v = {U"'", U"\"", U"”", U"“", U"`", U"‘", U"´", U"’", U"‚",
                                       U",", U"„", U"»", U"«", U"「", U"」", U"『", U"』",
                                       U"（", U"）", U"〔", U"〕", U"【", U"】", U"《", U"》",
                                       U"〈", U"〉", U"〈", U"〉", U"⟦", U"⟧"};
    return v;
}
const std::vector<u32> & list_currency() {
    static const std::vector<u32> v = {
        U"$", U"£", U"€", U"¥", U"฿", U"US$", U"C$", U"A$", U"₽", U"﷼", U"₴",
        U"₠", U"₡", U"₢", U"₣", U"₤", U"₥", U"₦", U"₧", U"₨", U"₩",
        U"₪", U"₫", U"€", U"₭", U"₮", U"₯", U"₰", U"₱", U"₲", U"₳",
        U"₴", U"₵", U"₶", U"₷", U"₸", U"₹", U"₺", U"₻", U"₼", U"₽",
        U"₾", U"₿"};
    return v;
}
const std::vector<u32> & list_units() {
    static std::vector<u32> v;
    static std::once_flag once;
    std::call_once(once, [] {
        const char * units =
            "km|km²|km³|m|m²|m³|dm|dm²|dm³|cm|cm²|cm³|mm|mm²|mm³|ha|µm|nm|yd|in|ft|"
            "kg|g|mg|µg|t|lb|oz|m/s|km/h|kmh|mph|hPa|Pa|mbar|mb|MB|kb|KB|gb|GB|tb|TB|T|G|M|K|%|"
            "км|км²|км³|м|м²|м³|дм|дм²|"
            "дм³|см|см²|см³|мм|мм²|мм³|"
            "нм|кг|г|мг|м/с|км/ч|кПа|Па|"
            "мбар|Кб|КБ|кб|Мб|МБ|мб|Гб|"
            "ГБ|гб|Тб|ТБ|тбكم|كم²|كم³|"
            "م|م²|م³|سم|سم²|سم³|مم|مم²|"
            "مم³|كم|غرام|جرام|جم|كغ|"
            "ملغ|كوب|اكواب";
        u32 all = U(units);
        size_t a = 0;
        for (size_t i = 0; i <= all.size(); i++) {
            if (i == all.size() || all[i] == U'|') {
                v.push_back(all.substr(a, i - a));
                a = i + 1;
            }
        }
    });
    return v;
}

bool is_digit09(char32_t c) { return c >= '0' && c <= '9'; }
bool is_nd(char32_t c) { return c < 0x80 ? is_digit09(c) : cat_Nd(c); }
bool is_space(char32_t c) { return py_isspace(c); }

// Length of the prefix the English prefix rules split off, 0 if none.
size_t find_prefix(const u32 & s) {
    if (s.empty()) return 0;
    auto lit = [&](const u32 & p) { return starts_with(s, p) ? p.size() : 0; };
    for (const char32_t * p : {U"§", U"%", U"=", U"—", U"–"}) {
        if (size_t n = lit(p)) return n;
    }
    if (s[0] == '+' && !(s.size() > 1 && is_digit09(s[1]))) return 1;
    for (const u32 & p : list_punct()) if (size_t n = lit(p)) return n;
    if (s.size() >= 2 && s[0] == '.' && s[1] == '.') {
        size_t n = 2;
        while (n < s.size() && s[n] == '.') n++;
        return n;
    }
    if (s[0] == U'…') return 1;
    for (const u32 & p : list_quotes()) if (size_t n = lit(p)) return n;
    for (const u32 & p : list_currency()) if (size_t n = lit(p)) return n;
    if (spacy_icon(s[0])) return 1;
    return 0;
}

// Does one suffix alternative match s[start:] exactly?
bool suffix_alt_at(const u32 & s, size_t start) {
    const u32 tail = s.substr(start);
    char32_t before = start > 0 ? s[start - 1] : 0;
    for (const u32 & p : list_punct()) if (tail == p) return true;
    if (tail.size() >= 2 && std::all_of(tail.begin(), tail.end(), [](char32_t c) { return c == '.'; })) return true;
    if (tail == U"…") return true;
    for (const u32 & p : list_quotes()) if (tail == p) return true;
    if (tail.size() == 1 && spacy_icon(tail[0])) return true;
    for (const char32_t * p : {U"'s", U"'S", U"’s", U"’S", U"—", U"–"}) if (tail == p) return true;
    if (tail == U"+" && start > 0 && is_digit09(before)) return true;
    if (tail == U"." && start >= 2 && s[start - 2] == U'°' && contains(U"FfCcKk", before)) return true;
    if (start > 0 && is_digit09(before)) {
        for (const u32 & p : list_currency()) if (tail == p) return true;
        for (const u32 & p : list_units()) if (tail == p) return true;
    }
    if (tail == U"." && start > 0 && spacy_suffix_dot_pre(before)) return true;
    if (tail == U"." && start >= 2 && spacy_alpha_upper(before) && spacy_alpha_upper(s[start - 2])) return true;
    return false;
}

size_t find_suffix(const u32 & s) {
    // re.search semantics: the leftmost start where an alternative matches
    // through the end. Every alternative is a literal of at most 6 code points
    // except the dot run (\.\.+$), so only the last 8 positions and the start
    // of the trailing dot run can match.
    const size_t n = s.size();
    size_t dots = n;
    while (dots > 0 && s[dots - 1] == '.') dots--;
    size_t best = (n - dots >= 2) ? dots : n;
    for (size_t start = n > 8 ? n - 8 : 0; start < best; start++) {
        if (suffix_alt_at(s, start)) return n - start;
    }
    return n - best;
}

// Infix matches (start, end) like re.finditer over the English infix rules.
std::vector<std::pair<size_t, size_t>> find_infix(const u32 & s) {
    std::vector<std::pair<size_t, size_t>> out;
    size_t p = 0;
    const size_t n = s.size();
    auto alpha = [&](size_t i) { return i < n && spacy_alpha(s[i]); };
    while (p < n) {
        size_t len = 0;
        char32_t c = s[p];
        char32_t prev = p > 0 ? s[p - 1] : 0;
        if (c == '.' && p + 1 < n && s[p + 1] == '.') {
            len = 2;
            while (p + len < n && s[p + len] == '.') len++;
        } else if (c == U'…') {
            len = 1;
        } else if (spacy_icon(c)) {
            len = 1;
        } else if (p > 0 && is_digit09(prev) && contains(U"+-*^", c) && p + 1 < n &&
                   (is_digit09(s[p + 1]) || s[p + 1] == '-')) {
            len = 1;
        } else if (c == '.' && p > 0 && (spacy_alpha_lower(prev) || contains(kConcatQuotes, prev)) && p + 1 < n &&
                   (spacy_alpha_upper(s[p + 1]) || contains(kConcatQuotes, s[p + 1]))) {
            len = 1;
        } else if (c == ',' && p > 0 && spacy_alpha(prev) && alpha(p + 1)) {
            len = 1;
        } else if (p > 0 && (spacy_alpha(prev) || is_digit09(prev))) {
            // (?:-|–|—|--|---|——|~)(?=[ALPHA]) tried in this order
            for (const char32_t * h : {U"-", U"–", U"—", U"--", U"---", U"——", U"~"}) {
                u32 hs(h);
                if (s.compare(p, hs.size(), hs) == 0 && alpha(p + hs.size())) {
                    len = hs.size();
                    break;
                }
            }
            if (!len && contains(U":<>=/", c) && alpha(p + 1)) len = 1;
        }
        if (len) {
            out.emplace_back(p, p + len);
            p += len;
        } else {
            p++;
        }
    }
    return out;
}

// --- URL_MATCH (spacy/lang/tokenizer_exceptions.py), hand-compiled
bool url_word_char(char32_t c) { return c == '_' || py_isalpha(c) || py_digit_value(c) >= 0 || cat_L(c) || cat_Nd(c); }
bool url_label_edge(char32_t c) { return (c < 0x80 && (std::isalnum((int) c) != 0)) || (c >= 0xA1 && c <= 0xFFFF); }
bool url_label_mid(char32_t c) { return url_label_edge(c) || c == '_' || c == '-'; }

bool url_tail(const u32 & s, size_t p) {
    // (?::\d{2,5})?(?:[/?#]\S*)?$
    size_t n = s.size();
    auto path_ok = [&](size_t q) {
        if (q == n) return true;
        if (!contains(U"/?#", s[q])) return false;
        for (size_t i = q + 1; i < n; i++) if (is_space(s[i])) return false;
        return true;
    };
    if (path_ok(p)) return true;
    if (p < n && s[p] == ':') {
        size_t q = p + 1, d = 0;
        while (q < n && is_nd(s[q]) && d < 5) { q++; d++; }
        // \d{2,5} is greedy but may back off; try each length
        for (size_t k = d; k >= 2; k--) {
            if (path_ok(p + 1 + k)) return true;
        }
    }
    return false;
}

bool url_host(const u32 & s, size_t p) {
    size_t n = s.size();
    // IPv4 public address
    {
        size_t q = p;
        int oct[4];
        int k = 0;
        for (; k < 4; k++) {
            size_t a = q;
            while (q < n && is_digit09(s[q]) && q - a < 3) q++;
            if (q == a) break;
            oct[k] = std::stoi(to_utf8(s.substr(a, q - a)));
            if (k < 3) {
                if (q < n && s[q] == '.') q++;
                else break;
            }
        }
        if (k == 4) {
            bool ok = oct[0] >= 1 && oct[0] <= 223 && oct[0] != 10 && oct[0] != 127 &&
                      !(oct[0] == 169 && oct[1] == 254) && !(oct[0] == 192 && oct[1] == 168) &&
                      !(oct[0] == 172 && oct[1] >= 16 && oct[1] <= 31) && oct[1] <= 255 && oct[2] <= 255 &&
                      oct[3] >= 1 && oct[3] <= 254;
            if (ok && url_tail(s, q)) return true;
        }
    }
    // (label .)+ tld
    size_t q = p;
    while (true) {
        size_t dot = s.find('.', q);
        if (dot == u32::npos) return false;
        size_t len = dot - q;
        if (len < 1 || len > 64 || !url_label_edge(s[q]) || !url_label_edge(s[dot - 1])) return false;
        for (size_t i = q + 1; i + 1 < dot; i++) if (!url_label_mid(s[i])) return false;
        q = dot + 1;
        size_t t = q;
        while (t < n && spacy_alpha_lower(s[t]) && t - q < 63) t++;
        for (size_t e = t; e >= q + 2; e--) {
            if (url_tail(s, e)) return true;
        }
    }
}

bool url_match(const u32 & s) {
    size_t n = s.size();
    if (n > 2048) return false;  // keeps the '@' scan bounded
    // scheme
    std::vector<size_t> starts;
    {
        size_t sep = s.find(U"://");
        if (sep != u32::npos && sep >= 2) {
            bool ok = true;
            for (size_t i = 0; i < sep; i++) {
                if (!(url_word_char(s[i]) || s[i] == '+' || s[i] == '-' || s[i] == '.')) { ok = false; break; }
            }
            if (ok) starts.push_back(sep + 3);
        }
        starts.push_back(0);
    }
    for (size_t st : starts) {
        // userinfo: \S+(?::\S*)?@ -- any '@' after at least one non-space char
        std::vector<size_t> hosts;
        for (size_t i = n; i-- > st + 1;) {
            if (s[i] == '@') {
                bool ok = true;
                for (size_t k = st; k < i; k++) if (is_space(s[k])) { ok = false; break; }
                if (ok) hosts.push_back(i + 1);
            }
        }
        hosts.push_back(st);
        for (size_t h : hosts) if (url_host(s, h)) return true;
    }
    return false;
}

const Rules & rules();

struct Piece {
    u32 text;
    bool space = false;
};

void split_affixes(const Rules & r, u32 & s, std::vector<u32> & prefixes, std::vector<u32> & suffixes, bool specials) {
    size_t last = 0;
    while (!s.empty() && s.size() != last) {
        if (specials && r.specials.count(s)) break;
        last = s.size();
        size_t pre = find_prefix(s);
        u32 prefix, minus_pre;
        if (pre) {
            prefix = s.substr(0, pre);
            minus_pre = s.substr(pre);
            if (!minus_pre.empty() && specials && r.specials.count(minus_pre)) {
                s = minus_pre;
                prefixes.push_back(prefix);
                break;
            }
        }
        size_t suf = find_suffix(s.substr(pre));
        u32 suffix, minus_suf;
        if (suf) {
            suffix = s.substr(s.size() - suf);
            minus_suf = s.substr(0, s.size() - suf);
            if (!minus_suf.empty() && specials && r.specials.count(minus_suf)) {
                s = minus_suf;
                suffixes.push_back(suffix);
                break;
            }
        }
        if (pre && suf && pre + suf <= s.size()) {
            s = s.substr(pre, s.size() - pre - suf);
            prefixes.push_back(prefix);
            suffixes.push_back(suffix);
        } else if (pre) {
            s = minus_pre;
            prefixes.push_back(prefix);
        } else if (suf) {
            s = minus_suf;
            suffixes.push_back(suffix);
        }
    }
}

void tokenize_span(const Rules & r, const u32 & span, std::vector<Piece> & out, bool specials) {
    if (specials) {
        auto it = r.specials.find(span);
        if (it != r.specials.end()) {
            for (const u32 & t : it->second) out.push_back({t, false});
            return;
        }
    }
    u32 s = span;
    std::vector<u32> prefixes, suffixes;
    split_affixes(r, s, prefixes, suffixes, specials);
    for (const u32 & p : prefixes) out.push_back({p, false});
    if (!s.empty()) {
        auto it = specials ? r.specials.find(s) : r.specials.end();
        if (it != r.specials.end()) {
            for (const u32 & t : it->second) out.push_back({t, false});
        } else if (url_match(s)) {
            out.push_back({s, false});
        } else {
            auto matches = find_infix(s);
            if (matches.empty()) {
                out.push_back({s, false});
            } else {
                size_t start = 0;
                const size_t start_before = 0;
                for (auto & m : matches) {
                    if (m.first == start_before) continue;
                    if (m.first != start) out.push_back({s.substr(start, m.first - start), false});
                    if (m.first != m.second) out.push_back({s.substr(m.first, m.second - m.first), false});
                    start = m.second;
                }
                if (start < s.size()) out.push_back({s.substr(start), false});
            }
        }
    }
    for (auto it = suffixes.rbegin(); it != suffixes.rend(); ++it) out.push_back({*it, false});
}

std::vector<Piece> tokenize_affixes(const Rules & r, const u32 & text, bool specials) {
    std::vector<Piece> doc;
    if (text.empty()) return doc;
    bool in_ws = is_space(text[0]);
    size_t start = 0;
    for (size_t i = 0; i < text.size(); i++) {
        char32_t uc = text[i];
        if (is_space(uc) != in_ws) {
            if (start < i) tokenize_span(r, text.substr(start, i - start), doc, specials);
            if (uc == ' ') {
                if (!doc.empty()) doc.back().space = true;
                start = i + 1;
            } else {
                start = i;
            }
            in_ws = !in_ws;
        }
    }
    if (start < text.size()) {
        tokenize_span(r, text.substr(start), doc, specials);
        if (!doc.empty()) doc.back().space = text.back() == ' ' && !in_ws;
    }
    return doc;
}

const Rules & rules() {
    static Rules r;
    static std::once_flag once;
    std::call_once(once, [] {
        for (const char * e : kSpacyRules) {
            u32 all = U(e);
            std::vector<u32> parts;
            size_t a = 0;
            for (size_t i = 0; i <= all.size(); i++) {
                if (i == all.size() || all[i] == 0x1F) {
                    parts.push_back(all.substr(a, i - a));
                    a = i + 1;
                }
            }
            u32 key = parts[0];
            parts.erase(parts.begin());
            r.specials[key] = parts;
        }
        // Tokenizer.add_special_case: with faster_heuristics, a special is
        // also matched as a token sequence when affix rules would split it.
        for (auto & kv : r.specials) {
            const u32 & k = kv.first;
            if (find_prefix(k) || !find_infix(k).empty() || find_suffix(k) || contains(k, U' ')) {
                std::vector<Piece> pieces = tokenize_affixes(r, k, false);
                std::vector<u32> key;
                for (auto & p : pieces) key.push_back(p.text);
                if (key.empty()) continue;
                r.max_phrase = std::max(r.max_phrase, key.size());
                r.phrase[key] = k;
            }
        }
    });
    return r;
}

// Tokenizer._apply_special_cases
void apply_special_cases(const Rules & r, std::vector<Piece> & doc) {
    struct Match { size_t start, end; u32 key; };
    std::vector<Match> matches;
    for (size_t i = 0; i < doc.size(); i++) {
        std::vector<u32> seq;
        for (size_t j = i; j < doc.size() && j - i < r.max_phrase; j++) {
            seq.push_back(doc[j].text);
            auto it = r.phrase.find(seq);
            if (it != r.phrase.end()) matches.push_back({i, j + 1, it->second});
        }
    }
    if (matches.empty()) return;
    // _filter_special_spans: longest first (then later start first)
    std::sort(matches.begin(), matches.end(), [](const Match & a, const Match & b) {
        size_t la = a.end - a.start, lb = b.end - b.start;
        return la != lb ? la < lb : a.start < b.start;
    });
    std::set<size_t> seen;
    std::vector<Match> kept;
    for (size_t k = matches.size(); k-- > 0;) {
        const Match & m = matches[k];
        if (!seen.count(m.start) && !seen.count(m.end - 1)) kept.push_back(m);
        for (size_t t = m.start; t < m.end; t++) seen.insert(t);
    }
    std::sort(kept.begin(), kept.end(), [](const Match & a, const Match & b) { return a.start < b.start; });
    std::vector<Piece> out;
    size_t i = 0, k = 0;
    while (i < doc.size()) {
        if (k < kept.size() && kept[k].start == i) {
            const Match & m = kept[k++];
            // _retokenize_special_spans looks the span text up again, spaces
            // included, so a match across whitespace changes nothing
            u32 span_text;
            for (size_t q = m.start; q < m.end; q++) {
                span_text += doc[q].text;
                if (q + 1 < m.end && doc[q].space) span_text += U' ';
            }
            auto it = r.specials.find(span_text);
            if (it == r.specials.end()) {
                for (size_t q = m.start; q < m.end; q++) out.push_back(doc[q]);
                i = m.end;
                continue;
            }
            bool final_space = doc[m.end - 1].space;
            for (const u32 & t : it->second) out.push_back({t, false});
            out.back().space = final_space;
            i = m.end;
        } else {
            out.push_back(doc[i++]);
        }
    }
    doc.swap(out);
}

}  // namespace

std::vector<SpacyToken> spacy_tokenize(const u32 & text) {
    const Rules & r = rules();
    std::vector<Piece> doc = tokenize_affixes(r, text, true);
    apply_special_cases(r, doc);
    std::vector<SpacyToken> out;
    size_t pos = 0;
    for (auto & p : doc) {
        SpacyToken t;
        t.text = p.text;
        t.space_after = p.space;
        t.start = pos;
        pos += p.text.size() + (p.space ? 1 : 0);
        out.push_back(std::move(t));
    }
    return out;
}

// misaki SUBTOKEN_REGEX:
//   ^['‘’]+ | \p{Lu}(?=\p{Lu}\p{Ll}) | (?:^-)?(?:\d?[,.]?\d)+ | [-_]+ | ['‘’]{2,}
//   | \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu}) | \p{L}+(?:['‘’]\p{L})* | [^-_\p{L}'‘’\d] | ['‘’]+$
std::vector<u32> subtokenize(const u32 & w) {
    std::vector<u32> out;
    const size_t n = w.size();
    auto q = [&](size_t i) { return i < n && (w[i] == '\'' || w[i] == U'‘' || w[i] == U'’'); };
    auto L = [&](size_t i) { return i < n && cat_L(w[i]); };
    auto Lu = [&](size_t i) { return i < n && cat_Lu(w[i]); };
    auto Ll = [&](size_t i) { return i < n && cat_Ll(w[i]); };
    auto D = [&](size_t i) { return i < n && is_nd(w[i]); };
    auto sep = [&](size_t i) { return i < n && (w[i] == ',' || w[i] == '.'); };
    size_t p = 0;
    while (p < n) {
        size_t len = 0;
        // 1. ^['‘’]+
        if (p == 0 && q(0)) {
            while (q(len)) len++;
        }
        // 2. \p{Lu}(?=\p{Lu}\p{Ll})
        if (!len && Lu(p) && Lu(p + 1) && Ll(p + 2)) len = 1;
        // 3. (?:^-)?(?:\d?[,.]?\d)+
        if (!len) {
            size_t s = p;
            if (p == 0 && w[0] == '-') s = 1;
            size_t e = s;
            while (true) {
                size_t t = e;
                // try \d [,.] \d, \d \d, [,.] \d, \d  in regex priority order
                if (D(t) && sep(t + 1) && D(t + 2)) t += 3;
                else if (D(t) && D(t + 1)) t += 2;
                else if (D(t)) t += 1;  // \d? consumed, [,.]? absent, then \d needed: handled by D(t+1) above
                else if (sep(t) && D(t + 1)) t += 2;
                else break;
                // A lone digit only matches as "\d?" absent + "\d" present.
                e = t;
            }
            if (e > s) len = e - p;
        }
        // 4. [-_]+
        if (!len && (w[p] == '-' || w[p] == '_')) {
            while (p + len < n && (w[p + len] == '-' || w[p + len] == '_')) len++;
        }
        // 5. ['‘’]{2,}
        if (!len && q(p) && q(p + 1)) {
            while (q(p + len)) len++;
        }
        // 6. \p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})  (lazy: shortest first)
        if (!len) {
            // bounded at 64 letters / apostrophe pairs so pathological input
            // stays linear; natural words never come close
            for (size_t a = 0; !len && a < 64 && (a == 0 || L(p + a - 1)); a++) {
                size_t s = p + a;
                for (size_t b = 0; b < 64; b++) {
                    size_t t = s + 2 * b;
                    if (b > 0 && !(q(t - 2) && L(t - 1))) break;
                    if (Ll(t) && Lu(t + 1)) {
                        len = t + 1 - p;
                        break;
                    }
                }
                if (p + a >= n) break;
            }
        }
        // 7. \p{L}+(?:['‘’]\p{L})*
        if (!len && L(p)) {
            size_t t = p;
            while (L(t)) t++;
            while (q(t) && L(t + 1)) t += 2;
            len = t - p;
        }
        // 8. [^-_\p{L}'‘’\d]
        if (!len && !(w[p] == '-' || w[p] == '_' || L(p) || q(p) || D(p))) len = 1;
        // 9. ['‘’]+$
        if (!len && q(p)) {
            size_t t = p;
            while (q(t)) t++;
            if (t == n) len = t - p;
        }
        if (len) {
            out.push_back(w.substr(p, len));
            p += len;
        } else {
            p++;
        }
    }
    return out;
}

}  // namespace misaki
