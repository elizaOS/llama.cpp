// SPDX-License-Identifier: Apache-2.0
// Python string semantics and spaCy character classes from generated tables.
#include "misaki-internal.h"

#include <algorithm>

namespace misaki {
namespace {

struct CpRange { char32_t lo, hi; };
struct CpDigit { char32_t cp; int value; };
struct CpMap { char32_t from, to; };
struct CpMulti { char32_t from; const char32_t * to; };
struct CpCompose { char32_t a, b, c; };

#include "unicode-tables.inc"

template <size_t N>
bool in_ranges(const CpRange (&t)[N], char32_t c) {
    size_t lo = 0, hi = N;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (c < t[mid].lo) hi = mid;
        else if (c > t[mid].hi) lo = mid + 1;
        else return true;
    }
    return false;
}

template <size_t N>
const CpMap * find_map(const CpMap (&t)[N], char32_t c) {
    auto it = std::lower_bound(t, t + N, c, [](const CpMap & m, char32_t v) { return m.from < v; });
    return (it != t + N && it->from == c) ? it : nullptr;
}

template <size_t N>
const CpMulti * find_multi(const CpMulti (&t)[N], char32_t c) {
    auto it = std::lower_bound(t, t + N, c, [](const CpMulti & m, char32_t v) { return m.from < v; });
    return (it != t + N && it->from == c) ? it : nullptr;
}

int ccc(char32_t c) {
    const CpMap * m = find_map(kCombiningClass, c);
    return m ? (int) m->to : 0;
}

bool compose_pair(char32_t a, char32_t b, char32_t & out) {
    auto it = std::lower_bound(kCompose, kCompose + sizeof(kCompose) / sizeof(kCompose[0]), std::make_pair(a, b),
                               [](const CpCompose & m, const std::pair<char32_t, char32_t> & v) {
                                   return m.a < v.first || (m.a == v.first && m.b < v.second);
                               });
    if (it != kCompose + sizeof(kCompose) / sizeof(kCompose[0]) && it->a == a && it->b == b) {
        out = it->c;
        return true;
    }
    return false;
}

}  // namespace

std::string to_utf8(const u32 & s) {
    std::string out;
    out.reserve(s.size());
    for (char32_t c : s) {
        if (c < 0x80) {
            out += (char) c;
        } else if (c < 0x800) {
            out += (char) (0xC0 | (c >> 6));
            out += (char) (0x80 | (c & 0x3F));
        } else if (c < 0x10000) {
            out += (char) (0xE0 | (c >> 12));
            out += (char) (0x80 | ((c >> 6) & 0x3F));
            out += (char) (0x80 | (c & 0x3F));
        } else {
            out += (char) (0xF0 | (c >> 18));
            out += (char) (0x80 | ((c >> 12) & 0x3F));
            out += (char) (0x80 | ((c >> 6) & 0x3F));
            out += (char) (0x80 | (c & 0x3F));
        }
    }
    return out;
}

u32 from_utf8(const char * p, size_t n) {
    u32 out;
    out.reserve(n);
    const auto * s = (const unsigned char *) p;
    size_t i = 0;
    while (i < n) {
        unsigned char b = s[i];
        char32_t c;
        size_t len;
        if (b < 0x80) { c = b; len = 1; }
        else if ((b & 0xE0) == 0xC0) { c = b & 0x1F; len = 2; }
        else if ((b & 0xF0) == 0xE0) { c = b & 0x0F; len = 3; }
        else if ((b & 0xF8) == 0xF0) { c = b & 0x07; len = 4; }
        else { out += 0xFFFD; i++; continue; }
        if (i + len > n) { out += 0xFFFD; break; }
        bool ok = true;
        for (size_t k = 1; k < len; k++) {
            if ((s[i + k] & 0xC0) != 0x80) { ok = false; break; }
            c = (c << 6) | (s[i + k] & 0x3F);
        }
        if (!ok || (len == 2 && c < 0x80) || (len == 3 && c < 0x800) || (len == 4 && (c < 0x10000 || c > 0x10FFFF)) ||
            (c >= 0xD800 && c <= 0xDFFF)) {
            out += 0xFFFD;
            i++;
            continue;
        }
        out += c;
        i += len;
    }
    return out;
}

u32 U(const char * utf8) { return from_utf8(utf8, std::char_traits<char>::length(utf8)); }

bool py_isalpha(char32_t c) { return c < 0x80 ? ((c | 0x20) >= 'a' && (c | 0x20) <= 'z') : in_ranges(kPyAlpha, c); }
bool py_isspace(char32_t c) { return in_ranges(kPySpace, c); }

int py_digit_value(char32_t c) {
    if (c >= '0' && c <= '9') return (int) (c - '0');
    auto it = std::lower_bound(kPyDigit, kPyDigit + sizeof(kPyDigit) / sizeof(kPyDigit[0]), c,
                               [](const CpDigit & d, char32_t v) { return d.cp < v; });
    if (it != kPyDigit + sizeof(kPyDigit) / sizeof(kPyDigit[0]) && it->cp == c) return it->value;
    return -1;
}

bool py_str_isalpha(const u32 & s) {
    if (s.empty()) return false;
    for (char32_t c : s) if (!py_isalpha(c)) return false;
    return true;
}

bool py_str_isdigit(const u32 & s) {
    if (s.empty()) return false;
    for (char32_t c : s) if (py_digit_value(c) < 0) return false;
    return true;
}

static void append_case(u32 & out, char32_t c, bool lower) {
    if (c < 0x80) {
        if (lower && c >= 'A' && c <= 'Z') c += 32;
        else if (!lower && c >= 'a' && c <= 'z') c -= 32;
        out += c;
        return;
    }
    if (lower) {
        // U+03A3 final sigma is context-dependent in CPython; it is not a
        // letter misaki ever looks up, so the plain mapping is used.
        if (const CpMulti * m = find_multi(kPyLowerMulti, c)) { out += m->to; return; }
        if (const CpMap * m = find_map(kPyLowerOne, c)) { out += m->to; return; }
    } else {
        if (const CpMulti * m = find_multi(kPyUpperMulti, c)) { out += m->to; return; }
        if (const CpMap * m = find_map(kPyUpperOne, c)) { out += m->to; return; }
    }
    out += c;
}

u32 py_lower(const u32 & s) {
    u32 out;
    out.reserve(s.size());
    for (char32_t c : s) append_case(out, c, true);
    return out;
}

u32 py_upper(const u32 & s) {
    u32 out;
    out.reserve(s.size());
    for (char32_t c : s) append_case(out, c, false);
    return out;
}

u32 py_capitalize(const u32 & s) {
    if (s.empty()) return s;
    u32 out;
    char32_t c = s[0];
    if (c < 0x80) {
        out += (c >= 'a' && c <= 'z') ? c - 32 : c;
    } else if (const CpMap * m = find_map(kPyTitleOne, c)) {
        out += m->to;
    } else {
        append_case(out, c, false);
    }
    out += py_lower(s.substr(1));
    return out;
}

u32 py_strip(const u32 & s) {
    size_t a = 0, b = s.size();
    while (a < b && py_isspace(s[a])) a++;
    while (b > a && py_isspace(s[b - 1])) b--;
    return s.substr(a, b - a);
}

u32 nfkc(const u32 & s) {
    bool ascii = true;
    for (char32_t c : s) if (c >= 0x80) { ascii = false; break; }
    if (ascii) return s;
    // 1. per-code-point compatibility mapping (each mapping is already NFKC)
    u32 d;
    for (char32_t c : s) {
        if (const CpMulti * m = find_multi(kNfkc, c)) d += m->to;
        else d += c;
    }
    // 2. canonical ordering of combining marks
    for (size_t i = 0; i < d.size();) {
        if (ccc(d[i]) == 0) { i++; continue; }
        size_t j = i;
        while (j < d.size() && ccc(d[j]) != 0) j++;
        std::stable_sort(d.begin() + i, d.begin() + j, [](char32_t a, char32_t b) { return ccc(a) < ccc(b); });
        i = j;
    }
    // 3. canonical composition
    u32 out;
    size_t starter = u32::npos;
    int last_ccc = -1;
    for (char32_t c : d) {
        int cc = ccc(c);
        char32_t comp;
        if (starter != u32::npos && (last_ccc < cc || (last_ccc == 0 && out.size() == starter + 1)) &&
            compose_pair(out[starter], c, comp)) {
            out[starter] = comp;
            continue;
        }
        if (cc == 0) {
            starter = out.size();
            last_ccc = 0;
        } else {
            last_ccc = cc;
        }
        out += c;
    }
    return out;
}

bool cat_L(char32_t c) { return in_ranges(kCatL, c); }
bool cat_Lu(char32_t c) { return in_ranges(kCatLu, c); }
bool cat_Ll(char32_t c) { return in_ranges(kCatLl, c); }
bool cat_Nd(char32_t c) { return in_ranges(kCatNd, c); }

bool spacy_alpha(char32_t c) { return in_ranges(kSpacyAlpha, c); }
bool spacy_alpha_lower(char32_t c) { return in_ranges(kSpacyAlphaLower, c); }
bool spacy_alpha_upper(char32_t c) { return in_ranges(kSpacyAlphaUpper, c); }
bool spacy_icon(char32_t c) { return in_ranges(kSpacyIcons, c); }
bool spacy_suffix_dot_pre(char32_t c) { return in_ranges(kSpacySuffixDotPre, c); }

u32 replace_all(u32 s, const u32 & from, const u32 & to) {
    if (from.empty()) return s;
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != u32::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
    return s;
}

}  // namespace misaki
