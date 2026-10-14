// SPDX-License-Identifier: Apache-2.0
// English number words for the misaki port. misaki got these strings from
// num2words; this is an independent implementation of the same English
// conventions (short scale, "and" before the last two digits, "oh" in
// years), checked against num2words output by tests/compare_misaki.py.
#include "misaki-internal.h"

#include <string>
#include <vector>

namespace misaki {
namespace {

const char * kOnes[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                        "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                        "eighteen", "nineteen"};
const char * kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};
// 10^3 .. 10^39
const char * kScales[] = {"", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion",
                          "sextillion", "septillion", "octillion", "nonillion", "decillion", "undecillion",
                          "duodecillion"};
constexpr size_t kMaxGroups = sizeof(kScales) / sizeof(kScales[0]);

std::string strip_zeros(const std::string & d) {
    size_t i = 0;
    while (i + 1 < d.size() && d[i] == '0') i++;
    return d.substr(i);
}

std::string under_100(int n) {
    if (n < 20) return kOnes[n];
    std::string s = kTens[n / 10];
    if (n % 10) s += std::string("-") + kOnes[n % 10];
    return s;
}

std::string under_1000(int n) {
    if (n < 100) return under_100(n);
    std::string s = std::string(kOnes[n / 100]) + " hundred";
    if (n % 100) s += " and " + under_100(n % 100);
    return s;
}

std::string cardinal(const std::string & digits) {
    std::string d = strip_zeros(digits);
    if (d.empty() || d == "0") return "zero";
    std::vector<int> groups;  // least significant first
    for (long end = (long) d.size(); end > 0; end -= 3) {
        long start = end - 3 < 0 ? 0 : end - 3;
        groups.push_back(std::stoi(d.substr((size_t) start, (size_t) (end - start))));
    }
    std::string out;
    int last = groups[0];
    bool higher = false;
    for (size_t g = groups.size(); g-- > 1;) {
        if (!groups[g]) continue;
        if (!out.empty()) out += ", ";
        out += under_1000(groups[g]) + " " + kScales[g];
        higher = true;
    }
    if (last) {
        if (higher) out += last < 100 ? " and " : ", ";
        out += under_1000(last);
    }
    return out;
}

std::string ordinalize_word(const std::string & w) {
    static const char * irregular[][2] = {{"one", "first"}, {"two", "second"}, {"three", "third"},
                                          {"five", "fifth"}, {"eight", "eighth"}, {"nine", "ninth"},
                                          {"twelve", "twelfth"}};
    for (auto & p : irregular) if (w == p[0]) return p[1];
    if (!w.empty() && w.back() == 'y') return w.substr(0, w.size() - 1) + "ieth";
    return w + "th";
}

std::string ordinal(const std::string & digits) {
    std::string c = cardinal(digits);
    size_t cut = c.find_last_of(" -");
    if (cut == std::string::npos) return ordinalize_word(c);
    return c.substr(0, cut + 1) + ordinalize_word(c.substr(cut + 1));
}

}  // namespace

bool number_fits(const std::string & digits) { return strip_zeros(digits).size() <= 3 * kMaxGroups; }

u32 number_cardinal(const std::string & digits) { return U(cardinal(digits).c_str()); }
u32 number_ordinal(const std::string & digits) { return U(ordinal(digits).c_str()); }

u32 number_year(const std::string & digits) {
    std::string d = strip_zeros(digits);
    long long v = d.size() > 6 ? 1000000 : std::stoll(d);
    long long high = v / 100, low = v % 100;
    if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) return number_cardinal(d);
    std::string s = cardinal(std::to_string(high)) + " ";
    if (low == 0) s += "hundred";
    else if (low < 10) s += "oh-" + cardinal(std::to_string(low));
    else s += cardinal(std::to_string(low));
    return U(s.c_str());
}

u32 number_float(const std::string & text) {
    size_t dot = text.find('.');
    std::string ip = dot == std::string::npos ? text : text.substr(0, dot);
    std::string fp = dot == std::string::npos ? "" : text.substr(dot + 1);
    while (!fp.empty() && fp.back() == '0') fp.pop_back();
    std::string s = cardinal(ip.empty() ? "0" : ip);
    if (!fp.empty()) {
        s += " point";
        for (char c : fp) s += std::string(" ") + kOnes[c - '0'];
    }
    return U(s.c_str());
}

}  // namespace misaki
