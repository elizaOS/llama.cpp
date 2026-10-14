// SPDX-License-Identifier: Apache-2.0
// misaki's out-of-vocabulary fallback: eSpeak NG (GPL-3.0-or-later, linked
// only when MISAKI_G2P_USE_ESPEAK is defined; see LICENSES/eSpeak-NG-NOTICE.txt)
// called the way phonemizer-fork 3.3.2 calls it for
// EspeakBackend('en-us', preserve_punctuation=True, with_stress=True, tie='^'),
// then misaki espeak.EspeakFallback's eSpeak-to-misaki (E2M) table.
#include "misaki-internal.h"

#include <mutex>
#include <stdexcept>

#if defined(MISAKI_G2P_USE_ESPEAK)
#    include <espeak-ng/speak_lib.h>
#endif

namespace misaki {
namespace {

const u32 kMarks = U";:,.!?¡¿—…\"«»“”(){}[]";  // phonemizer _DEFAULT_MARKS

bool is_mark(char32_t c) { return contains(kMarks, c); }

struct MarkIndex {
    u32 mark;
    char position;  // 'B', 'I', 'E' or 'A'
};

// re.finditer(r'(\s*[marks]+\s*)+', line)
std::vector<std::pair<size_t, size_t>> find_marks(const u32 & s) {
    std::vector<std::pair<size_t, size_t>> out;
    size_t p = 0;
    while (p < s.size()) {
        size_t q = p;
        while (q < s.size() && py_isspace(s[q])) q++;
        if (q >= s.size() || !is_mark(s[q])) {
            p++;
            continue;
        }
        size_t end = q;
        while (true) {
            size_t r = end;
            while (r < s.size() && py_isspace(s[r])) r++;
            if (r >= s.size() || !is_mark(s[r])) break;
            while (r < s.size() && is_mark(s[r])) r++;
            while (r < s.size() && py_isspace(s[r])) r++;
            end = r;
        }
        out.emplace_back(p, end);
        p = end;
    }
    return out;
}

std::vector<u32> py_str_split(const u32 & s, const u32 & sep) {
    std::vector<u32> out;
    size_t a = 0;
    while (true) {
        size_t b = s.find(sep, a);
        if (b == u32::npos) {
            out.push_back(s.substr(a));
            break;
        }
        out.push_back(s.substr(a, b - a));
        a = b + sep.size();
    }
    return out;
}

// Punctuation.preserve() for one line.
void preserve(const u32 & line_in, std::vector<u32> & chunks, std::vector<MarkIndex> & marks) {
    auto matches = find_marks(line_in);
    if (matches.empty()) {
        if (!line_in.empty()) chunks.push_back(line_in);
        return;
    }
    if (matches.size() == 1 && matches[0].first == 0 && matches[0].second == line_in.size()) {
        marks.push_back({line_in, 'A'});
        return;
    }
    for (size_t i = 0; i < matches.size(); i++) {
        u32 g = line_in.substr(matches[i].first, matches[i].second - matches[i].first);
        char pos = 'I';
        if (i == 0 && starts_with(line_in, g)) pos = 'B';
        else if (i + 1 == matches.size() && ends_with(line_in, g)) pos = 'E';
        marks.push_back({g, pos});
    }
    u32 line = line_in;
    std::vector<u32> preserved;
    for (auto & m : marks) {
        std::vector<u32> split = py_str_split(line, m.mark);
        u32 suffix;
        for (size_t k = 1; k < split.size(); k++) {
            if (k > 1) suffix += m.mark;
            suffix += split[k];
        }
        preserved.push_back(split[0]);
        line = suffix;
    }
    preserved.push_back(line);
    for (auto & p : preserved) if (!p.empty()) chunks.push_back(p);
}

// Punctuation.restore() with sep.word = ' ', strip = False, all marks on line 0.
std::vector<u32> restore(std::vector<u32> text, std::vector<MarkIndex> marks) {
    std::vector<u32> out;
    size_t pos = 0;
    const u32 sep = U" ";
    while (!text.empty() || !marks.empty()) {
        if (marks.empty()) {
            for (u32 line : text) {
                if (!ends_with(line, sep)) line += sep;
                out.push_back(line);
            }
            text.clear();
        } else if (text.empty()) {
            u32 all;
            for (auto & m : marks) all += m.mark;
            out.push_back(all);
            marks.clear();
        } else {
            MarkIndex cur = marks[0];
            if (pos == 0) {  // current_mark.index == pos
                marks.erase(marks.begin());
                const u32 & mark = cur.mark;
                if (ends_with(text[0], sep)) text[0] = text[0].substr(0, text[0].size() - sep.size());
                if (cur.position == 'B') {
                    text[0] = mark + text[0];
                } else if (cur.position == 'E') {
                    out.push_back(text[0] + mark + (ends_with(mark, sep) ? u32() : sep));
                    text.erase(text.begin());
                    pos++;
                } else if (cur.position == 'A') {
                    out.push_back(mark + (ends_with(mark, sep) ? u32() : sep));
                    pos++;
                } else {
                    if (text.size() == 1) {
                        text[0] = text[0] + mark;
                    } else {
                        u32 first = text[0];
                        text.erase(text.begin());
                        text[0] = first + mark + text[0];
                    }
                }
            } else {
                out.push_back(text[0]);
                text.erase(text.begin());
                pos++;
            }
        }
    }
    return out;
}

#if defined(MISAKI_G2P_USE_ESPEAK)
std::mutex & espeak_mutex() {
    static std::mutex m;
    return m;
}
bool g_espeak_ready = false;
std::string g_espeak_path;

// Caller holds espeak_mutex(). (Re)initialises libespeak-ng for data_dir.
bool ensure_espeak_locked(const std::string & data_dir, std::string & err) {
    if (g_espeak_ready && g_espeak_path == data_dir) return true;
    g_espeak_ready = false;
    int rate = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, data_dir.empty() ? nullptr : data_dir.c_str(), 0);
    if (rate <= 0) {
        err = "espeak_Initialize failed for data dir '" + data_dir + "'";
        return false;
    }
    // phonemizer selects the first voice whose language is en-us
    if (espeak_SetVoiceByName("gmw/en-US") != EE_OK) {
        err = "eSpeak NG voice gmw/en-US not found under '" + data_dir + "'";
        return false;
    }
    g_espeak_ready = true;
    g_espeak_path = data_dir;
    return true;
}
#endif

}  // namespace

bool espeak_linked() {
#if defined(MISAKI_G2P_USE_ESPEAK)
    return true;
#else
    return false;
#endif
}

u32 EspeakFallback::e2m(const u32 & in) {
    static const std::pair<const char32_t *, const char32_t *> kE2M[] = {
        // misaki EspeakFallback.E2M sorted by descending key length (stable)
        {U"ʔˌn̩", U"ʔn"},
        {U"ʔn̩", U"ʔn"}, {U"a^ɪ", U"I"}, {U"a^ʊ", U"W"}, {U"d^ʒ", U"ʤ"}, {U"e^ɪ", U"A"}, {U"t^ʃ", U"ʧ"},
        {U"ɔ^ɪ", U"Y"}, {U"ə^l", U"ᵊl"},
        {U"ʲo", U"jo"}, {U"ʲə", U"jə"},
        {U"e", U"A"}, {U"ʲ", U""}, {U"ɚ", U"əɹ"}, {U"r", U"ɹ"}, {U"x", U"k"}, {U"ç", U"k"}, {U"ɐ", U"ə"}, {U"ɬ", U"l"},
        {U"̃", U""},
    };
    u32 ps = in;
    for (auto & kv : kE2M) ps = replace_all(ps, kv.first, kv.second);
    // re.sub(r'(\S)̩', r'ᵊ\1', ps).replace(chr(809), '')
    {
        u32 o;
        for (size_t i = 0; i < ps.size(); i++) {
            if (i + 1 < ps.size() && ps[i + 1] == 0x329 && !py_isspace(ps[i])) {
                o += U'ᵊ';
                o += ps[i];
                i++;
            } else if (ps[i] != 0x329) {
                o += ps[i];
            }
        }
        ps = o;
    }
    // American English
    ps = replace_all(ps, U"o^ʊ", U"O");
    ps = replace_all(ps, U"ɜːɹ", U"ɜɹ");
    ps = replace_all(ps, U"ɜː", U"ɜɹ");
    ps = replace_all(ps, U"ɪə", U"iə");
    ps = replace_all(ps, U"ː", U"");
    ps = replace_all(ps, U"o", U"ɔ");  // for espeak < 1.52
    ps = replace_all(ps, U"ɾ", U"T");
    ps = replace_all(ps, U"ʔ", U"t");
    return replace_all(ps, U"^", U"");
}

std::unique_ptr<EspeakFallback> EspeakFallback::open(const std::string & data_dir, std::string & err) {
#if defined(MISAKI_G2P_USE_ESPEAK)
    std::lock_guard<std::mutex> lock(espeak_mutex());
    // espeak-ng keeps its data path in a 160-byte buffer (N_PATH_HOME) and
    // fails on longer ones with a misleading "No such file" error.
    if (data_dir.size() + sizeof("/espeak-ng-data/phontab") > 160) {
        err = "eSpeak NG data path is too long for libespeak-ng (max ~135 bytes): " + data_dir;
        return nullptr;
    }
    if (!ensure_espeak_locked(data_dir, err)) return nullptr;
    return std::unique_ptr<EspeakFallback>(new EspeakFallback(data_dir));
#else
    (void) data_dir;
    err = "misaki_g2p was built without libespeak-ng (MISAKI_G2P_ESPEAK=OFF)";
    return nullptr;
#endif
}

EspeakFallback::~EspeakFallback() = default;

u32 EspeakFallback::phonemize_line(const u32 & text) {
#if defined(MISAKI_G2P_USE_ESPEAK)
    std::lock_guard<std::mutex> lock(espeak_mutex());
    std::string init_err;
    if (!ensure_espeak_locked(data_dir_, init_err)) throw std::runtime_error(init_err);
    std::string utf8 = to_utf8(text);
    const void * ptr = utf8.c_str();
    // EspeakWrapper.text_to_phonemes(text, tie=True)
    const int mode = 0x02 | (0x01 << 7) | (0x0361 << 8);
    u32 joined;
    bool first = true;
    while (ptr != nullptr) {
        const char * ph = espeak_TextToPhonemes(&ptr, espeakCHARS_UTF8, mode);
        if (ph && *ph) {
            if (!first) joined += U" ";
            joined += from_utf8(ph, std::char_traits<char>::length(ph));
            first = false;
        }
    }
    // EspeakBackend._postprocess_line
    u32 line = replace_all(py_strip(joined), U"\n", U" ");
    line = replace_all(line, U"  ", U" ");  // str.replace is one left-to-right pass
    {
        u32 o;  // re.sub(r'_+', '_') then re.sub(r'_ ', ' ')
        for (char32_t c : line) {
            if (c == '_' && !o.empty() && o.back() == '_') continue;
            o += c;
        }
        line = replace_all(o, U"_ ", U" ");
    }
    if (line.empty()) return u32();
    u32 out;
    size_t a = 0;
    for (size_t i = 0; i <= line.size(); i++) {
        if (i == line.size() || line[i] == ' ') {
            u32 word = py_strip(line.substr(a, i - a));
            out += replace_all(word, U"͡", U"^") + U" ";
            a = i + 1;
        }
    }
    return out;
#else
    (void) text;
    return u32();
#endif
}

std::optional<u32> EspeakFallback::operator()(const u32 & text) {
    std::vector<u32> chunks;
    std::vector<MarkIndex> marks;
    preserve(text, chunks, marks);
    std::vector<u32> phonemized;
    for (auto & c : chunks) phonemized.push_back(phonemize_line(c));
    std::vector<u32> restored = restore(phonemized, marks);
    if (restored.empty()) return std::nullopt;
    u32 ps = py_strip(restored[0]);
    ps = e2m(ps);
    // EspeakFallback returns (None, None) only when phonemizer returns no line
    return ps;
}

}  // namespace misaki
