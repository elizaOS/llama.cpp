// SPDX-License-Identifier: Apache-2.0
// C API of misaki_g2p.
#include "misaki-g2p.h"

#include "misaki-internal.h"

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <string>

struct misaki_g2p {
    std::unique_ptr<misaki::LexFile> lex;
    std::unique_ptr<misaki::EspeakFallback> espeak;
    std::unique_ptr<misaki::G2P> g2p;
};

namespace {

char * dup(const std::string & s) {
    char * p = (char *) std::malloc(s.size() + 1);
    if (p) std::memcpy(p, s.c_str(), s.size() + 1);
    return p;
}

void set_err(char ** err, const std::string & msg) {
    if (err) *err = dup(msg);
}

}  // namespace

namespace misaki {

// Paradee-8M v1.0 config.json "vocab" keys (HF revision 8f34b01e, config.json
// sha256 f24046974a3a8c747affefb45c7c504263a99d5081787908b16abe8f5ac94fcd).
const u32 & paradee_vocab() {
    static const u32 v =
        U";:,.!?—…\"()“” ̃ʣʥʦʨᵝꭧAIOQSTWYᵊ"
        U"abcdefhijklmnopqrstuvwxyzɑɐɒæβɔɕçɖðʤə"
        U"ɚɛɜɟɡɥɨɪʝɯɰŋɳɲɴø"
        U"ɸθœɹɾɻʁɽʂʃʈʧʊʋʌɣ"
        U"ɤχʎʒʔˈˌːʰʲ↓→↗↘ᵻ";
    return v;
}

bool in_paradee_vocab(char32_t c) { return contains(paradee_vocab(), c); }

}  // namespace misaki

extern "C" {

misaki_g2p * misaki_g2p_open(const char * lex_path, const char * espeak_data_dir_or_null, char ** err) {
    if (err) *err = nullptr;
    if (!lex_path) {
        set_err(err, "misaki_g2p_open: lex_path is NULL");
        return nullptr;
    }
    try {
        std::unique_ptr<misaki_g2p> g(new misaki_g2p());
        std::string e;
        g->lex = misaki::LexFile::open(lex_path, e);
        if (!g->lex) {
            set_err(err, e);
            return nullptr;
        }
        if (espeak_data_dir_or_null) {
            g->espeak = misaki::EspeakFallback::open(espeak_data_dir_or_null, e);
            if (!g->espeak) {
                set_err(err, e);
                return nullptr;
            }
        }
        g->g2p.reset(new misaki::G2P(*g->lex, g->espeak.get()));
        return g.release();
    } catch (const std::exception & ex) {
        set_err(err, std::string("misaki_g2p_open: ") + ex.what());
        return nullptr;
    }
}

int misaki_g2p_phonemize(misaki_g2p * g, const char * utf8, size_t len, char ** out_utf8, char ** err) {
    if (err) *err = nullptr;
    if (out_utf8) *out_utf8 = nullptr;
    if (!g || !out_utf8 || (!utf8 && len)) {
        set_err(err, "misaki_g2p_phonemize: bad arguments");
        return -1;
    }
    try {
        misaki::u32 text = misaki::from_utf8(utf8 ? utf8 : "", len);
        std::vector<misaki::u32> chunks = misaki::paradee_chunks(*g->g2p, text, MISAKI_G2P_MAX_CHUNK);
        misaki::u32 joined;
        for (size_t i = 0; i < chunks.size(); i++) {
            const misaki::u32 & c = chunks[i];
            if (c.size() > MISAKI_G2P_MAX_CHUNK) {
                set_err(err, "misaki_g2p_phonemize: chunk longer than 510 symbols");
                return -2;
            }
            for (char32_t ch : c) {
                if (!misaki::in_paradee_vocab(ch)) {
                    set_err(err, "misaki_g2p_phonemize: symbol U+" + std::to_string((unsigned) ch) +
                                     " is not in the Paradee vocabulary (chunk: " + misaki::to_utf8(c) + ")");
                    return -3;
                }
            }
            if (i) joined += U'\n';
            joined += c;
        }
        *out_utf8 = dup(misaki::to_utf8(joined));
        if (!*out_utf8) {
            set_err(err, "misaki_g2p_phonemize: out of memory");
            return -4;
        }
        return 0;
    } catch (const std::exception & ex) {
        set_err(err, std::string("misaki_g2p_phonemize: ") + ex.what());
        return -5;
    }
}

int misaki_g2p_fallback_kind(const misaki_g2p * g) {
    return (g && g->espeak) ? MISAKI_G2P_FALLBACK_ESPEAK : MISAKI_G2P_FALLBACK_SPELL;
}

int misaki_g2p_espeak_linked(void) { return misaki::espeak_linked() ? 1 : 0; }

void misaki_g2p_free(char * s) { std::free(s); }

void misaki_g2p_close(misaki_g2p * g) { delete g; }

}  // extern "C"

// ------------------------------------------------------------ C++ interface
struct eliza_misaki::misaki_lexicon {
    misaki_g2p * g = nullptr;
};

void eliza_misaki::misaki_lexicon_deleter::operator()(misaki_lexicon * l) const noexcept {
    if (l) {
        misaki_g2p_close(l->g);
        delete l;
    }
}

eliza_misaki::misaki_lexicon_ptr eliza_misaki::misaki_load_lexicon(const std::string & lexicon_path,
                                                                   const char * espeak_data_dir,
                                                                   std::string & err_out) noexcept {
    char * err = nullptr;
    misaki_g2p * g = misaki_g2p_open(lexicon_path.c_str(), espeak_data_dir, &err);
    if (!g) {
        try {
            err_out = err ? err : "misaki_g2p_open failed";
        } catch (...) {
        }
        misaki_g2p_free(err);
        return nullptr;
    }
    misaki_lexicon * l = new (std::nothrow) misaki_lexicon();
    if (!l) {
        misaki_g2p_close(g);
        return nullptr;
    }
    l->g = g;
    return misaki_lexicon_ptr(l);
}

eliza_misaki::misaki_fallback_kind eliza_misaki::misaki_get_fallback_kind(const misaki_lexicon * l) noexcept {
    return (l && misaki_g2p_fallback_kind(l->g) == MISAKI_G2P_FALLBACK_ESPEAK) ? MISAKI_FALLBACK_ESPEAK
                                                                               : MISAKI_FALLBACK_NONE;
}

bool eliza_misaki::misaki_phonemize_chunks(const misaki_lexicon * l, const std::string & text,
                                           std::vector<std::string> & chunks_out, std::string & err_out) noexcept {
    try {
        chunks_out.clear();
        if (!l) {
            err_out = "misaki_phonemize: no lexicon";
            return false;
        }
        char * out = nullptr;
        char * err = nullptr;
        if (misaki_g2p_phonemize(l->g, text.data(), text.size(), &out, &err) != 0) {
            err_out = err ? err : "misaki_g2p_phonemize failed";
            misaki_g2p_free(err);
            return false;
        }
        std::string all = out;
        misaki_g2p_free(out);
        size_t a = 0;
        while (a <= all.size() && !all.empty()) {
            size_t b = all.find('\n', a);
            chunks_out.push_back(all.substr(a, b == std::string::npos ? std::string::npos : b - a));
            if (b == std::string::npos) break;
            a = b + 1;
        }
        return true;
    } catch (const std::exception & e) {
        err_out = e.what();
        return false;
    }
}

bool eliza_misaki::misaki_phonemize(const misaki_lexicon * l, const std::string & text, std::string & phonemes_out,
                                    std::string & err_out) noexcept {
    std::vector<std::string> chunks;
    if (!misaki_phonemize_chunks(l, text, chunks, err_out)) return false;
    try {
        phonemes_out.clear();
        for (size_t i = 0; i < chunks.size(); i++) {
            if (i) phonemes_out += ' ';
            phonemes_out += chunks[i];
        }
        return true;
    } catch (const std::exception & e) {
        err_out = e.what();
        return false;
    }
}
