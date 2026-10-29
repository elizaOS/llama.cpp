// SPDX-License-Identifier: Apache-2.0
// test-espeak-g2p: the native eSpeak-only G2P (misaki espeak.EspeakG2P)
// against the Python reference.
//
//   test-espeak-g2p --e2m                          offline: the E2M table alone
//   test-espeak-g2p --fixture F.tsv --espeak DIR [--language es]
//
// The fixture (tests/gen_espeak_g2p_fixture.py: misaki 0.9.4, phonemizer-fork
// 3.3.2, espeak-ng 1.52.0) has text, EspeakG2P output and that output in the
// Kokoro vocabulary per line. Every line must match exactly, both the raw
// output and the public API's (eliza_misaki::espeak_g2p_phonemize). Exits 77
// (skip) when libespeak-ng is not linked or DIR is empty.
#include "misaki-g2p.h"
#include "misaki-internal.h"

#include <cstdio>
#include <fstream>
#include <string>

using namespace misaki;

static int e2m_unit() {
    struct Case { const char * in; const char * want; };
    const Case cases[] = {
        {"t^ʃˈikoɾ", "ʧˈikoɾ"},          // ch
        {"pˈat^so", "pˈaʦo"},            // t^s
        {"d^ʒˈem", "ʤˈem"},              // d^ʒ
        {"ˈa^ɪɾe", "ˈIɾe"},              // a^ɪ
        {"bwˈeno-s", "bwˈenos"},         // '-' removed (version None)
        {"ˈe^u", "ˈeu"},                 // a tie not in the table is removed
        {"s^s o^ʊ ə^ʊ ɔ^ɪ e^ɪ a^ʊ d^z", "S O Q Y A W ʣ"},
    };
    int fails = 0;
    for (const Case & c : cases) {
        const std::string got = to_utf8(EspeakG2P::e2m(from_utf8(c.in)));
        if (got != c.want) {
            std::fprintf(stderr, "e2m(%s) = %s, want %s\n", c.in, got.c_str(), c.want);
            fails++;
        }
    }
    std::printf("e2m: %d/%zu\n", (int) (sizeof(cases) / sizeof(cases[0])) - fails, sizeof(cases) / sizeof(cases[0]));
    return fails ? 1 : 0;
}

int main(int argc, char ** argv) {
    std::string fixture, espeak, language = "es";
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--e2m") return e2m_unit();
        if (a == "--fixture" && i + 1 < argc) fixture = argv[++i];
        else if (a == "--espeak" && i + 1 < argc) espeak = argv[++i];
        else if (a == "--language" && i + 1 < argc) language = argv[++i];
        else { std::fprintf(stderr, "usage: test-espeak-g2p --e2m | --fixture F.tsv --espeak DIR [--language L]\n"); return 2; }
    }
    if (!misaki_g2p_espeak_linked() || espeak.empty()) {
        std::printf("test-espeak-g2p: libespeak-ng or its data not available, skipping\n");
        return 77;
    }
    std::string err;
    std::unique_ptr<EspeakG2P> raw = EspeakG2P::open(language, espeak, err);
    eliza_misaki::espeak_g2p_ptr api = eliza_misaki::espeak_g2p_open(language, espeak.c_str(), err);
    if (!raw || !api) {
        std::fprintf(stderr, "open failed: %s\n", err.c_str());
        return 1;
    }
    std::string bad_err;
    if (eliza_misaki::espeak_g2p_open("xx", espeak.c_str(), bad_err) || bad_err.find("supported: es") == std::string::npos) {
        std::fprintf(stderr, "an unsupported language was accepted (%s)\n", bad_err.c_str());
        return 1;
    }
    std::ifstream in(fixture);
    if (!in) { std::fprintf(stderr, "cannot read %s\n", fixture.c_str()); return 1; }
    std::string line;
    int n = 0, raw_ok = 0, api_ok = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        const size_t t1 = line.find('\t'), t2 = line.find('\t', t1 + 1);
        if (t1 == std::string::npos || t2 == std::string::npos) { std::fprintf(stderr, "bad fixture line\n"); return 1; }
        const std::string text = line.substr(0, t1), want_raw = line.substr(t1 + 1, t2 - t1 - 1), want = line.substr(t2 + 1);
        n++;
        const std::string got_raw = to_utf8((*raw)(from_utf8(text)));
        std::string got;
        if (!eliza_misaki::espeak_g2p_phonemize(api.get(), text, got, err)) {
            std::fprintf(stderr, "phonemize failed: %s\n", err.c_str());
            return 1;
        }
        if (got_raw == want_raw) raw_ok++;
        else std::fprintf(stderr, "RAW  %s\n  want %s\n  got  %s\n", text.c_str(), want_raw.c_str(), got_raw.c_str());
        if (got == want) api_ok++;
        else std::fprintf(stderr, "API  %s\n  want %s\n  got  %s\n", text.c_str(), want.c_str(), got.c_str());
    }
    std::printf("test-espeak-g2p (%s): %d lines, raw %d identical, vocabulary-filtered %d identical\n",
                language.c_str(), n, raw_ok, api_ok);
    return (n > 0 && raw_ok == n && api_ok == n) ? 0 : 1;
}
