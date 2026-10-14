// SPDX-License-Identifier: Apache-2.0
// misaki-g2p-cli: phonemize text from stdin, one input per line.
//
//   misaki-g2p-cli --lex misaki-en-us.lex [--espeak DIR] [MODE] < text.txt
//
// MODE
//   (default)    Paradee chunks via the C API, one line per input ('\n' inside
//                a result is printed as " | ")
//   --dump       misaki-style dump used by tests/compare_misaki.py:
//                  S<TAB>text, P<TAB>phonemes, C<TAB>chunk...,
//                  T<TAB>token<TAB>phonemes<TAB>whitespace<TAB>tag...
//   --tokenize   spaCy-compatible tokens and tags: text<TAB>tag<TAB>space
//   --numbers    lines "cardinal N" / "ordinal N" / "year N" / "float X"
#include "misaki-g2p.h"
#include "misaki-internal.h"

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

using namespace misaki;

static int usage() {
    std::fprintf(stderr, "usage: misaki-g2p-cli --lex FILE [--espeak DIR] [--dump|--tokenize|--numbers] < lines\n");
    return 2;
}

int main(int argc, char ** argv) {
    std::string lex, espeak, mode = "chunks";
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--lex" && i + 1 < argc) lex = argv[++i];
        else if (a == "--espeak" && i + 1 < argc) espeak = argv[++i];
        else if (a == "--dump") mode = "dump";
        else if (a == "--tokenize") mode = "tokenize";
        else if (a == "--numbers") mode = "numbers";
        else return usage();
    }
    std::string line;
    if (mode == "numbers") {
        while (std::getline(std::cin, line)) {
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string kind = line.substr(0, sp), v = line.substr(sp + 1);
            u32 w;
            if (kind == "cardinal") w = number_cardinal(v);
            else if (kind == "ordinal") w = number_ordinal(v);
            else if (kind == "year") w = number_year(v);
            else w = number_float(v);
            std::printf("%s\n", to_utf8(w).c_str());
        }
        return 0;
    }
    if (lex.empty()) return usage();
    std::string err;
    std::unique_ptr<LexFile> lf = LexFile::open(lex, err);
    if (!lf) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    if (mode == "tokenize") {
        while (std::getline(std::cin, line)) {
            u32 text = from_utf8(line);
            auto toks = spacy_tokenize(text);
            auto tags = tag_tokens(toks, lf.get());
            for (size_t i = 0; i < toks.size(); i++) {
                std::printf("%s\t%s\t%d\n", to_utf8(toks[i].text).c_str(), tags[i].c_str(), toks[i].space_after ? 1 : 0);
            }
            std::printf("\n");
        }
        return 0;
    }
    if (mode == "dump") {
        std::unique_ptr<EspeakFallback> fb;
        if (!espeak.empty()) {
            fb = EspeakFallback::open(espeak, err);
            if (!fb) {
                std::fprintf(stderr, "%s\n", err.c_str());
                return 1;
            }
        }
        G2P g2p(*lf, fb.get());
        while (std::getline(std::cin, line)) {
            u32 text = from_utf8(line);
            std::vector<MToken> toks;
            u32 ps = g2p(text, &toks);
            std::printf("S\t%s\nP\t%s\n", line.c_str(), to_utf8(ps).c_str());
            for (const u32 & c : paradee_chunks(g2p, text)) std::printf("C\t%s\n", to_utf8(c).c_str());
            for (auto & t : toks) {
                std::printf("T\t%s\t%s\t%s\t%s\n", to_utf8(t.text).c_str(), t.phonemes ? to_utf8(*t.phonemes).c_str() : "",
                            to_utf8(t.whitespace).c_str(), t.tag.c_str());
            }
        }
        return 0;
    }
    char * e = nullptr;
    misaki_g2p * g = misaki_g2p_open(lex.c_str(), espeak.empty() ? nullptr : espeak.c_str(), &e);
    if (!g) {
        std::fprintf(stderr, "%s\n", e ? e : "open failed");
        misaki_g2p_free(e);
        return 1;
    }
    int rc = 0;
    while (std::getline(std::cin, line)) {
        char * out = nullptr;
        if (misaki_g2p_phonemize(g, line.data(), line.size(), &out, &e) != 0) {
            std::fprintf(stderr, "%s\n", e ? e : "phonemize failed");
            misaki_g2p_free(e);
            e = nullptr;
            rc = 1;
            std::printf("\n");
            continue;
        }
        std::string s = out;
        for (size_t p; (p = s.find('\n')) != std::string::npos;) s.replace(p, 1, " | ");
        std::printf("%s\n", s.c_str());
        misaki_g2p_free(out);
    }
    misaki_g2p_close(g);
    return rc;
}
