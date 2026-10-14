// SPDX-License-Identifier: Apache-2.0
// Agreement gate: native misaki_g2p against the pinned Python misaki 0.9.4
// reference (fixtures written by compare_misaki.py reference).
//
//   test-misaki-g2p-agreement --lex misaki-en-us.lex --fixture fixtures/reference.txt
//                             --espeak ESPEAK_NG_DATA_DIR [--report out.txt]
//
// Pass: >= 99.5% of reference words and >= 98% of sentences identical after
// Paradee's vocabulary filter, the C API giving the same chunks as paradee's
// _chunks(). Every mismatch is listed in the report. Exit 77 (skip) when
// eSpeak NG is not available: without it the fallback cannot match training.
#include "misaki-g2p.h"
#include "misaki-internal.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace misaki;

namespace {

// --- SHA-256 (FIPS 180-4), to check the fixture against its .sha256 file
struct Sha256 {
    uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint64_t len = 0;
    uint8_t buf[64];
    size_t n = 0;
    static uint32_t rotr(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }
    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
            0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
            0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
            0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; i++) w[i] = (uint32_t) p[4 * i] << 24 | (uint32_t) p[4 * i + 1] << 16 | (uint32_t) p[4 * i + 2] << 8 | p[4 * i + 3];
        for (int i = 16; i < 64; i++) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = hh + S1 + ch + k[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }
    void update(const uint8_t * p, size_t m) {
        len += m;
        while (m--) {
            buf[n++] = *p++;
            if (n == 64) { block(buf); n = 0; }
        }
    }
    std::string hex() {
        uint64_t bits = len * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t z = 0;
        while (n != 56) update(&z, 1);
        for (int i = 7; i >= 0; i--) { uint8_t b = (uint8_t) (bits >> (8 * i)); update(&b, 1); }
        char out[65];
        for (int i = 0; i < 8; i++) std::snprintf(out + 8 * i, 9, "%08x", h[i]);
        return std::string(out, 64);
    }
};

struct RefTok {
    std::string text, ps, ws, tag;
};
struct RefSent {
    std::string id, text, ps;
    std::vector<std::string> chunks;
    std::vector<RefTok> toks;
};

std::vector<std::string> split_tab(const std::string & s) {
    std::vector<std::string> out;
    size_t a = 0;
    while (true) {
        size_t b = s.find('\t', a);
        out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

u32 vocab_filter(const std::string & s) {
    u32 out;
    for (char32_t c : from_utf8(s)) if (in_paradee_vocab(c)) out += c;
    return out;
}

bool is_word(const std::string & s) {
    for (char32_t c : from_utf8(s)) if (py_isalpha(c) || py_digit_value(c) >= 0) return true;
    return false;
}

// LCS alignment of token texts: ref index -> native index
std::map<size_t, size_t> align(const std::vector<std::string> & a, const std::vector<std::string> & b) {
    size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> dp(n + 1, std::vector<int>(m + 1, 0));
    for (size_t i = n; i-- > 0;)
        for (size_t j = m; j-- > 0;)
            dp[i][j] = a[i] == b[j] ? dp[i + 1][j + 1] + 1 : std::max(dp[i + 1][j], dp[i][j + 1]);
    std::map<size_t, size_t> out;
    size_t i = 0, j = 0;
    while (i < n && j < m) {
        if (a[i] == b[j]) { out[i++] = j++; }
        else if (dp[i + 1][j] >= dp[i][j + 1]) i++;
        else j++;
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string lex, fixture, espeak, report_path;
    double word_gate = 0.995, sent_gate = 0.98;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if (a == "--lex") lex = argv[i + 1];
        else if (a == "--fixture") fixture = argv[i + 1];
        else if (a == "--espeak") espeak = argv[i + 1];
        else if (a == "--report") report_path = argv[i + 1];
    }
    if (lex.empty() || fixture.empty()) {
        std::fprintf(stderr, "usage: %s --lex LEX --fixture FIXTURE --espeak DIR [--report FILE]\n", argv[0]);
        return 2;
    }
    if (!espeak_linked() || espeak.empty()) {
        std::printf("SKIP: the agreement gate needs libespeak-ng 1.52.0 and its data (misaki's fallback in training)\n");
        return 77;
    }
    // fixture integrity
    std::ifstream fin(fixture, std::ios::binary);
    if (!fin) {
        std::fprintf(stderr, "cannot read %s\n", fixture.c_str());
        return 1;
    }
    std::string raw((std::istreambuf_iterator<char>(fin)), std::istreambuf_iterator<char>());
    {
        Sha256 sh;
        sh.update((const uint8_t *) raw.data(), raw.size());
        std::string got = sh.hex();
        std::ifstream sf(fixture + ".sha256");
        std::string want;
        sf >> want;
        if (want != got) {
            std::fprintf(stderr, "fixture sha256 %s does not match %s.sha256 (%s)\n", got.c_str(), fixture.c_str(), want.c_str());
            return 1;
        }
    }
    std::vector<RefSent> ref;
    {
        std::istringstream in(raw);
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto f = split_tab(line);
            if (f[0] == "S" && f.size() >= 3) ref.push_back({f[1], f[2], "", {}, {}});
            else if (f[0] == "P" && !ref.empty()) ref.back().ps = f.size() > 1 ? f[1] : "";
            else if (f[0] == "C" && !ref.empty()) ref.back().chunks.push_back(f.size() > 1 ? f[1] : "");
            else if (f[0] == "T" && !ref.empty()) {
                f.resize(5);
                ref.back().toks.push_back({f[1], f[2], f[3], f[4]});
            }
        }
    }
    std::string err;
    auto lf = LexFile::open(lex, err);
    if (!lf) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    auto fb = EspeakFallback::open(espeak, err);
    if (!fb) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    G2P g2p(*lf, fb.get());
    char * cerr = nullptr;
    misaki_g2p * api = misaki_g2p_open(lex.c_str(), espeak.c_str(), &cerr);
    if (!api) {
        std::fprintf(stderr, "misaki_g2p_open: %s\n", cerr ? cerr : "?");
        return 1;
    }
    size_t words = 0, word_ok = 0, sents = 0, sent_ok = 0, api_ok = 0;
    std::ostringstream detail;
    for (const RefSent & r : ref) {
        sents++;
        std::vector<MToken> toks;
        u32 got = g2p(from_utf8(r.text), &toks);
        u32 want = vocab_filter(r.ps);
        std::vector<u32> chunks = paradee_chunks(g2p, from_utf8(r.text));
        bool chunks_ok = chunks.size() == r.chunks.size();
        for (size_t c = 0; chunks_ok && c < chunks.size(); c++) chunks_ok = chunks[c] == vocab_filter(r.chunks[c]);
        bool ok = got == want && chunks_ok;
        sent_ok += ok;
        // the C API must return exactly these chunks, '\n'-joined
        std::string want_api;
        for (size_t c = 0; c < chunks.size(); c++) want_api += (c ? "\n" : "") + to_utf8(chunks[c]);
        char * out = nullptr;
        char * e = nullptr;
        int rc = misaki_g2p_phonemize(api, r.text.data(), r.text.size(), &out, &e);
        bool api_same = rc == 0 && out && want_api == out;
        api_ok += api_same;
        misaki_g2p_free(out);
        misaki_g2p_free(e);
        std::vector<std::string> rt, nt;
        for (auto & t : r.toks) rt.push_back(t.text);
        for (auto & t : toks) nt.push_back(to_utf8(t.text));
        auto pairs = align(rt, nt);
        std::ostringstream bad;
        for (size_t i = 0; i < r.toks.size(); i++) {
            if (!is_word(r.toks[i].text)) continue;
            words++;
            auto it = pairs.find(i);
            std::string got_ps = "(no matching token)", ntag = "-";
            if (it != pairs.end()) {
                const MToken & nt_tok = toks[it->second];
                got_ps = nt_tok.phonemes ? to_utf8(*nt_tok.phonemes) : "(none)";
                ntag = nt_tok.tag;
                if (nt_tok.phonemes && vocab_filter(r.toks[i].ps) == *nt_tok.phonemes) {
                    word_ok++;
                    continue;
                }
            }
            bad << "    word '" << r.toks[i].text << "' [" << r.toks[i].tag << " / native " << ntag << "]: misaki '"
                << to_utf8(vocab_filter(r.toks[i].ps)) << "' native '" << got_ps << "'\n";
        }
        if (!ok || !bad.str().empty() || !api_same) {
            detail << "- " << r.id << " " << r.text << "\n  misaki: " << to_utf8(want) << "\n  native: " << to_utf8(got) << "\n"
                   << bad.str();
            if (!api_same) detail << "    C API output differs from the chunker (rc " << rc << ")\n";
        }
    }
    misaki_g2p_close(api);
    double wr = words ? (double) word_ok / words : 0, sr = sents ? (double) sent_ok / sents : 0;
    bool pass = wr >= word_gate && sr >= sent_gate && api_ok == sents && sents > 0;
    std::ostringstream rep;
    rep << "misaki-g2p agreement gate: " << fixture << "\n";
    rep << "words:     " << word_ok << "/" << words << " = " << 100.0 * wr << "% (gate >= " << 100 * word_gate << "%)\n";
    rep << "sentences: " << sent_ok << "/" << sents << " = " << 100.0 * sr << "% (gate >= " << 100 * sent_gate << "%)\n";
    rep << "C API:     " << api_ok << "/" << sents << " sentences: misaki_g2p_phonemize equals the chunker output\n";
    rep << "result:    " << (pass ? "PASS" : "FAIL") << "\n\nmismatches:\n" << detail.str();
    std::fputs(rep.str().c_str(), stdout);
    if (!report_path.empty()) {
        std::ofstream o(report_path);
        o << rep.str();
    }
    return pass ? 0 : 1;
}
