// SPDX-License-Identifier: Apache-2.0
// Unit tests for misaki_g2p. Expectation tables in data/ come from the pinned
// reference implementations (gen_unit_expectations.py); sentence expectations
// below were produced by Python misaki 0.9.4 with the full lexicon and use
// only words present in the small test lexicon.
//
// usage: test-misaki-g2p MINI_LEX DATA_DIR
#include "misaki-g2p.h"
#include "misaki-internal.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace misaki;

static int g_fail = 0, g_pass = 0;

#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (cond) g_pass++;                                                         \
        else {                                                                      \
            g_fail++;                                                               \
            std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); \
        }                                                                           \
    } while (0)

static void check_eq(const std::string & got, const std::string & want, const std::string & what) {
    if (got == want) {
        g_pass++;
    } else {
        g_fail++;
        std::fprintf(stderr, "FAIL %s\n  want: %s\n  got:  %s\n", what.c_str(), want.c_str(), got.c_str());
    }
}

static std::vector<std::vector<std::string>> read_tsv(const std::string & path) {
    std::vector<std::vector<std::string>> rows;
    std::ifstream in(path);
    if (!in) {
        std::fprintf(stderr, "cannot read %s\n", path.c_str());
        g_fail++;
        return rows;
    }
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> f;
        size_t a = 0;
        while (true) {
            size_t b = line.find('\t', a);
            f.push_back(line.substr(a, b == std::string::npos ? std::string::npos : b - a));
            if (b == std::string::npos) break;
            a = b + 1;
        }
        rows.push_back(f);
    }
    return rows;
}

static std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t a = 0;
    while (true) {
        size_t b = s.find(sep, a);
        out.push_back(s.substr(a, b == std::string::npos ? std::string::npos : b - a));
        if (b == std::string::npos) break;
        a = b + 1;
    }
    return out;
}

static void test_unicode() {
    check_eq(to_utf8(py_lower(U("ÀÉÎ ΣΑ ЖЁ"))), "àéî σα жё", "py_lower");
    check_eq(to_utf8(py_upper(U("straße"))), "STRASSE", "py_upper multi");
    check_eq(to_utf8(py_capitalize(U("hELLO"))), "Hello", "capitalize");
    CHECK(py_isalpha(U'é') && py_isalpha(U'ж') && !py_isalpha(U'1') && !py_isalpha(U'-'));
    CHECK(py_digit_value(U'²') == 2 && py_digit_value(U'٣') == 3 && py_digit_value(U'a') < 0);
    check_eq(to_utf8(nfkc(U("ﬁne"))), "fine", "nfkc ligature");
    check_eq(to_utf8(nfkc(U("é"))), "é", "nfkc compose");
    check_eq(to_utf8(nfkc(U("ＡＢＣ１"))), "ABC1", "nfkc fullwidth");
    check_eq(to_utf8(from_utf8("a\xff" "b", 3)), "a\xef\xbf\xbd" "b", "invalid utf-8 -> U+FFFD");
}

static void test_numbers() {
    check_eq(to_utf8(number_cardinal("0")), "zero", "cardinal 0");
    check_eq(to_utf8(number_cardinal("169")), "one hundred and sixty-nine", "cardinal 169");
    check_eq(to_utf8(number_cardinal("1234")), "one thousand, two hundred and thirty-four", "cardinal 1234");
    check_eq(to_utf8(number_cardinal("2000500")), "two million, five hundred", "cardinal 2000500");
    check_eq(to_utf8(number_cardinal("1000012")), "one million and twelve", "cardinal 1000012");
    check_eq(to_utf8(number_ordinal("21")), "twenty-first", "ordinal 21");
    check_eq(to_utf8(number_ordinal("112")), "one hundred and twelfth", "ordinal 112");
    check_eq(to_utf8(number_ordinal("230")), "two hundred and thirtieth", "ordinal 230");
    check_eq(to_utf8(number_year("2015")), "twenty fifteen", "year 2015");
    check_eq(to_utf8(number_year("2008")), "two thousand and eight", "year 2008");
    check_eq(to_utf8(number_year("1905")), "nineteen oh-five", "year 1905");
    check_eq(to_utf8(number_year("1900")), "nineteen hundred", "year 1900");
    check_eq(to_utf8(number_float("2.50")), "two point five", "float 2.50");
    check_eq(to_utf8(number_float("10.05")), "ten point zero five", "float 10.05");
    check_eq(to_utf8(number_float("3.0")), "three", "float 3.0");
    CHECK(number_fits(std::string(42, '9')) && !number_fits(std::string(43, '9')));
    // past the named scales cardinal() reads digit by digit instead of indexing out of bounds
    check_eq(to_utf8(number_cardinal(std::string(43, '1'))).substr(0, 11), "one one one", "cardinal 43 digits");
    check_eq(to_utf8(number_ordinal("1" + std::string(59, '0'))).substr(0, 9), "one zero ", "ordinal 60 digits");
    check_eq(to_utf8(number_float(std::string(60, '9') + ".5")).substr(0, 9), "nine nine", "float 60 digits");
}

static void test_tables(const std::string & data) {
    for (auto & r : read_tsv(data + "/tokenizer.tsv")) {
        if (r.size() < 2) continue;
        std::string got;
        for (auto & t : spacy_tokenize(U(r[0].c_str()))) {
            if (!got.empty()) got += '\x1f';
            got += to_utf8(t.text) + (t.space_after ? "\x1e" : "");
        }
        check_eq(got, r[1], "tokenize: " + r[0]);
    }
    for (auto & r : read_tsv(data + "/subtokenize.tsv")) {
        if (r.size() < 2) continue;
        std::string got;
        for (auto & t : subtokenize(U(r[0].c_str()))) {
            if (!got.empty()) got += '\x1f';
            got += to_utf8(t);
        }
        check_eq(got, r[1], "subtokenize: " + r[0]);
    }
    for (auto & r : read_tsv(data + "/stress.tsv")) {
        if (r.size() < 3) continue;
        std::optional<double> st;
        if (r[1] != "None") st = std::stod(r[1]);
        auto got = apply_stress_for_test(U(r[0].c_str()), st);
        check_eq(got ? to_utf8(*got) : "None", r[2], "apply_stress: " + r[0] + " " + r[1]);
    }
    for (auto & r : read_tsv(data + "/e2m.tsv")) {
        if (r.size() < 2) continue;
        check_eq(to_utf8(EspeakFallback::e2m(U(r[0].c_str()))), r[1], "e2m: " + r[0]);
    }
}

static std::string phonemize(misaki_g2p * g, const std::string & text, int * rc = nullptr) {
    char * out = nullptr;
    char * err = nullptr;
    int r = misaki_g2p_phonemize(g, text.data(), text.size(), &out, &err);
    if (rc) *rc = r;
    std::string s = r == 0 && out ? out : std::string("ERROR: ") + (err ? err : "?");
    misaki_g2p_free(out);
    misaki_g2p_free(err);
    return s;
}

static void test_api(const std::string & lex) {
    char * err = nullptr;
    CHECK(misaki_g2p_open("/nonexistent/misaki.lex", nullptr, &err) == nullptr && err != nullptr);
    misaki_g2p_free(err);
    err = nullptr;
    misaki_g2p * g = misaki_g2p_open(lex.c_str(), nullptr, &err);
    CHECK(g != nullptr);
    if (!g) {
        std::fprintf(stderr, "open: %s\n", err ? err : "?");
        misaki_g2p_free(err);
        return;
    }
    CHECK(misaki_g2p_fallback_kind(g) == MISAKI_G2P_FALLBACK_SPELL);
    // Python misaki 0.9.4, full lexicon (paradee configuration)
    check_eq(phonemize(g, "Hello world."), "həlˈO wˈɜɹld.", "hello world");
    check_eq(phonemize(g, "The cat is a test."), "ðə kˈæt ɪz ɐ tˈɛst.", "the cat");
    check_eq(phonemize(g, "I read that record."), "ˌI ɹˈɛd ðˈæt ɹˈɛkəɹd.", "read/that/record");
    check_eq(phonemize(g, "Hello, world! This is a test."), "həlˈO, wˈɜɹld!\nðˌɪs ɪz ɐ tˈɛst.", "two sentences");
    check_eq(phonemize(g, "It is 20."), "ˌɪt ɪz twˈɛnti.", "number");
    check_eq(phonemize(g, "That is $5."), "ðˈæt ɪz fˈIv dˈɑləɹz.", "currency");
    check_eq(phonemize(g, "To the cat, I am ten."), "tə ðə kˈæt, ˌI ɐm tˈɛn.", "to/the/am");
    check_eq(phonemize(g, "HELLO WORLD"), "həlˈO wˈɜɹld", "all caps");
    check_eq(phonemize(g, "Walking cats."), "wˈɔkɪŋ kˈæts.", "stems");
    // out of lexicon without eSpeak: letters, as misaki's get_NNP spells them
    check_eq(phonemize(g, "Zyx"), "zˌiwˌIˈɛks", "spell-out fallback");
    check_eq(phonemize(g, ""), "", "empty");
    check_eq(phonemize(g, "   \n\n  "), "", "blank");
    // newlines split chunks like paradee's _chunks
    check_eq(phonemize(g, "Hello\nworld"), "həlˈO\nwˈɜɹld", "newline split");
    // a very long sentence is cut at the last space within 510 symbols
    {
        std::string text;
        for (int i = 0; i < 200; i++) text += "hello ";
        text += "world.";
        std::string out = phonemize(g, text);
        auto chunks = split(out, '\n');
        CHECK(chunks.size() >= 2);
        size_t total = 0;
        for (auto & c : chunks) {
            u32 u = from_utf8(c);
            CHECK(u.size() <= MISAKI_G2P_MAX_CHUNK && !u.empty());
            CHECK(u.front() != U' ');
            total += u.size();
        }
        CHECK(total > 1000);
        check_eq(chunks.back().substr(chunks.back().size() - std::strlen("wˈɜɹld.")), "wˈɜɹld.", "last chunk ends the sentence");
    }
    // every output symbol is in the Paradee vocabulary
    {
        std::string out = phonemize(g, "Café naïve 😀 ✓ ® © ½ ⅓ – — … “quotes” ‘x’ [a] {b} <c> 100% 3/4 @home #tag ^_^ ~");
        CHECK(out.rfind("ERROR", 0) != 0);
        for (char32_t c : from_utf8(out)) CHECK(c == U'\n' || in_paradee_vocab(c));
    }
    // explicit phonemes outside the vocabulary are rejected, not dropped
    {
        int rc = 0;
        std::string out = phonemize(g, "[hello](/hɛ5o/)", &rc);
        CHECK(rc != 0 && out.find("vocabulary") != std::string::npos);
        // the error must not echo text derived from the caller's input (it may be logged)
        CHECK(out.find("hɛ") == std::string::npos);
        check_eq(phonemize(g, "[hello](/hɛlˈO/) world"), "hɛlˈO wˈɜɹld", "phoneme override");
    }
    // hostile input stays bounded: long runs, many subtokens, long numbers
    {
        std::string a(20000, 'a'), dots(5000, '.'), hy;
        for (int i = 0; i < 3000; i++) hy += "a-";
        const std::string d60(60, '9');
        for (const std::string & s : {a, dots, hy, std::string(3000, '9'), std::string(1000, '@') + "x.com",
                                      "It is " + d60 + ".5 units.", "Pay $" + d60 + " now.", "The " + d60 + "th one.",
                                      "It is 1.1" + std::string(59, '0') + ".5 units.", "Pay $1." + d60 + " now."}) {
            int rc = 0;
            phonemize(g, s, &rc);
            CHECK(rc == 0);
        }
    }
    // invalid arguments
    {
        char * out = nullptr;
        char * e = nullptr;
        CHECK(misaki_g2p_phonemize(nullptr, "x", 1, &out, &e) != 0);
        misaki_g2p_free(e);
        e = nullptr;
        CHECK(misaki_g2p_phonemize(g, nullptr, 3, &out, &e) != 0);
        misaki_g2p_free(e);
    }
    misaki_g2p_close(g);
    // eSpeak availability is reported, and requesting it without the library fails cleanly
    if (!misaki_g2p_espeak_linked()) {
        err = nullptr;
        CHECK(misaki_g2p_open(lex.c_str(), "/tmp", &err) == nullptr && err != nullptr);
        misaki_g2p_free(err);
    }
}

static void test_cpp_api(const std::string & lex) {
    std::string err;
    CHECK(!eliza_misaki::misaki_load_lexicon("/nonexistent.lex", nullptr, err) && !err.empty());
    auto l = eliza_misaki::misaki_load_lexicon(lex, nullptr, err);
    CHECK(l != nullptr);
    if (!l) return;
    CHECK(eliza_misaki::misaki_get_fallback_kind(l.get()) == eliza_misaki::MISAKI_FALLBACK_NONE);
    std::string ps;
    CHECK(eliza_misaki::misaki_phonemize(l.get(), "Hello, world! This is a test.", ps, err));
    check_eq(ps, "həlˈO, wˈɜɹld! ðˌɪs ɪz ɐ tˈɛst.", "C++ misaki_phonemize");
    std::vector<std::string> chunks;
    CHECK(eliza_misaki::misaki_phonemize_chunks(l.get(), "Hello, world! This is a test.", chunks, err));
    CHECK(chunks.size() == 2);
    CHECK(eliza_misaki::misaki_phonemize_chunks(l.get(), "", chunks, err) && chunks.empty());
    CHECK(!eliza_misaki::misaki_phonemize(nullptr, "x", ps, err));
}

static void test_bad_lexicon(const std::string & data) {
    std::string err;
    CHECK(LexFile::open(data + "/mini_gold.json", err) == nullptr && err.find("magic") != std::string::npos);
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MINI_LEX DATA_DIR\n", argv[0]);
        return 2;
    }
    test_unicode();
    test_numbers();
    test_tables(argv[2]);
    test_api(argv[1]);
    test_cpp_api(argv[1]);
    test_bad_lexicon(argv[2]);
    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
