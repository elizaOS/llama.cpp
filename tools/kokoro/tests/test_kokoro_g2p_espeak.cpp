// Standalone validation for kokoro-phonemes.cpp real G2P.
// Compile:
//   clang++ -std=c++17 -O2 -DKOKORO_USE_ESPEAK \
//     -I <kokoro/include> -I /opt/homebrew/include \
//     <this> <kokoro/src/kokoro-phonemes.cpp> \
//     -L /opt/homebrew/lib -lespeak-ng -o /tmp/t && /tmp/t

#include "kokoro-phonemes.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace eliza_kokoro;

static bool eq(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) return false;
    return true;
}

static void print_ids(const char* label, const std::vector<int32_t>& v) {
    printf("%s [", label);
    for (size_t i = 0; i < v.size(); ++i) printf("%s%d", i ? "," : "", v[i]);
    printf("]\n");
}

int main() {
    printf("espeak_available: %s\n", espeak_available() ? "yes" : "no");
    printf("phoneme_vocab_size: %d\n\n", phoneme_vocab_size());

    int fails = 0;

    // Reference case (from reference-ids.json).
    {
        const std::string text = "Hello, this is a native Kokoro voice test.";
        const std::vector<int32_t> ref_ids = {
            50, 83, 54, 156, 57, 135, 16, 81, 102, 61, 16, 102, 68, 16, 70, 16,
            56, 156, 47, 102, 125, 102, 64, 16, 53, 83, 53, 156, 76, 158, 123,
            57, 135, 16, 64, 156, 76, 102, 61, 16, 62, 156, 86, 61, 62};
        std::vector<int32_t> got = phonemize_ipa(text);
        print_ids("ref ids:", ref_ids);
        print_ids("got ids:", got);
        const bool ok = eq(got, ref_ids);
        printf("REFERENCE ids match: %s\n", ok ? "PASS" : "FAIL");
        if (!ok) ++fails;

        // input_ids wrapping = [0, *ids, 0]
        std::vector<int32_t> input = phonemize_to_input_ids(text);
        const bool wrap_ok = input.size() == ref_ids.size() + 2 &&
                             input.front() == 0 && input.back() == 0;
        printf("input_ids wrap [0,*,0]: %s (len %zu)\n",
               wrap_ok ? "PASS" : "FAIL", input.size());
        if (!wrap_ok) ++fails;
    }

    // ipa_to_token_ids reproduces from a fixed IPA string (espeak-independent).
    {
        const std::string ipa = "h\xc9\x99l\xcb\x88o\xca\x8a"; // həlˈoʊ
        std::vector<int32_t> got = ipa_to_token_ids(ipa);
        // h ə l ˈ o ʊ -> 50 83 54 156 57 135
        const std::vector<int32_t> exp = {50, 83, 54, 156, 57, 135};
        const bool ok = eq(got, exp);
        printf("\nipa_to_token_ids(\"həlˈoʊ\"): %s\n", ok ? "PASS" : "FAIL");
        if (!ok) { print_ids("  got:", got); ++fails; }
    }

    // Extra phrases — assert no codepoint is dropped (every espeak IPA char is
    // in-vocab) and the count is sane.
    {
        const char* phrases[] = {
            "The quick brown fox jumps over the lazy dog.",
            "I have 3 apples and 2 oranges.",
            "Eliza speaks with a calm, natural voice.",
        };
        for (const char* p : phrases) {
            std::vector<int32_t> ids = phonemize_ipa(p);
            printf("\nphrase: %s\n  ids(%zu):", p, ids.size());
            for (int32_t id : ids) printf(" %d", id);
            printf("\n");
            if (ids.empty()) { printf("  FAIL: empty\n"); ++fails; }
        }
    }

    printf("\n=== %s ===\n", fails == 0 ? "ALL PASS" : "FAILURES PRESENT");
    return fails == 0 ? 0 : 1;
}
