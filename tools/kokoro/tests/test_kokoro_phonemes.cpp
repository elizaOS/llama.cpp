// SPDX-License-Identifier: MIT
//
// test_kokoro_phonemes.cpp — sanity checks for the ASCII phoneme tokenizer.

#include "kokoro-phonemes.h"

#include <cassert>
#include <cstdio>
#include <string>

int main() {
    using namespace eliza_kokoro;

    {
        // Empty text → just BOS + EOS.
        auto ids = phonemize_ascii("");
        assert(ids.size() == 2);
        assert(ids.front() == 1);   // BOS
        assert(ids.back() == 2);    // EOS
    }
    {
        // Single word → BOS + letters + EOS.
        auto ids = phonemize_ascii("hi");
        assert(ids.size() == 4);
        assert(ids[0] == 1);        // BOS
        assert(ids[1] == 4 + 7);    // 'h' → offset 4 + 7
        assert(ids[2] == 4 + 8);    // 'i' → offset 4 + 8
        assert(ids[3] == 2);        // EOS
    }
    {
        // Text longer than 510 tokens is truncated.
        std::string s(2000, 'a');
        auto ids = phonemize_ascii(s);
        assert(ids.size() <= 510);
    }
    {
        // Vocab size matches Kokoro v1.0.
        assert(phoneme_vocab_size() == 178);
    }

    std::printf("test_kokoro_phonemes: OK\n");
    return 0;
}
