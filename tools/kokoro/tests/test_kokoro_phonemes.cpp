// SPDX-License-Identifier: MIT
//
// test_kokoro_phonemes.cpp — checks for the Kokoro G2P tokenizer.
//
// Validates the codepoint→id vocab mapping (which reproduces the kokoro
// reference ids) and the [PAD, …, PAD] input-id wrapping. The espeak path is
// validated separately by the standalone harness (it requires libespeak-ng);
// here we drive ipa_to_token_ids() with fixed IPA so the test is
// dependency-free.

#include "kokoro-phonemes.h"
#include "kokoro.h"

#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

int main() {
    using namespace eliza_kokoro;

    {
        // Vocab size matches Kokoro v1.0 (max id 177 → size 178).
        assert(phoneme_vocab_size() == 178);
    }
    {
        // Codepoint → id mapping (from tokenizer.json model.vocab).
        assert(kokoro_codepoint_to_id(U'h') == 50);
        assert(kokoro_codepoint_to_id(U' ') == 16);
        assert(kokoro_codepoint_to_id(0x02C8u) == 156); // ˈ primary stress
        assert(kokoro_codepoint_to_id(0x0259u) == 83);  // ə schwa
        assert(kokoro_codepoint_to_id(0x028Au) == 135); // ʊ
        assert(kokoro_codepoint_to_id(0x2603u) == -1);  // ☃ not in vocab
    }
    {
        // ipa_to_token_ids reproduces the reference ids for "həlˈoʊ".
        // UTF-8: h(0x68) ə(0xC9 0x99) l(0x6C) ˈ(0xCB 0x88) o(0x6F) ʊ(0xCA 0x8A)
        const std::string ipa = "h\xC9\x99l\xCB\x88o\xCA\x8A";
        std::vector<int32_t> ids = ipa_to_token_ids(ipa);
        const std::vector<int32_t> exp = {50, 83, 54, 156, 57, 135};
        assert(ids == exp);
    }
    {
        // Unmapped codepoints are dropped, not turned into a sentinel id.
        const std::string ipa = "h\x07i"; // 'h', BEL (unmapped), 'i'
        std::vector<int32_t> ids = ipa_to_token_ids(ipa);
        const std::vector<int32_t> exp = {50, 51}; // h, i
        assert(ids == exp);
    }
    {
        // wrap_input_ids → [PAD, *ids, PAD].
        std::vector<int32_t> ids = {50, 51};
        std::vector<int32_t> wrapped = wrap_input_ids(ids);
        assert(wrapped.size() == 4);
        assert(wrapped.front() == KOKORO_PAD_ID);
        assert(wrapped.back() == KOKORO_PAD_ID);
        assert(wrapped[1] == 50 && wrapped[2] == 51);
    }
    {
        // Empty ids → just the two pad tokens.
        std::vector<int32_t> wrapped = wrap_input_ids({});
        assert(wrapped.size() == 2);
        assert(wrapped[0] == KOKORO_PAD_ID && wrapped[1] == KOKORO_PAD_ID);
    }
    {
        // Wrapping caps the phoneme run at 510 (512 with both pads).
        std::vector<int32_t> ids(2000, 50);
        std::vector<int32_t> wrapped = wrap_input_ids(ids);
        assert(wrapped.size() == 512);
    }
    {
        // Legacy ASCII fallback still emits a valid wrapped sequence.
        std::vector<int32_t> ids = phonemize_ascii("hi");
        // [PAD, 'h'(50), 'i'(51), PAD]
        const std::vector<int32_t> exp = {KOKORO_PAD_ID, 50, 51, KOKORO_PAD_ID};
        assert(ids == exp);
    }
    {
        // G2P-kind capability query mirrors espeak_available() (ABI v14, #11776).
        // A build without libespeak-ng must report ASCII so the TS layer knows it
        // has to feed IPA through kokoro_synthesize_ipa.
        const kokoro_g2p_kind kind = kokoro_g2p_kind_of_build();
        assert(kind == (espeak_available() ? KOKORO_G2P_ESPEAK : KOKORO_G2P_ASCII));
    }
    {
        // The IPA-input synthesis path derives its model input_ids as
        // wrap_input_ids(ipa_to_token_ids(ipa)) — assert the exact wrapped run
        // the native kokoro_synthesize_ipa entry would feed the model for
        // "həlˈoʊ".
        const std::string ipa = "h\xC9\x99l\xCB\x88o\xCA\x8A";
        std::vector<int32_t> input = wrap_input_ids(ipa_to_token_ids(ipa));
        const std::vector<int32_t> exp = {
            KOKORO_PAD_ID, 50, 83, 54, 156, 57, 135, KOKORO_PAD_ID};
        assert(input == exp);
    }

    std::printf("test_kokoro_phonemes: OK\n");
    return 0;
}
