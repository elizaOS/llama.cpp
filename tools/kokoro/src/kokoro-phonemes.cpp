// SPDX-License-Identifier: MIT
//
// kokoro-phonemes.cpp — ASCII grapheme→phoneme mapping for the Kokoro
// fork path. See kokoro-phonemes.h for the contract.
//
// The mapping is intentionally minimal: it matches the phoneme-id offsets
// used by the kokoro-onnx tokenizer for the *single-character* phonemes only.
// Multi-char espeak-ng phonemes (eɪ, oʊ, ʊə, etc) are NOT emitted — those
// require a full G2P pass that is out of scope for this header. The
// downstream synthesis still runs, just with degraded acoustic quality.

#include "kokoro-phonemes.h"

#include <algorithm>
#include <cctype>
#include <unordered_map>

namespace eliza_kokoro {

namespace {

// Special tokens (match upstream kokoro-onnx).
static constexpr int32_t TOK_PAD = 0;
static constexpr int32_t TOK_BOS = 1;
static constexpr int32_t TOK_EOS = 2;
static constexpr int32_t TOK_BLANK = 3;

// First non-special id (matches kokoro-onnx tokenizer offset).
static constexpr int32_t PHONEME_OFFSET = 4;

// Coarse ASCII letter → phoneme-id table. The id space follows the
// kokoro-onnx tokenizer for single-letter mappings; everything else falls
// back to TOK_BLANK so the synthesis path still emits a valid sequence.
static const std::unordered_map<char, int32_t> & letter_table() {
    static const std::unordered_map<char, int32_t> kTable = {
        {'a', PHONEME_OFFSET + 0},
        {'b', PHONEME_OFFSET + 1},
        {'c', PHONEME_OFFSET + 2},
        {'d', PHONEME_OFFSET + 3},
        {'e', PHONEME_OFFSET + 4},
        {'f', PHONEME_OFFSET + 5},
        {'g', PHONEME_OFFSET + 6},
        {'h', PHONEME_OFFSET + 7},
        {'i', PHONEME_OFFSET + 8},
        {'j', PHONEME_OFFSET + 9},
        {'k', PHONEME_OFFSET + 10},
        {'l', PHONEME_OFFSET + 11},
        {'m', PHONEME_OFFSET + 12},
        {'n', PHONEME_OFFSET + 13},
        {'o', PHONEME_OFFSET + 14},
        {'p', PHONEME_OFFSET + 15},
        {'q', PHONEME_OFFSET + 16},
        {'r', PHONEME_OFFSET + 17},
        {'s', PHONEME_OFFSET + 18},
        {'t', PHONEME_OFFSET + 19},
        {'u', PHONEME_OFFSET + 20},
        {'v', PHONEME_OFFSET + 21},
        {'w', PHONEME_OFFSET + 22},
        {'x', PHONEME_OFFSET + 23},
        {'y', PHONEME_OFFSET + 24},
        {'z', PHONEME_OFFSET + 25},
        // Punctuation gets dedicated ids so the rhythm predictor sees them.
        {' ', PHONEME_OFFSET + 26},
        {'.', PHONEME_OFFSET + 27},
        {',', PHONEME_OFFSET + 28},
        {'!', PHONEME_OFFSET + 29},
        {'?', PHONEME_OFFSET + 30},
        {';', PHONEME_OFFSET + 31},
        {':', PHONEME_OFFSET + 32},
        {'\'', PHONEME_OFFSET + 33},
    };
    return kTable;
}

} // namespace

std::vector<int32_t> phonemize_ascii(const std::string & text) {
    std::vector<int32_t> out;
    out.reserve(text.size() + 4);
    out.push_back(TOK_BOS);

    const auto & table = letter_table();
    for (char c : text) {
        const char lc = (char) std::tolower((unsigned char) c);
        auto it = table.find(lc);
        if (it == table.end()) {
            out.push_back(TOK_BLANK);
        } else {
            out.push_back(it->second);
        }
        // Hard cap at the BERT encoder's 510-token limit (including specials).
        if (out.size() >= 509) break;
    }

    out.push_back(TOK_EOS);
    return out;
}

int phoneme_vocab_size() noexcept {
    // 4 specials + 34 mapped + 140 unused slots = 178 (Kokoro v1.0 vocab).
    return 178;
}

} // namespace eliza_kokoro
