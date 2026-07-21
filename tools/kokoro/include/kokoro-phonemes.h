// SPDX-License-Identifier: MIT
//
// kokoro-phonemes.h — text → Kokoro phoneme-id mapping.
//
// Kokoro v1.0 tokenizes espeak-ng IPA against a small fixed vocab
// (`tts/kokoro/tokenizer.json`, `model.vocab`). The reference Python path is:
//
//   text → espeak-ng (en-us, --ipa) → IPA string → per-codepoint vocab lookup
//        → ids → model input_ids = [0, *ids, 0]   (0 = the "$" pad symbol)
//
// Every vocab key is a single Unicode codepoint, so the mapping is a pure
// codepoint→id table lookup over the IPA string (no multi-char digraph
// handling is needed — espeak already emits the canonical IPA codepoints,
// e.g. eɪ is two codepoints 'e'+'ɪ', each with its own id).
//
// Two build modes:
//   * KOKORO_USE_ESPEAK (default when libespeak-ng is linked) — the real G2P
//     path: `phonemize_ipa()` drives espeak_TextToPhonemes() to get IPA, then
//     maps to ids. This reproduces the kokoro reference ids exactly.
//   * fallback — when espeak is unavailable the caller may pass pre-computed
//     IPA from the TS layer (which already runs espeak) into
//     `ipa_to_token_ids()`. `phonemize_ipa()` then returns an empty vector and
//     the caller must supply IPA.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace eliza_kokoro {

// Kokoro pad/boundary token. model.vocab maps '$' → 0; the reference wraps the
// phoneme ids as [PAD, *ids, PAD] to form the model input_ids.
inline constexpr int32_t KOKORO_PAD_ID = 0;

// Map a single Unicode codepoint (an espeak IPA symbol) to its Kokoro vocab id.
// Returns -1 if the codepoint is not in the vocab (caller drops it, matching
// the reference which silently skips unmapped codepoints).
int32_t kokoro_codepoint_to_id(char32_t cp) noexcept;

// Map an espeak-ng IPA string (UTF-8) to the bare Kokoro phoneme-id sequence
// (no pad wrapping). Codepoints absent from the vocab are dropped. This is the
// `ids` array in reference-ids.json — its length is the style-row index.
std::vector<int32_t> ipa_to_token_ids(const std::string & ipa);

// Phonemize text to bare Kokoro phoneme ids via espeak-ng (en-us IPA).
// Returns the same sequence as `ipa_to_token_ids(espeak_ipa(text))`.
// When KOKORO_USE_ESPEAK is not compiled in, returns an empty vector — the
// caller must supply IPA from the TS layer and call `ipa_to_token_ids()`.
std::vector<int32_t> phonemize_ipa(const std::string & text);

// Wrap a bare phoneme-id sequence as the model input_ids: [PAD, *ids, PAD].
std::vector<int32_t> wrap_input_ids(const std::vector<int32_t> & ids);

// Convenience: text → model input_ids [PAD, *ipa_ids, PAD] via espeak.
// Equivalent to `wrap_input_ids(phonemize_ipa(text))`.
std::vector<int32_t> phonemize_to_input_ids(const std::string & text);

// True when this build links libespeak-ng (the real G2P path is available).
bool espeak_available() noexcept;

// Total Kokoro vocab size (highest id + 1 = 178 for v1.0).
int phoneme_vocab_size() noexcept;

// --- Legacy ASCII fallback (retained only for callers not yet migrated) ---
// Deprecated: returns the degraded ASCII grapheme mapping. New code uses
// phonemize_to_input_ids().
std::vector<int32_t> phonemize_ascii(const std::string & text);

} // namespace eliza_kokoro
