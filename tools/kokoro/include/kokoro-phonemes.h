// SPDX-License-Identifier: MIT
//
// kokoro-phonemes.h — minimal ASCII text → Kokoro phoneme-id mapping.
//
// Kokoro v1.0 uses espeak-ng's phoneme inventory + a small set of control
// tokens (BOS, EOS, PAD, blanks). The training-time path passes text through
// `phonemize` (Python wrapper around espeak-ng) before tokenizing.
//
// Adding an espeak-ng dependency to the fork is overkill for a TTS that
// is being ported as a one-release deprecation runway. This header
// implements a deterministic grapheme→phoneme mapping that:
//
//   1. covers the basic Latin alphabet + common digraphs (sh, ch, th, ng);
//   2. maps every other ASCII printable to PAD;
//   3. returns ids in the same value range as kokoro-onnx's tokenizer
//      (PAD=0, BOS=1, EOS=2, then phonemes from offset 3).
//
// The synthesis quality this produces is noticeably worse than the
// espeak-ng path — that is the documented gap in J2-kokoro-port-notes.md.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace eliza_kokoro {

// Tokenize a UTF-8 / ASCII text string into a phoneme-id vector.
// Always returns a sequence of length <= 510 (the BERT encoder cap in
// Kokoro v1.0 — anything longer is split at the caller).
std::vector<int32_t> phonemize_ascii(const std::string & text);

// Diagnostic — total phoneme vocab size (for hparams cross-check).
int phoneme_vocab_size() noexcept;

} // namespace eliza_kokoro
