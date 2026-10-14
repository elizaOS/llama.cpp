// tools/omnivoice/tests/paradee-contract/misaki-g2p.h
//
// Test double for the public misaki G2P header
// (tools/misaki-g2p/include/misaki-g2p.h). The FFI bridge is written against
// exactly these declarations; keep this copy in sync with the real header.
#pragma once

#include <memory>
#include <string>

namespace eliza_misaki {

// How words missing from the lexicon are phonemized.
enum misaki_fallback_kind {
    MISAKI_FALLBACK_NONE   = 0,  // lexicon only; unknown words are dropped or spelled
    MISAKI_FALLBACK_RULES  = 1,  // native letter-to-sound rules, misaki spelling
    MISAKI_FALLBACK_ESPEAK = 2,  // eSpeak NG output converted to misaki spelling
};

struct misaki_lexicon;
struct misaki_lexicon_deleter { void operator()(misaki_lexicon *) const noexcept; };
using misaki_lexicon_ptr = std::unique_ptr<misaki_lexicon, misaki_lexicon_deleter>;

// `espeak_data_dir` may be nullptr: the lexicon then uses the rules fallback
// (or none, if the build has no rules).
misaki_lexicon_ptr misaki_load_lexicon(
    const std::string & lexicon_path,
    const char * espeak_data_dir,
    std::string & err_out) noexcept;

misaki_fallback_kind misaki_get_fallback_kind(const misaki_lexicon * lexicon) noexcept;

// English text -> misaki phoneme string (UTF-8). Returns false and sets
// `err_out` on failure.
bool misaki_phonemize(
    const misaki_lexicon * lexicon,
    const std::string & text,
    std::string & phonemes_out,
    std::string & err_out) noexcept;

} // namespace eliza_misaki
