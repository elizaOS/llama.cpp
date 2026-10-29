/* SPDX-License-Identifier: Apache-2.0
 *
 * misaki-g2p: native American English grapheme-to-phoneme conversion that
 * reproduces misaki 0.9.4 (hexgrad/misaki) as Paradee-8M was trained with it:
 *   en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk='')
 * and paradee/tts.py's sentence chunking. Output phonemes use Paradee's
 * config.json vocabulary; map each symbol to its id, add a 0 pad at each end.
 *
 * Thread safety: a misaki_g2p handle may be used from one thread at a time.
 * Separate handles may run concurrently; calls into eSpeak NG are serialised
 * internally because libespeak-ng is process-global, under the same lock
 * Kokoro uses (tools/espeak-shared/eliza-espeak-shared.h).
 */
#ifndef MISAKI_G2P_H
#define MISAKI_G2P_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(MISAKI_G2P_SHARED)
#    if defined(MISAKI_G2P_BUILD)
#        define MISAKI_G2P_API __declspec(dllexport)
#    else
#        define MISAKI_G2P_API __declspec(dllimport)
#    endif
#elif defined(MISAKI_G2P_SHARED)
#    define MISAKI_G2P_API __attribute__((visibility("default")))
#else
#    define MISAKI_G2P_API
#endif

/* Longest chunk misaki_g2p_phonemize returns, in phoneme symbols: Paradee's
 * text side has 512 positions, two of which are the pad tokens. */
#define MISAKI_G2P_MAX_CHUNK 510

/* misaki_g2p_fallback_kind() values */
#define MISAKI_G2P_FALLBACK_SPELL  0 /* no eSpeak: out-of-lexicon words are spelled with lexicon letters */
#define MISAKI_G2P_FALLBACK_ESPEAK 1 /* eSpeak NG + misaki's eSpeak-to-misaki table, as in training */

typedef struct misaki_g2p misaki_g2p;

/* Open a converter.
 *   lex_path: misaki-en-us.lex written by convert_misaki_lexicon.py.
 *   espeak_data_dir_or_null: an espeak-ng-data directory (or its parent) to
 *     enable the eSpeak fallback; NULL selects the spell-out fallback. When the
 *     library was built without libespeak-ng, a non-NULL value is an error.
 *   err: on failure, *err receives a message to release with misaki_g2p_free.
 * Returns NULL on failure. */
MISAKI_G2P_API misaki_g2p * misaki_g2p_open(const char * lex_path, const char * espeak_data_dir_or_null, char ** err);

/* Phonemize UTF-8 text. On success returns 0 and sets *out_utf8 to the
 * phoneme chunks separated by '\n', each at most MISAKI_G2P_MAX_CHUNK symbols
 * and made only of Paradee vocabulary symbols (an empty string when the text
 * has nothing to say). Release *out_utf8 with misaki_g2p_free.
 * On failure returns a negative value and sets *err. */
MISAKI_G2P_API int misaki_g2p_phonemize(misaki_g2p * g, const char * utf8, size_t len, char ** out_utf8, char ** err);

/* MISAKI_G2P_FALLBACK_ESPEAK or MISAKI_G2P_FALLBACK_SPELL. */
MISAKI_G2P_API int misaki_g2p_fallback_kind(const misaki_g2p * g);

/* 1 when the library was built with libespeak-ng (GPL-3.0-or-later). */
MISAKI_G2P_API int misaki_g2p_espeak_linked(void);

MISAKI_G2P_API void misaki_g2p_free(char * s);
MISAKI_G2P_API void misaki_g2p_close(misaki_g2p * g);

#ifdef __cplusplus
}

/* C++ interface used by libelizainference (tools/omnivoice, ABI v17). Same
 * engine as the C API above; declarations match
 * tools/omnivoice/tests/paradee-contract/misaki-g2p.h. */
#    include <memory>
#    include <string>
#    include <vector>

namespace eliza_misaki {

/* How words missing from the lexicon are phonemized. */
enum misaki_fallback_kind {
    MISAKI_FALLBACK_NONE   = 0, /* no eSpeak: unknown words are spelled out with lexicon letters */
    MISAKI_FALLBACK_RULES  = 1, /* not produced by this implementation */
    MISAKI_FALLBACK_ESPEAK = 2, /* eSpeak NG output converted to misaki spelling (as in training) */
};

struct misaki_lexicon;
struct misaki_lexicon_deleter {
    void operator()(misaki_lexicon *) const noexcept;
};
using misaki_lexicon_ptr = std::unique_ptr<misaki_lexicon, misaki_lexicon_deleter>;

/* `espeak_data_dir` may be nullptr: unknown words are then spelled out. */
misaki_lexicon_ptr misaki_load_lexicon(const std::string & lexicon_path, const char * espeak_data_dir,
                                       std::string & err_out) noexcept;

misaki_fallback_kind misaki_get_fallback_kind(const misaki_lexicon * lexicon) noexcept;

/* English text -> misaki phonemes. The result is paradee's chunks (each at
 * most 510 symbols, one sentence or less) joined by single spaces, so a
 * chunker that splits after sentence-final punctuation recovers them.
 * Returns false and sets `err_out` on failure. */
bool misaki_phonemize(const misaki_lexicon * lexicon, const std::string & text, std::string & phonemes_out,
                      std::string & err_out) noexcept;

/* The same, as separate chunks: paradee/tts.py synthesizes each one in its
 * own forward pass. */
bool misaki_phonemize_chunks(const misaki_lexicon * lexicon, const std::string & text,
                             std::vector<std::string> & chunks_out, std::string & err_out) noexcept;

/* eSpeak-only G2P for languages without a misaki lexicon: misaki 0.9.4
 * espeak.EspeakG2P(language), which Kokoro's KPipeline uses for those
 * languages and which made the training phonemes of Kokoro distillations
 * such as Delibes-8M-esES. phonemizer-fork 3.3.2 semantics
 * (preserve_punctuation, with_stress, tie '^', language_switch
 * 'remove-flags'), EspeakG2P's E2M ties (t^ʃ -> ʧ, t^s -> ʦ, d^ʒ -> ʤ, ...),
 * then '^' and '-' removed. Supported languages: "es" (eSpeak NG voice
 * roa/es, Castilian Spanish). Needs libespeak-ng (misaki_g2p_espeak_linked)
 * and an espeak-ng-data directory with that language's dictionary (es_dict,
 * lang/roa/es); it shares libespeak-ng's process-global state with the
 * English fallback and Kokoro under one lock and re-selects its voice on
 * every call. One handle may be used from one thread at a time. */
struct espeak_g2p;
struct espeak_g2p_deleter {
    void operator()(espeak_g2p *) const noexcept;
};
using espeak_g2p_ptr = std::unique_ptr<espeak_g2p, espeak_g2p_deleter>;

espeak_g2p_ptr espeak_g2p_open(const std::string & language, const char * espeak_data_dir,
                               std::string & err_out) noexcept;

/* The language the handle was opened for ("" for null). */
const char * espeak_g2p_language(const espeak_g2p * g) noexcept;

/* Text -> phonemes in the Kokoro vocabulary: EspeakG2P's output with every
 * symbol outside the vocabulary dropped (as Kokoro does when it maps symbols
 * to ids; for example the inverted marks '¿' and '¡'). The whole text is one
 * string: callers chunk it at sentence ends for the model. Returns false and
 * sets `err_out` on failure. */
bool espeak_g2p_phonemize(const espeak_g2p * g, const std::string & text, std::string & phonemes_out,
                          std::string & err_out) noexcept;

}  // namespace eliza_misaki
#endif

#endif /* MISAKI_G2P_H */
