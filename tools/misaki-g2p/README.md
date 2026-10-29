# misaki-g2p

Native American English grapheme-to-phoneme conversion for Paradee-8M v1.0.

Paradee only ever saw phonemes from misaki 0.9.4 in training, configured as in
`paradee/tts.py`:

```python
en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False), unk='')
```

This library reproduces that pipeline in C++17 with no Python, no ONNX and no
ggml: spaCy's English tokenizer, a rule-based stand-in for spaCy's tagger,
misaki's lexicon lookup, stemming, stress, numbers and token merging, the
eSpeak NG fallback with misaki's eSpeak-to-misaki (E2M) table, and paradee's
sentence chunking. Every input is pinned in `pins.json`.

## Build

```sh
cmake -S tools/misaki-g2p -B build -G Ninja      # standalone
cmake --build build && (cd build && ctest --output-on-failure)
```

Inside llama.cpp it is a plain `add_subdirectory(tools/misaki-g2p)`, giving
the static library `misaki_g2p` (no ggml dependency).

| option | default | meaning |
|---|---|---|
| `MISAKI_G2P_ESPEAK` | ON | link libespeak-ng for the OOV fallback. Uses an existing `espeak-ng` target, else `find_library` (hint `MISAKI_G2P_ESPEAK_ROOT`). **GPL-3.0-or-later**, see `LICENSES/eSpeak-NG-NOTICE.txt`. |
| `MISAKI_G2P_BUILD_TESTS` | ON standalone, `LLAMA_BUILD_TESTS` in-tree | unit tests and the agreement gate |
| `MISAKI_G2P_LEXICON` | empty | prebuilt `misaki-en-us.lex` for the tests (else fetched and converted) |
| `MISAKI_G2P_ESPEAK_DATA` | empty | `espeak-ng-data` for the gate (else found next to the library) |

## Lexicon

```sh
python3 tools/misaki-g2p/convert_misaki_lexicon.py --fetch CACHE_DIR -o misaki-en-us.lex
python3 tools/misaki-g2p/convert_misaki_lexicon.py --verify misaki-en-us.lex
```

`--fetch` downloads `us_gold.json` / `us_silver.json` from hexgrad/misaki at
the pinned commit and refuses bytes whose sha256 differs; the output is
deterministic and its sha256 is pinned (`pins.json` `lexicon`). The file is a
sorted, mmap-able table (format in the converter's docstring) with misaki's
capitalisation expansion applied and the per-tag variants (DEFAULT, NOUN, VERB,
ADJ, ADV, DT, None, VBD, VBN, VBP) kept, including null variants.

## C API (`include/misaki-g2p.h`)

```c
misaki_g2p* misaki_g2p_open(const char* lex_path, const char* espeak_data_dir_or_null, char** err);
int  misaki_g2p_phonemize(misaki_g2p*, const char* utf8, size_t len, char** out_utf8, char** err);
int  misaki_g2p_fallback_kind(const misaki_g2p*);   /* 1 = eSpeak + E2M, 0 = spell-out */
int  misaki_g2p_espeak_linked(void);
void misaki_g2p_free(char*);
void misaki_g2p_close(misaki_g2p*);
```

`misaki_g2p_phonemize` returns paradee's chunks joined by `'\n'`: text is split
at sentence ends and newlines, each sentence is phonemized, and a sentence
longer than 510 symbols is cut at the last space that fits. Every symbol is in
Paradee's `config.json` vocabulary or the call fails (only explicit
`[word](/phonemes/)` overrides can trigger that). Map each symbol to its id and
add a 0 pad at each end.

With `espeak_data_dir_or_null == NULL` (or a build without libespeak-ng) words
missing from the lexicon are spelled with the lexicon's letter pronunciations
and `misaki_g2p_fallback_kind()` returns 0. Those words then differ from
training, so ship with eSpeak NG when the GPL is acceptable. libespeak-ng
keeps its data path in a 160-byte buffer; keep the data directory path under
about 135 bytes.

## Agreement gate

`tests/compare_misaki.py reference` runs the pinned Python misaki (versions in
`pins.json` `reference_python`) on

- `samples/sentences.txt` from the Paradee model repo (5 sentences),
- the first 200 held-out sentences of paradee `training/data/wikitext_sents.json`,
- eliza's `packages/app/scripts/kokoro-voice/qualify/.../sentences.txt` (25),

and writes `tests/fixtures/reference.txt` with its sha256. The CTest
`misaki_g2p_agreement` checks the native output against it: at least 99.5% of
words and 98% of sentences must be identical after Paradee's vocabulary filter,
and the C API must return exactly the chunker's output. Every mismatch is
listed in `build/tests/agreement-report.txt`. `misaki_g2p_agent` applies the
same gate to 70 assistant-style sentences (`tests/corpus/agent_style.txt`).

Results with eSpeak NG 1.52.0:

| set | words | sentences |
|---|---|---|
| gate corpus (230 sentences) | 3751/3752 (99.97%) | 226/230 (98.26%) |
| assistant-style (70) | 619/619 (100%) | 70/70 (100%) |
| WikiText 200..1700, not gated | 99.93% | 96.1% |
| WikiText 9000..11000, not gated, never tuned on | 99.89% | 95.8% |

The remaining differences come from spaCy's statistical tagger, which the
rules cannot follow everywhere: mostly the direction of straight quotes
written with spaces on both sides (a WikiText artifact, `“` vs `”`), and a
few homographs and `that` readings. `compare_misaki.py extended` reproduces the
non-gated rows; `compare_misaki.py numbers` checks the number words against
num2words (22201/22201 identical).

## Deviations from Python misaki

- POS tags come from rules (`src/tagger.cpp`), not en_core_web_sm.
- Number words are an independent implementation (num2words is LGPL).
- NFKC normalisation composes base + mark pairs but not Hangul jamo.
- Hostile input is bounded: runs of more than 64 subtokens are split, the
  camelCase subtoken search stops at 64 letters, URL matching skips tokens over
  2048 code points. Natural text never reaches these limits.
- eSpeak symbols outside Paradee's vocabulary are dropped at the fallback,
  which gives the same ids paradee feeds the model.

## eSpeak-only G2P (Spanish)

`src/espeak-g2p.cpp` reproduces misaki 0.9.4 `espeak.EspeakG2P(language)`, the whole-text G2P Kokoro's
`KPipeline` uses for languages without a misaki lexicon (`lang_code='e'` -> `'es'`), and so the phonemes
Kokoro distillations such as Delibes-8M-esES were trained on: phonemizer-fork 3.3.2 semantics
(`preserve_punctuation`, `with_stress`, tie `^`, `language_switch='remove-flags'`) over libespeak-ng, then
`EspeakG2P`'s own E2M ties (`t^ʃ` -> `ʧ`, `t^s` -> `ʦ`, `d^ʒ` -> `ʤ`, ...), `^` and `-` removed. It shares the
English fallback's punctuation handling and the process-wide libespeak-ng lock, and re-selects its voice on
every call. Only `es` (eSpeak NG voice `roa/es`, Castilian) is supported; it needs `es_dict` and `lang/roa/es`
in the eSpeak NG data (pins.json `espeak_ng.data`).

C++ API: `eliza_misaki::espeak_g2p_open("es", espeak_data_dir, err)` and `espeak_g2p_phonemize` (symbols
outside the Kokoro vocabulary, such as `¿` and `¡`, are dropped, as Kokoro does when it maps symbols to ids).
CLI: `misaki-g2p-cli --espeak-g2p es --espeak DIR [--raw]`.

Gate: `misaki_g2p_espeak_g2p_es` runs `test-espeak-g2p` on `tests/fixtures/espeak_g2p_es.tsv`, written by
`tests/gen_espeak_g2p_fixture.py` (misaki 0.9.4, phonemizer-fork 3.3.2, espeak-ng 1.52.0) over 182
project-authored sentences (`tests/corpus/es_authored.txt`, `es_numbers.txt`: everyday assistant sentences,
numbers, years, times, dates, prices, percentages, ordinals, abbreviations, Roman numerals and English
loanwords). Every line must match exactly, raw and vocabulary-filtered: 182/182 on macOS arm64 with Homebrew's
libespeak-ng 1.52.0. `misaki_g2p_espeak_g2p_e2m` (label `main`) checks the E2M table offline. The native G2P
also reproduces the phonemes of all 5 Delibes-8M-esES samples exactly.

## Regenerating generated files

`scripts/gen_tables.py` (Python 3.12, spaCy 3.8.16) writes
`src/unicode-tables.inc` and `src/spacy-en-rules.inc`;
`tests/gen_unit_expectations.py` writes `tests/data/`.
