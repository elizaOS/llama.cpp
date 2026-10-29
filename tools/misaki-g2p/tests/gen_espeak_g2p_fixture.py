#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Reference fixture for the native eSpeak-only G2P (EspeakG2P).

Runs misaki 0.9.4 espeak.EspeakG2P(language) -- the G2P Kokoro's KPipeline uses
for languages without a misaki lexicon (lang_code 'e' -> 'es') and that made
Delibes-8M-esES's training phonemes -- on the corpus files and writes one TSV
line per input: text, EspeakG2P output, and that output with the symbols outside
the Kokoro vocabulary dropped (what the model sees).

Reference environment (pins.json "espeak_g2p"): misaki 0.9.4, phonemizer-fork
3.3.2, libespeak-ng 1.52.0 and its data. The library and data are given
explicitly (espeakng_loader is not needed):

  gen_espeak_g2p_fixture.py --language es --espeak-lib /path/libespeak-ng.so \\
      --espeak-data /path/espeak-ng-data --vocab <Kokoro-family config.json> \\
      --out tests/fixtures/espeak_g2p_es.tsv tests/corpus/es_authored.txt tests/corpus/es_numbers.txt
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import types
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--language", default="es")
    ap.add_argument("--espeak-lib", required=True)
    ap.add_argument("--espeak-data", required=True)
    ap.add_argument("--vocab", type=Path, required=True, help="config.json with a Kokoro 'vocab' map")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("corpus", type=Path, nargs="+")
    args = ap.parse_args()

    # misaki.espeak imports espeakng_loader only to locate the library and its data.
    loader = types.ModuleType("espeakng_loader")
    loader.get_library_path = lambda: args.espeak_lib
    loader.get_data_path = lambda: args.espeak_data
    sys.modules.setdefault("espeakng_loader", loader)
    from misaki import espeak  # noqa: E402

    g2p = espeak.EspeakG2P(language=args.language)
    version = ".".join(str(x) for x in g2p.backend.version())
    vocab = json.loads(args.vocab.read_text(encoding="utf-8"))["vocab"]
    lines = []
    for path in args.corpus:
        for text in path.read_text(encoding="utf-8").splitlines():
            if not text.strip():
                continue
            if "\t" in text:
                raise SystemExit(f"{path}: tab in {text!r}")
            ps, _ = g2p(text)
            kept = "".join(c for c in ps if c in vocab)
            lines.append(f"{text}\t{ps}\t{kept}")
    header = (f"# misaki {importlib.metadata.version('misaki')} espeak.EspeakG2P(language={args.language!r}); "
              f"phonemizer-fork {importlib.metadata.version('phonemizer-fork')}; espeak-ng {version}\n"
              f"# corpus: {' '.join(p.name + ':' + hashlib.sha256(p.read_bytes()).hexdigest()[:16] for p in args.corpus)}\n"
              f"# text<TAB>EspeakG2P output<TAB>output in the Kokoro vocabulary\n")
    args.out.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}: {len(lines)} lines, espeak-ng {version}, sha256 "
          f"{hashlib.sha256(args.out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
