#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write the expectation tables test_misaki_g2p.cpp checks, from the pinned
reference implementations (same environment as compare_misaki.py):

  data/tokenizer.tsv    text -> spaCy 3.8.16 tokens (en tokenizer, no model)
  data/subtokenize.tsv  word -> misaki en.subtokenize pieces
  data/stress.tsv       phonemes, stress -> misaki en.apply_stress
  data/e2m.tsv          phonemizer output -> misaki EspeakFallback E2M result
  data/mini_gold.json / mini_silver.json  a slice of us_gold/us_silver for unit tests

usage: python3 tests/gen_unit_expectations.py MISAKI_DATA_DIR
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
SEP = "\x1f"

TOKENIZER_CASES = [
    "I'm on it.",
    "Don't stop, I can't hear you!",
    "The U.S. Army and e.g. 10am or 5pm.",
    "It costs $5.99 (about €5).",
    "She co-founded the well-known start-up in 2015-16.",
    "Xenoceratops - (Alberta, Canada)",
    "Visit https://example.com/help?x=1 today.",
    "Email bob@example.com, please.",
    "Meet at 3:30 or 10:45.",
    "He said \"hello\" and left...",
    "Rock 'n' roll, y'all.",
    "Mr. Smith vs. Mrs. Jones at St. Paul's.",
    "It's 72°F outside :)",
    "Wait—what? Really…",
    "The 1990s and the '80s.",
    "Ph.D. students in N.Y.",
    "A 10-minute drive, 2.5 km/h, 30% off.",
    "Gold'cars and type'Desert",
    "I 've seen it , it 's fine .",
    "C++ and C# are languages; so is F#.",
]

SUBTOKEN_CASES = [
    "McCully", "co-founded", "don't", "rock'n'roll", "iPhone", "USA", "NASA's", "1,000.50", "-5", "3.14",
    "'90s", "Ma'am", "''quoted''", "hello--world", "snake_case", "ABCDef", "e.g.", "U.S.", "x2y", "Se8",
    "Heroes'Cemetery", "O'Brien", "ﬁne", "naïve", "über", "72°F", "a-b-c",
]

STRESS_CASES = [
    ("hɛlˈO", None), ("hɛlˈO", -2), ("hɛlˈO", -1), ("hɛlˈO", 0), ("hɛlˈO", 0.5), ("hɛlˈO", 1), ("hɛlˈO", 2),
    ("ðə", 0), ("ðə", 0.5), ("ðə", 2), ("bʌt", 1), ("tˌu", 1), ("tˌu", 2), ("tˌu", -0.5), ("kˈæt", -0.5),
    ("bdk", 2), ("ɐ", 0.5), ("ˌænd", -1),
]

E2M_CASES = [
    "hˈɜːlbʌts", "v^ɛlʊvˈɑːnə", "ɹˈæfa^ɪˈɛl", "t^ʃˈuːk", "kˈo^ʊən", "nˈɪkələ", "mˈɛdə^l", "bˈʌʔn̩",
    "ɹˈiːdɚ", "lˈɪndɪ", "ɡɹæn d^ʒˈɑːn", "ʔˌn̩ wˈʌn", "xˈɑːn", "ɬˈɑː", "ɔ^ɪl", "a^ʊt", "e^ɪ", "fˈiːʲo",
    "bˈɜːɹd", "ɪə", "sˈe", "ɐbˈaʊt", "ɾˈæ", "kɐ̃",
]


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    misaki_data = sys.argv[1]
    os.makedirs(DATA, exist_ok=True)
    import spacy
    from misaki import en
    from misaki.espeak import EspeakFallback

    nlp = spacy.blank("en")
    with open(os.path.join(DATA, "tokenizer.tsv"), "w", encoding="utf-8") as f:
        for text in TOKENIZER_CASES:
            doc = nlp(text)
            toks = [t.text + ("\x1e" if t.whitespace_ else "") for t in doc]
            f.write(text + "\t" + SEP.join(toks) + "\n")
    with open(os.path.join(DATA, "subtokenize.tsv"), "w", encoding="utf-8") as f:
        for w in SUBTOKEN_CASES:
            f.write(w + "\t" + SEP.join(en.subtokenize(w)) + "\n")
    with open(os.path.join(DATA, "stress.tsv"), "w", encoding="utf-8") as f:
        for ps, st in STRESS_CASES:
            f.write(f"{ps}\t{'None' if st is None else st}\t{en.apply_stress(ps, st)}\n")

    class Fake(EspeakFallback):
        def __init__(self):
            self.british = False
            self.version = None

            class B:
                def phonemize(_, texts):
                    return [texts[0]]
            self.backend = B()

    fb = Fake()

    class Tok:
        def __init__(self, text):
            self.text = text

    with open(os.path.join(DATA, "e2m.tsv"), "w", encoding="utf-8") as f:
        for ps in E2M_CASES:
            f.write(ps + "\t" + fb(Tok(ps))[0] + "\n")

    gold = json.load(open(os.path.join(misaki_data, "us_gold.json"), encoding="utf-8"))
    silver = json.load(open(os.path.join(misaki_data, "us_silver.json"), encoding="utf-8"))
    keep = ["hello", "world", "read", "the", "a", "to", "am", "used", "dollar", "cent", "and", "point", "minus",
            "O", "percent", "versus", "dot", "slash", "zero", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "twenty", "hundred", "thousand", "that", "record", "walk", "I", "in", "is",
            "it", "cat", "test", "this"]
    keep += [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    mini_gold = {k: gold[k] for k in keep if k in gold}
    mini_silver = {k: silver[k] for k in ["dollars", "testing", "cats"] if k in silver}
    for name, obj in (("mini_gold.json", mini_gold), ("mini_silver.json", mini_silver)):
        with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False, indent=0, sort_keys=True) + "\n")
    print("wrote", DATA)


if __name__ == "__main__":
    main()
