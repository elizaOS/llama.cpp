#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Convert misaki's American English lexicon (us_gold.json + us_silver.json)
into the binary, mmap-able `misaki-en-us.lex` read by tools/misaki-g2p.

The inputs are pinned in pins.json (hexgrad/misaki at a fixed commit, same
bytes as the misaki 0.9.4 wheel). The converter refuses inputs whose sha256
does not match the pin unless --allow-unpinned is given (used only by the unit
tests, which build a tiny lexicon from tests/data/).

misaki's Lexicon.grow_dictionary() is applied to each table, the same way
misaki does at load time: every lowercase key of two or more characters also
gets a Capitalized key, and every Capitalized key a lowercase one, unless the
dictionary already has that key.

File layout (little-endian, every section 8-byte aligned):

  header (128 bytes)
    0   char[8]  magic "MISAKILX"
    8   u32      format version (1)
    12  u32      flags (0 = American English)
    16  u32      gold entry count
    20  u32      silver entry count
    24  u32      variant count
    28  u32      string pool size in bytes
    32  u8[32]   sha256 of the us_gold.json bytes the file was built from
    64  u8[32]   sha256 of the us_silver.json bytes
    96  u32      offset of the gold entries
    100 u32      offset of the silver entries
    104 u32      offset of the variants
    108 u32      offset of the string pool
    112 u32      total file size
    116 u32[3]   reserved, zero
  gold entries, then silver entries (12 bytes each, sorted by key bytes)
    u32 key offset, u16 key length, u16 variant count, u32 first variant index
  variants (8 bytes each)
    u8 tag, u8 null flag, u16 value length, u32 value offset
  string pool (UTF-8, no terminators, sorted and de-duplicated)

A plain string entry has one variant with tag PLAIN. A tagged entry (for
example "read" or "that") has one variant per tag, in tag order; a null value
(JSON null) keeps the tag with the null flag set, because misaki treats a
null as "spell the word out" rather than as a missing tag.

The output is deterministic: the same inputs always give the same bytes.

usage:
  convert_misaki_lexicon.py --gold us_gold.json --silver us_silver.json -o misaki-en-us.lex
  convert_misaki_lexicon.py --fetch CACHE_DIR -o misaki-en-us.lex   # download the pinned JSON
  convert_misaki_lexicon.py --verify misaki-en-us.lex                # check against pins.json
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = os.path.join(HERE, "pins.json")

MAGIC = b"MISAKILX"
VERSION = 1
HEADER_SIZE = 128

# Tag ids shared with src/lexicon.h. PLAIN marks an entry whose JSON value is a
# plain string. "None" is the misaki tag used when nothing follows the word.
TAGS = ["PLAIN", "DEFAULT", "NOUN", "VERB", "ADJ", "ADV", "DT", "None", "VBD", "VBN", "VBP"]
TAG_ID = {t: i for i, t in enumerate(TAGS)}

# Phoneme vocabulary misaki asserts for American English gold entries (en.py US_VOCAB).
US_VOCAB = frozenset("AIOWYbdfhijklmnpstuvwzæðŋɑɔəɛɜɡɪɹɾʃʊʌʒʤʧˈˌθᵊᵻʔ")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def load_pins():
    with open(PINS, encoding="utf-8") as f:
        return json.load(f)


def grow_dictionary(d):
    """misaki en.Lexicon.grow_dictionary, for ASCII keys."""
    e = {}
    for k, v in d.items():
        if len(k) < 2:
            continue
        if k == k.lower():
            if k != k.capitalize():
                e[k.capitalize()] = v
        elif k == k.lower().capitalize():
            e[k.lower()] = v
    return {**e, **d}


def check_gold(d):
    for k, vs in d.items():
        if isinstance(vs, str):
            bad = [c for c in vs if c not in US_VOCAB]
            if bad:
                raise SystemExit(f"gold entry {k!r} has phonemes outside the US vocab: {bad}")
        elif isinstance(vs, dict):
            if "DEFAULT" not in vs:
                raise SystemExit(f"gold entry {k!r} has no DEFAULT tag")
            for t, v in vs.items():
                if t not in TAG_ID or t == "PLAIN":
                    raise SystemExit(f"gold entry {k!r} has unknown tag {t!r}")
                if v is not None and any(c not in US_VOCAB for c in v):
                    raise SystemExit(f"gold entry {k!r}/{t} has phonemes outside the US vocab")
        else:
            raise SystemExit(f"gold entry {k!r} is neither a string nor a dict")


def check_keys(d, name):
    for k in d:
        if not k.isascii():
            # The C++ reader compares raw bytes, so non-ASCII keys would still
            # work, but misaki's own data has none; flag a changed input.
            raise SystemExit(f"{name}: non-ASCII key {k!r}; the pinned data has none")


def build(gold_bytes, silver_bytes):
    gold = json.loads(gold_bytes.decode("utf-8"))
    silver = json.loads(silver_bytes.decode("utf-8"))
    check_keys(gold, "gold")
    check_keys(silver, "silver")
    gold = grow_dictionary(gold)
    silver = grow_dictionary(silver)
    check_gold(gold)
    for k, v in silver.items():
        if not isinstance(v, str):
            raise SystemExit(f"silver entry {k!r} is not a string")

    strings = set()
    for table in (gold, silver):
        for k, v in table.items():
            strings.add(k)
            if isinstance(v, str):
                strings.add(v)
            else:
                for x in v.values():
                    if x is not None:
                        strings.add(x)
    pool = bytearray()
    where = {}
    for s in sorted(strings, key=lambda s: s.encode("utf-8")):
        b = s.encode("utf-8")
        if len(b) > 0xFFFF:
            raise SystemExit(f"string too long: {s[:40]!r}")
        where[s] = (len(pool), len(b))
        pool += b

    variants = bytearray()
    nvar = 0

    def entries_for(table):
        nonlocal nvar
        out = bytearray()
        for k in sorted(table, key=lambda s: s.encode("utf-8")):
            v = table[k]
            koff, klen = where[k]
            first = nvar
            if isinstance(v, str):
                items = [("PLAIN", v)]
            else:
                items = sorted(v.items(), key=lambda kv: TAG_ID[kv[0]])
            for t, x in items:
                if x is None:
                    variants.extend(struct.pack("<BBHI", TAG_ID[t], 1, 0, 0))
                else:
                    voff, vlen = where[x]
                    variants.extend(struct.pack("<BBHI", TAG_ID[t], 0, vlen, voff))
                nvar += 1
            out.extend(struct.pack("<IHHI", koff, klen, len(items), first))
        return out

    gold_entries = entries_for(gold)
    silver_entries = entries_for(silver)

    def align(n):
        return (n + 7) & ~7

    off_gold = HEADER_SIZE
    off_silver = align(off_gold + len(gold_entries))
    off_var = align(off_silver + len(silver_entries))
    off_str = align(off_var + len(variants))
    total = off_str + len(pool)

    header = bytearray(HEADER_SIZE)
    header[0:8] = MAGIC
    struct.pack_into("<IIIIII", header, 8, VERSION, 0, len(gold), len(silver), nvar, len(pool))
    header[32:64] = hashlib.sha256(gold_bytes).digest()
    header[64:96] = hashlib.sha256(silver_bytes).digest()
    struct.pack_into("<IIIII", header, 96, off_gold, off_silver, off_var, off_str, total)

    out = bytearray(total)
    out[0:HEADER_SIZE] = header
    out[off_gold:off_gold + len(gold_entries)] = gold_entries
    out[off_silver:off_silver + len(silver_entries)] = silver_entries
    out[off_var:off_var + len(variants)] = variants
    out[off_str:off_str + len(pool)] = pool
    return bytes(out), len(gold), len(silver)


def fetch(cache_dir, pins):
    os.makedirs(cache_dir, exist_ok=True)
    got = {}
    for name in ("us_gold.json", "us_silver.json"):
        pin = pins["misaki"]["files"][f"misaki/data/{name}"]
        path = os.path.join(cache_dir, name)
        if not (os.path.exists(path) and sha256_bytes(open(path, "rb").read()) == pin["sha256"]):
            url = pins["misaki"]["raw_url"].format(commit=pins["misaki"]["commit"], path=f"misaki/data/{name}")
            print(f"fetching {url}", file=sys.stderr)
            with urllib.request.urlopen(url, timeout=120) as r:
                data = r.read()
            if sha256_bytes(data) != pin["sha256"]:
                raise SystemExit(f"{name}: downloaded sha256 {sha256_bytes(data)} != pinned {pin['sha256']}")
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)
        got[name] = path
    return got["us_gold.json"], got["us_silver.json"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold")
    ap.add_argument("--silver")
    ap.add_argument("--fetch", metavar="CACHE_DIR", help="download the pinned us_gold.json/us_silver.json into CACHE_DIR")
    ap.add_argument("-o", "--output")
    ap.add_argument("--allow-unpinned", action="store_true", help="accept inputs that differ from pins.json (tests only)")
    ap.add_argument("--verify", metavar="LEX", help="check an existing .lex against pins.json and exit")
    a = ap.parse_args()
    pins = load_pins()

    if a.verify:
        got = sha256_bytes(open(a.verify, "rb").read())
        want = pins["lexicon"]["sha256"]
        print(f"{a.verify}: {got}")
        if got != want:
            raise SystemExit(f"lexicon sha256 {got} != pinned {want}")
        return

    if a.fetch:
        a.gold, a.silver = fetch(a.fetch, pins)
    if not (a.gold and a.silver and a.output):
        ap.error("need --gold and --silver (or --fetch) and -o")
    gold_bytes = open(a.gold, "rb").read()
    silver_bytes = open(a.silver, "rb").read()
    if not a.allow_unpinned:
        for name, b in (("us_gold.json", gold_bytes), ("us_silver.json", silver_bytes)):
            pin = pins["misaki"]["files"][f"misaki/data/{name}"]
            if len(b) != pin["bytes"] or sha256_bytes(b) != pin["sha256"]:
                raise SystemExit(f"{name}: sha256 {sha256_bytes(b)} / {len(b)} B does not match the pin "
                                 f"{pin['sha256']} / {pin['bytes']} B (use --allow-unpinned for test data)")
    data, ng, ns = build(gold_bytes, silver_bytes)
    tmp = a.output + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, a.output)
    digest = sha256_bytes(data)
    print(f"{a.output}: {len(data)} B, {ng} gold + {ns} silver entries, sha256 {digest}")
    if not a.allow_unpinned and pins.get("lexicon", {}).get("sha256") not in (None, "", digest):
        raise SystemExit(f"output sha256 {digest} != pinned {pins['lexicon']['sha256']}: "
                         "the converter or its inputs changed; update pins.json deliberately")


if __name__ == "__main__":
    main()
