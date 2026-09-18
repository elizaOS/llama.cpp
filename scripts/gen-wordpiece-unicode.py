#!/usr/bin/env python3
"""Generate WordPiece canonical decomposition, ordering, and exact Mn tables.

Python 3.14's Unicode 16 database is pinned so regeneration is offline and does
not update the separate generic tokenizer tables from the latest online UCD.
"""
import argparse
import pathlib
import unicodedata as ud


def generate():
    if ud.unidata_version != "16.0.0":
        raise RuntimeError("Use Python 3.14 with Unicode 16.0.0")
    marks = []
    decompositions = []
    combining = []
    for cp in range(0x110000):
        char = chr(cp)
        if ud.category(char) == "Mn":
            if marks and marks[-1][1] + 1 == cp:
                marks[-1] = (marks[-1][0], cp)
            else:
                marks.append((cp, cp))
        ccc = ud.combining(char)
        if ccc:
            combining.append((cp, ccc))
        if 0xAC00 <= cp <= 0xD7A3:
            continue  # Hangul decomposition is algorithmic.
        nfd = tuple(map(ord, ud.normalize("NFD", char)))
        if nfd != (cp,):
            assert len(nfd) <= 4
            decompositions.append((cp, nfd))
    lines = [
        "/** Generated Unicode 16.0.0 data for WordPiece NFD and Mn removal.",
        " * Regenerate with scripts/gen-wordpiece-unicode.py; do not edit manually.",
        " */",
        "#pragma once", "#include <cstdint>",
        "namespace wordpiece_unicode_data {",
        "struct mark_range { uint32_t first; uint32_t last; };",
        "struct combining_class { uint32_t codepoint; uint8_t value; };",
        "struct decomposition { uint32_t codepoint; uint32_t values[4]; uint8_t count; };",
        "static constexpr mark_range nonspacing_marks[] = {",
    ]
    lines += [f"    {{0x{a:X}, 0x{b:X}}}," for a, b in marks]
    lines += ["};", "static constexpr combining_class combining_classes[] = {"]
    lines += [f"    {{0x{cp:X}, {ccc}}}," for cp, ccc in combining]
    lines += ["};", "static constexpr decomposition decompositions[] = {"]
    for cp, nfd in decompositions:
        values = ", ".join(f"0x{x:X}" for x in nfd)
        lines.append(f"    {{0x{cp:X}, {{{values}}}, {len(nfd)}}},")
    lines += ["};", "} // namespace wordpiece_unicode_data", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = pathlib.Path(__file__).resolve().parents[1] / "src/unicode-wordpiece-data.h"
    generated = generate()
    if args.check:
        if output.read_text() != generated:
            raise SystemExit("WordPiece Unicode data is stale; regenerate with Python 3.14")
    else:
        output.write_text(generated)
