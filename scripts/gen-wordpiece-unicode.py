#!/usr/bin/env python3
"""Generate WordPiece lowercase, canonical decomposition, ordering, and Mn tables.

Use Python 3.14 with its pinned Unicode 16 database; generation is offline.
"""
import argparse
import pathlib
import unicodedata as ud


def generate():
    if ud.unidata_version != "16.0.0":
        raise RuntimeError("Use Python 3.14 with Unicode 16.0.0")
    marks = []
    controls = []
    spaces = []
    def add_range(table, cp):
        if table and table[-1][1] + 1 == cp:
            table[-1] = (table[-1][0], cp)
        else:
            table.append((cp, cp))
    decompositions = []
    lowercase = []
    combining = []
    for cp in range(0x110000):
        char = chr(cp)
        category = ud.category(char)
        if category == "Mn":
            add_range(marks, cp)
        if category in ("Cc", "Cf", "Co", "Cs"):
            add_range(controls, cp)
        if category in ("Zs", "Zl", "Zp"):
            add_range(spaces, cp)
        lower = tuple(map(ord, char.lower()))
        if lower != (cp,):
            assert len(lower) <= 4
            lowercase.append((cp, lower))
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
        "/** Generated Unicode 16.0.0 data for uncased WordPiece normalization.",
        " * Regenerate with scripts/gen-wordpiece-unicode.py; do not edit manually.",
        " */",
        "#pragma once", "#include <cstdint>",
        "namespace wordpiece_unicode_data {",
        "struct codepoint_range { uint32_t first; uint32_t last; };",
        "struct combining_class { uint32_t codepoint; uint8_t value; };",
        "struct decomposition { uint32_t codepoint; uint32_t values[4]; uint8_t count; };",
        "static constexpr codepoint_range nonspacing_marks[] = {",
    ]
    lines += [f"    {{0x{a:X}, 0x{b:X}}}," for a, b in marks]
    lines += ["};"]
    for name, ranges in [("controls", controls), ("spaces", spaces)]:
        lines += [f"static constexpr codepoint_range {name}[] = {{"]
        lines += [f"    {{0x{a:X}, 0x{b:X}}}," for a, b in ranges]
        lines += ["};"]
    lines += ["static constexpr combining_class combining_classes[] = {"]
    lines += [f"    {{0x{cp:X}, {ccc}}}," for cp, ccc in combining]
    lines += ["};"]
    for name, mappings in [("lowercase_mappings", lowercase), ("decompositions", decompositions)]:
        lines += [f"static constexpr decomposition {name}[] = {{"]
        for cp, mapped in mappings:
            values = ", ".join(f"0x{x:X}" for x in mapped)
            lines.append(f"    {{0x{cp:X}, {{{values}}}, {len(mapped)}}},")
        lines += ["};"]
    lines += ["} // namespace wordpiece_unicode_data", ""]
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
