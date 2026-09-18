#!/usr/bin/env python3
"""Generate WordPiece lowercase, canonical decomposition, ordering, and Mn tables.

Use Python 3.14 and --derived-core-properties with the pinned Unicode 16 file
from https://www.unicode.org/Public/16.0.0/ucd/DerivedCoreProperties.txt.
Generation is offline; the supplied UCD bytes are verified before use.
"""
import argparse
import hashlib
import pathlib
import unicodedata as ud


def generate(properties_path):
    properties_bytes = properties_path.read_bytes()
    expected = "39d35161f2954497f69e08bdb9e701493f476a3d30222de20028feda36c1dabd"
    if hashlib.sha256(properties_bytes).hexdigest() != expected:
        raise RuntimeError("DerivedCoreProperties must contain the pinned Unicode 16 bytes")
    properties = {"Cased": [], "Case_Ignorable": []}
    for line in properties_bytes.decode("utf-8").splitlines():
        fields = line.split("#", 1)[0].strip().split(";")
        if len(fields) != 2 or fields[1].strip() not in properties:
            continue
        bounds = [int(cp, 16) for cp in fields[0].strip().split("..")]
        properties[fields[1].strip()].append((bounds[0], bounds[-1]))
    for ranges in properties.values():
        assert ranges and all(a <= b for a, b in ranges)
        assert all(ranges[i - 1][1] < ranges[i][0] for i in range(1, len(ranges)))
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
    for name, ranges in [("controls", controls), ("spaces", spaces), ("cased", properties["Cased"]),
                         ("case_ignorable", properties["Case_Ignorable"])]:
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
    parser.add_argument("--derived-core-properties", type=pathlib.Path, required=True)
    args = parser.parse_args()
    output = pathlib.Path(__file__).resolve().parents[1] / "src/unicode-wordpiece-data.h"
    generated = generate(args.derived_core_properties)
    if args.check:
        if output.read_text() != generated:
            raise SystemExit("WordPiece Unicode data is stale; regenerate with Python 3.14")
    else:
        output.write_text(generated)
