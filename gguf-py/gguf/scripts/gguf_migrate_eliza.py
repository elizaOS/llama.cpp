"""Rewrite legacy Eliza GGUF type IDs without changing quantized tensor bytes."""
import argparse
from pathlib import Path

from gguf import GGUFReader, GGUFWriter
from gguf.scripts.gguf_new_metadata import copy_with_new_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--legacy-type-ids", action="store_true", required=True,
                        help="Confirm that ID 42 means Eliza Q1_0_g32 in this file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    reader = GGUFReader(args.input, eliza_legacy_type_ids=True)
    architecture = reader.fields.get("general.architecture")
    if architecture is None:
        parser.error("input has no general.architecture")
    writer = GGUFWriter(args.output, architecture.contents(), endianess=reader.endianess)
    copy_with_new_metadata(reader, writer, {}, [])


if __name__ == "__main__":
    main()
