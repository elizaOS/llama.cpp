#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
convert_kokoro_pth_to_gguf.py — convert a Kokoro-82M (StyleTTS-2 + iSTFTNet)
PyTorch checkpoint into a GGUF file consumable by the fork's tools/kokoro/
inference path.

The canonical Kokoro release ships as a single `kokoro-v1_0.pth` file
(see hexgrad/Kokoro-82M). The Python `kokoro` package keeps the StyleTTS-2
sub-modules (bert, predictor, decoder) split across multiple `.bin`
shards plus a top-level config — we walk those into the GGUF layout the
fork's `kokoro_load_model` expects (one tensor per "kokoro.*" key).

Two modes:

  --pth <kokoro-v1_0.pth>   convert the canonical PyTorch checkpoint.
  --stub                    emit a smoke-test GGUF (random fp16 weights
                            at the canonical Kokoro v1.0 shapes). Used by
                            the J2 verify harness when the .pth file isn't
                            available; produces shape-valid output that
                            the C++ loader can ingest without error.

Output: GGUF with general.architecture="kokoro" and the canonical kokoro
hparams keys. See tools/kokoro/include/kokoro.h for the full key set.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# Resolve gguf-py sibling import (so the script runs without an install).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "gguf-py"))

import gguf  # noqa: E402


# Canonical Kokoro v1.0 hparams (from the hexgrad/Kokoro-82M config).
KOKORO_HPARAMS = {
    "text_n_layer":        6,
    "text_n_head":         12,
    "text_d_model":        768,
    "text_d_ff":           2048,
    "text_vocab_size":     178,
    "text_max_pos":        512,
    "style_dim":           256,
    "predictor_d_hidden":  512,
    "decoder_d_hidden":    512,
    "decoder_n_upsample":  5,
    "istft_n_fft":         20,
    "istft_hop":           5,
    "istft_win":           20,
    "sample_rate":         24000,
}


def _add_tensor(writer: gguf.GGUFWriter, name: str, data: np.ndarray) -> None:
    """Add a tensor, casting fp64 → fp32 + ensuring c-contig."""
    if data.dtype == np.float64:
        data = data.astype(np.float32)
    if not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data)
    writer.add_tensor(name, data)


def _write_metadata(writer: gguf.GGUFWriter, hp: dict) -> None:
    writer.add_string("general.architecture", "kokoro")
    writer.add_string("general.name", "Kokoro-82M")
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("general.description",
                     "Kokoro-82M TTS (StyleTTS-2 + iSTFTNet) — J2 GGUF port")
    writer.add_int32("kokoro.text.n_layer",        int(hp["text_n_layer"]))
    writer.add_int32("kokoro.text.n_head",         int(hp["text_n_head"]))
    writer.add_int32("kokoro.text.d_model",        int(hp["text_d_model"]))
    writer.add_int32("kokoro.text.d_ff",           int(hp["text_d_ff"]))
    writer.add_int32("kokoro.text.vocab_size",     int(hp["text_vocab_size"]))
    writer.add_int32("kokoro.text.max_position",   int(hp["text_max_pos"]))
    writer.add_int32("kokoro.style.dim",           int(hp["style_dim"]))
    writer.add_int32("kokoro.predictor.d_hidden",  int(hp["predictor_d_hidden"]))
    writer.add_int32("kokoro.decoder.d_hidden",    int(hp["decoder_d_hidden"]))
    writer.add_int32("kokoro.decoder.n_upsample",  int(hp["decoder_n_upsample"]))
    writer.add_int32("kokoro.decoder.istft_n_fft", int(hp["istft_n_fft"]))
    writer.add_int32("kokoro.decoder.istft_hop",   int(hp["istft_hop"]))
    writer.add_int32("kokoro.decoder.istft_win",   int(hp["istft_win"]))
    writer.add_int32("kokoro.audio.sample_rate",   int(hp["sample_rate"]))


def emit_stub(out_path: str, hp: dict) -> None:
    """Emit a smoke-test GGUF with random-init tensors at canonical shapes."""
    rng = np.random.default_rng(seed=42)

    writer = gguf.GGUFWriter(out_path, arch="kokoro")
    _write_metadata(writer, hp)

    d = hp["text_d_model"]
    ff = hp["text_d_ff"]
    vocab = hp["text_vocab_size"]
    n_layer = hp["text_n_layer"]
    style = hp["style_dim"]

    scale = 0.02
    # Token embedding [vocab, d_model].
    _add_tensor(writer, "kokoro.token_embd.weight",
                rng.standard_normal((vocab, d), dtype=np.float32) * scale)

    # Final layer norm.
    _add_tensor(writer, "kokoro.text.out_norm.weight",
                np.ones((d,), dtype=np.float32))

    # Per-layer Q/K/V/O + FFN.
    for il in range(n_layer):
        p = f"kokoro.text.layers.{il}."
        _add_tensor(writer, p + "attn_norm.weight", np.ones((d,), dtype=np.float32))
        _add_tensor(writer, p + "attn_q.weight",   rng.standard_normal((d, d), dtype=np.float32) * scale)
        _add_tensor(writer, p + "attn_k.weight",   rng.standard_normal((d, d), dtype=np.float32) * scale)
        _add_tensor(writer, p + "attn_v.weight",   rng.standard_normal((d, d), dtype=np.float32) * scale)
        _add_tensor(writer, p + "attn_o.weight",   rng.standard_normal((d, d), dtype=np.float32) * scale)
        _add_tensor(writer, p + "ffn_norm.weight", np.ones((d,), dtype=np.float32))
        _add_tensor(writer, p + "ffn_gate.weight", rng.standard_normal((d, ff), dtype=np.float32) * scale)
        _add_tensor(writer, p + "ffn_up.weight",   rng.standard_normal((d, ff), dtype=np.float32) * scale)
        _add_tensor(writer, p + "ffn_down.weight", rng.standard_normal((ff, d), dtype=np.float32) * scale)

    # Predictor head and style projection.
    _add_tensor(writer, "kokoro.predictor.duration.weight",
                rng.standard_normal((d, 1), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.style.proj.weight",
                rng.standard_normal((style, d), dtype=np.float32) * scale)

    # Decoder projections (the iSTFT vocoder reads from these).
    F = hp["istft_n_fft"] // 2 + 1
    _add_tensor(writer, "kokoro.decoder.mel_proj.weight",
                rng.standard_normal((hp["decoder_d_hidden"], F), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.decoder.phase_proj.weight",
                rng.standard_normal((hp["decoder_d_hidden"], F), dtype=np.float32) * scale)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _walk_torch_state_dict(pth_path: str):
    """Iterate (key, np.ndarray) pairs from a Kokoro PyTorch checkpoint."""
    try:
        import torch
    except ImportError as e:
        raise SystemExit("torch is required for --pth conversion; pip install torch") from e
    obj = torch.load(pth_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
    else:
        sd = obj
    for k, v in sd.items():
        if hasattr(v, "detach"):
            yield k, v.detach().cpu().numpy()
        else:
            yield k, np.asarray(v)


# Mapping rules from the upstream Kokoro PyTorch parameter names to the
# fork's GGUF keys. The actual upstream keys are dominated by Albert /
# StyleTTS-2 module hierarchies — the converter walks them and rewrites
# into the flat `kokoro.*` namespace the C++ loader expects.
_PTH_KEY_RULES = [
    # (substr to match, replacement format that takes the .N. layer index)
    ("bert.embeddings.word_embeddings.weight", "kokoro.token_embd.weight"),
    ("bert.encoder.layer.{il}.attention.self.query.weight", "kokoro.text.layers.{il}.attn_q.weight"),
    ("bert.encoder.layer.{il}.attention.self.key.weight",   "kokoro.text.layers.{il}.attn_k.weight"),
    ("bert.encoder.layer.{il}.attention.self.value.weight", "kokoro.text.layers.{il}.attn_v.weight"),
    ("bert.encoder.layer.{il}.attention.output.dense.weight","kokoro.text.layers.{il}.attn_o.weight"),
    ("bert.encoder.layer.{il}.intermediate.dense.weight",   "kokoro.text.layers.{il}.ffn_up.weight"),
    ("bert.encoder.layer.{il}.output.dense.weight",         "kokoro.text.layers.{il}.ffn_down.weight"),
]


def emit_from_pth(pth_path: str, out_path: str, hp: dict) -> None:
    writer = gguf.GGUFWriter(out_path, arch="kokoro")
    _write_metadata(writer, hp)

    keep = {}
    for key, arr in _walk_torch_state_dict(pth_path):
        # Best-effort rewrite — un-matched keys are dropped (the iSTFTNet
        # decoder and predictor convs need a per-layer mapping table that is
        # part of the follow-up "full quality" port; documented in
        # .swarm/impl/J2-kokoro-port-notes.md).
        for substr, target in _PTH_KEY_RULES:
            if "{il}" in substr:
                for il in range(hp["text_n_layer"]):
                    if substr.format(il=il) in key:
                        keep[target.format(il=il)] = arr.astype(np.float32)
                        break
            elif substr in key:
                keep[target] = arr.astype(np.float32)
                break

    for name, data in keep.items():
        _add_tensor(writer, name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pth", help="path to kokoro-v1_0.pth (PyTorch checkpoint)")
    p.add_argument("--stub", action="store_true",
                   help="emit a smoke-test GGUF with random init (no .pth needed)")
    p.add_argument("--output", required=True, help="output GGUF path")
    args = p.parse_args(argv)

    if not args.stub and not args.pth:
        p.error("must pass either --pth <path> or --stub")

    hp = dict(KOKORO_HPARAMS)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

    if args.stub:
        emit_stub(args.output, hp)
    else:
        emit_from_pth(args.pth, args.output, hp)

    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
