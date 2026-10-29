#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
convert_paradee_pt_to_gguf.py - convert the Paradee-8M v1.0 PyTorch weights
(sahilmahendrakar/Paradee-8M-v1.0, Apache-2.0) into the GGUF that
tools/paradee loads.

Paradee is Kokoro-82M's own StyleTTS-2 + iSTFTNet code at smaller widths,
distilled to one voice (af_heart). It ships two checkpoints:

  pytorch/text_side.pt   {'model': state_dict}  TextStudent preset 's+mlp'
  pytorch/decoder.pt     {'model': state_dict}  StudentDecoder preset 'A'

Each checkpoint holds its own learned constant style vector (text 32-d,
decoder 16-d); there is no voice .bin. The model's published runtime graph is
ONNX; this converter reads the PyTorch weights instead so that inference runs
natively in this fork.

Every input is checked against tools/paradee/pins.json (size + sha256) before
torch.load(weights_only=True) touches it. weight_norm pairs are folded with
torch._weight_norm, the same function PyTorch uses when it removes the
parametrisation, so the GGUF weights are bit-identical to the reference
module's effective weights. Every state-dict key must be consumed (only the
unused ALBERT pooler is skipped), and every tensor shape is checked against
the hparams written to the GGUF.

The output is byte-reproducible: the same inputs and the same converter
produce the same file. pins.json records the expected sha256. The metadata
records the input pins and the git blob sha of this script (not the commit
sha, which would make the expected output depend on the commit that pins it).

--version 1.1 converts sahilmahendrakar/Paradee-8M-v1.1 (pins.json "v1.1").
v1.1 keeps v1.0's text side (the same text_side.pt) and the decoder's AdaIN
trunk (retrained weights, same layout), and replaces the iSTFTNet generator
and the phase-locking filter with a frame-rate source-filter head
(training/scripts/vhead.py SourceFilterHead2). The GGUF says which decoder
head it holds in paradee.decoder.head ("istftnet_lock" or "source_filter_v2");
the runtime picks the graph from that key (a file without it is v1.0).

--version 1.2 converts sahilmahendrakar/Paradee-8M-v1.2 (pins.json "v1.2").
v1.2 is v1.1 with four voices: every tensor has v1.1's layout except the two
learned styles, which become tables with one row per voice (text [4, 32],
decoder [4, 16]). Both checkpoints carry a top-level 'voices' list next to
'model', which must equal config.json's. The GGUF stores the tables as 2-D
tensors (paradee.style.text [n_voices, 32], paradee.style.decoder
[n_voices, 16]), the voice names in model order (paradee.voices), each
voice's misaki accent (paradee.voices.accent: en-us for a*_ voices, en-gb
for b*_ voices) and the default voice (paradee.voice = af_heart). v1.0 and
v1.1 files keep their 1-D styles, so their bytes do not change.

usage:
  convert_paradee_pt_to_gguf.py --model-dir <dir with pytorch/, config.json, tokenizer.json> \
      --output paradee-8m-v1.0-f32.gguf [--version 1.0|1.1|1.2] [--outtype f32|f16] [--check-output]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "gguf-py"))

import gguf  # noqa: E402

PINS_PATH = _HERE / "pins.json"
ARCH = "paradee"
NS = "paradee"

# Architecture of Paradee-8M v1.0, read from the checkpoint shapes and from
# training/scripts/{student.py, student_decoder.py} at the pinned source
# commit. The converter checks every tensor against these values.
# HPARAMS_V11 below is the v1.1 file's set.
HPARAMS: dict[str, Any] = {
    # CustomAlbert (transformers AlbertConfig, num_hidden_groups=1).
    "text.n_layer": 6,
    "text.n_head": 4,
    "text.d_model": 256,
    "text.d_ff": 768,
    "text.embd_dim": 128,
    "text.vocab_size": 178,
    "text.max_position": 512,
    # ProsodyPredictor / TextEncoder.
    "hidden_dim": 192,
    "predictor.n_layer": 3,
    "predictor.max_dur": 50,
    "text_encoder.kernel": 5,
    # Styles (learned constants, one per half).
    "style.text_dim": 32,
    "style.decoder_dim": 16,
    # asr_proj MLP: Linear(192, 512) -> GELU -> Linear(512, 512).
    "asr.dim": 512,
    # StudentDecoder preset A + Kokoro iSTFTNet generator.
    "decoder.adain_dim": 256,
    "decoder.initial_ch": 128,
    "decoder.asr_res_dim": 64,
    "decoder.upsample_rates": [10, 6],
    "decoder.upsample_kernels": [20, 12],
    "decoder.resblock_kernels": [3, 7, 11],
    "decoder.resblock_dilations": [1, 3, 5, 1, 3, 5, 1, 3, 5],
    "decoder.istft_n_fft": 20,
    "decoder.istft_hop": 5,
    "decoder.harmonic_num": 8,
    "decoder.sine_amp": 0.1,
    "decoder.voiced_threshold": 10.0,
    "audio.sample_rate": 24000,
    # Phase-locking filter (training/scripts/phase_lock.py, export_paradee.py Lock(L=33)).
    "lock.n_fft": 1024,
    "lock.hop": 256,
    "lock.band_lo_hz": 2000,
    "lock.band_hi_hz": 8000,
    "lock.smooth_frames": 33,
    "lock.voiced_f0_hz": 60.0,
    "lock.f0_hop": 300,
    # Input contract (paradee/tts.py): [0] + ids + [0], at most 512 ids.
    "input.pad_id": 0,
    "input.max_ids": 512,
}

# Paradee-8M v1.1: the same text side and AdaIN trunk, then SourceFilterHead2
# (training/scripts/vhead.py at the pinned v1.1 source commit; constructor
# defaults, as VDecoder(head="v2") builds it). No generator, no phase lock.
_V10_GENERATOR_KEYS = ("decoder.upsample_rates", "decoder.upsample_kernels", "decoder.resblock_kernels",
                       "decoder.resblock_dilations", "decoder.istft_n_fft", "decoder.istft_hop",
                       "decoder.harmonic_num", "decoder.sine_amp", "decoder.voiced_threshold")
HPARAMS_V11: dict[str, Any] = {
    **{k: v for k, v in HPARAMS.items() if not k.startswith("lock.") and k not in _V10_GENERATOR_KEYS},
    "head.dim": 160,               # ConvNeXt width
    "head.blocks": 4,              # ConvNeXt blocks (depthwise 7, LayerNorm, Linear x3, GELU, Linear, gamma)
    "head.kernel": 7,              # inp and depthwise kernel
    "head.mult": 3,                # ConvNeXt expansion
    "head.f0_hop": 300,            # samples per trunk frame / F0 value (80 fps)
    "head.n_fft": 600,             # main STFT: periodic Hann, centred, reflect
    "head.hop": 150,               # 2 STFT frames per trunk frame
    "head.harmonics": 64,          # sine bank K
    "head.sine_amp": 0.1,
    "head.max_harmonic_hz": 11000.0,
    "head.voiced_f0_hz": 10.0,     # sine bank / pulsing noise mute below this
    "head.min_f0_hz": 60.0,        # F0 floor of the pulse-shift ratio
    "head.mod": 2,                 # pulsing noise: cos/sin(m * phi), m = 1..mod
    "head.points": 80,             # G: gains predicted at 80 mel-spaced points
    "head.gain_clamp": 6.0,        # exp(min(x, 6))
    "head.shift_scale": 0.5,       # pulse shift = 0.5 * tanh(x) cycles
    "head.burst_n_fft": 150,       # noise-burst STFT (window 150, hop 75)
    "head.burst_hop": 75,
    "head.burst_points": 24,       # Gb
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def check_pins(model_dir: Path, pins: dict) -> None:
    for rel in ("pytorch/text_side.pt", "pytorch/decoder.pt", "config.json", "tokenizer.json"):
        pin = pins["files"][rel]
        path = model_dir / rel
        if not path.is_file():
            raise SystemExit(f"missing input {path}")
        size = path.stat().st_size
        if size != pin["size"]:
            raise SystemExit(f"{rel}: size {size} != pinned {pin['size']}")
        digest = sha256_file(path)
        if digest != pin["sha256"]:
            raise SystemExit(f"{rel}: sha256 {digest} != pinned {pin['sha256']}")


def load_vocab(model_dir: Path, version: str = "1.0") -> list[tuple[str, int]]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tok = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = config["vocab"]
    if config.get("version") != version:
        raise SystemExit(f"config.json version {config.get('version')!r} != {version!r}")
    if config.get("sample_rate") != HPARAMS["audio.sample_rate"]:
        raise SystemExit(f"config.json sample_rate {config.get('sample_rate')} != 24000")
    if config.get("n_token") != HPARAMS["text.vocab_size"]:
        raise SystemExit(f"config.json n_token {config.get('n_token')} != 178")
    # tokenizer.json is the same vocabulary plus the '$' pad symbol at id 0.
    tok_vocab = dict(tok["model"]["vocab"])
    if tok_vocab.pop("$", None) != 0 or tok_vocab != vocab:
        raise SystemExit("tokenizer.json vocabulary differs from config.json")
    ids = sorted(vocab.values())
    if len(set(ids)) != len(ids) or ids[0] < 1 or ids[-1] >= HPARAMS["text.vocab_size"]:
        raise SystemExit("config.json vocabulary ids are not unique in 1..177")
    for sym in vocab:
        if len(sym) != 1:
            raise SystemExit(f"vocabulary symbol {sym!r} is not one code point")
    return sorted(vocab.items(), key=lambda kv: kv[1])


def fold_weight_norm(sd: dict) -> dict:
    """Fold weight_g/weight_v into weight with torch._weight_norm(v, g, 0). A state dict without
    weight_norm pairs (for example one read from ONNX, where the weights are already folded) passes
    through unchanged and does not need torch."""
    out = {}
    pairs: dict[str, dict] = {}
    for k, v in sd.items():
        if k.endswith(".weight_v"):
            pairs.setdefault(k[: -len(".weight_v")], {})["v"] = v
        elif k.endswith(".weight_g"):
            pairs.setdefault(k[: -len(".weight_g")], {})["g"] = v
        else:
            out[k] = v
    if pairs:
        import torch
    for base, p in pairs.items():
        if set(p) != {"v", "g"}:
            raise SystemExit(f"incomplete weight_norm pair for {base}")
        out[base + ".weight"] = torch._weight_norm(p["v"], p["g"], 0)
    return out


def lstm_names(src: str, dst: str) -> list[tuple[str, str]]:
    out = []
    for kind in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        out.append((f"{src}.{kind}_l0", f"{dst}.{kind}_l0"))
        out.append((f"{src}.{kind}_l0_reverse", f"{dst}.{kind}_l0_r"))
    return out


def has_weight(sd: dict, base: str) -> bool:
    return f"{base}.weight" in sd or f"{base}.weight_v" in sd


def adain_resblk_names(src: str, dst: str, sd: dict) -> list[tuple[str, str]]:
    """AdainResBlk1d (kokoro istftnet): conv1/conv2, AdaIN1d norm1/norm2 (fc +
    the trained InstanceNorm affine), optional conv1x1 (bias=False) and pool."""
    out = []
    for conv in ("conv1", "conv2"):
        out += [(f"{src}.{conv}.weight", f"{dst}.{conv}.weight"), (f"{src}.{conv}.bias", f"{dst}.{conv}.bias")]
    for norm in ("norm1", "norm2"):
        for part in ("fc.weight", "fc.bias", "norm.weight", "norm.bias"):
            out.append((f"{src}.{norm}.{part}", f"{dst}.{norm}.{part}"))
    if has_weight(sd, f"{src}.conv1x1"):
        out.append((f"{src}.conv1x1.weight", f"{dst}.conv1x1.weight"))
    if has_weight(sd, f"{src}.pool"):
        out += [(f"{src}.pool.weight", f"{dst}.pool.weight"), (f"{src}.pool.bias", f"{dst}.pool.bias")]
    return out


def gen_resblock_names(src: str, dst: str) -> list[tuple[str, str]]:
    """AdaINResBlock1 (Snake): convs1/convs2 x3, adain1/adain2 x3, alpha1/alpha2 x3."""
    out = []
    for j in range(3):
        for branch in ("convs1", "convs2"):
            out += [(f"{src}.{branch}.{j}.weight", f"{dst}.{branch}.{j}.weight"),
                    (f"{src}.{branch}.{j}.bias", f"{dst}.{branch}.{j}.bias")]
        for branch in ("adain1", "adain2"):
            for part in ("fc.weight", "fc.bias", "norm.weight", "norm.bias"):
                out.append((f"{src}.{branch}.{j}.{part}", f"{dst}.{branch}.{j}.{part}"))
        out += [(f"{src}.alpha1.{j}", f"{dst}.alpha1.{j}"), (f"{src}.alpha2.{j}", f"{dst}.alpha2.{j}")]
    return out


def text_side_map(sd: dict) -> list[tuple[str, str]]:
    m = [
        ("bert.embeddings.word_embeddings.weight", f"{NS}.bert.token_embd.weight"),
        ("bert.embeddings.position_embeddings.weight", f"{NS}.bert.position_embd.weight"),
        ("bert.embeddings.token_type_embeddings.weight", f"{NS}.bert.tok_type_embd.weight"),
        ("bert.embeddings.LayerNorm.weight", f"{NS}.bert.embd_ln.weight"),
        ("bert.embeddings.LayerNorm.bias", f"{NS}.bert.embd_ln.bias"),
        ("bert.encoder.embedding_hidden_mapping_in.weight", f"{NS}.bert.embd_proj.weight"),
        ("bert.encoder.embedding_hidden_mapping_in.bias", f"{NS}.bert.embd_proj.bias"),
        ("bert_encoder.weight", f"{NS}.bert_encoder.weight"),
        ("bert_encoder.bias", f"{NS}.bert_encoder.bias"),
    ]
    layer = "bert.encoder.albert_layer_groups.0.albert_layers.0."
    for src, dst in (("attention.query", "attn_q"), ("attention.key", "attn_k"), ("attention.value", "attn_v"),
                     ("attention.dense", "attn_o"), ("attention.LayerNorm", "attn_ln"),
                     ("full_layer_layer_norm", "full_ln"), ("ffn", "ffn"), ("ffn_output", "ffn_out")):
        m += [(f"{layer}{src}.weight", f"{NS}.bert.layer.{dst}.weight"), (f"{layer}{src}.bias", f"{NS}.bert.layer.{dst}.bias")]
    for i in range(HPARAMS["predictor.n_layer"]):
        m += lstm_names(f"predictor.text_encoder.lstms.{2 * i}", f"{NS}.predictor.de.lstm{i}")
        m += [(f"predictor.text_encoder.lstms.{2 * i + 1}.fc.weight", f"{NS}.predictor.de.adaln{i}.fc.weight"),
              (f"predictor.text_encoder.lstms.{2 * i + 1}.fc.bias", f"{NS}.predictor.de.adaln{i}.fc.bias")]
    m += lstm_names("predictor.lstm", f"{NS}.predictor.lstm")
    m += [("predictor.duration_proj.linear_layer.weight", f"{NS}.predictor.duration_proj.weight"),
          ("predictor.duration_proj.linear_layer.bias", f"{NS}.predictor.duration_proj.bias")]
    m += lstm_names("predictor.shared", f"{NS}.predictor.shared")
    for chain in ("F0", "N"):
        for i in range(3):
            m += adain_resblk_names(f"predictor.{chain}.{i}", f"{NS}.predictor.{chain}.{i}", sd)
        m += [(f"predictor.{chain}_proj.weight", f"{NS}.predictor.{chain}_proj.weight"),
              (f"predictor.{chain}_proj.bias", f"{NS}.predictor.{chain}_proj.bias")]
    m.append(("text_encoder.embedding.weight", f"{NS}.text_encoder.embd.weight"))
    for i in range(HPARAMS["predictor.n_layer"]):
        m += [(f"text_encoder.cnn.{i}.0.weight", f"{NS}.text_encoder.cnn.{i}.weight"),
              (f"text_encoder.cnn.{i}.0.bias", f"{NS}.text_encoder.cnn.{i}.bias"),
              (f"text_encoder.cnn.{i}.1.gamma", f"{NS}.text_encoder.ln.{i}.gamma"),
              (f"text_encoder.cnn.{i}.1.beta", f"{NS}.text_encoder.ln.{i}.beta")]
    m += lstm_names("text_encoder.lstm", f"{NS}.text_encoder.lstm")
    m += [("asr_proj.0.weight", f"{NS}.asr_proj.0.weight"), ("asr_proj.0.bias", f"{NS}.asr_proj.0.bias"),
          ("asr_proj.2.weight", f"{NS}.asr_proj.1.weight"), ("asr_proj.2.bias", f"{NS}.asr_proj.1.bias")]
    m.append(("style", f"{NS}.style.text"))
    return m


TEXT_SIDE_SKIPPED = ("bert.pooler.weight", "bert.pooler.bias")  # ALBERT pooler: not used by TextStudent


def decoder_map(sd: dict) -> list[tuple[str, str]]:
    m = adain_resblk_names("encode", f"{NS}.decoder.encode", sd)
    for i in range(4):
        m += adain_resblk_names(f"decode.{i}", f"{NS}.decoder.decode.{i}", sd)
    for conv in ("F0_conv", "N_conv"):
        m += [(f"{conv}.weight", f"{NS}.decoder.{conv}.weight"), (f"{conv}.bias", f"{NS}.decoder.{conv}.bias")]
    m += [("asr_res.0.weight", f"{NS}.decoder.asr_res.weight"), ("asr_res.0.bias", f"{NS}.decoder.asr_res.bias")]
    g = "generator"
    m += [(f"{g}.m_source.l_linear.weight", f"{NS}.gen.m_source.l_linear.weight"),
          (f"{g}.m_source.l_linear.bias", f"{NS}.gen.m_source.l_linear.bias")]
    for i in range(2):
        for part in ("weight", "bias"):
            m.append((f"{g}.noise_convs.{i}.{part}", f"{NS}.gen.noise_convs.{i}.{part}"))
            m.append((f"{g}.ups.{i}.{part}", f"{NS}.gen.ups.{i}.{part}"))
        m += gen_resblock_names(f"{g}.noise_res.{i}", f"{NS}.gen.noise_res.{i}")
    for i in range(6):
        m += gen_resblock_names(f"{g}.resblocks.{i}", f"{NS}.gen.resblocks.{i}")
    m += [(f"{g}.conv_post.weight", f"{NS}.gen.conv_post.weight"), (f"{g}.conv_post.bias", f"{NS}.gen.conv_post.bias")]
    m.append(("style", f"{NS}.style.decoder"))
    return m


def decoder_map_v11(sd: dict) -> list[tuple[str, str]]:
    """v1.1 decoder.pt: v1.0's AdaIN trunk + head.* (SourceFilterHead2)."""
    m = [p for p in decoder_map(sd) if not p[0].startswith("generator.")]
    hp = HPARAMS_V11
    for part in ("inp.weight", "inp.bias", "norm0.weight", "norm0.bias", "norm1.weight", "norm1.bias",
                 "out.weight", "out.bias", "out_b.weight", "out_b.bias"):
        m.append((f"head.{part}", f"{NS}.head.{part}"))
    for i in range(hp["head.blocks"]):
        for part in ("dw.weight", "dw.bias", "norm.weight", "norm.bias", "pw1.weight", "pw1.bias",
                     "pw2.weight", "pw2.bias", "gamma"):
            m.append((f"head.blocks.{i}.{part}", f"{NS}.head.blocks.{i}.{part}"))
    return m


def expected_shapes_v11() -> dict[str, tuple[int, ...]]:
    h = HPARAMS_V11
    D, uic, K = h["head.dim"], h["decoder.initial_ch"], h["head.kernel"]
    up = h["head.f0_hop"] // h["head.hop"]
    C = 4 + 2 * h["head.mod"]
    shapes = {k: v for k, v in expected_shapes().items() if ".gen." not in k}
    shapes.update({
        f"{NS}.head.inp.weight": (D, uic, K),
        f"{NS}.head.blocks.0.dw.weight": (D, 1, K),
        f"{NS}.head.blocks.0.pw1.weight": (D * h["head.mult"], D),
        f"{NS}.head.blocks.0.pw2.weight": (D, D * h["head.mult"]),
        f"{NS}.head.blocks.0.gamma": (D,),
        f"{NS}.head.out.weight": (up * (C * h["head.points"] + 1), D),
        f"{NS}.head.out_b.weight": ((h["head.f0_hop"] // h["head.burst_hop"]) * h["head.burst_points"], D),
    })
    return shapes


def expected_shapes_v12(n_voices: int) -> dict[str, tuple[int, ...]]:
    """v1.2: v1.1's shapes, with the two styles as [n_voices, dim] tables."""
    h = HPARAMS_V11
    shapes = expected_shapes_v11()
    shapes[f"{NS}.style.text"] = (n_voices, h["style.text_dim"])
    shapes[f"{NS}.style.decoder"] = (n_voices, h["style.decoder_dim"])
    return shapes


def voice_accent(name: str) -> str:
    """misaki accent a Paradee voice was trained on: Kokoro's a*_ voices are American English (lang_code
    'a'), b*_ voices British English (lang_code 'b'); paradee/tts.py picks the G2P the same way."""
    if len(name) > 3 and name[2] == "_" and name[0] in "ab" and name[1] in "fm":
        return "en-us" if name[0] == "a" else "en-gb"
    raise SystemExit(f"voice {name!r} is not a Kokoro English voice name (a/b, f/m, '_')")


def expected_shapes() -> dict[str, tuple[int, ...]]:
    """Shapes (PyTorch order) of the tensors whose widths the C++ graph reads
    from hparams. Everything else is checked for consistency at load time."""
    h = HPARAMS
    d, e, ff, hid = h["text.d_model"], h["text.embd_dim"], h["text.d_ff"], h["hidden_dim"]
    st, sd_, asr = h["style.text_dim"], h["style.decoder_dim"], h["asr.dim"]
    W, uic = h["decoder.adain_dim"], h["decoder.initial_ch"]
    return {
        f"{NS}.bert.token_embd.weight": (h["text.vocab_size"], e),
        f"{NS}.bert.position_embd.weight": (h["text.max_position"], e),
        f"{NS}.bert.embd_proj.weight": (d, e),
        f"{NS}.bert.layer.attn_q.weight": (d, d),
        f"{NS}.bert.layer.ffn.weight": (ff, d),
        f"{NS}.bert_encoder.weight": (hid, d),
        f"{NS}.predictor.de.lstm0.weight_ih_l0": (2 * hid, hid + st),
        f"{NS}.predictor.de.adaln0.fc.weight": (2 * hid, st),
        f"{NS}.predictor.duration_proj.weight": (h["predictor.max_dur"], hid),
        f"{NS}.predictor.F0.1.conv1.weight": (hid // 2, hid, 3),
        f"{NS}.predictor.F0_proj.weight": (1, hid // 2, 1),
        f"{NS}.text_encoder.embd.weight": (h["text.vocab_size"], hid),
        f"{NS}.text_encoder.cnn.0.weight": (hid, hid, h["text_encoder.kernel"]),
        f"{NS}.asr_proj.0.weight": (asr, hid),
        f"{NS}.asr_proj.1.weight": (asr, asr),
        f"{NS}.style.text": (st,),
        f"{NS}.style.decoder": (sd_,),
        f"{NS}.decoder.encode.conv1.weight": (W, asr + 2, 3),
        f"{NS}.decoder.decode.0.conv1.weight": (W, W + h["decoder.asr_res_dim"] + 2, 3),
        f"{NS}.decoder.decode.3.conv1.weight": (uic, W + h["decoder.asr_res_dim"] + 2, 3),
        f"{NS}.decoder.asr_res.weight": (h["decoder.asr_res_dim"], asr, 1),
        f"{NS}.decoder.encode.norm1.fc.weight": (2 * (asr + 2), sd_),
        f"{NS}.gen.ups.0.weight": (uic, uic // 2, h["decoder.upsample_kernels"][0]),
        f"{NS}.gen.ups.1.weight": (uic // 2, uic // 4, h["decoder.upsample_kernels"][1]),
        f"{NS}.gen.noise_convs.0.weight": (uic // 2, h["decoder.istft_n_fft"] + 2, 2 * h["decoder.upsample_rates"][1]),
        f"{NS}.gen.noise_convs.1.weight": (uic // 4, h["decoder.istft_n_fft"] + 2, 1),
        f"{NS}.gen.conv_post.weight": (h["decoder.istft_n_fft"] + 2, uic // 4, 7),
        f"{NS}.gen.m_source.l_linear.weight": (1, h["decoder.harmonic_num"] + 1),
    }


def collect(sd_raw: dict, mapping: list[tuple[str, str]], skipped: tuple[str, ...], label: str,
            style_table: bool = False) -> dict[str, np.ndarray]:
    sd = fold_weight_norm(sd_raw)
    out: dict[str, np.ndarray] = {}
    used = set()
    for src, dst in mapping:
        if src not in sd:
            raise SystemExit(f"{label}: missing tensor {src}")
        t = sd[src]
        if isinstance(t, np.ndarray):
            if t.dtype.kind != "f":
                raise SystemExit(f"{label}: {src} is not floating point")
            arr = t.astype(np.float32)
        else:
            t = t.detach().cpu().contiguous()
            if t.dtype.is_floating_point is False:
                raise SystemExit(f"{label}: {src} is not floating point")
            arr = t.float().numpy()
        if dst.endswith(".style.text") or dst.endswith(".style.decoder"):
            # v1.0 / v1.1: one voice, stored 1-D. v1.2: [n_voices, dim], one row per voice.
            arr = arr.reshape(arr.shape[0], -1) if style_table else arr.reshape(-1)
        out[dst] = np.ascontiguousarray(arr, dtype=np.float32)
        used.add(src)
    leftover = sorted(set(sd) - used - set(skipped))
    if leftover:
        raise SystemExit(f"{label}: unconsumed tensors: {leftover[:8]} ({len(leftover)} total)")
    return out


# paradee.decoder.head values. The runtime treats a file without the key as
# HEAD_V10 (GGUFs written before v1.1 support).
HEAD_V10 = "istftnet_lock"
HEAD_V11 = "source_filter_v2"


def write_gguf(out_path: Path, tensors: dict[str, np.ndarray], vocab: list[tuple[str, int]], pins: dict,
               outtype: str, version: str = "1.0", voices: list[str] | None = None) -> None:
    v11 = version in ("1.1", "1.2")
    voices = voices or ["af_heart"]
    writer = gguf.GGUFWriter(str(out_path), arch=ARCH)
    writer.add_string("general.name", f"Paradee-8M v{version}")
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("general.source.url", f"https://huggingface.co/{pins['model']['repo']}")
    writer.add_string("general.base_model.0.name", "Kokoro-82M")
    who = (f"{len(voices)}-voice ({', '.join(voices)})" if len(voices) > 1 else f"single-voice ({voices[0]})")
    writer.add_string("general.description",
                      f"Paradee-8M v{version} {who} TTS, distilled from Kokoro-82M. " +
                      ("StyleTTS-2 text side + AdaIN decoder trunk + frame-rate source-filter head." if v11 else
                       "StyleTTS-2 text side + iSTFTNet decoder + phase-locking filter."))
    writer.add_string(f"{NS}.version", version)
    writer.add_string(f"{NS}.decoder.head", HEAD_V11 if v11 else HEAD_V10)
    for key, val in (HPARAMS_V11 if v11 else HPARAMS).items():
        full = f"{NS}.{key}"
        if isinstance(val, list):
            writer.add_array(full, [int(x) for x in val])
        elif isinstance(val, float):
            writer.add_float32(full, val)
        else:
            writer.add_int32(full, int(val))
    writer.add_string(f"{NS}.voice", voices[0])
    if version == "1.2":
        # The style tables' rows, in model order (config.json "voices"), and each row's misaki accent.
        writer.add_array(f"{NS}.voices", list(voices))
        writer.add_array(f"{NS}.voices.accent", [voice_accent(v) for v in voices])
    writer.add_string(f"{NS}.phoneme_format", "misaki")
    writer.add_string(f"{NS}.input.pad_symbol", "$")
    writer.add_array(f"{NS}.vocab.symbols", [s for s, _ in vocab])
    writer.add_array(f"{NS}.vocab.ids", [int(i) for _, i in vocab])

    model = pins["model"]
    writer.add_string(f"{NS}.pins.hf_repo", model["repo"])
    writer.add_string(f"{NS}.pins.hf_revision", model["revision"])
    writer.add_string(f"{NS}.pins.hf_tag", model["tag"])
    for rel in ("pytorch/text_side.pt", "pytorch/decoder.pt", "config.json", "tokenizer.json"):
        writer.add_string(f"{NS}.pins.sha256.{rel}", pins["files"][rel]["sha256"])
    writer.add_string(f"{NS}.pins.source_repo", pins["source"]["repo"])
    writer.add_string(f"{NS}.pins.source_commit", pins["source"]["commit"])
    writer.add_string(f"{NS}.converter.git_blob_sha1", git_blob_sha1(Path(__file__)))
    writer.add_string(f"{NS}.converter.outtype", outtype)

    for name in sorted(tensors):
        arr = tensors[name]
        if outtype == "f16" and arr.ndim >= 2 and not name.endswith((".style.text", ".style.decoder")):
            arr = arr.astype(np.float16)
        writer.add_tensor(name, arr)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path,
                    help="local snapshot of the pinned HF revision (pytorch/*.pt, config.json, tokenizer.json)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--version", choices=("1.0", "1.1", "1.2"), default="1.0",
                    help="Paradee-8M release: 1.0 (default), 1.1 (source-filter head) or 1.2 (1.1 with four "
                         "voices); "
                         "selects the pins.json section the inputs are checked against")
    ap.add_argument("--outtype", choices=("f32", "f16"), default="f32",
                    help="f32 (default, the parity-gated file) or f16 for >=2-D weights "
                         "(experimental: fails test-paradee-parity, not pinned)")
    ap.add_argument("--check-output", action="store_true",
                    help="fail unless the output sha256 equals the one pinned for this outtype")
    args = ap.parse_args(argv)

    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    if args.version in ("1.1", "1.2"):
        pins = pins[f"v{args.version}"]
    check_pins(args.model_dir, pins)
    vocab = load_vocab(args.model_dir, args.version)
    v12 = args.version == "1.2"
    voices = None
    if v12:
        voices = json.loads((args.model_dir / "config.json").read_text(encoding="utf-8")).get("voices")
        if (not isinstance(voices, list) or not voices or voices[0] != "af_heart" or
                len(set(voices)) != len(voices) or not all(isinstance(v, str) for v in voices)):
            raise SystemExit(f"config.json voices {voices!r}: expected a list of unique names starting with af_heart")
        for v in voices:
            voice_accent(v)

    import torch

    text_sd = torch.load(args.model_dir / "pytorch/text_side.pt", map_location="cpu", weights_only=True)
    dec_sd = torch.load(args.model_dir / "pytorch/decoder.pt", map_location="cpu", weights_only=True)
    want_keys = {"model", "voices"} if v12 else {"model"}
    if set(text_sd) != want_keys or set(dec_sd) != want_keys:
        raise SystemExit(f"expected {sorted(want_keys)} checkpoints, got {sorted(text_sd)} / {sorted(dec_sd)}")
    if v12:
        for label, ck in (("text_side.pt", text_sd), ("decoder.pt", dec_sd)):
            if list(ck["voices"]) != voices:
                raise SystemExit(f"{label}: voices {ck['voices']!r} != config.json voices {voices!r}")
    text_sd, dec_sd = text_sd["model"], dec_sd["model"]

    tensors = collect(text_sd, text_side_map(text_sd), TEXT_SIDE_SKIPPED, "text_side.pt", style_table=v12)
    dec_map = decoder_map_v11(dec_sd) if args.version in ("1.1", "1.2") else decoder_map(dec_sd)
    dec = collect(dec_sd, dec_map, (), "decoder.pt", style_table=v12)
    overlap = set(tensors) & set(dec)
    if overlap:
        raise SystemExit(f"tensor name collision: {sorted(overlap)}")
    tensors.update(dec)

    shapes = (expected_shapes_v12(len(voices)) if v12 else
              expected_shapes_v11() if args.version == "1.1" else expected_shapes())
    for name, shape in shapes.items():
        if tuple(tensors[name].shape) != shape:
            raise SystemExit(f"{name}: shape {tuple(tensors[name].shape)} != expected {shape}")
    n_params = sum(int(a.size) for a in tensors.values())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(args.output, tensors, vocab, pins, args.outtype, args.version, voices)
    digest = sha256_file(args.output)
    print(f"wrote {args.output}: {len(tensors)} tensors, {n_params} parameters, "
          f"{args.output.stat().st_size} bytes, sha256 {digest}")

    if args.check_output:
        want = pins["gguf"].get(args.outtype, {}).get("sha256")
        if digest != want:
            print(f"error: sha256 {digest} != pinned {want}", file=sys.stderr)
            return 1
        print("output matches pins.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
