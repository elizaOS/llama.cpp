#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
convert_paradee_onnx_to_gguf.py - convert a Paradee-8M v1.1-architecture model
published only as ONNX into the GGUF that tools/paradee loads.

--family delibes: lapislabsai/Delibes-8M-esES (Apache-2.0), a Castilian Spanish
Kokoro-82M distillation (teacher voice em_alex) that uses Paradee-8M v1.1's
architecture and source code unchanged: the TextStudent 's+mlp' text side and
VDecoder(head='v2') (AdaIN trunk + SourceFilterHead2). It ships two fp32 ONNX
graphs (text and decoder) and no PyTorch checkpoint, so this converter reads
the graphs' weights:

  - named initializers carry the PyTorch module names under 'ts.' (text side)
    and 'vd.' (decoder); weight_norm is already folded;
  - the export anonymised the Linear weights it feeds to MatMul
    ('onnx::MatMul_*', stored transposed: [in, out]) and the LSTM weights
    ('onnx::LSTM_*', ONNX W [dirs, 4H, in], R [dirs, 4H, H], B [dirs, 8H] with
    gates in i, o, f, c order). Each one is traced to the node that consumes
    it, named after its module (for example '/lstms.2/LSTM'), and turned back
    into the PyTorch layout (Linear [out, in]; LSTM weight_ih / weight_hh /
    bias_ih / bias_hh in i, f, g, o order, forward and _reverse).

The result is the same state dict convert_paradee_pt_to_gguf.py builds from a
v1.1 checkpoint, mapped and shape-checked by the same tables (text_side_map,
decoder_map_v11, expected_shapes_v11). Every initializer must be consumed,
except constants the graph recomputes from the weights or builds from fixed
formulas; those are checked instead: the style copy the text graph folded
('onnx::Expand_*' == ts.style), and the decoder head's STFT bases (head.st,
head.sb), harmonic numbers (head.k, 2*pi*k), bin frequencies (head.fr) and
mel warp matrices (the constant inputs of the gain interpolation), which must
equal the ones the native head builds (vhead.py / export_paradee_v1_1.py
formulas).

Every input is checked against tools/paradee/pins.json ("delibes": size +
sha256) before it is parsed. The int8 graphs are not accepted: the GGUF is
the f32 file that test-paradee-parity gates against the fp32 ONNX graphs.

GGUF metadata, on top of the v1.1 hparams (paradee.decoder.head =
'source_filter_v2', paradee.version = '1.1'): general.name
'Delibes-8M-esES', paradee.family 'delibes', paradee.language 'es',
paradee.voice 'em_alex', paradee.phoneme_format 'espeak-es' (misaki
espeak.EspeakG2P('es') output, see tools/misaki-g2p), and the release's
validated envelope: paradee.input.validated_max_ids (360) and
paradee.validated_speed ([0.85, 1.15]). The output is byte-reproducible and
records the input pins and this script's git blob sha.

usage:
  convert_paradee_onnx_to_gguf.py --family delibes --model-dir <HF snapshot> \\
      --output delibes-8m-eses-f32.gguf [--check-output]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent / "gguf-py"))

import convert_paradee_pt_to_gguf as pt  # noqa: E402
import gguf  # noqa: E402

NS = pt.NS

# Node name of each anonymised weight's consumer -> PyTorch parameter (text side,
# without the 'ts.' prefix). The export names nodes after the module path below
# the top-level module, so the predictor's and the text encoder's LSTMs appear as
# '/lstm/LSTM' and '/lstm_1/LSTM'; the input widths tell them apart and are checked.
TEXT_MATMUL = {
    "/bert/encoder/embedding_hidden_mapping_in/MatMul": "bert.encoder.embedding_hidden_mapping_in.weight",
    "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/query/MatMul":
        "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.query.weight",
    "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/key/MatMul":
        "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.key.weight",
    "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/value/MatMul":
        "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.value.weight",
    "/bert/encoder/albert_layer_groups.0/albert_layers.0/attention/dense/MatMul":
        "bert.encoder.albert_layer_groups.0.albert_layers.0.attention.dense.weight",
    "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn/MatMul":
        "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight",
    "/bert/encoder/albert_layer_groups.0/albert_layers.0/ffn_output/MatMul":
        "bert.encoder.albert_layer_groups.0.albert_layers.0.ffn_output.weight",
    "/bert_encoder/MatMul": "bert_encoder.weight",
    "/duration_proj/linear_layer/MatMul": "predictor.duration_proj.linear_layer.weight",
    "/asr_proj/asr_proj.0/MatMul": "asr_proj.0.weight",
    "/asr_proj/asr_proj.2/MatMul": "asr_proj.2.weight",
}
# node -> (PyTorch LSTM module, expected input width)
TEXT_LSTM = {
    "/lstms.0/LSTM": ("predictor.text_encoder.lstms.0", 224),
    "/lstms.2/LSTM": ("predictor.text_encoder.lstms.2", 224),
    "/lstms.4/LSTM": ("predictor.text_encoder.lstms.4", 224),
    "/lstm/LSTM": ("predictor.lstm", 224),
    "/shared/LSTM": ("predictor.shared", 224),
    "/lstm_1/LSTM": ("text_encoder.lstm", 192),
}
DEC_MATMUL = {
    **{f"/head/blocks.{i}/pw{j}/MatMul": f"head.blocks.{i}.pw{j}.weight" for i in range(4) for j in (1, 2)},
    "/head/out/MatMul": "head.out.weight",
    "/head/out_b/MatMul": "head.out_b.weight",
}


def sha256_file(path: Path) -> str:
    return pt.sha256_file(path)


def check_pins(model_dir: Path, pins: dict) -> None:
    for rel in ("onnx/fp32/text.onnx", "onnx/fp32/decoder.onnx", "config.json"):
        pin = pins["files"][rel]
        path = model_dir / rel
        if not path.is_file():
            raise SystemExit(f"missing input {path}")
        if path.stat().st_size != pin["size"]:
            raise SystemExit(f"{rel}: size {path.stat().st_size} != pinned {pin['size']}")
        digest = sha256_file(path)
        if digest != pin["sha256"]:
            raise SystemExit(f"{rel}: sha256 {digest} != pinned {pin['sha256']}")


def load_config(model_dir: Path, pins: dict) -> tuple[dict, list[tuple[str, int]]]:
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    want = pins["config"]
    for key, value in want.items():
        if config.get(key) != value:
            raise SystemExit(f"config.json {key} {config.get(key)!r} != {value!r}")
    for prec_name in ("text", "decoder"):
        rec = config["models"]["fp32"][prec_name]
        pin = pins["files"][rec["path"]]
        if rec["sha256"] != pin["sha256"] or rec["bytes"] != pin["size"]:
            raise SystemExit(f"config.json records another {rec['path']} than pins.json")
    vocab = config["vocab"]
    ids = sorted(vocab.values())
    if len(set(ids)) != len(ids) or ids[0] < 1 or ids[-1] >= pt.HPARAMS["text.vocab_size"]:
        raise SystemExit("config.json vocabulary ids are not unique in 1..177")
    for sym in vocab:
        if len(sym) != 1:
            raise SystemExit(f"vocabulary symbol {sym!r} is not one code point")
    return config, sorted(vocab.items(), key=lambda kv: kv[1])


class Graph:
    """Initializers of one ONNX graph, and which of them the converter used."""

    def __init__(self, path: Path, prefix: str):
        import onnx
        from onnx import numpy_helper

        self.model = onnx.load(str(path))
        g = self.model.graph
        self.prefix = prefix
        self.inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
        self.consumers: dict[str, list] = {}
        for node in g.node:
            for k, name in enumerate(node.input):
                if name in self.inits:
                    self.consumers.setdefault(name, []).append((node, k))
        self.constants = {}
        for node in g.node:
            if node.op_type == "Constant":
                for a in node.attribute:
                    if a.name == "value":
                        self.constants[node.output[0]] = numpy_helper.to_array(a.t)
        self.nodes = {n.name: n for n in g.node}
        self.used: set[str] = set()

    def take(self, name: str) -> np.ndarray:
        if name not in self.inits:
            raise SystemExit(f"{self.prefix}: no initializer {name}")
        self.used.add(name)
        return self.inits[name]

    def input_of(self, node_name: str, k: int) -> str:
        node = self.nodes.get(node_name)
        if node is None:
            raise SystemExit(f"{self.prefix}: no node {node_name}")
        name = node.input[k]
        # The export shares the ALBERT weights across layers through Identity nodes; the
        # MatMul of the first layer reads the initializer itself.
        if name not in self.inits:
            raise SystemExit(f"{self.prefix}: input {k} of {node_name} is not an initializer")
        return name

    def named(self) -> dict[str, np.ndarray]:
        """Named initializers under the module prefix, prefix removed."""
        out = {}
        for name, arr in self.inits.items():
            if name.startswith(self.prefix + "."):
                out[name[len(self.prefix) + 1:]] = arr
                self.used.add(name)
        return out


def lstm_from_onnx(W: np.ndarray, R: np.ndarray, B: np.ndarray, in_dim: int, label: str) -> dict[str, np.ndarray]:
    """ONNX bidirectional LSTM weights -> PyTorch nn.LSTM(bidirectional=True) parameters."""
    if W.ndim != 3 or W.shape[0] != 2 or W.shape[2] != in_dim or W.shape[1] % 4:
        raise SystemExit(f"{label}: W shape {W.shape}, expected [2, 4H, {in_dim}]")
    H = W.shape[1] // 4
    if R.shape != (2, 4 * H, H) or B.shape != (2, 8 * H):
        raise SystemExit(f"{label}: R {R.shape} / B {B.shape} do not fit H = {H}")

    def gates(x: np.ndarray) -> np.ndarray:   # ONNX i, o, f, c -> PyTorch i, f, g, o
        i, o, f, c = np.split(x, 4, axis=0)
        return np.concatenate([i, f, c, o], axis=0)

    out = {}
    for d, suffix in ((0, "_l0"), (1, "_l0_reverse")):
        out[f"weight_ih{suffix}"] = gates(W[d])
        out[f"weight_hh{suffix}"] = gates(R[d])
        out[f"bias_ih{suffix}"] = gates(B[d, :4 * H])
        out[f"bias_hh{suffix}"] = gates(B[d, 4 * H:])
    return out


def text_state_dict(g: Graph) -> dict[str, np.ndarray]:
    sd = g.named()
    for node, key in TEXT_MATMUL.items():
        sd[key] = np.ascontiguousarray(g.take(g.input_of(node, 1)).T)
    for node, (module, in_dim) in TEXT_LSTM.items():
        W, R, B = (g.take(g.input_of(node, k)) for k in (1, 2, 3))
        for k, v in lstm_from_onnx(W, R, B, in_dim, node).items():
            sd[f"{module}.{k}"] = np.ascontiguousarray(v)
    # The duration encoder's style concat was constant-folded into a [1, sty, 1] copy.
    style = sd["style"]
    for name in [n for n in g.inits if n.startswith("onnx::Expand_")]:
        if not np.array_equal(g.take(name).reshape(-1), style.reshape(-1)):
            raise SystemExit(f"text graph: {name} is not a copy of ts.style")
    return sd


def head_constants_ok(g: Graph) -> list[str]:
    """Compare the decoder graph's fixed head constants with the formulas the native head
    (paradee-head.cpp) implements; returns the names checked."""
    hp = pt.HPARAMS_V11
    checked = []

    def conv_stft(n: int):
        w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)            # periodic Hann
        K = n // 2 + 1
        k = np.arange(K, dtype=np.float64)[:, None]
        ang = 2 * np.pi * ((k * np.arange(n, dtype=np.float64)) % n) / n
        c = np.full((K, 1), 2.0); c[0] = c[-1] = 1
        fwd = (np.concatenate([np.cos(ang), -np.sin(ang)]) * w).astype(np.float32)[:, None]
        inv = (np.concatenate([np.cos(ang) * c, -np.sin(ang) * c]) / n * w).astype(np.float32)[:, None]
        return fwd, inv, (w ** 2).astype(np.float32)[None, None]

    def same(name: str, want: np.ndarray, tol: float) -> None:
        got = g.take(name)
        if got.shape != want.shape or not np.allclose(got, want, rtol=0, atol=tol):
            err = np.abs(got.astype(np.float64) - want).max() if got.shape == want.shape else float("nan")
            raise SystemExit(f"decoder graph: {name} {got.shape} differs from the native head's constant "
                             f"(max |err| {err:.3g})")
        checked.append(name)

    for prefix, n in (("head.st", hp["head.n_fft"]), ("head.sb", hp["head.burst_n_fft"])):
        fwd, inv, w2 = conv_stft(n)
        same(f"{prefix}.fwd", fwd, 1e-6)
        same(f"{prefix}.inv", inv, 1e-8)
        same(f"{prefix}.w2", w2, 1e-7)
    K = hp["head.harmonics"]
    same("head.k", np.arange(1, K + 1, dtype=np.float32)[None, :, None], 0)
    nb = hp["head.n_fft"] // 2 + 1
    same("head.fr", np.linspace(0, hp["audio.sample_rate"] / 2, nb, dtype=np.float32)[None, :, None], 1e-3)
    two_pi_k = (2 * math.pi * np.arange(1, K + 1, dtype=np.float32)).astype(np.float32)[None, :, None]
    for name in [n for n in g.inits if n.startswith("onnx::Mul_")]:
        same(name, two_pi_k, 1e-4)

    def warp(nbins: int, G: int) -> np.ndarray:   # vhead.warp_matrix, float32 like torch
        f = np.linspace(0, hp["audio.sample_rate"] / 2, nbins, dtype=np.float32)
        mel = lambda x: np.log1p(x / np.float32(700)).astype(np.float32)
        pos = (mel(f) / mel(np.float32(hp["audio.sample_rate"] / 2.0)) * np.float32(G - 1)).astype(np.float32)
        i = np.clip(np.floor(pos), 0, G - 2).astype(np.int64)
        w = pos - i
        M = np.zeros((nbins, G), np.float32)
        M[np.arange(nbins), i] = 1 - w
        M[np.arange(nbins), i + 1] += w
        return M

    found = {}
    for name, arr in g.constants.items():
        if arr.ndim == 2 and arr.dtype == np.float32:
            found.setdefault(arr.shape, []).append((name, arr))
    for shape, want in (((hp["head.points"], nb), warp(nb, hp["head.points"]).T),
                        ((hp["head.burst_points"], hp["head.burst_n_fft"] // 2 + 1),
                         warp(hp["head.burst_n_fft"] // 2 + 1, hp["head.burst_points"]).T)):
        cands = found.get(shape, [])
        if len(cands) != 1:
            raise SystemExit(f"decoder graph: expected one {shape} warp matrix constant, found {len(cands)}")
        name, arr = cands[0]
        if not np.allclose(arr, want, rtol=0, atol=1e-5):
            raise SystemExit(f"decoder graph: {name} differs from vhead.warp_matrix (max |err| "
                             f"{np.abs(arr - want).max():.3g})")
        checked.append(name)
    return checked


def decoder_state_dict(g: Graph) -> tuple[dict[str, np.ndarray], list[str]]:
    sd = g.named()
    for node, key in DEC_MATMUL.items():
        sd[key] = np.ascontiguousarray(g.take(g.input_of(node, 1)).T)
    return sd, head_constants_ok(g)


def write_gguf(out_path: Path, tensors: dict[str, np.ndarray], vocab: list[tuple[str, int]], pins: dict,
               config: dict, family: dict) -> None:
    writer = gguf.GGUFWriter(str(out_path), arch=pt.ARCH)
    writer.add_string("general.name", family["name"])
    writer.add_string("general.license", "apache-2.0")
    writer.add_string("general.source.url", f"https://huggingface.co/{pins['model']['repo']}")
    writer.add_string("general.base_model.0.name", "Kokoro-82M")
    writer.add_string("general.base_model.1.name", "Paradee-8M v1.1")
    writer.add_string("general.description", family["description"])
    writer.add_string(f"{NS}.version", "1.1")
    writer.add_string(f"{NS}.decoder.head", pt.HEAD_V11)
    for key, val in pt.HPARAMS_V11.items():
        full = f"{NS}.{key}"
        if isinstance(val, list):
            writer.add_array(full, [int(x) for x in val])
        elif isinstance(val, float):
            writer.add_float32(full, val)
        else:
            writer.add_int32(full, int(val))
    writer.add_string(f"{NS}.family", family["family"])
    writer.add_string(f"{NS}.language", family["language"])
    writer.add_string(f"{NS}.voice", family["voice"])
    writer.add_string(f"{NS}.phoneme_format", family["phoneme_format"])
    writer.add_int32(f"{NS}.input.validated_max_ids", int(config["validated_max_tokens"]))
    lo, hi = config["validated_speed"]
    writer.add_array(f"{NS}.validated_speed", [float(lo), float(hi)])
    writer.add_string(f"{NS}.input.pad_symbol", "$")
    writer.add_array(f"{NS}.vocab.symbols", [s for s, _ in vocab])
    writer.add_array(f"{NS}.vocab.ids", [int(i) for _, i in vocab])
    model = pins["model"]
    writer.add_string(f"{NS}.pins.hf_repo", model["repo"])
    writer.add_string(f"{NS}.pins.hf_revision", model["revision"])
    for rel in ("onnx/fp32/text.onnx", "onnx/fp32/decoder.onnx", "config.json"):
        writer.add_string(f"{NS}.pins.sha256.{rel}", pins["files"][rel]["sha256"])
    writer.add_string(f"{NS}.converter.git_blob_sha1", pt.git_blob_sha1(Path(__file__)))
    writer.add_string(f"{NS}.converter.base_git_blob_sha1", pt.git_blob_sha1(Path(pt.__file__)))
    writer.add_string(f"{NS}.converter.outtype", "f32")
    for name in sorted(tensors):
        writer.add_tensor(name, tensors[name])
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


FAMILIES = {
    "delibes": {
        "pins": "delibes",
        "name": "Delibes-8M-esES",
        "family": "delibes",
        "language": "es",
        "voice": "em_alex",
        "phoneme_format": "espeak-es",
        "description": ("Delibes-8M-esES: Castilian Spanish TTS distilled from Kokoro-82M (teacher voice em_alex) "
                        "on Paradee-8M v1.1's architecture. StyleTTS-2 text side + AdaIN decoder trunk + "
                        "frame-rate source-filter head. Input: eSpeak NG es phonemes (misaki EspeakG2P)."),
    },
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=sorted(FAMILIES), required=True)
    ap.add_argument("--model-dir", required=True, type=Path, help="local snapshot of the pinned HF revision")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--check-output", action="store_true",
                    help="fail unless the output sha256 equals the one pinned in pins.json")
    args = ap.parse_args(argv)

    family = FAMILIES[args.family]
    pins = json.loads(pt.PINS_PATH.read_text(encoding="utf-8"))[family["pins"]]
    check_pins(args.model_dir, pins)
    config, vocab = load_config(args.model_dir, pins)

    tg = Graph(args.model_dir / "onnx/fp32/text.onnx", "ts")
    dg = Graph(args.model_dir / "onnx/fp32/decoder.onnx", "vd")
    text_sd = text_state_dict(tg)
    dec_sd, checked = decoder_state_dict(dg)
    for g in (tg, dg):
        left = sorted(set(g.inits) - g.used)
        if left:
            raise SystemExit(f"{g.prefix} graph: unconsumed initializers {left[:8]} ({len(left)} total)")

    tensors = pt.collect(text_sd, pt.text_side_map(text_sd), pt.TEXT_SIDE_SKIPPED, "text.onnx")
    dec = pt.collect(dec_sd, pt.decoder_map_v11(dec_sd), (), "decoder.onnx")
    overlap = set(tensors) & set(dec)
    if overlap:
        raise SystemExit(f"tensor name collision: {sorted(overlap)}")
    tensors.update(dec)
    for name, shape in pt.expected_shapes_v11().items():
        if tuple(tensors[name].shape) != shape:
            raise SystemExit(f"{name}: shape {tuple(tensors[name].shape)} != expected {shape}")
    n_params = sum(int(a.size) for a in tensors.values())
    if n_params != pins["parameters_converted"]:
        raise SystemExit(f"{n_params} parameters != pinned {pins['parameters_converted']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_gguf(args.output, tensors, vocab, pins, config, family)
    digest = sha256_file(args.output)
    print(f"checked decoder constants: {', '.join(checked)}")
    print(f"wrote {args.output}: {len(tensors)} tensors, {n_params} parameters, "
          f"{args.output.stat().st_size} bytes, sha256 {digest}")
    if args.check_output:
        want = pins["gguf"]["f32"]["sha256"]
        if digest != want:
            print(f"error: sha256 {digest} != pinned {want}", file=sys.stderr)
            return 1
        print("output matches pins.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
