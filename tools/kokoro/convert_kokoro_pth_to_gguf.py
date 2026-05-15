#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
convert_kokoro_pth_to_gguf.py — convert a Kokoro-82M (StyleTTS-2 + iSTFTNet)
PyTorch checkpoint into a GGUF consumable by the fork's tools/kokoro/
inference path.

Modes:

  --pth <kokoro-v1_0.pth>   convert the canonical hexgrad/Kokoro-82M checkpoint.
                            Walks all 548 PyTorch tensors and rewrites them
                            into the flat `kokoro.*` namespace the C++ loader
                            expects. weight_norm `weight_v / weight_g` pairs
                            are fused at convert time (no runtime re-param).

  --stub                    emit a smoke-test GGUF (random fp16 weights at the
                            canonical Kokoro v1.0 shapes). Used by the J2
                            verify harness when the .pth file isn't available;
                            produces shape-valid output that the C++ loader
                            can ingest without error.

The output GGUF has `general.architecture = "kokoro"` plus the canonical
kokoro hparams keys. See tools/kokoro/include/kokoro.h for the full key set
and tools/kokoro/src/kokoro.cpp for the loader that consumes the tensors.

K4 scope: this converter walks every parameter tensor — bert + bert_encoder
+ text_encoder (CNN + LSTM) + predictor (DurationEncoder LSTMs, AdaLayerNorm,
duration_proj, shared LSTM, F0 AdainResBlk1d chain, N AdainResBlk1d chain,
F0_proj, N_proj) + decoder (encode AdainResBlk1d, 4 decode AdainResBlk1d
stages, F0_conv, N_conv, asr_res, generator m_source + noise_convs +
noise_res + ups + 6 resblocks + conv_post). The fork's predictor + decoder
forward (kokoro_predictor.cpp + kokoro_decoder.cpp) consumes the resulting
tensors directly.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# Resolve gguf-py sibling import (so the script runs without an install).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "gguf-py"))

import gguf  # noqa: E402


# Canonical Kokoro v1.0 hparams (from hexgrad/Kokoro-82M config.json).
# - style_dim is per-half: ref_s is (256,) but split into ref_s[:128] (decoder)
#   and ref_s[128:] (predictor); each half is style_dim=128.
# - plbert / Albert: 12 hidden layers, num_hidden_groups=1 (shared layer).
# - predictor / text_encoder n_layer: 3.
# - generator: 2 upsample stages (rates [10,6], kernels [20,12]); 3 resblock
#   kernels per stage × dilations [[1,3,5],[1,3,5],[1,3,5]] → 6 resblocks total.
# - SineGen harmonic_num=8 → SourceModule l_linear takes 9 inputs.
KOKORO_HPARAMS = {
    # Albert / PLBERT text branch.
    "text_n_layer":          12,
    "text_n_head":           12,
    "text_d_model":          768,
    "text_d_ff":             2048,
    "text_vocab_size":       178,
    "text_max_pos":          512,
    "text_embd_dim":         128,   # Albert input embedding (factorized)

    # Style vector: ref_s is (n_pos, 1, 256). split is 128 + 128.
    "style_dim":             128,   # per-half style_dim

    # Hidden dim of the StyleTTS-2 prosody + text encoders.
    "hidden_dim":            512,
    "predictor_n_layer":     3,
    "max_dur":               50,
    "text_encoder_kernel":   5,
    "n_token":               178,
    "n_mels":                80,    # not used by inference; kept for record

    # iSTFTNet generator.
    "decoder_initial_ch":    512,
    "upsample_rates":        [10, 6],
    "upsample_kernels":      [20, 12],
    "resblock_kernels":      [3, 7, 11],
    "resblock_dilations":    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "istft_n_fft":           20,
    "istft_hop":             5,
    "harmonic_num":          8,    # SineGen harmonic count (l_linear: 9->1)

    # Output sample rate.
    "sample_rate":           24000,
}


def _add_tensor(writer: gguf.GGUFWriter, name: str, data: np.ndarray) -> None:
    """Add a tensor as fp32 (cast from fp64/fp16 if needed; ensure c-contig)."""
    if data.dtype != np.float32:
        data = data.astype(np.float32)
    if not data.flags["C_CONTIGUOUS"]:
        data = np.ascontiguousarray(data)
    writer.add_tensor(name, data)


def _write_metadata(writer: gguf.GGUFWriter, hp: dict) -> None:
    writer.add_string("general.architecture",   "kokoro")
    writer.add_string("general.name",           "Kokoro-82M")
    writer.add_string("general.license",        "apache-2.0")
    writer.add_string("general.description",
                     "Kokoro-82M TTS (StyleTTS-2 + iSTFTNet) — K4 GGUF port")

    # Albert / text branch.
    writer.add_int32("kokoro.text.n_layer",       int(hp["text_n_layer"]))
    writer.add_int32("kokoro.text.n_head",        int(hp["text_n_head"]))
    writer.add_int32("kokoro.text.d_model",       int(hp["text_d_model"]))
    writer.add_int32("kokoro.text.d_ff",          int(hp["text_d_ff"]))
    writer.add_int32("kokoro.text.vocab_size",    int(hp["text_vocab_size"]))
    writer.add_int32("kokoro.text.max_position",  int(hp["text_max_pos"]))
    writer.add_int32("kokoro.text.embd_dim",      int(hp["text_embd_dim"]))

    # Prosody / text encoder.
    writer.add_int32("kokoro.style.dim",          int(hp["style_dim"]))
    writer.add_int32("kokoro.hidden_dim",         int(hp["hidden_dim"]))
    writer.add_int32("kokoro.predictor.n_layer",  int(hp["predictor_n_layer"]))
    writer.add_int32("kokoro.predictor.max_dur",  int(hp["max_dur"]))
    writer.add_int32("kokoro.text_encoder.kernel", int(hp["text_encoder_kernel"]))

    # iSTFTNet generator.
    writer.add_int32("kokoro.decoder.initial_ch", int(hp["decoder_initial_ch"]))
    writer.add_array("kokoro.decoder.upsample_rates",   [int(x) for x in hp["upsample_rates"]])
    writer.add_array("kokoro.decoder.upsample_kernels", [int(x) for x in hp["upsample_kernels"]])
    writer.add_array("kokoro.decoder.resblock_kernels", [int(x) for x in hp["resblock_kernels"]])
    # Flatten dilations row-major: [3 kernels × 3 dilations].
    dil_flat = [int(x) for row in hp["resblock_dilations"] for x in row]
    writer.add_array("kokoro.decoder.resblock_dilations", dil_flat)
    writer.add_int32("kokoro.decoder.istft_n_fft",  int(hp["istft_n_fft"]))
    writer.add_int32("kokoro.decoder.istft_hop",    int(hp["istft_hop"]))
    writer.add_int32("kokoro.decoder.istft_win",    int(hp["istft_n_fft"]))
    writer.add_int32("kokoro.decoder.harmonic_num", int(hp["harmonic_num"]))

    writer.add_int32("kokoro.audio.sample_rate",  int(hp["sample_rate"]))


# ---------------------------------------------------------------------------
# weight_norm fusion.
#
# PyTorch's `torch.nn.utils.weight_norm(layer, name='weight')` stores
# `weight_v` and `weight_g` in the state dict — the runtime layer's
# `weight` is recomputed as:
#
#     weight = (weight_g / ||weight_v||) * weight_v
#
# where ||·|| is the L2 norm over all dims except `dim` (default dim=0).
# We fuse this at convert time so the C++ side doesn't need to re-parameterize.
# ---------------------------------------------------------------------------

def fuse_weight_norm(weight_v: np.ndarray, weight_g: np.ndarray, dim: int = 0) -> np.ndarray:
    """Return the fused `weight` tensor matching torch.nn.utils.weight_norm.

    weight_v has the layer's weight shape (e.g. Conv1d → [out, in, k]).
    weight_g has shape [out, 1, 1, ...] (size-1 except along `dim`).
    """
    # L2 norm of weight_v over all axes except `dim`.
    axes = tuple(i for i in range(weight_v.ndim) if i != dim)
    norm = np.sqrt((weight_v.astype(np.float64) ** 2).sum(axis=axes, keepdims=True))
    # Avoid /0 — same trick PyTorch uses (norm_except_dim adds eps=1e-12).
    norm = np.where(norm < 1e-12, 1.0, norm).astype(np.float32)
    return ((weight_g.astype(np.float32) / norm) * weight_v.astype(np.float32)).astype(np.float32)


# ---------------------------------------------------------------------------
# Stub emission (smoke-test path).
# ---------------------------------------------------------------------------

def emit_stub(out_path: str, hp: dict) -> None:
    """Emit a smoke-test GGUF with random-init tensors at canonical shapes.

    The stub matches the C++ loader's required tensors (token_embd + a flat
    per-layer Albert encoder + minimal predictor / decoder projections) so
    the loader smoke test passes without a real checkpoint. The stub is NOT
    used for audio-quality tests — the synthesis path with stub weights
    produces noise."""
    rng = np.random.default_rng(seed=42)
    scale = 0.02

    writer = gguf.GGUFWriter(out_path, arch="kokoro")
    _write_metadata(writer, hp)

    d  = hp["text_d_model"]
    ed = hp["text_embd_dim"]
    ff = hp["text_d_ff"]
    vocab = hp["text_vocab_size"]
    n_layer = hp["text_n_layer"]
    hid = hp["hidden_dim"]
    style = hp["style_dim"]

    # Albert factorized embedding: word_embeddings (vocab, embd_dim) +
    # embedding_hidden_mapping_in (embd_dim → d_model).
    _add_tensor(writer, "kokoro.bert.token_embd.weight",   rng.standard_normal((vocab, ed), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.bert.position_embd.weight", rng.standard_normal((hp["text_max_pos"], ed), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.bert.tok_type_embd.weight", rng.standard_normal((2, ed), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.bert.embd_ln.weight",      np.ones((ed,), dtype=np.float32))
    _add_tensor(writer, "kokoro.bert.embd_ln.bias",        np.zeros((ed,), dtype=np.float32))
    _add_tensor(writer, "kokoro.bert.embd_proj.weight",    rng.standard_normal((d, ed), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.bert.embd_proj.bias",      np.zeros((d,), dtype=np.float32))

    # Single shared Albert layer (Albert num_hidden_groups=1).
    _add_tensor(writer, "kokoro.bert.layer.full_ln.weight",  np.ones((d,), dtype=np.float32))
    _add_tensor(writer, "kokoro.bert.layer.full_ln.bias",    np.zeros((d,), dtype=np.float32))
    for name in ("attn_q", "attn_k", "attn_v", "attn_o"):
        _add_tensor(writer, f"kokoro.bert.layer.{name}.weight", rng.standard_normal((d, d), dtype=np.float32) * scale)
        _add_tensor(writer, f"kokoro.bert.layer.{name}.bias",   np.zeros((d,), dtype=np.float32))
    _add_tensor(writer, "kokoro.bert.layer.attn_ln.weight", np.ones((d,), dtype=np.float32))
    _add_tensor(writer, "kokoro.bert.layer.attn_ln.bias",   np.zeros((d,), dtype=np.float32))
    _add_tensor(writer, "kokoro.bert.layer.ffn.weight",     rng.standard_normal((ff, d), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.bert.layer.ffn.bias",       np.zeros((ff,), dtype=np.float32))
    _add_tensor(writer, "kokoro.bert.layer.ffn_out.weight", rng.standard_normal((d, ff), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.bert.layer.ffn_out.bias",   np.zeros((d,), dtype=np.float32))

    # bert_encoder: project 768 → 512 (hidden_dim).
    _add_tensor(writer, "kokoro.bert_encoder.weight", rng.standard_normal((hid, d), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.bert_encoder.bias",   np.zeros((hid,), dtype=np.float32))

    # text_encoder (StyleTTS-2 side, separate from Albert): 3 CNN+LN +
    # 1 BiLSTM. CNN kernels are size 5.
    for i in range(hp["predictor_n_layer"]):
        _add_tensor(writer, f"kokoro.text_encoder.cnn.{i}.weight", rng.standard_normal((hid, hid, hp["text_encoder_kernel"]), dtype=np.float32) * scale)
        _add_tensor(writer, f"kokoro.text_encoder.cnn.{i}.bias",   np.zeros((hid,), dtype=np.float32))
        _add_tensor(writer, f"kokoro.text_encoder.ln.{i}.weight",  np.ones((hid,), dtype=np.float32))
        _add_tensor(writer, f"kokoro.text_encoder.ln.{i}.bias",    np.zeros((hid,), dtype=np.float32))
    # BiLSTM.
    _add_tensor(writer, "kokoro.text_encoder.lstm.weight_ih_l0",   rng.standard_normal((4*hid//2, hid), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.text_encoder.lstm.weight_hh_l0",   rng.standard_normal((4*hid//2, hid//2), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.text_encoder.lstm.bias_ih_l0",     np.zeros((4*hid//2,), dtype=np.float32))
    _add_tensor(writer, "kokoro.text_encoder.lstm.bias_hh_l0",     np.zeros((4*hid//2,), dtype=np.float32))
    _add_tensor(writer, "kokoro.text_encoder.lstm.weight_ih_l0_r", rng.standard_normal((4*hid//2, hid), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.text_encoder.lstm.weight_hh_l0_r", rng.standard_normal((4*hid//2, hid//2), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.text_encoder.lstm.bias_ih_l0_r",   np.zeros((4*hid//2,), dtype=np.float32))
    _add_tensor(writer, "kokoro.text_encoder.lstm.bias_hh_l0_r",   np.zeros((4*hid//2,), dtype=np.float32))

    # Predictor DurationEncoder: nlayers BiLSTMs interleaved with AdaLayerNorm fc's.
    for i in range(hp["predictor_n_layer"]):
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.weight_ih_l0",   rng.standard_normal((4*hid//2, hid+style), dtype=np.float32) * scale)
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.weight_hh_l0",   rng.standard_normal((4*hid//2, hid//2), dtype=np.float32) * scale)
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.bias_ih_l0",     np.zeros((4*hid//2,), dtype=np.float32))
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.bias_hh_l0",     np.zeros((4*hid//2,), dtype=np.float32))
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.weight_ih_l0_r", rng.standard_normal((4*hid//2, hid+style), dtype=np.float32) * scale)
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.weight_hh_l0_r", rng.standard_normal((4*hid//2, hid//2), dtype=np.float32) * scale)
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.bias_ih_l0_r",   np.zeros((4*hid//2,), dtype=np.float32))
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.bias_hh_l0_r",   np.zeros((4*hid//2,), dtype=np.float32))
        _add_tensor(writer, f"kokoro.predictor.de.adaln{i}.fc.weight", rng.standard_normal((2*hid, style), dtype=np.float32) * scale)
        _add_tensor(writer, f"kokoro.predictor.de.adaln{i}.fc.bias",   np.zeros((2*hid,), dtype=np.float32))

    # Predictor main LSTM (post-DurationEncoder).
    _add_tensor(writer, "kokoro.predictor.lstm.weight_ih_l0",   rng.standard_normal((4*hid//2, hid+style), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.lstm.weight_hh_l0",   rng.standard_normal((4*hid//2, hid//2), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.lstm.bias_ih_l0",     np.zeros((4*hid//2,), dtype=np.float32))
    _add_tensor(writer, "kokoro.predictor.lstm.bias_hh_l0",     np.zeros((4*hid//2,), dtype=np.float32))
    _add_tensor(writer, "kokoro.predictor.lstm.weight_ih_l0_r", rng.standard_normal((4*hid//2, hid+style), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.lstm.weight_hh_l0_r", rng.standard_normal((4*hid//2, hid//2), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.lstm.bias_ih_l0_r",   np.zeros((4*hid//2,), dtype=np.float32))
    _add_tensor(writer, "kokoro.predictor.lstm.bias_hh_l0_r",   np.zeros((4*hid//2,), dtype=np.float32))

    # duration_proj.
    _add_tensor(writer, "kokoro.predictor.duration_proj.weight", rng.standard_normal((hp["max_dur"], hid), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.duration_proj.bias",   np.zeros((hp["max_dur"],), dtype=np.float32))

    # shared LSTM (feeds F0 + N branches).
    _add_tensor(writer, "kokoro.predictor.shared.weight_ih_l0",   rng.standard_normal((4*hid//2, hid+style), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.shared.weight_hh_l0",   rng.standard_normal((4*hid//2, hid//2), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.shared.bias_ih_l0",     np.zeros((4*hid//2,), dtype=np.float32))
    _add_tensor(writer, "kokoro.predictor.shared.bias_hh_l0",     np.zeros((4*hid//2,), dtype=np.float32))
    _add_tensor(writer, "kokoro.predictor.shared.weight_ih_l0_r", rng.standard_normal((4*hid//2, hid+style), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.shared.weight_hh_l0_r", rng.standard_normal((4*hid//2, hid//2), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.shared.bias_ih_l0_r",   np.zeros((4*hid//2,), dtype=np.float32))
    _add_tensor(writer, "kokoro.predictor.shared.bias_hh_l0_r",   np.zeros((4*hid//2,), dtype=np.float32))

    # F0 / N AdainResBlk1d chains. Stub: just emit projections.
    _add_tensor(writer, "kokoro.predictor.F0_proj.weight", rng.standard_normal((1, hid//2, 1), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.F0_proj.bias",   np.zeros((1,), dtype=np.float32))
    _add_tensor(writer, "kokoro.predictor.N_proj.weight",  rng.standard_normal((1, hid//2, 1), dtype=np.float32) * scale)
    _add_tensor(writer, "kokoro.predictor.N_proj.bias",    np.zeros((1,), dtype=np.float32))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


# ---------------------------------------------------------------------------
# .pth → GGUF (real weights).
#
# Walks all five top-level sections (bert / bert_encoder / text_encoder /
# predictor / decoder) and rewrites every tensor under a canonical
# `kokoro.*` name. weight_norm pairs are fused. The output is a complete,
# self-contained GGUF for inference.
# ---------------------------------------------------------------------------

def _strip_module(k: str) -> str:
    """Strip the `module.` DataParallel prefix that all kokoro state dicts
    wrap their parameters with."""
    return k[len("module."):] if k.startswith("module.") else k


def _fuse_section_weight_norm(sd: dict, prefix: str) -> dict:
    """Walk `sd` keys, fuse all `<x>.weight_v`/`<x>.weight_g` pairs into
    `<x>.weight`. Returns a new dict with the fused tensors and the original
    non-weight_norm tensors carried over.
    """
    out: dict = {}
    pairs: dict = {}
    for k, v in sd.items():
        ks = _strip_module(k)
        if ks.endswith(".weight_v"):
            pairs.setdefault(ks[:-len(".weight_v")], {})["v"] = v
        elif ks.endswith(".weight_g"):
            pairs.setdefault(ks[:-len(".weight_g")], {})["g"] = v
        else:
            out[ks] = v
    for base, pair in pairs.items():
        if "v" in pair and "g" in pair:
            out[base + ".weight"] = fuse_weight_norm(pair["v"], pair["g"], dim=0)
        elif "v" in pair:
            # weight_v without weight_g — rare; treat as identity weight
            # (no scaling).
            out[base + ".weight"] = pair["v"].astype(np.float32)
    return out


def emit_from_pth(pth_path: str, out_path: str, hp: dict) -> None:
    import torch
    obj = torch.load(pth_path, map_location="cpu", weights_only=True)

    # The kokoro .pth is a top-level dict with five sections, each itself a
    # state_dict. We process per-section so weight_norm fusion stays scoped.
    sections: dict = {}
    for sec_name, sec_sd in obj.items():
        # Materialize all tensors as numpy fp32.
        np_sd = {k: v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                 for k, v in sec_sd.items()}
        sections[sec_name] = _fuse_section_weight_norm(np_sd, sec_name)

    writer = gguf.GGUFWriter(out_path, arch="kokoro")
    _write_metadata(writer, hp)

    # ----- BERT (Albert / PLBERT 12 layers, single shared layer group). -----
    b = sections["bert"]

    def _emit_bert():
        _add_tensor(writer, "kokoro.bert.token_embd.weight",
                    b["embeddings.word_embeddings.weight"])
        _add_tensor(writer, "kokoro.bert.position_embd.weight",
                    b["embeddings.position_embeddings.weight"])
        _add_tensor(writer, "kokoro.bert.tok_type_embd.weight",
                    b["embeddings.token_type_embeddings.weight"])
        _add_tensor(writer, "kokoro.bert.embd_ln.weight",
                    b["embeddings.LayerNorm.weight"])
        _add_tensor(writer, "kokoro.bert.embd_ln.bias",
                    b["embeddings.LayerNorm.bias"])
        _add_tensor(writer, "kokoro.bert.embd_proj.weight",
                    b["encoder.embedding_hidden_mapping_in.weight"])
        _add_tensor(writer, "kokoro.bert.embd_proj.bias",
                    b["encoder.embedding_hidden_mapping_in.bias"])

        # Albert: num_hidden_groups=1 → single shared layer reused 12 times
        # in forward. Layer naming in the .pth: "encoder.albert_layer_groups.0.albert_layers.0.*"
        pfx_ag = "encoder.albert_layer_groups.0.albert_layers.0."
        _add_tensor(writer, "kokoro.bert.layer.full_ln.weight", b[pfx_ag + "full_layer_layer_norm.weight"])
        _add_tensor(writer, "kokoro.bert.layer.full_ln.bias",   b[pfx_ag + "full_layer_layer_norm.bias"])
        _add_tensor(writer, "kokoro.bert.layer.attn_q.weight",  b[pfx_ag + "attention.query.weight"])
        _add_tensor(writer, "kokoro.bert.layer.attn_q.bias",    b[pfx_ag + "attention.query.bias"])
        _add_tensor(writer, "kokoro.bert.layer.attn_k.weight",  b[pfx_ag + "attention.key.weight"])
        _add_tensor(writer, "kokoro.bert.layer.attn_k.bias",    b[pfx_ag + "attention.key.bias"])
        _add_tensor(writer, "kokoro.bert.layer.attn_v.weight",  b[pfx_ag + "attention.value.weight"])
        _add_tensor(writer, "kokoro.bert.layer.attn_v.bias",    b[pfx_ag + "attention.value.bias"])
        _add_tensor(writer, "kokoro.bert.layer.attn_o.weight",  b[pfx_ag + "attention.dense.weight"])
        _add_tensor(writer, "kokoro.bert.layer.attn_o.bias",    b[pfx_ag + "attention.dense.bias"])
        _add_tensor(writer, "kokoro.bert.layer.attn_ln.weight", b[pfx_ag + "attention.LayerNorm.weight"])
        _add_tensor(writer, "kokoro.bert.layer.attn_ln.bias",   b[pfx_ag + "attention.LayerNorm.bias"])
        _add_tensor(writer, "kokoro.bert.layer.ffn.weight",     b[pfx_ag + "ffn.weight"])
        _add_tensor(writer, "kokoro.bert.layer.ffn.bias",       b[pfx_ag + "ffn.bias"])
        _add_tensor(writer, "kokoro.bert.layer.ffn_out.weight", b[pfx_ag + "ffn_output.weight"])
        _add_tensor(writer, "kokoro.bert.layer.ffn_out.bias",   b[pfx_ag + "ffn_output.bias"])

        # Pooler is unused at inference time (we feed last_hidden_state).
    _emit_bert()

    # ----- bert_encoder: 768→512 linear -----
    be = sections["bert_encoder"]
    _add_tensor(writer, "kokoro.bert_encoder.weight", be["weight"])
    _add_tensor(writer, "kokoro.bert_encoder.bias",   be["bias"])

    # ----- text_encoder: 3 CNN + LN + BiLSTM -----
    te = sections["text_encoder"]
    _add_tensor(writer, "kokoro.text_encoder.embd.weight", te["embedding.weight"])
    for i in range(hp["predictor_n_layer"]):
        # `cnn.<i>.0` is the conv, `cnn.<i>.1` is the LayerNorm (the .2 / .3
        # are activation + dropout — no params).
        _add_tensor(writer, f"kokoro.text_encoder.cnn.{i}.weight", te[f"cnn.{i}.0.weight"])
        _add_tensor(writer, f"kokoro.text_encoder.cnn.{i}.bias",   te[f"cnn.{i}.0.bias"])
        _add_tensor(writer, f"kokoro.text_encoder.ln.{i}.gamma",   te[f"cnn.{i}.1.gamma"])
        _add_tensor(writer, f"kokoro.text_encoder.ln.{i}.beta",    te[f"cnn.{i}.1.beta"])
    _add_tensor(writer, "kokoro.text_encoder.lstm.weight_ih_l0",   te["lstm.weight_ih_l0"])
    _add_tensor(writer, "kokoro.text_encoder.lstm.weight_hh_l0",   te["lstm.weight_hh_l0"])
    _add_tensor(writer, "kokoro.text_encoder.lstm.bias_ih_l0",     te["lstm.bias_ih_l0"])
    _add_tensor(writer, "kokoro.text_encoder.lstm.bias_hh_l0",     te["lstm.bias_hh_l0"])
    _add_tensor(writer, "kokoro.text_encoder.lstm.weight_ih_l0_r", te["lstm.weight_ih_l0_reverse"])
    _add_tensor(writer, "kokoro.text_encoder.lstm.weight_hh_l0_r", te["lstm.weight_hh_l0_reverse"])
    _add_tensor(writer, "kokoro.text_encoder.lstm.bias_ih_l0_r",   te["lstm.bias_ih_l0_reverse"])
    _add_tensor(writer, "kokoro.text_encoder.lstm.bias_hh_l0_r",   te["lstm.bias_hh_l0_reverse"])

    # ----- predictor -----
    p = sections["predictor"]

    # DurationEncoder: nlayers (3) BiLSTM/AdaLayerNorm interleaved.
    # PyTorch naming: text_encoder.lstms.<2*i>   = BiLSTM
    #                  text_encoder.lstms.<2*i+1> = AdaLayerNorm (.fc.*)
    for i in range(hp["predictor_n_layer"]):
        lstm_idx = 2 * i
        adaln_idx = 2 * i + 1
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.weight_ih_l0",   p[f"text_encoder.lstms.{lstm_idx}.weight_ih_l0"])
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.weight_hh_l0",   p[f"text_encoder.lstms.{lstm_idx}.weight_hh_l0"])
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.bias_ih_l0",     p[f"text_encoder.lstms.{lstm_idx}.bias_ih_l0"])
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.bias_hh_l0",     p[f"text_encoder.lstms.{lstm_idx}.bias_hh_l0"])
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.weight_ih_l0_r", p[f"text_encoder.lstms.{lstm_idx}.weight_ih_l0_reverse"])
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.weight_hh_l0_r", p[f"text_encoder.lstms.{lstm_idx}.weight_hh_l0_reverse"])
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.bias_ih_l0_r",   p[f"text_encoder.lstms.{lstm_idx}.bias_ih_l0_reverse"])
        _add_tensor(writer, f"kokoro.predictor.de.lstm{i}.bias_hh_l0_r",   p[f"text_encoder.lstms.{lstm_idx}.bias_hh_l0_reverse"])
        _add_tensor(writer, f"kokoro.predictor.de.adaln{i}.fc.weight",     p[f"text_encoder.lstms.{adaln_idx}.fc.weight"])
        _add_tensor(writer, f"kokoro.predictor.de.adaln{i}.fc.bias",       p[f"text_encoder.lstms.{adaln_idx}.fc.bias"])

    # Predictor main LSTM (post-DurationEncoder).
    _add_tensor(writer, "kokoro.predictor.lstm.weight_ih_l0",   p["lstm.weight_ih_l0"])
    _add_tensor(writer, "kokoro.predictor.lstm.weight_hh_l0",   p["lstm.weight_hh_l0"])
    _add_tensor(writer, "kokoro.predictor.lstm.bias_ih_l0",     p["lstm.bias_ih_l0"])
    _add_tensor(writer, "kokoro.predictor.lstm.bias_hh_l0",     p["lstm.bias_hh_l0"])
    _add_tensor(writer, "kokoro.predictor.lstm.weight_ih_l0_r", p["lstm.weight_ih_l0_reverse"])
    _add_tensor(writer, "kokoro.predictor.lstm.weight_hh_l0_r", p["lstm.weight_hh_l0_reverse"])
    _add_tensor(writer, "kokoro.predictor.lstm.bias_ih_l0_r",   p["lstm.bias_ih_l0_reverse"])
    _add_tensor(writer, "kokoro.predictor.lstm.bias_hh_l0_r",   p["lstm.bias_hh_l0_reverse"])

    # duration_proj (LinearNorm wraps nn.Linear).
    _add_tensor(writer, "kokoro.predictor.duration_proj.weight", p["duration_proj.linear_layer.weight"])
    _add_tensor(writer, "kokoro.predictor.duration_proj.bias",   p["duration_proj.linear_layer.bias"])

    # shared LSTM (feeds F0 + N AdainResBlk1d chains).
    _add_tensor(writer, "kokoro.predictor.shared.weight_ih_l0",   p["shared.weight_ih_l0"])
    _add_tensor(writer, "kokoro.predictor.shared.weight_hh_l0",   p["shared.weight_hh_l0"])
    _add_tensor(writer, "kokoro.predictor.shared.bias_ih_l0",     p["shared.bias_ih_l0"])
    _add_tensor(writer, "kokoro.predictor.shared.bias_hh_l0",     p["shared.bias_hh_l0"])
    _add_tensor(writer, "kokoro.predictor.shared.weight_ih_l0_r", p["shared.weight_ih_l0_reverse"])
    _add_tensor(writer, "kokoro.predictor.shared.weight_hh_l0_r", p["shared.weight_hh_l0_reverse"])
    _add_tensor(writer, "kokoro.predictor.shared.bias_ih_l0_r",   p["shared.bias_ih_l0_reverse"])
    _add_tensor(writer, "kokoro.predictor.shared.bias_hh_l0_r",   p["shared.bias_hh_l0_reverse"])

    # F0 chain — 3 AdainResBlk1d blocks. Each has conv1/conv2 (fused
    # weight_norm) + norm1.fc / norm2.fc + optional conv1x1 + optional pool.
    def _emit_adainresblk1d(prefix_in: str, prefix_out: str, has_pool: bool):
        # convs (weight_norm fused already by _fuse_section_weight_norm).
        _add_tensor(writer, f"{prefix_out}.conv1.weight", p[f"{prefix_in}.conv1.weight"])
        _add_tensor(writer, f"{prefix_out}.conv1.bias",   p[f"{prefix_in}.conv1.bias"])
        _add_tensor(writer, f"{prefix_out}.conv2.weight", p[f"{prefix_in}.conv2.weight"])
        _add_tensor(writer, f"{prefix_out}.conv2.bias",   p[f"{prefix_in}.conv2.bias"])
        # AdaIN norms: fc weight + bias.
        _add_tensor(writer, f"{prefix_out}.norm1.fc.weight", p[f"{prefix_in}.norm1.fc.weight"])
        _add_tensor(writer, f"{prefix_out}.norm1.fc.bias",   p[f"{prefix_in}.norm1.fc.bias"])
        _add_tensor(writer, f"{prefix_out}.norm2.fc.weight", p[f"{prefix_in}.norm2.fc.weight"])
        _add_tensor(writer, f"{prefix_out}.norm2.fc.bias",   p[f"{prefix_in}.norm2.fc.bias"])
        # Optional conv1x1 (learned_sc when dim_in != dim_out, e.g. upsample blocks).
        if f"{prefix_in}.conv1x1.weight" in p:
            _add_tensor(writer, f"{prefix_out}.conv1x1.weight", p[f"{prefix_in}.conv1x1.weight"])
            if f"{prefix_in}.conv1x1.bias" in p:
                _add_tensor(writer, f"{prefix_out}.conv1x1.bias", p[f"{prefix_in}.conv1x1.bias"])
        # Optional pool (ConvTranspose1d, weight_norm) when upsample=True.
        if has_pool and f"{prefix_in}.pool.weight" in p:
            _add_tensor(writer, f"{prefix_out}.pool.weight", p[f"{prefix_in}.pool.weight"])
            _add_tensor(writer, f"{prefix_out}.pool.bias",   p[f"{prefix_in}.pool.bias"])

    for i in range(3):
        # F0.0: hid → hid; F0.1: hid → hid/2 upsample (has pool); F0.2: hid/2 → hid/2
        has_pool = (i == 1)
        _emit_adainresblk1d(f"F0.{i}", f"kokoro.predictor.F0.{i}", has_pool=has_pool)
    for i in range(3):
        has_pool = (i == 1)
        _emit_adainresblk1d(f"N.{i}", f"kokoro.predictor.N.{i}", has_pool=has_pool)

    # F0_proj, N_proj are plain Conv1d (no weight_norm).
    _add_tensor(writer, "kokoro.predictor.F0_proj.weight", p["F0_proj.weight"])
    _add_tensor(writer, "kokoro.predictor.F0_proj.bias",   p["F0_proj.bias"])
    _add_tensor(writer, "kokoro.predictor.N_proj.weight",  p["N_proj.weight"])
    _add_tensor(writer, "kokoro.predictor.N_proj.bias",    p["N_proj.bias"])

    # ----- decoder -----
    d = sections["decoder"]

    # encode AdainResBlk1d (no pool, no conv1x1 — dim_in=514 → dim_out=1024 has conv1x1).
    def _emit_decoder_adainresblk1d(prefix_in: str, prefix_out: str, has_pool: bool):
        _add_tensor(writer, f"{prefix_out}.conv1.weight", d[f"{prefix_in}.conv1.weight"])
        _add_tensor(writer, f"{prefix_out}.conv1.bias",   d[f"{prefix_in}.conv1.bias"])
        _add_tensor(writer, f"{prefix_out}.conv2.weight", d[f"{prefix_in}.conv2.weight"])
        _add_tensor(writer, f"{prefix_out}.conv2.bias",   d[f"{prefix_in}.conv2.bias"])
        _add_tensor(writer, f"{prefix_out}.norm1.fc.weight", d[f"{prefix_in}.norm1.fc.weight"])
        _add_tensor(writer, f"{prefix_out}.norm1.fc.bias",   d[f"{prefix_in}.norm1.fc.bias"])
        _add_tensor(writer, f"{prefix_out}.norm2.fc.weight", d[f"{prefix_in}.norm2.fc.weight"])
        _add_tensor(writer, f"{prefix_out}.norm2.fc.bias",   d[f"{prefix_in}.norm2.fc.bias"])
        if f"{prefix_in}.conv1x1.weight" in d:
            _add_tensor(writer, f"{prefix_out}.conv1x1.weight", d[f"{prefix_in}.conv1x1.weight"])
            if f"{prefix_in}.conv1x1.bias" in d:
                _add_tensor(writer, f"{prefix_out}.conv1x1.bias", d[f"{prefix_in}.conv1x1.bias"])
        if has_pool and f"{prefix_in}.pool.weight" in d:
            _add_tensor(writer, f"{prefix_out}.pool.weight", d[f"{prefix_in}.pool.weight"])
            _add_tensor(writer, f"{prefix_out}.pool.bias",   d[f"{prefix_in}.pool.bias"])

    _emit_decoder_adainresblk1d("encode", "kokoro.decoder.encode", has_pool=False)
    for i in range(4):
        has_pool = (i == 3)  # last decode block has upsample=True
        _emit_decoder_adainresblk1d(f"decode.{i}", f"kokoro.decoder.decode.{i}", has_pool=has_pool)

    # F0_conv / N_conv (Conv1d with weight_norm fused).
    _add_tensor(writer, "kokoro.decoder.F0_conv.weight", d["F0_conv.weight"])
    _add_tensor(writer, "kokoro.decoder.F0_conv.bias",   d["F0_conv.bias"])
    _add_tensor(writer, "kokoro.decoder.N_conv.weight",  d["N_conv.weight"])
    _add_tensor(writer, "kokoro.decoder.N_conv.bias",    d["N_conv.bias"])

    # asr_res: Sequential(Conv1d 512→64) with weight_norm.
    _add_tensor(writer, "kokoro.decoder.asr_res.weight", d["asr_res.0.weight"])
    _add_tensor(writer, "kokoro.decoder.asr_res.bias",   d["asr_res.0.bias"])

    # ---- generator ----
    # m_source.l_linear (Linear(9, 1)).
    _add_tensor(writer, "kokoro.gen.m_source.l_linear.weight", d["generator.m_source.l_linear.weight"])
    _add_tensor(writer, "kokoro.gen.m_source.l_linear.bias",   d["generator.m_source.l_linear.bias"])

    # noise_convs (plain Conv1d, NO weight_norm).
    # In the .pth: generator.noise_convs.0/1.weight, .bias.
    for i in range(2):
        _add_tensor(writer, f"kokoro.gen.noise_convs.{i}.weight",
                    d[f"generator.noise_convs.{i}.weight"])
        _add_tensor(writer, f"kokoro.gen.noise_convs.{i}.bias",
                    d[f"generator.noise_convs.{i}.bias"])

    # ups (ConvTranspose1d weight_norm).
    for i in range(2):
        _add_tensor(writer, f"kokoro.gen.ups.{i}.weight",
                    d[f"generator.ups.{i}.weight"])
        _add_tensor(writer, f"kokoro.gen.ups.{i}.bias",
                    d[f"generator.ups.{i}.bias"])

    # noise_res (2 AdaINResBlock1 — each has convs1.0/1/2, convs2.0/1/2,
    # adain1.0/1/2, adain2.0/1/2, alpha1.0/1/2, alpha2.0/1/2).
    for i in range(2):
        for j in range(3):
            for branch in ("convs1", "convs2"):
                _add_tensor(writer, f"kokoro.gen.noise_res.{i}.{branch}.{j}.weight",
                            d[f"generator.noise_res.{i}.{branch}.{j}.weight"])
                _add_tensor(writer, f"kokoro.gen.noise_res.{i}.{branch}.{j}.bias",
                            d[f"generator.noise_res.{i}.{branch}.{j}.bias"])
            for branch in ("adain1", "adain2"):
                _add_tensor(writer, f"kokoro.gen.noise_res.{i}.{branch}.{j}.fc.weight",
                            d[f"generator.noise_res.{i}.{branch}.{j}.fc.weight"])
                _add_tensor(writer, f"kokoro.gen.noise_res.{i}.{branch}.{j}.fc.bias",
                            d[f"generator.noise_res.{i}.{branch}.{j}.fc.bias"])
            _add_tensor(writer, f"kokoro.gen.noise_res.{i}.alpha1.{j}",
                        d[f"generator.noise_res.{i}.alpha1.{j}"])
            _add_tensor(writer, f"kokoro.gen.noise_res.{i}.alpha2.{j}",
                        d[f"generator.noise_res.{i}.alpha2.{j}"])

    # resblocks (6 of them — 3 per upsample stage × 2 stages).
    for i in range(6):
        for j in range(3):
            for branch in ("convs1", "convs2"):
                _add_tensor(writer, f"kokoro.gen.resblocks.{i}.{branch}.{j}.weight",
                            d[f"generator.resblocks.{i}.{branch}.{j}.weight"])
                _add_tensor(writer, f"kokoro.gen.resblocks.{i}.{branch}.{j}.bias",
                            d[f"generator.resblocks.{i}.{branch}.{j}.bias"])
            for branch in ("adain1", "adain2"):
                _add_tensor(writer, f"kokoro.gen.resblocks.{i}.{branch}.{j}.fc.weight",
                            d[f"generator.resblocks.{i}.{branch}.{j}.fc.weight"])
                _add_tensor(writer, f"kokoro.gen.resblocks.{i}.{branch}.{j}.fc.bias",
                            d[f"generator.resblocks.{i}.{branch}.{j}.fc.bias"])
            _add_tensor(writer, f"kokoro.gen.resblocks.{i}.alpha1.{j}",
                        d[f"generator.resblocks.{i}.alpha1.{j}"])
            _add_tensor(writer, f"kokoro.gen.resblocks.{i}.alpha2.{j}",
                        d[f"generator.resblocks.{i}.alpha2.{j}"])

    # conv_post (Conv1d weight_norm → 22 = n_fft + 2).
    _add_tensor(writer, "kokoro.gen.conv_post.weight", d["generator.conv_post.weight"])
    _add_tensor(writer, "kokoro.gen.conv_post.bias",   d["generator.conv_post.bias"])

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
