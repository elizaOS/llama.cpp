#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
convert_phonon2_to_gguf.py - convert FermionResearch/Phonon-2 (a five-value
QAT re-quantisation of nvidia/parakeet-tdt-0.6b-v3, CC-BY-4.0 weights) into the
GGUF that tools/parakeet loads (general.architecture = parakeet-tdt).

Input: the pinned release archive `phonon-2.bps.tar.zst` (or the already
unpacked `model.fermion` + NeMo `config.json`). Every input is checked against
tools/parakeet/pins.json (size + sha256) before it is parsed.

Container (`fermion-five-value-parakeet-v1`): u64 LE header length, JSON
header {format, index:[{n,k,shape,b}]}, then the records in index order.
  five_value [O,I]: O rows of ceil(I/5) base-3 bytes (LSD first, sign=code-1),
                    then a little-endian bit per non-zero code (1 = hi), then
                    fp16 lo[O], fp16 hi[O]. Weight = sign * (is_hi ? hi : lo).
  int6/int8 [O,...]: symmetric per-row integers (6-bit packed 4 per 3 bytes,
                    minus 32) followed by fp16 row scales.
  fp16:             raw half floats.

GGUF layout written here (all lossless with respect to the container):
  * five-value linears (level 0..4 = {-hi,-lo,0,+lo,+hi}) -> the codes plus
    `<name>.lohi` (F32 [O,2]). q/k/v are concatenated into one `attn.qkv`.
    The file is mmapped and used in place. `--layout` picks the code layout:
      b5x3t7 (default): `<name>.fv5t7`, I8 [O/8, fv5t7 block bytes], the
        `fv5-b5x3t7-v1` order (see pack_b5x3t7), 2.35-2.41 bits per weight;
      pair16: `<name>.fv5`, I8 [O/2, I], 4-bit level indices in the
        `fv5-pair16-v1` order (see pack_pair16), 4 bits per weight.
    Both decode to the same int8 panels, so they give identical results.
  * int6 tables -> `<name>.q` (I8, the exact integers) + `<name>.scale` (F32).
  * fp16 records -> F32 (exact widening).
  * BatchNorm is folded into the depthwise conv (F32 weight + bias); this is
    the only arithmetic transform and it is done in float64.
  * The LSTM's two bias vectors are summed (float64 -> F32).
  * The mel filterbank (librosa slaney, as ParakeetFeatureExtractor builds
    it) is stored as `feat.mel_fb` [128, 257] F32.

--verify re-reads the written GGUF and re-expands every tensor; five-value
and int tensors must equal the container's own expansion bit for bit (the
same check as Fermion's G-PLANE/G-LOAD gates), and all 723 records must be
consumed.

The output is byte-reproducible for the same inputs and converter.

usage:
  convert_phonon2_to_gguf.py --archive phonon-2.bps.tar.zst --output phonon-2-fv5.gguf [--verify]
  convert_phonon2_to_gguf.py --container model.fermion --nemo-config config.json \
      --notice NOTICE --output phonon-2-fv5.gguf [--verify]
  add --layout pair16 for the 4-bit `fv5-pair16-v1` layout
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent / "gguf-py"))

import gguf  # noqa: E402

PINS_PATH = _HERE / "pins.json"
ARCH = "parakeet-tdt"
NS = "parakeet"
FORMAT = "fermion-five-value-parakeet-v1"

HP: dict[str, Any] = {
    "d_model": 1024,
    "n_layers": 24,
    "n_heads": 8,
    "ff_dim": 4096,
    "conv_kernel": 9,
    "sub_channels": 256,
    "sub_factor": 8,
    "n_mels": 128,
    "sample_rate": 16000,
    "n_fft": 512,
    "win_length": 400,
    "hop_length": 160,
    "preemph": 0.97,
    "log_guard": 2.0 ** -24,
    "norm_eps": 1e-5,
    "ln_eps": 1e-5,
    "bn_eps": 1e-5,
    "pos_max_len": 5000,
    "pred_hidden": 640,
    "pred_layers": 2,
    "joint_hidden": 640,
    "vocab": 8192,
    "blank_id": 8192,
    "durations": [0, 1, 2, 3, 4],
    "max_symbols": 10,
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def die(msg: str) -> NoReturn:
    print(f"convert_phonon2_to_gguf: {msg}", file=sys.stderr)
    sys.exit(1)


def check_pin(name: str, data: bytes, pin: dict) -> None:
    if len(data) != pin["size"] or sha256_bytes(data) != pin["sha256"]:
        die(f"{name}: size/sha256 mismatch against pins.json "
            f"(got {len(data)} bytes, sha256 {sha256_bytes(data)})")


# --------------------------------------------------------------- inputs
def read_archive(path: Path, pins: dict) -> dict[str, bytes]:
    data = path.read_bytes()
    check_pin(path.name, data, pins["files"]["phonon-2.bps.tar.zst"])
    try:
        import zstandard  # type: ignore
        raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(data)).read()
    except ImportError:
        raw = subprocess.run(["zstd", "-q", "-d", "-c"], input=data, check=True,
                             stdout=subprocess.PIPE).stdout
    members: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        manifest = None
        for m in tar:
            fh = tar.extractfile(m)
            if fh is None:
                continue
            blob = fh.read()
            if m.name == "bps_manifest.json":
                manifest = json.loads(blob)
                continue
            if manifest is None:
                die("bps_manifest.json must be the first archive member")
            rows = {r["path"]: r for r in manifest["files"]}
            rel = m.name[:-4] if m.name.endswith(".bps") else m.name
            row = rows[rel]
            if "transform" in row:
                die(f"{rel}: byte-plane transform present; the pinned release stores members verbatim")
            if sha256_bytes(blob) != row["original_sha256"] or len(blob) != row["original_bytes"]:
                die(f"{rel}: archive member checksum mismatch")
            members[rel] = blob
    for name, pin in pins["archive_members"].items():
        if name not in members:
            die(f"archive is missing {name}")
        check_pin(name, members[name], pin)
    return members


# ------------------------------------------------------------ container
def trits(buf: np.ndarray, o: int, i: int) -> np.ndarray:
    rb = (i + 4) // 5
    x = buf[: o * rb].reshape(o, rb).astype(np.uint16)
    d = np.empty((o, rb, 5), dtype=np.uint8)
    p = 1
    for k in range(5):
        d[:, :, k] = (x // p) % 3
        p *= 3
    return d.reshape(o, rb * 5)[:, :i]


def parse_five_value(blob: bytes, shape) -> dict:
    o, i = shape
    rb = (i + 4) // 5
    u8 = np.frombuffer(blob, dtype=np.uint8)
    codes = trits(u8, o, i)
    if codes.max() > 2:
        die("five_value: base-3 digit out of range")
    nz = codes != 1
    nnz = int(nz.sum())
    off = o * rb
    nbytes = (nnz + 7) // 8
    bits = np.unpackbits(u8[off: off + nbytes], bitorder="little")[:nnz].astype(bool)
    off += nbytes
    lo = np.frombuffer(blob[off: off + 2 * o], dtype=np.float16).copy()
    hi = np.frombuffer(blob[off + 2 * o: off + 4 * o], dtype=np.float16).copy()
    if off + 4 * o != len(blob):
        die("five_value: record length mismatch")
    is_hi = np.zeros((o, i), dtype=bool)
    is_hi[nz] = bits
    sign = codes.astype(np.int8) - 1
    return {"sign": sign, "is_hi": is_hi, "lo": lo, "hi": hi}


def parse_intn(blob: bytes, shape, bits: int) -> dict:
    o = shape[0]
    total = int(np.prod(shape))
    sc = np.frombuffer(blob[-2 * o:], dtype=np.float16).copy()
    body = blob[: -2 * o]
    if bits == 8:
        q = np.frombuffer(body, dtype=np.int8)[:total].astype(np.int32)
    elif bits == 6:
        b = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3).astype(np.uint32)
        packed = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        u = np.stack([(packed >> s) & 0x3F for s in (0, 6, 12, 18)], axis=1).ravel()
        q = u[:total].astype(np.int32) - 32
    else:
        die(f"int{bits}: unsupported")
    return {"q": q.reshape(o, total // o).astype(np.int8), "scale": sc, "shape": tuple(shape)}


def read_container(blob: bytes):
    n = int.from_bytes(blob[:8], "little")
    header = json.loads(blob[8: 8 + n])
    if header.get("format") != FORMAT:
        die(f"container format {header.get('format')!r} != {FORMAT}")
    off = 8 + n
    recs = {}
    for e in header["index"]:
        b = blob[off: off + e["b"]]
        off += e["b"]
        k, shape = e["k"], tuple(e["shape"])
        if k == "five_value":
            recs[e["n"]] = ("fv5", parse_five_value(b, shape))
        elif k.startswith("int"):
            recs[e["n"]] = ("int", parse_intn(b, shape, int(k[3:])))
        elif k == "fp16":
            recs[e["n"]] = ("f16", np.frombuffer(b, dtype=np.float16).reshape(shape).copy())
        else:
            die(f"unknown record kind {k}")
    if off != len(blob):
        die("container has trailing bytes")
    return recs, header["index"]


# ------------------------------------------------------------- features
def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """librosa.filters.mel(sr, n_fft, n_mels, fmin=0, fmax=sr/2, htk=False, norm='slaney')."""
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(f):
        f = np.asanyarray(f, dtype=np.float64)
        m = f / f_sp
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep, m)

    def mel_to_hz(m):
        m = np.asanyarray(m, dtype=np.float64)
        f = f_sp * m
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f)

    fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2.0), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    w = np.zeros((n_mels, 1 + n_fft // 2), dtype=np.float32)
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        w[i] = np.maximum(0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2: n_mels + 2] - mel_f[:n_mels])
    w *= enorm[:, np.newaxis]
    return w


# ------------------------------------------------------------- packing
def fv5_levels(r: dict) -> np.ndarray:
    """Level index 0..4 = {-hi,-lo,0,+lo,+hi}."""
    s = r["sign"].astype(np.int16)
    mag = np.where(r["is_hi"], 2, 1).astype(np.int16)
    return (2 + s * mag).astype(np.uint8)


def pack_pair16(levels: np.ndarray) -> np.ndarray:
    """Kernel layout `fv5-pair16-v1`, [O/2, K] bytes.

    Rows are taken in pairs (2p, 2p+1) and K in groups of 16. For pair p and
    group g, chunk A = [r0 k16g..16g+7, r1 k16g..16g+7] and chunk B =
    [r0 k16g+8..16g+15, r1 k16g+8..16g+15], with 8 added to every r1 level.
    Byte j of the 16-byte block = A[j] | B[j] << 4. A NEON kernel decodes 32
    weights with one load, an AND, a shift and two table lookups (one 16-byte
    table per row pair holds both rows' five int8 levels), straight into the
    2x8 operand order of SMMLA / 8-wide SDOT pairs."""
    o, k = levels.shape
    if o % 2 or k % 16:
        die(f"five-value shape {levels.shape}: rows must be even and width a multiple of 16")
    lv = levels.reshape(o // 2, 2, k // 16, 2, 8)        # pair, row, group, half, 8
    lv = lv.astype(np.uint8).copy()
    lv[:, 1] += 8
    seq = lv.transpose(0, 2, 3, 1, 4).reshape(o // 2, k // 16, 2, 16)   # pair, group, half(A/B), 16
    out = seq[:, :, 0, :] | (seq[:, :, 1, :] << 4)
    return out.reshape(o // 2, k).astype(np.uint8)


def unpack_pair16(packed: np.ndarray) -> np.ndarray:
    p, k = packed.shape
    b = packed.reshape(p, k // 16, 16)
    seq = np.stack([b & 0x0F, b >> 4], axis=2)            # pair, group, half, 16
    seq = seq.reshape(p, k // 16, 2, 2, 8).transpose(0, 3, 1, 2, 4)   # pair, row, group, half, 8
    seq = seq.astype(np.int16)
    seq[:, 1] -= 8
    if seq.min() < 0 or seq.max() > 4:
        die("unpack_pair16: level out of range")
    return seq.reshape(2 * p, k).astype(np.uint8)


def fv5t7_block_bytes(k: int) -> int:
    t = (k // 2 + 2) // 3
    return (t + 7) // 8 * 7 * 16


def pack_b5x3t7(levels: np.ndarray) -> np.ndarray:
    """Kernel layout `fv5-b5x3t7-v1`, [O/8, fv5t7_block_bytes(K)] bytes.

    Per block of 8 rows, take the K/2 16-byte panel vectors of the int8 GEMM
    (vector v = 4c + p holds row 2p at k 8c..8c+7, then row 2p+1) as level
    indices, three vectors at a time: code lane j = l(3t) + 5 l(3t+1) +
    25 l(3t+2) < 125 (vectors past K/2 count as level 2). Codes fit in 7 bits,
    so 8 code vectors share 7 stored vectors: byte m of lane j = code m |
    (bit m of code 7) << 7; a short last group is padded with code 62 (levels
    2, 2, 2). A NEON kernel recovers the digits with two multiply-highs and
    two multiply-subtracts per 16 codes, then makes each panel vector with
    the same 16-byte LUT lookup as pair16."""
    o, k = levels.shape
    if o % 8 or k % 16:
        die(f"five-value shape {levels.shape}: rows must be a multiple of 8 and width a multiple of 16")
    nb, v = o // 8, k // 2
    t = (v + 2) // 3
    g = (t + 7) // 8
    pan = levels.astype(np.uint8).reshape(nb, 4, 2, k // 8, 8).transpose(0, 3, 1, 2, 4).reshape(nb, v, 16)
    pan = np.concatenate([pan, np.full((nb, 3 * t - v, 16), 2, np.uint8)], axis=1).astype(np.uint16)
    pan = pan.reshape(nb, t, 3, 16)
    code = pan[:, :, 0] + 5 * pan[:, :, 1] + 25 * pan[:, :, 2]
    code = np.concatenate([code, np.full((nb, 8 * g - t, 16), 62, np.uint16)], axis=1).reshape(nb, g, 8, 16)
    out = np.empty((nb, g, 7, 16), np.uint8)
    for m in range(7):
        out[:, :, m] = code[:, :, m] | (((code[:, :, 7] >> m) & 1) << 7)
    return out.reshape(nb, g * 112)


def unpack_b5x3t7(packed: np.ndarray, k: int) -> np.ndarray:
    nb, nbytes = packed.shape
    if nbytes != fv5t7_block_bytes(k):
        die(f"unpack_b5x3t7: {nbytes} bytes per block, expected {fv5t7_block_bytes(k)} for K={k}")
    v = k // 2
    t = (v + 2) // 3
    g = (t + 7) // 8
    b = packed.reshape(nb, g, 7, 16).astype(np.uint16)
    code = np.empty((nb, g, 8, 16), np.uint16)
    code[:, :, :7] = b & 0x7F
    code[:, :, 7] = sum(((b[:, :, m] >> 7) << m) for m in range(7))
    code = code.reshape(nb, 8 * g, 16)
    if code[:, :t].max() > 124 or (8 * g > t and np.any(code[:, t:] != 62)):
        die("unpack_b5x3t7: code out of range")
    pan = np.stack([code[:, :t] % 5, code[:, :t] // 5 % 5, code[:, :t] // 25], axis=2).reshape(nb, 3 * t, 16)
    if np.any(pan[:, v:] != 2):
        die("unpack_b5x3t7: padding is not level 2")
    pan = pan[:, :v].reshape(nb, k // 8, 4, 2, 8).transpose(0, 2, 3, 1, 4)
    return pan.reshape(nb * 8, k).astype(np.uint8)


LAYOUTS = {"b5x3t7": ("fv5t7", "fv5-b5x3t7-v1"), "pair16": ("fv5", "fv5-pair16-v1")}


def fv5_expand(levels: np.ndarray, lohi: np.ndarray) -> np.ndarray:
    lo = lohi[:, 0:1].astype(np.float16)
    hi = lohi[:, 1:2].astype(np.float16)
    table = np.stack([-hi, -lo, np.zeros_like(lo), lo, hi], axis=-1)[:, 0, :]  # [O,5] fp16
    return np.take_along_axis(table, levels.astype(np.int64), axis=1)


def container_fv5_expand(r: dict) -> np.ndarray:
    mag = np.where(r["is_hi"], r["hi"][:, None], r["lo"][:, None])
    return (r["sign"].astype(np.float16) * mag).astype(np.float16)


# --------------------------------------------------------------- writer
class Builder:
    def __init__(self, recs: dict, layout: str):
        self.recs = recs
        self.layout = layout
        self.used: set[str] = set()
        self.tensors: list[tuple[str, np.ndarray]] = []
        self.fv5_src: dict[str, list[str]] = {}
        self.fv5_k: dict[str, int] = {}
        self.int_src: dict[str, str] = {}

    def take(self, name: str, kind: str):
        if name not in self.recs:
            die(f"missing record {name}")
        k, v = self.recs[name]
        if k != kind:
            die(f"record {name}: kind {k} != {kind}")
        self.used.add(name)
        return v

    def add(self, name: str, arr: np.ndarray):
        self.tensors.append((name, np.ascontiguousarray(arr)))

    def fv5(self, out: str, names: list[str], o_expect: int, i_expect: int):
        lv, lohi = [], []
        for n in names:
            r = self.take(n, "fv5")
            o, i = r["sign"].shape
            if i != i_expect:
                die(f"{n}: width {i} != {i_expect}")
            lv.append(fv5_levels(r))
            lohi.append(np.stack([r["lo"].astype(np.float32), r["hi"].astype(np.float32)], axis=1))
        levels = np.concatenate(lv, axis=0)
        if levels.shape[0] != o_expect:
            die(f"{out}: rows {levels.shape[0]} != {o_expect}")
        if self.layout == "b5x3t7":
            self.add(out + ".fv5t7", pack_b5x3t7(levels).view(np.int8))
        else:
            self.add(out + ".fv5", pack_pair16(levels).view(np.int8))
        self.add(out + ".lohi", np.concatenate(lohi, axis=0).astype(np.float32))
        self.fv5_src[out] = names
        self.fv5_k[out] = i_expect

    def intq(self, out: str, name: str, shape2d: tuple[int, int] | None = None):
        r = self.take(name, "int")
        q = r["q"]
        if shape2d is not None and q.shape != shape2d:
            die(f"{name}: shape {q.shape} != {shape2d}")
        self.add(out + ".q", q.astype(np.int8))
        self.add(out + ".scale", r["scale"].astype(np.float32))
        self.int_src[out] = name

    def f32(self, out: str, name: str, shape=None):
        v = self.take(name, "f16").astype(np.float32)
        if shape is not None:
            v = v.reshape(shape)
        self.add(out, v)


def build(recs: dict, layout: str) -> Builder:
    b = Builder(recs, layout)
    d, ff, L = HP["d_model"], HP["ff_dim"], HP["n_layers"]
    C = HP["sub_channels"]
    # subsampling (dw_striding): conv0 (1->C, 3x3 s2) relu, [dw 3x3 s2, pw 1x1, relu] x2, linear
    b.f32("enc.sub.conv0.w", "encoder.subsampling.layers.0.weight", (C, 9))
    b.f32("enc.sub.conv0.b", "encoder.subsampling.layers.0.bias")
    for src, dst in ((2, "dw1"), (5, "dw2")):
        b.f32(f"enc.sub.{dst}.w", f"encoder.subsampling.layers.{src}.weight", (C, 9))
        b.f32(f"enc.sub.{dst}.b", f"encoder.subsampling.layers.{src}.bias")
    for src, dst in ((3, "pw1"), (6, "pw2")):
        b.intq(f"enc.sub.{dst}", f"encoder.subsampling.layers.{src}.weight", (C, C))
        b.f32(f"enc.sub.{dst}.b", f"encoder.subsampling.layers.{src}.bias")
    b.intq("enc.sub.out", "encoder.subsampling.linear.weight", (d, C * (HP["n_mels"] // 8)))
    b.f32("enc.sub.out.b", "encoder.subsampling.linear.bias")

    for l in range(L):
        p = f"encoder.layers.{l}."
        o = f"enc.l{l}."
        b.fv5(o + "ff1.w1", [p + "feed_forward1.linear1"], ff, d)
        b.fv5(o + "ff1.w2", [p + "feed_forward1.linear2"], d, ff)
        b.fv5(o + "attn.qkv", [p + "self_attn.q_proj", p + "self_attn.k_proj", p + "self_attn.v_proj"], 3 * d, d)
        b.fv5(o + "attn.out", [p + "self_attn.o_proj"], d, d)
        b.fv5(o + "attn.pos", [p + "self_attn.relative_k_proj"], d, d)
        b.fv5(o + "conv.pw1", [p + "conv.pointwise_conv1"], 2 * d, d)
        b.fv5(o + "conv.pw2", [p + "conv.pointwise_conv2"], d, d)
        b.fv5(o + "ff2.w1", [p + "feed_forward2.linear1"], ff, d)
        b.fv5(o + "ff2.w2", [p + "feed_forward2.linear2"], d, ff)
        b.f32(o + "attn.bias_u", p + "self_attn.bias_u")
        b.f32(o + "attn.bias_v", p + "self_attn.bias_v")
        for nm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            b.f32(o + nm + ".w", p + nm + ".weight")
            b.f32(o + nm + ".b", p + nm + ".bias")
        # depthwise conv (int6, no bias) with BatchNorm folded, in float64
        r = b.take(p + "conv.depthwise_conv.weight", "int")
        k = HP["conv_kernel"]
        w = r["q"].astype(np.float64).reshape(d, k) * r["scale"].astype(np.float64)[:, None]
        gamma = b.take(p + "conv.norm.weight", "f16").astype(np.float64)
        beta = b.take(p + "conv.norm.bias", "f16").astype(np.float64)
        mean = b.take(p + "conv.norm.running_mean", "f16").astype(np.float64)
        var = b.take(p + "conv.norm.running_var", "f16").astype(np.float64)
        b.take(p + "conv.norm.num_batches_tracked", "f16")  # unused at inference (stored as fp16, may be inf)
        inv = gamma / np.sqrt(var + HP["bn_eps"])
        b.add(o + "conv.dw.w", (w * inv[:, None]).astype(np.float32))
        b.add(o + "conv.dw.b", (beta - mean * inv).astype(np.float32))

    H, V = HP["pred_hidden"], HP["vocab"]
    b.intq("enc_proj", "encoder_projector.weight", (HP["joint_hidden"], d))
    b.f32("enc_proj.b", "encoder_projector.bias")
    b.intq("dec.embed", "decoder.embedding.weight", (V + 1, H))
    for l in range(HP["pred_layers"]):
        b.intq(f"dec.lstm{l}.ih", f"decoder.lstm.weight_ih_l{l}", (4 * H, H))
        b.intq(f"dec.lstm{l}.hh", f"decoder.lstm.weight_hh_l{l}", (4 * H, H))
        bi = b.take(f"decoder.lstm.bias_ih_l{l}", "f16").astype(np.float64)
        bh = b.take(f"decoder.lstm.bias_hh_l{l}", "f16").astype(np.float64)
        b.add(f"dec.lstm{l}.b", (bi + bh).astype(np.float32))
    b.intq("dec.proj", "decoder.decoder_projector.weight", (HP["joint_hidden"], H))
    b.f32("dec.proj.b", "decoder.decoder_projector.bias")
    b.intq("joint.head", "joint.head.weight", (V + 1 + len(HP["durations"]), HP["joint_hidden"]))
    b.f32("joint.head.b", "joint.head.bias")

    unused = sorted(set(recs) - b.used)
    if unused:
        die(f"{len(unused)} container records not consumed, e.g. {unused[:4]}")
    b.add("feat.mel_fb", mel_filterbank(HP["sample_rate"], HP["n_fft"], HP["n_mels"]))
    return b


def git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def write_gguf(out: Path, b: Builder, labels: list[str], notice: str, pins: dict, src: dict):
    w = gguf.GGUFWriter(str(out), ARCH)
    w.add_name("Phonon-2")
    w.add_string("general.license", "CC-BY-4.0")
    w.add_string("general.license.link", "https://creativecommons.org/licenses/by/4.0/")
    w.add_string("general.base_model", pins["model"]["base_model"])
    w.add_string(f"{NS}.source.repo", pins["model"]["repo"])
    w.add_string(f"{NS}.source.revision", pins["model"]["revision"])
    for k, v in src.items():
        w.add_string(f"{NS}.source.{k}", v)
    w.add_string(f"{NS}.notice", notice)
    w.add_string(f"{NS}.converter.blob_sha1", git_blob_sha(Path(__file__)))
    w.add_string(f"{NS}.weights.format", LAYOUTS[b.layout][1])
    for key in ("d_model", "n_layers", "n_heads", "ff_dim", "conv_kernel", "sub_channels", "sub_factor",
                "n_mels", "sample_rate", "n_fft", "win_length", "hop_length", "pos_max_len",
                "pred_hidden", "pred_layers", "joint_hidden", "vocab", "blank_id", "max_symbols"):
        w.add_uint32(f"{NS}.{key}", int(HP[key]))
    for key in ("preemph", "log_guard", "norm_eps", "ln_eps"):
        w.add_float32(f"{NS}.{key}", float(HP[key]))
    w.add_array(f"{NS}.durations", [int(x) for x in HP["durations"]])
    w.add_string(f"{NS}.language", "en")
    w.add_tokenizer_model("sentencepiece-bpe")
    w.add_token_list(labels)
    for name, arr in b.tensors:
        if arr.dtype == np.float32:
            w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.F32)
        elif arr.dtype == np.int8:
            w.add_tensor(name, arr, raw_dtype=gguf.GGMLQuantizationType.I8)
        else:
            die(f"{name}: unexpected dtype {arr.dtype}")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def verify(out: Path, b: Builder, recs: dict):
    r = gguf.GGUFReader(str(out))
    t = {x.name: np.array(x.data) for x in r.tensors}
    nfv = nint = 0
    for name, srcs in b.fv5_src.items():
        lohi = t[name + ".lohi"]
        if b.layout == "b5x3t7":
            lv = unpack_b5x3t7(t[name + ".fv5t7"].view(np.uint8), b.fv5_k[name])
        else:
            lv = unpack_pair16(t[name + ".fv5"].view(np.uint8))
        got = fv5_expand(lv, lohi)
        row = 0
        for s in srcs:
            ref = container_fv5_expand(recs[s][1])
            o = ref.shape[0]
            if not np.array_equal(got[row: row + o].view(np.uint16), ref.view(np.uint16)):
                die(f"verify: {name} rows {row}:{row + o} differ from container record {s}")
            row += o
            nfv += 1
    for name, s in b.int_src.items():
        q = t[name + ".q"].astype(np.float32)
        sc = t[name + ".scale"].astype(np.float32)
        r0 = recs[s][1]
        ref = r0["q"].astype(np.float32) * r0["scale"].astype(np.float32)[:, None]
        if not np.array_equal((q * sc[:, None]).view(np.uint32), ref.view(np.uint32)):
            die(f"verify: {name} differs from container record {s}")
        nint += 1
    try:
        import librosa  # type: ignore
        ref = librosa.filters.mel(sr=HP["sample_rate"], n_fft=HP["n_fft"], n_mels=HP["n_mels"],
                                  fmin=0.0, fmax=HP["sample_rate"] / 2, norm="slaney")
        err = float(np.max(np.abs(ref.astype(np.float32) - t["feat.mel_fb"])))
        if err > 1e-7:
            die(f"verify: mel filterbank differs from librosa by {err}")
        mel = f"librosa {getattr(librosa, '__version__', '?')} max abs diff {err:.1e}"
    except ImportError:
        mel = "librosa not installed; filterbank not cross-checked"
    print(f"verify: OK, {nfv} five-value records and {nint} int records bit-identical; "
          f"{len(recs)} records consumed; mel: {mel}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", type=Path, help="phonon-2.bps.tar.zst at the pinned revision")
    ap.add_argument("--container", type=Path, help="unpacked model.fermion")
    ap.add_argument("--nemo-config", type=Path, help="unpacked config.json (with 'labels')")
    ap.add_argument("--notice", type=Path, help="NOTICE from the model repo (default: next to --archive)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--layout", choices=sorted(LAYOUTS), default="b5x3t7",
                    help="five-value code layout (default b5x3t7, 2.35-2.41 bits per weight; pair16 is 4 bits)")
    a = ap.parse_args()
    pins = json.loads(PINS_PATH.read_text())

    if a.archive:
        members = read_archive(a.archive, pins)
        container, cfg_bytes = members["model.fermion"], members["config.json"]
        notice_path = a.notice or a.archive.parent / "NOTICE"
    elif a.container and a.nemo_config:
        container = a.container.read_bytes()
        check_pin("model.fermion", container, pins["archive_members"]["model.fermion"])
        cfg_bytes = a.nemo_config.read_bytes()
        check_pin("config.json", cfg_bytes, pins["archive_members"]["config.json"])
        notice_path = a.notice or a.container.parent / "NOTICE"
    else:
        ap.error("give --archive, or --container and --nemo-config")
    if not notice_path.exists():
        die(f"NOTICE not found at {notice_path} (pass --notice); CC-BY-4.0 attribution must ship with the GGUF")
    notice = notice_path.read_bytes()
    check_pin("NOTICE", notice, pins["files"]["NOTICE"])

    cfg = json.loads(cfg_bytes)
    labels = list(cfg["labels"])
    if len(labels) != HP["vocab"]:
        die(f"labels: {len(labels)} != {HP['vocab']}")
    pre = cfg["preprocessor"]
    enc = cfg["encoder"]
    checks = [
        (pre["sample_rate"], HP["sample_rate"]), (pre["features"], HP["n_mels"]), (pre["n_fft"], HP["n_fft"]),
        (round(pre["window_size"] * pre["sample_rate"]), HP["win_length"]),
        (round(pre["window_stride"] * pre["sample_rate"]), HP["hop_length"]), (pre["normalize"], "per_feature"),
        (enc["d_model"], HP["d_model"]), (enc["n_layers"], HP["n_layers"]), (enc["n_heads"], HP["n_heads"]),
        (enc["conv_kernel_size"], HP["conv_kernel"]), (enc["subsampling"], "dw_striding"),
        (enc["subsampling_factor"], HP["sub_factor"]), (enc["subsampling_conv_channels"], HP["sub_channels"]),
        (enc["self_attention_model"], "rel_pos"), (list(enc["att_context_size"]), [-1, -1]),
        (enc["xscaling"], False), (enc["untie_biases"], True), (enc["use_bias"], False),
        (cfg["model_defaults"]["tdt_durations"], HP["durations"]),
    ]
    for got, want in checks:
        if got != want:
            die(f"NeMo config mismatch: {got!r} != {want!r}")

    recs, index = read_container(container)
    if len(index) != 723:
        die(f"container has {len(index)} records, expected 723")
    b = build(recs, a.layout)
    src = {"container_sha256": pins["archive_members"]["model.fermion"]["sha256"],
           "archive_sha256": pins["files"]["phonon-2.bps.tar.zst"]["sha256"]}
    write_gguf(a.output, b, labels, notice.decode("utf-8"), pins, src)
    print(f"wrote {a.output} ({a.output.stat().st_size} bytes, {len(b.tensors)} tensors)")
    if a.verify:
        verify(a.output, b, recs)


if __name__ == "__main__":
    main()
