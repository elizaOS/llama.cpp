#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
dump_paradee_reference.py - PyTorch reference for the tools/paradee parity gate.

Three subcommands:

  phonemes  Phonemize samples/sentences.txt with misaki 0.9.4
            (en.G2P(trf=False, british=False, fallback=EspeakFallback(british=False), unk=""),
            the setup paradee/tts.py and kokoro.KPipeline(lang_code="a") use) and write
            tests/fixtures/sentences.misaki.json. The fixture is committed; `dump` reads it.

  dump      Run the pinned PyTorch Paradee (TextStudent 's+mlp' + StudentDecoder 'A' from
            the pinned github.com/sahilmahendrakar/paradee checkout, kokoro 0.9.4 modules)
            on each fixture sentence and write every stage plus manifest.json for
            test-paradee-parity.

  lock-fixture  Write tests/fixtures/phase_lock_synthetic.f32: phase_lock.lock() (float64
            and the unmodified float32) on deterministic synthetic input, for the
            self-contained test-paradee-phase-lock.

  report    Compare native waveforms (written by test-paradee-parity with
            PARADEE_PARITY_OUT) with the deterministic reference and with the published,
            seeded samples/*_paradee.wav: log-mel L1 (raw and DTW). With --whisper-cli,
            also the word error rate of whisper.cpp on the native and the published audio.
            Report only.

The reference is deterministic. Three documented changes make it so and make it a
well-conditioned target; the native graph implements the same three:

  1. SineGen: rand_ini = 0 and no additive noise (tools/kokoro does the same).
  2. SineGen runs in float64. In float32 the accumulated phase of the 9th harmonic
     reaches ~1e5 rad in a few seconds, where one float32 step is ~1e-2 rad; float64
     makes the source a defined value rather than an accident of float32 rounding.
  3. The harmonic source's STFT (the generator's `har` input) is computed in float64
     and values below the DFT's rounding floor are set to exactly zero before the
     phase is taken. Without noise, an unvoiced source is a constant, its STFT bins
     2..n_fft/2 are pure rounding noise, and their phase is otherwise arbitrary.
     The inverse STFT is export_paradee.py's ConvSTFT.inverse (= torch.istft).

The final waveform applies phase_lock.lock() in float64. lock_f32 is the unmodified
float32 phase_lock.lock() on the same decoder output, for the Lock-only check.

`dump --version 1.1` runs Paradee-8M v1.1 (pins.json "v1.1"): the same text side, then
vhead.VDecoder(head="v2") loaded with load_ckpt, as export_paradee_v1_1.py builds it. The
v1.1 head draws Gaussian noise, so the reference fixes it: each sentence's noise is
torch.randn from torch.Generator().manual_seed(--noise-seed + index), or read from
--noise-dir/<index>.noise.f32, and is written out as the `noise` stage for the native
test to feed back in. Nothing else in the v1.1 reference is changed; it runs the
upstream module in float32. Extra v1.1 stages: trunk (AdaIN trunk output), hn (the
head's ConvNeXt stack after norm1), sine (vhead.sine_bank), sine64 (the same sine bank
with the harmonics evaluated in float64: float32's 2*pi*k*cyc reaches ~440 rad, where one
float32 step is ~3e-5 rad, so the float32 bank is up to ~9e-4 x RMS from its exact value),
main (the n_fft 600 path, head run with burst=False), burst (the 150/75 noise-burst path)
and wav (head output).

`dump --version 1.2 --voice NAME` runs Paradee-8M v1.2 (pins.json "v1.2"). Its training scripts are
byte-identical to v1.1's and its checkpoints have v1.1's layout except the two style tables, one row
per voice. The reference loads row `NAME` of each table into the v1.1 modules' (1, dim) style
parameter (what selecting a voice means: style.expand(B, -1) of that row) and is otherwise the v1.1
reference. The manifest records the voice and its row.

usage:
  dump_paradee_reference.py phonemes --model-dir HF_SNAPSHOT --out tools/paradee/tests/fixtures/sentences.misaki.json
  dump_paradee_reference.py dump --model-dir HF_SNAPSHOT --paradee-src PARADEE_CHECKOUT --out REF_DIR
  dump_paradee_reference.py report --model-dir HF_SNAPSHOT --ref REF_DIR --native NATIVE_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARADEE_DIR = HERE.parent
PINS = json.loads((PARADEE_DIR / "pins.json").read_text(encoding="utf-8"))
DEFAULT_FIXTURE = PARADEE_DIR / "tests" / "fixtures" / "sentences.misaki.json"
SR = 24000


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check_file(root: Path, rel: str, pin: dict | str) -> Path:
    path = root / rel
    want = pin if isinstance(pin, str) else pin["sha256"]
    got = sha256_file(path)
    if got != want:
        raise SystemExit(f"{path}: sha256 {got} != pinned {want}")
    return path


def check_versions(*names: str) -> None:
    env = PINS["reference_env"]
    for name in names:
        got = importlib.metadata.version(name)
        if got != env[name]:
            raise SystemExit(f"{name} {got} != pinned {env[name]} (see pins.json reference_env)")


def read_sentences(model_dir: Path) -> list[str]:
    path = check_file(model_dir, "samples/sentences.txt", PINS["files"]["samples/sentences.txt"])
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        num, _, text = line.partition(". ")   # "1. As of August 2015, ..."
        if not num.isdigit():
            raise SystemExit(f"unexpected sentences.txt line {line!r}")
        out.append(text)
    return out


# ---------------------------------------------------------------------------
# phonemes
# ---------------------------------------------------------------------------

def cmd_phonemes(args) -> int:
    check_versions("misaki", "spacy", "en_core_web_sm")
    import espeakng_loader
    if args.espeak_lib:
        espeakng_loader.get_library_path = lambda: args.espeak_lib
    if args.espeak_data:
        espeakng_loader.get_data_path = lambda: args.espeak_data
    from misaki import en, espeak

    fb = espeak.EspeakFallback(british=False)
    g2p = en.G2P(trf=False, british=False, fallback=fb, unk="")
    espeak_version = fb.backend.version()
    vocab = json.loads(check_file(args.model_dir, "config.json", PINS["files"]["config.json"]).read_text(encoding="utf-8"))["vocab"]
    items = []
    for i, text in enumerate(read_sentences(args.model_dir), 1):
        ps, _ = g2p(text)
        missing = sorted({c for c in ps if c not in vocab})
        if missing:
            raise SystemExit(f"sentence {i}: phonemes outside the vocabulary: {missing}")
        items.append({"index": i, "text": text, "phonemes": ps,
                      "ids": [0] + [vocab[c] for c in ps] + [0]})
    doc = {
        "source": f"{PINS['model']['repo']}@{PINS['model']['revision']} samples/sentences.txt",
        "g2p": "misaki en.G2P(trf=False, british=False, fallback=EspeakFallback(british=False), unk='')",
        "versions": {"misaki": importlib.metadata.version("misaki"), "spacy": importlib.metadata.version("spacy"),
                     "en_core_web_sm": importlib.metadata.version("en_core_web_sm"),
                     "espeak-ng": ".".join(str(x) for x in espeak_version)},
        "sentences": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} sha256 {sha256_file(args.out)}")
    return 0


# ---------------------------------------------------------------------------
# reference model
# ---------------------------------------------------------------------------

def load_reference_v11(model_dir: Path, paradee_src: Path, version: str = "1.1", voice: str = "af_heart"):
    """Paradee-8M v1.1 in float32, as export_paradee_v1_1.py builds it. version 1.2: the same
    modules with row `voice` of v1.2's style tables."""
    pins = PINS[f"v{version}"]
    check_versions("torch", "kokoro", "misaki")
    for rel, digest in pins["source"]["files"].items():
        check_file(paradee_src, rel, digest)
    for rel in ("pytorch/text_side.pt", "pytorch/decoder.pt"):
        check_file(model_dir, rel, pins["files"][rel])
    import torch
    sys.path.insert(0, str(paradee_src / "training" / "scripts"))
    student = importlib.import_module("student")
    vhead = importlib.import_module("vhead")
    from torch.nn.utils import remove_weight_norm
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations
    torch.set_grad_enabled(False)

    def strip(m):  # export_paradee_v1_1.py strip()
        for x in m.modules():
            try:
                remove_weight_norm(x)
            except Exception:
                pass
            if is_parametrized(x, "weight"):
                try:
                    remove_parametrizations(x, "weight", leave_parametrized=True)
                except Exception:
                    pass

    ts = student.TextStudent("s+mlp").eval()
    vd = vhead.VDecoder(head="v2").eval()
    if version == "1.2":
        text_ck = torch.load(model_dir / "pytorch/text_side.pt", map_location="cpu", weights_only=True)
        dec_ck = torch.load(model_dir / "pytorch/decoder.pt", map_location="cpu", weights_only=True)
        voices = list(text_ck["voices"])
        if voices != pins["voices"] or list(dec_ck["voices"]) != voices or voice not in voices:
            raise SystemExit(f"voice {voice!r} / checkpoint voices {voices!r} do not match pins.json")
        row = voices.index(voice)
        tsd, dsd = dict(text_ck["model"]), dict(dec_ck["model"])
        tsd["style"], dsd["style"] = tsd["style"][row:row + 1].clone(), dsd["style"][row:row + 1].clone()
        ts.load_state_dict(tsd)
        res = vd.load_state_dict(dsd, strict=False)   # VDecoder.load_ckpt's checks
        if res.missing_keys or res.unexpected_keys:
            raise SystemExit(f"v1.2 decoder.pt does not fit VDecoder(head='v2'): {res}")
    else:
        ts.load_state_dict(torch.load(model_dir / "pytorch/text_side.pt", map_location="cpu", weights_only=True)["model"])
        vd.load_ckpt(str(model_dir / "pytorch/decoder.pt"))
    if not vd.head.burst:
        raise SystemExit("v1.1 decoder.pt has no noise-burst path")
    strip(ts)
    strip(vd)
    return torch, ts, vd, vhead, student


def text_side(torch, ts, student, ids: list[int], speed: float):
    """export_paradee.py Paradee.text() (the same in export_paradee_v1_1.py), with the stages kept."""
    import torch.nn as nn
    x_ids = torch.tensor([ids], dtype=torch.long)
    p, s = ts.predictor, ts.style
    bert = ts.bert(x_ids, attention_mask=torch.ones_like(x_ids))                  # [1,T,256]
    d_en = ts.bert_encoder(bert).transpose(-1, -2)
    T = x_ids.shape[1]
    sx = s[:, :, None].expand(-1, -1, T)
    x = torch.cat([d_en, sx], 1)
    for b in p.text_encoder.lstms:
        if isinstance(b, nn.LSTM):
            x = b(x.transpose(-1, -2))[0].transpose(-1, -2)
        else:
            x = torch.cat([b(x.transpose(-1, -2), s).transpose(-1, -2), sx], 1)
    d = x.transpose(-1, -2)                                                        # [1,T,224]
    dur = torch.sigmoid(p.duration_proj(p.lstm(d)[0])).sum(-1)[0]                  # [T]
    e = ts.text_encoder
    h = e.embedding(x_ids).transpose(1, 2)
    for c in e.cnn:
        h = c(h)
    t_en = e.lstm(h.transpose(1, 2))[0]
    asr_tok = ts.asr_proj(t_en).transpose(-1, -2)                                  # [1,512,T]
    n = torch.clamp(torch.round(dur / speed), min=1).long()
    aln = student.alignment(n)[None]                                               # [1,T,F]
    F0, N = p.F0Ntrain(d.transpose(-1, -2) @ aln, s)
    asr = asr_tok @ aln                                                            # [1,512,F]
    return bert, d, dur, n, F0, N, asr


def sine_bank64(torch, F0, K=64, hop=300, sr=24000):
    """vhead.sine_bank with sin(2 pi k cyc) and the harmonic sum in float64. The running
    phase c, its step d, the interpolation weight and cyc are float32 as in the module."""
    import torch.nn.functional as F
    n = F0.shape[1]
    r = F0.double() * hop / sr
    c = (torch.cumsum(r, 1) % 1).float()
    d = F.pad(r[:, 1:], (0, 1)).float()
    p = (torch.arange(n * hop) + 0.5) / hop - 0.5
    i = p.floor().clamp(0, n - 1).long()
    fr = (p - i).clamp(0, 1)
    cyc = c[:, i] + fr * d[:, i]
    f0s = F0.repeat_interleave(hop, 1)
    k = torch.arange(1, K + 1)[None, :, None]
    amp = ((f0s[:, None] * k < 11000) & (f0s[:, None] > 10)).double()
    return (torch.sin(2 * math.pi * k.double() * cyc.double()[:, None]) * amp).sum(1) * 0.1


def run_reference_v11(torch, ts, vd, vhead, student, ids: list[int], speed: float, noise) -> dict:
    """export_paradee_v1_1.py Paradee2.forward with the torch head (vhead.SourceFilterHead2)
    and the given noise, with the stages kept."""
    bert, d, dur, n, F0, N, asr = text_side(torch, ts, student, ids, speed)
    h = vd.head
    L = F0.shape[1] * vhead.FHOP
    if noise is None:
        raise SystemExit("v1.1 reference needs its noise")
    nz = torch.as_tensor(noise, dtype=torch.float32)[None]
    if nz.shape[1] != L:
        raise SystemExit(f"noise has {nz.shape[1]} samples, the sentence needs {L}")
    cap = {}
    hook = h.norm1.register_forward_hook(lambda m, i, o: cap.__setitem__("hn", o))
    s = vd.style.expand(1, -1)                                                     # VDecoder.forward up to the head
    F0c = vd.F0_conv(F0.unsqueeze(1)); Nc = vd.N_conv(N.unsqueeze(1))
    x = vd.encode(torch.cat([asr, F0c, Nc], 1), s); asr_res = vd.asr_res(asr); res = True
    for block in vd.decode:
        if res:
            x = torch.cat([x, asr_res, F0c, Nc], 1)
        x = block(x, s)
        if block.upsample_type != "none":
            res = False
    wav = h(x, F0, nz)
    gb = h.last["gb"]
    hook.remove()
    h.burst = False
    main = h(x, F0, nz)
    h.burst = True
    Nb = torch.stft(nz.flip(-1), 150, 75, window=h.wb, return_complex=True)
    burst = torch.istft(Nb * gb, 150, 75, window=h.wb, length=L)
    sine = vhead.sine_bank(F0, h.K)[0]
    sine64 = sine_bank64(torch, F0, h.K)
    full = vd(asr, F0, N, noise=nz)                                                # VDecoder.forward itself
    if not torch.equal(full, wav):
        raise SystemExit("VDecoder.forward differs from the staged trunk + head")
    f = lambda t: t.detach().contiguous().float().numpy()
    return {
        "ids": np.asarray(ids, dtype=np.int32),
        "bert": f(bert[0]), "d": f(d[0]), "dur": f(dur / speed), "dur_int": n.numpy().astype(np.int32),
        "F0": f(F0[0]), "N": f(N[0]), "asr": f(asr[0]), "noise": f(nz[0]),
        "trunk": f(x[0]), "hn": f(cap["hn"][0]), "sine": f(sine[0]), "sine64": f(sine64[0]), "main": f(main[0]), "burst": f(burst[0]),
        "wav": f(wav[0]),
    }


def load_reference(model_dir: Path, paradee_src: Path):
    check_versions("torch", "kokoro", "misaki")
    for rel, digest in PINS["source"]["files"].items():
        check_file(paradee_src, rel, digest)
    for rel in ("pytorch/text_side.pt", "pytorch/decoder.pt"):
        check_file(model_dir, rel, PINS["files"][rel])

    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(paradee_src / "training" / "scripts"))
    student = importlib.import_module("student")
    student_decoder = importlib.import_module("student_decoder")
    phase_lock = importlib.import_module("phase_lock")
    from kokoro.istftnet import SineGen
    from torch.nn.utils import remove_weight_norm
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    torch.set_grad_enabled(False)

    def strip(m):  # export_paradee.py strip()
        for x in m.modules():
            try:
                remove_weight_norm(x)
            except Exception:
                pass
            if is_parametrized(x, "weight"):
                try:
                    remove_parametrizations(x, "weight", leave_parametrized=True)
                except Exception:
                    pass

    # (1) + (2): deterministic float64 SineGen (kokoro 0.9.4 SineGen with
    # flag_for_pulse=False, minus rand_ini and noise).
    def f02sine_det(self, f0_values):
        rad_values = (f0_values / self.sampling_rate) % 1
        rad_values = F.interpolate(rad_values.transpose(1, 2), scale_factor=1 / self.upsample_scale,
                                   mode="linear").transpose(1, 2)
        phase = torch.cumsum(rad_values, dim=1) * 2 * torch.pi
        phase = F.interpolate(phase.transpose(1, 2) * self.upsample_scale, scale_factor=self.upsample_scale,
                              mode="linear").transpose(1, 2)
        return torch.sin(phase)

    def sinegen_forward_det(self, f0):
        assert not self.flag_for_pulse
        f0d = f0.double()
        fn = f0d * torch.arange(1, self.harmonic_num + 2, dtype=torch.float64)[None, None, :]
        sine_waves = f02sine_det(self, fn) * self.sine_amp
        uv = (f0d > self.voiced_threshold).double()
        sine_waves = sine_waves * uv
        return sine_waves.float(), uv.float(), torch.zeros_like(sine_waves, dtype=torch.float32)

    SineGen.forward = sinegen_forward_det

    # (3): canonical float64 source STFT + ConvSTFT inverse (export_paradee.py).
    class RefSTFT(torch.nn.Module):
        def __init__(self, n, hop):
            super().__init__()
            self.n, self.hop = n, hop
            w = torch.hann_window(n, dtype=torch.float64)
            k = torch.arange(n // 2 + 1, dtype=torch.float64)[:, None]
            ang = 2 * math.pi * k * torch.arange(n, dtype=torch.float64) / n
            self.re_b, self.im_b = torch.cos(ang) * w, -torch.sin(ang) * w
            self.w64 = w
            # inverse: ConvSTFT.inverse, float32 like the exported graph
            wf = torch.hann_window(n)
            kf = torch.arange(n // 2 + 1).float()[:, None]
            angf = 2 * torch.pi * kf * torch.arange(n).float() / n
            c = torch.full((n // 2 + 1, 1), 2.0); c[0] = c[-1] = 1
            self.inv = (torch.cat([torch.cos(angf) * c, -torch.sin(angf) * c]) / n * wf)[:, None]
            self.w2 = (wf ** 2)[None, None]

        def transform(self, x):
            xd = F.pad(x.double()[:, None], (self.n // 2, self.n // 2), mode="reflect")[:, 0]
            frames = xd.unfold(-1, self.n, self.hop)                       # [B, T, n]
            re = frames @ self.re_b.T                                      # [B, T, K]
            im = frames @ self.im_b.T
            floor = 1e-12 * (frames * self.w64).abs().sum(-1, keepdim=True)
            re = torch.where(re.abs() <= floor, torch.zeros_like(re), re)
            im = torch.where(im.abs() <= floor, torch.zeros_like(im), im)
            mag = torch.sqrt(re ** 2 + im ** 2)
            ph = torch.where((im == 0) & (re < 0), torch.full_like(re, math.pi), torch.atan2(im, re))
            return mag.transpose(1, 2).float(), ph.transpose(1, 2).float()

        def inverse(self, mag, phase):
            z = torch.cat([mag * torch.cos(phase), mag * torch.sin(phase)], 1)
            y = F.conv_transpose1d(z, self.inv, stride=self.hop)[:, 0]
            env = F.conv_transpose1d(torch.ones(1, 1, mag.shape[-1]), self.w2, stride=self.hop)[:, 0]
            return (y / env.clamp_min(1e-8))[:, self.n // 2: -(self.n // 2)][:, None]

    ts = student.TextStudent("s+mlp").eval()
    ts.load_state_dict(torch.load(model_dir / "pytorch/text_side.pt", map_location="cpu", weights_only=True)["model"])
    ds = student_decoder.load_student_decoder("A", str(model_dir / "pytorch/decoder.pt"), disable_complex=True)
    strip(ts)
    strip(ds)
    g = ds.generator
    g.stft = RefSTFT(g.stft.filter_length, g.stft.hop_length)
    cap = {}
    g.m_source.register_forward_hook(lambda m, i, o: cap.__setitem__("src", o[0]))
    return torch, ts, ds, cap, phase_lock, student


def lock64(torch, phase_lock, s, src, f0, L):
    """phase_lock.lock() evaluated in float64 (same operations; the module's
    window and smoothing kernel are float32 constants, so it cannot take
    float64 input as is)."""
    import torch.nn.functional as F
    n, hop = phase_lock.n, phase_lock.hop
    W = torch.hann_window(n, dtype=torch.float64)
    k = min(len(s), len(src))
    s, src = s[:k].double(), src[:k].double()
    S = torch.stft(s, n, hop, window=W, return_complex=True)
    E = torch.stft(src, n, hop, window=W, return_complex=True)
    v = torch.tensor([float(f0[min(j * hop // 300, len(f0) - 1)]) > 60 for j in range(S.shape[1])])[None]
    rel = S * E.conj() / E.abs().clamp_min(1e-6)
    ker = torch.hann_window(L + 2, dtype=torch.float64)[1:-1]
    ker = (ker / ker.sum())[None, None]
    sm = torch.complex(F.conv1d(rel.real[:, None], ker, padding=L // 2)[:, 0],
                       F.conv1d(rel.imag[:, None], ker, padding=L // 2)[:, 0])
    ph = torch.where(phase_lock.BAND & v, E.angle() + sm.angle(), S.angle())
    return torch.istft(torch.polar(S.abs(), ph), n, hop, window=W, length=k)


def run_reference(torch, ts, ds, cap, phase_lock, student, ids: list[int], speed: float) -> dict:
    """export_paradee.py Paradee.text() + forward(), with the stages kept."""
    bert, d, dur, n, F0, N, asr = text_side(torch, ts, student, ids, speed)
    dec = ds(asr, F0, N).squeeze()                                                 # [L]
    src = cap["src"].squeeze()
    k = min(len(dec), len(src))
    lock_f32 = phase_lock.lock(dec, src, F0[0], 33)
    wav = lock64(torch, phase_lock, dec, src, F0[0], 33).float()
    f = lambda t: t.detach().contiguous().float().numpy()
    return {
        "ids": np.asarray(ids, dtype=np.int32),
        "bert": f(bert[0]), "d": f(d[0]), "dur": f(dur / speed), "dur_int": n.numpy().astype(np.int32),
        "F0": f(F0[0]), "N": f(N[0]), "asr": f(asr[0]), "dec": f(dec[:k]), "src": f(src[:k]),
        "wav": f(wav), "lock_f32": f(lock_f32),
    }


def cmd_dump(args) -> int:
    fixture = json.loads(args.phonemes.read_text(encoding="utf-8"))
    v11 = args.version in ("1.1", "1.2")
    pins = PINS[f"v{args.version}"] if v11 else PINS
    if v11:
        torch, ts, vd, vhead, student = load_reference_v11(args.model_dir, args.paradee_src, args.version, args.voice)
    else:
        torch, ts, ds, cap, phase_lock, student = load_reference(args.model_dir, args.paradee_src)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": pins["model"], "source_commit": pins["source"]["commit"], "version": args.version,
        "phonemes_fixture_sha256": sha256_file(args.phonemes),
        "speed": args.speed,
        **({"voice": args.voice, "voice_row": pins["voices"].index(args.voice)} if args.version == "1.2" else {}),
        "versions": {k: importlib.metadata.version(k) for k in ("torch", "kokoro", "misaki")},
        "reference": ("float32 upstream modules; head noise fixed (the noise stage), "
                      f"{'read from ' + str(args.noise_dir) if args.noise_dir else 'torch.randn seeded with %d + index' % args.noise_seed}"
                      if v11 else
                      "deterministic: SineGen rand_ini=0, no noise, float64; canonical float64 source STFT; float64 phase lock"),
        "sentences": [],
    }
    for item in fixture["sentences"]:
        if v11:
            frames = int(text_side(torch, ts, student, item["ids"], args.speed)[3].sum())
            L = 2 * frames * 300
            if args.noise_dir:
                noise = np.fromfile(args.noise_dir / f"{item['index']}.noise.f32", dtype="<f4")
            else:
                noise = torch.randn(L, generator=torch.Generator().manual_seed(args.noise_seed + item["index"])).numpy()
            out = run_reference_v11(torch, ts, vd, vhead, student, item["ids"], args.speed, noise)
        else:
            out = run_reference(torch, ts, ds, cap, phase_lock, student, item["ids"], args.speed)
        files = {}
        for name, arr in out.items():
            ext = "i32" if arr.dtype == np.int32 else "f32"
            fn = f"{item['index']}.{name}.{ext}"
            arr.astype("<i4" if ext == "i32" else "<f4").tofile(args.out / fn)
            files[name] = {"file": fn, "shape": list(arr.shape), "sha256": sha256_file(args.out / fn)}
        manifest["sentences"].append({"index": item["index"], "phonemes": item["phonemes"],
                                      "T_phon": len(item["ids"]), "T_frame": int(out["dur_int"].sum()),
                                      "files": files})
        print(f"sentence {item['index']}: T={len(item['ids'])} frames={int(out['dur_int'].sum())} "
              f"samples={len(out['wav'])}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}/manifest.json")
    return 0


# ---------------------------------------------------------------------------
# lock-fixture: a small, self-contained check of the phase lock alone
# ---------------------------------------------------------------------------

LOCK_FIXTURE_SAMPLES = 9600   # 0.4 s: 38 STFT frames, more than the 33-frame smoothing


def lock_fixture_inputs():
    """Deterministic synthetic lock inputs. tests/test_paradee_phase_lock.cpp
    builds the same arrays with the same formulas (double, then rounded to
    float32). F0 glides 110 -> 220 Hz with an unvoiced gap; src is a tanh-
    shaped harmonic excitation of that F0; s is a sum of its harmonics with a
    slowly drifting phase offset, so the lock has something to correct."""
    n = LOCK_FIXTURE_SAMPLES
    nf0 = n // 300
    f0 = np.array([0.0 if 12 <= j < 16 else 110.0 + 110.0 * j / (nf0 - 1) for j in range(nf0)])
    ph = 0.0
    s = np.zeros(n); src = np.zeros(n)
    for t in range(n):
        f = f0[t // 300]
        ph += 2 * np.pi * f / SR
        voiced = f > 0
        src[t] = np.tanh(0.8 * np.sin(ph)) * (1.0 if voiced else 0.0) + 0.01 * np.sin(0.37 * t)
        acc = 0.0
        for h in range(1, 40):
            if f * h >= 11000:
                break
            acc += np.sin(h * ph + 0.002 * h * t / 7.0 + 0.5 * h) / h
        s[t] = 0.2 * acc * (1.0 if voiced else 0.3) + 0.003 * np.sin(1.3 * t)
    return s.astype(np.float32), src.astype(np.float32), f0.astype(np.float32)


def cmd_lock_fixture(args) -> int:
    for rel, digest in PINS["source"]["files"].items():
        check_file(args.paradee_src, rel, digest)
    import torch
    sys.path.insert(0, str(args.paradee_src / "training" / "scripts"))
    phase_lock = importlib.import_module("phase_lock")
    s, src, f0 = lock_fixture_inputs()
    ts_, tsrc, tf0 = torch.from_numpy(s), torch.from_numpy(src), torch.from_numpy(f0)
    with torch.no_grad():
        out64 = lock64(torch, phase_lock, ts_, tsrc, tf0, 33).float().numpy()
        out32 = phase_lock.lock(ts_, tsrc, tf0, 33).numpy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.concatenate([out64, out32]).astype("<f4").tofile(args.out)
    print(f"wrote {args.out}: lock64 + lock32 ({LOCK_FIXTURE_SAMPLES} samples each), "
          f"change rms {np.sqrt(np.mean((out64 - s) ** 2)):.4g}, |lock64-lock32| max {np.abs(out64 - out32).max():.3g}, "
          f"sha256 {sha256_file(args.out)}")
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """HTK-scale triangular mel filters (torchaudio MelSpectrogram defaults: htk, no norm)."""
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    pts = mel2hz(np.linspace(hz2mel(0.0), hz2mel(sr / 2), n_mels + 2))
    fb = np.zeros((n_fft // 2 + 1, n_mels))
    for m in range(n_mels):
        lo, c, hi = pts[m], pts[m + 1], pts[m + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[:, m] = np.maximum(0.0, np.minimum(up, down))
    return fb


def logmel(x: np.ndarray) -> np.ndarray:
    """log(MelSpectrogram(24000, n_fft=1024, hop=256, n_mels=80)) as training/scripts/common.py."""
    n_fft, hop = 1024, 256
    x = x / max(np.sqrt(np.mean(x.astype(np.float64) ** 2)), 1e-8)
    xp = np.pad(x.astype(np.float64), (n_fft // 2, n_fft // 2), mode="reflect")
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft)[::hop] * w
    power = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
    return np.log(np.maximum(power @ mel_filterbank(SR, n_fft, 80), 1e-5)).T   # [80, T]


def dtw_l1(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.T, b.T
    C = np.abs(a[:, None, :] - b[None, :, :]).mean(-1)
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        Di, Dp, Ci = D[i], D[i - 1], C[i - 1]
        for j in range(1, m + 1):
            Di[j] = Ci[j - 1] + min(Dp[j], Di[j - 1], Dp[j - 1])
    i, j, path = n, m, 0
    while i > 0 and j > 0:
        path += 1
        k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return float(D[n, m] / path)


def read_wav_f32(path: Path) -> np.ndarray:
    import wave
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def compare(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    la, lb = logmel(a), logmel(b)
    k = min(la.shape[1], lb.shape[1])
    return float(np.abs(la[:, :k] - lb[:, :k]).mean()), dtw_l1(la, lb)


def write_wav16k(path: Path, x: np.ndarray) -> None:
    """24 kHz float -> 16 kHz 16-bit WAV (band-limited FFT resampling) for whisper.cpp."""
    import wave
    n = len(x); m = int(round(n * 16000 / SR))
    X = np.fft.rfft(x.astype(np.float64))
    Y = np.zeros(m // 2 + 1, complex); k = min(len(X), len(Y)); Y[:k] = X[:k]
    y = np.clip(np.fft.irfft(Y, m) * (m / n), -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((y * 32767).astype("<i2").tobytes())


def wer_words(text: str) -> list[str]:
    """Lower-case words with digits spelled out (years as years), punctuation and 'and' dropped,
    so '2015' and 'twenty fifteen' compare equal."""
    import re
    from num2words import num2words
    t = text.lower().replace("-", " ")
    t = re.sub(r"(\d+)'s", r"\1s", t)
    t = re.sub(r"\d+", lambda m: num2words(int(m.group(0)), to="year") if 1100 <= int(m.group(0)) <= 2099
               else num2words(int(m.group(0))), t)
    t = re.sub(r"[^a-z' ]", " ", t.replace("-", " "))
    return [w for w in t.split() if w != "and"]


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)]


def report_wer(args, manifest, texts) -> None:
    import subprocess
    import tempfile
    totals = {"native": [0, 0], "published": [0, 0]}
    with tempfile.TemporaryDirectory() as tmp:
        for s in manifest["sentences"]:
            i = s["index"]
            audio = {"native": np.fromfile(args.native / f"{i}.wav.f32", dtype="<f4"),
                     "published": read_wav_f32(args.model_dir / f"samples/{i}_paradee.wav")}
            for name, x in audio.items():
                wav = Path(tmp) / f"{i}_{name}.wav"
                write_wav16k(wav, x)
                hyp = subprocess.run([str(args.whisper_cli), "-m", str(args.whisper_model), "-f", str(wav), "-nt", "-np"],
                                     capture_output=True, text=True, check=True).stdout.strip()
                ref_words = wer_words(texts[i])
                errors = edit_distance(ref_words, wer_words(hyp))
                totals[name][0] += errors; totals[name][1] += len(ref_words)
                print(f"{i} {name:9s} {errors:2d}/{len(ref_words):2d}  {hyp}")
    for name, (e, n) in totals.items():
        print(f"{name}: WER {100.0 * e / n:.1f}% ({e}/{n} words)")


def cmd_report(args) -> int:
    manifest = json.loads((args.ref / "manifest.json").read_text(encoding="utf-8"))
    rows = []
    for s in manifest["sentences"]:
        i = s["index"]
        ref = np.fromfile(args.ref / s["files"]["wav"]["file"], dtype="<f4")
        nat = np.fromfile(args.native / f"{i}.wav.f32", dtype="<f4")
        pub = read_wav_f32(check_file(args.model_dir, f"samples/{i}_paradee.wav", PINS["files"][f"samples/{i}_paradee.wav"]))
        rows.append((i, len(nat) / SR, len(pub) / SR, *compare(nat, ref), *compare(nat, pub), *compare(ref, pub)))
    print("log-mel L1 (RMS-normalised, raw / DTW). 'published' = seeded samples/*_paradee.wav (ONNX, with noise).")
    print(f"{'#':>2} {'native s':>8} {'pub s':>6} | {'nat-ref raw':>11} {'dtw':>6} | {'nat-pub raw':>11} {'dtw':>6} | {'ref-pub raw':>11} {'dtw':>6}")
    for r in rows:
        print(f"{r[0]:>2} {r[1]:8.2f} {r[2]:6.2f} | {r[3]:11.4f} {r[4]:6.4f} | {r[5]:11.4f} {r[6]:6.4f} | {r[7]:11.4f} {r[8]:6.4f}")
    m = np.mean(np.array([r[3:] for r in rows]), axis=0)
    print(f"mean               | {m[0]:11.4f} {m[1]:6.4f} | {m[2]:11.4f} {m[3]:6.4f} | {m[4]:11.4f} {m[5]:6.4f}")
    if args.whisper_cli:
        fixture = json.loads(args.phonemes.read_text(encoding="utf-8"))
        texts = {s["index"]: s["text"] for s in fixture["sentences"]}
        print(f"\nWER with whisper.cpp ({args.whisper_model.name}), native vs the published samples:")
        report_wer(args, manifest, texts)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("phonemes")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument("--espeak-lib", help="libespeak-ng to use instead of espeakng_loader's")
    p.add_argument("--espeak-data", help="espeak-ng-data directory to use instead of espeakng_loader's")
    p = sub.add_parser("dump")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--paradee-src", type=Path, required=True)
    p.add_argument("--phonemes", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--version", choices=("1.0", "1.1", "1.2"), default="1.0",
                   help="1.1: Paradee-8M v1.1 (source-filter head; --model-dir and --paradee-src at the v1.1 pins); "
                        "1.2: v1.2 (four voices, see --voice)")
    p.add_argument("--voice", default="af_heart", help="1.2: the voice (style row) to run")
    p.add_argument("--noise-seed", type=int, default=0, help="v1.1: noise = torch.randn seeded with this + index")
    p.add_argument("--noise-dir", type=Path, help="v1.1: read <index>.noise.f32 from here instead")
    p = sub.add_parser("lock-fixture")
    p.add_argument("--paradee-src", type=Path, required=True)
    p.add_argument("--out", type=Path, default=PARADEE_DIR / "tests" / "fixtures" / "phase_lock_synthetic.f32")
    p = sub.add_parser("report")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--phonemes", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument("--whisper-cli", type=Path, help="whisper.cpp whisper-cli binary: also report WER")
    p.add_argument("--whisper-model", type=Path, help="whisper.cpp ggml model (e.g. ggml-base.en.bin)")
    args = ap.parse_args(argv)
    return {"phonemes": cmd_phonemes, "dump": cmd_dump, "lock-fixture": cmd_lock_fixture,
            "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
