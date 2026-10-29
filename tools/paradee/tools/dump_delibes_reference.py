#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
dump_delibes_reference.py - onnxruntime reference for the Delibes-8M-esES case of
test-paradee-parity (tools/paradee).

Delibes-8M-esES (lapislabsai/Delibes-8M-esES) publishes only ONNX, so its
reference is the release's own fp32 graphs run with onnxruntime exactly as its
inference.py runs them: the text graph on samples/0N.json's token_ids at speed
1.0 (asr [512, F], f0 / noise_condition [2F], durations [T]), then the decoder
graph with white_noise = numpy.random.default_rng(noise_seed).standard_normal(
(1, 600F)). The native head draws its own Gaussian noise, so the test feeds this
noise to it (paradee_stages::noise_in).

Two subcommands:

  fixture  write tests/fixtures/delibes_samples.json: each sample's text,
           phonemes and token ids (Apache-2.0, Lapis Labs; from the pinned
           samples/0N.json). The native G2P must reproduce the phonemes and
           paradee_phonemes_to_ids the ids.

  dump     write <out>/<i>.{ids,dur_int}.i32, <i>.{F0,N,asr,noise,wav}.f32 and
           manifest.json (the layout dump_paradee_reference.py uses; stages the
           ONNX graphs do not expose are absent and the test skips them). It
           also checks the reference against the release's own samples/0N_fp32.wav
           (16-bit PCM of the same run, made with another onnxruntime build and thread count):
           max |diff| <= 4 LSB.

usage:
  dump_delibes_reference.py fixture --model-dir HF_SNAPSHOT [--out tests/fixtures/delibes_samples.json]
  dump_delibes_reference.py dump --model-dir HF_SNAPSHOT --out REF_DIR
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARADEE_DIR = HERE.parent
PINS = json.loads((PARADEE_DIR / "pins.json").read_text(encoding="utf-8"))["delibes"]
DEFAULT_FIXTURE = PARADEE_DIR / "tests" / "fixtures" / "delibes_samples.json"
N_SAMPLES = 5


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def checked(model_dir: Path, rel: str) -> Path:
    path = model_dir / rel
    pin = PINS["files"][rel]
    if path.stat().st_size != pin["size"] or sha256_file(path) != pin["sha256"]:
        raise SystemExit(f"{path}: does not match pins.json delibes.files[{rel!r}]")
    return path


def samples(model_dir: Path) -> list[dict]:
    out = []
    for i in range(1, N_SAMPLES + 1):
        s = json.loads(checked(model_dir, f"samples/0{i}.json").read_text(encoding="utf-8"))
        s["index"] = i
        out.append(s)
    return out


def cmd_fixture(args) -> int:
    vocab = json.loads(checked(args.model_dir, "config.json").read_text(encoding="utf-8"))["vocab"]
    items = []
    for s in samples(args.model_dir):
        ids = [0] + [vocab[c] for c in s["phonemes"]] + [0]
        if ids != s["token_ids"]:
            raise SystemExit(f"sample {s['index']}: phonemes do not map to its token_ids")
        items.append({"index": s["index"], "text": s["text_normalized"], "phonemes": s["phonemes"], "ids": ids,
                      "noise_seed": s["noise_seed"]})
    doc = {
        "source": f"{PINS['model']['repo']}@{PINS['model']['revision']} samples/0N.json (Apache-2.0, Lapis Labs)",
        "g2p": "misaki 0.9.4 espeak.EspeakG2P('es') with eSpeak NG 1.52.0, symbols outside the vocabulary dropped",
        "sentences": items,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} sha256 {sha256_file(args.out)}")
    return 0


def read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1 or w.getframerate() != 24000:
            raise SystemExit(f"{path}: not 24 kHz mono 16-bit")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0


def cmd_dump(args) -> int:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    text = ort.InferenceSession(str(checked(args.model_dir, "onnx/fp32/text.onnx")), opts,
                                providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(checked(args.model_dir, "onnx/fp32/decoder.onnx")), opts,
                               providers=["CPUExecutionProvider"])
    fixture_path = args.fixture
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model": PINS["model"], "family": "delibes", "version": "1.1", "speed": 1.0,
        "fixture": fixture_path.name, "phonemes_fixture_sha256": sha256_file(fixture_path),
        "versions": {"onnxruntime": ort.__version__, "numpy": np.__version__,
                     "onnx": importlib.metadata.version("onnx")},
        "reference": ("onnxruntime fp32 text + decoder graphs (CPU, 1 thread), as inference.py runs them; "
                      "white_noise = numpy default_rng(noise_seed).standard_normal((1, 600F))"),
        "sentences": [],
    }
    for item in fixture["sentences"]:
        i = item["index"]
        ids = np.asarray([item["ids"]], np.int64)
        asr, f0, nc, durations = text.run(None, {"input_ids": ids, "speed": np.asarray([1.0], np.float32)})
        frames = int(durations.sum())
        white = np.random.default_rng(item["noise_seed"]).standard_normal((1, frames * 600)).astype(np.float32)
        wav = dec.run(None, {"asr": asr, "f0": f0, "noise_condition": nc, "white_noise": white})[0]
        if wav.shape != (1, frames * 600) or asr.shape != (1, 512, frames) or f0.shape != (1, 2 * frames):
            raise SystemExit(f"sentence {i}: unexpected graph output shapes")
        published = read_pcm16(checked(args.model_dir, f"samples/0{i}_fp32.wav"))
        lsb = np.abs(np.clip(wav[0].astype(np.float64), -1, 1) - published).max() * 32768 if len(published) == wav.shape[1] else float("inf")
        if lsb > 4.0:
            raise SystemExit(f"sentence {i}: reference differs from samples/0{i}_fp32.wav by {lsb:.1f} LSB")
        out = {"ids": ids[0].astype(np.int32), "dur_int": durations.astype(np.int32).reshape(-1),
               "F0": f0[0], "N": nc[0], "asr": asr[0], "noise": white[0], "wav": wav[0]}
        files = {}
        for name, arr in out.items():
            ext = "i32" if arr.dtype == np.int32 else "f32"
            fn = f"{i}.{name}.{ext}"
            arr.astype("<i4" if ext == "i32" else "<f4").tofile(args.out / fn)
            files[name] = {"file": fn, "shape": list(arr.shape), "sha256": sha256_file(args.out / fn)}
        manifest["sentences"].append({"index": i, "phonemes": item["phonemes"], "T_phon": int(ids.shape[1]),
                                      "T_frame": frames, "published_fp32_wav_max_lsb": round(float(lsb), 3),
                                      "files": files})
        print(f"sentence {i}: T={ids.shape[1]} frames={frames} samples={wav.shape[1]} "
              f"vs published fp32 wav {lsb:.2f} LSB")
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}/manifest.json")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fixture")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_FIXTURE)
    p = sub.add_parser("dump")
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return {"fixture": cmd_fixture, "dump": cmd_dump}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
