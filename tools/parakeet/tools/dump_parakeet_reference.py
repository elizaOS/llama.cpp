#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
dump_parakeet_reference.py - write the reference outputs the tools/parakeet
parity tests compare against.

The reference is Fermion's own `reference_transformers.py` (dense fp32
`transformers.ParakeetForTDT`, every container record expanded exactly) from
the pinned FermionResearch/Phonon-2 snapshot, with the processor of the base
repo nvidia/parakeet-tdt-0.6b-v3. Both scripts are checked against
tools/parakeet/pins.json before they are imported.

For every utterance it writes, into --out-dir:
  <id>.mel.f32     normalised log-mel features [n_frames, 128] (valid frames only)
  <id>.sub.f32     subsampling output [T, 1024]
  <id>.l0.f32      encoder layer 0 output [T, 1024]
  <id>.enc.f32     encoder output [T, 1024]
  <id>.pcm.f32     the 16 kHz mono input
and manifest.json (shapes, token ids without blanks, text) plus the same rows
as manifest.tsv for the C++ test:
  <id> TAB <n_samples> TAB <n_mel> TAB <T> TAB <ids, comma separated> TAB <text>

usage:
  dump_parakeet_reference.py --snapshot <Phonon-2 snapshot dir> --base-dir <parakeet-tdt-0.6b-v3 dir> \
      --container model.fermion --utts rows.json [--n 8] [--text-only] --out-dir fixtures/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
PINS = json.loads((_HERE.parent / "pins.json").read_text())


def check(path: Path, pin: dict):
    data = path.read_bytes()
    if len(data) != pin["size"] or hashlib.sha256(data).hexdigest() != pin["sha256"]:
        sys.exit(f"{path}: does not match pins.json")


def load_pcm(path: str) -> np.ndarray:
    import soundfile as sf
    w, sr = sf.read(path, dtype="float32", always_2d=True)
    w = w.mean(axis=1) if w.shape[1] > 1 else w[:, 0]
    if sr != 16000:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=16000).astype(np.float32)
    return np.ascontiguousarray(w, dtype=np.float32)


def write_tsv(path: Path, rows: list[dict]):
    with open(path, "w") as fh:
        for r in rows:
            text = r["text"].replace("\t", " ").replace("\n", " ")
            fh.write(f"{r['id']}\t{r['n_samples']}\t{r['n_mel']}\t{r['T']}\t"
                     f"{','.join(str(i) for i in r['token_ids'])}\t{text}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--container", type=Path, required=True)
    ap.add_argument("--utts", type=Path, required=True, help="JSON list of {id, path, ref}")
    ap.add_argument("--n", type=int, default=0, help="first N utterances (0 = all)")
    ap.add_argument("--text-only", action="store_true", help="write transcripts only, no tensors")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    for f in ("reference_transformers.py", "fermion_container.py"):
        check(a.snapshot / f, PINS["files"][f])
    check(a.container, PINS["archive_members"]["model.fermion"])
    sys.path.insert(0, str(a.snapshot))
    import torch
    import reference_transformers as rt  # type: ignore  # noqa: E402  (from the pinned snapshot)

    torch.set_num_threads(4)
    model, processor, info = rt.load_model(str(a.container), a.base_dir)
    utts = json.loads(a.utts.read_text())
    if a.n:
        utts = utts[: a.n]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    caps = {}
    hooks = [model.encoder.subsampling.register_forward_hook(lambda m, i, o: caps.__setitem__("sub", o)),
             model.encoder.layers[0].register_forward_hook(lambda m, i, o: caps.__setitem__("l0", o))]
    rows = []
    for u in utts:
        pcm = load_pcm(u["path"])
        inp = processor([pcm], sampling_rate=16000, return_tensors="pt", padding=True)
        feats = inp["input_features"]
        mask = inp["attention_mask"]
        n_valid = int(mask.sum())
        with torch.no_grad():
            enc = model.encoder(input_features=feats, attention_mask=mask)
            out = model.generate(input_features=feats, attention_mask=mask)
        seq = getattr(out, "sequences", out)[0].tolist()
        text = processor.batch_decode(getattr(out, "sequences", out), skip_special_tokens=True)[0].strip()
        blank = model.config.blank_token_id
        ids = [int(x) for x in seq if int(x) != blank][0:]
        # the generate output starts with decoder_start_token_id (= blank) and keeps blank steps
        T = int(model._get_subsampling_output_length(torch.tensor([n_valid])).item())
        row = {"id": u["id"], "ref": u.get("ref"), "text": text, "token_ids": ids,
               "n_samples": int(pcm.size), "n_mel": n_valid, "T": T, "split": u.get("split")}
        if not a.text_only:
            pcm.tofile(a.out_dir / f"{u['id']}.pcm.f32")
            feats[0, :n_valid].float().numpy().astype(np.float32).tofile(a.out_dir / f"{u['id']}.mel.f32")
            caps["sub"][0, :T].float().numpy().astype(np.float32).tofile(a.out_dir / f"{u['id']}.sub.f32")
            l0 = caps["l0"][0] if isinstance(caps["l0"], torch.Tensor) else caps["l0"][0][0]
            l0[:T].float().numpy().astype(np.float32).tofile(a.out_dir / f"{u['id']}.l0.f32")
            enc.last_hidden_state[0, :T].float().numpy().astype(np.float32).tofile(a.out_dir / f"{u['id']}.enc.f32")
        rows.append(row)
        print(f"{u['id']:>20} T={T:4d} {text[:90]}", flush=True)
    for h in hooks:
        h.remove()
    write_tsv(a.out_dir / "manifest.tsv", rows)
    (a.out_dir / "manifest.json").write_text(json.dumps(
        {"reference": "reference_transformers.py (dense fp32 ParakeetForTDT)", "load": {
            k: v for k, v in info.items() if k in ("container_records", "params")},
         "torch": torch.__version__, "rows": rows}, indent=1))


if __name__ == "__main__":
    main()
