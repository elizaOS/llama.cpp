#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
bench_parakeet.py - run parakeet-bench and score it: RTFx (wall and CPU time),
first-window partial latency, peak RSS, WER against the references and the
share of transcripts identical to reference_transformers.py.

  bench_parakeet.py --bench build/bin/parakeet-bench --model phonon-2-fv5.gguf \
      --rows rows.json [--rows cmd_rows.json] [--reference ref/manifest.json] \
      [--threads 1 4] [--stream] [--out summary.json] [--gate-wer 5.0]

rows.json: [{id, path, ref, split}] (the same rows dump_parakeet_reference.py
reads). WER uses whisper_normalizer's EnglishTextNormalizer when installed
(the Open ASR Leaderboard normaliser), otherwise lower-case + strip
punctuation. CPU-time RTFx is the robust number on a shared host; wall-clock
RTFx is only meaningful on an idle machine. Host numbers are not device
evidence.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import subprocess
import sys
import tempfile
from pathlib import Path


def normaliser():
    try:
        from whisper_normalizer.english import EnglishTextNormalizer  # type: ignore
        n = EnglishTextNormalizer()
        return n, "whisper_normalizer.EnglishTextNormalizer"
    except ImportError:
        def n(s: str) -> str:
            s = s.lower().replace("-", " ")
            s = re.sub(r"[^a-z0-9' ]+", " ", s)
            return " ".join(s.split())
        return n, "lowercase+strip-punctuation"


def wer(refs: list[str], hyps: list[str]) -> float:
    errs = words = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        d = list(range(len(hw) + 1))
        for i in range(1, len(rw) + 1):
            prev, d[0] = d[0], i
            for j in range(1, len(hw) + 1):
                cur = d[j]
                d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (rw[i - 1] != hw[j - 1]))
                prev = cur
        errs += d[len(hw)]
        words += len(rw)
    return 100.0 * errs / max(1, words)


def run(bench: str, model: str, rows: list[dict], threads: int, stream: bool, extra: list[str]) -> list[dict]:
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as fh:
        for r in rows:
            fh.write(f"{r['id']}\t{r['path']}\n")
        lst = fh.name
    cmd = [bench, "-m", model, "-t", str(threads), "--list", lst] + (["--stream"] if stream else []) + extra
    out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, text=True).stdout
    return [json.loads(l) for l in out.splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", action="append", required=True)
    ap.add_argument("--reference", action="append", default=[], help="manifest.json from dump_parakeet_reference.py")
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--extra", default="", help="extra parakeet-bench arguments")
    ap.add_argument("--out")
    ap.add_argument("--gate-wer", type=float, default=None, help="fail when the overall WER exceeds this")
    ap.add_argument("--gate-ref-identical", type=float, default=None,
                    help="fail when fewer than this fraction of transcripts equal the reference")
    a = ap.parse_args()
    norm, norm_name = normaliser()
    rows = []
    for p in a.rows:
        rows += json.loads(Path(p).read_text())
    ref_text = {}
    for p in a.reference:
        for r in json.loads(Path(p).read_text())["rows"]:
            ref_text[r["id"]] = r["text"]
    summary = {"normaliser": norm_name, "n": len(rows), "host_note": "host CPU numbers; not device evidence", "runs": {}}
    ok = True
    for th in a.threads:
        res = run(a.bench, a.model, rows, th, a.stream, a.extra.split())
        tail = res[-1]
        per = {r["id"]: r for r in res if not r.get("summary")}
        byid = {r["id"]: r for r in rows}
        splits = sorted({r.get("split", "all") for r in rows})
        s = {k: tail[k] for k in ("kernel", "threads", "audio_s", "wall_s", "cpu_s", "rtfx_wall", "rtfx_cpu",
                                  "peak_rss_mb", "load_ms", "loadavg")}
        refs = [norm(byid[i]["ref"]) for i in per]
        hyps = [norm(per[i]["text"]) for i in per]
        s["wer"] = round(wer(refs, hyps), 2)
        for sp in splits:
            ids = [i for i in per if byid[i].get("split", "all") == sp]
            s[f"wer_{sp}"] = round(wer([norm(byid[i]["ref"]) for i in ids], [norm(per[i]["text"]) for i in ids]), 2)
        s["prefix_cpu_ms_median"] = round(st.median(r["prefix_cpu_ms"] for r in per.values()), 1)
        s["prefix_wall_ms_median"] = round(st.median(r["prefix_wall_ms"] for r in per.values()), 1)
        s["utt_cpu_ms_median"] = round(st.median(r["cpu_ms"] for r in per.values()), 1)
        if ref_text:
            ids = [i for i in per if i in ref_text]
            same = sum(per[i]["text"] == ref_text[i] for i in ids)
            s["identical_to_reference"] = f"{same}/{len(ids)}"
            s["reference_wer"] = round(wer([norm(byid[i]["ref"]) for i in ids], [norm(ref_text[i]) for i in ids]), 2)
            if a.gate_ref_identical is not None and same < a.gate_ref_identical * len(ids):
                ok = False
        if a.stream:
            s["stream_wer"] = round(wer(refs, [norm(per[i]["stream_text"]) for i in per]), 2)
            same = sum(per[i]["stream_text"] == per[i]["text"] for i in per)
            s["stream_equals_batch"] = f"{same}/{len(per)}"
            s["stream_first_partial_ms_median"] = round(
                st.median(r["stream_first_partial_ms"] for r in per.values() if r["stream_first_partial_ms"] >= 0), 1)
            s["stream_cpu_rtfx"] = round(sum(r["audio_s"] for r in per.values()) * 1e3 /
                                         sum(r["stream_cpu_ms"] for r in per.values()), 2)
            s["stream_finish_ms_median"] = round(st.median(r["stream_finish_ms"] for r in per.values()), 1)
        if a.gate_wer is not None and s["wer"] > a.gate_wer:
            ok = False
        summary["runs"][f"t{th}"] = s
        summary.setdefault("rows", {})[f"t{th}"] = res
        print(json.dumps(s), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(summary, indent=1))
    if not ok:
        print("GATE FAILED", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
