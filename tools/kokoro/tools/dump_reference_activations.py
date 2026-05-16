#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dump reference activations from the upstream `kokoro` Python package for
parity-testing the C++ forward.

Each boundary in the StyleTTS-2/iSTFTNet pipeline is serialized to a binary
fp32 file the C++ test harness can read. The C++ predictor + decoder
implementations must reproduce these activations within a tight ulp tolerance.

Usage:

    python3 dump_reference_activations.py \\
        --voice voices/af_bella.bin \\
        --text "Hello world." \\
        --output-dir /tmp/kokoro-ref/

Outputs:
    input_ids.bin           int32 [T+2]
    ref_s.bin               fp32  [1, 256]
    bert_out.bin            fp32  [T+2, 768]
    d_en.bin                fp32  [T+2, 512]
    duration.bin            fp32  [T+2]
    pred_dur.bin            int32 [T+2]
    en.bin                  fp32  [T_frame, 512] (after length-regulate)
    F0_pred.bin             fp32  [T_frame]
    N_pred.bin              fp32  [T_frame]
    t_en.bin                fp32  [T+2, 512]
    asr.bin                 fp32  [T_frame, 512]
    audio.bin               fp32  [num_samples]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch


def dump(t: torch.Tensor, path: str) -> None:
    arr = t.detach().cpu().numpy().astype(np.float32 if t.dtype.is_floating_point else np.int32)
    arr.tofile(path)
    print(f"  wrote {path}: shape={tuple(arr.shape)} dtype={arr.dtype}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--voice", required=True, help="voice .bin (256 fp32 × N positions)")
    p.add_argument("--text",  required=True, help="text to phonemize and synthesize")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--lang", default="a", help="kokoro KPipeline lang_code (default 'a')")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load voice (raw fp32 (N,1,256)).
    voice = np.fromfile(args.voice, dtype=np.float32)
    assert voice.size % 256 == 0, f"voice size {voice.size} not multiple of 256"
    voice = voice.reshape(-1, 1, 256)

    from kokoro import KPipeline
    from kokoro.model import KModel
    model = KModel(repo_id="hexgrad/Kokoro-82M").eval()

    pipeline = KPipeline(lang_code=args.lang, model=False)
    # pipeline.g2p returns (phoneme_string, mtokens). Pull the string.
    ph_str, _ = pipeline.g2p(args.text)
    input_ids = list(filter(lambda i: i is not None,
                            map(lambda c: model.vocab.get(c), ph_str)))
    print(f"phonemes ({len(ph_str)}): {ph_str!r}")
    print(f"input_ids ({len(input_ids)}): {input_ids}")

    # Add BOS+EOS (zeros) per KModel.forward.
    input_ids_t = torch.LongTensor([[0, *input_ids, 0]])
    T = input_ids_t.shape[-1]
    # Pick the per-position style row that kokoro-onnx uses.
    slot = min(voice.shape[0] - 1, max(0, len(input_ids)))
    ref_s = torch.from_numpy(voice[slot]).float()  # [1, 256]
    assert ref_s.shape == (1, 256)

    dump(input_ids_t.squeeze(), os.path.join(args.output_dir, "input_ids.bin"))
    dump(ref_s, os.path.join(args.output_dir, "ref_s.bin"))

    # Now do the same forward as KModel.forward_with_tokens, dumping at each
    # boundary. We can patch via hooks or replicate inline.
    with torch.no_grad():
        input_lengths = torch.full((input_ids_t.shape[0],),
                                   input_ids_t.shape[-1],
                                   dtype=torch.long)
        text_mask = torch.arange(int(input_lengths.max())).unsqueeze(0).expand(
            input_ids_t.shape[0], -1).type_as(input_lengths)
        text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

        bert_dur = model.bert(input_ids_t, attention_mask=(~text_mask).int())
        dump(bert_dur.squeeze(0), os.path.join(args.output_dir, "bert_out.bin"))

        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)  # [B, 512, T]
        dump(d_en.squeeze(0).transpose(0, 1), os.path.join(args.output_dir, "d_en.bin"))

        s = ref_s[:, 128:]  # predictor style half: [1, 128]
        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        # `d` is [B, T, d_hid+sty_dim] = [1, T, 512+128=640].
        dump(d.squeeze(0), os.path.join(args.output_dir, "duration_encoder_out.bin"))

        x_lstm, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x_lstm)  # [1, T, max_dur=50]
        duration_sig = torch.sigmoid(duration).sum(dim=-1) / 1.0  # speed=1
        dump(duration_sig.squeeze(0), os.path.join(args.output_dir, "duration.bin"))

        pred_dur = torch.round(duration_sig).clamp(min=1).long().squeeze()
        dump(pred_dur, os.path.join(args.output_dir, "pred_dur.bin"))

        # Length-regulate
        indices = torch.repeat_interleave(torch.arange(input_ids_t.shape[1]), pred_dur)
        pred_aln_trg = torch.zeros((input_ids_t.shape[1], indices.shape[0]))
        pred_aln_trg[indices, torch.arange(indices.shape[0])] = 1
        pred_aln_trg = pred_aln_trg.unsqueeze(0)
        en = d.transpose(-1, -2) @ pred_aln_trg  # [B, 512, T_frame]
        dump(en.squeeze(0).transpose(0, 1), os.path.join(args.output_dir, "en.bin"))

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
        dump(F0_pred.squeeze(0), os.path.join(args.output_dir, "F0_pred.bin"))
        dump(N_pred.squeeze(0), os.path.join(args.output_dir, "N_pred.bin"))

        t_en = model.text_encoder(input_ids_t, input_lengths, text_mask)  # [B, 512, T]
        dump(t_en.squeeze(0).transpose(0, 1), os.path.join(args.output_dir, "t_en.bin"))
        asr = t_en @ pred_aln_trg  # [B, 512, T_frame]
        dump(asr.squeeze(0).transpose(0, 1), os.path.join(args.output_dir, "asr.bin"))

        audio = model.decoder(asr, F0_pred, N_pred, ref_s[:, :128]).squeeze()
        dump(audio, os.path.join(args.output_dir, "audio.bin"))

    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
