# tools/parakeet: Parakeet-TDT ASR for Phonon-2

[FermionResearch/Phonon-2](https://huggingface.co/FermionResearch/Phonon-2) is
[nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) (FastConformer encoder +
TDT decoder, 627M parameters, English) with every encoder linear retrained to a five-value alphabet
`{-hi, -lo, 0, +lo, +hi}` per row and everything else stored as 6-bit integer tables. This directory runs it
natively on the CPU inside libelizainference. It has no ONNX, MLX, Core ML or Python dependency at run time.

**Licence (flagged, not a blocker).** The weights are CC-BY-4.0, derived from NVIDIA parakeet-tdt-0.6b-v3
(CC-BY-4.0). The converter copies Fermion's `NOTICE` into the GGUF (`parakeet.notice`), and
`pk_model_notice()` returns it. Apps that ship the model must show the attribution. According to the NOTICE,
the training audio includes CHiME-6 (CC-BY-SA-4.0) and SPGISpeech (Kensho terms). Fermion's
`fermion_container.py` and `reference_transformers.py` are Apache-2.0. Fermion's fast CPU kernels ship only as
closed binaries in the `fermion-research` wheel, so none of their code is used here. Everything in this
directory is new code.

## Pipeline

| Stage | What runs |
|---|---|
| front end | pre-emphasis 0.97, 512-point STFT (400-sample periodic=False Hann, hop 160, centre zero-pad), slaney 128-mel, `log(x + 2^-24)`, per-feature normalisation over the utterance |
| subsampling | conv 3x3/2 (1→256) + ReLU, then [depthwise 3x3/2, pointwise 256→256 (int8), ReLU] x2, then a 4096→1024 linear (int8). 8x in time, 80 ms per frame. Computed in 1.28 s chunks, so memory stays flat |
| 24 conformer blocks | FF½ → rel-pos MHSA (Transformer-XL, 8 heads, full context) → conv module (pw1, GLU, depthwise k=9 with BatchNorm folded in, SiLU, pw2) → FF½ → LayerNorm. All 264 linears are five-value int8 GEMMs |
| TDT decoder | 2-layer LSTM prediction net (640), joint `ReLU(enc_proj + dec_proj)` → 8198 logits (8193 token/blank + 5 durations {0..4}), greedy, max 10 symbols per frame |

## Numerics

- **Five-value weights in int8.** `wmat_from_fv5` puts each row's `{lo, hi}` on an int8 grid `{L, H}` with
  `H <= 127`. It picks the `H` in 64..127 that best keeps `lo/hi`, so the only weight error is
  `|L*s - lo| <= 0.4% of hi` (reported as `fv5_grid_err`). `hi` is exact.
- **Activations.** Activations entering a linear are quantised to int8 per token (symmetric,
  round-half-even). The products are exact int32, and the per-row and per-token scales are applied in fp32.
  Fermion's own CPU engine takes the same approach.
- **Everything else is fp32:** attention, softmax, LayerNorm, convolutions, LSTM state and the joint
  epilogue.
- **Position keys.** `W_pos · e(p)` depends only on the relative position `p`. It is computed once per layer,
  stored as fp16 and grown on demand: 12 MB at 10 s windows, 37 MB at 30 s.
- **Decoder.** The LSTM, projection and joint-head matvecs use the exact int6 tables as int8 weights, with
  int8 activations.

## Kernels (`src/pk-kernels*.cpp`)

The weights stay in the GGUF in kernel order and are used in place from the mmapped file. Per 8-row block,
the codes are decoded into an int8 panel that stays in L1, and the panel is then multiplied against all
tokens. Two code layouts decode to the same panel, so they give bit-identical results:

| layout | bits per weight | Phonon-2 GGUF | decode per 16 weights |
|---|---|---|---|
| `fv5-b5x3t7-v1` (default) | 2.35–2.41 | 208.9 MB | 1/3 of: 2 `UMULL` + `UZP2` (x / 5), `TBL` (/ 25), 2 `MLS`, then 3 x (`ORR` + `TBL`); 1/8 of a 7-`SRI` bit gather |
| `fv5-pair16-v1` | 4 | 330.6 MB | `AND` or `USHR`, one `vqtbl1q_s8` |

- **`fv5-pair16-v1`**: 4-bit level codes, two rows by 16 k per 16-byte block.
- **`fv5-b5x3t7-v1`**: the panel's 16-byte vectors are taken three at a time. Lane j of a code vector is the
  base-5 number of the three levels, which is below 125, so 8 code vectors share 7 stored vectors (the eighth
  code is spread over their top bits). `pk-kernels.h` specifies it.
- The five-value alphabet carries log2(5) = 2.32 bits, and the levels' entropy is 1.95 bits. Fermion's own
  container gets 2.11 bits with a variable-length bit per non-zero weight, but that cannot be decoded at an
  arbitrary block without prefix counts.

What the denser layout costs is decode time. It is fixed per encoder pass, so it only shows on short windows.
These are `tools/bench-fv5-packing.cpp` results, decode plus tile over all 24 layers, on one thread:

| N tokens (80 ms each) | 8 | 16 | 32 | 64 | 125 | 375 |
|---|---|---|---|---|---|---|
| TB336FU Cortex-A76, dotprod: pair16 → b5x3t7 (ms) | 101 → 121 | 176 → 191 | 324 → 330 | 596 → 613 | 1139 → 1142 | 3661 → 3602 |
| M4 Max, i8mm: pair16 → b5x3t7 (ms) | 25 → 33 | 45 → 52 | 85 → 91 | 167 → 171 | 321 → 326 | 957 → 960 |

The converter writes `fv5-b5x3t7-v1` unless given `--layout pair16`. The loader accepts both.


| tier | instruction | runtime check |
|---|---|---|
| `i8mm` | `SMMLA` 2x8·8x2, 8 rows × 8 tokens per tile | `HWCAP2_I8MM` (Linux/Android), `hw.optional.arm.FEAT_I8MM` (Apple) |
| `dotprod` | `SDOT` on straight and half-swapped token pairs | `HWCAP_ASIMDDP`, `FEAT_DotProd` |
| `avx2` | `VPMADDWD` on sign-extended straight and half-swapped token pairs (x86_64) | CPUID AVX2 + `XGETBV` YMM state |
| `scalar` | portable C++ (old ARM, pre-AVX2 x86) | always |

The dotprod and i8mm translation units are compiled with their own `-march`, and the avx2 one with `-mavx2`
(`/arch:AVX2` on MSVC), so the library baseline stays armv8-a / x86-64. The avx2 tier widens int8 to int16
before `VPMADDWD`, so it is exact for every int8 input (no `VPMADDUBSW` saturation). All tiers produce
bit-identical int32 results (`tests/test_parakeet_kernels.cpp`, which also runs random full-range int8
operands). `PK_KERNEL` set to `scalar`, `dotprod`, `i8mm` or `avx2` forces a tier.

On an Apple silicon Mac the x86_64 build can be checked under Rosetta 2, which runs AVX2 but only reports it
when asked: `arch -x86_64 -e ROSETTA_ADVERTISE_AVX=1 build-x64/bin/test-parakeet-kernels` (configure with
`-DCMAKE_OSX_ARCHITECTURES=x86_64`). Rosetta timings are a sanity check, not x86 performance evidence.

Threading uses a persistent pool (`src/pk-pool.cpp`) that spins briefly between jobs and then sleeps. By
default it starts one thread per fast core, at most 4. On Linux/Android those are the cores whose
`cpuinfo_max_freq` is within 20% of the fastest, and workers are pinned to them. On Apple it uses the
performance cores, with QoS user-interactive.

## Streaming (`src/pk-stream.cpp`)

The model attends over the whole input and has no cache-aware mode, so a stream re-decodes a window. A window
must not start in the middle of speech. When its input begins mid-word, the model drops or invents words: on
the same audio slices, the dense fp32 reference does exactly what this port does. Each decode therefore covers
a **pause-delimited segment**, from the segment start to the newest sample.

- **Committing.** A token is committed when two consecutive decodes agree on it and it ends at least
  `right_context_ms` before the newest sample. Commits happen on word boundaries only. Committed text never
  changes.
- **Starting a new segment.** A segment ends when a pause of at least 150 ms lies between the last committed
  token and the first uncommitted one. The next segment starts inside that pause, so windows begin in silence
  and stay short.
- **Bounded windows.** Past `max_window_ms`, stable tokens are committed without agreement and the segment is
  cut at the quietest point of that gap, which bounds every decode.
- **Finished words.** The newest word is committed once the agreed hypothesis has no token in the last
  `right_context_ms`, so it does not wait for the next word.
- **Quiet audio.** With nothing uncommitted and only quiet audio since the last decode (every 10 ms frame below
  -60 dBFS, or 30 dB under the speech level of the last window that held tokens), the decode is skipped and the
  window keeps at most the newest 1.5 s of the pause. A long pause therefore costs no decodes and does not grow
  the next window, and the model is not run on short windows of near-silence, which is where it invents words.
- **No speech at all.** A window past `max_window_ms` with no token keeps its newest half, which bounds noise
  without speech.
- **Decode interval.** It is at least `min_hop_ms` and grows to a sixth of the window.
- **`finish()`** decodes the open segment with full right context.

## Files

- `convert_phonon2_to_gguf.py` reads the pinned `phonon-2.bps.tar.zst`, or an unpacked `model.fermion` and
  `config.json`, and writes the GGUF.
  - It checks every input against `pins.json` (HF revision `7e153e40…`, size and sha256).
  - `--layout b5x3t7` (default) or `--layout pair16` picks the code layout.
  - `--verify` re-reads the GGUF and checks that all 264 five-value records and 11 int tables expand
    bit-identically to the container, that all 723 records are consumed, and that the mel filterbank equals
    librosa's.
- `include/parakeet.h` is the C API: load, transcribe, word times, and stream open/feed/partial/finish.
- `tools/dump_parakeet_reference.py` writes fixtures from Fermion's `reference_transformers.py`, the dense
  fp32 `transformers.ParakeetForTDT`. The fixtures hold the PCM, the normalised mel, the subsampling, layer-0
  and encoder outputs, and the token ids and text.
- `tests/test_parakeet_parity.cpp` compares the native pipeline with those fixtures stage by stage.
- `tools/bench-fv5-packing.cpp` times the two code layouts' decode and decode + tile per GEMM shape, and
  checks that they decode to the same panel.
- `tools/parakeet-bench.cpp` and `tools/bench_parakeet.py` measure RTFx (wall and CPU time), first-window
  partial latency, peak RSS and WER, plus stream WER and stream/batch agreement with `--stream`.
- libelizainference integration lives in `tools/omnivoice/src/eliza-asr-parakeet.inc`.

## Build and use

```sh
python tools/parakeet/convert_phonon2_to_gguf.py --archive phonon-2.bps.tar.zst --output phonon-2-fv5.gguf --verify
cmake -B build -DLLAMA_BUILD_PARAKEET=ON -DPARAKEET_BUILD_TESTS=ON ...
build/bin/parakeet-cli -m phonon-2-fv5.gguf [-t 4] [--stream] [--words] clip.wav
build/bin/test-parakeet-parity phonon-2-fv5.gguf fixtures/
python tools/parakeet/tools/bench_parakeet.py --bench build/bin/parakeet-bench --model phonon-2-fv5.gguf \
    --rows rows.json --threads 1 4 --stream
```

In libelizainference, put the GGUF at `<bundle>/asr/phonon-2-fv5.gguf`. On `mmap_acquire("asr")`, a bundle
whose `asr/` holds a GGUF with `general.architecture = parakeet-tdt` loads this backend.

- `ELIZA_ASR_BACKEND=qwen3` keeps the Qwen3-ASR path; `=parakeet` requires this backend.
- With the backend loaded, `eliza_inference_asr_transcribe` and `_timed` run here. `_timed` returns real TDT
  word times.
- `eliza_inference_asr_stream_supported()` returns 1 and the `asr_stream_*` session runs `pk-stream`. The
  stream does not return token ids, because Parakeet's vocabulary is not the text model's.
- `ELIZA_ASR_THREADS` sets the thread count. `ELIZA_ASR_STREAM_{RIGHT,HOP,MAX_WINDOW}_MS` tune the stream.

## Evidence boundary

The measurements in the PR come from an Apple M4 Max used as an ARM CPU proxy, on a shared, heavily loaded
host. They are not phone evidence. The Android arm64-v8a and x86_64 builds are compile evidence only. Speed,
memory and thermals on a phone need a device run.
