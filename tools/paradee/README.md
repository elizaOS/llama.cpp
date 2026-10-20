# tools/paradee: Paradee-8M v1.0 TTS

Paradee-8M v1.0 ([sahilmahendrakar/Paradee-8M-v1.0](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0), Apache-2.0)
is Kokoro-82M's StyleTTS-2 + iSTFTNet code at smaller widths. It is distilled to one voice (`af_heart`), has
8.07M parameters and outputs 24 kHz audio. Its input is misaki phoneme ids with a `0` pad at each end, at most
512 ids, plus a speed value.

Upstream ships the model as an ONNX graph and as PyTorch weights. This directory runs the PyTorch weights
natively, with no ONNX:

- `convert_paradee_pt_to_gguf.py` writes a GGUF (`general.architecture = paradee`) from the pinned
  `pytorch/text_side.pt` and `pytorch/decoder.pt`.
- `paradee_lib` runs it on the `tools/kokoro` layers. Those layers now read their widths from the tensor
  shapes, and Kokoro-82M's output is unchanged bit for bit.
- `src/paradee-phase-lock.cpp` is a native implementation of the model's phase-locking filter.

## Pipeline

| Stage | Kokoro-82M | Paradee-8M |
|---|---|---|
| ALBERT | 768 wide, 12 heads, 12 layers | 256 wide, 4 heads, 6 layers (one shared group) |
| predictor hidden / style | 512 / 128 (a voice row) | 192 / 32 (learned constant `paradee.style.text`) |
| text encoder → decoder `asr` | 512, used directly | 192 → `asr_proj` (Linear 192→512, exact GELU, Linear 512→512) |
| decoder AdaIN width / style | 1024 / 128 | 256 / 16 (learned constant `paradee.style.decoder`) |
| generator channels | 512 → 256 → 128 | 128 → 64 → 32 |
| AdaIN1d InstanceNorm affine | identity (untrained) | trained, applied |
| durations | `std::round` | `torch.round` (half to even) |
| after the generator | none | phase lock: STFT 1024/256, 2–8 kHz, voiced frames (F0 > 60 Hz), 33-frame smoothing |

The harmonic source is deterministic, as in `tools/kokoro`: it has no random initial phase and no additive
noise. Paradee runs it in double precision. Its STFT sets values below the DFT rounding floor to exactly
zero, so the empty bins of an unvoiced, noise-free source have a defined phase.

## API (`include/paradee.h`)

```cpp
namespace eliza_paradee {
  paradee_model_ptr paradee_load_model(const std::string & gguf, std::string & err);
  int paradee_sample_rate(const paradee_model *);
  paradee_status paradee_phonemes_to_ids(const paradee_model *, const std::string & misaki_utf8,
                                         std::vector<int32_t> & ids, std::string & err);
  paradee_status paradee_synthesize_ids(const paradee_model *, const int32_t * ids, size_t n, float speed,
                                        bool (*cancel)(void *), void * user,
                                        std::vector<float> & pcm, std::string & err);
}
```

- `paradee_phonemes_to_ids` adds the pads. It rejects more than 510 symbols (`PARADEE_E_TOO_LONG`), any
  symbol outside the vocabulary (`PARADEE_E_UNKNOWN_SYMBOL`, with the code point named), and invalid UTF-8.
  It never drops a symbol without saying so.
- `paradee_synthesize_ids` returns `PARADEE_CANCELLED`, with `pcm` empty, as soon as `cancel` returns true.
  It checks `cancel` between stages and inside the generator and phase-lock loops. Calls on one model run
  one at a time.
- `paradee_synthesize` takes the same arguments in two structs (`paradee_synth_options` and
  `paradee_audio`).
- Diagnostics:
  - `paradee_synthesize_ids_staged` and `paradee_decode_staged` return every stage.
  - `paradee_phase_lock` runs the filter alone.

## Build and use

`LLAMA_BUILD_PARADEE` is ON by default. It needs `LLAMA_BUILD_KOKORO`. It builds `paradee_lib`, `paradee-tts`
and `paradee-stage-dump`.

```sh
python tools/paradee/convert_paradee_pt_to_gguf.py --model-dir <HF snapshot at the pinned revision> \
    --output paradee-8m-v1.0-f32.gguf --check-output
build/bin/paradee-tts --model paradee-8m-v1.0-f32.gguf --phonemes "həlˈO wˈɜɹld." --output hello.wav
```

The input is misaki's phoneme spelling, not text and not eSpeak IPA. `tools/misaki-g2p` provides the text
front end.

## Pins (`pins.json`)

- **Model:** HF revision `c7f83f86c04e84d0ae94cbe1aad7618e0544f80a`. The `v1.0` tag resolves to commit
  `8f34b01e`, which differs only in `README.md`.
- **Files:** size and sha256 of the `.pt` files, `config.json`, `tokenizer.json`, `samples/sentences.txt` and
  `samples/*_paradee.wav`.
- **Source:** github.com/sahilmahendrakar/paradee at `9c8b4d75`. The sha256 of each training script the
  reference imports is pinned.
- **Reference environment:** torch 2.14.0, kokoro 0.9.4 and misaki 0.9.4, from `training/uv.lock`.
- **GGUF:** the expected sha256 of the F32 converter output. The converter is byte-reproducible.
  `--check-output` enforces this sha256.
- **No F16 pin:** `--outtype f16` halves the file (16 MB) but fails the parity gate (end-to-end SNR
  16–23 dB), so it is not pinned or shipped. See [F16 and v1.1](#f16-and-v11) for the full evaluation.

## Memory

`paradee_load_model` memory-maps an all-F32 GGUF (the converter's default) read-only and the layers read the
weights in place. The 32 MB of weights are then clean, reclaimable page cache rather than private memory.
`PARADEE_USE_MMAP=0` reads the file into private memory instead. Any other file (for example `--outtype f16`)
is read and widened to F32 at load, so it keeps a private F32 copy of every weight as well as the file data.

On a Lenovo TB336FU (4 GB), synthesizing 19.8 s of speech with `paradee-tts` gives byte-identical audio either
way. Peak private memory is 170 MB mapped against 202 MB read, and peak file-backed memory is 34 MB against
3 MB. Activations make up most of the peak.

## F16 and v1.1

Measured October 19, 2026 against the pinned v1.0 GGUF, with this directory's parity test, its reference dump,
and 40 more sentences (20 assistant-style prompts and 20 Harvard sentences). Those sentences were phonemized
with `tools/misaki-g2p` and transcribed with Phonon-2 (`tools/parakeet`) and with whisper.cpp `base.en`.

| File | Size (deflate -9) | Parity test | Durations (40 sentences) | WER Phonon-2 / whisper | Log-mel vs F32 | RTF, TB336FU / M4 Max |
|---|---|---|---|---|---|---|
| F32 (shipped) | 32.1 MB (29.7) | pass, end to end 58–76 dB | – | 3.90% / 3.90% | – | 0.30 / 0.024 |
| `--outtype f16` | 16.2 MB (14.9) | fail: end to end 9.8–23 dB, stage errors up to 1.1e-2 × RMS | 1 of 40 shorter by 25 ms | 3.90% / 3.90% | 0.039 mean, 0.56 max | 0.32 / 0.024 |
| F16 decoder, F32 BERT and predictor (experiment, not in the converter) | 22.1 MB (20.4) | end to end 56–61 dB; fails the 1e-3 × RMS stage gate on `asr` and `dec` only | identical | 3.90% / 3.90% | 0.0026 | 0.30 / 0.024 |

- The end-to-end error of `--outtype f16` comes from the BERT and the duration/F0 predictor. The decoder in F16
  costs little. F0 errors near 1e-3 accumulate in the harmonic source's phase, which the waveform SNR punishes
  while the spectrum barely changes.
- F16 does not speed anything up here, because the runtime computes in F32. It does not save RAM either: an F16
  file cannot be mapped and is widened at load, so private memory goes up by the size of the file (about 16–22 MB).
- So F32 stays the shipped and pinned file. F16 would only make sense with layers that read F16 weights from
  the mapping. The ggml conv1d path already converts its weights to a cached F16 copy (`cached_f16_weight`,
  about 10 MB for the 3-D weights).
- **v1.1** ([sahilmahendrakar/Paradee-8M-v1.1](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.1) at
  `330617ad`, Apache-2.0) is not a weight update this runtime can load. It keeps the text side (same
  `text_side.pt`) and the decoder's AdaIN trunk, but it replaces the iSTFTNet generator and the phase-locking
  filter with a frame-rate source-filter head (`training/scripts/vhead.py` at github.com/sahilmahendrakar/paradee
  `606e16e1`). That head is ConvNeXt blocks at 80 fps that shape the STFT of a 64-harmonic sine bank and of
  random noise, plus a 3 ms noise path. Its output is not deterministic, because it draws random noise.
- Upstream's fp32 ONNX graphs, one onnxruntime thread on the M4 Max, the same 40 sentences:

  | Graph | RTF | WER Phonon-2 / whisper |
  |---|---|---|
  | v1.0 | 0.051 | 4.16% / 3.90% |
  | v1.1 | 0.010 | 3.90% / 3.12% |

  int8 graphs: v1.0 0.054, v1.1 0.011, with WER within one word of fp32.
- Running v1.1 natively needs a new decoder head (sine bank, ConvNeXt, two STFT/iSTFT sizes, a seeded noise
  source) and a new parity gate that does not compare the noise.

## Parity gate

`tools/dump_paradee_reference.py` (run it in the reference environment) has four subcommands:

| Subcommand | What it does |
|---|---|
| `phonemes` | Writes `tests/fixtures/sentences.misaki.json`: misaki 0.9.4 phonemes of `samples/sentences.txt`, from `en.G2P(trf=False, british=False, fallback=EspeakFallback)`. The fixture is committed. |
| `dump` | Runs the pinned PyTorch model on those phonemes and writes every stage: bert, d, dur, integer durations, F0, N, asr, decoder output, source, final waveform, and the float32 lock. |
| `lock-fixture` | Writes `tests/fixtures/phase_lock_synthetic.f32` for the self-contained lock test. |
| `report` | Log-mel distance to the published seeded samples, and WER with whisper.cpp. Report only. |

The reference is deterministic and makes three documented changes, which the native graph also makes:

- `SineGen` has `rand_ini = 0` and no noise.
- `SineGen` runs in float64.
- The source STFT is the canonical float64 STFT described above.

```sh
python tools/paradee/tools/dump_paradee_reference.py dump --model-dir <snapshot> --paradee-src <paradee checkout> --out ref
cmake -B build -DLLAMA_BUILD_TESTS=ON -DPARADEE_GGUF=$PWD/paradee-8m-v1.0-f32.gguf -DPARADEE_REF_DIR=$PWD/ref
ctest --test-dir build -R paradee --output-on-failure
```

| Test | Needs | Gates |
|---|---|---|
| `test-paradee-phase-lock` | nothing (committed fixture) | Lock within 1e-4 of `phase_lock.lock()` evaluated in float64; no further from the shipped float32 `phase_lock.lock()` than float32 is from float64; pass-through, length and cancel behaviour |
| `test-paradee-api` | `PARADEE_GGUF` | ids, vocabulary and length errors, input validation, determinism, speed, cancellation at several points |
| `test-paradee-parity` | `PARADEE_GGUF`, `PARADEE_REF_DIR` | integer durations identical; bert/d/dur/F0/N/asr within 1e-3 × RMS; source (from the reference F0) and decoder output (from the reference asr/F0/N/source) within 1e-3 × RMS; lock alone within 1e-4 of float64 `phase_lock.lock()`; final waveform SNR ≥ 35 dB |

The model-backed tests exit 77 (CTest SKIP) when the model or the reference is not configured.

Each acoustic stage in the parity test gets the reference's inputs. The harmonic source integrates F0
over the whole utterance, so F0's float32 rounding (about 1e-6 relative) would otherwise show up as a
source phase error and be charged to the wrong stage. The end-to-end SNR covers the whole chain.

The shipped float32 `phase_lock.lock()` differs from its own float64 evaluation by up to about 1e-3, at
bins where the source is nearly silent. No independent implementation can match it within 1e-4. The gate
therefore uses the float64 evaluation and reports the float32 distance.
