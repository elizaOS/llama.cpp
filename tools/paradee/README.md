# tools/paradee: Paradee-8M TTS (v1.0, v1.1 and v1.2)

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
- `src/paradee-head.cpp` runs Paradee-8M v1.1's source-filter head. See [v1.1](#v11-source-filter-head).
- Paradee-8M v1.2 is v1.1 with four voices (a style table, one row per voice). See [v1.2](#v12-four-voices).

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
  `paradee_audio`). For v1.1, `paradee_synth_options::noise_seed` picks the head's noise.
- `paradee_get_head` and `paradee_version` say which release a GGUF holds.
- Diagnostics:
  - `paradee_synthesize_ids_staged` and `paradee_decode_staged` return every stage.
  - `paradee_phase_lock` runs the filter alone.
  - `paradee_head_staged` runs the v1.1 head alone; `paradee_stages::noise_in` replaces its noise.

## Build and use

`LLAMA_BUILD_PARADEE` is ON by default. It needs `LLAMA_BUILD_KOKORO`. It builds `paradee_lib`, `paradee-tts`
and `paradee-stage-dump`.

```sh
python tools/paradee/convert_paradee_pt_to_gguf.py --model-dir <HF snapshot at the pinned revision> \
    --output paradee-8m-v1.0-f32.gguf --check-output
python tools/paradee/convert_paradee_pt_to_gguf.py --version 1.1 --model-dir <v1.1 snapshot> \
    --output paradee-8m-v1.1-f32.gguf --check-output
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
- **v1.1:** the `"v1.1"` section pins the same things for sahilmahendrakar/Paradee-8M-v1.1 at `330617ad`
  (the `v1.1` tag) and github.com/sahilmahendrakar/paradee at `606e16e1` (`vhead.py` and
  `export_paradee_v1_1.py` as well as the v1.0 scripts, which are unchanged there). The reference
  environment is the same: `training/uv.lock` did not change.
- **v1.0 GGUF sha256:** writing `paradee.version` and `paradee.decoder.head` changed the converter, and so the
  v1.0 output's sha256 (the tensors are unchanged). Files from the earlier converter, such as `357477…`
  (32,072,768 bytes), still load as v1.0 and give byte-identical audio.
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
| `test-paradee-parity` | `PARADEE_GGUF`, `PARADEE_REF_DIR` | integer durations identical; bert/d/dur/F0/N/asr within 1e-3 × RMS; source (from the reference F0) and decoder output (from the reference asr/F0/N/source) within 1e-3 × RMS; lock alone within 1e-4 of float64 `phase_lock.lock()`; final waveform SNR ≥ 35 dB. For v1.1, see [v1.1 parity](#v11-parity) |

The model-backed tests exit 77 (CTest SKIP) when the model or the reference is not configured.

Each acoustic stage in the parity test gets the reference's inputs. The harmonic source integrates F0
over the whole utterance, so F0's float32 rounding (about 1e-6 relative) would otherwise show up as a
source phase error and be charged to the wrong stage. The end-to-end SNR covers the whole chain.

The shipped float32 `phase_lock.lock()` differs from its own float64 evaluation by up to about 1e-3, at
bins where the source is nearly silent. No independent implementation can match it within 1e-4. The gate
therefore uses the float64 evaluation and reports the float32 distance.

## v1.1: source-filter head

[sahilmahendrakar/Paradee-8M-v1.1](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.1) (`330617ad`,
Apache-2.0) keeps v1.0's text side (the same `text_side.pt`) and the decoder's AdaIN trunk (same layout,
retrained weights). It replaces the iSTFTNet generator and the phase-locking filter with `SourceFilterHead2`
from `training/scripts/vhead.py` (github.com/sahilmahendrakar/paradee `606e16e1`). `src/paradee-head.cpp`
implements that head:

| Part | What it does |
|---|---|
| ConvNeXt stack | Conv1d 128→160 (k 7), LayerNorm, 4 × (depthwise conv 7, LayerNorm, Linear 160→480, GELU, Linear 480→160, gamma, residual), LayerNorm. 80 fps, on the kokoro layers. |
| Output layers | Per STFT frame (n_fft 600, hop 150, 2 frames per trunk frame): harmonic log-gain, phase (cos, sin), noise log-gain and 4 pulsing-noise gains at 80 mel-spaced points, linearly interpolated to the 301 bins; one pulse shift (0.5 tanh, in pitch periods). Noise-burst gains at 24 points for a 150/75 STFT. |
| Sources | A 64-harmonic sine bank from F0 (harmonics above 11 kHz and unvoiced frames muted), white noise, and noise × voicing × cos/sin(2πm·phase) for m = 1, 2. |
| Output | iSTFT(E · gain · phase · e^{i2π(f/F0)·shift} + (noise + Σ pulsing noise · gain) · noise gain) + iSTFT_150/75(STFT_150/75(reversed noise) · burst gain). |

The STFTs are mixed-radix FFTs (600 = 4·2·3·5·5, 150 = 2·3·5·5). The main path runs one STFT frame at a time,
so no spectrogram is kept.

- **Routing.** The converter writes `paradee.version` and `paradee.decoder.head` (`istftnet_lock` for v1.0,
  `source_filter_v2` for v1.1). `paradee_load_model` picks the graph from that key. A file without it is v1.0,
  so v1.0 GGUFs written before this change still load and give byte-identical audio (checked with the
  `357477…` file). The v1.1 file has no generator or lock tensors and no `paradee.lock.*` keys.
  `paradee_get_lock_params` returns null for it.
- **Noise.** The head draws Gaussian noise. Natively, that is xoshiro256** seeded by splitmix64, with
  Box-Muller in double. The seed is `PARADEE_DEFAULT_NOISE_SEED` (0) unless `paradee_synth_options::noise_seed`
  says otherwise, so the same input gives the same audio. `paradee_stages::noise_in` replaces the noise, and
  the parity test uses that to feed in the reference's noise. libelizainference needs no change: it calls
  `paradee_synthesize` with the default options.
- **Rounding.** The native head follows the module's float32 arithmetic, with three deliberate differences:
  - The harmonics come from a rotation recurrence on exp(i2π·phase), computed in double, instead of one
    float32 `sin(2πk·phase)` per harmonic. That float32 argument reaches about 440 rad, where one float32 step
    is about 3e-5 rad.
  - The pulse shift's phase steps across the bins in double.
  - The running phase `c + fr·d` is rounded twice, exactly as torch rounds it, and never fused into one FMA.

### v1.1 parity

`dump_paradee_reference.py dump --version 1.1` runs the upstream module (`VDecoder(head="v2")` with
`load_ckpt`, as `export_paradee_v1_1.py` builds it) in float32, with no other changes. The noise is
`torch.randn` from `Generator().manual_seed(--noise-seed + index)`, or `--noise-dir/<i>.noise.f32`, and it is
written out as the `noise` stage.

```sh
python tools/paradee/tools/dump_paradee_reference.py dump --version 1.1 --model-dir <v1.1 snapshot> \
    --paradee-src <paradee checkout at 606e16e1> --out ref11
PARADEE_GGUF=paradee-8m-v1.1-f32.gguf PARADEE_REF_DIR=ref11 PARADEE_GGUF_V10=paradee-8m-v1.0-f32.gguf \
    build/bin/test-paradee-parity
```

| Gate | Input | Limit | Measured (5 sentences, M4 Max and TB336FU) |
|---|---|---|---|
| ids, integer durations | – | identical | identical |
| text side unchanged | `PARADEE_GGUF_V10` | bert/d/dur/dur_int/F0/N/asr bit-identical between the v1.0 and v1.1 GGUFs | bit-identical |
| bert, d, dur, F0, N, asr | – | max\|err\| ≤ 1e-3 × RMS | ≤ 6e-6 × RMS |
| trunk | reference asr/F0/N | ≤ 1e-3 × RMS | ≤ 9e-6 × RMS |
| hn (ConvNeXt stack) | reference trunk | ≤ 1e-3 × RMS | ≤ 1e-5 × RMS |
| sine bank | reference F0 | ≤ 1e-4 × RMS of the float64 bank; no further from float32 `sine_bank` than float32 is from float64 (+1e-4) | 1.3–1.4e-5; float32 itself is 2.6–3.1e-4 away |
| main path, burst path, head output | reference trunk, F0, noise | ≤ 1e-3 × RMS | ≤ 9.3e-5 × RMS |
| end to end | reference noise | SNR ≥ 35 dB | 68.5–81.9 dB (Mac), 69.0–77.3 dB (tablet) |

The end-to-end figure is set by the text side's float32 rounding, as in v1.0. F0 differs by about 1e-6 relative,
and the harmonic phase integrates that error.

### v1.1 against v1.0

Measured October 19, 2026, with the 40 sentences of the F16 and v1.1 evaluation in elizaOS/llama.cpp#85: 20
assistant prompts and 20 Harvard sentences (130.1 s of speech), phonemized with `tools/misaki-g2p`. Both releases give the same durations, because the text side is the same.
Both GGUFs are F32 and read into memory, as on this branch. RTF is the sum of the best of 3 runs per sentence
from `paradee-tts`, divided by the audio length.

| | v1.0 | v1.1 |
|---|---|---|
| GGUF (F32) | 32,072,864 B (deflate -9: 29.7 MB) | 31,275,232 B (deflate -9: 29.0 MB) |
| RTF, M4 Max, 1 thread | 0.025 | 0.0083 |
| RTF, M4 Max, default threads | 0.025 | 0.0081 |
| RTF, TB336FU (MT6835), 8 threads | 0.263 | 0.091 |
| RTF, TB336FU, 1 thread on an A76 core | 0.533 | 0.105 |
| WER, Phonon-2 / whisper.cpp `base.en` | 3.90% / 3.90% | 3.90% / 3.12% |
| Peak RssAnon, TB336FU, 19.8 s utterance | 202 MB | 69–73 MB |
| Peak footprint, M4 Max, 19.8 s utterance | 300 MB | 110–115 MB |

- The WER figures equal upstream's ONNX graphs on the same sentences (v1.1 fp32: 3.90% / 3.12%). Phonon-2's
  transcripts of the native v1.1 audio are identical to its transcripts of the ONNX v1.1 audio for all 40
  sentences. That holds even though the two draw different noise.
- On the tablet, v1.1 is 2.9× faster with 8 threads and 5× faster on one core. The head runs on one thread,
  so most of the 8-thread gain is the text side's.
- The Mac was busy with other work (load average ~50) and the tablet had load ~11–14 from its own background
  work. Treat the absolute RTFs as approximate; the two releases were measured back to back.
- Peak memory is lower mainly because the head never builds the generator's 24 kHz feature maps. With
  #85's memory mapping, both files would also stop counting their 31–32 MB of weights as private memory.

## v1.2: four voices

[sahilmahendrakar/Paradee-8M-v1.2](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.2) at `ed2934d4`
(Apache-2.0; pins.json `"v1.2"`) is v1.1 retrained with four voices: `af_heart`, `bf_isabella`, `am_puck` and
`bm_fable`, in that order (config.json `voices`). Every tensor has v1.1's name and shape except the two learned
styles, which become tables: text `(4, 32)` and decoder `(4, 16)`. Both checkpoints carry a top-level `voices`
list next to `model`. The training scripts the reference imports are byte-identical to v1.1's, and the weights
are retrained, so `af_heart` sounds different from v1.1.

- `convert_paradee_pt_to_gguf.py --version 1.2` writes the styles as 2-D tensors `paradee.style.text [4, 32]` and
  `paradee.style.decoder [4, 16]`, plus `paradee.voices` (names in model order), `paradee.voices.accent`
  (`en-us` for `a*_` voices, `en-gb` for `b*_` voices: the misaki dialect each was trained on) and
  `paradee.voice = af_heart` (the default). v1.0 and v1.1 files keep their 1-D styles.
- The runtime loads a 1-D style as one voice and a 2-D style as a table. `paradee_voice_count`,
  `paradee_voice_name`, `paradee_voice_accent`, `paradee_find_voice` and `paradee_default_voice` describe the
  voices; `paradee_synth_options::voice` (and `paradee_stages::voice` for the staged calls) selects one, -1 is
  the default. `paradee-tts --voice NAME` (`--voice list` prints them).
- The British voices need misaki's British lexicon (`en.G2P(british=True)`), which `tools/misaki-g2p` does not
  ship.
- `dump_paradee_reference.py dump --version 1.2 --voice NAME` runs v1.1's modules with row `NAME` of each
  table loaded into their `(1, dim)` style parameter.

Parity (`test-paradee-parity`, M4 Max, the 5 fixture sentences, the reference's noise; torch 2.14.1 in the
reference env, pins say 2.14.0): every stage gate passes, end to end 66.2–80.1 dB for `af_heart` and
72.2–77.3 dB for `am_puck`. `test-paradee-api` checks the table (names, accents, row selection, out-of-range
rows) on the v1.2 file.
