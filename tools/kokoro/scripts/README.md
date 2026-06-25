# Kokoro GGUF converters (for the CrispASR runtime)

These produce the **`bert.*` / `pred.*` / `dec.gen.*`** GGUF tensor layout that
`tools/kokoro/src/kokoro-crispasr.cpp` (the vendored CrispStrobe/CrispASR forward)
requires. The older `../convert_kokoro_pth_to_gguf.py` emits the *mainline*
`kokoro.*` layout, which this runtime cannot load — use these instead.

- `convert-kokoro-to-gguf.py` — hexgrad/Kokoro-82M (or StyleTTS2 `net.*`) `.pth`
  → `bert.*` GGUF (459 tensors).
- `convert-kokoro-voice-to-gguf.py` — a Kokoro voice `.pt` → GGUF voice pack
  (the runtime reads voice packs as GGUF; raw `.bin` voices fail with
  "invalid magic").

Source: **CrispStrobe/CrispASR** (MIT), vendored verbatim except a portability
fix in `convert-kokoro-to-gguf.py` (a `tensor or tensor` fallback that raised
"Boolean value of Tensor is ambiguous" on current torch — replaced with explicit
`is None` checks).

## Usage

```bash
pip install torch gguf numpy huggingface_hub pyyaml   # NOT ultralytics
python convert-kokoro-to-gguf.py --input hexgrad/Kokoro-82M --output kokoro-82m-v1_0.gguf
python convert-kokoro-voice-to-gguf.py --input voices/af_bella.pt --output af_bella.bin

# Verify: stage both into <state>/local-inference/models/kokoro/{*.gguf,voices/*.bin}
# then `bun plugins/plugin-local-inference/scripts/kokoro-real-smoke.ts`
# → "loaded 459 tensors ... synthesized N samples @ 24000Hz". ASR round-trips exactly.
```

See elizaOS/eliza#9588.
