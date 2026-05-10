# milady-kernels — Metal kernel sources for milady/integration

These five `.metal` files are the Metal-side implementations of the
quantization techniques carried on `milady/integration`. They land on
their own branch (`milady/metal`) so the kernel SOURCES live in the
fork tree (instead of being stamped in by `build-llama-cpp-dflash.mjs`
patches), and so a future agent can wire them into the standard
`ggml-metal` dispatcher with one CMake change instead of N runtime
patches.

## Files

| File | Backs | Status |
|---|---|---|
| `tbq3_0.metal` | `GGML_TYPE_TBQ3_0` (slot 43) — 3-bit TurboQuant V-cache | source landed; needs `ggml-metal.metal` dispatcher entry |
| `tbq4_0.metal` | `GGML_TYPE_TBQ4_0` (slot 44) — 4-bit TurboQuant V-cache | same |
| `tbq3_tcq.metal` | TBQ3 with Trellis-Coded Quantization (decode-only) | same |
| `qjl.metal` | `GGML_TYPE_QJL1_256` (slot 46) — 1-bit JL-transform K-cache | same; pairs with `GGML_OP_ATTN_SCORE_QJL` op |
| `polar.metal` | `GGML_TYPE_Q4_POLAR` (slot 47) — 4-bit PolarQuant weights | same |

## Naming note

Source files are named after the canonical `GGML_TYPE_*` slot they back
(e.g. `tbq3_0.metal` for `GGML_TYPE_TBQ3_0`). The W1-D agent originally
authored `turbo3.metal` and `turbo4.metal` against the spiritbuun
naming convention; renamed here to match the apothic + milady TBQ
scheme used everywhere else in the consumer (`aosp-llama-adapter.ts`
hard-codes `GGML_TYPE_TBQ3_0 = 43`, `GGML_TYPE_TBQ4_0 = 44`).

## Wiring TODO (next agent — Apple Silicon required)

1. Add to `ggml/src/ggml-metal/CMakeLists.txt` so these sources are
   compiled into `default.metallib`.
2. Add dispatcher entries in `ggml/src/ggml-metal/ggml-metal.metal` so
   `GGML_TYPE_TBQ3_0` / `GGML_TYPE_TBQ4_0` / `GGML_TYPE_QJL1_256` /
   `GGML_TYPE_Q4_POLAR` resolve to the new kernels instead of falling
   back to the CPU path.
3. Run `local-inference/kernels/verify/metal_verify` against
   `local-inference/kernels/verify/fixtures/*.json` to confirm parity.
   Target: max abs diff <= 1e-4 vs the reference C dequantize/score.
4. Once green, drop the `ELIZA_DFLASH_PATCH_METAL_*` opt-in env flags
   in `scripts/build-llama-cpp-dflash.mjs` — these kernels become
   always-on.

## Verification harness

The companion verifier lives in the consumer repo at
`local-inference/kernels/verify/` (W1-D agent). Fixtures are JSON
golden files; harness is `metal_verify.mm` + `Makefile` + a
`qjl_polar_ref.{c,h}` reference dequantizer.
