# Eliza quantization compatibility

Upstream GGUF type 42 is Q2_0. Eliza types now occupy 256 through 264,
respectively Q1_0_g32, Q1_0_g128, TBQ3_0, TBQ4_0, QJL1_256, Q4_POLAR,
TBQ3_TCQ, TBQ3_K and TBQ4_K. Tensor block layouts are unchanged.
New writers mark these files with `eliza.quantization_type_ids = private-256`.

The native and Python readers translate historical IDs 43–50 automatically.
Historical ID 42 is translated only when `general.file_type` is 100–102
(without a private-256 marker), or the explicit layout marker is `legacy-42`.
Unmarked ID 42 retains its upstream Q2_0 meaning. File names are not evidence
of a tensor format.

For a known old Eliza file that lacks identifying metadata, rewrite its type
IDs with an explicit declaration of the original format:

```sh
PYTHONPATH=gguf-py python3 -m gguf.scripts.gguf_migrate_eliza old.gguf new.gguf --legacy-type-ids
```

This writes a separate file and preserves quantized tensor bytes. Do not use
this option on upstream Q2_0 files. Old binaries cannot read the new private
IDs; update consumers together with their native library pin.

RPC peers must both use this Eliza build. Its private major version 128 rejects
stock upstream peers and old Eliza peers, whose operation IDs are incompatible.
