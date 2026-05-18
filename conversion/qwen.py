"""Qwen-family conversion helpers.

Ported from upstream PR ggml-org/llama.cpp#22673. In upstream, this
file contains the full set of Qwen* model classes; in this fork we
keep `convert_hf_to_gguf.py` monolithic, so only the MTP mixin lives
here — the Qwen3.5 / Qwen3.5 MoE classes themselves are still defined
in `convert_hf_to_gguf.py` and now mix in `_Qwen35MtpMixin` from this
module.

Source (upstream): conversion/qwen.py, lines ~538-617 of PR #22673.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

# Mirror the gguf-py path resolution from convert_hf_to_gguf.py so this
# module can be imported even when invoked from outside the llama.cpp
# tree (e.g. in tests).
if 'NO_LOCAL_GGUF' not in os.environ:
    sys.path.insert(1, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
import gguf  # noqa: E402


class _Qwen35MtpMixin:
    """Shared MTP wiring for Qwen3.5/3.6 text variants.

    The HF config carries the MTP block under `mtp_num_hidden_layers`
    and the tensors under `mtp.*`; we extend block_count, emit the
    nextn metadata key, and remap `mtp.*` to the standard layer-indexed
    nextn naming so the existing tensor_map handles them.

    Class-level `no_mtp` / `mtp_only` flags are toggled by the CLI:
      --mtp     -> mtp_only=True  (emit only the MTP head split file)
      --no-mtp  -> no_mtp=True    (emit only the trunk, drop MTP tensors)
      (neither) -> bundled default (trunk + MTP in one file)
    """

    hparams: dict[str, Any]
    model_arch: "gguf.MODEL_ARCH"
    gguf_writer: "gguf.GGUFWriter"
    block_count: int
    tensor_map: "gguf.TensorNameMap"
    no_mtp: bool = False
    mtp_only: bool = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_count = self.hparams["num_hidden_layers"]
        if not self.no_mtp:
            self.block_count += self.hparams.get("mtp_num_hidden_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    @classmethod
    def filter_tensors(cls, item):
        name, _ = item
        if name.startswith("mtp."):
            if cls.no_mtp:
                return None
            return item
        if cls.mtp_only:
            canonical = name.replace("language_model.", "")
            keep = canonical in (
                "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
                "embed_tokens.weight", "norm.weight",
            )
            if not keep:
                return None
        return super().filter_tensors(item)  # type: ignore[misc]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()  # type: ignore[misc]
        if self.no_mtp:
            return
        if (n := self.hparams.get("mtp_num_hidden_layers", 0)) > 0:
            self.gguf_writer.add_nextn_predict_layers(n)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()  # type: ignore[attr-defined]
        super().prepare_metadata(vocab_only=vocab_only)  # type: ignore[misc]

        if not self.mtp_only or not from_dir:
            return

        output_type: str = self.ftype.name.partition("_")[2]  # type: ignore[attr-defined]
        fname_default: str = gguf.naming_convention(
            self.metadata.name, self.metadata.basename, self.metadata.finetune,             # type: ignore[attr-defined]
            self.metadata.version, size_label=None, output_type=output_type, model_type=None)  # type: ignore[attr-defined]
        self.fname_out = self.fname_out.parent / f"mtp-{fname_default}.gguf"  # type: ignore[attr-defined]

    def modify_tensors(self, data_torch: "Tensor", name: str, bid: int | None) -> Iterable[tuple[str, "Tensor"]]:
        if name.startswith("mtp."):
            n_layer = self.hparams["num_hidden_layers"]
            if name.find("layers.") != -1:
                assert bid is not None
                name = name.replace(f"mtp.layers.{bid}", f"model.layers.{bid + n_layer}")
            else:
                remapper = {
                    "mtp.fc":                    "model.layers.{bid}.eh_proj",
                    "mtp.pre_fc_norm_embedding": "model.layers.{bid}.enorm",
                    "mtp.pre_fc_norm_hidden":    "model.layers.{bid}.hnorm",
                    "mtp.norm":                  "model.layers.{bid}.shared_head.norm",
                }
                stem   = Path(name).stem
                suffix = Path(name).suffix
                tmpl   = remapper[stem] + suffix
                for b in range(n_layer, self.block_count):
                    yield from super().modify_tensors(data_torch, tmpl.format(bid=b), b)  # type: ignore[misc]
                return

        yield from super().modify_tensors(data_torch, name, bid)  # type: ignore[misc]


__all__ = ["_Qwen35MtpMixin"]
