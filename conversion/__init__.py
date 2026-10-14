"""Eliza-1 minimal `conversion` package.

Upstream PR ggml-org/llama.cpp#22673 refactored `convert_hf_to_gguf.py`
into a modular `conversion/` package (one file per model family). The
MTP cherry-pick (commit 276163573 in this fork) imported
`_Qwen35MtpMixin` from `conversion.qwen` but the package itself was
backed out because we still ship the monolithic `convert_hf_to_gguf.py`.

This shim restores only the MTP mixin so that
`python convert_hf_to_gguf.py --mtp <dir>` can resolve its import. The
full modular refactor is intentionally NOT vendored — we only need
the bits required to bake MTP heads into Eliza-1 Qwen3.5 GGUFs.
"""

from .qwen import _Qwen35MtpMixin

__all__ = ["_Qwen35MtpMixin"]
