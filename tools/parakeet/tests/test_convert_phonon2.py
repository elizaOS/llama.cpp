#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Byte-level checks for the Phonon-2 converter, without model downloads."""
import hashlib
import os
from pathlib import Path
import platform
import sys
import unittest

# Select the qualified Linux model-conversion dispatch before NumPy is imported.
if sys.platform == "linux" and platform.machine().lower() in ("x86_64", "amd64"):
    os.environ["NPY_DISABLE_CPU_FEATURES"] = "X86_V4 AVX512_ICL AVX512_SPR"

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import convert_phonon2_to_gguf as converter


class MelFilterbankTest(unittest.TestCase):
    def test_pinned_mel_bytes_are_canonical(self):
        mel = converter.mel_filterbank(16000, 512, 128)
        self.assertEqual(mel.shape, (128, 257))
        self.assertFalse(converter.np.signbit(mel[mel == 0]).any())
        # The original AArch64 GGUF's feat.mel_fb bytes, including positive zero.
        self.assertEqual(hashlib.sha256(mel.tobytes()).hexdigest(),
                         "bce5ec5f194a5913f6508cee5a85512e7bad2352db8fc28f5c6ff75af8b09137")


if __name__ == "__main__":
    unittest.main()
