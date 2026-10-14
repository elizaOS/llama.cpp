import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gguf import GGUFReader, GGUFWriter, GGMLQuantizationType, GGML_QUANT_SIZES
from gguf.scripts.gguf_new_metadata import copy_with_new_metadata


def text(value):
    data = value.encode()
    return struct.pack('<Q', len(data)) + data


def fixture(path, type_id, *, ftype=None, layout=None):
    metadata = [('general.architecture', 8, text('llama'))]
    if ftype is not None:
        metadata.append(('general.file_type', 4, struct.pack('<I', ftype)))
    if layout is not None:
        metadata.append(('eliza.quantization_type_ids', 8, text(layout)))
    data = b'GGUF' + struct.pack('<IQQ', 3, 1, len(metadata))
    for name, kind, value in metadata:
        data += text(name) + struct.pack('<I', kind) + value
    data += text('weight') + struct.pack('<IQIQ', 1, 256, type_id, 0)
    data += bytes((-len(data)) % 32)
    data += bytes(i % 251 for i in range(1024))
    path.write_bytes(data)


class ElizaTypeIds(unittest.TestCase):
    def test_legacy_and_upstream(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.gguf'
            cases = [(42, 42, None, None), (42, 256, 100, None),
                     (42, 256, None, 'legacy-42'), (42, 42, 100, 'private-256')]
            cases += [(i, i + 214, None, None) for i in range(43, 51)]
            cases += [(i, i, None, None) for i in range(256, 265)]
            for raw, expected, ftype, layout in cases:
                with self.subTest(raw=raw, ftype=ftype, layout=layout):
                    fixture(path, raw, ftype=ftype, layout=layout)
                    self.assertEqual(GGUFReader(path).tensors[0].tensor_type, expected)

    def test_explicit_migration_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            old = Path(directory) / 'old.gguf'
            new = Path(directory) / 'new.gguf'
            fixture(old, 42)
            reader = GGUFReader(old, eliza_legacy_type_ids=True)
            expected = reader.tensors[0].data.copy()
            writer = GGUFWriter(new, 'llama')
            copy_with_new_metadata(reader, writer, {}, [])
            actual = GGUFReader(new)
            self.assertEqual(actual.tensors[0].tensor_type, GGMLQuantizationType.Q1_0_g32)
            self.assertEqual(actual.fields['eliza.quantization_type_ids'].contents(), 'private-256')
            np.testing.assert_array_equal(actual.tensors[0].data, expected)

    def test_new_writer_all_types(self):
        with tempfile.TemporaryDirectory() as directory:
            for type_id in [42, *range(256, 265)]:
                kind = GGMLQuantizationType(type_id)
                block, size = GGML_QUANT_SIZES[kind]
                path = Path(directory) / f'{type_id}.gguf'
                writer = GGUFWriter(path, 'llama')
                writer.add_file_type(100)
                writer.add_tensor('weight', np.zeros((1, size), dtype=np.uint8), raw_dtype=kind)
                writer.write_header_to_file()
                writer.write_kv_data_to_file()
                writer.write_tensors_to_file()
                writer.close()
                result = GGUFReader(path)
                self.assertEqual(result.tensors[0].tensor_type, kind)
                self.assertEqual(int(result.tensors[0].shape[0]), block)


if __name__ == '__main__':
    unittest.main()
