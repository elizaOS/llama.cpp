#!/usr/bin/env python3
"""Exercise the backend console boundary: incomplete and unsupported runs cannot qualify."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from ci_verify_vulkan import LANES, SHARDED, SHARDS, aggregate, verify


def completed(operation="ADD", statuses=("OK",)):
    rows = [f"  {operation}(case={i}): {status}" for i, status in enumerate(statuses)]
    count = statuses.count("OK")
    return "\n".join(rows + [f"  {count}/{count} tests passed", "  Backend Vulkan0: OK", ""])


class VulkanCompletionTests(unittest.TestCase):
    def test_counts_numeric_and_explicit_unsupported_separately(self):
        text = completed(statuses=("OK", "not supported [Vulkan0]"))
        result = verify(text.replace("OK", "\x1b[32mOK\x1b[0m"), "ADD")
        self.assertEqual(result["numericalPasses"], 1)
        self.assertEqual(result["unsupportedCases"], ["case=1"])

    def test_rejects_invalid_or_incomplete_backend_output(self):
        valid = completed()
        invalid = {
            "empty": "",
            "unsupported-only": completed(statuses=("not supported [Vulkan0]",)),
            "divergence": completed(statuses=("FAIL",)),
            "partial-result": "  ADD(case=0): ",
            "missing-summary": valid.replace("  1/1 tests passed\n", ""),
            "repeated-summary": valid + "  1/1 tests passed\n",
            "wrong-count": valid.replace("1/1", "2/2"),
            "missing-backend": valid.replace("  Backend Vulkan0: OK\n", ""),
            "wrong-operation": completed("MUL"),
            "unexpected-status": completed(statuses=("skipping large tensors for speed",)),
        }
        for name, text in invalid.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify(text, "ADD")

    def test_istft_requires_four_numeric_cases_without_unsupported(self):
        self.assertEqual(verify(completed("ISTFT", ("OK",) * 4), "ISTFT")["numericalPasses"], 4)
        for statuses in [("OK",) * 3, ("OK",) * 4 + ("not supported [Vulkan0]",)]:
            with self.subTest(statuses=statuses), self.assertRaises(ValueError):
                verify(completed("ISTFT", statuses), "ISTFT")


def shard_output(op, index, count, inventory):
    rows = [f"Shard inventory {i}: {op}({params})" for i, params in enumerate(inventory)]
    selected = inventory[index::count]
    rows += [f"  {op}({params}): OK" for params in selected]
    rows += [f"Shard {index}/{count}: {len(selected)} selected of {len(inventory)} matching cases",
             f"  {len(selected)}/{len(selected)} tests passed", "  Backend Vulkan0: OK", ""]
    return "\n".join(rows)


class ShardCoverageTests(unittest.TestCase):
    def records(self):
        records = []
        for group, ops in LANES.items():
            for op in ops:
                for index in (range(2) if (group, op) in SHARDED else [None]):
                    if index is None:
                        r = verify(completed(op, ("OK",) * (4 if op == "ISTFT" else 1)), op)
                    else:
                        # Duplicate identities and a whole-graph case must remain occurrences.
                        r = verify(shard_output(op, index, 2, ["o=1", "o=1", "o=3"]), op, (index, 2))
                    r.update(group=group, head="head", buildSha256="build")
                    records.append(r)
        return records

    def test_union_preserves_duplicate_and_whole_graph_occurrences(self):
        result = aggregate(self.records(), "head", "build", shards=2)
        for r in result:
            if (r["group"], r["operation"]) in SHARDED:
                self.assertEqual(r["numericalPasses"], 3)

    def test_rejects_incomplete_or_inconsistent_worker_sets(self):
        original = self.records()
        controls = {}
        controls["missing-shard"] = original[1:]
        controls["duplicate-shard"] = original + [original[0]]
        for name, key, value in [
            ("wrong-source", "head", "other"),
            ("wrong-build", "buildSha256", "other"),
            ("wrong-inventory", "inventory", ["different"]),
            ("duplicate-index", "selectedIndices", [0, 0]),
            ("wrong-count", "shard", [0, 3]),
        ]:
            records = copy.deepcopy(original)
            records[0][key] = value
            controls[name] = records
        records = copy.deepcopy(original)
        for r in records:
            if r["operation"] == "MUL_MAT":
                r["numericalPasses"] = 0
        controls["all-unsupported-family"] = records
        for name, records in controls.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                aggregate(records, "head", "build", shards=2)

    def test_rejects_missing_extra_or_reordered_occurrence_output(self):
        valid = shard_output("MUL_MAT", 0, 2, ["o=1", "o=1", "o=3"])
        for text in [valid.replace("  MUL_MAT(o=3): OK\n", ""),
                     valid.replace("  MUL_MAT(o=3): OK", "  MUL_MAT(o=1): OK"),
                     valid.replace("Shard inventory 1:", "Shard inventory 2:"),
                     valid.replace("2 selected of 3", "1 selected of 3")]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                verify(text, "MUL_MAT", (0, 2))
        for shard in [(-1, 2), (2, 2), (0, 0), (0, 257)]:
            with self.subTest(shard=shard), self.assertRaises(ValueError):
                verify(valid, "MUL_MAT", shard)


class AggregateCommandTests(unittest.TestCase):
    def test_cli_rechecks_files_and_rejects_changed_or_missing_artifacts(self):
        script = Path(__file__).with_name("ci_verify_vulkan.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(SHARDS):
                folder = root / f"worker-{index}"
                folder.mkdir()
                records = []
                for group, ops in LANES.items():
                    for op in ops:
                        sharded = (group, op) in SHARDED
                        if not sharded and index != 0:
                            continue
                        # Some shards have no assigned occurrences; the complete family must still execute.
                        text = (shard_output(op, index, SHARDS, ["o=1", "o=1", "o=3"])
                                if sharded else completed(op, ("OK",) * (4 if op == "ISTFT" else 1)))
                        log = folder / f"{group}-{op}.log"
                        log.write_text(text)
                        record = verify(text, op, (index, SHARDS) if sharded else None)
                        record.update(group=group, head="head", buildSha256="build", log=log.name,
                                      logSha256=hashlib.sha256(log.read_bytes()).hexdigest())
                        records.append(record)
                (folder / "ledger.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            command = [sys.executable, str(script), "aggregate", str(root), "--head", "head", "--build-sha", "build"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(json.loads(result.stdout)), sum(map(len, LANES.values())))
            log = root / "worker-0" / "broad-MUL_MAT.log"
            original = log.read_bytes()
            log.write_text("changed artifact")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            log.write_bytes(original)
            ledger = root / "worker-1" / "ledger.jsonl"
            ledger.unlink()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
