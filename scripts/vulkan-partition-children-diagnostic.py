"""Measure two smaller partitions of one timed-out shard without rebuilding the runtime."""

from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import time

ROOT = Path.cwd()
OUT = ROOT / "partition-child-results"
OUT.mkdir(exist_ok=False)
BUILD_SHA = "4a8824b685224613468c7be9211244058e2d49e5e7a5de31f3f7d887d03e102a"
REFERENCE_SHA = "19fb24bdc4fa6fef547c62435492040de832aca929bf1aece6200c989f552136"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


assert sha(ROOT / "vulkan-runtime.tar.gz") == BUILD_SHA
reference = ROOT / "reference-shard/broad-FLASH_ATTN_EXT.log"
assert sha(reference) == REFERENCE_SHA
spec = importlib.util.spec_from_file_location(
    "verifier", ROOT / "scripts/ci_verify_vulkan.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load the pinned Vulkan verifier")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
canonical = verifier.verify(reference.read_text(), "FLASH_ATTN_EXT", [0, 32])[
    "inventory"
]
assert len(canonical) == 5096
parent_indices = list(range(1, len(canonical), 32))
expected_children = {index: list(range(index, len(canonical), 64)) for index in [1, 33]}
assert sorted(expected_children[1] + expected_children[33]) == parent_indices
assert set(expected_children[1]).isdisjoint(expected_children[33])
runtime = {
    str(f.relative_to(ROOT)): sha(f)
    for f in (ROOT / "build-vulkan").rglob("*")
    if f.is_file()
}
save(
    "binding.json",
    {
        "buildSha256": BUILD_SHA,
        "referenceLogSha256": REFERENCE_SHA,
        "runtime": runtime,
        "parentIndices": parent_indices,
        "childIndices": expected_children,
        "sourceInventory": canonical,
        "scope": "Fixed-artifact partition experiment, not full matrix qualification",
    },
)
rows = []
verified_indices = []
for index in [1, 33]:
    name = f"child-{index}-of-64"
    command = [
        "timeout",
        "--kill-after=10s",
        "600",
        str(ROOT / "build-vulkan/bin/test-backend-ops"),
        "-b",
        "Vulkan0",
        "-o",
        "FLASH_ATTN_EXT",
        "--test-shard",
        f"{index}/64",
    ]
    start = time.monotonic()
    with (
        (OUT / (name + ".stdout")).open("x") as stdout,
        (OUT / (name + ".stderr")).open("x") as stderr,
    ):
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    row = {
        "id": name,
        "command": command,
        "seconds": time.monotonic() - start,
        "exit": result.returncode,
        "stdoutSha256": sha(OUT / (name + ".stdout")),
        "stderrSha256": sha(OUT / (name + ".stderr")),
        "completeNumericalResult": False,
    }
    if result.returncode == 0:
        try:
            verified = verifier.verify(
                (OUT / (name + ".stdout")).read_text(), "FLASH_ATTN_EXT", [index, 64]
            )
            assert verified["inventory"] == canonical
            assert verified["selectedIndices"] == expected_children[index]
            assert verified["numericalPasses"] > 0
            verified_indices.extend(verified["selectedIndices"])
            row.update(
                completeNumericalResult=True,
                numericalPasses=verified["numericalPasses"],
                explicitUnsupported=len(verified["unsupportedCases"]),
                selectedIndices=verified["selectedIndices"],
            )
        except (ValueError, AssertionError) as error:
            row["verificationFailure"] = (
                str(error) or "Exact occurrence inventory mismatch"
            )
    rows.append(row)
    save(
        "receipt.json",
        {
            "cases": rows,
            "runtimeHashesUnchanged": all(
                sha(ROOT / f) == h for f, h in runtime.items()
            ),
            "completeParentOccurrenceUnion": len(rows) == 2
            and sorted(verified_indices) == parent_indices,
            "scope": "Diagnostic only; all other partitions remain unqualified",
        },
    )
assert all(sha(ROOT / f) == h for f, h in runtime.items())
assert sha(reference) == REFERENCE_SHA
raise SystemExit(
    0
    if all(r["completeNumericalResult"] for r in rows)
    and sorted(verified_indices) == parent_indices
    else 1
)
