"""Diagnose exact hosted Vulkan cases without rebuilding or qualifying the matrix."""

from pathlib import Path
import hashlib
import importlib.util
import json
import re
import subprocess
import time


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


root = Path.cwd()
out = root / "diagnostic-results"
out.mkdir(exist_ok=False)
capsule = json.loads((root / "diagnostic-cases.json").read_text())
assert sha(root / "vulkan-runtime.tar.gz") == capsule["buildSha256"]
spec = importlib.util.spec_from_file_location(
    "verifier", root / "scripts/ci_verify_vulkan.py"
)
if spec is None or spec.loader is None:
    raise RuntimeError("Cannot load the pinned Vulkan verifier")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
binary = root / "build-vulkan/bin/test-backend-ops"
artifacts = {
    str(p.relative_to(root)): sha(p)
    for p in (root / "build-vulkan").rglob("*")
    if p.is_file()
}
(out / "runtime-hashes.json").write_text(json.dumps(artifacts, indent=2) + "\n")
rows = []
inputs = [
    (
        c["id"],
        ["-p", "^" + re.sub(r"([\\.^$|?*+()\[\]{}])", r"\\\1", c["params"]) + "$"],
        c,
    )
    for c in capsule["cases"]
]
inputs.append(("original-shard1", ["--test-shard", "1/32"], None))
for name, extra, case in inputs:
    command = [
        "timeout",
        "600",
        str(binary),
        "-b",
        "Vulkan0",
        "-o",
        "FLASH_ATTN_EXT",
        *extra,
    ]
    start = time.monotonic()
    with (
        (out / (name + ".stdout")).open("x") as stdout,
        (out / (name + ".stderr")).open("x") as stderr,
    ):
        result = subprocess.run(command, stdout=stdout, stderr=stderr)
    row = {
        "id": name,
        "command": command,
        "seconds": time.monotonic() - start,
        "exit": result.returncode,
        "stdoutSha256": sha(out / (name + ".stdout")),
        "stderrSha256": sha(out / (name + ".stderr")),
        "completeNumericalResult": False,
    }
    if result.returncode == 0:
        try:
            verified = verifier.verify(
                (out / (name + ".stdout")).read_text(),
                "FLASH_ATTN_EXT",
                None if case else [1, 32],
            )
            if case:
                assert verified["numericalPasses"] == case["expectedOccurrences"] > 0
                assert (
                    verified["numericalCases"]
                    == [case["params"]] * case["expectedOccurrences"]
                )
                assert not verified["unsupportedCases"]
            else:
                assert verified["numericalPasses"] > 0
            row.update(
                completeNumericalResult=True,
                numericalPasses=verified["numericalPasses"],
            )
        except (ValueError, AssertionError) as error:
            row["verificationFailure"] = (
                str(error) or "Exact numerical occurrence mismatch"
            )
    rows.append(row)
    (out / "receipt.json").write_text(
        json.dumps(
            {
                "scope": "Diagnostic only, not matrix qualification",
                "cases": rows,
                "runtimeHashesUnchanged": all(
                    sha(root / p) == h for p, h in artifacts.items()
                ),
            },
            indent=2,
        )
        + "\n"
    )
assert all(sha(root / p) == h for p, h in artifacts.items())
raise SystemExit(0 if all(r["completeNumericalResult"] for r in rows) else 1)
