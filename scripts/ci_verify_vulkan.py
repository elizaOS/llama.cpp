#!/usr/bin/env python3
"""Verify complete Vulkan results and the union of deterministic test shards.

Unsupported outcomes remain explicit and never count as numerical passes.
Occurrence indices preserve duplicate cases and whole-graph tests; support-mode
inventories are deliberately not used because they omit whole-graph tests.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LANES = {
    "broad": ["MUL_MAT", "MUL_MAT_ID", "ADD", "MUL", "SOFT_MAX", "RMS_NORM", "CPY", "ROPE", "FLASH_ATTN_EXT"],
    "custom": ["GET_ROWS", "CPY", "MUL_MAT", "ATTN_SCORE_TBQ", "ATTN_SCORE_POLAR", "ISTFT"],
}
SHARDED = {("broad", "MUL_MAT"), ("broad", "FLASH_ATTN_EXT"), ("custom", "MUL_MAT")}
SHARDS = 64


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(text, operation, shard=None):
    text = ANSI.sub("", text)
    results = re.findall(r"^  ([A-Z0-9_]+)\((.*)\): (.*)$", text, re.M)
    passed, unsupported, observed = [], [], []
    for op, params, status in results:
        if op != operation:
            raise ValueError("Unexpected operation: " + op)
        observed.append(params)
        if status.strip() == "OK":
            passed.append(params)
        elif status.strip() == "not supported [Vulkan0]":
            unsupported.append(params)
        else:
            raise ValueError("Nonpassing or incomplete outcome: " + op + "(" + params + "): " + status)
    inventory = []
    selected = []
    inventory_rows = re.findall(r"^Shard inventory (\d+): ([A-Z0-9_]+)\((.*)\)$", text, re.M)
    shard_rows = re.findall(r"^Shard (\d+)/(\d+): (\d+) selected of (\d+) matching cases$", text, re.M)
    if shard is not None:
        index, count = shard
        if not 0 <= index < count <= 256:
            raise ValueError("Invalid shard index/count")
        for ordinal, (actual, op, params) in enumerate(inventory_rows):
            if int(actual) != ordinal or op != operation:
                raise ValueError("Noncontiguous or unexpected inventory")
            inventory.append(params)
        if not inventory:
            raise ValueError("Missing full occurrence inventory")
        selected = list(range(index, len(inventory), count))
        if shard_rows != [(str(index), str(count), str(len(selected)), str(len(inventory)))]:
            raise ValueError("Missing or inconsistent shard completion")
        if observed != [inventory[i] for i in selected]:
            raise ValueError("Shard results do not equal assigned occurrence sequence")
    elif inventory_rows or shard_rows or not passed:
        raise ValueError("Unexpected shard output or no numerical cases")
    summaries = re.findall(r"^  (\d+)/(\d+) tests passed$", text, re.M)
    if summaries != [(str(len(passed)), str(len(passed)))]:
        raise ValueError("Missing, repeated or inconsistent numerical completion summary")
    if len(re.findall(r"^  Backend Vulkan0: OK$", text, re.M)) != 1:
        raise ValueError("Vulkan backend did not complete exactly once")
    if operation == "ISTFT" and (unsupported or len(passed) < 4):
        raise ValueError("All four required ISTFT cases must execute numerically")
    return {"operation": operation, "numericalPasses": len(passed),
            "unsupportedCases": unsupported, "numericalCases": passed,
            "inventory": inventory, "selectedIndices": selected,
            "shard": list(shard) if shard is not None else None}


def aggregate(records, head, build_sha, shards=SHARDS):
    expected = {(group, op, index if (group, op) in SHARDED else None)
                for group, ops in LANES.items() for op in ops
                for index in (range(shards) if (group, op) in SHARDED else [0])}
    found = {}
    for r in records:
        if r["head"] != head or r["buildSha256"] != build_sha:
            raise ValueError("Source or build artifact mismatch")
        shard = r["shard"]
        key = (r["group"], r["operation"], shard[0] if shard else None)
        if key not in expected or key in found or (shard and shard[1] != shards):
            raise ValueError("Unexpected, duplicate or misconfigured invocation")
        found[key] = r
    if set(found) != expected:
        raise ValueError("Missing terminal invocations")
    totals = []
    for group, ops in LANES.items():
        for op in ops:
            family = [r for (g, o, _), r in found.items() if (g, o) == (group, op)]
            numeric = sum(r["numericalPasses"] for r in family)
            if numeric == 0:
                raise ValueError("No numerical execution for " + group + "/" + op)
            if (group, op) in SHARDED:
                inventory = family[0]["inventory"]
                indices = []
                for r in family:
                    if r["inventory"] != inventory:
                        raise ValueError("Workers generated different complete inventories")
                    indices.extend(r["selectedIndices"])
                if Counter(indices) != Counter(range(len(inventory))):
                    raise ValueError("Shard union omits or repeats occurrences")
            totals.append({"group": group, "operation": op, "numericalPasses": numeric,
                           "explicitUnsupported": sum(len(r["unsupportedCases"]) for r in family)})
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    record = sub.add_parser("record")
    record.add_argument("group", choices=LANES)
    record.add_argument("operation")
    record.add_argument("log", type=Path)
    record.add_argument("ledger", type=Path)
    record.add_argument("--shard", type=int)
    collect = sub.add_parser("aggregate")
    collect.add_argument("root", type=Path)
    for command in (record, collect):
        command.add_argument("--head", required=True)
        command.add_argument("--build-sha", required=True)
    args = parser.parse_args()
    if args.mode == "record":
        shard = (args.shard, SHARDS) if args.shard is not None else None
        r = verify(args.log.read_text(), args.operation, shard)
        r.update(group=args.group, head=args.head, buildSha256=args.build_sha,
                 log=args.log.name, logSha256=sha(args.log))
        with args.ledger.open("a") as output:
            output.write(json.dumps(r) + "\n")
        # stdout is the machine-readable CLI result; failures retain stderr and nonzero exit.
        sys.stdout.write(json.dumps({k: r[k] for k in ("group", "operation", "shard", "numericalPasses")}) + "\n")
    else:
        records = []
        for ledger in sorted(args.root.glob("*/ledger.jsonl")):
            for line in ledger.read_text().splitlines():
                r = json.loads(line)
                log = ledger.parent / r["log"]
                if log.name != r["log"] or sha(log) != r["logSha256"]:
                    raise ValueError("Log path or digest mismatch")
                verified = verify(log.read_text(), r["operation"], r["shard"])
                if any(r[key] != value for key, value in verified.items()):
                    raise ValueError("Receipt does not match raw output")
                records.append(r)
        sys.stdout.write(json.dumps(aggregate(records, args.head, args.build_sha), indent=2) + "\n")


if __name__ == "__main__":
    main()
