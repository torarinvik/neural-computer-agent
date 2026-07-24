#!/usr/bin/env python3
"""Cross-check native Elisa challenge output against the independent solver."""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

from .challenge_reference import generated

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", default="./experiments/syllogimous_realtime/syllogimous_challenge_dump")
    parser.add_argument("--output", type=Path, default=Path("native_challenge_crosscheck.json"))
    args = parser.parse_args()
    raw = subprocess.check_output([args.binary])
    if len(raw) % 20:
        raise SystemExit(f"native dump has partial record: {len(raw)} bytes")
    records = len(raw) // 20
    mismatches = []
    for seed in range(records):
        native = struct.unpack_from("<5i", raw, seed * 20)
        reference = generated(seed, "intro")
        if tuple(reference.values[:4]) != native[:4] or reference.answer != native[4]:
            mismatches.append({"seed": seed, "native": native,
                               "reference_values": list(reference.values),
                               "reference_answer": reference.answer})
            if len(mismatches) >= 100:
                break
    result = {"schema": "syllogimous.native-crosscheck.v1", "records": records,
              "mismatch_count": len(mismatches), "mismatches": mismatches,
              "solver": "challenge_reference.generated"}
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"records": records, "mismatch_count": len(mismatches),
                      "output": str(args.output)}))
    return 0 if not mismatches else 1

if __name__ == "__main__": raise SystemExit(main())
