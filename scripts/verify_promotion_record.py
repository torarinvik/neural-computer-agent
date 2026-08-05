#!/usr/bin/env python3
"""Re-evaluate a serialized promotion record against its holdout ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from neural_computer.promotion import (
    HoldoutLedger,
    evaluate_promotion,
    read_promotion_record,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("record", type=Path)
    parser.add_argument(
        "--holdout-ledger",
        type=Path,
        required=True,
        help="append-only ledger containing the exact consumed holdout lease",
    )
    args = parser.parse_args()
    gate, evidence, recorded = read_promotion_record(args.record)
    decision = evaluate_promotion(
        gate,
        evidence,
        holdout_ledger=HoldoutLedger(args.holdout_ledger),
    )
    print(
        json.dumps(
            {
                "recorded_decision": recorded.canonical(),
                "verified_decision": decision.canonical(),
            },
            sort_keys=True,
        )
    )
    return 0 if decision.eligible else 2


if __name__ == "__main__":
    raise SystemExit(main())
