from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.reports:
        raw = path.read_bytes()
        report = json.loads(raw)
        generalization = report["generalization"]
        rows.append({
            "core": report["core"],
            "parameters": report["parameters"],
            "trained_length_final_accuracy": report["history"][-1]["validation"]["final_accuracy"],
            "heldout_final_accuracy": generalization["final_accuracy"],
            "final_accuracy_by_cards": generalization["final_accuracy_by_cards"],
            "model_milliseconds_per_event": generalization["model_milliseconds_per_event"],
            "report": path.name,
            "report_sha256": hashlib.sha256(raw).hexdigest(),
        })
    payload = {
        "schema": "syllogimous-latent-comparison-v1",
        "training_premises": [2, 3, 4, 5, 6],
        "evaluation_premises": [2, 4, 6, 8, 16],
        "strict_inference_inputs": ["rgb", "pcm", "padding_mask"],
        "models": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

