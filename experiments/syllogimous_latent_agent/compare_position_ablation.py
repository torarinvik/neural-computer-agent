from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-positions-eval", type=Path, required=True)
    parser.add_argument("--without-positions-eval", type=Path, required=True)
    parser.add_argument("--with-positions-latency", type=Path, required=True)
    parser.add_argument("--without-positions-latency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    variants = []
    for name, enabled, eval_path, latency_path in (
        ("with_positions", True, args.with_positions_eval, args.with_positions_latency),
        ("without_positions", False, args.without_positions_eval, args.without_positions_latency),
    ):
        evaluation, eval_hash = read(eval_path)
        latency, latency_hash = read(latency_path)
        variants.append({
            "variant": name,
            "use_positions": enabled,
            "overall_accuracy": evaluation["metrics"]["final_accuracy"],
            "accuracy_by_cards": evaluation["metrics"]["final_accuracy_by_cards"],
            "confusion_by_cards": evaluation["metrics"]["final_confusion_by_cards"],
            "incremental_latency": latency["rows"],
            "evaluation_report": eval_path.name,
            "evaluation_sha256": eval_hash,
            "latency_report": latency_path.name,
            "latency_sha256": latency_hash,
        })
    payload = {"schema": "syllogimous-position-ablation-v1",
               "controlled_change": "learned absolute position embeddings",
               "evaluation_episodes_per_variant": 6000,
               "variants": variants}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

