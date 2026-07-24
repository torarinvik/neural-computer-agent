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
    parser.add_argument("--graph-eval", type=Path, required=True)
    parser.add_argument("--graph-latency", type=Path, required=True)
    parser.add_argument("--closure-eval", type=Path, required=True)
    parser.add_argument("--closure-latency", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for core, eval_path, latency_path in (
        ("learned_cached_graph", args.graph_eval, args.graph_latency),
        ("neural_transitive_closure", args.closure_eval, args.closure_latency),
    ):
        evaluation, eval_hash = read(eval_path)
        latency, latency_hash = read(latency_path)
        rows.append({"core": core,
                     "overall_accuracy": evaluation["metrics"]["final_accuracy"],
                     "accuracy_by_cards": evaluation["metrics"]["final_accuracy_by_cards"],
                     "perception": {key: evaluation["metrics"][key] for key in
                                    ("subject_accuracy", "relation_accuracy", "object_accuracy",
                                     "final_detection_accuracy")},
                     "incremental_latency": latency["rows"],
                     "evaluation_report": eval_path.name,
                     "evaluation_sha256": eval_hash,
                     "latency_report": latency_path.name,
                     "latency_sha256": latency_hash})
    payload = {"schema": "syllogimous-reasoner-comparison-v1",
               "controlled_inputs": {"train_samples_per_epoch": 75000,
                                     "epochs": 6,
                                     "train_premises": [2, 3, 4, 6, 8, 12, 16],
                                     "test_premises": [24, 32, 64],
                                     "evaluation_episodes_per_core": 6000,
                                     "entity_count": 128,
                                     "public_conclusion_marker": True,
                                     "absolute_positions": False},
               "reasoners": rows}
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

