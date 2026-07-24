from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import NeuralComputerAgent
from .train_lifetime import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--thresholds", default=".01,.02,.03,.05,.1,.2,.3,.5")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    payload = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    config = payload["arguments"]
    model = NeuralComputerAgent(config["hidden"], config["workspace_slots"], config["heads"],
                                config["thought_steps"], config["choices"]).to(args.device)
    incompatible = model.load_state_dict(payload["model"], strict=False)
    if set(incompatible.missing_keys) - {"log_read_scale"} or incompatible.unexpected_keys:
        raise ValueError(f"incompatible checkpoint: {incompatible}")
    results = {}
    for threshold in map(float, args.thresholds.split(",")):
        metrics = evaluate(
            model, "learned_memory", torch.device(args.device), samples=args.samples,
            batch_size=64, associations=config["associations"], delay=config["delay"],
            choices=config["choices"], seed=1_000_000,
            admission_mode="deterministic_hard", admission_threshold=threshold)
        results[str(threshold)] = {
            "accuracy": metrics["accuracy"],
            "writes_per_lifetime": metrics["writes_per_lifetime"],
        }
    args.report.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
