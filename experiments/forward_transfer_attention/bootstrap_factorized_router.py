"""Install a supervised diagnostic router into the optional agent module."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--router-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--ood-threshold", type=float)
    args = parser.parse_args()

    payload = torch.load(
        args.controller_checkpoint, map_location="cpu", weights_only=False)
    router = torch.load(
        args.router_checkpoint, map_location="cpu", weights_only=False)
    state = dict(payload["model"])
    for key, value in router["model"].items():
        state["latest_row_factorized_router." + key] = value
    state["latest_row_factorized_strength"] = torch.tensor(args.strength)

    config_key = ("controller_arguments" if "controller_arguments" in payload
                  else "arguments")
    config = dict(payload[config_key])
    config["latest_row_reader"] = True
    config["latest_row_answer_fusion"] = False
    config["latest_row_answer_gate"] = False
    config["latest_row_answer_pairwise"] = False
    config["latest_row_answer_event_binding"] = False
    config["latest_row_answer_event_linear"] = False
    config["latest_row_answer_factorized_router"] = True
    config["latest_row_factorized_ood_threshold"] = args.ood_threshold
    config["preserve_raw_write"] = False
    config["preserve_first_raw_write"] = True
    payload["model"] = state
    payload[config_key] = config
    payload["factorized_router_bootstrap"] = {
        "source_controller": str(args.controller_checkpoint),
        "source_router": str(args.router_checkpoint),
        "strength": args.strength,
        "ood_threshold": args.ood_threshold,
        "supervised": True,
        "sensory_only": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
