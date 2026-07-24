"""Install an audited linear event/action probe as the real agent readout.

This is an explicitly supervised bootstrap.  It changes only the generic
event-binding answer head and records its provenance in the checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--linear-head-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    payload = torch.load(
        args.controller_checkpoint, map_location="cpu", weights_only=False)
    probe = torch.load(
        args.linear_head_checkpoint, map_location="cpu", weights_only=False)
    state = dict(payload["model"])
    prefix = "latest_row_answer_event_head."
    state = {key: value for key, value in state.items()
             if not key.startswith(prefix)}
    state[prefix + "weight"] = probe["weight"] * args.scale
    state[prefix + "bias"] = probe["bias"] * args.scale

    config_key = ("controller_arguments" if "controller_arguments" in payload
                  else "arguments")
    config = dict(payload[config_key])
    config["latest_row_answer_event_binding"] = True
    config["latest_row_answer_event_linear"] = True
    config["preserve_raw_write"] = False
    config["preserve_first_raw_write"] = True

    payload["model"] = state
    payload[config_key] = config
    payload["event_linear_bootstrap"] = {
        "source_controller": str(args.controller_checkpoint),
        "source_probe": str(args.linear_head_checkpoint),
        "supervised": True,
        "target": "counterfactual_action",
        "scale": args.scale,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
