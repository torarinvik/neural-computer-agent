"""Audit causal complementary N=2 composition across visual appearances."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .amodal_runtime import AmodalInputBus, runtime_from_legacy_payload
from .train_complementary_input_bus import evaluate_bus


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=150_001)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument(
        "--device",
        default=(
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    args = parser.parse_args()
    if args.count < 64 or args.count % 4:
        raise ValueError("count must be at least 64 and divisible by four")

    device = torch.device(args.device)
    controller_payload = torch.load(
        args.controller, map_location=device, weights_only=False
    )
    bus_payload = torch.load(args.input_bus, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    bus.eval()
    before = {
        f"runtime.{name}": value.detach().cpu().clone()
        for name, value in runtime.state_dict().items()
    }
    before.update(
        {
            f"bus.{name}": value.detach().cpu().clone()
            for name, value in bus.state_dict().items()
        }
    )

    results = {
        appearance: evaluate_bus(
            runtime,
            bus,
            count=args.count,
            seed=args.seed + offset * 10_000,
            device=device,
            appearance=appearance,
        )
        for offset, appearance in enumerate(("bars", "diamonds", "dot_pairs"))
    }
    after = {
        f"runtime.{name}": value.detach().cpu()
        for name, value in runtime.state_dict().items()
    }
    after.update(
        {
            f"bus.{name}": value.detach().cpu()
            for name, value in bus.state_dict().items()
        }
    )
    unchanged = all(torch.equal(value, after[name]) for name, value in before.items())
    thresholds = {"bars": 0.90, "diamonds": 0.85, "dot_pairs": 0.90}
    flip_thresholds = {"bars": 0.80, "diamonds": 0.70, "dot_pairs": 0.80}
    gates = {
        appearance: bool(
            row["fused_accuracy"] >= thresholds[appearance]
            and row["stream_a_accuracy"] <= 0.65
            and row["stream_b_accuracy"] <= 0.65
            and row["shuffled_partner_accuracy"] <= 0.60
            and row["contradictory_partner_accuracy"] <= 0.25
            and row["contradictory_prediction_flip_rate"] >= flip_thresholds[appearance]
            and row["full_n1_accuracy"] >= 0.95
            and row["duplicate_actions_exact"]
            and row["duplicate_max_logit_difference"] == 0.0
            and row["permuted_stream_order_actions_exact"]
            and row["permuted_stream_order_max_logit_difference"] == 0.0
        )
        for appearance, row in results.items()
    }
    passed = unchanged and all(gates.values())
    report = {
        "schema": "amodal-complementary-input-audit-v1",
        "claim": (
            "A frozen cognitive controller composes two encoded sensory events "
            "whose individual views are insufficient, using a generic set bus "
            "trained only from attempted actions and scalar outcomes."
        ),
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "device": str(device),
        },
        "appearance_gates": gates,
        "all_parameters_unchanged": unchanged,
        "results": results,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "appearance_gates": gates}, sort_keys=True))


if __name__ == "__main__":
    main()
