"""Audit generic confidence routing for an opaque third input stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .legacy_interface import AmodalEvent
from .legacy_runtime import AmodalInputBus, runtime_from_legacy_payload
from .environment import NULL_ACTION, generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _accuracy(runtime, bus, first, second, third, labels, confidence) -> float:
    count = labels.shape[0]
    state = runtime.initial_state(count, device=labels.device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=labels.device)
    reward = torch.zeros(count, device=labels.device)
    actions = []
    for trial in range(labels.shape[1]):
        feedback = torch.full_like(reward, float(trial == 1))
        events = [
            runtime.encode(first[:, trial]),
            runtime.encode(second[:, trial]),
            AmodalEvent(
                runtime.encode(third[:, trial]).payload,
                confidence=torch.full_like(reward, confidence),
            ),
        ]
        core, state = runtime.step_intention_event(
            bus(events), state, action, reward * feedback, feedback
        )
        action = runtime.decode(core.intent_event).argmax(dim=-1)
        reward = (action == labels[:, trial]).float()
        actions.append(action)
    return float((torch.stack(actions, dim=1)[:, 1:] == labels[:, 1:]).float().mean())


def _load_bus(path: Path, device: torch.device) -> AmodalInputBus:
    payload = torch.load(path, map_location=device, weights_only=False)
    bus = AmodalInputBus(
        int(payload["event_width"]), int(payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(payload["state_dict"])
    return bus.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=172_001)
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
    device = torch.device(args.device)
    controller_payload = torch.load(args.controller, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = _load_bus(args.bus, device)
    batch = generate_lifetimes(
        args.count,
        6,
        seed=args.seed,
        heldout=True,
        task="pair_relation",
        appearance="bars",
        support_trials=1,
        device=device,
    )
    distractor = generate_lifetimes(
        args.count,
        6,
        seed=args.seed + 1,
        heldout=True,
        task="pair_relation",
        appearance="bars",
        support_trials=1,
        device=device,
    )
    first, second = split_complementary_views(batch.frames)
    rows = [
        {
            "confidence": confidence,
            "accuracy": _accuracy(
                runtime,
                bus,
                first,
                second,
                distractor.frames,
                batch.correct_actions,
                confidence,
            ),
        }
        for confidence in (1.0, 0.5, 0.1, 0.01, 0.0)
    ]
    n2_accuracy = _accuracy(
        runtime,
        bus,
        first,
        second,
        batch.frames,
        batch.correct_actions,
        0.0,
    )
    full_confidence = rows[0]["accuracy"]
    low_confidence = rows[-2]["accuracy"]
    passed = bool(
        n2_accuracy >= 0.90
        and full_confidence <= 0.65
        and low_confidence >= 0.90
        and low_confidence - n2_accuracy >= -0.02
        and low_confidence - full_confidence >= 0.25
    )
    report = {
        "schema": "amodal-input-n3-confidence-audit-v1",
        "claim": "Generic encoder confidence suppresses an opaque third-stream distractor.",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "bus": str(args.bus),
        "bus_sha256": _sha256(args.bus),
        "configuration": {"seed": args.seed, "count": args.count, "device": str(device)},
        "n2_accuracy": n2_accuracy,
        "rows": rows,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"n2_accuracy": n2_accuracy, "rows": rows, "passed": passed}))


if __name__ == "__main__":
    main()

