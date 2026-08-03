"""Audit N=3 distractor rejection and N=2 retention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

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
def _accuracy(runtime, bus, streams, labels) -> float:
    count = labels.shape[0]
    state = runtime.initial_state(count, device=labels.device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=labels.device)
    reward = torch.zeros(count, device=labels.device)
    actions = []
    for trial in range(labels.shape[1]):
        feedback = torch.full_like(reward, float(trial == 1))
        event = bus([runtime.encode(stream[:, trial]) for stream in streams])
        core, state = runtime.step_intention_event(
            event, state, action, reward * feedback, feedback
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
    parser.add_argument("--baseline-bus", type=Path, required=True)
    parser.add_argument("--adapted-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=166_001)
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
    runtime_payload = torch.load(args.controller, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(runtime_payload, device=device).eval()
    baseline = _load_bus(args.baseline_bus, device)
    adapted = _load_bus(args.adapted_bus, device)
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
    distractor_first, _ = split_complementary_views(distractor.frames)
    shuffled = distractor_first.roll(1, 0)
    rows = []
    for name, bus in (("baseline", baseline), ("adapted", adapted)):
        n2 = _accuracy(runtime, bus, (first, second), batch.correct_actions)
        n3 = _accuracy(
            runtime, bus, (first, second, distractor.frames), batch.correct_actions
        )
        shuffled_n3 = _accuracy(
            runtime, bus, (first, second, shuffled), batch.correct_actions
        )
        rows.append(
            {
                "bus": name,
                "n2_accuracy": n2,
                "n3_distractor_accuracy": n3,
                "n3_shuffled_distractor_accuracy": shuffled_n3,
            }
        )
    base, candidate = rows
    n3_gain = candidate["n3_distractor_accuracy"] - base["n3_distractor_accuracy"]
    n2_delta = candidate["n2_accuracy"] - base["n2_accuracy"]
    passed = bool(
        candidate["n3_distractor_accuracy"] >= 0.85
        and candidate["n3_shuffled_distractor_accuracy"] <= 0.65
        and n2_delta >= -0.02
        and n3_gain >= 0.20
    )
    report = {
        "schema": "amodal-input-n3-distractor-audit-v1",
        "claim": "A learned generic input bus rejects an opaque third-stream distractor.",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "baseline_bus": str(args.baseline_bus),
        "baseline_bus_sha256": _sha256(args.baseline_bus),
        "adapted_bus": str(args.adapted_bus),
        "adapted_bus_sha256": _sha256(args.adapted_bus),
        "configuration": {"seed": args.seed, "count": args.count, "device": str(device)},
        "rows": rows,
        "n3_gain": n3_gain,
        "n2_delta": n2_delta,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": rows, "passed": passed}, sort_keys=True))


if __name__ == "__main__":
    main()

