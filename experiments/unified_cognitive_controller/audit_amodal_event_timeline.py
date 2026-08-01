"""Audit timestamp-aware out-of-order event delivery into one frozen controller."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .amodal_interface import AmodalEvent
from .amodal_runtime import (
    AmodalEventTimeline,
    AmodalInputBus,
    runtime_from_legacy_payload,
)
from .environment import NULL_ACTION, generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _rollout(runtime, bus, batch, *, mode: str, tolerance: float) -> torch.Tensor:
    first, second = split_complementary_views(batch.frames)
    state = runtime.initial_state(batch.batch_size, device=batch.frames.device)
    action = torch.full(
        (batch.batch_size,), NULL_ACTION, dtype=torch.long, device=batch.frames.device
    )
    reward = torch.zeros(batch.batch_size, device=batch.frames.device)
    actions = []
    for trial in range(batch.trials):
        has_feedback = torch.full_like(reward, float(trial == 1))
        timestamp = torch.full(
            (batch.batch_size,), float(trial), device=batch.frames.device
        )
        if mode == "single":
            event = runtime.encode(first[:, trial])
            collection = [event]
        else:
            first_event = AmodalEvent(
                runtime.encode(first[:, trial]).payload, timestamp=timestamp
            )
            second_time = timestamp + (0.25 if mode == "jitter" else 0.0)
            second_event = AmodalEvent(
                runtime.encode(second[:, trial]).payload, timestamp=second_time
            )
            delivered = (
                [second_event, first_event]
                if mode in ("out_of_order", "jitter")
                else [first_event, second_event]
            )
            collection = AmodalEventTimeline(delivered, tolerance=tolerance).windows()
        if mode == "single":
            event = bus(collection)
        else:
            if len(collection) != 1:
                raise AssertionError("aligned timeline should produce one window")
            event = bus(collection[0])
        core, state = runtime.step_intention_event(
            event, state, action, reward * has_feedback, has_feedback
        )
        logits = runtime.decode(core.intent_event)
        action = logits.argmax(dim=-1)
        reward = (action == batch.correct_actions[:, trial]).float()
        actions.append(action)
    return torch.stack(actions, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=157_001)
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
    modes = {
        "single": _rollout(runtime, bus, batch, mode="single", tolerance=0.0),
        "synchronous": _rollout(runtime, bus, batch, mode="synchronous", tolerance=0.0),
        "out_of_order": _rollout(
            runtime, bus, batch, mode="out_of_order", tolerance=0.0
        ),
        "jitter_tolerance": _rollout(runtime, bus, batch, mode="jitter", tolerance=0.5),
    }
    query = slice(1, None)
    accuracies = {
        name: float(
            (actions[:, query] == batch.correct_actions[:, query]).float().mean()
        )
        for name, actions in modes.items()
    }
    out_of_order_exact = torch.equal(modes["synchronous"], modes["out_of_order"])
    jitter_exact = torch.equal(modes["synchronous"], modes["jitter_tolerance"])
    wrong_timestamp_events = [
        AmodalEvent(
            runtime.encode(batch.frames[:, 0]).payload,
            timestamp=torch.zeros(batch.batch_size, device=device),
        ),
        AmodalEvent(
            runtime.encode(batch.frames[:, 0]).payload,
            timestamp=torch.ones(batch.batch_size, device=device),
        ),
    ]
    wrong_timestamp_windows = len(
        AmodalEventTimeline(wrong_timestamp_events, tolerance=0.0).windows()
    )
    passed = bool(
        accuracies["synchronous"] >= 0.90
        and accuracies["single"] <= 0.65
        and accuracies["out_of_order"] >= 0.90
        and accuracies["jitter_tolerance"] >= 0.90
        and out_of_order_exact
        and jitter_exact
        and wrong_timestamp_windows == 2
    )
    report = {
        "schema": "amodal-event-timeline-audit-v1",
        "claim": (
            "Timestamp-preserving out-of-order delivery reconstructs the same "
            "synchronous event collection; mismatched timestamps remain separate."
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
        "accuracies": accuracies,
        "out_of_order_actions_exact": out_of_order_exact,
        "jitter_actions_exact": jitter_exact,
        "wrong_timestamp_windows": wrong_timestamp_windows,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "accuracies": accuracies}, sort_keys=True))


if __name__ == "__main__":
    main()
