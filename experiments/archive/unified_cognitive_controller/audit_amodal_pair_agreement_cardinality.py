"""Audit pair-agreement routing as the number of distractors increases."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import torch

from .legacy_interface import AmodalEvent
from .legacy_runtime import AmodalInputBus, runtime_from_legacy_payload
from .environment import NULL_ACTION, generate_lifetimes
from .train_amodal_pair_agreement import AmodalPairAgreement
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_bus(path: Path, device: torch.device) -> AmodalInputBus:
    payload = torch.load(path, map_location=device, weights_only=False)
    bus = AmodalInputBus(
        int(payload["event_width"]), int(payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(payload["state_dict"])
    return bus.eval()


def _load_agreement(path: Path, device: torch.device) -> AmodalPairAgreement:
    payload = torch.load(path, map_location=device, weights_only=False)
    agreement = AmodalPairAgreement(
        int(payload["event_width"]), int(payload["hidden"])
    ).to(device)
    agreement.load_state_dict(payload["state_dict"])
    return agreement.eval()


@torch.no_grad()
def _accuracy(
    runtime,
    bus,
    agreement,
    streams,
    labels,
    threshold: float | None,
) -> tuple[float, float, float]:
    count = labels.shape[0]
    state = runtime.initial_state(count, device=labels.device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=labels.device)
    reward = torch.zeros(count, device=labels.device)
    actions = []
    confidence_rows = []
    for trial in range(labels.shape[1]):
        feedback = torch.full_like(reward, float(trial == 1))
        encoded = [runtime.encode(stream[:, trial]) for stream in streams]
        if threshold is None:
            confidence = torch.ones(count, len(encoded), device=labels.device)
        else:
            pair_scores = {
                pair: agreement(encoded[pair[0]].payload, encoded[pair[1]].payload)
                for pair in combinations(range(len(encoded)), 2)
            }
            confidence = []
            for index in range(len(encoded)):
                strongest = torch.stack(
                    [
                        pair_scores[tuple(sorted((index, other)))]
                        for other in range(len(encoded))
                        if other != index
                    ],
                    dim=1,
                ).max(dim=1).values
                confidence.append(
                    ((strongest - threshold) / (1.0 - threshold)).clamp_min(0.0)
                )
            confidence = torch.stack(confidence, dim=1)
        confidence_rows.append(confidence)
        events = [
            AmodalEvent(event.payload, confidence=confidence[:, index])
            for index, event in enumerate(encoded)
        ]
        core, state = runtime.step_intention_event(
            bus(events), state, action, reward * feedback, feedback
        )
        action = runtime.decode(core.intent_event).argmax(dim=-1)
        reward = (action == labels[:, trial]).float()
        actions.append(action)
    accuracy = float((torch.stack(actions, dim=1)[:, 1:] == labels[:, 1:]).float().mean())
    confidence = torch.stack(confidence_rows, dim=1)
    return accuracy, float(confidence[:, :, :2].mean()), float(confidence[:, :, 2:].mean()) if confidence.shape[2] > 2 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--bus", type=Path, required=True)
    parser.add_argument("--agreement", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=177_001)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--max-cardinality", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.8)
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
    if not 0.0 <= args.threshold < 1.0:
        raise ValueError("threshold must be within [0, 1)")
    if not 2 <= args.max_cardinality <= 16:
        raise ValueError("max cardinality must be between 2 and 16")
    device = torch.device(args.device)
    runtime_payload = torch.load(args.controller, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(runtime_payload, device=device).eval()
    bus = _load_bus(args.bus, device)
    agreement = _load_agreement(args.agreement, device)
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
    first, second = split_complementary_views(batch.frames)
    distractors = [
        generate_lifetimes(
            args.count,
            6,
            seed=args.seed + offset,
            heldout=True,
            task="pair_relation",
            appearance="bars",
            support_trials=1,
            device=device,
        ).frames
        for offset in range(2, args.max_cardinality)
    ]
    rows = {}
    for cardinality in range(2, args.max_cardinality + 1):
        streams = (first, second, *distractors[: cardinality - 2])
        no_agreement = _accuracy(
            runtime, bus, agreement, streams, batch.correct_actions, None
        )
        learned = _accuracy(
            runtime, bus, agreement, streams, batch.correct_actions, args.threshold
        )
        rows[str(cardinality)] = {
            "without_agreement": no_agreement[0],
            "with_agreement": learned[0],
            "gain": learned[0] - no_agreement[0],
            "useful_confidence": learned[1],
            "distractor_confidence": learned[2],
        }
    passed = bool(
        rows["2"]["with_agreement"] >= 0.90
        and all(
            rows[str(cardinality)]["with_agreement"] >= 0.85
            for cardinality in range(3, args.max_cardinality + 1)
        )
        and all(
            rows[str(cardinality)]["gain"] >= 0.25
            for cardinality in range(3, args.max_cardinality + 1)
        )
    )
    report = {
        "schema": "amodal-pair-agreement-cardinality-audit-v1",
        "claim": f"Pair agreement scales relevance routing from N=2 to N={args.max_cardinality}.",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "bus": str(args.bus),
        "bus_sha256": _sha256(args.bus),
        "agreement": str(args.agreement),
        "agreement_sha256": _sha256(args.agreement),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "threshold": args.threshold,
            "max_cardinality": args.max_cardinality,
            "device": str(device),
        },
        "rows": rows,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": rows, "passed": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
