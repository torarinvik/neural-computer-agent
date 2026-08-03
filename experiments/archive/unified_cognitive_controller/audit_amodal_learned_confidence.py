"""Audit a learned confidence head on corrupted N=3 streams."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .legacy_interface import AmodalEvent
from .legacy_runtime import AmodalInputBus, runtime_from_legacy_payload
from .environment import NULL_ACTION, generate_lifetimes
from .train_amodal_confidence_estimator import AmodalConfidenceEstimator
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _erase(frames: torch.Tensor, fraction: float, generator: torch.Generator) -> torch.Tensor:
    background = frames[..., :1, :1]
    mask = torch.rand(
        frames.shape[0],
        frames.shape[1],
        1,
        frames.shape[-2],
        frames.shape[-1],
        generator=generator,
        device=frames.device,
    ) < fraction
    return torch.where(mask, background, frames)


@torch.no_grad()
def _accuracy(
    runtime,
    bus,
    estimator,
    first,
    second,
    third,
    labels,
    use_confidence,
    confidence_power: float,
) -> tuple[float, float, float]:
    count = labels.shape[0]
    state = runtime.initial_state(count, device=labels.device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=labels.device)
    reward = torch.zeros(count, device=labels.device)
    actions = []
    third_confidences = []
    for trial in range(labels.shape[1]):
        feedback = torch.full_like(reward, float(trial == 1))
        encoded = [
            runtime.encode(first[:, trial]),
            runtime.encode(second[:, trial]),
            runtime.encode(third[:, trial]),
        ]
        if estimator is None:
            third_confidence = torch.ones_like(reward)
            events = encoded
        else:
            confidences = [estimator(event.payload).pow(confidence_power) for event in encoded]
            third_confidence = confidences[-1]
            events = [
                AmodalEvent(event.payload, confidence=confidence)
                for event, confidence in zip(encoded, confidences, strict=True)
            ]
        if not use_confidence:
            events = [AmodalEvent(event.payload) for event in encoded]
        event = bus(events)
        core, state = runtime.step_intention_event(
            event, state, action, reward * feedback, feedback
        )
        action = runtime.decode(core.intent_event).argmax(dim=-1)
        reward = (action == labels[:, trial]).float()
        actions.append(action)
        third_confidences.append(third_confidence)
    accuracy = float((torch.stack(actions, dim=1)[:, 1:] == labels[:, 1:]).float().mean())
    confidence = torch.stack(third_confidences, dim=1)
    return accuracy, float(confidence.mean()), float(confidence[:, 1:].mean())


def _load_bus(path: Path, device: torch.device) -> AmodalInputBus:
    payload = torch.load(path, map_location=device, weights_only=False)
    bus = AmodalInputBus(
        int(payload["event_width"]), int(payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(payload["state_dict"])
    return bus.eval()


def _load_estimator(path: Path, device: torch.device) -> AmodalConfidenceEstimator:
    payload = torch.load(path, map_location=device, weights_only=False)
    estimator = AmodalConfidenceEstimator(
        int(payload["event_width"]), int(payload["hidden"])
    ).to(device)
    estimator.load_state_dict(payload["state_dict"])
    return estimator.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--bus", type=Path, required=True)
    parser.add_argument("--estimator", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=174_001)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--third-erase", type=float, default=0.8)
    parser.add_argument("--confidence-power", type=float, default=2.0)
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
    if args.confidence_power <= 0:
        raise ValueError("confidence power must be positive")
    runtime_payload = torch.load(args.controller, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(runtime_payload, device=device).eval()
    bus = _load_bus(args.bus, device)
    estimator = _load_estimator(args.estimator, device)
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
    generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    corrupted_third = _erase(distractor.frames, args.third_erase, generator)
    learned, learned_conf, learned_query_conf = _accuracy(
        runtime,
        bus,
        estimator,
        first,
        second,
        corrupted_third,
        batch.correct_actions,
        True,
        args.confidence_power,
    )
    no_confidence, _, _ = _accuracy(
        runtime,
        bus,
        None,
        first,
        second,
        corrupted_third,
        batch.correct_actions,
        False,
        args.confidence_power,
    )
    n2, _, _ = _accuracy(
        runtime,
        bus,
        estimator,
        first,
        second,
        batch.frames,
        batch.correct_actions,
        True,
        args.confidence_power,
    )
    rows = {
        "n2_clean_with_learned_confidence": n2,
        "n3_corrupted_without_confidence": no_confidence,
        "n3_corrupted_with_learned_confidence": learned,
        "mean_third_confidence_all_trials": learned_conf,
        "mean_third_confidence_query_trials": learned_query_conf,
    }
    passed = bool(
        n2 >= 0.90
        and learned >= 0.85
        and learned - no_confidence >= 0.05
        and learned_conf <= 0.35
    )
    report = {
        "schema": "amodal-learned-confidence-audit-v1",
        "claim": "A learned generic quality head suppresses a corrupted N=3 stream.",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "bus": str(args.bus),
        "bus_sha256": _sha256(args.bus),
        "estimator": str(args.estimator),
        "estimator_sha256": _sha256(args.estimator),
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "third_erase": args.third_erase,
            "confidence_power": args.confidence_power,
            "device": str(device),
        },
        "rows": rows,
        "pass_bar": {
            "n2_min": 0.90,
            "learned_n3_min": 0.85,
            "learned_gain_min": 0.05,
            "mean_third_confidence_max": 0.35,
        },
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": rows, "passed": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
