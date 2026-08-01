"""Audit pair-agreement confidence for valid N=3 distractor rejection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .amodal_interface import AmodalEvent
from .amodal_runtime import AmodalInputBus, runtime_from_legacy_payload
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
    agreement = AmodalPairAgreement(int(payload["event_width"]), int(payload["hidden"])).to(device)
    agreement.load_state_dict(payload["state_dict"])
    return agreement.eval()


@torch.no_grad()
def _accuracy(runtime, bus, agreement, streams, labels, use_agreement) -> tuple[float, float, float]:
    count = labels.shape[0]
    state = runtime.initial_state(count, device=labels.device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=labels.device)
    reward = torch.zeros(count, device=labels.device)
    actions = []
    useful_confidences = []
    distractor_confidences = []
    for trial in range(labels.shape[1]):
        feedback = torch.full_like(reward, float(trial == 1))
        encoded = [runtime.encode(stream[:, trial]) for stream in streams]
        if use_agreement:
            pair_scores = torch.stack(
                [
                    agreement(encoded[0].payload, encoded[1].payload),
                    agreement(encoded[0].payload, encoded[2].payload),
                    agreement(encoded[1].payload, encoded[2].payload),
                ],
                dim=1,
            )
            confidences = torch.stack(
                [pair_scores[:, 0] + pair_scores[:, 1],
                 pair_scores[:, 0] + pair_scores[:, 2],
                 pair_scores[:, 1] + pair_scores[:, 2]],
                dim=1,
            ) / 2.0
            useful_confidences.append(confidences[:, :2])
            distractor_confidences.append(confidences[:, 2])
            events = [
                AmodalEvent(event.payload, confidence=confidences[:, index])
                for index, event in enumerate(encoded)
            ]
        else:
            events = encoded
        core, state = runtime.step_intention_event(
            bus(events), state, action, reward * feedback, feedback
        )
        action = runtime.decode(core.intent_event).argmax(dim=-1)
        reward = (action == labels[:, trial]).float()
        actions.append(action)
    accuracy = float((torch.stack(actions, dim=1)[:, 1:] == labels[:, 1:]).float().mean())
    if use_agreement:
        useful = float(torch.cat(useful_confidences, dim=0).mean())
        distractor = float(torch.cat(distractor_confidences, dim=0).mean())
    else:
        useful = distractor = 1.0
    return accuracy, useful, distractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--bus", type=Path, required=True)
    parser.add_argument("--agreement", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=176_001)
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
    no_agreement = _accuracy(
        runtime,
        bus,
        agreement,
        (first, second, distractor.frames),
        batch.correct_actions,
        False,
    )
    learned = _accuracy(
        runtime,
        bus,
        agreement,
        (first, second, distractor.frames),
        batch.correct_actions,
        True,
    )
    n2 = _accuracy(
        runtime,
        bus,
        agreement,
        (first, second, batch.frames),
        batch.correct_actions,
        True,
    )
    passed = bool(
        n2[0] >= 0.90
        and learned[0] >= 0.85
        and learned[0] - no_agreement[0] >= 0.25
        and learned[1] - learned[2] >= 0.20
    )
    rows = {
        "n2_with_agreement": n2[0],
        "n3_without_agreement": no_agreement[0],
        "n3_with_agreement": learned[0],
        "mean_useful_confidence": learned[1],
        "mean_distractor_confidence": learned[2],
        "confidence_separation": learned[1] - learned[2],
    }
    report = {
        "schema": "amodal-pair-agreement-audit-v1",
        "claim": "Self-supervised pair agreement rejects a valid N=3 distractor.",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "bus": str(args.bus),
        "bus_sha256": _sha256(args.bus),
        "agreement": str(args.agreement),
        "agreement_sha256": _sha256(args.agreement),
        "configuration": {"seed": args.seed, "count": args.count, "device": str(device)},
        "rows": rows,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rows": rows, "passed": passed}, sort_keys=True))


if __name__ == "__main__":
    main()
