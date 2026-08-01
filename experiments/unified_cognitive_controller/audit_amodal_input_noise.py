"""Compare outcome-only noise adaptation with the clean input bus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .amodal_interface import AmodalEvent
from .amodal_runtime import AmodalInputBus, runtime_from_legacy_payload
from .environment import NULL_ACTION, generate_lifetimes
from .train_amodal_event_denoiser import AmodalEventDenoiser
from .train_complementary_input_bus import split_complementary_views


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def _accuracy(runtime, bus, first, second, labels, denoiser=None) -> float:
    count = first.shape[0]
    state = runtime.initial_state(count, device=first.device)
    action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=first.device)
    reward = torch.zeros(count, device=first.device)
    actions = []
    for trial in range(first.shape[1]):
        has_feedback = torch.full_like(reward, float(trial == 1))
        events = [runtime.encode(first[:, trial]), runtime.encode(second[:, trial])]
        if denoiser is not None:
            events = [
                AmodalEvent(payload=denoiser(event.payload)).validate(
                    width=runtime.controller.width
                )
                for event in events
            ]
        event = bus(events)
        core, state = runtime.step_intention_event(
            event, state, action, reward * has_feedback, has_feedback
        )
        logits = runtime.decode(core.intent_event)
        action = logits.argmax(dim=-1)
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


def _load_denoiser(path: Path, device: torch.device) -> AmodalEventDenoiser:
    payload = torch.load(path, map_location=device, weights_only=False)
    denoiser = AmodalEventDenoiser(
        int(payload["event_width"]), int(payload["hidden"])
    ).to(device)
    denoiser.load_state_dict(payload["state_dict"])
    return denoiser.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--baseline-bus", type=Path, required=True)
    parser.add_argument("--adapted-bus", type=Path, required=True)
    parser.add_argument("--denoiser", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=161_001)
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
    runtime = runtime_from_legacy_payload(controller_payload, device=device).eval()
    baseline = _load_bus(args.baseline_bus, device)
    adapted = _load_bus(args.adapted_bus, device)
    denoiser = _load_denoiser(args.denoiser, device) if args.denoiser else None
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
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    levels = (0.0, 0.2, 0.4, 0.6, 0.8)
    rows = []
    for level in levels:
        background = second[..., :1, :1]
        erase = (
            torch.rand(
                second.shape[0],
                second.shape[1],
                1,
                second.shape[-2],
                second.shape[-1],
                generator=generator,
                device=device,
            )
            < level
        )
        corrupted = torch.where(erase, background, second)
        rows.append(
            {
                "erase_fraction": level,
                "baseline_accuracy": _accuracy(
                    runtime, baseline, first, corrupted, batch.correct_actions
                ),
                "adapted_accuracy": _accuracy(
                    runtime,
                    adapted,
                    first,
                    corrupted,
                    batch.correct_actions,
                    denoiser,
                ),
            }
        )
    gains = [row["adapted_accuracy"] - row["baseline_accuracy"] for row in rows]
    clean_retained = rows[0]["adapted_accuracy"] >= 0.90
    noisy_gain = max(gains[1:])
    high_noise_gain = gains[-2]
    passed = bool(clean_retained and noisy_gain >= 0.05 and high_noise_gain >= 0.05)
    report = {
        "schema": "amodal-input-noise-adaptation-audit-v1",
        "claim": (
            "Outcome-only adaptation improves a frozen amodal input bus under "
            "known pixel corruption while retaining clean behavior."
        ),
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "baseline_bus": str(args.baseline_bus),
        "baseline_bus_sha256": _sha256(args.baseline_bus),
        "adapted_bus": str(args.adapted_bus),
        "adapted_bus_sha256": _sha256(args.adapted_bus),
        "denoiser": str(args.denoiser) if args.denoiser else None,
        "configuration": {
            "seed": args.seed,
            "count": args.count,
            "device": str(device),
        },
        "curve": rows,
        "gains": gains,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "curve": rows}, sort_keys=True))


if __name__ == "__main__":
    main()
