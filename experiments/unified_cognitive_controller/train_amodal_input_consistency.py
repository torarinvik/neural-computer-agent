"""Train an input bus to preserve clean latent evidence under pixel corruption.

The target is a frozen copy of the promoted bus on the same uncorrupted event
collection. No task labels, action labels, or verifier outcomes are used. This
is a disposable self-distillation diagnostic for corruption invariance; the
student is promoted only after the independent behavioral noise audit passes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .amodal_runtime import AmodalInputBus, runtime_from_legacy_payload
from .environment import generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


def _load_bus(path: Path, device: torch.device) -> AmodalInputBus:
    payload = torch.load(path, map_location=device, weights_only=False)
    bus = AmodalInputBus(
        int(payload["event_width"]), int(payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(payload["state_dict"])
    return bus


def _erase(
    frames: torch.Tensor, fraction: float, generator: torch.Generator
) -> torch.Tensor:
    background = frames[..., :1, :1]
    mask = (
        torch.rand(
            frames.shape[0],
            frames.shape[1],
            1,
            frames.shape[-2],
            frames.shape[-1],
            generator=generator,
            device=frames.device,
        )
        < fraction
    )
    return torch.where(mask, background, frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--baseline-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bus-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=163_001)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--erase-low", type=float, default=0.2)
    parser.add_argument("--erase-high", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--clean-weight", type=float, default=4.0)
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
    if args.updates < 1 or args.batch_size < 4 or args.batch_size % 4:
        raise ValueError("updates and batch size are invalid")
    if not 0.0 <= args.erase_low <= args.erase_high <= 1.0:
        raise ValueError("erase range must satisfy 0 <= low <= high <= 1")
    if args.clean_weight < 0:
        raise ValueError("clean weight must be nonnegative")

    device = torch.device(args.device)
    controller_payload = torch.load(
        args.controller, map_location=device, weights_only=False
    )
    runtime = runtime_from_legacy_payload(controller_payload, device=device).eval()
    teacher = _load_bus(args.baseline_bus, device)
    student = _load_bus(args.baseline_bus, device)
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    start = time.perf_counter()
    curve = []

    for update in range(1, args.updates + 1):
        batch = generate_lifetimes(
            args.batch_size,
            6,
            seed=args.seed + update,
            task="pair_relation",
            appearance="bars",
            support_trials=1,
            device=device,
        )
        first, second = split_complementary_views(batch.frames)
        fraction = float(
            torch.empty((), device=device).uniform_(
                args.erase_low, args.erase_high, generator=generator
            )
        )
        corrupted_second = _erase(second, fraction, generator)
        losses = []
        for trial in range(batch.trials):
            with torch.no_grad():
                clean_events = [
                    runtime.encode(first[:, trial]), runtime.encode(second[:, trial])
                ]
                target = teacher(clean_events).payload
                corrupted_events = [
                    runtime.encode(first[:, trial]),
                    runtime.encode(corrupted_second[:, trial]),
                ]
            prediction = student(corrupted_events).payload
            corrupted_loss = torch.nn.functional.smooth_l1_loss(prediction, target)
            clean_prediction = student(clean_events).payload
            clean_loss = torch.nn.functional.smooth_l1_loss(clean_prediction, target)
            losses.append(corrupted_loss + args.clean_weight * clean_loss)
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        curve.append(
            {
                "update": update,
                "erase_fraction": fraction,
                "loss": float(loss.detach()),
            }
        )

    args.bus_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "amodal-input-bus-v1",
            "event_width": runtime.controller.width,
            "residual_hidden": student.residual[0].out_features
            if student.residual is not None
            else 0,
            "state_dict": {
                name: value.detach().cpu() for name, value in student.state_dict().items()
            },
            "training": {
                "method": "clean-latent-consistency-distillation",
                "controller_checkpoint": str(args.controller),
                "baseline_bus": str(args.baseline_bus),
                "erase_range": [args.erase_low, args.erase_high],
                "updates": args.updates,
                "batch_size": args.batch_size,
            },
        },
        args.bus_out,
    )
    report = {
        "schema": "amodal-input-consistency-training-v1",
        "learner_visible": ["encoded event payloads", "synthetic pixel corruption"],
        "labels_used": [],
        "controller": str(args.controller),
        "baseline_bus": str(args.baseline_bus),
        "bus_out": str(args.bus_out),
        "configuration": {
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "erase_range": [args.erase_low, args.erase_high],
                "learning_rate": args.learning_rate,
                "clean_weight": args.clean_weight,
            "device": str(device),
        },
        "curve": curve,
        "final_loss": curve[-1]["loss"],
        "wall_seconds": time.perf_counter() - start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"final_loss": report["final_loss"], "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
