"""Self-distill N=3 set composition from the promoted N=2 bus output."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .legacy_runtime import AmodalInputBus, runtime_from_legacy_payload
from .environment import generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


def _load_bus(path: Path, device: torch.device) -> AmodalInputBus:
    payload = torch.load(path, map_location=device, weights_only=False)
    bus = AmodalInputBus(
        int(payload["event_width"]), int(payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(payload["state_dict"])
    return bus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--baseline-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bus-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=167_001)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
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
    if args.clean_weight < 0:
        raise ValueError("clean weight must be nonnegative")
    device = torch.device(args.device)
    payload = torch.load(args.controller, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(payload, device=device).eval()
    teacher = _load_bus(args.baseline_bus, device)
    student = _load_bus(args.baseline_bus, device)
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
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
        distractor = generate_lifetimes(
            args.batch_size,
            6,
            seed=args.seed + 100_000 + update,
            task="pair_relation",
            appearance="bars",
            support_trials=1,
            device=device,
        )
        first, second = split_complementary_views(batch.frames)
        losses = []
        for trial in range(batch.trials):
            with torch.no_grad():
                first_event = runtime.encode(first[:, trial])
                second_event = runtime.encode(second[:, trial])
                distractor_event = runtime.encode(distractor.frames[:, trial])
                target = teacher([first_event, second_event]).payload
            n3_prediction = student([first_event, second_event, distractor_event]).payload
            n2_prediction = student([first_event, second_event]).payload
            losses.append(
                torch.nn.functional.smooth_l1_loss(n3_prediction, target)
                + args.clean_weight
                * torch.nn.functional.smooth_l1_loss(n2_prediction, target)
            )
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        curve.append({"update": update, "loss": float(loss.detach())})
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
                "method": "n3-to-n2-clean-latent-consistency",
                "controller_checkpoint": str(args.controller),
                "baseline_bus": str(args.baseline_bus),
                "updates": args.updates,
                "batch_size": args.batch_size,
                "clean_weight": args.clean_weight,
            },
        },
        args.bus_out,
    )
    report = {
        "schema": "amodal-input-n3-consistency-training-v1",
        "labels_used": [],
        "configuration": {
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "clean_weight": args.clean_weight,
            "device": str(device),
        },
        "curve": curve,
        "wall_seconds": time.perf_counter() - start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"final_loss": curve[-1]["loss"], "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()

