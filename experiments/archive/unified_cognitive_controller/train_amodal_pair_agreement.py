"""Learn task-agnostic pair agreement from same-frame view augmentations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .legacy_runtime import runtime_from_legacy_payload
from .environment import generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


class AmodalPairAgreement(nn.Module):
    """Symmetric opaque-latent compatibility score in [0, 1]."""

    def __init__(self, width: int, hidden: int = 32) -> None:
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError("agreement dimensions are invalid")
        self.width = width
        self.hidden = hidden
        self.network = nn.Sequential(
            nn.Linear(width * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        if first.shape != second.shape or first.ndim != 2 or first.shape[1] != self.width:
            raise ValueError("agreement payload shapes do not match")
        features = torch.cat([torch.abs(first - second), first * second], dim=-1)
        return self.network(features).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--agreement-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=175_001)
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
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
    device = torch.device(args.device)
    payload = torch.load(args.controller, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(payload, device=device).eval()
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    agreement = AmodalPairAgreement(runtime.controller.width, args.hidden).to(device)
    optimizer = torch.optim.Adam(agreement.parameters(), lr=args.learning_rate)
    start = time.perf_counter()
    curve = []
    for update in range(1, args.updates + 1):
        batch = generate_lifetimes(
            args.batch_size,
            6,
            seed=args.seed + update,
            heldout=True,
            task="pair_relation",
            appearance="bars",
            support_trials=1,
            device=device,
        )
        distractor = generate_lifetimes(
            args.batch_size,
            6,
            seed=args.seed + 100_000 + update,
            heldout=True,
            task="pair_relation",
            appearance="bars",
            support_trials=1,
            device=device,
        )
        first, second = split_complementary_views(batch.frames)
        with torch.no_grad():
            first_latent = runtime.encoder(first.reshape(-1, *first.shape[2:]))
            second_latent = runtime.encoder(second.reshape(-1, *second.shape[2:]))
            distractor_latent = runtime.encoder(
                distractor.frames.reshape(-1, *distractor.frames.shape[2:])
            )
        positive = agreement(first_latent, second_latent)
        negative_a = agreement(first_latent, distractor_latent)
        negative_b = agreement(second_latent, distractor_latent)
        positive_loss = torch.nn.functional.binary_cross_entropy(
            positive, torch.ones_like(positive)
        )
        negative_loss = torch.nn.functional.binary_cross_entropy(
            torch.cat([negative_a, negative_b]),
            torch.zeros(negative_a.shape[0] * 2, device=device),
        )
        loss = positive_loss + negative_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        curve.append(
            {
                "update": update,
                "loss": float(loss.detach()),
                "positive_mean": float(positive.detach().mean()),
                "negative_mean": float(
                    torch.cat([negative_a, negative_b]).detach().mean()
                ),
            }
        )
    args.agreement_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "amodal-pair-agreement-v1",
            "event_width": runtime.controller.width,
            "hidden": args.hidden,
            "state_dict": {
                name: value.detach().cpu() for name, value in agreement.state_dict().items()
            },
            "training": {
                "method": "same-frame-view-agreement",
                "controller_checkpoint": str(args.controller),
                "updates": args.updates,
                "batch_size": args.batch_size,
            },
        },
        args.agreement_out,
    )
    report = {
        "schema": "amodal-pair-agreement-training-v1",
        "labels_used": [],
        "positive_source": "two complementary views of the same rendered frame",
        "negative_source": "view paired with an independently rendered frame",
        "configuration": {
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "learning_rate": args.learning_rate,
            "device": str(device),
        },
        "curve": curve,
        "wall_seconds": time.perf_counter() - start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"final": curve[-1], "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    main()
