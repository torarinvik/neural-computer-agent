"""Learn a generic event-quality confidence from latent corruption consistency."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .amodal_runtime import runtime_from_legacy_payload
from .environment import generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


class AmodalConfidenceEstimator(nn.Module):
    """Opaque event latent to generic quality confidence in [0, 1]."""

    def __init__(self, width: int, hidden: int = 32) -> None:
        super().__init__()
        if width < 1 or hidden < 1:
            raise ValueError("confidence estimator dimensions are invalid")
        self.width = width
        self.hidden = hidden
        self.network = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, payload: torch.Tensor) -> torch.Tensor:
        if payload.ndim != 2 or payload.shape[1] != self.width:
            raise ValueError("payload shape does not match confidence estimator")
        return self.network(payload).squeeze(-1)


def _erase(
    frames: torch.Tensor, fraction: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    background = frames[..., :1, :1]
    mask = torch.rand(
        frames.shape[0],
        frames.shape[1],
        1,
        frames.shape[-2],
        frames.shape[-1],
        generator=generator,
        device=frames.device,
    ) < fraction.reshape(-1, 1, 1, 1, 1)
    return torch.where(mask, background, frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--estimator-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=173_001)
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--quality-scale", type=float, default=8.0)
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
    if args.quality_scale <= 0:
        raise ValueError("quality scale must be positive")
    device = torch.device(args.device)
    payload = torch.load(args.controller, map_location=device, weights_only=False)
    runtime = runtime_from_legacy_payload(payload, device=device).eval()
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    estimator = AmodalConfidenceEstimator(runtime.controller.width, args.hidden).to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=args.learning_rate)
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
        source_frames = torch.cat([batch.frames, first, second], dim=0)
        source_count = source_frames.shape[0]
        fractions = torch.rand(source_count, generator=generator, device=device) * 0.9
        corrupted = _erase(source_frames, fractions, generator)
        with torch.no_grad():
            clean_latent = runtime.encoder(source_frames.reshape(-1, *source_frames.shape[2:]))
            noisy_latent = runtime.encoder(corrupted.reshape(-1, *corrupted.shape[2:]))
            latent_mse = (clean_latent - noisy_latent).square().mean(dim=1)
            target_noisy = torch.exp(-args.quality_scale * latent_mse).reshape(
                source_count, batch.trials
            )
            clean_latent = clean_latent.reshape(
                source_count * batch.trials, -1
            )
            noisy_latent = noisy_latent.reshape(
                source_count * batch.trials, -1
            )
        predicted_noisy = estimator(noisy_latent).reshape(source_count, batch.trials)
        predicted_clean = estimator(clean_latent).reshape(source_count, batch.trials)
        loss = torch.nn.functional.mse_loss(predicted_noisy, target_noisy) + torch.nn.functional.mse_loss(
            predicted_clean, torch.ones_like(predicted_clean)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        curve.append({"update": update, "loss": float(loss.detach())})
    args.estimator_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "amodal-confidence-estimator-v1",
            "event_width": runtime.controller.width,
            "hidden": args.hidden,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in estimator.state_dict().items()
            },
            "training": {
                "method": "clean-corrupted-latent-consistency",
                "controller_checkpoint": str(args.controller),
                "quality_scale": args.quality_scale,
                "updates": args.updates,
                "batch_size": args.batch_size,
            },
        },
        args.estimator_out,
    )
    report = {
        "schema": "amodal-confidence-estimator-training-v1",
        "labels_used": [],
        "configuration": {
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "quality_scale": args.quality_scale,
            "learning_rate": args.learning_rate,
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
