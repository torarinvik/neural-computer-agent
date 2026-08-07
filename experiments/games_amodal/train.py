"""Composition smoke for the game verifiers behind the amodal event bus.

This runs a random policy only.  It exists to prove the wiring: the frontend
encodes raw observation grids into validated opaque events, the verifier keeps
all game state private, and per-step scalar outcomes are the only feedback.
It makes no learning or retention claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from experiments.games_amodal.environments import PongVerifier, SnakeVerifier
from neural_computer import AmodalEvent


class GridEventEncoder(nn.Module):
    """Caller-owned frontend mapping observation grids to opaque events."""

    def __init__(self, *, channels: int, height: int, width: int, event_width: int) -> None:
        super().__init__()
        self.project = nn.Linear(channels * height * width, event_width)
        self.event_width = event_width

    def forward(self, observation: torch.Tensor) -> AmodalEvent:
        payload = torch.tanh(self.project(observation.flatten(start_dim=1)))
        return AmodalEvent(payload=payload).validate(width=self.event_width)


class ConvGridEventEncoder(nn.Module):
    """Screen frontend with translation-equivariant features.

    The linear frontend maps every pixel through its own weight, so a
    pattern learned at one avatar position says nothing about the same
    pattern elsewhere. F22 measured the cost: motor games that need the
    agent to approach an object sat at 0.03-0.13 of ceiling regardless of
    budget, while fixed-geometry decision games reached 1.00. Convolution
    shares weights across positions, which is the structural version of
    the egocentric-roll fix -- and unlike the roll it survives walls and
    boundaries, which a toroidal shift distorts.
    """

    def __init__(
        self,
        *,
        channels: int,
        height: int,
        width: int,
        event_width: int,
        hidden: int = 16,
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.project = nn.Linear(hidden * height * width, event_width)
        self.event_width = event_width

    def forward(self, observation: torch.Tensor) -> AmodalEvent:
        features = self.features(observation)
        payload = torch.tanh(self.project(features.flatten(start_dim=1)))
        return AmodalEvent(payload=payload).validate(width=self.event_width)


def _rollout(
    verifier: SnakeVerifier | PongVerifier,
    encoder: GridEventEncoder,
    *,
    steps: int,
    seed: int,
) -> dict[str, float]:
    verifier.reset(seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    total_reward = 0.0
    events = 0
    for _ in range(steps):
        with torch.no_grad():
            event = encoder(verifier.observation())
        events += event.payload.shape[0]
        actions = torch.randint(
            0, verifier.action_count, (verifier.batch_size,), generator=generator
        )
        outcome = verifier.step(actions)
        total_reward += float(outcome.reward.sum())
        if not bool(outcome.alive.any()):
            break
    return {"total_reward": total_reward, "events_encoded": float(events)}


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.manual_seed(args.seed)
    report: dict[str, object] = {"seed": args.seed, "steps": args.steps}
    for name, verifier in (
        (
            "snake",
            SnakeVerifier(batch_size=args.batch_size, seed=args.seed),
        ),
        (
            "pong",
            PongVerifier(batch_size=args.batch_size, seed=args.seed),
        ),
    ):
        encoder = GridEventEncoder(
            channels=verifier.observation_channels,
            height=verifier.height,
            width=verifier.width,
            event_width=args.event_width,
        )
        report[name] = _rollout(verifier, encoder, steps=args.steps, seed=args.seed)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--event-width", type=int, default=32)
    parser.add_argument("--report-out", type=Path, default=None)
    args = parser.parse_args()
    report = run(args)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
