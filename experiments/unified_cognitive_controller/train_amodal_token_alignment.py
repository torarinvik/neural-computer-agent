"""Align an independent token sensor to the frozen amodal neural-IR.

The token stream is made by a fixed, modality-specific sensor: a view is
pooled into a small raster of RGB patch tokens. The trainable frontend sees
only those sensor readings. It is aligned to the existing vision encoder by
paired encoded-event consistency; no task labels, verifier bits, action
labels, or semantic token names are used. The controller, original vision
encoder, input bus, and decoder stay frozen.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from .amodal_runtime import (
    AmodalControllerRuntime,
    AmodalInputBus,
    AmodalOutputBus,
    runtime_from_legacy_payload,
)
from .environment import NULL_ACTION, generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


TOKEN_GRID = 16
TOKEN_LEVELS = 8
TOKEN_VOCAB = TOKEN_LEVELS**3


def render_pair_relation_tokens(
    frames: torch.Tensor,
    *,
    grid: int = TOKEN_GRID,
    levels: int = TOKEN_LEVELS,
) -> torch.Tensor:
    """Convert ordinary RGB pixels into a fixed raster token stream.

    This is a sensor operation, not a task shortcut.  Token identity encodes
    only local RGB measurements and sequence position is the fixed raster
    order. In particular, no stimulus identity, relation, action, or verifier
    field is appended to the tokens.
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must have shape [batch, 3, height, width]")
    if grid < 1 or levels < 2:
        raise ValueError("grid and levels are invalid")
    pooled = F.adaptive_avg_pool2d(frames, (grid, grid)).clamp(0.0, 1.0)
    # Keep the sensor reading itself as the token payload.  ``levels`` is
    # retained in the API for a discrete variant, but the promoted stream uses
    # continuous RGB measurements so quantization cannot erase thin contours.
    return pooled.permute(0, 2, 3, 1).reshape(-1, grid * grid, 3)


class PairRelationTokenEncoder(nn.Module):
    """Small learned projection from raw patch tokens into the opaque event IR."""

    def __init__(
        self,
        event_width: int,
        *,
        grid: int = TOKEN_GRID,
        levels: int = TOKEN_LEVELS,
        embedding_width: int = 8,
    ) -> None:
        super().__init__()
        if grid < 1 or levels < 2 or embedding_width < 1:
            raise ValueError("token encoder dimensions are invalid")
        self.grid = grid
        self.levels = levels
        self.vocab = levels**3
        self.embedding_width = embedding_width
        self.head = nn.Sequential(
            # Keep the full fixed sensor grid available to the projection,
            # matching the information budget of the successful spectral
            # audio frontend while retaining token boundaries at the API.
            nn.Linear(grid * grid * 3, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, event_width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (self.grid * self.grid, 3):
            raise ValueError("tokens must have shape [batch, grid*grid, 3]")
        if not torch.is_floating_point(tokens):
            raise ValueError("continuous token readings must be floating point")
        return self.head(tokens.flatten(1))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_runtime(
    controller_path: Path,
    bus_path: Path,
    device: torch.device,
) -> tuple[AmodalControllerRuntime, PairRelationTokenEncoder]:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    extracted = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    tokens = PairRelationTokenEncoder(extracted.controller.width).to(device)
    runtime = AmodalControllerRuntime(
        extracted.controller,
        encoders={
            "stream_a": copy.deepcopy(extracted.encoder),
            "stream_b": tokens,
        },
        input_bus=bus,
        output_bus=AmodalOutputBus({"action": extracted.decoder}),
    ).to(device).eval()
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    for parameter in tokens.parameters():
        parameter.requires_grad_(True)
    return runtime, tokens


@torch.no_grad()
def evaluate_token_alignment(
    runtime: AmodalControllerRuntime,
    *,
    count: int,
    seed: int,
    device: torch.device,
    appearance: str,
) -> dict[str, float]:
    normal = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task="pair_relation",
        appearance=appearance, support_trials=1, device=device,
    )
    contradictory = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task="pair_relation",
        appearance=appearance, support_trials=1, reverse_contexts=True,
        device=device,
    )
    first, second = split_complementary_views(normal.frames)
    _, contradictory_second = split_complementary_views(contradictory.frames)
    token_second = render_pair_relation_tokens(second.flatten(0, 1)).reshape(
        count, normal.trials, TOKEN_GRID * TOKEN_GRID, 3
    )
    contradictory_tokens = render_pair_relation_tokens(
        contradictory_second.flatten(0, 1)
    ).reshape(count, normal.trials, TOKEN_GRID * TOKEN_GRID, 3)

    def run(
        streams: tuple[tuple[torch.Tensor, str], ...],
    ) -> torch.Tensor:
        state = runtime.initial_state(count, device=device)
        action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
        reward = torch.zeros(count, device=device)
        actions = []
        for trial in range(normal.trials):
            feedback = torch.full_like(reward, float(trial == 1))
            named = {
                name: values[:, trial] for values, name in streams
            }
            output, state = runtime.step_streams(
                named, state, action, reward * feedback, feedback
            )
            action = output.decoded["action"].argmax(dim=-1)
            reward = (action == normal.correct_actions[:, trial]).float()
            actions.append(action)
        return torch.stack(actions, dim=1)

    first_actions = run(((first, "stream_a"),))
    token_actions = run(((token_second, "stream_b"),))
    fused_actions = run(((first, "stream_a"), (token_second, "stream_b")))
    shuffled_actions = run(
        ((first, "stream_a"), (token_second.roll(1, 0), "stream_b"))
    )
    contradictory_actions = run(
        ((first, "stream_a"), (contradictory_tokens, "stream_b"))
    )
    full_actions = run(((normal.frames, "stream_a"),))
    query = slice(1, None)

    def accuracy(actions: torch.Tensor) -> float:
        return float(
            (actions[:, query] == normal.correct_actions[:, query])
            .float().mean()
        )

    return {
        "stream_a_accuracy": accuracy(first_actions),
        "stream_b_accuracy": accuracy(token_actions),
        "fused_accuracy": accuracy(fused_actions),
        "shuffled_partner_accuracy": accuracy(shuffled_actions),
        "contradictory_partner_accuracy": accuracy(contradictory_actions),
        "contradictory_prediction_flip_rate": float(
            (fused_actions[:, query] != contradictory_actions[:, query])
            .float().mean()
        ),
        "full_n1_accuracy": accuracy(full_actions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--adapter-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=996001)
    parser.add_argument("--updates", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--device",
        default=(
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    )
    args = parser.parse_args()
    if args.updates < 1 or args.batch_size < 4 or args.eval_count < 64:
        raise ValueError("updates, batch-size, and eval-count are too small")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    runtime, tokens = _load_runtime(args.controller, args.input_bus, device)
    controller_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    optimizer = torch.optim.Adam(tokens.parameters(), lr=args.learning_rate)
    appearances = ("bars", "diamonds", "dot_pairs")
    initial_by_appearance = {
        appearance: evaluate_token_alignment(
            runtime, count=args.eval_count, seed=args.seed + index * 10_000,
            device=device, appearance=appearance,
        )
        for index, appearance in enumerate(appearances)
    }
    curve: list[dict[str, object]] = []
    paired_frames = 0
    for update in range(0, args.updates + 1):
        if update:
            runtime.train()
            losses = []
            for appearance_index, appearance in enumerate(appearances):
                batch = generate_lifetimes(
                    args.batch_size, 6, seed=args.seed + update * 100 + appearance_index,
                    heldout=True, task="pair_relation", appearance=appearance,
                    support_trials=1,
                    device=device,
                )
                first, second = split_complementary_views(batch.frames)
                frames = torch.cat((first.flatten(0, 1), second.flatten(0, 1)))
                tokens_input = render_pair_relation_tokens(frames)
                with torch.no_grad():
                    target = runtime.encoders["stream_a"](frames)
                losses.append(F.mse_loss(tokens(tokens_input), target))
                paired_frames += int(frames.shape[0])
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tokens.parameters(), 5.0)
            optimizer.step()
            runtime.eval()
        else:
            loss = None
        curve.append({
            "update": update,
            "paired_frames": paired_frames,
            "loss": None if loss is None else float(loss.detach()),
            **evaluate_token_alignment(
                runtime, count=args.eval_count, seed=args.seed,
                device=device, appearance="bars",
            ),
        })

    final_by_appearance = {
        appearance: evaluate_token_alignment(
            runtime, count=args.eval_count * 2,
            seed=args.seed + 60_000 + index * 10_000,
            device=device, appearance=appearance,
        )
        for index, appearance in enumerate(appearances)
    }
    controller_unchanged = all(
        torch.equal(value, runtime.controller.state_dict()[name].detach().cpu())
        for name, value in controller_before.items()
    )
    passed = bool(
        controller_unchanged and all(
            row["fused_accuracy"] >= threshold
            and row["stream_a_accuracy"] <= 0.65
            and row["stream_b_accuracy"] <= 0.65
            and row["shuffled_partner_accuracy"] <= 0.60
            and row["contradictory_partner_accuracy"] <= 0.25
            and row["contradictory_prediction_flip_rate"] >= flip_threshold
            and row["full_n1_accuracy"] >= 0.95
            for row, threshold, flip_threshold in zip(
                final_by_appearance.values(), (0.90, 0.85, 0.90),
                (0.80, 0.70, 0.80),
            )
        )
    )
    report = {
        "schema": "amodal-token-basis-alignment-v1",
        "claim": (
            "A fixed RGB token sensor can align an independent encoder into a "
            "frozen vision neural-IR basis using paired sensory consistency."
        ),
        "training_signal": "paired_encoded_event_consistency",
        "controller": str(args.controller),
        "controller_sha256": _sha256(args.controller),
        "input_bus": str(args.input_bus),
        "input_bus_sha256": _sha256(args.input_bus),
        "configuration": {
            "seed": args.seed,
            "updates": args.updates,
            "batch_size": args.batch_size,
            "eval_count": args.eval_count,
            "learning_rate": args.learning_rate,
            "device": str(device),
            "token_grid": TOKEN_GRID,
            "token_levels": TOKEN_LEVELS,
            "token_vocab": TOKEN_VOCAB,
            "paired_views_per_update": 2,
            "appearances": appearances,
            "trainable_components": ["token_encoder"],
        },
        "initial_by_appearance": initial_by_appearance,
        "curve": curve,
        "final_by_appearance": final_by_appearance,
        "paired_frames": paired_frames,
        "controller_parameters_unchanged": controller_unchanged,
        "passed": passed,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.adapter_out.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "schema": "amodal-token-aligned-frontend-v1",
            "controller_sha256": _sha256(args.controller),
            "input_bus_sha256": _sha256(args.input_bus),
            "latent_width": runtime.controller.width,
            "frontend_kind": "pair_relation_rgb_tokens",
            "token_grid": TOKEN_GRID,
            "token_levels": TOKEN_LEVELS,
            "token_vocab": TOKEN_VOCAB,
            "embedding_width": tokens.embedding_width,
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in tokens.state_dict().items()
            },
            "training_signal": report["training_signal"],
            "source_report": str(args.report),
            "passed": passed,
        },
        args.adapter_out,
    )
    print(json.dumps({"passed": passed, "final": final_by_appearance}, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
