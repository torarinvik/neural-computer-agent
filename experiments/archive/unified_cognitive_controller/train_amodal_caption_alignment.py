"""Align synthetic grounded captions to the frozen amodal neural IR.

An external caption source describes only visible object properties (shape,
colour, and location). It never receives the lifetime rule, correct action, or
feedback. The caption frontend is trained only by paired consistency against
the frozen vision encoder. This is a grounded-language adapter diagnostic, not
a claim about a pretrained language model or natural-language reasoning.
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

from .legacy_runtime import (
    AmodalControllerRuntime,
    AmodalInputBus,
    AmodalOutputBus,
    runtime_from_legacy_payload,
)
from .environment import NULL_ACTION, generate_lifetimes
from .train_complementary_input_bus import split_complementary_views


CAPTION_LENGTH = 8
CAPTION_VOCAB = 64


def render_pair_relation_captions(
    frames: torch.Tensor,
    object_ids: torch.Tensor,
    *,
    vocabulary: int = CAPTION_VOCAB,
) -> torch.Tensor:
    """Emit a short caption from observable object descriptors.

    ``object_ids`` is the descriptor selected by the external caption source
    for the visible glyph; it is not a rule, action, reward, or task ID. Colour
    and location are measured from pixels. The output is a word-ID sequence,
    not a controller-side semantic shortcut.
    """
    if frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("frames must have shape [batch, 3, height, width]")
    if object_ids.ndim != 1 or object_ids.shape[0] != frames.shape[0]:
        raise ValueError("object_ids must match the frame batch")
    if vocabulary < 40:
        raise ValueError("caption vocabulary is too small")
    device = frames.device
    batch = frames.shape[0]
    max_value = frames.max(dim=1).values
    min_value = frames.min(dim=1).values
    visible = (max_value - min_value > 0.12) & (max_value > 0.15)
    mass = visible.float().sum((1, 2)).clamp_min(1.0)
    rgb = (frames * visible.unsqueeze(1)).sum((2, 3)) / mass.unsqueeze(1)
    colour = 10 + rgb.argmax(dim=-1)
    brightness = 20 + (rgb.mean(dim=-1) > 0.45).long()
    height, width = frames.shape[-2:]
    ys = torch.arange(height, device=device).view(1, height, 1)
    xs = torch.arange(width, device=device).view(1, 1, width)
    centre_y = (visible * ys).sum((1, 2)) / mass
    centre_x = (visible * xs).sum((1, 2)) / mass
    x_position = 30 + (centre_x > width / 2).long()
    y_position = 32 + (centre_y > height / 2).long()
    # Identity zero is the tall/vertical visible glyph; identity one is the
    # wide/horizontal glyph across the audited bar/diamond/dot renderers.
    shape = 2 + object_ids.long().clamp(0, 1)
    return torch.stack(
        (
            torch.ones(batch, dtype=torch.long, device=device),  # a
            torch.full((batch,), 4, dtype=torch.long, device=device),  # object
            shape,
            colour.clamp_max(vocabulary - 1),
            brightness,
            x_position,
            y_position,
            torch.full((batch,), 5, dtype=torch.long, device=device),  # end
        ),
        dim=1,
    )


class PairRelationCaptionEncoder(nn.Module):
    """Small word embedding frontend for the opaque event basis."""

    def __init__(
        self,
        event_width: int,
        *,
        vocabulary: int = CAPTION_VOCAB,
        embedding_width: int = 16,
    ) -> None:
        super().__init__()
        if vocabulary < 40 or embedding_width < 1:
            raise ValueError("caption encoder dimensions are invalid")
        self.vocabulary = vocabulary
        self.embedding_width = embedding_width
        self.embedding = nn.Embedding(vocabulary, embedding_width)
        self.head = nn.Sequential(
            nn.Linear(CAPTION_LENGTH * embedding_width, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, event_width),
        )

    def forward(self, captions: torch.Tensor) -> torch.Tensor:
        if captions.ndim != 2 or captions.shape[1] != CAPTION_LENGTH:
            raise ValueError("captions must have shape [batch, caption_length]")
        if captions.dtype != torch.long:
            raise ValueError("captions must contain integer word IDs")
        if torch.any(captions < 0) or torch.any(captions >= self.vocabulary):
            raise ValueError("caption word is outside the vocabulary")
        return self.head(self.embedding(captions).flatten(1))


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
) -> tuple[AmodalControllerRuntime, PairRelationCaptionEncoder]:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    extracted = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    captions = PairRelationCaptionEncoder(extracted.controller.width).to(device)
    runtime = AmodalControllerRuntime(
        extracted.controller,
        encoders={
            "stream_a": copy.deepcopy(extracted.encoder),
            "stream_b": captions,
        },
        input_bus=bus,
        output_bus=AmodalOutputBus({"action": extracted.decoder}),
    ).to(device).eval()
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    for parameter in captions.parameters():
        parameter.requires_grad_(True)
    return runtime, captions


@torch.no_grad()
def evaluate_caption_alignment(
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
    captions = render_pair_relation_captions(
        second.flatten(0, 1), normal.context_ids.flatten(0, 1)
    ).reshape(count, normal.trials, CAPTION_LENGTH)
    contradictory_captions = render_pair_relation_captions(
        contradictory_second.flatten(0, 1), contradictory.context_ids.flatten(0, 1)
    ).reshape(count, normal.trials, CAPTION_LENGTH)

    def run(streams: tuple[tuple[torch.Tensor, str], ...]) -> torch.Tensor:
        state = runtime.initial_state(count, device=device)
        action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
        reward = torch.zeros(count, device=device)
        actions = []
        for trial in range(normal.trials):
            feedback = torch.full_like(reward, float(trial == 1))
            named = {name: values[:, trial] for values, name in streams}
            output, state = runtime.step_streams(
                named, state, action, reward * feedback, feedback
            )
            action = output.decoded["action"].argmax(dim=-1)
            reward = (action == normal.correct_actions[:, trial]).float()
            actions.append(action)
        return torch.stack(actions, dim=1)

    caption_actions = run(((captions, "stream_b"),))
    fused_actions = run(((first, "stream_a"), (captions, "stream_b")))
    shuffled_actions = run(
        ((first, "stream_a"), (captions.roll(1, 0), "stream_b"))
    )
    contradictory_actions = run(
        ((first, "stream_a"), (contradictory_captions, "stream_b"))
    )
    full_actions = run(((normal.frames, "stream_a"),))
    query = slice(1, None)

    def accuracy(actions: torch.Tensor) -> float:
        return float(
            (actions[:, query] == normal.correct_actions[:, query])
            .float().mean()
        )

    return {
        "caption_only_accuracy": accuracy(caption_actions),
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
    parser.add_argument("--seed", type=int, default=999001)
    parser.add_argument("--updates", type=int, default=32)
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
    runtime, captions = _load_runtime(args.controller, args.input_bus, device)
    controller_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    optimizer = torch.optim.Adam(captions.parameters(), lr=args.learning_rate)
    appearances = ("bars", "diamonds", "dot_pairs")
    curve = []
    paired_frames = 0
    for update in range(args.updates + 1):
        if update:
            runtime.train()
            losses = []
            for appearance_index, appearance in enumerate(appearances):
                batch = generate_lifetimes(
                    args.batch_size, 6,
                    seed=args.seed + update * 100 + appearance_index,
                    heldout=True, task="pair_relation", appearance=appearance,
                    support_trials=1, device=device,
                )
                _, second = split_complementary_views(batch.frames)
                frames = second.flatten(0, 1)
                caption_ids = render_pair_relation_captions(
                    frames, batch.context_ids.flatten(0, 1)
                )
                with torch.no_grad():
                    target = runtime.encoders["stream_a"](frames)
                losses.append(F.mse_loss(captions(caption_ids), target))
                paired_frames += int(frames.shape[0])
            loss = torch.stack(losses).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(captions.parameters(), 5.0)
            optimizer.step()
            runtime.eval()
        else:
            loss = None
        curve.append({
            "update": update,
            "paired_frames": paired_frames,
            "loss": None if loss is None else float(loss.detach()),
            **evaluate_caption_alignment(
                runtime, count=args.eval_count, seed=args.seed,
                device=device, appearance="bars",
            ),
        })
    final_by_appearance = {
        appearance: evaluate_caption_alignment(
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
        "schema": "amodal-caption-basis-alignment-v1",
        "claim": (
            "A grounded caption frontend can align visible object descriptors "
            "into a frozen vision neural-IR basis using paired consistency."
        ),
        "training_signal": "paired_encoded_event_consistency",
        "caption_provenance": "external_visible_object_descriptor_source",
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
            "caption_length": CAPTION_LENGTH,
            "caption_vocabulary": CAPTION_VOCAB,
            "appearances": appearances,
            "trainable_components": ["caption_encoder"],
        },
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
            "schema": "amodal-caption-aligned-frontend-v1",
            "controller_sha256": _sha256(args.controller),
            "input_bus_sha256": _sha256(args.input_bus),
            "latent_width": runtime.controller.width,
            "frontend_kind": "pair_relation_grounded_caption",
            "caption_length": CAPTION_LENGTH,
            "caption_vocabulary": CAPTION_VOCAB,
            "embedding_width": captions.embedding_width,
            "caption_provenance": report["caption_provenance"],
            "state_dict": {
                name: value.detach().cpu().clone()
                for name, value in captions.state_dict().items()
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
