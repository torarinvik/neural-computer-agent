"""Tiny alignment diagnostics for a second encoder with a rotated latent basis.

The controller remains frozen. A copied external vision frontend emits a
deterministically permuted latent basis, and a trainable generic adapter can be
tested either with scalar verifier outcomes or with paired encoded-event
consistency. This is an alignment diagnostic, not a production checkpoint
promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .amodal_runtime import (
    AmodalControllerRuntime,
    AmodalInputBus,
    AmodalOutputBus,
    runtime_from_legacy_payload,
)
from .environment import NULL_ACTION, generate_lifetimes
from .train import attempted_success_loss
from .train_complementary_input_bus import split_complementary_views


class LatentBasisFrontend(nn.Module):
    """Frozen frontend plus trainable adapter after a fixed latent permutation."""

    def __init__(
        self,
        frontend: nn.Module,
        width: int,
        *,
        random_init: bool = False,
    ) -> None:
        super().__init__()
        self.frontend = frontend.eval()
        for parameter in self.frontend.parameters():
            parameter.requires_grad_(False)
        self.adapter = nn.Linear(width, width)
        if random_init:
            nn.init.normal_(self.adapter.weight, mean=0.0, std=width ** -0.5)
            nn.init.zeros_(self.adapter.bias)
        else:
            # The adapter starts as a no-op in its own coordinates.  It must
            # learn the inverse basis from reward; no target latent is exposed.
            with torch.no_grad():
                self.adapter.weight.copy_(torch.eye(width))
                self.adapter.bias.zero_()
        self.register_buffer("basis_permutation", torch.arange(width - 1, -1, -1))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        encoded = self.frontend(raw).detach()
        encoded = encoded.index_select(-1, self.basis_permutation)
        return self.adapter(encoded)


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
    *,
    random_init: bool,
    train_bus: bool,
) -> tuple[AmodalControllerRuntime, LatentBasisFrontend, AmodalInputBus]:
    controller_payload = torch.load(
        controller_path, map_location=device, weights_only=False
    )
    bus_payload = torch.load(bus_path, map_location=device, weights_only=False)
    extracted = runtime_from_legacy_payload(controller_payload, device=device).eval()
    bus = AmodalInputBus(
        int(bus_payload["event_width"]), int(bus_payload["residual_hidden"])
    ).to(device)
    bus.load_state_dict(bus_payload["state_dict"])
    adapter = LatentBasisFrontend(
        copy.deepcopy(extracted.encoder),
        extracted.controller.width,
        random_init=random_init,
    ).to(device)
    runtime = AmodalControllerRuntime(
        extracted.controller,
        encoders={
            "stream_a": copy.deepcopy(extracted.encoder),
            "stream_b": adapter,
        },
        input_bus=bus,
        output_bus=AmodalOutputBus({"action": extracted.decoder}),
    ).to(device)
    runtime.eval()
    for parameter in runtime.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.adapter.parameters():
        parameter.requires_grad_(True)
    if train_bus:
        for parameter in bus.parameters():
            parameter.requires_grad_(True)
    return runtime, adapter, bus


@torch.no_grad()
def evaluate_alignment(
    runtime: AmodalControllerRuntime,
    *,
    count: int,
    seed: int,
    device: torch.device,
    appearance: str = "bars",
) -> dict[str, float | bool]:
    """Evaluate composition and causal controls without updating weights."""
    normal = generate_lifetimes(
        count,
        6,
        seed=seed,
        heldout=True,
        task="pair_relation",
        appearance=appearance,
        support_trials=1,
        device=device,
    )
    contradictory = generate_lifetimes(
        count,
        6,
        seed=seed,
        heldout=True,
        task="pair_relation",
        appearance=appearance,
        support_trials=1,
        reverse_contexts=True,
        device=device,
    )
    first, second = split_complementary_views(normal.frames)
    _, contradictory_second = split_complementary_views(contradictory.frames)

    def run(streams: tuple[torch.Tensor, ...]) -> torch.Tensor:
        state = runtime.initial_state(count, device=device)
        action = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
        reward = torch.zeros(count, device=device)
        actions = []
        for trial in range(normal.trials):
            feedback = torch.full_like(reward, float(trial == 1))
            named = {
                name: stream[:, trial]
                for name, stream in zip(("stream_a", "stream_b"), streams)
            }
            output, state = runtime.step_streams(
                named, state, action, reward * feedback, feedback
            )
            action = output.decoded["action"].argmax(dim=-1)
            reward = (action == normal.correct_actions[:, trial]).float()
            actions.append(action)
        return torch.stack(actions, dim=1)

    first_actions = run((first,))
    second_actions = run((second,))
    fused_actions = run((first, second))
    shuffled_actions = run((first, second.roll(1, 0)))
    contradictory_actions = run((first, contradictory_second))
    full_actions = run((normal.frames,))
    query = slice(1, None)

    def accuracy(actions: torch.Tensor) -> float:
        return float(
            (actions[:, query] == normal.correct_actions[:, query]).float().mean()
        )

    return {
        "stream_a_accuracy": accuracy(first_actions),
        "stream_b_accuracy": accuracy(second_actions),
        "fused_accuracy": accuracy(fused_actions),
        "shuffled_partner_accuracy": accuracy(shuffled_actions),
        "contradictory_partner_accuracy": accuracy(contradictory_actions),
        "contradictory_prediction_flip_rate": float(
            (fused_actions[:, query] != contradictory_actions[:, query])
            .float()
            .mean()
        ),
        "full_n1_accuracy": accuracy(full_actions),
    }


def _train_update(
    runtime: AmodalControllerRuntime,
    batch,
    *,
    exploration: float,
) -> tuple[torch.Tensor, float]:
    first, second = split_complementary_views(batch.frames)
    state = runtime.initial_state(batch.batch_size, device=batch.frames.device)
    action = torch.full(
        (batch.batch_size,), NULL_ACTION, dtype=torch.long, device=batch.frames.device
    )
    reward = torch.zeros(batch.batch_size, device=batch.frames.device)
    losses = []
    successes = []
    for trial in range(batch.trials):
        feedback = torch.full_like(reward, float(trial == 1))
        output, state = runtime.step_streams(
            {"stream_a": first[:, trial], "stream_b": second[:, trial]},
            state,
            action,
            reward * feedback,
            feedback,
        )
        logits = output.decoded["action"]
        behavior = torch.softmax(logits, dim=-1)
        behavior = behavior * (1.0 - exploration) + exploration / logits.shape[-1]
        action = torch.multinomial(behavior, 1).squeeze(1)
        reward = (action == batch.correct_actions[:, trial]).to(logits.dtype)
        losses.append(attempted_success_loss(logits, action, reward))
        successes.append(float(reward.mean()))
    return torch.stack(losses).mean(), sum(successes) / len(successes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--input-bus", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--adapter-out",
        type=Path,
        help="optional independently loadable adapter artifact",
    )
    parser.add_argument("--seed", type=int, default=285_001)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--exploration", type=float, default=0.25)
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument(
        "--self-supervised",
        action="store_true",
        help="align paired raw frames with no verifier outcomes or task labels",
    )
    parser.add_argument(
        "--calibration-appearances",
        nargs="+",
        choices=("bars", "diamonds", "dot_pairs"),
        default=["bars", "diamonds", "dot_pairs"],
    )
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
    if args.updates < 1 or args.batch_size < 4 or args.eval_count < 64:
        raise ValueError("updates, batch-size, and eval-count are too small")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    runtime, adapter, bus = _load_runtime(
        args.controller,
        args.input_bus,
        device,
        random_init=args.random_init,
        train_bus=not args.adapter_only and not args.self_supervised,
    )
    controller_before = {
        name: value.detach().cpu().clone()
        for name, value in runtime.controller.state_dict().items()
    }
    trainable = list(adapter.adapter.parameters())
    if not args.adapter_only and not args.self_supervised:
        trainable += list(bus.parameters())
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    initial = evaluate_alignment(
        runtime,
        count=args.eval_count,
        seed=args.seed + 50_000,
        device=device,
    )
    initial_by_appearance = {
        appearance: evaluate_alignment(
            runtime,
            count=args.eval_count,
            seed=args.seed + 50_000 + index * 10_000,
            device=device,
            appearance=appearance,
        )
        for index, appearance in enumerate(("bars", "diamonds", "dot_pairs"))
    }
    curve = [{"update": 0, "verifier_bits": 0, **initial}]
    alignment_bits = 0
    for update in range(1, args.updates + 1):
        runtime.train()
        if args.self_supervised:
            calibration_losses = []
            for appearance_index, appearance in enumerate(
                args.calibration_appearances
            ):
                batch = generate_lifetimes(
                    args.batch_size,
                    6,
                    seed=args.seed + update * 100 + appearance_index,
                    task="pair_relation",
                    appearance=appearance,
                    support_trials=1,
                    device=device,
                )
                frames = batch.frames.reshape(
                    -1, *batch.frames.shape[2:]
                )
                with torch.no_grad():
                    target = runtime.encoders["stream_a"](frames)
                prediction = runtime.encoders["stream_b"](frames)
                calibration_losses.append(
                    torch.nn.functional.mse_loss(prediction, target)
                )
            loss = torch.stack(calibration_losses).mean()
            train_accuracy = None
            alignment_bits += (
                args.batch_size * 6 * len(args.calibration_appearances)
            )
        else:
            batch = generate_lifetimes(
                args.batch_size,
                6,
                seed=args.seed + update,
                task="pair_relation",
                appearance="bars",
                support_trials=1,
                device=device,
            )
            loss, train_accuracy = _train_update(
                runtime, batch, exploration=args.exploration
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        optimizer.step()
        runtime.eval()
        metrics = evaluate_alignment(
            runtime,
            count=args.eval_count,
            seed=args.seed + 50_000,
            device=device,
        )
        curve.append(
            {
                "update": update,
                "verifier_bits": (
                    0
                    if args.self_supervised
                    else update * args.batch_size * 6
                ),
                "alignment_frames": alignment_bits,
                "loss": float(loss.detach()),
                "train_accuracy": train_accuracy,
                **metrics,
            }
        )
    final_by_appearance = {
        appearance: evaluate_alignment(
            runtime,
            count=args.eval_count * 2,
            seed=args.seed + 60_000 + index * 10_000,
            device=device,
            appearance=appearance,
        )
        for index, appearance in enumerate(("bars", "diamonds", "dot_pairs"))
    }
    final = final_by_appearance["bars"]
    controller_unchanged = all(
        torch.equal(value, runtime.controller.state_dict()[name].detach().cpu())
        for name, value in controller_before.items()
    )
    report = {
        "schema": "amodal-latent-basis-alignment-v1",
        "claim": (
            "A frozen controller can use an externally aligned second encoder "
            "whose latent basis was permuted, using paired sensory consistency."
            if args.self_supervised
            else
            "A frozen controller can use an externally aligned second encoder "
            "whose latent basis was permuted, using only attempted actions and "
            "scalar outcomes."
        ),
        "training_signal": (
            "paired_encoded_event_consistency"
            if args.self_supervised
            else "attempted_action_scalar_outcome"
        ),
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
            "exploration": args.exploration,
            "device": str(device),
            "random_init": args.random_init,
            "adapter_only": args.adapter_only,
            "self_supervised": args.self_supervised,
            "calibration_appearances": args.calibration_appearances,
            "trainable_components": [
                "second_encoder_adapter"
            ] if args.adapter_only or args.self_supervised else [
                "second_encoder_adapter", "input_bus"
            ],
        },
        "initial": initial,
        "initial_by_appearance": initial_by_appearance,
        "curve": curve,
        "final": final,
        "final_by_appearance": final_by_appearance,
        "alignment_frames": alignment_bits,
        "controller_parameters_unchanged": controller_unchanged,
        "passed": bool(
            controller_unchanged
            and all(
                row["fused_accuracy"] >= threshold
                and row["stream_a_accuracy"] <= 0.65
                and row["stream_b_accuracy"] <= 0.65
                and row["shuffled_partner_accuracy"] <= 0.60
                and row["contradictory_partner_accuracy"] <= 0.25
                and row["contradictory_prediction_flip_rate"] >= flip_threshold
                and row["full_n1_accuracy"] >= 0.95
                and row["full_n1_accuracy"]
                >= initial_by_appearance[appearance]["full_n1_accuracy"] - 0.02
                for appearance, row, threshold, flip_threshold in zip(
                    ("bars", "diamonds", "dot_pairs"),
                    final_by_appearance.values(),
                    (0.90, 0.85, 0.90),
                    (0.80, 0.70, 0.80),
                )
            )
        )
    }
    if args.adapter_out is not None:
        args.adapter_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema": "amodal-latent-basis-adapter-v1",
                "controller_sha256": _sha256(args.controller),
                "input_bus_sha256": _sha256(args.input_bus),
                "latent_width": runtime.controller.width,
                "frontend_kind": "vision_reverse_basis",
                "basis_permutation": adapter.basis_permutation.detach().cpu(),
                "adapter_state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in adapter.adapter.state_dict().items()
                },
                "training_signal": report["training_signal"],
                "source_report": str(args.report),
                "passed": report["passed"],
            },
            args.adapter_out,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "final": final}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
