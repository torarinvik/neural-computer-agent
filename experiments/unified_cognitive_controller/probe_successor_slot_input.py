"""Probe whether a new successor slot can decode the next action.

This is a verifier-side diagnostic only.  A frozen controller renders RGB
episodes and the probe receives the raw input to an appended slot's first
linear layer.  The throwaway linear/MLP heads see correct actions only for
diagnostic labels; their weights are discarded.  A lifetime-disjoint test and
shuffled-label null separate an absent representation from a reward/optimizer
failure without changing the controller.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .audit_sequence_multi_skill_bank import _model
from .train_pair_numerosity_transfer import _build_student
from .train_sequence_working_memory import generate_sequence_memory_batch


def _collect(
        model: nn.Module, *, count: int, span: int, distractors: int,
        seed: int, device: torch.device, slot: int,
        position_augmentation: bool,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(model, "skill_adapters") or len(model.skill_adapters) <= slot:
        raise ValueError("requested successor slot does not exist")
    batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation="mixed", position_augmentation=position_augmentation,
        device=device)
    captured: list[torch.Tensor] = []

    def capture(_module, inputs) -> None:
        captured.append(inputs[0].detach())

    handle = model.skill_adapters[slot][0].register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            state = model.initial_state(count, device=device)
            null = torch.full((count,), 2, dtype=torch.long, device=device)
            zeros = torch.zeros(count, device=device)
            for index in range(span):
                _, state = model.step(
                    batch.input_frames[:, index], state, null, zeros, zeros)
            for index in range(distractors):
                _, state = model.step(
                    batch.distractor_frames[:, index], state, null, zeros,
                    zeros)
            previous_action = null
            previous_reward = zeros
            for index in range(span):
                feedback = torch.full_like(previous_reward, float(index > 0))
                output, state = model.step(
                    batch.query_frames[:, index], state, previous_action,
                    previous_reward * feedback, feedback)
                previous_action = output.logits.argmax(dim=-1)
                previous_reward = (
                    previous_action == batch.correct_actions[:, index]
                ).float()
    finally:
        handle.remove()
    expected_calls = span + distractors + span
    if len(captured) != expected_calls:
        raise AssertionError(
            f"captured {len(captured)} calls, expected {expected_calls}")
    features = torch.stack(captured[-span:], dim=1)
    return features.reshape(count * span, -1), batch.correct_actions.reshape(-1)


def _fit(
        train_x: torch.Tensor, train_y: torch.Tensor,
        test_x: torch.Tensor, test_y: torch.Tensor, *, hidden: int,
        updates: int, seed: int,
        ) -> float:
    torch.manual_seed(seed)
    model: nn.Module = (
        nn.Linear(train_x.shape[-1], 2)
        if hidden == 0 else nn.Sequential(
            nn.Linear(train_x.shape[-1], hidden), nn.GELU(),
            nn.Linear(hidden, 2)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3,
                                  weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1000)
    for _ in range(updates):
        indices = torch.randint(
            train_x.shape[0], (min(256, train_x.shape[0]),),
            generator=generator)
        loss = nn.functional.cross_entropy(
            model(train_x[indices]), train_y[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float((model(test_x).argmax(-1) == test_y).float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=93601)
    parser.add_argument("--span", type=int, default=11)
    parser.add_argument("--slot-width", type=int, default=256)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--test-count", type=int, default=256)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--read-prior-slot", action="store_true")
    parser.add_argument("--position-augmentation", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if min(args.span, args.slot_width, args.train_count, args.test_count,
           args.updates) < 1:
        raise ValueError("span, counts, and updates must be positive")
    if args.train_count % 2 or args.test_count % 2:
        raise ValueError("train and test counts must be even")
    device = torch.device(args.device)
    payload = torch.load(args.parent, map_location=device, weights_only=False)
    model = _model(payload, device)
    overrides = None
    if args.read_prior_slot:
        overrides = {
            "skill_adapter_reads_prior": True,
            "skill_adapter_reads_prior_from": len(
                payload["model_configuration"].get("skill_adapter_widths", ())),
            "skill_adapter_prior_read_limit": 1,
        }
    student, configuration, slot, _ = _build_student(
        payload, device=device, slot_width=args.slot_width,
        continue_last_slot=False, configuration_overrides=overrides)
    student.eval()
    train_x, train_y = _collect(
        student, count=args.train_count, span=args.span,
        distractors=args.distractors, seed=args.seed + 10_000,
        device=device, slot=slot,
        position_augmentation=args.position_augmentation)
    test_x, test_y = _collect(
        student, count=args.test_count, span=args.span,
        distractors=args.distractors, seed=args.seed + 20_000,
        device=device, slot=slot,
        position_augmentation=args.position_augmentation)
    # Probe training uses CPU to avoid tying a diagnostic head to the runtime
    # device; the controller itself remains frozen throughout extraction.
    train_x, test_x = train_x.cpu().float(), test_x.cpu().float()
    train_y, test_y = train_y.cpu(), test_y.cpu()
    linear = _fit(
        train_x, train_y, test_x, test_y, hidden=0, updates=args.updates,
        seed=args.seed + 30_000)
    mlp = _fit(
        train_x, train_y, test_x, test_y, hidden=args.hidden,
        updates=args.updates, seed=args.seed + 40_000)
    torch.manual_seed(args.seed + 50_000)
    null_train_y = torch.randint(0, 2, train_y.shape)
    torch.manual_seed(args.seed + 60_000)
    null_test_y = torch.randint(0, 2, test_y.shape)
    shuffled = _fit(
        train_x, null_train_y, test_x, test_y, hidden=args.hidden,
        updates=args.updates, seed=args.seed + 70_000)
    independent_null = _fit(
        train_x, null_train_y, test_x, null_test_y, hidden=args.hidden,
        updates=args.updates, seed=args.seed + 80_000)
    report = {
        "schema": "successor-slot-input-probe-v1",
        "claim_boundary": (
            "The controller and appended slot are frozen. Throwaway probes "
            "receive only raw successor-slot input; correct actions are "
            "verifier-side diagnostic labels and are discarded."),
        "parent": str(args.parent),
        "seed": args.seed,
        "span": args.span,
        "slot": slot,
        "slot_width": args.slot_width,
        "read_prior_slot": args.read_prior_slot,
        "train_lifetimes": args.train_count,
        "test_lifetimes": args.test_count,
        "feature_dim": int(train_x.shape[-1]),
        "distractors": args.distractors,
        "position_augmentation": args.position_augmentation,
        "linear_accuracy": linear,
        "mlp_accuracy": mlp,
        "random_train_normal_test_accuracy": shuffled,
        "independent_random_accuracy": independent_null,
        "controller_frozen": True,
        "gates": {
            "mlp_above_chance": mlp > 0.60,
            "shuffled_near_chance": independent_null <= 0.60,
        },
    }
    report["accepted_probe"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "linear_accuracy", "mlp_accuracy",
            "random_train_normal_test_accuracy", "independent_random_accuracy",
            "accepted_probe")
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
