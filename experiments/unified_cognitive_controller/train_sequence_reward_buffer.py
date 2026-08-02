"""Train a generic action adapter from a latent replay buffer.

The buffer contains only the controller-visible recurrent/event latent, the
opaque action that was attempted, and the scalar outcome for that attempt.
Verifier-private correct actions are used only inside the environment while
the buffer is collected.  The inherited controller is frozen; only a
zero-initialized generic action adapter is plastic.  This isolates whether
offline reuse of sparse outcomes improves sample efficiency without changing
the representation or adding semantic labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .environment import ACTIONS, NULL_ACTION
from .model import UnifiedCognitiveController
from .train import attempted_success_loss, seed_everything
from .train_sequence_working_memory import (
    evaluate_sequence_memory, generate_sequence_memory_batch)


def _collect_buffer(
        model: UnifiedCognitiveController, *, count: int, span: int,
        distractors: int, seed: int, device: torch.device,
        position_augmentation: bool,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect latent transitions using uniformly random opaque actions."""
    batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation="mixed", position_augmentation=position_augmentation,
        device=device)
    null = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    state = model.initial_state(count, device=device)
    previous_action = null
    previous_reward = zeros
    features: list[torch.Tensor] = []
    base_logits: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    outcomes: list[torch.Tensor] = []
    with torch.no_grad():
        for index in range(span):
            _, state = model.step(
                batch.input_frames[:, index], state, null, zeros, zeros)
        for index in range(distractors):
            _, state = model.step(
                batch.distractor_frames[:, index], state, null, zeros, zeros)
        for query in range(span):
            has_feedback = torch.full_like(
                previous_reward, float(query > 0))
            frame = batch.query_frames[:, query]
            event = model.vision(frame)
            hidden_before = state.hidden.clone()
            output, state = model.step(
                frame, state, previous_action,
                previous_reward * has_feedback, has_feedback)
            action = torch.randint(ACTIONS, (count,), device=device)
            outcome = (
                action == batch.correct_actions[:, query]).to(torch.float32)
            features.append(torch.cat([hidden_before, event], dim=-1))
            base_logits.append(output.logits.detach())
            actions.append(action)
            outcomes.append(outcome)
            previous_action = action
            previous_reward = outcome
    return (
        torch.cat(features), torch.cat(base_logits), torch.cat(actions),
        torch.cat(outcomes))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=30941)
    parser.add_argument("--train-lifetimes", type=int, default=2048)
    parser.add_argument("--rehearse-spans", default="")
    parser.add_argument("--rehearsal-lifetimes", type=int, default=512)
    parser.add_argument("--span", type=int, default=8)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--action-adapter-width", type=int, default=64)
    parser.add_argument("--test-episodes", type=int, default=512)
    parser.add_argument("--shuffle-outcomes", action="store_true")
    parser.add_argument("--position-augmentation", action="store_true")
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if (
            args.train_lifetimes < 2
            or args.train_lifetimes % 2
            or args.rehearsal_lifetimes < 2
            or args.rehearsal_lifetimes % 2
            or args.span < 1
            or args.distractors < 0
            or args.epochs < 1
            or args.batch_size < 1
            or args.action_adapter_width < 1
            or args.test_episodes < 2):
        raise ValueError("invalid replay-buffer dimensions")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.parent, map_location=device, weights_only=False)
    base_configuration = dict(payload["model_configuration"])
    base = UnifiedCognitiveController(**base_configuration).to(device)
    base.load_state_dict(payload["state_dict"])
    base.eval()
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    features, base_logits, actions, outcomes = _collect_buffer(
        base, count=args.train_lifetimes, span=args.span,
        distractors=args.distractors, seed=args.seed, device=device,
        position_augmentation=args.position_augmentation)
    rehearsal_spans = tuple(
        int(value) for value in args.rehearse_spans.split(",") if value)
    if any(value < 1 for value in rehearsal_spans):
        raise ValueError("rehearsal spans must be positive")
    replay_buffers = []
    for index, rehearsal_span in enumerate(rehearsal_spans):
        replay_buffers.append(_collect_buffer(
            base, count=args.rehearsal_lifetimes, span=rehearsal_span,
            distractors=args.distractors,
            seed=args.seed + (index + 1) * 1_000_003,
            device=device, position_augmentation=args.position_augmentation))
    if replay_buffers:
        features = torch.cat([features] + [value[0] for value in replay_buffers])
        base_logits = torch.cat(
            [base_logits] + [value[1] for value in replay_buffers])
        actions = torch.cat([actions] + [value[2] for value in replay_buffers])
        outcomes = torch.cat(
            [outcomes] + [value[3] for value in replay_buffers])
    rehearsal_lifetime_count = (
        args.rehearsal_lifetimes * len(rehearsal_spans))
    if args.shuffle_outcomes:
        generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        permutation = torch.randperm(
            outcomes.shape[0], generator=generator).to(device)
        outcomes = outcomes[permutation]
    configuration = dict(
        base_configuration,
        action_adapter_width=args.action_adapter_width,
        action_adapter_gated=False)
    student = UnifiedCognitiveController(**configuration).to(device)
    student.load_state_dict(payload["state_dict"], strict=False)
    assert student.action_adapter is not None
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    for parameter in student.action_adapter.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        student.action_adapter.parameters(), lr=args.learning_rate,
        weight_decay=1e-5)
    for _ in range(args.epochs):
        permutation = torch.randperm(features.shape[0], device=device)
        for start in range(0, features.shape[0], args.batch_size):
            indices = permutation[start:start + args.batch_size]
            logits = base_logits[indices] + student.action_adapter(
                features[indices])
            loss = attempted_success_loss(
                logits, actions[indices], outcomes[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    student.eval()
    audit = evaluate_sequence_memory(
        student, count=args.test_episodes, span=args.span,
        distractors=args.distractors, seed=args.seed + 90_000,
        operation="mixed", device=device)
    report = {
        "schema": "span8-reward-buffer-readout-v1",
        "claim_boundary": (
            "The learner sees only RGB streams, opaque attempted actions, "
            "and scalar attempted-action outcomes. No correct unattempted "
            "action or semantic task label is stored in the buffer."),
        "parent": str(args.parent),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "unique_train_lifetimes": args.train_lifetimes,
        "unique_rehearsal_lifetimes": rehearsal_lifetime_count,
        "total_unique_lifetimes": args.train_lifetimes + rehearsal_lifetime_count,
        "query_transition_count": int(features.shape[0]),
        "rehearse_spans": rehearsal_spans,
        "rehearsal_lifetimes_per_stream": args.rehearsal_lifetimes,
        "buffer_outcomes_shuffled": args.shuffle_outcomes,
        "audit": audit,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "unified-cognitive-controller-v1",
        "model_configuration": configuration,
        "state_dict": student.state_dict(),
        "source_report": str(args.report),
        "admission_status": "diagnostic_reward_buffer_candidate",
    }, args.checkpoint_out)
    print(json.dumps({
        "accuracy": audit["accuracy"],
        "reverse_flip": audit[
            "reverse_operation_prediction_flip_rate_nonpalindrome"],
        "blank": audit["blank_sequence_accuracy"],
        "reset": audit["all_memory_reset_accuracy"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
