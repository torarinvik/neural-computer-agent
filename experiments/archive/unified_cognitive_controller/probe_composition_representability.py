"""Diagnostic: can this controller represent the fourth-primitive function?

DIAGNOSTIC ONLY -- NOT AN ADMISSIBLE RESULT. This probe trains on the
verifier's correct actions, including actions the controller never attempted.
That is exactly what the project forbids for any capability claim, so nothing
here may be reported as learned capability or promoted to a checkpoint.

Its only job is to separate two very different failure modes when a
reward-only rung sits at chance:

  representational  -- even with dense correct-action supervision the
                       architecture cannot compute the mapping, so the rung
                       needs a different structure;
  bootstrapping     -- supervision solves it easily, so the architecture is
                       fine and the reward-only objective, budget, or
                       curriculum is what fails.

Report which one it is and then fix the right thing.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn

from .environment import generate_lifetimes
from .legacy_model import UnifiedCognitiveController
from .train import rollout, seed_everything


def _supervised_loss(
        model: UnifiedCognitiveController, batch, *,
        feedback_trials: int) -> torch.Tensor:
    """Cross-entropy against verifier answers -- diagnostic use only."""
    result = rollout(
        model, batch, sample_actions=False, feedback_trials=feedback_trials)
    logits = result["logits"][:, feedback_trials:]
    targets = batch.correct_actions[:, feedback_trials:]
    return nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def _accuracy(
        model: UnifiedCognitiveController, *, task: str, count: int,
        seed: int, feedback_trials: int, device: torch.device,
        blank: bool = False) -> float:
    batch = generate_lifetimes(
        count, 6, seed=seed, heldout=True, task=task,
        support_trials=feedback_trials, device=device)
    if blank:
        batch = type(batch)(
            frames=torch.zeros_like(batch.frames),
            correct_actions=batch.correct_actions,
            stimulus_identities=batch.stimulus_identities,
            rule_bits=batch.rule_bits,
            seeds=batch.seeds,
            context_ids=batch.context_ids)
    result = rollout(
        model, batch, sample_actions=False, feedback_trials=feedback_trials)
    return float(result["rewards"][:, feedback_trials:].float().mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--task", default="contextual_composition")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feedback-trials", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--workspace-slots", type=int, default=8)
    parser.add_argument("--intention-width", type=int, default=24)
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device(args.device)
    model = UnifiedCognitiveController(
        width=args.width, workspace_slots=args.workspace_slots,
        intention_width=args.intention_width).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    started = time.perf_counter()
    history = []
    for update in range(1, args.steps + 1):
        model.train()
        batch = generate_lifetimes(
            args.batch_size, 6, seed=args.seed * 1_000_000 + update,
            task=args.task, support_trials=args.feedback_trials,
            device=device)
        loss = _supervised_loss(
            model, batch, feedback_trials=args.feedback_trials)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if update % max(1, args.steps // 10) == 0 or update == 1:
            history.append({"update": update, "loss": float(loss.detach())})

    model.eval()
    held_out = _accuracy(
        model, task=args.task, count=args.test_lifetimes,
        seed=args.seed + 70_000_000, feedback_trials=args.feedback_trials,
        device=device)
    blank_vision = _accuracy(
        model, task=args.task, count=args.test_lifetimes,
        seed=args.seed + 70_000_000, feedback_trials=args.feedback_trials,
        device=device, blank=True)
    report = {
        "schema": "composition-representability-probe-v1",
        "diagnostic_only": True,
        "semantic_labels_used_for_training": True,
        "unattempted_correct_actions_used_as_targets": True,
        "claim_boundary": (
            "Dense correct-action supervision. Measures representability "
            "only; it is not evidence of learned capability and must never "
            "be promoted or cited as a rung result."),
        "configuration": {**vars(args), "report": str(args.report)},
        "history": history,
        "held_out_query_accuracy": held_out,
        "blank_vision_query_accuracy": blank_vision,
        "representable_at_90": held_out >= 0.90,
        "vision_causally_used": blank_vision <= held_out - 0.15,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "task": args.task,
        "held_out_query_accuracy": held_out,
        "blank_vision_query_accuracy": blank_vision,
        "representable_at_90": report["representable_at_90"],
        "final_loss": history[-1]["loss"],
        "total_seconds": report["total_seconds"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
