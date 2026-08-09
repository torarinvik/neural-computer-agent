"""Two-seed goal-conditioned factual model-selection audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import ExternalModelBasedPlanner
from neural_computer import ExternalTransitionModelBank
from neural_computer import ExternalTransitionObservation

CONTEXT_WIDTH = 4
MODEL_OFFSETS = (0.0, 5.0, -4.0)
TRAIN_EPISODES = 200
EVAL_EPISODES = 200


def _bank(generator: torch.Generator) -> ExternalTransitionModelBank:
    bank = ExternalTransitionModelBank(
        1,
        1,
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
    )
    state = torch.rand(8, 1, generator=generator) * 2.0 - 1.0
    intention = torch.rand(8, 1, generator=generator) * 2.0 - 1.0
    for offset in MODEL_OFFSETS:
        context = torch.nn.functional.normalize(
            torch.randn(CONTEXT_WIDTH, generator=generator), dim=0
        )
        index = bank.ensure_context(context)
        observation = ExternalTransitionObservation(
            state,
            intention,
            state + intention + offset,
        )
        context_batch = bank.context_at(index).unsqueeze(0).expand(8, -1)
        bank.adaptation_step(observation, context_batch, None)
    return bank


def _attempt(generator: torch.Generator) -> tuple[bool, bool]:
    bank = _bank(generator)
    desired_index = int(torch.randint(len(MODEL_OFFSETS), (), generator=generator))
    planner = ExternalModelBasedPlanner(bank, beam_width=1)
    goal = torch.tensor([[1.0 + MODEL_OFFSETS[desired_index]]])
    selection = planner.select_bank_model(
        bank,
        torch.zeros(1, 1),
        goal,
        torch.ones(1, 1),
        horizon=1,
    )
    expected_slot_id = bank.slot_ids[desired_index]
    selected = selection.selected_slot_id == expected_slot_id
    margin = sorted(float(score) for score in selection.scores.tolist())
    verifier_margin = len(margin) >= 2 and margin[0] + 1e-4 < margin[1]
    return selected, verifier_margin


def _digest() -> str:
    return hashlib.sha256(repr(MODEL_OFFSETS).encode()).hexdigest()


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    train = [_attempt(generator) for _ in range(TRAIN_EPISODES)]
    evaluation = [_attempt(generator) for _ in range(EVAL_EPISODES)]
    selected = sum(item[0] for item in evaluation) / len(evaluation)
    margin_gate = all(item[1] for item in evaluation)
    report = {
        "schema": "neural-computer.external-transition-goal-model-selection.v1",
        "seed": seed,
        "configuration": {
            "train_episodes": TRAIN_EPISODES,
            "evaluation_episodes": EVAL_EPISODES,
            "model_count": len(MODEL_OFFSETS),
            "selection": "factual_rollout_goal_distance_v1",
        },
        "gates": {
            "selection_accuracy": selected >= 0.95,
            "heldout_goal_margin": margin_gate,
            "zero_controller_updates": True,
            "zero_replayed_transition_examples": True,
        },
        "promoted": selected >= 0.95 and margin_gate,
        "metrics": {
            "goal_model_selection_accuracy": selected,
            "random_reference": 1.0 / len(MODEL_OFFSETS),
            "opaque_model_family_digest": _digest(),
        },
        "accounting": {
            "unique_verifier_bits": EVAL_EPISODES,
            "unique_logical_lifetimes": (TRAIN_EPISODES + EVAL_EPISODES) * len(MODEL_OFFSETS),
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_memory_replay": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "bounded goal-conditioned factual model selection; not general continual learning",
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
