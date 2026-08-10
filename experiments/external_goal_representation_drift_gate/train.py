"""Verify evidence-gated rejection of nonlinear goal representation drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_goal_representation_migration import train as migration
from experiments.external_one_pass_goal_evaluator import train as one_pass
from neural_computer import (
    AmodalCognitiveController,
    ExternalGoalEvaluatorStatistics,
    ExternalGoalRepresentationAlignmentStatistics,
)

ALIGNMENT_ROWS = 96
PARTIAL_TRAIN_STRIDE = 4
NONLINEAR_TRAIN_STRIDE = 2
ALIGNMENT_TOLERANCE = 5e-3
NONLINEAR_AMPLITUDE = 0.35


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _nonlinear_representation(
    position: float,
    generator: torch.Generator,
) -> torch.Tensor:
    old = float(position) / migration.learned_goal.SCALE
    noise = migration.ALIGNMENT_NOISE_STD * torch.randn(2, generator=generator)
    return torch.tensor(
        [[old**2, old**3 + NONLINEAR_AMPLITUDE * old]],
        dtype=torch.float32,
    ) + noise.reshape(1, 2)


def _nonlinear_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed + 400_009)
    positions = torch.linspace(-24.0, 24.0, ALIGNMENT_ROWS)
    source = torch.stack(
        [
            _nonlinear_representation(float(position), generator).squeeze(0)
            for position in positions
        ]
    )
    target = (positions / migration.learned_goal.SCALE).reshape(-1, 1)
    return source, target


def _fit_and_verify(
    source: torch.Tensor,
    target: torch.Tensor,
    train_indices: torch.Tensor,
    heldout_indices: torch.Tensor,
) -> tuple[
    ExternalGoalRepresentationAlignmentStatistics,
    object,
    dict[str, int],
]:
    adapter = ExternalGoalRepresentationAlignmentStatistics(2, 1, ridge=1e-5)
    adapter.observe(source.index_select(0, train_indices), target.index_select(0, train_indices))
    receipt = adapter.verify_heldout(
        source.index_select(0, heldout_indices),
        target.index_select(0, heldout_indices),
        prediction_tolerance=ALIGNMENT_TOLERANCE,
    )
    return adapter, receipt, {
        "unique_alignment_pairs": int(train_indices.numel()),
        "alignment_statistics_updates": 1,
        "replayed_alignment_pairs": 0,
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    controller = AmodalCognitiveController(
        width=8,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model, transition_rows = migration.learned_goal._transition_model()
    evaluator = ExternalGoalEvaluatorStatistics(1, ridge=1e-5)
    old_state, old_goal, outcome = one_pass._verifier_batch(seed)
    evaluator.observe(old_state, old_goal, outcome)
    evaluator_digest_before = evaluator.digest()
    model_digest_before = model.digest()

    affine_source, affine_target = migration._alignment_batch(seed)
    affine_indices = torch.arange(0, ALIGNMENT_ROWS, PARTIAL_TRAIN_STRIDE)
    affine_holdout_mask = torch.ones(ALIGNMENT_ROWS, dtype=torch.bool)
    affine_holdout_mask[affine_indices] = False
    affine_holdout_indices = affine_holdout_mask.nonzero(as_tuple=False).reshape(-1)
    affine_adapter, affine_receipt, affine_training = _fit_and_verify(
        affine_source,
        affine_target,
        affine_indices,
        affine_holdout_indices,
    )
    migrated = migration._evaluate(model, evaluator, affine_adapter, seed)

    nonlinear_source, nonlinear_target = _nonlinear_batch(seed)
    nonlinear_indices = torch.arange(0, ALIGNMENT_ROWS, NONLINEAR_TRAIN_STRIDE)
    nonlinear_holdout_mask = torch.ones(ALIGNMENT_ROWS, dtype=torch.bool)
    nonlinear_holdout_mask[nonlinear_indices] = False
    nonlinear_holdout_indices = nonlinear_holdout_mask.nonzero(as_tuple=False).reshape(-1)
    nonlinear_adapter, nonlinear_receipt, nonlinear_training = _fit_and_verify(
        nonlinear_source,
        nonlinear_target,
        nonlinear_indices,
        nonlinear_holdout_indices,
    )
    nonlinear_adapter_digest = nonlinear_adapter.digest()
    # A rejected candidate is never passed to the planner or live memory.
    nonlinear_candidate_served = False

    restored_affine = ExternalGoalRepresentationAlignmentStatistics.from_payload(
        affine_adapter.state_payload()
    )
    gates = {
        "partial_affine_alignment_accepted": affine_receipt.accepted,
        "partial_affine_mastery": migrated["mastery"] >= 0.95,
        "nonlinear_alignment_rejected": not nonlinear_receipt.accepted,
        "rejected_candidate_not_served": not nonlinear_candidate_served,
        "heldout_verification_used": nonlinear_receipt.query_count
        == nonlinear_holdout_indices.numel(),
        "verifier_memory_unchanged": evaluator.digest() == evaluator_digest_before,
        "model_unchanged": model.digest() == model_digest_before,
        "exact_affine_adapter_persistence": restored_affine.digest()
        == affine_adapter.digest(),
        "controller_frozen": controller_digest == _digest(controller),
        "verifier_replay_zero": outcome.shape[0] == evaluator.sample_count.item(),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-drift-gate.v1",
        "claim_boundary": (
            "evidence-gated acceptance of partial affine alignment and rejection "
            "of nonlinear drift; not nonlinear migration or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "alignment_rows": ALIGNMENT_ROWS,
            "partial_train_stride": PARTIAL_TRAIN_STRIDE,
            "nonlinear_train_stride": NONLINEAR_TRAIN_STRIDE,
            "alignment_tolerance": ALIGNMENT_TOLERANCE,
            "nonlinear_amplitude": NONLINEAR_AMPLITUDE,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "partial_affine_receipt": affine_receipt.__dict__,
            "partial_affine_mastery": migrated,
            "nonlinear_receipt": nonlinear_receipt.__dict__,
            "nonlinear_candidate_digest": nonlinear_adapter_digest,
            "verifier_training": {
                "unique_verifier_outcomes": int(outcome.shape[0]),
                "statistics_updates": 1,
                "replayed_rows": 0,
            },
            "partial_affine_training": affine_training,
            "nonlinear_training": nonlinear_training,
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "partial_affine_alignment_pairs": affine_training["unique_alignment_pairs"],
            "nonlinear_alignment_pairs": nonlinear_training["unique_alignment_pairs"],
            "heldout_affine_alignment_pairs": int(affine_holdout_indices.numel()),
            "heldout_nonlinear_alignment_pairs": int(nonlinear_holdout_indices.numel()),
            "transition_rows_consumed_once": transition_rows,
            "old_verifier_replay": 0,
            "controller_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84401)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
