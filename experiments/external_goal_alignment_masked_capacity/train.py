"""Pressure-test verifier-gated masked identity replacement under capacity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.external_goal_alignment_drift_missing_reversal import train as drift
from experiments.external_goal_alignment_identity import train as identity
from experiments.external_goal_representation_drift_gate import train as drift_data
from experiments.external_goal_representation_migration import train as migration
from experiments.external_one_pass_goal_evaluator import train as one_pass
from neural_computer import (
    AmodalCognitiveController,
    ExternalGoalEvaluatorStatistics,
    ExternalGoalRepresentationAlignmentBank,
    ExternalTransitionRouteMemory,
)

CAPACITY = 2
MAX_PROTOTYPES_PER_SLOT = 2
IDENTITY_MIN_SCORE = 0.55
IDENTITY_MIN_MARGIN = 0.04
MASKED_PROBE_MASK = torch.tensor(
    [False, True, True, False, True, True, True, True],
    dtype=torch.bool,
)


def _route_summary(result) -> dict[str, object]:
    return {
        "selected_slot_id": result.selected_slot_id,
        "eligible_slot_ids": list(result.eligible_slot_ids),
        "scores": [float(value) for value in result.scores.tolist()],
        "margin": result.margin,
        "reason": result.reason,
    }


def _digest(module: torch.nn.Module) -> str:
    return migration._digest(module)


def _probe(
    candidate: ExternalTransitionRouteMemory,
    *,
    affine_base: torch.Tensor,
    nonlinear_base: torch.Tensor,
    old_masked: torch.Tensor,
    new_masked: torch.Tensor,
    old_query_mask: torch.Tensor,
    new_query_mask: torch.Tensor,
) -> bool:
    checks = (
        (affine_base, None, 0),
        (nonlinear_base, None, 1),
        (old_masked, old_query_mask, 0),
        (new_masked, new_query_mask, 0),
    )
    for query, query_mask, expected_slot in checks:
        proposal = candidate.propose(
            query,
            (0, 1),
            minimum_score=IDENTITY_MIN_SCORE,
            query_mask=query_mask,
        )
        if proposal.selected_slot_id != expected_slot:
            return False
    return candidate.prototype_count(0) == MAX_PROTOTYPES_PER_SLOT


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
    evaluator_digest = evaluator.digest()
    model_digest = model.digest()

    affine, affine_source, affine_target, affine_base = identity._fit_affine(seed)
    nonlinear_source, nonlinear_target = drift_data._nonlinear_batch(seed)
    nonlinear, nonlinear_holdout_source, nonlinear_holdout_target, nonlinear_base = (
        identity._fit_nonlinear(nonlinear_source, nonlinear_target, seed)
    )
    affine_base = torch.nn.functional.normalize(affine_base, dim=0)
    nonlinear_base = torch.nn.functional.normalize(nonlinear_base, dim=0)
    affine_direction = drift._orthogonal_direction(
        affine_base,
        nonlinear_base,
        seed + 17,
    )
    old_masked = drift._drifted_signature(affine_base, affine_direction, 0.6)
    new_masked = drift._drifted_signature(affine_base, affine_direction, -0.6)
    old_query_mask = MASKED_PROBE_MASK.clone()
    new_query_mask = old_query_mask.clone()

    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=CAPACITY,
        identity_width=8,
        identity_min_score=IDENTITY_MIN_SCORE,
        identity_min_margin=IDENTITY_MIN_MARGIN,
        identity_max_prototypes_per_slot=MAX_PROTOTYPES_PER_SLOT,
        identity_merge_cosine=0.94,
    )
    affine_receipt = bank.admit_verified(
        "opaque-affine",
        affine,
        affine_source,
        affine_target,
        prediction_tolerance=identity.ALIGNMENT_TOLERANCE,
        identity_signature=affine_base,
    )
    nonlinear_receipt = bank.admit_verified(
        "opaque-nonlinear",
        nonlinear,
        nonlinear_holdout_source,
        nonlinear_holdout_target,
        prediction_tolerance=identity.ALIGNMENT_TOLERANCE,
        identity_signature=nonlinear_base,
    )

    reinforced = [
        bank.accept_identity_anchor(affine_base, verifier_accepted=True)
        for _ in range(2)
    ]
    old_masked_anchor = bank.accept_identity_anchor(
        old_masked,
        signature_mask=old_query_mask,
        verifier_accepted=True,
    )
    before_rejected = bank.digest()
    rejected = bank.accept_identity_anchor(
        new_masked,
        signature_mask=new_query_mask,
        verifier_accepted=True,
        retention_probe=lambda _candidate: False,
    )
    after_rejected = bank.digest()

    retention_probe = lambda candidate: _probe(
        candidate,
        affine_base=affine_base,
        nonlinear_base=nonlinear_base,
        old_masked=old_masked,
        new_masked=new_masked,
        old_query_mask=old_query_mask,
        new_query_mask=new_query_mask,
    )
    accepted = bank.accept_identity_anchor(
        new_masked,
        signature_mask=new_query_mask,
        verifier_accepted=True,
        retention_probe=retention_probe,
    )
    retained_affine = bank.route_by_signature(affine_base, affine_source[:1])
    retained_nonlinear = bank.route_by_signature(nonlinear_base, nonlinear_source[:1])
    retained_old = bank.route_by_signature(
        old_masked,
        affine_source[:1],
        signature_mask=old_query_mask,
    )
    routed_new = bank.route_by_signature(
        new_masked,
        affine_source[:1],
        signature_mask=new_query_mask,
    )
    affine_eval, affine_route = identity._evaluate_routed(
        bank,
        model,
        evaluator,
        migration._new_representation,
        seed + 301,
        query_expected_slot=affine_receipt.slot_id,
    )
    nonlinear_eval, nonlinear_route = identity._evaluate_routed(
        bank,
        model,
        evaluator,
        drift_data._nonlinear_representation,
        seed + 302,
        query_expected_slot=nonlinear_receipt.slot_id,
    )
    restored = ExternalGoalRepresentationAlignmentBank.from_payload(bank.state_payload())
    restored_new = restored.route_by_signature(
        new_masked,
        affine_source[:1],
        signature_mask=new_query_mask,
    )

    gates = {
        "both_frontends_admitted": affine_receipt.accepted and nonlinear_receipt.accepted,
        "core_identity_reinforced": all(receipt.accepted for receipt in reinforced),
        "old_masked_variant_appended": (
            old_masked_anchor.accepted
            and old_masked_anchor.anchor_update_stored
            and bank.identity_memory.masked_prototype_count == 1
        ),
        "rejected_replacement_preserves_state": (
            not rejected.accepted and before_rejected == after_rejected
        ),
        "accepted_replacement_passes_retention": accepted.accepted,
        "retained_core_affine_route": retained_affine.selected_slot_id == affine_receipt.slot_id,
        "retained_core_nonlinear_route": retained_nonlinear.selected_slot_id == nonlinear_receipt.slot_id,
        "retained_old_masked_route": retained_old.selected_slot_id == affine_receipt.slot_id,
        "new_masked_route": routed_new.selected_slot_id == affine_receipt.slot_id,
        "restored_new_masked_route": restored_new.selected_slot_id == affine_receipt.slot_id,
        "affine_mastery_after_replacement": affine_eval["mastery"] >= 0.95,
        "nonlinear_mastery_after_replacement": nonlinear_eval["mastery"] >= 0.95,
        "exact_final_persistence": restored.digest() == bank.digest(),
        "verifier_memory_unchanged": evaluator.digest() == evaluator_digest,
        "model_unchanged": model.digest() == model_digest,
        "controller_frozen": controller_digest == _digest(controller),
        "verifier_replay_zero": evaluator.sample_count.item() == outcome.shape[0],
        "alignment_replay_zero": all(
            adapter.sample_count.item() == affine_source.shape[0]
            for adapter in (affine, nonlinear)
        ),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-alignment-masked-capacity.v1",
        "claim_boundary": (
            "bounded verifier-gated masked-prototype replacement under fixed per-slot "
            "capacity; not autonomous retention policy, unbounded growth, semantic "
            "open-world identity, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "capacity": CAPACITY,
            "max_prototypes_per_slot": MAX_PROTOTYPES_PER_SLOT,
            "identity_min_score": IDENTITY_MIN_SCORE,
            "identity_min_margin": IDENTITY_MIN_MARGIN,
            "runtime_frontend_ids_used_for_updates": False,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "old_masked_anchor": old_masked_anchor.__dict__,
            "rejected_replacement": rejected.__dict__,
            "accepted_replacement": accepted.__dict__,
            "retained_affine_route": _route_summary(retained_affine),
            "retained_nonlinear_route": _route_summary(retained_nonlinear),
            "retained_old_masked_route": _route_summary(retained_old),
            "new_masked_route": _route_summary(routed_new),
            "restored_new_masked_route": _route_summary(restored_new),
            "affine_route": _route_summary(affine_route),
            "nonlinear_route": _route_summary(nonlinear_route),
            "affine_mastery": affine_eval,
            "nonlinear_mastery": nonlinear_eval,
            "masked_prototype_count": bank.identity_memory.masked_prototype_count,
            "restored_masked_prototype_count": restored.identity_memory.masked_prototype_count,
            "active_slots": list(bank.slot_ids),
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "unique_alignment_pairs": int(affine.sample_count.item() + nonlinear.sample_count.item()),
            "heldout_alignment_pairs": int(affine_source.shape[0] + nonlinear_holdout_source.shape[0]),
            "unique_identity_anchor_queries": 3,
            "identity_anchor_operations": 5,
            "replayed_examples": 0,
            "transition_rows_consumed_once": transition_rows,
            "verifier_statistics_updates": 1,
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
    parser.add_argument("--seed", type=int, default=85101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
