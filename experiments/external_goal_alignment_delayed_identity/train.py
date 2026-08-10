"""Pressure-test delayed resolution of overlapping goal-frontend identity."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.external_goal_alignment_identity import train as identity
from experiments.external_goal_representation_drift_gate import train as drift
from experiments.external_goal_representation_migration import train as migration
from experiments.external_one_pass_goal_evaluator import train as one_pass
from neural_computer import (
    AmodalCognitiveController,
    ExternalGoalEvaluatorStatistics,
    ExternalGoalRepresentationAlignmentBank,
)

CAPACITY = 2
IDENTITY_QUARANTINE_CAPACITY = 2
ALIGNMENT_TOLERANCE = identity.ALIGNMENT_TOLERANCE


def _route_summary(result) -> dict[str, object]:
    return {
        "selected_slot_id": result.selected_slot_id,
        "eligible_slot_ids": list(result.eligible_slot_ids),
        "scores": [float(value) for value in result.scores.tolist()],
        "margin": result.margin,
        "reason": result.reason,
    }


def _anchor_passive(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 910_007)
    positions = torch.linspace(48.0, 72.0, 32)
    return torch.stack(
        [drift._nonlinear_representation(float(position), generator).squeeze(0) for position in positions]
    )


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
    controller_digest = migration._digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model, transition_rows = migration.learned_goal._transition_model()
    evaluator = ExternalGoalEvaluatorStatistics(1, ridge=1e-5)
    old_state, old_goal, outcome = one_pass._verifier_batch(seed)
    evaluator.observe(old_state, old_goal, outcome)
    evaluator_digest = evaluator.digest()
    model_digest = model.digest()

    affine, affine_source, affine_target, affine_signature = identity._fit_affine(seed)
    nonlinear_source, nonlinear_target = drift._nonlinear_batch(seed)
    nonlinear, nonlinear_source_holdout, nonlinear_target_holdout, nonlinear_signature = (
        identity._fit_nonlinear(nonlinear_source, nonlinear_target, seed)
    )
    shared_signature = torch.nn.functional.normalize(
        affine_signature + nonlinear_signature,
        dim=0,
    )
    anchor_passive = _anchor_passive(seed + 20)
    anchor_signature = identity._signature(anchor_passive)

    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=CAPACITY,
        identity_width=identity.IDENTITY_WIDTH,
        identity_min_score=identity.IDENTITY_MIN_SCORE,
        identity_min_margin=identity.IDENTITY_MIN_MARGIN,
        identity_quarantine_capacity=IDENTITY_QUARANTINE_CAPACITY,
    )
    nonlinear_receipt = bank.admit_verified(
        "opaque-nonlinear",
        nonlinear,
        nonlinear_source_holdout,
        nonlinear_target_holdout,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
        identity_signature=shared_signature,
    )
    affine_receipt = bank.admit_verified(
        "opaque-affine",
        affine,
        affine_source,
        affine_target,
        prediction_tolerance=ALIGNMENT_TOLERANCE,
        identity_signature=shared_signature,
    )

    before_ambiguous_route = bank.digest()
    ambiguous = bank.route_by_signature(shared_signature, affine_source[:1])
    after_ambiguous_route = bank.digest()
    deferred_one = bank.defer_identity_signature(
        shared_signature,
        candidate_slot_ids=(nonlinear_receipt.slot_id, affine_receipt.slot_id),
    )
    deferred_two = bank.defer_identity_signature(
        shared_signature,
        candidate_slot_ids=(nonlinear_receipt.slot_id, affine_receipt.slot_id),
    )
    overflow = bank.defer_identity_signature(
        shared_signature,
        candidate_slot_ids=(nonlinear_receipt.slot_id, affine_receipt.slot_id),
    )
    blocked_eviction = bank.evict_verified(
        nonlinear_receipt.slot_id,
        lambda _candidate: True,
    )
    before_rejected_resolution = bank.digest()
    rejected_resolution = bank.resolve_identity_quarantine(
        nonlinear_receipt.slot_id,
        verifier_accepted=False,
    )
    after_rejected_resolution = bank.digest()
    persisted_deferred = ExternalGoalRepresentationAlignmentBank.from_payload(
        bank.state_payload()
    )
    deferred_digest = bank.digest()
    anchor_update = bank.observe_identity_verified(
        nonlinear_receipt.slot_id,
        anchor_signature,
    )
    anchor_route = bank.route_by_signature(anchor_signature, anchor_passive[:1])
    accepted_resolution = bank.resolve_identity_quarantine(
        nonlinear_receipt.slot_id,
        verifier_accepted=True,
    )
    evicted = bank.evict_verified(
        affine_receipt.slot_id,
        lambda candidate: candidate.active_count == 1,
    )
    mastery, routed = identity._evaluate_routed(
        bank,
        model,
        evaluator,
        drift._nonlinear_representation,
        seed + 20,
        query_expected_slot=nonlinear_receipt.slot_id,
    )
    restored = ExternalGoalRepresentationAlignmentBank.from_payload(bank.state_payload())
    restored_anchor_route = restored.route_by_signature(
        anchor_signature,
        anchor_passive[:1],
    )

    gates = {
        "both_overlapping_frontends_admitted": (
            nonlinear_receipt.accepted and affine_receipt.accepted
        ),
        "overlap_route_refused": ambiguous.selected_slot_id is None,
        "overlap_route_read_only": after_ambiguous_route == before_ambiguous_route,
        "first_deferred_signature_retained": deferred_one.accepted,
        "second_deferred_signature_retained": deferred_two.accepted,
        "quarantine_overflow_refused": not overflow.accepted,
        "eviction_blocked_by_deferred_reference": not blocked_eviction.accepted,
        "verifier_rejection_preserves_deferred_state": (
            not rejected_resolution.accepted
            and before_rejected_resolution == after_rejected_resolution
        ),
        "deferred_state_persists_exactly": (
            persisted_deferred.digest() == deferred_digest
            and persisted_deferred.identity_quarantined_count == 2
        ),
        "anchor_update_written": anchor_update,
        "anchor_route_selected_nonlinear": (
            anchor_route.selected_slot_id == nonlinear_receipt.slot_id
        ),
        "verifier_acceptance_resolved_deferred": (
            accepted_resolution.accepted
            and accepted_resolution.resolved_count == 2
            and bank.identity_quarantined_count == 0
        ),
        "eviction_after_resolution_accepted": evicted.accepted,
        "resolved_frontend_mastery": mastery["mastery"] >= 0.95,
        "resolved_runtime_route": routed.selected_slot_id == nonlinear_receipt.slot_id,
        "exact_final_persistence": restored.digest() == bank.digest(),
        "restored_anchor_route": restored_anchor_route.selected_slot_id
        == nonlinear_receipt.slot_id,
        "verifier_memory_unchanged": evaluator.digest() == evaluator_digest,
        "model_unchanged": model.digest() == model_digest,
        "controller_frozen": controller_digest == migration._digest(controller),
        "verifier_replay_zero": evaluator.sample_count.item() == outcome.shape[0],
        "alignment_replay_zero": all(
            adapter.sample_count.item() == nonlinear_source.shape[0] // 2
            for adapter in (affine, nonlinear)
        ),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-alignment-delayed-identity.v1",
        "claim_boundary": (
            "bounded verifier-gated quarantine and delayed resolution of overlapping "
            "opaque identity signatures; not semantic open-world identity, unrestricted "
            "memory growth, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "capacity": CAPACITY,
            "identity_quarantine_capacity": IDENTITY_QUARANTINE_CAPACITY,
            "overlapping_signature": True,
            "runtime_frontend_ids_used_for_routing": False,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "ambiguous_route": _route_summary(ambiguous),
            "deferred_one": deferred_one.__dict__,
            "deferred_two": deferred_two.__dict__,
            "overflow": overflow.__dict__,
            "blocked_eviction": blocked_eviction.__dict__,
            "rejected_resolution": rejected_resolution.__dict__,
            "anchor_route": _route_summary(anchor_route),
            "accepted_resolution": accepted_resolution.__dict__,
            "evicted": evicted.__dict__,
            "resolved_mastery": mastery,
            "resolved_route": _route_summary(routed),
            "restored_anchor_route": _route_summary(restored_anchor_route),
            "final_active_slots": list(bank.slot_ids),
            "final_identity_quarantine_count": bank.identity_quarantined_count,
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "unique_alignment_pairs": sum(
                int(adapter.sample_count.item()) for adapter in (affine, nonlinear)
            ),
            "heldout_alignment_pairs": int(
                affine_source.shape[0] + nonlinear_source_holdout.shape[0]
            ),
            "unique_identity_signatures": 2,
            "deferred_identity_signatures": 2,
            "resolved_identity_signatures": 2,
            "transition_rows_consumed_once": transition_rows,
            "verifier_statistics_updates": 1,
            "alignment_statistics_updates": 2,
            "identity_route_updates_after_admission": 1,
            "old_verifier_replay": 0,
            "old_alignment_replay": 0,
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
    parser.add_argument("--seed", type=int, default=84901)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
