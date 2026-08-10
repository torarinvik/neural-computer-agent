"""Pressure-test caller-free identity under drift, missing evidence, and reversals."""

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
IDENTITY_MIN_SCORE = 0.55
IDENTITY_MIN_MARGIN = 0.04
DRIFT_AMPLITUDE = 0.16
DRIFT_PHASES = (0.0, 0.25, 0.5, 0.75, 1.0, 0.75, 0.5, 0.25, 0.0, -0.25, -0.5, -0.75, -1.0, -0.75, -0.5, -0.25)


def _route_summary(result) -> dict[str, object]:
    return {
        "selected_slot_id": result.selected_slot_id,
        "eligible_slot_ids": list(result.eligible_slot_ids),
        "scores": [float(value) for value in result.scores.tolist()],
        "margin": result.margin,
        "reason": result.reason,
    }


def _orthogonal_direction(
    base: torch.Tensor,
    other: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(base.shape, generator=generator)
    direction = direction - (direction @ base) * base
    direction = direction - (direction @ other) * other
    return torch.nn.functional.normalize(direction, dim=0)


def _drifted_signature(
    base: torch.Tensor,
    direction: torch.Tensor,
    amplitude: float,
) -> torch.Tensor:
    return torch.nn.functional.normalize(base + amplitude * direction, dim=0)


def _mask_for_phase(width: int, phase: int) -> torch.Tensor | None:
    if phase % 4 == 0:
        return None
    mask = torch.ones(width, dtype=torch.bool)
    first = (phase * 3) % width
    mask[first] = False
    mask[(first + 1) % width] = False
    return mask


def _identity_query(
    base: torch.Tensor,
    direction: torch.Tensor,
    phase: int,
    *,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    amplitude = DRIFT_AMPLITUDE * DRIFT_PHASES[phase]
    signature = _drifted_signature(base, direction, amplitude)
    mask = _mask_for_phase(width, phase)
    return signature, mask


def _digest(module: torch.nn.Module) -> str:
    return migration._digest(module)


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
    nonlinear_source, nonlinear_target = drift._nonlinear_batch(seed)
    nonlinear, nonlinear_holdout_source, nonlinear_holdout_target, nonlinear_base = (
        identity._fit_nonlinear(nonlinear_source, nonlinear_target, seed)
    )
    width = int(affine_base.shape[0])
    affine_base = torch.nn.functional.normalize(affine_base, dim=0)
    nonlinear_base = torch.nn.functional.normalize(nonlinear_base, dim=0)
    affine_direction = _orthogonal_direction(affine_base, nonlinear_base, seed + 17)
    nonlinear_direction = _orthogonal_direction(nonlinear_base, affine_base, seed + 23)

    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=CAPACITY,
        identity_width=width,
        identity_min_score=IDENTITY_MIN_SCORE,
        identity_min_margin=IDENTITY_MIN_MARGIN,
        identity_max_prototypes_per_slot=8,
        identity_merge_cosine=0.94,
        identity_quarantine_capacity=IDENTITY_QUARANTINE_CAPACITY,
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

    ambiguous = torch.nn.functional.normalize(affine_base + nonlinear_base, dim=0)
    deferred_one = bank.defer_identity_signature(ambiguous)
    deferred_two = bank.defer_identity_signature(ambiguous)
    before_rejected_anchor = bank.digest()
    rejected_anchor = bank.accept_identity_anchor(
        affine_base,
        verifier_accepted=False,
    )
    after_rejected_anchor = bank.digest()
    accepted_anchor = bank.accept_identity_anchor(
        affine_base,
        verifier_accepted=True,
    )

    ordered_queries: list[dict[str, object]] = []
    route_correct = 0
    partial_count = 0
    full_anchor_count = 0
    anchor_updates = 0
    reversal_count = 0
    last_order: tuple[str, str] | None = None
    for phase in range(len(DRIFT_PHASES)):
        affine_signature, affine_mask = _identity_query(
            affine_base,
            affine_direction,
            phase,
            width=width,
        )
        nonlinear_signature, nonlinear_mask = _identity_query(
            nonlinear_base,
            nonlinear_direction,
            phase,
            width=width,
        )
        entries = [
            ("affine", affine_signature, affine_mask, affine_receipt.slot_id),
            ("nonlinear", nonlinear_signature, nonlinear_mask, nonlinear_receipt.slot_id),
        ]
        if phase % 2:
            entries.reverse()
        current_order = (str(entries[0][0]), str(entries[1][0]))
        if last_order is not None and current_order != last_order:
            reversal_count += 1
        last_order = current_order
        for name, signature, mask, expected_slot in entries:
            source = affine_source[:1] if name == "affine" else nonlinear_source[:1]
            route = bank.route_by_signature(
                signature,
                source,
                signature_mask=mask,
            )
            selected = route.selected_slot_id
            route_correct += int(selected == expected_slot)
            partial_count += int(mask is not None)
            anchor = bank.accept_identity_anchor(
                signature,
                signature_mask=mask,
                verifier_accepted=selected == expected_slot,
            )
            full_anchor_count += int(mask is None)
            anchor_updates += int(anchor.anchor_update_stored)
            ordered_queries.append(
                {
                    "phase": phase,
                    "name": name,
                    "expected_slot": expected_slot,
                    "mask_present_count": width if mask is None else int(mask.sum()),
                    "route": _route_summary(route),
                    "anchor": anchor.__dict__,
                }
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
        drift._nonlinear_representation,
        seed + 302,
        query_expected_slot=nonlinear_receipt.slot_id,
    )
    restored = ExternalGoalRepresentationAlignmentBank.from_payload(bank.state_payload())
    restored_affine_route = restored.route_by_signature(affine_base, affine_source[:1])
    restored_nonlinear_route = restored.route_by_signature(
        nonlinear_base,
        nonlinear_source[:1],
    )

    total_queries = len(DRIFT_PHASES) * 2
    gates = {
        "both_frontends_admitted": affine_receipt.accepted and nonlinear_receipt.accepted,
        "deferred_ambiguous_evidence_retained": deferred_one.accepted and deferred_two.accepted,
        "rejected_anchor_preserves_state": (
            not rejected_anchor.accepted
            and before_rejected_anchor == after_rejected_anchor
            and bank.identity_quarantined_count == 0
        ),
        "accepted_anchor_resolves_without_caller_slot_id": (
            accepted_anchor.accepted
            and accepted_anchor.selected_slot_id == affine_receipt.slot_id
            and accepted_anchor.resolved_count == 2
        ),
        "repeated_order_reversals_exercised": reversal_count >= len(DRIFT_PHASES) - 2,
        "missing_evidence_exercised": partial_count >= total_queries // 2,
        "all_drift_routes_correct": route_correct == total_queries,
        "all_anchor_proposals_safe": all(
            item["anchor"]["verifier_accepted"]
            and item["anchor"]["selected_slot_id"] == item["expected_slot"]
            for item in ordered_queries
        ),
        "full_anchors_updated_without_ids": anchor_updates == full_anchor_count,
        "affine_mastery_after_drift": affine_eval["mastery"] >= 0.95,
        "nonlinear_mastery_after_drift": nonlinear_eval["mastery"] >= 0.95,
        "exact_final_persistence": restored.digest() == bank.digest(),
        "restored_affine_route": restored_affine_route.selected_slot_id == affine_receipt.slot_id,
        "restored_nonlinear_route": restored_nonlinear_route.selected_slot_id == nonlinear_receipt.slot_id,
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
        "schema": "neural-computer.external-goal-representation-alignment-drift-missing-reversal.v1",
        "claim_boundary": (
            "bounded replay-free verifier-gated identity retention under gradual and "
            "reversible drift with partial learned evidence; not semantic open-world "
            "identity, unrestricted memory growth, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "capacity": CAPACITY,
            "identity_width": width,
            "identity_min_score": IDENTITY_MIN_SCORE,
            "identity_min_margin": IDENTITY_MIN_MARGIN,
            "identity_quarantine_capacity": IDENTITY_QUARANTINE_CAPACITY,
            "drift_phases": len(DRIFT_PHASES),
            "drift_amplitude": DRIFT_AMPLITUDE,
            "runtime_frontend_ids_used_for_routing": False,
            "runtime_frontend_ids_used_for_identity_updates": False,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "route_accuracy": route_correct / total_queries,
            "partial_query_count": partial_count,
            "full_anchor_count": full_anchor_count,
            "anchor_updates": anchor_updates,
            "reversal_count": reversal_count,
            "accepted_anchor": accepted_anchor.__dict__,
            "affine_route": _route_summary(affine_route),
            "nonlinear_route": _route_summary(nonlinear_route),
            "affine_mastery": affine_eval,
            "nonlinear_mastery": nonlinear_eval,
            "ordered_queries": ordered_queries,
            "restored_affine_route": _route_summary(restored_affine_route),
            "restored_nonlinear_route": _route_summary(restored_nonlinear_route),
            "final_active_slots": list(bank.slot_ids),
            "final_identity_quarantine_count": bank.identity_quarantined_count,
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "unique_alignment_pairs": int(affine.sample_count.item() + nonlinear.sample_count.item()),
            "heldout_alignment_pairs": int(affine_source.shape[0] + nonlinear_holdout_source.shape[0]),
            "unique_identity_queries": total_queries,
            "partial_identity_queries": partial_count,
            "full_identity_anchor_updates": anchor_updates,
            "transition_rows_consumed_once": transition_rows,
            "verifier_statistics_updates": 1,
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
    parser.add_argument("--seed", type=int, default=85001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
