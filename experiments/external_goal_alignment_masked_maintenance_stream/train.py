"""Pressure-test repeated opaque masked-memory maintenance under reversal."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.external_goal_alignment_drift_missing_reversal import train as drift
from experiments.external_goal_alignment_masked_capacity import train as capacity
from experiments.external_goal_representation_drift_gate import train as drift_data
from experiments.external_goal_representation_migration import train as migration
from experiments.external_one_pass_goal_evaluator import train as one_pass
from neural_computer import (
    AmodalCognitiveController,
    ExternalGoalEvaluatorStatistics,
    ExternalGoalRepresentationAlignmentBank,
    OpaqueCapacityPlanner,
)

INITIAL_PROTOTYPES_PER_SLOT = 2
GROWN_PROTOTYPES_PER_SLOT = 5
IDENTITY_MIN_SCORE = 0.55
IDENTITY_MIN_MARGIN = 0.04
MASKS = (
    torch.tensor(
        [False, True, True, False, True, True, True, True],
        dtype=torch.bool,
    ),
    torch.tensor(
        [True, False, True, True, True, True, False, True],
        dtype=torch.bool,
    ),
    torch.tensor(
        [True, True, False, True, True, False, True, True],
        dtype=torch.bool,
    ),
    torch.tensor(
        [False, True, False, True, True, False, True, True],
        dtype=torch.bool,
    ),
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
    model_digest = model.digest()
    old_state, old_goal, outcome = one_pass._verifier_batch(seed)
    evaluator = ExternalGoalEvaluatorStatistics(1, ridge=1e-5)
    evaluator.observe(old_state, old_goal, outcome)
    evaluator_digest = evaluator.digest()

    affine, affine_source, affine_target, affine_base = capacity.identity._fit_affine(seed)
    nonlinear_source, nonlinear_target = drift_data._nonlinear_batch(seed)
    nonlinear, nonlinear_holdout_source, nonlinear_holdout_target, nonlinear_base = (
        capacity.identity._fit_nonlinear(nonlinear_source, nonlinear_target, seed)
    )
    affine_base = torch.nn.functional.normalize(affine_base, dim=0)
    nonlinear_base = torch.nn.functional.normalize(nonlinear_base, dim=0)
    direction = drift._orthogonal_direction(affine_base, nonlinear_base, seed + 17)
    pattern_values = tuple(
        drift._drifted_signature(affine_base, direction, scale)
        for scale in (0.6, -0.6, 0.0, 0.25)
    )
    replacement_value = drift._drifted_signature(affine_base, direction, -0.25)
    patterns = tuple(zip(pattern_values, MASKS, strict=True))
    replacement_pattern = (replacement_value, MASKS[3])

    bank = ExternalGoalRepresentationAlignmentBank(
        1,
        capacity=2,
        identity_width=8,
        identity_min_score=IDENTITY_MIN_SCORE,
        identity_min_margin=IDENTITY_MIN_MARGIN,
        identity_max_prototypes_per_slot=INITIAL_PROTOTYPES_PER_SLOT,
        identity_merge_cosine=0.94,
    )
    affine_receipt = bank.admit_verified(
        "opaque-affine",
        affine,
        affine_source,
        affine_target,
        prediction_tolerance=capacity.identity.ALIGNMENT_TOLERANCE,
        identity_signature=affine_base,
    )
    nonlinear_receipt = bank.admit_verified(
        "opaque-nonlinear",
        nonlinear,
        nonlinear_holdout_source,
        nonlinear_holdout_target,
        prediction_tolerance=capacity.identity.ALIGNMENT_TOLERANCE,
        identity_signature=nonlinear_base,
    )
    identity_memory = bank.identity_memory
    assert identity_memory is not None

    before_growth = identity_memory.digest()
    rejected_growth = identity_memory.grow_verified(
        GROWN_PROTOTYPES_PER_SLOT,
        lambda _candidate: False,
    )
    after_rejected_growth = identity_memory.digest()
    rejected_growth_preserved_state = (
        not rejected_growth.accepted
        and before_growth == after_rejected_growth
        and identity_memory.max_prototypes_per_slot == INITIAL_PROTOTYPES_PER_SLOT
    )
    accepted_growth = identity_memory.grow_verified(
        GROWN_PROTOTYPES_PER_SLOT,
        lambda candidate: all(
            candidate.propose(
                query,
                bank.slot_ids,
                minimum_score=IDENTITY_MIN_SCORE,
            ).selected_slot_id
            == expected_slot
            for query, expected_slot in (
                (affine_base, affine_receipt.slot_id),
                (nonlinear_base, nonlinear_receipt.slot_id),
            )
        ),
    )

    planner = OpaqueCapacityPlanner(width=8, hidden=16).eval()
    planner_digest = _digest(planner)
    planner_before = identity_memory.digest()
    planner_plan_before = identity_memory.maintenance_plan(
        affine_receipt.slot_id,
        replacement_pattern[0],
        planner=planner,
        query_mask=replacement_pattern[1],
        protected_indices=(0,),
    )
    planner_after = identity_memory.digest()

    for query, query_mask in patterns:
        bank.accept_identity_anchor(
            query,
            signature_mask=query_mask,
            verifier_accepted=True,
        )
    insertion_count = identity_memory.prototype_count(affine_receipt.slot_id)

    # The same online stream is observed in forward and reverse order. These
    # are new timeline observations, not reads from a replay buffer.
    online_order = list(range(len(patterns))) + list(reversed(range(len(patterns))))
    route_checks: list[bool] = []
    for _ in range(3):
        for pattern_index in online_order:
            query, query_mask = patterns[pattern_index]
            bank.accept_identity_anchor(
                query,
                signature_mask=query_mask,
                verifier_accepted=True,
            )
            route_checks.append(
                bank.route_by_signature(
                    query,
                    affine_source[:1],
                    signature_mask=query_mask,
                ).selected_slot_id
                == affine_receipt.slot_id
            )

    for query, query_mask, repeats in (
        (affine_base, None, 4),
        (patterns[0][0], patterns[0][1], 2),
        (patterns[1][0], patterns[1][1], 2),
        (patterns[2][0], patterns[2][1], 2),
    ):
        for _ in range(repeats):
            bank.accept_identity_anchor(
                query,
                signature_mask=query_mask,
                verifier_accepted=True,
            )

    desired_after_replacement = (
        (affine_base, None),
        (patterns[0][0], patterns[0][1]),
        (patterns[1][0], patterns[1][1]),
        (patterns[2][0], patterns[2][1]),
        replacement_pattern,
    )

    def replacement_probe(candidate) -> bool:
        return all(
            candidate.propose(
                query,
                bank.slot_ids,
                minimum_score=IDENTITY_MIN_SCORE,
                query_mask=query_mask,
            ).selected_slot_id
            == affine_receipt.slot_id
            for query, query_mask in desired_after_replacement
        )

    before_rejected_replacement = identity_memory.digest()
    rejected_replacement = bank.accept_identity_anchor(
        replacement_pattern[0],
        signature_mask=replacement_pattern[1],
        verifier_accepted=True,
        retention_probe=lambda _candidate: False,
    )
    after_rejected_replacement = identity_memory.digest()
    accepted_replacement = bank.accept_identity_anchor(
        replacement_pattern[0],
        signature_mask=replacement_pattern[1],
        verifier_accepted=True,
        retention_probe=replacement_probe,
    )

    before_rejected_consolidation = identity_memory.digest()
    rejected_consolidation = identity_memory.consolidate_verified(
        affine_receipt.slot_id,
        (1, 2),
        retention_probe=lambda _candidate: False,
    )
    after_rejected_consolidation = identity_memory.digest()

    def consolidation_probe(candidate) -> bool:
        return all(
            candidate.propose(
                query,
                bank.slot_ids,
                minimum_score=IDENTITY_MIN_SCORE,
                query_mask=query_mask,
            ).selected_slot_id
            == affine_receipt.slot_id
            for query, query_mask in desired_after_replacement
        )

    accepted_consolidation = identity_memory.consolidate_verified(
        affine_receipt.slot_id,
        (1, 2),
        consolidation_probe,
    )
    after_consolidation_routes = [
        bank.route_by_signature(
            query,
            affine_source[:1],
            signature_mask=query_mask,
        ).selected_slot_id
        == affine_receipt.slot_id
        for query, query_mask in desired_after_replacement
    ]

    # Re-admit the reversed fourth pattern after compaction, then traverse the
    # reverse stream once more to test reuse after maintenance.
    re_admitted = bank.accept_identity_anchor(
        patterns[3][0],
        signature_mask=patterns[3][1],
        verifier_accepted=True,
    )
    final_reverse_routes = []
    for pattern_index in reversed(range(len(patterns))):
        query, query_mask = patterns[pattern_index]
        final_reverse_routes.append(
            bank.route_by_signature(
                query,
                affine_source[:1],
                signature_mask=query_mask,
            ).selected_slot_id
            == affine_receipt.slot_id
        )

    retained_affine = bank.route_by_signature(affine_base, affine_source[:1])
    retained_nonlinear = bank.route_by_signature(
        nonlinear_base,
        nonlinear_source[:1],
    )
    affine_eval, affine_route = capacity.identity._evaluate_routed(
        bank,
        model,
        evaluator,
        migration._new_representation,
        seed + 601,
        query_expected_slot=affine_receipt.slot_id,
    )
    nonlinear_eval, nonlinear_route = capacity.identity._evaluate_routed(
        bank,
        model,
        evaluator,
        drift_data._nonlinear_representation,
        seed + 602,
        query_expected_slot=nonlinear_receipt.slot_id,
    )
    restored = ExternalGoalRepresentationAlignmentBank.from_payload(
        bank.state_payload()
    )
    restored_affine = restored.route_by_signature(affine_base, affine_source[:1])
    restored_replacement = restored.route_by_signature(
        replacement_pattern[0],
        affine_source[:1],
        signature_mask=replacement_pattern[1],
    )

    gates = {
        "both_frontends_admitted": affine_receipt.accepted and nonlinear_receipt.accepted,
        "rejected_growth_preserves_state": rejected_growth_preserved_state,
        "accepted_growth_passes_retention": accepted_growth.accepted,
        "planner_is_side_effect_free": planner_before == planner_after,
        "four_mask_patterns_inserted": insertion_count == 5,
        "forward_reverse_stream_routes": all(route_checks),
        "rejected_replacement_preserves_state": (
            not rejected_replacement.anchor_update_stored
            and before_rejected_replacement == after_rejected_replacement
        ),
        "accepted_replacement_passes_retention": accepted_replacement.anchor_update_stored,
        "rejected_consolidation_preserves_state": (
            not rejected_consolidation.accepted
            and before_rejected_consolidation == after_rejected_consolidation
        ),
        "accepted_consolidation_passes_retention": accepted_consolidation.accepted,
        "all_post_consolidation_routes": all(after_consolidation_routes),
        "reversal_re_admission": re_admitted.anchor_update_stored,
        "final_reverse_stream_routes": all(final_reverse_routes),
        "full_affine_route_retained": retained_affine.selected_slot_id
        == affine_receipt.slot_id,
        "nonlinear_route_retained": retained_nonlinear.selected_slot_id
        == nonlinear_receipt.slot_id,
        "restored_affine_route": restored_affine.selected_slot_id
        == affine_receipt.slot_id,
        "restored_replacement_route": restored_replacement.selected_slot_id
        == affine_receipt.slot_id,
        "affine_mastery_after_stream": affine_eval["mastery"] >= 0.95,
        "nonlinear_mastery_after_stream": nonlinear_eval["mastery"] >= 0.95,
        "exact_final_persistence": restored.digest() == bank.digest(),
        "verifier_memory_unchanged": evaluator.digest() == evaluator_digest,
        "model_unchanged": model.digest() == model_digest,
        "controller_frozen": controller_digest == _digest(controller),
        "planner_frozen": planner_digest == _digest(planner),
        "verifier_replay_zero": evaluator.sample_count.item() == outcome.shape[0],
        "alignment_replay_zero": all(
            adapter.sample_count.item() == affine_source.shape[0]
            for adapter in (affine, nonlinear)
        ),
    }
    report = {
        "schema": "neural-computer.external-goal-representation-alignment-masked-maintenance-stream.v1",
        "claim_boundary": (
            "bounded online external-memory maintenance with four changed masks, "
            "forward/reverse observation order, verifier-gated growth, replacement, "
            "and consolidation; the capacity planner is advisory and untrained, "
            "so this is not autonomous maintenance policy or general continual "
            "learning"
        ),
        "seed": seed,
        "configuration": {
            "initial_max_prototypes_per_slot": INITIAL_PROTOTYPES_PER_SLOT,
            "grown_max_prototypes_per_slot": GROWN_PROTOTYPES_PER_SLOT,
            "mask_count": len(MASKS),
            "online_reversal_rounds": 3,
            "runtime_frontend_ids_used_for_updates": False,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "planner_plan_before_maintenance": {
                "action": planner_plan_before.action,
                "action_index": planner_plan_before.action_index,
                "eviction_index": planner_plan_before.eviction_index,
                "pair": planner_plan_before.pair,
                "score": float(planner_plan_before.score),
            },
            "rejected_growth": rejected_growth.__dict__,
            "accepted_growth": accepted_growth.__dict__,
            "rejected_replacement": rejected_replacement.__dict__,
            "accepted_replacement": accepted_replacement.__dict__,
            "rejected_consolidation": rejected_consolidation.__dict__,
            "accepted_consolidation": accepted_consolidation.__dict__,
            "retained_affine_route": _route_summary(retained_affine),
            "retained_nonlinear_route": _route_summary(retained_nonlinear),
            "affine_route": _route_summary(affine_route),
            "nonlinear_route": _route_summary(nonlinear_route),
            "affine_mastery": affine_eval,
            "nonlinear_mastery": nonlinear_eval,
            "prototype_counts": {
                str(slot_id): identity_memory.prototype_count(slot_id)
                for slot_id in identity_memory.slot_ids
            },
            "masked_prototype_count": identity_memory.masked_prototype_count,
            "restored_masked_prototype_count": restored.identity_memory.masked_prototype_count,
            "stream_steps": len(route_checks) + len(final_reverse_routes),
            "active_slots": list(bank.slot_ids),
        },
        "accounting": {
            "unique_verifier_outcomes": int(outcome.shape[0]),
            "unique_alignment_pairs": int(
                affine.sample_count.item() + nonlinear.sample_count.item()
            ),
            "heldout_alignment_pairs": int(
                affine_source.shape[0] + nonlinear_holdout_source.shape[0]
            ),
            "unique_masked_patterns": len(MASKS) + 1,
            "online_logical_observations": len(route_checks),
            "replayed_examples": 0,
            "identity_memory_growth_operations": 2,
            "identity_memory_replacement_operations": 2,
            "identity_memory_consolidation_operations": 2,
            "identity_anchor_operations": 4 + len(route_checks) + 4 + 2,
            "transition_rows_consumed_once": transition_rows,
            "verifier_statistics_updates": 1,
            "controller_optimizer_updates": 0,
            "planner_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85401)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
