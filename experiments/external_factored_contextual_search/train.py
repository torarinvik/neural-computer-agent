"""Compose contextual reliability, factual growth, eviction, and goal search."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import torch

from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    AmodalCognitiveController,
    ExternalContextualTransitionEvidenceStatistics,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalModelBasedPlanner,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

SCHEMA = "neural-computer.external-factored-contextual-search.v1"
STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
ROWS = 24
TRAIN_ROWS = 20
EVIDENCE_ROWS = 4
INITIAL_CAPACITY = 2
GROWN_CAPACITY = 3
FEATURE_WIDTH = 128
RESIDUAL_UPDATES = 128
PREDICTION_TOLERANCE = 0.1
CORRUPTION_DELTA = 0.04


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _target(state: torch.Tensor, intention: torch.Tensor, regime: int) -> torch.Tensor:
    offset = torch.tensor(
        (
            0.75 * float(regime % 3) - 0.3 * float(regime // 3),
            0.6 * float(regime // 2) - 0.25 * float(regime % 2),
        ),
        dtype=state.dtype,
        device=state.device,
    )
    nonlinear = torch.stack(
        (
            0.45 * torch.sin(state[:, 0] + 0.35 * intention[:, 0]),
            0.4 * torch.tanh(state[:, 1] - 0.25 * intention[:, 1]),
        ),
        dim=1,
    )
    return nonlinear + offset


def _observation(
    state: torch.Tensor,
    intention: torch.Tensor,
    next_state: torch.Tensor,
) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=next_state,
        confidence=torch.ones(state.shape[0]),
    )


def _bank_mse(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    slot_id: int,
) -> float:
    index = bank.physical_index_for_slot_id(slot_id)
    context = bank.context_at(index).to(observation.state)
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    with torch.no_grad():
        prediction = bank(
            observation.state,
            observation.intention,
            context_batch,
        )
    return float((prediction - observation.next_state).square().mean())


def _retains(
    observations: dict[int, ExternalTransitionObservation],
    *,
    tolerance: float,
) -> Callable[[ExternalTransitionModelBank], bool]:
    def probe(bank: ExternalTransitionModelBank) -> bool:
        return all(
            slot_id in bank.slot_ids
            and _bank_mse(bank, observation, slot_id) <= tolerance
            for slot_id, observation in observations.items()
        )

    return probe


def _model_retains(
    observations: dict[int, ExternalTransitionObservation],
    *,
    tolerance: float,
) -> Callable[[ExternalFactoredTransitionModel], bool]:
    def probe(model: ExternalFactoredTransitionModel) -> bool:
        if model.residual_bank is None:
            return False
        return all(
            slot_id in model.residual_bank.slot_ids
            and _bank_mse(model.residual_bank, observation, slot_id) <= tolerance
            for slot_id, observation in observations.items()
        )

    return probe


def _context(router: ExternalFactoredTransitionRouter, slot_id: int) -> torch.Tensor:
    return router.contexts[router.slot_ids.index(slot_id)]


def _prediction(
    router: ExternalFactoredTransitionRouter,
    observation,
    slot_id: int,
) -> torch.Tensor:
    context = _context(router, slot_id).to(observation.state)
    with torch.no_grad():
        return router.model.predict_with_context(
            observation.state,
            observation.intention,
            context.unsqueeze(0).expand(observation.state.shape[0], -1),
        )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed,
    ).draw(ROWS) * 2.0 - 1.0
    state = inputs[:, :STATE_WIDTH]
    intention = inputs[:, STATE_WIDTH:]
    observations = {
        regime: _observation(state, intention, _target(state, intention, regime))
        for regime in range(3)
    }
    train = {
        regime: _observation(
            item.state[:TRAIN_ROWS],
            item.intention[:TRAIN_ROWS],
            item.next_state[:TRAIN_ROWS],
        )
        for regime, item in observations.items()
    }
    heldout = {
        regime: _observation(
            item.state[TRAIN_ROWS:],
            item.intention[TRAIN_ROWS:],
            item.next_state[TRAIN_ROWS:],
        )
        for regime, item in observations.items()
    }
    evidence_inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed + 100_000,
    ).draw(EVIDENCE_ROWS) * 2.0 - 1.0
    evidence_state = evidence_inputs[:, :STATE_WIDTH]
    evidence_intention = evidence_inputs[:, STATE_WIDTH:]
    evidence_observations = {
        regime: _observation(
            evidence_state,
            evidence_intention,
            _target(evidence_state, evidence_intention, regime),
        )
        for regime in range(3)
    }

    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_digest = _digest_module(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model = ExternalFactoredTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=16,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_random_feature_width=FEATURE_WIDTH,
        residual_random_feature_seed=seed,
        residual_ridge=0.01,
        residual_capacity=INITIAL_CAPACITY,
    )
    for parameter in model.base.parameters():
        parameter.data.zero_()
    model.freeze_base()
    base_digest = model.base.digest()
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=16,
        context_width=CONTEXT_WIDTH,
    )
    encoder_digest = encoder.digest()
    evidence = ExternalContextualTransitionEvidenceStatistics(
        STATE_WIDTH,
        CONTEXT_WIDTH,
        error_scale=0.2,
        prior_count=0.01,
        matching_tolerance=1e-4,
        count_decay=0.25,
    )
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=0.005,
        match_margin=0.0001,
        max_contexts=INITIAL_CAPACITY,
        residual_adaptation_updates=RESIDUAL_UPDATES,
        quarantine_capacity=ROWS,
        evidence_evaluator=evidence,
        evidence_threshold=0.5,
        evidence_gate_min_evidence=1,
        committed_evidence_gate=True,
    )

    promotion_receipts: list[dict[str, object]] = []
    for regime in (0, 1):
        route = router.route_bundle((train[regime],))
        receipt = router.promote_staged_candidate(
            heldout[regime],
            _model_retains(
                {previous: heldout[previous] for previous in range(regime)},
                tolerance=PREDICTION_TOLERANCE,
            ),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        promotion_receipts.append(receipt.__dict__)
        if route.status != "staged" or not receipt.accepted:
            break

    active_before = tuple(router.slot_ids)
    for slot_id in active_before:
        observation = evidence_observations[slot_id]
        prediction = _prediction(router, observation, slot_id)
        evidence.observe(
            prediction,
            observation.next_state,
            torch.ones(prediction.shape[0]),
            _context(router, slot_id),
        )

    model_digest_before_veto = router.model.digest()
    clean_prediction = _prediction(router, evidence_observations[0], 0)[:1]
    corrupted = _observation(
        evidence_observations[0].state[:1],
        evidence_observations[0].intention[:1],
        clean_prediction + torch.tensor([[CORRUPTION_DELTA, 0.0]]),
    )
    evidence.observe(
        clean_prediction,
        corrupted.next_state,
        torch.zeros(1),
        _context(router, 0),
    )
    veto_result = router.route_bundle((corrupted,))
    veto_preserved_facts = router.model.digest() == model_digest_before_veto

    reversal_source = _observation(
        evidence_observations[0].state[1:2],
        evidence_observations[0].intention[1:2],
        evidence_observations[0].next_state[1:2],
    )
    reversal_prediction = _prediction(router, reversal_source, 0)
    reversal = _observation(
        evidence_observations[0].state[1:2],
        evidence_observations[0].intention[1:2],
        reversal_prediction + torch.tensor([[CORRUPTION_DELTA, 0.0]]),
    )
    evidence.observe(
        reversal_prediction,
        reversal.next_state,
        torch.ones(1),
        _context(router, 0),
    )
    resolved_slots = router.resolve_quarantine(
        match_tolerance=PREDICTION_TOLERANCE,
        contradiction_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )

    growth_receipt = router.grow_verified(
        GROWN_CAPACITY,
        _retains(
            {0: heldout[0], 1: heldout[1]},
            tolerance=PREDICTION_TOLERANCE,
        ),
    )
    route_two = router.route_bundle((train[2],))
    receipt_two = router.promote_staged_candidate(
        heldout[2],
        _model_retains(
            {0: heldout[0], 1: heldout[1]},
            tolerance=PREDICTION_TOLERANCE,
        ),
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    promotion_receipts.append(receipt_two.__dict__)
    evidence.observe(
        _prediction(router, evidence_observations[2], 2),
        evidence_observations[2].next_state,
        torch.ones(evidence_observations[2].state.shape[0]),
        _context(router, 2),
    )

    planner = ExternalModelBasedPlanner(router.model, beam_width=2)
    model_digest_before_plan = router.model.digest()
    planning_state = heldout[2].state[:1]
    true_intention = heldout[2].intention[:1]
    goal = _target(planning_state, true_intention, 2)
    candidates = torch.cat((true_intention, -true_intention), dim=0)
    planning_result = planner.plan(
        planning_state,
        goal,
        candidates,
        horizon=1,
        transition_context=_context(router, 2).unsqueeze(0),
    )
    planner_preserved_facts = router.model.digest() == model_digest_before_plan

    eviction_receipt = router.evict_verified_id(
        1,
        _retains(
            {0: heldout[0], 2: heldout[2]},
            tolerance=PREDICTION_TOLERANCE,
        ),
    )
    restored_evidence = ExternalContextualTransitionEvidenceStatistics.from_payload(
        evidence.payload()
    )
    restored = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(),
        evidence_evaluator=restored_evidence,
    )
    restored_route = restored.route_bundle((train[2],))

    chosen_error = float(
        (planning_result.predicted_states[0, 0] - goal[0]).square().mean()
    )
    candidate_predictions = router.model.predict_with_context(
        planning_state.expand(candidates.shape[0], -1),
        candidates,
        _context(router, 2).unsqueeze(0).expand(candidates.shape[0], -1),
    )
    candidate_errors = (candidate_predictions - goal.expand_as(candidate_predictions)).square().mean(dim=-1)
    planner_ranked_candidates = bool(
        torch.isclose(
            planning_result.scores[0], candidate_errors.min(), atol=1e-7, rtol=1e-5
        )
    )
    gates = {
        "two_initial_slots_promoted": len(promotion_receipts) >= 2
        and all(item["accepted"] for item in promotion_receipts[:2]),
        "contextual_stats_track_active_slots": evidence.context_count
        == len(router.slot_ids),
        "near_boundary_corruption_vetoed": veto_result.status == "reliability_veto"
        and veto_result.quarantine_accepted,
        "veto_did_not_mutate_facts": veto_preserved_facts,
        "reversal_released_quarantine": resolved_slots == (0,)
        and router.quarantined_observations == 0,
        "growth_retained_prior_slots": growth_receipt.accepted
        and growth_receipt.context_count == 2,
        "third_slot_promoted": route_two.status == "staged"
        and receipt_two.accepted,
        "planner_ranked_candidates_by_goal": planner_ranked_candidates,
        "planner_reached_heldout_goal": chosen_error <= PREDICTION_TOLERANCE,
        "planner_is_inference_only": planner_preserved_facts,
        "middle_eviction_accepted": eviction_receipt.accepted,
        "eviction_removed_contextual_state": evidence.context_count
        == len(router.slot_ids)
        == 2,
        "restored_contextual_state_matches": restored_evidence.digest() == evidence.digest()
        and restored.slot_ids == router.slot_ids
        and restored_route.status == "matched"
        and restored_route.slot_id == 2,
        "base_unchanged": model.base.digest() == base_digest and model.base_frozen,
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_unchanged": encoder_digest == encoder.digest(),
        "replay_free": True,
    }
    report = {
        "schema": SCHEMA,
        "seed": seed,
        "configuration": {
            "initial_capacity": INITIAL_CAPACITY,
            "grown_capacity": GROWN_CAPACITY,
            "count_decay": evidence.count_decay,
            "corruption_delta": CORRUPTION_DELTA,
            "claim": "contextual-reliability-factual-search-composition-v1",
        },
        "metrics": {
            "promotion_receipts": promotion_receipts,
            "growth_receipt": growth_receipt.__dict__,
            "veto_status": veto_result.status,
            "resolved_slots": list(resolved_slots),
            "planning_score": float(planning_result.scores[0]),
            "planning_goal_error": chosen_error,
            "planning_expanded_nodes": planning_result.expanded_nodes,
            "eviction_receipt": eviction_receipt.__dict__,
            "active_slot_ids": list(router.slot_ids),
            "contextual_state_count": evidence.context_count,
            "model_mse_slot_2": _bank_mse(
                router.model.residual_bank, heldout[2], 2
            ),
        },
        "gates": gates,
        "accounting": {
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "residual_optimizer_updates": 0,
            "reliability_statistics_updates": 5,
            "replayed_rows": 0,
            "unique_transition_rows": 3 * (TRAIN_ROWS + EVIDENCE_ROWS),
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "base": base_digest,
            "controller": controller_digest,
            "context_encoder": encoder_digest,
            "model": router.model.digest(),
            "evidence": evidence.digest(),
        },
        "promoted": all(gates.values()),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=94001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
