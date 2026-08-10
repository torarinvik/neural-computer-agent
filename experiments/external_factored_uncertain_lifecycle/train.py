"""Combine reliability, delayed evidence, and factored memory lifecycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_factored_memory_lifecycle.train import (
    _model_retains,
    _observation,
    _retains,
    _rows,
    _target,
)
from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    AmodalCognitiveController,
    EventWaitPolicy,
    EventWaitStatistics,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionEvidenceStatistics,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
REGIME_COUNT = 4
ROWS_PER_REGIME = 24
TRAIN_ROWS = 20
INITIAL_CAPACITY = 2
GROWN_CAPACITY = 3
FEATURE_WIDTH = 128
RESIDUAL_UPDATES = 128
PREDICTION_TOLERANCE = 0.1
MATCH_TOLERANCE = 0.005
CORRUPTION_DELTA = 0.04
RELIABILITY_THRESHOLD = 0.9
RELIABILITY_WARMUP = 4


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _slice(
    observation: ExternalTransitionObservation,
    start: int,
    stop: int,
) -> ExternalTransitionObservation:
    return _observation(
        observation.state[start:stop],
        observation.intention[start:stop],
        observation.next_state[start:stop],
    )


def _observe_verifier(
    reliability: ExternalTransitionEvidenceStatistics,
    model: ExternalFactoredTransitionModel,
    context: torch.Tensor,
    observation: ExternalTransitionObservation,
    outcome: float,
) -> None:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    with torch.no_grad():
        prediction = model.predict_with_context(
            observation.state,
            observation.intention,
            context_batch,
        )
    reliability.observe(
        prediction,
        observation.next_state,
        torch.full((observation.state.shape[0],), outcome),
    )


def _wait_statistics() -> EventWaitStatistics:
    statistics = EventWaitStatistics(
        bin_count=4,
        ridge=1e-3,
        outcome_scale=8.0,
        minimum_context_observations=1,
    )
    delayed = EventWaitPolicy.features(
        age=torch.full((16,), 3.0),
        present_fraction=torch.full((16,), 0.5),
        complete=torch.zeros(16),
        arrival_count=torch.full((16,), 8.0),
        arrival_delta=torch.full((16,), 3.0),
    )
    immediate = EventWaitPolicy.features(
        age=torch.full((16,), 0.1),
        present_fraction=torch.full((16,), 0.5),
        complete=torch.zeros(16),
        arrival_count=torch.ones(16),
        arrival_delta=torch.full((16,), 0.1),
    )
    statistics.observe(delayed, torch.ones(16))
    statistics.observe(immediate, torch.zeros(16))
    return statistics


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed,
    ).draw(ROWS_PER_REGIME) * 2.0 - 1.0
    state, intention = inputs[:, :STATE_WIDTH], inputs[:, STATE_WIDTH:]
    full = {
        regime: _observation(
            state,
            intention,
            _target(state, intention, regime),
        )
        for regime in range(REGIME_COUNT)
    }
    train = {regime: _slice(item, 0, TRAIN_ROWS) for regime, item in full.items()}
    heldout = {
        regime: _slice(item, TRAIN_ROWS, ROWS_PER_REGIME)
        for regime, item in full.items()
    }

    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_digest = _digest(controller)
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
    reliability = ExternalTransitionEvidenceStatistics(
        STATE_WIDTH,
        bin_count=16,
        error_scale=0.2,
        prior_count=0.01,
    )
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=MATCH_TOLERANCE,
        match_margin=0.0,
        max_contexts=INITIAL_CAPACITY,
        residual_adaptation_updates=RESIDUAL_UPDATES,
        quarantine_capacity=2 * TRAIN_ROWS,
        evidence_evaluator=reliability,
        evidence_threshold=RELIABILITY_THRESHOLD,
        evidence_gate_min_evidence=RELIABILITY_WARMUP,
        committed_evidence_gate=True,
    )

    source_route = router.route_bundle(_rows(train[0]))
    source_receipt = router.promote_staged_candidate(
        heldout[0],
        _model_retains({}, tolerance=PREDICTION_TOLERANCE),
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    source_context = router.contexts[0].clone()
    target_route = router.route_bundle(_rows(train[1]))
    target_receipt = router.promote_staged_candidate(
        heldout[1],
        _model_retains({0: heldout[0]}, tolerance=PREDICTION_TOLERANCE),
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    target_context = router.contexts[1].clone()

    clean_source_status = router.route_bundle(_rows(train[0])).status
    _observe_verifier(reliability, router.model, source_context, train[0], 1.0)
    clean_target_status = router.route_bundle(_rows(train[1])).status
    _observe_verifier(reliability, router.model, target_context, train[1], 1.0)
    gate_open = int(reliability.observation_count) >= RELIABILITY_WARMUP

    corrupted = _observation(
        train[0].state,
        train[0].intention,
        train[0].next_state + CORRUPTION_DELTA,
    )
    corruption_status = router.route_bundle(_rows(corrupted)).status
    _observe_verifier(reliability, router.model, source_context, corrupted, 0.0)
    source_reversal_status = router.route_bundle(_rows(train[0])).status
    target_reversal_status = router.route_bundle(_rows(train[1])).status

    wait_statistics = _wait_statistics()
    delayed_features = EventWaitPolicy.features(
        age=torch.tensor([3.0]),
        present_fraction=torch.tensor([0.5]),
        complete=torch.tensor([0.0]),
        arrival_count=torch.tensor([8.0]),
        arrival_delta=torch.tensor([3.0]),
    )
    immediate_features = EventWaitPolicy.features(
        age=torch.tensor([0.1]),
        present_fraction=torch.tensor([0.5]),
        complete=torch.tensor([0.0]),
        arrival_count=torch.tensor([1.0]),
        arrival_delta=torch.tensor([0.1]),
    )
    with torch.no_grad():
        delayed_wait_probability = float(wait_statistics(delayed_features)[0])
        immediate_wait_probability = float(wait_statistics(immediate_features)[0])

    partial = _slice(train[0], 0, 5)
    partial_read = router.route_partial_bundle(
        _rows(partial),
        min_match_fraction=1.0,
        match_tolerance=PREDICTION_TOLERANCE,
        contradiction_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    delayed_quarantine = router.quarantine_partial_bundle((partial,))
    resolved = router.resolve_quarantine(
        match_tolerance=PREDICTION_TOLERANCE,
        contradiction_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    digest_before_absence = router.digest()
    immediate_absence_released = immediate_wait_probability < 0.5
    digest_after_absence = router.digest()

    full_capacity_novel = router.route_bundle(_rows(train[2]))
    capacity_unchanged = router.slot_ids == (0, 1) and not router.candidate_active
    growth = router.grow_verified(
        GROWN_CAPACITY,
        _retains({0: heldout[0], 1: heldout[1]}, tolerance=PREDICTION_TOLERANCE),
    )
    novel_route = router.route_bundle(_rows(train[2]))
    third_receipt = router.promote_staged_candidate(
        heldout[2],
        _model_retains(
            {0: heldout[0], 1: heldout[1]},
            tolerance=PREDICTION_TOLERANCE,
        ),
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    eviction = router.evict_verified_id(
        1,
        _retains({0: heldout[0], 2: heldout[2]}, tolerance=PREDICTION_TOLERANCE),
    )
    fourth_route = router.route_bundle(_rows(train[3]))
    fourth_receipt = router.promote_staged_candidate(
        heldout[3],
        _model_retains(
            {0: heldout[0], 2: heldout[2]},
            tolerance=PREDICTION_TOLERANCE,
        ),
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    source_after_lifecycle = router.route_bundle(_rows(train[0]))
    third_after_lifecycle = router.route_bundle(_rows(train[2]))
    fourth_after_lifecycle = router.route_bundle(_rows(train[3]))

    restored_reliability = ExternalTransitionEvidenceStatistics.from_payload(
        reliability.payload()
    )
    restored_wait = EventWaitStatistics.from_payload(wait_statistics.payload())
    restored_router = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=restored_reliability
    )
    persistence_exact = (
        restored_router.digest() == router.digest()
        and restored_router.configuration() == router.configuration()
        and restored_reliability.digest() == reliability.digest()
        and restored_wait.digest() == wait_statistics.digest()
    )
    gates = {
        "source_promoted": source_route.status == "staged" and source_receipt.accepted,
        "target_promoted": target_route.status == "staged" and target_receipt.accepted,
        "clean_revisits_route": clean_source_status == "matched"
        and clean_target_status == "matched",
        "reliability_warmup_reached": gate_open,
        "corruption_vetoed": corruption_status == "reliability_veto",
        "active_gate_reversals_route": source_reversal_status == "matched"
        and target_reversal_status == "matched",
        "delayed_wait_learned": delayed_wait_probability > 0.75,
        "immediate_absence_released": immediate_wait_probability < 0.25
        and immediate_absence_released
        and digest_before_absence == digest_after_absence,
        "partial_read_routes": partial_read.status == "matched"
        and partial_read.slot_id == 0,
        "delayed_partial_quarantined": delayed_quarantine.accepted,
        "quarantine_resolved_once": resolved == (0,)
        and router.quarantined_observations == TRAIN_ROWS,
        "corruption_quarantine_retained": router.quarantined_observations == TRAIN_ROWS,
        "full_capacity_novelty_refused": full_capacity_novel.status == "ambiguous"
        and capacity_unchanged,
        "growth_accepted": growth.accepted,
        "novel_promoted_after_growth": novel_route.status == "staged"
        and third_receipt.accepted,
        "eviction_accepted": eviction.accepted and router.slot_ids[:2] == (0, 2),
        "fourth_promoted_after_eviction": fourth_route.status == "staged"
        and fourth_receipt.accepted,
        "survivors_route_after_lifecycle": source_after_lifecycle.slot_id == 0
        and third_after_lifecycle.slot_id == 2
        and fourth_after_lifecycle.slot_id == 3,
        "base_frozen": model.base_frozen and model.base.digest() == base_digest,
        "controller_frozen": _digest(controller) == controller_digest,
        "context_encoder_unchanged": encoder.digest() == encoder_digest,
        "exact_persistence": persistence_exact,
        "replay_free": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"uncertain lifecycle gates failed: {gates}")
    report: dict[str, object] = {
        "schema": "neural-computer.external-factored-uncertain-lifecycle.v1",
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "initial_capacity": INITIAL_CAPACITY,
            "grown_capacity": GROWN_CAPACITY,
            "train_rows": TRAIN_ROWS,
            "corruption_delta": CORRUPTION_DELTA,
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "reliability_warmup": RELIABILITY_WARMUP,
            "wait_statistics_samples": int(wait_statistics.sample_count),
            "claim": "online-reliability-wait-lifecycle-v1",
        },
        "metrics": {
            "source_route_status": source_route.status,
            "target_route_status": target_route.status,
            "clean_source_status": clean_source_status,
            "clean_target_status": clean_target_status,
            "corruption_status": corruption_status,
            "source_reversal_status": source_reversal_status,
            "target_reversal_status": target_reversal_status,
            "delayed_wait_probability": delayed_wait_probability,
            "immediate_wait_probability": immediate_wait_probability,
            "partial_read": partial_read.__dict__,
            "delayed_quarantine": delayed_quarantine.__dict__,
            "resolved_slots": list(resolved),
            "full_capacity_novel_status": full_capacity_novel.status,
            "growth": growth.__dict__,
            "third_promotion": third_receipt.__dict__,
            "eviction": eviction.__dict__,
            "fourth_promotion": fourth_receipt.__dict__,
            "final_slot_ids": list(router.slot_ids),
            "reliability_observation_count": int(reliability.observation_count),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_outcomes": int(reliability.observation_count),
            "unique_wait_outcomes": int(wait_statistics.sample_count),
            "unique_logical_transition_lifetimes": TRAIN_ROWS * 5,
            "wait_sufficient_statistics_updates": 2,
            "reliability_sufficient_statistics_updates": 3,
            "base_optimizer_updates": 0,
            "residual_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "base": model.base.digest(),
            "controller": controller_digest,
            "context_encoder": encoder_digest,
            "router": router.digest(),
            "reliability": reliability.digest(),
            "wait": wait_statistics.digest(),
        },
        "promoted": True,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=89001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
