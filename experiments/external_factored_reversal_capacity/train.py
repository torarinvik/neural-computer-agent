"""Adversarial reversal and capacity-pressure audit for factored memory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.external_factored_online_reliability_growth.train import (
    _digest,
    _mse,
    _observation,
    _rows,
    _slice,
    _source_dynamics,
    _train_base,
)
from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    AmodalCognitiveController,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionEvidenceStatistics,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
BASE_ROWS = 48
ONLINE_ROWS = 48
TRAIN_ROWS = 16
HELDOUT_START = 16
HELDOUT_STOP = 20
BASE_UPDATES = 1_000
RESIDUAL_UPDATES = 400
MATCH_TOLERANCE = 0.005
PREDICTION_TOLERANCE = 0.1
CORRUPTION_DELTA = 0.04
RELIABILITY_THRESHOLD = 0.9
RELIABILITY_WARMUP = 4


def _target_dynamics(state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    return _source_dynamics(state, intention) + torch.stack(
        (
            0.35 * torch.sin(0.8 * state[:, 0] - 0.5 * intention[:, 1]),
            0.25 * torch.tanh(0.7 * state[:, 1] + 0.4 * intention[:, 0]),
        ),
        dim=1,
    )


def _third_dynamics(state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    return _target_dynamics(state, intention) + torch.stack(
        (
            0.55 * torch.sin(1.1 * state[:, 0] + 0.2 * intention[:, 0]),
            -0.4 * torch.tanh(0.9 * state[:, 1] - 0.3 * intention[:, 1]),
        ),
        dim=1,
    )


def _observe_verifier(
    statistics: ExternalTransitionEvidenceStatistics,
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
    statistics.observe(
        prediction,
        observation.next_state,
        torch.full((observation.state.shape[0],), outcome),
    )


def _fresh_streams(seed: int) -> tuple[
    ExternalTransitionObservation,
    ExternalTransitionObservation,
    ExternalTransitionObservation,
]:
    inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed,
    ).draw(ONLINE_ROWS) * 2.0 - 1.0
    state, intention = inputs[:, :STATE_WIDTH], inputs[:, STATE_WIDTH:]
    return (
        _observation(state, intention, _source_dynamics(state, intention)),
        _observation(state, intention, _target_dynamics(state, intention)),
        _observation(state, intention, _third_dynamics(state, intention)),
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    base_inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed + 1,
    ).draw(BASE_ROWS) * 2.0 - 1.0
    base_state, base_intention = (
        base_inputs[:, :STATE_WIDTH],
        base_inputs[:, STATE_WIDTH:],
    )
    base_observation = _observation(
        base_state,
        base_intention,
        _source_dynamics(base_state, base_intention),
    )
    source, target, third = _fresh_streams(seed)
    source_train = _slice(source, 0, TRAIN_ROWS)
    source_heldout = _slice(source, HELDOUT_START, HELDOUT_STOP)
    target_train = _slice(target, 0, TRAIN_ROWS)
    target_heldout = _slice(target, HELDOUT_START, HELDOUT_STOP)
    third_train = _slice(third, 0, TRAIN_ROWS)
    third_heldout = _slice(third, HELDOUT_START, HELDOUT_STOP)

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
        hidden_width=32,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_hidden_width=64,
        residual_random_feature_width=128,
        residual_random_feature_seed=seed,
        residual_capacity=2,
    )
    base_loss, base_updates = _train_base(model, base_observation)
    model.freeze_base()
    base_digest = model.base.digest()
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=32,
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
        max_contexts=2,
        auto_grow=False,
        residual_adaptation_updates=RESIDUAL_UPDATES,
        quarantine_capacity=TRAIN_ROWS,
        evidence_evaluator=reliability,
        evidence_threshold=RELIABILITY_THRESHOLD,
        evidence_gate_min_evidence=RELIABILITY_WARMUP,
        committed_evidence_gate=True,
    )

    source_route = router.route_bundle(_rows(source_train))
    source_receipt = router.promote_staged_candidate(
        source_heldout,
        lambda candidate: candidate.residual_context_count == 1,
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    source_context = router.contexts[0].clone()
    target_route = router.route_bundle(_rows(target_train))
    target_receipt = router.promote_staged_candidate(
        target_heldout,
        lambda candidate: _mse(candidate, source_heldout, source_context)
        <= PREDICTION_TOLERANCE,
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    target_context = router.contexts[1].clone()

    # These are new verifier outcomes for clean revisits. They warm the
    # reliability state before the adversarial corruption arrives.
    clean_source_status = router.route_bundle(_rows(source_train)).status
    _observe_verifier(reliability, router.model, source_context, source_train, 1.0)
    clean_target_status = router.route_bundle(_rows(target_train)).status
    _observe_verifier(reliability, router.model, target_context, target_train, 1.0)
    gate_open = int(reliability.observation_count) >= RELIABILITY_WARMUP

    fresh_control = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=reliability
    )
    fresh_control.committed_evidence_gate = False
    corrupted_source = _observation(
        source_train.state,
        source_train.intention,
        source_train.next_state + CORRUPTION_DELTA,
    )
    corruption_status = router.route_bundle(_rows(corrupted_source)).status
    _observe_verifier(reliability, router.model, source_context, corrupted_source, 0.0)
    fresh_corruption_status = fresh_control.route_bundle(
        _rows(corrupted_source)
    ).status

    # A real return to a known regime must work while the reliability gate is
    # still active, after corruption has been observed and quarantined.
    source_reversal_status = router.route_bundle(_rows(source_train)).status
    target_reversal_status = router.route_bundle(_rows(target_train)).status

    full_capacity_novel = router.route_bundle(_rows(third_train))
    full_capacity_unchanged = router.slot_ids == (0, 1) and not router.candidate_active
    growth = router.grow_verified(
        3,
        lambda candidate: candidate.context_count == 2,
    )
    novel_route = router.route_bundle(_rows(third_train))
    third_receipt = router.promote_staged_candidate(
        third_heldout,
        lambda candidate: _mse(candidate, source_heldout, source_context)
        <= PREDICTION_TOLERANCE
        and _mse(candidate, target_heldout, target_context) <= PREDICTION_TOLERANCE,
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    source_after_growth = router.route_bundle(_rows(source_train))
    target_after_growth = router.route_bundle(_rows(target_train))
    third_after_growth = router.route_bundle(_rows(third_train))
    restored = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=reliability
    )
    persistence_exact = restored.configuration() == router.configuration() and (
        restored.digest() == router.digest()
    )
    gates = {
        "source_promoted": source_route.status == "staged" and source_receipt.accepted,
        "target_promoted": target_route.status == "staged" and target_receipt.accepted,
        "clean_revisits_route": clean_source_status == "matched"
        and clean_target_status == "matched",
        "gate_warmup_reached": gate_open,
        "corruption_vetoed": corruption_status == "reliability_veto",
        "corruption_did_not_stage": not router.candidate_active,
        "fresh_gate_disabled_control_matches": fresh_corruption_status == "matched",
        "legitimate_source_reversal_routes": source_reversal_status == "matched",
        "legitimate_target_reversal_routes": target_reversal_status == "matched",
        "full_capacity_novelty_refused": full_capacity_novel.status == "ambiguous"
        and full_capacity_unchanged,
        "verified_growth_accepted": growth.accepted and router.max_contexts == 3,
        "novel_candidate_promoted_after_growth": novel_route.status == "staged"
        and third_receipt.accepted,
        "source_retained_after_growth": source_after_growth.slot_id == 0,
        "target_retained_after_growth": target_after_growth.slot_id == 1,
        "third_regime_routes_after_growth": third_after_growth.slot_id == 2,
        "base_frozen": model.base_frozen and model.base.digest() == base_digest,
        "controller_frozen": _digest(controller) == controller_digest,
        "context_encoder_unchanged": encoder.digest() == encoder_digest,
        "exact_persistence": persistence_exact,
        "replay_free": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"reversal/capacity gates failed: {gates}")
    report: dict[str, object] = {
        "schema": "neural-computer.external-factored-reversal-capacity.v1",
        "seed": seed,
        "configuration": {
            "regimes": 3,
            "initial_capacity": 2,
            "grown_capacity": 3,
            "train_rows": TRAIN_ROWS,
            "corruption_delta": CORRUPTION_DELTA,
            "match_tolerance": MATCH_TOLERANCE,
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "reliability_warmup": RELIABILITY_WARMUP,
            "base_updates": BASE_UPDATES,
            "residual_updates": RESIDUAL_UPDATES,
            "claim": "adversarial_reversal_capacity_pressure_v1",
        },
        "metrics": {
            "source_route_status": source_route.status,
            "target_route_status": target_route.status,
            "clean_source_status": clean_source_status,
            "clean_target_status": clean_target_status,
            "corruption_status": corruption_status,
            "fresh_corruption_status": fresh_corruption_status,
            "source_reversal_status": source_reversal_status,
            "target_reversal_status": target_reversal_status,
            "full_capacity_novel_status": full_capacity_novel.status,
            "growth": growth.__dict__,
            "novel_route_status": novel_route.status,
            "third_promotion": third_receipt.__dict__,
            "post_growth_slots": [
                source_after_growth.slot_id,
                target_after_growth.slot_id,
                third_after_growth.slot_id,
            ],
            "reliability_observation_count": int(reliability.observation_count),
            "base_loss": base_loss,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_outcomes": int(reliability.observation_count),
            "unique_logical_transition_lifetimes": TRAIN_ROWS * 6,
            "base_optimizer_updates": base_updates,
            "residual_optimizer_updates": 0,
            "reliability_optimizer_updates": 0,
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
        },
        "promoted": True,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=87001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
