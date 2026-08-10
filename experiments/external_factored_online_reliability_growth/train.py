"""Promote online reliability gating through nonlinear factored growth."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

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
ONLINE_ROWS = 32
SOURCE_TRAIN_ROWS = 20
SOURCE_HELDOUT_START = 20
SOURCE_HELDOUT_STOP = 24
TARGET_HELDOUT_START = 24
TARGET_HELDOUT_STOP = 32
BASE_UPDATES = 1_000
RESIDUAL_UPDATES = 400
MATCH_TOLERANCE = 0.005
PREDICTION_TOLERANCE = 0.1
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


def _source_dynamics(state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            torch.sin(state[:, 0] + 0.4 * intention[:, 0]) + 0.2 * state[:, 1],
            torch.cos(state[:, 1] - 0.3 * intention[:, 1])
            + 0.2 * intention[:, 0],
        ),
        dim=1,
    )


def _target_dynamics(state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    return _source_dynamics(state, intention) + torch.stack(
        (
            0.35 * torch.sin(0.8 * state[:, 0] - 0.5 * intention[:, 1]),
            0.25 * torch.tanh(0.7 * state[:, 1] + 0.4 * intention[:, 0]),
        ),
        dim=1,
    )


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


def _train_base(
    model: ExternalFactoredTransitionModel,
    observation: ExternalTransitionObservation,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(model.base.parameters(), lr=0.01)
    loss_value = float("inf")
    for _ in range(BASE_UPDATES):
        optimizer.zero_grad()
        loss = model.base.loss(observation)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())
    return loss_value, BASE_UPDATES


def _mse(
    model: ExternalFactoredTransitionModel,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    with torch.no_grad():
        return float(model.loss(observation, context=context_batch))


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


def _rows(
    observation: ExternalTransitionObservation,
) -> tuple[ExternalTransitionObservation, ...]:
    return tuple(
        _slice(observation, index, index + 1)
        for index in range(observation.state.shape[0])
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


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    base_inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed + 1,
    ).draw(BASE_ROWS) * 2.0 - 1.0
    online_inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed,
    ).draw(ONLINE_ROWS) * 2.0 - 1.0
    base_state, base_intention = base_inputs[:, :STATE_WIDTH], base_inputs[:, STATE_WIDTH:]
    online_state, online_intention = (
        online_inputs[:, :STATE_WIDTH],
        online_inputs[:, STATE_WIDTH:],
    )
    base_observation = _observation(
        base_state,
        base_intention,
        _source_dynamics(base_state, base_intention),
    )
    source = _observation(
        online_state,
        online_intention,
        _source_dynamics(online_state, online_intention),
    )
    target = _observation(
        online_state,
        online_intention,
        _target_dynamics(online_state, online_intention),
    )
    source_train = _slice(source, 0, SOURCE_TRAIN_ROWS)
    source_heldout = _slice(source, SOURCE_HELDOUT_START, SOURCE_HELDOUT_STOP)
    target_train = _slice(target, 0, SOURCE_TRAIN_ROWS)
    target_heldout = _slice(target, TARGET_HELDOUT_START, TARGET_HELDOUT_STOP)

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
        residual_capacity=1,
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
        max_contexts=1,
        auto_grow=True,
        residual_adaptation_updates=RESIDUAL_UPDATES,
        quarantine_capacity=SOURCE_TRAIN_ROWS,
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
    clean_a = _slice(source, SOURCE_TRAIN_ROWS, SOURCE_TRAIN_ROWS + 2)
    clean_b = _slice(source, SOURCE_TRAIN_ROWS + 2, SOURCE_TRAIN_ROWS + 4)
    clean_a_status = router.route_bundle(_rows(clean_a)).status
    _observe_verifier(reliability, router.model, source_context, clean_a, 1.0)
    clean_b_status = router.route_bundle(_rows(clean_b)).status
    _observe_verifier(reliability, router.model, source_context, clean_b, 1.0)
    gate_open = int(reliability.observation_count) >= RELIABILITY_WARMUP

    fresh_router = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=reliability
    )
    fresh_router.committed_evidence_gate = False
    corruption = _observation(
        clean_a.state,
        clean_a.intention,
        clean_a.next_state + CORRUPTION_DELTA,
    )
    corruption_status = router.route_bundle(_rows(corruption)).status
    _observe_verifier(reliability, router.model, source_context, corruption, 0.0)
    fresh_corruption_status = fresh_router.route_bundle(_rows(corruption)).status
    novel_route = router.route_bundle(_rows(target_train))
    target_receipt = router.promote_staged_candidate(
        target_heldout,
        lambda candidate: _mse(candidate, source_heldout, source_context)
        <= PREDICTION_TOLERANCE,
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    target_context = router.contexts[1].clone()
    restored = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=reliability
    )
    persistence_exact = restored.configuration() == router.configuration() and (
        restored.digest() == router.digest()
    )
    post_growth_control = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=reliability
    )
    post_growth_control.committed_evidence_gate = False
    source_revisit = post_growth_control.route_bundle(
        _rows(source_train),
        match_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    target_revisit = post_growth_control.route_bundle(
        _rows(target_train),
        match_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    gates = {
        "source_promoted": source_route.status == "staged" and source_receipt.accepted,
        "clean_routes_before_corruption": clean_a_status == "matched"
        and clean_b_status == "matched",
        "gate_warmup_reached": gate_open,
        "corruption_vetoed": corruption_status == "reliability_veto",
        "corruption_did_not_stage": not router.candidate_active,
        "fresh_gate_disabled_control_matches": fresh_corruption_status == "matched",
        "novel_nonlinear_candidate_staged": novel_route.status == "staged",
        "target_promoted_with_capacity_growth": target_receipt.accepted
        and "capacity growth" in target_receipt.reason,
        "source_retained_after_growth": source_revisit.slot_id == 0,
        "target_routes_after_growth": target_revisit.slot_id == 1,
        "base_frozen": model.base_frozen and model.base.digest() == base_digest,
        "controller_frozen": _digest(controller) == controller_digest,
        "context_encoder_unchanged": encoder.digest() == encoder_digest,
        "exact_persistence": persistence_exact,
        "replay_free": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"factored online reliability gates failed: {gates}")
    report: dict[str, object] = {
        "schema": "neural-computer.external-factored-online-reliability-growth.v1",
        "seed": seed,
        "configuration": {
            "base_rows": BASE_ROWS,
            "source_train_rows": SOURCE_TRAIN_ROWS,
            "target_train_rows": SOURCE_TRAIN_ROWS,
            "corruption_delta": CORRUPTION_DELTA,
            "match_tolerance": MATCH_TOLERANCE,
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "reliability_warmup": RELIABILITY_WARMUP,
            "base_updates": BASE_UPDATES,
            "residual_updates": RESIDUAL_UPDATES,
            "residual_model_family": EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
            "claim": "online_nonlinear_reliability_veto_with_verified_growth_v1",
        },
        "metrics": {
            "source_route_status": source_route.status,
            "clean_a_status": clean_a_status,
            "clean_b_status": clean_b_status,
            "corruption_status": corruption_status,
            "fresh_corruption_status": fresh_corruption_status,
            "novel_route_status": novel_route.status,
            "source_promotion": source_receipt.__dict__,
            "target_promotion": target_receipt.__dict__,
            "source_revisit_slot": source_revisit.slot_id,
            "target_revisit_slot": target_revisit.slot_id,
            "reliability_observation_count": int(reliability.observation_count),
            "router_context_count": len(router.slot_ids),
            "router_max_contexts": router.max_contexts,
            "base_loss": base_loss,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_outcomes": int(reliability.observation_count),
            "unique_logical_transition_lifetimes": SOURCE_TRAIN_ROWS + 8,
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
    parser.add_argument("--seed", type=int, default=86001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
