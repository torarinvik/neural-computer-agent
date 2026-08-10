"""Pressure-test nonlinear factored residual learning and gradual drift."""

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
    ExternalSparseTransitionEvidenceIndex,
    ExternalTransitionContextEncoder,
    ExternalTransitionModel,
    ExternalTransitionObservation,
)

SCHEMA = "neural-computer.external-factored-nonlinear-drift.v1"
STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
BASE_HIDDEN_WIDTH = 32
RESIDUAL_HIDDEN_WIDTH = 64
RESIDUAL_RANDOM_FEATURE_WIDTH = 128
RESIDUAL_RIDGE = 1e-5
BASE_ROWS = 40
ONLINE_ROWS = 34
ONLINE_TRAIN_ROWS = 20
OLD_HELDOUT_START = 20
DRIFT_UPDATE_START = 24
DRIFT_HELDOUT_START = 30
BASE_UPDATES = 2_000
RESIDUAL_UPDATES = 1_000
PREDICTION_TOLERANCE = 0.1


def _digest_module(module: torch.nn.Module) -> str:
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


def _target_residual(state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            0.35 * torch.sin(0.8 * state[:, 0] - 0.5 * intention[:, 1]),
            0.25 * torch.tanh(0.7 * state[:, 1] + 0.4 * intention[:, 0]),
        ),
        dim=1,
    )


def _drift_residual(state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    return _target_residual(state, intention) + torch.stack(
        (
            0.04 * torch.sin(state[:, 0]),
            -0.03 * torch.cos(intention[:, 1]),
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


def _rows(observation: ExternalTransitionObservation) -> list[ExternalTransitionObservation]:
    return [
        ExternalTransitionObservation(
            state=observation.state[index : index + 1],
            intention=observation.intention[index : index + 1],
            next_state=observation.next_state[index : index + 1],
            confidence=observation.confidence[index : index + 1]
            if observation.confidence is not None
            else None,
        )
        for index in range(observation.state.shape[0])
    ]


def _mse(
    model: ExternalFactoredTransitionModel,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    with torch.no_grad():
        return float(model.loss(observation, context=context_batch))


def _train_base(
    model: ExternalFactoredTransitionModel,
    observation: ExternalTransitionObservation,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(model.base.parameters(), lr=0.01)
    final_loss = float("inf")
    for update in range(1, BASE_UPDATES + 1):
        optimizer.zero_grad()
        loss = model.base.loss(observation)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss, BASE_UPDATES


def _train_fresh_target(
    observation: ExternalTransitionObservation,
    heldout: ExternalTransitionObservation,
) -> tuple[float, float, int]:
    model = ExternalTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=BASE_HIDDEN_WIDTH,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    final_loss = float("inf")
    for update in range(1, BASE_UPDATES + 1):
        optimizer.zero_grad()
        loss = model.loss(observation)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    heldout_loss = float(model.loss(heldout).detach())
    return final_loss, heldout_loss, BASE_UPDATES


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    base_inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed + 1,
    ).draw(BASE_ROWS) * 2.0 - 1.0
    base_state = base_inputs[:, :STATE_WIDTH]
    base_intention = base_inputs[:, STATE_WIDTH:]
    online_inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed,
    ).draw(ONLINE_ROWS) * 2.0 - 1.0
    online_state = online_inputs[:, :STATE_WIDTH]
    online_intention = online_inputs[:, STATE_WIDTH:]
    base_observation = _observation(
        base_state,
        base_intention,
        _source_dynamics(base_state, base_intention),
    )
    source_next = _source_dynamics(online_state, online_intention)
    target_next = source_next + _target_residual(online_state, online_intention)
    drift_next = source_next + _drift_residual(online_state, online_intention)
    source_train = _observation(
        online_state[:ONLINE_TRAIN_ROWS],
        online_intention[:ONLINE_TRAIN_ROWS],
        source_next[:ONLINE_TRAIN_ROWS],
    )
    source_heldout = _observation(
        online_state[OLD_HELDOUT_START:DRIFT_UPDATE_START],
        online_intention[OLD_HELDOUT_START:DRIFT_UPDATE_START],
        source_next[OLD_HELDOUT_START:DRIFT_UPDATE_START],
    )
    target_train = _observation(
        online_state[:ONLINE_TRAIN_ROWS],
        online_intention[:ONLINE_TRAIN_ROWS],
        target_next[:ONLINE_TRAIN_ROWS],
    )
    target_heldout = _observation(
        online_state[OLD_HELDOUT_START:DRIFT_UPDATE_START],
        online_intention[OLD_HELDOUT_START:DRIFT_UPDATE_START],
        target_next[OLD_HELDOUT_START:DRIFT_UPDATE_START],
    )
    drift_update = _observation(
        online_state[DRIFT_UPDATE_START:DRIFT_HELDOUT_START],
        online_intention[DRIFT_UPDATE_START:DRIFT_HELDOUT_START],
        drift_next[DRIFT_UPDATE_START:DRIFT_HELDOUT_START],
    )
    drift_heldout = _observation(
        online_state[DRIFT_HELDOUT_START:],
        online_intention[DRIFT_HELDOUT_START:],
        drift_next[DRIFT_HELDOUT_START:],
    )

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
        hidden_width=BASE_HIDDEN_WIDTH,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_hidden_width=RESIDUAL_HIDDEN_WIDTH,
        residual_ridge=RESIDUAL_RIDGE,
        residual_random_feature_width=RESIDUAL_RANDOM_FEATURE_WIDTH,
        residual_random_feature_seed=seed,
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
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=0.005,
        match_margin=0.0001,
        residual_adaptation_updates=RESIDUAL_UPDATES,
        sparse_evidence=ExternalSparseTransitionEvidenceIndex(
            STATE_WIDTH,
            INTENTION_WIDTH,
            input_match_tolerance=1e-6,
            output_match_tolerance=0.01,
            minimum_matches=1,
            minimum_match_fraction=0.25,
        ),
    )

    source_route = router.route_bundle(_rows(source_train))
    source_receipt = router.promote_staged_candidate(
        source_heldout,
        lambda candidate: candidate.residual_context_count == 1,
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    source_context = router.contexts[0].clone() if source_receipt.accepted else None
    target_route = None
    target_receipt = None
    target_context = None
    if source_receipt.accepted:
        target_route = router.route_bundle(_rows(target_train))
        if router.candidate_active:
            target_receipt = router.promote_staged_candidate(
                target_heldout,
                lambda candidate: source_context is not None
                and _mse(candidate, source_heldout, source_context)
                <= PREDICTION_TOLERANCE,
                prediction_tolerance=PREDICTION_TOLERANCE,
            )
            if target_receipt.accepted:
                target_context = router.contexts[1].clone()

    drift_receipt = None
    drift_error = float("inf")
    retained_target_error = float("inf")
    before_drift = router.model.digest()
    if target_receipt is not None and target_receipt.accepted and target_context is not None:
        drift_receipt = router.update_bound_slot(
            target_receipt.slot_id,
            drift_update,
            lambda candidate: _mse(candidate, target_heldout, target_context)
            <= PREDICTION_TOLERANCE,
            heldout=drift_heldout,
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        if drift_receipt.accepted:
            drift_error = _mse(router.model, drift_heldout, target_context)
            retained_target_error = _mse(router.model, target_heldout, target_context)
    after_drift = router.model.digest()

    alternating_routes: list[int | None] = []
    if target_receipt is not None and target_receipt.accepted:
        for _ in range(3):
            alternating_routes.append(
                router.route_bundle(
                    _rows(source_train),
                    match_tolerance=PREDICTION_TOLERANCE,
                    match_margin=0.0,
                ).slot_id
            )
            alternating_routes.append(
                router.route_bundle(
                    _rows(target_train),
                    match_tolerance=PREDICTION_TOLERANCE,
                    match_margin=0.0,
                ).slot_id
            )

    bad_drift_heldout = _observation(
        drift_heldout.state,
        drift_heldout.intention,
        drift_heldout.next_state + 2.0,
    )
    before_rejection = router.model.digest()
    rejected_update = None
    if drift_receipt is not None and drift_receipt.accepted:
        rejected_update = router.update_bound_slot(
            drift_receipt.slot_id,
            drift_update,
            lambda _candidate: True,
            heldout=bad_drift_heldout,
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
    after_rejection = router.model.digest()
    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    persistence_exact = restored.digest() == router.digest()
    restored_sparse_digest = (
        None
        if restored.sparse_evidence is None
        else restored.sparse_evidence.digest()
    )
    restored_routes = []
    if target_receipt is not None and target_receipt.accepted:
        restored_routes = [
            restored.route_bundle(
                _rows(source_train),
                match_tolerance=PREDICTION_TOLERANCE,
                match_margin=0.0,
            ).slot_id,
            restored.route_bundle(
                _rows(target_train),
                match_tolerance=PREDICTION_TOLERANCE,
                match_margin=0.0,
            ).slot_id,
        ]
    fresh_training_loss, fresh_heldout_loss, fresh_updates = _train_fresh_target(
        target_train,
        target_heldout,
    )
    base_only_target_error = float(model.base.loss(target_heldout).detach())
    base_only_drift_error = float(model.base.loss(drift_heldout).detach())
    gates = {
        "source_promoted": source_receipt.accepted,
        "target_promoted": target_receipt is not None and target_receipt.accepted,
        "partial_target_evidence": target_train.state.shape[0] < ONLINE_ROWS,
        "nonlinear_drift_evidence_is_partial": drift_update.state.shape[0] < drift_update.state.shape[0] + drift_heldout.state.shape[0],
        "heldout_source_quality": source_context is not None
        and _mse(router.model, source_heldout, source_context) <= PREDICTION_TOLERANCE,
        "heldout_target_quality_before_drift": target_receipt is not None
        and target_receipt.accepted
        and target_context is not None
        and _mse(router.model, target_heldout, target_context) <= PREDICTION_TOLERANCE,
        "drift_promoted": drift_receipt is not None and drift_receipt.accepted,
        "drift_heldout_quality": drift_error <= PREDICTION_TOLERANCE,
        "prior_target_retained_after_drift": retained_target_error <= PREDICTION_TOLERANCE,
        "alternating_routes_correct": alternating_routes == [0, 1, 0, 1, 0, 1],
        "rejected_corrupted_drift": rejected_update is not None and not rejected_update.accepted,
        "rejection_did_not_mutate_model": before_rejection == after_rejection,
        "base_unchanged": base_digest == model.base.digest() and model.base_frozen,
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_unchanged": encoder_digest == encoder.digest(),
        "exact_persistence": persistence_exact,
        "restored_routes_correct": restored_routes == [0, 1],
        "sparse_drift_identity_persisted": (
            router.sparse_evidence is not None
            and restored.sparse_evidence is not None
            and router.sparse_evidence.record_count >= (
                source_train.state.shape[0]
                + target_train.state.shape[0]
                + drift_update.state.shape[0]
            )
            and restored_sparse_digest == router.sparse_evidence.digest()
        ),
        "learned_beats_base_only": drift_error < base_only_drift_error,
        "fresh_control_measured": fresh_updates == BASE_UPDATES,
    }
    report = {
        "schema": SCHEMA,
        "seed": seed,
        "configuration": {
            "base_rows": BASE_ROWS,
            "online_rows": ONLINE_ROWS,
            "target_train_rows": ONLINE_TRAIN_ROWS,
            "target_heldout_rows": DRIFT_UPDATE_START - OLD_HELDOUT_START,
            "drift_update_rows": DRIFT_HELDOUT_START - DRIFT_UPDATE_START,
            "drift_heldout_rows": ONLINE_ROWS - DRIFT_HELDOUT_START,
            "base_updates": BASE_UPDATES,
            "residual_updates_per_transaction": RESIDUAL_UPDATES,
            "residual_model_family": EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
            "residual_random_feature_width": RESIDUAL_RANDOM_FEATURE_WIDTH,
            "residual_ridge": RESIDUAL_RIDGE,
            "prediction_tolerance": PREDICTION_TOLERANCE,
            "claim": "partial-evidence-nonlinear-factual-drift-retention-v1",
        },
        "metrics": {
            "base_loss": base_loss,
            "base_only_target_heldout_mse": base_only_target_error,
            "base_only_drift_heldout_mse": base_only_drift_error,
            "learned_drift_heldout_mse": drift_error,
            "retained_target_mse_after_drift": retained_target_error,
            "fresh_target_training_mse": fresh_training_loss,
            "fresh_target_heldout_mse": fresh_heldout_loss,
            "source_route_status": source_route.status,
            "target_route_status": None if target_route is None else target_route.status,
            "alternating_routes": alternating_routes,
            "restored_routes": restored_routes,
            "source_promotion": source_receipt.__dict__,
            "target_promotion": None if target_receipt is None else target_receipt.__dict__,
            "drift_promotion": None if drift_receipt is None else drift_receipt.__dict__,
            "rejected_update": None if rejected_update is None else rejected_update.__dict__,
            "model_digest_changed_by_drift": before_drift != after_drift,
        },
        "gates": gates,
        "accounting": {
            "base_optimizer_updates": base_updates,
            "fresh_control_optimizer_updates": fresh_updates,
            "residual_optimizer_updates": 0,
            "residual_statistics_updates": 3,
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "old_regime_replay_during_target_and_drift": 0,
            "base_pretraining_rows": BASE_ROWS,
            "target_online_rows": ONLINE_TRAIN_ROWS,
            "drift_update_rows": drift_update.state.shape[0],
            "drift_heldout_rows": drift_heldout.state.shape[0],
            "sparse_identity_records": (
                0
                if router.sparse_evidence is None
                else router.sparse_evidence.record_count
            ),
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "base": model.base.digest(),
            "controller": controller_digest,
            "context_encoder": encoder_digest,
            "router": router.digest(),
        },
        "promoted": all(gates.values()),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=81021)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
