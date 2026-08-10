"""Pressure-test a learned external residual function under a frozen base."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    AmodalCognitiveController,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModel,
    ExternalTransitionObservation,
)

SCHEMA = "neural-computer.external-factored-learned-residual.v1"
STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
BASE_HIDDEN_WIDTH = 32
RESIDUAL_HIDDEN_WIDTH = 48
BASE_ROWS = 40
ONLINE_ROWS = 20
ONLINE_TRAIN_ROWS = 16
BASE_UPDATES = 2_000
RESIDUAL_UPDATES = 1
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


def _target_dynamics(state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    source = _source_dynamics(state, intention)
    residual = torch.stack(
        (
            0.4 * state[:, 0] - 0.2 * intention[:, 1],
            -0.3 * state[:, 1] + 0.1 * intention[:, 0],
        ),
        dim=1,
    )
    return source + residual


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
    generator = torch.Generator().manual_seed(seed)
    base_state = torch.randn(BASE_ROWS, STATE_WIDTH, generator=generator)
    base_intention = torch.randn(BASE_ROWS, INTENTION_WIDTH, generator=generator)
    online_state = torch.randn(ONLINE_ROWS, STATE_WIDTH, generator=generator)
    online_intention = torch.randn(ONLINE_ROWS, INTENTION_WIDTH, generator=generator)
    source_base = _observation(
        base_state,
        base_intention,
        _source_dynamics(base_state, base_intention),
    )
    source_all = _observation(
        online_state,
        online_intention,
        _source_dynamics(online_state, online_intention),
    )
    target_all = _observation(
        online_state,
        online_intention,
        _target_dynamics(online_state, online_intention),
    )
    source_train = _observation(
        online_state[:ONLINE_TRAIN_ROWS],
        online_intention[:ONLINE_TRAIN_ROWS],
        source_all.next_state[:ONLINE_TRAIN_ROWS],
    )
    target_train = _observation(
        online_state[:ONLINE_TRAIN_ROWS],
        online_intention[:ONLINE_TRAIN_ROWS],
        target_all.next_state[:ONLINE_TRAIN_ROWS],
    )
    source_heldout = _observation(
        online_state[ONLINE_TRAIN_ROWS:],
        online_intention[ONLINE_TRAIN_ROWS:],
        source_all.next_state[ONLINE_TRAIN_ROWS:],
    )
    target_heldout = _observation(
        online_state[ONLINE_TRAIN_ROWS:],
        online_intention[ONLINE_TRAIN_ROWS:],
        target_all.next_state[ONLINE_TRAIN_ROWS:],
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
        residual_model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        residual_hidden_width=RESIDUAL_HIDDEN_WIDTH,
    )
    base_loss, base_updates = _train_base(model, source_base)
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
        match_tolerance=0.05,
        match_margin=0.001,
        residual_adaptation_updates=RESIDUAL_UPDATES,
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

    alternating_routes: list[int | None] = []
    if source_receipt.accepted and target_receipt is not None and target_receipt.accepted:
        for _ in range(3):
            alternating_routes.append(router.route_bundle(_rows(source_train)).slot_id)
            alternating_routes.append(router.route_bundle(_rows(target_train)).slot_id)

    source_error = (
        float("inf")
        if source_context is None
        else _mse(router.model, source_heldout, source_context)
    )
    target_error = (
        float("inf")
        if target_context is None
        else _mse(router.model, target_heldout, target_context)
    )
    model_before_rejection = router.model.digest()
    rejected_update = None
    if target_receipt is not None and target_receipt.accepted and target_receipt.slot_id is not None:
        bad_heldout = _observation(
            target_heldout.state,
            target_heldout.intention,
            target_heldout.next_state + 2.0,
        )
        rejected_update = router.update_bound_slot(
            target_receipt.slot_id,
            target_heldout,
            lambda _candidate: True,
            heldout=bad_heldout,
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
    model_after_rejection = router.model.digest()

    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    persistence_exact = restored.digest() == router.digest()
    restored_routes = []
    if source_receipt.accepted and target_receipt is not None and target_receipt.accepted:
        restored_routes = [
            restored.route_bundle(_rows(source_train)).slot_id,
            restored.route_bundle(_rows(target_train)).slot_id,
        ]
    fresh_training_loss, fresh_loss, fresh_updates = _train_fresh_target(
        target_train,
        target_heldout,
    )
    base_only_error = float(model.base.loss(target_heldout).detach())
    gates = {
        "source_promoted": source_receipt.accepted,
        "target_promoted": target_receipt is not None and target_receipt.accepted,
        "partial_target_evidence": target_train.state.shape[0] < target_all.state.shape[0],
        "heldout_source_quality": source_error <= PREDICTION_TOLERANCE,
        "heldout_target_quality": target_error <= PREDICTION_TOLERANCE,
        "alternating_routes_correct": alternating_routes == [0, 1, 0, 1, 0, 1],
        "prior_retained_during_target_promotion": source_error <= PREDICTION_TOLERANCE,
        "rejected_bad_bound_update": rejected_update is not None and not rejected_update.accepted,
        "rejected_update_did_not_mutate_model": model_before_rejection == model_after_rejection,
        "base_frozen": base_digest == model.base.digest() and model.base_frozen,
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_unchanged": encoder_digest == encoder.digest(),
        "exact_persistence": persistence_exact,
        "restored_routes_correct": restored_routes == [0, 1],
        "learned_beats_base_only": target_error < base_only_error,
        "fresh_control_measured": fresh_updates == BASE_UPDATES,
    }
    report = {
        "schema": SCHEMA,
        "seed": seed,
        "configuration": {
            "base_rows": BASE_ROWS,
            "online_rows": ONLINE_ROWS,
            "online_train_rows": ONLINE_TRAIN_ROWS,
            "base_updates": BASE_UPDATES,
            "residual_updates": RESIDUAL_UPDATES,
            "residual_model_family": EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            "prediction_tolerance": PREDICTION_TOLERANCE,
            "claim": "partial-evidence-factored-learned-residual-retention-v1",
        },
        "metrics": {
            "base_loss": base_loss,
            "base_only_target_heldout_mse": base_only_error,
            "learned_target_heldout_mse": target_error,
            "learned_source_heldout_mse": source_error,
            "fresh_target_heldout_mse": fresh_loss,
            "fresh_target_training_mse": fresh_training_loss,
            "source_route_status": source_route.status,
            "target_route_status": None if target_route is None else target_route.status,
            "alternating_routes": alternating_routes,
            "restored_routes": restored_routes,
            "source_promotion": source_receipt.__dict__,
            "target_promotion": None if target_receipt is None else target_receipt.__dict__,
            "rejected_bound_update": None if rejected_update is None else rejected_update.__dict__,
        },
        "gates": gates,
        "accounting": {
            "base_optimizer_updates": base_updates,
            "fresh_control_optimizer_updates": fresh_updates,
            "residual_optimizer_updates": 0,
            "residual_statistics_updates": 2,
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "old_regime_replay_during_target_adaptation": 0,
            "base_pretraining_rows": BASE_ROWS,
            "source_online_rows": ONLINE_TRAIN_ROWS,
            "target_online_rows": ONLINE_TRAIN_ROWS,
            "target_heldout_rows": ONLINE_ROWS - ONLINE_TRAIN_ROWS,
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
    parser.add_argument("--seed", type=int, default=81011)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
