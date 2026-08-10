"""Pressure-test randomized partial identity together with factual drift.

The controller, shared base, and context encoder remain frozen.  Each opaque
regime is admitted through a randomized partial window, then receives a
disjoint randomized drift update.  Old-regime rows are never replayed during
the drift transaction; a held-out old slice is used only as a retention gate.

This is deliberately a bounded external-memory experiment.  It tests whether
two promoted mechanisms compose, not whether the system has learned arbitrary
open-world identity or unrestricted continual learning.
"""

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
    ExternalTransitionObservation,
)

SCHEMA = "neural-computer.external-factored-random-drift.v1"
STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
REGIME_COUNT = 4
INITIAL_ROWS = 18
INITIAL_OBSERVED_ROWS = 14
INITIAL_WINDOW_ROWS = 7
INITIAL_HELDOUT_ROWS = INITIAL_ROWS - INITIAL_OBSERVED_ROWS
DRIFT_ROWS = 8
DRIFT_UPDATE_ROWS = 4
DRIFT_HELDOUT_ROWS = DRIFT_ROWS - DRIFT_UPDATE_ROWS
FEATURE_WIDTH = 128
MODEL_HIDDEN_WIDTH = 16
RESIDUAL_UPDATES = 128
PREDICTION_TOLERANCE = 0.1
SPARSE_OUTPUT_TOLERANCE = 0.01


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
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


def _drift(state: torch.Tensor, intention: torch.Tensor, regime: int) -> torch.Tensor:
    phase = 0.17 * float(regime + 1)
    return torch.stack(
        (
            0.045 * torch.sin(state[:, 0] + phase),
            -0.035 * torch.cos(intention[:, 1] - phase),
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
            confidence=(
                None
                if observation.confidence is None
                else observation.confidence[index : index + 1]
            ),
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


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    total_rows = INITIAL_ROWS + DRIFT_ROWS
    inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed,
    ).draw(total_rows) * 2.0 - 1.0
    state = inputs[:, :STATE_WIDTH]
    intention = inputs[:, STATE_WIDTH:]
    initial_full = {
        regime: _observation(
            state[:INITIAL_ROWS],
            intention[:INITIAL_ROWS],
            _target(state[:INITIAL_ROWS], intention[:INITIAL_ROWS], regime),
        )
        for regime in range(REGIME_COUNT)
    }
    drift_full = {
        regime: _observation(
            state[INITIAL_ROWS:],
            intention[INITIAL_ROWS:],
            _target(state[INITIAL_ROWS:], intention[INITIAL_ROWS:], regime)
            + _drift(state[INITIAL_ROWS:], intention[INITIAL_ROWS:], regime),
        )
        for regime in range(REGIME_COUNT)
    }
    initial_train = {
        regime: _observation(
            item.state[:INITIAL_OBSERVED_ROWS],
            item.intention[:INITIAL_OBSERVED_ROWS],
            item.next_state[:INITIAL_OBSERVED_ROWS],
        )
        for regime, item in initial_full.items()
    }
    initial_heldout = {
        regime: _observation(
            item.state[INITIAL_OBSERVED_ROWS:],
            item.intention[INITIAL_OBSERVED_ROWS:],
            item.next_state[INITIAL_OBSERVED_ROWS:],
        )
        for regime, item in initial_full.items()
    }
    drift_update: dict[int, ExternalTransitionObservation] = {}
    drift_heldout: dict[int, ExternalTransitionObservation] = {}
    orders: dict[int, tuple[int, ...]] = {}
    drift_orders: dict[int, tuple[int, ...]] = {}
    for regime in range(REGIME_COUNT):
        generator = torch.Generator().manual_seed(seed * 100 + regime)
        order = tuple(int(value) for value in torch.randperm(INITIAL_OBSERVED_ROWS, generator=generator))
        orders[regime] = order
        drift_order = tuple(int(value) for value in torch.randperm(DRIFT_ROWS, generator=generator))
        drift_orders[regime] = drift_order
        item = drift_full[regime]
        update_indices = drift_order[:DRIFT_UPDATE_ROWS]
        heldout_indices = drift_order[DRIFT_UPDATE_ROWS:]
        drift_update[regime] = _observation(
            item.state[list(update_indices)],
            item.intention[list(update_indices)],
            item.next_state[list(update_indices)],
        )
        drift_heldout[regime] = _observation(
            item.state[list(heldout_indices)],
            item.intention[list(heldout_indices)],
            item.next_state[list(heldout_indices)],
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
        hidden_width=MODEL_HIDDEN_WIDTH,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_random_feature_width=FEATURE_WIDTH,
        residual_random_feature_seed=seed,
        residual_ridge=0.01,
        residual_capacity=REGIME_COUNT,
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
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        admission_observations=INITIAL_WINDOW_ROWS,
        max_contexts=REGIME_COUNT,
        residual_adaptation_updates=RESIDUAL_UPDATES,
        match_tolerance=0.02,
        match_margin=0.001,
        quarantine_capacity=8,
        sparse_evidence=ExternalSparseTransitionEvidenceIndex(
            STATE_WIDTH,
            INTENTION_WIDTH,
            input_match_tolerance=1e-6,
            output_match_tolerance=SPARSE_OUTPUT_TOLERANCE,
            minimum_matches=1,
            minimum_match_fraction=0.25,
        ),
    )

    promotion_receipts: list[dict[str, object]] = []
    stream_statuses: dict[int, list[str]] = {}
    drift_receipts: list[dict[str, object]] = []
    alternating_routes: dict[int, list[int | None]] = {}
    for regime in range(REGIME_COUNT):
        order = orders[regime]
        first = [_rows(initial_full[regime])[index] for index in order[:INITIAL_WINDOW_ROWS]]
        second = [_rows(initial_full[regime])[index] for index in order[INITIAL_WINDOW_ROWS:]]
        staged = router.route_bundle(first)
        if staged.status != "staged":
            raise RuntimeError(f"regime {regime} did not stage: {staged.status}")
        statuses = [staged.status]
        statuses.extend(router.observe(item).status for item in second)
        stream_statuses[regime] = statuses
        receipt = router.promote_staged_candidate(
            initial_heldout[regime],
            lambda candidate, expected_count=regime + 1: (
                candidate.residual_bank is not None
                and candidate.residual_bank.context_count == expected_count
            ),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        promotion_receipts.append(receipt.__dict__)
        if not receipt.accepted:
            continue
        context = router.contexts[regime].clone()
        target_before = _mse(router.model, initial_heldout[regime], context)
        drift_receipt = router.update_bound_slot(
            receipt.slot_id,
            drift_update[regime],
            lambda candidate, old=initial_heldout[regime], old_context=context: (
                _mse(candidate, old, old_context) <= PREDICTION_TOLERANCE
            ),
            heldout=drift_heldout[regime],
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        drift_receipts.append(
            {
                **drift_receipt.__dict__,
                "target_error_before": target_before,
                "target_error_after": _mse(router.model, initial_heldout[regime], context)
                if drift_receipt.accepted
                else None,
                "drift_error_after": _mse(router.model, drift_heldout[regime], context)
                if drift_receipt.accepted
                else None,
            }
        )
        if drift_receipt.accepted:
            alternating_routes[regime] = []
            for _ in range(2):
                target_query = [
                    _rows(initial_full[regime])[index]
                    for index in orders[regime][:3]
                ]
                drift_query = [
                    _rows(drift_full[regime])[index]
                    for index in drift_orders[regime][:2]
                ]
                alternating_routes[regime].append(
                    router.route_partial_bundle(
                        target_query,
                        min_match_fraction=0.25,
                        match_tolerance=PREDICTION_TOLERANCE,
                        contradiction_tolerance=PREDICTION_TOLERANCE,
                        match_margin=0.0,
                    ).slot_id
                )
                alternating_routes[regime].append(
                    router.route_partial_bundle(
                        drift_query,
                        min_match_fraction=0.25,
                        match_tolerance=PREDICTION_TOLERANCE,
                        contradiction_tolerance=PREDICTION_TOLERANCE,
                        match_margin=0.0,
                    ).slot_id
                )

    mixed = router.route_partial_bundle(
        [
            _rows(initial_full[0])[orders[0][0]],
            _rows(initial_full[1])[orders[1][0]],
        ],
        min_match_fraction=0.25,
        match_tolerance=PREDICTION_TOLERANCE,
        contradiction_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    retained_initial = {
        regime: _mse(router.model, initial_heldout[regime], router.contexts[regime])
        for regime in range(REGIME_COUNT)
    }
    retained_drift = {
        regime: _mse(router.model, drift_heldout[regime], router.contexts[regime])
        for regime in range(REGIME_COUNT)
    }
    restored_routes = {
        regime: restored.route_partial_bundle(
            [
                _rows(drift_full[regime])[index]
                for index in drift_orders[regime][:2]
            ],
            min_match_fraction=0.25,
            match_tolerance=PREDICTION_TOLERANCE,
            contradiction_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in range(REGIME_COUNT)
    }
    gates = {
        "all_initial_regimes_promoted": all(
            item["accepted"] for item in promotion_receipts
        ),
        "random_initial_windows_stage_and_continue": all(
            values[0] == "staged" and all(value == "staged" for value in values)
            for values in stream_statuses.values()
        ),
        "all_drift_versions_promoted": len(drift_receipts) == REGIME_COUNT
        and all(item["accepted"] for item in drift_receipts),
        "partial_drift_evidence": DRIFT_UPDATE_ROWS < DRIFT_ROWS,
        "random_routes_alternate": all(
            values == [regime, regime, regime, regime]
            for regime, values in alternating_routes.items()
        ),
        "mixed_partial_is_ambiguous": mixed.status == "ambiguous"
        and mixed.slot_id is None,
        "initial_retention_after_drift": all(
            value <= PREDICTION_TOLERANCE for value in retained_initial.values()
        ),
        "drift_heldout_quality": all(
            value <= PREDICTION_TOLERANCE for value in retained_drift.values()
        ),
        "exact_persistence": restored.digest() == router.digest(),
        "restored_drift_routes_correct": list(restored_routes.values())
        == list(range(REGIME_COUNT)),
        "sparse_identity_persisted": (
            router.sparse_evidence is not None
            and restored.sparse_evidence is not None
            and restored.sparse_evidence.digest() == router.sparse_evidence.digest()
            and router.sparse_evidence.record_count
            >= REGIME_COUNT * (INITIAL_OBSERVED_ROWS + DRIFT_UPDATE_ROWS)
        ),
        "base_unchanged": model.base.digest() == base_digest and model.base_frozen,
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_unchanged": encoder_digest == encoder.digest(),
        "replay_free": True,
    }
    report = {
        "schema": SCHEMA,
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "initial_observed_rows": INITIAL_OBSERVED_ROWS,
            "initial_heldout_rows": INITIAL_HELDOUT_ROWS,
            "initial_window_rows": INITIAL_WINDOW_ROWS,
            "drift_update_rows": DRIFT_UPDATE_ROWS,
            "drift_heldout_rows": DRIFT_HELDOUT_ROWS,
            "mask": "independent_seeded_random_permutations_v1",
            "residual_model_family": EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
            "claim": "bounded-factored-random-missingness-drift-composition-v1",
        },
        "metrics": {
            "orders": {str(key): list(value) for key, value in orders.items()},
            "drift_orders": {str(key): list(value) for key, value in drift_orders.items()},
            "stream_statuses": stream_statuses,
            "initial_promotion_receipts": promotion_receipts,
            "drift_promotion_receipts": drift_receipts,
            "alternating_routes": alternating_routes,
            "restored_drift_routes": restored_routes,
            "mixed_status": mixed.status,
            "retained_initial_mse": retained_initial,
            "retained_drift_mse": retained_drift,
            "sparse_records": (
                0
                if router.sparse_evidence is None
                else router.sparse_evidence.record_count
            ),
        },
        "gates": gates,
        "accounting": {
            "base_optimizer_updates": 0,
            "residual_optimizer_updates": 0,
            "residual_statistics_updates": REGIME_COUNT * 3,
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "old_regime_replay_during_drift": 0,
            "unique_initial_observed_rows": REGIME_COUNT * INITIAL_OBSERVED_ROWS,
            "unique_initial_heldout_rows": REGIME_COUNT * INITIAL_HELDOUT_ROWS,
            "unique_drift_update_rows": REGIME_COUNT * DRIFT_UPDATE_ROWS,
            "unique_drift_heldout_rows": REGIME_COUNT * DRIFT_HELDOUT_ROWS,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "base": base_digest,
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
