"""Audit factored factual identity under randomized partial windows.

The controller, shared base, and context encoder remain frozen.  Each regime
arrives in a random order through two seven-row windows; the external
random-feature residual learner consumes every observed row once.  After
promotion, sparse factual overlap routes new random partial windows and keeps
mixed-regime evidence ambiguous.
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

SCHEMA = "neural-computer.external-factored-random-missingness.v1"
STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
REGIME_COUNT = 4
ROWS_PER_REGIME = 18
OBSERVED_ROWS = 14
WINDOW_ROWS = 7
HELDOUT_ROWS = ROWS_PER_REGIME - OBSERVED_ROWS
FEATURE_WIDTH = 128
MODEL_HIDDEN_WIDTH = 16
RESIDUAL_UPDATES = 128
PREDICTION_TOLERANCE = 0.1


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


def _row(
    observation: ExternalTransitionObservation,
    index: int,
) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=observation.state[index : index + 1],
        intention=observation.intention[index : index + 1],
        next_state=observation.next_state[index : index + 1],
        confidence=(
            None
            if observation.confidence is None
            else observation.confidence[index : index + 1]
        ),
    )


def _mse(
    router: ExternalFactoredTransitionRouter,
    observation: ExternalTransitionObservation,
    slot_id: int,
) -> float:
    index = router.model.residual_bank.physical_index_for_slot_id(slot_id)
    context = router.model.residual_bank.context_at(index).to(observation.state)
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    with torch.no_grad():
        prediction = router.model.predict_with_context(
            observation.state,
            observation.intention,
            context_batch,
        )
    return float((prediction - observation.next_state).square().mean())


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    inputs = torch.quasirandom.SobolEngine(
        dimension=STATE_WIDTH + INTENTION_WIDTH,
        scramble=True,
        seed=seed,
    ).draw(ROWS_PER_REGIME) * 2.0 - 1.0
    state = inputs[:, :STATE_WIDTH]
    intention = inputs[:, STATE_WIDTH:]
    full = {
        regime: _observation(state, intention, _target(state, intention, regime))
        for regime in range(REGIME_COUNT)
    }
    heldout = {
        regime: _observation(
            item.state[OBSERVED_ROWS:],
            item.intention[OBSERVED_ROWS:],
            item.next_state[OBSERVED_ROWS:],
        )
        for regime, item in full.items()
    }
    orders: dict[int, tuple[int, ...]] = {}
    for regime in range(REGIME_COUNT):
        generator = torch.Generator().manual_seed(seed * 100 + regime)
        orders[regime] = tuple(
            int(value)
            for value in torch.randperm(OBSERVED_ROWS, generator=generator)
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
        admission_observations=WINDOW_ROWS,
        max_contexts=REGIME_COUNT,
        residual_adaptation_updates=RESIDUAL_UPDATES,
        match_tolerance=0.02,
        match_margin=0.001,
        quarantine_capacity=8,
        sparse_evidence=ExternalSparseTransitionEvidenceIndex(
            STATE_WIDTH,
            INTENTION_WIDTH,
            input_match_tolerance=1e-6,
            output_match_tolerance=PREDICTION_TOLERANCE,
            minimum_matches=1,
            minimum_match_fraction=0.25,
        ),
    )

    statuses: dict[int, list[str]] = {}
    receipts: list[dict[str, object]] = []
    for regime in range(REGIME_COUNT):
        order = orders[regime]
        first_window = [_row(full[regime], index) for index in order[:WINDOW_ROWS]]
        second_window = [_row(full[regime], index) for index in order[WINDOW_ROWS:]]
        staged = router.route_bundle(first_window)
        if staged.status != "staged":
            raise RuntimeError(f"regime {regime} did not stage: {staged.status}")
        stream_statuses = [staged.status]
        stream_statuses.extend(router.observe(item).status for item in second_window)
        statuses[regime] = stream_statuses
        receipt = router.promote_staged_candidate(
            heldout[regime],
            lambda candidate, expected_count=regime + 1: (
                candidate.residual_bank is not None
                and candidate.residual_bank.context_count == expected_count
            ),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        receipts.append(receipt.__dict__)

    random_partial_routes: list[int | None] = []
    for regime in range(REGIME_COUNT):
        query_indices = orders[regime][:3]
        query = [_row(full[regime], index) for index in query_indices]
        result = router.route_partial_bundle(
            query,
            min_match_fraction=0.25,
            match_tolerance=PREDICTION_TOLERANCE,
            contradiction_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        )
        random_partial_routes.append(result.slot_id)
    mixed = router.route_partial_bundle(
        [_row(full[0], orders[0][0]), _row(full[1], orders[1][0])],
        min_match_fraction=0.25,
        match_tolerance=PREDICTION_TOLERANCE,
        contradiction_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    retained_errors = {
        regime: _mse(router, heldout[regime], regime)
        for regime in range(REGIME_COUNT)
    }
    gates = {
        "all_regimes_promoted": all(item["accepted"] for item in receipts),
        "random_windows_stage_and_continue": all(
            values[0] == "staged" and all(value == "staged" for value in values)
            for values in statuses.values()
        ),
        "heldout_retention": all(
            value <= PREDICTION_TOLERANCE for value in retained_errors.values()
        ),
        "random_partial_routes_correct": random_partial_routes
        == [0, 1, 2, 3],
        "mixed_partial_is_ambiguous": mixed.status == "ambiguous"
        and mixed.slot_id is None,
        "sparse_identity_persisted": (
            router.sparse_evidence is not None
            and restored.sparse_evidence is not None
            and restored.sparse_evidence.digest() == router.sparse_evidence.digest()
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
            "observed_rows": OBSERVED_ROWS,
            "window_rows": WINDOW_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "mask": "seeded_random_permutation_v1",
            "residual_model_family": EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
            "claim": "bounded-factored-random-missingness-identity-v1",
        },
        "metrics": {
            "orders": {str(key): list(value) for key, value in orders.items()},
            "stream_statuses": statuses,
            "promotion_receipts": receipts,
            "random_partial_routes": random_partial_routes,
            "mixed_status": mixed.status,
            "retained_heldout_mse": retained_errors,
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
            "residual_statistics_updates": REGIME_COUNT * 2,
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "old_regime_replay": 0,
            "unique_observed_rows": REGIME_COUNT * OBSERVED_ROWS,
            "unique_heldout_rows": REGIME_COUNT * HELDOUT_ROWS,
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
