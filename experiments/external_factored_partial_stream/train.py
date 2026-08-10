"""Pressure-test replay-free factored acquisition from partial streams."""

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
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

SCHEMA = "neural-computer.external-factored-partial-stream.v1"
STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
REGIME_COUNT = 4
ROWS_PER_REGIME = 18
STREAM_ROWS = 14
ADMISSION_ROWS = 7
HELDOUT_ROWS = ROWS_PER_REGIME - STREAM_ROWS
FEATURE_WIDTH = 128
MODEL_HIDDEN_WIDTH = 16
CAPACITY = REGIME_COUNT
RESIDUAL_UPDATES = 128
PREDICTION_TOLERANCE = 0.1
MATCH_TOLERANCE = 0.02


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
        regime: _observation(
            state,
            intention,
            _target(state, intention, regime),
        )
        for regime in range(REGIME_COUNT)
    }
    streams = {
        regime: _observation(
            item.state[:STREAM_ROWS],
            item.intention[:STREAM_ROWS],
            item.next_state[:STREAM_ROWS],
        )
        for regime, item in full.items()
    }
    heldout = {
        regime: _observation(
            item.state[STREAM_ROWS:],
            item.intention[STREAM_ROWS:],
            item.next_state[STREAM_ROWS:],
        )
        for regime, item in full.items()
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
        hidden_width=MODEL_HIDDEN_WIDTH,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_random_feature_width=FEATURE_WIDTH,
        residual_random_feature_seed=seed,
        residual_ridge=0.01,
        residual_capacity=CAPACITY,
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
        match_tolerance=MATCH_TOLERANCE,
        match_margin=0.001,
        admission_observations=ADMISSION_ROWS,
        max_contexts=CAPACITY,
        residual_adaptation_updates=RESIDUAL_UPDATES,
    )

    promotion_receipts: list[dict[str, object]] = []
    stream_statuses: dict[int, list[str]] = {}
    for regime in range(REGIME_COUNT):
        statuses: list[str] = []
        for row in _rows(streams[regime]):
            statuses.append(router.observe(row).status)
        stream_statuses[regime] = statuses
        previous = {index: heldout[index] for index in range(regime)}
        receipt = router.promote_staged_candidate(
            heldout[regime],
            _model_retains(previous, tolerance=PREDICTION_TOLERANCE),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        promotion_receipts.append(receipt.__dict__)

    alternating_routes = [
        router.route_bundle(
            _rows(streams[regime]),
            match_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in (0, 1, 2, 3, 0, 3, 1, 2)
    ]
    partial_reads = [
        router.route_partial_bundle(
            _rows(streams[regime])[:1],
            match_tolerance=PREDICTION_TOLERANCE,
            contradiction_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in range(REGIME_COUNT)
    ]
    contradictory = router.route_partial_bundle(
        _rows(streams[0])[:2] + _rows(streams[1])[:2],
        min_match_fraction=0.5,
        match_tolerance=PREDICTION_TOLERANCE,
        contradiction_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    before_reads = router.digest()
    empty = router.route_partial_bundle([])
    after_reads = router.digest()
    retained_errors = {
        regime: _bank_mse(router.model.residual_bank, heldout[regime], regime)
        for regime in range(REGIME_COUNT)
    }
    gates = {
        "all_regimes_promoted": all(
            receipt["accepted"] for receipt in promotion_receipts
        ),
        "prefix_admission_reached_stage_at_seven": all(
            statuses[ADMISSION_ROWS - 1] == "staged"
            for statuses in stream_statuses.values()
        ),
        "later_stream_rows_updated_candidate": all(
            all(status == "staged" for status in statuses[ADMISSION_ROWS:])
            for statuses in stream_statuses.values()
        ),
        "heldout_retention": all(
            error <= PREDICTION_TOLERANCE for error in retained_errors.values()
        ),
        "alternating_routes_correct": alternating_routes
        == [0, 1, 2, 3, 0, 3, 1, 2],
        "partial_known_reads_correct": partial_reads == [0, 1, 2, 3],
        "contradictory_partial_is_ambiguous": contradictory.status == "ambiguous"
        and contradictory.slot_id is None,
        "empty_partial_is_noop": empty.status == "ambiguous" and empty.slot_id is None,
        "read_only_digest_stable": before_reads == after_reads,
        "base_unchanged": model.base.digest() == base_digest and model.base_frozen,
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_unchanged": encoder_digest == encoder.digest(),
        "no_old_regime_replay": True,
    }
    report = {
        "schema": SCHEMA,
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "rows_per_regime": ROWS_PER_REGIME,
            "stream_rows": STREAM_ROWS,
            "admission_rows": ADMISSION_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "residual_model_family": EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
            "feature_width": FEATURE_WIDTH,
            "prediction_tolerance": PREDICTION_TOLERANCE,
            "claim": "bounded-replay-free-partial-stream-acquisition-v1",
        },
        "metrics": {
            "promotion_receipts": promotion_receipts,
            "stream_statuses": stream_statuses,
            "alternating_routes": alternating_routes,
            "partial_reads": partial_reads,
            "contradictory_status": contradictory.status,
            "empty_status": empty.status,
            "retained_heldout_mse": retained_errors,
            "slot_ids": list(router.slot_ids),
        },
        "gates": gates,
        "accounting": {
            "base_optimizer_updates": 0,
            "residual_optimizer_updates": 0,
            "residual_statistics_updates": REGIME_COUNT,
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "old_regime_replay": 0,
            "unique_stream_rows": REGIME_COUNT * STREAM_ROWS,
            "admission_rows_per_regime": ADMISSION_ROWS,
            "heldout_rows_per_regime": HELDOUT_ROWS,
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
    parser.add_argument("--seed", type=int, default=81041)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
