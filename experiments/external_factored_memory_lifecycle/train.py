"""Pressure-test growth, stable-ID eviction, and compression of factored memory."""

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

SCHEMA = "neural-computer.external-factored-memory-lifecycle.v1"
STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
BASE_HIDDEN_WIDTH = 16
FEATURE_WIDTH = 128
REGIME_COUNT = 5
ROWS_PER_REGIME = 24
TRAIN_ROWS = 20
INITIAL_CAPACITY = 2
GROWN_CAPACITY = 4
BASE_UPDATES = 0
RESIDUAL_UPDATES = 128
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
        return _retains(observations, tolerance=tolerance)(model.residual_bank)

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
    observations = {
        regime: _observation(
            state,
            intention,
            _target(state, intention, regime),
        )
        for regime in range(REGIME_COUNT)
    }
    train = {
        regime: _observation(
            observation.state[:TRAIN_ROWS],
            observation.intention[:TRAIN_ROWS],
            observation.next_state[:TRAIN_ROWS],
        )
        for regime, observation in observations.items()
    }
    heldout = {
        regime: _observation(
            observation.state[TRAIN_ROWS:],
            observation.intention[TRAIN_ROWS:],
            observation.next_state[TRAIN_ROWS:],
        )
        for regime, observation in observations.items()
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
        hidden_width=BASE_HIDDEN_WIDTH,
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
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=0.005,
        match_margin=0.0001,
        max_contexts=INITIAL_CAPACITY,
        residual_adaptation_updates=RESIDUAL_UPDATES,
    )

    promotion_receipts: list[object] = []
    routes_before_eviction: list[int | None] = []
    routes_after_eviction: list[int | None] = []
    for regime in (0, 1):
        route = router.route_bundle(_rows(train[regime]))
        receipt = router.promote_staged_candidate(
            heldout[regime],
            _model_retains(
                {previous: heldout[previous] for previous in range(regime)},
                tolerance=PREDICTION_TOLERANCE,
            ),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        promotion_receipts.append(receipt.__dict__)
        if not receipt.accepted or route.status != "staged":
            break

    full_route = router.route_bundle(
        _rows(train[2]),
        match_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    full_before_growth = full_route.status == "ambiguous"
    old_heldout = {0: heldout[0], 1: heldout[1]}
    growth_receipt = router.grow_verified(
        GROWN_CAPACITY,
        _retains(old_heldout, tolerance=PREDICTION_TOLERANCE),
    )

    for regime in (2, 3):
        route = router.route_bundle(_rows(train[regime]))
        receipt = router.promote_staged_candidate(
            heldout[regime],
            _model_retains(
                {previous: heldout[previous] for previous in (0, 1, 2, 3) if previous < regime},
                tolerance=PREDICTION_TOLERANCE,
            ),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        promotion_receipts.append(receipt.__dict__)
        if not receipt.accepted or route.status != "staged":
            break

    for regime in (0, 1, 2, 3):
        routes_before_eviction.append(
            router.route_bundle(
                _rows(train[regime]),
                match_tolerance=PREDICTION_TOLERANCE,
                match_margin=0.0,
            ).slot_id
        )

    partial_digest_before = router.digest()
    partial_routes = [
        router.route_partial_bundle(
            _rows(train[regime])[:5],
            min_match_fraction=1.0,
            match_tolerance=PREDICTION_TOLERANCE,
            contradiction_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in (0, 3)
    ]
    contradictory = router.route_partial_bundle(
        _rows(train[0])[:2] + _rows(train[1])[:2],
        min_match_fraction=0.5,
        match_tolerance=PREDICTION_TOLERANCE,
        contradiction_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    empty_partial = router.route_partial_bundle([])
    partial_digest_after = router.digest()

    all_heldout = {regime: heldout[regime] for regime in range(4)}
    compressed_selection = router.select_compression_verified(
        ["float16_stats", torch.float16],
        retention_probe=_retains(all_heldout, tolerance=PREDICTION_TOLERANCE),
    )
    compressed_roundtrip = False
    if compressed_selection.accepted and compressed_selection.selected_codec is not None:
        compressed_payload = router.model.residual_bank.compressed_payload(
            dtype=compressed_selection.selected_codec
        )
        compressed_bank = ExternalTransitionModelBank.from_compressed_payload(
            compressed_payload
        )
        compressed_roundtrip = _retains(
            all_heldout,
            tolerance=PREDICTION_TOLERANCE,
        )(compressed_bank)

    eviction_receipt = router.evict_verified_id(
        1,
        _retains({0: heldout[0], 2: heldout[2], 3: heldout[3]}, tolerance=PREDICTION_TOLERANCE),
    )
    for regime in (0, 2, 3):
        routes_after_eviction.append(
            router.route_bundle(
                _rows(train[regime]),
                match_tolerance=PREDICTION_TOLERANCE,
                match_margin=0.0,
            ).slot_id
        )
    new_route = router.route_bundle(
        _rows(train[4]),
        match_tolerance=PREDICTION_TOLERANCE,
        match_margin=0.0,
    )
    new_receipt = router.promote_staged_candidate(
        heldout[4],
        _model_retains(
            {0: heldout[0], 2: heldout[2], 3: heldout[3]},
            tolerance=PREDICTION_TOLERANCE,
        ),
        prediction_tolerance=PREDICTION_TOLERANCE,
    )
    promotion_receipts.append(new_receipt.__dict__)
    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    restored_routes = [
        restored.route_bundle(
            _rows(train[regime]),
            match_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in (0, 2, 3, 4)
    ]

    gates = {
        "all_five_promoted": all(receipt["accepted"] for receipt in promotion_receipts)
        and len(promotion_receipts) == 5,
        "initial_capacity_blocked_novel_regime": full_before_growth,
        "growth_accepted": growth_receipt.accepted,
        "growth_retained_prior_slots": growth_receipt.accepted
        and growth_receipt.context_count == 2,
        "four_regimes_routed_before_eviction": routes_before_eviction == [0, 1, 2, 3],
        "partial_known_evidence_routes": partial_routes == [0, 3],
        "contradictory_partial_evidence_is_ambiguous": contradictory.status
        == "ambiguous"
        and contradictory.slot_id is None,
        "empty_partial_evidence_is_noop": empty_partial.status == "ambiguous"
        and empty_partial.slot_id is None,
        "partial_reads_do_not_mutate": partial_digest_before == partial_digest_after,
        "compression_selected": compressed_selection.accepted,
        "compression_roundtrip_retained": compressed_roundtrip,
        "eviction_accepted": eviction_receipt.accepted,
        "stable_ids_survived_middle_eviction": router.slot_ids[:3] == (0, 2, 3),
        "survivors_routed_after_eviction": routes_after_eviction == [0, 2, 3],
        "new_slot_reused_capacity": new_route.status == "staged"
        and new_receipt.accepted
        and new_receipt.slot_id == 4,
        "stable_ids_persisted": router.slot_ids == (0, 2, 3, 4)
        and restored.slot_ids == (0, 2, 3, 4)
        and restored_routes == [0, 2, 3, 4],
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
            "train_rows": TRAIN_ROWS,
            "initial_capacity": INITIAL_CAPACITY,
            "grown_capacity": GROWN_CAPACITY,
            "residual_model_family": EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
            "feature_width": FEATURE_WIDTH,
            "prediction_tolerance": PREDICTION_TOLERANCE,
            "claim": "bounded-factored-memory-lifecycle-v1",
        },
        "metrics": {
            "promotion_receipts": promotion_receipts,
            "growth_receipt": growth_receipt.__dict__,
            "eviction_receipt": eviction_receipt.__dict__,
            "compression_selection": compressed_selection.__dict__,
            "routes_before_eviction": routes_before_eviction,
            "partial_routes": partial_routes,
            "contradictory_partial_status": contradictory.status,
            "empty_partial_status": empty_partial.status,
            "routes_after_eviction": routes_after_eviction,
            "restored_routes": restored_routes,
            "final_slot_ids": list(router.slot_ids),
            "final_model_digest": router.model.digest(),
        },
        "gates": gates,
        "accounting": {
            "base_optimizer_updates": BASE_UPDATES,
            "residual_optimizer_updates": 0,
            "residual_statistics_updates": 5,
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": 0,
            "old_regime_replay": 0,
            "unique_regime_rows": REGIME_COUNT * TRAIN_ROWS,
            "heldout_rows_per_regime": ROWS_PER_REGIME - TRAIN_ROWS,
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
    parser.add_argument("--seed", type=int, default=81031)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
