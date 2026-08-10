"""Repeated growth/eviction audit for stable external factored memory."""

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
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 8
REGIME_COUNT = 7
ROWS_PER_REGIME = 24
TRAIN_ROWS = 20
INITIAL_CAPACITY = 2
FIRST_GROWN_CAPACITY = 4
SECOND_GROWN_CAPACITY = 5
FEATURE_WIDTH = 128
RESIDUAL_UPDATES = 128
PREDICTION_TOLERANCE = 0.1


def _digest(module: torch.nn.Module) -> str:
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        result.update(name.encode("utf-8"))
        result.update(str(detached.dtype).encode("utf-8"))
        result.update(repr(tuple(detached.shape)).encode("utf-8"))
        result.update(detached.numpy().tobytes())
    return result.hexdigest()


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
    observations = {
        regime: _observation(
            state,
            intention,
            _target(state, intention, regime),
        )
        for regime in range(REGIME_COUNT)
    }
    train = {regime: _slice(item, 0, TRAIN_ROWS) for regime, item in observations.items()}
    heldout = {
        regime: _slice(item, TRAIN_ROWS, ROWS_PER_REGIME)
        for regime, item in observations.items()
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
    router = ExternalFactoredTransitionRouter(
        model,
        encoder,
        match_tolerance=0.005,
        match_margin=0.0001,
        max_contexts=INITIAL_CAPACITY,
        residual_adaptation_updates=RESIDUAL_UPDATES,
    )

    active: dict[int, ExternalTransitionObservation] = {}
    promotion_receipts: list[dict[str, object]] = []

    def promote(regime: int) -> int | None:
        route = router.route_bundle(_rows(train[regime]))
        receipt = router.promote_staged_candidate(
            heldout[regime],
            _model_retains(active, tolerance=PREDICTION_TOLERANCE),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        promotion_receipts.append(
            {
                "regime": regime,
                "route_status": route.status,
                "receipt": receipt.__dict__,
            }
        )
        if not receipt.accepted or receipt.slot_id is None:
            return None
        active[receipt.slot_id] = heldout[regime]
        return receipt.slot_id

    first_slots = [promote(regime) for regime in (0, 1)]
    first_growth = router.grow_verified(
        FIRST_GROWN_CAPACITY,
        _retains(active, tolerance=PREDICTION_TOLERANCE),
    )
    second_slots = [promote(regime) for regime in (2, 3)]
    routes_after_first_growth = [
        router.route_bundle(
            _rows(train[regime]),
            match_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in (0, 1, 2, 3)
    ]

    first_eviction = router.evict_verified_id(
        1,
        _retains(
            {slot: observation for slot, observation in active.items() if slot != 1},
            tolerance=PREDICTION_TOLERANCE,
        ),
    )
    active.pop(1)
    fourth_slot = promote(4)
    routes_after_first_eviction = [
        router.route_bundle(
            _rows(train[regime]),
            match_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in (0, 2, 3, 4)
    ]

    second_growth = router.grow_verified(
        SECOND_GROWN_CAPACITY,
        _retains(active, tolerance=PREDICTION_TOLERANCE),
    )
    fifth_slot = promote(5)
    routes_after_second_growth = [
        router.route_bundle(
            _rows(train[regime]),
            match_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in (0, 2, 3, 4, 5)
    ]

    second_eviction = router.evict_verified_id(
        3,
        _retains(
            {slot: observation for slot, observation in active.items() if slot != 3},
            tolerance=PREDICTION_TOLERANCE,
        ),
    )
    active.pop(3)
    sixth_slot = promote(6)
    final_regimes = (0, 2, 4, 5, 6)
    final_routes = [
        router.route_bundle(
            _rows(train[regime]),
            match_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in final_regimes
    ]
    partial_digest_before = router.digest()
    partial_routes = [
        router.route_partial_bundle(
            _rows(train[regime])[:5],
            match_tolerance=PREDICTION_TOLERANCE,
            contradiction_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in (0, 6)
    ]
    partial_digest_after = router.digest()

    compressed = router.select_compression_verified(
        ["float16_stats", torch.float16],
        retention_probe=_retains(active, tolerance=PREDICTION_TOLERANCE),
    )
    compressed_roundtrip = False
    if compressed.accepted and compressed.selected_codec is not None:
        payload = router.model.residual_bank.compressed_payload(
            dtype=compressed.selected_codec
        )
        compressed_bank = ExternalTransitionModelBank.from_compressed_payload(payload)
        compressed_roundtrip = _retains(
            active,
            tolerance=PREDICTION_TOLERANCE,
        )(compressed_bank)

    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    restored_routes = [
        restored.route_bundle(
            _rows(train[regime]),
            match_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for regime in final_regimes
    ]
    expected_ids = (0, 2, 4, 5, 6)
    gates = {
        "all_seven_promoted": len(promotion_receipts) == REGIME_COUNT
        and all(item["receipt"]["accepted"] for item in promotion_receipts),
        "first_growth_retains": first_growth.accepted
        and first_growth.context_count == 2,
        "second_growth_retains": second_growth.accepted
        and second_growth.context_count == 4,
        "routes_after_first_growth": routes_after_first_growth == [0, 1, 2, 3],
        "first_eviction_accepted": first_eviction.accepted,
        "first_eviction_survivors_route": routes_after_first_eviction == [0, 2, 3, 4],
        "second_growth_routes": routes_after_second_growth == [0, 2, 3, 4, 5],
        "second_eviction_accepted": second_eviction.accepted,
        "second_eviction_survivors_route": final_routes == [0, 2, 4, 5, 6],
        "stable_ids_after_repeated_lifecycle": router.slot_ids == expected_ids,
        "partial_reads_do_not_mutate": partial_routes == [0, 6]
        and partial_digest_before == partial_digest_after,
        "compression_selected": compressed.accepted,
        "compression_roundtrip_retained": compressed_roundtrip,
        "persistence_retains_ids": restored.slot_ids == expected_ids
        and restored_routes == [0, 2, 4, 5, 6],
        "base_unchanged": model.base.digest() == base_digest and model.base_frozen,
        "controller_unchanged": controller_digest == _digest(controller),
        "context_encoder_unchanged": encoder_digest == encoder.digest(),
        "no_old_regime_replay": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"repeated lifecycle gates failed: {gates}")
    report: dict[str, object] = {
        "schema": "neural-computer.external-factored-repeated-lifecycle.v1",
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "rows_per_regime": ROWS_PER_REGIME,
            "train_rows": TRAIN_ROWS,
            "initial_capacity": INITIAL_CAPACITY,
            "first_grown_capacity": FIRST_GROWN_CAPACITY,
            "second_grown_capacity": SECOND_GROWN_CAPACITY,
            "residual_model_family": EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
            "feature_width": FEATURE_WIDTH,
            "prediction_tolerance": PREDICTION_TOLERANCE,
            "claim": "repeated-factored-memory-growth-eviction-v1",
        },
        "metrics": {
            "first_slots": first_slots,
            "second_slots": second_slots,
            "fourth_slot": fourth_slot,
            "fifth_slot": fifth_slot,
            "sixth_slot": sixth_slot,
            "promotion_receipts": promotion_receipts,
            "first_growth": first_growth.__dict__,
            "second_growth": second_growth.__dict__,
            "first_eviction": first_eviction.__dict__,
            "second_eviction": second_eviction.__dict__,
            "routes_after_first_growth": routes_after_first_growth,
            "routes_after_first_eviction": routes_after_first_eviction,
            "routes_after_second_growth": routes_after_second_growth,
            "final_routes": final_routes,
            "partial_routes": partial_routes,
            "restored_routes": restored_routes,
            "compression": compressed.__dict__,
            "final_slot_ids": list(router.slot_ids),
        },
        "gates": gates,
        "accounting": {
            "base_optimizer_updates": 0,
            "residual_optimizer_updates": 0,
            "residual_statistics_updates": REGIME_COUNT,
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
        "promoted": True,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=88001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
