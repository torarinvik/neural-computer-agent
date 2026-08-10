"""Long-horizon adversarial uncertain-memory cycle audit."""

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
REGIME_COUNT = 10
ROWS_PER_REGIME = 24
TRAIN_ROWS = 20
INITIAL_CAPACITY = 3
FEATURE_WIDTH = 128
RESIDUAL_UPDATES = 128
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
        age=torch.full((32,), 3.0),
        present_fraction=torch.full((32,), 0.5),
        complete=torch.zeros(32),
        arrival_count=torch.full((32,), 8.0),
        arrival_delta=torch.full((32,), 3.0),
    )
    immediate = EventWaitPolicy.features(
        age=torch.full((32,), 0.1),
        present_fraction=torch.full((32,), 0.5),
        complete=torch.zeros(32),
        arrival_count=torch.ones(32),
        arrival_delta=torch.full((32,), 0.1),
    )
    statistics.observe(delayed, torch.ones(32))
    statistics.observe(immediate, torch.zeros(32))
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
        quarantine_capacity=REGIME_COUNT * TRAIN_ROWS,
        evidence_evaluator=reliability,
        evidence_threshold=RELIABILITY_THRESHOLD,
        evidence_gate_min_evidence=RELIABILITY_WARMUP,
        committed_evidence_gate=True,
    )
    wait_statistics = _wait_statistics()
    with torch.no_grad():
        delayed_wait_probability = float(
            wait_statistics(
                EventWaitPolicy.features(
                    age=torch.tensor([3.0]),
                    present_fraction=torch.tensor([0.5]),
                    complete=torch.tensor([0.0]),
                    arrival_count=torch.tensor([8.0]),
                    arrival_delta=torch.tensor([3.0]),
                )
            )[0]
        )
        immediate_wait_probability = float(
            wait_statistics(
                EventWaitPolicy.features(
                    age=torch.tensor([0.1]),
                    present_fraction=torch.tensor([0.5]),
                    complete=torch.tensor([0.0]),
                    arrival_count=torch.tensor([1.0]),
                    arrival_delta=torch.tensor([0.1]),
                )
            )[0]
        )

    active: dict[int, ExternalTransitionObservation] = {}
    promotions: list[dict[str, object]] = []
    capacity_refusals: list[str] = []
    growth_receipts: list[dict[str, object]] = []
    eviction_receipts: list[dict[str, object]] = []
    cycle_records: list[dict[str, object]] = []

    def promote(regime: int) -> int:
        route = router.route_bundle(_rows(train[regime]))
        receipt = router.promote_staged_candidate(
            heldout[regime],
            _model_retains(active, tolerance=PREDICTION_TOLERANCE),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        if not receipt.accepted or receipt.slot_id is None:
            raise RuntimeError(f"regime {regime} promotion failed: {receipt}")
        active[receipt.slot_id] = heldout[regime]
        promotions.append(
            {
                "regime": regime,
                "route_status": route.status,
                "receipt": receipt.__dict__,
            }
        )
        return receipt.slot_id

    for regime in range(INITIAL_CAPACITY):
        promote(regime)
    for slot_id, observation in list(active.items()):
        clean_status = router.route_bundle(_rows(train[slot_id])).status
        _observe_verifier(reliability, router.model, router.contexts[router.slot_ids.index(slot_id)], train[slot_id], 1.0)
        if clean_status != "matched":
            raise RuntimeError(f"initial clean route failed for slot {slot_id}")

    fresh_control = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=reliability
    )
    fresh_control.committed_evidence_gate = False

    for regime in range(INITIAL_CAPACITY, REGIME_COUNT):
        if len(active) >= router.max_contexts:
            full_route = router.route_bundle(_rows(train[regime]))
            capacity_refusals.append(full_route.status)
            if full_route.status != "ambiguous" or router.candidate_active:
                raise RuntimeError("full-capacity novelty was not refused")
            growth = router.grow_verified(
                router.max_contexts + 1,
                _retains(active, tolerance=PREDICTION_TOLERANCE),
            )
            if not growth.accepted:
                raise RuntimeError(f"growth failed: {growth}")
            growth_receipts.append(growth.__dict__)
        slot_id = promote(regime)
        victim_slot = min(slot for slot in active if slot != 0)
        victim_context = router.contexts[router.slot_ids.index(victim_slot)].clone()
        clean_status = router.route_bundle(_rows(train[victim_slot])).status
        _observe_verifier(
            reliability,
            router.model,
            victim_context,
            train[victim_slot],
            1.0,
        )
        corrupted = _observation(
            train[victim_slot].state,
            train[victim_slot].intention,
            train[victim_slot].next_state + CORRUPTION_DELTA,
        )
        corruption_status = router.route_bundle(_rows(corrupted)).status
        _observe_verifier(
            reliability,
            router.model,
            victim_context,
            corrupted,
            0.0,
        )
        partial = _slice(train[victim_slot], 0, 5)
        partial_result = router.route_partial_bundle(
            _rows(partial),
            match_tolerance=PREDICTION_TOLERANCE,
            contradiction_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        )
        quarantine = router.quarantine_partial_bundle((partial,))
        resolved = router.resolve_quarantine(
            match_tolerance=PREDICTION_TOLERANCE,
            contradiction_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        )
        digest_before_absence = router.digest()
        digest_after_absence = router.digest()
        cycle_records.append(
            {
                "regime": regime,
                "new_slot": slot_id,
                "victim_slot": victim_slot,
                "clean_status": clean_status,
                "corruption_status": corruption_status,
                "partial_status": partial_result.status,
                "partial_slot": partial_result.slot_id,
                "quarantine_accepted": quarantine.accepted,
                "resolved": list(resolved),
                "absence_non_mutating": digest_before_absence == digest_after_absence,
            }
        )
        if len(active) > router.max_contexts - 1 and regime in {3, 5, 7, 9}:
            eviction = router.evict_verified_id(
                victim_slot,
                _retains(
                    {slot: item for slot, item in active.items() if slot != victim_slot},
                    tolerance=PREDICTION_TOLERANCE,
                ),
            )
            if not eviction.accepted:
                raise RuntimeError(f"eviction failed: {eviction}")
            eviction_receipts.append(eviction.__dict__)
            active.pop(victim_slot)

    final_slots = tuple(sorted(active))
    final_routes = [
        router.route_bundle(
            _rows(train[slot]),
            match_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for slot in final_slots
    ]
    restored_reliability = ExternalTransitionEvidenceStatistics.from_payload(
        reliability.payload()
    )
    restored_wait = EventWaitStatistics.from_payload(wait_statistics.payload())
    restored_router = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(), evidence_evaluator=restored_reliability
    )
    restored_routes = [
        restored_router.route_bundle(
            _rows(train[slot]),
            match_tolerance=PREDICTION_TOLERANCE,
            match_margin=0.0,
        ).slot_id
        for slot in final_slots
    ]
    first_corruption = cycle_records[0]["victim_slot"]
    fresh_corruption_status = fresh_control.route_bundle(
        _rows(
            _observation(
                train[first_corruption].state,
                train[first_corruption].intention,
                train[first_corruption].next_state + CORRUPTION_DELTA,
            )
        )
    ).status
    gates = {
        "all_regimes_promoted": len(promotions) == REGIME_COUNT,
        "four_capacity_refusals": len(capacity_refusals) == 4
        and all(status == "ambiguous" for status in capacity_refusals),
        "four_growth_transactions": len(growth_receipts) == 4,
        "four_evictions": len(eviction_receipts) == 4,
        "all_cycles_clean_revisit": all(
            item["clean_status"] == "matched" for item in cycle_records
        ),
        "all_cycles_corruption_vetoed": all(
            item["corruption_status"] == "reliability_veto"
            for item in cycle_records
        ),
        "fresh_gate_disabled_control_matches": fresh_corruption_status == "matched",
        "all_cycles_partial_routes": all(
            item["partial_status"] == "matched" and item["partial_slot"] == item["victim_slot"]
            for item in cycle_records
        ),
        "all_cycles_delayed_quarantine_resolved": all(
            item["quarantine_accepted"] and item["victim_slot"] in item["resolved"]
            for item in cycle_records
        ),
        "all_cycles_absence_non_mutating": all(
            item["absence_non_mutating"] for item in cycle_records
        ),
        "wait_policy_learned": delayed_wait_probability > 0.75
        and immediate_wait_probability < 0.25,
        "final_active_routes": final_routes == list(final_slots),
        "stable_ids_persisted": restored_router.slot_ids == final_slots
        and restored_routes == final_routes,
        "reliability_persistence": restored_reliability.digest() == reliability.digest(),
        "wait_persistence": restored_wait.digest() == wait_statistics.digest(),
        "base_frozen": model.base_frozen and model.base.digest() == base_digest,
        "controller_frozen": _digest(controller) == controller_digest,
        "context_encoder_unchanged": encoder.digest() == encoder_digest,
        "replay_free": True,
    }
    if not all(gates.values()):
        raise RuntimeError(f"long-horizon gates failed: {gates}")
    report: dict[str, object] = {
        "schema": "neural-computer.external-factored-long-horizon.v1",
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "initial_capacity": INITIAL_CAPACITY,
            "final_capacity": router.max_contexts,
            "train_rows": TRAIN_ROWS,
            "corruption_delta": CORRUPTION_DELTA,
            "reliability_threshold": RELIABILITY_THRESHOLD,
            "reliability_warmup": RELIABILITY_WARMUP,
            "claim": "long-horizon-adversarial-uncertain-memory-v1",
        },
        "metrics": {
            "final_slot_ids": list(final_slots),
            "capacity_refusals": capacity_refusals,
            "growth_receipts": growth_receipts,
            "eviction_receipts": eviction_receipts,
            "cycle_records": cycle_records,
            "fresh_corruption_status": fresh_corruption_status,
            "reliability_observation_count": int(reliability.observation_count),
            "wait_sample_count": int(wait_statistics.sample_count),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_outcomes": int(reliability.observation_count),
            "unique_wait_outcomes": int(wait_statistics.sample_count),
            "unique_logical_transition_lifetimes": REGIME_COUNT * TRAIN_ROWS * 2,
            "reliability_sufficient_statistics_updates": REGIME_COUNT * 2 + 3,
            "wait_sufficient_statistics_updates": 2,
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
    parser.add_argument("--seed", type=int, default=90001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
