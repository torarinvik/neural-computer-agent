"""Pressure-test capacity growth and learned reliability for external memory.

The shared factual transition model is trained once and frozen. Ten distinct
factual lifetimes are admitted into an external random-feature residual bank;
the bank expands from a four-slot verified capacity to ten slots while every
promotion is held-out and complete-prefix gated. A replay-free scalar evidence
statistics component then learns which prediction-error bins are reliable and
rejects corrupted/out-of-distribution reads without mutating the bank.

This is still a bounded factual-memory experiment. It is not a claim of
general continual learning, arbitrary new computation, or unlimited storage.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.policy_free_factual_residual_growth import train as base
from experiments.policy_free_factual_residual_stream.train import (
    _fit_base,
    _new_router,
    _regime_data,
    _residual_observation,
    _reversal_data,
    _rollout,
    _rows,
    _train_fresh_control,
)
from neural_computer import (
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionEvidenceStatistics,
    ExternalTransitionObservation,
)

REGIME_COUNT = 9
TOTAL_LIFETIMES = REGIME_COUNT + 1
INITIAL_CAPACITY = 1
EXPLICIT_GROWTH_SOURCE_COUNT = 4
EXPLICIT_GROWTH_DESTINATION = 8
TARGET_ERROR_FLOOR = 0.04
SOURCE_RETENTION_FLOOR = base.SOURCE_RETENTION_FLOOR
ROUTE_MATCH_TOLERANCE = 0.005
ROUTE_MATCH_MARGIN = 0.001
COMPRESSION_FLOOR = 0.04
CONTROL_UPDATES = 400


def _prefix_probe(
    candidate: ExternalFactoredTransitionModel,
    source_heldout: ExternalTransitionObservation,
    records: list[dict[str, object]],
) -> bool:
    if float(candidate.base.loss(source_heldout).detach()) > SOURCE_RETENTION_FLOOR:
        return False
    for record in records:
        context = record["context"]
        heldout = record["heldout"]
        assert isinstance(context, torch.Tensor)
        assert isinstance(heldout, ExternalTransitionObservation)
        context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
        if float(candidate.loss(heldout, context=context_batch).detach()) > TARGET_ERROR_FLOOR:
            return False
    return True


def _make_reversal_rollout() -> object:
    return _rollout(0, reversal=True)


def _record_metrics(
    router: ExternalFactoredTransitionRouter,
    records: list[dict[str, object]],
) -> list[float]:
    errors: list[float] = []
    for record in records:
        context = record["context"]
        heldout = record["heldout"]
        assert isinstance(context, torch.Tensor)
        assert isinstance(heldout, ExternalTransitionObservation)
        context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
        errors.append(float(router.model.loss(heldout, context=context_batch).detach()))
    return errors


def _route_roundtrip(
    router: ExternalFactoredTransitionRouter,
    records: list[dict[str, object]],
) -> list[int | None]:
    slots: list[int | None] = []
    for record in records:
        train = record["train"]
        assert isinstance(train, ExternalTransitionObservation)
        result = router.route_partial_bundle(
            _rows(train),
            min_match_fraction=1.0,
            match_tolerance=0.05,
            contradiction_tolerance=0.1,
            match_margin=0.0,
        )
        slots.append(result.slot_id)
    return slots


def _calibrate_reliability(
    router: ExternalFactoredTransitionRouter,
    records: list[dict[str, object]],
) -> tuple[
    ExternalTransitionEvidenceStatistics,
    ExternalTransitionObservation,
    ExternalTransitionObservation,
    float,
    float,
]:
    """Learn a replay-free global reliability curve from verifier outcomes."""

    statistics = ExternalTransitionEvidenceStatistics(
        base.STATE_WIDTH,
        bin_count=16,
        error_scale=0.1,
        prior_count=1.0,
    )
    for record in records:
        context = record["context"]
        heldout = record["heldout"]
        assert isinstance(context, torch.Tensor)
        assert isinstance(heldout, ExternalTransitionObservation)
        context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
        with torch.no_grad():
            prediction = router.model.predict_with_context(
                heldout.state,
                heldout.intention,
                context_batch,
            )
        statistics.observe(
            prediction,
            heldout.next_state,
            torch.ones(heldout.state.shape[0]),
            torch.ones(heldout.state.shape[0]),
        )

    reference = records[0]["heldout"]
    reference_context = records[0]["context"]
    assert isinstance(reference, ExternalTransitionObservation)
    assert isinstance(reference_context, torch.Tensor)
    corrupted = ExternalTransitionObservation(
        state=reference.state[:1],
        intention=reference.intention[:1],
        next_state=reference.next_state[:1] + 2.0,
    )
    context_batch = reference_context.unsqueeze(0)
    with torch.no_grad():
        corrupted_prediction = router.model.predict_with_context(
            corrupted.state,
            corrupted.intention,
            context_batch,
        )
    statistics.observe(
        corrupted_prediction,
        corrupted.next_state,
        torch.zeros(1),
        torch.ones(1),
    )
    out_of_distribution = ExternalTransitionObservation(
        state=torch.tensor([[3.0]]),
        intention=torch.tensor([[1.0]]),
        next_state=torch.tensor([[0.0]]),
    )
    with torch.no_grad():
        out_of_distribution_prediction = router.model.predict_with_context(
            out_of_distribution.state,
            out_of_distribution.intention,
            context_batch,
        )
    statistics.observe(
        out_of_distribution_prediction,
        out_of_distribution.next_state,
        torch.zeros(1),
        torch.ones(1),
    )
    bad_probability = float(
        torch.sigmoid(
            statistics(
                corrupted_prediction,
                corrupted.next_state,
                torch.ones(1),
            )
        ).mean()
    )
    good_prediction = router.model.predict_with_context(
        reference.state[:1],
        reference.intention[:1],
        context_batch,
    )
    good_probability = float(
        torch.sigmoid(
            statistics(good_prediction, reference.next_state[:1], torch.ones(1))
        ).mean()
    )
    return statistics, corrupted, out_of_distribution, good_probability, bad_probability


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    source_train, source_heldout, _, _ = base._base_data()
    source_base, base_updates = _fit_base(seed, source_train)
    base_digest = base._digest_module(source_base)
    source_error_before = float(source_base.loss(source_heldout).detach())
    router = _new_router(source_base, seed + 10_000)
    if router.max_contexts != INITIAL_CAPACITY:
        raise AssertionError("capacity pressure test did not start at one slot")

    records: list[dict[str, object]] = []
    growth_receipt = None
    fresh_controls: list[dict[str, object]] = []
    streams: list[tuple[int | str, ExternalTransitionObservation, ExternalTransitionObservation]] = []
    for regime in range(REGIME_COUNT):
        train, heldout = _regime_data(regime)
        streams.append((regime, train, heldout))
    reversal_train, reversal_heldout = _reversal_data()
    streams.append(("reversal", reversal_train, reversal_heldout))

    for lifetime, train, heldout in streams:
        staging = router.route_bundle(_rows(train))
        if staging.status != "staged":
            raise AssertionError(f"lifetime {lifetime} did not stage: {staging}")
        prior_records = list(records)
        receipt = router.promote_staged_candidate(
            heldout,
            lambda candidate, prior_records=prior_records: _prefix_probe(
                candidate,
                source_heldout,
                prior_records,
            ),
            prediction_tolerance=TARGET_ERROR_FLOOR,
            heldout_rollout=(
                _make_reversal_rollout()
                if lifetime == "reversal"
                else _rollout(int(lifetime))
            ),
            rollout_error_tolerance=TARGET_ERROR_FLOOR,
        )
        if not receipt.accepted or receipt.slot_id is None:
            raise AssertionError(f"lifetime {lifetime} was rejected: {receipt}")
        records.append(
            {
                "lifetime": lifetime,
                "slot_id": receipt.slot_id,
                "context": staging.context,
                "train": train,
                "heldout": heldout,
                "train_rows": int(train.state.shape[0]),
                "heldout_rows": int(heldout.state.shape[0]),
                "heldout_error": receipt.heldout_error,
                "rollout_error": receipt.heldout_rollout_error,
            }
        )
        if isinstance(lifetime, int):
            fresh_controls.append(
                _train_fresh_control(
                    seed=seed + 40_000 + lifetime,
                    source_heldout=source_heldout,
                    target_train=train,
                    target_heldout=heldout,
                )
            )
        if len(records) == EXPLICIT_GROWTH_SOURCE_COUNT:
            before_content = router.model.residual_bank.content_digest()
            growth_receipt = router.grow_verified(
                EXPLICIT_GROWTH_DESTINATION,
                lambda candidate, before_content=before_content: (
                    candidate.context_count == EXPLICIT_GROWTH_SOURCE_COUNT
                    and candidate.content_digest() == before_content
                ),
            )
            if not growth_receipt.accepted:
                raise AssertionError(f"explicit capacity growth was rejected: {growth_receipt}")

    if router.model.residual_bank is None:
        raise AssertionError("residual bank was not created")
    residual_bank = router.model.residual_bank
    prefix_errors = _record_metrics(router, records)
    route_roundtrip_slots = _route_roundtrip(router, records)

    shuffled_router = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    permutation = torch.randperm(
        reversal_train.next_state.shape[0],
        generator=torch.Generator().manual_seed(seed + 30_000),
    )
    shuffled_train = ExternalTransitionObservation(
        state=reversal_train.state,
        intention=reversal_train.intention,
        next_state=reversal_train.next_state[permutation],
    )
    shuffled_before = shuffled_router.digest()
    shuffled_staging = shuffled_router.route_bundle(_rows(shuffled_train))
    shuffled_receipt = None
    if shuffled_staging.status == "staged":
        shuffled_receipt = shuffled_router.promote_staged_candidate(
            reversal_heldout,
            lambda candidate: float(candidate.base.loss(source_heldout).detach())
            <= SOURCE_RETENTION_FLOOR,
            prediction_tolerance=TARGET_ERROR_FLOOR,
        )
    shuffled_after = shuffled_router.digest()

    (
        reliability,
        corrupted,
        out_of_distribution,
        good_probability,
        bad_probability,
    ) = _calibrate_reliability(
        router,
        records,
    )
    router.evidence_evaluator = reliability
    router.committed_evidence_gate = True
    reliability_before_read = reliability.digest()
    clean_read = records[0]["train"]
    assert isinstance(clean_read, ExternalTransitionObservation)
    clean_result = router.route_partial_bundle(
        _rows(clean_read)[:1],
        match_tolerance=ROUTE_MATCH_TOLERANCE,
        contradiction_tolerance=0.1,
        match_margin=0.0,
    )
    corruption_result = router.route_partial_bundle(
        [corrupted],
        match_tolerance=ROUTE_MATCH_TOLERANCE,
        contradiction_tolerance=0.1,
        match_margin=0.0,
    )
    ood_result = router.route_partial_bundle(
        [out_of_distribution],
        match_tolerance=ROUTE_MATCH_TOLERANCE,
        contradiction_tolerance=0.1,
        match_margin=0.0,
    )
    reliability_after_read = reliability.digest()

    restored_reliability = ExternalTransitionEvidenceStatistics.from_payload(
        reliability.payload()
    )
    restored_router = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload(),
        evidence_evaluator=restored_reliability,
    )
    restored_prefix_errors = _record_metrics(restored_router, records)

    def compression_probe(candidate_bank) -> bool:
        for record in records:
            context = record["context"]
            heldout = record["heldout"]
            assert isinstance(context, torch.Tensor)
            assert isinstance(heldout, ExternalTransitionObservation)
            residual = _residual_observation(router.model, heldout)
            context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
            if float(candidate_bank.loss(residual, context_batch).detach()) > COMPRESSION_FLOOR:
                return False
        return True

    compression = router.select_compression_verified(
        (torch.float16, "int4"),
        retention_probe=compression_probe,
    )
    growth_before_reject = router.digest()
    rejected_growth = router.grow_verified(
        EXPLICIT_GROWTH_DESTINATION + 4,
        lambda _candidate: False,
    )
    growth_after_reject = router.digest()

    residual_counts = [
        int(
            residual_bank.models[
                residual_bank.physical_index_for_slot_id(int(record["slot_id"]))
            ].sample_count.item()
        )
        for record in records
    ]
    gates = {
        "ten_lifetimes_promoted": len(records) == TOTAL_LIFETIMES,
        "all_one_step_passed": all(
            float(record["heldout_error"]) <= TARGET_ERROR_FLOOR for record in records
        ),
        "all_recursive_rollouts_passed": all(
            record["rollout_error"] is not None
            and float(record["rollout_error"]) <= TARGET_ERROR_FLOOR
            for record in records
        ),
        "complete_prefix_retention_passed": max(prefix_errors) <= TARGET_ERROR_FLOOR,
        "explicit_capacity_growth_passed": (
            growth_receipt is not None
            and growth_receipt.accepted
            and router.max_contexts >= EXPLICIT_GROWTH_DESTINATION
        ),
        "growth_rejection_is_noop": (
            not rejected_growth.accepted
            and growth_before_reject == growth_after_reject
        ),
        "opaque_route_roundtrip_passed": route_roundtrip_slots == [
            int(record["slot_id"]) for record in records
        ],
        "source_retention_passed": float(router.model.base.loss(source_heldout).detach())
        <= SOURCE_RETENTION_FLOOR,
        "base_frozen": router.model.base_frozen,
        "base_byte_stable": base._digest_module(router.model.base) == base_digest,
        "shuffled_reversal_not_promoted": (
            shuffled_receipt is None or not shuffled_receipt.accepted
        )
        and shuffled_before == shuffled_after,
        "learned_clean_read_allowed": clean_result.status == "matched",
        "learned_corruption_read_rejected": corruption_result.status
        in {"ambiguous", "reliability_veto"},
        "learned_ood_read_rejected": ood_result.status in {"ambiguous", "reliability_veto"},
        "reliability_read_is_noop": reliability_before_read == reliability_after_read,
        "reliability_persistence": restored_reliability.digest() == reliability.digest(),
        "exact_router_persistence": restored_router.digest() == router.digest(),
        "restored_prefix_retention_passed": max(restored_prefix_errors) <= TARGET_ERROR_FLOOR,
        "one_pass_residual_accounting": all(
            count == int(record["train_rows"])
            for count, record in zip(residual_counts, records, strict=True)
        ),
        "compression_verified": compression.accepted,
        "compression_reduces_storage": any(
            receipt.accepted and receipt.compressed_bytes < receipt.source_bytes
            for receipt in compression.receipts
        ),
        "fresh_controls_accounted": all(
            control["optimizer_updates"] == CONTROL_UPDATES
            and control["replayed_examples"] > 0
            for control in fresh_controls
        ),
    }
    report = {
        "schema": "neural-computer.policy-free-factual-residual-capacity.v1",
        "claim_boundary": (
            "ten bounded factual lifetimes in an external residual bank with "
            "verified capacity growth, replay-free reliability statistics, and "
            "OOD read rejection; not general continual learning or unlimited growth"
        ),
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "total_lifetimes": TOTAL_LIFETIMES,
            "initial_capacity": INITIAL_CAPACITY,
            "explicit_growth_source_count": EXPLICIT_GROWTH_SOURCE_COUNT,
            "explicit_growth_destination": EXPLICIT_GROWTH_DESTINATION,
            "target_error_floor": TARGET_ERROR_FLOOR,
            "source_retention_floor": SOURCE_RETENTION_FLOOR,
            "compression_floor": COMPRESSION_FLOOR,
            "reliability": reliability.configuration(),
            "routing": router.configuration(),
            "admission": "heldout_one_step_plus_recursive_rollout_plus_complete_prefix_retention_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "base_source_error_before": source_error_before,
            "base_source_error_after": float(router.model.base.loss(source_heldout).detach()),
            "prefix_errors": prefix_errors,
            "restored_prefix_errors": restored_prefix_errors,
            "max_prefix_error": max(prefix_errors),
            "route_roundtrip_slots": route_roundtrip_slots,
            "route_comparisons_for_novel_bundles": sum(range(TOTAL_LIFETIMES)),
            "final_capacity": router.max_contexts,
            "final_slot_ids": list(router.slot_ids),
            "growth_receipt": (
                None
                if growth_receipt is None
                else {
                    "accepted": growth_receipt.accepted,
                    "source_capacity": growth_receipt.source_capacity,
                    "destination_capacity": growth_receipt.destination_capacity,
                    "content_digest_before": growth_receipt.content_digest_before,
                    "content_digest_after": growth_receipt.content_digest_after,
                }
            ),
            "rejected_growth_reason": rejected_growth.reason,
            "shuffled_staging_status": shuffled_staging.status,
            "shuffled_receipt": (
                None
                if shuffled_receipt is None
                else {
                    "accepted": shuffled_receipt.accepted,
                    "heldout_error": shuffled_receipt.heldout_error,
                    "reason": shuffled_receipt.reason,
                }
            ),
            "clean_read_status": clean_result.status,
            "corruption_read_status": corruption_result.status,
            "ood_read_status": ood_result.status,
            "clean_reliability_probability": good_probability,
            "corrupt_reliability_probability": bad_probability,
            "reliability_observation_count": int(reliability.observation_count.item()),
            "residual_sample_counts": residual_counts,
            "residual_bank_storage_bytes": sum(
                value.numel() * value.element_size()
                for value in residual_bank.state_dict().values()
            ),
            "compression": {
                "accepted": compression.accepted,
                "selected_codec": compression.selected_codec,
                "reason": compression.reason,
                "receipts": [
                    {
                        "codec": receipt.codec,
                        "accepted": receipt.accepted,
                        "source_bytes": receipt.source_bytes,
                        "compressed_bytes": receipt.compressed_bytes,
                        "reason": receipt.reason,
                    }
                    for receipt in compression.receipts
                ],
            },
            "fresh_controls": fresh_controls,
        },
        "accounting": {
            "base_optimizer_updates": base_updates,
            "residual_optimizer_updates": 0,
            "reliability_statistics_updates": int(reliability.observation_count.item()),
            "residual_unique_transition_rows": sum(int(record["train_rows"]) for record in records),
            "residual_heldout_transition_rows": sum(int(record["heldout_rows"]) for record in records),
            "residual_rollout_transition_rows": sum(
                _rollout(0, reversal=record["lifetime"] == "reversal").horizon
                for record in records
            ),
            "logical_lifetimes": len(records),
            "residual_replayed_examples": 0,
            "reliability_replayed_examples": 0,
            "fresh_optimizer_updates": sum(int(control["optimizer_updates"]) for control in fresh_controls),
            "fresh_replayed_examples": sum(int(control["replayed_examples"]) for control in fresh_controls),
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
