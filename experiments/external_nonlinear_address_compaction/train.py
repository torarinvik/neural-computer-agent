"""Lifecycle pressure test for learned-address nonlinear factual memory.

This composes the promoted long alternating stream with two reversible
external-memory operations: verifier-gated parameter sharing and retention-
verified storage compression. The controller remains frozen, and the source
transition rows are never replayed during target acquisition or compaction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

from experiments.external_nonlinear_address_shift_stream.train import (
    INITIAL_CAPACITY,
    REGIME_NAMES,
    SOURCE_REGIMES,
    TARGET_REGIMES,
    _consume,
    _error,
)
from experiments.external_nonlinear_drift_learned_context.train import (
    ADMISSION_ROWS,
    CONTEXT_HIDDEN_WIDTH,
    CONTEXT_UPDATES,
    FEATURE_WIDTH,
    HELDOUT_ROWS,
    INTENTION_WIDTH,
    LOSS_THRESHOLD,
    PRESENTED_ROWS,
    RANDOM_FEATURE_FAMILY,
    STATE_WIDTH,
    TRAIN_ROWS,
    _fixture,
    _train_context_encoder,
    _transition,
)
from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextAddressAdapter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

COMPRESSION_RETENTION_DELTA = 1e-3
ALTERNATION_ROUNDS = 16


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _new_bank(capacity: int) -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        12,
        model_family=RANDOM_FEATURE_FAMILY,
        random_feature_width=FEATURE_WIDTH,
        random_feature_seed=17,
        affine_ridge=1e-4,
        capacity=capacity,
    )


def _retains(
    bank: ExternalTransitionModelBank,
    heldout: dict[str, ExternalTransitionObservation],
    contexts: dict[str, torch.Tensor],
) -> bool:
    return all(
        _error(bank, heldout[name], context) < LOSS_THRESHOLD
        for name, context in contexts.items()
    )


def _fresh_copy_on_write_observation(seed: int) -> ExternalTransitionObservation:
    """Generate new target-regime evidence, never reuse an earlier row."""

    generator = torch.Generator().manual_seed(seed + 901_337)
    state = torch.rand(TRAIN_ROWS, STATE_WIDTH, generator=generator) * 2.0 - 1.0
    intention = torch.rand(
        TRAIN_ROWS,
        INTENTION_WIDTH,
        generator=generator,
    ) * 2.0 - 1.0
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=_transition(2, state, intention),
        confidence=torch.ones(TRAIN_ROWS),
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    fixtures = {
        name: _fixture(seed, regime)
        for regime, name in enumerate(REGIME_NAMES)
    }
    observations = {name: pair[0] for name, pair in fixtures.items()}
    heldout = {name: pair[1] for name, pair in fixtures.items()}

    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=CONTEXT_HIDDEN_WIDTH,
        context_width=12,
        aggregation="mean_pool",
    )
    context_loss, context_updates = _train_context_encoder(
        encoder,
        {name: observations[name] for name in REGIME_NAMES[:SOURCE_REGIMES]},
        seed=seed,
    )
    encoder.eval()
    base_adapter = ExternalTransitionContextAddressAdapter(
        encoder,
        learning_rate=0.001,
        adaptation_steps=4,
        anchor_cosine_ceiling=0.75,
    )
    base_adapter_digest = base_adapter.digest()
    with torch.no_grad():
        source_contexts = {
            name: encoder.encode_observation(observations[name])
            for name in REGIME_NAMES[:SOURCE_REGIMES]
        }

    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=1,
        intention_width=INTENTION_WIDTH,
        feedback_width=2,
        event_window_capacity=ADMISSION_ROWS,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    bank = _new_bank(INITIAL_CAPACITY)
    prior_contexts: dict[str, torch.Tensor] = dict(source_contexts)
    prior_slot_ids: dict[str, int] = {}
    prior_model_digests: dict[int, str] = {}
    for name in REGIME_NAMES[:SOURCE_REGIMES]:
        context = source_contexts[name]
        index = bank.ensure_context(context)
        bank.adaptation_step(
            observations[name],
            context.unsqueeze(0).expand(TRAIN_ROWS, -1),
            None,
        )
        prior_slot_ids[name] = bank.slot_id_at(index)
        prior_model_digests[bank.slot_id_at(index)] = bank.models[index].digest()

    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=LOSS_THRESHOLD,
        match_margin=0.005,
        continuation_tolerance=LOSS_THRESHOLD,
        provisional_continuation_tolerance=1e9,
        admission_observations=ADMISSION_ROWS,
        max_contexts=INITIAL_CAPACITY,
        defer_admission=True,
        candidate_model_families=(RANDOM_FEATURE_FAMILY,),
        provisional_evidence_policy="streaming_statistics",
        address_adapter=base_adapter,
    )
    status_counts: Counter[str] = Counter()
    growth_records: list[dict[str, object]] = []
    acquisitions: list[dict[str, object]] = []
    all_prior_digests_stable = True

    for regime in TARGET_REGIMES:
        name = REGIME_NAMES[regime]
        if name == "target_e":

            def growth_probe(candidate: ExternalTransitionModelBank) -> bool:
                return _retains(candidate, heldout, prior_contexts)

            receipt = router.grow_verified(6, growth_probe)
            if not receipt.accepted:
                raise RuntimeError(f"first capacity growth failed: {receipt.reason}")
            growth_records.append(
                {
                    "source_capacity": receipt.source_capacity,
                    "destination_capacity": receipt.destination_capacity,
                    "context_count": receipt.context_count,
                    "reason": receipt.reason,
                }
            )
        statuses = _consume(router, observations[name])
        status_counts.update(
            f"{name}:{key}" for key in statuses for _ in range(statuses[key])
        )
        if router.provisional_candidate_count != 1:
            raise RuntimeError(f"{name} did not leave one isolated candidate")
        context = router.provisional_context_at(0)
        receipt = router.promote_staged_candidate(
            heldout[name],
            lambda candidate: _retains(candidate, heldout, prior_contexts),
            prediction_tolerance=LOSS_THRESHOLD,
        )
        if not receipt.accepted or receipt.slot_index is None:
            raise RuntimeError(f"{name} promotion failed: {receipt.reason}")
        prior_contexts[name] = context
        prior_slot_ids[name] = bank.slot_id_at(receipt.slot_index)
        prior_model_digests[bank.slot_id_at(receipt.slot_index)] = bank.models[
            receipt.slot_index
        ].digest()
        acquisitions.append(
            {
                "name": name,
                "statuses": dict(statuses),
                "slot_id": bank.slot_id_at(receipt.slot_index),
                "heldout_error": _error(bank, heldout[name], context),
            }
        )

    for name in REGIME_NAMES:
        if bank.slot_id_at(bank.physical_index_for_slot_id(prior_slot_ids[name])) != (
            prior_slot_ids[name]
        ):
            raise RuntimeError(f"slot identity changed before compaction: {name}")

    def second_growth_probe(candidate: ExternalTransitionModelBank) -> bool:
        return _retains(candidate, heldout, prior_contexts)

    second_growth = router.grow_verified(7, second_growth_probe)
    if not second_growth.accepted:
        raise RuntimeError(f"second capacity growth failed: {second_growth.reason}")
    growth_records.append(
        {
            "source_capacity": second_growth.source_capacity,
            "destination_capacity": second_growth.destination_capacity,
            "context_count": second_growth.context_count,
            "reason": second_growth.reason,
        }
    )

    historical_contexts = bank.contexts
    historical_slot_ids = bank.slot_ids
    historical_digests = dict(prior_model_digests)
    duplicate_adapter = router.address_adapter.copy_on_write(
        observations["target_c"],
        historical_contexts,
    ) if router.address_adapter is not None else None
    if duplicate_adapter is None:
        raise RuntimeError("address adapter unexpectedly missing")
    duplicate_context = duplicate_adapter.encode_observation(
        observations["target_c"]
    ).detach()
    duplicate_index = bank.ensure_context(duplicate_context)
    if duplicate_index != bank.context_count - 1:
        raise RuntimeError("copy-on-write address reused an existing slot")
    source_c_index = bank.physical_index_for_slot_id(prior_slot_ids["target_c"])
    bank.models[duplicate_index].load_state_dict(
        bank.models[source_c_index].state_dict()
    )
    duplicate_slot_id = bank.slot_id_at(duplicate_index)
    compaction_contexts = dict(prior_contexts)
    compaction_contexts["duplicate_target_c"] = duplicate_context
    compaction_heldout = dict(heldout)
    compaction_heldout["duplicate_target_c"] = heldout["target_c"]
    compaction_baseline = {
        name: _error(bank, observation, context)
        for name, (observation, context) in {
            name: (compaction_heldout[name], compaction_contexts[name])
            for name in compaction_contexts
        }.items()
    }

    def retains_with_compression_delta(
        candidate: ExternalTransitionModelBank,
    ) -> bool:
        return all(
            _error(candidate, compaction_heldout[name], context)
            <= compaction_baseline[name] + COMPRESSION_RETENTION_DELTA
            for name, context in compaction_contexts.items()
        )

    consolidation = bank.consolidate_verified(
        source_c_index,
        duplicate_index,
        [heldout["target_c"]],
        prediction_tolerance=LOSS_THRESHOLD,
        retention_probe=retains_with_compression_delta,
    )
    if not consolidation.accepted:
        raise RuntimeError(f"verified consolidation failed: {consolidation.reason}")

    aliases_before_copy_on_write = bank.model_aliases()
    pre_copy_on_write_alternation = [
        {
            "regime": name,
            "error": _error(bank, heldout[name], prior_contexts[name]),
        }
        for _round in range(ALTERNATION_ROUNDS)
        for name in REGIME_NAMES
    ]
    source_c_digest_before_copy_on_write = bank.models[source_c_index].digest()
    fresh_copy_on_write_observation = _fresh_copy_on_write_observation(seed)
    bank.adaptation_step(
        fresh_copy_on_write_observation,
        duplicate_context.unsqueeze(0).expand(
            fresh_copy_on_write_observation.state.shape[0],
            -1,
        ),
        None,
    )
    aliases_after_copy_on_write = bank.model_aliases()
    source_c_error_after_copy_on_write = _error(
        bank,
        heldout["target_c"],
        prior_contexts["target_c"],
    )
    duplicate_error_after_copy_on_write = _error(
        bank,
        heldout["target_c"],
        duplicate_context,
    )
    post_copy_on_write_alternation = [
        {
            "regime": name,
            "error": _error(bank, heldout[name], prior_contexts[name]),
        }
        for _round in range(ALTERNATION_ROUNDS)
        for name in REGIME_NAMES
    ]

    compression = bank.select_compression_verified(
        (torch.float16, torch.int8, "int8_row", "float16_stats"),
        retention_probe=retains_with_compression_delta,
    )
    selected_codec = (
        compression.selected_codec
        if compression.selected_codec in {"int8_row", "float16_stats"}
        else torch.float16
    )
    compressed_restored = ExternalTransitionModelBank.from_compressed_payload(
        bank.compressed_payload(dtype=selected_codec)
    )
    compression_deltas: dict[str, dict[str, float]] = {}
    for codec in (torch.float16, torch.int8, "int8_row", "float16_stats"):
        candidate = ExternalTransitionModelBank.from_compressed_payload(
            bank.compressed_payload(dtype=codec)
        )
        compression_deltas[str(codec)] = {
            name: _error(candidate, compaction_heldout[name], context)
            - compaction_baseline[name]
            for name, context in compaction_contexts.items()
        }
    compressed_alternation = [
        {
            "regime": name,
            "error": _error(compressed_restored, heldout[name], prior_contexts[name]),
        }
        for _round in range(ALTERNATION_ROUNDS)
        for name in REGIME_NAMES
    ]

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    compressed_retention = compression.accepted and retains_with_compression_delta(
        compressed_restored
    )
    historical_ids_stable = bank.slot_ids == historical_slot_ids + (duplicate_slot_id,)
    for slot_id, digest in historical_digests.items():
        index = bank.physical_index_for_slot_id(slot_id)
        all_prior_digests_stable = all_prior_digests_stable and (
            bank.models[index].digest() == digest
        )

    corruption_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    corruption_router.bank.capacity = bank.context_count + 1
    corruption_router.max_contexts = bank.context_count + 1
    corrupted = ExternalTransitionObservation(
        state=observations["target_f"].state[:PRESENTED_ROWS],
        intention=observations["target_f"].intention[:PRESENTED_ROWS],
        next_state=observations["target_f"].next_state[:PRESENTED_ROWS].roll(1, 0),
        confidence=torch.ones(PRESENTED_ROWS),
    )
    corruption_statuses = _consume(corruption_router, corrupted)
    corruption_before = corruption_router.bank.content_digest()
    corruption_receipt = corruption_router.promote_staged_candidate(
        heldout["target_f"],
        lambda _candidate: False,
        prediction_tolerance=LOSS_THRESHOLD,
    )
    corruption_rejected = (
        corruption_statuses["staged"] > 0
        and not corruption_receipt.accepted
        and corruption_router.bank.content_digest() == corruption_before
    )

    gates = {
        "context_encoder_converged": context_loss < 0.05,
        "all_targets_acquired": len(acquisitions) == len(TARGET_REGIMES),
        "two_capacity_growths_verified": (
            len(growth_records) == 2
            and growth_records[0]["source_capacity"] == INITIAL_CAPACITY
            and growth_records[0]["destination_capacity"] == 6
            and growth_records[1]["source_capacity"] == 6
            and growth_records[1]["destination_capacity"] == 7
        ),
        "all_acquisition_heldout_errors_pass": all(
            float(record["heldout_error"]) < LOSS_THRESHOLD
            for record in acquisitions
        ),
        "consolidation_verified": (
            consolidation.accepted
            and consolidation.physical_models_after
            == consolidation.physical_models_before - 1
        ),
        "pre_copy_on_write_alternation_retained": (
            aliases_before_copy_on_write[duplicate_index] == source_c_index
            and all(
                float(row["error"]) < LOSS_THRESHOLD
                for row in pre_copy_on_write_alternation
            )
        ),
        "copy_on_write_preserves_source_during_long_use": (
            aliases_after_copy_on_write[duplicate_index] != source_c_index
            and bank.models[source_c_index].digest()
            == source_c_digest_before_copy_on_write
            and source_c_error_after_copy_on_write < LOSS_THRESHOLD
            and duplicate_error_after_copy_on_write < LOSS_THRESHOLD
        ),
        "post_copy_on_write_alternation_retained": all(
            float(row["error"]) < LOSS_THRESHOLD
            for row in post_copy_on_write_alternation
        ),
        "statistics_aware_compression_verified": (
            compression.accepted
            and compression.selected_codec in {"int8_row", "float16_stats"}
        ),
        "selected_compressed_retention_pass": compressed_retention,
            "legacy_codecs_rejected_without_promotion": (
            all(
                not receipt.accepted
                for receipt in compression.receipts
                if receipt.codec in {"torch.float16", "torch.int8"}
            )
            and any(
                max(compression_deltas[codec].values())
                > COMPRESSION_RETENTION_DELTA
                for codec in ("torch.float16", "torch.int8")
            )
        ),
        "compression_payload_roundtrip_exact": (
            compressed_restored.slot_ids == bank.slot_ids
            and compressed_restored.model_aliases() == bank.model_aliases()
        ),
        "compressed_long_alternation_retained": all(
            float(row["error"]) < LOSS_THRESHOLD
            for row in compressed_alternation
        ),
        "stable_logical_addresses": historical_ids_stable
        and compressed_restored.slot_ids == bank.slot_ids,
        "historical_model_digests_stable": all_prior_digests_stable,
        "alias_relationship_persisted": (
            bank.model_aliases() == compressed_restored.model_aliases()
            and bank.model_aliases()[duplicate_index] != source_c_index
        ),
        "corruption_rejected_without_bank_write": corruption_rejected,
        "controller_unchanged": controller_digest == _digest(controller),
        "no_raw_candidate_rows_retained": all(
            not candidate.observations for candidate in router._provisional_candidates
        ),
        "adapter_copy_on_write": (
            router.address_adapter is not None
            and router.address_adapter.digest() != base_adapter_digest
            and base_adapter.digest() == base_adapter_digest
        ),
        "router_persistence_exact": (
            restored.bank.digest() == router.bank.digest()
            and restored.context_encoder.digest() == router.context_encoder.digest()
            and restored.address_adapter is not None
            and router.address_adapter is not None
            and restored.address_adapter.digest() == router.address_adapter.digest()
        ),
    }
    report = {
        "schema": "neural-computer.external-nonlinear-address-compaction.v2",
        "seed": seed,
        "claim_boundary": (
            "retention-verified six-regime 16-round factual-memory growth, alias "
            "consolidation, copy-on-write, and storage compression; not semantic "
            "merging or unrestricted general continual learning"
        ),
        "configuration": {
            "regimes": list(REGIME_NAMES),
            "initial_capacity": INITIAL_CAPACITY,
            "final_capacity": bank.capacity,
            "presented_rows_per_regime": PRESENTED_ROWS,
            "heldout_rows_per_regime": HELDOUT_ROWS,
            "admission_rows": ADMISSION_ROWS,
            "context_encoder_updates": CONTEXT_UPDATES,
            "model_family": RANDOM_FEATURE_FAMILY,
            "compression_candidates": [
                str(torch.float16),
                str(torch.int8),
                "int8_row",
                "float16_stats",
            ],
            "compression_retention_delta": COMPRESSION_RETENTION_DELTA,
            "alternation_rounds": ALTERNATION_ROUNDS,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "context_encoder": {
            "optimizer_updates": context_updates,
            "loss": context_loss,
            "aggregation": "mean_pool",
        },
        "acquisitions": acquisitions,
        "capacity_growth": growth_records,
        "consolidation": {
            "accepted": consolidation.accepted,
            "physical_models_before": consolidation.physical_models_before,
            "physical_models_after": consolidation.physical_models_after,
            "max_heldout_difference": consolidation.max_heldout_difference,
            "source_slot_id": bank.slot_id_at(source_c_index),
            "duplicate_slot_id": duplicate_slot_id,
        },
        "compression": {
            "accepted": compression.accepted,
            "selected_codec": compression.selected_codec,
            "retention_pass": compressed_retention,
            "retention_deltas": compression_deltas,
            "receipts": [
                {
                    "codec": receipt.codec,
                    "accepted": receipt.accepted,
                    "source_bytes": receipt.source_bytes,
                    "compressed_bytes": receipt.compressed_bytes,
                }
                for receipt in compression.receipts
            ],
        },
        "alternation": {
            "rounds": ALTERNATION_ROUNDS,
            "aliases_before_copy_on_write": aliases_before_copy_on_write,
            "aliases_after_copy_on_write": aliases_after_copy_on_write,
            "pre_copy_on_write": pre_copy_on_write_alternation,
            "post_copy_on_write": post_copy_on_write_alternation,
            "compressed": compressed_alternation,
            "source_c_error_after_copy_on_write": source_c_error_after_copy_on_write,
            "duplicate_error_after_copy_on_write": duplicate_error_after_copy_on_write,
        },
        "corruption_control": {
            "statuses": dict(corruption_statuses),
            "accepted": corruption_receipt.accepted,
            "bank_unchanged": corruption_rejected,
        },
        "status_counts": dict(status_counts),
        "address_adapter": {
            "base_digest": base_adapter_digest,
            "final_digest": router.address_adapter.digest()
            if router.address_adapter is not None
            else None,
            "final_version": router.address_adapter.version
            if router.address_adapter is not None
            else None,
        },
        "accounting": {
            "unique_verifier_bits": (
                len(REGIME_NAMES) * HELDOUT_ROWS * STATE_WIDTH
                + TRAIN_ROWS * STATE_WIDTH
            ),
            "unique_logical_lifetimes": (
                len(REGIME_NAMES) * (TRAIN_ROWS + HELDOUT_ROWS) + TRAIN_ROWS
            ),
            "context_encoder_optimizer_updates": context_updates,
            "model_statistics_updates": len(REGIME_NAMES[:SOURCE_REGIMES])
            + PRESENTED_ROWS // ADMISSION_ROWS * len(TARGET_REGIMES)
            + 1,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "compaction_optimizer_updates": 0,
            "copy_on_write_statistics_updates": 1,
            "copy_on_write_fresh_rows": TRAIN_ROWS,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=82301)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
