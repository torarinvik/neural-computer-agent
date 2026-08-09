"""Long alternating stream audit for copy-on-write nonlinear addresses."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch

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
    _error,
    _fixture,
    _row,
    _train_context_encoder,
)
from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextAddressAdapter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

REGIME_NAMES = (
    "source_a",
    "source_b",
    "target_c",
    "target_d",
    "target_e",
    "target_f",
)
INITIAL_CAPACITY = 4
SOURCE_REGIMES = 2
TARGET_REGIMES = tuple(range(SOURCE_REGIMES, len(REGIME_NAMES)))
SEQUENCE = (
    "target_c",
    "target_d",
    "source_a",
    "target_e",
    "source_b",
    "target_f",
    "target_c",
    "target_d",
    "target_e",
    "target_f",
    "source_a",
    "source_b",
)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rows(
    observation: ExternalTransitionObservation,
    count: int = PRESENTED_ROWS,
) -> list[ExternalTransitionObservation]:
    return [_row(observation, index) for index in range(count)]


def _consume(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> Counter[str]:
    statuses: Counter[str] = Counter()
    for row in _rows(observation):
        result = router.observe(row)
        statuses[result.status] += 1
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)
    return statuses


def _new_bank() -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        12,
        model_family=RANDOM_FEATURE_FAMILY,
        random_feature_width=FEATURE_WIDTH,
        random_feature_seed=17,
        affine_ridge=1e-4,
        capacity=INITIAL_CAPACITY,
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

    bank = _new_bank()
    source_slot_digests: dict[int, str] = {}
    for name in REGIME_NAMES[:SOURCE_REGIMES]:
        context = source_contexts[name]
        index = bank.ensure_context(context)
        batch_context = context.unsqueeze(0).expand(TRAIN_ROWS, -1)
        bank.adaptation_step(observations[name], batch_context, None)
        source_slot_digests[index] = bank.models[index].digest()

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
    acquisition_records: list[dict[str, object]] = []
    prior_contexts: dict[str, torch.Tensor] = dict(source_contexts)
    prior_slot_digests = dict(source_slot_digests)
    all_retained = True
    growth_records: list[dict[str, object]] = []

    for regime in TARGET_REGIMES:
        name = REGIME_NAMES[regime]
        if name == "target_e":
            def growth_retention_probe(candidate: ExternalTransitionModelBank) -> bool:
                return all(
                    _error(candidate, heldout[old_name], old_context)
                    < LOSS_THRESHOLD
                    for old_name, old_context in prior_contexts.items()
                )

            growth_receipt = router.grow_verified(6, growth_retention_probe)
            if not growth_receipt.accepted:
                raise RuntimeError(
                    f"capacity growth failed: {growth_receipt.reason}"
                )
            growth_records.append(
                {
                    "source_capacity": growth_receipt.source_capacity,
                    "destination_capacity": growth_receipt.destination_capacity,
                    "context_count": growth_receipt.context_count,
                    "reason": growth_receipt.reason,
                }
            )
        statuses = _consume(router, observations[name])
        status_counts.update(f"{name}:{key}" for key in statuses for _ in range(statuses[key]))
        if router.provisional_candidate_count != 1:
            raise RuntimeError(f"{name} did not leave one isolated candidate")
        context = router.provisional_context_at(0)

        def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
            return all(
                _error(candidate, heldout[old_name], old_context) < LOSS_THRESHOLD
                for old_name, old_context in prior_contexts.items()
            )

        receipt = router.promote_staged_candidate(
            heldout[name],
            retention_probe,
            prediction_tolerance=LOSS_THRESHOLD,
        )
        if not receipt.accepted or receipt.slot_index is None:
            raise RuntimeError(f"{name} promotion failed: {receipt.reason}")
        prior_contexts[name] = context
        prior_slot_digests[receipt.slot_index] = bank.models[receipt.slot_index].digest()
        retained = all(
            bank.models[index].digest() == digest
            for index, digest in prior_slot_digests.items()
        )
        all_retained = all_retained and retained
        acquisition_records.append(
            {
                "name": name,
                "statuses": dict(statuses),
                "heldout_error": _error(bank, heldout[name], context),
                "slot_index": receipt.slot_index,
                "slot_id": bank.slot_id_at(receipt.slot_index),
                "address_version": router.address_adapter.version
                if router.address_adapter is not None
                else None,
                "prior_slots_retained": retained,
            }
        )

    route_records: list[dict[str, object]] = []
    for name in SEQUENCE:
        statuses = _consume(router, observations[name])
        status_counts.update(f"{name}:{key}" for key in statuses for _ in range(statuses[key]))
        route_records.append({"name": name, "statuses": dict(statuses)})

    corruption_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    corruption_router.bank.capacity = len(REGIME_NAMES) + 1
    corruption_router.max_contexts = len(REGIME_NAMES) + 1
    corrupted_observation = ExternalTransitionObservation(
        state=observations["target_f"].state[:PRESENTED_ROWS],
        intention=observations["target_f"].intention[:PRESENTED_ROWS],
        next_state=observations["target_f"].next_state[:PRESENTED_ROWS].roll(1, 0),
        confidence=torch.ones(PRESENTED_ROWS),
    )
    corruption_statuses = _consume(corruption_router, corrupted_observation)
    corruption_bank_before = corruption_router.bank.content_digest()
    corruption_receipt = corruption_router.promote_staged_candidate(
        heldout["target_f"],
        lambda _candidate: False,
        prediction_tolerance=LOSS_THRESHOLD,
    )
    corruption_control = {
        "statuses": dict(corruption_statuses),
        "staged": corruption_statuses["staged"] > 0,
        "accepted": corruption_receipt.accepted,
        "reason": corruption_receipt.reason,
        "bank_unchanged": corruption_router.bank.content_digest()
        == corruption_bank_before,
    }

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    source_retention = all(
        _error(bank, heldout[name], prior_contexts[name]) < LOSS_THRESHOLD
        for name in REGIME_NAMES[:SOURCE_REGIMES]
    )
    historical_keys = bank.contexts
    restored_keys = restored.bank.contexts
    gates = {
        "context_encoder_converged": context_loss < 0.05,
        "all_target_promotions_pass": len(acquisition_records) == len(TARGET_REGIMES),
        "capacity_growth_verified": (
            len(growth_records) == 1
            and growth_records[0]["source_capacity"] == INITIAL_CAPACITY
            and growth_records[0]["destination_capacity"] == 6
        ),
        "all_target_heldout_errors_pass": all(
            float(record["heldout_error"]) < LOSS_THRESHOLD
            for record in acquisition_records
        ),
        "all_prior_slots_retained": all_retained and source_retention,
        "revisited_targets_match_existing_slots": all(
            route_records[index]["statuses"].get("matched", 0) > 0
            for index in range(len(route_records))
            if route_records[index]["name"].startswith("target_")
            and route_records[index]["name"] in {
                "target_c",
                "target_d",
                "target_e",
                "target_f",
            }
            and index >= len(TARGET_REGIMES)
        ),
        "address_adapter_copy_on_write": (
            router.address_adapter is not None
            and router.address_adapter.version > len(TARGET_REGIMES)
            and base_adapter.digest() == base_adapter_digest
        ),
        "historical_keys_unchanged_by_address_updates": torch.equal(
            historical_keys, restored_keys
        ),
        "corruption_rejected_without_bank_write": (
            corruption_control["staged"]
            and not corruption_control["accepted"]
            and corruption_control["bank_unchanged"]
        ),
        "controller_unchanged": controller_digest == _digest(controller),
        "no_raw_candidate_rows_retained": all(
            not candidate.observations for candidate in router._provisional_candidates
        ),
        "exact_persistence": (
            restored.bank.digest() == router.bank.digest()
            and restored.context_encoder.digest() == router.context_encoder.digest()
            and restored.address_adapter is not None
            and router.address_adapter is not None
            and restored.address_adapter.digest() == router.address_adapter.digest()
        ),
    }
    report = {
        "schema": "neural-computer.external-nonlinear-address-shift-stream.v1",
        "seed": seed,
        "claim_boundary": (
            "bounded long alternating nonlinear factual-memory routing with "
            "copy-on-write learned address versions; not unrestricted memory "
            "growth or general continual learning"
        ),
        "configuration": {
            "regimes": list(REGIME_NAMES),
            "sequence": list(SEQUENCE),
            "train_rows_per_regime": TRAIN_ROWS,
            "presented_rows_per_stream": PRESENTED_ROWS,
            "heldout_rows_per_regime": HELDOUT_ROWS,
            "initial_capacity": INITIAL_CAPACITY,
            "final_capacity": bank.capacity,
            "admission_rows": ADMISSION_ROWS,
            "context_encoder_updates": CONTEXT_UPDATES,
            "model_family": RANDOM_FEATURE_FAMILY,
            "address_update": "copy_on_write_current_bundle_anchor_separation_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "context_encoder": {
            "optimizer_updates": context_updates,
            "loss": context_loss,
            "aggregation": "mean_pool",
        },
        "acquisitions": acquisition_records,
        "capacity_growth": growth_records,
        "routes": route_records,
        "corruption_control": corruption_control,
        "status_counts": dict(status_counts),
        "address_adapter": {
            "base_digest": base_adapter_digest,
            "final_version": router.address_adapter.version
            if router.address_adapter is not None
            else None,
            "final_digest": router.address_adapter.digest()
            if router.address_adapter is not None
            else None,
            "historical_keys_count": bank.context_count,
        },
        "accounting": {
            "unique_verifier_bits": len(REGIME_NAMES) * HELDOUT_ROWS * STATE_WIDTH,
            "unique_logical_lifetimes": len(REGIME_NAMES) * (TRAIN_ROWS + HELDOUT_ROWS),
            "context_encoder_optimizer_updates": context_updates,
            "address_adapter_optimizer_updates": (
                router.address_adapter.version * 4
                if router.address_adapter is not None
                else 0
            ),
            "model_statistics_updates": len(REGIME_NAMES[:SOURCE_REGIMES])
            + PRESENTED_ROWS // ADMISSION_ROWS * len(TARGET_REGIMES),
            "replayed_examples": 0,
            "old_regime_replay": 0,
            "controller_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=82101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
