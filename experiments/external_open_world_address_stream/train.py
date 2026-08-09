"""Pressure test online address formation without a pretrained encoder."""

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
    FEATURE_WIDTH,
    HELDOUT_ROWS,
    INTENTION_WIDTH,
    LOSS_THRESHOLD,
    PRESENTED_ROWS,
    RANDOM_FEATURE_FAMILY,
    STATE_WIDTH,
    TRAIN_ROWS,
    _fixture,
    _row,
)
from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextAddressAdapter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _error(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    return float(bank.loss(observation, context_batch).detach())


def _consume(
    router: ExternalOnlineTransitionContextRouter,
    observation: ExternalTransitionObservation,
) -> Counter[str]:
    statuses: Counter[str] = Counter()
    for index in range(PRESENTED_ROWS):
        result = router.observe(_row(observation, index))
        statuses[result.status] += 1
        if result.status == "staged":
            router.adaptation_step(result, None, replay_evidence=False)
    return statuses


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


def run(
    seed: int,
    report_out: Path,
    *,
    regimes: int = 8,
) -> dict[str, object]:
    if regimes < 2:
        raise ValueError("open-world stream requires at least two regimes")
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    names = tuple(f"regime_{index:02d}" for index in range(regimes))
    fixtures = {
        name: _fixture(seed, index)
        for index, name in enumerate(names)
    }
    observations = {name: pair[0] for name, pair in fixtures.items()}
    heldout = {name: pair[1] for name, pair in fixtures.items()}

    # Deliberately untrained: the stream must create identity online through
    # isolated address adaptation rather than a source-label pretraining pass.
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=CONTEXT_HIDDEN_WIDTH,
        context_width=12,
        aggregation="mean_pool",
    )
    encoder.eval()
    encoder_digest = encoder.digest()
    address_adapter = ExternalTransitionContextAddressAdapter(
        encoder,
        learning_rate=0.001,
        adaptation_steps=4,
        anchor_cosine_ceiling=0.75,
    )
    base_adapter_digest = address_adapter.digest()

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

    bank = _new_bank(1)
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=LOSS_THRESHOLD,
        match_margin=0.005,
        continuation_tolerance=LOSS_THRESHOLD,
        provisional_continuation_tolerance=1e9,
        admission_observations=ADMISSION_ROWS,
        max_contexts=1,
        defer_admission=True,
        candidate_model_families=(RANDOM_FEATURE_FAMILY,),
        provisional_evidence_policy="streaming_statistics",
        address_adapter=address_adapter,
    )
    prior_contexts: dict[str, torch.Tensor] = {}
    prior_slot_digests: dict[int, str] = {}
    acquisitions: list[dict[str, object]] = []
    growth_records: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()

    for index, name in enumerate(names):
        if index:

            def growth_probe(candidate: ExternalTransitionModelBank) -> bool:
                return all(
                    _error(candidate, heldout[old_name], old_context)
                    < LOSS_THRESHOLD
                    for old_name, old_context in prior_contexts.items()
                )

            growth = router.grow_verified(index + 1, growth_probe)
            if not growth.accepted:
                raise RuntimeError(f"capacity growth failed for {name}: {growth.reason}")
            growth_records.append(
                {
                    "source_capacity": growth.source_capacity,
                    "destination_capacity": growth.destination_capacity,
                    "context_count": growth.context_count,
                }
            )

        statuses = _consume(router, observations[name])
        status_counts.update(
            f"{name}:{key}" for key in statuses for _ in range(statuses[key])
        )
        if router.provisional_candidate_count != 1:
            raise RuntimeError(f"{name} did not produce one isolated candidate")
        context = router.provisional_context_at(0)
        receipt = router.promote_staged_candidate(
            heldout[name],
            lambda candidate: all(
                _error(candidate, heldout[old_name], old_context)
                < LOSS_THRESHOLD
                for old_name, old_context in prior_contexts.items()
            ),
            prediction_tolerance=LOSS_THRESHOLD,
        )
        if not receipt.accepted or receipt.slot_index is None:
            raise RuntimeError(f"{name} promotion failed: {receipt.reason}")
        prior_contexts[name] = context
        slot_id = bank.slot_id_at(receipt.slot_index)
        prior_slot_digests[slot_id] = bank.models[receipt.slot_index].digest()
        acquisitions.append(
            {
                "name": name,
                "statuses": dict(statuses),
                "slot_id": slot_id,
                "heldout_error": _error(bank, heldout[name], context),
                "address_version": router.address_adapter.version
                if router.address_adapter is not None
                else None,
            }
        )

    first_pass_context_count = bank.context_count
    revisit_records: list[dict[str, object]] = []
    for name in tuple(reversed(names)) + names[1:-1]:
        statuses = _consume(router, observations[name])
        status_counts.update(
            f"{name}:revisit:{key}" for key in statuses for _ in range(statuses[key])
        )
        revisit_records.append({"name": name, "statuses": dict(statuses)})

    corruption_router = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    corruption_router.bank.capacity = bank.context_count + 1
    corruption_router.max_contexts = bank.context_count + 1
    corrupted = ExternalTransitionObservation(
        state=observations[names[-1]].state[:PRESENTED_ROWS],
        intention=observations[names[-1]].intention[:PRESENTED_ROWS],
        next_state=observations[names[-1]].next_state[:PRESENTED_ROWS].roll(1, 0),
        confidence=torch.ones(PRESENTED_ROWS),
    )
    corruption_statuses = _consume(corruption_router, corrupted)
    corruption_before = corruption_router.bank.content_digest()
    corruption_receipt = corruption_router.promote_staged_candidate(
        heldout[names[-1]],
        lambda _candidate: False,
        prediction_tolerance=LOSS_THRESHOLD,
    )
    corruption_rejected = (
        corruption_statuses["staged"] > 0
        and not corruption_receipt.accepted
        and corruption_router.bank.content_digest() == corruption_before
    )

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    prior_retained = all(
        bank.models[bank.physical_index_for_slot_id(slot_id)].digest() == digest
        for slot_id, digest in prior_slot_digests.items()
    )
    revisit_matched = all(
        record["statuses"].get("matched", 0) > 0
        for record in revisit_records
    )
    gates = {
        "untrained_encoder": encoder.digest() == encoder_digest,
        "zero_encoder_pretraining_updates": True,
        "all_regimes_acquired": bank.context_count == regimes,
        "one_growth_per_new_regime": len(growth_records) == regimes - 1,
        "all_growth_retention_verified": len(growth_records) == regimes - 1,
        "all_heldout_errors_pass": all(
            float(record["heldout_error"]) < LOSS_THRESHOLD
            for record in acquisitions
        ),
        "revisits_match_existing_slots": revisit_matched,
        "no_duplicate_slots": first_pass_context_count == bank.context_count,
        "all_prior_slots_retained": prior_retained,
        "corruption_rejected_without_bank_write": corruption_rejected,
        "address_adapter_learned_online": (
            router.address_adapter is not None
            and router.address_adapter.version >= regimes
            and router.address_adapter.digest() != base_adapter_digest
        ),
        "base_adapter_copy_on_write": base_adapter_digest == address_adapter.digest(),
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
        "schema": "neural-computer.external-open-world-address-stream.v1",
        "seed": seed,
        "claim_boundary": (
            "bounded open-world-style nonlinear address formation from an "
            "untrained encoder; not unrestricted continual learning"
        ),
        "configuration": {
            "regime_count": regimes,
            "capacity_schedule": list(range(1, regimes + 1)),
            "presented_rows_per_regime": PRESENTED_ROWS,
            "heldout_rows_per_regime": HELDOUT_ROWS,
            "admission_rows": ADMISSION_ROWS,
            "context_encoder_optimizer_updates": 0,
            "model_family": RANDOM_FEATURE_FAMILY,
            "address_update": "copy_on_write_current_bundle_anchor_separation_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "acquisitions": acquisitions,
        "revisits": revisit_records,
        "capacity_growth": growth_records,
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
            "unique_verifier_bits": regimes * HELDOUT_ROWS * STATE_WIDTH,
            "unique_logical_lifetimes": regimes * (TRAIN_ROWS + HELDOUT_ROWS),
            "context_encoder_optimizer_updates": 0,
            "address_adapter_optimizer_updates": (
                router.address_adapter.version * 4
                if router.address_adapter is not None
                else 0
            ),
            "model_statistics_updates": regimes * PRESENTED_ROWS // ADMISSION_ROWS,
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
    parser.add_argument("--seed", type=int, default=82401)
    parser.add_argument("--regimes", type=int, default=8)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out, regimes=args.regimes)


if __name__ == "__main__":
    main()
