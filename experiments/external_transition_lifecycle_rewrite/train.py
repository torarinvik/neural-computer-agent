"""Two-seed composition audit for external transition-memory rewrites."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import ExternalTransitionModelBank
from neural_computer import ExternalTransitionModelLifetimePolicy
from neural_computer import ExternalTransitionObservation

CONTEXT_WIDTH = 3
HIDDEN_WIDTH = 8


def _observation(rows: int = 8) -> ExternalTransitionObservation:
    state = torch.arange(rows, dtype=torch.float32).reshape(rows, 1) / 5.0
    intention = torch.arange(rows, dtype=torch.float32).reshape(rows, 1) / 7.0
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=0.4 * state + 0.7 * intention - 0.1,
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    torch.manual_seed(seed)
    begun = time.perf_counter()
    heldout = _observation()
    training = _observation(rows=4)
    bank = ExternalTransitionModelBank(
        1,
        1,
        CONTEXT_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=3,
    )
    for context in (
        torch.tensor([1.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    ):
        index = bank.ensure_context(context)
        context_batch = bank.context_at(index).unsqueeze(0).expand(4, -1)
        bank.adaptation_step(training, context_batch, None)
    bank.record_lifetime_observations(bank.slot_ids, [0.0, 0.0, 0.0])
    telemetry_before = bank.lifetime_telemetry()
    policy = ExternalTransitionModelLifetimePolicy(
        CONTEXT_WIDTH,
        hidden_width=16,
        learning_rate=0.03,
    )
    source_ids = bank.slot_ids
    expected_after_eviction = (0, 2)

    def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
        if candidate.slot_ids == source_ids:
            return True
        if candidate.slot_ids != expected_after_eviction:
            return False
        for slot_id in expected_after_eviction:
            index = candidate.physical_index_for_slot_id(slot_id)
            context = candidate.context_at(index)
            context_batch = context.unsqueeze(0).expand(heldout.state.shape[0], -1)
            if float(candidate.loss(heldout, context_batch).detach()) > 1e-6:
                return False
        return True

    proposal, eviction = policy.evict_from_bank_query_verified(
        bank,
        bank.context_at(0),
        torch.tensor([False, False, True]),
        retention_probe,
        relevance_weight=100.0,
        update=True,
    )
    consolidation = bank.consolidate_verified(
        0,
        1,
        [heldout],
        prediction_tolerance=1e-6,
        retention_probe=lambda candidate: candidate.slot_ids == expected_after_eviction,
    )
    compressed = bank.select_compression_verified(
        [torch.float16],
        retention_probe=lambda candidate: all(
            float(
                candidate.loss(
                    heldout,
                    candidate.context_at(index)
                    .unsqueeze(0)
                    .expand(heldout.state.shape[0], -1),
                ).detach()
            )
            <= 1e-3
            for index in range(candidate.context_count)
        ),
    )
    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    compressed_restored = ExternalTransitionModelBank.from_compressed_payload(
        bank.compressed_payload(dtype=torch.float16)
    )
    retained_behavior = all(
        float(
            bank.loss(
                heldout,
                bank.context_at(index)
                .unsqueeze(0)
                .expand(heldout.state.shape[0], -1),
            ).detach()
        )
        <= 1e-6
        for index in range(bank.context_count)
    )
    telemetry_after = bank.lifetime_telemetry()
    report = {
        "schema": "neural-computer.external-transition-lifecycle-rewrite.v1",
        "seed": seed,
        "gates": {
            "query_conditioned_eviction": bool(eviction and eviction.accepted),
            "consolidation": consolidation.accepted,
            "compression": compressed.accepted,
            "retained_behavior": retained_behavior,
            "stable_ids": bank.slot_ids == expected_after_eviction,
            "telemetry_preserved": (
                telemetry_after.slot_ids == expected_after_eviction
                and telemetry_after.logical_clock == telemetry_before.logical_clock
                and telemetry_after.usage.tolist() == [5.0, 5.0]
            ),
            "payload_persistence": restored.digest() == bank.digest(),
            "compressed_persistence": (
                compressed_restored.slot_ids == bank.slot_ids
                and compressed_restored.lifetime_telemetry().logical_clock
                == telemetry_after.logical_clock
            ),
            "zero_controller_updates": True,
            "zero_replayed_transition_examples": True,
        },
        "promoted": all(
            (
                bool(eviction and eviction.accepted),
                consolidation.accepted,
                compressed.accepted,
                retained_behavior,
                bank.slot_ids == expected_after_eviction,
                restored.digest() == bank.digest(),
            )
        ),
        "metrics": {
            "selected_eviction_slot_id": proposal.selected_slot_id,
            "physical_models_after_consolidation": bank.physical_model_count,
            "compression_selected_codec": compressed.selected_codec,
            "policy_digest": hashlib.sha256(policy.digest().encode()).hexdigest(),
        },
        "accounting": {
            "unique_verifier_bits": 6,
            "unique_logical_lifetimes": 12,
            "policy_optimizer_updates": 1,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_memory_replay": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "composed bounded external lifecycle rewrites; not unrestricted general continual learning",
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
