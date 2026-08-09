"""Pressure test online context admission with interleaved opaque streams."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineContextAddressResolver,
    ExternalTransitionMemory,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 6
POSITION_COUNT = 3
ADMISSION_OBSERVATIONS = 3
REGIME_DELTAS = ((1, -1), (2, -2), (-1, 1))


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(
    seed: int,
) -> tuple[tuple[ExternalTransitionObservation, ...], torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )
    positions = (0, 1, 2)
    actions = (0, 1, 0)
    regimes: list[tuple[ExternalTransitionObservation, ...]] = []
    for deltas in REGIME_DELTAS:
        rows: list[ExternalTransitionObservation] = []
        for position, action in zip(positions, actions, strict=True):
            next_position = min(
                POSITION_COUNT - 1,
                max(0, position + deltas[action]),
            )
            rows.append(
                ExternalTransitionObservation(
                    state=state_codes[position].unsqueeze(0),
                    intention=intention_codes[action].unsqueeze(0),
                    next_state=state_codes[next_position].unsqueeze(0),
                    confidence=torch.ones(1),
                )
            )
        regimes.append(tuple(rows))
    stream_keys = F.normalize(
        torch.randn(3, CONTEXT_WIDTH, generator=generator), dim=-1
    )
    return tuple(regimes), stream_keys


def _batch(rows: tuple[ExternalTransitionObservation, ...]) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=torch.cat([row.state for row in rows]),
        intention=torch.cat([row.intention for row in rows]),
        next_state=torch.cat([row.next_state for row in rows]),
        confidence=torch.ones(len(rows)),
    )


def _diagnostic(
    memory: ExternalTransitionMemory,
    rows: tuple[ExternalTransitionObservation, ...],
    context: torch.Tensor,
) -> dict[str, float]:
    observation = _batch(rows)
    prediction, hits = memory.predict_with_hit(
        observation.state,
        observation.intention,
        context=context.unsqueeze(0).expand(len(rows), -1),
    )
    return {
        "hit_rate": float(hits.float().mean()),
        "next_state_mse": float(
            (prediction - observation.next_state).square().mean()
        ),
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    regimes, stream_keys = _fixture(seed)
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

    memory = ExternalTransitionMemory(
        STATE_WIDTH,
        INTENTION_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    resolver = ExternalOnlineContextAddressResolver(
        CONTEXT_WIDTH,
        address_seed=seed + 1000,
        admission_observations=ADMISSION_OBSERVATIONS,
        contradiction_observations=2,
    )
    trace: list[dict[str, object]] = []
    admissions: list[torch.Tensor] = []
    for row_index in range(ADMISSION_OBSERVATIONS):
        for regime_index, rows in enumerate(regimes):
            resolution = resolver.observe(
                rows[row_index],
                stream_keys[regime_index],
                memory,
            )
            if resolution.context is not None and regime_index not in range(len(admissions)):
                admissions.append(resolution.context)
            trace.append(
                {
                    "row_index": row_index,
                    "stream_index": regime_index,
                    "status": resolution.status,
                    "committed_observations": resolution.committed_observations,
                    "pending_observations": resolution.pending_observations,
                    "record_count": memory.record_count,
                }
            )

    # The first stream's context is retained before testing a reversal on that
    # same opaque stream binding.
    source_context = resolver.addresses()[0].clone()
    initial_context_count = resolver.context_count
    duplicate_key = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    record_count_before_duplicate = memory.record_count
    duplicate = resolver.observe(regimes[0][0], duplicate_key, memory)
    source_record_count = memory.record_count

    reversal_rows = (
        ExternalTransitionObservation(
            state=regimes[0][0].state,
            intention=regimes[0][0].intention,
            next_state=regimes[2][0].next_state,
            confidence=torch.ones(1),
        ),
        ExternalTransitionObservation(
            state=regimes[0][1].state,
            intention=regimes[0][1].intention,
            next_state=regimes[2][1].next_state,
            confidence=torch.ones(1),
        ),
    )
    reversal_first = resolver.observe(reversal_rows[0], stream_keys[0], memory)
    reversal_second = resolver.observe(reversal_rows[1], stream_keys[0], memory)
    reversal_context = resolver.addresses()[-1].clone()

    contexts = resolver.addresses()
    retained = [
        _diagnostic(memory, rows, contexts[index])
        for index, rows in enumerate(regimes)
    ]
    reversal_retention = _diagnostic(memory, regimes[0], source_context)
    wrong_context = _diagnostic(memory, regimes[1], source_context)

    corrupted = ExternalTransitionMemory(
        STATE_WIDTH,
        INTENTION_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    for index, rows in enumerate(regimes):
        observation = _batch(rows)
        corrupted.write(
            ExternalTransitionObservation(
                state=observation.state,
                intention=observation.intention,
                next_state=observation.next_state.roll(1, 0),
                confidence=observation.confidence,
            ),
            context=contexts[index].unsqueeze(0).expand(len(rows), -1),
        )
    corrupted_result = _diagnostic(corrupted, regimes[0], contexts[0])
    fresh = ExternalTransitionMemory(
        STATE_WIDTH,
        INTENTION_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    fresh_result = _diagnostic(fresh, regimes[0], contexts[0])

    restored_memory = ExternalTransitionMemory(
        STATE_WIDTH,
        INTENTION_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    restored_memory.store.load_state_dict(memory.store.state_dict())
    restored_resolver = ExternalOnlineContextAddressResolver.from_payload(
        resolver.payload()
    )
    restored_reversal = _diagnostic(
        restored_memory,
        reversal_rows,
        reversal_context,
    )
    controller_unchanged = controller_digest == _digest_module(controller)
    first_statuses = [item["status"] for item in trace[:6]]
    gates = {
        "controller_unchanged": controller_unchanged,
        "no_early_writes": all(
            item["record_count"] == 0 for item in trace[:6]
        ),
        "interleaved_admission": all(
            item["status"] == "admitted" for item in trace[6:9]
        ),
        "three_contexts_discovered": initial_context_count == 3,
        "duplicate_context_reused": duplicate.status == "reused"
        and source_record_count == record_count_before_duplicate,
        "all_regimes_retained": min(
            item["hit_rate"] for item in retained
        ) == 1.0,
        "reversal_waits_before_write": reversal_first.status == "conflict"
        and reversal_first.committed_observations == 0,
        "reversal_allocates_without_overwrite": reversal_second.status == "admitted"
        and reversal_retention["next_state_mse"] < 1e-6,
        "wrong_context_factual_control": wrong_context["next_state_mse"] > 1e-5,
        "corruption_factual_control": corrupted_result["next_state_mse"] > 0.1,
        "fresh_factual_control": fresh_result["hit_rate"] == 0.0,
        "persistence_exact": restored_reversal["next_state_mse"] < 1e-6,
    }
    report = {
        "schema": "neural-computer.external-online-context-admission-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "stream_count": len(regimes),
            "admission_observations": ADMISSION_OBSERVATIONS,
            "contradiction_observations": 2,
            "policy": "uncertain_partial_evidence_no_write_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "trace": trace,
        "first_statuses": first_statuses,
        "duplicate": {
            "status": duplicate.status,
            "committed_observations": duplicate.committed_observations,
        },
        "reversal": {
            "first_status": reversal_first.status,
            "second_status": reversal_second.status,
            "retention": reversal_retention,
        },
        "retained": retained,
        "wrong_context": wrong_context,
        "corrupted": corrupted_result,
        "fresh": fresh_result,
        "persisted_reversal": restored_reversal,
        "accounting": {
            "unique_transition_lifetimes": len(regimes) * ADMISSION_OBSERVATIONS + 2,
            "unique_verifier_bits": len(regimes) * ADMISSION_OBSERVATIONS + 2,
            "target_optimizer_updates": 0,
            "replayed_examples": 0,
            "memory_records_before_reversal": source_record_count,
            "memory_records_after_reversal": memory.record_count,
            "learned_context_addresses": resolver.context_count,
        },
        "controller_digest": controller_digest,
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69601)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
