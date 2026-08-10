"""Verify reversible online factual versioning without overwriting old facts."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineContextAddressResolver,
    ExternalTransitionMemory,
    ExternalTransitionObservation,
)

STATE_WIDTH = 3
INTENTION_WIDTH = 2
CONTEXT_WIDTH = 4
ADMISSION_OBSERVATIONS = 3
CONTRADICTION_OBSERVATIONS = 2


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _row(state: int, intention: int, next_state: int) -> ExternalTransitionObservation:
    state_code = torch.nn.functional.one_hot(
        torch.tensor([state]), num_classes=STATE_WIDTH
    ).float()
    intention_code = torch.nn.functional.one_hot(
        torch.tensor([intention]), num_classes=INTENTION_WIDTH
    ).float()
    next_state_code = torch.nn.functional.one_hot(
        torch.tensor([next_state]), num_classes=STATE_WIDTH
    ).float()
    return ExternalTransitionObservation(
        state=state_code,
        intention=intention_code,
        next_state=next_state_code,
        confidence=torch.ones(1),
    )


def _fixture(seed: int) -> tuple[tuple[ExternalTransitionObservation, ...], tuple[ExternalTransitionObservation, ...], torch.Tensor]:
    # Seed controls only the opaque stream binding.  The two regimes share the
    # same input space but encode different facts, forcing versioning rather
    # than a new modality or a hand-written semantic label.
    torch.manual_seed(seed)
    source = (
        _row(0, 0, 1),
        _row(1, 1, 2),
        _row(2, 0, 0),
    )
    reversal = (
        _row(0, 0, 2),
        _row(1, 1, 0),
    )
    stream_key = torch.tensor([1.0, 0.0, 1.0, 0.0])
    return source, reversal, stream_key


def _predict(
    memory: ExternalTransitionMemory,
    rows: tuple[ExternalTransitionObservation, ...],
    context: torch.Tensor,
) -> tuple[float, float]:
    observation = ExternalTransitionObservation(
        state=torch.cat([item.state for item in rows]),
        intention=torch.cat([item.intention for item in rows]),
        next_state=torch.cat([item.next_state for item in rows]),
        confidence=torch.ones(len(rows)),
    )
    prediction, hits = memory.predict_with_hit(
        observation.state,
        observation.intention,
        context=context.unsqueeze(0).expand(len(rows), -1),
    )
    return (
        float(hits.float().mean()),
        float((prediction - observation.next_state).square().mean()),
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    source, reversal, stream_key = _fixture(seed)
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
        contradiction_observations=CONTRADICTION_OBSERVATIONS,
    )
    trace: list[dict[str, object]] = []

    source_results = [resolver.observe(item, stream_key, memory) for item in source]
    source_context = source_results[-1].context
    if source_context is None:
        raise RuntimeError("source regime did not admit")
    source_context = source_context.clone()
    source_record_count = memory.record_count
    source_before = _predict(memory, source, source_context)

    conflict = resolver.observe(reversal[0], stream_key, memory)
    reversal_admission = resolver.observe(reversal[1], stream_key, memory)
    reversal_context = reversal_admission.context
    if reversal_context is None:
        raise RuntimeError("reversal regime did not admit")
    reversal_context = reversal_context.clone()
    reversal_record_count = memory.record_count
    reversal_before = _predict(memory, reversal, reversal_context)

    # Alternate regimes after the reversal.  These are routing-only reads;
    # neither return path may allocate or mutate the factual store.
    source_return = resolver.observe(source[0], stream_key, memory)
    reversal_return = resolver.observe(reversal[0], stream_key, memory)
    source_after = _predict(memory, source, source_context)
    reversal_after = _predict(memory, reversal, reversal_context)
    trace.extend(
        [
            {
                "phase": "source_admission",
                "statuses": [item.status for item in source_results],
                "committed": [item.committed_observations for item in source_results],
            },
            {
                "phase": "reversal_admission",
                "statuses": [conflict.status, reversal_admission.status],
                "committed": [
                    conflict.committed_observations,
                    reversal_admission.committed_observations,
                ],
            },
            {
                "phase": "version_reactivation",
                "statuses": [source_return.status, reversal_return.status],
                "contexts": [
                    "source" if torch.allclose(source_return.context, source_context) else "other",
                    "reversal" if torch.allclose(reversal_return.context, reversal_context) else "other",
                ],
            },
        ]
    )

    restored_memory = ExternalTransitionMemory(
        STATE_WIDTH,
        INTENTION_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    restored_memory.store.load_state_dict(memory.store.state_dict())
    restored_resolver = ExternalOnlineContextAddressResolver.from_payload(
        resolver.payload()
    )
    persisted_source = restored_resolver.observe(source[0], stream_key, restored_memory)
    persisted_reversal = restored_resolver.observe(reversal[0], stream_key, restored_memory)
    persisted_reversal_prediction = _predict(
        restored_memory,
        reversal,
        restored_resolver.addresses()[1],
    )

    corrupted = ExternalTransitionMemory(
        STATE_WIDTH,
        INTENTION_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    for rows, context in ((source, source_context), (reversal, reversal_context)):
        observation = ExternalTransitionObservation(
            state=torch.cat([item.state for item in rows]),
            intention=torch.cat([item.intention for item in rows]),
            next_state=torch.cat([item.next_state for item in rows]).roll(1, 0),
            confidence=torch.ones(len(rows)),
        )
        corrupted.write(
            observation,
            context=context.unsqueeze(0).expand(len(rows), -1),
        )
    corrupted_source = _predict(corrupted, source, source_context)
    fresh_source = _predict(
        ExternalTransitionMemory(STATE_WIDTH, INTENTION_WIDTH, context_width=CONTEXT_WIDTH),
        source,
        source_context,
    )
    controller_unchanged = controller_digest == _digest_module(controller)
    gates = {
        "controller_unchanged": controller_unchanged,
        "source_waits_then_admits": [item.status for item in source_results]
        == ["uncertain", "uncertain", "admitted"],
        "reversal_waits_before_write": conflict.status == "conflict"
        and conflict.committed_observations == 0,
        "new_version_without_overwrite": reversal_admission.status == "admitted"
        and reversal_record_count == source_record_count + len(reversal),
        "two_versions_retained": resolver.context_count == 2
        and source_after[0] == 1.0
        and source_after[1] < 1e-8
        and reversal_after[0] == 1.0
        and reversal_after[1] < 1e-8,
        "source_reactivated": source_return.status == "reused"
        and torch.allclose(source_return.context, source_context),
        "reversal_reactivated": reversal_return.status == "reused"
        and torch.allclose(reversal_return.context, reversal_context),
        "reactivation_does_not_grow": resolver.context_count == 2
        and memory.record_count == reversal_record_count,
        "persisted_versions_route": persisted_source.status == "reused"
        and persisted_reversal.status == "reused",
        "corruption_control": corrupted_source[1] > 0.1,
        "fresh_control": fresh_source[0] == 0.0,
    }
    report = {
        "schema": "neural-computer.external-online-context-versioning.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "admission_observations": ADMISSION_OBSERVATIONS,
            "contradiction_observations": CONTRADICTION_OBSERVATIONS,
            "policy": "same_stream_version_history_read_only_reactivation_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "trace": trace,
        "source": {"before": source_before, "after": source_after},
        "reversal": {"before": reversal_before, "after": reversal_after},
        "persisted_routing": {
            "source_status": persisted_source.status,
            "reversal_status": persisted_reversal.status,
            "reversal_prediction": persisted_reversal_prediction,
        },
        "corrupted": corrupted_source,
        "fresh": fresh_source,
        "accounting": {
            "unique_transition_lifetimes": len(source) + len(reversal) + 2,
            "unique_verifier_bits": len(source) + len(reversal) + 2,
            "target_optimizer_updates": 0,
            "replayed_examples": 0,
            "memory_records_before_reversal": source_record_count,
            "memory_records_after_reversal": reversal_record_count,
            "learned_context_versions": resolver.context_count,
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
    parser.add_argument("--seed", type=int, default=96001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
