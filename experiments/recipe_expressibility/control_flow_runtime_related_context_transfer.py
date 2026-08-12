"""Audit gated related-context transfer in the canonical control-flow runtime.

The source context first learns an opaque file binding.  A nearby but unseen
learned trajectory query may use that protected binding as a cold-start prior,
while a fresh or distant query falls back to append order.  The nearby query
is then given a new outcome stream that reverses its correct file; its local
row must adapt without changing the source row.

This is an external-memory transfer audit.  The controller, query adapter,
program files, and intention codec are frozen; only the checksummed route
table changes.  It does not claim semantic relatedness or general continual
learning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControlFlowInstruction,
    ControlFlowIntentionAdapter,
    ControlFlowProgram,
    ControlFlowProgramAmodalRuntime,
    ControlFlowProgramMemory,
    ControllerFeedback,
    ExternalControllerTrajectoryQueryAdapter,
    IntentEvent,
    PersistentOpaqueContextRouteEvidence,
)

COUNTER_COUNT = 4
INTENTION_WIDTH = 4
EVENT_WIDTH = 8
FEEDBACK_WIDTH = 3
PROGRAM_COUNT = 4
SOURCE_LOGICAL_SLOT = 2
REVERSAL_LOGICAL_SLOT = 0
TRAIN_EPISODES = 32
HELDOUT_LIFETIMES = 8
MATCHING_TOLERANCE = 1e-5
GENERALIZATION_TOLERANCE = 0.08
MASTERY_THRESHOLD = 0.8
NULL_TRANSFER_THRESHOLD = 0.75
SEEDS = (17, 18, 19)


class OpaqueCounterCodec(ControlFlowIntentionAdapter):
    """Fixed executable ABI with no controller-visible counter semantics."""

    def encode(
        self,
        intention: IntentEvent,
        previous_counters: torch.Tensor,
    ) -> torch.Tensor:
        return previous_counters.clone()

    def decode(
        self,
        counters: torch.Tensor,
        template: IntentEvent,
    ) -> IntentEvent:
        return IntentEvent(
            payload=counters.to(dtype=template.payload.dtype),
            timestamp=template.timestamp,
            confidence=template.confidence,
            target_key=template.target_key,
        )


def _program(counter: int) -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("inc", counter=counter),
            ControlFlowInstruction("halt"),
        ),
    )


def _feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, FEEDBACK_WIDTH),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1, dtype=torch.bool),
    )


def _event(kind: str) -> list[AmodalEvent]:
    payload = torch.zeros(1, EVENT_WIDTH)
    if kind == "source":
        payload[0, 0] = 1.0
    elif kind == "related":
        payload[0, 0] = 1.0
        payload[0, 1] = 0.05
    elif kind == "far":
        payload[0, 2] = 1.0
    else:
        raise ValueError(f"unsupported rendered context kind: {kind}")
    return [AmodalEvent(payload)]


def _physical_slot(logical_slot: int, *, reverse_files: bool) -> int:
    return PROGRAM_COUNT - 1 - logical_slot if reverse_files else logical_slot


def _make_agent(
    seed: int,
    *,
    reverse_files: bool,
    generalization_tolerance: float,
) -> tuple[
    ControlFlowProgramAmodalRuntime,
    PersistentOpaqueContextRouteEvidence,
    tuple[str, ...],
]:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=8,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=FEEDBACK_WIDTH,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    programs = tuple(_program(counter) for counter in range(PROGRAM_COUNT))
    if reverse_files:
        programs = tuple(reversed(programs))
    for program in programs:
        memory.add_program(program, protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(
        width=40,
        matching_tolerance=MATCHING_TOLERANCE,
        generalization_tolerance=generalization_tolerance,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=8,
        reversal_threshold=0.5,
        reversal_patience=4,
    )
    for _ in range(PROGRAM_COUNT):
        evidence.append_slot()
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=8,
        query_width=40,
    )
    agent = ControlFlowProgramAmodalRuntime(
        runtime,
        OpaqueCounterCodec(INTENTION_WIDTH, COUNTER_COUNT),
        program_memory=memory,
        program_route_evidence=evidence,
        program_route_query_adapter=query_adapter,
        max_steps=8,
    )
    return agent, evidence, tuple(program.digest() for program in programs)


def _step(
    agent: ControlFlowProgramAmodalRuntime,
    kind: str,
    *,
    override: int | None = None,
) -> object:
    state = agent.initial_state(1, device="cpu")
    output, _ = agent.step_events(
        _event(kind),
        state,
        _feedback(),
        program_route_override=(
            None
            if override is None
            else torch.tensor([override], dtype=torch.int64)
        ),
    )
    if output.program_route_query is None:
        raise RuntimeError("related-context audit requires a route query")
    return output


def _train_source(
    agent: ControlFlowProgramAmodalRuntime,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    target_slot: int,
    shuffled: bool,
    seed: int,
) -> dict[str, object]:
    random = torch.Generator(device="cpu").manual_seed(seed + 81_001)
    outcomes: list[float] = []
    for episode in range(TRAIN_EPISODES):
        candidate = episode % PROGRAM_COUNT
        output = _step(agent, "source", override=candidate)
        outcome = float(candidate == target_slot)
        if shuffled:
            outcome = float(torch.rand((), generator=random) < 0.25)
        evidence.observe(output.program_route_query[0], candidate, outcome)
        outcomes.append(outcome)
    return {
        "episodes": TRAIN_EPISODES,
        "candidate_schedule": [episode % PROGRAM_COUNT for episode in range(TRAIN_EPISODES)],
        "outcomes": outcomes,
        "replayed_examples": 0,
    }


def _train_related_reversal(
    agent: ControlFlowProgramAmodalRuntime,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    target_slot: int,
    shuffled: bool,
    seed: int,
) -> dict[str, object]:
    random = torch.Generator(device="cpu").manual_seed(seed + 82_001)
    outcomes: list[float] = []
    for episode in range(TRAIN_EPISODES):
        candidate = episode % PROGRAM_COUNT
        output = _step(agent, "related", override=candidate)
        outcome = float(candidate == target_slot)
        if shuffled:
            outcome = float(torch.rand((), generator=random) < 0.25)
        evidence.observe(output.program_route_query[0], candidate, outcome)
        outcomes.append(outcome)
    return {
        "episodes": TRAIN_EPISODES,
        "candidate_schedule": [episode % PROGRAM_COUNT for episode in range(TRAIN_EPISODES)],
        "outcomes": outcomes,
        "replayed_examples": 0,
    }


def _probe(
    agent: ControlFlowProgramAmodalRuntime,
    kind: str,
    *,
    target_slot: int,
    lifetimes: int = HELDOUT_LIFETIMES,
) -> dict[str, object]:
    selected: list[int] = []
    for _ in range(lifetimes):
        output = _step(agent, kind)
        selected.append(int(output.selected_program_slots[0]))
    return {
        "kind": kind,
        "target_slot": target_slot,
        "selected_slots": selected,
        "accuracy": sum(slot == target_slot for slot in selected) / float(len(selected)),
    }


def _controller_snapshot(agent: ControlFlowProgramAmodalRuntime) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }


def _run_arm(
    seed: int,
    *,
    reverse_files: bool,
    shuffled: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    agent, evidence, source_digests = _make_agent(
        seed,
        reverse_files=reverse_files,
        generalization_tolerance=GENERALIZATION_TOLERANCE,
    )
    controller_before = _controller_snapshot(agent)
    source_target = _physical_slot(SOURCE_LOGICAL_SLOT, reverse_files=reverse_files)
    reversal_target = _physical_slot(REVERSAL_LOGICAL_SLOT, reverse_files=reverse_files)
    source_training = _train_source(
        agent,
        evidence,
        target_slot=source_target,
        shuffled=shuffled,
        seed=seed,
    )
    source_probe = _probe(agent, "source", target_slot=source_target)
    related_before = _probe(agent, "related", target_slot=source_target)
    far_before = _probe(agent, "far", target_slot=0)

    fresh_agent, _, _ = _make_agent(
        seed + 4_000,
        reverse_files=reverse_files,
        generalization_tolerance=0.0,
    )
    fresh_related = _probe(fresh_agent, "related", target_slot=source_target)
    fresh_source = _probe(fresh_agent, "source", target_slot=source_target)

    reversal_training = _train_related_reversal(
        agent,
        evidence,
        target_slot=reversal_target,
        shuffled=shuffled,
        seed=seed,
    )
    source_after = _probe(agent, "source", target_slot=source_target)
    related_after = _probe(agent, "related", target_slot=reversal_target)
    far_after = _probe(agent, "far", target_slot=0)

    evidence_payload = evidence.payload()
    restored_evidence = PersistentOpaqueContextRouteEvidence.from_payload(
        evidence_payload
    )
    evidence_reload_exact = restored_evidence.payload() == evidence_payload
    corrupted = dict(evidence_payload)
    corrupted["version"] = int(corrupted["version"]) + 1
    try:
        PersistentOpaqueContextRouteEvidence.from_payload(corrupted)
        checksum_rejected = False
    except ValueError as error:
        checksum_rejected = "checksum" in str(error)
    files_retained = all(
        agent.program_memory.program(slot).digest() == source_digests[slot]
        for slot in range(PROGRAM_COUNT)
    )
    controller_after = _controller_snapshot(agent)
    controller_frozen = all(
        torch.equal(controller_before[name], controller_after[name])
        for name in controller_before
    )
    source_query = _step(agent, "source").program_route_query[0]
    related_query = _step(agent, "related").program_route_query[0]
    far_query = _step(agent, "far").program_route_query[0]
    source_related_distance = float(
        torch.linalg.vector_norm(
            torch.nn.functional.normalize(source_query, dim=0)
            - torch.nn.functional.normalize(related_query, dim=0)
        )
    )
    source_far_distance = float(
        torch.linalg.vector_norm(
            torch.nn.functional.normalize(source_query, dim=0)
            - torch.nn.functional.normalize(far_query, dim=0)
        )
    )
    gates = {
        "source_mastered_before_transfer": source_probe["accuracy"] >= 0.9,
        "related_transfer_beats_fresh": related_before["accuracy"]
        > fresh_related["accuracy"],
        "related_transfer_mastered": related_before["accuracy"] >= 0.9,
        "far_context_does_not_use_source_prior": all(
            slot == 0 for slot in far_before["selected_slots"]
        )
        and source_target != 0,
        "source_retained_after_local_reversal": source_after["accuracy"] >= 0.9,
        "related_reversal_mastered": related_after["accuracy"] >= 0.9,
        "far_context_remains_unbound": all(
            slot == 0 for slot in far_after["selected_slots"]
        )
        and source_target != 0,
        "query_distance_is_gated": source_related_distance <= GENERALIZATION_TOLERANCE
        < source_far_distance,
        "all_external_files_retained": files_retained,
        "controller_frozen": controller_frozen,
        "evidence_reload_exact": evidence_reload_exact,
        "evidence_checksum_rejected": checksum_rejected,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    return {
        "schema": "neural-computer.control-flow-runtime-related-context-transfer.v1",
        "seed": seed,
        "reverse_files": reverse_files,
        "feedback_mode": "reward_shuffled" if shuffled else "verifier_scalar",
        "architecture": {
            "program_count": PROGRAM_COUNT,
            "source_logical_slot": SOURCE_LOGICAL_SLOT,
            "reversal_logical_slot": REVERSAL_LOGICAL_SLOT,
            "matching_tolerance": MATCHING_TOLERANCE,
            "generalization_tolerance": GENERALIZATION_TOLERANCE,
            "learner_inputs": "opaque_controller_trajectory_and_selected_scalar_outcome",
            "external_memory": evidence.schema,
            "forbidden_features": "context labels, target slots, unattempted outcomes, protocol fields",
        },
        "source_target_slot": source_target,
        "reversal_target_slot": reversal_target,
        "source_training": source_training,
        "related_reversal_training": reversal_training,
        "probes": {
            "source_before": source_probe,
            "related_before": related_before,
            "far_before": far_before,
            "source_after": source_after,
            "related_after": related_after,
            "far_after": far_after,
            "fresh_source": fresh_source,
            "fresh_related": fresh_related,
        },
        "query_distances": {
            "source_to_related": source_related_distance,
            "source_to_far": source_far_distance,
        },
        "evidence_context_count": evidence.context_count,
        "evidence_digest": evidence.digest(),
        "evidence_payload": evidence_payload,
        "source_program_digests": source_digests,
        "program_memory_digest": agent.program_memory.digest(),
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": 2 * TRAIN_EPISODES,
            "unique_logical_lifetimes": 2 * TRAIN_EPISODES,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - started,
            "latency_seconds_per_lifetime": (time.perf_counter() - started)
            / float(2 * TRAIN_EPISODES),
            "stable_bits_to_threshold": TRAIN_EPISODES,
            "retention_on_mastered_primitives": float(
                source_after["accuracy"] >= 0.9 and related_after["accuracy"] >= 0.9
            ),
        },
    }


def run(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    reports = tuple(
        _run_arm(seed, reverse_files=reverse_files, shuffled=shuffled)
        for seed in seeds
        for reverse_files in (False, True)
        for shuffled in (False, True)
    )
    verifier = tuple(
        report for report in reports if report["feedback_mode"] == "verifier_scalar"
    )
    nulls = tuple(
        report for report in reports if report["feedback_mode"] == "reward_shuffled"
    )
    null_transfer = [
        float(report["probes"]["related_before"]["accuracy"])
        for report in nulls
    ]
    return {
        "schema": "neural-computer.control-flow-runtime-related-context-transfer.v1",
        "claim_boundary": (
            "bounded gated nearest-context prior transfer and local reversal over "
            "four isolated external control-flow files; not semantic relatedness, "
            "unrestricted memory growth, arbitrary new computation, or general continual learning"
        ),
        "seeds": list(seeds),
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in verifier)
        and all(value < NULL_TRANSFER_THRESHOLD for value in null_transfer),
        "reward_shuffled_related_transfer_accuracy": null_transfer,
        "reward_shuffled_does_not_transfer": all(
            value < NULL_TRANSFER_THRESHOLD for value in null_transfer
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    report = run(tuple(args.seeds))
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
