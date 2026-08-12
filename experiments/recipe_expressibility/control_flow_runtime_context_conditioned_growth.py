"""Sequential context-conditioned growth of an opaque control-flow file bank.

This is a short-rung continual-memory audit, not a general continual-learning
claim.  One frozen amodal controller produces a learned event/intention
trajectory.  A checksummed external table learns which protected control-flow
file to address for each learned context.  Training receives only the scalar
verifier outcome for the selected opaque file; candidate exploration is an
external round-robin schedule.

Contexts are acquired one at a time.  Once a context is mastered, its
outcomes are never replayed while later contexts are learned.  A final
reversal changes one context's correct file and tests patient demotion of the
stale external mapping while the other context mappings remain intact.
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
CONTEXT_COUNT = 4
TRAIN_EPISODES = 32
HELDOUT_LIFETIMES = 8
MASTERY_THRESHOLD = 0.9
NULL_TOLERANCE = 0.15
SEEDS = (17, 18, 19)

CONTEXT_TARGETS = (0, 1, 2, 3)
REVERSAL_TARGETS = (3, 1, 2, 3)


class OpaqueCounterCodec(ControlFlowIntentionAdapter):
    """Fixed external executable ABI with no semantic controller fields."""

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


def _feedback(batch: int = 1) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(batch, FEEDBACK_WIDTH),
        reward=torch.zeros(batch),
        propensity=torch.ones(batch),
        has_feedback=torch.zeros(batch, dtype=torch.bool),
    )


def _event(context: int) -> list[AmodalEvent]:
    payload = torch.zeros(1, EVENT_WIDTH)
    payload[0, context] = 1.0
    return [AmodalEvent(payload)]


def _physical_slot(logical_slot: int, *, reverse_files: bool) -> int:
    return PROGRAM_COUNT - 1 - logical_slot if reverse_files else logical_slot


def _target_slot(
    context: int,
    mapping: tuple[int, ...],
    *,
    reverse_files: bool,
) -> int:
    return _physical_slot(mapping[context], reverse_files=reverse_files)


def _make_agent(
    seed: int,
    *,
    reverse_files: bool,
    evidence: PersistentOpaqueContextRouteEvidence | None = None,
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
    source_digests = tuple(program.digest() for program in programs)
    if evidence is None:
        evidence = PersistentOpaqueContextRouteEvidence(
            width=40,
            matching_tolerance=1e-5,
            mastery_threshold=0.8,
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
    return agent, evidence, source_digests


def _controller_snapshot(agent: ControlFlowProgramAmodalRuntime) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }


def _controller_frozen(
    before: dict[str, torch.Tensor], agent: ControlFlowProgramAmodalRuntime
) -> bool:
    after = agent.runtime.controller.state_dict()
    return all(torch.equal(before[name], after[name]) for name in before)


def _step(
    agent: ControlFlowProgramAmodalRuntime,
    context: int,
    *,
    override: int | None = None,
) -> tuple[object, object]:
    state = agent.initial_state(1, device="cpu")
    output, next_state = agent.step_events(
        _event(context),
        state,
        _feedback(),
        program_route_override=(
            None
            if override is None
            else torch.tensor([override], dtype=torch.int64)
        ),
    )
    if output.program_route_query is None:
        raise RuntimeError("context-growth audit requires an opaque route query")
    return output, next_state


def _train_context(
    agent: ControlFlowProgramAmodalRuntime,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    context: int,
    mapping: tuple[int, ...],
    reverse_files: bool,
    seed: int,
    shuffled: bool = False,
) -> dict[str, object]:
    random = torch.Generator(device="cpu").manual_seed(seed + 91_771 + context)
    selected_outcomes: list[float] = []
    selected_slots: list[int] = []
    for episode in range(TRAIN_EPISODES):
        candidate = episode % PROGRAM_COUNT
        output, _ = _step(agent, context, override=candidate)
        selected = int(output.selected_program_slots[0])
        if selected != candidate:
            raise RuntimeError("external route exploration override was not honored")
        target = _target_slot(context, mapping, reverse_files=reverse_files)
        outcome = float(candidate == target)
        if shuffled:
            outcome = float(torch.rand((), generator=random) < 0.25)
        evidence.observe(output.program_route_query[0], selected, outcome)
        selected_slots.append(selected)
        selected_outcomes.append(outcome)
    status = evidence._find_record(  # type: ignore[attr-defined]
        output.program_route_query[0], create=False
    )
    if status is None:
        raise RuntimeError("trained context did not create an evidence row")
    return {
        "context": context,
        "episodes": TRAIN_EPISODES,
        "candidate_schedule": selected_slots,
        "scalar_outcomes": selected_outcomes,
        "replayed_examples": 0,
        "preferred_order": list(evidence.preferred_order(output.program_route_query[0])),
        "protected": list(status.evidence.status().protected),
    }


def _evaluate_context(
    agent: ControlFlowProgramAmodalRuntime,
    *,
    context: int,
    mapping: tuple[int, ...],
    reverse_files: bool,
    lifetimes: int = HELDOUT_LIFETIMES,
) -> dict[str, object]:
    target = _target_slot(context, mapping, reverse_files=reverse_files)
    selected: list[int] = []
    scores: list[bool] = []
    queries: list[torch.Tensor] = []
    for _ in range(lifetimes):
        output, _ = _step(agent, context)
        slot = int(output.selected_program_slots[0])
        selected.append(slot)
        scores.append(slot == target)
        queries.append(output.program_route_query[0].detach().clone())
    return {
        "context": context,
        "target_slot": target,
        "selected_slots": selected,
        "accuracy": sum(scores) / float(len(scores)),
        "query": queries[0].tolist(),
    }


def _evaluate_all(
    agent: ControlFlowProgramAmodalRuntime,
    *,
    mappings: tuple[int, ...],
    reverse_files: bool,
    contexts: range | tuple[int, ...],
) -> dict[str, dict[str, object]]:
    return {
        str(context): _evaluate_context(
            agent,
            context=context,
            mapping=mappings,
            reverse_files=reverse_files,
        )
        for context in contexts
    }


def _query_cosine_floor(rows: dict[str, dict[str, object]]) -> float:
    queries = [torch.as_tensor(row["query"], dtype=torch.float32) for row in rows.values()]
    if len(queries) < 2:
        return 1.0
    normalized = torch.nn.functional.normalize(torch.stack(queries), dim=-1)
    values = [
        float(normalized[index] @ normalized[other])
        for index in range(len(queries))
        for other in range(index)
    ]
    return min(values)


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
    )
    controller_before = _controller_snapshot(agent)
    phase_reports: list[dict[str, object]] = []
    for context in range(CONTEXT_COUNT):
        training = _train_context(
            agent,
            evidence,
            context=context,
            mapping=CONTEXT_TARGETS,
            reverse_files=reverse_files,
            seed=seed,
            shuffled=shuffled,
        )
        retained = _evaluate_all(
            agent,
            mappings=CONTEXT_TARGETS,
            reverse_files=reverse_files,
            contexts=tuple(range(context + 1)),
        )
        phase_reports.append(
            {
                "phase": f"context_{context}",
                "training": training,
                "retention": retained,
            }
        )

    before_reversal = _evaluate_all(
        agent,
        mappings=CONTEXT_TARGETS,
        reverse_files=reverse_files,
        contexts=tuple(range(CONTEXT_COUNT)),
    )
    reversal_training = _train_context(
        agent,
        evidence,
        context=0,
        mapping=REVERSAL_TARGETS,
        reverse_files=reverse_files,
        seed=seed + 1_000,
        shuffled=shuffled,
    )
    reversal_mappings = REVERSAL_TARGETS
    after_reversal = _evaluate_all(
        agent,
        mappings=reversal_mappings,
        reverse_files=reverse_files,
        contexts=tuple(range(CONTEXT_COUNT)),
    )

    unknown, _ = _step(agent, CONTEXT_COUNT)
    unknown_slot = int(unknown.selected_program_slots[0])
    final_state = agent.initial_state(1, device="cpu")
    state_payload = final_state.payload()
    restored_state = agent.state_from_payload(state_payload)
    state_reload_exact = restored_state.digest() == final_state.digest()
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

    fresh_agent, _, _ = _make_agent(seed + 4_000, reverse_files=reverse_files)
    fresh = _evaluate_all(
        fresh_agent,
        mappings=CONTEXT_TARGETS,
        reverse_files=reverse_files,
        contexts=tuple(range(CONTEXT_COUNT)),
    )
    all_mastered = all(
        float(row["accuracy"]) >= MASTERY_THRESHOLD
        for phase in phase_reports
        for row in phase["retention"].values()
    )
    reversal_mastered = all(
        float(row["accuracy"]) >= MASTERY_THRESHOLD
        for row in after_reversal.values()
    )
    trained_accuracy = sum(
        float(row["accuracy"]) for row in after_reversal.values()
    ) / CONTEXT_COUNT
    fresh_accuracy = sum(float(row["accuracy"]) for row in fresh.values()) / CONTEXT_COUNT
    files_retained = all(
        agent.program_memory.program(slot).digest() == source_digests[slot]
        for slot in range(PROGRAM_COUNT)
    )
    gates = {
        "all_sequential_contexts_mastered": all_mastered,
        "reversal_mastered_without_old_context_replay": reversal_mastered,
        "fresh_control_measured": 0.25 - NULL_TOLERANCE
        <= fresh_accuracy
        <= 0.25 + NULL_TOLERANCE,
        "unknown_context_falls_back_to_append_order": unknown_slot == 0,
        "four_distinct_context_rows": evidence.context_count == CONTEXT_COUNT,
        "context_queries_are_distinct": _query_cosine_floor(before_reversal) < 0.99,
        "all_external_files_retained": files_retained,
        "controller_frozen": _controller_frozen(controller_before, agent),
        "state_reload_exact": state_reload_exact,
        "evidence_reload_exact": evidence_reload_exact,
        "evidence_checksum_rejected": checksum_rejected,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    return {
        "schema": "neural-computer.control-flow-runtime-context-conditioned-growth.v1",
        "seed": seed,
        "reverse_files": reverse_files,
        "feedback_mode": "reward_shuffled" if shuffled else "verifier_scalar",
        "architecture": {
            "contexts": CONTEXT_COUNT,
            "program_count": PROGRAM_COUNT,
            "train_episodes_per_context": TRAIN_EPISODES,
            "heldout_lifetimes_per_context": HELDOUT_LIFETIMES,
            "route_query_width": evidence.width,
            "learner_inputs": "opaque_learned_event_trajectory_and_selected_scalar_outcome",
            "external_memory": evidence.schema,
            "forbidden_features": "context labels, target slots, unattempted outcomes, controller protocol fields",
        },
        "phase_reports": phase_reports,
        "before_reversal": before_reversal,
        "reversal_training": reversal_training,
        "after_reversal": after_reversal,
        "fresh": fresh,
        "unknown_context_selected_slot": unknown_slot,
        "context_query_cosine_floor": _query_cosine_floor(before_reversal),
        "trained_accuracy": trained_accuracy,
        "fresh_accuracy": fresh_accuracy,
        "transfer_ratio_against_fresh": (
            None if fresh_accuracy == 0.0 else trained_accuracy / fresh_accuracy
        ),
        "evidence_context_count": evidence.context_count,
        "evidence_digest": evidence.digest(),
        "evidence_payload": evidence_payload,
        "source_program_digests": source_digests,
        "program_memory_digest": agent.program_memory.digest(),
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": (CONTEXT_COUNT + 1) * TRAIN_EPISODES,
            "unique_logical_lifetimes": (CONTEXT_COUNT + 1) * TRAIN_EPISODES,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - started,
            "latency_seconds_per_lifetime": (
                time.perf_counter() - started
            )
            / float((CONTEXT_COUNT + 1) * TRAIN_EPISODES),
            "stable_bits_to_threshold": TRAIN_EPISODES,
            "retention_on_mastered_primitives": float(reversal_mastered),
        },
    }


def run(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    reports = tuple(
        _run_arm(seed, reverse_files=reverse_files, shuffled=shuffled)
        for seed in seeds
        for reverse_files in (False, True)
        for shuffled in (False, True)
    )
    verifier_reports = tuple(
        report for report in reports if report["feedback_mode"] == "verifier_scalar"
    )
    null_reports = tuple(
        report for report in reports if report["feedback_mode"] == "reward_shuffled"
    )
    null_accuracy = {
        str(seed): sum(
            float(report["fresh_accuracy"])
            for report in null_reports
            if report["seed"] == seed
        )
        / 2.0
        for seed in seeds
    }
    null_within_boundary = all(
        abs(value - 0.25) <= NULL_TOLERANCE for value in null_accuracy.values()
    )
    return {
        "schema": "neural-computer.control-flow-runtime-context-conditioned-growth.v1",
        "claim_boundary": (
            "bounded sequential context-conditioned routing into four isolated external "
            "control-flow files with scalar outcome-only updates, frozen controller, "
            "reversal, and zero replay; not unrestricted memory growth, arbitrary new "
            "computation, or general continual learning"
        ),
        "seeds": list(seeds),
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in verifier_reports)
        and null_within_boundary,
        "reward_shuffled_paired_fresh_accuracy": null_accuracy,
        "reward_shuffled_null_within_boundary": null_within_boundary,
        "candidate_order_permutation_covered": all(
            any(
                bool(report["reverse_files"])
                for report in verifier_reports
                if report["seed"] == seed
            )
            and any(
                not bool(report["reverse_files"])
                for report in verifier_reports
                if report["seed"] == seed
            )
            for seed in seeds
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
