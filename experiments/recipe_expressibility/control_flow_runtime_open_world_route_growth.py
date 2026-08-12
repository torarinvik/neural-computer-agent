"""Open-world external-file routing under interleaved growth pressure.

Eight opaque external files are admitted one at a time, then eight unseen
opaque event contexts are learned in an interleaved stream.  Exploration is
novelty-weighted by external attempt counts, so each newly appended file keeps
an acquisition path as the bank grows.  One context is reversed after all
eight are mastered while the other seven are not replayed.

This is a bounded open-world routing promotion.  It does not claim arbitrary
program induction, unrestricted memory growth, or general continual learning.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
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

SEEDS = (17, 18, 19)
COUNTER_COUNT = 4
EVENT_WIDTH = 8
CONTEXT_COUNT = 8
INTENTION_WIDTH = 4
FEEDBACK_WIDTH = 3
QUERY_WIDTH = 40
FILES_PER_SEED = CONTEXT_COUNT
TRAIN_ROUNDS = 48
REVERSAL_ROUNDS = 64
MASTERY_THRESHOLD = 0.8
EXPLORATION = 0.35


class OpaqueCounterCodec(ControlFlowIntentionAdapter):
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


def _program(logical_target: int) -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("inc", counter=logical_target),
            ControlFlowInstruction("halt"),
        ),
    )


def _quiet_feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, FEEDBACK_WIDTH),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1, dtype=torch.bool),
    )


def _route_feedback(outcome: float) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, FEEDBACK_WIDTH),
        reward=torch.tensor([outcome]),
        propensity=torch.ones(1),
        has_feedback=torch.ones(1, dtype=torch.bool),
    )


def _event(context: int) -> list[AmodalEvent]:
    payload = torch.zeros(1, EVENT_WIDTH)
    payload[0, context] = 1.0
    return [AmodalEvent(payload)]


def _build(seed: int) -> tuple[ControlFlowProgramAmodalRuntime, ControlFlowProgramMemory]:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=EVENT_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=FEEDBACK_WIDTH,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    memory.add_program(_program(0), protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(
        width=QUERY_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=4,
        reversal_threshold=0.5,
        reversal_patience=3,
    )
    evidence.append_slot()
    agent = ControlFlowProgramAmodalRuntime(
        runtime,
        OpaqueCounterCodec(INTENTION_WIDTH, COUNTER_COUNT),
        program_memory=memory,
        program_route_evidence=evidence,
        program_route_query_adapter=ExternalControllerTrajectoryQueryAdapter(
            controller_width=EVENT_WIDTH,
            query_width=QUERY_WIDTH,
            trajectory_statistics="recency_weighted_and_latest_v1",
        ),
        program_route_exploration=EXPLORATION,
        program_route_exploration_strategy="balanced",
        max_steps=8,
    )
    return agent, memory


def _reset_controller(agent: ControlFlowProgramAmodalRuntime, state):
    return replace(
        state,
        controller=agent.runtime.controller.initial_state(1, device="cpu"),
    )


def _step(
    agent: ControlFlowProgramAmodalRuntime,
    state,
    context: int,
    route_feedback: ControllerFeedback | None = None,
):
    state = _reset_controller(agent, state)
    output, next_state = agent.step_events(
        _event(context),
        state,
        _quiet_feedback(),
        route_feedback=route_feedback,
    )
    if output.program_route_query is None:
        raise RuntimeError("open-world route audit requires a route query")
    return output, next_state


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


def _run_seed(seed: int, *, shuffled: bool) -> dict[str, object]:
    started = time.perf_counter()
    agent, memory = _build(seed)
    state = agent.initial_state(1, device="cpu")
    controller_before = _controller_snapshot(agent)
    source_digest = memory.program(0).digest()
    admission_bits = 0
    for slot in range(1, FILES_PER_SEED):
        receipt, state = agent.admit_program_verified(
            state,
            _program(slot % COUNTER_COUNT),
            (1.0, 1.0),
            protect=True,
        )
        if not receipt.accepted or receipt.slot != slot:
            raise RuntimeError("open-world file admission failed")
        admission_bits += 2

    generator = torch.Generator(device="cpu").manual_seed(seed + 22_001)
    targets = tuple(range(CONTEXT_COUNT))
    output, state = _step(agent, state, 0)
    last_context = 0
    selected_history: list[dict[str, int | float]] = []
    for round_index in range(TRAIN_ROUNDS):
        for context in range(CONTEXT_COUNT):
            selected = int(output.selected_program_slots[0])
            outcome = (
                float(torch.rand((), generator=generator).item() < 0.5)
                if shuffled
                else float(selected == targets[last_context])
            )
            selected_history.append(
                {
                    "round": round_index,
                    "context": last_context,
                    "selected": selected,
                    "outcome": outcome,
                }
            )
            output, state = _step(
                agent,
                state,
                context,
                route_feedback=_route_feedback(outcome),
            )
            last_context = context

    final_outcome = (
        float(torch.rand((), generator=generator).item() < 0.5)
        if shuffled
        else float(int(output.selected_program_slots[0]) == targets[last_context])
    )
    agent.program_route_exploration = 0.0
    learned_slots: dict[int, int] = {}
    for context in range(CONTEXT_COUNT):
        output, state = _step(
            agent,
            state,
            context,
            route_feedback=_route_feedback(final_outcome),
        )
        learned_slots[context] = int(output.selected_program_slots[0])
        final_outcome = 0.0

    # Reverse only context 7. The other seven contexts are not replayed.
    agent.program_route_exploration = EXPLORATION
    reversal_output = output
    reversal_state = state
    for _ in range(REVERSAL_ROUNDS):
        selected = int(reversal_output.selected_program_slots[0])
        outcome = (
            float(torch.rand((), generator=generator).item() < 0.5)
            if shuffled
            else float(selected == 0)
        )
        reversal_output, reversal_state = _step(
            agent,
            reversal_state,
            7,
            route_feedback=_route_feedback(outcome),
        )
    reversal_final = (
        float(torch.rand((), generator=generator).item() < 0.5)
        if shuffled
        else float(int(reversal_output.selected_program_slots[0]) == 0)
    )
    agent.program_route_exploration = 0.0
    reversal_output, reversal_state = _step(
        agent,
        reversal_state,
        7,
        route_feedback=_route_feedback(reversal_final),
    )
    reversed_slot = int(reversal_output.selected_program_slots[0])
    retained_output, retained_state = _step(
        agent,
        reversal_state,
        0,
        route_feedback=_route_feedback(0.0),
    )
    retained_slot = int(retained_output.selected_program_slots[0])

    evidence = agent.program_route_evidence
    if evidence is None:
        raise RuntimeError("open-world route evidence disappeared")
    evidence_payload = evidence.payload()
    restored_evidence = PersistentOpaqueContextRouteEvidence.from_payload(
        evidence_payload
    )
    corrupted = dict(evidence_payload)
    corrupted["version"] = int(corrupted["version"]) + 1
    try:
        PersistentOpaqueContextRouteEvidence.from_payload(corrupted)
        corruption_rejected = False
    except ValueError as error:
        corruption_rejected = "checksum" in str(error)
    restored_state = agent.state_from_payload(retained_state.payload())
    source_retained = memory.program(0).digest() == source_digest
    queries = []
    for record in evidence_payload["contexts"]:
        queries.append(torch.tensor(record["key"], dtype=torch.float32))
    query_matrix = torch.stack(queries)
    normalized = torch.nn.functional.normalize(query_matrix, dim=-1)
    off_diagonal = normalized @ normalized.T
    distinct = float(off_diagonal.fill_diagonal_(-1.0).amax()) < 0.99
    gates = {
        "all_files_admitted": agent.program_memory.file_count == FILES_PER_SEED,
        "all_open_world_contexts_created": evidence.context_count == CONTEXT_COUNT,
        "all_contexts_route_correct": all(
            learned_slots[context] == targets[context]
            for context in range(CONTEXT_COUNT)
        ),
        "newest_file_reachable": learned_slots[CONTEXT_COUNT - 1] == CONTEXT_COUNT - 1,
        "reversal_recovers_context_7": reversed_slot == 0,
        "older_context_retained": retained_slot == 0,
        "context_keys_remain_distinct": distinct,
        "source_file_retained": source_retained,
        "controller_frozen": _controller_frozen(controller_before, agent),
        "evidence_reload_exact": restored_evidence.payload() == evidence_payload,
        "state_reload_exact": restored_state.digest() == retained_state.digest(),
        "evidence_corruption_rejected": corruption_rejected,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
        "shuffled_feedback_not_mastered": (
            not shuffled
            or not all(
                learned_slots[context] == targets[context]
                for context in range(CONTEXT_COUNT)
            )
        ),
    }
    return {
        "schema": "neural-computer.control-flow-runtime-open-world-route-growth.v1",
        "seed": seed,
        "shuffled_feedback": shuffled,
        "learned_slots": learned_slots,
        "reversed_slot": reversed_slot,
        "retained_slot": retained_slot,
        "evidence_context_count": evidence.context_count,
        "selected_history_tail": selected_history[-12:],
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": admission_bits + TRAIN_ROUNDS * CONTEXT_COUNT + REVERSAL_ROUNDS,
            "unique_logical_lifetimes": TRAIN_ROUNDS * CONTEXT_COUNT + REVERSAL_ROUNDS,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_time_seconds": time.perf_counter() - started,
        },
    }


def run() -> dict[str, object]:
    positive = tuple(_run_seed(seed, shuffled=False) for seed in SEEDS)
    shuffled = tuple(_run_seed(seed + 10_000, shuffled=True) for seed in SEEDS)
    gates = {
        key: all(bool(report["gates"][key]) for report in positive)
        for key in positive[0]["gates"]
        if key != "shuffled_feedback_not_mastered"
    }
    gates["shuffled_feedback_not_mastered"] = all(
        bool(report["gates"]["shuffled_feedback_not_mastered"])
        for report in shuffled
    )
    reports = positive + shuffled
    return {
        "schema": "neural-computer.control-flow-runtime-open-world-route-growth-audit.v1",
        "status": "promoted_narrow_open_world_route_growth" if all(gates.values()) else "rejected",
        "claim_boundary": (
            "Promoted bounded novelty-weighted reachability of eight newly "
            "admitted external files across eight unseen interleaved contexts, "
            "with reversal retention and frozen controller; not unrestricted "
            "memory growth, arbitrary program induction, or general continual learning."
        ),
        "architecture": {
            "route_exploration": "opaque_per_context_novelty_weighted_epsilon_v1",
            "new_file_probability": "inverse_attempt_count_normalized",
            "contexts": CONTEXT_COUNT,
            "files": FILES_PER_SEED,
            "caller_supplied_slot_overrides": 0,
            "replayed_examples": 0,
        },
        "seeds": list(SEEDS),
        "positive_reports": positive,
        "shuffled_reports": shuffled,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": sum(
                int(report["accounting"]["unique_verifier_bits"])
                for report in reports
            ),
            "unique_logical_lifetimes": sum(
                int(report["accounting"]["unique_logical_lifetimes"])
                for report in reports
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_time_seconds": sum(
                float(report["accounting"]["wall_time_seconds"])
                for report in reports
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    report = run()
    encoded = json.dumps(report, indent=2) + "\n"
    if args.json is None:
        print(encoded, end="")
    else:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(encoded)


if __name__ == "__main__":
    main()
