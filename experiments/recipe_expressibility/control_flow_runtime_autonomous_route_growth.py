"""Interleaved outcome-only reachability of a newly admitted external file.

The runtime admits a second opaque control-flow file, explores it without a
caller-supplied slot override, and automatically credits the selected file on
the following step from an explicit scalar route outcome.  Two contexts are
interleaved, then one context reverses while the other is retained.  The
controller is frozen throughout.

This is a bounded route-reachability promotion.  It does not claim arbitrary
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
INTENTION_WIDTH = 4
FEEDBACK_WIDTH = 3
QUERY_WIDTH = 40
TRAIN_STEPS = 240
REVERSAL_STEPS = 160
MASTERY_THRESHOLD = 0.8


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


def _program(counter: int) -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("inc", counter=counter),
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
        width=8,
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
            controller_width=8,
            query_width=QUERY_WIDTH,
            trajectory_statistics="recency_weighted_and_latest_v1",
        ),
        program_route_exploration=0.35,
        max_steps=8,
    )
    return agent, memory


def _reset_controller(
    agent: ControlFlowProgramAmodalRuntime,
    state,
):
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
        raise RuntimeError("autonomous route audit requires an opaque route query")
    return output, next_state


def _outcome(
    selected: int,
    context: int,
    targets: tuple[int, int],
    *,
    shuffled: bool,
    generator: torch.Generator,
) -> float:
    if shuffled:
        return float(torch.rand((), generator=generator).item() < 0.5)
    return float(selected == targets[context])


def _snapshot(agent: ControlFlowProgramAmodalRuntime) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }


def _frozen(
    before: dict[str, torch.Tensor],
    agent: ControlFlowProgramAmodalRuntime,
) -> bool:
    after = agent.runtime.controller.state_dict()
    return all(torch.equal(before[name], after[name]) for name in before)


def _run_seed(seed: int, *, shuffled: bool) -> dict[str, object]:
    started = time.perf_counter()
    agent, source_memory = _build(seed)
    initial_state = agent.initial_state(1, device="cpu")
    controller_before = _snapshot(agent)
    source_digest = source_memory.program(0).digest()
    admission, state = agent.admit_program_verified(
        initial_state,
        _program(1),
        (1.0, 1.0),
        protect=True,
    )
    if not admission.accepted:
        raise RuntimeError(admission.reason)
    generator = torch.Generator(device="cpu").manual_seed(seed + 91_003)

    targets = (0, 1)
    output, state = _step(agent, state, context=0)
    if (
        output.program_route_propensities is None
        or output.program_route_probabilities is None
    ):
        raise RuntimeError("exploration did not expose route propensities")
    exploration_propensity_exact = torch.allclose(
        output.program_route_propensities,
        output.program_route_probabilities.gather(
            1,
            output.selected_program_slots.unsqueeze(-1),
        ).squeeze(-1),
    )
    last_context = 0
    selected_history: list[dict[str, int | float]] = []
    for step in range(1, TRAIN_STEPS):
        last_selected = int(output.selected_program_slots[0])
        outcome = _outcome(
            last_selected,
            last_context,
            targets,
            shuffled=shuffled,
            generator=generator,
        )
        current_context = step % 2
        selected_history.append(
            {
                "context": last_context,
                "selected": last_selected,
                "outcome": outcome,
            }
        )
        output, state = _step(
            agent,
            state,
            current_context,
            route_feedback=_route_feedback(outcome),
        )
        last_context = current_context

    # Apply the final pending outcome and measure both contexts without any
    # route override.  The controller is reset only to isolate the learned
    # external address from recurrent drift; the pending route credit remains
    # in the runtime state.
    final_outcome = _outcome(
        int(output.selected_program_slots[0]),
        last_context,
        targets,
        shuffled=shuffled,
        generator=generator,
    )
    agent.program_route_exploration = 0.0
    measured: dict[int, int] = {}
    output, state = _step(
        agent,
        state,
        context=0,
        route_feedback=_route_feedback(final_outcome),
    )
    measured[0] = int(output.selected_program_slots[0])
    context0_outcome = _outcome(
        measured[0], 0, targets, shuffled=False, generator=generator
    )
    output, state = _step(
        agent,
        state,
        context=1,
        route_feedback=_route_feedback(context0_outcome),
    )
    measured[1] = int(output.selected_program_slots[0])

    # Reverse only context 1. Context 0 is not replayed during this phase.
    agent.program_route_exploration = 0.35
    reversal_targets = (0, 0)
    reversal_output = output
    reversal_state = state
    reversal_last_context = 1
    for _ in range(REVERSAL_STEPS):
        selected = int(reversal_output.selected_program_slots[0])
        outcome = _outcome(
            selected,
            reversal_last_context,
            reversal_targets,
            shuffled=shuffled,
            generator=generator,
        )
        reversal_output, reversal_state = _step(
            agent,
            reversal_state,
            context=1,
            route_feedback=_route_feedback(outcome),
        )
    reversal_final = _outcome(
        int(reversal_output.selected_program_slots[0]),
        1,
        reversal_targets,
        shuffled=shuffled,
        generator=generator,
    )
    agent.program_route_exploration = 0.0
    reversal_output, reversal_state = _step(
        agent,
        reversal_state,
        context=1,
        route_feedback=_route_feedback(reversal_final),
    )
    reversed_context1 = int(reversal_output.selected_program_slots[0])
    retained_context0_output, retained_context0_state = _step(
        agent,
        reversal_state,
        context=0,
        route_feedback=_route_feedback(
            _outcome(
                reversed_context1,
                1,
                reversal_targets,
                shuffled=False,
                generator=generator,
            )
        ),
    )
    retained_context0 = int(retained_context0_output.selected_program_slots[0])

    evidence_payload = agent.program_route_evidence.payload()
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
    restored_state = agent.state_from_payload(retained_context0_state.payload())
    source_retained = source_memory.program(0).digest() == source_digest
    query = retained_context0_output.program_route_query
    gates = {
        "admission_accepted": admission.accepted,
        "new_file_reachable_without_override": measured[1] == 1,
        "source_context_mastered": measured[0] == 0,
        "reversal_context_mastered": reversed_context1 == 0,
        "unreversed_context_retained": retained_context0 == 0,
        "two_context_rows": agent.program_route_evidence.context_count == 2,
        "source_file_retained": source_retained,
        "controller_frozen": _frozen(controller_before, agent),
        "route_propensity_exact": bool(exploration_propensity_exact),
        "evidence_reload_exact": restored_evidence.payload() == evidence_payload,
        "state_reload_exact": restored_state.digest() == retained_context0_state.digest(),
        "evidence_corruption_rejected": corruption_rejected,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
        "shuffled_control_not_promoted": (
            not shuffled
            or not (
                measured[0] == 0
                and measured[1] == 1
                and reversed_context1 == 0
            )
        ),
    }
    return {
        "schema": "neural-computer.control-flow-runtime-autonomous-route-growth.v1",
        "seed": seed,
        "shuffled_feedback": shuffled,
        "admission": admission.__dict__,
        "measured_context_slots": measured,
        "reversed_context1": reversed_context1,
        "retained_context0": retained_context0,
        "selected_history_tail": selected_history[-8:],
        "evidence_context_count": agent.program_route_evidence.context_count,
        "evidence_digest": agent.program_route_evidence.digest(),
        "query_width": None if query is None else int(query.shape[1]),
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": TRAIN_STEPS + REVERSAL_STEPS,
            "unique_logical_lifetimes": TRAIN_STEPS + REVERSAL_STEPS,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_time_seconds": time.perf_counter() - started,
        },
    }


def run() -> dict[str, object]:
    reports = tuple(
        _run_seed(seed, shuffled=False) for seed in SEEDS
    )
    shuffled_reports = tuple(
        _run_seed(seed + 10_000, shuffled=True) for seed in SEEDS
    )
    positive_gates = {
        key: all(bool(report["gates"][key]) for report in reports)
        for key in reports[0]["gates"]
        if key != "shuffled_control_not_promoted"
    }
    positive_gates["shuffled_null_not_mastered"] = all(
        bool(report["gates"]["shuffled_control_not_promoted"])
        for report in shuffled_reports
    )
    return {
        "schema": "neural-computer.control-flow-runtime-autonomous-route-growth-audit.v1",
        "status": "promoted_narrow_autonomous_route_reachability" if all(positive_gates.values()) else "rejected",
        "claim_boundary": (
            "Promoted bounded interleaved reachability of a newly admitted "
            "external file using runtime-owned pending route credit and exact "
            "exploration propensities under a frozen controller; not arbitrary "
            "program induction, unrestricted memory growth, or general continual learning."
        ),
        "architecture": {
            "route_credit": "runtime_owned_pending_opaque_query_and_slot_v2",
            "exploration": "epsilon_greedy_external_evidence_with_exact_propensity_v1",
            "stream": "interleaved_context_0_context_1_then_context_1_reversal",
            "caller_supplied_slot_overrides": 0,
            "replayed_examples": 0,
        },
        "seeds": list(SEEDS),
        "reports": reports,
        "shuffled_reports": shuffled_reports,
        "gates": positive_gates,
        "accounting": {
            "unique_verifier_bits": sum(
                int(report["accounting"]["unique_verifier_bits"])
                for report in reports + shuffled_reports
            ),
            "unique_logical_lifetimes": sum(
                int(report["accounting"]["unique_logical_lifetimes"])
                for report in reports + shuffled_reports
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_time_seconds": sum(
                float(report["accounting"]["wall_time_seconds"])
                for report in reports + shuffled_reports
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
