"""Outcome-only acquisition, external composition, and canonical execution.

The structural frontier acquires a generic transfer loop.  A separate external
file is then composed with that acquired file by the typed control-flow ABI;
the materialized composition is admitted through scalar verifier evidence and
is routed by a frozen amodal controller.  The controller sees only learned
events, opaque feedback, and opaque intentions throughout.

This is a bounded reusable-computation rung.  It does not claim arbitrary
program induction, unrestricted memory growth, or general continual learning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from experiments.recipe_expressibility.control_flow_frontier_growth import (
        HELDOUT_AMOUNTS,
        MAX_STEPS,
        TRAIN_AMOUNTS,
        _fresh_root,
        _outcomes,
        _private_target,
        _search,
        _warm_root,
    )
    from experiments.recipe_expressibility.control_flow_runtime_acquired_program import (
        OpaqueAmountCodec,
    )
except ModuleNotFoundError:
    from control_flow_frontier_growth import (
        HELDOUT_AMOUNTS,
        MAX_STEPS,
        TRAIN_AMOUNTS,
        _fresh_root,
        _outcomes,
        _private_target,
        _search,
        _warm_root,
    )
    from control_flow_runtime_acquired_program import OpaqueAmountCodec

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControlFlowInstruction,
    ControlFlowProgram,
    ControlFlowProgramAmodalRuntime,
    ControlFlowProgramMemory,
    ControllerFeedback,
    ExternalControllerTrajectoryQueryAdapter,
    PersistentOpaqueContextRouteEvidence,
    compose_control_flow_programs,
)

PROGRAM_COUNT = 5
INTENTION_WIDTH = 2
COUNTER_COUNT = 2
EVENT_WIDTH = 4
FEEDBACK_WIDTH = 3
SOURCE_LOGICAL_SLOT = 0
ACQUIRED_COMPONENT_LOGICAL_SLOT = 3
SUFFIX_LOGICAL_SLOT = 4
COMPOSED_LOGICAL_SLOT = 2
TRAIN_ROUTE_EPISODES = 40
HELDOUT_LIFETIMES = 8
SEEDS = (17, 18, 19)
ROUTE_MASTERY_THRESHOLD = 0.8
VERIFIER_THRESHOLD = 1.0


def _suffix() -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("halt"),
        ),
    )


def _event(context: str) -> list[AmodalEvent]:
    payload = torch.zeros(1, EVENT_WIDTH)
    if context == "source":
        payload[0, 0] = 1.0
    elif context == "composed":
        payload[0, 1] = 1.0
    else:
        raise ValueError(f"unsupported runtime context: {context}")
    return [AmodalEvent(payload)]


def _feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, FEEDBACK_WIDTH),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1, dtype=torch.bool),
    )


def _physical_slot(logical_slot: int, *, reverse_files: bool) -> int:
    return PROGRAM_COUNT - 1 - logical_slot if reverse_files else logical_slot


def _acquire(seed: int, *, reverse_inputs: bool) -> dict[str, object]:
    amounts = tuple(reversed(TRAIN_AMOUNTS)) if reverse_inputs else TRAIN_AMOUNTS
    warm = _search(_warm_root(), _private_target(), amounts, seed=seed + 10_000)
    fresh = _search(_fresh_root(), _private_target(), amounts, seed=seed + 20_000)
    candidate = warm["candidate"]
    if not isinstance(candidate, ControlFlowProgram):
        raise TypeError("structural frontier did not acquire an external program")
    heldout = _outcomes(candidate, _private_target(), HELDOUT_AMOUNTS)
    return {
        "candidate": candidate,
        "warm": warm,
        "fresh": fresh,
        "amounts": amounts,
        "heldout_accuracy": sum(heldout) / len(heldout),
    }


def _build_logical_memory(
    candidate: ControlFlowProgram,
) -> tuple[ControlFlowProgramMemory, ControlFlowProgram]:
    composition_memory = ControlFlowProgramMemory(COUNTER_COUNT)
    composition_memory.add_program(_warm_root(), protect=True)
    composition_memory.add_program(_fresh_root(), protect=True)
    acquired_slot = composition_memory.add_program(candidate, protect=True)
    suffix_slot = composition_memory.add_program(_suffix(), protect=True)
    if (acquired_slot, suffix_slot) != (2, 3):
        raise RuntimeError("composition source slots are not stable")
    private_composed = compose_control_flow_programs((_private_target(), _suffix()))
    candidate_composed = composition_memory.compose((acquired_slot, suffix_slot))
    outcomes = tuple(
        float(
            candidate_composed.execute((amount, 0), max_steps=MAX_STEPS).status
            == "halted"
            and candidate_composed.execute((amount, 0), max_steps=MAX_STEPS).counters
            == private_composed.execute((amount, 0), max_steps=MAX_STEPS).counters
        )
        for amount in TRAIN_AMOUNTS
    )
    receipt = composition_memory.compose_verified(
        (acquired_slot, suffix_slot),
        outcomes,
        threshold=VERIFIER_THRESHOLD,
        min_observations=len(TRAIN_AMOUNTS),
        min_stable_observations=len(TRAIN_AMOUNTS),
        protect=True,
    )
    if not receipt.accepted or receipt.slot != 4:
        raise RuntimeError(f"composed program admission failed: {receipt.reason}")
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    memory.add_program(_warm_root(), protect=True)
    memory.add_program(_fresh_root(), protect=True)
    memory.add_program(composition_memory.program(4), protect=True)
    memory.add_program(candidate, protect=True)
    memory.add_program(_suffix(), protect=True)
    return memory, private_composed


def _make_runtime(
    seed: int,
    programs: tuple[ControlFlowProgram, ...],
) -> tuple[
    ControlFlowProgramAmodalRuntime,
    PersistentOpaqueContextRouteEvidence,
    tuple[str, ...],
]:
    if len(programs) != PROGRAM_COUNT:
        raise ValueError("composed runtime needs the configured file count")
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=FEEDBACK_WIDTH,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    for program in programs:
        memory.add_program(program, protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(
        width=20,
        matching_tolerance=1e-5,
        mastery_threshold=ROUTE_MASTERY_THRESHOLD,
        min_mastery_observations=8,
        reversal_threshold=0.5,
        reversal_patience=4,
    )
    for _ in range(PROGRAM_COUNT):
        evidence.append_slot()
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=4,
        query_width=20,
    )
    agent = ControlFlowProgramAmodalRuntime(
        runtime,
        OpaqueAmountCodec(INTENTION_WIDTH, COUNTER_COUNT),
        program_memory=memory,
        program_route_evidence=evidence,
        program_route_query_adapter=query_adapter,
        max_steps=MAX_STEPS,
    )
    return agent, evidence, tuple(program.digest() for program in programs)


def _step(
    agent: ControlFlowProgramAmodalRuntime,
    context: str,
    *,
    override: int | None = None,
) -> tuple[object, torch.Tensor]:
    output, _ = agent.step_events(
        _event(context),
        agent.initial_state(1, device="cpu"),
        _feedback(),
        program_route_override=(
            None
            if override is None
            else torch.tensor([override], dtype=torch.int64)
        ),
    )
    if output.program_route_query is None:
        raise RuntimeError("composed-program audit requires a route query")
    initial = agent.adapter.encode(
        output.controller.intention,
        torch.zeros(1, COUNTER_COUNT, dtype=torch.int64),
    )[0]
    return output, initial


def _outcome(
    output: object,
    initial: torch.Tensor,
    reference: ControlFlowProgram,
) -> float:
    if len(output.executions) != 1:
        raise RuntimeError("composed-program audit expected one execution")
    expected = reference.execute(
        tuple(int(value) for value in initial.tolist()),
        max_steps=MAX_STEPS,
    )
    actual = output.executions[0]
    return float(
        actual.status == "halted"
        and expected.status == "halted"
        and actual.counters == expected.counters
    )


def _train_route(
    agent: ControlFlowProgramAmodalRuntime,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    context: str,
    target_slot: int,
    reference: ControlFlowProgram,
    shuffled: bool,
    seed: int,
) -> int:
    random = torch.Generator(device="cpu").manual_seed(seed + 90_000)
    for episode in range(TRAIN_ROUTE_EPISODES):
        candidate = episode % PROGRAM_COUNT
        output, initial = _step(agent, context, override=candidate)
        outcome = _outcome(output, initial, reference)
        if shuffled:
            outcome = float(torch.rand((), generator=random) < 0.25)
        assert output.program_route_query is not None
        evidence.observe(output.program_route_query[0], candidate, outcome)
    return TRAIN_ROUTE_EPISODES


def _evaluate(
    agent: ControlFlowProgramAmodalRuntime,
    *,
    context: str,
    target_slot: int,
    reference: ControlFlowProgram,
) -> dict[str, object]:
    selected: list[int] = []
    outcomes: list[float] = []
    for _ in range(HELDOUT_LIFETIMES):
        output, initial = _step(agent, context)
        selected.append(int(output.selected_program_slots[0]))
        outcomes.append(_outcome(output, initial, reference))
    return {
        "context": context,
        "target_slot": target_slot,
        "selected_slots": selected,
        "selection_accuracy": sum(slot == target_slot for slot in selected)
        / float(len(selected)),
        "execution_accuracy": sum(outcomes) / float(len(outcomes)),
    }


def _run_arm(
    seed: int,
    *,
    reverse_files: bool,
    shuffled: bool,
) -> dict[str, object]:
    started = time.perf_counter()
    acquisition = _acquire(seed, reverse_inputs=reverse_files)
    candidate = acquisition["candidate"]
    assert isinstance(candidate, ControlFlowProgram)
    logical_memory, private_composed = _build_logical_memory(candidate)
    logical_programs = tuple(
        logical_memory.program(slot) for slot in range(logical_memory.file_count)
    )
    programs = tuple(reversed(logical_programs)) if reverse_files else logical_programs
    agent, evidence, source_digests = _make_runtime(seed + 50_000, programs)
    source_slot = _physical_slot(SOURCE_LOGICAL_SLOT, reverse_files=reverse_files)
    target_slot = _physical_slot(COMPOSED_LOGICAL_SLOT, reverse_files=reverse_files)
    controller_before = {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }
    source_updates = _train_route(
        agent,
        evidence,
        context="source",
        target_slot=source_slot,
        reference=_warm_root(),
        shuffled=shuffled,
        seed=seed,
    )
    composed_updates = _train_route(
        agent,
        evidence,
        context="composed",
        target_slot=target_slot,
        reference=private_composed,
        shuffled=shuffled,
        seed=seed + 1,
    )
    source_result = _evaluate(
        agent,
        context="source",
        target_slot=source_slot,
        reference=_warm_root(),
    )
    composed_result = _evaluate(
        agent,
        context="composed",
        target_slot=target_slot,
        reference=private_composed,
    )
    fresh_agent, _, _ = _make_runtime(seed + 60_000, programs)
    fresh_result = _evaluate(
        fresh_agent,
        context="composed",
        target_slot=target_slot,
        reference=private_composed,
    )
    controller_after = {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }
    payload = evidence.payload()
    restored = PersistentOpaqueContextRouteEvidence.from_payload(payload)
    corrupted = dict(payload)
    corrupted["version"] = int(corrupted["version"]) + 1
    try:
        PersistentOpaqueContextRouteEvidence.from_payload(corrupted)
        corruption_rejected = False
    except ValueError as error:
        corruption_rejected = "checksum" in str(error)
    files_retained = all(
        agent.program_memory.program(slot).digest() == source_digests[slot]
        for slot in range(PROGRAM_COUNT)
    )
    composed_program = logical_memory.program(COMPOSED_LOGICAL_SLOT)
    heldout_composed = _outcomes(
        composed_program,
        private_composed,
        HELDOUT_AMOUNTS,
    )
    gates = {
        "frontier_acquired_component": acquisition["warm"]["status"] == "expressible",
        "acquired_component_heldout_mastery": acquisition["heldout_accuracy"] >= 0.95,
        "composition_admitted": composed_program.digest() != candidate.digest(),
        "composed_program_heldout_mastery": min(heldout_composed) >= VERIFIER_THRESHOLD,
        "canonical_composed_execution_mastered": composed_result["execution_accuracy"] >= 0.95,
        "canonical_composed_route_mastered": composed_result["selection_accuracy"] >= 0.95,
        "source_route_retention": source_result["selection_accuracy"] >= 0.95,
        "source_execution_retention": source_result["execution_accuracy"] >= 0.95,
        "fresh_composed_control_measured": fresh_result["selection_accuracy"] < 0.95,
        "all_external_files_retained": files_retained,
        "controller_frozen": all(
            torch.equal(controller_before[name], controller_after[name])
            for name in controller_before
        ),
        "evidence_reload_exact": restored.payload() == payload,
        "corruption_rejected": corruption_rejected,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    return {
        "schema": "neural-computer.control-flow-runtime-composed-program.v1",
        "seed": seed,
        "reverse_inputs": reverse_files,
        "feedback_mode": "reward_shuffled" if shuffled else "verifier_scalar",
        "architecture": {
            "program_count": PROGRAM_COUNT,
            "learner_inputs": "opaque_programs_and_selected_runtime_scalar_outcomes",
            "canonical_boundary": "amodal_event_to_frozen_controller_to_external_composed_program_to_intention_bus",
            "forbidden_features": "target program names, correct unattempted actions, protocol fields",
        },
        "acquisition": {
            "warm_status": acquisition["warm"]["status"],
            "warm_evaluations": acquisition["warm"]["evaluations"],
            "fresh_status": acquisition["fresh"]["status"],
            "fresh_evaluations": acquisition["fresh"]["evaluations"],
            "acquired_component_heldout_accuracy": acquisition["heldout_accuracy"],
            "acquired_component_digest": candidate.digest(),
            "composed_program_digest": composed_program.digest(),
            "private_composed_digest": private_composed.digest(),
            "composed_heldout_accuracy": sum(heldout_composed) / len(heldout_composed),
        },
        "route_training": {
            "source_updates": source_updates,
            "composed_updates": composed_updates,
            "unique_selected_runtime_verifier_bits": source_updates + composed_updates,
            "replayed_examples": 0,
        },
        "source_result": source_result,
        "composed_result": composed_result,
        "fresh_composed_result": fresh_result,
        "source_program_digests": source_digests,
        "program_memory_digest": agent.program_memory.digest(),
        "evidence_digest": evidence.digest(),
        "evidence_payload": payload,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": int(acquisition["warm"]["evaluations"])
            * len(TRAIN_AMOUNTS)
            + source_updates
            + composed_updates,
            "unique_logical_lifetimes": int(acquisition["warm"]["evaluations"])
            + source_updates
            + composed_updates,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - started,
            "stable_bits_to_threshold": None,
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
    shuffled_reports = tuple(
        report for report in reports if report["feedback_mode"] == "reward_shuffled"
    )
    return {
        "schema": "neural-computer.control-flow-runtime-composed-program.v1",
        "claim_boundary": (
            "bounded outcome-only acquisition of one generic external component, "
            "verified external composition, and canonical frozen-controller "
            "execution; not arbitrary program induction, unrestricted memory "
            "growth, or general continual learning"
        ),
        "seeds": list(seeds),
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in verifier)
        and all(
            not bool(report["gates"]["canonical_composed_route_mastered"])
            for report in shuffled_reports
        ),
        "reward_shuffled_route_mastery": [
            float(report["composed_result"]["selection_accuracy"])
            for report in shuffled_reports
        ],
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
