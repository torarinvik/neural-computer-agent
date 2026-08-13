"""End-to-end acquisition and canonical execution of a new external program.

The structural frontier first acquires a data-dependent loop from scalar
verifier outcomes.  The acquired file is then admitted beside a protected
source and two decoys.  A frozen amodal controller emits opaque intentions;
the canonical runtime routes those intentions through an external context
ledger, executes the selected file, and returns an opaque intention to the
output bus.

This closes the integration gap between outcome-only program search and the
production INPUT -> PROCESS -> OUTPUT boundary.  It remains bounded external
program acquisition, not unrestricted program induction or general continual
learning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

try:
    from experiments.recipe_expressibility.control_flow_frontier_growth import (
        FRONTIER_MINIMUM_QUALITY,
        HELDOUT_AMOUNTS,
        MAX_STEPS,
        TRAIN_AMOUNTS,
        _fresh_root,
        _outcomes,
        _private_target,
        _search,
        _warm_root,
    )
except ModuleNotFoundError:
    from control_flow_frontier_growth import (
        FRONTIER_MINIMUM_QUALITY,
        HELDOUT_AMOUNTS,
        MAX_STEPS,
        TRAIN_AMOUNTS,
        _fresh_root,
        _outcomes,
        _private_target,
        _search,
        _warm_root,
    )

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

PROGRAM_COUNT = 4
INTENTION_WIDTH = 2
COUNTER_COUNT = 2
EVENT_WIDTH = 4
FEEDBACK_WIDTH = 3
SOURCE_LOGICAL_SLOT = 0
TARGET_LOGICAL_SLOT = 2
TRAIN_ROUTE_EPISODES = 32
HELDOUT_LIFETIMES = 8
SEEDS = (17, 18, 19)
ROUTE_MASTERY_THRESHOLD = 0.8


class OpaqueAmountCodec(ControlFlowIntentionAdapter):
    """External opaque codec that supplies bounded counter input to files."""

    def encode(
        self,
        intention: IntentEvent,
        previous_counters: torch.Tensor,
    ) -> torch.Tensor:
        # Use the complete opaque intention so a single coordinate quantization
        # cannot collapse a route challenge to the programs' shared zero case.
        value = (intention.payload.abs().sum(dim=1) * 8.0).to(torch.int64)
        counters = torch.zeros_like(previous_counters)
        counters[:, 0] = value.clamp(0, 8)
        return counters

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


def _event(context: str) -> list[AmodalEvent]:
    payload = torch.zeros(1, EVENT_WIDTH)
    if context == "source":
        payload[0, 0] = 1.0
    elif context == "acquired":
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


def _make_runtime(
    seed: int,
    candidate: ControlFlowProgram,
    *,
    reverse_files: bool,
    generalization_tolerance: float = 0.0,
) -> tuple[
    ControlFlowProgramAmodalRuntime,
    PersistentOpaqueContextRouteEvidence,
    tuple[str, ...],
]:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=FEEDBACK_WIDTH,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    decoy_a = _fresh_root()
    decoy_b = ControlFlowProgram(
        COUNTER_COUNT,
        (
            # A second generic file with a distinct wrong behavior.
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )
    programs = (_warm_root(), decoy_a, candidate, decoy_b)
    if reverse_files:
        programs = tuple(reversed(programs))
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    for program in programs:
        memory.add_program(program, protect=True)
    evidence = PersistentOpaqueContextRouteEvidence(
        width=20,
        matching_tolerance=1e-5,
        generalization_tolerance=generalization_tolerance,
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
        raise RuntimeError("acquired-program audit requires a route query")
    initial = agent.adapter.encode(
        output.controller.intention,
        torch.zeros(1, COUNTER_COUNT, dtype=torch.int64),
    )[0]
    return output, initial


def _runtime_outcome(
    output: object,
    initial: torch.Tensor,
    reference: ControlFlowProgram,
) -> float:
    if len(output.executions) != 1:
        raise RuntimeError("single-file acquired-program audit expected one execution")
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
        outcome = _runtime_outcome(output, initial, reference)
        if shuffled:
            outcome = float(torch.rand((), generator=random) < 0.25)
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
        outcomes.append(_runtime_outcome(output, initial, reference))
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
    agent, evidence, source_digests = _make_runtime(
        seed + 50_000,
        candidate,
        reverse_files=reverse_files,
    )
    source_slot = _physical_slot(SOURCE_LOGICAL_SLOT, reverse_files=reverse_files)
    target_slot = _physical_slot(TARGET_LOGICAL_SLOT, reverse_files=reverse_files)
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
    target_updates = _train_route(
        agent,
        evidence,
        context="acquired",
        target_slot=target_slot,
        reference=_private_target(),
        shuffled=shuffled,
        seed=seed + 1,
    )
    source_result = _evaluate(
        agent,
        context="source",
        target_slot=source_slot,
        reference=_warm_root(),
    )
    acquired_result = _evaluate(
        agent,
        context="acquired",
        target_slot=target_slot,
        reference=_private_target(),
    )
    fresh_agent, _, _ = _make_runtime(
        seed + 60_000,
        candidate,
        reverse_files=reverse_files,
    )
    fresh_acquired = _evaluate(
        fresh_agent,
        context="acquired",
        target_slot=target_slot,
        reference=_private_target(),
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
    gates = {
        "frontier_acquired_program": acquisition["warm"]["status"] == "expressible",
        "acquired_program_heldout_mastery": acquisition["heldout_accuracy"] >= 0.95,
        "canonical_acquired_execution_mastered": (
            acquired_result["execution_accuracy"] >= 0.95
        ),
        "canonical_acquired_route_mastered": (
            acquired_result["selection_accuracy"] >= 0.95
        ),
        "source_route_retention": source_result["selection_accuracy"] >= 0.95,
        "source_execution_retention": source_result["execution_accuracy"] >= 0.95,
        "fresh_acquired_control_measured": (
            fresh_acquired["selection_accuracy"] < 0.95
        ),
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
        "schema": "neural-computer.control-flow-runtime-acquired-program.v1",
        "seed": seed,
        "reverse_inputs": reverse_files,
        "feedback_mode": "reward_shuffled" if shuffled else "verifier_scalar",
        "architecture": {
            "program_count": PROGRAM_COUNT,
            "frontier_minimum_quality": FRONTIER_MINIMUM_QUALITY,
            "learner_inputs": "opaque_programs_and_selected_runtime_scalar_outcomes",
            "canonical_boundary": "amodal_event_to_frozen_controller_to_external_program_to_intention_bus",
            "forbidden_features": "target program names, correct unattempted actions, protocol fields",
        },
        "acquisition": {
            "warm_status": acquisition["warm"]["status"],
            "warm_evaluations": acquisition["warm"]["evaluations"],
            "fresh_status": acquisition["fresh"]["status"],
            "fresh_evaluations": acquisition["fresh"]["evaluations"],
            "target_digest": _private_target().digest(),
            "acquired_digest": candidate.digest(),
            "heldout_accuracy": acquisition["heldout_accuracy"],
        },
        "route_training": {
            "source_updates": source_updates,
            "acquired_updates": target_updates,
            "unique_selected_runtime_verifier_bits": source_updates + target_updates,
            "replayed_examples": 0,
        },
        "source_result": source_result,
        "acquired_result": acquired_result,
        "fresh_acquired_result": fresh_acquired,
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
            + target_updates,
            "unique_logical_lifetimes": int(acquisition["warm"]["evaluations"])
            + source_updates
            + target_updates,
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
        "schema": "neural-computer.control-flow-runtime-acquired-program.v1",
        "claim_boundary": (
            "bounded outcome-only structural acquisition of one generic external "
            "control-flow file followed by canonical frozen-controller execution "
            "and route learning; not arbitrary program induction, unrestricted "
            "memory growth, or general continual learning"
        ),
        "seeds": list(seeds),
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in verifier)
        and all(
            not bool(report["gates"]["canonical_acquired_route_mastered"])
            for report in shuffled_reports
        ),
        "reward_shuffled_route_mastery": [
            float(report["acquired_result"]["selection_accuracy"])
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
