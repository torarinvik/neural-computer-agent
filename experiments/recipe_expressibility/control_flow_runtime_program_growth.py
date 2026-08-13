"""Three-seed audit of synchronized external control-flow file growth.

This pressure test exercises the memory-side lifecycle rather than controller
learning: a scalar verifier admits one new opaque program file while the
runtime keeps its frozen controller, route state, context evidence, and
per-file counters synchronized.  It is a bounded systems promotion, not a
claim of autonomous program induction or general continual learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControlFlowInstruction,
    ControlFlowIntentionAdapter,
    ControlFlowProgram,
    ControlFlowProgramAmodalRuntime,
    ControlFlowProgramGrowthReceipt,
    ControlFlowProgramMemory,
    ControllerFeedback,
    ExternalControllerTrajectoryQueryAdapter,
    ExternalOutcomeProgramRouter,
    PersistentOpaqueContextRouteEvidence,
)

SEEDS = (17, 18, 19)
COUNTER_COUNT = 2
EVENT_WIDTH = 4
INTENTION_WIDTH = 2
FEEDBACK_WIDTH = 3


class OpaqueCounterCodec(ControlFlowIntentionAdapter):
    """Test-only opaque adapter; no production semantic mapping is implied."""

    def encode(
        self,
        intention,
        previous_counters: torch.Tensor,
    ) -> torch.Tensor:
        counters = previous_counters.clone()
        counters[:, 0] = (intention.payload[:, 0] > 0.0).to(torch.int64)
        return counters

    def decode(self, counters: torch.Tensor, template):
        from neural_computer import IntentEvent

        return IntentEvent(
            payload=counters.to(dtype=template.payload.dtype),
            timestamp=template.timestamp,
            confidence=template.confidence,
            target_key=template.target_key,
        )


class EchoDecoder(nn.Module):
    def forward(self, intention):
        return intention.payload


def _source_program() -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("inc", counter=0),
            ControlFlowInstruction("halt"),
        ),
    )


def _target_program() -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("inc", counter=1),
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


def _build(
    seed: int,
    *,
    route: str,
) -> tuple[ControlFlowProgramAmodalRuntime, ControlFlowProgramMemory]:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=4,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=FEEDBACK_WIDTH,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder("echo", EchoDecoder())
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    memory.add_program(_source_program(), protect=True)
    if route == "evidence":
        evidence = PersistentOpaqueContextRouteEvidence(width=20)
        evidence.append_slot()
        return (
            ControlFlowProgramAmodalRuntime(
                runtime,
                OpaqueCounterCodec(INTENTION_WIDTH, COUNTER_COUNT),
                program_memory=memory,
                program_route_evidence=evidence,
                program_route_query_adapter=ExternalControllerTrajectoryQueryAdapter(
                    controller_width=4,
                    query_width=20,
                ),
                max_steps=8,
            ),
            memory,
        )
    if route == "router":
        return (
            ControlFlowProgramAmodalRuntime(
                runtime,
                OpaqueCounterCodec(INTENTION_WIDTH, COUNTER_COUNT),
                program_memory=memory,
                program_router=ExternalOutcomeProgramRouter(
                    feature_width=INTENTION_WIDTH,
                    program_capacity=1,
                    initial_programs=1,
                ),
                max_steps=8,
            ),
            memory,
        )
    raise ValueError(f"unsupported route arm: {route!r}")


def _step(
    agent: ControlFlowProgramAmodalRuntime,
    state,
    *,
    slot: int | None,
) -> tuple[object, object]:
    return agent.step_events(
        [AmodalEvent(torch.ones(1, EVENT_WIDTH))],
        state,
        _feedback(),
        program_route_override=(
            None if slot is None else torch.tensor([slot], dtype=torch.int64)
        ),
    )


def _digest_controller(agent: ControlFlowProgramAmodalRuntime) -> str:
    digest = hashlib.sha256()
    for name, value in agent.runtime.controller.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _run_seed(seed: int) -> dict[str, object]:
    started = time.perf_counter()
    arms: dict[str, dict[str, object]] = {}
    for route in ("evidence", "router"):
        agent, source_memory = _build(seed + (0 if route == "evidence" else 1000), route=route)
        state = agent.initial_state(1, device="cpu")
        controller_before = _digest_controller(agent)
        source_digest = source_memory.program(0).digest()
        source_output, _ = _step(
            agent,
            state,
            slot=0 if route == "evidence" else None,
        )

        receipt, grown = agent.admit_program_verified(
            state,
            _target_program(),
            (1.0, 1.0),
            protect=True,
        )
        if not isinstance(receipt, ControlFlowProgramGrowthReceipt):
            raise TypeError("program growth did not return its receipt ABI")
        if route == "evidence":
            target_output, _ = _step(agent, grown, slot=1)
            target_execution = target_output.executions[0]
        else:
            target_execution = agent.program_memory.program(1).execute(
                (0, 0),
                max_steps=8,
            )
        restored_memory = ControlFlowProgramMemory.from_payload(
            agent.program_memory.payload()
        )
        controller_after = _digest_controller(agent)
        source_retained = source_memory.program(0).digest() == source_digest
        source_execution = source_output.executions[0]
        route_state = grown.program_router
        route_evidence = agent.program_route_evidence
        arms[route] = {
            "accepted": receipt.accepted,
            "slot": receipt.slot,
            "source_execution": source_execution.counters,
            "target_execution": target_execution.counters,
            "memory_file_count": agent.program_memory.file_count,
            "memory_reload_exact": restored_memory.digest() == agent.program_memory.digest(),
            "source_retained": source_retained,
            "controller_frozen": controller_before == controller_after,
            "state_program_slots": sorted(grown.program_counters),
            "router_capacity": None if agent.program_router is None else agent.program_router.program_capacity,
            "router_active_programs": None if route_state is None else route_state.active_programs,
            "evidence_slots": None if route_evidence is None else route_evidence.slot_count,
            "state_digest_before": receipt.state_digest_before,
            "state_digest_after": receipt.state_digest_after,
        }

    reject_agent, reject_memory = _build(seed + 2000, route="evidence")
    reject_state = reject_agent.initial_state(1, device="cpu")
    before = reject_state.digest()
    rejected, unchanged = reject_agent.admit_program_verified(
        reject_state,
        _target_program(),
        (0.0, 0.0),
        min_observations=2,
        min_stable_observations=2,
    )
    arms["rejected"] = {
        "accepted": rejected.accepted,
        "state_identity_preserved": unchanged is reject_state,
        "state_digest_preserved": before == rejected.state_digest_after,
        "memory_file_count": reject_memory.file_count,
        "evidence_slots": reject_agent.program_route_evidence.slot_count,
    }

    gates = {
        "evidence_growth_admitted": arms["evidence"]["accepted"],
        "evidence_target_executes_second_counter": arms["evidence"]["target_execution"][1] == 1,
        "evidence_source_retained": arms["evidence"]["source_retained"],
        "evidence_route_slot_synchronized": arms["evidence"]["evidence_slots"] == 2,
        "evidence_program_counters_synchronized": arms["evidence"]["state_program_slots"] == [0, 1],
        "router_growth_admitted": arms["router"]["accepted"],
        "router_capacity_expanded": arms["router"]["router_capacity"] == 2,
        "router_active_route_synchronized": arms["router"]["router_active_programs"] == 2,
        "router_program_counters_synchronized": arms["router"]["state_program_slots"] == [0, 1],
        "memory_reload_exact": arms["evidence"]["memory_reload_exact"] and arms["router"]["memory_reload_exact"],
        "controller_frozen": arms["evidence"]["controller_frozen"] and arms["router"]["controller_frozen"],
        "rejected_candidate_not_promoted": not arms["rejected"]["accepted"] and arms["rejected"]["memory_file_count"] == 1,
        "rejected_state_unchanged": arms["rejected"]["state_identity_preserved"] and arms["rejected"]["state_digest_preserved"],
    }
    return {
        "schema": "neural-computer.control-flow-runtime-program-growth.v1",
        "seed": seed,
        "arms": arms,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": 6,
            "unique_logical_lifetimes": 6,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_time_seconds": time.perf_counter() - started,
        },
    }


def run() -> dict[str, object]:
    reports = tuple(_run_seed(seed) for seed in SEEDS)
    gates = {
        key: all(bool(report["gates"][key]) for report in reports)
        for key in reports[0]["gates"]
    }
    return {
        "schema": "neural-computer.control-flow-runtime-program-growth-audit.v1",
        "status": "promoted_narrow_synchronized_external_growth" if all(gates.values()) else "rejected",
        "claim_boundary": (
            "Promoted one verifier-gated, copy-on-write external program-file "
            "growth transaction with synchronized route/evidence/counter state "
            "under a frozen controller; not arbitrary program induction, "
            "unrestricted memory growth, or general continual learning."
        ),
        "architecture": {
            "boundary": "frozen_amodal_controller_to_checksummed_external_program_memory",
            "growth": "copy_on_write_verifier_gated_synchronized_file_route_state_v1",
            "learner_visible_feedback": "deterministic_scalar_verifier_outcome",
            "replayed_examples": 0,
            "controller_updates": 0,
        },
        "seeds": list(SEEDS),
        "reports": reports,
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
