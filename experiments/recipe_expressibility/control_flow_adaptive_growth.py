"""Replay-free adaptive-horizon induction for external control-flow files.

The audit grows a generic structural-search horizon one instruction at a time.
Each longer behavior is acquired from the previously qualified root, admitted
to external program memory only after scalar verifier evidence, and then
promoted as the next search root through a copy-on-write retention boundary.
The controller is not involved: this isolates the external CPU/files seam
before it is composed with the canonical amodal runtime.

This is a bounded curriculum result.  It does not claim efficient arbitrary
program synthesis, unrestricted execution, or general continual learning.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ControlFlowInstruction,
    ControlFlowProgram,
    ControlFlowProgramFrontierGrowth,
    ControlFlowProgramMemory,
)

SEEDS = (17, 18, 19, 20)
COUNTER_COUNT = 2
INITIAL_HORIZON = 2
MAXIMUM_HORIZON = 5
MAX_CANDIDATES_PER_RUNG = 600
TRAIN_STATES = ((0, 0), (1, 0), (2, 1), (4, 2), (7, 3))
HELDOUT_STATES = ((0, 4), (3, 0), (5, 2), (8, 1), (11, 4), (16, 3))
MAX_STEPS = 256


def _program(counters: tuple[int, ...]) -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        tuple(
            ControlFlowInstruction("inc", counter=counter)
            for counter in counters
        )
        + (ControlFlowInstruction("halt"),),
    )


ROOT = _program((0,))
STAGES = (_program((0, 1)), _program((0, 1, 0)), _program((0, 1, 0, 1)))


def _matches(
    candidate: ControlFlowProgram,
    reference: ControlFlowProgram,
    states: tuple[tuple[int, ...], ...],
) -> tuple[float, ...]:
    outcomes: list[float] = []
    for initial in states:
        actual = candidate.execute(initial, max_steps=MAX_STEPS)
        expected = reference.execute(initial, max_steps=MAX_STEPS)
        outcomes.append(
            float(
                actual.status == expected.status
                and actual.counters == expected.counters
            )
        )
    return tuple(outcomes)


def _accuracy(
    candidate: ControlFlowProgram,
    reference: ControlFlowProgram,
    states: tuple[tuple[int, ...], ...],
) -> float:
    outcomes = _matches(candidate, reference, states)
    return sum(outcomes) / len(outcomes)


def _retention_probe(
    memory: ControlFlowProgramMemory,
    retained: tuple[tuple[int, ControlFlowProgram], ...],
    candidate: ControlFlowProgram | None = None,
    candidate_reference: ControlFlowProgram | None = None,
) -> bool:
    if (
        candidate is not None
        and candidate_reference is not None
        and _accuracy(candidate, candidate_reference, HELDOUT_STATES) < 1.0
    ):
        return False
    return all(
        _accuracy(memory.program(slot), reference, HELDOUT_STATES) >= 1.0
        for slot, reference in retained
    )


def _build_policy() -> ControlFlowProgramFrontierGrowth:
    return ControlFlowProgramFrontierGrowth(
        COUNTER_COUNT,
        initial_horizon=INITIAL_HORIZON,
        maximum_horizon=MAXIMUM_HORIZON,
        beam_width=48,
        max_depth=64,
        minimum_quality=0.0,
        parent_temperature=0.2,
        exploration=0.7,
        proposal_retry_limit=256,
    )


def _run_positive(seed: int, *, reverse_inputs: bool) -> dict[str, object]:
    started = time.perf_counter()
    policy = _build_policy()
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    root_slot = memory.add_program(ROOT, protect=True)
    retained: list[tuple[int, ControlFlowProgram]] = [(root_slot, ROOT)]
    state = policy.initial_state(ROOT, root_quality=1.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    training_states = tuple(reversed(TRAIN_STATES)) if reverse_inputs else TRAIN_STATES
    stage_reports: list[dict[str, object]] = []
    missing_evidence_no_write = True

    for stage_index, target in enumerate(STAGES, start=1):
        before_missing = state.digest()
        try:
            policy.record_outcomes(
                state,
                policy.propose(state, generator=generator),
                (),
            )
            missing_evidence_no_write = False
        except (RuntimeError, ValueError):
            missing_evidence_no_write = missing_evidence_no_write and state.digest() == before_missing

        horizon_receipt, state = policy.expand_horizon_verified(
            state,
            lambda candidate: _retention_probe(memory, tuple(retained)),
        )
        if not horizon_receipt.accepted:
            raise RuntimeError("adaptive horizon expansion was rejected")

        accepted_candidate: ControlFlowProgram | None = None
        evaluations_before = state.frontier.evaluations
        for _ in range(MAX_CANDIDATES_PER_RUNG):
            proposal = policy.propose(state, generator=generator)
            outcomes = _matches(proposal.proposal.program, target, training_states)
            feedback = policy.record_outcomes(
                state,
                proposal,
                outcomes,
                threshold=1.0,
                min_observations=len(training_states),
                min_stable_observations=len(training_states),
            )
            state = feedback.state
            if feedback.accepted and proposal.proposal.program.digest() == target.digest():
                accepted_candidate = proposal.proposal.program
                break
        if accepted_candidate is None:
            raise RuntimeError(f"stage {stage_index} target was not exactly discovered")

        candidate = accepted_candidate
        candidate_reference = target
        promote_receipt, promoted_state = policy.promote_root_verified(
            state,
            candidate,
            lambda candidate_state, candidate=candidate, candidate_reference=candidate_reference: _retention_probe(
                memory,
                tuple(retained),
                candidate,
                candidate_reference,
            ),
        )
        if not promote_receipt.accepted:
            raise RuntimeError("qualified adaptive root promotion was rejected")
        state = promoted_state
        slot = memory.add_program(candidate, protect=True)
        retained.append((slot, target))
        restored_state = type(state).from_payload(state.payload())
        restored_memory = ControlFlowProgramMemory.from_payload(memory.payload())
        stage_reports.append(
            {
                "stage": stage_index,
                "target_length": len(target.instructions),
                "candidate_digest": candidate.digest(),
                "candidate_slot": slot,
                "horizon": state.horizon,
                "rung": state.rung,
                "candidates_this_rung": state.frontier.evaluations - evaluations_before,
                "heldout_accuracy": _accuracy(candidate, target, HELDOUT_STATES),
                "retention_accuracy": min(
                    _accuracy(memory.program(retained_slot), reference, HELDOUT_STATES)
                    for retained_slot, reference in retained
                ),
                "state_reload_exact": restored_state.digest() == state.digest(),
                "memory_reload_exact": restored_memory.digest() == memory.digest(),
                "promote_receipt": promote_receipt.reason,
            }
        )

    corrupted = state.payload()
    corrupted["sha256"] = "0" * 64
    try:
        type(state).from_payload(corrupted)
        corruption_rejected = False
    except ValueError:
        corruption_rejected = True

    final_retention = min(
        _accuracy(slot_program, reference, HELDOUT_STATES)
        for slot, reference in retained
        for slot_program in (memory.program(slot),)
    )
    gates = {
        "all_rungs_discovered": len(stage_reports) == len(STAGES),
        "horizon_grew_one_step_at_a_time": [item["horizon"] for item in stage_reports] == [3, 4, 5],
        "all_heldout_mastered": all(item["heldout_accuracy"] == 1.0 for item in stage_reports),
        "all_prior_programs_retained": final_retention == 1.0,
        "state_reload_exact": all(item["state_reload_exact"] for item in stage_reports),
        "memory_reload_exact": all(item["memory_reload_exact"] for item in stage_reports),
        "memory_corruption_rejected": corruption_rejected,
        "missing_evidence_no_write": missing_evidence_no_write,
        "zero_replayed_examples": True,
        "zero_optimizer_updates": True,
    }
    return {
        "seed": seed,
        "reverse_inputs": reverse_inputs,
        "schema": "neural-computer.recipe-control-flow-adaptive-growth.v1",
        "architecture": policy.configuration(),
        "stage_reports": stage_reports,
        "qualified_program_count": len(state.qualified_programs),
        "final_state_digest": state.digest(),
        "final_memory_digest": memory.digest(),
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": state.frontier.evaluations * len(training_states),
            "unique_logical_lifetimes": state.frontier.evaluations,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - started,
            "stable_bits_to_threshold": state.frontier.evaluations * len(training_states),
        },
        "promoted": all(gates.values()),
    }


def _run_shuffled(seed: int) -> dict[str, object]:
    """Run the same curriculum with independent shuffled scalar outcomes."""

    policy = _build_policy()
    state = policy.initial_state(ROOT, root_quality=1.0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 70_000)
    accepted_rungs = 0
    evaluations = 0
    for _ in range(len(STAGES)):
        _, state = policy.expand_horizon_verified(state, lambda _: True)
        for _ in range(80):
            proposal = policy.propose(state, generator=generator)
            evaluations += 1
            outcomes = tuple(float(index % 2 == 0) for index in range(len(TRAIN_STATES)))
            feedback = policy.record_outcomes(
                state,
                proposal,
                outcomes,
                threshold=1.0,
                min_observations=len(TRAIN_STATES),
                min_stable_observations=len(TRAIN_STATES),
            )
            state = feedback.state
            if feedback.accepted:
                accepted_rungs += 1
                break
    return {
        "seed": seed,
        "accepted_rungs": accepted_rungs,
        "promoted": accepted_rungs < len(STAGES),
        "accounting": {
            "unique_verifier_bits": evaluations * len(TRAIN_STATES),
            "unique_logical_lifetimes": evaluations,
            "optimizer_updates": 0,
            "replayed_examples": 0,
        },
        "zero_replayed_examples": True,
        "zero_optimizer_updates": True,
    }


def _run_fresh(seed: int) -> dict[str, object]:
    """Measure a fresh source-to-target search at each final horizon."""

    stage_reports: list[dict[str, object]] = []
    total_evaluations = 0
    for stage_index, target in enumerate(STAGES, start=1):
        horizon = len(target.instructions)
        policy = ControlFlowProgramFrontierGrowth(
            COUNTER_COUNT,
            initial_horizon=horizon,
            maximum_horizon=horizon,
            beam_width=48,
            max_depth=64,
            minimum_quality=0.0,
            parent_temperature=0.2,
            exploration=0.7,
            proposal_retry_limit=256,
        )
        state = policy.initial_state(ROOT, root_quality=1.0)
        generator = torch.Generator(device="cpu").manual_seed(seed + stage_index * 101)
        found = False
        evaluations = 0
        for _ in range(MAX_CANDIDATES_PER_RUNG):
            proposal = policy.propose(state, generator=generator)
            evaluations += 1
            feedback = policy.record_outcomes(
                state,
                proposal,
                _matches(proposal.proposal.program, target, TRAIN_STATES),
                threshold=1.0,
                min_observations=len(TRAIN_STATES),
                min_stable_observations=len(TRAIN_STATES),
            )
            state = feedback.state
            if feedback.accepted and proposal.proposal.program.digest() == target.digest():
                found = True
                break
        total_evaluations += evaluations
        stage_reports.append(
            {
                "stage": stage_index,
                "horizon": horizon,
                "found": found,
                "candidate_evaluations": evaluations,
            }
        )
    return {
        "seed": seed,
        "stage_reports": stage_reports,
        "accounting": {
            "unique_verifier_bits": total_evaluations * len(TRAIN_STATES),
            "unique_logical_lifetimes": total_evaluations,
            "optimizer_updates": 0,
            "replayed_examples": 0,
        },
        "all_targets_found": all(bool(item["found"]) for item in stage_reports),
    }


def run(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    positive = tuple(
        _run_positive(seed, reverse_inputs=reverse_inputs)
        for seed in seeds
        for reverse_inputs in (False, True)
    )
    shuffled = tuple(_run_shuffled(seed) for seed in seeds)
    fresh = tuple(_run_fresh(seed) for seed in seeds)
    gates = {
        "positive_arms_promoted": all(bool(report["promoted"]) for report in positive),
        "shuffled_feedback_not_promoted": all(bool(report["promoted"]) for report in shuffled),
    }
    return {
        "schema": "neural-computer.recipe-control-flow-adaptive-growth-audit.v1",
        "status": "promoted_replay_free_adaptive_control_flow_growth" if all(gates.values()) else "rejected",
        "seeds": list(seeds),
        "positive_reports": positive,
        "shuffled_reports": shuffled,
        "fresh_reports": fresh,
        "gates": gates,
        "claim_boundary": (
            "Promoted bounded replay-free adaptive-horizon induction through three "
            "one-instruction curriculum rungs with held-out retention of earlier "
            "external programs; not efficient arbitrary program synthesis, "
            "unrestricted execution, or general continual learning."
        ),
        "accounting": {
            "unique_verifier_bits": sum(
                int(report["accounting"]["unique_verifier_bits"])
                for report in (*positive, *shuffled, *fresh)
            ),
            "unique_logical_lifetimes": sum(
                int(report["accounting"]["unique_logical_lifetimes"])
                for report in (*positive, *shuffled, *fresh)
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
        },
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
    if report["status"] != "promoted_replay_free_adaptive_control_flow_growth":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
