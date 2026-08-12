"""Audit factorized, replay-free credit for structural control-flow growth.

The frontier previously learned only which edit operator was useful.  This
audit adds an external proposal memory that also credits an opaque
instruction identity and a position relative to the program's non-terminal
boundary.  The same insertion can therefore be reused when the parent loop
gets longer.  The controller is absent/frozen; all learning is external state
updated from deterministic scalar verifier outcomes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ControlFlowFrontierProposalMemory,
    ControlFlowInstruction,
    ControlFlowProgram,
    ControlFlowProgramFrontierGrowth,
    ControlFlowProgramMemory,
)

COUNTER_COUNT = 2
INITIAL_HORIZON = 4
MAXIMUM_HORIZON = 7
MAX_CANDIDATES = 1_200
TRAIN_STATES = ((0, 0), (1, 0), (2, 1), (4, 2), (7, 3))
HELDOUT_STATES = ((0, 4), (3, 0), (5, 2), (8, 1), (11, 4))
MAX_STEPS = 256


def _program(body: tuple[ControlFlowInstruction, ...]) -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        (*body, ControlFlowInstruction("halt")),
    )


ROOT = _program(
    (
        ControlFlowInstruction("jump_if_zero", counter=0, target=3),
        ControlFlowInstruction("dec", counter=0),
        ControlFlowInstruction("jump", target=0),
    )
)
STAGES = (
    _program(
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=4),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("jump", target=0),
        )
    ),
    _program(
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=5),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("jump", target=0),
        )
    ),
    _program(
        (
            ControlFlowInstruction("jump_if_zero", counter=0, target=6),
            ControlFlowInstruction("dec", counter=0),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("inc", counter=1),
            ControlFlowInstruction("jump", target=0),
        )
    ),
)

SEEDS = (17, 18, 19, 20)
CONTEXT = "opaque-noncommuting-loop-growth"


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
            float(actual.status == expected.status and actual.counters == expected.counters)
        )
    return tuple(outcomes)


def _accuracy(
    candidate: ControlFlowProgram,
    reference: ControlFlowProgram,
    states: tuple[tuple[int, ...], ...],
) -> float:
    values = _matches(candidate, reference, states)
    return sum(values) / len(values)


def _proposal_memory() -> ControlFlowFrontierProposalMemory:
    return ControlFlowFrontierProposalMemory(
        exploration_floor=0.05,
        shared_prior_weight=0.25,
        exploration_bonus=0.25,
        instruction_weight=1.0,
        position_weight=0.75,
        operator_weight=0.25,
        temperature=0.05,
    )


def _growth(memory: ControlFlowFrontierProposalMemory) -> ControlFlowProgramFrontierGrowth:
    return ControlFlowProgramFrontierGrowth(
        COUNTER_COUNT,
        initial_horizon=INITIAL_HORIZON,
        maximum_horizon=MAXIMUM_HORIZON,
        beam_width=64,
        max_depth=96,
        minimum_quality=0.0,
        parent_temperature=0.2,
        exploration=0.7,
        proposal_retry_limit=256,
        proposal_policy=memory,
    )


def _retains(
    memory: ControlFlowProgramMemory,
    retained: tuple[tuple[int, ControlFlowProgram], ...],
    *,
    candidate: ControlFlowProgram | None = None,
    reference: ControlFlowProgram | None = None,
) -> bool:
    if (
        candidate is not None
        and reference is not None
        and _accuracy(candidate, reference, HELDOUT_STATES) < 1.0
    ):
        return False
    return all(
        _accuracy(memory.program(slot), expected, HELDOUT_STATES) == 1.0
        for slot, expected in retained
    )


def _run_warm(seed: int, *, reverse_inputs: bool) -> dict[str, object]:
    started = time.perf_counter()
    proposal_memory = _proposal_memory()
    growth = _growth(proposal_memory)
    executable_memory = ControlFlowProgramMemory(COUNTER_COUNT)
    root_slot = executable_memory.add_program(ROOT, protect=True)
    retained: list[tuple[int, ControlFlowProgram]] = [(root_slot, ROOT)]
    state = growth.initial_state(ROOT, root_quality=1.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    training_states = tuple(reversed(TRAIN_STATES)) if reverse_inputs else TRAIN_STATES
    missing_evidence_no_write = True
    verifier_seconds = 0.0
    reports: list[dict[str, object]] = []

    for stage_index, target in enumerate(STAGES, start=1):
        before_state = state.digest()
        before_policy = proposal_memory.digest()
        try:
            probe = growth.propose(state, generator=generator, context=CONTEXT)
            growth.record_outcomes(state, probe, ())
            missing_evidence_no_write = False
        except (RuntimeError, ValueError):
            missing_evidence_no_write = (
                missing_evidence_no_write
                and state.digest() == before_state
                and proposal_memory.digest() == before_policy
            )

        receipt, state = growth.expand_horizon_verified(
            state,
            lambda candidate_state: _retains(executable_memory, tuple(retained)),
        )
        if not receipt.accepted:
            raise RuntimeError("factorized loop-growth horizon expansion was rejected")

        evaluations_before = state.frontier.evaluations
        candidate: ControlFlowProgram | None = None
        stable_bits_to_threshold: int | None = None
        for _ in range(MAX_CANDIDATES):
            proposal = growth.propose(state, generator=generator, context=CONTEXT)
            verifier_started = time.perf_counter()
            outcomes = _matches(proposal.proposal.program, target, training_states)
            verifier_seconds += time.perf_counter() - verifier_started
            feedback = growth.record_outcomes(
                state,
                proposal,
                outcomes,
                threshold=1.0,
                min_observations=len(training_states),
                min_stable_observations=len(training_states),
            )
            state = feedback.state
            if feedback.accepted and proposal.proposal.program.digest() == target.digest():
                candidate = proposal.proposal.program
                stable_bits_to_threshold = feedback.stable_bits_to_threshold
                break
        if candidate is None:
            raise RuntimeError(f"factorized loop target at stage {stage_index} was not found")

        promote_receipt, state = growth.promote_root_verified(
            state,
            candidate,
            lambda candidate_state, candidate=candidate, target=target: _retains(
                executable_memory,
                tuple(retained),
                candidate=candidate,
                reference=target,
            ),
        )
        if not promote_receipt.accepted:
            raise RuntimeError("factorized loop-growth root promotion was rejected")
        slot = executable_memory.add_program(candidate, protect=True)
        retained.append((slot, target))
        restored_state = type(state).from_payload(state.payload())
        restored_memory = ControlFlowProgramMemory.from_payload(executable_memory.payload())
        restored_policy = ControlFlowFrontierProposalMemory.from_payload(
            proposal_memory.payload()
        )
        reports.append(
            {
                "stage": stage_index,
                "target_length": len(target.instructions),
                "candidate_digest": candidate.digest(),
                "candidate_evaluations": state.frontier.evaluations - evaluations_before,
                "stable_bits_to_threshold": stable_bits_to_threshold,
                "heldout_accuracy": _accuracy(candidate, target, HELDOUT_STATES),
                "retention_accuracy": min(
                    _accuracy(executable_memory.program(retained_slot), expected, HELDOUT_STATES)
                    for retained_slot, expected in retained
                ),
                "state_reload_exact": restored_state.digest() == state.digest(),
                "memory_reload_exact": restored_memory.digest() == executable_memory.digest(),
                "proposal_memory_reload_exact": restored_policy.digest() == proposal_memory.digest(),
            }
        )

    corrupted = proposal_memory.payload()
    corrupted["sha256"] = "0" * 64
    try:
        ControlFlowFrontierProposalMemory.from_payload(corrupted)
        policy_corruption_rejected = False
    except ValueError:
        policy_corruption_rejected = True

    gates = {
        "all_loop_rungs_discovered": len(reports) == len(STAGES),
        "all_heldout_mastered": all(item["heldout_accuracy"] == 1.0 for item in reports),
        "all_prior_loops_retained": all(
            item["retention_accuracy"] == 1.0 for item in reports
        ),
        "horizon_grew_one_step_at_a_time": [item["target_length"] for item in reports]
        == [5, 6, 7],
        "state_reload_exact": all(item["state_reload_exact"] for item in reports),
        "memory_reload_exact": all(item["memory_reload_exact"] for item in reports),
        "proposal_memory_reload_exact": all(
            item["proposal_memory_reload_exact"] for item in reports
        ),
        "proposal_memory_corruption_rejected": policy_corruption_rejected,
        "missing_evidence_no_write": missing_evidence_no_write,
        "zero_replayed_examples": True,
        "zero_optimizer_updates": True,
    }
    return {
        "seed": seed,
        "reverse_inputs": reverse_inputs,
        "stage_reports": reports,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": state.frontier.evaluations * len(training_states),
            "unique_logical_lifetimes": state.frontier.evaluations,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - started,
            "verifier_seconds": verifier_seconds,
            "mean_verifier_latency_seconds": (
                verifier_seconds / state.frontier.evaluations
                if state.frontier.evaluations
                else 0.0
            ),
        },
        "promoted": all(gates.values()),
    }


def _run_fresh(seed: int) -> dict[str, object]:
    """Matched fresh-policy control with the mastered previous root retained."""

    started = time.perf_counter()
    parent = ROOT
    reports: list[dict[str, object]] = []
    verifier_seconds = 0.0
    for stage_index, target in enumerate(STAGES, start=1):
        proposal_memory = _proposal_memory()
        growth = ControlFlowProgramFrontierGrowth(
            COUNTER_COUNT,
            initial_horizon=len(parent.instructions),
            maximum_horizon=len(target.instructions),
            beam_width=64,
            max_depth=96,
            minimum_quality=0.0,
            parent_temperature=0.2,
            exploration=0.7,
            proposal_retry_limit=256,
            proposal_policy=proposal_memory,
        )
        state = growth.initial_state(parent, root_quality=1.0)
        receipt, state = growth.expand_horizon_verified(state, lambda _: True)
        if not receipt.accepted:
            raise RuntimeError("fresh control horizon expansion was rejected")
        generator = torch.Generator(device="cpu").manual_seed(seed + stage_index * 101)
        found = False
        stable_bits_to_threshold: int | None = None
        evaluations = 0
        for _ in range(MAX_CANDIDATES):
            proposal = growth.propose(state, generator=generator, context=CONTEXT)
            evaluations += 1
            verifier_started = time.perf_counter()
            outcomes = _matches(proposal.proposal.program, target, TRAIN_STATES)
            verifier_seconds += time.perf_counter() - verifier_started
            feedback = growth.record_outcomes(
                state,
                proposal,
                outcomes,
                threshold=1.0,
                min_observations=len(TRAIN_STATES),
                min_stable_observations=len(TRAIN_STATES),
            )
            state = feedback.state
            if feedback.accepted and proposal.proposal.program.digest() == target.digest():
                found = True
                stable_bits_to_threshold = feedback.stable_bits_to_threshold
                break
        reports.append(
            {
                "stage": stage_index,
                "found": found,
                "candidate_evaluations": evaluations,
                "stable_bits_to_threshold": stable_bits_to_threshold,
            }
        )
        if not found:
            break
        parent = target
    return {
        "seed": seed,
        "stage_reports": reports,
        "all_targets_found": len(reports) == len(STAGES)
        and all(item["found"] for item in reports),
        "accounting": {
            "unique_verifier_bits": sum(
                int(item["candidate_evaluations"]) * len(TRAIN_STATES)
                for item in reports
            ),
            "unique_logical_lifetimes": sum(
                int(item["candidate_evaluations"])
                for item in reports
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "verifier_seconds": verifier_seconds,
            "mean_verifier_latency_seconds": (
                verifier_seconds / sum(
                    int(item["candidate_evaluations"]) for item in reports
                )
                if reports
                else 0.0
            ),
            "wall_seconds": time.perf_counter() - started,
        },
    }


def _run_shuffled(seed: int) -> dict[str, object]:
    started = time.perf_counter()
    proposal_memory = _proposal_memory()
    growth = _growth(proposal_memory)
    state = growth.initial_state(ROOT, root_quality=1.0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 70_000)
    accepted_rungs = 0
    evaluations = 0
    for _ in STAGES:
        receipt, state = growth.expand_horizon_verified(state, lambda _: True)
        if not receipt.accepted:
            raise RuntimeError("shuffled control horizon expansion was rejected")
        for _ in range(160):
            proposal = growth.propose(state, generator=generator, context=CONTEXT)
            evaluations += 1
            feedback = growth.record_outcomes(
                state,
                proposal,
                tuple(float(index % 2 == 0) for index in range(len(TRAIN_STATES))),
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
        "not_promoted": accepted_rungs < len(STAGES),
        "accounting": {
            "unique_verifier_bits": evaluations * len(TRAIN_STATES),
            "unique_logical_lifetimes": evaluations,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "verifier_seconds": 0.0,
            "mean_verifier_latency_seconds": 0.0,
            "wall_seconds": time.perf_counter() - started,
        },
    }


def run(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    warm = tuple(
        _run_warm(seed, reverse_inputs=reverse_inputs)
        for seed in seeds
        for reverse_inputs in (False, True)
    )
    fresh = tuple(_run_fresh(seed) for seed in seeds)
    shuffled = tuple(_run_shuffled(seed) for seed in seeds)

    later_transfer_ratios: list[float] = []
    for report in warm:
        fresh_report = next(item for item in fresh if item["seed"] == report["seed"])
        fresh_by_stage = {
            item["stage"]: item for item in fresh_report["stage_reports"]
        }
        for stage_report in report["stage_reports"][1:]:
            fresh_stage = fresh_by_stage.get(stage_report["stage"])
            if fresh_stage is not None and fresh_stage["found"]:
                later_transfer_ratios.append(
                    int(stage_report["candidate_evaluations"])
                    / int(fresh_stage["candidate_evaluations"])
                )

    gates = {
        "warm_arms_promoted": all(item["promoted"] for item in warm),
        "warm_later_transfer_has_margin": bool(later_transfer_ratios)
        and sum(later_transfer_ratios) / len(later_transfer_ratios) <= 0.75,
        "fresh_control_is_matched": all(
            len(item["stage_reports"]) == len(STAGES) for item in fresh
        ),
        "shuffled_feedback_not_promoted": all(item["not_promoted"] for item in shuffled),
    }
    arms = (*warm, *fresh, *shuffled)
    total_lifetimes = sum(
        int(arm["accounting"]["unique_logical_lifetimes"]) for arm in arms
    )
    total_verifier_seconds = sum(
        float(arm["accounting"].get("verifier_seconds", 0.0)) for arm in arms
    )
    return {
        "schema": "neural-computer.recipe-control-flow-factorized-loop-growth-audit.v1",
        "status": (
            "promoted_factorized_replay_free_loop_growth"
            if all(gates.values())
            else "rejected"
        ),
        "seeds": list(seeds),
        "warm_reports": warm,
        "fresh_reports": fresh,
        "shuffled_reports": shuffled,
        "later_transfer_ratios": later_transfer_ratios,
        "gates": gates,
        "claim_boundary": (
            "Promoted bounded factorized replay-free structural credit transfer "
            "through non-commuting external loop growth; not efficient arbitrary "
            "program synthesis, unrestricted memory growth, or general continual learning."
        ),
        "accounting": {
            "unique_verifier_bits": sum(
                int(arm["accounting"]["unique_verifier_bits"]) for arm in arms
            ),
            "unique_logical_lifetimes": sum(
                int(arm["accounting"]["unique_logical_lifetimes"]) for arm in arms
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": sum(
                float(arm["accounting"].get("wall_seconds", 0.0)) for arm in arms
            ),
            "verifier_seconds": total_verifier_seconds,
            "mean_verifier_latency_seconds": (
                total_verifier_seconds / total_lifetimes
                if total_lifetimes
                else 0.0
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args()
    started = time.perf_counter()
    report = run(tuple(args.seeds))
    report["wall_seconds"] = time.perf_counter() - started
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "promoted_factorized_replay_free_loop_growth":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
