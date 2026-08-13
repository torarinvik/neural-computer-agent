"""Replay-free adaptive growth through a non-commuting control-flow loop.

This pressure test uses a clear-loop root and grows it into transfer loops
whose jump targets must move with each inserted instruction. It therefore
exercises the structural edit ABI, not only straight-line instruction growth.
The controller is frozen/absent; the experiment isolates the external
CPU/files learner and keeps target identities private to the verifier.
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
    values = _matches(candidate, reference, states)
    return sum(values) / len(values)


def _policy(
    *,
    initial_horizon: int = INITIAL_HORIZON,
    maximum_horizon: int = MAXIMUM_HORIZON,
) -> ControlFlowProgramFrontierGrowth:
    return ControlFlowProgramFrontierGrowth(
        COUNTER_COUNT,
        initial_horizon=initial_horizon,
        maximum_horizon=maximum_horizon,
        beam_width=64,
        max_depth=96,
        minimum_quality=0.0,
        parent_temperature=0.2,
        exploration=0.7,
        proposal_retry_limit=256,
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


def _run_positive(seed: int, *, reverse_inputs: bool) -> dict[str, object]:
    started = time.perf_counter()
    policy = _policy()
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    root_slot = memory.add_program(ROOT, protect=True)
    retained: list[tuple[int, ControlFlowProgram]] = [(root_slot, ROOT)]
    state = policy.initial_state(ROOT, root_quality=1.0)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    training_states = tuple(reversed(TRAIN_STATES)) if reverse_inputs else TRAIN_STATES
    reports: list[dict[str, object]] = []
    missing_evidence_no_write = True

    for stage_index, target in enumerate(STAGES, start=1):
        before_missing = state.digest()
        try:
            proposal = policy.propose(state, generator=generator)
            policy.record_outcomes(state, proposal, ())
            missing_evidence_no_write = False
        except (RuntimeError, ValueError):
            missing_evidence_no_write = (
                missing_evidence_no_write and state.digest() == before_missing
            )

        horizon_receipt, state = policy.expand_horizon_verified(
            state,
            lambda candidate_state: _retains(memory, tuple(retained)),
        )
        if not horizon_receipt.accepted:
            raise RuntimeError("loop-growth horizon expansion was rejected")

        candidate: ControlFlowProgram | None = None
        evaluations_before = state.frontier.evaluations
        for _ in range(MAX_CANDIDATES):
            proposal = policy.propose(state, generator=generator)
            feedback = policy.record_outcomes(
                state,
                proposal,
                _matches(proposal.proposal.program, target, training_states),
                threshold=1.0,
                min_observations=len(training_states),
                min_stable_observations=len(training_states),
            )
            state = feedback.state
            if feedback.accepted and proposal.proposal.program.digest() == target.digest():
                candidate = proposal.proposal.program
                break
        if candidate is None:
            raise RuntimeError(f"loop-growth target at stage {stage_index} was not found")

        promote_receipt, state = policy.promote_root_verified(
            state,
            candidate,
            lambda candidate_state, candidate=candidate, target=target: _retains(
                memory,
                tuple(retained),
                candidate=candidate,
                reference=target,
            ),
        )
        if not promote_receipt.accepted:
            raise RuntimeError("loop-growth root promotion was rejected")
        slot = memory.add_program(candidate, protect=True)
        retained.append((slot, target))
        restored_state = type(state).from_payload(state.payload())
        restored_memory = ControlFlowProgramMemory.from_payload(memory.payload())
        reports.append(
            {
                "stage": stage_index,
                "target_length": len(target.instructions),
                "candidate_digest": candidate.digest(),
                "candidate_slot": slot,
                "horizon": state.horizon,
                "rung": state.rung,
                "candidate_evaluations": state.frontier.evaluations - evaluations_before,
                "heldout_accuracy": _accuracy(candidate, target, HELDOUT_STATES),
                "retention_accuracy": min(
                    _accuracy(memory.program(retained_slot), expected, HELDOUT_STATES)
                    for retained_slot, expected in retained
                ),
                "state_reload_exact": restored_state.digest() == state.digest(),
                "memory_reload_exact": restored_memory.digest() == memory.digest(),
            }
        )

    corrupted = state.payload()
    corrupted["sha256"] = "0" * 64
    try:
        type(state).from_payload(corrupted)
        corruption_rejected = False
    except ValueError:
        corruption_rejected = True

    gates = {
        "all_loop_rungs_discovered": len(reports) == len(STAGES),
        "all_heldout_mastered": all(item["heldout_accuracy"] == 1.0 for item in reports),
        "all_prior_loops_retained": all(
            item["retention_accuracy"] == 1.0 for item in reports
        ),
        "horizon_grew_one_step_at_a_time": [item["horizon"] for item in reports]
        == [5, 6, 7],
        "state_reload_exact": all(item["state_reload_exact"] for item in reports),
        "memory_reload_exact": all(item["memory_reload_exact"] for item in reports),
        "memory_corruption_rejected": corruption_rejected,
        "missing_evidence_no_write": missing_evidence_no_write,
        "zero_replayed_examples": True,
        "zero_optimizer_updates": True,
    }
    return {
        "seed": seed,
        "reverse_inputs": reverse_inputs,
        "schema": "neural-computer.recipe-control-flow-adaptive-loop-growth.v1",
        "stage_reports": reports,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": state.frontier.evaluations * len(training_states),
            "unique_logical_lifetimes": state.frontier.evaluations,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - started,
        },
        "promoted": all(gates.values()),
    }


def _run_fresh(seed: int) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    total_evaluations = 0
    for stage_index, target in enumerate(STAGES, start=1):
        horizon = len(target.instructions)
        policy = _policy(initial_horizon=horizon, maximum_horizon=horizon)
        state = policy.initial_state(ROOT, root_quality=1.0)
        generator = torch.Generator(device="cpu").manual_seed(seed + stage_index * 101)
        found = False
        evaluations = 0
        for _ in range(MAX_CANDIDATES):
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
        reports.append(
            {
                "stage": stage_index,
                "horizon": horizon,
                "found": found,
                "candidate_evaluations": evaluations,
            }
        )
    return {
        "seed": seed,
        "stage_reports": reports,
        "all_targets_found": all(item["found"] for item in reports),
        "accounting": {
            "unique_verifier_bits": total_evaluations * len(TRAIN_STATES),
            "unique_logical_lifetimes": total_evaluations,
            "optimizer_updates": 0,
            "replayed_examples": 0,
        },
    }


def _run_shuffled(seed: int) -> dict[str, object]:
    policy = _policy()
    state = policy.initial_state(ROOT, root_quality=1.0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 70_000)
    accepted_rungs = 0
    evaluations = 0
    for _ in STAGES:
        _, state = policy.expand_horizon_verified(state, lambda _: True)
        for _ in range(160):
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
        "not_promoted": accepted_rungs < len(STAGES),
        "accounting": {
            "unique_verifier_bits": evaluations * len(TRAIN_STATES),
            "unique_logical_lifetimes": evaluations,
            "optimizer_updates": 0,
            "replayed_examples": 0,
        },
    }


def run(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    positive = tuple(
        _run_positive(seed, reverse_inputs=reverse_inputs)
        for seed in seeds
        for reverse_inputs in (False, True)
    )
    fresh = tuple(_run_fresh(seed) for seed in seeds)
    shuffled = tuple(_run_shuffled(seed) for seed in seeds)
    gates = {
        "positive_arms_promoted": all(item["promoted"] for item in positive),
        "fresh_final_rung_not_found": all(
            not item["stage_reports"][-1]["found"] for item in fresh
        ),
        "shuffled_feedback_not_promoted": all(item["not_promoted"] for item in shuffled),
    }
    arms = (*positive, *fresh, *shuffled)
    return {
        "schema": "neural-computer.recipe-control-flow-adaptive-loop-growth-audit.v1",
        "status": "promoted_replay_free_adaptive_loop_growth" if all(gates.values()) else "rejected",
        "seeds": list(seeds),
        "positive_reports": positive,
        "fresh_reports": fresh,
        "shuffled_reports": shuffled,
        "gates": gates,
        "claim_boundary": (
            "Promoted bounded replay-free adaptive growth through three non-commuting "
            "control-flow loop rungs with held-out retention; not efficient arbitrary "
            "program synthesis, unrestricted execution, unrestricted memory growth, "
            "or general continual learning."
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
    if report["status"] != "promoted_replay_free_adaptive_loop_growth":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
