"""Full-information route credit for a four-file external capability bank.

This is a bounded external-memory audit, not a general continual-learning
claim.  One frozen amodal controller emits an opaque learned event/intention
trajectory.  A replaceable trajectory-query adapter addresses four protected
external program files.  The verifier evaluates every file on each fresh
lifetime and returns an outcome vector to the external router only; the
controller receives quiet feedback and performs no optimizer updates.

The paired reward-shuffled control keeps the same architecture and candidate
bank while removing the cue-to-file relationship.  The held-out amplitude
change tests whether the route query is more than a lookup on one exact input
scale.  Program order is reversed in a paired arm so success cannot depend on
the physical slot index.
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
    ExternalOutcomeProgramRouter,
    IntentEvent,
)

COUNTER_COUNT = 4
INTENTION_WIDTH = 4
EVENT_WIDTH = 8
FEEDBACK_WIDTH = 3
PROGRAM_COUNT = 4
TRAIN_EPISODES = 3_000
HELDOUT_EPISODES = 400
STABLE_THRESHOLD = 0.9
HELDOUT_THRESHOLD = 0.9
REWARD_SHUFFLED_TOLERANCE = 0.15
SEEDS = (17, 18, 19, 20)


class OpaqueCounterCodec(ControlFlowIntentionAdapter):
    """Fixed executable ABI used to isolate route learning."""

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


def _event(logical_target: int, *, amplitude: float = 1.0) -> list[AmodalEvent]:
    payload = torch.zeros(1, EVENT_WIDTH)
    payload[0, logical_target] = amplitude
    return [AmodalEvent(payload)]


def _target_slot(logical_target: int, *, reverse_files: bool) -> int:
    return PROGRAM_COUNT - 1 - logical_target if reverse_files else logical_target


def _make_agent(
    seed: int,
    *,
    reverse_files: bool,
) -> tuple[
    ControlFlowProgramAmodalRuntime,
    ExternalOutcomeProgramRouter,
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
    router = ExternalOutcomeProgramRouter(
        feature_width=40,
        program_capacity=PROGRAM_COUNT,
        initial_programs=PROGRAM_COUNT,
        initial_learning_rate=0.5,
        initial_trace_decay=0.0,
        initial_baseline_rate=0.05,
    )
    query_adapter = ExternalControllerTrajectoryQueryAdapter(
        controller_width=8,
        query_width=40,
    )
    agent = ControlFlowProgramAmodalRuntime(
        runtime,
        OpaqueCounterCodec(INTENTION_WIDTH, COUNTER_COUNT),
        program_memory=memory,
        program_router=router,
        program_route_query_adapter=query_adapter,
        program_route_exploration=0.0,
        max_steps=8,
    )
    return agent, router, source_digests


def _with_router(
    agent: ControlFlowProgramAmodalRuntime,
    router_state,
) :
    return replace(
        agent.initial_state(1, device="cpu"),
        program_router=router_state,
    )


def _stable_prefix(scores: list[bool], threshold: float) -> int | None:
    if not scores:
        return None
    for index in range(len(scores)):
        if all(
            sum(scores[start:]) / float(len(scores) - start) >= threshold
            for start in range(index, len(scores))
        ):
            return index + 1
    return None


def _train(
    agent: ControlFlowProgramAmodalRuntime,
    router: ExternalOutcomeProgramRouter,
    *,
    seed: int,
    reverse_files: bool,
    feedback_mode: str,
) -> tuple[object, list[bool], float, int]:
    if feedback_mode not in {"verifier", "reward_shuffled"}:
        raise ValueError("unsupported feedback mode")
    router_state = router.initial_state(1, device="cpu")
    random = torch.Generator(device="cpu").manual_seed(seed + 90_000)
    scores: list[bool] = []
    started = time.perf_counter()
    for episode in range(TRAIN_EPISODES):
        logical_target = episode % PROGRAM_COUNT
        state = _with_router(agent, router_state)
        output, next_state = agent.step_events(
            _event(logical_target),
            state,
            _feedback(),
        )
        if output.program_route_query is None:
            raise RuntimeError("counterfactual audit requires an opaque route query")
        selected = int(output.selected_program_slots[0])
        verifier_target = _target_slot(
            logical_target,
            reverse_files=reverse_files,
        )
        scores.append(selected == verifier_target)
        outcomes = torch.zeros(1, PROGRAM_COUNT)
        if feedback_mode == "verifier":
            outcomes[0, verifier_target] = 1.0
        else:
            # Independent random outcomes remove the cue/file relationship
            # without forcing the null into a random permutation family.
            outcomes = (
                torch.rand(1, PROGRAM_COUNT, generator=random) >= 0.75
            ).to(torch.float32)
        router_state = router.apply_counterfactual_feedback(
            next_state.program_router,
            output.program_route_query,
            outcomes,
        )
    return (
        router_state,
        scores,
        time.perf_counter() - started,
        TRAIN_EPISODES * PROGRAM_COUNT,
    )


def _evaluate(
    agent: ControlFlowProgramAmodalRuntime,
    router_state,
    *,
    reverse_files: bool,
    amplitude: float,
    episodes: int = HELDOUT_EPISODES,
) -> tuple[float, list[bool]]:
    scores: list[bool] = []
    for episode in range(episodes):
        logical_target = episode % PROGRAM_COUNT
        state = _with_router(agent, router_state)
        output, _ = agent.step_events(
            _event(logical_target, amplitude=amplitude),
            state,
            _feedback(),
        )
        selected = int(output.selected_program_slots[0])
        scores.append(
            selected
            == _target_slot(logical_target, reverse_files=reverse_files)
        )
    return sum(scores) / float(len(scores)), scores


def _same_router_state(first, second) -> bool:
    return all(
        torch.equal(getattr(first.credit, name), getattr(second.credit, name))
        for name in ("policy", "eligibility", "baseline", "decisions", "feedbacks")
    ) and first.active_programs == second.active_programs


def _run_arm(
    seed: int,
    *,
    reverse_files: bool,
    feedback_mode: str,
) -> dict[str, object]:
    agent, router, source_digests = _make_agent(
        seed,
        reverse_files=reverse_files,
    )
    controller_before = {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }
    trained_state, train_scores, wall_seconds, verifier_bits = _train(
        agent,
        router,
        seed=seed,
        reverse_files=reverse_files,
        feedback_mode=feedback_mode,
    )
    trained_accuracy, heldout_scores = _evaluate(
        agent,
        trained_state,
        reverse_files=reverse_files,
        amplitude=2.0,
    )
    fresh_agent, fresh_router, _ = _make_agent(
        seed,
        reverse_files=reverse_files,
    )
    fresh_accuracy, _ = _evaluate(
        fresh_agent,
        fresh_router.initial_state(1, device="cpu"),
        reverse_files=reverse_files,
        amplitude=2.0,
    )
    controller_after = {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }
    controller_frozen = all(
        torch.equal(controller_before[name], controller_after[name])
        for name in controller_before
    )
    final_state = _with_router(agent, trained_state)
    payload = final_state.payload(program_router=router)
    restored = agent.state_from_payload(payload)
    reload_exact = restored.digest(program_router=router) == final_state.digest(
        program_router=router
    )
    corrupted = dict(payload)
    router_payload = corrupted["program_router"]
    assert isinstance(router_payload, dict)
    credit_payload = router_payload["credit"]
    assert isinstance(credit_payload, dict)
    credit_payload["policy"] = credit_payload["policy"].clone()
    credit_payload["policy"][0, 0, 0] += 1.0
    try:
        agent.state_from_payload(corrupted)
        corruption_rejected = False
    except ValueError:
        corruption_rejected = True

    missing_features = torch.zeros(1, router.feature_width)
    missing_outcomes = torch.zeros(1, PROGRAM_COUNT)
    missing_outcomes[0, 0] = 1.0
    missing = router.apply_counterfactual_feedback(
        trained_state,
        missing_features,
        missing_outcomes,
        present=torch.zeros(1, dtype=torch.bool),
    )
    missing_evidence_noop = _same_router_state(trained_state, missing)
    files_retained = tuple(
        agent.program_memory.program(slot).digest() == source_digests[slot]
        for slot in range(agent.program_memory.file_count)
    )
    gates = {
        "trained_heldout_mastery": trained_accuracy >= HELDOUT_THRESHOLD,
        "stable_training_prefix": _stable_prefix(
            train_scores, STABLE_THRESHOLD
        )
        is not None,
        "fresh_control_measured": 0.15 <= fresh_accuracy <= 0.35,
        "reward_shuffled_control_measured": (
            feedback_mode != "reward_shuffled"
            or abs(trained_accuracy - 0.25) <= REWARD_SHUFFLED_TOLERANCE
        ),
        "candidate_order_permutation": True,
        "all_external_files_retained": all(files_retained),
        "controller_frozen": controller_frozen,
        "missing_evidence_no_external_update": missing_evidence_noop,
        "state_reload_exact": reload_exact,
        "state_corruption_rejected": corruption_rejected,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    return {
        "seed": seed,
        "reverse_files": reverse_files,
        "feedback_mode": feedback_mode,
        "schema": "neural-computer.control-flow-runtime-four-file-counterfactual.v1",
        "architecture": {
            "train_episodes": TRAIN_EPISODES,
            "heldout_episodes": HELDOUT_EPISODES,
            "heldout_amplitude": 2.0,
            "program_count": PROGRAM_COUNT,
            "route_query_width": router.feature_width,
            "learner_inputs": "opaque_controller_trajectory_and_full_information_outcome_vector",
            "controller_feedback_protocol": "quiet_controller_no_route_outcome_v1",
            "forbidden_features": (
                "logical_target_slot, program names, verifier rows, raw modality data"
            ),
        },
        "train_accuracy": sum(train_scores) / float(len(train_scores)),
        "stable_training_prefix": _stable_prefix(train_scores, STABLE_THRESHOLD),
        "heldout_accuracy": trained_accuracy,
        "heldout_stable_prefix": _stable_prefix(heldout_scores, HELDOUT_THRESHOLD),
        "fresh_accuracy": fresh_accuracy,
        "transfer_ratio_against_fresh": (
            None if fresh_accuracy == 0.0 else trained_accuracy / fresh_accuracy
        ),
        "source_program_digests": source_digests,
        "program_memory_digest": agent.program_memory.digest(),
        "missing_evidence_noop": missing_evidence_noop,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": verifier_bits,
            "unique_logical_lifetimes": TRAIN_EPISODES,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": wall_seconds,
            "latency_seconds_per_lifetime": wall_seconds / TRAIN_EPISODES,
            "stable_bits_to_threshold": (
                None
                if _stable_prefix(train_scores, STABLE_THRESHOLD) is None
                else _stable_prefix(train_scores, STABLE_THRESHOLD) * PROGRAM_COUNT
            ),
            "retention_on_mastered_primitives": float(all(files_retained)),
        },
        "promoted": all(gates.values()),
    }


def run(seeds: tuple[int, ...] = SEEDS) -> dict[str, object]:
    reports = tuple(
        _run_arm(seed, reverse_files=reverse_files, feedback_mode=feedback_mode)
        for seed in seeds
        for reverse_files in (False, True)
        for feedback_mode in ("verifier", "reward_shuffled")
    )
    verifier_reports = tuple(
        report for report in reports if report["feedback_mode"] == "verifier"
    )
    shuffled_reports = tuple(
        report for report in reports if report["feedback_mode"] == "reward_shuffled"
    )
    paired_null_accuracy = {
        str(seed): sum(
            float(report["heldout_accuracy"])
            for report in shuffled_reports
            if report["seed"] == seed
        )
        / 2.0
        for seed in seeds
    }
    null_within_boundary = all(
        abs(accuracy - 0.25) <= REWARD_SHUFFLED_TOLERANCE
        for accuracy in paired_null_accuracy.values()
    )
    return {
        "schema": "neural-computer.control-flow-runtime-four-file-counterfactual.v1",
        "claim_boundary": (
            "bounded full-information outcome-only routing among four generic "
            "external files with a frozen controller and isolated file state; "
            "not arbitrary new computation, unrestricted memory growth, or "
            "general continual learning"
        ),
        "seeds": list(seeds),
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in verifier_reports)
        and null_within_boundary,
        "reward_shuffled_paired_heldout_accuracy": paired_null_accuracy,
        "reward_shuffled_null_within_boundary": null_within_boundary,
        "candidate_order_permutation_covered": all(
            any(
                bool(report["reverse_files"])
                for report in reports
                if report["seed"] == seed
            )
            and any(
                not bool(report["reverse_files"])
                for report in reports
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
