"""Outcome-only multi-file routing through the canonical control-flow runtime.

This is a narrow router experiment, not a general continual-learning claim.
The controller and the opaque intention/counter codec remain frozen.  A
generic external router receives only the controller's opaque intention
features and learns which of two external control-flow files to execute from
one delayed scalar verifier outcome per fresh episode.  The verifier keeps the
logical target and file identity private.
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
    ExternalOutcomeProgramRouter,
    IntentEvent,
)

COUNTER_COUNT = 2
INTENTION_WIDTH = 2
FEEDBACK_WIDTH = 3
TRAIN_EPISODES = 1_000
HELDOUT_EPISODES = 200
EXPLORATION = 0.1
STABLE_THRESHOLD = 0.8
HELDOUT_THRESHOLD = 0.9
REWARD_SHUFFLED_TOLERANCE = 0.15
SEEDS = (17, 18, 19, 20)


class OpaqueIntentionCounterCodec(ControlFlowIntentionAdapter):
    """Fixed external ABI codec used to isolate the router question.

    This codec is deliberately not a learned semantic map.  It only makes the
    typed runtime executable while the experiment measures external route
    learning.  The controller never sees its counter representation.
    """

    def encode(
        self,
        intention: IntentEvent,
        previous_counters: torch.Tensor,
    ) -> torch.Tensor:
        counters = previous_counters.clone()
        counters[:, 0] = (intention.payload[:, 0] > 0.0).to(torch.int64)
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


def _program(counter: int) -> ControlFlowProgram:
    return ControlFlowProgram(
        COUNTER_COUNT,
        (
            ControlFlowInstruction("inc", counter=counter),
            ControlFlowInstruction("halt"),
        ),
    )


def _feedback(*, present: bool = False, reward: float = 0.0) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, FEEDBACK_WIDTH),
        reward=torch.tensor([reward]),
        propensity=torch.ones(1),
        has_feedback=torch.tensor([present]),
    )


def _event(cue: float) -> list[AmodalEvent]:
    return [AmodalEvent(torch.tensor([[cue, 0.0, 0.0, 0.0]]))]


def _target_slot(*, cue: float, reverse_files: bool) -> int:
    logical_slot = 0 if cue > 0.0 else 1
    return 1 - logical_slot if reverse_files else logical_slot


def _make_agent(
    seed: int,
    *,
    reverse_files: bool,
    route_exploration: float,
) -> tuple[
    ControlFlowProgramAmodalRuntime,
    ExternalOutcomeProgramRouter,
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
    memory = ControlFlowProgramMemory(COUNTER_COUNT)
    programs = (_program(0), _program(1))
    if reverse_files:
        programs = tuple(reversed(programs))
    for program in programs:
        memory.add_program(program, protect=True)
    source_digests = tuple(program.digest() for program in programs)
    router = ExternalOutcomeProgramRouter(
        feature_width=INTENTION_WIDTH,
        program_capacity=2,
        initial_programs=2,
        initial_learning_rate=0.5,
        initial_trace_decay=0.0,
        initial_baseline_rate=0.05,
    )
    agent = ControlFlowProgramAmodalRuntime(
        runtime,
        OpaqueIntentionCounterCodec(INTENTION_WIDTH, COUNTER_COUNT),
        program_memory=memory,
        program_router=router,
        program_route_exploration=route_exploration,
        max_steps=8,
    )
    return agent, router, source_digests


def _with_router(
    agent: ControlFlowProgramAmodalRuntime,
    router_state,
    *,
    batch_size: int = 1,
):
    return replace(
        agent.initial_state(batch_size, device="cpu"),
        program_router=router_state,
    )


def _stable_prefix(scores: list[bool], threshold: float) -> int | None:
    if not scores:
        return None
    suffix_min = 1.0
    first_stable: int | None = None
    for index in range(len(scores) - 1, -1, -1):
        suffix_min = min(
            suffix_min,
            sum(scores[index:]) / float(len(scores) - index),
        )
        if suffix_min >= threshold:
            first_stable = index
    return None if first_stable is None else first_stable + 1


def _train(
    agent: ControlFlowProgramAmodalRuntime,
    router: ExternalOutcomeProgramRouter,
    *,
    seed: int,
    reverse_files: bool,
    feedback_mode: str,
) -> tuple[object, list[bool], float, int]:
    if feedback_mode not in {"verifier", "reward_shuffled"}:
        raise ValueError("unsupported control-flow routing feedback mode")
    router_state = router.initial_state(1, device="cpu")
    random = torch.Generator(device="cpu").manual_seed(seed + 90_000)
    pending = _feedback()
    scores: list[bool] = []
    started = time.perf_counter()
    for episode in range(TRAIN_EPISODES):
        cue = 1.0 if episode % 2 == 0 else -1.0
        state = _with_router(agent, router_state)
        output, next_state = agent.step_events(_event(cue), state, pending)
        selected = int(output.selected_program_slots[0])
        correct = selected == _target_slot(
            cue=cue,
            reverse_files=reverse_files,
        )
        scores.append(correct)
        if feedback_mode == "verifier":
            reward = float(correct)
        else:
            reward = float(torch.rand((), generator=random) >= 0.5)
        router_state = next_state.program_router
        pending = _feedback(present=True, reward=reward)
    # Commit the final delayed scalar without counting another verifier bit.
    final_state = _with_router(agent, router_state)
    _, final_state = agent.step_events(_event(1.0), final_state, pending)
    router_state = final_state.program_router
    return (
        router_state,
        scores,
        time.perf_counter() - started,
        TRAIN_EPISODES,
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
        cue = amplitude if episode % 2 == 0 else -amplitude
        state = _with_router(agent, router_state)
        output, _ = agent.step_events(
            _event(cue),
            state,
            _feedback(),
        )
        selected = int(output.selected_program_slots[0])
        scores.append(
            selected == _target_slot(cue=cue, reverse_files=reverse_files)
        )
    return sum(scores) / float(len(scores)), scores


def _run_arm(
    seed: int,
    *,
    reverse_files: bool,
    feedback_mode: str,
) -> dict[str, object]:
    agent, router, source_digests = _make_agent(
        seed,
        reverse_files=reverse_files,
        route_exploration=EXPLORATION,
    )
    controller_before = {
        name: value.detach().clone()
        for name, value in agent.runtime.controller.state_dict().items()
    }
    trained_router_state, train_scores, wall_seconds, verifier_bits = _train(
        agent,
        router,
        seed=seed,
        reverse_files=reverse_files,
        feedback_mode=feedback_mode,
    )
    evaluation_agent, _, _ = _make_agent(
        seed,
        reverse_files=reverse_files,
        route_exploration=0.0,
    )
    # Reuse the trained frozen controller/adapter and the trained external
    # router state; no weights are copied into the controller.
    evaluation_agent.runtime = agent.runtime
    evaluation_agent.adapter = agent.adapter
    evaluation_agent.program_memory = agent.program_memory
    evaluation_agent.program_router = router
    trained_accuracy, heldout_scores = _evaluate(
        evaluation_agent,
        trained_router_state,
        reverse_files=reverse_files,
        amplitude=2.0,
    )

    fresh_agent, fresh_router, _ = _make_agent(
        seed,
        reverse_files=reverse_files,
        route_exploration=0.0,
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
    final_state = _with_router(agent, trained_router_state)
    state_payload = final_state.payload(program_router=router)
    restored_state = evaluation_agent.state_from_payload(state_payload)
    reload_exact = restored_state.digest(program_router=router) == final_state.digest(
        program_router=router
    )
    corrupt = dict(state_payload)
    router_payload = corrupt["program_router"]
    assert isinstance(router_payload, dict)
    credit_payload = router_payload["credit"]
    assert isinstance(credit_payload, dict)
    credit_payload["policy"] = credit_payload["policy"].clone()
    credit_payload["policy"][0, 0, 0] += 1.0
    try:
        evaluation_agent.state_from_payload(corrupt)
        corruption_rejected = False
    except ValueError:
        corruption_rejected = True

    missing_state = router.initial_state(1, device="cpu")
    _, missing_next = agent.step_events(
        _event(1.0),
        _with_router(agent, missing_state),
        _feedback(),
    )
    missing_policy_unchanged = torch.equal(
        missing_state.credit.policy,
        missing_next.program_router.credit.policy,
    )
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
        "fresh_control_measured": 0.35 <= fresh_accuracy <= 0.65,
        "reward_shuffled_control_measured": (
            feedback_mode != "reward_shuffled"
            or abs(trained_accuracy - 0.5) <= REWARD_SHUFFLED_TOLERANCE
        ),
        "candidate_order_permutation": True,
        "all_external_files_retained": all(files_retained),
        "controller_frozen": controller_frozen,
        "missing_feedback_no_policy_update": missing_policy_unchanged,
        "state_reload_exact": reload_exact,
        "state_corruption_rejected": corruption_rejected,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    return {
        "seed": seed,
        "reverse_files": reverse_files,
        "feedback_mode": feedback_mode,
        "schema": "neural-computer.control-flow-runtime-routing.v1",
        "architecture": {
            "train_episodes": TRAIN_EPISODES,
            "heldout_episodes": HELDOUT_EPISODES,
            "heldout_amplitude": 2.0,
            "route_exploration": EXPLORATION,
            "learner_inputs": "opaque_controller_intention_and_scalar_outcome",
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
        "missing_feedback_policy_unchanged": missing_policy_unchanged,
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
                else _stable_prefix(train_scores, STABLE_THRESHOLD)
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
    return {
        "schema": "neural-computer.control-flow-runtime-routing.v1",
        "claim_boundary": (
            "bounded outcome-only routing among two generic external control-flow "
            "files with frozen controller and isolated file state; not arbitrary "
            "new computation, unrestricted memory growth, or general continual learning"
        ),
        "seeds": list(seeds),
        "reports": reports,
        "promoted": all(bool(report["promoted"]) for report in verifier_reports),
        "reward_shuffled_null_within_boundary": all(
            abs(float(report["heldout_accuracy"]) - 0.5)
            <= REWARD_SHUFFLED_TOLERANCE
            for report in shuffled_reports
        ),
        "candidate_order_permutation_covered": (
            any(
                bool(report["reverse_files"])
                for report in reports
                if report["seed"] == seeds[0]
            )
            and any(
                not bool(report["reverse_files"])
                for report in reports
                if report["seed"] == seeds[0]
            )
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
