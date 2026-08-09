"""Audit positive transfer in an external outcome-only program router."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_outcome_program_router.train import (
    _evaluate,
    _hidden_choices,
    _stable_prefix,
    _train_executable_bank,
    _train_stream,
)
from neural_computer import (
    AmodalCognitiveController,
    ExternalOutcomeProgramRouter,
    ExternalOutcomeProgramRouterState,
    ExternalOutcomeValueBaseline,
)

EVENT_WIDTH = 4
PHASES = 2
SOURCE_PROGRAMS = 2
PROGRAM_CAPACITY = 3
SOURCE_EPISODES = 2000
TARGET_EPISODES = 7000
EVALUATION_EPISODES = 300
PROBE_EPISODES = 64
MASTER_THRESHOLD = 0.8
EXECUTION_TOLERANCE = 1e-5


def _new_value_baseline() -> ExternalOutcomeValueBaseline:
    return ExternalOutcomeValueBaseline(
        feature_width=EVENT_WIDTH * PHASES + PHASES,
        initial_learning_rate=0.05,
        initial_trace_decay=0.90,
    )


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _events(count: int, *, include_new: bool, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    events = torch.randn(count, EVENT_WIDTH, generator=generator)
    if not include_new:
        events[:, -1] = 0.0
        return events
    events[::4, -1] = 0.0
    events[1::4, :-1] = 0.0
    events[1::4, -1] = 3.0
    events[2::4, -1] = 0.0
    events[3::4, -1] = 0.0
    return events


def _relations() -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.randn(
        PHASES,
        SOURCE_PROGRAMS,
        EVENT_WIDTH - 1,
        generator=torch.Generator(device="cpu").manual_seed(240101),
    )
    source_relation = torch.zeros(PHASES, SOURCE_PROGRAMS, EVENT_WIDTH)
    source_relation[..., :-1] = source
    target = torch.zeros(PHASES, PROGRAM_CAPACITY, EVENT_WIDTH)
    target[:, :SOURCE_PROGRAMS] = source_relation
    target[:, 2, -1] = 1.5
    return source_relation, target


def _accuracy(
    router: ExternalOutcomeProgramRouter,
    state: ExternalOutcomeProgramRouterState,
    machine: torch.nn.Module,
    memory: torch.nn.Module,
    events: torch.Tensor,
    relation: torch.Tensor,
) -> float:
    return _evaluate(
        router=router,
        state=state,
        machine=machine,
        memory=memory,
        events=events,
        relation=relation,
        phase_count=PHASES,
        tolerance=EXECUTION_TOLERANCE,
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    source_relation, target_relation = _relations()
    machine, memory, _interpreter_loss = _train_executable_bank(
        seed=seed + 40_000,
        program_count=PROGRAM_CAPACITY,
        event_width=EVENT_WIDTH,
        register_width=4,
        instruction_width=8,
        updates=900,
    )
    machine_digest = _digest(machine)
    memory_digest = _digest(memory)
    controller = AmodalCognitiveController(
        width=8,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    source_train = _events(SOURCE_EPISODES, include_new=False, seed=seed + 1)
    source_eval = _events(EVALUATION_EPISODES, include_new=False, seed=seed + 2)
    target_train = _events(TARGET_EPISODES, include_new=True, seed=seed + 3)
    target_eval = _events(EVALUATION_EPISODES, include_new=True, seed=seed + 4)
    source_router = ExternalOutcomeProgramRouter(
        feature_width=EVENT_WIDTH * PHASES + PHASES,
        program_capacity=PROGRAM_CAPACITY,
        initial_programs=SOURCE_PROGRAMS,
        initial_learning_rate=0.03,
        initial_trace_decay=0.95,
        initial_baseline_rate=0.02,
    )
    source_state = source_router.initial_state(1)
    source_value_baseline = _new_value_baseline()
    source_value_state = source_value_baseline.initial_state(1)
    torch.manual_seed(seed + 10)
    source_state, source_value_state, source_outcomes, _ = _train_stream(
        router=source_router,
        state=source_state,
        value_baseline=source_value_baseline,
        value_state=source_value_state,
        machine=machine,
        memory=memory,
        events=source_train,
        relation=source_relation,
        eval_events=source_eval,
        eval_relation=source_relation,
        phase_count=PHASES,
        eval_every=source_train.shape[0] + 1,
        tolerance=EXECUTION_TOLERANCE,
    )
    source_before = _accuracy(
        source_router,
        source_state,
        machine,
        memory,
        source_eval,
        source_relation,
    )
    source_state_digest = source_router._state_digest(source_state)

    target_probe = target_train[:PROBE_EPISODES]
    target_continuation = target_train[PROBE_EPISODES:]
    candidate_value_states: dict[int, object] = {}
    candidate_value_baselines: dict[int, ExternalOutcomeValueBaseline] = {}

    def probe(
        transfer_router: ExternalOutcomeProgramRouter,
        transfer_state: ExternalOutcomeProgramRouterState,
        fresh_router: ExternalOutcomeProgramRouter,
        fresh_state: ExternalOutcomeProgramRouterState,
    ) -> tuple[
        float,
        float,
        ExternalOutcomeProgramRouterState,
        ExternalOutcomeProgramRouterState,
    ]:
        transfer_value_baseline = copy.deepcopy(source_value_baseline)
        transfer_value_state = copy.deepcopy(source_value_state)
        torch.manual_seed(seed + 20)
        transfer_state, transfer_value_state, _, _ = _train_stream(
            router=transfer_router,
            state=transfer_state,
            value_baseline=transfer_value_baseline,
            value_state=transfer_value_state,
            machine=machine,
            memory=memory,
            events=target_probe,
            relation=target_relation,
            eval_events=target_eval,
            eval_relation=target_relation,
            phase_count=PHASES,
            eval_every=target_probe.shape[0] + 1,
            tolerance=EXECUTION_TOLERANCE,
            protected_programs=SOURCE_PROGRAMS,
        )
        candidate_value_states[id(transfer_router)] = transfer_value_state
        candidate_value_baselines[id(transfer_router)] = transfer_value_baseline
        transfer_score = 1.0 - _accuracy(
            transfer_router,
            transfer_state,
            machine,
            memory,
            target_eval,
            target_relation,
        )
        fresh_value_baseline = copy.deepcopy(source_value_baseline)
        fresh_value_state = fresh_value_baseline.initial_state(1)
        torch.manual_seed(seed + 20)
        fresh_state, fresh_value_state, _, _ = _train_stream(
            router=fresh_router,
            state=fresh_state,
            value_baseline=fresh_value_baseline,
            value_state=fresh_value_state,
            machine=machine,
            memory=memory,
            events=target_probe,
            relation=target_relation,
            eval_events=target_eval,
            eval_relation=target_relation,
            phase_count=PHASES,
            eval_every=target_probe.shape[0] + 1,
            tolerance=EXECUTION_TOLERANCE,
        )
        candidate_value_states[id(fresh_router)] = fresh_value_state
        candidate_value_baselines[id(fresh_router)] = fresh_value_baseline
        fresh_score = 1.0 - _accuracy(
            fresh_router,
            fresh_state,
            machine,
            memory,
            target_eval,
            target_relation,
        )
        return transfer_score, fresh_score, transfer_state, fresh_state

    receipt, warm_router, warm_state = source_router.select_verified_transfer_prior(
        source_state,
        PROGRAM_CAPACITY,
        PROGRAM_CAPACITY,
        probe,
        probe_updates=PROBE_EPISODES,
    )
    torch.manual_seed(seed + 21)
    warm_state, _warm_value_state, warm_outcomes, warm_progress = _train_stream(
        router=warm_router,
        state=warm_state,
        value_baseline=candidate_value_baselines[id(warm_router)],
        value_state=candidate_value_states[id(warm_router)],
        machine=machine,
        memory=memory,
        events=target_continuation,
        relation=target_relation,
        eval_events=target_eval,
        eval_relation=target_relation,
        phase_count=PHASES,
        eval_every=500,
        tolerance=EXECUTION_TOLERANCE,
        protected_programs=SOURCE_PROGRAMS,
    )
    warm_target = _accuracy(
        warm_router,
        warm_state,
        machine,
        memory,
        target_eval,
        target_relation,
    )
    warm_source = _accuracy(
        warm_router,
        warm_state,
        machine,
        memory,
        source_eval,
        source_relation,
    )

    fresh_router = ExternalOutcomeProgramRouter(
        feature_width=EVENT_WIDTH * PHASES + PHASES,
        program_capacity=PROGRAM_CAPACITY,
        initial_programs=PROGRAM_CAPACITY,
        initial_learning_rate=0.03,
        initial_trace_decay=0.95,
        initial_baseline_rate=0.02,
    )
    fresh_state = fresh_router.initial_state(1)
    fresh_value_baseline = _new_value_baseline()
    fresh_value_state = fresh_value_baseline.initial_state(1)
    torch.manual_seed(seed + 21)
    fresh_state, fresh_value_state, fresh_outcomes, fresh_progress = _train_stream(
        router=fresh_router,
        state=fresh_state,
        value_baseline=fresh_value_baseline,
        value_state=fresh_value_state,
        machine=machine,
        memory=memory,
        events=target_train,
        relation=target_relation,
        eval_events=target_eval,
        eval_relation=target_relation,
        phase_count=PHASES,
        eval_every=500,
        tolerance=EXECUTION_TOLERANCE,
    )
    fresh_target = _accuracy(
        fresh_router,
        fresh_state,
        machine,
        memory,
        target_eval,
        target_relation,
    )

    shuffled_router = ExternalOutcomeProgramRouter(
        feature_width=EVENT_WIDTH * PHASES + PHASES,
        program_capacity=PROGRAM_CAPACITY,
        initial_programs=PROGRAM_CAPACITY,
        initial_learning_rate=0.03,
        initial_trace_decay=0.95,
        initial_baseline_rate=0.02,
    )
    permutation = torch.randperm(
        fresh_outcomes.shape[0],
        generator=torch.Generator(device="cpu").manual_seed(seed + 30),
    )
    shuffled_state = shuffled_router.initial_state(1)
    shuffled_value_baseline = _new_value_baseline()
    shuffled_value_state = shuffled_value_baseline.initial_state(1)
    torch.manual_seed(seed + 21)
    shuffled_state, shuffled_value_state, shuffled_outcomes, _ = _train_stream(
        router=shuffled_router,
        state=shuffled_state,
        value_baseline=shuffled_value_baseline,
        value_state=shuffled_value_state,
        machine=machine,
        memory=memory,
        events=target_train,
        relation=target_relation,
        eval_events=target_eval,
        eval_relation=target_relation,
        phase_count=PHASES,
        eval_every=target_train.shape[0] + 1,
        tolerance=EXECUTION_TOLERANCE,
        feedback_override=fresh_outcomes[permutation],
    )
    shuffled_target = _accuracy(
        shuffled_router,
        shuffled_state,
        machine,
        memory,
        target_eval,
        target_relation,
    )

    warm_stable = _stable_prefix(warm_progress, MASTER_THRESHOLD)
    fresh_stable = _stable_prefix(fresh_progress, MASTER_THRESHOLD)
    old_cell_retention = _accuracy(
        source_router,
        source_state,
        machine,
        memory,
        source_eval,
        source_relation,
    )
    warm_target_cost = (
        None
        if warm_stable is None
        else 2 * PROBE_EPISODES + warm_stable
    )
    fresh_target_cost = fresh_stable
    warm_stable_unique_bits = (
        None
        if warm_stable is None
        else SOURCE_EPISODES + PROBE_EPISODES + warm_stable
    )
    fresh_stable_unique_bits = (
        None if fresh_stable is None else SOURCE_EPISODES + fresh_stable
    )
    gates = {
        "controller_frozen": controller_digest == _digest(controller),
        "executor_frozen": machine_digest == _digest(machine),
        "program_memory_frozen": memory_digest == _digest(memory),
        "source_mastery": source_before >= MASTER_THRESHOLD,
        "warm_target_mastery": warm_target >= MASTER_THRESHOLD,
        "fresh_target_mastery": fresh_target >= MASTER_THRESHOLD,
        "warm_beats_fresh": (
            warm_target_cost is not None
            and fresh_target_cost is not None
            and warm_target_cost < fresh_target_cost
        ),
        "old_external_cell_retained": old_cell_retention >= MASTER_THRESHOLD,
        "shuffled_outcome_control_rejected": shuffled_target < MASTER_THRESHOLD,
        "source_state_unchanged_during_selection": (
            source_state_digest == source_router._state_digest(source_state)
        ),
        "new_program_present_in_target": bool(
            (_hidden_choices(target_eval, target_relation) == 2).any()
        ),
        "zero_old_regime_replay": True,
    }
    report = {
        "schema": "neural-computer.external-program-compounding-pressure-test.v1",
        "claim_boundary": (
            "A frozen-controller external program router can retain a mastered "
            "route state while acquiring an added opaque program at lower "
            "accounted cost than a fresh route state; this is not arbitrary "
            "program induction or general continual learning."
        ),
        "seed": seed,
        "configuration": {
            "event_width": EVENT_WIDTH,
            "phases": PHASES,
            "source_programs": SOURCE_PROGRAMS,
            "program_capacity": PROGRAM_CAPACITY,
            "source_episodes": SOURCE_EPISODES,
            "target_episodes": TARGET_EPISODES,
            "evaluation_episodes": EVALUATION_EPISODES,
            "probe_episodes_per_candidate": PROBE_EPISODES,
            "mastery_threshold": MASTER_THRESHOLD,
            "policy": "external_outcome_program_compounding_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "source": {
            "accuracy": source_before,
            "outcome_mean": float(source_outcomes.mean()),
            "state_digest": source_state_digest,
        },
        "prior_selection": {
            "schema": receipt.schema,
            "selected_initialization": receipt.selected_initialization,
            "source_active_programs": receipt.source_active_programs,
            "destination_capacity": receipt.destination_capacity,
            "destination_active_programs": receipt.destination_active_programs,
            "transfer_probe_score": receipt.transfer_probe_score,
            "fresh_probe_score": receipt.fresh_probe_score,
            "probe_updates_per_candidate": receipt.probe_updates,
            "source_state_digest": receipt.source_state_digest,
            "selected_state_digest": receipt.selected_state_digest,
            "reason": receipt.reason,
        },
        "warm": {
            "target_accuracy": warm_target,
            "mutable_target_cell_source_probe": warm_source,
            "continuation_updates": int(warm_outcomes.shape[0]),
            "accounted_updates": 2 * PROBE_EPISODES + int(warm_outcomes.shape[0]),
            "stable_continuation_episodes": warm_stable,
            "stable_target_cost_including_probe": warm_target_cost,
        },
        "fresh": {
            "target_accuracy": fresh_target,
            "updates": int(fresh_outcomes.shape[0]),
            "stable_target_episodes": fresh_stable,
            "stable_target_cost": fresh_target_cost,
        },
        "retention": {
            "old_external_cell_accuracy_after_target_learning": old_cell_retention,
        },
        "shuffled_outcome_control": {
            "target_accuracy": shuffled_target,
            "updates": int(shuffled_outcomes.shape[0]),
        },
        "accounting": {
            "unique_verifier_bits": SOURCE_EPISODES + TARGET_EPISODES,
            "unique_logical_lifetimes": SOURCE_EPISODES + TARGET_EPISODES,
            "warm_model_state_updates": SOURCE_EPISODES
            + 2 * PROBE_EPISODES
            + int(warm_outcomes.shape[0]),
            "fresh_model_state_updates": SOURCE_EPISODES + TARGET_EPISODES,
            "shadow_prior_probe_updates": 2 * PROBE_EPISODES,
            "old_regime_replay_during_target_adaptation": 0,
            "controller_optimizer_updates": 0,
            "interpreter_optimizer_updates": 900,
            "stable_bits_to_threshold": {
                "warm": warm_stable_unique_bits,
                "fresh": fresh_stable_unique_bits,
            },
            "warm_stable_model_updates": (
                None
                if warm_target_cost is None
                else SOURCE_EPISODES + warm_target_cost
            ),
            "fresh_stable_model_updates": (
                None
                if fresh_target_cost is None
                else SOURCE_EPISODES + fresh_target_cost
            ),
            "warm_stable_unique_verifier_bits": warm_stable_unique_bits,
            "fresh_stable_unique_verifier_bits": fresh_stable_unique_bits,
            "warm_stable_target_cost": warm_target_cost,
            "fresh_stable_target_cost": fresh_target_cost,
            "transfer_ratio_against_fresh_target": (
                None
                if warm_target_cost is None or fresh_target_cost is None
                else fresh_target_cost / max(warm_target_cost, 1)
            ),
        },
        "digests": {
            "controller": controller_digest,
            "executor": machine_digest,
            "program_memory": memory_digest,
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
