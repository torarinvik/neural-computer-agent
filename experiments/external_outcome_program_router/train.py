"""Pressure-test outcome-only routing over executable external programs.

The executable programs are pre-admitted opaque artifacts executed by the
shared external register interpreter.  The only component trained during the
continual-learning audit is the memory-side program router: it samples opaque
program choices, records exact propensities, and receives one terminal scalar
verifier outcome after a multi-step execution.  The controller, interpreter,
program artifacts, router rule, and value-baseline rule remain frozen.

This is an architecture pressure test, not a game task.  The verifier keeps a
private relation from learned event tensors to program sequences.  The learner
never receives that relation, a correct program index, or a semantic program
name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from neural_computer import (
    EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA,
    EXTERNAL_REGISTER_SCHEMA,
    ExternalCapabilityRegisterMachine,
    ExternalOutcomeProgramRouter,
    ExternalOutcomeProgramRouterState,
    ExternalOutcomeValueBaseline,
    ExternalProgramArtifact,
    ExternalSequenceProgramMemory,
)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_digest(
    router: ExternalOutcomeProgramRouter,
    state: ExternalOutcomeProgramRouterState,
) -> str:
    payload = router.state_payload(state)
    digest = hashlib.sha256()

    def visit(value: object) -> None:
        if isinstance(value, torch.Tensor):
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                digest.update(str(key).encode("utf-8"))
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode("utf-8"))

    visit(payload)
    return digest.hexdigest()


def _phase_feature(
    event: torch.Tensor,
    phase: int,
    phase_count: int,
) -> torch.Tensor:
    if event.ndim != 2 or phase_count < 1 or phase not in range(phase_count):
        raise ValueError("invalid phase feature inputs")
    blocks = [torch.zeros_like(event) for _ in range(phase_count)]
    blocks[phase] = event
    token = torch.zeros(
        event.shape[0],
        phase_count,
        device=event.device,
        dtype=event.dtype,
    )
    token[:, phase] = 1.0
    return torch.cat((*blocks, token), dim=-1)


def _hidden_choices(
    events: torch.Tensor,
    relation: torch.Tensor,
) -> torch.Tensor:
    """Verifier-private relation; never passed to the router."""

    if events.ndim != 2 or relation.ndim != 3:
        raise ValueError("hidden relation inputs have the wrong rank")
    if relation.shape[2] != events.shape[1]:
        raise ValueError("hidden relation width does not match events")
    return torch.einsum("be,pae->bpa", events, relation).argmax(dim=-1)


def _train_executable_bank(
    *,
    seed: int,
    program_count: int,
    event_width: int,
    register_width: int,
    instruction_width: int,
    updates: int,
) -> tuple[ExternalCapabilityRegisterMachine, ExternalSequenceProgramMemory, float]:
    """Pre-admit executable artifacts, then freeze the shared interpreter."""

    if program_count < 1 or updates < 1:
        raise ValueError("program count and interpreter updates must be positive")
    torch.manual_seed(seed)
    machine = ExternalCapabilityRegisterMachine(
        event_width,
        1,
        1,
        register_width,
        instruction_width,
        interpreter_hidden=32,
        operator_rank=4,
        operator_mode="factorized_low_rank",
    )
    codes = torch.nn.Parameter(torch.randn(program_count, instruction_width) * 0.1)
    generator = torch.Generator(device="cpu").manual_seed(seed + 700)
    matrices = torch.randn(
        program_count,
        register_width,
        register_width,
        generator=generator,
    ) * 0.25
    biases = torch.randn(program_count, register_width, generator=generator) * 0.1
    examples = torch.randn(
        512,
        register_width,
        generator=torch.Generator(device="cpu").manual_seed(seed + 701),
    )
    optimizer = torch.optim.Adam(
        [*machine.parameters(), codes],
        lr=0.01,
    )
    final_loss = torch.tensor(0.0)
    for update in range(updates):
        program = update % program_count
        output = machine.execute_code_chain(
            examples,
            codes[program].view(1, 1, instruction_width).expand(
                examples.shape[0], -1, -1
            ),
        )
        expected = examples @ matrices[program].T + biases[program]
        final_loss = (output - expected).square().mean()
        optimizer.zero_grad(set_to_none=True)
        final_loss.backward()
        optimizer.step()

    for parameter in machine.parameters():
        parameter.requires_grad_(False)
    codes.requires_grad_(False)
    machine.eval()
    memory = ExternalSequenceProgramMemory(
        instruction_width,
        content_addressing=True,
    )
    for program in range(program_count):
        memory.add_artifact(
            ExternalProgramArtifact(
                codes=codes[program : program + 1].detach(),
                interpreter_schema=EXTERNAL_REGISTER_SCHEMA,
                execution_schema=EXTERNAL_REGISTER_READ_EXECUTE_SCHEMA,
            )
        )
    for parameter in memory.parameters():
        parameter.requires_grad_(False)
    return machine, memory, float(final_loss.detach())


@torch.no_grad()
def _execute_program(
    machine: ExternalCapabilityRegisterMachine,
    memory: ExternalSequenceProgramMemory,
    register: torch.Tensor,
    program_index: int,
) -> torch.Tensor:
    codes = memory.program_codes(
        program_index,
        batch_size=register.shape[0],
        device=register.device,
        dtype=register.dtype,
    )
    return machine.execute_code_chain(register, codes)


def _evaluate(
    *,
    router: ExternalOutcomeProgramRouter,
    state: ExternalOutcomeProgramRouterState,
    machine: ExternalCapabilityRegisterMachine,
    memory: ExternalSequenceProgramMemory,
    events: torch.Tensor,
    relation: torch.Tensor,
    phase_count: int,
    tolerance: float,
) -> float:
    scores: list[float] = []
    with torch.no_grad():
        state = router.begin_episode(state)
        for event in events:
            event = event.unsqueeze(0)
            hidden = _hidden_choices(event, relation)
            actual = event.clone()
            expected = event.clone()
            for phase in range(phase_count):
                feature = _phase_feature(event, phase, phase_count)
                selected = int(router.logits(state, feature).argmax(dim=-1).item())
                actual = _execute_program(machine, memory, actual, selected)
                expected = _execute_program(
                    machine,
                    memory,
                    expected,
                    int(hidden[:, phase].item()),
                )
            scores.append(
                float(
                    (actual - expected).abs().amax(dim=-1).le(tolerance).item()
                )
            )
    return sum(scores) / len(scores)


def _stable_prefix(
    progress: list[dict[str, object]],
    threshold: float,
) -> int | None:
    for index, point in enumerate(progress):
        if min(
            float(later["exact_sequence_accuracy"])
            for later in progress[index:]
        ) >= threshold:
            return int(point["episodes"])
    return None


def _train_stream(
    *,
    router: ExternalOutcomeProgramRouter,
    state: ExternalOutcomeProgramRouterState,
    value_baseline: ExternalOutcomeValueBaseline,
    value_state: object,
    machine: ExternalCapabilityRegisterMachine,
    memory: ExternalSequenceProgramMemory,
    events: torch.Tensor,
    relation: torch.Tensor,
    eval_events: torch.Tensor,
    eval_relation: torch.Tensor,
    phase_count: int,
    eval_every: int,
    tolerance: float,
    feedback_override: torch.Tensor | None = None,
) -> tuple[
    ExternalOutcomeProgramRouterState,
    object,
    torch.Tensor,
    list[dict[str, object]],
]:
    if feedback_override is not None and feedback_override.shape != (events.shape[0],):
        raise ValueError("feedback override must have one value per episode")
    outcomes: list[torch.Tensor] = []
    progress: list[dict[str, object]] = []
    with torch.no_grad():
        for index, event in enumerate(events):
            event = event.unsqueeze(0)
            state = router.begin_episode(state)
            value_state = value_baseline.begin_episode(value_state)
            hidden = _hidden_choices(event, relation)
            actual = event.clone()
            for phase in range(phase_count):
                feature = _phase_feature(event, phase, phase_count)
                choice, propensity = router.sample_program(
                    state,
                    feature,
                    exploration=0.1,
                )
                state = router.record_decision(
                    state,
                    feature,
                    choice,
                    propensity,
                )
                _, value_state = value_baseline.record_decision(
                    value_state,
                    feature,
                )
                actual = _execute_program(
                    machine,
                    memory,
                    actual,
                    int(choice.item()),
                )
            expected = event.clone()
            for phase in range(phase_count):
                expected = _execute_program(
                    machine,
                    memory,
                    expected,
                    int(hidden[:, phase].item()),
                )
            outcome = (
                (actual - expected).abs().amax(dim=-1).le(tolerance).to(torch.float32)
            )
            outcomes.append(outcome)
            feedback = (
                outcome
                if feedback_override is None
                else feedback_override[index : index + 1]
            )
            baseline = value_baseline.episode_baseline(value_state)
            state = router.apply_feedback(
                state,
                feedback,
                terminal=torch.ones(1, dtype=torch.bool),
                baseline_override=baseline,
            )
            value_state = value_baseline.apply_feedback(
                value_state,
                feedback,
                terminal=torch.ones(1, dtype=torch.bool),
            )
            if (index + 1) % eval_every == 0:
                progress.append(
                    {
                        "episodes": index + 1,
                        "exact_sequence_accuracy": _evaluate(
                            router=router,
                            state=state,
                            machine=machine,
                            memory=memory,
                            events=eval_events,
                            relation=eval_relation,
                            phase_count=phase_count,
                            tolerance=tolerance,
                        ),
                    }
                )
    return state, value_state, torch.cat(outcomes), progress


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    if min(
        args.source_episodes,
        args.target_episodes,
        args.evaluation_episodes,
        args.eval_every,
        args.event_width,
        args.register_width,
        args.instruction_width,
        args.phases,
        args.interpreter_updates,
    ) < 1:
        raise ValueError("episode, dimension, and update counts must be positive")
    if not 0.0 < args.mastery_threshold <= 1.0:
        raise ValueError("mastery threshold must lie in (0, 1]")
    if args.phases < 2:
        raise ValueError("at least two executable program phases are required")
    if args.program_capacity != 3:
        raise ValueError("this rung requires exactly three program candidates")
    if not 0.0 < args.tolerance:
        raise ValueError("execution tolerance must be positive")

    torch.manual_seed(args.seed)
    machine, memory, interpreter_loss = _train_executable_bank(
        seed=args.seed + 40_000,
        program_count=args.program_capacity,
        event_width=args.event_width,
        register_width=args.register_width,
        instruction_width=args.instruction_width,
        updates=args.interpreter_updates,
    )
    machine_digest_before = _digest(machine)
    memory_digest_before = _digest(memory)

    generator = torch.Generator(device="cpu").manual_seed(args.seed + 10_001)
    source_events = torch.randn(
        args.source_episodes + args.evaluation_episodes,
        args.event_width,
        generator=generator,
    )
    target_events = torch.randn(
        args.target_episodes + args.evaluation_episodes,
        args.event_width,
        generator=generator,
    )
    source_relation = torch.randn(
        args.phases,
        2,
        args.event_width,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 20_001),
    )
    target_relation = torch.randn(
        args.phases,
        args.program_capacity,
        args.event_width,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 20_002),
    )
    source_train = source_events[: args.source_episodes]
    source_eval = source_events[args.source_episodes :]
    target_train = target_events[: args.target_episodes]
    target_eval = target_events[args.target_episodes :]
    feature_width = args.event_width * args.phases + args.phases

    router = ExternalOutcomeProgramRouter(
        feature_width,
        args.program_capacity,
        initial_programs=2,
        initial_learning_rate=args.learning_rate,
        initial_trace_decay=args.trace_decay,
        initial_baseline_rate=args.baseline_rate,
    )
    value_baseline = ExternalOutcomeValueBaseline(
        feature_width,
        initial_learning_rate=args.value_learning_rate,
        initial_trace_decay=args.value_trace_decay,
    )
    router_digest_before = _digest(router)
    value_digest_before = _digest(value_baseline)

    source_state = router.initial_state(1)
    source_value_state = value_baseline.initial_state(1)
    source_state, source_value_state, _, source_progress = _train_stream(
        router=router,
        state=source_state,
        value_baseline=value_baseline,
        value_state=source_value_state,
        machine=machine,
        memory=memory,
        events=source_train,
        relation=source_relation,
        eval_events=source_eval,
        eval_relation=source_relation,
        phase_count=args.phases,
        eval_every=args.eval_every,
        tolerance=args.tolerance,
    )
    source_before = _evaluate(
        router=router,
        state=source_state,
        machine=machine,
        memory=memory,
        events=source_eval,
        relation=source_relation,
        phase_count=args.phases,
        tolerance=args.tolerance,
    )
    source_state_digest_before_target = _state_digest(router, source_state)

    target_state = router.initial_state(1)
    target_state = router.append_program(target_state)
    target_value_state = value_baseline.initial_state(1)
    target_state, target_value_state, target_outcomes, target_progress = _train_stream(
        router=router,
        state=target_state,
        value_baseline=value_baseline,
        value_state=target_value_state,
        machine=machine,
        memory=memory,
        events=target_train,
        relation=target_relation,
        eval_events=target_eval,
        eval_relation=target_relation,
        phase_count=args.phases,
        eval_every=args.eval_every,
        tolerance=args.tolerance,
    )
    target_final = _evaluate(
        router=router,
        state=target_state,
        machine=machine,
        memory=memory,
        events=target_eval,
        relation=target_relation,
        phase_count=args.phases,
        tolerance=args.tolerance,
    )

    no_trace_router = ExternalOutcomeProgramRouter(
        feature_width,
        args.program_capacity,
        initial_programs=2,
        initial_learning_rate=args.learning_rate,
        initial_trace_decay=0.0,
        initial_baseline_rate=args.baseline_rate,
    )
    no_trace_state = no_trace_router.append_program(no_trace_router.initial_state(1))
    no_trace_value = value_baseline.initial_state(1)
    no_trace_state, no_trace_value, _, no_trace_progress = _train_stream(
        router=no_trace_router,
        state=no_trace_state,
        value_baseline=value_baseline,
        value_state=no_trace_value,
        machine=machine,
        memory=memory,
        events=target_train,
        relation=target_relation,
        eval_events=target_eval,
        eval_relation=target_relation,
        phase_count=args.phases,
        eval_every=args.eval_every,
        tolerance=args.tolerance,
    )
    no_trace_final = _evaluate(
        router=no_trace_router,
        state=no_trace_state,
        machine=machine,
        memory=memory,
        events=target_eval,
        relation=target_relation,
        phase_count=args.phases,
        tolerance=args.tolerance,
    )

    permutation = torch.randperm(
        target_outcomes.shape[0],
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 30_001),
    )
    shuffled_router = ExternalOutcomeProgramRouter(
        feature_width,
        args.program_capacity,
        initial_programs=2,
        initial_learning_rate=args.learning_rate,
        initial_trace_decay=args.trace_decay,
        initial_baseline_rate=args.baseline_rate,
    )
    shuffled_state = shuffled_router.append_program(shuffled_router.initial_state(1))
    shuffled_value = value_baseline.initial_state(1)
    shuffled_state, shuffled_value, _, shuffled_progress = _train_stream(
        router=shuffled_router,
        state=shuffled_state,
        value_baseline=value_baseline,
        value_state=shuffled_value,
        machine=machine,
        memory=memory,
        events=target_train,
        relation=target_relation,
        eval_events=target_eval,
        eval_relation=target_relation,
        phase_count=args.phases,
        eval_every=args.eval_every,
        tolerance=args.tolerance,
        feedback_override=target_outcomes[permutation],
    )
    shuffled_final = _evaluate(
        router=shuffled_router,
        state=shuffled_state,
        machine=machine,
        memory=memory,
        events=target_eval,
        relation=target_relation,
        phase_count=args.phases,
        tolerance=args.tolerance,
    )

    capacity_router = ExternalOutcomeProgramRouter(
        feature_width,
        args.program_capacity,
        initial_programs=2,
        initial_learning_rate=args.learning_rate,
        initial_trace_decay=args.trace_decay,
        initial_baseline_rate=args.baseline_rate,
    )
    capacity_state = capacity_router.initial_state(1)
    capacity_value = value_baseline.initial_state(1)
    capacity_state, capacity_value, _, capacity_progress = _train_stream(
        router=capacity_router,
        state=capacity_state,
        value_baseline=value_baseline,
        value_state=capacity_value,
        machine=machine,
        memory=memory,
        events=target_train,
        relation=target_relation,
        eval_events=target_eval,
        eval_relation=target_relation,
        phase_count=args.phases,
        eval_every=args.eval_every,
        tolerance=args.tolerance,
    )
    capacity_final = _evaluate(
        router=capacity_router,
        state=capacity_state,
        machine=machine,
        memory=memory,
        events=target_eval,
        relation=target_relation,
        phase_count=args.phases,
        tolerance=args.tolerance,
    )

    source_after = _evaluate(
        router=router,
        state=source_state,
        machine=machine,
        memory=memory,
        events=source_eval,
        relation=source_relation,
        phase_count=args.phases,
        tolerance=args.tolerance,
    )
    source_state_unchanged = (
        source_state_digest_before_target == _state_digest(router, source_state)
    )
    probe_feature = _phase_feature(target_train[:1], 0, args.phases)
    probe_state = router.record_decision(
        router.begin_episode(target_state),
        probe_feature,
        torch.zeros(1, dtype=torch.long),
        torch.ones(1),
    )
    missing_state = router.apply_feedback(
        probe_state,
        torch.ones(1),
        present=torch.zeros(1, dtype=torch.bool),
        terminal=torch.ones(1, dtype=torch.bool),
    )
    missing_feedback_no_write = all(
        torch.equal(
            getattr(missing_state.credit, name),
            getattr(probe_state.credit, name),
        )
        for name in ("policy", "eligibility", "baseline", "feedbacks")
    ) and missing_state.active_programs == probe_state.active_programs

    restored_router_state = router.state_from_payload(
        router.state_payload(target_state)
    )
    persistence_exact = (
        restored_router_state.active_programs == target_state.active_programs
        and torch.equal(
            restored_router_state.credit.policy,
            target_state.credit.policy,
        )
        and torch.equal(
            restored_router_state.credit.eligibility,
            target_state.credit.eligibility,
        )
    )
    restored_artifacts = [
        ExternalProgramArtifact.from_payload(memory.artifact(index).payload())
        for index in range(args.program_capacity)
    ]
    artifact_persistence_exact = all(
        restored.digest() == memory.artifact(index).digest()
        for index, restored in enumerate(restored_artifacts)
    )
    machine_digest_after = _digest(machine)
    memory_digest_after = _digest(memory)
    router_digest_after = _digest(router)
    value_digest_after = _digest(value_baseline)

    inherited_stable = _stable_prefix(target_progress, args.mastery_threshold)
    no_trace_stable = _stable_prefix(no_trace_progress, args.mastery_threshold)
    shuffled_stable = _stable_prefix(shuffled_progress, args.mastery_threshold)
    capacity_stable = _stable_prefix(capacity_progress, args.mastery_threshold)
    gates = {
        "source_mastery": source_before >= args.mastery_threshold,
        "source_retention": source_after >= args.mastery_threshold,
        "source_state_unchanged": source_state_unchanged,
        "target_mastery": target_final >= args.mastery_threshold,
        "target_stable": inherited_stable is not None,
        "appended_program_used": bool(
            (_hidden_choices(target_eval, target_relation) == 2).any()
        ),
        "no_trace_control_rejected": no_trace_final < 0.75,
        "reward_shuffled_control_rejected": shuffled_final < 0.75,
        "capacity_control_rejected": capacity_final < 0.75,
        "missing_feedback_no_write": missing_feedback_no_write,
        "router_persistence_exact": persistence_exact,
        "artifact_persistence_exact": artifact_persistence_exact,
        "executor_frozen": machine_digest_before == machine_digest_after,
        "program_memory_frozen": memory_digest_before == memory_digest_after,
        "router_rule_frozen": router_digest_before == router_digest_after,
        "value_baseline_rule_frozen": value_digest_before == value_digest_after,
        "no_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-outcome-program-router-pressure-test.v1",
        "claim_boundary": (
            "A memory-side outcome-credit router acquires a new multi-step "
            "capability by selecting pre-admitted opaque executable programs "
            "with one terminal scalar outcome; this is not program induction "
            "or general continual learning."
        ),
        "seed": args.seed,
        "source_episodes": args.source_episodes,
        "target_episodes": args.target_episodes,
        "evaluation_episodes": args.evaluation_episodes,
        "phase_count": args.phases,
        "event_width": args.event_width,
        "feature_width": feature_width,
        "program_capacity": args.program_capacity,
        "initial_programs": 2,
        "interpreter_pretraining_updates": args.interpreter_updates,
        "interpreter_final_loss": interpreter_loss,
        "mastery_threshold": args.mastery_threshold,
        "source_before": source_before,
        "source_after": source_after,
        "source_progress": source_progress,
        "target_final": target_final,
        "no_trace_final": no_trace_final,
        "reward_shuffled_final": shuffled_final,
        "capacity_control_final": capacity_final,
        "inherited_stable_episodes": inherited_stable,
        "no_trace_stable_episodes": no_trace_stable,
        "reward_shuffled_stable_episodes": shuffled_stable,
        "capacity_stable_episodes": capacity_stable,
        "source_state_unchanged": source_state_unchanged,
        "missing_feedback_no_write": missing_feedback_no_write,
        "router_persistence_exact": persistence_exact,
        "artifact_persistence_exact": artifact_persistence_exact,
        "executor_frozen": machine_digest_before == machine_digest_after,
        "program_memory_frozen": memory_digest_before == memory_digest_after,
        "router_rule_frozen": router_digest_before == router_digest_after,
        "value_baseline_rule_frozen": value_digest_before == value_digest_after,
        "accounting": {
            "unique_verifier_bits": args.source_episodes + args.target_episodes,
            "unique_logical_lifetimes": args.source_episodes + args.target_episodes,
            "external_route_decision_updates": args.phases
            * (args.source_episodes + args.target_episodes),
            "external_feedback_updates": args.source_episodes + args.target_episodes,
            "interpreter_optimizer_updates": args.interpreter_updates,
            "router_optimizer_updates": 0,
            "replayed_examples": 0,
            "paired_no_trace_control_lifetimes": args.target_episodes,
            "paired_reward_shuffled_control_lifetimes": args.target_episodes,
            "paired_capacity_control_lifetimes": args.target_episodes,
            "stable_bits_to_threshold": inherited_stable,
        },
        "gates": gates,
        "promoted": all(gates.values()),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--source-episodes", type=int, default=2000)
    parser.add_argument("--target-episodes", type=int, default=7000)
    parser.add_argument("--evaluation-episodes", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--phases", type=int, default=2)
    parser.add_argument("--event-width", type=int, default=4)
    parser.add_argument("--register-width", type=int, default=4)
    parser.add_argument("--instruction-width", type=int, default=8)
    parser.add_argument("--program-capacity", type=int, default=3)
    parser.add_argument("--interpreter-updates", type=int, default=900)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--trace-decay", type=float, default=0.95)
    parser.add_argument("--baseline-rate", type=float, default=0.02)
    parser.add_argument("--value-learning-rate", type=float, default=0.05)
    parser.add_argument("--value-trace-decay", type=float, default=0.90)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--mastery-threshold", type=float, default=0.90)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
