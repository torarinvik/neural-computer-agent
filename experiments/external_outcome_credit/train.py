"""Pressure-test external eligibility traces on learned multi-step relations.

The learner receives one standardized event tensor, samples opaque choices,
logs their exact behavior propensities, and receives only one terminal scalar
verifier outcome.  A hidden verifier relation maps the event to the choice
sequence; its parameters and labels never enter the learner.

The main comparison is a matched external state with eligibility traces versus
a fresh state with the trace disabled.  A reward-shuffled control receives the
same scalar outcomes in the wrong temporal order.  The controller analogue is
fully frozen because this experiment contains only the memory-side policy
state and plasticity rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from neural_computer import (
    ExternalOutcomeCreditPlasticity,
    ExternalOutcomeCreditState,
)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_digest(state: ExternalOutcomeCreditState) -> str:
    digest = hashlib.sha256()
    for value in (
        state.policy,
        state.eligibility,
        state.baseline,
        state.decisions,
        state.feedbacks,
    ):
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _phase_feature(
    event: torch.Tensor,
    phase: int,
    phase_count: int,
) -> torch.Tensor:
    if event.ndim != 2:
        raise ValueError("events must have shape [batch, width]")
    if phase_count < 2 or phase not in range(phase_count):
        raise ValueError("phase must be inside a sequence of at least two phases")
    event_blocks = [torch.zeros_like(event) for _ in range(phase_count)]
    event_blocks[phase] = event
    phase_token = torch.zeros(
        event.shape[0],
        phase_count,
        device=event.device,
        dtype=event.dtype,
    )
    phase_token[:, phase] = 1.0
    return torch.cat((*event_blocks, phase_token), dim=-1)


def _hidden_target(
    event: torch.Tensor,
    relation: torch.Tensor,
) -> torch.Tensor:
    """Verifier-private two-choice target; never passed to the learner."""

    return (event @ relation.T > 0.0).to(torch.long)


def _sample_choice(
    rule: ExternalOutcomeCreditPlasticity,
    state: ExternalOutcomeCreditState,
    feature: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    probabilities = rule.logits(state, feature).softmax(dim=-1)
    behavior = 0.9 * probabilities + 0.05
    choice = torch.multinomial(behavior, 1).squeeze(-1)
    propensity = behavior.gather(1, choice.unsqueeze(-1)).squeeze(-1)
    return choice, propensity


def _train_stream(
    rule: ExternalOutcomeCreditPlasticity,
    state: ExternalOutcomeCreditState,
    events: torch.Tensor,
    relation: torch.Tensor,
    *,
    eval_events: torch.Tensor,
    eval_relation: torch.Tensor,
    eval_every: int,
    phase_count: int,
    feedback_override: torch.Tensor | None = None,
) -> tuple[
    ExternalOutcomeCreditState,
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
            state = rule.begin_episode(state)
            choices: list[torch.Tensor] = []
            for phase in range(phase_count):
                feature = _phase_feature(event, phase, phase_count)
                choice, propensity = _sample_choice(rule, state, feature)
                state = rule.record_decision(
                    state,
                    feature,
                    choice,
                    propensity,
                )
                choices.append(choice)
            chosen = torch.stack(choices, dim=1)
            target = _hidden_target(event, relation)
            outcome = (chosen == target).all(dim=1).to(torch.float32)
            outcomes.append(outcome)
            feedback = (
                outcome
                if feedback_override is None
                else feedback_override[index : index + 1]
            )
            state = rule.apply_feedback(
                state,
                feedback,
                terminal=torch.ones(1, dtype=torch.bool),
            )
            if (index + 1) % eval_every == 0:
                exact, phase_scores = _evaluate(
                    rule,
                    state,
                    eval_events,
                    eval_relation,
                    phase_count,
                )
                progress.append(
                    {
                        "episodes": index + 1,
                        "exact_sequence_accuracy": exact,
                        "phase_accuracy": phase_scores,
                    }
                )
    return state, torch.cat(outcomes), progress


def _evaluate(
    rule: ExternalOutcomeCreditPlasticity,
    state: ExternalOutcomeCreditState,
    events: torch.Tensor,
    relation: torch.Tensor,
    phase_count: int,
) -> tuple[float, list[float]]:
    exact: list[float] = []
    phase_scores = [[] for _ in range(phase_count)]
    with torch.no_grad():
        state = rule.begin_episode(state)
        for event in events:
            event = event.unsqueeze(0)
            choices: list[torch.Tensor] = []
            target = _hidden_target(event, relation)
            for phase in range(phase_count):
                feature = _phase_feature(event, phase, phase_count)
                choice = rule.logits(state, feature).argmax(dim=-1)
                choices.append(choice)
                phase_scores[phase].append(float((choice == target[:, phase]).item()))
            chosen = torch.stack(choices, dim=1)
            exact.append(float((chosen == target).all().item()))
    return sum(exact) / len(exact), [
        sum(scores) / len(scores) for scores in phase_scores
    ]


def _stable_episode_prefix(
    progress: list[dict[str, object]],
    threshold: float,
) -> int | None:
    for index, point in enumerate(progress):
        scores = [
            float(later["exact_sequence_accuracy"])
            for later in progress[index:]
        ]
        if min(scores) >= threshold:
            return int(point["episodes"])
    return None


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    if min(
        args.source_episodes,
        args.target_episodes,
        args.evaluation_episodes,
        args.eval_every,
        args.event_width,
        args.phases,
    ) < 1:
        raise ValueError("episode counts, eval interval, and width must be positive")
    if not 0.0 < args.mastery_threshold <= 1.0:
        raise ValueError("mastery threshold must lie in (0, 1]")
    if args.phases < 2:
        raise ValueError("at least two delayed-credit phases are required")
    torch.manual_seed(args.seed)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 10_001)
    relation_source = torch.randn(args.phases, args.event_width, generator=generator)
    relation_target = torch.randn(args.phases, args.event_width, generator=generator)
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
    source_train = source_events[: args.source_episodes]
    source_eval = source_events[args.source_episodes :]
    target_train = target_events[: args.target_episodes]
    target_eval = target_events[args.target_episodes :]
    feature_width = args.event_width * args.phases + args.phases

    rule = ExternalOutcomeCreditPlasticity(
        feature_width,
        2,
        initial_learning_rate=args.learning_rate,
        initial_trace_decay=args.trace_decay,
        initial_baseline_rate=args.baseline_rate,
    )
    rule_digest_before = _digest(rule)
    source_state, _, source_progress = _train_stream(
        rule,
        rule.initial_state(1),
        source_train,
        relation_source,
        eval_events=source_eval,
        eval_relation=relation_source,
        eval_every=args.eval_every,
        phase_count=args.phases,
    )
    source_before, source_phase_before = _evaluate(
        rule, source_state, source_eval, relation_source, args.phases
    )

    target_state, target_outcomes, target_progress = _train_stream(
        rule,
        rule.initial_state(1),
        target_train,
        relation_target,
        eval_events=target_eval,
        eval_relation=relation_target,
        eval_every=args.eval_every,
        phase_count=args.phases,
    )
    no_trace_rule = ExternalOutcomeCreditPlasticity(
        feature_width,
        2,
        initial_learning_rate=args.learning_rate,
        initial_trace_decay=0.0,
        initial_baseline_rate=args.baseline_rate,
    )
    no_trace_state, _, no_trace_progress = _train_stream(
        no_trace_rule,
        no_trace_rule.initial_state(1),
        target_train,
        relation_target,
        eval_events=target_eval,
        eval_relation=relation_target,
        eval_every=args.eval_every,
        phase_count=args.phases,
    )
    permutation = torch.randperm(
        target_outcomes.shape[0], generator=torch.Generator().manual_seed(args.seed + 20_002)
    )
    shuffled_outcomes = target_outcomes[permutation]
    shuffled_state, _, shuffled_progress = _train_stream(
        rule,
        rule.initial_state(1),
        target_train,
        relation_target,
        eval_events=target_eval,
        eval_relation=relation_target,
        eval_every=args.eval_every,
        phase_count=args.phases,
        feedback_override=shuffled_outcomes,
    )

    source_after, source_phase_after = _evaluate(
        rule, source_state, source_eval, relation_source, args.phases
    )
    target_final, target_phase_final = _evaluate(
        rule, target_state, target_eval, relation_target, args.phases
    )
    no_trace_final, no_trace_phase_final = _evaluate(
        no_trace_rule, no_trace_state, target_eval, relation_target, args.phases
    )
    shuffled_final, shuffled_phase_final = _evaluate(
        rule, shuffled_state, target_eval, relation_target, args.phases
    )

    with torch.no_grad():
        probe_feature = _phase_feature(target_train[:1], 0, args.phases)
        probe_state = rule.record_decision(
            rule.begin_episode(target_state),
            probe_feature,
            torch.zeros(1, dtype=torch.long),
            torch.ones(1),
        )
        missing_feedback = rule.apply_feedback(
            probe_state,
            torch.ones(1),
            present=torch.zeros(1, dtype=torch.bool),
            terminal=torch.ones(1, dtype=torch.bool),
        )
    missing_feedback_no_write = all(
        torch.equal(getattr(missing_feedback, name), getattr(probe_state, name))
        for name in ("policy", "eligibility", "baseline", "feedbacks")
    )
    payload = rule.state_payload(target_state)
    restored = rule.state_from_payload(payload)
    persistence_exact = all(
        torch.equal(getattr(restored, name), getattr(target_state, name))
        for name in ("policy", "eligibility", "baseline", "decisions", "feedbacks")
    )
    rule_digest_after = _digest(rule)
    inherited_stable = _stable_episode_prefix(
        target_progress, args.mastery_threshold
    )
    no_trace_stable = _stable_episode_prefix(
        no_trace_progress, args.mastery_threshold
    )
    shuffled_stable = _stable_episode_prefix(
        shuffled_progress, args.mastery_threshold
    )
    source_retention = min(source_before, source_after)
    gates = {
        "source_mastery": source_before >= args.mastery_threshold,
        "source_retention": source_after >= args.mastery_threshold,
        "delayed_credit_mastery": target_final >= args.mastery_threshold,
        "delayed_credit_stable": inherited_stable is not None,
        "no_trace_control_rejected": no_trace_final < 0.75,
        "reward_shuffled_control_rejected": shuffled_final < 0.75,
        "missing_feedback_no_write": missing_feedback_no_write,
        "persistence_exact": persistence_exact,
        "plasticity_rule_frozen": rule_digest_before == rule_digest_after,
        "no_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-outcome-credit-pressure-test.v1",
        "claim_boundary": (
            "An external eligibility-trace policy learns a hidden multi-step "
            "event-to-choice relation from terminal scalar outcomes without "
            "controller updates or replay. This is a bounded causal-credit "
            "primitive, not general continual learning or arbitrary program "
            "induction."
        ),
        "seed": args.seed,
        "source_episodes": args.source_episodes,
        "target_episodes": args.target_episodes,
        "evaluation_episodes": args.evaluation_episodes,
        "phase_count": args.phases,
        "feature_width": feature_width,
        "trace_decay": float(rule.trace_decay.detach()),
        "mastery_threshold": args.mastery_threshold,
        "source_progress": source_progress,
        "target_progress": target_progress,
        "no_trace_progress": no_trace_progress,
        "reward_shuffled_progress": shuffled_progress,
        "source_before": source_before,
        "source_after": source_after,
        "source_phase_before": source_phase_before,
        "source_phase_after": source_phase_after,
        "target_final": target_final,
        "target_phase_final": target_phase_final,
        "no_trace_final": no_trace_final,
        "no_trace_phase_final": no_trace_phase_final,
        "reward_shuffled_final": shuffled_final,
        "reward_shuffled_phase_final": shuffled_phase_final,
        "inherited_stable_episodes": inherited_stable,
        "no_trace_stable_episodes": no_trace_stable,
        "reward_shuffled_stable_episodes": shuffled_stable,
        "source_retention_floor": source_retention,
        "missing_feedback_no_write": missing_feedback_no_write,
        "persistence_exact": persistence_exact,
        "plasticity_rule_frozen": rule_digest_before == rule_digest_after,
        "accounting": {
            "unique_verifier_bits": args.source_episodes + args.target_episodes,
            "unique_logical_lifetimes": args.source_episodes + args.target_episodes,
            "external_decision_updates": args.phases * (
                args.source_episodes + args.target_episodes
            ),
            "external_feedback_updates": args.source_episodes + args.target_episodes,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "paired_no_trace_control_lifetimes": args.target_episodes,
            "paired_reward_shuffled_control_lifetimes": args.target_episodes,
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
    parser.add_argument("--target-episodes", type=int, default=5000)
    parser.add_argument("--evaluation-episodes", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--phases", type=int, default=2)
    parser.add_argument("--event-width", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--trace-decay", type=float, default=0.95)
    parser.add_argument("--baseline-rate", type=float, default=0.02)
    parser.add_argument("--mastery-threshold", type=float, default=0.90)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
