"""Audit transfer from a trained intention adapter into fresh external state.

This is deliberately a narrow continual-learning pressure test.  A shared
memory-side adapter is trained once from unique opaque action logs, then
frozen.  New capabilities receive fresh fast-weight state and only positive
outcomes; they must use the inherited adapter without replaying the source
stream.  A matched fresh adapter sees the same target stream and is trained
online as the control.

The target is an interface-level association, not a game task.  The action
record is learner-visible and is used only as the opaque value written to
external state and as the post-outcome verifier for the intention residual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import ExternalFastWeightCapabilityProgram, IntentEvent


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_digest(program: ExternalFastWeightCapabilityProgram, state) -> str:
    payload = program.fast_weight.state_payload(state)
    digest = hashlib.sha256()
    digest.update(payload["weights"].numpy().tobytes())
    digest.update(payload["updates"].numpy().tobytes())
    return digest.hexdigest()


def _cosine_score(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = F.normalize(actual, dim=-1, eps=1e-8)
    expected = F.normalize(expected, dim=-1, eps=1e-8)
    return float((actual * expected).sum(dim=-1).mean().detach())


def _stable_prefix(scores: list[float], threshold: float) -> int | None:
    for index in range(len(scores)):
        if min(scores[index:]) >= threshold:
            return index + 1
    return None


def _make_program(args: argparse.Namespace) -> ExternalFastWeightCapabilityProgram:
    return ExternalFastWeightCapabilityProgram(
        event_width=args.event_width,
        action_width=args.action_width,
        intention_width=args.intention_width,
        key_width=args.key_width,
        query_hidden=args.query_hidden,
        fast_weight_hidden=args.fast_weight_hidden,
    )


def _stream(
    *,
    count: int,
    event_width: int,
    action_width: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    events = torch.randn(count, event_width, generator=generator)
    actions = F.normalize(
        torch.randn(count, action_width, generator=generator), dim=-1
    )
    return events, actions


def _write_and_read(
    program: ExternalFastWeightCapabilityProgram,
    event: torch.Tensor,
    action: torch.Tensor,
) -> tuple[torch.Tensor, object]:
    """Write one successful lifetime, then read the old external state."""

    intention = IntentEvent(torch.zeros(event.shape[0], program.intention_width))
    state = program.initial_state(event.shape[0], device=event.device)
    with torch.no_grad():
        query = program._query(event, intention)
        state = program.fast_weight.update(
            state,
            query,
            action,
            torch.ones(event.shape[0]),
        )
        memory_value = program.fast_weight.read(state, query)
    adapted = program.intent_adapter(memory_value)
    return adapted, state


def _read_existing(
    program: ExternalFastWeightCapabilityProgram,
    event: torch.Tensor,
    state,
) -> torch.Tensor:
    intention = IntentEvent(torch.zeros(event.shape[0], program.intention_width))
    with torch.no_grad():
        query = program._query(event, intention)
        memory_value = program.fast_weight.read(state, query)
    return program.intent_adapter(memory_value)


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_examples,
        args.target_examples,
        args.event_width,
        args.action_width,
        args.intention_width,
        args.key_width,
        args.query_hidden,
        args.fast_weight_hidden,
    ) < 1:
        raise ValueError("all dimensions and stream lengths must be positive")
    if args.action_width != args.intention_width:
        raise ValueError(
            "this interface audit requires equal action and intention widths"
        )
    torch.manual_seed(args.seed)

    source_events, source_actions = _stream(
        count=args.source_examples,
        event_width=args.event_width,
        action_width=args.action_width,
        seed=args.seed + 101,
    )
    target_events, target_actions = _stream(
        count=args.target_examples,
        event_width=args.event_width,
        action_width=args.action_width,
        seed=args.seed + 202,
    )

    inherited = _make_program(args)
    optimizer = torch.optim.AdamW(
        inherited.intent_adapter.parameters(),
        lr=args.source_learning_rate,
        weight_decay=1e-5,
    )
    source_states: list[tuple[torch.Tensor, object, torch.Tensor]] = []
    source_training_losses: list[float] = []
    for index in range(args.source_examples):
        event = source_events[index : index + 1]
        action = source_actions[index : index + 1]
        adapted, state = _write_and_read(inherited, event, action)
        loss = F.mse_loss(adapted, action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inherited.intent_adapter.parameters(), 1.0)
        optimizer.step()
        source_training_losses.append(float(loss.detach()))
        source_states.append((event, state, action))

    inherited.eval()
    for parameter in inherited.parameters():
        parameter.requires_grad_(False)
    frozen_digest_before_target = _digest(inherited)

    inherited_target_scores: list[float] = []
    for index in range(args.target_examples):
        event = target_events[index : index + 1]
        action = target_actions[index : index + 1]
        adapted, _ = _write_and_read(inherited, event, action)
        inherited_target_scores.append(_cosine_score(adapted, action))

    source_retention_scores = [
        _cosine_score(_read_existing(inherited, event, state).detach(), action)
        for event, state, action in source_states
    ]

    fresh = _make_program(args)
    fresh.eval()
    fresh_optimizer = torch.optim.AdamW(
        fresh.intent_adapter.parameters(),
        lr=args.target_learning_rate,
        weight_decay=1e-5,
    )
    fresh_target_scores: list[float] = []
    fresh_training_losses: list[float] = []
    for index in range(args.target_examples):
        event = target_events[index : index + 1]
        action = target_actions[index : index + 1]
        adapted, _ = _write_and_read(fresh, event, action)
        fresh_target_scores.append(_cosine_score(adapted, action))
        loss = F.mse_loss(adapted, action)
        fresh_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fresh.intent_adapter.parameters(), 1.0)
        fresh_optimizer.step()
        fresh_training_losses.append(float(loss.detach()))

    # Failure and missing-evidence controls use an already learned source
    # state.  They must not alter its computation, even though observation
    # accounting may still advance the external update counter.
    control_event, control_state, control_action = source_states[0]
    control_intention = IntentEvent(
        torch.zeros(control_event.shape[0], inherited.intention_width)
    )
    with torch.no_grad():
        control_query = inherited._query(control_event, control_intention)
        failed_state = inherited.fast_weight.update(
            control_state,
            control_query,
            -control_action,
            torch.zeros(1),
        )
        missing_state = inherited.fast_weight.update(
            control_state,
            control_query,
            -control_action,
            torch.ones(1),
            present=torch.zeros(1, dtype=torch.bool),
        )
    failed_no_write = torch.equal(failed_state.weights, control_state.weights)
    missing_no_write = torch.equal(missing_state.weights, control_state.weights)

    payload = inherited.fast_weight.state_payload(control_state)
    restored = inherited.fast_weight.state_from_payload(payload)
    persistence_exact = (
        torch.equal(restored.weights, control_state.weights)
        and torch.equal(restored.updates, control_state.updates)
    )
    frozen_digest_after_target = _digest(inherited)
    threshold = args.threshold
    inherited_stable = _stable_prefix(inherited_target_scores, threshold)
    fresh_stable = _stable_prefix(fresh_target_scores, threshold)
    source_retention = min(source_retention_scores)
    inherited_target_score = min(inherited_target_scores)
    fresh_target_score = min(fresh_target_scores)
    positive_transfer = (
        inherited_stable is not None
        and fresh_stable is not None
        and inherited_stable < fresh_stable
        and inherited_target_score >= threshold
        and source_retention >= threshold
    )
    report = {
        "schema": "neural-computer.external-fast-capability-transfer.v1",
        "claim_boundary": (
            "A source-trained shared intention adapter transfers across fresh "
            "opaque capability state without target optimizer updates or source "
            "replay. This is interface-prior transfer, not general continual "
            "learning or unrestricted memory growth."
        ),
        "seed": args.seed,
        "source_examples": args.source_examples,
        "target_examples": args.target_examples,
        "threshold": threshold,
        "source_training_losses": source_training_losses,
        "inherited_target_scores": inherited_target_scores,
        "fresh_control_target_scores": fresh_target_scores,
        "fresh_control_training_losses": fresh_training_losses,
        "source_retention_scores": source_retention_scores,
        "source_retention_floor": source_retention,
        "inherited_target_floor": inherited_target_score,
        "fresh_control_target_floor": fresh_target_score,
        "inherited_stable_examples": inherited_stable,
        "fresh_control_stable_examples": fresh_stable,
        "positive_transfer": positive_transfer,
        "failed_outcome_no_write": failed_no_write,
        "missing_evidence_no_write": missing_no_write,
        "persistence_exact": persistence_exact,
        "inherited_program_frozen": (
            frozen_digest_before_target == frozen_digest_after_target
        ),
        "inherited_target_optimizer_updates": 0,
        "source_optimizer_updates": args.source_examples,
        "fresh_control_optimizer_updates": args.target_examples,
        "replayed_examples": 0,
        "accounting": {
            "unique_verifier_bits": args.source_examples + args.target_examples,
            "unique_logical_lifetimes": args.source_examples + args.target_examples,
            "paired_fresh_control_lifetimes": args.target_examples,
            "stable_bits_to_inherited_threshold": inherited_stable,
            "stable_bits_to_fresh_control_threshold": fresh_stable,
        },
        "promoted": bool(
            positive_transfer
            and failed_no_write
            and missing_no_write
            and persistence_exact
            and frozen_digest_before_target == frozen_digest_after_target
        ),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--source-examples", type=int, default=64)
    parser.add_argument("--target-examples", type=int, default=16)
    parser.add_argument("--event-width", type=int, default=4)
    parser.add_argument("--action-width", type=int, default=2)
    parser.add_argument("--intention-width", type=int, default=2)
    parser.add_argument("--key-width", type=int, default=8)
    parser.add_argument("--query-hidden", type=int, default=16)
    parser.add_argument("--fast-weight-hidden", type=int, default=8)
    parser.add_argument("--source-learning-rate", type=float, default=0.12)
    parser.add_argument("--target-learning-rate", type=float, default=0.12)
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
