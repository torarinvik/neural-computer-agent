"""Audit adapter transfer into fresh per-file executable-cell state.

The external program cell's query/value path is trained once on a source
stream.  It is then frozen while a fresh target cell state learns from new
opaque action/outcome records.  A matched fresh cell trains its adapter on the
same target stream.  This is an interface-prior pressure test for the new
execution-context seam, not a claim of arbitrary new computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from neural_computer import ExternalProgramFastCell, IntentEvent

EVENT_WIDTH = 8
ACTION_WIDTH = 4
INTENTION_WIDTH = 6
REGISTER_WIDTH = ACTION_WIDTH
KEY_WIDTH = 12
CELL_HIDDEN = 24
ADAPTER_HIDDEN = 24
FAST_WEIGHT_HIDDEN = 16
MASTERY_THRESHOLD = 0.95


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(repr(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _make_cell() -> ExternalProgramFastCell:
    return ExternalProgramFastCell(
        EVENT_WIDTH,
        ACTION_WIDTH,
        INTENTION_WIDTH,
        REGISTER_WIDTH,
        key_width=KEY_WIDTH,
        query_hidden=CELL_HIDDEN,
        adapter_hidden=ADAPTER_HIDDEN,
        fast_weight_hidden=FAST_WEIGHT_HIDDEN,
    )


def _stream(
    count: int,
    seed: int,
    action_codebook: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    events = torch.randn(count, EVENT_WIDTH, generator=generator)
    indices = (torch.arange(count) + seed) % action_codebook.shape[0]
    actions = action_codebook[indices]
    return events, actions


def _intention(batch_size: int) -> IntentEvent:
    return IntentEvent(torch.zeros(batch_size, INTENTION_WIDTH))


def _write(
    cell: ExternalProgramFastCell,
    event: torch.Tensor,
    action: torch.Tensor,
    *,
    outcome: torch.Tensor,
    present: torch.Tensor | None = None,
):
    state = cell.initial_state(event.shape[0], device=event.device)
    with torch.no_grad():
        _context, state = cell.step(
            event=event,
            action=action,
            outcome=outcome,
            intention=_intention(event.shape[0]),
            state=state,
            present=present,
        )
    return state


def _read(
    cell: ExternalProgramFastCell,
    state,
    event: torch.Tensor,
) -> torch.Tensor:
    return cell.read(state, event, _intention(event.shape[0]))


def _score(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = F.normalize(actual, dim=-1, eps=1e-8)
    expected = F.normalize(expected, dim=-1, eps=1e-8)
    return float((actual * expected).sum(dim=-1).mean().detach())


def _stable_prefix(scores: list[float]) -> int | None:
    for index in range(len(scores)):
        if min(scores[index:]) >= MASTERY_THRESHOLD:
            return index + 1
    return None


def _train_source(
    cell: ExternalProgramFastCell,
    events: torch.Tensor,
    actions: torch.Tensor,
    *,
    learning_rate: float,
) -> tuple[
    list[float],
    list[tuple[torch.Tensor, object, torch.Tensor]],
    float,
]:
    # The source run learns the replaceable memory-side codec.  Once frozen,
    # the same codec must decode values written into fresh logical files.
    parameters = list(cell.value_encoder.parameters()) + list(
        cell.context_adapter.parameters()
    )
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    progress: list[float] = []
    retained: list[tuple[torch.Tensor, object, torch.Tensor]] = []
    begun = perf_counter()
    for event, action in zip(events, actions, strict=True):
        event = event.unsqueeze(0)
        action = action.unsqueeze(0)
        state = _write(
            cell,
            event,
            action,
            outcome=torch.ones(1),
        )
        prediction = _read(cell, state, event)
        loss = F.mse_loss(prediction, action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        progress.append(_score(prediction, action))
        retained.append((event.detach(), state, action.detach()))
    return progress, retained, (perf_counter() - begun) * 1000.0 / len(progress)


def _target_scores(
    cell: ExternalProgramFastCell,
    events: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[list[float], float]:
    scores: list[float] = []
    begun = perf_counter()
    for event, action in zip(events, actions, strict=True):
        event = event.unsqueeze(0)
        action = action.unsqueeze(0)
        state = _write(
            cell,
            event,
            action,
            outcome=torch.ones(1),
        )
        scores.append(_score(_read(cell, state, event), action))
    return scores, (perf_counter() - begun) * 1000.0 / len(scores)


def _train_fresh_target(
    cell: ExternalProgramFastCell,
    events: torch.Tensor,
    actions: torch.Tensor,
    *,
    learning_rate: float,
) -> tuple[list[float], float]:
    parameters = list(cell.value_encoder.parameters()) + list(
        cell.context_adapter.parameters()
    )
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    scores: list[float] = []
    begun = perf_counter()
    for event, action in zip(events, actions, strict=True):
        event = event.unsqueeze(0)
        action = action.unsqueeze(0)
        state = _write(
            cell,
            event,
            action,
            outcome=torch.ones(1),
        )
        prediction = _read(cell, state, event)
        scores.append(_score(prediction, action))
        loss = F.mse_loss(prediction, action)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
    return scores, (perf_counter() - begun) * 1000.0 / len(scores)


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(args.source_examples, args.target_examples) < 1:
        raise ValueError("stream lengths must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("learning rate must be positive")
    begun = perf_counter()
    torch.manual_seed(args.seed)
    codebook_generator = torch.Generator(device="cpu").manual_seed(args.seed + 303)
    action_codebook = F.normalize(
        torch.randn(
            ACTION_WIDTH,
            ACTION_WIDTH,
            generator=codebook_generator,
        ),
        dim=-1,
    )
    source_events, source_actions = _stream(
        args.source_examples,
        args.seed + 101,
        action_codebook,
    )
    target_events, target_actions = _stream(
        args.target_examples,
        args.seed + 202,
        action_codebook,
    )

    inherited = _make_cell()
    source_progress, retained, source_latency_ms = _train_source(
        inherited,
        source_events,
        source_actions,
        learning_rate=args.learning_rate,
    )
    source_digest_before_target = _digest(inherited)
    for parameter in inherited.parameters():
        parameter.requires_grad_(False)

    inherited_target, inherited_latency_ms = _target_scores(
        inherited,
        target_events,
        target_actions,
    )
    source_retention = [
        _score(_read(inherited, state, event), action)
        for event, state, action in retained
    ]

    fresh = _make_cell()
    fresh_target, fresh_latency_ms = _train_fresh_target(
        fresh,
        target_events,
        target_actions,
        learning_rate=args.learning_rate,
    )

    source_event, source_state, source_action = retained[0]
    source_query = inherited.query(source_event, _intention(1))
    source_value = inherited.value_encoder(torch.randn_like(source_action))
    failed = inherited.fast_weight.update(
        source_state,
        source_query,
        source_value,
        torch.zeros(1),
    )
    missing = inherited.fast_weight.update(
        source_state,
        source_query,
        source_value,
        torch.ones(1),
        present=torch.zeros(1, dtype=torch.bool),
    )
    restored = inherited.state_from_payload(
        inherited.state_payload(source_state)
    )
    source_digest_after_target = _digest(inherited)
    inherited_stable = _stable_prefix(inherited_target)
    fresh_stable = _stable_prefix(fresh_target)
    gates = {
        "source_mastery": min(source_retention) >= MASTERY_THRESHOLD,
        "target_mastery": (
            inherited_stable is not None
            and min(inherited_target) >= MASTERY_THRESHOLD
        ),
        "fresh_control_measured": fresh_stable is not None,
        "positive_transfer": (
            inherited_stable is not None
            and fresh_stable is not None
            and inherited_stable < fresh_stable
        ),
        "failed_outcome_no_write": torch.equal(
            failed.weights,
            source_state.weights,
        ),
        "missing_evidence_no_write": torch.equal(
            missing.weights,
            source_state.weights,
        ),
        "persistence_exact": (
            torch.equal(restored.weights, source_state.weights)
            and torch.equal(restored.updates, source_state.updates)
        ),
        "cell_frozen_during_target": source_digest_before_target
        == source_digest_after_target,
    }
    report = {
        "schema": "neural-computer.external-program-fast-cell-transfer.v1",
        "claim_boundary": (
            "A source-trained external execution-context adapter transfers into "
            "fresh per-file fast state; this is not arbitrary new computation "
            "or general continual learning."
        ),
        "seed": args.seed,
        "source_examples": args.source_examples,
        "target_examples": args.target_examples,
        "mastery_threshold": MASTERY_THRESHOLD,
        "source_progress": source_progress,
        "inherited_target_scores": inherited_target,
        "fresh_target_scores": fresh_target,
        "source_retention_scores": source_retention,
        "inherited_stable_examples": inherited_stable,
        "fresh_stable_examples": fresh_stable,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": args.source_examples + args.target_examples,
            "unique_logical_lifetimes": args.source_examples + args.target_examples,
            "optimizer_updates": {
                "source_codec": args.source_examples,
                "inherited_target": 0,
                "fresh_control": args.target_examples,
                "total": args.source_examples + args.target_examples,
            },
            "replayed_examples": 0,
            "matched_control_reused_examples": args.target_examples,
            "mean_step_latency_ms": {
                "source_training": source_latency_ms,
                "inherited_target": inherited_latency_ms,
                "fresh_control": fresh_latency_ms,
            },
            "inherited_target_stable_bits": inherited_stable,
            "fresh_target_stable_bits": fresh_stable,
            "transfer_ratio_against_fresh": (
                float(fresh_stable) / float(inherited_stable)
                if inherited_stable and fresh_stable
                else None
            ),
            "wall_seconds": perf_counter() - begun,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--source-examples", type=int, default=256)
    parser.add_argument("--target-examples", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
