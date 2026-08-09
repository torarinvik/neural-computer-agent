"""Pressure-test outcome-only external fast-weight plasticity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import ExternalFastWeightPlasticity


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _state_digest(rule: ExternalFastWeightPlasticity, state) -> str:
    payload = rule.state_payload(state)
    digest = hashlib.sha256()
    digest.update(payload["weights"].numpy().tobytes())
    digest.update(payload["updates"].numpy().tobytes())
    return digest.hexdigest()


def _score(rule, state, query, expected) -> float:
    read = rule.read(state, query)
    expected = F.normalize(expected, dim=-1)
    actual = F.normalize(read, dim=-1, eps=1e-8)
    return float((actual * expected).sum(dim=-1).mean().detach())


def _stable_updates(scores: list[float], threshold: float) -> int | None:
    for index in range(len(scores)):
        if min(scores[index:]) >= threshold:
            return index + 1
    return None


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(args.key_width, args.value_width, args.updates, args.audit_count) < 1:
        raise ValueError("dimensions, updates, and audit count must be positive")
    torch.manual_seed(args.seed)
    rule = ExternalFastWeightPlasticity(
        args.key_width,
        args.value_width,
        hidden=args.hidden,
    )
    rule_digest_before = _digest(rule)

    source_query = F.normalize(
        torch.arange(1, args.key_width + 1, dtype=torch.float32).unsqueeze(0),
        dim=-1,
    )
    target_query = F.normalize(
        torch.arange(args.key_width, 0, -1, dtype=torch.float32).unsqueeze(0),
        dim=-1,
    )
    source_value = F.normalize(
        torch.arange(1, args.value_width + 1, dtype=torch.float32).unsqueeze(0),
        dim=-1,
    )
    target_value = F.normalize(
        torch.arange(args.value_width, 0, -1, dtype=torch.float32).unsqueeze(0),
        dim=-1,
    )
    success = torch.ones(1)

    source_state = rule.initial_state(1)
    source_progress: list[float] = []
    for _ in range(args.updates):
        source_state = rule.update(
            source_state, source_query, source_value, success
        )
        source_progress.append(_score(rule, source_state, source_query, source_value))
    source_digest_before_target = _state_digest(rule, source_state)

    # This is a distinct external file/state. No source outcome or tensor is
    # replayed while the target state is learned.
    target_state = rule.initial_state(1)
    target_progress: list[float] = []
    for _ in range(args.updates):
        target_state = rule.update(
            target_state, target_query, target_value, success
        )
        target_progress.append(_score(rule, target_state, target_query, target_value))

    source_after_target = _score(
        rule, source_state, source_query, source_value
    )
    target_score = _score(rule, target_state, target_query, target_value)

    failed_state = rule.update(
        source_state,
        source_query,
        -source_value,
        torch.zeros(1),
    )
    missing_state = rule.update(
        source_state,
        source_query,
        -source_value,
        success,
        present=torch.zeros(1, dtype=torch.bool),
    )
    source_unchanged_on_failure = torch.equal(
        failed_state.weights, source_state.weights
    )
    source_unchanged_on_missing = _state_digest(rule, missing_state) == source_digest_before_target

    payload = rule.state_payload(source_state)
    restored = rule.state_from_payload(payload)
    persistence_exact = (
        torch.equal(restored.weights, source_state.weights)
        and torch.equal(restored.updates, source_state.updates)
    )
    rule_digest_after = _digest(rule)
    threshold = 0.95
    source_stable = all(score >= threshold for score in source_progress)
    target_stable = all(score >= threshold for score in target_progress)
    report = {
        "schema": "neural-computer.external-fast-plasticity-pressure-test.v1",
        "claim_boundary": (
            "One outcome-only external fast-weight state learns two isolated "
            "opaque associations without controller updates or replay; this "
            "is not general continual learning."
        ),
        "seed": args.seed,
        "key_width": args.key_width,
        "value_width": args.value_width,
        "updates_per_capability": args.updates,
        "source_progress": source_progress,
        "target_progress": target_progress,
        "source_stable": source_stable,
        "target_stable": target_stable,
        "source_retention_after_target": source_after_target,
        "target_score": target_score,
        "failed_outcome_no_write": source_unchanged_on_failure,
        "missing_evidence_no_write": source_unchanged_on_missing,
        "persistence_exact": persistence_exact,
        "plasticity_rule_frozen": rule_digest_before == rule_digest_after,
        "replayed_examples": 0,
        "accounting": {
            "unique_verifier_bits": args.updates * 2 + 2,
            "unique_logical_lifetimes": args.updates * 2,
            "optimizer_updates": 0,
            "stable_bits_to_threshold": (
                _stable_updates(source_progress, threshold)
                if source_stable
                else None
            ),
            "target_stable_bits_to_threshold": (
                _stable_updates(target_progress, threshold)
                if target_stable
                else None
            ),
        },
        "promoted": bool(
            source_stable
            and target_stable
            and source_after_target >= threshold
            and source_unchanged_on_failure
            and source_unchanged_on_missing
            and persistence_exact
            and rule_digest_before == rule_digest_after
        ),
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--key-width", type=int, default=8)
    parser.add_argument("--value-width", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--audit-count", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
