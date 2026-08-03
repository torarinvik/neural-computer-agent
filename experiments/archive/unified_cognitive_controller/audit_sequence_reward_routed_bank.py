"""End-to-end audit for reward-trained routing through a real skill bank.

The bank stores real opaque successor-slot artifacts behind fixed random row
addresses.  A frozen controller supplies context queries.  The selector is
trained only from an attempted physical row and that attempt's scalar
outcome; span names and correct rows remain verifier-private.  The audit then
reloads the bank and checks routing, artifact integrity, direct-vs-rehydrated
behavior, and wrong-skill controls.  Learned routing is opt-in and never
changes the bank's default nearest-key path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .audit_sequence_multi_skill_bank import _model
from .audit_sequence_skill_bank import _context_key
from .audit_sequence_skill_memory import _build_skill_memory, _load, _rehydrate
from .audit_skill_bank_reward_router import (
    _accuracy,
    _CosineSelector,
    _permuted_accuracy,
    _queries,
    _random_opaque_keys,
    _train,
)
from .skill_memory_bank import SkillArtifactBank
from .train_sequence_working_memory import evaluate_sequence_memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--children", type=Path, nargs="+", required=True)
    parser.add_argument("--spans", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=93111)
    parser.add_argument("--train-queries-per-skill", type=int, default=16)
    parser.add_argument("--test-queries-per-skill", type=int, default=32)
    parser.add_argument("--updates", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--behavior-count", type=int, default=512)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    spans = tuple(int(value) for value in args.spans.split(",") if value)
    if len(spans) != len(args.children) or len(spans) < 2:
        raise ValueError("spans must contain one entry per child and at least two")
    if args.behavior_count < 2 or args.behavior_count % 2:
        raise ValueError("behavior-count must be positive and even")
    device = torch.device(args.device)
    base_payload = _load(args.base, device)
    child_payloads = [_load(path, device) for path in args.children]
    controller = _model(base_payload, device)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    train_queries, train_targets = _queries(
        controller, spans, per_span=args.train_queries_per_skill,
        seed=args.seed + 10_000, device=device)
    test_queries, test_targets = _queries(
        controller, spans, per_span=args.test_queries_per_skill,
        seed=args.seed + 20_000, device=device)
    keys = _random_opaque_keys(
        len(spans), int(train_queries.shape[-1]), seed=args.seed + 1_000,
        device=device)
    selector = _train(
        keys, train_queries, train_targets, updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 30_000,
        shuffle_rewards=False)
    shuffled_selector = _train(
        keys, train_queries, train_targets, updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 40_000,
        shuffle_rewards=True)
    permutation = torch.arange(
        len(spans) - 1, -1, -1, device=device)
    normal_accuracy = _accuracy(
        selector, test_queries, test_targets, keys)
    shuffled_accuracy = _accuracy(
        shuffled_selector, test_queries, test_targets, keys)
    permuted_accuracy = _permuted_accuracy(
        selector, test_queries, test_targets, keys, permutation)
    cosine_accuracy = _accuracy(
        _CosineSelector(), test_queries, test_targets, keys)

    skill_memories = [
        _build_skill_memory(
            base_payload, child, parent_path=args.base, child_path=path)
        for path, child in zip(args.children, child_payloads)
    ]
    args.bank.mkdir(parents=True, exist_ok=True)
    bank = SkillArtifactBank(
        args.bank, width=int(keys.shape[-1]), capacity=len(spans), device="cpu")
    physical_indices = [
        bank.put(
            keys[index].detach().cpu(), skill_memories[index],
            name=f"skill-span{span}.pt")
        for index, span in enumerate(spans)
    ]
    bank.save()
    restored = SkillArtifactBank.load(args.bank, device="cpu")

    routed: list[dict[str, object]] = []
    selected_artifacts: dict[int, dict[str, object]] = {}
    for query, target in zip(test_queries, test_targets):
        selected, confidence, artifact = restored.promote_with_selector(
            query.detach().cpu(), selector)
        expected = int(target)
        routed.append({
            "expected_index": physical_indices[expected],
            "selected_index": selected,
            "confidence": confidence,
        })
        selected_artifacts.setdefault(expected, artifact)
    if any(row["expected_index"] != row["selected_index"] for row in routed):
        raise AssertionError("reward-trained selector routed a held-out query wrong")

    behavior: dict[str, dict[str, object]] = {}
    for index, span in enumerate(spans):
        rehydrated = _rehydrate(
            base_payload, selected_artifacts[index], device=device)
        direct = _model(child_payloads[index], device)
        direct_audit = evaluate_sequence_memory(
            direct, count=args.behavior_count, span=span,
            distractors=args.distractors, seed=args.seed + 50_000 + index,
            operation="mixed", device=device)
        routed_audit = evaluate_sequence_memory(
            rehydrated, count=args.behavior_count, span=span,
            distractors=args.distractors, seed=args.seed + 50_000 + index,
            operation="mixed", device=device)
        wrong_index = (index + 1) % len(spans)
        wrong_model = _rehydrate(
            base_payload, skill_memories[wrong_index], device=device)
        wrong_audit = evaluate_sequence_memory(
            wrong_model, count=args.behavior_count, span=span,
            distractors=args.distractors, seed=args.seed + 50_000 + index,
            operation="mixed", device=device)
        behavior[str(span)] = {
            "direct": direct_audit,
            "routed": routed_audit,
            "wrong_skill": wrong_audit,
            "routed_matches_direct_accuracy": (
                routed_audit["accuracy"] == direct_audit["accuracy"]),
        }

    report = {
        "schema": "sequence-reward-routed-bank-audit-v1",
        "claim_boundary": (
            "The frozen controller supplies queries. The selector sees only "
            "opaque candidate keys, attempted rows, and scalar outcomes. "
            "Span identities and correct rows remain verifier-private."),
        "base": str(args.base),
        "children": [str(path) for path in args.children],
        "spans_private_to_verifier": list(spans),
        "bank": str(args.bank),
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "verifier_bits": args.updates * args.batch_size,
        "train_queries_per_skill": args.train_queries_per_skill,
        "test_queries_per_skill": args.test_queries_per_skill,
        "normal_accuracy": normal_accuracy,
        "reward_shuffled_accuracy": shuffled_accuracy,
        "candidate_permutation_accuracy": permuted_accuracy,
        "cosine_baseline_accuracy": cosine_accuracy,
        "bank_reload_exact": all(
            torch.equal(
                getattr(bank.memory.store, name),
                getattr(restored.memory.store, name))
            for name in ("keys", "usage", "valid", "age")),
        "routed_queries": len(routed),
        "routed_all_correct": all(
            row["expected_index"] == row["selected_index"] for row in routed),
        "access_count_total": int(restored.memory.store.access_count.sum()),
        "controller_weights_frozen": True,
        "behavior": behavior,
        "gates": {
            "normal_at_least_90": normal_accuracy >= 0.90,
            "reward_shuffle_near_chance": shuffled_accuracy <= (
                0.60 if len(spans) == 2 else 0.45),
            "candidate_permutation_invariant": permuted_accuracy >= 0.90,
            "cosine_near_chance": cosine_accuracy <= (
                0.60 if len(spans) == 2 else 0.45),
            "bank_reload_exact": all(
                torch.equal(
                    getattr(bank.memory.store, name),
                    getattr(restored.memory.store, name))
                for name in ("keys", "usage", "valid", "age")),
            "routed_all_correct": all(
                row["expected_index"] == row["selected_index"]
                for row in routed),
            "routed_matches_direct": all(
                item["routed_matches_direct_accuracy"]
                for item in behavior.values()),
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "normal_accuracy", "reward_shuffled_accuracy",
            "candidate_permutation_accuracy", "cosine_baseline_accuracy",
            "bank_reload_exact", "routed_all_correct",
            "accepted_diagnostic")
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
