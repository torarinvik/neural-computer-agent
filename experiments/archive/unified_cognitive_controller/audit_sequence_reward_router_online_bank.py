"""Audit continual reward routing with real disk-backed skill replacement.

An old artifact and an unused opaque address are first stored in the bounded
bank.  The selector learns the old route from scalar outcomes.  The unused
address is then replaced by a real new artifact, and phase-two updates compare
new-only learning with output-distilled replay.  The accepted arm must route
both rows after save/reload and reproduce each direct child's behavior.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .audit_sequence_multi_skill_bank import _model
from .audit_sequence_skill_memory import _build_skill_memory, _load, _rehydrate
from .audit_skill_bank_reward_router import (
    _accuracy,
    _permuted_accuracy,
    _queries,
    _random_opaque_keys,
)
from .audit_skill_bank_router_online import _initial_train, _online_train
from .skill_bank_router import SkillAddressSelector
from .skill_memory_bank import SkillArtifactBank
from .train_sequence_working_memory import evaluate_sequence_memory


def _digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--old-child", type=Path, required=True)
    parser.add_argument("--new-child", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=93301)
    parser.add_argument("--old-span", type=int, default=9)
    parser.add_argument("--new-span", type=int, default=10)
    parser.add_argument("--train-queries-per-skill", type=int, default=32)
    parser.add_argument("--test-queries-per-skill", type=int, default=32)
    parser.add_argument("--phase-one-updates", type=int, default=512)
    parser.add_argument("--phase-two-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument("--behavior-count", type=int, default=128)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.old_span == args.new_span or min(args.old_span, args.new_span) < 1:
        raise ValueError("old and new spans must be distinct and positive")
    if args.behavior_count < 2 or args.behavior_count % 2:
        raise ValueError("behavior-count must be positive and even")
    device = torch.device(args.device)
    base = _load(args.base, device)
    old_child = _load(args.old_child, device)
    new_child = _load(args.new_child, device)
    controller = _model(base, device)
    controller_digest_before = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    old_train, old_train_target = _queries(
        controller, (args.old_span,), per_span=args.train_queries_per_skill,
        seed=args.seed + 10_000, device=device)
    new_train, new_train_target = _queries(
        controller, (args.new_span,), per_span=args.train_queries_per_skill,
        seed=args.seed + 20_000, device=device)
    old_test, old_test_target = _queries(
        controller, (args.old_span,), per_span=args.test_queries_per_skill,
        seed=args.seed + 30_000, device=device)
    new_test, new_test_target = _queries(
        controller, (args.new_span,), per_span=args.test_queries_per_skill,
        seed=args.seed + 40_000, device=device)
    keys = _random_opaque_keys(2, int(old_train.shape[-1]),
                               seed=args.seed + 50_000, device=device)
    old_memory = _build_skill_memory(
        base, old_child, parent_path=args.base, child_path=args.old_child)
    new_memory = _build_skill_memory(
        base, new_child, parent_path=args.base, child_path=args.new_child)
    placeholder = dict(old_memory)
    placeholder["skill_state"] = {
        key: torch.zeros_like(value)
        for key, value in old_memory["skill_state"].items()}

    args.bank.mkdir(parents=True, exist_ok=True)
    bank = SkillArtifactBank(
        args.bank, width=int(keys.shape[-1]), capacity=2, device="cpu")
    old_index = bank.put(
        keys[0].detach().cpu(), old_memory, name="old-skill.pt")
    placeholder_index = bank.put(
        keys[1].detach().cpu(), placeholder, name="new-placeholder.pt",
        strength=0.0)
    bank.save()
    before_replacement = SkillArtifactBank.load(args.bank, device="cpu")
    initial_bank_exact = all(torch.equal(
        getattr(bank.memory.store, name),
        getattr(before_replacement.memory.store, name))
        for name in ("keys", "usage", "valid", "age"))
    # Learn the old route while the second address is still a placeholder.
    phase_one = _initial_train(
        keys, old_train, torch.zeros_like(old_train_target),
        updates=args.phase_one_updates, batch_size=args.batch_size,
        seed=args.seed + 60_000)
    old_teacher_logits = phase_one(old_train, keys).detach()
    # A successful old read makes the unused address the least-used row, so
    # replacement is driven by physical bank usage rather than a row label.
    bank.promote_with_selector(old_train[0].detach().cpu(), phase_one)
    # Replace the least-used placeholder row with the real new artifact.
    new_index = bank.put(
        keys[1].detach().cpu(), new_memory, name="new-skill.pt")
    bank.save()
    restored = SkillArtifactBank.load(args.bank, device="cpu")
    replacement_reload_exact = all(torch.equal(
        getattr(bank.memory.store, name),
        getattr(restored.memory.store, name))
        for name in ("keys", "usage", "valid", "age"))
    naive = _online_train(
        copy.deepcopy(phase_one), keys, old_train, old_teacher_logits,
        new_train, torch.ones_like(new_train_target),
        updates=args.phase_two_updates, batch_size=args.batch_size,
        seed=args.seed + 70_000, distill_weight=0.0,
        shuffle_rewards=False)
    distilled = _online_train(
        copy.deepcopy(phase_one), keys, old_train, old_teacher_logits,
        new_train, torch.ones_like(new_train_target),
        updates=args.phase_two_updates, batch_size=args.batch_size,
        seed=args.seed + 80_000, distill_weight=args.distill_weight,
        shuffle_rewards=False)
    shuffled = _online_train(
        copy.deepcopy(phase_one), keys, old_train, old_teacher_logits,
        new_train, torch.ones_like(new_train_target),
        updates=args.phase_two_updates, batch_size=args.batch_size,
        seed=args.seed + 90_000, distill_weight=args.distill_weight,
        shuffle_rewards=True)

    def route_scores(selector: SkillAddressSelector) -> dict[str, float]:
        return {
            "old": _accuracy(selector, old_test, old_test_target, keys),
            "new": _accuracy(
                selector, new_test, torch.ones_like(new_test_target), keys),
            "old_permuted": _permuted_accuracy(
                selector, old_test, old_test_target, keys,
                torch.tensor([1, 0], device=device)),
            "new_permuted": _permuted_accuracy(
                selector, new_test, torch.ones_like(new_test_target), keys,
                torch.tensor([1, 0], device=device)),
        }
    phase_one_scores = route_scores(phase_one)
    naive_scores = route_scores(naive)
    distilled_scores = route_scores(distilled)
    shuffled_scores = route_scores(shuffled)

    def promote_and_audit(
            selector: SkillAddressSelector, query: torch.Tensor,
            child: dict[str, object], span: int, seed: int,
            ) -> tuple[int, dict[str, object], dict[str, object]]:
        selected, confidence, artifact = restored.promote_with_selector(
            query.detach().cpu(), selector)
        routed_model = _rehydrate(base, artifact, device=device)
        direct_model = _model(child, device)
        routed_audit = evaluate_sequence_memory(
            routed_model, count=args.behavior_count, span=span,
            distractors=args.distractors, seed=seed,
            operation="mixed", device=device)
        direct_audit = evaluate_sequence_memory(
            direct_model, count=args.behavior_count, span=span,
            distractors=args.distractors, seed=seed,
            operation="mixed", device=device)
        return selected, {
            "confidence": confidence,
            "routed": routed_audit,
            "direct": direct_audit,
            "matches_direct": routed_audit["accuracy"] == direct_audit["accuracy"],
        }, artifact

    old_selected, old_behavior, _ = promote_and_audit(
        distilled, old_test[0], old_child, args.old_span, args.seed + 100_000)
    new_selected, new_behavior, _ = promote_and_audit(
        distilled, new_test[0], new_child, args.new_span, args.seed + 101_000)
    controller_digest_after = _digest(controller)
    report = {
        "schema": "sequence-reward-router-online-bank-audit-v1",
        "claim_boundary": (
            "The controller is frozen. The selector sees opaque random row "
            "keys, attempted rows, and scalar outcomes. Span identities and "
            "correct rows remain verifier-private."),
        "base": str(args.base),
        "old_child": str(args.old_child),
        "new_child": str(args.new_child),
        "bank": str(args.bank),
        "seed": args.seed,
        "old_span_private": args.old_span,
        "new_span_private": args.new_span,
        "phase_one_updates": args.phase_one_updates,
        "phase_two_updates": args.phase_two_updates,
        "batch_size": args.batch_size,
        "distill_weight": args.distill_weight,
        "verifier_bits": args.batch_size * (
            args.phase_one_updates + 3 * args.phase_two_updates),
        "old_index": old_index,
        "placeholder_index": placeholder_index,
        "new_index": new_index,
        "initial_bank_reload_exact": initial_bank_exact,
        "replacement_reload_exact": replacement_reload_exact,
        "phase_one": phase_one_scores,
        "naive_new_only": naive_scores,
        "distilled_replay": distilled_scores,
        "shuffled_distilled": shuffled_scores,
        "distilled_old_selected": old_selected,
        "distilled_new_selected": new_selected,
        "distilled_old_behavior": old_behavior,
        "distilled_new_behavior": new_behavior,
        "controller_weights_unchanged": (
            controller_digest_before == controller_digest_after),
        "gates": {
            "phase_one_old_mastery": phase_one_scores["old"] >= 0.90,
            "naive_exposes_forgetting": (
                naive_scores["new"] >= 0.90
                and naive_scores["old"] <= 0.60),
            "distilled_old_retained": distilled_scores["old"] >= 0.90,
            "distilled_new_mastery": distilled_scores["new"] >= 0.90,
            "shuffled_new_near_chance": shuffled_scores["new"] <= 0.60,
            "distilled_permutation": (
                distilled_scores["old_permuted"] >= 0.90
                and distilled_scores["new_permuted"] >= 0.90),
            "bank_reload_exact": initial_bank_exact and replacement_reload_exact,
            "old_row_selected": old_selected == old_index,
            "new_row_selected": new_selected == new_index,
            "old_behavior_matches_direct": old_behavior["matches_direct"],
            "new_behavior_matches_direct": new_behavior["matches_direct"],
            "controller_frozen": (
                controller_digest_before == controller_digest_after),
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "phase_one", "naive_new_only", "distilled_replay",
            "shuffled_distilled", "old_selected", "new_selected",
            "accepted_diagnostic")
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
