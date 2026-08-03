"""Audit online reward routing with a task-agnostic retention objective.

The selector first learns one opaque skill address from scalar attempted-row
outcomes.  A second skill then arrives.  Three phase-two arms are compared:
new-skill-only (expected forgetting), reward training plus replay distillation
of the old selector outputs, and the same distillation with shuffled rewards.
The controller and candidate keys remain frozen.  No span or correct-row label
is given to the learner; verifier-private targets only generate the scalar
outcomes and score the audit.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .audit_sequence_multi_skill_bank import _model
from .audit_sequence_skill_memory import _load
from .audit_skill_bank_reward_router import (
    _accuracy,
    _permuted_accuracy,
    _queries,
    _random_opaque_keys,
)
from .skill_bank_router import (
    SkillAddressSelector,
    attempted_outcome_loss,
    selector_distillation_loss,
)


def _digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _initial_train(
        keys: torch.Tensor, queries: torch.Tensor, targets: torch.Tensor, *,
        updates: int, batch_size: int, seed: int,
        ) -> SkillAddressSelector:
    torch.manual_seed(seed)
    selector = SkillAddressSelector(int(keys.shape[-1]), hidden=64)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000)
    for _ in range(updates):
        indices = torch.randint(
            queries.shape[0], (batch_size,), generator=generator)
        attempted = torch.randint(
            keys.shape[0], (batch_size,), generator=generator)
        outcomes = (attempted == targets[indices]).to(torch.float32)
        loss = attempted_outcome_loss(
            selector(queries[indices], keys), attempted, outcomes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    selector.eval()
    return selector


def _online_train(
        selector: SkillAddressSelector, keys: torch.Tensor,
        old_queries: torch.Tensor, old_teacher_logits: torch.Tensor,
        new_queries: torch.Tensor, new_targets: torch.Tensor, *,
        updates: int, batch_size: int, seed: int, distill_weight: float,
        shuffle_rewards: bool,
        ) -> SkillAddressSelector:
    if distill_weight < 0.0:
        raise ValueError("distill_weight must be nonnegative")
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed + 2_000)
    for _ in range(updates):
        new_indices = torch.randint(
            new_queries.shape[0], (batch_size,), generator=generator)
        old_indices = torch.randint(
            old_queries.shape[0], (batch_size,), generator=generator)
        attempted = torch.randint(
            keys.shape[0], (batch_size,), generator=generator)
        outcomes = (attempted == new_targets[new_indices]).to(torch.float32)
        if shuffle_rewards:
            outcomes = outcomes[torch.randperm(
                outcomes.shape[0], generator=generator)]
        new_loss = attempted_outcome_loss(
            selector(new_queries[new_indices], keys), attempted, outcomes)
        old_logits = selector(old_queries[old_indices], keys)
        distill_loss = selector_distillation_loss(
            old_logits, old_teacher_logits[old_indices])
        loss = new_loss + distill_weight * distill_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    selector.eval()
    return selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=93201)
    parser.add_argument("--old-span", type=int, default=9)
    parser.add_argument("--new-span", type=int, default=10)
    parser.add_argument("--address-width", type=int, default=64)
    parser.add_argument("--train-queries-per-skill", type=int, default=32)
    parser.add_argument("--test-queries-per-skill", type=int, default=64)
    parser.add_argument("--phase-one-updates", type=int, default=512)
    parser.add_argument("--phase-two-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.old_span < 1 or args.new_span < 1:
        raise ValueError("spans must be positive")
    if args.old_span == args.new_span:
        raise ValueError("old and new spans must differ")
    if min(args.train_queries_per_skill, args.test_queries_per_skill,
           args.phase_one_updates, args.phase_two_updates, args.batch_size) < 2:
        raise ValueError("query, update, and batch counts must be at least two")
    device = torch.device(args.device)
    payload = _load(args.parent, device)
    controller = _model(payload, device)
    controller_digest_before = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    spans = (args.old_span, args.new_span)
    train_old, train_old_target = _queries(
        controller, (args.old_span,), per_span=args.train_queries_per_skill,
        seed=args.seed + 10_000, device=device)
    train_new, train_new_target = _queries(
        controller, (args.new_span,), per_span=args.train_queries_per_skill,
        seed=args.seed + 20_000, device=device)
    test_old, test_old_target = _queries(
        controller, (args.old_span,), per_span=args.test_queries_per_skill,
        seed=args.seed + 30_000, device=device)
    test_new, test_new_target = _queries(
        controller, (args.new_span,), per_span=args.test_queries_per_skill,
        seed=args.seed + 40_000, device=device)
    # The key vectors are opaque and independent of the controller queries.
    keys = _random_opaque_keys(
        2, args.address_width, seed=args.seed + 50_000, device=device)
    if keys.shape[-1] != train_old.shape[-1]:
        raise ValueError(
            "address-width must equal the controller query width for this audit")

    phase_one = _initial_train(
        keys, train_old, torch.zeros_like(train_old_target),
        updates=args.phase_one_updates, batch_size=args.batch_size,
        seed=args.seed + 60_000)
    teacher_logits = phase_one(train_old, keys).detach()
    naive = _online_train(
        copy.deepcopy(phase_one), keys, train_old, teacher_logits,
        train_new, torch.ones_like(train_new_target),
        updates=args.phase_two_updates, batch_size=args.batch_size,
        seed=args.seed + 70_000, distill_weight=0.0,
        shuffle_rewards=False)
    distilled = _online_train(
        copy.deepcopy(phase_one), keys, train_old, teacher_logits,
        train_new, torch.ones_like(train_new_target),
        updates=args.phase_two_updates, batch_size=args.batch_size,
        seed=args.seed + 80_000, distill_weight=args.distill_weight,
        shuffle_rewards=False)
    shuffled = _online_train(
        copy.deepcopy(phase_one), keys, train_old, teacher_logits,
        train_new, torch.ones_like(train_new_target),
        updates=args.phase_two_updates, batch_size=args.batch_size,
        seed=args.seed + 90_000, distill_weight=args.distill_weight,
        shuffle_rewards=True)

    def scores(selector: SkillAddressSelector) -> dict[str, float]:
        return {
            "old": _accuracy(selector, test_old, test_old_target, keys),
            "new": _accuracy(
                selector, test_new, torch.ones_like(test_new_target), keys),
            "old_permuted": _permuted_accuracy(
                selector, test_old, test_old_target, keys,
                torch.tensor([1, 0], device=device)),
            "new_permuted": _permuted_accuracy(
                selector, test_new, torch.ones_like(test_new_target), keys,
                torch.tensor([1, 0], device=device)),
        }

    phase_one_scores = scores(phase_one)
    naive_scores = scores(naive)
    distilled_scores = scores(distilled)
    shuffled_scores = scores(shuffled)
    controller_digest_after = _digest(controller)
    report = {
        "schema": "skill-bank-router-online-audit-v1",
        "claim_boundary": (
            "The controller is frozen. The selector sees opaque random row "
            "keys, attempted rows, and scalar outcomes. Span identities and "
            "correct rows are verifier-private."),
        "parent": str(args.parent),
        "seed": args.seed,
        "old_span_private": args.old_span,
        "new_span_private": args.new_span,
        "phase_one_updates": args.phase_one_updates,
        "phase_two_updates": args.phase_two_updates,
        "batch_size": args.batch_size,
        "distill_weight": args.distill_weight,
        "verifier_bits": args.batch_size * (
            args.phase_one_updates + 3 * args.phase_two_updates),
        "phase_one": phase_one_scores,
        "naive_new_only": naive_scores,
        "distilled_replay": distilled_scores,
        "shuffled_distilled": shuffled_scores,
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
            "distilled_old_permutation": (
                distilled_scores["old_permuted"] >= 0.90),
            "distilled_new_permutation": (
                distilled_scores["new_permuted"] >= 0.90),
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
            "shuffled_distilled", "accepted_diagnostic")
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
