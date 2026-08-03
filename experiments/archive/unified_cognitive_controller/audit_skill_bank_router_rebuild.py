"""Compare incremental router updates with rebuilding from opaque replay.

Three skill families arrive sequentially.  The controller and random opaque
row keys are frozen.  The incremental arm uses output-distilled replay and is
expected to expose the three-row interference found in the exploratory probe.
The rebuild arm discards only the small selector weights, keeps the external
replay rows, and relearns the selector from attempted rows plus scalar
outcomes.  No span or correct-row label reaches either learner.
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
from .audit_skill_bank_router_online import _initial_train, _online_train
from .skill_bank_router import SkillAddressSelector, attempted_outcome_loss


def _digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _rebuild(
        keys: torch.Tensor, query_sets: list[torch.Tensor],
        target_sets: list[torch.Tensor], *, updates: int, batch_size: int,
        seed: int, shuffle_rewards: bool,
        ) -> SkillAddressSelector:
    if len(query_sets) != len(target_sets) or not query_sets:
        raise ValueError("rebuild replay sets must be non-empty and paired")
    torch.manual_seed(seed)
    selector = SkillAddressSelector(int(keys.shape[-1]), hidden=64)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=3e-3, weight_decay=1e-5)
    queries = torch.cat(query_sets, dim=0)
    targets = torch.cat(target_sets, dim=0)
    generator = torch.Generator(device="cpu").manual_seed(seed + 1_000)
    for _ in range(updates):
        indices = torch.randint(
            queries.shape[0], (batch_size,), generator=generator)
        attempted = torch.randint(
            keys.shape[0], (batch_size,), generator=generator)
        outcomes = (attempted == targets[indices]).to(torch.float32)
        if shuffle_rewards:
            # Independent Bernoulli nulls remove batch-composition and
            # attempted-row correlations that can survive a permutation.
            outcomes = torch.randint(
                0, 2, outcomes.shape, generator=generator,
                dtype=torch.float32)
        loss = attempted_outcome_loss(
            selector(queries[indices], keys), attempted, outcomes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    selector.eval()
    return selector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=93401)
    parser.add_argument("--spans", default="9,10,11")
    parser.add_argument("--address-width", type=int, default=64)
    parser.add_argument("--train-queries-per-skill", type=int, default=32)
    parser.add_argument("--test-queries-per-skill", type=int, default=64)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--distill-weight", type=float, default=0.1)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    spans = tuple(int(value) for value in args.spans.split(",") if value)
    if len(spans) < 3 or len(set(spans)) != len(spans):
        raise ValueError("at least three distinct spans are required")
    if min(args.train_queries_per_skill, args.test_queries_per_skill,
           args.updates, args.batch_size) < 2:
        raise ValueError("query, update, and batch counts must be at least two")
    device = torch.device(args.device)
    payload = _load(args.parent, device)
    controller = _model(payload, device)
    controller_digest_before = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    train_sets: list[torch.Tensor] = []
    train_targets: list[torch.Tensor] = []
    test_sets: list[torch.Tensor] = []
    test_targets: list[torch.Tensor] = []
    for index, span in enumerate(spans):
        train_query, _ = _queries(
            controller, (span,), per_span=args.train_queries_per_skill,
            seed=args.seed + 10_000 + index * 1_000, device=device)
        test_query, _ = _queries(
            controller, (span,), per_span=args.test_queries_per_skill,
            seed=args.seed + 20_000 + index * 1_000, device=device)
        train_sets.append(train_query)
        train_targets.append(torch.full(
            (train_query.shape[0],), index, dtype=torch.long, device=device))
        test_sets.append(test_query)
        test_targets.append(torch.full(
            (test_query.shape[0],), index, dtype=torch.long, device=device))
    keys = _random_opaque_keys(
        len(spans), args.address_width, seed=args.seed + 30_000,
        device=device)
    if keys.shape[-1] != train_sets[0].shape[-1]:
        raise ValueError("address-width must equal controller query width")

    phase_one = _initial_train(
        keys, train_sets[0], train_targets[0], updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 40_000)
    stage_one_teacher = phase_one(train_sets[0], keys).detach()
    incremental_stage_one = _online_train(
        copy.deepcopy(phase_one), keys, train_sets[0], stage_one_teacher,
        train_sets[1], train_targets[1], updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 50_000,
        distill_weight=args.distill_weight, shuffle_rewards=False)
    prior_queries = torch.cat(train_sets[:2], dim=0)
    prior_teacher = incremental_stage_one(prior_queries, keys).detach()
    incremental_stage_two = _online_train(
        copy.deepcopy(incremental_stage_one), keys, prior_queries,
        prior_teacher, train_sets[2], train_targets[2], updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 60_000,
        distill_weight=args.distill_weight, shuffle_rewards=False)

    rebuilt_stage_one = _rebuild(
        keys, train_sets[:2], train_targets[:2], updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 70_000,
        shuffle_rewards=False)
    rebuilt_stage_two = _rebuild(
        keys, train_sets, train_targets, updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 80_000,
        shuffle_rewards=False)
    shuffled_rebuilt_stage_two = _rebuild(
        keys, train_sets, train_targets, updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 90_000,
        shuffle_rewards=True)

    def score(selector: SkillAddressSelector) -> list[float]:
        return [
            _accuracy(selector, query, target, keys)
            for query, target in zip(test_sets, test_targets)]

    incremental_stage_one_scores = score(incremental_stage_one)
    incremental_stage_two_scores = score(incremental_stage_two)
    rebuilt_stage_one_scores = score(rebuilt_stage_one)
    rebuilt_stage_two_scores = score(rebuilt_stage_two)
    shuffled_scores = score(shuffled_rebuilt_stage_two)
    permutation = torch.arange(
        len(spans) - 1, -1, -1, device=device)
    rebuilt_permuted = [
        _permuted_accuracy(rebuilt_stage_two, query, target, keys, permutation)
        for query, target in zip(test_sets, test_targets)
    ]
    controller_digest_after = _digest(controller)
    report = {
        "schema": "skill-bank-router-rebuild-audit-v1",
        "claim_boundary": (
            "The frozen controller supplies queries. The selector sees opaque "
            "random row keys, attempted rows, and scalar outcomes. Span "
            "identities and correct rows are verifier-private."),
        "parent": str(args.parent),
        "seed": args.seed,
        "spans_private_to_verifier": list(spans),
        "updates_per_arm": args.updates,
        "batch_size": args.batch_size,
        "distill_weight": args.distill_weight,
        "shuffle_control": "independent random binary outcomes",
        "verifier_bits_per_rebuild": args.updates * args.batch_size,
        "incremental_stage_one": incremental_stage_one_scores,
        "incremental_stage_two": incremental_stage_two_scores,
        "rebuilt_stage_one": rebuilt_stage_one_scores,
        "rebuilt_stage_two": rebuilt_stage_two_scores,
        "shuffled_rebuilt_stage_two": shuffled_scores,
        "rebuilt_stage_two_permuted": rebuilt_permuted,
        "controller_weights_unchanged": (
            controller_digest_before == controller_digest_after),
        "gates": {
            "incremental_reference_recorded": True,
            "rebuild_stage_one_mastery": all(
                value >= 0.90 for value in rebuilt_stage_one_scores[:2]),
            "rebuild_stage_two_mastery": all(
                value >= 0.90 for value in rebuilt_stage_two_scores),
            # A null can favor one physical row by chance; judge its aggregate
            # accuracy rather than requiring every class to be individually
            # chance-level.
            "shuffled_rebuild_mean_near_chance": (
                sum(shuffled_scores) / len(shuffled_scores) <= 0.60),
            "rebuild_permutation_invariant": all(
                value >= 0.90 for value in rebuilt_permuted),
            "controller_frozen": (
                controller_digest_before == controller_digest_after),
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        key: report[key] for key in (
            "incremental_stage_one", "incremental_stage_two",
            "rebuilt_stage_one", "rebuilt_stage_two",
            "shuffled_rebuilt_stage_two", "accepted_diagnostic")
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
