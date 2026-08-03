"""Tiny reward-only diagnostic for learned multi-skill address routing.

The frozen controller produces one latent query per opaque context family and
one latent key per cold-bank row.  A selector sees only those queries, keys,
the attempted row, and the scalar success/failure of that attempt.  The
verifier keeps the family index private for evaluation and to generate the
scalar outcome.  This is a diagnostic for learned routing, not a promoted
replacement for nearest-key bank behavior.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .audit_sequence_multi_skill_bank import _model
from .audit_sequence_skill_bank import _context_key
from .audit_sequence_skill_memory import _load
from .skill_bank_router import SkillAddressSelector, attempted_outcome_loss


def _digest(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _queries(
        controller: nn.Module, spans: tuple[int, ...], *, per_span: int,
        seed: int, device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    queries: list[torch.Tensor] = []
    private_targets: list[int] = []
    for row, span in enumerate(spans):
        for offset in range(per_span):
            queries.append(_context_key(
                controller,
                seed=seed + row * 1000 + offset,
                count=64,
                span=span,
                distractors=2,
                device=device))
            private_targets.append(row)
    return torch.stack(queries), torch.tensor(
        private_targets, dtype=torch.long, device=device)


def _train(
        keys: torch.Tensor, queries: torch.Tensor, private_targets: torch.Tensor,
        *, updates: int, batch_size: int, seed: int,
        shuffle_rewards: bool,
        ) -> SkillAddressSelector:
    selector = SkillAddressSelector(
        width=int(keys.shape[-1]), hidden=64).to(keys.device)
    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(updates):
        indices = torch.randint(
            queries.shape[0], (batch_size,), generator=generator).to(
                queries.device)
        attempted = torch.randint(
            keys.shape[0], (batch_size,), generator=generator).to(
                queries.device)
        outcomes = (attempted == private_targets[indices]).to(torch.float32)
        if shuffle_rewards:
            outcomes = outcomes[torch.randperm(
                outcomes.shape[0], generator=generator).to(queries.device)]
        logits = selector(queries[indices], keys)
        loss = attempted_outcome_loss(logits, attempted, outcomes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    selector.eval()
    return selector


def _random_opaque_keys(
        rows: int, width: int, *, seed: int, device: torch.device,
        ) -> torch.Tensor:
    """Create fixed opaque row addresses with no query-aligned geometry."""
    if rows < 1 or width < 1:
        raise ValueError("rows and width must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = torch.randn((rows, width), generator=generator)
    return F.normalize(keys, dim=-1).to(device)


@torch.no_grad()
def _accuracy(
        selector: SkillAddressSelector, queries: torch.Tensor,
        targets: torch.Tensor, keys: torch.Tensor,
        ) -> float:
    predicted = selector(queries, keys).argmax(-1)
    return float((predicted == targets).float().mean())


@torch.no_grad()
def _permuted_accuracy(
        selector: SkillAddressSelector, queries: torch.Tensor,
        targets: torch.Tensor, keys: torch.Tensor,
        permutation: torch.Tensor,
        ) -> float:
    permuted_keys = keys[permutation]
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), device=permutation.device)
    return _accuracy(selector, queries, inverse[targets], permuted_keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--router-out", type=Path)
    parser.add_argument("--seed", type=int, default=93001)
    parser.add_argument("--spans", default="8,9,10")
    parser.add_argument(
        "--address-mode", choices=("controller", "random"),
        default="controller",
        help=(
            "controller uses the original query-aligned keys; random uses "
            "fixed opaque row addresses so reward is necessary."))
    parser.add_argument("--train-queries-per-skill", type=int, default=8)
    parser.add_argument("--test-queries-per-skill", type=int, default=16)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    spans = tuple(int(value) for value in args.spans.split(",") if value)
    if len(spans) < 2 or any(span < 1 for span in spans):
        raise ValueError("at least two positive spans are required")
    if args.train_queries_per_skill < 2 or args.test_queries_per_skill < 2:
        raise ValueError("each skill needs at least two queries")
    if args.updates < 1 or args.batch_size < 2:
        raise ValueError("updates and batch size must be positive")
    device = torch.device(args.device)
    payload = _load(args.parent, device)
    controller = _model(payload, device)
    controller_digest_before = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    train_queries, train_targets = _queries(
        controller, spans, per_span=args.train_queries_per_skill,
        seed=args.seed + 10_000, device=device)
    test_queries, test_targets = _queries(
        controller, spans, per_span=args.test_queries_per_skill,
        seed=args.seed + 20_000, device=device)
    if args.address_mode == "controller":
        keys, _ = _queries(
            controller, spans, per_span=1, seed=args.seed + 1_000,
            device=device)
        # One controller-produced address per row is retained.  The row-family
        # index is verifier-private and never enters selector training.
        keys = keys.detach()
    else:
        # Random row addresses are fixed across train/test, but independent of
        # the controller queries.  The only route from a family to its row is
        # the attempted-row scalar outcome observed during training.
        keys = _random_opaque_keys(
            len(spans), int(train_queries.shape[-1]),
            seed=args.seed + 1_000, device=device)
    selector = _train(
        keys, train_queries, train_targets,
        updates=args.updates, batch_size=args.batch_size,
        seed=args.seed + 30_000, shuffle_rewards=False)
    shuffled_selector = _train(
        keys, train_queries, train_targets,
        updates=args.updates, batch_size=args.batch_size,
        seed=args.seed + 40_000, shuffle_rewards=True)
    permutation = torch.arange(len(spans) - 1, -1, -1, device=device)
    normal_accuracy = _accuracy(selector, test_queries, test_targets, keys)
    shuffled_accuracy = _accuracy(
        shuffled_selector, test_queries, test_targets, keys)
    permuted_accuracy = _permuted_accuracy(
        selector, test_queries, test_targets, keys, permutation)
    cosine_accuracy = _accuracy(
        # The cosine baseline is evaluated without training and is retained as
        # a control against accidentally calling any static heuristic learned.
        _CosineSelector(), test_queries, test_targets, keys)
    controller_digest_after = _digest(controller)
    report = {
        "schema": "skill-bank-reward-router-audit-v1",
        "claim_boundary": (
            "The selector sees controller-produced queries, opaque candidate "
            "keys, an attempted row, and its scalar outcome. Family indices are "
            "private verifier data used only to generate outcomes and evaluate "
            "routing."),
        "parent": str(args.parent),
        "seed": args.seed,
        "spans_private_to_verifier": list(spans),
        "address_mode": args.address_mode,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "verifier_bits": args.updates * args.batch_size,
        "key_cosine": (
            torch.nn.functional.normalize(keys, dim=-1)
            @ torch.nn.functional.normalize(keys, dim=-1).T
        ).detach().cpu().tolist(),
        "normal_accuracy": normal_accuracy,
        "reward_shuffled_accuracy": shuffled_accuracy,
        "candidate_permutation_accuracy": permuted_accuracy,
        "cosine_baseline_accuracy": cosine_accuracy,
        "controller_weights_unchanged": (
            controller_digest_before == controller_digest_after),
        "training_inputs": [
            "controller-produced query",
            ("controller-produced candidate keys" if args.address_mode ==
             "controller" else "fixed random opaque candidate keys"),
            "attempted opaque row index",
            "scalar attempted-row outcome",
        ],
        "gates": {
            "normal_at_least_90": normal_accuracy >= 0.90,
            "reward_shuffle_near_chance": shuffled_accuracy <= 0.60,
            "candidate_permutation_invariant": permuted_accuracy >= 0.90,
            "controller_frozen": controller_digest_before == controller_digest_after,
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    if args.router_out:
        args.router_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "skill-bank-reward-router-v1",
            "width": selector.width,
            "hidden": 64,
            "state_dict": selector.state_dict(),
            "source_report": str(args.report),
            "admission_status": "diagnostic_only",
        }, args.router_out)
    print(json.dumps({
        key: report[key] for key in (
            "normal_accuracy", "reward_shuffled_accuracy",
            "candidate_permutation_accuracy", "cosine_baseline_accuracy",
            "accepted_diagnostic")
    }, sort_keys=True), flush=True)


class _CosineSelector(nn.Module):
    """Adapter exposing the fixed cosine rule through the selector interface."""

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        if keys.ndim == 2:
            keys = keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        query = torch.nn.functional.normalize(query, dim=-1)
        keys = torch.nn.functional.normalize(keys, dim=-1)
        return torch.einsum("bw,brw->br", query, keys)


if __name__ == "__main__":
    main()
