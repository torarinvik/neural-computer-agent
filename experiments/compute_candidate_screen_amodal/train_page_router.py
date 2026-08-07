"""Audit learned opaque retrieval of independently trained source pages.

Each page has a frozen local candidate screen. A page router receives only a
learned query and an opaque page summary, then learns from the scalar verifier
outcome of the local page's selected candidate. This removes physical page
order from source retrieval while leaving append-only growth and the sovereign
controller unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.compute_candidate_screen_amodal.train import (
    EVENT_WIDTH,
    _candidate_key_diagnostics,
    _candidate_keys,
    _event_query,
    _per_target_top1_accuracy,
)
from experiments.compute_candidate_screen_amodal.train_page_local_source_sharded import (
    _merge_training_accounting,
    _train_local_screen,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    generate_runtime_program_grammar,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
    _runtime,
)
from neural_computer import (
    LearnedComputeCandidateScreen,
    OpaqueCandidateSignatureNormalizer,
)

MASTERY_THRESHOLD = 0.75
DEFAULT_CANDIDATES = 64
DEFAULT_UNSEEN = 34
DEFAULT_SOURCE_PAGE_SIZE = 10


def _page_outcomes(
    query: torch.Tensor,
    families: list[int],
    source_pages: list[LearnedComputeCandidateScreen],
    candidate_keys: torch.Tensor,
    *,
    page_size: int,
    family_offset: int = 0,
) -> torch.Tensor:
    """Attempt each opaque page and retain only its scalar verifier result."""

    outcomes = torch.zeros(query.shape[0], len(source_pages))
    for page_index, page in enumerate(source_pages):
        local_predictions = page(
            query,
            candidate_keys[
                page_index * page_size : (page_index + 1) * page_size
            ],
        ).argmax(dim=-1)
        targets = torch.tensor(
            [
                family - family_offset - page_index * page_size
                for family in families
            ],
            dtype=torch.long,
        )
        outcomes[:, page_index] = (local_predictions == targets).float()
    return outcomes


def _train_page_router(
    router: LearnedComputeCandidateScreen,
    parent,
    grammar,
    page_keys: torch.Tensor,
    candidate_keys: torch.Tensor,
    source_pages: list[LearnedComputeCandidateScreen],
    *,
    page_size: int,
    source_count: int,
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    normalizer: OpaqueCandidateSignatureNormalizer,
    shuffle_outcomes: bool = False,
    family_offset: int = 0,
) -> dict[str, int | float]:
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    informative_pairs = 0
    last_loss = 0.0
    for update in range(updates):
        families = [
            family_offset + (update * batch_size + row) % source_count
            for row in range(batch_size)
        ]
        query = normalizer(
            _event_query(
                parent,
                grammar,
                families,
                seed=seed + update * 10_007,
            )
        )
        with torch.no_grad():
            outcomes = _page_outcomes(
                query,
                families,
                source_pages,
                candidate_keys,
                page_size=page_size,
                family_offset=family_offset,
            )
        if shuffle_outcomes:
            shuffled = []
            for row in range(batch_size):
                permutation = torch.randperm(
                    len(source_pages),
                    generator=torch.Generator().manual_seed(
                        seed + update * 101 + row
                    ),
                )
                shuffled.append(outcomes[row][permutation])
            outcomes = torch.stack(shuffled)
        if not bool(router.enabled.item()):
            router.enable()
        loss, pairs = router.outcome_ranking_loss(query, page_keys, outcomes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        informative_pairs += pairs
        last_loss = float(loss.detach())
    router.eval()
    return {
        "optimizer_updates": updates,
        "unique_verifier_bits": updates * batch_size * len(source_pages),
        "unique_logical_lifetimes": updates * batch_size * len(source_pages),
        "informative_candidate_pairs": informative_pairs,
        "informative_outcomes": updates * batch_size * len(source_pages),
        "replayed_examples": 0,
        "final_loss": last_loss,
    }


def _route(
    router: LearnedComputeCandidateScreen,
    source_pages: list[LearnedComputeCandidateScreen],
    query: torch.Tensor,
    page_keys: torch.Tensor,
    candidate_keys: torch.Tensor,
    *,
    normalizer: OpaqueCandidateSignatureNormalizer,
    page_size: int,
    page_order: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = router(normalizer(query), page_keys).argmax(dim=-1)
    predictions = torch.empty(slots.shape[0], dtype=torch.long)
    for slot, physical_page in enumerate(page_order):
        rows = slots == slot
        if rows.any():
            local = source_pages[physical_page](
                normalizer(query[rows]),
                candidate_keys[physical_page * page_size : (physical_page + 1) * page_size],
            ).argmax(dim=-1)
            predictions[rows] = physical_page * page_size + local
    return predictions, slots


def _metrics(
    predictions: torch.Tensor,
    page_slots: torch.Tensor,
    targets: torch.Tensor,
    *,
    page_order: list[int],
    page_size: int,
) -> dict[str, object]:
    physical_pages = torch.tensor([page_order[slot] for slot in page_slots.tolist()])
    target_pages = targets // page_size
    return {
        "top1_accuracy": float((predictions == targets).float().mean()),
        "per_target_top1_accuracy": _per_target_top1_accuracy(predictions, targets),
        "page_top1_accuracy": float((physical_pages == target_pages).float().mean()),
        "per_page_top1_accuracy": [
            float((physical_pages[target_pages == page] == page).float().mean())
            for page in sorted(set(target_pages.tolist()))
        ],
        "mean_fresh_attempts": 1.0,
        "fresh_verifier_authorized_rate": float((predictions == targets).float().mean()),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    source_count = args.candidate_count - args.unseen_candidates
    if source_count < 1 or source_count % args.source_page_size:
        raise ValueError("source candidates must form equal opaque pages")
    if min(
        args.parent_updates,
        args.source_updates_per_page,
        args.router_updates,
        args.batch_size,
        args.audit_count,
        args.key_samples,
    ) < 1:
        raise ValueError("all training and audit budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")
    page_count = source_count // args.source_page_size
    grammar = generate_runtime_program_grammar(
        seed=args.program_seed,
        count=args.candidate_count,
        depth=args.program_depth,
        primitive_family=args.primitive_family,
    )
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    parent = _runtime(seed=args.seed, growth=False)
    parent_history, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=max(1, args.parent_updates // 2),
        credit_mode="sampled",
    )
    parent.eval()
    parent_digest_before = _digest_core(parent, ())
    raw_keys = _candidate_keys(
        parent,
        grammar,
        args.candidate_count,
        seed=args.seed + 20_000,
        samples_per_candidate=args.key_samples,
    )
    normalizer = OpaqueCandidateSignatureNormalizer(EVENT_WIDTH)
    normalizer.fit(raw_keys)
    normalized_keys = normalizer(raw_keys).detach()
    source_pages = []
    source_training = []
    for page_index in range(page_count):
        page = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
        )
        source_pages.append(page)
        start = page_index * args.source_page_size
        source_training.append(
            _train_local_screen(
                page,
                parent,
                grammar,
                normalized_keys[start : start + args.source_page_size],
                list(range(start, start + args.source_page_size)),
                updates=args.source_updates_per_page,
                batch_size=args.batch_size,
                seed=args.seed + 30_000 + page_index * 10_000,
                learning_rate=args.learning_rate,
                query_transform=normalizer,
            )
        )
    page_keys = F.normalize(
        normalized_keys[:source_count]
        .reshape(page_count, args.source_page_size, EVENT_WIDTH)
        .mean(dim=1),
        dim=-1,
    )
    router = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    router_training = _train_page_router(
        router,
        parent,
        grammar,
        page_keys,
        normalized_keys,
        source_pages,
        page_size=args.source_page_size,
        source_count=source_count,
        updates=args.router_updates,
        batch_size=args.batch_size,
        seed=args.seed + 50_000,
        learning_rate=args.learning_rate,
        normalizer=normalizer,
    )
    shuffled_router = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    shuffled_training = _train_page_router(
        shuffled_router,
        parent,
        grammar,
        page_keys,
        normalized_keys,
        source_pages,
        page_size=args.source_page_size,
        source_count=source_count,
        updates=args.router_updates,
        batch_size=args.batch_size,
        seed=args.seed + 60_000,
        learning_rate=args.learning_rate,
        normalizer=normalizer,
        shuffle_outcomes=True,
    )
    families = [index % source_count for index in range(args.audit_count)]
    queries = _event_query(parent, grammar, families, seed=args.seed + 70_000)
    targets = torch.tensor(families)
    predictions, slots = _route(
        router,
        source_pages,
        queries,
        page_keys,
        normalized_keys,
        normalizer=normalizer,
        page_size=args.source_page_size,
        page_order=list(range(page_count)),
    )
    metrics = _metrics(
        predictions,
        slots,
        targets,
        page_order=list(range(page_count)),
        page_size=args.source_page_size,
    )
    permutation = torch.randperm(
        page_count,
        generator=torch.Generator().manual_seed(args.seed + 80_000),
    )
    permuted_predictions, permuted_slots = _route(
        router,
        source_pages,
        queries,
        page_keys[permutation],
        normalized_keys,
        normalizer=normalizer,
        page_size=args.source_page_size,
        page_order=permutation.tolist(),
    )
    permuted_metrics = _metrics(
        permuted_predictions,
        permuted_slots,
        targets,
        page_order=permutation.tolist(),
        page_size=args.source_page_size,
    )
    shuffled_predictions, shuffled_slots = _route(
        shuffled_router,
        source_pages,
        queries,
        page_keys,
        normalized_keys,
        normalizer=normalizer,
        page_size=args.source_page_size,
        page_order=list(range(page_count)),
    )
    shuffled_metrics = _metrics(
        shuffled_predictions,
        shuffled_slots,
        targets,
        page_order=list(range(page_count)),
        page_size=args.source_page_size,
    )
    state = {
        name: value.detach().clone() for name, value in router.state_dict().items()
    }
    restored = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    restored.load_state_dict(state, strict=True)
    restored_predictions, restored_slots = _route(
        restored,
        source_pages,
        queries,
        page_keys,
        normalized_keys,
        normalizer=normalizer,
        page_size=args.source_page_size,
        page_order=list(range(page_count)),
    )
    reload_exact = all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in state.items()
    )
    reload_metrics = _metrics(
        restored_predictions,
        restored_slots,
        targets,
        page_order=list(range(page_count)),
        page_size=args.source_page_size,
    )
    parent_digest_after = _digest_core(parent, ())
    parent_accounting = parent_history[-1] if parent_history else {}
    source_accounting = _merge_training_accounting(
        [*source_training, router_training, shuffled_training]
    )
    accounting = {
        "unique_verifier_bits": int(parent_accounting.get("unique_verifier_bits", 0))
        + int(source_accounting["unique_verifier_bits"]),
        "unique_logical_lifetimes": int(
            parent_accounting.get("unique_logical_lifetimes", 0)
        )
        + int(source_accounting["unique_logical_lifetimes"]),
        "optimizer_updates": args.parent_updates
        + int(source_accounting["optimizer_updates"]),
        "replayed_examples": int(source_accounting["replayed_examples"]),
        "wall_seconds": perf_counter() - started,
    }
    gates = {
        "parent_stable": any(
            float(row["heldout_accuracy"]) >= MASTERY_THRESHOLD
            for row in parent_progress
        ),
        "page_generalization": metrics["page_top1_accuracy"] >= MASTERY_THRESHOLD,
        "candidate_generalization": metrics["top1_accuracy"] >= MASTERY_THRESHOLD,
        "page_per_candidate_mastery": min(metrics["per_page_top1_accuracy"])
        >= MASTERY_THRESHOLD,
        "candidate_per_candidate_mastery": min(metrics["per_target_top1_accuracy"])
        >= MASTERY_THRESHOLD,
        "page_permutation": permuted_metrics["page_top1_accuracy"]
        >= MASTERY_THRESHOLD,
        "candidate_permutation": permuted_metrics["top1_accuracy"]
        >= MASTERY_THRESHOLD,
        "reward_shuffled_null": shuffled_metrics["page_top1_accuracy"]
        <= (1.0 / page_count) + 0.15,
        "reload_exact": reload_exact,
        "reload_behavior_exact": reload_metrics == metrics,
        "core_unchanged": parent_digest_before == parent_digest_after,
        "no_replayed_examples": accounting["replayed_examples"] == 0,
    }
    report = {
        "schema": "neural-computer.opaque-page-router-report.v1",
        "claim_boundary": (
            "An external page router learned opaque source-page retrieval from "
            "scalar outcomes of attempted local pages. The controller remains "
            "frozen; this is bounded learned page addressing, not general "
            "continual learning."
        ),
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "source_candidate_count": source_count,
        "source_page_size": args.source_page_size,
        "source_page_count": page_count,
        "page_key_construction": "normalized_opaque_candidate_mean_v1",
        "learning_signal": "scalar_verifier_outcome_of_attempted_local_page_v1",
        "candidate_key_diagnostics": {
            "raw": _candidate_key_diagnostics(raw_keys),
            "normalized": _candidate_key_diagnostics(normalized_keys),
            "page": _candidate_key_diagnostics(page_keys),
        },
        "budgets": {
            "parent_updates": args.parent_updates,
            "source_updates_per_page": args.source_updates_per_page,
            "router_updates": args.router_updates,
            "reward_shuffled_router_updates": args.router_updates,
            "batch_size": args.batch_size,
            "audit_count": args.audit_count,
            "key_samples": args.key_samples,
        },
        "training": {
            "source_pages": source_training,
            "router": router_training,
            "reward_shuffled_router": shuffled_training,
        },
        "accounting": accounting,
        "metrics": metrics,
        "permuted_metrics": permuted_metrics,
        "reward_shuffled_metrics": shuffled_metrics,
        "reload_metrics": reload_metrics,
        "controls": {
            "parent_digest_before": parent_digest_before,
            "parent_digest_after": parent_digest_after,
            "parent_unchanged": parent_digest_before == parent_digest_after,
            "router_reload_exact": reload_exact,
            "fresh_admission_required": True,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "wall_seconds": accounting["wall_seconds"],
    }
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--program-seed", type=int, default=4242)
    parser.add_argument("--program-depth", type=int, default=8)
    parser.add_argument(
        "--primitive-family",
        choices=("registry", "opaque_rule"),
        default="opaque_rule",
    )
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--unseen-candidates", type=int, default=DEFAULT_UNSEEN)
    parser.add_argument("--source-page-size", type=int, default=DEFAULT_SOURCE_PAGE_SIZE)
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--source-updates-per-page", type=int, default=512)
    parser.add_argument("--router-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--audit-count", type=int, default=96)
    parser.add_argument("--key-samples", type=int, default=12)
    parser.add_argument("--screen-hidden", type=int, default=64)
    parser.add_argument("--latent-width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args()
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise ValueError("torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)
    report = run(args)
    print(
        json.dumps(
            {
                "seed": report["seed"],
                "promoted": report["promoted"],
                "metrics": report["metrics"],
                "permuted_metrics": report["permuted_metrics"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
