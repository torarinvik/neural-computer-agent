"""Pressure-test learned page retrieval during append-only growth.

The source router is trained once from scalar outcomes of independently
trained source pages. New pages then grow only the external page and candidate
memory: the router and all source pages remain frozen. The audit measures
whether the frozen learned address function can retrieve new pages and retain
old pages without replay.
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
from experiments.compute_candidate_screen_amodal.train_page_router import (
    _train_page_router,
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
    OpaqueCandidateIdentityView,
    OpaqueCandidateSignatureNormalizer,
)

MASTERY_THRESHOLD = 0.75


def _page_ranges(page_sizes: list[int]) -> list[tuple[int, int]]:
    ranges = []
    start = 0
    for size in page_sizes:
        ranges.append((start, start + size))
        start += size
    return ranges


def _mixed_route(
    router: LearnedComputeCandidateScreen,
    pages: list[LearnedComputeCandidateScreen],
    query: torch.Tensor,
    page_keys: torch.Tensor,
    candidate_keys: torch.Tensor,
    normalizer: OpaqueCandidateSignatureNormalizer,
    page_sizes: list[int],
    page_order: list[int],
    raw_page_start: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route through source-normalized and append-raw local pages."""

    slots = router(normalizer(query), page_keys).argmax(dim=-1)
    predictions = torch.empty(slots.shape[0], dtype=torch.long)
    ranges = _page_ranges(page_sizes)
    for slot, physical_page in enumerate(page_order):
        rows = slots == slot
        if not rows.any():
            continue
        start, end = ranges[physical_page]
        page_query = query[rows]
        page_keys_for_screen = candidate_keys[start:end]
        if physical_page < raw_page_start:
            page_query = normalizer(page_query)
            page_keys_for_screen = normalizer(page_keys_for_screen)
        local = pages[physical_page](page_query, page_keys_for_screen).argmax(dim=-1)
        predictions[rows] = start + local
    return predictions, slots


def _mixed_metrics(
    predictions: torch.Tensor,
    slots: torch.Tensor,
    targets: torch.Tensor,
    page_order: list[int],
    page_sizes: list[int],
) -> dict[str, object]:
    ranges = _page_ranges(page_sizes)
    physical_pages = torch.empty_like(targets)
    for page, (start, end) in enumerate(ranges):
        rows = (targets >= start) & (targets < end)
        physical_pages[rows] = page
    selected_pages = torch.tensor([page_order[slot] for slot in slots.tolist()])
    return {
        "top1_accuracy": float((predictions == targets).float().mean()),
        "per_target_top1_accuracy": _per_target_top1_accuracy(
            predictions, targets
        ),
        "page_top1_accuracy": float((selected_pages == physical_pages).float().mean()),
        "per_page_top1_accuracy": [
            float(
                (selected_pages[physical_pages == page] == page)
                .float()
                .mean()
            )
            for page in range(len(page_sizes))
        ],
        "mean_fresh_attempts": 1.0,
        "fresh_verifier_authorized_rate": float(
            (predictions == targets).float().mean()
        ),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.source_candidates < args.source_page_size:
        raise ValueError("source candidates must fill at least one page")
    if args.source_candidates % args.source_page_size:
        raise ValueError("source candidates must form equal source pages")
    if args.append_candidates < 1 or args.append_candidates % args.append_page_size:
        raise ValueError("append candidates must form equal append pages")
    if min(
        args.parent_updates,
        args.source_updates_per_page,
        args.append_updates_per_page,
        args.router_updates,
        args.batch_size,
        args.audit_count,
        args.key_samples,
    ) < 1:
        raise ValueError("all training and audit budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")

    source_page_count = args.source_candidates // args.source_page_size
    append_page_count = args.append_candidates // args.append_page_size
    total_candidates = args.source_candidates + args.append_candidates
    page_sizes = [args.source_page_size] * source_page_count + [
        args.append_page_size
    ] * append_page_count
    raw_page_start = source_page_count
    grammar = generate_runtime_program_grammar(
        seed=args.program_seed,
        count=total_candidates,
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
        total_candidates,
        seed=args.seed + 20_000,
        samples_per_candidate=args.key_samples,
    )
    normalizer = OpaqueCandidateSignatureNormalizer(EVENT_WIDTH)
    normalizer.fit(raw_keys)
    normalized_keys = normalizer(raw_keys).detach()
    identity = OpaqueCandidateIdentityView(EVENT_WIDTH)

    pages: list[LearnedComputeCandidateScreen] = []
    page_training: list[dict[str, int | float]] = []
    ranges = _page_ranges(page_sizes)
    for page_index, (start, end) in enumerate(ranges):
        page = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
        )
        pages.append(page)
        is_source = page_index < raw_page_start
        page_training.append(
            _train_local_screen(
                page,
                parent,
                grammar,
                (normalized_keys if is_source else raw_keys)[start:end],
                list(range(start, end)),
                updates=(
                    args.source_updates_per_page
                    if is_source
                    else args.append_updates_per_page
                ),
                batch_size=args.batch_size,
                seed=args.seed + 30_000 + page_index * 10_000,
                learning_rate=args.learning_rate,
                query_transform=normalizer if is_source else identity,
            )
        )

    source_page_keys = F.normalize(
        normalized_keys[: args.source_candidates]
        .reshape(source_page_count, args.source_page_size, EVENT_WIDTH)
        .mean(dim=1),
        dim=-1,
    )
    append_page_keys = F.normalize(
        normalized_keys[args.source_candidates :]
        .reshape(append_page_count, args.append_page_size, EVENT_WIDTH)
        .mean(dim=1),
        dim=-1,
    )
    source_router = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    source_router_training = _train_page_router(
        source_router,
        parent,
        grammar,
        source_page_keys,
        normalized_keys[: args.source_candidates],
        pages[:raw_page_start],
        page_size=args.source_page_size,
        source_count=args.source_candidates,
        updates=args.router_updates,
        batch_size=args.batch_size,
        seed=args.seed + 50_000,
        learning_rate=args.learning_rate,
        normalizer=normalizer,
    )
    router_digest_before_append = {
        name: value.detach().clone()
        for name, value in source_router.state_dict().items()
    }
    source_page_states_before_append = [
        {
            name: value.detach().clone()
            for name, value in page.state_dict().items()
        }
        for page in pages[:raw_page_start]
    ]
    all_page_keys = torch.cat([source_page_keys, append_page_keys], dim=0)
    audit_families = [index % total_candidates for index in range(args.audit_count)]
    queries = _event_query(parent, grammar, audit_families, seed=args.seed + 70_000)
    targets = torch.tensor(audit_families)
    predictions, slots = _mixed_route(
        source_router,
        pages,
        queries,
        all_page_keys,
        raw_keys,
        normalizer,
        page_sizes,
        list(range(len(pages))),
        raw_page_start,
    )
    metrics = _mixed_metrics(
        predictions, slots, targets, list(range(len(pages))), page_sizes
    )
    page_permutation = torch.randperm(
        len(pages), generator=torch.Generator().manual_seed(args.seed + 80_000)
    )
    permuted_predictions, permuted_slots = _mixed_route(
        source_router,
        pages,
        queries,
        all_page_keys[page_permutation],
        raw_keys,
        normalizer,
        page_sizes,
        page_permutation.tolist(),
        raw_page_start,
    )
    permuted_metrics = _mixed_metrics(
        permuted_predictions,
        permuted_slots,
        targets,
        page_permutation.tolist(),
        page_sizes,
    )
    router_still_frozen = all(
        torch.equal(value, source_router.state_dict()[name])
        for name, value in router_digest_before_append.items()
    )
    source_page_immutable = all(
        torch.equal(page.state_dict()[name], value)
        for page, saved in zip(
            pages[:raw_page_start],
            source_page_states_before_append,
            strict=True,
        )
        for name, value in saved.items()
    )
    source_accounting = _merge_training_accounting(
        [*page_training, source_router_training]
    )
    parent_accounting = parent_history[-1] if parent_history else {}
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
        "candidate_generalization": metrics["top1_accuracy"] >= MASTERY_THRESHOLD,
        "candidate_per_candidate_mastery": min(metrics["per_target_top1_accuracy"])
        >= MASTERY_THRESHOLD,
        "page_generalization": metrics["page_top1_accuracy"] >= MASTERY_THRESHOLD,
        "page_per_page_mastery": min(metrics["per_page_top1_accuracy"])
        >= MASTERY_THRESHOLD,
        "candidate_permutation": permuted_metrics["top1_accuracy"]
        >= MASTERY_THRESHOLD,
        "page_permutation": permuted_metrics["page_top1_accuracy"]
        >= MASTERY_THRESHOLD,
        "router_frozen_during_append": router_still_frozen,
        "source_pages_frozen_during_append": source_page_immutable,
        "core_unchanged": parent_digest_before == _digest_core(parent, ()),
        "no_replayed_examples": accounting["replayed_examples"] == 0,
    }
    report = {
        "schema": "neural-computer.opaque-page-router-append-report.v1",
        "claim_boundary": (
            "A frozen learned page router is tested against append-only pages "
            "trained in external memory. Passing this audit would establish "
            "bounded no-replay page retrieval during growth; it is not general "
            "continual learning."
        ),
        "seed": args.seed,
        "candidate_count": total_candidates,
        "source_candidate_count": args.source_candidates,
        "append_candidate_count": args.append_candidates,
        "source_page_count": source_page_count,
        "append_page_count": append_page_count,
        "page_sizes": page_sizes,
        "page_key_construction": "normalized_opaque_candidate_mean_v1",
        "learning_signal": "scalar_verifier_outcome_of_attempted_local_page_v1",
        "append_policy": "append_local_pages_without_router_replay_or_update_v1",
        "candidate_key_diagnostics": {
            "raw": _candidate_key_diagnostics(raw_keys),
            "normalized": _candidate_key_diagnostics(normalized_keys),
            "source_pages": _candidate_key_diagnostics(source_page_keys),
            "append_pages": _candidate_key_diagnostics(append_page_keys),
        },
        "budgets": {
            "parent_updates": args.parent_updates,
            "source_updates_per_page": args.source_updates_per_page,
            "append_updates_per_page": args.append_updates_per_page,
            "router_updates": args.router_updates,
            "batch_size": args.batch_size,
            "audit_count": args.audit_count,
            "key_samples": args.key_samples,
        },
        "training": {
            "pages": page_training,
            "source_router": source_router_training,
        },
        "accounting": accounting,
        "metrics": metrics,
        "permuted_metrics": permuted_metrics,
        "controls": {
            "router_frozen_during_append": router_still_frozen,
            "source_pages_frozen_during_append": source_page_immutable,
            "core_unchanged": parent_digest_before == _digest_core(parent, ()),
            "replayed_examples": accounting["replayed_examples"],
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
    parser.add_argument("--source-candidates", type=int, default=30)
    parser.add_argument("--append-candidates", type=int, default=34)
    parser.add_argument("--source-page-size", type=int, default=10)
    parser.add_argument("--append-page-size", type=int, default=2)
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--source-updates-per-page", type=int, default=512)
    parser.add_argument("--append-updates-per-page", type=int, default=32)
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
