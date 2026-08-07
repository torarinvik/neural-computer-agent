"""Audit repeated no-replay generations of external page memory.

The source router is trained once. Each append generation receives an
independent token-preserving page router trained only on that generation's
scalar verifier outcomes. Inference cascades through the frozen source router
and then the append overlays, advancing only after verifier failure.
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
    _merge_training_accounting,
    _per_target_top1_accuracy,
)
from experiments.compute_candidate_screen_amodal.train_page_local_source_sharded import (
    _train_local_screen,
)
from experiments.compute_candidate_screen_amodal.train_page_router import (
    _train_page_router,
)
from experiments.compute_candidate_screen_amodal.train_page_router_append import (
    _mixed_metrics,
    _token_append_route,
    _train_token_page_router,
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


def _source_route(
    router: LearnedComputeCandidateScreen,
    pages: list[LearnedComputeCandidateScreen],
    query: torch.Tensor,
    source_page_keys: torch.Tensor,
    raw_keys: torch.Tensor,
    normalizer: OpaqueCandidateSignatureNormalizer,
    source_page_size: int,
    page_order: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = router(normalizer(query), source_page_keys).argmax(dim=-1)
    predictions = torch.empty(slots.shape[0], dtype=torch.long)
    for slot, physical_page in enumerate(page_order):
        rows = slots == slot
        if not rows.any():
            continue
        start = physical_page * source_page_size
        end = start + source_page_size
        predictions[rows] = start + pages[physical_page](
            normalizer(query[rows]), normalizer(raw_keys[start:end])
        ).argmax(dim=-1)
    return predictions, slots


def _cascade_route(
    source_router: LearnedComputeCandidateScreen,
    generation_routers: list[LearnedComputeCandidateScreen],
    pages: list[LearnedComputeCandidateScreen],
    query: torch.Tensor,
    targets: torch.Tensor,
    source_page_keys: torch.Tensor,
    generation_keys: list[torch.Tensor],
    raw_keys: torch.Tensor,
    normalizer: OpaqueCandidateSignatureNormalizer,
    source_page_size: int,
    generation_page_size: int,
    source_page_count: int,
    source_page_order: list[int],
    generation_page_starts: list[int],
    generation_candidate_starts: list[int],
    generation_page_orders: list[list[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cascade through learned external overlays using scalar failure gates."""

    predictions, source_slots = _source_route(
        source_router,
        pages,
        query,
        source_page_keys,
        raw_keys,
        normalizer,
        source_page_size,
        source_page_order,
    )
    selected_slots = torch.tensor(
        [source_page_order[slot] for slot in source_slots.tolist()]
    )
    attempts = torch.ones(query.shape[0])
    remaining = predictions != targets
    for generation_index, router in enumerate(generation_routers):
        if not remaining.any():
            break
        rows = remaining.clone()
        generation_predictions, generation_slots = _token_append_route(
            router,
            pages,
            query[rows],
            generation_keys[generation_index],
            raw_keys[generation_candidate_starts[generation_index] :],
            normalizer,
            generation_candidate_starts[generation_index],
            generation_page_starts[generation_index],
            generation_page_size,
            generation_page_orders[generation_index],
        )
        predictions[rows] = generation_predictions
        selected_slots[rows] = torch.tensor(
            [generation_page_orders[generation_index][slot] for slot in generation_slots.tolist()]
        )
        attempts[rows] += 1.0
        remaining[rows] = generation_predictions != targets[rows]
    return predictions, selected_slots, attempts, remaining


def _generation_metrics(
    predictions: torch.Tensor,
    slots: torch.Tensor,
    targets: torch.Tensor,
    generation_candidate_start: int,
    generation_page_start: int,
    generation_page_size: int,
    generation_page_order: list[int],
) -> dict[str, object]:
    target_pages = generation_page_start + (
        (targets - generation_candidate_start) // generation_page_size
    )
    selected_pages = torch.tensor(
        [generation_page_order[slot] for slot in slots.tolist()]
    )
    return {
        "top1_accuracy": float((predictions == targets).float().mean()),
        "per_target_top1_accuracy": _per_target_top1_accuracy(
            predictions, targets
        ),
        "page_top1_accuracy": float((selected_pages == target_pages).float().mean()),
        "per_page_top1_accuracy": [
            float(
                (selected_pages[target_pages == page] == page)
                .float()
                .mean()
            )
            for page in generation_page_order
        ],
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.generations < 2:
        raise ValueError("this audit requires at least two append generations")
    if args.source_candidates % args.source_page_size:
        raise ValueError("source candidates must form equal pages")
    if args.generation_candidates % args.generation_page_size:
        raise ValueError("generation candidates must form equal pages")
    if min(
        args.parent_updates,
        args.source_updates_per_page,
        args.generation_updates_per_page,
        args.source_router_updates,
        args.generation_router_updates,
        args.batch_size,
        args.audit_count,
        args.key_samples,
    ) < 1:
        raise ValueError("all budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")

    source_page_count = args.source_candidates // args.source_page_size
    generation_page_count = args.generation_candidates // args.generation_page_size
    total_candidates = args.source_candidates + (
        args.generations * args.generation_candidates
    )
    generation_candidate_starts = [
        args.source_candidates + index * args.generation_candidates
        for index in range(args.generations)
    ]
    generation_page_starts = [
        source_page_count + index * generation_page_count
        for index in range(args.generations)
    ]
    page_sizes = [args.source_page_size] + []
    page_sizes = [args.source_page_size] * source_page_count + [
        args.generation_page_size
    ] * (args.generations * generation_page_count)
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
    page_training = []
    for page_index, (start, end) in enumerate(_page_ranges(page_sizes)):
        is_source = page_index < source_page_count
        page = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
        )
        pages.append(page)
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
                    else args.generation_updates_per_page
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
        pages[:source_page_count],
        page_size=args.source_page_size,
        source_count=args.source_candidates,
        updates=args.source_router_updates,
        batch_size=args.batch_size,
        seed=args.seed + 50_000,
        learning_rate=args.learning_rate,
        normalizer=normalizer,
    )
    source_router_state = {
        name: value.detach().clone()
        for name, value in source_router.state_dict().items()
    }
    source_page_states = [
        {
            name: value.detach().clone()
            for name, value in page.state_dict().items()
        }
        for page in pages[:source_page_count]
    ]

    generation_routers = []
    shuffled_generation_routers = []
    generation_training = []
    shuffled_generation_training = []
    generation_keys = []
    for generation_index in range(args.generations):
        candidate_start = generation_candidate_starts[generation_index]
        candidate_end = candidate_start + args.generation_candidates
        page_start = generation_page_starts[generation_index]
        page_end = page_start + generation_page_count
        keys = normalized_keys[candidate_start:candidate_end]
        generation_keys.append(keys)
        router = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
        )
        generation_routers.append(router)
        generation_training.append(
            _train_token_page_router(
                router,
                parent,
                grammar,
                keys,
                raw_keys[candidate_start:candidate_end],
                pages[page_start:page_end],
                page_size=args.generation_page_size,
                family_offset=candidate_start,
                updates=args.generation_router_updates,
                batch_size=args.batch_size,
                seed=args.seed + 60_000 + generation_index * 20_000,
                learning_rate=args.learning_rate,
                normalizer=normalizer,
            )
        )
        shuffled = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
        )
        shuffled_generation_routers.append(shuffled)
        shuffled_generation_training.append(
            _train_token_page_router(
                shuffled,
                parent,
                grammar,
                keys,
                raw_keys[candidate_start:candidate_end],
                pages[page_start:page_end],
                page_size=args.generation_page_size,
                family_offset=candidate_start,
                updates=args.generation_router_updates,
                batch_size=args.batch_size,
                seed=args.seed + 70_000 + generation_index * 20_000,
                learning_rate=args.learning_rate,
                normalizer=normalizer,
                shuffle_outcomes=True,
            )
        )

    audit_families = [index % total_candidates for index in range(args.audit_count)]
    queries = _event_query(parent, grammar, audit_families, seed=args.seed + 80_000)
    targets = torch.tensor(audit_families)
    generation_orders = [
        list(range(start, start + generation_page_count))
        for start in generation_page_starts
    ]
    predictions, selected_slots, attempts, unresolved = _cascade_route(
        source_router,
        generation_routers,
        pages,
        queries,
        targets,
        source_page_keys,
        generation_keys,
        raw_keys,
        normalizer,
        args.source_page_size,
        args.generation_page_size,
        source_page_count,
        list(range(source_page_count)),
        generation_page_starts,
        generation_candidate_starts,
        generation_orders,
    )
    metrics = _mixed_metrics(
        predictions,
        selected_slots,
        targets,
        list(range(len(pages))),
        page_sizes,
    )
    metrics["mean_fresh_attempts"] = float(attempts.mean())

    generation_metrics = []
    shuffled_generation_metrics = []
    for generation_index in range(args.generations):
        candidate_start = generation_candidate_starts[generation_index]
        candidate_families = [
            candidate_start + index % args.generation_candidates
            for index in range(args.audit_count)
        ]
        generation_queries = _event_query(
            parent,
            grammar,
            candidate_families,
            seed=args.seed + 90_000 + generation_index * 10_000,
        )
        generation_targets = torch.tensor(candidate_families)
        generation_predictions, generation_slots = _token_append_route(
            generation_routers[generation_index],
            pages,
            generation_queries,
            generation_keys[generation_index],
            raw_keys[candidate_start:],
            normalizer,
            candidate_start,
            generation_page_starts[generation_index],
            args.generation_page_size,
            generation_orders[generation_index],
        )
        generation_metrics.append(
            _generation_metrics(
                generation_predictions,
                generation_slots,
                generation_targets,
                candidate_start,
                generation_page_starts[generation_index],
                args.generation_page_size,
                generation_orders[generation_index],
            )
        )
        shuffled_predictions, shuffled_slots = _token_append_route(
            shuffled_generation_routers[generation_index],
            pages,
            generation_queries,
            generation_keys[generation_index],
            raw_keys[candidate_start:],
            normalizer,
            candidate_start,
            generation_page_starts[generation_index],
            args.generation_page_size,
            generation_orders[generation_index],
        )
        shuffled_generation_metrics.append(
            _generation_metrics(
                shuffled_predictions,
                shuffled_slots,
                generation_targets,
                candidate_start,
                generation_page_starts[generation_index],
                args.generation_page_size,
                generation_orders[generation_index],
            )
        )

    source_permutation = torch.randperm(
        source_page_count,
        generator=torch.Generator().manual_seed(args.seed + 100_000),
    )
    permuted_source_keys = source_page_keys[source_permutation]
    permuted_generation_keys = []
    permuted_generation_orders = []
    for generation_index in range(args.generations):
        permutation = torch.randperm(
            generation_page_count,
            generator=torch.Generator().manual_seed(
                args.seed + 110_000 + generation_index
            ),
        )
        keys = generation_keys[generation_index].reshape(
            generation_page_count, args.generation_page_size, EVENT_WIDTH
        )
        permuted_generation_keys.append(keys[permutation].reshape_as(generation_keys[generation_index]))
        page_start = generation_page_starts[generation_index]
        permuted_generation_orders.append(
            [page_start + index for index in permutation.tolist()]
        )
    permuted_predictions, permuted_slots, permuted_attempts, permuted_unresolved = _cascade_route(
        source_router,
        generation_routers,
        pages,
        queries,
        targets,
        permuted_source_keys,
        permuted_generation_keys,
        raw_keys,
        normalizer,
        args.source_page_size,
        args.generation_page_size,
        source_page_count,
        source_permutation.tolist(),
        generation_page_starts,
        generation_candidate_starts,
        permuted_generation_orders,
    )
    permuted_metrics = _mixed_metrics(
        permuted_predictions,
        permuted_slots,
        targets,
        list(range(len(pages))),
        page_sizes,
    )
    permuted_metrics["mean_fresh_attempts"] = float(permuted_attempts.mean())
    source_router_immutable = all(
        torch.equal(value, source_router.state_dict()[name])
        for name, value in source_router_state.items()
    )
    source_pages_immutable = all(
        torch.equal(page.state_dict()[name], value)
        for page, saved in zip(
            pages[:source_page_count], source_page_states, strict=True
        )
        for name, value in saved.items()
    )
    parent_accounting = parent_history[-1] if parent_history else {}
    training_accounting = _merge_training_accounting(
        [
            *page_training,
            source_router_training,
            *generation_training,
            *shuffled_generation_training,
        ]
    )
    accounting = {
        "unique_verifier_bits": int(parent_accounting.get("unique_verifier_bits", 0))
        + int(training_accounting["unique_verifier_bits"]),
        "unique_logical_lifetimes": int(
            parent_accounting.get("unique_logical_lifetimes", 0)
        )
        + int(training_accounting["unique_logical_lifetimes"]),
        "optimizer_updates": args.parent_updates
        + int(training_accounting["optimizer_updates"]),
        "replayed_examples": int(training_accounting["replayed_examples"]),
        "wall_seconds": perf_counter() - started,
    }
    gates = {
        "parent_stable": any(
            float(row["heldout_accuracy"]) >= MASTERY_THRESHOLD
            for row in parent_progress
        ),
        "candidate_generalization": metrics["top1_accuracy"]
        >= MASTERY_THRESHOLD,
        "candidate_per_target_mastery": min(metrics["per_target_top1_accuracy"])
        >= MASTERY_THRESHOLD,
        "page_generalization": metrics["page_top1_accuracy"]
        >= MASTERY_THRESHOLD,
        "page_per_page_mastery": min(metrics["per_page_top1_accuracy"])
        >= MASTERY_THRESHOLD,
        "permutation": permuted_metrics["top1_accuracy"] >= MASTERY_THRESHOLD
        and permuted_metrics["page_top1_accuracy"] >= MASTERY_THRESHOLD,
        "generation_mastery": all(
            row["top1_accuracy"] >= MASTERY_THRESHOLD
            and min(row["per_target_top1_accuracy"]) >= MASTERY_THRESHOLD
            and row["page_top1_accuracy"] >= MASTERY_THRESHOLD
            and min(row["per_page_top1_accuracy"]) >= MASTERY_THRESHOLD
            for row in generation_metrics
        ),
        "generation_reward_shuffled_null": all(
            row["page_top1_accuracy"]
            <= (1.0 / generation_page_count) + 0.15
            for row in shuffled_generation_metrics
        ),
        "verifier_cascade_used": bool((attempts > 1.0).any()),
        "source_router_immutable": source_router_immutable,
        "source_pages_immutable": source_pages_immutable,
        "no_unresolved_audit_rows": not bool(unresolved.any()),
        "no_permuted_unresolved_rows": not bool(permuted_unresolved.any()),
        "core_unchanged": parent_digest_before == _digest_core(parent, ()),
        "no_replayed_examples": accounting["replayed_examples"] == 0,
    }
    report = {
        "schema": "neural-computer.opaque-page-router-generations-report.v1",
        "claim_boundary": (
            "Independent token-preserving append routers retrieve repeated "
            "external generations through verifier-gated cascading while the "
            "source router/controller remain frozen. This is bounded repeated "
            "growth, not general continual learning."
        ),
        "seed": args.seed,
        "candidate_count": total_candidates,
        "source_candidate_count": args.source_candidates,
        "generation_count": args.generations,
        "generation_candidate_count": args.generation_candidates,
        "source_page_count": source_page_count,
        "generation_page_count": generation_page_count,
        "page_sizes": page_sizes,
        "representation": "normalized_opaque_candidate_tokens_v1",
        "learning_signal": "scalar_verifier_outcome_of_attempted_generation_page_v1",
        "growth_policy": "independent_generation_overlay_without_prior_replay_v1",
        "candidate_key_diagnostics": {
            "raw": _candidate_key_diagnostics(raw_keys),
            "normalized": _candidate_key_diagnostics(normalized_keys),
        },
        "budgets": {
            "parent_updates": args.parent_updates,
            "source_updates_per_page": args.source_updates_per_page,
            "generation_updates_per_page": args.generation_updates_per_page,
            "source_router_updates": args.source_router_updates,
            "generation_router_updates": args.generation_router_updates,
            "batch_size": args.batch_size,
            "audit_count": args.audit_count,
            "key_samples": args.key_samples,
        },
        "training": {
            "pages": page_training,
            "source_router": source_router_training,
            "generation_routers": generation_training,
            "reward_shuffled_generation_routers": shuffled_generation_training,
        },
        "accounting": accounting,
        "metrics": metrics,
        "generation_metrics": generation_metrics,
        "reward_shuffled_generation_metrics": shuffled_generation_metrics,
        "permuted_metrics": permuted_metrics,
        "fallback_rate": float((attempts > 1.0).float().mean()),
        "mean_fresh_attempts": float(attempts.mean()),
        "controls": {
            "source_router_immutable": source_router_immutable,
            "source_pages_immutable": source_pages_immutable,
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
    parser.add_argument("--generation-candidates", type=int, default=18)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--source-page-size", type=int, default=10)
    parser.add_argument("--generation-page-size", type=int, default=2)
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--source-updates-per-page", type=int, default=512)
    parser.add_argument("--generation-updates-per-page", type=int, default=32)
    parser.add_argument("--source-router-updates", type=int, default=512)
    parser.add_argument("--generation-router-updates", type=int, default=3072)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--audit-count", type=int, default=132)
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
                "generation_metrics": report["generation_metrics"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
