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


def _global_generation_metrics(
    predictions: torch.Tensor,
    slots: torch.Tensor,
    targets: torch.Tensor,
    generation_candidate_start: int,
    generation_page_start: int,
    generation_page_size: int,
    generation_page_order: list[int],
) -> dict[str, object]:
    """Measure one router whose page slots span every append generation."""

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


def _page_metrics_from_selected_pages(
    predictions: torch.Tensor,
    selected_pages: torch.Tensor,
    targets: torch.Tensor,
    generation_candidate_start: int,
    generation_page_start: int,
    generation_page_size: int,
    generation_page_order: list[int],
) -> dict[str, object]:
    """Measure predictions when the caller already resolved physical pages."""

    target_pages = generation_page_start + (
        (targets - generation_candidate_start) // generation_page_size
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


def _train_shared_local_router(
    router: LearnedComputeCandidateScreen,
    parent,
    grammar,
    normalized_generation_keys: list[torch.Tensor],
    raw_generation_keys: list[torch.Tensor],
    generation_pages: list[list[LearnedComputeCandidateScreen]],
    generation_candidate_starts: list[int],
    *,
    page_size: int,
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    normalizer: OpaqueCandidateSignatureNormalizer,
    query_offsets: torch.Tensor | None = None,
    query_adapters: torch.nn.ModuleList | None = None,
    shuffle_outcomes: bool = False,
) -> dict[str, int | float]:
    """Train one local-page router across generations without flat interference."""

    if not normalized_generation_keys or len(normalized_generation_keys) != len(
        generation_pages
    ):
        raise ValueError("shared local router generations must align")
    page_count = len(generation_pages[0])
    if page_count < 1 or any(len(pages) != page_count for pages in generation_pages):
        raise ValueError("shared local router needs equal page counts")
    if any(
        keys.shape[0] != page_count * page_size
        for keys in normalized_generation_keys
    ):
        raise ValueError("shared local router keys must align with pages")
    if query_offsets is not None and query_offsets.shape != (
        len(generation_pages),
        EVENT_WIDTH,
    ):
        raise ValueError("shared local router query offsets have the wrong shape")
    if query_adapters is not None and len(query_adapters) != len(generation_pages):
        raise ValueError("shared local router adapters must align with generations")
    trainable = list(router.parameters())
    if query_offsets is not None:
        trainable.append(query_offsets)
    if query_adapters is not None:
        trainable.extend(query_adapters.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1e-5
    )
    informative_pairs = 0
    last_loss = 0.0
    generation_count = len(generation_pages)
    for update in range(updates):
        generation_index = update % generation_count
        candidate_start = generation_candidate_starts[generation_index]
        families = [
            candidate_start
            + (update * batch_size + row) % (page_count * page_size)
            for row in range(batch_size)
        ]
        raw_query = _event_query(
            parent,
            grammar,
            families,
            seed=seed + update * 10_007,
        )
        page_outcomes = torch.zeros(batch_size, page_count)
        with torch.no_grad():
            for page_index, page in enumerate(generation_pages[generation_index]):
                start = page_index * page_size
                end = start + page_size
                local_predictions = page(
                    raw_query,
                    raw_generation_keys[generation_index][start:end],
                ).argmax(dim=-1)
                local_targets = torch.tensor(
                    [family - candidate_start - start for family in families],
                    dtype=torch.long,
                )
                page_outcomes[:, page_index] = (
                    local_predictions == local_targets
                ).float()
        if shuffle_outcomes:
            shuffled = []
            for row in range(batch_size):
                permutation = torch.randperm(
                    page_count,
                    generator=torch.Generator().manual_seed(
                        seed + update * 101 + row
                    ),
                )
                shuffled.append(page_outcomes[row][permutation])
            page_outcomes = torch.stack(shuffled)
        candidate_outcomes = page_outcomes.repeat_interleave(page_size, dim=1)
        query = normalizer(raw_query)
        if query_adapters is not None:
            query = query_adapters[generation_index](query)
        if query_offsets is not None:
            query = query + query_offsets[generation_index]
        if not bool(router.enabled.item()):
            router.enable()
        loss, pairs = router.outcome_ranking_loss(
            query,
            normalized_generation_keys[generation_index],
            candidate_outcomes,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        informative_pairs += pairs
        last_loss = float(loss.detach())
    router.eval()
    return {
        "optimizer_updates": updates,
        "unique_verifier_bits": updates * batch_size * page_count,
        "unique_logical_lifetimes": updates * batch_size * page_count,
        "informative_candidate_pairs": informative_pairs,
        "informative_outcomes": updates * batch_size * page_count,
        "replayed_examples": 0,
        "final_loss": last_loss,
    }


def _train_generation_selector(
    selector: LearnedComputeCandidateScreen,
    parent,
    grammar,
    generation_selector_keys: torch.Tensor,
    generation_candidate_starts: list[int],
    generation_candidate_count: int,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    normalizer: OpaqueCandidateSignatureNormalizer,
    shuffle_outcomes: bool = False,
) -> dict[str, int | float]:
    """Learn generation addressing separately from local page addressing."""

    optimizer = torch.optim.AdamW(
        selector.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    generation_count = len(generation_candidate_starts)
    informative_pairs = 0
    last_loss = 0.0
    for update in range(updates):
        generation_index = update % generation_count
        families = [
            generation_candidate_starts[generation_index]
            + (update * batch_size + row) % generation_candidate_count
            for row in range(batch_size)
        ]
        raw_query = _event_query(
            parent,
            grammar,
            families,
            seed=seed + update * 10_007,
        )
        outcomes = torch.zeros(
            batch_size, generation_count * generation_candidate_count
        )
        start = generation_index * generation_candidate_count
        outcomes[:, start : start + generation_candidate_count] = 1.0
        if shuffle_outcomes:
            for row in range(batch_size):
                permutation = torch.randperm(
                    generation_count,
                    generator=torch.Generator().manual_seed(
                        seed + update * 101 + row
                    ),
                )
                blocks = outcomes[row].reshape(
                    generation_count, generation_candidate_count
                )
                outcomes[row] = blocks[permutation].reshape(-1)
        if not bool(selector.enabled.item()):
            selector.enable()
        loss, pairs = selector.outcome_ranking_loss(
            normalizer(raw_query), generation_selector_keys, outcomes
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(selector.parameters(), 1.0)
        optimizer.step()
        informative_pairs += pairs
        last_loss = float(loss.detach())
    selector.eval()
    return {
        "optimizer_updates": updates,
        "unique_verifier_bits": updates * batch_size * generation_count,
        "unique_logical_lifetimes": updates * batch_size * generation_count,
        "informative_candidate_pairs": informative_pairs,
        "informative_outcomes": updates * batch_size * generation_count,
        "replayed_examples": 0,
        "final_loss": last_loss,
    }


def _factorized_route(
    selector: LearnedComputeCandidateScreen,
    local_router: LearnedComputeCandidateScreen,
    generation_selector_keys: torch.Tensor,
    normalized_generation_keys: list[torch.Tensor],
    raw_generation_keys: list[torch.Tensor],
    generation_pages: list[list[LearnedComputeCandidateScreen]],
    generation_candidate_starts: list[int],
    generation_page_starts: list[int],
    generation_page_ids: list[list[int]],
    query: torch.Tensor,
    normalizer: OpaqueCandidateSignatureNormalizer,
    page_size: int,
    query_offsets: torch.Tensor | None = None,
    query_adapters: torch.nn.ModuleList | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route through a generation selector and one shared local router."""

    generation_scores = selector(
        normalizer(query), generation_selector_keys
    ).reshape(query.shape[0], len(generation_pages), -1)
    generation_slots = generation_scores.amax(dim=-1).argmax(dim=-1)
    predictions = torch.empty(query.shape[0], dtype=torch.long)
    selected_pages = torch.empty(query.shape[0], dtype=torch.long)
    for generation_index in range(len(generation_pages)):
        rows = generation_slots == generation_index
        if not rows.any():
            continue
        local_query = normalizer(query[rows])
        if query_adapters is not None:
            local_query = query_adapters[generation_index](local_query)
        if query_offsets is not None:
            local_query = local_query + query_offsets[generation_index]
        candidate_scores = local_router(
            local_query,
            normalized_generation_keys[generation_index],
        )
        local_page_scores = candidate_scores.reshape(
            query[rows].shape[0], len(generation_pages[generation_index]), page_size
        ).amax(dim=-1)
        local_slots = local_page_scores.argmax(dim=-1)
        for local_page in range(len(generation_pages[generation_index])):
            local_rows = local_slots == local_page
            if not local_rows.any():
                continue
            page = generation_pages[generation_index][local_page]
            start = local_page * page_size
            end = start + page_size
            local_predictions = page(
                query[rows][local_rows], raw_generation_keys[generation_index][start:end]
            ).argmax(dim=-1)
            physical_page = generation_page_ids[generation_index][local_page]
            predictions[rows.nonzero().flatten()[local_rows]] = (
                generation_candidate_starts[generation_index]
                + (physical_page - generation_page_starts[generation_index])
                * page_size
                + local_predictions
            )
            selected_pages[rows.nonzero().flatten()[local_rows]] = physical_page
    return predictions, selected_pages


def _shared_local_generation_route(
    local_router: LearnedComputeCandidateScreen,
    normalized_keys: torch.Tensor,
    raw_keys: torch.Tensor,
    pages: list[LearnedComputeCandidateScreen],
    candidate_start: int,
    page_start: int,
    page_ids: list[int],
    query: torch.Tensor,
    normalizer: OpaqueCandidateSignatureNormalizer,
    page_size: int,
    query_offsets: torch.Tensor | None = None,
    query_adapter: torch.nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the shared local router for one external generation binding."""

    local_query = normalizer(query)
    if query_adapter is not None:
        local_query = query_adapter(local_query)
    if query_offsets is not None:
        local_query = local_query + query_offsets
    scores = local_router(local_query, normalized_keys)
    page_scores = scores.reshape(query.shape[0], len(pages), page_size).amax(dim=-1)
    local_slots = page_scores.argmax(dim=-1)
    predictions = torch.empty(query.shape[0], dtype=torch.long)
    selected_pages = torch.empty(query.shape[0], dtype=torch.long)
    for local_page in range(len(pages)):
        rows = local_slots == local_page
        if not rows.any():
            continue
        start = local_page * page_size
        end = start + page_size
        local_predictions = pages[local_page](
            query[rows], raw_keys[start:end]
        ).argmax(dim=-1)
        physical_page = page_ids[local_page]
        predictions[rows] = (
            candidate_start
            + (physical_page - page_start) * page_size
            + local_predictions
        )
        selected_pages[rows] = physical_page
    return predictions, selected_pages


def _shared_local_cascade_route(
    local_router: LearnedComputeCandidateScreen,
    normalized_generation_keys: list[torch.Tensor],
    raw_generation_keys: list[torch.Tensor],
    generation_pages: list[list[LearnedComputeCandidateScreen]],
    generation_candidate_starts: list[int],
    generation_page_starts: list[int],
    generation_page_ids: list[list[int]],
    query: torch.Tensor,
    targets: torch.Tensor,
    normalizer: OpaqueCandidateSignatureNormalizer,
    page_size: int,
    query_offsets: torch.Tensor | None = None,
    query_adapters: torch.nn.ModuleList | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cascade one shared compute core through generation-local bindings."""

    predictions = torch.empty(query.shape[0], dtype=torch.long)
    selected_pages = torch.empty(query.shape[0], dtype=torch.long)
    attempts = torch.zeros(query.shape[0])
    remaining = torch.ones(query.shape[0], dtype=torch.bool)
    for generation_index in range(len(generation_pages)):
        rows = remaining.clone()
        if not rows.any():
            break
        generation_predictions, generation_pages_selected = (
            _shared_local_generation_route(
                local_router,
                normalized_generation_keys[generation_index],
                raw_generation_keys[generation_index],
                generation_pages[generation_index],
                generation_candidate_starts[generation_index],
                generation_page_starts[generation_index],
                generation_page_ids[generation_index],
                query[rows],
                normalizer,
                page_size,
                None
                if query_offsets is None
                else query_offsets[generation_index],
                None
                if query_adapters is None
                else query_adapters[generation_index],
            )
        )
        indices = rows.nonzero().flatten()
        predictions[indices] = generation_predictions
        selected_pages[indices] = generation_pages_selected
        attempts[indices] += 1.0
        remaining[indices] = generation_predictions != targets[indices]
    return predictions, selected_pages, attempts, remaining


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.generations < 2:
        raise ValueError("this audit requires at least two append generations")
    if args.consolidate_generations and args.factorized_consolidation:
        raise ValueError("choose flat or factorized consolidation, not both")
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
        args.consolidation_router_updates,
        args.factorized_local_router_updates,
        args.factorized_selector_updates,
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
    normalizer.fit(
        raw_keys[: args.source_candidates]
        if args.normalizer_fit == "source"
        else raw_keys
    )
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

    consolidated_router = None
    shuffled_consolidated_router = None
    consolidated_training = None
    shuffled_consolidated_training = None
    consolidated_page_order = list(range(source_page_count, len(pages)))
    if args.consolidate_generations:
        consolidated_router = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.consolidation_latent_width,
            hidden=args.consolidation_hidden,
        )
        consolidated_training = _train_token_page_router(
            consolidated_router,
            parent,
            grammar,
            normalized_keys[args.source_candidates :],
            raw_keys[args.source_candidates :],
            pages[source_page_count:],
            page_size=args.generation_page_size,
            family_offset=args.source_candidates,
            updates=args.consolidation_router_updates,
            batch_size=args.batch_size,
            seed=args.seed + 160_000,
            learning_rate=args.learning_rate,
            normalizer=normalizer,
        )
        shuffled_consolidated_router = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.consolidation_latent_width,
            hidden=args.consolidation_hidden,
        )
        shuffled_consolidated_training = _train_token_page_router(
            shuffled_consolidated_router,
            parent,
            grammar,
            normalized_keys[args.source_candidates :],
            raw_keys[args.source_candidates :],
            pages[source_page_count:],
            page_size=args.generation_page_size,
            family_offset=args.source_candidates,
            updates=args.consolidation_router_updates,
            batch_size=args.batch_size,
            seed=args.seed + 170_000,
            learning_rate=args.learning_rate,
            normalizer=normalizer,
            shuffle_outcomes=True,
        )

    factorized_selector = None
    factorized_local_router = None
    shuffled_factorized_selector = None
    shuffled_factorized_local_router = None
    factorized_selector_training = None
    factorized_local_training = None
    shuffled_factorized_selector_training = None
    factorized_query_offsets = None
    shuffled_factorized_query_offsets = None
    factorized_query_adapters = None
    shuffled_factorized_query_adapters = None
    shuffled_factorized_local_training = None
    factorized_generation_keys = generation_keys
    factorized_raw_keys = [
        raw_keys[start : start + args.generation_candidates]
        for start in generation_candidate_starts
    ]
    factorized_pages = [
        pages[start : start + generation_page_count]
        for start in generation_page_starts
    ]
    factorized_page_ids = [
        list(range(start, start + generation_page_count))
        for start in generation_page_starts
    ]
    if args.factorized_consolidation:
        if args.factorized_query_offset:
            factorized_query_offsets = torch.nn.Parameter(
                torch.zeros(args.generations, EVENT_WIDTH)
            )
            shuffled_factorized_query_offsets = torch.nn.Parameter(
                torch.zeros(args.generations, EVENT_WIDTH)
            )
        if args.factorized_query_adapter:
            factorized_query_adapters = torch.nn.ModuleList(
                [
                    torch.nn.Linear(EVENT_WIDTH, EVENT_WIDTH)
                    for _ in range(args.generations)
                ]
            )
            shuffled_factorized_query_adapters = torch.nn.ModuleList(
                [
                    torch.nn.Linear(EVENT_WIDTH, EVENT_WIDTH)
                    for _ in range(args.generations)
                ]
            )
            for adapters in (
                factorized_query_adapters,
                shuffled_factorized_query_adapters,
            ):
                for adapter in adapters:
                    torch.nn.init.eye_(adapter.weight)
                    torch.nn.init.zeros_(adapter.bias)
        generation_selector_keys = torch.cat(factorized_generation_keys, dim=0)
        factorized_local_router = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.factorized_latent_width,
            hidden=args.factorized_hidden,
        )
        factorized_local_training = _train_shared_local_router(
            factorized_local_router,
            parent,
            grammar,
            factorized_generation_keys,
            factorized_raw_keys,
            factorized_pages,
            generation_candidate_starts,
            page_size=args.generation_page_size,
            updates=args.factorized_local_router_updates,
            batch_size=args.batch_size,
            seed=args.seed + 200_000,
            learning_rate=args.learning_rate,
            normalizer=normalizer,
            query_offsets=factorized_query_offsets,
            query_adapters=factorized_query_adapters,
        )
        factorized_selector = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.factorized_selector_latent_width,
            hidden=args.factorized_selector_hidden,
        )
        factorized_selector_training = _train_generation_selector(
            factorized_selector,
            parent,
            grammar,
            generation_selector_keys,
            generation_candidate_starts,
            args.generation_candidates,
            updates=args.factorized_selector_updates,
            batch_size=args.batch_size,
            seed=args.seed + 210_000,
            learning_rate=args.learning_rate,
            normalizer=normalizer,
        )
        shuffled_factorized_local_router = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.factorized_latent_width,
            hidden=args.factorized_hidden,
        )
        shuffled_factorized_local_training = _train_shared_local_router(
            shuffled_factorized_local_router,
            parent,
            grammar,
            factorized_generation_keys,
            factorized_raw_keys,
            factorized_pages,
            generation_candidate_starts,
            page_size=args.generation_page_size,
            updates=args.factorized_local_router_updates,
            batch_size=args.batch_size,
            seed=args.seed + 220_000,
            learning_rate=args.learning_rate,
            normalizer=normalizer,
            query_offsets=shuffled_factorized_query_offsets,
            query_adapters=shuffled_factorized_query_adapters,
            shuffle_outcomes=True,
        )
        shuffled_factorized_selector = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.factorized_selector_latent_width,
            hidden=args.factorized_selector_hidden,
        )
        shuffled_factorized_selector_training = _train_generation_selector(
            shuffled_factorized_selector,
            parent,
            grammar,
            generation_selector_keys,
            generation_candidate_starts,
            args.generation_candidates,
            updates=args.factorized_selector_updates,
            batch_size=args.batch_size,
            seed=args.seed + 230_000,
            learning_rate=args.learning_rate,
            normalizer=normalizer,
            shuffle_outcomes=True,
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
    generation_audit_batches = []
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
        generation_audit_batches.append((generation_queries, generation_targets))
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

    consolidated_metrics = []
    shuffled_consolidated_metrics = []
    consolidated_permuted_metrics = None
    if consolidated_router is not None and shuffled_consolidated_router is not None:
        for generation_index, (generation_queries, generation_targets) in enumerate(
            generation_audit_batches
        ):
            candidate_start = generation_candidate_starts[generation_index]
            consolidated_predictions, consolidated_slots = _token_append_route(
                consolidated_router,
                pages,
                generation_queries,
                normalized_keys[args.source_candidates :],
                raw_keys[args.source_candidates :],
                normalizer,
                args.source_candidates,
                source_page_count,
                args.generation_page_size,
                consolidated_page_order,
            )
            consolidated_metrics.append(
                _global_generation_metrics(
                    consolidated_predictions,
                    consolidated_slots,
                    generation_targets,
                    candidate_start,
                    source_page_count,
                    args.generation_page_size,
                    consolidated_page_order,
                )
            )
            shuffled_predictions, shuffled_slots = _token_append_route(
                shuffled_consolidated_router,
                pages,
                generation_queries,
                normalized_keys[args.source_candidates :],
                raw_keys[args.source_candidates :],
                normalizer,
                args.source_candidates,
                source_page_count,
                args.generation_page_size,
                consolidated_page_order,
            )
            shuffled_consolidated_metrics.append(
                _global_generation_metrics(
                    shuffled_predictions,
                    shuffled_slots,
                    generation_targets,
                    candidate_start,
                    source_page_count,
                    args.generation_page_size,
                    consolidated_page_order,
                )
            )

        all_generation_queries = torch.cat(
            [batch[0] for batch in generation_audit_batches], dim=0
        )
        all_generation_targets = torch.cat(
            [batch[1] for batch in generation_audit_batches], dim=0
        )
        consolidation_permutation = torch.randperm(
            len(consolidated_page_order),
            generator=torch.Generator().manual_seed(args.seed + 180_000),
        )
        permuted_consolidated_keys = normalized_keys[args.source_candidates :].reshape(
            len(consolidated_page_order), args.generation_page_size, EVENT_WIDTH
        )[consolidation_permutation].reshape_as(
            normalized_keys[args.source_candidates :]
        )
        permuted_consolidated_order = [
            consolidated_page_order[index]
            for index in consolidation_permutation.tolist()
        ]
        consolidated_predictions, consolidated_slots = _token_append_route(
            consolidated_router,
            pages,
            all_generation_queries,
            permuted_consolidated_keys,
            raw_keys[args.source_candidates :],
            normalizer,
            args.source_candidates,
            source_page_count,
            args.generation_page_size,
            permuted_consolidated_order,
        )
        consolidated_permuted_metrics = _global_generation_metrics(
            consolidated_predictions,
            consolidated_slots,
            all_generation_targets,
            args.source_candidates,
            source_page_count,
            args.generation_page_size,
            permuted_consolidated_order,
        )

    factorized_metrics = []
    shuffled_factorized_metrics = []
    factorized_permuted_metrics = None
    factorized_local_metrics = []
    shuffled_factorized_local_metrics = []
    factorized_local_permuted_metrics = None
    factorized_cascade_metrics = None
    shuffled_factorized_cascade_metrics = None
    if (
        factorized_selector is not None
        and factorized_local_router is not None
        and shuffled_factorized_selector is not None
        and shuffled_factorized_local_router is not None
    ):
        generation_selector_keys = torch.cat(factorized_generation_keys, dim=0)
        for generation_index, (generation_queries, generation_targets) in enumerate(
            generation_audit_batches
        ):
            factorized_predictions, factorized_pages_selected = _factorized_route(
                factorized_selector,
                factorized_local_router,
                generation_selector_keys,
                factorized_generation_keys,
                factorized_raw_keys,
                factorized_pages,
                generation_candidate_starts,
                generation_page_starts,
                factorized_page_ids,
                generation_queries,
                normalizer,
                args.generation_page_size,
                factorized_query_offsets,
                factorized_query_adapters,
            )
            factorized_metrics.append(
                _page_metrics_from_selected_pages(
                    factorized_predictions,
                    factorized_pages_selected,
                    generation_targets,
                    generation_candidate_starts[generation_index],
                    generation_page_starts[generation_index],
                    args.generation_page_size,
                    factorized_page_ids[generation_index],
                )
            )
            shuffled_predictions, shuffled_pages_selected = _factorized_route(
                shuffled_factorized_selector,
                shuffled_factorized_local_router,
                generation_selector_keys,
                factorized_generation_keys,
                factorized_raw_keys,
                factorized_pages,
                generation_candidate_starts,
                generation_page_starts,
                factorized_page_ids,
                generation_queries,
                normalizer,
                args.generation_page_size,
                shuffled_factorized_query_offsets,
                shuffled_factorized_query_adapters,
            )
            shuffled_factorized_metrics.append(
                _page_metrics_from_selected_pages(
                    shuffled_predictions,
                    shuffled_pages_selected,
                    generation_targets,
                    generation_candidate_starts[generation_index],
                    generation_page_starts[generation_index],
                    args.generation_page_size,
                    factorized_page_ids[generation_index],
                )
            )
        generation_permutation = torch.randperm(
            args.generations,
            generator=torch.Generator().manual_seed(args.seed + 240_000),
        ).tolist()
        permuted_generation_keys = [
            factorized_generation_keys[index] for index in generation_permutation
        ]
        permuted_factorized_raw_keys = [
            factorized_raw_keys[index] for index in generation_permutation
        ]
        permuted_factorized_pages = [
            factorized_pages[index] for index in generation_permutation
        ]
        permuted_generation_starts = [
            generation_candidate_starts[index] for index in generation_permutation
        ]
        permuted_page_starts = [
            generation_page_starts[index] for index in generation_permutation
        ]
        permuted_page_ids = [
            factorized_page_ids[index] for index in generation_permutation
        ]
        permuted_selector_keys = torch.cat(permuted_generation_keys, dim=0)
        all_generation_queries = torch.cat(
            [batch[0] for batch in generation_audit_batches], dim=0
        )
        all_generation_targets = torch.cat(
            [batch[1] for batch in generation_audit_batches], dim=0
        )
        factorized_predictions, factorized_pages_selected = _factorized_route(
            factorized_selector,
            factorized_local_router,
            permuted_selector_keys,
            permuted_generation_keys,
            permuted_factorized_raw_keys,
            permuted_factorized_pages,
            permuted_generation_starts,
            permuted_page_starts,
            permuted_page_ids,
            all_generation_queries,
            normalizer,
            args.generation_page_size,
            factorized_query_offsets,
            factorized_query_adapters,
        )
        factorized_permuted_metrics = _page_metrics_from_selected_pages(
            factorized_predictions,
            factorized_pages_selected,
            all_generation_targets,
            args.source_candidates,
            source_page_count,
            args.generation_page_size,
            consolidated_page_order,
        )

        if args.factorized_verifier_cascade:
            for generation_index, (generation_queries, generation_targets) in enumerate(
                generation_audit_batches
            ):
                local_predictions, local_pages_selected = (
                    _shared_local_generation_route(
                        factorized_local_router,
                        factorized_generation_keys[generation_index],
                        factorized_raw_keys[generation_index],
                        factorized_pages[generation_index],
                        generation_candidate_starts[generation_index],
                        generation_page_starts[generation_index],
                        factorized_page_ids[generation_index],
                        generation_queries,
                        normalizer,
                        args.generation_page_size,
                        None
                        if factorized_query_offsets is None
                        else factorized_query_offsets[generation_index],
                        None
                        if factorized_query_adapters is None
                        else factorized_query_adapters[generation_index],
                    )
                )
                factorized_local_metrics.append(
                    _page_metrics_from_selected_pages(
                        local_predictions,
                        local_pages_selected,
                        generation_targets,
                        generation_candidate_starts[generation_index],
                        generation_page_starts[generation_index],
                        args.generation_page_size,
                        factorized_page_ids[generation_index],
                    )
                )
                shuffled_predictions, shuffled_pages_selected = (
                    _shared_local_generation_route(
                        shuffled_factorized_local_router,
                        factorized_generation_keys[generation_index],
                        factorized_raw_keys[generation_index],
                        factorized_pages[generation_index],
                        generation_candidate_starts[generation_index],
                        generation_page_starts[generation_index],
                        factorized_page_ids[generation_index],
                        generation_queries,
                        normalizer,
                        args.generation_page_size,
                        None
                        if shuffled_factorized_query_offsets is None
                        else shuffled_factorized_query_offsets[generation_index],
                        None
                        if shuffled_factorized_query_adapters is None
                        else shuffled_factorized_query_adapters[generation_index],
                    )
                )
                shuffled_factorized_local_metrics.append(
                    _page_metrics_from_selected_pages(
                        shuffled_predictions,
                        shuffled_pages_selected,
                        generation_targets,
                        generation_candidate_starts[generation_index],
                        generation_page_starts[generation_index],
                        args.generation_page_size,
                        factorized_page_ids[generation_index],
                    )
                )
            page_permutations = [
                torch.randperm(
                    generation_page_count,
                    generator=torch.Generator().manual_seed(
                        args.seed + 250_000 + generation_index
                    ),
                ).tolist()
                for generation_index in range(args.generations)
            ]
            all_local_predictions = []
            all_local_pages = []
            all_local_targets = []
            for generation_index, (generation_queries, generation_targets) in enumerate(
                generation_audit_batches
            ):
                permutation = page_permutations[generation_index]
                permuted_keys = factorized_generation_keys[generation_index].reshape(
                    generation_page_count,
                    args.generation_page_size,
                    EVENT_WIDTH,
                )[permutation].reshape_as(factorized_generation_keys[generation_index])
                permuted_raw_keys = factorized_raw_keys[generation_index].reshape(
                    generation_page_count,
                    args.generation_page_size,
                    EVENT_WIDTH,
                )[permutation].reshape_as(factorized_raw_keys[generation_index])
                permuted_pages = [
                    factorized_pages[generation_index][index] for index in permutation
                ]
                permuted_page_ids = [
                    factorized_page_ids[generation_index][index]
                    for index in permutation
                ]
                local_predictions, local_pages_selected = (
                    _shared_local_generation_route(
                        factorized_local_router,
                        permuted_keys,
                        permuted_raw_keys,
                        permuted_pages,
                        generation_candidate_starts[generation_index],
                        generation_page_starts[generation_index],
                        permuted_page_ids,
                        generation_queries,
                        normalizer,
                        args.generation_page_size,
                        None
                        if factorized_query_offsets is None
                        else factorized_query_offsets[generation_index],
                        None
                        if factorized_query_adapters is None
                        else factorized_query_adapters[generation_index],
                    )
                )
                all_local_predictions.append(local_predictions)
                all_local_pages.append(local_pages_selected)
                all_local_targets.append(generation_targets)
            factorized_local_permuted_metrics = _page_metrics_from_selected_pages(
                torch.cat(all_local_predictions),
                torch.cat(all_local_pages),
                torch.cat(all_local_targets),
                args.source_candidates,
                source_page_count,
                args.generation_page_size,
                consolidated_page_order,
            )
            all_generation_queries = torch.cat(
                [batch[0] for batch in generation_audit_batches], dim=0
            )
            all_generation_targets = torch.cat(
                [batch[1] for batch in generation_audit_batches], dim=0
            )
            cascade_predictions, cascade_pages, cascade_attempts, cascade_unresolved = (
                _shared_local_cascade_route(
                    factorized_local_router,
                    factorized_generation_keys,
                    factorized_raw_keys,
                    factorized_pages,
                    generation_candidate_starts,
                    generation_page_starts,
                    factorized_page_ids,
                    all_generation_queries,
                    all_generation_targets,
                    normalizer,
                    args.generation_page_size,
                    factorized_query_offsets,
                    factorized_query_adapters,
                )
            )
            factorized_cascade_metrics = _page_metrics_from_selected_pages(
                cascade_predictions,
                cascade_pages,
                all_generation_targets,
                args.source_candidates,
                source_page_count,
                args.generation_page_size,
                consolidated_page_order,
            )
            factorized_cascade_metrics["mean_fresh_attempts"] = float(
                cascade_attempts.mean()
            )
            factorized_cascade_metrics["no_unresolved_rows"] = not bool(
                cascade_unresolved.any()
            )
            shuffled_cascade_predictions, shuffled_cascade_pages, _, shuffled_unresolved = (
                _shared_local_cascade_route(
                    shuffled_factorized_local_router,
                    factorized_generation_keys,
                    factorized_raw_keys,
                    factorized_pages,
                    generation_candidate_starts,
                    generation_page_starts,
                    factorized_page_ids,
                    all_generation_queries,
                    all_generation_targets,
                    normalizer,
                    args.generation_page_size,
                    shuffled_factorized_query_offsets,
                    shuffled_factorized_query_adapters,
                )
            )
            shuffled_factorized_cascade_metrics = _page_metrics_from_selected_pages(
                shuffled_cascade_predictions,
                shuffled_cascade_pages,
                all_generation_targets,
                args.source_candidates,
                source_page_count,
                args.generation_page_size,
                consolidated_page_order,
            )
            shuffled_factorized_cascade_metrics["no_unresolved_rows"] = not bool(
                shuffled_unresolved.any()
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
            *(
                [consolidated_training, shuffled_consolidated_training]
                if args.consolidate_generations
                else []
            ),
            *(
                [
                    factorized_local_training,
                    factorized_selector_training,
                    shuffled_factorized_local_training,
                    shuffled_factorized_selector_training,
                ]
                if args.factorized_consolidation
                else []
            ),
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
    if args.consolidate_generations:
        gates.update(
            {
                "consolidated_generation_mastery": all(
                    row["top1_accuracy"] >= MASTERY_THRESHOLD
                    and min(row["per_target_top1_accuracy"])
                    >= MASTERY_THRESHOLD
                    and row["page_top1_accuracy"] >= MASTERY_THRESHOLD
                    and min(row["per_page_top1_accuracy"])
                    >= MASTERY_THRESHOLD
                    for row in consolidated_metrics
                ),
                "consolidated_reward_shuffled_null": all(
                    row["page_top1_accuracy"]
                    <= (1.0 / len(consolidated_page_order)) + 0.15
                    for row in shuffled_consolidated_metrics
                ),
                "consolidated_permutation": (
                    consolidated_permuted_metrics is not None
                    and consolidated_permuted_metrics["top1_accuracy"]
                    >= MASTERY_THRESHOLD
                    and consolidated_permuted_metrics["page_top1_accuracy"]
                    >= MASTERY_THRESHOLD
                ),
                "router_count_reduced": 1 < len(generation_routers),
                "consolidated_no_replayed_examples": (
                    consolidated_training is not None
                    and shuffled_consolidated_training is not None
                    and consolidated_training["replayed_examples"] == 0
                    and shuffled_consolidated_training["replayed_examples"] == 0
                ),
            }
        )
    if args.factorized_consolidation:
        factorized_eval_metrics = (
            factorized_local_metrics
            if args.factorized_verifier_cascade
            else factorized_metrics
        )
        factorized_eval_shuffled_metrics = (
            shuffled_factorized_local_metrics
            if args.factorized_verifier_cascade
            else shuffled_factorized_metrics
        )
        factorized_eval_permuted_metrics = (
            factorized_local_permuted_metrics
            if args.factorized_verifier_cascade
            else factorized_permuted_metrics
        )
        gates.update(
            {
                "factorized_generation_mastery": all(
                    row["top1_accuracy"] >= MASTERY_THRESHOLD
                    and min(row["per_target_top1_accuracy"])
                    >= MASTERY_THRESHOLD
                    and row["page_top1_accuracy"] >= MASTERY_THRESHOLD
                    and min(row["per_page_top1_accuracy"])
                    >= MASTERY_THRESHOLD
                    for row in factorized_eval_metrics
                ),
                "factorized_reward_shuffled_null": all(
                    row["page_top1_accuracy"]
                    <= (1.0 / generation_page_count) + 0.15
                    for row in factorized_eval_shuffled_metrics
                ),
                "factorized_permutation": (
                    factorized_eval_permuted_metrics is not None
                    and factorized_eval_permuted_metrics["top1_accuracy"]
                    >= MASTERY_THRESHOLD
                    and factorized_eval_permuted_metrics["page_top1_accuracy"]
                    >= MASTERY_THRESHOLD
                ),
                "factorized_router_count_reduced": (
                    2 < len(generation_routers)
                ),
                "factorized_no_replayed_examples": all(
                    training is not None
                    and training["replayed_examples"] == 0
                    for training in (
                        factorized_local_training,
                        factorized_selector_training,
                        shuffled_factorized_local_training,
                        shuffled_factorized_selector_training,
                    )
                ),
                "factorized_cascade_retention": (
                    not args.factorized_verifier_cascade
                    or (
                        factorized_cascade_metrics is not None
                        and factorized_cascade_metrics["top1_accuracy"]
                        >= MASTERY_THRESHOLD
                        and factorized_cascade_metrics["no_unresolved_rows"]
                        and shuffled_factorized_cascade_metrics is not None
                    )
                ),
            }
        )
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
        "normalizer_fit": args.normalizer_fit,
        "consolidation": {
            "enabled": args.consolidate_generations,
            "router_count_before": len(generation_routers),
            "router_count_after": 1 if args.consolidate_generations else None,
            "updates": args.consolidation_router_updates
            if args.consolidate_generations
            else None,
            "latent_width": args.consolidation_latent_width
            if args.consolidate_generations
            else None,
            "hidden": args.consolidation_hidden
            if args.consolidate_generations
            else None,
            "training": consolidated_training,
            "reward_shuffled_training": shuffled_consolidated_training,
            "metrics": consolidated_metrics,
            "reward_shuffled_metrics": shuffled_consolidated_metrics,
            "permuted_metrics": consolidated_permuted_metrics,
        },
        "factorized_consolidation": {
            "enabled": args.factorized_consolidation,
            "router_count_before": len(generation_routers),
            "router_count_after": 2 if args.factorized_consolidation else None,
            "local_router_updates": args.factorized_local_router_updates
            if args.factorized_consolidation
            else None,
            "selector_updates": args.factorized_selector_updates
            if args.factorized_consolidation
            else None,
            "query_offset": args.factorized_query_offset,
            "query_adapter": args.factorized_query_adapter,
            "local_training": factorized_local_training,
            "selector_training": factorized_selector_training,
            "reward_shuffled_local_training": shuffled_factorized_local_training,
            "reward_shuffled_selector_training": shuffled_factorized_selector_training,
            "metrics": factorized_metrics,
            "reward_shuffled_metrics": shuffled_factorized_metrics,
            "permuted_metrics": factorized_permuted_metrics,
            "local_metrics": factorized_local_metrics,
            "local_reward_shuffled_metrics": shuffled_factorized_local_metrics,
            "local_permuted_metrics": factorized_local_permuted_metrics,
            "verifier_cascade_enabled": args.factorized_verifier_cascade,
            "verifier_cascade_metrics": factorized_cascade_metrics,
            "verifier_cascade_reward_shuffled_metrics": (
                shuffled_factorized_cascade_metrics
            ),
        },
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
    parser.add_argument("--consolidate-generations", action="store_true")
    parser.add_argument("--consolidation-router-updates", type=int, default=3072)
    parser.add_argument("--consolidation-latent-width", type=int, default=64)
    parser.add_argument("--consolidation-hidden", type=int, default=128)
    parser.add_argument("--factorized-consolidation", action="store_true")
    parser.add_argument("--factorized-local-router-updates", type=int, default=3072)
    parser.add_argument("--factorized-selector-updates", type=int, default=512)
    parser.add_argument("--factorized-latent-width", type=int, default=32)
    parser.add_argument("--factorized-hidden", type=int, default=64)
    parser.add_argument("--factorized-selector-latent-width", type=int, default=16)
    parser.add_argument("--factorized-selector-hidden", type=int, default=32)
    parser.add_argument("--factorized-query-offset", action="store_true")
    parser.add_argument("--factorized-query-adapter", action="store_true")
    parser.add_argument("--factorized-verifier-cascade", action="store_true")
    parser.add_argument(
        "--normalizer-fit",
        choices=("all", "source"),
        default="all",
    )
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
