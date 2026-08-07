"""Audit source-page sharding behind the page-local memory boundary.

The initial source bank is divided into independent normalized pages. A later
source page is exposed only after cumulative scalar verifier failure for the
earlier page, exactly like an appended page. Unseen pages retain the raw
representation-matched prior from the page-local growth audit.

The experiment tests whether source competition, rather than scorer width or
signature rank, is the 46-candidate bottleneck. Page membership is physical
memory organization; no task or semantic labels enter the controller or the
page screens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.compute_candidate_screen_amodal.train import (
    EVENT_WIDTH,
    _append_only_permuted_accuracy,
    _append_only_route_metrics,
    _candidate_key_diagnostics,
    _candidate_keys,
    _event_query,
    _merge_training_accounting,
    _per_target_top1_accuracy,
    _train_append_only_extension,
    _train_screen,
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
    PageLocalLearnedComputeCandidateScreen,
)

MASTERY_THRESHOLD = 0.75
DEFAULT_CANDIDATES = 46
DEFAULT_UNSEEN = 26
DEFAULT_SOURCE_PAGE_SIZE = 10
DEFAULT_STAGES = 13


def _all_targets_clear(metrics: dict[str, object]) -> bool:
    values = metrics["per_target_top1_accuracy"]
    return isinstance(values, list) and bool(values) and min(values) >= MASTERY_THRESHOLD


def _train_local_screen(
    screen: LearnedComputeCandidateScreen,
    parent,
    grammar,
    keys: torch.Tensor,
    families: list[int],
    *,
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    query_transform,
    shuffle_outcomes: bool = False,
) -> dict[str, int | float]:
    """Train a source page using local opaque candidate indices only."""

    if not families:
        raise ValueError("source page requires at least one family")
    if keys.ndim != 2 or keys.shape[0] != len(families):
        raise ValueError("source page keys must align with its families")
    optimizer = torch.optim.AdamW(
        screen.parameters(),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    informative_pairs = 0
    last_loss = 0.0
    for update in range(updates):
        batch_families = [
            families[(update * batch_size + row) % len(families)]
            for row in range(batch_size)
        ]
        query = _event_query(
            parent,
            grammar,
            batch_families,
            seed=seed + update * 10_007,
        )
        query = query_transform(query)
        outcomes = torch.zeros(batch_size, len(families))
        local_targets = [families.index(family) for family in batch_families]
        outcomes[torch.arange(batch_size), torch.tensor(local_targets)] = 1.0
        if shuffle_outcomes:
            shuffled = []
            for row in range(batch_size):
                permutation = torch.randperm(
                    len(families),
                    generator=torch.Generator().manual_seed(
                        seed + update * 101 + row
                    ),
                )
                shuffled.append(outcomes[row][permutation])
            outcomes = torch.stack(shuffled)
        if not bool(screen.enabled.item()):
            screen.enable()
        loss, pairs = screen.outcome_ranking_loss(query, keys, outcomes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(screen.parameters(), 1.0)
        optimizer.step()
        informative_pairs += pairs
        last_loss = float(loss.detach())
    screen.eval()
    return {
        "optimizer_updates": updates,
        "unique_verifier_bits": updates * batch_size * len(families),
        "unique_logical_lifetimes": updates * batch_size * len(families),
        "informative_candidate_pairs": informative_pairs,
        "replayed_examples": 0,
        "final_loss": last_loss,
    }


def _source_failure_schedule(
    targets: torch.Tensor,
    *,
    source_count: int,
    source_page_size: int,
    stage_sizes: list[int],
) -> torch.Tensor:
    source_page_count = source_count // source_page_size
    extension_count = source_page_count - 1 + len(stage_sizes)
    schedule = torch.zeros(
        targets.shape[0],
        extension_count,
        dtype=torch.bool,
    )
    for row, target in enumerate(targets.tolist()):
        if target < source_count:
            page_index = target // source_page_size
            if page_index > 0:
                schedule[row, : page_index] = True
            continue
        stage_offset = target - source_count
        stage_index = 0
        cumulative = stage_sizes[0]
        while stage_offset >= cumulative:
            stage_index += 1
            cumulative += stage_sizes[stage_index]
        schedule[row, : source_page_count - 1 + stage_index + 1] = True
    return schedule


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.candidate_count < 4 or args.candidate_count % 2:
        raise ValueError("candidate-count must be an even number of at least four")
    if args.unseen_candidates < 1 or args.unseen_candidates >= args.candidate_count:
        raise ValueError("unseen-candidates must be between one and candidate-count")
    source_count = args.candidate_count - args.unseen_candidates
    if source_count % args.source_page_size:
        raise ValueError("source candidate count must divide into equal pages")
    if args.append_only_stages < 1 or args.append_only_stages > args.unseen_candidates:
        raise ValueError("append-only-stages must fit within unseen candidates")
    if min(
        args.parent_updates,
        args.source_updates_per_page,
        args.raw_template_updates,
        args.calibration_updates,
        args.batch_size,
        args.audit_count,
        args.key_samples,
    ) < 1:
        raise ValueError("all training and audit budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")
    unseen_size, remainder = divmod(
        args.unseen_candidates,
        args.append_only_stages,
    )
    stage_sizes = [unseen_size] * args.append_only_stages
    for index in range(remainder):
        stage_sizes[-(index + 1)] += 1
    source_page_count = source_count // args.source_page_size
    known_families = list(range(source_count))
    unseen_families = list(range(source_count, args.candidate_count))
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

    source_pages: list[LearnedComputeCandidateScreen] = []
    source_page_training = []
    for page_index in range(source_page_count):
        start = page_index * args.source_page_size
        families = list(range(start, start + args.source_page_size))
        page = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
        )
        source_pages.append(page)
        source_page_training.append(
            _train_local_screen(
                page,
                parent,
                grammar,
                normalized_keys[start : start + args.source_page_size],
                families,
                updates=args.source_updates_per_page,
                batch_size=args.batch_size,
                seed=args.seed + 30_000 + page_index * 10_000,
                learning_rate=args.learning_rate,
                query_transform=normalizer,
            )
        )
    raw_template = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    raw_template_training = _train_screen(
        raw_template,
        parent,
        grammar,
        raw_keys,
        families=known_families,
        updates=args.raw_template_updates,
        batch_size=args.batch_size,
        seed=args.seed + 60_000,
        learning_rate=args.learning_rate,
    )

    raw_container = PageLocalLearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
        base_screen=raw_template,
    )
    for size in stage_sizes:
        raw_container.append_extension(size)
    raw_container.freeze_base()
    for index in range(len(stage_sizes)):
        raw_container.initialize_extension_from_template(index, raw_template)
    extension_training = []
    start = 0
    for stage_index, extension_size in enumerate(stage_sizes):
        extension_training.append(
            _train_append_only_extension(
                raw_container,
                parent,
                grammar,
                raw_keys[source_count + start : source_count + start + extension_size],
                extension_index=stage_index,
                families=unseen_families[start : start + extension_size],
                updates=args.calibration_updates,
                batch_size=args.batch_size,
                seed=args.seed + 81_000 + stage_index * 10_000,
                learning_rate=args.learning_rate,
                probe_families=known_families + unseen_families,
            )
        )
        start += extension_size
    extension_accounting = _merge_training_accounting(extension_training)

    shuffled_page = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    shuffled_training = _train_local_screen(
        shuffled_page,
        parent,
        grammar,
        normalized_keys[: args.source_page_size],
        list(range(args.source_page_size)),
        updates=args.source_updates_per_page,
        batch_size=args.batch_size,
        seed=args.seed + 70_000,
        learning_rate=args.learning_rate,
        query_transform=normalizer,
        shuffle_outcomes=True,
    )

    page_screen = PageLocalLearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
        base_screen=source_pages[0],
        base_query_view=normalizer,
        base_key_view=normalizer,
    )
    for page in source_pages[1:]:
        page_screen.append_extension(
            args.source_page_size,
            screen=page,
            query_view=normalizer,
            key_view=normalizer,
        )
    for extension, extension_size in zip(
        raw_container.extensions,
        stage_sizes,
        strict=True,
    ):
        page_screen.append_extension(
            extension_size,
            screen=extension,
        )
    page_screen.freeze_base()
    base_state_before_append = {
        name: value.detach().clone()
        for name, value in source_pages[0].state_dict().items()
    }
    source_extension_states_before_append = [
        {
            name: value.detach().clone()
            for name, value in page.state_dict().items()
        }
        for page in source_pages[1:]
    ]

    known_query_families = [
        known_families[index % len(known_families)]
        for index in range(args.audit_count)
    ]
    unseen_query_families = [
        unseen_families[index % len(unseen_families)]
        for index in range(args.audit_count)
    ]
    known_queries = _event_query(
        parent,
        grammar,
        known_query_families,
        seed=args.seed + 80_000,
    )
    unseen_queries = _event_query(
        parent,
        grammar,
        unseen_query_families,
        seed=args.seed + 90_000,
    )
    known_targets = torch.tensor(known_query_families)
    unseen_targets = torch.tensor(unseen_query_families)
    known_failures = _source_failure_schedule(
        known_targets,
        source_count=source_count,
        source_page_size=args.source_page_size,
        stage_sizes=stage_sizes,
    )
    unseen_failures = _source_failure_schedule(
        unseen_targets,
        source_count=source_count,
        source_page_size=args.source_page_size,
        stage_sizes=stage_sizes,
    )
    base_keys = raw_keys[: args.source_page_size]
    extension_keys = raw_keys[args.source_page_size :]
    known_metrics = _append_only_route_metrics(
        page_screen,
        known_queries,
        known_targets,
        base_keys,
        extension_keys,
        failed_extensions=known_failures,
    )
    unseen_metrics = _append_only_route_metrics(
        page_screen,
        unseen_queries,
        unseen_targets,
        base_keys,
        extension_keys,
        failed_extensions=unseen_failures,
    )
    known_permutation_accuracy = _append_only_permuted_accuracy(
        page_screen,
        known_queries,
        known_targets,
        base_keys,
        extension_keys,
        seed=args.seed + 100_000,
        failed_extensions=known_failures,
    )
    unseen_permutation_accuracy = _append_only_permuted_accuracy(
        page_screen,
        unseen_queries,
        unseen_targets,
        base_keys,
        extension_keys,
        seed=args.seed + 101_000,
        failed_extensions=unseen_failures,
    )

    shuffled_families = [index % args.source_page_size for index in range(args.audit_count)]
    shuffled_queries = _event_query(
        parent,
        grammar,
        shuffled_families,
        seed=args.seed + 110_000,
    )
    shuffled_predictions = shuffled_page(
        normalizer(shuffled_queries),
        normalized_keys[: args.source_page_size],
    ).argmax(dim=-1)
    shuffled_targets = torch.tensor(shuffled_families)
    shuffled_metrics = {
        "top1_accuracy": float(
            (shuffled_predictions == shuffled_targets).float().mean()
        ),
        "per_target_top1_accuracy": _per_target_top1_accuracy(
            shuffled_predictions,
            shuffled_targets,
        ),
    }
    cold_screen = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    cold_predictions = cold_screen(
        normalizer(known_queries),
        normalized_keys[:source_count],
    ).argmax(dim=-1)
    cold_metrics = {
        "top1_accuracy": float((cold_predictions == known_targets).float().mean()),
        "per_target_top1_accuracy": _per_target_top1_accuracy(
            cold_predictions,
            known_targets,
        ),
    }

    state = {
        name: value.detach().clone() for name, value in page_screen.state_dict().items()
    }
    restored_normalizer = OpaqueCandidateSignatureNormalizer(EVENT_WIDTH)
    restored_normalizer.fit(raw_keys)
    restored = PageLocalLearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
        base_screen=LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
        ),
        base_query_view=restored_normalizer,
        base_key_view=restored_normalizer,
    )
    for index in range(source_page_count - 1):
        restored.append_extension(
            args.source_page_size,
            query_view=restored_normalizer,
            key_view=restored_normalizer,
        )
    for size in stage_sizes:
        restored.append_extension(size)
    restored.load_state_dict(state, strict=True)
    reload_exact = all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in state.items()
    )
    reload_metrics = _append_only_route_metrics(
        restored,
        known_queries,
        known_targets,
        base_keys,
        extension_keys,
        failed_extensions=known_failures,
    )
    parent_digest_after = _digest_core(parent, ())
    source_immutable = all(
        torch.equal(value, source_pages[0].state_dict()[name])
        for name, value in base_state_before_append.items()
    ) and all(
        torch.equal(value, page.state_dict()[name])
        for page, saved in zip(
            source_pages[1:],
            source_extension_states_before_append,
            strict=True,
        )
        for name, value in saved.items()
    )
    source_runs = [*source_page_training, raw_template_training, shuffled_training]
    source_accounting = _merge_training_accounting(source_runs)
    parent_accounting = parent_history[-1] if parent_history else {}
    accounting = {
        "unique_verifier_bits": int(parent_accounting.get("unique_verifier_bits", 0))
        + int(source_accounting["unique_verifier_bits"])
        + int(extension_accounting["unique_verifier_bits"]),
        "unique_logical_lifetimes": int(
            parent_accounting.get("unique_logical_lifetimes", 0)
        )
        + int(source_accounting["unique_logical_lifetimes"])
        + int(extension_accounting["unique_logical_lifetimes"]),
        "optimizer_updates": args.parent_updates
        + int(source_accounting["optimizer_updates"])
        + int(extension_accounting["optimizer_updates"]),
        "replayed_examples": int(
            source_accounting["replayed_examples"]
            + extension_accounting["replayed_examples"]
        ),
        "wall_seconds": perf_counter() - started,
        "retention_floor": min(known_metrics["per_target_top1_accuracy"]),
    }
    gates = {
        "parent_stable": any(
            float(row["heldout_accuracy"]) >= MASTERY_THRESHOLD
            for row in parent_progress
        ),
        "known_context_generalization": known_metrics["top1_accuracy"]
        >= MASTERY_THRESHOLD,
        "known_context_per_candidate_mastery": _all_targets_clear(known_metrics),
        "unseen_candidate_acquired": unseen_metrics["top1_accuracy"]
        >= MASTERY_THRESHOLD,
        "unseen_candidate_per_candidate_mastery": _all_targets_clear(unseen_metrics),
        "known_context_beats_cold_start": known_metrics["top1_accuracy"]
        > cold_metrics["top1_accuracy"] + 0.15,
        "known_candidate_permutation": known_permutation_accuracy
        >= MASTERY_THRESHOLD,
        "unseen_candidate_permutation": unseen_permutation_accuracy
        >= MASTERY_THRESHOLD,
        "reward_shuffled_null": shuffled_metrics["top1_accuracy"]
        <= (1.0 / args.source_page_size) + 0.15,
        "source_pages_immutable_during_growth": source_immutable,
        "reload_exact": reload_exact,
        "reload_behavior_exact": reload_metrics == known_metrics,
        "core_unchanged": parent_digest_before == parent_digest_after,
        "no_replayed_examples": accounting["replayed_examples"] == 0,
    }
    report = {
        "schema": "neural-computer.page-local-source-sharded-report.v1",
        "claim_boundary": (
            "Independent normalized source pages reduce protected candidate "
            "competition behind cumulative scalar verifier gates while raw "
            "append pages retain representation-matched growth. This is a "
            "bounded source-sharding result, not general continual learning."
        ),
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "source_candidate_count": source_count,
        "unseen_candidate_count": args.unseen_candidates,
        "source_page_size": args.source_page_size,
        "source_page_count": source_page_count,
        "append_stage_sizes": stage_sizes,
        "representation_assignment": {
            "source_pages": "frozen_affine_normalized_v1",
            "unseen_pages": "raw_identity_v1",
            "selection": "physical_page_order_experiment_v1",
        },
        "candidate_key_normalizer": normalizer.configuration(),
        "candidate_key_diagnostics": {
            "raw": _candidate_key_diagnostics(raw_keys),
            "normalized": _candidate_key_diagnostics(normalized_keys),
        },
        "configuration": page_screen.configuration(),
        "budgets": {
            "parent_updates": args.parent_updates,
            "source_updates_per_page": args.source_updates_per_page,
            "raw_template_updates": args.raw_template_updates,
            "calibration_updates_per_stage": args.calibration_updates,
            "batch_size": args.batch_size,
            "audit_count": args.audit_count,
            "key_samples": args.key_samples,
        },
        "training": {
            "source_pages": source_page_training,
            "raw_template": raw_template_training,
            "reward_shuffled_page": shuffled_training,
            "unseen_pages": extension_accounting,
        },
        "accounting": accounting,
        "known_context_metrics": known_metrics,
        "unseen_candidate_metrics": unseen_metrics,
        "cold_metrics": cold_metrics,
        "reward_shuffled_metrics": shuffled_metrics,
        "known_candidate_permutation_accuracy": known_permutation_accuracy,
        "unseen_candidate_permutation_accuracy": unseen_permutation_accuracy,
        "controls": {
            "parent_digest_before": parent_digest_before,
            "parent_digest_after": parent_digest_after,
            "parent_unchanged": parent_digest_before == parent_digest_after,
            "reload_exact": reload_exact,
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
    parser.add_argument("--raw-template-updates", type=int, default=512)
    parser.add_argument("--calibration-updates", type=int, default=32)
    parser.add_argument("--append-only-stages", type=int, default=DEFAULT_STAGES)
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
                "known_context_metrics": report["known_context_metrics"],
                "unseen_candidate_metrics": report["unseen_candidate_metrics"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
