"""Audit generalizing opaque compute-candidate screening.

The screen receives learned event queries, opaque candidate keys, and scalar
outcomes for attempted candidates.  It is trained on one half of a candidate
bank and evaluated on held-out candidates and fresh query contexts.  The
screen only orders candidates; a private fresh verifier remains the authority
for whether the selected candidate is admissible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
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
    AppendOnlyLearnedComputeCandidateScreen,
    LearnedComputeCandidateScreen,
)

EVENT_WIDTH = 32
SPAN = 4
DEFAULT_CANDIDATES = 6
UNSEEN_CANDIDATES = 2
MASTERY_THRESHOLD = 0.75


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _event_query(
    parent,
    grammar,
    families: list[int],
    *,
    seed: int,
) -> torch.Tensor:
    """Create a query from rendered frames through the frozen event encoder."""

    if not families:
        raise ValueError("event query requires at least one family")
    encoder = parent.encoders["vision"]
    rows: list[torch.Tensor] = []
    for row, family in enumerate(families):
        batch = generate_sequence_memory_batch(
            2,
            span=SPAN,
            distractors=1,
            seed=seed + row * 1009,
            operation="generated_composition",
            generated_composition_ids=(family,),
            generated_compositions=grammar,
        )
        events = torch.stack(
            [encoder(frame[:1]) for frame in batch.query_frames.transpose(0, 1)],
            dim=1,
        )
        rows.append(F.normalize(events.mean(dim=(0, 1)), dim=0))
    return torch.stack(rows)


@torch.no_grad()
def _candidate_keys(
    parent,
    grammar,
    candidate_count: int,
    *,
    seed: int,
    samples_per_candidate: int,
) -> torch.Tensor:
    keys: list[torch.Tensor] = []
    for candidate in range(candidate_count):
        query = _event_query(
            parent,
            grammar,
            [candidate] * samples_per_candidate,
            seed=seed + candidate * 10_003,
        )
        keys.append(F.normalize(query.mean(dim=0), dim=0))
    return torch.stack(keys)


def _outcomes(families: list[int], candidate_count: int) -> torch.Tensor:
    result = torch.zeros(len(families), candidate_count)
    result[torch.arange(len(families)), torch.tensor(families)] = 1.0
    return result


@torch.no_grad()
def _route_metrics(
    screen: LearnedComputeCandidateScreen,
    queries: torch.Tensor,
    targets: torch.Tensor,
    keys: torch.Tensor,
) -> dict[str, float]:
    scores = screen(queries, keys)
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    top = order[:, 0]
    rank = (order == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "top1_accuracy": float((top == targets).float().mean()),
        "mean_fresh_attempts": float(rank.float().mean()),
        "fresh_verifier_authorized_rate": float((top == targets).float().mean()),
    }


@torch.no_grad()
def _append_only_route_metrics(
    screen: AppendOnlyLearnedComputeCandidateScreen,
    queries: torch.Tensor,
    targets: torch.Tensor,
    base_keys: torch.Tensor,
    extension_keys: torch.Tensor,
    *,
    failed_extensions: torch.Tensor | bool,
) -> dict[str, float]:
    """Measure an append-only screen after an explicit old-route outcome."""

    scores = screen(
        queries,
        base_keys,
        extension_keys,
        failed_extensions=failed_extensions,
    )
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    top = order[:, 0]
    rank = (order == targets.unsqueeze(1)).nonzero(as_tuple=False)[:, 1] + 1
    return {
        "top1_accuracy": float((top == targets).float().mean()),
        "mean_fresh_attempts": float(rank.float().mean()),
        "fresh_verifier_authorized_rate": float((top == targets).float().mean()),
        "failed_extension_gate": float(
            torch.as_tensor(failed_extensions, dtype=torch.float32).mean()
        ),
    }


@torch.no_grad()
def _permuted_accuracy(
    screen: LearnedComputeCandidateScreen,
    queries: torch.Tensor,
    targets: torch.Tensor,
    keys: torch.Tensor,
    *,
    seed: int,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(keys.shape[0], generator=generator)
    local_predictions = screen(queries, keys[permutation]).argmax(dim=-1)
    predictions = permutation[local_predictions]
    return float((predictions == targets).float().mean())


@torch.no_grad()
def _append_only_permuted_accuracy(
    screen: AppendOnlyLearnedComputeCandidateScreen,
    queries: torch.Tensor,
    targets: torch.Tensor,
    base_keys: torch.Tensor,
    extension_keys: torch.Tensor,
    *,
    seed: int,
    failed_extensions: torch.Tensor | bool,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    base_permutation = torch.randperm(base_keys.shape[0], generator=generator)
    extension_permutation_parts = []
    offset = 0
    for size in screen.extension_sizes:
        local = torch.randperm(size, generator=generator) + offset
        extension_permutation_parts.append(local)
        offset += size
    extension_permutation = torch.cat(extension_permutation_parts)
    scores = screen(
        queries,
        base_keys[base_permutation],
        extension_keys[extension_permutation],
        failed_extensions=failed_extensions,
    )
    local_predictions = scores.argmax(dim=-1)
    base_count = base_keys.shape[0]
    predictions = torch.where(
        local_predictions < base_count,
        base_permutation[local_predictions.clamp_max(base_count - 1)],
        base_count
        + extension_permutation[(local_predictions - base_count).clamp_min(0)],
    )
    return float((predictions == targets).float().mean())


def _train_screen(
    screen: LearnedComputeCandidateScreen,
    parent,
    grammar,
    keys: torch.Tensor,
    *,
    families: list[int],
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    shuffle_outcomes: bool = False,
) -> dict[str, int | float]:
    optimizer = torch.optim.AdamW(screen.parameters(), lr=learning_rate, weight_decay=1e-5)
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
        outcomes = _outcomes(batch_families, keys.shape[0])
        if shuffle_outcomes:
            shuffled = []
            for row in range(batch_size):
                permutation = torch.randperm(
                    keys.shape[0],
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
        "unique_verifier_bits": updates * batch_size * keys.shape[0],
        "unique_logical_lifetimes": updates * batch_size * keys.shape[0],
        "informative_candidate_pairs": informative_pairs,
        "replayed_examples": 0,
        "final_loss": last_loss,
    }


def _train_append_only_extension(
    screen: AppendOnlyLearnedComputeCandidateScreen,
    parent,
    grammar,
    extension_keys: torch.Tensor,
    *,
    extension_index: int,
    families: list[int],
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    probe_families: list[int] | None = None,
) -> dict[str, int | float]:
    """Train only one appended screen from fresh outcomes for its candidates."""

    extension = screen.extensions[extension_index]
    optimizer = torch.optim.AdamW(
        extension.parameters(), lr=learning_rate, weight_decay=1e-5
    )
    screen.enable_extension(extension_index)
    if len(families) == 1 and probe_families is None:
        raise ValueError("singleton extension training requires probe families")
    training_families = probe_families if len(families) == 1 else families
    if not training_families:
        raise ValueError("extension training requires probe families")
    local_family = {family: index for index, family in enumerate(families)}
    informative_pairs = 0
    informative_outcomes = 0
    last_loss = 0.0
    for update in range(updates):
        batch_families = [
            training_families[(update * batch_size + row) % len(training_families)]
            for row in range(batch_size)
        ]
        query = _event_query(
            parent,
            grammar,
            batch_families,
            seed=seed + update * 10_007,
        )
        if len(families) == 1:
            outcomes = torch.tensor(
                [float(family == families[0]) for family in batch_families]
            )
            loss, signals = extension.outcome_calibration_loss(
                query,
                extension_keys,
                torch.zeros(batch_size, dtype=torch.long),
                outcomes,
            )
            informative_outcomes += signals
        else:
            outcomes = torch.zeros(batch_size, len(families))
            outcomes[
                torch.arange(batch_size),
                torch.tensor([local_family[family] for family in batch_families]),
            ] = 1.0
            loss, signals = extension.outcome_ranking_loss(
                query,
                extension_keys,
                outcomes,
            )
            informative_pairs += signals
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(extension.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
    extension.eval()
    return {
        "optimizer_updates": updates,
        "unique_verifier_bits": updates * batch_size * extension_keys.shape[0],
        "unique_logical_lifetimes": updates * batch_size * extension_keys.shape[0],
        "informative_candidate_pairs": informative_pairs,
        "informative_outcomes": informative_outcomes,
        "replayed_examples": 0,
        "final_loss": last_loss,
    }


def _merge_training_accounting(
    runs: list[dict[str, int | float]],
) -> dict[str, int | float]:
    """Combine independent append-stage accounting without hiding any cost."""

    if not runs:
        raise ValueError("at least one accounting run is required")
    merged: dict[str, int | float] = {}
    for key in runs[0]:
        if key == "final_loss":
            merged[key] = runs[-1][key]
        else:
            merged[key] = sum(run[key] for run in runs)
    return merged


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.candidate_count < 4 or args.candidate_count % 2:
        raise ValueError("candidate-count must be an even number of at least four")
    if min(args.parent_updates, args.screen_updates, args.batch_size, args.audit_count) < 1:
        raise ValueError("all training and audit budgets must be positive")
    if args.calibration_updates < 0:
        raise ValueError("calibration-updates must be non-negative")
    if args.append_only_calibration and args.calibration_updates < 1:
        raise ValueError(
            "append-only-calibration requires positive calibration-updates"
        )
    if args.append_only_stages < 1:
        raise ValueError("append-only-stages must be positive")
    if args.append_only_inherit_base and args.append_only_prior_mode != "none":
        raise ValueError("choose one append-only prior mode")
    append_only_prior_mode = (
        "full" if args.append_only_inherit_base else args.append_only_prior_mode
    )
    if append_only_prior_mode != "none" and not args.append_only_calibration:
        raise ValueError("append-only prior mode requires append-only-calibration")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")
    if args.unseen_candidates < 1 or args.unseen_candidates >= args.candidate_count:
        raise ValueError("unseen-candidates must be between one and candidate-count")
    train_count = args.candidate_count - args.unseen_candidates
    known_families = list(range(train_count))
    unseen_families = list(range(train_count, args.candidate_count))
    if args.append_only_stages > len(unseen_families):
        raise ValueError("append-only-stages cannot exceed unseen candidates")
    base_stage_size, remainder = divmod(
        len(unseen_families), args.append_only_stages
    )
    append_stage_sizes = [base_stage_size] * args.append_only_stages
    for index in range(remainder):
        append_stage_sizes[-(index + 1)] += 1
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
    keys = _candidate_keys(
        parent,
        grammar,
        args.candidate_count,
        seed=args.seed + 20_000,
        samples_per_candidate=args.key_samples,
    )
    screen = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    training_accounting = _train_screen(
        screen,
        parent,
        grammar,
        keys,
        families=known_families,
        updates=args.screen_updates,
        batch_size=args.batch_size,
        seed=args.seed + 30_000,
        learning_rate=args.learning_rate,
    )
    shuffled_screen = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    shuffled_accounting = _train_screen(
        shuffled_screen,
        parent,
        grammar,
        keys,
        families=known_families,
        updates=args.screen_updates,
        batch_size=args.batch_size,
        seed=args.seed + 40_000,
        learning_rate=args.learning_rate,
        shuffle_outcomes=True,
    )
    train_query_families = [
        known_families[index % len(known_families)]
        for index in range(args.audit_count)
    ]
    novel_context_query_families = [
        known_families[index % len(known_families)]
        for index in range(args.audit_count)
    ]
    unseen_candidate_query_families = [
        unseen_families[index % len(unseen_families)]
        for index in range(args.audit_count)
    ]
    train_queries = _event_query(
        parent,
        grammar,
        train_query_families,
        seed=args.seed + 50_000,
    )
    novel_context_queries = _event_query(
        parent,
        grammar,
        novel_context_query_families,
        seed=args.seed + 60_000,
    )
    unseen_candidate_queries = _event_query(
        parent,
        grammar,
        unseen_candidate_query_families,
        seed=args.seed + 70_000,
    )
    train_targets = torch.tensor(train_query_families)
    novel_context_targets = torch.tensor(novel_context_query_families)
    unseen_candidate_targets = torch.tensor(unseen_candidate_query_families)
    train_metrics = _route_metrics(screen, train_queries, train_targets, keys)
    novel_context_metrics = _route_metrics(
        screen,
        novel_context_queries,
        novel_context_targets,
        keys,
    )
    unseen_candidate_metrics = _route_metrics(
        screen,
        unseen_candidate_queries,
        unseen_candidate_targets,
        keys,
    )
    initial_novel_context_metrics = novel_context_metrics
    initial_unseen_candidate_metrics = unseen_candidate_metrics
    calibration_accounting: dict[str, int | float] = {
        "optimizer_updates": 0,
        "unique_verifier_bits": 0,
        "unique_logical_lifetimes": 0,
        "informative_candidate_pairs": 0,
        "informative_outcomes": 0,
        "replayed_examples": 0,
        "final_loss": 0.0,
    }
    append_only_screen: AppendOnlyLearnedComputeCandidateScreen | None = None
    append_only_training: dict[str, int | float] | None = None
    append_only_unseen_pre_failure_metrics: dict[str, float] | None = None
    append_only_unseen_permuted_accuracy: float | None = None
    if args.append_only_calibration:
        known_keys = keys[:train_count]
        extension_keys = keys[train_count:]
        append_only_screen = AppendOnlyLearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
            extension_sizes=tuple(append_stage_sizes),
        )
        append_only_screen.base_screen.load_state_dict(screen.state_dict(), strict=True)
        if append_only_prior_mode != "none":
            for stage_index in range(args.append_only_stages):
                append_only_screen.initialize_extension_from_base(
                    stage_index,
                    mode=append_only_prior_mode,
                )
        append_only_screen.freeze_base()
        stage_accounting = []
        start = 0
        for stage_index, extension_size in enumerate(append_stage_sizes):
            stage_accounting.append(
                _train_append_only_extension(
                    append_only_screen,
                    parent,
                    grammar,
                    extension_keys[start : start + extension_size],
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
        calibration_accounting = _merge_training_accounting(stage_accounting)
        failure_schedule = torch.zeros(
            unseen_candidate_targets.shape[0],
            args.append_only_stages,
            dtype=torch.bool,
        )
        for row, target in enumerate(unseen_candidate_targets.tolist()):
            stage_offset = target - train_count
            stage_index = 0
            cumulative = append_stage_sizes[0]
            while stage_offset >= cumulative:
                stage_index += 1
                cumulative += append_stage_sizes[stage_index]
            failure_schedule[row, : stage_index + 1] = True
        novel_context_metrics = _append_only_route_metrics(
            append_only_screen,
            novel_context_queries,
            novel_context_targets,
            known_keys,
            extension_keys,
            failed_extensions=False,
        )
        append_only_unseen_pre_failure_metrics = _append_only_route_metrics(
            append_only_screen,
            unseen_candidate_queries,
            unseen_candidate_targets,
            known_keys,
            extension_keys,
            failed_extensions=False,
        )
        unseen_candidate_metrics = _append_only_route_metrics(
            append_only_screen,
            unseen_candidate_queries,
            unseen_candidate_targets,
            known_keys,
            extension_keys,
            failed_extensions=failure_schedule,
        )
        append_only_unseen_permuted_accuracy = _append_only_permuted_accuracy(
            append_only_screen,
            unseen_candidate_queries,
            unseen_candidate_targets,
            known_keys,
            extension_keys,
            seed=args.seed + 71_000,
            failed_extensions=failure_schedule,
        )
    elif args.calibration_updates:
        calibration_accounting = _train_screen(
            screen,
            parent,
            grammar,
            keys,
            families=unseen_families,
            updates=args.calibration_updates,
            batch_size=args.batch_size,
            seed=args.seed + 80_000,
            learning_rate=args.learning_rate,
        )
        novel_context_metrics = _route_metrics(
            screen,
            novel_context_queries,
            novel_context_targets,
            keys,
        )
        unseen_candidate_metrics = _route_metrics(
            screen,
            unseen_candidate_queries,
            unseen_candidate_targets,
            keys,
        )
    calibrated_novel_context_metrics = novel_context_metrics
    calibrated_unseen_candidate_metrics = unseen_candidate_metrics
    shuffled_novel_context_metrics = _route_metrics(
        shuffled_screen,
        novel_context_queries,
        novel_context_targets,
        keys,
    )
    if append_only_screen is None:
        permuted_accuracy = _permuted_accuracy(
            screen,
            novel_context_queries,
            novel_context_targets,
            keys,
            seed=args.seed + 70_000,
        )
    else:
        permuted_accuracy = _append_only_permuted_accuracy(
            append_only_screen,
            novel_context_queries,
            novel_context_targets,
            keys[:train_count],
            keys[train_count:],
            seed=args.seed + 70_000,
            failed_extensions=False,
        )
    cold_screen = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    cold_metrics = _route_metrics(
        cold_screen,
        novel_context_queries,
        novel_context_targets,
        keys,
    )
    evaluated_screen: torch.nn.Module = (
        append_only_screen if append_only_screen is not None else screen
    )
    screen_state = {
        name: value.detach().clone()
        for name, value in evaluated_screen.state_dict().items()
    }
    if append_only_screen is None:
        reloaded = LearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
        )
    else:
        reloaded = AppendOnlyLearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
            extension_sizes=tuple(append_stage_sizes),
        )
    reloaded.load_state_dict(screen_state, strict=True)
    reload_exact = all(
        torch.equal(value, reloaded.state_dict()[name])
        for name, value in screen_state.items()
    )
    if append_only_screen is None:
        reload_metrics = _route_metrics(
            reloaded,
            novel_context_queries,
            novel_context_targets,
            keys,
        )
    else:
        reload_metrics = _append_only_route_metrics(
            reloaded,
            novel_context_queries,
            novel_context_targets,
            keys[:train_count],
            keys[train_count:],
            failed_extensions=False,
        )
    parent_digest_after = _digest_core(parent, ())
    chance = 1.0 / args.candidate_count
    calibration_enabled = args.calibration_updates > 0
    gates = {
        "parent_stable": any(
            float(row["heldout_accuracy"]) >= MASTERY_THRESHOLD
            for row in parent_progress
        ),
        "novel_context_generalization": (
            novel_context_metrics["top1_accuracy"] >= MASTERY_THRESHOLD
        ),
        "novel_context_beats_cold_start": (
            novel_context_metrics["top1_accuracy"]
            > cold_metrics["top1_accuracy"] + 0.15
        ),
        "unseen_candidate_acquired": (
            calibrated_unseen_candidate_metrics["top1_accuracy"] >= MASTERY_THRESHOLD
            if calibration_enabled
            else True
        ),
        "known_context_retained_after_calibration": (
            calibrated_novel_context_metrics["top1_accuracy"] >= MASTERY_THRESHOLD
            if calibration_enabled
            else True
        ),
        "candidate_permutation": permuted_accuracy >= MASTERY_THRESHOLD,
        "append_only_unseen_candidate_permutation": (
            append_only_unseen_permuted_accuracy is None
            or append_only_unseen_permuted_accuracy >= MASTERY_THRESHOLD
        ),
        "reward_shuffled_null": (
            shuffled_novel_context_metrics["top1_accuracy"] <= chance + 0.15
        ),
        "reload_exact": reload_exact,
        "reload_behavior_exact": reload_metrics == novel_context_metrics,
        "core_unchanged": parent_digest_before == parent_digest_after,
        "no_replayed_examples": training_accounting["replayed_examples"] == 0,
    }
    report = {
        "schema": "neural-computer.learned-compute-candidate-screen-report.v1",
        "claim_boundary": (
            "A replaceable memory-side screen learned an opaque query/key "
            "compatibility relation from attempted scalar outcomes. The "
            "screen only orders candidates; fresh verifier admission remains "
            "required. This is a bounded generalization audit, not general "
            "continual learning."
        ),
        "seed": args.seed,
        "primitive_family": args.primitive_family,
        "candidate_count": args.candidate_count,
        "train_candidate_count": train_count,
        "known_candidate_count": len(known_families),
        "unseen_candidate_count": len(unseen_families),
        "calibration_enabled": calibration_enabled,
        "append_only_enabled": append_only_screen is not None,
        "append_only_inherit_base": args.append_only_inherit_base,
        "append_only_prior_mode": append_only_prior_mode,
        "append_only_stages": args.append_only_stages,
        "append_only_stage_sizes": append_stage_sizes,
        "configuration": evaluated_screen.configuration(),
        "budgets": {
            "parent_updates": args.parent_updates,
            "screen_updates": args.screen_updates,
            "calibration_updates": args.calibration_updates,
            "append_only_stages": args.append_only_stages,
            "append_only_inherit_base": args.append_only_inherit_base,
            "append_only_prior_mode": append_only_prior_mode,
            "append_only_stage_sizes": append_stage_sizes,
            "batch_size": args.batch_size,
            "audit_count": args.audit_count,
            "key_samples": args.key_samples,
            "screen_hidden": args.screen_hidden,
            "latent_width": args.latent_width,
        },
        "candidate_key_digest": hashlib.sha256(
            keys.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest(),
        "training": training_accounting,
        "append_only_training": append_only_training,
        "train_metrics": train_metrics,
        "novel_context_metrics": novel_context_metrics,
        "unseen_candidate_metrics": unseen_candidate_metrics,
        "append_only_unseen_pre_failure_metrics": append_only_unseen_pre_failure_metrics,
        "append_only_unseen_permuted_accuracy": append_only_unseen_permuted_accuracy,
        "initial_novel_context_metrics": initial_novel_context_metrics,
        "initial_unseen_candidate_metrics": initial_unseen_candidate_metrics,
        "calibrated_novel_context_metrics": calibrated_novel_context_metrics,
        "calibrated_unseen_candidate_metrics": calibrated_unseen_candidate_metrics,
        "cold_metrics": cold_metrics,
        "reward_shuffled_training": shuffled_accounting,
        "reward_shuffled_novel_context_metrics": shuffled_novel_context_metrics,
        "candidate_permutation_accuracy": permuted_accuracy,
        "reload_metrics": reload_metrics,
        "accounting": {
            "unique_verifier_bits": (
                parent_history[-1]["unique_verifier_bits"]
                if parent_history
                else 0
            )
            + training_accounting["unique_verifier_bits"]
            + (
                append_only_training["unique_verifier_bits"]
                if append_only_training is not None
                else 0
            )
            + calibration_accounting["unique_verifier_bits"],
            "unique_logical_lifetimes": (
                parent_history[-1]["unique_logical_lifetimes"]
                if parent_history
                else 0
            )
            + training_accounting["unique_logical_lifetimes"]
            + (
                append_only_training["unique_logical_lifetimes"]
                if append_only_training is not None
                else 0
            )
            + calibration_accounting["unique_logical_lifetimes"],
            "optimizer_updates": (
                args.parent_updates
                + args.screen_updates
                + (
                    append_only_training["optimizer_updates"]
                    if append_only_training is not None
                    else 0
                )
                + calibration_accounting["optimizer_updates"]
            ),
            "replayed_examples": 0,
            "screen_informative_pairs": training_accounting[
                "informative_candidate_pairs"
            ]
            + (
                append_only_training["informative_candidate_pairs"]
                if append_only_training is not None
                else 0
            ),
            "calibration_optimizer_updates": calibration_accounting[
                "optimizer_updates"
            ],
            "calibration_unique_verifier_bits": calibration_accounting[
                "unique_verifier_bits"
            ],
            "calibration_unique_logical_lifetimes": calibration_accounting[
                "unique_logical_lifetimes"
            ],
            "calibration_informative_outcomes": calibration_accounting.get(
                "informative_outcomes", 0
            ),
            "calibration_replayed_examples": calibration_accounting[
                "replayed_examples"
            ],
        },
        "controls": {
            "parent_digest_before": parent_digest_before,
            "parent_digest_after": parent_digest_after,
            "parent_unchanged": parent_digest_before == parent_digest_after,
            "screen_reload_exact": reload_exact,
            "fresh_admission_required": True,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "wall_seconds": perf_counter() - started,
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
        "--primitive-family", choices=("registry", "opaque_rule"), default="opaque_rule"
    )
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--unseen-candidates", type=int, default=UNSEEN_CANDIDATES)
    parser.add_argument("--parent-updates", type=int, default=64)
    parser.add_argument("--screen-updates", type=int, default=512)
    parser.add_argument(
        "--calibration-updates",
        type=int,
        default=0,
        help="fresh-outcome updates for candidates that had no positive evidence",
    )
    parser.add_argument(
        "--append-only-calibration",
        action="store_true",
        help="calibrate an isolated extension after the frozen base route fails",
    )
    parser.add_argument(
        "--append-only-stages",
        type=int,
        default=1,
        help="number of sequential isolated extension stages",
    )
    parser.add_argument(
        "--append-only-inherit-base",
        action="store_true",
        help="copy the frozen base address blueprint into each extension",
    )
    parser.add_argument(
        "--append-only-prior-mode",
        choices=("none", "full", "query_path"),
        default="none",
        help="selective copy-on-write prior for new extensions",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=96)
    parser.add_argument("--key-samples", type=int, default=16)
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
                "promoted": report["promoted"],
                "novel_context_metrics": report["novel_context_metrics"],
                "unseen_candidate_metrics": report["unseen_candidate_metrics"],
                "append_only_unseen_pre_failure_metrics": report[
                    "append_only_unseen_pre_failure_metrics"
                ],
                "calibrated_unseen_candidate_metrics": report[
                    "calibrated_unseen_candidate_metrics"
                ],
                "cold_metrics": report["cold_metrics"],
                "candidate_permutation_accuracy": report[
                    "candidate_permutation_accuracy"
                ],
                "reward_shuffled_novel_context_metrics": report[
                    "reward_shuffled_novel_context_metrics"
                ],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
