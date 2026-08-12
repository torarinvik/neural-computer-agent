"""Audit verifier-gated consolidation of learned append-only screen stages.

The source screen grows through isolated external stages.  A separately
trained replacement screen then attempts to represent two consecutive stages
as one physical extension.  Adoption is transactional and requires fresh
behavioral verification; the frozen controller and source screen are never
mutated by a rejected proposal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.compute_candidate_screen_amodal.train import (
    EVENT_WIDTH,
    _append_only_permuted_accuracy,
    _append_only_route_metrics,
    _candidate_keys,
    _event_query,
    _runtime,
    _train_append_only_extension,
    _train_screen,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    generate_runtime_program_grammar,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
)
from neural_computer import (
    AppendOnlyLearnedComputeCandidateScreen,
    LearnedComputeCandidateScreen,
    selector_distillation_loss,
)

MASTERY_THRESHOLD = 0.75


def _failure_schedule(
    targets: torch.Tensor,
    *,
    train_count: int,
    stage_sizes: list[int],
) -> torch.Tensor:
    schedule = torch.zeros(
        targets.shape[0], len(stage_sizes), dtype=torch.bool
    )
    for row, target in enumerate(targets.tolist()):
        offset = int(target) - train_count
        stage = 0
        cumulative = stage_sizes[0]
        while offset >= cumulative:
            stage += 1
            cumulative += stage_sizes[stage]
        schedule[row, : stage + 1] = True
    return schedule


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _metrics(
    screen: AppendOnlyLearnedComputeCandidateScreen,
    queries: torch.Tensor,
    targets: torch.Tensor,
    base_keys: torch.Tensor,
    extension_keys: torch.Tensor,
    failures: torch.Tensor | bool,
    *,
    permutation_seed: int,
) -> dict[str, object]:
    result = _append_only_route_metrics(
        screen,
        queries,
        targets,
        base_keys,
        extension_keys,
        failed_extensions=failures,
    )
    result["permutation_accuracy"] = _append_only_permuted_accuracy(
        screen,
        queries,
        targets,
        base_keys,
        extension_keys,
        seed=permutation_seed,
        failed_extensions=failures,
    )
    scores = screen(
        queries,
        base_keys,
        extension_keys,
        failed_extensions=failures,
    )
    predictions = scores.argmax(dim=-1)
    result["per_target_top1_accuracy"] = [
        float((predictions[targets == target] == target).float().mean())
        for target in sorted(set(targets.tolist()))
    ]
    return result


def _all_targets_clear(metrics: dict[str, object]) -> bool:
    values = metrics["per_target_top1_accuracy"]
    return isinstance(values, list) and bool(values) and min(values) >= MASTERY_THRESHOLD


def _train_distilled_replacement(
    replacement_screen: AppendOnlyLearnedComputeCandidateScreen,
    source: AppendOnlyLearnedComputeCandidateScreen,
    parent,
    grammar,
    base_keys: torch.Tensor,
    replacement_keys: torch.Tensor,
    all_extension_keys: torch.Tensor,
    *,
    families: list[int],
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    distillation_weight: float,
) -> dict[str, int | float]:
    """Train a compact stage from fresh outcomes plus source behavior."""

    if len(families) < 2:
        raise ValueError("distilled replacement needs multiple candidates")
    if distillation_weight < 0.0:
        raise ValueError("distillation weight cannot be negative")
    replacement_screen.enable_extension(0)
    extension = replacement_screen.extensions[0]
    optimizer = torch.optim.AdamW(
        extension.parameters(), lr=learning_rate, weight_decay=1e-5
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
        outcomes = torch.zeros(batch_size, len(families))
        outcomes[
            torch.arange(batch_size),
            torch.tensor([families.index(family) for family in batch_families]),
        ] = 1.0
        loss, pairs = extension.outcome_ranking_loss(
            query,
            replacement_keys,
            outcomes,
        )
        with torch.no_grad():
            teacher = source(
                query,
                base_keys,
                all_extension_keys,
                failed_extensions=True,
            )[:, base_keys.shape[0] : base_keys.shape[0] + len(families)]
        student_centered = extension(query, replacement_keys)
        student_centered = student_centered - student_centered.mean(
            dim=-1, keepdim=True
        )
        teacher_centered = teacher - teacher.mean(dim=-1, keepdim=True)
        loss = loss + distillation_weight * selector_distillation_loss(
            student_centered,
            teacher_centered,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(extension.parameters(), 1.0)
        optimizer.step()
        informative_pairs += pairs
        last_loss = float(loss.detach())
    extension.eval()
    return {
        "optimizer_updates": updates,
        "unique_verifier_bits": updates * batch_size * replacement_keys.shape[0],
        "unique_logical_lifetimes": updates * batch_size * replacement_keys.shape[0],
        "informative_candidate_pairs": informative_pairs,
        "distillation_updates": updates,
        "replayed_examples": 0,
        "final_loss": last_loss,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.candidate_count < 6 or args.candidate_count % 2:
        raise ValueError("candidate-count must be an even number of at least six")
    if args.unseen_candidates != 10 or args.append_only_stages != 5:
        raise ValueError("this audit fixes ten unseen candidates across five stages")
    if args.replacement_updates < 1:
        raise ValueError("replacement-updates must be positive")
    train_count = args.candidate_count - args.unseen_candidates
    if train_count < 1:
        raise ValueError("candidate-count must exceed unseen-candidates")
    grammar = generate_runtime_program_grammar(
        seed=args.program_seed,
        count=args.candidate_count,
        depth=args.program_depth,
        primitive_family="opaque_rule",
    )
    known_families = list(range(train_count))
    unseen_families = list(range(train_count, args.candidate_count))
    source_stage_sizes = [2] * args.append_only_stages
    parent = _runtime(seed=args.seed, growth=False)
    _, parent_progress = _train_with_progress(
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
    parent_digest = _digest_core(parent, ())
    keys = _candidate_keys(
        parent,
        grammar,
        args.candidate_count,
        seed=args.seed + 20_000,
        samples_per_candidate=args.key_samples,
    )
    base_screen = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    base_accounting = _train_screen(
        base_screen,
        parent,
        grammar,
        keys,
        families=known_families,
        updates=args.screen_updates,
        batch_size=args.batch_size,
        seed=args.seed + 30_000,
        learning_rate=args.learning_rate,
    )
    source = AppendOnlyLearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
        extension_sizes=tuple(source_stage_sizes),
    )
    source.base_screen.load_state_dict(base_screen.state_dict(), strict=True)
    source.freeze_base()
    source_accounting: list[dict[str, int | float]] = []
    extension_start = 0
    for stage_index, stage_size in enumerate(source_stage_sizes):
        source_accounting.append(
            _train_append_only_extension(
                source,
                parent,
                grammar,
                keys[train_count + extension_start : train_count + extension_start + stage_size],
                extension_index=stage_index,
                families=unseen_families[extension_start : extension_start + stage_size],
                updates=args.stage_updates,
                batch_size=args.batch_size,
                seed=args.seed + 81_000 + stage_index * 10_000,
                learning_rate=args.learning_rate,
            )
        )
        extension_start += stage_size
    extension_keys = keys[train_count:]
    base_keys = keys[:train_count]
    unseen_families_audit = [
        unseen_families[index % len(unseen_families)]
        for index in range(args.audit_count)
    ]
    all_unseen_targets = torch.tensor(unseen_families_audit)
    all_unseen_queries = _event_query(
        parent,
        grammar,
        unseen_families_audit,
        seed=args.seed + 120_000,
    )
    source_failures = _failure_schedule(
        all_unseen_targets,
        train_count=train_count,
        stage_sizes=source_stage_sizes,
    )
    source_digest_before = _digest_module(source)
    source_before = _metrics(
        source,
        all_unseen_queries,
        all_unseen_targets,
        base_keys,
        extension_keys,
        source_failures,
        permutation_seed=args.seed + 121_000,
    )
    known_families_audit = [known_families[index % len(known_families)] for index in range(args.audit_count)]
    known_queries = _event_query(
        parent,
        grammar,
        known_families_audit,
        seed=args.seed + 122_000,
    )
    known_targets = torch.tensor(known_families_audit)

    replacement_training = AppendOnlyLearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
        extension_sizes=(len(unseen_families[:4]),),
    )
    replacement_accounting = _train_distilled_replacement(
        replacement_training,
        source,
        parent,
        grammar,
        base_keys,
        extension_keys[:4],
        extension_keys,
        families=unseen_families[:4],
        updates=args.replacement_updates,
        batch_size=args.batch_size,
        seed=args.seed + 150_000,
        learning_rate=args.learning_rate,
        distillation_weight=args.distillation_weight,
    )
    replacement = replacement_training.extensions[0]
    compact_stage_sizes = [4, 2, 2, 2]
    compact_failures = _failure_schedule(
        all_unseen_targets,
        train_count=train_count,
        stage_sizes=compact_stage_sizes,
    )
    compacted, receipt = source.consolidate_verified(
        (0, 1),
        replacement,
        verifier=lambda candidate: (
            _metrics(
                candidate,
                all_unseen_queries,
                all_unseen_targets,
                base_keys,
                extension_keys,
                compact_failures,
                permutation_seed=args.seed + 123_000,
            )["top1_accuracy"]
            >= MASTERY_THRESHOLD
            and _all_targets_clear(
                _metrics(
                    candidate,
                    all_unseen_queries,
                    all_unseen_targets,
                    base_keys,
                    extension_keys,
                    compact_failures,
                    permutation_seed=args.seed + 123_000,
                )
            )
            and _append_only_route_metrics(
                candidate,
                known_queries,
                known_targets,
                base_keys,
                extension_keys,
                failed_extensions=False,
            )["top1_accuracy"]
            >= MASTERY_THRESHOLD
        ),
    )
    compact_after = (
        _metrics(
            compacted,
            all_unseen_queries,
            all_unseen_targets,
            base_keys,
            extension_keys,
            compact_failures,
            permutation_seed=args.seed + 123_000,
        )
        if compacted is not None
        else None
    )
    compact_known = (
        _append_only_route_metrics(
            compacted,
            known_queries,
            known_targets,
            base_keys,
            extension_keys,
            failed_extensions=False,
        )
        if compacted is not None
        else None
    )

    naive = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    naive.load_state_dict(source.extensions[0].state_dict(), strict=True)
    naive_compacted, naive_receipt = source.consolidate_verified(
        (0, 1),
        naive,
        verifier=lambda candidate: (
            _metrics(
                candidate,
                all_unseen_queries,
                all_unseen_targets,
                base_keys,
                extension_keys,
                compact_failures,
                permutation_seed=args.seed + 124_000,
            )["top1_accuracy"]
            >= MASTERY_THRESHOLD
            and _all_targets_clear(
                _metrics(
                    candidate,
                    all_unseen_queries,
                    all_unseen_targets,
                    base_keys,
                    extension_keys,
                    compact_failures,
                    permutation_seed=args.seed + 124_000,
                )
            )
        ),
    )
    source_digest_after = _digest_module(source)
    reloaded = None
    reload_exact = False
    reload_metrics = None
    if compacted is not None:
        reloaded = AppendOnlyLearnedComputeCandidateScreen(
            EVENT_WIDTH,
            EVENT_WIDTH,
            latent_width=args.latent_width,
            hidden=args.screen_hidden,
            extension_sizes=tuple(compact_stage_sizes),
        )
        reloaded.load_state_dict(compacted.state_dict(), strict=True)
        reload_exact = all(
            torch.equal(value, reloaded.state_dict()[name])
            for name, value in compacted.state_dict().items()
        )
        reload_metrics = _metrics(
            reloaded,
            all_unseen_queries,
            all_unseen_targets,
            base_keys,
            extension_keys,
            compact_failures,
            permutation_seed=args.seed + 123_000,
        )
    parent_digest_after = _digest_core(parent, ())
    training_accounting = {
        "optimizer_updates": (
            args.parent_updates
            + args.screen_updates
            + sum(int(run["optimizer_updates"]) for run in source_accounting)
            + int(replacement_accounting["optimizer_updates"])
        ),
        "unique_verifier_bits": (
            int(base_accounting["unique_verifier_bits"])
            + sum(int(run["unique_verifier_bits"]) for run in source_accounting)
            + int(replacement_accounting["unique_verifier_bits"])
        ),
        "unique_logical_lifetimes": (
            int(base_accounting["unique_logical_lifetimes"])
            + sum(int(run["unique_logical_lifetimes"]) for run in source_accounting)
            + int(replacement_accounting["unique_logical_lifetimes"])
        ),
        "replayed_examples": 0,
    }
    gates = {
        "parent_stable": any(
            float(row["heldout_accuracy"]) >= MASTERY_THRESHOLD
            for row in parent_progress
        ),
        "consolidation_accepted": compacted is not None and receipt.accepted,
        "logical_candidates_preserved": (
            receipt.logical_candidates_before == receipt.logical_candidates_after
        ),
        "extensions_reduced": receipt.extensions_saved == 1,
        "compact_unseen_retained": (
            compact_after is not None
            and compact_after["top1_accuracy"] >= MASTERY_THRESHOLD
        ),
        "compact_known_retained": (
            compact_known is not None
            and compact_known["top1_accuracy"] >= MASTERY_THRESHOLD
        ),
        "compact_permutation": (
            compact_after is not None
            and compact_after["permutation_accuracy"] >= MASTERY_THRESHOLD
        ),
        "source_unchanged": source_digest_before == source_digest_after,
        "parent_unchanged": parent_digest == parent_digest_after,
        "naive_replacement_rejected": naive_compacted is None and not naive_receipt.accepted,
        "reload_exact": reload_exact,
        "reload_behavior_exact": (
            reload_metrics == compact_after
            if compact_after is not None
            else False
        ),
        "no_replayed_examples": training_accounting["replayed_examples"] == 0,
    }
    report = {
        "schema": "neural-computer.learned-compute-screen-consolidation-report.v1",
        "claim_boundary": (
            "A fresh-outcome-trained replacement can compact consecutive external "
            "screen stages only when a behavior verifier accepts the immutable "
            "candidate. This is bounded verified compaction, not general continual "
            "learning or arbitrary learned compression."
        ),
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "known_candidate_count": train_count,
        "unseen_candidate_count": len(unseen_families),
        "source_stage_sizes": source_stage_sizes,
        "compact_stage_sizes": compact_stage_sizes,
        "budgets": {
            "parent_updates": args.parent_updates,
            "screen_updates": args.screen_updates,
            "stage_updates": args.stage_updates,
            "replacement_updates": args.replacement_updates,
            "distillation_weight": args.distillation_weight,
            "batch_size": args.batch_size,
            "audit_count": args.audit_count,
            "key_samples": args.key_samples,
        },
        "source_before": source_before,
        "replacement_training": replacement_accounting,
        "consolidation_receipt": {
            "accepted": receipt.accepted,
            "source_indices": receipt.source_indices,
            "extension_count_before": receipt.extension_count_before,
            "extension_count_after": receipt.extension_count_after,
            "logical_candidates_before": receipt.logical_candidates_before,
            "logical_candidates_after": receipt.logical_candidates_after,
            "extensions_saved": receipt.extensions_saved,
            "reason": receipt.reason,
        },
        "compact_after": compact_after,
        "compact_known": compact_known,
        "naive_replacement_receipt": {
            "accepted": naive_receipt.accepted,
            "extensions_saved": naive_receipt.extensions_saved,
            "reason": naive_receipt.reason,
        },
        "accounting": training_accounting,
        "gates": gates,
        "promoted": all(gates.values()),
        "wall_seconds": perf_counter() - started,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--program-seed", type=int, default=4242)
    parser.add_argument("--program-depth", type=int, default=8)
    parser.add_argument("--candidate-count", type=int, default=20)
    parser.add_argument("--unseen-candidates", type=int, default=10)
    parser.add_argument("--append-only-stages", type=int, default=5)
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--screen-updates", type=int, default=512)
    parser.add_argument("--stage-updates", type=int, default=32)
    parser.add_argument("--replacement-updates", type=int, default=64)
    parser.add_argument("--distillation-weight", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--audit-count", type=int, default=96)
    parser.add_argument("--key-samples", type=int, default=12)
    parser.add_argument("--screen-hidden", type=int, default=64)
    parser.add_argument("--latent-width", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    args = parser.parse_args()
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "source_before": report["source_before"],
                "compact_after": report["compact_after"],
                "compact_known": report["compact_known"],
                "consolidation_receipt": report["consolidation_receipt"],
                "naive_replacement_receipt": report["naive_replacement_receipt"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
