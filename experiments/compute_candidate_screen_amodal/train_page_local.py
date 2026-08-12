"""Audit heterogeneous page-local representations for append-only screening.

The source page uses a frozen affine-normalized opaque signature while each
new page keeps a raw opaque signature.  The controller and event encoder are
unchanged.  A representation-matched raw source screen supplies copy-on-write
priors for raw extension pages, and cumulative scalar verifier failures are
the only activation signal for later pages.

This is an integration audit for the page-local memory ABI.  It does not claim
that the system has learned which representation to choose in general; the
representation assignment is an external experimental factor and fresh
verifier admission remains mandatory.
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
DEFAULT_CANDIDATES = 26
DEFAULT_UNSEEN = 12
DEFAULT_STAGES = 6
SPAN = 4


def _all_targets_clear(metrics: dict[str, object]) -> bool:
    values = metrics["per_target_top1_accuracy"]
    return isinstance(values, list) and bool(values) and min(values) >= MASTERY_THRESHOLD


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _failure_schedule(
    targets: torch.Tensor,
    *,
    train_count: int,
    stage_sizes: list[int],
) -> torch.Tensor:
    schedule = torch.zeros(
        targets.shape[0],
        len(stage_sizes),
        dtype=torch.bool,
    )
    for row, target in enumerate(targets.tolist()):
        offset = target - train_count
        stage_index = 0
        cumulative = stage_sizes[0]
        while offset >= cumulative:
            stage_index += 1
            cumulative += stage_sizes[stage_index]
        schedule[row, : stage_index + 1] = True
    return schedule


def _build_page_screen(
    normalized_base: LearnedComputeCandidateScreen,
    normalizer: OpaqueCandidateSignatureNormalizer,
    raw_template: LearnedComputeCandidateScreen,
    *,
    stage_sizes: list[int],
) -> PageLocalLearnedComputeCandidateScreen:
    screen = PageLocalLearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=normalized_base.latent_width,
        hidden=normalized_base.hidden,
        base_screen=normalized_base,
        base_query_view=normalizer,
        base_key_view=normalizer,
        activation_margin=1.0,
    )
    for size in stage_sizes:
        screen.append_extension(size)
    screen.freeze_base()
    for index in range(len(stage_sizes)):
        screen.initialize_extension_from_template(index, raw_template, mode="full")
    return screen


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.candidate_count < 4 or args.candidate_count % 2:
        raise ValueError("candidate-count must be an even number of at least four")
    if args.unseen_candidates < 1 or args.unseen_candidates >= args.candidate_count:
        raise ValueError("unseen-candidates must be between one and candidate-count")
    if min(args.parent_updates, args.screen_updates, args.calibration_updates) < 1:
        raise ValueError("all training budgets must be positive")
    template_updates = (
        args.screen_updates
        if args.template_updates is None
        else args.template_updates
    )
    if template_updates < 1:
        raise ValueError("template-updates must be positive")
    if min(args.batch_size, args.audit_count, args.key_samples) < 1:
        raise ValueError("batch-size, audit-count, and key-samples must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch-size and audit-count must be even")
    if args.append_only_stages < 1 or args.append_only_stages > args.unseen_candidates:
        raise ValueError("append-only-stages must fit within unseen candidates")
    train_count = args.candidate_count - args.unseen_candidates
    unseen_count, remainder = divmod(
        args.unseen_candidates,
        args.append_only_stages,
    )
    stage_sizes = [unseen_count] * args.append_only_stages
    for index in range(remainder):
        stage_sizes[-(index + 1)] += 1
    known_families = list(range(train_count))
    unseen_families = list(range(train_count, args.candidate_count))
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

    normalized_base = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    normalized_training = _train_screen(
        normalized_base,
        parent,
        grammar,
        normalized_keys,
        families=known_families,
        updates=args.screen_updates,
        batch_size=args.batch_size,
        seed=args.seed + 30_000,
        learning_rate=args.learning_rate,
        query_transform=normalizer,
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
        updates=template_updates,
        batch_size=args.batch_size,
        seed=args.seed + 31_000,
        learning_rate=args.learning_rate,
    )
    page_screen = _build_page_screen(
        normalized_base,
        normalizer,
        raw_template,
        stage_sizes=stage_sizes,
    )
    base_state_before_extensions = {
        name: value.detach().clone()
        for name, value in page_screen.base_screen.state_dict().items()
    }
    stage_accounting = []
    start = 0
    for stage_index, extension_size in enumerate(stage_sizes):
        stage_accounting.append(
            _train_append_only_extension(
                page_screen,
                parent,
                grammar,
                raw_keys[train_count + start : train_count + start + extension_size],
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

    shuffled_base = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    shuffled_training = _train_screen(
        shuffled_base,
        parent,
        grammar,
        normalized_keys,
        families=known_families,
        updates=args.screen_updates,
        batch_size=args.batch_size,
        seed=args.seed + 40_000,
        learning_rate=args.learning_rate,
        shuffle_outcomes=True,
        query_transform=normalizer,
    )

    known_query_families = [
        known_families[index % len(known_families)] for index in range(args.audit_count)
    ]
    unseen_query_families = [
        unseen_families[index % len(unseen_families)]
        for index in range(args.audit_count)
    ]
    known_queries = _event_query(
        parent,
        grammar,
        known_query_families,
        seed=args.seed + 60_000,
    )
    unseen_queries = _event_query(
        parent,
        grammar,
        unseen_query_families,
        seed=args.seed + 70_000,
    )
    known_targets = torch.tensor(known_query_families)
    unseen_targets = torch.tensor(unseen_query_families)
    failure_schedule = _failure_schedule(
        unseen_targets,
        train_count=train_count,
        stage_sizes=stage_sizes,
    )
    known_metrics = _append_only_route_metrics(
        page_screen,
        known_queries,
        known_targets,
        raw_keys[:train_count],
        raw_keys[train_count:],
        failed_extensions=False,
    )
    unseen_pre_failure_metrics = _append_only_route_metrics(
        page_screen,
        unseen_queries,
        unseen_targets,
        raw_keys[:train_count],
        raw_keys[train_count:],
        failed_extensions=False,
    )
    unseen_metrics = _append_only_route_metrics(
        page_screen,
        unseen_queries,
        unseen_targets,
        raw_keys[:train_count],
        raw_keys[train_count:],
        failed_extensions=failure_schedule,
    )
    permutation_accuracy = _append_only_permuted_accuracy(
        page_screen,
        known_queries,
        known_targets,
        raw_keys[:train_count],
        raw_keys[train_count:],
        seed=args.seed + 70_000,
        failed_extensions=False,
    )
    unseen_permutation_accuracy = _append_only_permuted_accuracy(
        page_screen,
        unseen_queries,
        unseen_targets,
        raw_keys[:train_count],
        raw_keys[train_count:],
        seed=args.seed + 71_000,
        failed_extensions=failure_schedule,
    )
    shuffled_queries = normalizer(known_queries)
    shuffled_scores = shuffled_base(shuffled_queries, normalized_keys)
    shuffled_predictions = shuffled_scores.argmax(dim=-1)
    shuffled_metrics = {
        "top1_accuracy": float((shuffled_predictions == known_targets).float().mean()),
        "per_target_top1_accuracy": _per_target_top1_accuracy(
            shuffled_predictions,
            known_targets,
        ),
    }

    cold_base = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    cold_scores = cold_base(normalizer(known_queries), normalized_keys[:train_count])
    cold_predictions = cold_scores.argmax(dim=-1)
    cold_metrics = {
        "top1_accuracy": float((cold_predictions == known_targets).float().mean()),
        "per_target_top1_accuracy": _per_target_top1_accuracy(
            cold_predictions,
            known_targets,
        ),
    }

    page_state = {
        name: value.detach().clone() for name, value in page_screen.state_dict().items()
    }
    restored_normalizer = OpaqueCandidateSignatureNormalizer(EVENT_WIDTH)
    restored_normalizer.fit(raw_keys)
    restored_base = LearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
    )
    restored = PageLocalLearnedComputeCandidateScreen(
        EVENT_WIDTH,
        EVENT_WIDTH,
        latent_width=args.latent_width,
        hidden=args.screen_hidden,
        base_screen=restored_base,
        base_query_view=restored_normalizer,
        base_key_view=restored_normalizer,
        activation_margin=1.0,
    )
    for size in stage_sizes:
        restored.append_extension(size)
    restored.load_state_dict(page_state, strict=True)
    reload_exact = all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in page_state.items()
    )
    reload_metrics = _append_only_route_metrics(
        restored,
        known_queries,
        known_targets,
        raw_keys[:train_count],
        raw_keys[train_count:],
        failed_extensions=False,
    )
    parent_digest_after = _digest_core(parent, ())
    base_unchanged = all(
        torch.equal(value, page_screen.base_screen.state_dict()[name])
        for name, value in base_state_before_extensions.items()
    )
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
        "candidate_permutation": permutation_accuracy >= MASTERY_THRESHOLD,
        "unseen_candidate_permutation": unseen_permutation_accuracy
        >= MASTERY_THRESHOLD,
        "reward_shuffled_null": shuffled_metrics["top1_accuracy"]
        <= (1.0 / args.candidate_count) + 0.15,
        "base_unchanged_during_page_growth": base_unchanged,
        "reload_exact": reload_exact,
        "reload_behavior_exact": reload_metrics == known_metrics,
        "core_unchanged": parent_digest_before == parent_digest_after,
        "no_replayed_examples": (
            normalized_training["replayed_examples"]
            + raw_template_training["replayed_examples"]
            + shuffled_training["replayed_examples"]
            + calibration_accounting["replayed_examples"]
            == 0
        ),
    }
    representation_runs = [
        normalized_training,
        raw_template_training,
        shuffled_training,
        calibration_accounting,
    ]
    representation_accounting = _merge_training_accounting(representation_runs)
    parent_accounting = parent_history[-1] if parent_history else {}
    accounting = {
        "unique_verifier_bits": int(
            parent_accounting.get("unique_verifier_bits", 0)
        )
        + int(representation_accounting["unique_verifier_bits"]),
        "unique_logical_lifetimes": int(
            parent_accounting.get("unique_logical_lifetimes", 0)
        )
        + int(representation_accounting["unique_logical_lifetimes"]),
        "optimizer_updates": args.parent_updates
        + int(representation_accounting["optimizer_updates"]),
        "replayed_examples": int(representation_accounting["replayed_examples"]),
        "wall_seconds": perf_counter() - started,
        "retention_floor": min(known_metrics["per_target_top1_accuracy"]),
        "transfer_ratio_against_cold": (
            known_metrics["top1_accuracy"]
            / max(float(cold_metrics["top1_accuracy"]), 1.0 / args.candidate_count)
        ),
    }
    report = {
        "schema": "neural-computer.page-local-compute-candidate-screen-report.v1",
        "claim_boundary": (
            "A memory-side screen can preserve a normalized source page and "
            "raw representation-matched append pages behind cumulative scalar "
            "verifier gates. Representation assignment is an experimental "
            "factor; this is not learned representation selection or general "
            "continual learning."
        ),
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "train_candidate_count": train_count,
        "unseen_candidate_count": args.unseen_candidates,
        "append_only_stage_sizes": stage_sizes,
        "representation_assignment": {
            "base": "frozen_affine_normalized_v1",
            "extensions": "raw_identity_v1",
            "selection": "external_experimental_factor_v1",
        },
        "candidate_key_normalizer": normalizer.configuration(),
        "candidate_key_diagnostics": {
            "raw": _candidate_key_diagnostics(raw_keys),
            "normalized": _candidate_key_diagnostics(normalized_keys),
        },
        "configuration": page_screen.configuration(),
        "budgets": {
            "parent_updates": args.parent_updates,
            "normalized_base_updates": args.screen_updates,
            "raw_template_updates": template_updates,
            "calibration_updates_per_stage": args.calibration_updates,
            "batch_size": args.batch_size,
            "audit_count": args.audit_count,
            "key_samples": args.key_samples,
        },
        "training": {
            "parent": parent_history,
            "normalized_base": normalized_training,
            "raw_template": raw_template_training,
            "reward_shuffled_normalized_base": shuffled_training,
            "page_extensions": calibration_accounting,
        },
        "accounting": accounting,
        "known_context_metrics": known_metrics,
        "unseen_candidate_metrics": unseen_metrics,
        "unseen_pre_failure_metrics": unseen_pre_failure_metrics,
        "cold_metrics": cold_metrics,
        "reward_shuffled_metrics": shuffled_metrics,
        "candidate_permutation_accuracy": permutation_accuracy,
        "unseen_candidate_permutation_accuracy": unseen_permutation_accuracy,
        "controls": {
            "parent_digest_before": parent_digest_before,
            "parent_digest_after": parent_digest_after,
            "parent_unchanged": parent_digest_before == parent_digest_after,
            "base_digest_before_page_growth": _digest_module(
                page_screen.base_screen
            ),
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
    parser.add_argument("--parent-updates", type=int, default=64)
    parser.add_argument("--screen-updates", type=int, default=1024)
    parser.add_argument(
        "--template-updates",
        type=int,
        default=None,
        help="raw representation prior updates; defaults to screen-updates",
    )
    parser.add_argument("--calibration-updates", type=int, default=32)
    parser.add_argument("--append-only-stages", type=int, default=DEFAULT_STAGES)
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
