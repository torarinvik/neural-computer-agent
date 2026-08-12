"""Audit multi-source external transfer and verified finite-capacity compaction.

Two independently learned external files are protected in a bounded artifact
bank.  Three fresh-outcome candidates (one initialized from each source and
one fresh) learn a new runtime program while the parent controller remains
frozen.  The selected candidate is admitted only after stable-prefix evidence.

The two protected files are then transactionally compacted into one physical
row with two opaque executable views.  This is logical storage consolidation,
not a claim that two neural programs have been compressed into one program.
Each view is reverified independently before the compacted bank is adopted;
the new target is then admitted by growing capacity rather than evicting a
protected capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    SequenceMemoryBatch,
    generate_sequence_memory_batch,
)
from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    SPAN,
    THRESHOLD,
    _generated_key,
    _load_stack_artifact,
    _parse_program_specs,
    _stack_artifact,
)
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    _capability_accuracy,
    _new_capability,
    _rollout_capability,
    _stable_bits,
    _train_capability,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
    _runtime,
)
from neural_computer import (
    CapabilityRetentionProbe,
    ExecutableArtifactMemory,
    ExternalCapabilityLifecycle,
    RetentionPolicyConfig,
    select_capability_candidate,
)

from .train_pipeline import _new_stack

DEFAULT_RUNTIME_GRAMMAR = (
    ("reverse", "adjacent_xor", "complement", "prefix_parity"),
    ("prefix_parity", "global_parity", "rotate", "complement"),
    ("global_parity", "reverse", "adjacent_xor", "rotate"),
)


def _digest_artifact(artifact: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(artifact.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _candidate_scores(
    progress: list[dict[str, float | int]],
) -> tuple[float, ...]:
    return tuple(float(row["heldout_accuracy"]) for row in progress)


def _view_artifact(
    artifact: dict[str, torch.Tensor], view: str
) -> dict[str, torch.Tensor]:
    prefix = f"view.{view}."
    selected = {
        name.removeprefix(prefix): value
        for name, value in artifact.items()
        if name.startswith(prefix)
    }
    if not selected:
        raise ValueError(f"consolidated artifact is missing view {view!r}")
    return selected


def _pack_views(
    artifacts: tuple[dict[str, torch.Tensor], ...],
    views: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    if len(artifacts) != len(views) or not artifacts:
        raise ValueError("view artifacts and names must be nonempty and aligned")
    packed: dict[str, torch.Tensor] = {}
    for view, artifact in zip(views, artifacts, strict=True):
        if not view or "." in view:
            raise ValueError("view names must be nonempty and dot-free")
        for name, value in artifact.items():
            packed[f"view.{view}.{name}"] = value.detach().cpu().clone()
    return packed


def _source_behavior(
    parent,
    artifact: dict[str, torch.Tensor],
    program_id: int,
    grammar,
    *,
    count: int,
    seed: int,
) -> float:
    stack, decoder = _load_stack_artifact(artifact)
    return _capability_accuracy(
        parent,
        stack,
        decoder,
        operation="generated_composition",
        span=SPAN,
        count=count,
        seed=seed,
        generated_composition_ids=(program_id,),
        generated_compositions=grammar,
    )


@torch.no_grad()
def _loaded_source_probe_outcomes(
    parent,
    stack,
    decoder,
    program_id: int,
    grammar,
    *,
    count: int,
    probes: int,
    seed: int,
) -> list[float]:
    """Run independent retention probes in one batched visual rollout."""

    batches = tuple(
        generate_sequence_memory_batch(
            count,
            span=SPAN,
            distractors=1,
            seed=seed + probe * 101,
            operation="generated_composition",
            generated_composition_ids=(program_id,),
            generated_compositions=grammar,
        )
        for probe in range(probes)
    )
    combined = SequenceMemoryBatch(
        *(torch.cat(tuple(getattr(batch, field) for batch in batches), dim=0)
          for field in SequenceMemoryBatch.__dataclass_fields__)
    )
    rewards = _rollout_capability(
        parent,
        stack,
        decoder,
        combined,
        train=False,
    )["rewards"]
    return [
        float(rewards[offset : offset + count].mean())
        for offset in range(0, count * probes, count)
    ]


def _probe_raw_artifact(
    parent,
    artifact: dict[str, torch.Tensor],
    program_id: int,
    grammar,
    *,
    count: int,
    probes: int,
    seed: int,
) -> list[float]:
    return [
        _source_behavior(
            parent,
            artifact,
            program_id,
            grammar,
            count=count,
            seed=seed + probe * 101,
        )
        for probe in range(probes)
    ]


def _probe_views(
    parent,
    candidate: ExecutableArtifactMemory,
    source_keys: tuple[torch.Tensor, ...],
    views: tuple[str, ...],
    source_ids: tuple[int, ...],
    grammar,
    *,
    count: int,
    probes: int,
    seed: int,
) -> tuple[CapabilityRetentionProbe, ...]:
    probes_by_view: list[list[float]] = [[] for _ in views]
    loaded_by_view: list[tuple[object, object] | None] = []
    for key, view in zip(source_keys, views, strict=True):
        try:
            handle, packed = candidate.promote(key)
            if handle.view != view:
                raise ValueError("opaque alias resolved to the wrong neural slot")
            loaded_by_view.append(_load_stack_artifact(_view_artifact(packed, view)))
        except (LookupError, ValueError):
            loaded_by_view.append(None)
    for index, loaded in enumerate(loaded_by_view):
        if loaded is None:
            probes_by_view[index].extend([0.0] * probes)
            continue
        probes_by_view[index].extend(
            _loaded_source_probe_outcomes(
                parent,
                loaded[0],
                loaded[1],
                source_ids[index],
                grammar,
                count=count,
                probes=probes,
                seed=seed + index,
            )
        )
    return tuple(
        CapabilityRetentionProbe(key, outcomes)
        for key, outcomes in zip(source_keys, probes_by_view, strict=True)
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    budgets = (
        args.parent_updates,
        args.source_updates,
        args.candidate_updates,
        args.batch_size,
        args.audit_count,
        args.retention_probes,
        args.eval_every,
    )
    if min(budgets) < 1:
        raise ValueError("all update and audit budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")
    source_ids = tuple(args.source_ids)
    if len(source_ids) != 2 or len(set(source_ids)) != 2:
        raise ValueError("exactly two distinct source IDs are required")
    grammar = _parse_program_specs(
        args.program_spec
        or [
            ",".join(program)
            for program in DEFAULT_RUNTIME_GRAMMAR
        ]
    )
    required_ids = (*source_ids, args.target_id)
    if any(program_id < 0 or program_id >= len(grammar) for program_id in required_ids):
        raise ValueError("source or target program is out of range")
    if args.target_id in source_ids:
        raise ValueError("target program must differ from both source programs")

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    parent = _runtime(seed=args.seed, growth=False)
    _parent_history, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        credit_mode="sampled",
    )
    parent.eval()
    parent_digest_before = _digest_core(parent, ())

    source_artifacts: list[dict[str, torch.Tensor]] = []
    source_progress: list[list[dict[str, float | int]]] = []
    source_stable: list[int | None] = []
    source_behavior: list[float] = []
    source_digests: list[str] = []
    for source_position, source_id in enumerate(source_ids):
        stack = _new_stack(
            args.seed + 1_000 + source_position,
            program_count=2,
            stack="routed",
        )
        decoder = _new_capability(args.seed + 2_000 + source_position)[1]
        _history, progress = _train_capability(
            parent,
            stack,
            decoder,
            operation="generated_composition",
            span=SPAN,
            updates=args.source_updates,
            batch_size=args.batch_size,
            seed=args.seed + 20_000 + source_position * 10_003,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            generated_composition_ids=(source_id,),
            generated_compositions=grammar,
        )
        artifact = _stack_artifact(stack, decoder)
        source_artifacts.append(artifact)
        source_progress.append(progress)
        source_stable.append(
            _stable_bits(
                progress,
                threshold=THRESHOLD,
                bits_per_update=args.batch_size * SPAN,
            )
        )
        source_behavior.append(
            _source_behavior(
                parent,
                artifact,
                source_id,
                grammar,
                count=args.audit_count,
                seed=args.seed + 40_000 + source_position,
            )
        )
        source_digests.append(_digest_artifact(artifact))

    candidate_artifacts: list[dict[str, torch.Tensor]] = []
    candidate_progress: list[list[dict[str, float | int]]] = []
    candidate_names = tuple(f"source_{source_id}" for source_id in source_ids) + ("fresh",)
    for candidate_index, source_artifact in enumerate((*source_artifacts, None)):
        if source_artifact is None:
            stack = _new_stack(args.seed + 3_000, program_count=2, stack="routed")
            decoder = _new_capability(args.seed + 4_000)[1]
        else:
            stack, decoder = _load_stack_artifact(source_artifact)
        torch.manual_seed(args.seed + 50_001)
        _history, progress = _train_capability(
            parent,
            stack,
            decoder,
            operation="generated_composition",
            span=SPAN,
            updates=args.candidate_updates,
            batch_size=args.batch_size,
            seed=args.seed + 50_001,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            generated_composition_ids=(args.target_id,),
            generated_compositions=grammar,
        )
        candidate_artifacts.append(_stack_artifact(stack, decoder))
        candidate_progress.append(progress)

    bits_per_update = args.batch_size * SPAN
    candidate_stable = [
        _stable_bits(progress, threshold=THRESHOLD, bits_per_update=bits_per_update)
        for progress in candidate_progress
    ]
    selection = select_capability_candidate(
        tuple(_candidate_scores(progress) for progress in candidate_progress),
        threshold=THRESHOLD,
        bits_per_observation=args.eval_every * bits_per_update,
    )
    candidate_behavior = [
        _source_behavior(
            parent,
            artifact,
            args.target_id,
            grammar,
            count=args.audit_count,
            seed=args.seed + 60_000,
        )
        for artifact in candidate_artifacts
    ]

    source_path = args.report_out.parent / "source_bank"
    if source_path.exists():
        shutil.rmtree(source_path)
    bank = ExecutableArtifactMemory(
        source_path,
        width=48,
        capacity=2,
        write_match_threshold=0.99999,
    )
    bank.retention.config = RetentionPolicyConfig(
        mastery_threshold=THRESHOLD,
        min_mastery_observations=args.retention_probes,
    )
    lifecycle = ExternalCapabilityLifecycle(bank)
    source_keys = tuple(
        _generated_key(
            parent,
            source_id,
            count=args.audit_count,
            seed=args.seed + 70_000 + index,
            generated_compositions=grammar,
        )
        for index, source_id in enumerate(source_ids)
    )
    address_similarity = torch.nn.functional.normalize(
        torch.stack(source_keys), dim=-1
    ) @ torch.nn.functional.normalize(torch.stack(source_keys), dim=-1).T
    off_diagonal = address_similarity[~torch.eye(2, dtype=torch.bool)]
    addresses_separated = bool(torch.all(off_diagonal < 0.99999))
    if not addresses_separated:
        raise RuntimeError(
            "source address keys are too similar for an independent two-row "
            f"memory pressure test: max cosine={float(off_diagonal.max()):.8f}"
        )
    source_rows: list[int] = []
    source_retention: list[list[float]] = []
    for index, (key, artifact, source_id) in enumerate(
        zip(source_keys, source_artifacts, source_ids, strict=True)
    ):
        receipt = lifecycle.admit(key, artifact)
        if not receipt.accepted or receipt.index is None:
            raise RuntimeError(f"source admission failed: {receipt.reason}")
        source_rows.append(receipt.index)
        outcomes = _probe_raw_artifact(
            parent,
            artifact,
            source_id,
            grammar,
            count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 80_000 + index * 1_000,
        )
        source_retention.append(outcomes)
        for outcome in outcomes:
            bank.observe_retention(key, outcome)
    bank.save()
    source_protected = bool(bank.protection_mask().all())

    views = tuple(f"source_{source_id}" for source_id in source_ids)
    packed_artifact = _pack_views(tuple(source_artifacts), views)
    replacement_key = F.normalize(torch.stack(source_keys).mean(dim=0), dim=0)
    consolidation_path = args.report_out.parent / "consolidated_bank"
    if consolidation_path.exists():
        shutil.rmtree(consolidation_path)
    consolidation_capture: dict[str, object] = {}

    def retention_probe(candidate: ExecutableArtifactMemory):
        outcomes = _probe_views(
            parent,
            candidate,
            source_keys,
            views,
            source_ids,
            grammar,
            count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 90_000,
        )
        consolidation_capture["retention"] = [
            list(map(float, probe.outcomes)) for probe in outcomes
        ]
        return outcomes

    def verifier(candidate: ExecutableArtifactMemory) -> bool:
        outcomes = _probe_views(
            parent,
            candidate,
            source_keys,
            views,
            source_ids,
            grammar,
            count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 91_000,
        )
        passed = all(
            min(float(value) for value in probe.outcomes) >= THRESHOLD
            for probe in outcomes
        )
        consolidation_capture["verified"] = passed
        return passed

    consolidation_receipt = lifecycle.consolidate(
        tuple(source_rows),
        replacement_key,
        packed_artifact,
        consolidation_path,
        replacement_aliases=source_keys,
        replacement_alias_views=views,
        verifier=verifier,
        candidate_outcome_probe=retention_probe,
        retained_scores=source_behavior,
        candidate_threshold=THRESHOLD,
        retention_floor=THRESHOLD,
        min_candidate_observations=args.retention_probes,
    )
    compacted_reload = (
        ExecutableArtifactMemory.load(consolidation_path)
        if consolidation_receipt.accepted
        else None
    )
    compacted_behavior: dict[str, float] = {}
    compacted_views: dict[str, str | None] = {}
    if compacted_reload is not None:
        for key, view, source_id in zip(source_keys, views, source_ids, strict=True):
            handle, packed = compacted_reload.promote(key)
            compacted_views[view] = handle.view
            compacted_behavior[view] = _source_behavior(
                parent,
                _view_artifact(packed, view),
                source_id,
                grammar,
                count=args.audit_count,
                seed=args.seed + 92_000,
            )

    selected_artifact = (
        None
        if selection.selected_index is None
        else candidate_artifacts[selection.selected_index]
    )
    selected_behavior = (
        None
        if selection.selected_index is None
        else candidate_behavior[selection.selected_index]
    )
    target_key = _generated_key(
        parent,
        args.target_id,
        count=args.audit_count,
        seed=args.seed + 93_000,
        generated_compositions=grammar,
    )
    target_admission = None
    target_reload_preserved = False
    final_bank = None
    if (
        selection.accepted
        and selected_artifact is not None
        and selected_behavior is not None
        and selected_behavior >= THRESHOLD
        and consolidation_receipt.accepted
    ):
        if compacted_reload is None:
            raise RuntimeError("target cannot be admitted without compacted memory")
        compact_lifecycle = ExternalCapabilityLifecycle(compacted_reload)
        plan = compact_lifecycle.plan_admission(target_key, selected_artifact)
        target_admission = compact_lifecycle.admit(
            target_key,
            selected_artifact,
            plan=plan,
            grow_destination=args.report_out.parent / "grown_bank",
        )
        if target_admission.accepted:
            final_bank = ExecutableArtifactMemory.load(
                args.report_out.parent / "grown_bank"
            )

    parent_digest_after = _digest_core(parent, ())
    source_after = [compacted_behavior.get(view, 0.0) for view in views]
    inherited_stable = candidate_stable[:2]
    fresh_stable = candidate_stable[2]
    inherited_winner = any(
        value is not None
        and fresh_stable is not None
        and value < fresh_stable
        for value in inherited_stable
    )
    target_behavior_after = (
        _source_behavior(
            parent,
            final_bank.promote(target_key)[1]
            if final_bank is not None and target_admission is not None
            else selected_artifact,
            args.target_id,
            grammar,
            count=args.audit_count,
            seed=args.seed + 94_000,
        )
        if target_admission is not None and target_admission.accepted
        else None
    )
    target_reload_preserved = (
        target_admission is not None
        and target_admission.accepted
        and target_behavior_after is not None
        and target_behavior_after >= THRESHOLD
    )
    report = {
        "schema": "neural-computer.generated-composition-multi-transfer-report.v1",
        "claim_boundary": (
            "Two protected external artifacts were compared as frozen-file "
            "initializations for a new runtime program, then transactionally "
            "co-located as independently addressable opaque views. This is "
            "multi-source bounded external transfer and logical compaction, "
            "not unrestricted continual learning or neural weight compression."
        ),
        "seed": args.seed,
        "source_ids": list(source_ids),
        "target_id": args.target_id,
        "programs": [list(program) for program in grammar],
        "sources": [
            {
                "id": source_id,
                "stable_bits_to_threshold": source_stable[index],
                "behavior": source_behavior[index],
                "retention": source_retention[index],
                "artifact_digest": source_digests[index],
                "progress": source_progress[index],
            }
            for index, source_id in enumerate(source_ids)
        ],
        "candidates": [
            {
                "name": candidate_names[index],
                "stable_bits_to_threshold": candidate_stable[index],
                "behavior": candidate_behavior[index],
                "progress": candidate_progress[index],
            }
            for index in range(len(candidate_names))
        ],
        "candidate_selection": {
            "accepted": selection.accepted,
            "selected_index": selection.selected_index,
            "stable_bits_to_threshold": selection.stable_bits_to_threshold,
            "reason": selection.reason,
            "inherited_winner": inherited_winner,
        },
        "source_bank": {
            "capacity": 2,
            "rows": source_rows,
            "protected": source_protected,
            "address_similarity": address_similarity.tolist(),
            "addresses_separated": addresses_separated,
        },
        "consolidation": {
            "accepted": consolidation_receipt.accepted,
            "rows_before": consolidation_receipt.rows_before,
            "rows_after": consolidation_receipt.rows_after,
            "rows_saved": consolidation_receipt.rows_saved,
            "views": views,
            "resolved_views_after_reload": compacted_views,
            "behavior_after_reload": compacted_behavior,
            "retention_probe_outcomes": consolidation_capture.get("retention", []),
            "verifier_passed": consolidation_capture.get("verified", False),
            "protected_after_reload": [
                compacted_reload is not None
                and compacted_reload.retention.is_protected(key)
                for key in source_keys
            ],
        },
        "target_admission": None
        if target_admission is None
        else {
            "accepted": target_admission.accepted,
            "action": target_admission.action,
            "source_capacity": target_admission.source_capacity,
            "destination_capacity": target_admission.destination_capacity,
            "reason": target_admission.reason,
            "behavior_after_admission": target_behavior_after,
        },
        "frozen_core": {
            "digest_before": parent_digest_before,
            "digest_after": parent_digest_after,
            "unchanged": parent_digest_before == parent_digest_after,
        },
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + len(source_ids) * args.source_updates * args.batch_size * (SPAN + 2)
                + 3 * args.candidate_updates * args.batch_size * (SPAN + 2)
                + len(source_ids) * args.retention_probes * args.audit_count * SPAN
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + len(source_ids) * args.source_updates * args.batch_size * 2
                + 3 * args.candidate_updates * args.batch_size * 2
                + len(source_ids) * args.retention_probes * args.audit_count
            ),
            "optimizer_updates": (
                args.parent_updates
                + len(source_ids) * args.source_updates
                + 3 * args.candidate_updates
            ),
            "replayed_examples": 0,
        },
        "gates": {
            "parent_stable": _stable_bits(
                parent_progress,
                threshold=THRESHOLD,
                bits_per_update=args.batch_size * 2,
            )
            is not None,
            "sources_stable": all(value is not None for value in source_stable),
            "sources_mastered": min(source_behavior) >= THRESHOLD,
            "sources_protected": source_protected,
            "source_addresses_separated": addresses_separated,
            "candidate_stable": selection.accepted,
            "candidate_mastered": selected_behavior is not None
            and selected_behavior >= THRESHOLD,
            "positive_multi_source_transfer": inherited_winner,
            "consolidation_accepted": consolidation_receipt.accepted,
            "consolidation_saved_row": consolidation_receipt.rows_saved == 1,
            "all_views_reloaded": all(
                compacted_views.get(view) == view for view in views
            ),
            "all_views_retained": min(source_after) >= THRESHOLD,
            "all_views_protected": all(
                compacted_reload is not None
                and compacted_reload.retention.is_protected(key)
                for key in source_keys
            ),
            "target_grown_after_compaction": target_admission is not None
            and target_admission.accepted
            and target_admission.action == "grow",
            "target_mastered_after_admission": target_behavior_after is not None
            and target_behavior_after >= THRESHOLD,
            "target_reload_preserved": target_reload_preserved,
            "core_unchanged": parent_digest_before == parent_digest_after,
            "no_replayed_examples": True,
        },
        "wall_seconds": perf_counter() - started,
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--source-updates", type=int, default=256)
    parser.add_argument("--candidate-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--retention-probes", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--source-ids", type=int, nargs=2, default=(0, 2))
    parser.add_argument("--target-id", type=int, default=1)
    parser.add_argument(
        "--program-spec",
        action="append",
        default=None,
        help="verifier-private runtime program as comma-separated primitives",
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "candidate_selection": report["candidate_selection"],
                "consolidation": report["consolidation"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
