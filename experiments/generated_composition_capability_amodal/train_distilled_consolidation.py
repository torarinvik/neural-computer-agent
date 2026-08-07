"""Audit fresh-outcome neural consolidation of multiple external files.

The earlier multi-source audit compacted two files into one row while retaining
two separate executable views.  This audit has a stricter replacement: one
external routed stack must execute both source procedures through one set of
weights.  The inherited student starts from one protected file and receives
fresh outcomes from both source procedures; a fresh student receives the same
budget.  Only a unique stable-prefix winner can replace the protected source
rows.  An explicit fresh-rebuild opt-in allows the fresh winner to replace
them after independent retention verification.  The accepted replacement is
then tested for target transfer and capacity-safe admission.

This is still bounded learned consolidation.  A fresh rebuild is not claimed
as positive transfer from inherited weights.  The audit does not claim
arbitrary program induction or general continual learning.
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

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    SPAN,
    THRESHOLD,
    _generated_key,
    _load_stack_artifact,
    _parse_program_specs,
    _stack_artifact,
)
from experiments.generated_composition_capability_amodal.train_multi_transfer import (
    DEFAULT_RUNTIME_GRAMMAR,
    _candidate_scores,
    _source_behavior,
)
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    _new_capability,
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

from .train_pipeline import _new_stack, expand_routed_stack


def _digest_artifact(artifact: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(artifact.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _payload_bytes(artifact: dict[str, torch.Tensor]) -> int:
    return sum(int(value.numel() * value.element_size()) for value in artifact.values())


def _load_or_fresh(
    artifact: dict[str, torch.Tensor] | None,
    *,
    stack_seed: int,
    decoder_seed: int,
    program_count: int,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if artifact is not None:
        stack, decoder = _load_stack_artifact(artifact)
        while len(stack.programs) < program_count:
            stack = expand_routed_stack(stack, seed=stack_seed + len(stack.programs))
        if len(stack.programs) != program_count:
            raise ValueError("artifact has more routed slots than requested")
        return stack, decoder
    return (
        _new_stack(stack_seed, program_count=program_count, stack="routed"),
        _new_capability(decoder_seed)[1],
    )


def _train_program(
    parent,
    artifact: dict[str, torch.Tensor] | None,
    program_ids: tuple[int, ...],
    grammar,
    *,
    updates: int,
    batch_size: int,
    audit_count: int,
    eval_every: int,
    seed: int,
    learning_rate: float,
    program_count: int = 2,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]]]:
    stack, decoder = _load_or_fresh(
        artifact,
        stack_seed=seed + 1_000,
        decoder_seed=seed + 2_000,
        program_count=program_count,
    )
    torch.manual_seed(seed + 3_000)
    _history, progress = _train_capability(
        parent,
        stack,
        decoder,
        operation="generated_composition",
        span=SPAN,
        updates=updates,
        batch_size=batch_size,
        seed=seed,
        audit_count=audit_count,
        eval_every=eval_every,
        learning_rate=learning_rate,
        generated_composition_ids=program_ids,
        generated_compositions=grammar,
    )
    stack.eval()
    decoder.eval()
    return _stack_artifact(stack, decoder), progress


def _probe_artifact(
    parent,
    artifact: dict[str, torch.Tensor],
    program_ids: tuple[int, ...],
    grammar,
    *,
    count: int,
    probes: int,
    seed: int,
) -> tuple[CapabilityRetentionProbe, ...]:
    result: list[CapabilityRetentionProbe] = []
    for index, program_id in enumerate(program_ids):
        outcomes = [
            _source_behavior(
                parent,
                artifact,
                program_id,
                grammar,
                count=count,
                seed=seed + index * 10_003 + probe * 101,
            )
            for probe in range(probes)
        ]
        result.append(CapabilityRetentionProbe(torch.empty(0), outcomes))
    return tuple(result)


def _probe_bank_aliases(
    parent,
    candidate: ExecutableArtifactMemory,
    keys: tuple[torch.Tensor, ...],
    program_ids: tuple[int, ...],
    grammar,
    *,
    count: int,
    probes: int,
    seed: int,
) -> tuple[CapabilityRetentionProbe, ...]:
    result: list[CapabilityRetentionProbe] = []
    for index, (key, program_id) in enumerate(zip(keys, program_ids, strict=True)):
        outcomes: list[float] = []
        for probe in range(probes):
            try:
                handle, artifact = candidate.promote(key)
                valid = handle.view is None
            except (LookupError, ValueError):
                valid = False
                artifact = {}
            outcomes.append(
                _source_behavior(
                    parent,
                    artifact,
                    program_id,
                    grammar,
                    count=count,
                    seed=seed + index * 10_003 + probe * 101,
                )
                if valid
                else 0.0
            )
        result.append(CapabilityRetentionProbe(key, outcomes))
    return tuple(result)


def _probe_candidate_behaviors(
    parent,
    artifact: dict[str, torch.Tensor],
    program_ids: tuple[int, ...],
    grammar,
    *,
    count: int,
    seed: int,
) -> dict[str, float]:
    return {
        str(program_id): _source_behavior(
            parent,
            artifact,
            program_id,
            grammar,
            count=count,
            seed=seed + index * 10_003,
        )
        for index, program_id in enumerate(program_ids)
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.parent_updates,
        args.source_updates,
        args.consolidation_updates,
        args.target_updates,
        args.batch_size,
        args.audit_count,
        args.retention_probes,
        args.eval_every,
    ) < 1:
        raise ValueError("all update and audit budgets must be positive")
    if min(args.source_program_count, args.student_program_count) < 2:
        raise ValueError("external routed program counts must be at least two")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")
    if not 0.0 <= args.behavior_margin <= 1.0:
        raise ValueError("behavior margin must lie in [0, 1]")
    source_ids = tuple(args.source_ids)
    if len(source_ids) < 2 or len(set(source_ids)) != len(source_ids):
        raise ValueError("at least two distinct source IDs are required")
    grammar = _parse_program_specs(
        args.program_spec
        or [",".join(program) for program in DEFAULT_RUNTIME_GRAMMAR]
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
    source_behavior: list[float] = []
    for index, program_id in enumerate(source_ids):
        artifact, progress = _train_program(
            parent,
            None,
            (program_id,),
            grammar,
            updates=args.source_updates,
            batch_size=args.batch_size,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            seed=args.seed + 20_000 + index * 10_003,
            learning_rate=args.learning_rate,
            program_count=args.source_program_count,
        )
        source_artifacts.append(artifact)
        source_progress.append(progress)
        source_behavior.append(
            _source_behavior(
                parent,
                artifact,
                program_id,
                grammar,
                count=args.audit_count,
                seed=args.seed + 40_000 + index,
            )
        )

    # The inherited student owns one stack and starts from source 0 only. It
    # then sees fresh outcomes from both source procedures, never old source
    # trajectories or source artifact snapshots as training examples.
    inherited_student, inherited_progress = _train_program(
        parent,
        source_artifacts[0],
        source_ids,
        grammar,
        updates=args.consolidation_updates,
        batch_size=args.batch_size,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        seed=args.seed + 50_000,
        learning_rate=args.learning_rate,
        program_count=args.student_program_count,
    )
    fresh_student, fresh_progress = _train_program(
        parent,
        None,
        source_ids,
        grammar,
        updates=args.consolidation_updates,
        batch_size=args.batch_size,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        seed=args.seed + 50_000,
        learning_rate=args.learning_rate,
        program_count=args.student_program_count,
    )
    inherited_behavior = _probe_candidate_behaviors(
        parent,
        inherited_student,
        source_ids,
        grammar,
        count=args.audit_count,
        seed=args.seed + 60_000,
    )
    fresh_behavior = _probe_candidate_behaviors(
        parent,
        fresh_student,
        source_ids,
        grammar,
        count=args.audit_count,
        seed=args.seed + 60_000,
    )
    bits_per_observation = args.eval_every * args.batch_size * SPAN
    student_selection = select_capability_candidate(
        (
            _candidate_scores(inherited_progress),
            _candidate_scores(fresh_progress),
        ),
        threshold=THRESHOLD,
        bits_per_observation=bits_per_observation,
    )
    inherited_min_behavior = min(inherited_behavior.values())
    fresh_min_behavior = min(fresh_behavior.values())
    primary_inherited_winner = (
        student_selection.accepted and student_selection.selected_index == 0
    )
    inherited_weights_help = (
        primary_inherited_winner
        or (
            inherited_min_behavior >= fresh_min_behavior + args.behavior_margin
            and not student_selection.accepted
        )
    )
    selected_student = inherited_student if inherited_weights_help else None
    selected_student_source = "inherited" if inherited_weights_help else None
    if (
        selected_student is None
        and args.allow_fresh_consolidation
        and student_selection.accepted
        and student_selection.selected_index == 1
    ):
        selected_student = fresh_student
        selected_student_source = "fresh_rebuild"
    selected_behavior = (
        inherited_behavior
        if selected_student_source == "inherited"
        else fresh_behavior
        if selected_student_source == "fresh_rebuild"
        else {}
    )

    source_path = args.report_out.parent / "source_bank"
    if source_path.exists():
        shutil.rmtree(source_path)
    bank = ExecutableArtifactMemory(
        source_path,
        width=48,
        capacity=len(source_ids),
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
            program_id,
            count=args.audit_count,
            seed=args.seed + 70_000 + index,
            generated_compositions=grammar,
        )
        for index, program_id in enumerate(source_ids)
    )
    address_similarity = F.normalize(torch.stack(source_keys), dim=-1) @ F.normalize(
        torch.stack(source_keys), dim=-1
    ).T
    addresses_separated = bool(
        torch.all(
            address_similarity[~torch.eye(len(source_ids), dtype=torch.bool)]
            < 0.99999
        )
    )
    if not addresses_separated:
        raise RuntimeError("source memory addresses are not independently resolvable")
    source_rows: list[int] = []
    for source_index, (key, artifact) in enumerate(
        zip(source_keys, source_artifacts, strict=True)
    ):
        receipt = lifecycle.admit(key, artifact)
        if not receipt.accepted or receipt.index is None:
            raise RuntimeError(f"source admission failed: {receipt.reason}")
        source_rows.append(receipt.index)
        outcomes = [
            _source_behavior(
                parent,
                artifact,
                program_id,
                grammar,
                count=args.audit_count,
                seed=args.seed + 80_000 + receipt.index * 10_003 + probe * 101,
            )
            for probe in range(args.retention_probes)
            for program_id in source_ids[source_index : source_index + 1]
        ]
        for outcome in outcomes:
            bank.observe_retention(key, outcome)
    bank.save()
    sources_protected = bool(bank.protection_mask().all())

    consolidation_path = args.report_out.parent / "neural_consolidated_bank"
    if consolidation_path.exists():
        shutil.rmtree(consolidation_path)
    consolidation_capture: dict[str, object] = {}

    def retention_probe(candidate: ExecutableArtifactMemory):
        outcomes = _probe_bank_aliases(
            parent,
            candidate,
            source_keys,
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
        outcomes = _probe_bank_aliases(
            parent,
            candidate,
            source_keys,
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

    replacement_key = F.normalize(torch.stack(source_keys).mean(dim=0), dim=0)
    consolidation_receipt = None
    if selected_student is not None:
        consolidation_receipt = lifecycle.consolidate(
            tuple(source_rows),
            replacement_key,
            selected_student,
            consolidation_path,
            replacement_aliases=source_keys,
            verifier=verifier,
            candidate_outcome_probe=retention_probe,
            retained_scores=source_behavior,
            candidate_threshold=THRESHOLD,
            retention_floor=THRESHOLD,
            min_candidate_observations=args.retention_probes,
        )
    compacted_reload = (
        ExecutableArtifactMemory.load(consolidation_path)
        if consolidation_receipt is not None and consolidation_receipt.accepted
        else None
    )
    compacted_behavior: dict[str, float] = {}
    compacted_alias_digests: dict[str, str] = {}
    if compacted_reload is not None:
        for key, program_id in zip(source_keys, source_ids, strict=True):
            handle, artifact = compacted_reload.promote(key)
            if handle.view is not None:
                raise RuntimeError("neural consolidation unexpectedly returned a view")
            compacted_behavior[str(program_id)] = _source_behavior(
                parent,
                artifact,
                program_id,
                grammar,
                count=args.audit_count,
                seed=args.seed + 92_000,
            )
            compacted_alias_digests[str(program_id)] = _digest_artifact(artifact)

    # Test transfer from the single consolidated network only after its
    # behavior has passed. This measures whether shared computation is useful,
    # not merely smaller on disk.
    target_candidates: list[dict[str, torch.Tensor]] = []
    target_progress: list[list[dict[str, float | int]]] = []
    if selected_student is not None:
        for candidate_index, initial in enumerate((selected_student, None)):
            artifact, progress = _train_program(
                parent,
                initial,
                (args.target_id,),
                grammar,
                updates=args.target_updates,
                batch_size=args.batch_size,
                audit_count=args.audit_count,
                eval_every=args.eval_every,
                seed=args.seed + 100_000,
                learning_rate=args.learning_rate,
                program_count=args.student_program_count,
            )
            target_candidates.append(artifact)
            target_progress.append(progress)
    target_selection = (
        select_capability_candidate(
            tuple(_candidate_scores(progress) for progress in target_progress),
            threshold=THRESHOLD,
            bits_per_observation=bits_per_observation,
        )
        if target_progress
        else None
    )
    selected_target = (
        None
        if target_selection is None or target_selection.selected_index is None
        else target_candidates[target_selection.selected_index]
    )
    target_behavior = (
        [
            _source_behavior(
                parent,
                artifact,
                args.target_id,
                grammar,
                count=args.audit_count,
                seed=args.seed + 110_000,
            )
            for artifact in target_candidates
        ]
        if target_candidates
        else []
    )
    target_admission = None
    final_bank = None
    target_behavior_after = None
    if (
        compacted_reload is not None
        and target_selection is not None
        and target_selection.accepted
        and selected_target is not None
        and target_behavior[target_selection.selected_index] >= THRESHOLD
    ):
        compact_lifecycle = ExternalCapabilityLifecycle(compacted_reload)
        plan = compact_lifecycle.plan_admission(
            _generated_key(
                parent,
                args.target_id,
                count=args.audit_count,
                seed=args.seed + 120_000,
                generated_compositions=grammar,
            ),
            selected_target,
        )
        target_key = _generated_key(
            parent,
            args.target_id,
            count=args.audit_count,
            seed=args.seed + 120_000,
            generated_compositions=grammar,
        )
        target_admission = compact_lifecycle.admit(
            target_key,
            selected_target,
            plan=plan,
            grow_destination=args.report_out.parent / "grown_bank",
        )
        if target_admission.accepted:
            final_bank = ExecutableArtifactMemory.load(
                args.report_out.parent / "grown_bank"
            )
            _, reloaded_target = final_bank.promote(target_key)
            target_behavior_after = _source_behavior(
                parent,
                reloaded_target,
                args.target_id,
                grammar,
                count=args.audit_count,
                seed=args.seed + 111_000,
            )

    parent_digest_after = _digest_core(parent, ())
    source_payload_bytes = sum(_payload_bytes(artifact) for artifact in source_artifacts)
    selected_payload_bytes = (
        _payload_bytes(selected_student) if selected_student is not None else 0
    )
    consolidation_accepted = (
        consolidation_receipt is not None and consolidation_receipt.accepted
    )
    inherited_transfer_required = selected_student_source == "inherited"
    report = {
        "schema": "neural-computer.generated-composition-distilled-consolidation-report.v1",
        "claim_boundary": (
            "One external routed stack was trained from fresh outcomes to execute "
            "prior procedures, replacing protected source files through one "
            "shared executable artifact. Inherited and explicitly permitted "
            "fresh-rebuild winners are behavior-verified separately; a fresh "
            "rebuild is not positive transfer from inherited weights. Target "
            "transfer is required only for inherited candidates and remains "
            "unqualified for fresh rebuilds. This is bounded neural "
            "consolidation, not general continual learning."
        ),
        "seed": args.seed,
        "source_ids": list(source_ids),
        "target_id": args.target_id,
        "source_program_count": args.source_program_count,
        "student_program_count": args.student_program_count,
        "programs": [list(program) for program in grammar],
        "sources": [
            {
                "id": program_id,
                "behavior": source_behavior[index],
                "stable_bits_to_threshold": _stable_bits(
                    source_progress[index],
                    threshold=THRESHOLD,
                    bits_per_update=args.batch_size * SPAN,
                ),
                "progress": source_progress[index],
                "artifact_digest": _digest_artifact(source_artifacts[index]),
            }
            for index, program_id in enumerate(source_ids)
        ],
        "student_candidates": {
            "inherited": {
                "behavior": inherited_behavior,
                "stable_bits_to_threshold": student_selection.stable_bits_to_threshold[0],
                "progress": inherited_progress,
            },
            "fresh": {
                "behavior": fresh_behavior,
                "stable_bits_to_threshold": student_selection.stable_bits_to_threshold[1],
                "progress": fresh_progress,
            },
        },
        "student_selection": {
            "accepted": student_selection.accepted,
            "selected_index": student_selection.selected_index,
            "stable_bits_to_threshold": student_selection.stable_bits_to_threshold,
            "reason": student_selection.reason,
            "selection_policy": (
                "stable-prefix selector; tied prefixes require a fresh "
                "maximin behavior margin for inherited weights"
            ),
            "inherited_min_behavior": inherited_min_behavior,
            "fresh_min_behavior": fresh_min_behavior,
            "primary_inherited_winner": primary_inherited_winner,
            "inherited_weights_help": inherited_weights_help,
            "selected_student_source": selected_student_source,
            "allow_fresh_consolidation": args.allow_fresh_consolidation,
            "inherited_transfer_required": inherited_transfer_required,
            "behavior_margin": args.behavior_margin,
        },
        "source_bank": {
            "rows": source_rows,
            "capacity": len(source_ids),
            "protected": sources_protected,
            "address_similarity": address_similarity.tolist(),
            "addresses_separated": addresses_separated,
        },
        "consolidation": {
            "accepted": consolidation_accepted,
            "rows_before": (
                consolidation_receipt.rows_before
                if consolidation_receipt
                else len(source_ids)
            ),
            "rows_after": (
                consolidation_receipt.rows_after
                if consolidation_receipt
                else len(source_ids)
            ),
            "rows_saved": (
                consolidation_receipt.rows_saved if consolidation_receipt else 0
            ),
            "shared_artifact": consolidation_accepted,
            "behavior_after_reload": compacted_behavior,
            "alias_digests_after_reload": compacted_alias_digests,
            "retention_probe_outcomes": consolidation_capture.get("retention", []),
            "verifier_passed": consolidation_capture.get("verified", False),
            "protected_after_reload": [
                compacted_reload.retention.is_protected(key)
                if compacted_reload is not None
                else False
                for key in source_keys
            ],
            "source_payload_bytes": source_payload_bytes,
            "replacement_payload_bytes": selected_payload_bytes,
            "payload_ratio": selected_payload_bytes / source_payload_bytes
            if source_payload_bytes
            else None,
        },
        "target_candidates": {
            "stable_bits_to_threshold": (
                target_selection.stable_bits_to_threshold
                if target_selection is not None
                else []
            ),
            "behavior": target_behavior,
            "selected_index": (
                target_selection.selected_index if target_selection is not None else None
            ),
        },
        "target_admission": None
        if target_admission is None
        else {
            "accepted": target_admission.accepted,
            "action": target_admission.action,
            "source_capacity": target_admission.source_capacity,
            "destination_capacity": target_admission.destination_capacity,
            "behavior_after_reload": target_behavior_after,
        },
        "frozen_core": {
            "digest_before": parent_digest_before,
            "digest_after": parent_digest_after,
            "unchanged": parent_digest_before == parent_digest_after,
        },
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + len(source_ids)
                * args.source_updates
                * args.batch_size
                * (SPAN + 2)
                + 2 * args.consolidation_updates * args.batch_size * (SPAN + 2)
                + 2 * args.target_updates * args.batch_size * (SPAN + 2)
                + 2 * args.retention_probes * args.audit_count * SPAN
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + len(source_ids) * args.source_updates * args.batch_size * 2
                + 2 * args.consolidation_updates * args.batch_size * 2
                + 2 * args.target_updates * args.batch_size * 2
                + 2 * args.retention_probes * args.audit_count
            ),
            "optimizer_updates": (
                args.parent_updates
                + len(source_ids) * args.source_updates
                + 2 * args.consolidation_updates
                + 2 * args.target_updates
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
            "sources_stable": all(
                _stable_bits(
                    source_progress[index],
                    threshold=THRESHOLD,
                    bits_per_update=args.batch_size * SPAN,
                )
                is not None
                for index in range(len(source_ids))
            ),
            "sources_mastered": min(source_behavior) >= THRESHOLD,
            "sources_protected": sources_protected,
            "source_addresses_separated": addresses_separated,
            "student_candidate_verified": selected_student is not None,
            "student_all_sources_mastered": len(selected_behavior) == len(source_ids)
            and min(selected_behavior.values()) >= THRESHOLD,
            "shared_artifact_payload_reduced": selected_payload_bytes
            < source_payload_bytes,
            "consolidation_accepted": consolidation_accepted,
            "consolidation_saved_rows": consolidation_accepted
            and consolidation_receipt.rows_saved == len(source_ids) - 1,
            "consolidated_aliases_share_artifact": consolidation_accepted
            and len(set(compacted_alias_digests.values())) == 1,
            "consolidated_sources_retained": consolidation_accepted
            and min(compacted_behavior.values(), default=0.0) >= THRESHOLD,
            "consolidated_sources_protected": all(
                compacted_reload is not None
                and compacted_reload.retention.is_protected(key)
                for key in source_keys
            ),
            "target_unique_winner": (
                not inherited_transfer_required
                or (
                    target_selection is not None
                    and target_selection.accepted
                    and target_selection.selected_index == 0
                )
            ),
            "target_grown_after_consolidation": (
                not inherited_transfer_required
                or (
                    target_admission is not None
                    and target_admission.accepted
                    and target_admission.action == "grow"
                )
            ),
            "target_reloaded_mastered": (
                not inherited_transfer_required
                or (
                    target_behavior_after is not None
                    and target_behavior_after >= THRESHOLD
                )
            ),
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
    parser.add_argument("--consolidation-updates", type=int, default=256)
    parser.add_argument("--target-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--retention-probes", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--source-program-count",
        type=int,
        default=2,
        help="external routed slots used by independently acquired source files",
    )
    parser.add_argument(
        "--student-program-count",
        type=int,
        default=2,
        help="external routed slots available to consolidation and target transfer",
    )
    parser.add_argument(
        "--allow-fresh-consolidation",
        action="store_true",
        help=(
            "permit a fresh-outcome winner to replace protected rows after the "
            "independent retention verifier passes"
        ),
    )
    parser.add_argument(
        "--behavior-margin",
        type=float,
        default=0.02,
        help="minimum inherited-over-fresh worst-source behavior margin",
    )
    parser.add_argument("--source-ids", type=int, nargs="+", default=(0, 2))
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
                "student_selection": report["student_selection"],
                "consolidation": report["consolidation"],
                "target_admission": report["target_admission"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
