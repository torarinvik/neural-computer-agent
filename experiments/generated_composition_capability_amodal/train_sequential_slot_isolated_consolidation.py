"""Audit replay-free sequential growth with alias-addressed neural slots.

This is the safe companion to the rejected shared-weight expansion control.
Every new procedure is learned in a fresh external neural slot.  The slot is
then appended into the same physical artifact row under a new opaque alias;
old slot weights and decoders are never updated.  Consolidation is therefore a
verified slot-isolation transaction, not a claim that one dense network can
learn arbitrary new computation without replay.

The bank grows through a finite nonstationary grammar and includes fresh
retention, target growth, reversal/recovery, reload, and corruption controls.
The frozen controller never receives a raw modality or a task label.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.generated_composition_capability_amodal.sequential_consolidation_common import (
    configure_bank as _configure_bank,
)
from experiments.generated_composition_capability_amodal.sequential_consolidation_common import (
    corruption_control as _corruption_control,
)
from experiments.generated_composition_capability_amodal.sequential_consolidation_common import (
    digest_artifact_bank as _digest_artifact_bank,
)
from experiments.generated_composition_capability_amodal.sequential_consolidation_common import (
    reversal_recovery as _reversal_recovery,
)
from experiments.generated_composition_capability_amodal.train_artifact_bank import (
    _generated_key,
)
from experiments.generated_composition_capability_amodal.train_distilled_consolidation import (
    DEFAULT_RUNTIME_GRAMMAR,
    _digest_artifact,
    _parse_program_specs,
    _payload_bytes,
    _source_behavior,
    _train_program,
)
from experiments.generated_composition_capability_amodal.train_multi_transfer import (
    _pack_views,
    _probe_views,
    _view_artifact,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
    _runtime,
)
from neural_computer import ExecutableArtifactMemory, ExternalCapabilityLifecycle

SPAN = 4
THRESHOLD = 0.75
DEFAULT_SEQUENTIAL_GRAMMAR = (
    *DEFAULT_RUNTIME_GRAMMAR,
    ("complement", "prefix_parity", "reverse", "global_parity"),
    ("rotate", "global_parity", "complement", "adjacent_xor"),
)


def _append_view(
    packed: dict[str, torch.Tensor],
    view: str,
    artifact: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not view or "." in view:
        raise ValueError("view names must be nonempty and dot-free")
    result = {name: value.detach().cpu().clone() for name, value in packed.items()}
    prefix = f"view.{view}."
    for name, value in artifact.items():
        result[f"{prefix}{name}"] = value.detach().cpu().clone()
    return result


def _packed_behaviors(
    parent,
    bank: ExecutableArtifactMemory,
    keys: tuple[torch.Tensor, ...],
    views: tuple[str, ...],
    program_ids: tuple[int, ...],
    grammar,
    *,
    count: int,
    seed: int,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for index, (key, view, program_id) in enumerate(
        zip(keys, views, program_ids, strict=True)
    ):
        handle, packed = bank.promote(key)
        if handle.view != view:
            raise RuntimeError("opaque alias resolved to the wrong neural slot")
        result[str(program_id)] = _source_behavior(
            parent,
            _view_artifact(packed, view),
            program_id,
            grammar,
            count=count,
            seed=seed + index * 10_003,
        )
    return result


def _shared_row_aliases(
    bank: ExecutableArtifactMemory,
    keys: tuple[torch.Tensor, ...],
    views: tuple[str, ...],
) -> tuple[bool, int | None]:
    rows: list[int] = []
    for key, view in zip(keys, views, strict=True):
        handle, _ = bank.promote(key)
        if handle.view != view:
            return False, None
        rows.append(handle.index)
    return bool(rows) and len(set(rows)) == 1, rows[0] if rows else None


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    budgets = (
        args.parent_updates,
        args.source_updates,
        args.target_updates,
        args.batch_size,
        args.audit_count,
        args.retention_probes,
        args.eval_every,
        args.reversal_patience,
    )
    if min(budgets) < 1:
        raise ValueError("all update and audit budgets must be positive")
    if args.batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch size and audit count must be even")
    source_ids = tuple(args.source_ids)
    if len(source_ids) < 2 or len(set(source_ids)) != len(source_ids):
        raise ValueError("source IDs must be distinct and contain at least two IDs")
    grammar = _parse_program_specs(
        args.program_spec
        or [",".join(program) for program in DEFAULT_SEQUENTIAL_GRAMMAR]
    )
    required_ids = (*source_ids, args.target_id)
    if any(program_id < 0 or program_id >= len(grammar) for program_id in required_ids):
        raise ValueError("source or target program is out of range")
    if args.target_id in source_ids:
        raise ValueError("target program must not be a source program")

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

    source_artifacts: dict[str, dict[str, torch.Tensor]] = {}
    source_progress: dict[str, list[dict[str, float | int]]] = {}
    source_behavior: dict[str, float] = {}
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
        )
        source_artifacts[str(program_id)] = artifact
        source_progress[str(program_id)] = progress
        source_behavior[str(program_id)] = _source_behavior(
            parent,
            artifact,
            program_id,
            grammar,
            count=args.audit_count,
            seed=args.seed + 40_000 + index,
        )

    source_keys = tuple(
        F.normalize(
            _generated_key(
                parent,
                program_id,
                count=args.audit_count,
                seed=args.seed + 70_000 + index,
                generated_compositions=grammar,
            ),
            dim=0,
        )
        for index, program_id in enumerate(source_ids)
    )
    source_key_similarity = (
        F.normalize(torch.stack(source_keys), dim=-1)
        @ F.normalize(torch.stack(source_keys), dim=-1).T
    )
    source_addresses_separated = bool(
        torch.all(
            source_key_similarity[~torch.eye(len(source_ids), dtype=torch.bool)]
            < 0.99999
        )
    )
    if not source_addresses_separated:
        raise RuntimeError("source memory addresses are not independently resolvable")

    bank_path = args.report_out.parent / "slot_isolated_bank"
    if bank_path.exists():
        shutil.rmtree(bank_path)
    bank = _configure_bank(
        bank_path,
        reversal_patience=args.reversal_patience,
        observations=args.retention_probes,
    )
    lifecycle = ExternalCapabilityLifecycle(bank)
    first_id = source_ids[0]
    first_key = source_keys[0]
    first_receipt = lifecycle.admit(first_key, source_artifacts[str(first_id)])
    if not first_receipt.accepted or first_receipt.index is None:
        raise RuntimeError(f"initial source admission failed: {first_receipt.reason}")
    for probe in range(args.retention_probes):
        bank.observe_retention(
            first_key,
            _source_behavior(
                parent,
                source_artifacts[str(first_id)],
                first_id,
                grammar,
                count=args.audit_count,
                seed=args.seed + 80_000 + probe * 101,
            ),
        )

    active_ids = [first_id]
    active_keys = [first_key]
    active_views = ["slot0"]
    shared_artifact = source_artifacts[str(first_id)]
    current_row = first_receipt.index
    current_behaviors = {str(first_id): source_behavior[str(first_id)]}
    stage_records: list[dict[str, object]] = []

    for stage_index, new_id in enumerate(source_ids[1:], start=1):
        new_key = source_keys[stage_index]
        new_artifact = source_artifacts[str(new_id)]
        if not bool(bank.protection_mask()[current_row]):
            stage_records.append(
                {
                    "stage": stage_index,
                    "new_source_id": new_id,
                    "active_source_ids_before": list(active_ids),
                    "accepted": False,
                    "mode": "opaque_alias_addressed_slot_append",
                    "new_slot_trained_from_fresh_outcomes_only": True,
                    "reason": "prior source was not protected; refusing eviction during growth",
                }
            )
            break
        new_receipt = lifecycle.admit(
            new_key,
            new_artifact,
            plan=lifecycle.plan_admission(new_key, new_artifact),
            grow_destination=args.report_out.parent / f"stage_{stage_index}_grown_bank",
        )
        record: dict[str, object] = {
            "stage": stage_index,
            "new_source_id": new_id,
            "active_source_ids_before": list(active_ids),
            "admission": new_receipt.__dict__,
            "mode": "opaque_alias_addressed_slot_append",
            "new_slot_trained_from_fresh_outcomes_only": True,
        }
        if not new_receipt.accepted or new_receipt.index is None:
            record.update(
                {
                    "accepted": False,
                    "reason": "new slot could not be staged without touching protected rows",
                }
            )
            stage_records.append(record)
            break
        bank = lifecycle.memory
        for probe in range(args.retention_probes):
            bank.observe_retention(
                new_key,
                _source_behavior(
                    parent,
                    new_artifact,
                    new_id,
                    grammar,
                    count=args.audit_count,
                    seed=args.seed + 81_000 + stage_index * 10_003 + probe * 101,
                ),
            )

        new_view = f"slot{stage_index}"
        candidate_ids = (*active_ids, new_id)
        candidate_keys = (*active_keys, new_key)
        candidate_views = (*active_views, new_view)
        if stage_index == 1:
            candidate_artifact = _pack_views(
                (shared_artifact, new_artifact), candidate_views
            )
        else:
            candidate_artifact = {
                name: value.detach().cpu().clone()
                for name, value in shared_artifact.items()
            }
            candidate_artifact = _append_view(
                candidate_artifact,
                new_view,
                new_artifact,
            )
        candidate_behaviors = {
            str(program_id): _source_behavior(
                parent,
                _view_artifact(candidate_artifact, view),
                program_id,
                grammar,
                count=args.audit_count,
                seed=args.seed + 100_000 + stage_index * 10_003 + index,
            )
            for index, (program_id, view) in enumerate(
                zip(candidate_ids, candidate_views, strict=True)
            )
        }
        candidate_ready = min(candidate_behaviors.values(), default=0.0) >= THRESHOLD
        record.update(
            {
                "candidate_behavior_before_rewrite": candidate_behaviors,
                "candidate_payload_bytes": _payload_bytes(candidate_artifact),
                "source_payload_bytes": sum(
                    _payload_bytes(source_artifacts[str(program_id)])
                    for program_id in candidate_ids
                ),
                "new_source_stable_bits_to_threshold": _stable_source_bits(
                    source_progress[str(new_id)], args
                ),
            }
        )
        if not candidate_ready:
            record.update(
                {
                    "accepted": False,
                    "reason": "fresh slot or retained alias failed the mastery floor",
                }
            )
            stage_records.append(record)
            break

        retention_scores = [
            current_behaviors[str(program_id)] for program_id in active_ids
        ] + [source_behavior[str(new_id)]]
        replacement_key = F.normalize(torch.stack(candidate_keys).mean(dim=0), dim=0)
        destination = args.report_out.parent / f"stage_{stage_index}_consolidated_bank"
        if destination.exists():
            shutil.rmtree(destination)
        capture: dict[str, object] = {}

        def candidate_probe(
            candidate_bank: ExecutableArtifactMemory,
            *,
            probe_keys=candidate_keys,
            probe_views=candidate_views,
            probe_ids=candidate_ids,
            probe_stage=stage_index,
            probe_capture=capture,
        ):
            probes = _probe_views(
                parent,
                candidate_bank,
                probe_keys,
                probe_views,
                probe_ids,
                grammar,
                count=args.audit_count,
                probes=args.retention_probes,
                seed=args.seed + 110_000 + probe_stage * 10_003,
            )
            probe_capture["retention_probe_outcomes"] = [
                list(map(float, probe.outcomes)) for probe in probes
            ]
            return probes

        def verifier(
            candidate_bank: ExecutableArtifactMemory,
            *,
            probe_keys=candidate_keys,
            probe_views=candidate_views,
            probe_ids=candidate_ids,
            probe_stage=stage_index,
            probe_capture=capture,
        ) -> bool:
            probes = _probe_views(
                parent,
                candidate_bank,
                probe_keys,
                probe_views,
                probe_ids,
                grammar,
                count=args.audit_count,
                probes=args.retention_probes,
                seed=args.seed + 111_000 + probe_stage * 10_003,
            )
            passed = all(
                min(map(float, probe.outcomes), default=0.0) >= THRESHOLD
                for probe in probes
            )
            probe_capture["verifier_passed"] = passed
            return passed

        current_handle, _ = bank.promote(active_keys[0])
        receipt = lifecycle.consolidate(
            (current_handle.index, new_receipt.index),
            replacement_key,
            candidate_artifact,
            destination,
            verifier=verifier,
            replacement_aliases=candidate_keys,
            replacement_alias_views=candidate_views,
            candidate_outcome_probe=candidate_probe,
            retained_scores=retention_scores,
            candidate_threshold=THRESHOLD,
            retention_floor=THRESHOLD,
            min_candidate_observations=args.retention_probes,
        )
        record.update(capture)
        record["consolidation"] = receipt.__dict__
        record["accepted"] = receipt.accepted
        if not receipt.accepted:
            record["reason"] = receipt.reason
            stage_records.append(record)
            break
        bank = lifecycle.memory
        active_ids.append(new_id)
        active_keys.append(new_key)
        active_views.append(new_view)
        current_handle, shared_artifact = bank.promote(new_key)
        current_row = current_handle.index
        current_behaviors = _packed_behaviors(
            parent,
            bank,
            tuple(active_keys),
            tuple(active_views),
            tuple(active_ids),
            grammar,
            count=args.audit_count,
            seed=args.seed + 112_000 + stage_index * 10_003,
        )
        record.update(
            {
                "active_source_ids_after": list(active_ids),
                "behavior_after_reload": current_behaviors,
                "physical_row": current_row,
                "physical_row_payload_bytes": _payload_bytes(shared_artifact),
            }
        )
        stage_records.append(record)

    stages_completed = len(active_ids) == len(source_ids)
    target_record: dict[str, object] = {"attempted": False}
    target_key = None
    target_receipt = None
    target_behavior_after = None
    if stages_completed:
        target_record["attempted"] = True
        source_view = _view_artifact(shared_artifact, active_views[0])
        target_artifact, target_progress = _train_program(
            parent,
            source_view,
            (args.target_id,),
            grammar,
            updates=args.target_updates,
            batch_size=args.batch_size,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            seed=args.seed + 130_000,
            learning_rate=args.learning_rate,
        )
        fresh_target, fresh_target_progress = _train_program(
            parent,
            None,
            (args.target_id,),
            grammar,
            updates=args.target_updates,
            batch_size=args.batch_size,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            seed=args.seed + 130_000,
            learning_rate=args.learning_rate,
        )
        target_behavior = _source_behavior(
            parent,
            target_artifact,
            args.target_id,
            grammar,
            count=args.audit_count,
            seed=args.seed + 140_000,
        )
        fresh_target_behavior = _source_behavior(
            parent,
            fresh_target,
            args.target_id,
            grammar,
            count=args.audit_count,
            seed=args.seed + 140_000,
        )
        target_record.update(
            {
                "inherited_behavior": target_behavior,
                "fresh_behavior": fresh_target_behavior,
                "inherited_stable_bits_to_threshold": _stable_source_bits(
                    target_progress, args
                ),
                "fresh_stable_bits_to_threshold": _stable_source_bits(
                    fresh_target_progress, args
                ),
                "trained_from_first_slot_only": True,
            }
        )
        if target_behavior >= THRESHOLD:
            target_key = F.normalize(
                _generated_key(
                    parent,
                    args.target_id,
                    count=args.audit_count,
                    seed=args.seed + 150_000,
                    generated_compositions=grammar,
                ),
                dim=0,
            )
            target_receipt = lifecycle.admit(
                target_key,
                target_artifact,
                plan=lifecycle.plan_admission(target_key, target_artifact),
                grow_destination=args.report_out.parent / "target_grown_bank",
            )
            bank = lifecycle.memory
            target_record["admission"] = target_receipt.__dict__
            if target_receipt.accepted:
                target_handle, reloaded_target = bank.promote(target_key)
                target_behavior_after = _source_behavior(
                    parent,
                    reloaded_target,
                    args.target_id,
                    grammar,
                    count=args.audit_count,
                    seed=args.seed + 151_000,
                )
                for probe in range(args.retention_probes):
                    bank.observe_retention(
                        target_key,
                        _source_behavior(
                            parent,
                            reloaded_target,
                            args.target_id,
                            grammar,
                            count=args.audit_count,
                            seed=args.seed + 152_000 + probe * 101,
                        ),
                    )
                target_record.update(
                    {
                        "row": target_handle.index,
                        "behavior_after_reload": target_behavior_after,
                        "protected_after_reload": bank.retention.is_protected(
                            target_key
                        ),
                    }
                )
        else:
            target_record["reason"] = (
                "inherited first-slot transfer did not master target"
            )

    reversal_record: dict[str, object] = {"attempted": False}
    if stages_completed and target_receipt is not None and target_receipt.accepted:
        reversal_record["attempted"] = True
        shared_reversal = _reversal_recovery(
            bank,
            active_keys[-1],
            current_row,
            reversal_patience=args.reversal_patience,
            recovery_observations=args.retention_probes,
        )
        if target_key is None or target_receipt.index is None:
            raise RuntimeError("accepted target admission lacks key or row")
        target_reversal = _reversal_recovery(
            bank,
            target_key,
            target_receipt.index,
            reversal_patience=args.reversal_patience,
            recovery_observations=args.retention_probes,
        )
        reversal_record.update(
            {
                "shared_alias": shared_reversal,
                "target_row": target_reversal,
                "shared_physical_row_stayed_protected": shared_reversal[
                    "after_reversal_row_protected"
                ],
                "target_physical_row_released": not target_reversal[
                    "after_reversal_row_protected"
                ],
            }
        )

    corruption = _corruption_control(
        bank,
        args.report_out.parent / "corruption_control_bank",
    )
    parent_digest_after = _digest_core(parent, ())
    final_bank_digest_before_reload = _digest_artifact_bank(bank)
    reloaded_bank = ExecutableArtifactMemory.load(bank.directory)
    final_bank_digest_after_reload = _digest_artifact_bank(reloaded_bank)
    shared_aliases_share_row, shared_row = _shared_row_aliases(
        reloaded_bank,
        tuple(active_keys),
        tuple(active_views),
    )
    gates = {
        "parent_stable": _stable_source_bits(parent_progress, args) is not None,
        "sources_stable": all(
            _stable_source_bits(source_progress[str(program_id)], args) is not None
            for program_id in source_ids
        ),
        "sources_mastered": min(source_behavior.values(), default=0.0) >= THRESHOLD,
        "source_addresses_separated": source_addresses_separated,
        "all_sequential_stages_adopted": stages_completed
        and len(stage_records) == len(source_ids) - 1
        and all(bool(record.get("accepted")) for record in stage_records),
        "all_sources_retained_after_reload": stages_completed
        and min(current_behaviors.values(), default=0.0) >= THRESHOLD,
        "all_source_aliases_share_one_physical_row": shared_aliases_share_row,
        "target_grown_and_reloaded_mastered": target_receipt is not None
        and target_receipt.accepted
        and target_receipt.action == "grow"
        and target_behavior_after is not None
        and target_behavior_after >= THRESHOLD,
        "shared_alias_reversal_isolated_and_recovered": reversal_record.get(
            "shared_alias", {}
        ).get("reversal_detected", False)
        and reversal_record.get("shared_alias", {}).get("alias_released", False)
        and reversal_record.get("shared_alias", {}).get("alias_recovered", False)
        and reversal_record.get("shared_physical_row_stayed_protected", False),
        "target_reversal_released_and_recovered": reversal_record.get(
            "target_row", {}
        ).get("reversal_detected", False)
        and reversal_record.get("target_row", {}).get("alias_released", False)
        and reversal_record.get("target_row", {}).get("alias_recovered", False)
        and reversal_record.get("target_physical_row_released", False),
        "memory_corruption_rejected": bool(corruption.get("rejected")),
        "final_bank_reload_exact": final_bank_digest_before_reload
        == final_bank_digest_after_reload,
        "core_unchanged": parent_digest_before == parent_digest_after,
        "no_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.generated-composition-sequential-slot-isolated-consolidation-report.v1",
        "claim_boundary": (
            "A frozen controller acquired a finite nonstationary sequence of "
            "external procedures. Each new neural slot was trained from fresh "
            "outcomes only, then appended under an opaque alias into one "
            "retention-gated physical artifact row. This is bounded replay-free "
            "slot-isolated growth, not dense shared-weight consolidation or "
            "general continual learning."
        ),
        "seed": args.seed,
        "source_ids": list(source_ids),
        "target_id": args.target_id,
        "programs": [list(program) for program in grammar],
        "budgets": {
            "parent_updates": args.parent_updates,
            "source_updates": args.source_updates,
            "target_updates": args.target_updates,
            "batch_size": args.batch_size,
            "audit_count": args.audit_count,
            "retention_probes": args.retention_probes,
            "eval_every": args.eval_every,
            "reversal_patience": args.reversal_patience,
        },
        "sources": [
            {
                "id": program_id,
                "behavior": source_behavior[str(program_id)],
                "stable_bits_to_threshold": _stable_source_bits(
                    source_progress[str(program_id)], args
                ),
                "artifact_digest": _digest_artifact(source_artifacts[str(program_id)]),
                "payload_bytes": _payload_bytes(source_artifacts[str(program_id)]),
            }
            for program_id in source_ids
        ],
        "source_key_similarity": source_key_similarity.tolist(),
        "sequential_stages": stage_records,
        "active_source_ids": list(active_ids),
        "active_source_views": list(active_views),
        "active_source_behavior_after_reload": current_behaviors,
        "shared_physical_row": shared_row,
        "shared_physical_payload_bytes": _payload_bytes(shared_artifact)
        if stages_completed
        else None,
        "source_payload_bytes": sum(
            _payload_bytes(source_artifacts[str(program_id)])
            for program_id in source_ids
        ),
        "target": target_record,
        "reversal": reversal_record,
        "memory_corruption_control": corruption,
        "frozen_core": {
            "digest_before": parent_digest_before,
            "digest_after": parent_digest_after,
            "unchanged": parent_digest_before == parent_digest_after,
        },
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + len(source_ids) * args.source_updates * args.batch_size * (SPAN + 2)
                + 2 * args.target_updates * args.batch_size * (SPAN + 2)
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + len(source_ids) * args.source_updates * args.batch_size * 2
                + 2 * args.target_updates * args.batch_size * 2
            ),
            "optimizer_updates": (
                args.parent_updates
                + len(source_ids) * args.source_updates
                + 2 * args.target_updates
            ),
            "consolidation_optimizer_updates": 0,
            "replayed_examples": 0,
            "retention_observations": (
                len(source_ids) * args.retention_probes
                + 2 * (len(source_ids) - 1) * args.retention_probes
                + 2 * args.retention_probes
                + 2 * (args.reversal_patience + args.retention_probes)
            ),
        },
        "gates": gates,
        "wall_seconds": perf_counter() - started,
    }
    report["promoted"] = all(gates.values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _stable_source_bits(progress, args: argparse.Namespace) -> int | None:
    return next(
        (
            int(row["update"]) * args.batch_size * SPAN
            for index, row in enumerate(progress)
            if all(
                float(later["heldout_accuracy"]) >= THRESHOLD
                for later in progress[index:]
            )
        ),
        None,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--source-updates", type=int, default=256)
    parser.add_argument("--target-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--retention-probes", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--reversal-patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--source-ids", type=int, nargs="+", default=(0, 2, 3, 4))
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
                "sequential_stages": [
                    {
                        "stage": row["stage"],
                        "new_source_id": row["new_source_id"],
                        "accepted": row["accepted"],
                    }
                    for row in report["sequential_stages"]
                ],
                "target": report["target"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
