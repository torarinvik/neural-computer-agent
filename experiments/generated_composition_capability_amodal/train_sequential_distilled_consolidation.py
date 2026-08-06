"""Audit replay-free sequential neural consolidation under nonstationarity.

Each source procedure is first acquired as an independent external artifact.
The bank then grows one source at a time.  At every stage, a single shared
routed artifact is fine-tuned only on the newly arrived procedure and is
adopted only after fresh held-out probes retain every earlier alias.  The old
source procedures are never replayed during those updates.

The audit also exercises memory-side reversal state, target growth, and
checksum corruption rejection.  It is deliberately a pressure test, not a
claim of unrestricted memory growth or general continual learning: the
external blueprint and verifier-private grammar remain finite.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
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
    _load_stack_artifact,
    _stack_artifact,
)
from experiments.generated_composition_capability_amodal.train_distilled_consolidation import (
    DEFAULT_RUNTIME_GRAMMAR,
    _digest_artifact,
    _parse_program_specs,
    _payload_bytes,
    _probe_bank_aliases,
    _source_behavior,
    _train_program,
)
from experiments.generated_composition_capability_amodal.train_pipeline import (
    _new_stack,
)
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    _rollout_capability,
    _stable_bits,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
    _runtime,
)
from neural_computer import (
    ExecutableArtifactMemory,
    ExternalCapabilityLifecycle,
)

SPAN = 4
THRESHOLD = 0.75
DEFAULT_SEQUENTIAL_GRAMMAR = (
    *DEFAULT_RUNTIME_GRAMMAR,
    ("complement", "prefix_parity", "reverse", "global_parity"),
    ("rotate", "global_parity", "complement", "adjacent_xor"),
)


def _expand_routed_stack(
    stack: torch.nn.Module,
    *,
    seed: int,
) -> torch.nn.Module:
    """Add one isolated routed slot while preserving every old slot exactly."""

    old_count = len(stack.programs)
    expanded = _new_stack(seed, program_count=old_count + 1, stack="routed")
    if expanded.composition_steps != stack.composition_steps:
        raise ValueError("expanded stack changed composition step count")
    with torch.no_grad():
        for old_program, new_program in zip(
            stack.programs,
            expanded.programs[:old_count],
            strict=True,
        ):
            new_program.load_state_dict(old_program.state_dict(), strict=True)
        expanded.router[0].load_state_dict(stack.router[0].state_dict(), strict=True)
        old_router = stack.router[2]
        new_router = expanded.router[2]
        old_width = old_count
        new_width = old_count + 1
        for step in range(stack.composition_steps):
            old_slice = slice(step * old_width, (step + 1) * old_width)
            new_slice = slice(step * new_width, step * new_width + old_width)
            new_router.weight[new_slice].copy_(old_router.weight[old_slice])
            new_router.bias[new_slice].copy_(old_router.bias[old_slice])
            new_router.weight[step * new_width + old_count].zero_()
            new_router.bias[step * new_width + old_count].fill_(-8.0)
    expanded.eval()
    return expanded


def _train_expanded_new_only(
    parent,
    artifact: dict[str, torch.Tensor],
    new_id: int,
    grammar,
    *,
    updates: int,
    batch_size: int,
    audit_count: int,
    eval_every: int,
    seed: int,
    learning_rate: float,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float | int]]]:
    """Train only a newly added slot; old slots, routing, and decoder are locked."""

    stack, decoder = _load_stack_artifact(artifact)
    old_count = len(stack.programs)
    stack = _expand_routed_stack(stack, seed=seed + 1_000)
    locked: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, parameter in stack.named_parameters():
        mask = torch.zeros_like(parameter, dtype=torch.bool)
        if name.startswith("programs."):
            slot = int(name.split(".")[1])
            if slot < old_count:
                mask.fill_(True)
        elif name.startswith("router.0."):
            mask.fill_(True)
        elif name.startswith("router.2."):
            if name.endswith("weight"):
                for step in range(stack.composition_steps):
                    mask[
                        step * (old_count + 1) : step * (old_count + 1) + old_count
                    ] = True
            else:
                for step in range(stack.composition_steps):
                    mask[
                        step * (old_count + 1) : step * (old_count + 1) + old_count
                    ] = True
        else:
            mask.fill_(True)
        locked[name] = (parameter.detach().clone(), mask)
        parameter.requires_grad_(not bool(mask.all()))
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    trainable = [
        parameter for parameter in stack.parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("expanded stack has no trainable new-slot parameters")
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    progress: list[dict[str, float | int]] = []
    torch.manual_seed(seed + 2_000)
    stack.train()
    decoder.eval()
    for update in range(1, updates + 1):
        target = generate_sequence_memory_batch(
            batch_size,
            span=SPAN,
            distractors=1,
            seed=seed + update * 10_007,
            operation="generated_composition",
            generated_composition_ids=(new_id,),
            generated_compositions=grammar,
        )
        target_result = _rollout_capability(parent, stack, decoder, target, train=True)
        auxiliary = generate_sequence_memory_batch(
            batch_size,
            span=2,
            distractors=1,
            seed=seed + 5_000_003 + update * 20_021,
            operation="forward",
        )
        auxiliary_result = _rollout_capability(
            parent, stack, decoder, auxiliary, train=True
        )
        loss = target_result["loss"] + auxiliary_result["loss"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        with torch.no_grad():
            for name, parameter in stack.named_parameters():
                values, mask = locked[name]
                parameter.data.copy_(torch.where(mask, values, parameter.data))
        history.append(
            {
                "update": update,
                "unique_logical_lifetimes": update * batch_size * 2,
                "unique_verifier_bits": update * batch_size * (SPAN + 2),
                "training_accuracy": float(target_result["rewards"].mean()),
                "loss": float(loss.detach()),
            }
        )
        if update == updates or (eval_every > 0 and update % eval_every == 0):
            stack.eval()
            heldout = generate_sequence_memory_batch(
                audit_count,
                span=SPAN,
                distractors=1,
                seed=seed + 1_000_000 + update,
                operation="generated_composition",
                generated_composition_ids=(new_id,),
                generated_compositions=grammar,
            )
            progress.append(
                {
                    "update": update,
                    "unique_verifier_bits": update * batch_size * SPAN,
                    "heldout_accuracy": float(
                        _rollout_capability(
                            parent, stack, decoder, heldout, train=False
                        )["rewards"].mean()
                    ),
                }
            )
            stack.train()
    stack.eval()
    return _stack_artifact(stack, decoder), progress


def _observe_mastery(
    bank: ExecutableArtifactMemory,
    parent,
    artifact: dict[str, torch.Tensor],
    key: torch.Tensor,
    program_id: int,
    grammar,
    *,
    count: int,
    observations: int,
    seed: int,
) -> list[float]:
    outcomes = [
        _source_behavior(
            parent,
            artifact,
            program_id,
            grammar,
            count=count,
            seed=seed + probe * 101,
        )
        for probe in range(observations)
    ]
    for outcome in outcomes:
        bank.observe_retention(key, outcome)
    return outcomes


def _behaviors(
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
    budgets = (
        args.parent_updates,
        args.source_updates,
        args.consolidation_updates,
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

    bank_path = args.report_out.parent / "sequential_bank"
    if bank_path.exists():
        shutil.rmtree(bank_path)
    bank = _configure_bank(
        bank_path,
        reversal_patience=args.reversal_patience,
        observations=args.retention_probes,
    )
    lifecycle = ExternalCapabilityLifecycle(bank)
    first_key = source_keys[0]
    first_id = source_ids[0]
    first_receipt = lifecycle.admit(first_key, source_artifacts[str(first_id)])
    if not first_receipt.accepted or first_receipt.index is None:
        raise RuntimeError(f"initial source admission failed: {first_receipt.reason}")
    _observe_mastery(
        lifecycle.memory,
        parent,
        source_artifacts[str(first_id)],
        first_key,
        first_id,
        grammar,
        count=args.audit_count,
        observations=args.retention_probes,
        seed=args.seed + 80_000,
    )
    bank = lifecycle.memory
    active_ids = [first_id]
    active_keys = [first_key]
    shared_artifact = source_artifacts[str(first_id)]
    current_row = first_receipt.index
    current_behaviors = {str(first_id): source_behavior[str(first_id)]}
    stage_records: list[dict[str, object]] = []

    for stage_index, new_id in enumerate(source_ids[1:], start=1):
        new_key = source_keys[stage_index]
        new_artifact = source_artifacts[str(new_id)]
        plan = lifecycle.plan_admission(new_key, new_artifact)
        new_receipt = lifecycle.admit(
            new_key,
            new_artifact,
            plan=plan,
            grow_destination=args.report_out.parent / f"stage_{stage_index}_grown_bank",
        )
        if not new_receipt.accepted or new_receipt.index is None:
            stage_records.append(
                {
                    "stage": stage_index,
                    "new_source_id": new_id,
                    "accepted": False,
                    "admission": new_receipt.__dict__,
                    "reason": "new source could not be staged",
                }
            )
            break
        bank = lifecycle.memory
        _observe_mastery(
            bank,
            parent,
            new_artifact,
            new_key,
            new_id,
            grammar,
            count=args.audit_count,
            observations=args.retention_probes,
            seed=args.seed + 81_000 + stage_index * 101,
        )
        candidate, candidate_progress = _train_expanded_new_only(
            parent,
            shared_artifact,
            new_id,
            grammar,
            updates=args.consolidation_updates,
            batch_size=args.batch_size,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            seed=args.seed + 90_000 + stage_index * 10_003,
            learning_rate=args.learning_rate,
        )
        fresh, fresh_progress = _train_program(
            parent,
            None,
            (new_id,),
            grammar,
            updates=args.consolidation_updates,
            batch_size=args.batch_size,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            seed=args.seed + 90_000 + stage_index * 10_003,
            learning_rate=args.learning_rate,
        )
        candidate_ids = (*active_ids, new_id)
        candidate_keys = (*active_keys, new_key)
        candidate_behavior = _behaviors(
            parent,
            candidate,
            candidate_ids,
            grammar,
            count=args.audit_count,
            seed=args.seed + 100_000 + stage_index * 10_003,
        )
        fresh_behavior = _source_behavior(
            parent,
            fresh,
            new_id,
            grammar,
            count=args.audit_count,
            seed=args.seed + 100_001 + stage_index * 10_003,
        )
        candidate_stable = _stable_bits(
            candidate_progress,
            threshold=THRESHOLD,
            bits_per_update=args.batch_size * SPAN,
        )
        fresh_stable = _stable_bits(
            fresh_progress,
            threshold=THRESHOLD,
            bits_per_update=args.batch_size * SPAN,
        )
        candidate_ready = (
            candidate_stable is not None
            and min(candidate_behavior.values(), default=0.0) >= THRESHOLD
        )
        record: dict[str, object] = {
            "stage": stage_index,
            "new_source_id": new_id,
            "active_source_ids_before": list(active_ids),
            "admission": new_receipt.__dict__,
            "candidate_behavior_before_rewrite": candidate_behavior,
            "fresh_new_behavior": fresh_behavior,
            "candidate_stable_bits_to_threshold": candidate_stable,
            "fresh_stable_bits_to_threshold": fresh_stable,
            "candidate_progress": candidate_progress,
            "fresh_progress": fresh_progress,
            "candidate_trained_on_new_source_only": True,
        }
        if not candidate_ready:
            record["accepted"] = False
            record["reason"] = (
                "candidate did not master the new source while retaining old aliases"
            )
            stage_records.append(record)
            break

        retained_scores = [
            current_behaviors[str(program_id)] for program_id in active_ids
        ] + [source_behavior[str(new_id)]]
        replacement_key = F.normalize(torch.stack(candidate_keys).mean(dim=0), dim=0)
        consolidation_path = (
            args.report_out.parent / f"stage_{stage_index}_consolidated_bank"
        )
        if consolidation_path.exists():
            shutil.rmtree(consolidation_path)
        capture: dict[str, object] = {}

        def candidate_probe(
            candidate_bank: ExecutableArtifactMemory,
            *,
            probe_keys=candidate_keys,
            probe_ids=candidate_ids,
            probe_stage=stage_index,
            probe_capture=capture,
        ):
            probes = _probe_bank_aliases(
                parent,
                candidate_bank,
                probe_keys,
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
            probe_ids=candidate_ids,
            probe_stage=stage_index,
            probe_capture=capture,
        ) -> bool:
            probes = _probe_bank_aliases(
                parent,
                candidate_bank,
                probe_keys,
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
            candidate,
            consolidation_path,
            verifier=verifier,
            replacement_aliases=candidate_keys,
            candidate_outcome_probe=candidate_probe,
            retained_scores=retained_scores,
            candidate_threshold=THRESHOLD,
            retention_floor=THRESHOLD,
            min_candidate_observations=args.retention_probes,
        )
        record["consolidation"] = receipt.__dict__
        record.update(capture)
        record["accepted"] = receipt.accepted
        if not receipt.accepted:
            record["reason"] = receipt.reason
            stage_records.append(record)
            break
        bank = lifecycle.memory
        active_ids.append(new_id)
        active_keys.append(new_key)
        current_handle, shared_artifact = bank.promote(new_key)
        current_row = current_handle.index
        current_behaviors = _behaviors(
            parent,
            shared_artifact,
            tuple(active_ids),
            grammar,
            count=args.audit_count,
            seed=args.seed + 112_000 + stage_index * 10_003,
        )
        record["active_source_ids_after"] = list(active_ids)
        record["behavior_after_reload"] = current_behaviors
        record["shared_alias_digest"] = _digest_artifact(shared_artifact)
        stage_records.append(record)

    stages_completed = len(active_ids) == len(source_ids)
    target_record: dict[str, object] = {"attempted": False}
    target_key = None
    target_receipt = None
    target_behavior_after = None
    if stages_completed:
        target_record["attempted"] = True
        target_artifact, target_progress = _train_program(
            parent,
            shared_artifact,
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
        target_stable = _stable_bits(
            target_progress,
            threshold=THRESHOLD,
            bits_per_update=args.batch_size * SPAN,
        )
        fresh_target_stable = _stable_bits(
            fresh_target_progress,
            threshold=THRESHOLD,
            bits_per_update=args.batch_size * SPAN,
        )
        target_record.update(
            {
                "inherited_behavior": target_behavior,
                "fresh_behavior": fresh_target_behavior,
                "inherited_stable_bits_to_threshold": target_stable,
                "fresh_stable_bits_to_threshold": fresh_target_stable,
                "inherited_progress": target_progress,
                "fresh_progress": fresh_target_progress,
                "trained_on_target_only": True,
            }
        )
        if target_stable is not None and target_behavior >= THRESHOLD:
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
            plan = lifecycle.plan_admission(target_key, target_artifact)
            target_receipt = lifecycle.admit(
                target_key,
                target_artifact,
                plan=plan,
                grow_destination=args.report_out.parent / "target_grown_bank",
            )
            bank = lifecycle.memory
            target_record["admission"] = target_receipt.__dict__
            if target_receipt.accepted:
                target_row = target_receipt.index
                if target_row is None:
                    raise RuntimeError("target admission omitted its row index")
                reloaded_handle, reloaded_target = bank.promote(target_key)
                target_behavior_after = _source_behavior(
                    parent,
                    reloaded_target,
                    args.target_id,
                    grammar,
                    count=args.audit_count,
                    seed=args.seed + 151_000,
                )
                _observe_mastery(
                    bank,
                    parent,
                    reloaded_target,
                    target_key,
                    args.target_id,
                    grammar,
                    count=args.audit_count,
                    observations=args.retention_probes,
                    seed=args.seed + 152_000,
                )
                target_record.update(
                    {
                        "row": reloaded_handle.index,
                        "behavior_after_reload": target_behavior_after,
                        "protected_after_reload": bank.retention.is_protected(
                            target_key
                        ),
                    }
                )
        else:
            target_record["reason"] = (
                "inherited shared artifact did not master the new target"
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
        target_row = target_receipt.index
        if target_row is None or target_key is None:
            raise RuntimeError("accepted target admission lacks key or row")
        target_reversal = _reversal_recovery(
            bank,
            target_key,
            target_row,
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
    final_alias_digests = {
        str(program_id): _digest_artifact(reloaded_bank.promote(key)[1])
        for program_id, key in zip(active_ids, active_keys, strict=True)
    }
    gates = {
        "parent_stable": _stable_bits(
            parent_progress,
            threshold=THRESHOLD,
            bits_per_update=args.batch_size * 2,
        )
        is not None,
        "sources_stable": all(
            _stable_bits(
                source_progress[str(program_id)],
                threshold=THRESHOLD,
                bits_per_update=args.batch_size * SPAN,
            )
            is not None
            for program_id in source_ids
        ),
        "sources_mastered": min(source_behavior.values(), default=0.0) >= THRESHOLD,
        "source_addresses_separated": source_addresses_separated,
        "all_sequential_stages_adopted": stages_completed
        and len(stage_records) == len(source_ids) - 1
        and all(bool(record.get("accepted")) for record in stage_records),
        "all_sources_retained_after_reload": stages_completed
        and min(current_behaviors.values(), default=0.0) >= THRESHOLD,
        "shared_aliases_resolve_to_one_artifact": stages_completed
        and len(set(final_alias_digests.values())) == 1,
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
        "schema": "neural-computer.generated-composition-sequential-distilled-consolidation-report.v1",
        "claim_boundary": (
            "A frozen controller acquired a finite sequence of verifier-private "
            "external procedures. Each shared neural rewrite trained only on "
            "the newly arrived source and was retention-gated against earlier "
            "aliases. This is bounded replay-free sequential consolidation, "
            "not general continual learning."
        ),
        "seed": args.seed,
        "source_ids": list(source_ids),
        "target_id": args.target_id,
        "programs": [list(program) for program in grammar],
        "budgets": {
            "parent_updates": args.parent_updates,
            "source_updates": args.source_updates,
            "consolidation_updates": args.consolidation_updates,
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
                "stable_bits_to_threshold": _stable_bits(
                    source_progress[str(program_id)],
                    threshold=THRESHOLD,
                    bits_per_update=args.batch_size * SPAN,
                ),
                "artifact_digest": _digest_artifact(source_artifacts[str(program_id)]),
                "payload_bytes": _payload_bytes(source_artifacts[str(program_id)]),
            }
            for program_id in source_ids
        ],
        "source_key_similarity": source_key_similarity.tolist(),
        "source_bank": {
            "initial_capacity": 1,
            "final_capacity": bank.capacity,
            "final_rows": list(bank.occupied),
            "final_protection_mask": bank.protection_mask().tolist(),
        },
        "sequential_stages": stage_records,
        "active_source_ids": list(active_ids),
        "active_source_behavior_after_reload": current_behaviors,
        "active_alias_digests_after_reload": final_alias_digests,
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
                + 2
                * (len(source_ids) - 1)
                * args.consolidation_updates
                * args.batch_size
                * (SPAN + 2)
                + 2 * args.target_updates * args.batch_size * (SPAN + 2)
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + len(source_ids) * args.source_updates * args.batch_size * 2
                + 2
                * (len(source_ids) - 1)
                * args.consolidation_updates
                * args.batch_size
                * 2
                + 2 * args.target_updates * args.batch_size * 2
            ),
            "optimizer_updates": (
                args.parent_updates
                + len(source_ids) * args.source_updates
                + 2 * (len(source_ids) - 1) * args.consolidation_updates
                + 2 * args.target_updates
            ),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--source-updates", type=int, default=256)
    parser.add_argument("--consolidation-updates", type=int, default=512)
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
