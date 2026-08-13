"""Grow an external compute archive behind a bounded hot cache.

This is the next lifecycle pressure test after indexed external-file route
reversal.  Four independently learned opaque compute files are acquired while
the controller, event frontend, and shared interpreter remain fixed.  Only two
files may be resident in the executable hot cache; the remaining files live as
portable external artifacts in a cold archive.  Fresh learned event keys find
archive records, and a memory-side reliability/recency policy selects an
unprotected resident for verifier-gated reactivation.

The archive stores only normalized learned keys, opaque artifact handles, and
deterministic scalar verifier telemetry.  It does not receive task names,
family IDs, target actions, or semantic labels.  This qualifies the missing
external-compute lifecycle seam, not unrestricted storage, learned
compression, or general continual learning.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import torch

from neural_computer import (
    EpisodicBindingArtifactIndex,
    ExternalRegisterComputeBasisArtifact,
)

from .external_compute_growth import (
    EVENT_WIDTH,
    ComputeGrowthSystem,
    _build,
    _common_modules,
    _digest,
    _evaluate,
    _set_requires_grad,
    _slot_modules,
    _train_stage,
)
from .external_compute_open_growth import (
    _append_compute_file,
    _discard_newest_compute_file,
)
from .external_compute_route_bank import _all_modules, _family_steps


SCHEMA = "neural-computer.brainworkshop-external-compute-artifact-cache-pressure.v1"
SOURCE_FAMILY = "symbol_parity"
SCHEDULE = (
    ("symbol_parity", 7),
    ("triplet_parity", 8),
    ("parity2", 10),
    ("switch_binary", 11),
)
UNKNOWN_CUE = 6
ACTIVE_CACHE_SLOTS = 2
MASTERY_THRESHOLD = 0.80
MATCHING_THRESHOLD = 0.999
SOURCE_PROTECTION_OBSERVATIONS = 16


@dataclass(frozen=True)
class _FileSnapshot:
    """Portable state for one opaque executable file."""

    basis: ExternalRegisterComputeBasisArtifact
    instruction: Mapping[str, torch.Tensor]
    readout: Mapping[str, torch.Tensor]
    decoder: Mapping[str, torch.Tensor]
    digest: str


def _copy_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


def _snapshot(system: ComputeGrowthSystem, slot: int) -> _FileSnapshot:
    modules = _slot_modules(system, slot)
    return _FileSnapshot(
        basis=system.machine.basis_artifact(slot),
        instruction=_copy_state(system.instructions[slot]),
        readout=_copy_state(system.readouts[slot]),
        decoder=_copy_state(system.decoders[slot]),
        digest=_digest(*modules),
    )


def _restore_snapshot(
    system: ComputeGrowthSystem,
    slot: int,
    snapshot: _FileSnapshot,
) -> None:
    """Load one validated cold artifact into an executable physical slot."""

    basis = system.machine.basis_slots[slot]
    if dict(basis.configuration()) != dict(snapshot.basis.configuration):
        raise ValueError("external compute artifact ABI does not match the cache")
    ExternalRegisterComputeBasisArtifact.from_payload(snapshot.basis.payload())
    basis.load_state_dict(snapshot.basis.state, strict=True)
    system.instructions[slot].load_state_dict(snapshot.instruction, strict=True)
    system.readouts[slot].load_state_dict(snapshot.readout, strict=True)
    system.decoders[slot].load_state_dict(snapshot.decoder, strict=True)
    if _digest(*_slot_modules(system, slot)) != snapshot.digest:
        raise RuntimeError("external compute artifact failed round-trip integrity")


def _route_key(system: ComputeGrowthSystem, cue_symbol: int) -> torch.Tensor:
    with torch.no_grad():
        collection = system.agent.runtime.encode_streams(
            {"stimulus": torch.tensor([cue_symbol], dtype=torch.long)}
        )
    return collection.payload[0, 0].detach().cpu()


def _stable(rows: list[dict[str, float | int]]) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= MASTERY_THRESHOLD


def _mean(rows: list[dict[str, float | int]]) -> float:
    if not rows:
        raise ValueError("cannot average an empty evaluation")
    return sum(float(row["accuracy"]) for row in rows) / len(rows)


def _choose_victim(
    index: EpisodicBindingArtifactIndex,
    *,
    step: int,
) -> tuple[int, int] | None:
    """Choose an unprotected resident using only generic telemetry.

    The score is deliberately non-semantic: older and less reliable residents
    are more disposable.  Protection is a verifier-owned stable-prefix latch;
    a protected file is never selected.
    """

    candidates: list[tuple[float, float, int, int]] = []
    for slot, binding_id in enumerate(index.active_binding_ids):
        if binding_id is None or index.archive.is_protected(binding_id):
            continue
        reliability, age = index.archive.telemetry(
            binding_id,
            step=step,
            age_horizon=max(1, step + 1),
        )
        candidates.append((age, 1.0 - reliability, -binding_id, slot))
    if not candidates:
        return None
    _age, _risk, _binding_order, slot = max(candidates)
    binding_id = index.archive.active_binding(slot)
    if binding_id is None:
        raise RuntimeError("victim selection returned an empty cache slot")
    return slot, binding_id


def _active_digests(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
) -> bool:
    for slot, binding_id in enumerate(index.active_binding_ids):
        if binding_id is None:
            continue
        handle = index.artifact_handle(binding_id)
        if handle is None or handle not in snapshots:
            return False
        if _digest(*_slot_modules(system, slot)) != snapshots[handle].digest:
            return False
    return True


def _train_file(
    system: ComputeGrowthSystem,
    *,
    slot: int,
    family: str,
    cue_symbol: int,
    updates: int,
    batch_size: int,
    seed: int,
    learning_rate: float,
    first_file: bool,
    shuffle_outcomes: bool = False,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    _set_requires_grad(_all_modules(system), False)
    train_modules = _slot_modules(system, slot)
    if first_file:
        train_modules = _common_modules(system) + train_modules
    _set_requires_grad(train_modules, True)
    steps = _family_steps(family)
    history = _train_stage(
        system,
        family=family,
        slot=slot,
        cue_symbol=cue_symbol,
        updates=updates,
        batch_size=batch_size,
        steps=steps,
        seed=seed,
        learning_rate=learning_rate,
        entropy_weight=0.01,
        credit_mode="attempted_bce",
        shuffle_outcomes=shuffle_outcomes,
    )
    _set_requires_grad(_all_modules(system), False)
    fresh = _evaluate(
        system,
        family=family,
        slot=slot,
        cue_symbol=cue_symbol,
        lifetimes=2,
        batch_size=batch_size,
        steps=steps,
        seed=seed + 50_000,
    )
    return history, fresh


def _install_candidate(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    *,
    binding_id: int,
    candidate_snapshot: _FileSnapshot,
    family: str,
    cue_symbol: int,
    candidate_slot: int,
    candidate_is_scratch: bool,
    batch_size: int,
    retention_lifetimes: int,
    step: int,
) -> dict[str, object]:
    """Commit an admitted file to hot cache or leave it safely cold."""

    free_slot = next(
        (slot for slot, resident in enumerate(index.active_binding_ids) if resident is None),
        None,
    )
    victim = None if free_slot is not None else _choose_victim(index, step=step)
    destination = free_slot if free_slot is not None else (
        None if victim is None else victim[0]
    )
    if destination is None:
        if candidate_is_scratch:
            _discard_newest_compute_file(system)
        return {
            "accepted": False,
            "resident": False,
            "reason": "all active files are protected",
            "victim_slot": None,
        }

    displaced_binding = index.archive.active_binding(destination)
    displaced_snapshot = None
    if displaced_binding is not None:
        displaced_handle = index.artifact_handle(displaced_binding)
        if displaced_handle is None or displaced_handle not in snapshots:
            raise RuntimeError("active cache resident has no cold artifact")
        displaced_snapshot = snapshots[displaced_handle]
    if destination != candidate_slot:
        _restore_snapshot(system, destination, candidate_snapshot)

    probe: list[dict[str, float | int]] = []

    def retention_probe(_candidate: EpisodicBindingArtifactIndex) -> bool:
        probe.extend(
            _evaluate(
                system,
                family=family,
                slot=destination,
                cue_symbol=cue_symbol,
                lifetimes=retention_lifetimes,
                batch_size=batch_size,
                steps=_family_steps(family),
                seed=step + 700_000,
            )
        )
        return _stable(probe)

    if free_slot is not None:
        index.activate(binding_id, destination)
        accepted = True
        receipt: dict[str, object] = {
            "accepted": True,
            "resident": True,
            "reason": "free hot-cache slot",
            "victim_slot": None,
            "retention_probe": None,
        }
    else:
        activation = index.reactivate_verified(
            binding_id,
            destination,
            retention_probe,
        )
        accepted = activation.accepted
        receipt = {
            "accepted": activation.accepted,
            "resident": activation.accepted,
            "reason": activation.reason,
            "victim_slot": destination,
            "displaced_binding_id": displaced_binding,
            "retention_probe": probe,
        }
        if not accepted and displaced_snapshot is not None:
            _restore_snapshot(system, destination, displaced_snapshot)

    if candidate_is_scratch:
        _discard_newest_compute_file(system)
    if not accepted:
        return receipt
    if probe:
        index.archive.observe(
            binding_id,
            min(float(row["accuracy"]) for row in probe),
            step=step,
        )
    return receipt


def _route_probe(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    specs: Mapping[int, tuple[str, int]],
    *,
    route_id: int,
    batch_size: int,
    retention_lifetimes: int,
    step: int,
) -> dict[str, object]:
    family, cue_symbol = specs[route_id]
    key = _route_key(system, cue_symbol)
    lookup = index.lookup(key)
    if lookup.binding_id is None:
        return {
            "route_id": route_id,
            "cue": cue_symbol,
            "lookup": False,
            "reactivated": False,
            "accuracy": None,
            "selected_slot": None,
            "replayed_examples": 0,
        }
    binding_id = lookup.binding_id
    slot = lookup.active_slot
    reactivated = False
    activation_receipt: dict[str, object] | None = None
    if slot is None:
        handle = lookup.artifact_handle
        if handle is None or handle not in snapshots:
            raise RuntimeError("cold route has no executable artifact")
        candidate = snapshots[handle]
        victim = _choose_victim(index, step=step)
        if victim is None:
            return {
                "route_id": route_id,
                "cue": cue_symbol,
                "lookup": True,
                "reactivated": False,
                "accuracy": None,
                "selected_slot": None,
                "reactivation_blocked": True,
                "replayed_examples": 0,
            }
        destination, displaced_binding = victim
        displaced_snapshot = None
        if displaced_binding is not None:
            displaced_handle = index.artifact_handle(displaced_binding)
            if displaced_handle is None or displaced_handle not in snapshots:
                raise RuntimeError("victim has no executable cold artifact")
            displaced_snapshot = snapshots[displaced_handle]
        _restore_snapshot(system, destination, candidate)
        probe: list[dict[str, float | int]] = []

        def retention_probe(_candidate: EpisodicBindingArtifactIndex) -> bool:
            probe.extend(
                _evaluate(
                    system,
                    family=family,
                    slot=destination,
                    cue_symbol=cue_symbol,
                    lifetimes=retention_lifetimes,
                    batch_size=batch_size,
                    steps=_family_steps(family),
                    seed=step + 900_000,
                )
            )
            return _stable(probe)

        activation = index.reactivate_verified(
            binding_id,
            destination,
            retention_probe,
        )
        activation_receipt = {
            "accepted": activation.accepted,
            "reason": activation.reason,
            "victim_slot": destination,
            "displaced_binding_id": displaced_binding,
            "probe": probe,
        }
        if not activation.accepted:
            if displaced_snapshot is not None:
                _restore_snapshot(system, destination, displaced_snapshot)
            return {
                "route_id": route_id,
                "cue": cue_symbol,
                "lookup": True,
                "reactivated": False,
                "accuracy": None,
                "selected_slot": None,
                "activation": activation_receipt,
                "replayed_examples": 0,
            }
        slot = destination
        reactivated = True
        if probe:
            index.archive.observe(
                binding_id,
                min(float(row["accuracy"]) for row in probe),
                step=step,
            )

    rows = _evaluate(
        system,
        family=family,
        slot=slot,
        cue_symbol=cue_symbol,
        lifetimes=retention_lifetimes,
        batch_size=batch_size,
        steps=_family_steps(family),
        seed=step + 1_000_000,
    )
    outcome = min(float(row["accuracy"]) for row in rows)
    index.archive.observe(binding_id, outcome, step=step)
    return {
        "route_id": route_id,
        "cue": cue_symbol,
        "lookup": True,
        "reactivated": reactivated,
        "accuracy": outcome,
        "selected_slot": slot,
        "activation": activation_receipt,
        "unique_verifier_bits": sum(int(row["unique_verifier_bits"]) for row in rows),
        "replayed_examples": 0,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.target_file_count != len(SCHEDULE):
        raise ValueError("the calibrated cache-pressure schedule has four files")
    if min(
        args.file_updates,
        args.batch_size,
        args.retention_lifetimes,
        args.route_revisits,
    ) < 1:
        raise ValueError("cache-pressure budgets must be positive")
    if args.batch_size != 32:
        raise ValueError("the calibrated cache-pressure harness requires batch size 32")
    system = _build(args.seed, slot_count=ACTIVE_CACHE_SLOTS)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    index = EpisodicBindingArtifactIndex.create(
        EVENT_WIDTH,
        EVENT_WIDTH,
        active_slots=ACTIVE_CACHE_SLOTS,
        matching_threshold=MATCHING_THRESHOLD,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=SOURCE_PROTECTION_OBSERVATIONS,
        reversal_threshold=0.5,
        reversal_patience=4,
    )
    snapshots: dict[str, _FileSnapshot] = {}
    specs: dict[int, tuple[str, int]] = {
        route_id: spec for route_id, spec in enumerate(SCHEDULE)
    }
    allocation: list[dict[str, object]] = []
    direct: list[list[dict[str, float | int]]] = []
    histories: list[list[dict[str, float | int]]] = []
    source_binding_id: int | None = None
    source_digest: str | None = None
    source_probes: list[list[dict[str, float | int]]] = []
    total_training_bits = 0
    optimizer_updates = 0

    for route_id, (family, cue_symbol) in enumerate(SCHEDULE):
        active_free = next(
            (slot for slot, binding_id in enumerate(index.active_binding_ids) if binding_id is None),
            None,
        )
        candidate_is_scratch = active_free is None
        candidate_slot = (
            _append_compute_file(system, seed=args.seed + 80_000 + route_id)
            if candidate_is_scratch
            else int(active_free)
        )
        history, fresh = _train_file(
            system,
            slot=candidate_slot,
            family=family,
            cue_symbol=cue_symbol,
            updates=args.file_updates,
            batch_size=args.batch_size,
            seed=args.seed + 10_000 * (route_id + 1),
            learning_rate=args.learning_rate,
            first_file=route_id == 0,
        )
        histories.append(history)
        direct.append(fresh)
        total_training_bits += sum(int(row["unique_verifier_bits"]) for row in history)
        optimizer_updates += args.file_updates
        if not _stable(fresh):
            if candidate_is_scratch:
                _discard_newest_compute_file(system)
            allocation.append(
                {
                    "route_id": route_id,
                    "family": family,
                    "candidate_slot": candidate_slot,
                    "accepted": False,
                    "resident": False,
                    "fresh_probe": fresh,
                }
            )
            break
        candidate_snapshot = _snapshot(system, candidate_slot)
        handle = candidate_snapshot.digest
        snapshots[handle] = candidate_snapshot
        key = _route_key(system, cue_symbol)
        binding_id = index.register(key, key, handle)
        if route_id == 0:
            source_binding_id = binding_id
            source_digest = handle
        install = _install_candidate(
            system,
            index,
            snapshots,
            binding_id=binding_id,
            candidate_snapshot=candidate_snapshot,
            family=family,
            cue_symbol=cue_symbol,
            candidate_slot=candidate_slot,
            candidate_is_scratch=candidate_is_scratch,
            batch_size=args.batch_size,
            retention_lifetimes=args.retention_lifetimes,
            step=route_id + 1,
        )
        if not install["accepted"]:
            allocation.append(
                {
                    "route_id": route_id,
                    "family": family,
                    "candidate_slot": candidate_slot,
                    "artifact_handle": handle,
                    "accepted": True,
                    "resident": False,
                    "install": install,
                    "fresh_probe": fresh,
                }
            )
            break
        index.archive.observe(
            binding_id,
            min(float(row["accuracy"]) for row in fresh),
            step=route_id + 1,
        )
        if route_id == 0:
            # Protect the source before any later candidate can create cache
            # pressure.  These are fresh verifier lifetimes, not replayed
            # source examples.
            source_slot = index.archive.binding_slot(source_binding_id)
            if source_slot is None:
                raise RuntimeError("source file was not resident after admission")
            for observation in range(SOURCE_PROTECTION_OBSERVATIONS):
                rows = _evaluate(
                    system,
                    family=SOURCE_FAMILY,
                    slot=source_slot,
                    cue_symbol=SCHEDULE[0][1],
                    lifetimes=1,
                    batch_size=args.batch_size,
                    steps=_family_steps(SOURCE_FAMILY),
                    seed=args.seed + 300_000 + observation,
                )
                source_probes.append(rows)
                index.archive.observe(
                    source_binding_id,
                    min(float(row["accuracy"]) for row in rows),
                    step=100 + observation,
                )
        allocation.append(
            {
                "route_id": route_id,
                "family": family,
                "candidate_slot": candidate_slot,
                "artifact_handle": handle,
                "binding_id": binding_id,
                "accepted": True,
                "resident": True,
                "install": install,
                "fresh_probe": fresh,
            }
        )

    accepted_count = sum(1 for item in allocation if bool(item["resident"]))
    archive_count = index.record_count
    if source_binding_id is None or source_digest is None:
        rejected_report = {
            "schema": SCHEMA,
            "claim_boundary": (
                "Outcome-only bounded hot-cache lifecycle for independently "
                "learned external compute artifacts; source admission failed "
                "at this budget."
            ),
            "seed": args.seed,
            "architecture": {
                "active_cache_slots": ACTIVE_CACHE_SLOTS,
                "archive": "episodic_binding_artifact_index_v1",
                "allocation": "fresh_candidate_then_verifier_gated_cache_installation_v1",
            },
            "allocation": allocation,
            "direct": direct,
            "gates": {
                "source_candidate_mastered": False,
                "source_not_promoted": True,
                "zero_replayed_examples": True,
                "frozen_controller": controller_before
                == _digest(system.agent.controller),
                "frozen_event_encoder": encoder_before
                == _digest(system.agent.runtime.encoders["stimulus"]),
            },
            "accounting": {
                "unique_verifier_bits": total_training_bits,
                "optimizer_updates": optimizer_updates,
                "replayed_examples": 0,
                "stable_bits_to_threshold": None,
            },
            "status": "rejected",
        }
        if args.report_out is not None:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(json.dumps(rejected_report, indent=2) + "\n")
        return rejected_report

    # Protect only the source through a stable fresh prefix.  Later files stay
    # replaceable until they accumulate their own independent evidence.
    route_history: list[dict[str, object]] = []
    route_order = list(range(accepted_count))
    route_order += list(reversed(route_order))
    route_order += [route_id for route_id in (1, 2, 3) if route_id < accepted_count]
    route_order *= args.route_revisits
    for ordinal, route_id in enumerate(route_order):
        row = _route_probe(
            system,
            index,
            snapshots,
            specs,
            route_id=route_id,
            batch_size=args.batch_size,
            retention_lifetimes=args.retention_lifetimes,
            step=200 + ordinal,
        )
        row["ordinal"] = ordinal
        route_history.append(row)

    unknown_key_before = _route_key(system, UNKNOWN_CUE)
    unknown_lookup = index.lookup(unknown_key_before)
    unknown_active_before = index.active_binding_ids
    unknown_route = {
        "lookup": unknown_lookup.binding_id is not None,
        "binding_id": unknown_lookup.binding_id,
        "active_slot": unknown_lookup.active_slot,
        "active_before": unknown_active_before,
        "active_after": index.active_binding_ids,
    }

    index_payload = index.payload()
    restored = EpisodicBindingArtifactIndex.from_payload(index_payload)
    restored_payload = restored.payload()
    index_reload_exact = restored_payload == index_payload
    corrupted = dict(index_payload)
    corrupted["checksum"] = "tampered"
    corruption_rejected = False
    try:
        EpisodicBindingArtifactIndex.from_payload(corrupted)
    except ValueError as error:
        corruption_rejected = "checksum" in str(error).lower()
    artifact_corruption_rejected = False
    first_artifact = next(iter(snapshots.values())).basis.payload()
    first_artifact["sha256"] = "tampered"
    try:
        ExternalRegisterComputeBasisArtifact.from_payload(first_artifact)
    except ValueError as error:
        artifact_corruption_rejected = "checksum" in str(error).lower()

    control = _build(args.seed + 700_000, slot_count=1)
    _set_requires_grad(_all_modules(control), False)
    _set_requires_grad(_common_modules(control) + _slot_modules(control, 0), True)
    control_history = _train_stage(
        control,
        family="triplet_parity",
        slot=0,
        cue_symbol=8,
        updates=args.file_updates,
        batch_size=args.batch_size,
        steps=_family_steps("triplet_parity"),
        seed=args.seed + 710_000,
        learning_rate=args.learning_rate,
        entropy_weight=0.01,
        credit_mode="attempted_bce",
        shuffle_outcomes=True,
    )
    _set_requires_grad(_all_modules(control), False)
    shuffled_control = _evaluate(
        control,
        family="triplet_parity",
        slot=0,
        cue_symbol=8,
        lifetimes=args.retention_lifetimes,
        batch_size=args.batch_size,
        steps=_family_steps("triplet_parity"),
        seed=args.seed + 720_000,
    )

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    all_route_rows = [row for row in route_history if row.get("accuracy") is not None]
    route_accuracy_by_id: dict[str, list[float]] = {}
    for row in all_route_rows:
        route_accuracy_by_id.setdefault(str(row["route_id"]), []).append(
            float(row["accuracy"])
        )
    reactivation_count = sum(bool(row.get("reactivated")) for row in route_history)
    replacement_count = sum(
        1
        for row in route_history
        if (row.get("activation") or {}).get("victim_slot") is not None
    )
    source_binding_slot = index.archive.binding_slot(source_binding_id)
    gates = {
        "target_file_count_archived": archive_count >= len(SCHEDULE),
        "every_admitted_file_mastered": len(direct) == len(SCHEDULE)
        and all(_stable(rows) for rows in direct),
        "hot_cache_capacity_bounded": len(index.active_binding_ids) == ACTIVE_CACHE_SLOTS,
        "cold_archive_reactivation_mastery": bool(all_route_rows)
        and all(
            len(values) >= 1 and min(values) >= MASTERY_THRESHOLD
            for values in route_accuracy_by_id.values()
        )
        and len(route_accuracy_by_id) == accepted_count,
        "multiple_reactivations_occurred": reactivation_count >= 3,
        "multiple_replacements_occurred": replacement_count >= 3,
        "source_protected": index.archive.is_protected(source_binding_id),
        "protected_source_never_evicted": source_binding_slot == 0,
        "unknown_key_fails_closed": unknown_lookup.binding_id is None,
        "unknown_key_did_not_change_cache": unknown_route["active_before"]
        == unknown_route["active_after"],
        "shuffled_outcome_control_rejects_mastery": max(
            float(row["accuracy"]) for row in shuffled_control
        )
        < MASTERY_THRESHOLD,
        "archive_reload_exact": index_reload_exact,
        "archive_corruption_rejected": corruption_rejected,
        "artifact_corruption_rejected": artifact_corruption_rejected,
        "all_active_files_match_immutable_snapshots": _active_digests(
            system, index, snapshots
        ),
        "all_archived_snapshot_digests_stable": all(
            handle == snapshot.digest for handle, snapshot in snapshots.items()
        ),
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    route_bits = sum(
        int(row.get("unique_verifier_bits", 0)) for row in route_history
    )
    source_bits = sum(
        int(row["unique_verifier_bits"])
        for rows in source_probes
        for row in rows
    )
    training_bits = total_training_bits + source_bits + route_bits
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Outcome-only bounded hot-cache lifecycle for independently learned "
            "external compute artifacts; not unrestricted memory growth, learned "
            "compression, arbitrary program induction, or general continual learning."
        ),
        "architecture": {
            "active_cache_slots": ACTIVE_CACHE_SLOTS,
            "archive": "episodic_binding_artifact_index_v1",
            "artifact": "external_register_compute_basis_artifact_v1_plus_opaque_decoder_state",
            "allocation": "fresh_candidate_then_verifier_gated_cache_installation_v1",
            "victim_policy": "unprotected_reliability_recency_telemetry_v1",
            "route_query": "learned_event_tensor_key",
            "route_feedback": "deterministic_scalar_verifier_accuracy",
            "source_family": SOURCE_FAMILY,
            "schedule": [
                {"family": family, "cue": cue} for family, cue in SCHEDULE
            ],
            "unknown_cue": UNKNOWN_CUE,
        },
        "seed": args.seed,
        "allocation": allocation,
        "direct": direct,
        "source_protection": source_probes,
        "route_history": route_history,
        "route_accuracy_by_id": route_accuracy_by_id,
        "archive": {
            "record_count": index.record_count,
            "active_binding_ids": list(index.active_binding_ids),
            "source_binding_id": source_binding_id,
            "source_binding_slot": source_binding_slot,
            "protected": list(index.archive.status().protected),
            "handles": sorted(snapshots),
            "payload_checksum": index_payload["checksum"],
            "reloaded_payload_checksum": restored_payload["checksum"],
        },
        "unknown_route": unknown_route,
        "shuffled_control": {
            "history_tail": control_history[-5:],
            "evaluation": shuffled_control,
            "replayed_examples": 0,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": training_bits,
            "control_verifier_bits": sum(
                int(row["unique_verifier_bits"]) for row in control_history
            ),
            "unique_logical_lifetimes": args.batch_size
            * (args.file_updates * len(direct) + len(source_probes) + len(route_history)),
            "optimizer_updates": optimizer_updates,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": training_bits if all(gates.values()) else None,
            "retention_threshold": MASTERY_THRESHOLD,
            "transfer_ratio_against_fresh_learner": "not measured in this lifecycle rung",
        },
        "status": "promoted_external_compute_artifact_cache_pressure"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-file-count", type=int, default=4)
    parser.add_argument("--file-updates", type=int, default=256)
    parser.add_argument("--route-revisits", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retention-lifetimes", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
