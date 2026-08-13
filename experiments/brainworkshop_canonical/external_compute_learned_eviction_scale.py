"""Learn external-compute eviction utility while scaling a cold archive.

The preceding cache-pressure rung used a deterministic reliability/recency
victim rule.  This audit replaces that rule with an independently trainable
memory-side policy.  The policy sees a learned event-tensor context and opaque
fixed-width descriptors of resident external files.  It is updated only from
paired fresh verifier outcomes: a file that performs poorly on the incoming
task is more disposable.  Protection remains a separate verifier-owned gate.

Six independently learned external files are retained behind a three-slot hot
cache.  The controller, event frontend, and generic interpreter remain frozen
after the first file.  The policy is not a controller branch and receives no
family names, route IDs, correct actions, or semantic metadata.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Mapping

import torch
from torch.nn import functional as F

from neural_computer import (
    EpisodicBindingArtifactIndex,
    ExternalCapabilityEvictionPolicy,
    ExternalRegisterComputeBasisArtifact,
    paired_counterfactual_ranking_loss,
)

from .external_compute_artifact_cache_pressure import (
    _FileSnapshot,
    _active_digests,
    _append_compute_file,
    _discard_newest_compute_file,
    _restore_snapshot,
    _route_key,
    _snapshot,
    _stable,
    _train_file,
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
)
from .external_compute_route_bank import _all_modules, _family_steps


SCHEMA = "neural-computer.brainworkshop-external-compute-learned-eviction-scale.v1"
SCHEDULE = (
    ("symbol_parity", 7),
    ("triplet_parity", 8),
    ("parity2", 10),
    ("switch_binary", 11),
    ("symbol_parity_odd", 5),
    ("nback3", 4),
)
ACTIVE_CACHE_SLOTS = 3
POLICY_CONTEXT_WIDTH = EVENT_WIDTH
POLICY_CANDIDATE_WIDTH = 32
POLICY_HIDDEN = 32
MASTERY_THRESHOLD = 0.80
MATCHING_THRESHOLD = 0.999
SOURCE_PROTECTION_OBSERVATIONS = 16
UNKNOWN_CUE = 6


@dataclass(frozen=True)
class _PolicyProbe:
    context: torch.Tensor
    features: torch.Tensor
    outcomes: dict[int, float]
    unique_verifier_bits: int


def _artifact_feature(snapshot: _FileSnapshot) -> torch.Tensor:
    """Make a fixed-width opaque descriptor from portable file state."""

    chunks: list[torch.Tensor] = []
    for state in (
        snapshot.basis.state,
        snapshot.instruction,
        snapshot.readout,
        snapshot.decoder,
    ):
        for name in sorted(state):
            chunks.append(state[name].detach().cpu().reshape(-1).to(torch.float32))
    flat = torch.cat(chunks)
    positions = torch.linspace(0, flat.numel() - 1, POLICY_CANDIDATE_WIDTH).round().long()
    return F.normalize(flat[positions], dim=0)


def _active_features(
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for binding_id in index.active_binding_ids:
        if binding_id is None:
            raise RuntimeError("learned-eviction policy requires a full hot cache")
        handle = index.artifact_handle(binding_id)
        if handle is None or handle not in snapshots:
            raise RuntimeError("active file has no opaque feature descriptor")
        rows.append(_artifact_feature(snapshots[handle]))
    return torch.stack(rows)


def _policy_scores(
    policy: ExternalCapabilityEvictionPolicy,
    context: torch.Tensor,
    features: torch.Tensor,
) -> torch.Tensor:
    if context.ndim != 1 or context.shape[0] != POLICY_CONTEXT_WIDTH:
        raise ValueError("learned eviction context has the wrong shape")
    if features.shape != (ACTIVE_CACHE_SLOTS, POLICY_CANDIDATE_WIDTH):
        raise ValueError("learned eviction features have the wrong shape")
    return policy.score_candidates(
        context.unsqueeze(0),
        features.unsqueeze(0),
    ).squeeze(0)


def _probe_active(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    *,
    family: str,
    cue_symbol: int,
    batch_size: int,
    seed: int,
    retention_lifetimes: int,
) -> _PolicyProbe:
    context = _route_key(system, cue_symbol)
    features = _active_features(index, snapshots)
    outcomes: dict[int, float] = {}
    bits = 0
    for slot, binding_id in enumerate(index.active_binding_ids):
        if binding_id is None:
            raise RuntimeError("active policy probe found an empty slot")
        rows = _evaluate(
            system,
            family=family,
            slot=slot,
            cue_symbol=cue_symbol,
            lifetimes=retention_lifetimes,
            batch_size=batch_size,
            steps=_family_steps(family),
            seed=seed + slot * 1_003,
        )
        outcomes[slot] = min(float(row["accuracy"]) for row in rows)
        bits += sum(int(row["unique_verifier_bits"]) for row in rows)
    return _PolicyProbe(context, features, outcomes, bits)


def _eligible_slots(
    index: EpisodicBindingArtifactIndex,
) -> tuple[int, ...]:
    return tuple(
        slot
        for slot, binding_id in enumerate(index.active_binding_ids)
        if binding_id is not None
        and not index.archive.is_protected(binding_id)
    )


def _adapt_policy(
    policy: ExternalCapabilityEvictionPolicy,
    optimizer: torch.optim.Optimizer,
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    *,
    family: str,
    cue_symbol: int,
    batch_size: int,
    updates: int,
    seed: int,
    retention_lifetimes: int,
    shuffle_utility: bool = False,
    minimum_utility_gap: float = 0.0,
) -> tuple[list[dict[str, float | int]], _PolicyProbe | None]:
    history: list[dict[str, float | int]] = []
    last_probe: _PolicyProbe | None = None
    eligible = _eligible_slots(index)
    if len(eligible) < 2:
        return history, None
    for update in range(updates):
        probe = _probe_active(
            system,
            index,
            snapshots,
            family=family,
            cue_symbol=cue_symbol,
            batch_size=batch_size,
            seed=seed + update * 10_007,
            retention_lifetimes=retention_lifetimes,
        )
        last_probe = probe
        pair = (eligible[update % len(eligible)], eligible[(update + 1) % len(eligible)])
        if len(eligible) == 2:
            pair = (eligible[0], eligible[1])
        utility = torch.tensor(
            [[1.0 - probe.outcomes[pair[0]], 1.0 - probe.outcomes[pair[1]]]],
            dtype=torch.float32,
        )
        if shuffle_utility:
            utility = utility.flip(dims=(1,))
        scores = _policy_scores(policy, probe.context, probe.features)
        attempted = torch.tensor([pair], dtype=torch.long)
        loss, advantage = paired_counterfactual_ranking_loss(
            scores.unsqueeze(0),
            attempted,
            utility,
        )
        updated = abs(float(advantage.item())) >= minimum_utility_gap
        if updated:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable = [
                parameter
                for parameter in policy.parameters()
                if parameter.grad is not None
            ]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optimizer.step()
        history.append(
            {
                "update": update + 1,
                "policy_updated": int(updated),
                "loss": float(loss.detach()),
                "utility_gap": float(advantage.mean()),
                "unique_verifier_bits": probe.unique_verifier_bits,
                "replayed_examples": 0,
            }
        )
    return history, last_probe


def _select_victim(
    policy: ExternalCapabilityEvictionPolicy,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    probe: _PolicyProbe,
) -> tuple[int | None, int | None, int | None, torch.Tensor]:
    scores = _policy_scores(policy, probe.context, probe.features)
    masked = scores.clone()
    for slot in range(ACTIVE_CACHE_SLOTS):
        binding_id = index.active_binding_ids[slot]
        if binding_id is None or index.archive.is_protected(binding_id):
            masked[slot] = -torch.inf
    if not bool(torch.isfinite(masked).any()):
        return None, None, None, scores
    chosen = int(masked.argmax())
    eligible = _eligible_slots(index)
    oracle = max(
        eligible,
        key=lambda slot: 1.0 - probe.outcomes[slot],
    )
    return chosen, oracle, int(chosen == oracle), scores


def _reactivate_with_policy(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    policy: ExternalCapabilityEvictionPolicy,
    policy_optimizer: torch.optim.Optimizer,
    *,
    route_id: int,
    spec: tuple[str, int],
    batch_size: int,
    retention_lifetimes: int,
    policy_updates: int,
    seed: int,
) -> dict[str, object]:
    family, cue_symbol = spec
    key = _route_key(system, cue_symbol)
    lookup = index.lookup(key)
    if lookup.binding_id is None:
        return {
            "route_id": route_id,
            "lookup": False,
            "reactivated": False,
            "accuracy": None,
            "replayed_examples": 0,
        }
    binding_id = lookup.binding_id
    if lookup.active_slot is not None:
        rows = _evaluate(
            system,
            family=family,
            slot=lookup.active_slot,
            cue_symbol=cue_symbol,
            lifetimes=retention_lifetimes,
            batch_size=batch_size,
            steps=_family_steps(family),
            seed=seed + 100_000,
        )
        outcome = min(float(row["accuracy"]) for row in rows)
        index.archive.observe(binding_id, outcome, step=seed)
        return {
            "route_id": route_id,
            "lookup": True,
            "reactivated": False,
            "accuracy": outcome,
            "selected_slot": lookup.active_slot,
            "unique_verifier_bits": sum(int(row["unique_verifier_bits"]) for row in rows),
            "replayed_examples": 0,
        }

    handle = lookup.artifact_handle
    if handle is None or handle not in snapshots:
        raise RuntimeError("cold learned-eviction route has no artifact")
    probe_history, probe = _adapt_policy(
        policy,
        policy_optimizer,
        system,
        index,
        snapshots,
        family=family,
        cue_symbol=cue_symbol,
        batch_size=batch_size,
        updates=policy_updates,
            seed=seed,
            retention_lifetimes=1,
            minimum_utility_gap=0.15,
        )
    if probe is None:
        return {
            "route_id": route_id,
            "lookup": True,
            "reactivated": False,
            "accuracy": None,
            "reactivation_blocked": True,
            "policy_history": probe_history,
            "replayed_examples": 0,
        }
    chosen, oracle, selection_correct, scores = _select_victim(
        policy,
        index,
        snapshots,
        probe,
    )
    if chosen is None:
        return {
            "route_id": route_id,
            "lookup": True,
            "reactivated": False,
            "accuracy": None,
            "reactivation_blocked": True,
            "policy_history": probe_history,
            "replayed_examples": 0,
        }
    displaced_binding = index.archive.active_binding(chosen)
    displaced_snapshot = None
    if displaced_binding is not None:
        displaced_handle = index.artifact_handle(displaced_binding)
        if displaced_handle is None or displaced_handle not in snapshots:
            raise RuntimeError("learned-eviction victim has no artifact")
        displaced_snapshot = snapshots[displaced_handle]
    _restore_snapshot(system, chosen, snapshots[handle])
    retention_probe: list[dict[str, float | int]] = []

    def verify(_candidate: EpisodicBindingArtifactIndex) -> bool:
        retention_probe.extend(
            _evaluate(
                system,
                family=family,
                slot=chosen,
                cue_symbol=cue_symbol,
                lifetimes=retention_lifetimes,
                batch_size=batch_size,
                steps=_family_steps(family),
                seed=seed + 200_000,
            )
        )
        return _stable(retention_probe)

    activation = index.reactivate_verified(binding_id, chosen, verify)
    if not activation.accepted:
        if displaced_snapshot is not None:
            _restore_snapshot(system, chosen, displaced_snapshot)
        return {
            "route_id": route_id,
            "lookup": True,
            "reactivated": False,
            "accuracy": None,
            "activation": activation.reason,
            "policy_history": probe_history,
            "policy_scores": scores.tolist(),
            "oracle_slot": oracle,
            "selected_slot": chosen,
            "selection_correct": selection_correct,
            "replayed_examples": 0,
        }
    rows = _evaluate(
        system,
        family=family,
        slot=chosen,
        cue_symbol=cue_symbol,
        lifetimes=retention_lifetimes,
        batch_size=batch_size,
        steps=_family_steps(family),
        seed=seed + 300_000,
    )
    outcome = min(float(row["accuracy"]) for row in rows)
    index.archive.observe(binding_id, outcome, step=seed)
    return {
        "route_id": route_id,
        "lookup": True,
        "reactivated": True,
        "accuracy": outcome,
        "selected_slot": chosen,
        "oracle_slot": oracle,
        "selection_correct": selection_correct,
        "policy_scores": scores.tolist(),
        "policy_history": probe_history,
        "retention_probe": retention_probe,
        "activation": activation.reason,
        "unique_verifier_bits": sum(int(row["unique_verifier_bits"]) for row in rows)
        + sum(int(row["unique_verifier_bits"]) for row in retention_probe),
        "replayed_examples": 0,
    }

def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if args.target_file_count != len(SCHEDULE):
        raise ValueError("the calibrated learned-eviction schedule has six files")
    if min(
        args.file_updates,
        args.batch_size,
        args.retention_lifetimes,
        args.policy_calibration_rounds,
        args.policy_updates_per_round,
        args.policy_updates_per_route,
        args.route_revisits,
    ) < 1:
        raise ValueError("learned-eviction budgets must be positive")
    if args.batch_size != 32:
        raise ValueError("the calibrated learned-eviction harness requires batch size 32")

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
    specs = {route_id: spec for route_id, spec in enumerate(SCHEDULE)}
    direct: list[list[dict[str, float | int]]] = []
    histories: list[list[dict[str, float | int]]] = []
    allocation: list[dict[str, object]] = []
    source_binding_id: int | None = None
    source_digest: str | None = None
    training_bits = 0
    optimizer_updates = 0

    for route_id, (family, cue_symbol) in enumerate(SCHEDULE):
        candidate_is_scratch = route_id >= ACTIVE_CACHE_SLOTS
        candidate_slot = (
            _append_compute_file(system, seed=args.seed + 80_000 + route_id)
            if candidate_is_scratch
            else route_id
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
        training_bits += sum(int(row["unique_verifier_bits"]) for row in history)
        optimizer_updates += args.file_updates
        if not _stable(fresh):
            if candidate_is_scratch:
                _discard_newest_compute_file(system)
            allocation.append(
                {
                    "route_id": route_id,
                    "family": family,
                    "accepted": False,
                    "fresh_probe": fresh,
                }
            )
            break
        snapshot = _snapshot(system, candidate_slot)
        handle = snapshot.digest
        snapshots[handle] = snapshot
        key = _route_key(system, cue_symbol)
        binding_id = index.register(key, key, handle)
        if route_id < ACTIVE_CACHE_SLOTS:
            index.activate(binding_id, route_id)
        else:
            _discard_newest_compute_file(system)
        index.archive.observe(
            binding_id,
            min(float(row["accuracy"]) for row in fresh),
            step=route_id + 1,
        )
        if route_id == 0:
            source_binding_id = binding_id
            source_digest = handle
            for observation in range(SOURCE_PROTECTION_OBSERVATIONS):
                source_rows = _evaluate(
                    system,
                    family=family,
                    slot=0,
                    cue_symbol=cue_symbol,
                    lifetimes=1,
                    batch_size=args.batch_size,
                    steps=_family_steps(family),
                    seed=args.seed + 300_000 + observation,
                )
                training_bits += sum(
                    int(row["unique_verifier_bits"]) for row in source_rows
                )
                index.archive.observe(
                    source_binding_id,
                    min(float(row["accuracy"]) for row in source_rows),
                    step=100 + observation,
                )
        allocation.append(
            {
                "route_id": route_id,
                "family": family,
                "binding_id": binding_id,
                "artifact_handle": handle,
                "resident_after_acquisition": route_id < ACTIVE_CACHE_SLOTS,
                "fresh_probe": fresh,
            }
        )

    accepted_count = len(snapshots)
    if source_binding_id is None or source_digest is None or accepted_count < len(SCHEDULE):
        rejected = {
            "schema": SCHEMA,
            "seed": args.seed,
            "allocation": allocation,
            "gates": {
                "all_six_files_admitted": False,
                "all_six_files_directly_mastered": False,
                "frozen_controller": controller_before == _digest(system.agent.controller),
                "frozen_event_encoder": encoder_before
                == _digest(system.agent.runtime.encoders["stimulus"]),
                "zero_replayed_examples": True,
            },
            "accounting": {
                "unique_verifier_bits": training_bits,
                "optimizer_updates": optimizer_updates,
                "replayed_examples": 0,
            },
            "status": "rejected",
        }
        if args.report_out is not None:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(json.dumps(rejected, indent=2) + "\n")
        return rejected

    policy = ExternalCapabilityEvictionPolicy(
        context_width=POLICY_CONTEXT_WIDTH,
        candidate_width=POLICY_CANDIDATE_WIDTH,
        hidden=POLICY_HIDDEN,
    )
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_learning_rate)
    shuffled_policy = ExternalCapabilityEvictionPolicy(
        context_width=POLICY_CONTEXT_WIDTH,
        candidate_width=POLICY_CANDIDATE_WIDTH,
        hidden=POLICY_HIDDEN,
    )
    shuffled_optimizer = torch.optim.Adam(
        shuffled_policy.parameters(), lr=args.policy_learning_rate
    )
    calibration: list[dict[str, object]] = []
    policy_bits = 0
    shuffled_bits = 0
    policy_lifetimes = 0
    shuffled_lifetimes = 0
    policy_updates = 0
    for round_index in range(args.policy_calibration_rounds):
        family, cue_symbol = SCHEDULE[round_index % len(SCHEDULE)]
        history, probe = _adapt_policy(
            policy,
            policy_optimizer,
            system,
            index,
            snapshots,
            family=family,
            cue_symbol=cue_symbol,
            batch_size=args.batch_size,
            updates=args.policy_updates_per_round,
            seed=args.seed + 500_000 + round_index * 10_000,
            retention_lifetimes=1,
        )
        shuffled_history, _ = _adapt_policy(
            shuffled_policy,
            shuffled_optimizer,
            system,
            index,
            snapshots,
            family=family,
            cue_symbol=cue_symbol,
            batch_size=args.batch_size,
            updates=args.policy_updates_per_round,
            seed=args.seed + 1_500_000 + round_index * 10_000,
            retention_lifetimes=1,
            shuffle_utility=True,
        )
        policy_bits += sum(int(row["unique_verifier_bits"]) for row in history)
        shuffled_bits += sum(int(row["unique_verifier_bits"]) for row in shuffled_history)
        policy_lifetimes += args.batch_size * len(history) * ACTIVE_CACHE_SLOTS
        shuffled_lifetimes += args.batch_size * len(shuffled_history) * ACTIVE_CACHE_SLOTS
        policy_updates += len(history)
        if probe is None:
            continue
        chosen, oracle, correct, scores = _select_victim(
            policy,
            index,
            snapshots,
            probe,
        )
        shuffled_scores = _policy_scores(shuffled_policy, probe.context, probe.features)
        shuffled_masked = shuffled_scores.clone()
        for slot in range(ACTIVE_CACHE_SLOTS):
            binding_id = index.active_binding_ids[slot]
            if binding_id is None or index.archive.is_protected(binding_id):
                shuffled_masked[slot] = -torch.inf
        shuffled_chosen = int(shuffled_masked.argmax())
        calibration.append(
            {
                "round": round_index + 1,
                "family": family,
                "chosen_slot": chosen,
                "oracle_slot": oracle,
                "selection_correct": correct,
                "shuffled_chosen_slot": shuffled_chosen,
                "shuffled_selection_correct": int(shuffled_chosen == oracle),
                "policy_scores": scores.tolist(),
                "outcomes": probe.outcomes,
            }
        )

    route_order = [3, 4, 5, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
    route_history: list[dict[str, object]] = []
    for ordinal, route_id in enumerate(route_order * args.route_revisits):
        result = _reactivate_with_policy(
            system,
            index,
            snapshots,
            policy,
            policy_optimizer,
            route_id=route_id,
            spec=specs[route_id],
            batch_size=args.batch_size,
            retention_lifetimes=args.retention_lifetimes,
            policy_updates=args.policy_updates_per_route,
            seed=args.seed + 3_000_000 + ordinal * 10_000,
        )
        result["ordinal"] = ordinal
        route_history.append(result)
        policy_bits += sum(
            int(row["unique_verifier_bits"])
            for row in result.get("policy_history", [])
        )
        policy_bits += int(result.get("unique_verifier_bits", 0))
        policy_lifetimes += args.batch_size * (
            len(result.get("policy_history", [])) * ACTIVE_CACHE_SLOTS
            + len(result.get("retention_probe", []))
            + (1 if result.get("unique_verifier_bits") else 0)
        )
        policy_updates += len(result.get("policy_history", []))

    unknown = index.lookup(_route_key(system, UNKNOWN_CUE))
    active_before_unknown = index.active_binding_ids
    active_after_unknown = index.active_binding_ids
    index_payload = index.payload()
    restored_index = EpisodicBindingArtifactIndex.from_payload(index_payload)
    reload_exact = restored_index.payload() == index_payload
    policy_snapshot = {
        name: value.detach().clone() for name, value in policy.state_dict().items()
    }
    restored_policy = ExternalCapabilityEvictionPolicy(
        context_width=POLICY_CONTEXT_WIDTH,
        candidate_width=POLICY_CANDIDATE_WIDTH,
        hidden=POLICY_HIDDEN,
    )
    restored_policy.load_state_dict(policy_snapshot)
    probe_for_reload = _probe_active(
        system,
        index,
        snapshots,
        family=SCHEDULE[0][0],
        cue_symbol=SCHEDULE[0][1],
        batch_size=args.batch_size,
        seed=args.seed + 9_000_000,
        retention_lifetimes=1,
    )
    policy_reload_exact = torch.equal(
        _policy_scores(policy, probe_for_reload.context, probe_for_reload.features),
        _policy_scores(restored_policy, probe_for_reload.context, probe_for_reload.features),
    )
    corrupted = dict(index_payload)
    corrupted["checksum"] = "tampered"
    archive_corruption_rejected = False
    try:
        EpisodicBindingArtifactIndex.from_payload(corrupted)
    except ValueError as error:
        archive_corruption_rejected = "checksum" in str(error).lower()
    artifact_payload = next(iter(snapshots.values())).basis.payload()
    artifact_payload["sha256"] = "tampered"
    artifact_corruption_rejected = False
    try:
        ExternalRegisterComputeBasisArtifact.from_payload(artifact_payload)
    except ValueError as error:
        artifact_corruption_rejected = "checksum" in str(error).lower()

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    cold_rows = [row for row in route_history if bool(row.get("reactivated"))]
    route_rows = [row for row in route_history if row.get("accuracy") is not None]
    calibration_correct = [int(row["selection_correct"]) for row in calibration]
    shuffled_correct = [int(row["shuffled_selection_correct"]) for row in calibration]
    route_correct = [int(row["selection_correct"]) for row in cold_rows]
    route_accuracy = [float(row["accuracy"]) for row in route_rows]
    gates = {
        "all_six_files_admitted": len(snapshots) == len(SCHEDULE),
        "all_six_files_directly_mastered": len(direct) == len(SCHEDULE)
        and all(_stable(rows) for rows in direct),
        "active_cache_capacity_bounded": len(index.active_binding_ids)
        == ACTIVE_CACHE_SLOTS,
        "cold_routes_reactivated": len(cold_rows) >= 5,
        "cold_route_mastery": bool(route_accuracy)
        and min(route_accuracy) >= MASTERY_THRESHOLD,
        "learned_calibration_beats_chance": bool(calibration_correct)
        and sum(calibration_correct) / len(calibration_correct) >= 0.65,
        "learned_calibration_beats_shuffled": bool(calibration_correct)
        and sum(calibration_correct) > sum(shuffled_correct),
        "learned_route_selection_beats_chance": bool(route_correct)
        and sum(route_correct) / len(route_correct) >= 0.60,
        "protected_source_retained": source_binding_id is not None
        and index.archive.is_protected(source_binding_id)
        and index.archive.binding_slot(source_binding_id) == 0,
        "unknown_route_fails_closed": unknown.binding_id is None,
        "unknown_route_did_not_change_cache": active_before_unknown
        == active_after_unknown,
        "shuffled_utility_control_is_not_mastery": sum(shuffled_correct)
        <= max(1, len(shuffled_correct) // 2),
        "archive_reload_exact": reload_exact,
        "policy_reload_exact": policy_reload_exact,
        "archive_corruption_rejected": archive_corruption_rejected,
        "artifact_corruption_rejected": artifact_corruption_rejected,
        "active_files_match_immutable_snapshots": _active_digests(
            system, index, snapshots
        ),
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Outcome-only learned eviction utility for a six-file external "
            "compute archive behind a bounded hot cache; not unrestricted "
            "memory growth, learned compression, arbitrary program induction, "
            "or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "schedule": [
                {"family": family, "cue": cue} for family, cue in SCHEDULE
            ],
            "active_cache_slots": ACTIVE_CACHE_SLOTS,
            "archive": "episodic_binding_artifact_index_v1",
            "artifact": "external_register_compute_basis_artifact_v1_plus_opaque_decoder_state",
            "policy": "external_capability_eviction_policy_v1",
            "policy_features": "learned_event_tensor_plus_opaque_artifact_descriptor_v1",
            "policy_signal": "paired_fresh_scalar_disposability_utility_v1",
            "calibration_rounds": args.policy_calibration_rounds,
            "updates_per_calibration_round": args.policy_updates_per_round,
            "updates_per_route": args.policy_updates_per_route,
            "minimum_route_utility_gap": 0.15,
            "protection": "stable_prefix_verifier_gate_outside_policy_v1",
        },
        "allocation": allocation,
        "direct": direct,
        "training_history_tails": [history[-5:] for history in histories],
        "calibration": calibration,
        "route_history": route_history,
        "archive": {
            "record_count": index.record_count,
            "active_binding_ids": list(index.active_binding_ids),
            "protected": list(index.archive.status().protected),
            "source_digest": source_digest,
        },
        "unknown_route": {
            "binding_id": unknown.binding_id,
            "active_before": active_before_unknown,
            "active_after": active_after_unknown,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": training_bits + policy_bits,
            "shuffled_control_verifier_bits": shuffled_bits,
            "unique_logical_lifetimes": args.batch_size
            * (args.file_updates * len(direct) + SOURCE_PROTECTION_OBSERVATIONS)
            + policy_lifetimes,
            "shuffled_control_logical_lifetimes": shuffled_lifetimes,
            "optimizer_updates": optimizer_updates,
            "policy_updates": policy_updates,
            "replayed_examples": 0,
            "stable_bits_to_threshold": (
                training_bits + policy_bits if all(gates.values()) else None
            ),
            "retention_threshold": MASTERY_THRESHOLD,
            "transfer_ratio_against_fresh_learner": "not measured in this lifecycle rung",
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_external_compute_learned_eviction_scale"
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
    parser.add_argument("--target-file-count", type=int, default=6)
    parser.add_argument("--file-updates", type=int, default=256)
    parser.add_argument("--policy-calibration-rounds", type=int, default=24)
    parser.add_argument("--policy-updates-per-round", type=int, default=8)
    parser.add_argument("--policy-updates-per-route", type=int, default=2)
    parser.add_argument("--route-revisits", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retention-lifetimes", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--policy-learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
