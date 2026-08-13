"""Test learned eviction transfer on a held-out external compute family.

The promoted six-file eviction rung calibrates and serves one family cohort.
This audit freezes that cohort, introduces a held-out n-back-2 artifact, and
compares the inherited memory-side policy with a matched fresh policy. Both
policies receive the same fresh verifier probes and may update only external
policy state; the controller, event encoder, and executable artifacts remain
frozen.

This is a transfer measurement, not an assumption that the inherited policy
must win. A failed transfer is valuable evidence about the remaining
continual-learning bottleneck.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping

import torch

from neural_computer import (
    EpisodicBindingArtifactIndex,
    ExternalCapabilityEvictionPolicy,
    GatedResidualCapabilityEvictionPolicyBank,
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
    _digest,
    _evaluate,
)
from .external_compute_learned_eviction_scale import (
    ACTIVE_CACHE_SLOTS,
    MATCHING_THRESHOLD,
    MASTERY_THRESHOLD,
    POLICY_CANDIDATE_WIDTH,
    POLICY_CONTEXT_WIDTH,
    POLICY_HIDDEN,
    SCHEDULE,
    SOURCE_PROTECTION_OBSERVATIONS,
    _adapt_policy,
    _eligible_slots,
    _policy_scores,
    _probe_active,
    _select_victim,
)
from .external_compute_route_bank import _family_steps


SCHEMA = "neural-computer.brainworkshop-external-compute-eviction-transfer.v1"
TRANSFER_FAMILY = "nback2"
TRANSFER_CUE = 9
UTILITY_GAP_GATE = 0.15
STABILITY_WINDOW = 3


def _acquire_source_cohort(
    args: argparse.Namespace,
) -> tuple[
    ComputeGrowthSystem,
    EpisodicBindingArtifactIndex,
    dict[str, _FileSnapshot],
    list[list[dict[str, float | int]]],
    int,
    int,
    int,
]:
    """Acquire the six-file source cohort and protect its first file."""

    system = _build(args.seed, slot_count=ACTIVE_CACHE_SLOTS)
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
    direct: list[list[dict[str, float | int]]] = []
    verifier_bits = 0
    optimizer_updates = 0
    logical_lifetimes = 0
    source_binding_id: int | None = None

    for route_id, (family, cue_symbol) in enumerate(SCHEDULE):
        scratch = route_id >= ACTIVE_CACHE_SLOTS
        slot = (
            _append_compute_file(system, seed=args.seed + 80_000 + route_id)
            if scratch
            else route_id
        )
        history, fresh = _train_file(
            system,
            slot=slot,
            family=family,
            cue_symbol=cue_symbol,
            updates=args.file_updates,
            batch_size=args.batch_size,
            seed=args.seed + 10_000 * (route_id + 1),
            learning_rate=args.learning_rate,
            first_file=route_id == 0,
        )
        optimizer_updates += len(history)
        logical_lifetimes += args.batch_size * (len(history) + len(fresh))
        verifier_bits += sum(int(row["unique_verifier_bits"]) for row in history)
        verifier_bits += sum(int(row["unique_verifier_bits"]) for row in fresh)
        direct.append(fresh)
        if not _stable(fresh):
            if scratch:
                _discard_newest_compute_file(system)
            raise RuntimeError(f"source family {family} failed fresh mastery")

        artifact = _snapshot(system, slot)
        snapshots[artifact.digest] = artifact
        key = _route_key(system, cue_symbol)
        binding_id = index.register(key, key, artifact.digest)
        if scratch:
            _discard_newest_compute_file(system)
        else:
            index.activate(binding_id, slot)
        index.archive.observe(
            binding_id,
            min(float(row["accuracy"]) for row in fresh),
            step=route_id + 1,
        )

        if route_id == 0:
            source_binding_id = binding_id
            for observation in range(SOURCE_PROTECTION_OBSERVATIONS):
                rows = _evaluate(
                    system,
                    family=family,
                    slot=0,
                    cue_symbol=cue_symbol,
                    lifetimes=1,
                    batch_size=args.batch_size,
                    steps=_family_steps(family),
                    seed=args.seed + 300_000 + observation,
                )
                verifier_bits += sum(
                    int(row["unique_verifier_bits"]) for row in rows
                )
                logical_lifetimes += args.batch_size * len(rows)
                index.archive.observe(
                    binding_id,
                    min(float(row["accuracy"]) for row in rows),
                    step=100 + observation,
                )

    if source_binding_id is None:
        raise RuntimeError("source protection binding was not created")
    return (
        system,
        index,
        snapshots,
        direct,
        verifier_bits,
        optimizer_updates,
        logical_lifetimes,
    )


def _acquire_transfer_file(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: dict[str, _FileSnapshot],
    *,
    args: argparse.Namespace,
) -> tuple[int, str, list[dict[str, float | int]], int, int]:
    """Train the held-out file in scratch capacity and archive it cold."""

    slot = _append_compute_file(system, seed=args.seed + 90_000)
    history, fresh = _train_file(
        system,
        slot=slot,
        family=TRANSFER_FAMILY,
        cue_symbol=TRANSFER_CUE,
        updates=args.file_updates,
        batch_size=args.batch_size,
        seed=args.seed + 700_000,
        learning_rate=args.learning_rate,
        first_file=False,
    )
    bits = sum(int(row["unique_verifier_bits"]) for row in history)
    bits += sum(int(row["unique_verifier_bits"]) for row in fresh)
    logical_lifetimes = args.batch_size * (len(history) + len(fresh))
    if not _stable(fresh):
        _discard_newest_compute_file(system)
        raise RuntimeError("held-out family failed fresh mastery")
    artifact = _snapshot(system, slot)
    snapshots[artifact.digest] = artifact
    key = _route_key(system, TRANSFER_CUE)
    binding_id = index.register(key, key, artifact.digest)
    _discard_newest_compute_file(system)
    index.archive.observe(
        binding_id,
        min(float(row["accuracy"]) for row in fresh),
        step=10_000,
    )
    return binding_id, artifact.digest, fresh, bits, logical_lifetimes


def _install_related_source(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    *,
    args: argparse.Namespace,
) -> tuple[bool, int, int, dict[str, object]]:
    """Place the known n-back-3 source in the hot cache through verification."""

    lookup = index.lookup(_route_key(system, 4))
    if lookup.binding_id is None or lookup.artifact_handle is None:
        raise RuntimeError("related n-back-3 source is missing")
    if lookup.active_slot is not None:
        return True, 0, 0, {"already_active_slot": lookup.active_slot}
    destination = 2
    displaced = index.archive.active_binding(destination)
    displaced_handle = (
        index.artifact_handle(displaced) if displaced is not None else None
    )
    _restore_snapshot(system, destination, snapshots[lookup.artifact_handle])
    rows: list[dict[str, float | int]] = []

    def verify(_candidate: EpisodicBindingArtifactIndex) -> bool:
        rows.extend(
            _evaluate(
                system,
                family="nback3",
                slot=destination,
                cue_symbol=4,
                lifetimes=1,
                batch_size=args.batch_size,
                steps=_family_steps("nback3"),
                seed=args.seed + 850_000,
            )
        )
        return _stable(rows)

    receipt = index.reactivate_verified(lookup.binding_id, destination, verify)
    if not receipt.accepted and displaced_handle is not None:
        _restore_snapshot(system, destination, snapshots[displaced_handle])
    return (
        receipt.accepted,
        sum(int(row["unique_verifier_bits"]) for row in rows),
        args.batch_size * len(rows),
        {
            "binding_id": lookup.binding_id,
            "destination_slot": destination,
            "receipt": receipt.__dict__,
            "retention_probe": rows,
        },
    )


def _policy_update(
    policy,
    optimizer: torch.optim.Optimizer,
    probe,
    *,
    eligible: tuple[int, ...],
) -> dict[str, float | int]:
    """Update one policy using only a shared probe's paired scalar outcomes."""

    if len(eligible) < 2:
        raise RuntimeError("transfer policy needs two eligible residents")
    pair = (eligible[0], eligible[1])
    utility = torch.tensor(
        [[1.0 - probe.outcomes[pair[0]], 1.0 - probe.outcomes[pair[1]]]],
        dtype=torch.float32,
    )
    scores = _policy_scores(policy, probe.context, probe.features)
    masked = scores.clone()
    for slot in range(ACTIVE_CACHE_SLOTS):
        if slot not in eligible:
            masked[slot] = -torch.inf
    chosen = int(masked.argmax())
    oracle = max(eligible, key=lambda slot: 1.0 - probe.outcomes[slot])
    loss, advantage = paired_counterfactual_ranking_loss(
        scores.unsqueeze(0),
        torch.tensor([pair], dtype=torch.long),
        utility,
    )
    gap = abs(float(advantage.item()))
    updated = gap >= UTILITY_GAP_GATE
    if updated:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        trainable = [
            parameter for parameter in policy.parameters() if parameter.grad is not None
        ]
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
    return {
        "chosen_slot": chosen,
        "oracle_slot": oracle,
        "selection_correct": int(chosen == oracle),
        "policy_updated": int(updated),
        "utility_gap": float(advantage.item()),
        "loss": float(loss.detach()),
        "unique_verifier_bits": probe.unique_verifier_bits,
        "replayed_examples": 0,
    }


def _transfer_curve(
    system: ComputeGrowthSystem,
    index: EpisodicBindingArtifactIndex,
    snapshots: Mapping[str, _FileSnapshot],
    inherited: ExternalCapabilityEvictionPolicy,
    fresh: ExternalCapabilityEvictionPolicy,
    inherited_optimizer: torch.optim.Optimizer,
    fresh_optimizer: torch.optim.Optimizer,
    *,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], int, int, int]:
    """Give both policies the same fresh held-out-family probes."""

    eligible = _eligible_slots(index)
    rows: list[dict[str, object]] = []
    bits = 0
    policy_updates = 0
    logical_lifetimes = 0
    for update in range(args.transfer_updates):
        probe = _probe_active(
            system,
            index,
            snapshots,
            family=TRANSFER_FAMILY,
            cue_symbol=TRANSFER_CUE,
            batch_size=args.batch_size,
            seed=args.seed + 1_000_000 + update * 10_007,
            retention_lifetimes=1,
        )
        bits += probe.unique_verifier_bits
        logical_lifetimes += args.batch_size * ACTIVE_CACHE_SLOTS
        inherited_row = _policy_update(
            inherited, inherited_optimizer, probe, eligible=eligible
        )
        fresh_row = _policy_update(fresh, fresh_optimizer, probe, eligible=eligible)
        policy_updates += int(inherited_row["policy_updated"])
        policy_updates += int(fresh_row["policy_updated"])
        rows.append(
            {
                "update": update + 1,
                "inherited": inherited_row,
                "fresh": fresh_row,
            }
        )
    for row in rows:
        prefix = rows[: int(row["update"])]
        row["inherited_cumulative_accuracy"] = sum(
            int(item["inherited"]["selection_correct"]) for item in prefix
        ) / len(prefix)
        row["fresh_cumulative_accuracy"] = sum(
            int(item["fresh"]["selection_correct"]) for item in prefix
        ) / len(prefix)
    return rows, bits, policy_updates, logical_lifetimes


def _stable_window(values: list[int]) -> bool:
    return any(
        sum(values[index : index + STABILITY_WINDOW])
        >= 0.60 * STABILITY_WINDOW
        for index in range(len(values) - STABILITY_WINDOW + 1)
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.file_updates,
        args.batch_size,
        args.policy_calibration_rounds,
        args.policy_updates_per_round,
        args.transfer_updates,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("transfer budgets must be positive")
    if args.batch_size != 32:
        raise ValueError("the calibrated transfer harness requires batch size 32")

    try:
        (
            system,
            index,
            snapshots,
            direct,
            verifier_bits,
            optimizer_updates,
            logical_lifetimes,
        ) = _acquire_source_cohort(args)
    except RuntimeError as error:
        report = {
            "schema": SCHEMA,
            "seed": args.seed,
            "claim_boundary": (
                "Rejected before source-cohort qualification; no transfer "
                "claim is made."
            ),
            "error": str(error),
            "gates": {
                "source_cohort_mastered": False,
                "frozen_controller": False,
                "frozen_event_encoder": False,
                "zero_replayed_examples": True,
            },
            "accounting": {
                "replayed_examples": 0,
                "unique_logical_lifetimes": 0,
                "stable_bits_to_threshold": None,
            },
            "status": "rejected",
        }
        if args.report_out is not None:
            args.report_out.parent.mkdir(parents=True, exist_ok=True)
            args.report_out.write_text(json.dumps(report, indent=2) + "\n")
        return report
    (
        transfer_id,
        transfer_digest,
        transfer_direct,
        transfer_bits,
        transfer_lifetimes,
    ) = (
        _acquire_transfer_file(system, index, snapshots, args=args)
    )
    verifier_bits += transfer_bits
    logical_lifetimes += transfer_lifetimes
    optimizer_updates += args.file_updates
    (
        related_installed,
        related_bits,
        related_lifetimes,
        related_setup,
    ) = _install_related_source(
        system,
        index,
        snapshots,
        args=args,
    )
    verifier_bits += related_bits
    logical_lifetimes += related_lifetimes
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])

    base_policy = ExternalCapabilityEvictionPolicy(
        context_width=POLICY_CONTEXT_WIDTH,
        candidate_width=POLICY_CANDIDATE_WIDTH,
        hidden=POLICY_HIDDEN,
    )
    base_optimizer = torch.optim.Adam(
        base_policy.parameters(), lr=args.policy_learning_rate
    )
    source_calibration: list[dict[str, object]] = []
    policy_bits = 0
    policy_updates = 0
    for round_index in range(args.policy_calibration_rounds):
        family, cue_symbol = SCHEDULE[round_index % len(SCHEDULE)]
        history, probe = _adapt_policy(
            base_policy,
            base_optimizer,
            system,
            index,
            snapshots,
            family=family,
            cue_symbol=cue_symbol,
            batch_size=args.batch_size,
            updates=args.policy_updates_per_round,
            seed=args.seed + 500_000 + round_index * 10_000,
            retention_lifetimes=1,
            minimum_utility_gap=UTILITY_GAP_GATE,
        )
        policy_bits += sum(int(row["unique_verifier_bits"]) for row in history)
        policy_updates += len(history)
        logical_lifetimes += args.batch_size * len(history) * ACTIVE_CACHE_SLOTS
        if probe is not None:
            chosen, oracle, correct, scores = _select_victim(
                base_policy, index, snapshots, probe
            )
            source_calibration.append(
                {
                    "round": round_index + 1,
                    "family": family,
                    "chosen_slot": chosen,
                    "oracle_slot": oracle,
                    "selection_correct": correct,
                    "scores": scores.tolist(),
                }
            )

    residual_policy = GatedResidualCapabilityEvictionPolicyBank(
        base_policy,
        context_width=POLICY_CONTEXT_WIDTH,
        candidate_width=POLICY_CANDIDATE_WIDTH,
        max_slots=1,
        route_threshold=0.75,
        residual_gain=32.0,
    )
    residual_slot = residual_policy.add_slot(_route_key(system, TRANSFER_CUE))
    residual_policy.activate_slot(residual_slot)
    residual_optimizer = torch.optim.Adam(
        residual_policy.trainable_parameters(residual_slot),
        lr=args.policy_learning_rate,
    )
    fresh = ExternalCapabilityEvictionPolicy(
        context_width=POLICY_CONTEXT_WIDTH,
        candidate_width=POLICY_CANDIDATE_WIDTH,
        hidden=POLICY_HIDDEN,
    )
    fresh_optimizer = torch.optim.Adam(
        fresh.parameters(), lr=args.policy_learning_rate
    )
    active_before = _active_digests(system, index, snapshots)
    transfer_curve, curve_bits, curve_updates, curve_lifetimes = _transfer_curve(
        system,
        index,
        snapshots,
        residual_policy,
        fresh,
        residual_optimizer,
        fresh_optimizer,
        args=args,
    )
    policy_bits += curve_bits
    policy_updates += curve_updates
    logical_lifetimes += curve_lifetimes

    probe = _probe_active(
        system,
        index,
        snapshots,
        family=TRANSFER_FAMILY,
        cue_symbol=TRANSFER_CUE,
        batch_size=args.batch_size,
        seed=args.seed + 2_000_000,
        retention_lifetimes=1,
    )
    chosen, oracle, selected_correctly, scores = _select_victim(
        residual_policy, index, snapshots, probe
    )
    logical_lifetimes += args.batch_size * ACTIVE_CACHE_SLOTS
    if chosen is None:
        raise RuntimeError("inherited transfer policy selected no victim")
    displaced = index.archive.active_binding(chosen)
    displaced_snapshot = (
        snapshots[index.artifact_handle(displaced)] if displaced is not None else None
    )
    _restore_snapshot(system, chosen, snapshots[transfer_digest])
    retention_rows: list[dict[str, float | int]] = []

    def verify(_candidate: EpisodicBindingArtifactIndex) -> bool:
        retention_rows.extend(
            _evaluate(
                system,
                family=TRANSFER_FAMILY,
                slot=chosen,
                cue_symbol=TRANSFER_CUE,
                lifetimes=args.retention_lifetimes,
                batch_size=args.batch_size,
                steps=_family_steps(TRANSFER_FAMILY),
                seed=args.seed + 2_100_000,
            )
        )
        return _stable(retention_rows)

    activation = index.reactivate_verified(transfer_id, chosen, verify)
    if not activation.accepted and displaced_snapshot is not None:
        _restore_snapshot(system, chosen, displaced_snapshot)
    post_activation = (
        _evaluate(
            system,
            family=TRANSFER_FAMILY,
            slot=chosen,
            cue_symbol=TRANSFER_CUE,
            lifetimes=args.retention_lifetimes,
            batch_size=args.batch_size,
            steps=_family_steps(TRANSFER_FAMILY),
            seed=args.seed + 2_200_000,
        )
        if activation.accepted
        else []
    )
    verifier_bits += sum(int(row["unique_verifier_bits"]) for row in retention_rows)
    verifier_bits += sum(int(row["unique_verifier_bits"]) for row in post_activation)
    logical_lifetimes += args.batch_size * (
        len(retention_rows) + len(post_activation)
    )

    inherited_correct = [
        int(row["inherited"]["selection_correct"]) for row in transfer_curve
    ]
    fresh_correct = [
        int(row["fresh"]["selection_correct"]) for row in transfer_curve
    ]
    inherited_early = sum(inherited_correct[:4])
    fresh_early = sum(fresh_correct[:4])
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    source_id = index.lookup(_route_key(system, SCHEDULE[0][1])).binding_id
    gates = {
        "source_cohort_mastered": len(direct) == len(SCHEDULE)
        and all(_stable(rows) for rows in direct),
        "related_source_installed": related_installed,
        "held_out_transfer_mastered": _stable(transfer_direct),
        "source_calibration_present": bool(source_calibration),
        "inherited_beats_fresh_early": inherited_early > fresh_early,
        "inherited_reaches_stable_window": _stable_window(inherited_correct),
        "fresh_baseline_measured": len(fresh_correct) == args.transfer_updates,
        "inherited_reactivation_accepted": activation.accepted,
        "held_out_retention_mastery": _stable(post_activation),
        "active_cache_capacity_bounded": len(index.active_binding_ids)
        == ACTIVE_CACHE_SLOTS,
        "protected_source_retained": source_id is not None
        and index.archive.is_protected(source_id),
        "active_files_match_snapshots_before": active_before,
        "active_files_match_snapshots_after": _active_digests(
            system, index, snapshots
        ),
        "selected_transfer_victim": selected_correctly == 1,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": SCHEMA,
        "claim_boundary": (
            "Matched fresh-policy transfer measurement for outcome-trained "
            "external eviction on a held-out n-back-2 family; not unrestricted "
            "memory growth, semantic compression, arbitrary program induction, "
            "or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "source_schedule": [
                {"family": family, "cue": cue} for family, cue in SCHEDULE
            ],
            "held_out_family": TRANSFER_FAMILY,
            "held_out_cue": TRANSFER_CUE,
            "active_cache_slots": ACTIVE_CACHE_SLOTS,
            "policy": "external_capability_eviction_policy_v1",
            "transfer_protocol": "shared_fresh_verifier_outcomes_v1",
            "transfer_policy": "frozen_base_plus_opaque_context_residual_v1",
            "utility_gap_gate": UTILITY_GAP_GATE,
            "fresh_baseline": "same architecture and updates, zero inherited state",
        },
        "direct_source": direct,
        "direct_transfer": transfer_direct,
        "related_source_setup": related_setup,
        "source_calibration": source_calibration,
        "residual": {
            "slot_count": residual_policy.slot_count,
            "active_slots": int(residual_policy.slot_active.sum().item()),
            "frozen_base": True,
        },
        "transfer_curve": transfer_curve,
        "reactivation": {
            "transfer_binding_id": transfer_id,
            "selected_slot": chosen,
            "oracle_slot": oracle,
            "selected_correctly": selected_correctly,
            "scores": scores.tolist(),
            "activation": activation.__dict__,
            "retention_probe": retention_rows,
            "post_activation": post_activation,
        },
        "archive": {
            "record_count": index.record_count,
            "active_binding_ids": list(index.active_binding_ids),
            "transfer_artifact_digest": transfer_digest,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": verifier_bits + policy_bits,
            "policy_verifier_bits": policy_bits,
            "optimizer_updates": optimizer_updates,
            "policy_updates": policy_updates,
            "replayed_examples": 0,
            "unique_logical_lifetimes": logical_lifetimes,
            "inherited_early_correct": inherited_early,
            "fresh_early_correct": fresh_early,
            "inherited_transfer_accuracy": sum(inherited_correct)
            / len(inherited_correct),
            "fresh_transfer_accuracy": sum(fresh_correct) / len(fresh_correct),
            "stable_bits_to_threshold": (
                verifier_bits + policy_bits if all(gates.values()) else None
            ),
            "retention_threshold": MASTERY_THRESHOLD,
            "transfer_ratio_against_fresh_learner": inherited_early
            / max(1, fresh_early),
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_external_compute_eviction_transfer"
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
    parser.add_argument("--file-updates", type=int, default=256)
    parser.add_argument("--policy-calibration-rounds", type=int, default=48)
    parser.add_argument("--policy-updates-per-round", type=int, default=8)
    parser.add_argument("--transfer-updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--retention-lifetimes", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--policy-learning-rate", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
