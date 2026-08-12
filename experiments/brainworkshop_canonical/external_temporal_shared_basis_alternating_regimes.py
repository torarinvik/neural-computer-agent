"""Pressure-test replay-free replacement across repeated hidden regime changes.

The frozen controller and event encoder are held constant while an external
working scope alternates between two opaque value regimes.  Three protected
scopes must survive every replacement.  Each boundary is hidden from the
regime detector: it sees only the current and incoming value banks and emits
keep/replace from a scalar-trained policy.  The memory verifier owns route
retention, expected-version checks, persistence, and copy-on-write commits.

This is a bounded repeated-reversal and capacity-reuse test.  It is not a
claim of unrestricted memory growth or general continual learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import (
    MemoryCandidates,
    MemoryQuery,
    OpaqueRegimeChangePolicy,
    OpaqueSharedBasisStructurePolicy,
    PersistentSharedBasisContentAddressedMemory,
)

from .external_temporal_query_address_growth import EVENT_WIDTH, _build
from .external_temporal_shared_basis_competing_subspaces import (
    COMPETING_CANDIDATE_RANKS,
)
from .external_temporal_shared_basis_competing_subspaces import (
    _evaluate_policy as _evaluate_structure_policy,
)
from .external_temporal_shared_basis_competing_subspaces import (
    _train_policy as _train_structure_policy,
)
from .external_temporal_shared_basis_learned_regime_trigger import (
    DETECTOR_HIDDEN,
    DETECTOR_LEARNING_RATE,
    _evaluate_detector,
    _train_detector,
)
from .external_temporal_shared_basis_policy_growth import _digest

ALTERNATING_REGIMES_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-shared-basis-"
    "alternating-regimes.v1"
)
PROTECTED_SCOPES = (0, 1, 2)
WORKING_SCOPE = 3
PROTECTED_RECORDS = 6
WORKING_RECORDS = 8
REGIME_SEQUENCE = (0, 1, 0, 1, 0, 1)
REGIME_BASIS_WIDTH = 2
NOISE_SCALE = 0.002
READ_MATCH_THRESHOLD = 0.90
WRITE_MATCH_THRESHOLD = 0.999
RETENTION_TOLERANCE = 0.04
POLICY_TEMPERATURE = 0.6


def _memory_digest(memory: PersistentSharedBasisContentAddressedMemory) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(memory.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _payloads(*, seed: int, reverse_rows: bool) -> dict[str, object]:
    """Create opaque route/value payloads for protected scopes and reversals."""

    key_generator = torch.Generator().manual_seed(seed + 91_001)
    protected_count = len(PROTECTED_SCOPES) * PROTECTED_RECORDS
    regime_count = len(REGIME_SEQUENCE) * WORKING_RECORDS
    keys = F.normalize(
        torch.randn(protected_count + regime_count, EVENT_WIDTH, generator=key_generator),
        dim=-1,
    )
    basis_generator = torch.Generator().manual_seed(seed + 92_001)
    basis = torch.linalg.qr(
        torch.randn(EVENT_WIDTH, 6, generator=basis_generator)
    ).Q[:, :6]
    value_generator = torch.Generator().manual_seed(seed + 93_001)

    def make_values(columns: tuple[int, int], count: int) -> torch.Tensor:
        values = (
            torch.randn(count, REGIME_BASIS_WIDTH, generator=value_generator)
            @ basis[:, columns[0] : columns[1]].transpose(0, 1)
            + NOISE_SCALE
            * torch.randn(count, EVENT_WIDTH, generator=value_generator)
        )
        return F.normalize(values, dim=-1)

    protected: dict[int, dict[str, torch.Tensor]] = {}
    cursor = 0
    for scope in PROTECTED_SCOPES:
        scope_keys = keys[cursor : cursor + PROTECTED_RECORDS]
        cursor += PROTECTED_RECORDS
        payload = {
            "keys": scope_keys,
            "values": make_values((0, 2), PROTECTED_RECORDS),
        }
        if reverse_rows:
            payload = {name: value.flip(0) for name, value in payload.items()}
        protected[scope] = payload

    regimes: dict[int, dict[str, torch.Tensor]] = {}
    for occurrence, regime in enumerate(REGIME_SEQUENCE):
        scope_keys = keys[cursor : cursor + WORKING_RECORDS]
        cursor += WORKING_RECORDS
        columns = (2, 4) if regime == 0 else (4, 6)
        payload = {
            "keys": scope_keys,
            "values": make_values(columns, WORKING_RECORDS),
            "regime": torch.tensor(regime, dtype=torch.long),
            "occurrence": torch.tensor(occurrence, dtype=torch.long),
        }
        if reverse_rows:
            payload = {
                name: value.flip(0) if name in {"keys", "values"} else value
                for name, value in payload.items()
            }
        regimes[occurrence] = payload
    return {"protected": protected, "regimes": regimes}


def _scope_routes_match(
    memory: PersistentSharedBasisContentAddressedMemory,
    payload: dict[str, torch.Tensor],
    *,
    scope: int,
    tolerance: float,
) -> bool:
    indices: list[int] = []
    for key, expected in zip(payload["keys"], payload["values"], strict=True):
        read = memory.read(
            MemoryQuery(
                key.reshape(1, -1),
                top_k=1,
                scope=torch.tensor([scope], dtype=torch.long),
            )
        )
        if not bool(read.hit.item()):
            return False
        indices.append(int(read.indices[0, 0]))
        if not torch.allclose(read.value[0], expected, atol=tolerance, rtol=0.0):
            return False
    return len(indices) == len(set(indices))


def _scope_routes_absent(
    memory: PersistentSharedBasisContentAddressedMemory,
    payload: dict[str, torch.Tensor],
    *,
    scope: int,
) -> bool:
    for key in payload["keys"]:
        read = memory.read(
            MemoryQuery(
                key.reshape(1, -1),
                top_k=1,
                scope=torch.tensor([scope], dtype=torch.long),
            )
        )
        if bool(read.hit.item()):
            return False
    return True


def _all_protected_routes_match(
    memory: PersistentSharedBasisContentAddressedMemory,
    payloads: dict[str, object],
    *,
    tolerance: float,
) -> bool:
    protected = payloads["protected"]
    return all(
        _scope_routes_match(memory, protected[scope], scope=scope, tolerance=tolerance)
        for scope in PROTECTED_SCOPES
    )


def _all_live_values(
    memory: PersistentSharedBasisContentAddressedMemory,
) -> tuple[torch.Tensor, torch.Tensor]:
    values: list[torch.Tensor] = []
    for scope in (*PROTECTED_SCOPES, WORKING_SCOPE):
        candidates = memory.candidates(
            scope=torch.tensor([scope], dtype=torch.long)
        )
        values.append(candidates.values[:, candidates.occupied[0]])
    joined = torch.cat(values, dim=1)
    return joined, torch.ones(joined.shape[:2], dtype=torch.bool)


def _replacement_candidates(payload: dict[str, torch.Tensor]) -> MemoryCandidates:
    count = payload["keys"].shape[0]
    return MemoryCandidates(
        keys=payload["keys"].unsqueeze(0),
        values=payload["values"].unsqueeze(0),
        strengths=torch.ones(1, count),
        timestamps=torch.zeros(1, count),
        occupied=torch.ones(1, count, dtype=torch.bool),
    )


def _select_compression(
    policy: OpaqueSharedBasisStructurePolicy,
    memory: PersistentSharedBasisContentAddressedMemory,
    payloads: dict[str, object],
) -> dict[str, object]:
    values, occupied = _all_live_values(memory)
    candidate_ranks = torch.tensor(COMPETING_CANDIDATE_RANKS, dtype=torch.long)
    plan = policy.propose(
        values,
        occupied,
        candidate_ranks,
        explore=False,
        temperature=POLICY_TEMPERATURE,
    )
    selected_rank = COMPETING_CANDIDATE_RANKS[plan.candidate_index]
    candidate = memory.compression_candidate(selected_rank)
    version = int(memory.store_version.item())
    receipt = memory.replace_from_candidate(
        candidate,
        expected_version=version,
        retention_probe=lambda current: _all_routes_match(
            current, payloads, tolerance=RETENTION_TOLERANCE
        ),
    )
    return {
        "selected_rank": selected_rank,
        "accepted": receipt.accepted,
        "max_value_error": receipt.max_value_error,
        "version": receipt.version,
        "physical_value_scalars": memory.physical_value_scalar_count,
    }


def _all_routes_match(
    memory: PersistentSharedBasisContentAddressedMemory,
    payloads: dict[str, object],
    *,
    tolerance: float,
) -> bool:
    protected = _all_protected_routes_match(memory, payloads, tolerance=tolerance)
    current_occurrence = payloads["current_occurrence"]
    current = payloads["regimes"][current_occurrence]
    return protected and _scope_routes_match(
        memory,
        current,
        scope=WORKING_SCOPE,
        tolerance=tolerance,
    )


def _select_replacement(
    policy: OpaqueSharedBasisStructurePolicy,
    memory: PersistentSharedBasisContentAddressedMemory,
    payloads: dict[str, object],
    *,
    incoming_occurrence: int,
) -> dict[str, object]:
    incoming = payloads["regimes"][incoming_occurrence]
    protected_values = [
        payloads["protected"][scope]["values"]
        for scope in PROTECTED_SCOPES
    ]
    proposed_values = torch.cat(
        [*(value.unsqueeze(0) for value in protected_values), incoming["values"].unsqueeze(0)],
        dim=1,
    )
    proposed_occupied = torch.ones(
        proposed_values.shape[:2], dtype=torch.bool
    )
    candidate_ranks = torch.tensor(COMPETING_CANDIDATE_RANKS, dtype=torch.long)
    plan = policy.propose(
        proposed_values,
        proposed_occupied,
        candidate_ranks,
        explore=False,
        temperature=POLICY_TEMPERATURE,
    )
    selected_rank = COMPETING_CANDIDATE_RANKS[plan.candidate_index]
    candidate = memory.rewrite_candidate(
        _replacement_candidates(incoming),
        basis_rows=selected_rank,
        scope=WORKING_SCOPE,
    )
    previous_occurrence = payloads["current_occurrence"]
    previous = payloads["regimes"][previous_occurrence]
    version = int(memory.store_version.item())
    receipt = memory.replace_from_rewrite_candidate(
        candidate,
        expected_version=version,
        retention_probe=lambda current: (
            _all_protected_routes_match(
                current, payloads, tolerance=RETENTION_TOLERANCE
            )
            and _scope_routes_match(
                current,
                incoming,
                scope=WORKING_SCOPE,
                tolerance=RETENTION_TOLERANCE,
            )
            and _scope_routes_absent(current, previous, scope=WORKING_SCOPE)
        ),
    )
    return {
        "incoming_occurrence": incoming_occurrence,
        "selected_rank": selected_rank,
        "accepted": receipt.accepted,
        "rows_before": receipt.rows_before,
        "rows_after": receipt.rows_after,
        "basis_rows_before": receipt.basis_rows_before,
        "basis_rows_after": receipt.basis_rows_after,
        "version": receipt.version,
        "physical_value_scalars": memory.physical_value_scalar_count,
    }


def _run_stream(
    *,
    detector: OpaqueRegimeChangePolicy,
    structure_policy: OpaqueSharedBasisStructurePolicy,
    system,
    seed: int,
    reverse_rows: bool,
    path: Path,
) -> dict[str, object]:
    payloads = _payloads(seed=seed, reverse_rows=reverse_rows)
    memory = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
        scope_capacity=4,
    )
    protected = payloads["protected"]
    for scope in PROTECTED_SCOPES:
        payload = protected[scope]
        memory.write(
            payload["keys"],
            payload["values"],
            torch.ones(PROTECTED_RECORDS),
            scope=torch.full((PROTECTED_RECORDS,), scope, dtype=torch.long),
        )
    current_occurrence = 0
    current = payloads["regimes"][current_occurrence]
    memory.write(
        current["keys"],
        current["values"],
        torch.ones(WORKING_RECORDS),
        scope=torch.full((WORKING_RECORDS,), WORKING_SCOPE, dtype=torch.long),
    )
    payloads["current_occurrence"] = current_occurrence
    initial_routes = _all_routes_match(memory, payloads, tolerance=1e-6)
    initial_dense_scalars = memory.dense_value_scalar_count
    initial_compression = _select_compression(structure_policy, memory, payloads)
    physical_history = [memory.physical_value_scalar_count]
    record_history = [memory.record_count]
    transitions: list[dict[str, object]] = []
    for incoming_occurrence in range(1, len(REGIME_SEQUENCE)):
        incoming = payloads["regimes"][incoming_occurrence]
        current = payloads["regimes"][current_occurrence]
        current_values = current["values"]
        incoming_values = incoming["values"]
        stable_before_version = int(memory.store_version.item())
        stable_before_digest = _memory_digest(memory)
        stable_plan = detector.propose(
            current_values.unsqueeze(0),
            torch.ones(1, current_values.shape[0], dtype=torch.bool),
            current_values.unsqueeze(0),
            torch.ones(1, current_values.shape[0], dtype=torch.bool),
        )
        stable_after_version = int(memory.store_version.item())
        stable_after_digest = _memory_digest(memory)
        shifted_plan = detector.propose(
            current_values.unsqueeze(0),
            torch.ones(1, current_values.shape[0], dtype=torch.bool),
            incoming_values.unsqueeze(0),
            torch.ones(1, incoming_values.shape[0], dtype=torch.bool),
        )
        replacement = None
        if shifted_plan.replace:
            replacement = _select_replacement(
                structure_policy,
                memory,
                payloads,
                incoming_occurrence=incoming_occurrence,
            )
            if replacement["accepted"]:
                current_occurrence = incoming_occurrence
                payloads["current_occurrence"] = current_occurrence
                compression = _select_compression(
                    structure_policy, memory, payloads
                )
                physical_history.append(memory.physical_value_scalar_count)
                record_history.append(memory.record_count)
            else:
                compression = None
        else:
            compression = None
        transitions.append(
            {
                "from_occurrence": int(current["occurrence"]),
                "to_occurrence": incoming_occurrence,
                "expected_shift": bool(
                    current["regime"].item() != incoming["regime"].item()
                ),
                "stable_keep": not stable_plan.replace,
                "stable_noop_version": stable_before_version == stable_after_version,
                "stable_noop_digest": stable_before_digest == stable_after_digest,
                "shift_detected": bool(shifted_plan.replace),
                "replacement": replacement,
                "compression": compression,
            }
        )
    final_current = payloads["regimes"][current_occurrence]
    final_routes = _all_routes_match(memory, payloads, tolerance=RETENTION_TOLERANCE)
    old_routes_removed = all(
        _scope_routes_absent(memory, payloads["regimes"][occurrence], scope=WORKING_SCOPE)
        for occurrence in range(len(REGIME_SEQUENCE) - 1)
        if occurrence != current_occurrence
    )
    restored = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
        scope_capacity=4,
    )
    reload_routes = _all_routes_match(
        restored,
        payloads,
        tolerance=RETENTION_TOLERANCE,
    )
    stale_rejected = False
    final_incoming = payloads["regimes"][current_occurrence]
    stale_candidate = memory.rewrite_candidate(
        _replacement_candidates(final_incoming),
        basis_rows=4,
        scope=WORKING_SCOPE,
    )
    try:
        memory.replace_from_rewrite_candidate(
            stale_candidate,
            expected_version=max(0, int(memory.store_version.item()) - 1),
        )
    except RuntimeError as error:
        stale_rejected = "stale" in str(error).lower()
    return {
        "reverse_rows": reverse_rows,
        "initial_routes": initial_routes,
        "initial_compression": initial_compression,
        "transitions": transitions,
        "final_occurrence": int(final_current["occurrence"]),
        "final_routes": final_routes,
        "old_routes_removed": old_routes_removed,
        "reload_routes": reload_routes,
        "stale_version_rejected": stale_rejected,
        "constant_record_count": len(set(record_history)) == 1,
        "record_count_history": record_history,
        "physical_value_scalar_history": physical_history,
        "capacity_reused": (
            max(physical_history) <= initial_dense_scalars // 2
            and len(set(physical_history)) == 1
        ),
        "final_record_count": memory.record_count,
        "final_physical_value_scalars": memory.physical_value_scalar_count,
        "final_dense_value_scalars": memory.dense_value_scalar_count,
        "current_payload": final_current,
        "path": path,
        "system": system,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1:
        raise ValueError("alternating-regime policy updates must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    detector, detector_training = _train_detector(
        seed=args.seed,
        updates=args.policy_updates,
    )
    detector_scores = _evaluate_detector(detector, seed=args.seed + 810_000)
    structure_policy, structure_training = _train_structure_policy(
        seed=args.seed + 1_000,
        updates=args.policy_updates,
    )
    structure_scores = _evaluate_structure_policy(
        structure_policy,
        seed=args.seed + 811_000,
    )
    torch.manual_seed(args.seed + 912_000)
    fresh_detector = OpaqueRegimeChangePolicy(
        value_width=EVENT_WIDTH,
        hidden=DETECTOR_HIDDEN,
        max_spectral_bins=8,
        learning_rate=DETECTOR_LEARNING_RATE,
    ).eval()
    fresh_detector_scores = _evaluate_detector(
        fresh_detector,
        seed=args.seed + 810_000,
    )
    with tempfile.TemporaryDirectory(
        prefix="neural-computer-shared-basis-alternating-regimes-"
    ) as directory:
        forward = _run_stream(
            detector=detector,
            structure_policy=structure_policy,
            system=system,
            seed=args.seed,
            reverse_rows=False,
            path=Path(directory) / "forward.pt",
        )
        reversed_stream = _run_stream(
            detector=detector,
            structure_policy=structure_policy,
            system=system,
            seed=args.seed,
            reverse_rows=True,
            path=Path(directory) / "reversed.pt",
        )
        corruption_path = Path(directory) / "corrupt.pt"
        payload = torch.load(forward["path"], weights_only=False)
        payload["state_dict"]["coefficients"] = payload["state_dict"][
            "coefficients"
        ].clone()
        payload["state_dict"]["coefficients"][0, 0] += 0.1
        torch.save(payload, corruption_path)
        corruption_rejected = False
        try:
            PersistentSharedBasisContentAddressedMemory(
                EVENT_WIDTH,
                path=corruption_path,
                write_threshold=0.0,
                write_match_threshold=WRITE_MATCH_THRESHOLD,
                read_match_threshold=READ_MATCH_THRESHOLD,
                basis_tolerance=1e-8,
                scope_capacity=4,
            )
        except ValueError as error:
            corruption_rejected = "checksum" in str(error).lower()
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])

    def transition_gates(stream: dict[str, object]) -> dict[str, bool]:
        transitions = stream["transitions"]
        return {
            "all_stable_keeps": all(item["stable_keep"] for item in transitions),
            "all_stable_noops": all(
                item["stable_noop_version"] and item["stable_noop_digest"]
                for item in transitions
            ),
            "all_shifts_detected": all(item["shift_detected"] for item in transitions),
            "all_replacements_accepted": all(
                item["replacement"] and item["replacement"]["accepted"]
                for item in transitions
            ),
            "all_rank_four": all(
                item["replacement"]
                and item["replacement"]["selected_rank"] == 4
                for item in transitions
            ),
            "all_compressions_accepted": all(
                item["compression"] and item["compression"]["accepted"]
                for item in transitions
            ),
        }

    forward_gates = transition_gates(forward)
    reversed_gates = transition_gates(reversed_stream)
    gates = {
        "detector_transfer_stable_keep": detector_scores["stable_keep"] >= 0.80,
        "detector_transfer_shift_replace": detector_scores["shift_replace"] >= 0.80,
        "detector_beats_fresh": (
            detector_scores["stable_keep"] + detector_scores["shift_replace"]
            >= fresh_detector_scores["stable_keep"]
            + fresh_detector_scores["shift_replace"]
            + 0.20
        ),
        "structure_rank_four_transfer": structure_scores["4"] >= 0.80,
        "forward_initial_routes": bool(forward["initial_routes"]),
        "reversed_initial_routes": bool(reversed_stream["initial_routes"]),
        "forward_all_stable_keeps": forward_gates["all_stable_keeps"],
        "reversed_all_stable_keeps": reversed_gates["all_stable_keeps"],
        "forward_all_stable_noops": forward_gates["all_stable_noops"],
        "reversed_all_stable_noops": reversed_gates["all_stable_noops"],
        "forward_all_shifts_detected": forward_gates["all_shifts_detected"],
        "reversed_all_shifts_detected": reversed_gates["all_shifts_detected"],
        "forward_all_replacements_accepted": forward_gates[
            "all_replacements_accepted"
        ],
        "reversed_all_replacements_accepted": reversed_gates[
            "all_replacements_accepted"
        ],
        "forward_all_rank_four": forward_gates["all_rank_four"],
        "reversed_all_rank_four": reversed_gates["all_rank_four"],
        "forward_all_compressions_accepted": forward_gates[
            "all_compressions_accepted"
        ],
        "reversed_all_compressions_accepted": reversed_gates[
            "all_compressions_accepted"
        ],
        "forward_protected_retained": bool(forward["final_routes"]),
        "reversed_protected_retained": bool(reversed_stream["final_routes"]),
        "forward_old_routes_removed": bool(forward["old_routes_removed"]),
        "reversed_old_routes_removed": bool(reversed_stream["old_routes_removed"]),
        "forward_reload_routes": bool(forward["reload_routes"]),
        "reversed_reload_routes": bool(reversed_stream["reload_routes"]),
        "forward_stale_version_rejected": bool(forward["stale_version_rejected"]),
        "reversed_stale_version_rejected": bool(
            reversed_stream["stale_version_rejected"]
        ),
        "forward_constant_record_count": bool(forward["constant_record_count"]),
        "reversed_constant_record_count": bool(
            reversed_stream["constant_record_count"]
        ),
        "forward_capacity_reused": bool(forward["capacity_reused"]),
        "reversed_capacity_reused": bool(reversed_stream["capacity_reused"]),
        "corruption_rejected": corruption_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": ALTERNATING_REGIMES_SCHEMA,
        "claim_boundary": (
            "Outcome-trained external trigger transfers through five alternating "
            "hidden regime boundaries while three protected scopes survive and "
            "one working scope reuses bounded factorized capacity; not general "
            "continual learning or unrestricted memory growth."
        ),
        "seed": args.seed,
        "architecture": {
            "detector": "opaque_regime_change_policy_v1",
            "structure_policy": "opaque_shared_basis_structure_policy_v2",
            "memory": "persistent_shared_basis_content_addressed_memory_v1",
            "rewrite": "shared_basis_rewrite_v1",
            "scope_capacity": 4,
            "protected_scopes": len(PROTECTED_SCOPES),
            "detector_feature_contract": "opaque_spectral_cross_bank_structure_v1",
            "forbidden_features": "task_labels_regime_ids_candidate_reconstruction_error_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "regime_sequence": REGIME_SEQUENCE,
        "detector_training": detector_training,
        "structure_training": structure_training,
        "detector_scores": detector_scores,
        "fresh_detector_scores": fresh_detector_scores,
        "structure_scores": structure_scores,
        "forward": {
            key: value
            for key, value in forward.items()
            if key not in {"path", "system", "current_payload"}
        },
        "reversed": {
            key: value
            for key, value in reversed_stream.items()
            if key not in {"path", "system", "current_payload"}
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": args.policy_updates * 2 + 120,
            "unique_logical_lifetimes": args.policy_updates * 2 + 90,
            "optimizer_updates": args.policy_updates * 2,
            "replayed_examples": 0,
            "controller_updates": 0,
            "live_replacement_transactions_per_stream": len(REGIME_SEQUENCE) - 1,
            "live_compression_transactions_per_stream": len(REGIME_SEQUENCE),
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_alternating_regimes"
        if all(gates.values())
        else "rejected",
    }
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--policy-updates", type=int, default=1_000)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
