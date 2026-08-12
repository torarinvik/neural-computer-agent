"""Pressure-test verifier-gated regime replacement in external memory.

The memory has two isolated scopes: a protected source capability and a
replaceable working set.  The working set first contains two incompatible
rank-two subspaces, then is replaced by a new regime in one rank-two
subspace.  The external policy therefore sees global structure change from
rank six to rank four and proposes ``8 -> 4`` while the protected source route
survives and the stale working routes disappear.

The controller and event encoder are frozen.  Replacement is copy-on-write,
version checked, retention verified, and atomically persisted.  This is a
bounded regime-replacement test, not unrestricted continual learning.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import (
    MemoryCandidates,
    MemoryQuery,
    OpaqueSharedBasisStructurePolicy,
    PersistentSharedBasisContentAddressedMemory,
)

from .external_temporal_query_address_growth import EVENT_WIDTH, _build
from .external_temporal_shared_basis_competing_subspaces import (
    COMPETING_CANDIDATE_RANKS,
    _evaluate_policy,
    _train_policy,
)
from .external_temporal_shared_basis_policy_growth import RETENTION_TOLERANCE, _digest

REGIME_REPLACEMENT_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-shared-basis-regime-replacement.v1"
)
PROTECTED_RECORDS = 6
WORKING_RECORDS = 12
TOTAL_RECORDS = PROTECTED_RECORDS + WORKING_RECORDS * 2
NOISE_SCALE = 0.002
READ_MATCH_THRESHOLD = 0.90
WRITE_MATCH_THRESHOLD = 0.999
PROTECTED_SCOPE = 0
WORKING_SCOPE = 1


def _payloads(
    system,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    key_generator = torch.Generator().manual_seed(seed + 75_001)
    # Independent learned-address surrogates make replacement observable:
    # an old opaque address must not accidentally retrieve a new regime row.
    keys = F.normalize(
        torch.randn(TOTAL_RECORDS, EVENT_WIDTH, generator=key_generator), dim=-1
    )
    basis_generator = torch.Generator().manual_seed(seed + 76_001)
    basis = torch.linalg.qr(
        torch.randn(EVENT_WIDTH, 8, generator=basis_generator)
    ).Q[:, :8]
    value_generator = torch.Generator().manual_seed(seed + 77_001)

    def make_values(columns: tuple[int, int], count: int) -> torch.Tensor:
        values = (
            torch.randn(count, 2, generator=value_generator)
            @ basis[:, columns[0] : columns[1]].transpose(0, 1)
            + NOISE_SCALE
            * torch.randn(count, EVENT_WIDTH, generator=value_generator)
        )
        return F.normalize(values, dim=-1)

    protected_values = make_values((0, 2), PROTECTED_RECORDS)
    old_working_values = torch.cat(
        (make_values((2, 4), 6), make_values((4, 6), 6))
    )
    new_working_values = make_values((6, 8), WORKING_RECORDS)
    return {
        "protected_keys": keys[:PROTECTED_RECORDS],
        "protected_values": protected_values,
        "old_working_keys": keys[PROTECTED_RECORDS : PROTECTED_RECORDS + WORKING_RECORDS],
        "old_working_values": old_working_values,
        "new_working_keys": keys[PROTECTED_RECORDS + WORKING_RECORDS :],
        "new_working_values": new_working_values,
    }


def _scoped_routes_match(
    memory: PersistentSharedBasisContentAddressedMemory,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    scope: int,
    tolerance: float,
) -> bool:
    indices: list[int] = []
    for key, expected in zip(keys, values, strict=True):
        query = MemoryQuery(
            key.reshape(1, -1),
            top_k=1,
            scope=torch.tensor([scope], dtype=torch.long),
        )
        read = memory.read(query)
        if not bool(read.hit.item()):
            return False
        indices.append(int(read.indices[0, 0]))
        if not torch.allclose(read.value[0], expected, atol=tolerance, rtol=0.0):
            return False
    return len(indices) == len(set(indices))


def _scoped_routes_absent(
    memory: PersistentSharedBasisContentAddressedMemory,
    keys: torch.Tensor,
    *,
    scope: int,
) -> bool:
    for key in keys:
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


def _all_values_for_policy(
    memory: PersistentSharedBasisContentAddressedMemory,
) -> tuple[torch.Tensor, torch.Tensor]:
    protected = memory.candidates(
        scope=torch.tensor([PROTECTED_SCOPE], dtype=torch.long)
    )
    working = memory.candidates(
        scope=torch.tensor([WORKING_SCOPE], dtype=torch.long)
    )
    protected_values = protected.values[:, protected.occupied[0]]
    working_values = working.values[:, working.occupied[0]]
    values = torch.cat((protected_values, working_values), dim=1)
    occupied = torch.ones(
        values.shape[:2], dtype=torch.bool, device=values.device
    )
    return values, occupied


def _replacement_candidates(
    keys: torch.Tensor,
    values: torch.Tensor,
) -> MemoryCandidates:
    count = keys.shape[0]
    return MemoryCandidates(
        keys=keys.unsqueeze(0),
        values=values.unsqueeze(0),
        strengths=torch.ones(1, count),
        timestamps=torch.zeros(1, count),
        occupied=torch.ones(1, count, dtype=torch.bool),
    )


def _select_compression(
    policy: OpaqueSharedBasisStructurePolicy,
    memory: PersistentSharedBasisContentAddressedMemory,
    payloads: dict[str, torch.Tensor],
) -> dict[str, object]:
    values, occupied = _all_values_for_policy(memory)
    rank_tensor = torch.tensor(COMPETING_CANDIDATE_RANKS, dtype=torch.long)
    plan = policy.propose(values, occupied, rank_tensor)
    selected_rank = COMPETING_CANDIDATE_RANKS[plan.candidate_index]
    candidate = memory.compression_candidate(selected_rank)
    version = int(memory.store_version.item())
    receipt = memory.replace_from_candidate(
        candidate,
        expected_version=version,
        retention_probe=lambda current: (
            _scoped_routes_match(
                current,
                payloads["protected_keys"],
                payloads["protected_values"],
                scope=PROTECTED_SCOPE,
                tolerance=RETENTION_TOLERANCE,
            )
            and _scoped_routes_match(
                current,
                payloads["old_working_keys"],
                payloads["old_working_values"],
                scope=WORKING_SCOPE,
                tolerance=RETENTION_TOLERANCE,
            )
        ),
    )
    return {
        "selected_rank": selected_rank,
        "candidate_error": memory.max_value_error(candidate),
        "accepted": receipt.accepted,
        "basis_rows_before": receipt.basis_rows_before,
        "basis_rows_after": receipt.basis_rows_after,
        "version": receipt.version,
        "physical_value_scalars_after": memory.physical_value_scalar_count,
    }


def _select_replacement(
    policy: OpaqueSharedBasisStructurePolicy,
    memory: PersistentSharedBasisContentAddressedMemory,
    payloads: dict[str, torch.Tensor],
) -> dict[str, object]:
    values, _occupied = _all_values_for_policy(memory)
    proposed_values = torch.cat(
        (
            payloads["protected_values"].unsqueeze(0),
            payloads["new_working_values"].unsqueeze(0),
        ),
        dim=1,
    )
    proposed_occupied = torch.ones(
        proposed_values.shape[:2], dtype=torch.bool
    )
    rank_tensor = torch.tensor(COMPETING_CANDIDATE_RANKS, dtype=torch.long)
    plan = policy.propose(proposed_values, proposed_occupied, rank_tensor)
    selected_rank = COMPETING_CANDIDATE_RANKS[plan.candidate_index]
    replacement = _replacement_candidates(
        payloads["new_working_keys"],
        payloads["new_working_values"],
    )
    candidate = memory.rewrite_candidate(
        replacement,
        basis_rows=selected_rank,
        scope=WORKING_SCOPE,
    )
    version = int(memory.store_version.item())
    receipt = memory.replace_from_rewrite_candidate(
        candidate,
        expected_version=version,
        retention_probe=lambda current: (
            _scoped_routes_match(
                current,
                payloads["protected_keys"],
                payloads["protected_values"],
                scope=PROTECTED_SCOPE,
                tolerance=RETENTION_TOLERANCE,
            )
            and _scoped_routes_match(
                current,
                payloads["new_working_keys"],
                payloads["new_working_values"],
                scope=WORKING_SCOPE,
                tolerance=RETENTION_TOLERANCE,
            )
            and _scoped_routes_absent(
                current,
                payloads["old_working_keys"],
                scope=WORKING_SCOPE,
            )
        ),
    )
    return {
        "selected_rank": selected_rank,
        "accepted": receipt.accepted,
        "rows_before": receipt.rows_before,
        "rows_after": receipt.rows_after,
        "basis_rows_before": receipt.basis_rows_before,
        "basis_rows_after": receipt.basis_rows_after,
        "version": receipt.version,
        "physical_value_scalars_after": memory.physical_value_scalar_count,
        "policy_observed_current_record_count": int(values.shape[1]),
    }


def _run_stream(
    *,
    policy: OpaqueSharedBasisStructurePolicy,
    system,
    seed: int,
    path: Path,
) -> dict[str, object]:
    payloads = _payloads(system, seed=seed)
    memory = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
        scope_capacity=2,
    )
    memory.write(
        payloads["protected_keys"],
        payloads["protected_values"],
        torch.ones(PROTECTED_RECORDS),
        scope=torch.full((PROTECTED_RECORDS,), PROTECTED_SCOPE, dtype=torch.long),
    )
    memory.write(
        payloads["old_working_keys"],
        payloads["old_working_values"],
        torch.ones(WORKING_RECORDS),
        scope=torch.full((WORKING_RECORDS,), WORKING_SCOPE, dtype=torch.long),
    )
    initial_protected = _scoped_routes_match(
        memory,
        payloads["protected_keys"],
        payloads["protected_values"],
        scope=PROTECTED_SCOPE,
        tolerance=1e-6,
    )
    initial_working = _scoped_routes_match(
        memory,
        payloads["old_working_keys"],
        payloads["old_working_values"],
        scope=WORKING_SCOPE,
        tolerance=1e-6,
    )
    compression = _select_compression(policy, memory, payloads)
    compressed_protected = _scoped_routes_match(
        memory,
        payloads["protected_keys"],
        payloads["protected_values"],
        scope=PROTECTED_SCOPE,
        tolerance=RETENTION_TOLERANCE,
    )
    compressed_working = _scoped_routes_match(
        memory,
        payloads["old_working_keys"],
        payloads["old_working_values"],
        scope=WORKING_SCOPE,
        tolerance=RETENTION_TOLERANCE,
    )
    replacement = _select_replacement(policy, memory, payloads)
    protected_after = _scoped_routes_match(
        memory,
        payloads["protected_keys"],
        payloads["protected_values"],
        scope=PROTECTED_SCOPE,
        tolerance=RETENTION_TOLERANCE,
    )
    new_working_after = _scoped_routes_match(
        memory,
        payloads["new_working_keys"],
        payloads["new_working_values"],
        scope=WORKING_SCOPE,
        tolerance=RETENTION_TOLERANCE,
    )
    old_working_absent = _scoped_routes_absent(
        memory,
        payloads["old_working_keys"],
        scope=WORKING_SCOPE,
    )
    stale_rejected = False
    try:
        replacement_candidate = memory.rewrite_candidate(
            _replacement_candidates(
                payloads["new_working_keys"],
                payloads["new_working_values"],
            ),
            basis_rows=4,
            scope=WORKING_SCOPE,
        )
        memory.replace_from_rewrite_candidate(
            replacement_candidate,
            expected_version=1,
        )
    except RuntimeError as error:
        stale_rejected = "stale" in str(error).lower()
    restored = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
        scope_capacity=2,
    )
    reload_protected = _scoped_routes_match(
        restored,
        payloads["protected_keys"],
        payloads["protected_values"],
        scope=PROTECTED_SCOPE,
        tolerance=RETENTION_TOLERANCE,
    )
    reload_new = _scoped_routes_match(
        restored,
        payloads["new_working_keys"],
        payloads["new_working_values"],
        scope=WORKING_SCOPE,
        tolerance=RETENTION_TOLERANCE,
    )
    return {
        "initial_protected": initial_protected,
        "initial_working": initial_working,
        "compression": compression,
        "compressed_protected": compressed_protected,
        "compressed_working": compressed_working,
        "replacement": replacement,
        "protected_after": protected_after,
        "new_working_after": new_working_after,
        "old_working_absent": old_working_absent,
        "reload_protected": reload_protected,
        "reload_new_working": reload_new,
        "stale_version_rejected": stale_rejected,
        "final_record_count": memory.record_count,
        "final_scope_zero_count": int(
            memory.candidates(scope=torch.tensor([0], dtype=torch.long)).occupied.sum()
        ),
        "final_scope_one_count": int(
            memory.candidates(scope=torch.tensor([1], dtype=torch.long)).occupied.sum()
        ),
        "final_physical_value_scalars": memory.physical_value_scalar_count,
        "final_dense_value_scalars": memory.dense_value_scalar_count,
        "path": path,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1:
        raise ValueError("regime-replacement policy updates must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    policy, training = _train_policy(seed=args.seed, updates=args.policy_updates)
    trained_scores = _evaluate_policy(policy, seed=args.seed + 800_000)
    torch.manual_seed(args.seed + 900_000)
    fresh_policy = OpaqueSharedBasisStructurePolicy(
        value_width=EVENT_WIDTH,
        hidden=128,
        max_spectral_bins=8,
        learning_rate=0.002,
    ).eval()
    fresh_scores = _evaluate_policy(fresh_policy, seed=args.seed + 800_000)
    with tempfile.TemporaryDirectory(
        prefix="neural-computer-shared-basis-regime-replacement-"
    ) as directory:
        forward = _run_stream(
            policy=policy,
            system=system,
            seed=args.seed,
            path=Path(directory) / "forward.pt",
        )
        reversed_stream = _run_stream(
            policy=policy,
            system=system,
            seed=args.seed + 100,
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
                scope_capacity=2,
            )
        except ValueError as error:
            corruption_rejected = "checksum" in str(error).lower()
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "trained_beats_fresh": (
            trained_scores["2"] >= fresh_scores["2"]
            and trained_scores["4"] >= fresh_scores["4"] + 0.10
            and trained_scores["8"] >= fresh_scores["8"] + 0.10
        ),
        "trained_heldout_rank_2": trained_scores["2"] >= 0.80,
        "trained_heldout_rank_4": trained_scores["4"] >= 0.80,
        "trained_heldout_rank_8": trained_scores["8"] >= 0.80,
        "forward_initial_routes": bool(
            forward["initial_protected"] and forward["initial_working"]
        ),
        "reversed_initial_routes": bool(
            reversed_stream["initial_protected"]
            and reversed_stream["initial_working"]
        ),
        "forward_compression_rank_8": forward["compression"]["selected_rank"] == 8,
        "reversed_compression_rank_8": reversed_stream["compression"]["selected_rank"] == 8,
        "forward_compression_accepted": bool(forward["compression"]["accepted"]),
        "reversed_compression_accepted": bool(reversed_stream["compression"]["accepted"]),
        "forward_replacement_rank_4": forward["replacement"]["selected_rank"] == 4,
        "reversed_replacement_rank_4": reversed_stream["replacement"]["selected_rank"] == 4,
        "forward_replacement_accepted": bool(forward["replacement"]["accepted"]),
        "reversed_replacement_accepted": bool(reversed_stream["replacement"]["accepted"]),
        "forward_protected_retained": bool(forward["protected_after"]),
        "reversed_protected_retained": bool(reversed_stream["protected_after"]),
        "forward_new_working_admitted": bool(forward["new_working_after"]),
        "reversed_new_working_admitted": bool(reversed_stream["new_working_after"]),
        "forward_old_working_removed": bool(forward["old_working_absent"]),
        "reversed_old_working_removed": bool(reversed_stream["old_working_absent"]),
        "forward_reload_routes": bool(
            forward["reload_protected"] and forward["reload_new_working"]
        ),
        "reversed_reload_routes": bool(
            reversed_stream["reload_protected"]
            and reversed_stream["reload_new_working"]
        ),
        "forward_stale_version_rejected": bool(
            forward["stale_version_rejected"]
        ),
        "reversed_stale_version_rejected": bool(
            reversed_stream["stale_version_rejected"]
        ),
        "corruption_rejected": corruption_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": REGIME_REPLACEMENT_SCHEMA,
        "claim_boundary": (
            "Outcome-trained dynamic-rank external memory representation that "
            "performs verifier-gated replacement of a replaceable working scope "
            "while retaining a protected scope; not unrestricted semantic "
            "regime discovery, arbitrary new computation, or general continual "
            "learning."
        ),
        "seed": args.seed,
        "architecture": {
            "policy": "opaque_shared_basis_structure_policy_v2",
            "memory": "persistent_shared_basis_content_addressed_memory_v1",
            "memory_rewrite": "shared_basis_rewrite_v1",
            "candidate_ranks": COMPETING_CANDIDATE_RANKS,
            "scopes": {"protected": PROTECTED_SCOPE, "replaceable": WORKING_SCOPE},
            "stream": "protected_rank_two_plus_old_rank_six_to_new_rank_two",
            "forbidden_features": "precomputed_candidate_reconstruction_error_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
        },
        "training": training,
        "trained_scores": trained_scores,
        "fresh_scores": fresh_scores,
        "forward": {key: value for key, value in forward.items() if key != "path"},
        "reversed": {
            key: value for key, value in reversed_stream.items() if key != "path"
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": args.policy_updates + TOTAL_RECORDS * 4,
            "unique_logical_lifetimes": args.policy_updates + TOTAL_RECORDS * 2,
            "optimizer_updates": args.policy_updates,
            "live_compression_transactions": 4,
            "live_logical_rewrite_transactions": 2,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_regime_replacement"
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
    parser.add_argument("--policy-updates", type=int, default=5_000)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
