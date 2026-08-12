"""Learn shared structure from opaque values without candidate-error features.

This pressure test removes the previous compression policy's precomputed
reconstruction-error shortcut.  An external permutation-invariant structure
policy receives only the current opaque value rows and occupancy mask.  It
computes a fixed-width singular-spectrum summary, learns a generic candidate
preference from scalar verifier utility, and proposes a rank.  The memory
backend's independent route/value verifier still authorizes every commit.

The live stream repeats the old six-record rank-two cohort followed by a new
six-record rank-four cohort.  The controller, event encoder, and logical keys
remain frozen; old routes must survive successor growth without replay.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import (
    OpaqueSharedBasisStructurePolicy,
    PersistentSharedBasisContentAddressedMemory,
)

from .external_temporal_query_address_growth import EVENT_WIDTH, _build
from .external_temporal_shared_basis_policy_growth import (
    CANDIDATE_RANKS,
    INITIAL_RECORDS,
    NEW_RECORDS,
    RETENTION_TOLERANCE,
    _digest,
    _live_payloads,
    _minimal_safe_index,
    _route_verifier,
    _synthetic_values,
)

SHARED_BASIS_STRUCTURE_GROWTH_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-shared-basis-structure-growth.v1"
)
FEATURE_WIDTH = EVENT_WIDTH
POLICY_HIDDEN = 128
POLICY_LEARNING_RATE = 0.002
POLICY_TEMPERATURE = 0.6
READ_MATCH_THRESHOLD = 0.75
WRITE_MATCH_THRESHOLD = 0.999


def _target_index(values: torch.Tensor) -> int:
    """Private verifier target; never passed to the structure policy."""

    _left, _singular, right = torch.linalg.svd(values, full_matrices=False)
    errors: list[float] = []
    physical_sizes: list[int] = []
    for rank in CANDIDATE_RANKS:
        basis = right[:rank]
        errors.append(float((values - values @ basis.transpose(0, 1) @ basis).abs().max()))
        physical_sizes.append(rank * values.shape[1] + values.shape[0] * rank)
    return _minimal_safe_index(
        tuple(errors),
        tuple(physical_sizes),
        dense_scalars=values.shape[0] * values.shape[1],
    )


def _train_policy(
    *,
    seed: int,
    updates: int,
) -> tuple[OpaqueSharedBasisStructurePolicy, dict[str, float | int]]:
    if updates < 1:
        raise ValueError("shared-basis structure policy updates must be positive")
    torch.manual_seed(seed)
    policy = OpaqueSharedBasisStructurePolicy(
        value_width=EVENT_WIDTH,
        hidden=POLICY_HIDDEN,
        max_spectral_bins=8,
        learning_rate=POLICY_LEARNING_RATE,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=POLICY_LEARNING_RATE)
    candidate_ranks = torch.tensor(CANDIDATE_RANKS, dtype=torch.long)
    explorer = torch.Generator().manual_seed(seed + 50_001)
    utilities: list[float] = []
    for update in range(updates):
        values, _target_rank = _synthetic_values(seed=seed + 10_000 + update)
        occupied = torch.ones(1, values.shape[0], dtype=torch.bool)
        plan = policy.propose(
            values.unsqueeze(0),
            occupied,
            candidate_ranks,
            explore=True,
            temperature=POLICY_TEMPERATURE,
            generator=explorer,
        )
        utility = float(plan.candidate_index == _target_index(values))
        policy.adaptation_step(
            values.unsqueeze(0),
            occupied,
            candidate_ranks,
            plan,
            utility,
            optimizer=optimizer,
        )
        utilities.append(utility)
    policy.eval()
    return policy, {
        "optimizer_updates": updates,
        "unique_scalar_utilities": updates,
        "first_window_utility": sum(utilities[:100]) / min(100, len(utilities)),
        "last_window_utility": sum(utilities[-100:]) / min(100, len(utilities)),
    }


@torch.no_grad()
def _evaluate_policy(
    policy: OpaqueSharedBasisStructurePolicy,
    *,
    seed: int,
    episodes_per_rank: int = 64,
) -> dict[str, float]:
    candidate_ranks = torch.tensor(CANDIDATE_RANKS, dtype=torch.long)
    scores: dict[str, list[float]] = {str(rank): [] for rank in CANDIDATE_RANKS}
    for rank_index, rank in enumerate(CANDIDATE_RANKS):
        for episode in range(episodes_per_rank):
            values, _ = _synthetic_values(
                seed=seed + rank_index * 10_000 + episode,
                target_rank=rank,
            )
            occupied = torch.ones(1, values.shape[0], dtype=torch.bool)
            plan = policy.propose(
                values.unsqueeze(0), occupied, candidate_ranks
            )
            scores[str(rank)].append(
                float(plan.candidate_index == _target_index(values))
            )
    return {rank: sum(values) / len(values) for rank, values in scores.items()}


def _select_and_commit(
    policy: OpaqueSharedBasisStructurePolicy,
    memory: PersistentSharedBasisContentAddressedMemory,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> dict[str, object]:
    candidates = memory.candidates()
    basis_before = memory.basis_count
    candidate_ranks = torch.tensor(
        CANDIDATE_RANKS,
        dtype=torch.long,
        device=candidates.values.device,
    )
    plan = policy.propose(
        candidates.values,
        candidates.occupied,
        candidate_ranks,
    )
    selected_rank = CANDIDATE_RANKS[plan.candidate_index]
    selected_candidate = memory.compression_candidate(selected_rank)
    candidate_error = memory.max_value_error(selected_candidate)
    version = int(memory.store_version.item())
    receipt = memory.replace_from_candidate(
        selected_candidate,
        expected_version=version,
        retention_probe=lambda candidate: _route_verifier(
            candidate,
            keys,
            values,
            tolerance=RETENTION_TOLERANCE,
        ),
    )
    return {
        "selected_rank": selected_rank,
        "candidate_index": plan.candidate_index,
        "candidate_error": candidate_error,
        "accepted": receipt.accepted,
        "basis_rows_before": basis_before,
        "basis_rows_after": memory.basis_count,
        "physical_value_scalars_after": memory.physical_value_scalar_count,
        "version": int(memory.store_version.item()),
    }


def _run_live_stream(
    *,
    policy: OpaqueSharedBasisStructurePolicy,
    system,
    seed: int,
    path: Path,
    reversed_order: bool,
) -> dict[str, object]:
    keys, initial_values, successor_values = _live_payloads(system, seed=seed)
    memory = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
    )
    initial_order = (
        torch.arange(INITIAL_RECORDS - 1, -1, -1)
        if reversed_order
        else torch.arange(INITIAL_RECORDS)
    )
    memory.write(
        keys[:INITIAL_RECORDS][initial_order],
        initial_values[initial_order],
        torch.ones(INITIAL_RECORDS),
    )
    initial_before = _route_verifier(
        memory, keys[:INITIAL_RECORDS], initial_values, tolerance=1e-6
    )
    initial_selection = _select_and_commit(
        policy, memory, keys[:INITIAL_RECORDS], initial_values
    )
    initial_after = _route_verifier(
        memory,
        keys[:INITIAL_RECORDS],
        initial_values,
        tolerance=RETENTION_TOLERANCE,
    )
    successor_order = (
        torch.arange(NEW_RECORDS - 1, -1, -1)
        if reversed_order
        else torch.arange(NEW_RECORDS)
    )
    memory.write(
        keys[INITIAL_RECORDS:][successor_order],
        successor_values[successor_order],
        torch.ones(NEW_RECORDS),
    )
    basis_before_successor = memory.basis_count
    all_values = torch.cat((initial_values, successor_values))
    successor_selection = _select_and_commit(policy, memory, keys, all_values)
    old_after_growth = _route_verifier(
        memory,
        keys[:INITIAL_RECORDS],
        initial_values,
        tolerance=RETENTION_TOLERANCE,
    )
    new_after_growth = _route_verifier(
        memory,
        keys[INITIAL_RECORDS:],
        successor_values,
        tolerance=RETENTION_TOLERANCE,
    )
    stale_rejected = False
    try:
        memory.replace_from_candidate(
            memory.compression_candidate(CANDIDATE_RANKS[-1]),
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
    )
    reload_routes = _route_verifier(
        restored,
        keys,
        all_values,
        tolerance=RETENTION_TOLERANCE,
    )
    return {
        "physical_order": "reversed" if reversed_order else "forward",
        "initial_before": initial_before,
        "initial_selection": initial_selection,
        "initial_after": initial_after,
        "basis_before_successor": basis_before_successor,
        "successor_selection": successor_selection,
        "old_after_growth": old_after_growth,
        "new_after_growth": new_after_growth,
        "all_after_growth": old_after_growth and new_after_growth,
        "reload_routes": reload_routes,
        "stale_version_rejected": stale_rejected,
        "final_record_count": memory.record_count,
        "final_physical_value_scalars": memory.physical_value_scalar_count,
        "final_dense_value_scalars": memory.dense_value_scalar_count,
        "path": path,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1:
        raise ValueError("shared-basis structure policy updates must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    policy, training = _train_policy(seed=args.seed, updates=args.policy_updates)
    trained_scores = _evaluate_policy(policy, seed=args.seed + 800_000)
    torch.manual_seed(args.seed + 900_000)
    fresh_policy = OpaqueSharedBasisStructurePolicy(
        value_width=EVENT_WIDTH,
        hidden=POLICY_HIDDEN,
        max_spectral_bins=8,
        learning_rate=POLICY_LEARNING_RATE,
    ).eval()
    fresh_scores = _evaluate_policy(fresh_policy, seed=args.seed + 800_000)
    with tempfile.TemporaryDirectory(
        prefix="neural-computer-shared-basis-structure-"
    ) as directory:
        forward = _run_live_stream(
            policy=policy,
            system=system,
            seed=args.seed,
            path=Path(directory) / "forward.pt",
            reversed_order=False,
        )
        reversed_stream = _run_live_stream(
            policy=policy,
            system=system,
            seed=args.seed + 100,
            path=Path(directory) / "reversed.pt",
            reversed_order=True,
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
            )
        except ValueError as error:
            corruption_rejected = "checksum" in str(error).lower()
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "trained_beats_fresh": (
            trained_scores["1"] >= fresh_scores["1"] + 0.10
            and trained_scores["2"] >= fresh_scores["2"]
            and trained_scores["4"] >= fresh_scores["4"]
        ),
        "trained_heldout_rank_1": trained_scores["1"] >= 0.80,
        "trained_heldout_rank_2": trained_scores["2"] >= 0.80,
        "trained_heldout_rank_4": trained_scores["4"] >= 0.80,
        "forward_initial_route": bool(forward["initial_before"]),
        "forward_initial_rank_2": forward["initial_selection"]["selected_rank"] == 2,
        "forward_initial_accepted": bool(forward["initial_selection"]["accepted"]),
        "forward_initial_retained": bool(forward["initial_after"]),
        "forward_successor_rank_4": forward["successor_selection"]["selected_rank"] == 4,
        "forward_successor_accepted": bool(forward["successor_selection"]["accepted"]),
        "forward_old_retained_after_growth": bool(forward["old_after_growth"]),
        "forward_new_retained_after_growth": bool(forward["new_after_growth"]),
        "forward_reload_routes": bool(forward["reload_routes"]),
        "forward_stale_version_rejected": bool(forward["stale_version_rejected"]),
        "reversed_initial_rank_2": reversed_stream["initial_selection"]["selected_rank"] == 2,
        "reversed_successor_rank_4": reversed_stream["successor_selection"]["selected_rank"] == 4,
        "reversed_all_routes": bool(reversed_stream["all_after_growth"]),
        "reversed_reload_routes": bool(reversed_stream["reload_routes"]),
        "corruption_rejected": corruption_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": SHARED_BASIS_STRUCTURE_GROWTH_SCHEMA,
        "claim_boundary": (
            "Outcome-trained permutation-invariant external structure policy "
            "that consumes opaque value rows without precomputed candidate "
            "reconstruction error, transferred across one nonstationary "
            "frozen canonical memory growth event with verifier-gated commit; "
            "not unrestricted semantic structure discovery, new computation, "
            "or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "policy": "opaque_shared_basis_structure_policy_v1",
            "policy_signal": "single_scalar_verifier_utility_without_replay",
            "memory": "persistent_shared_basis_content_addressed_memory_v1",
            "candidate_ranks": CANDIDATE_RANKS,
            "feature_contract": "opaque_singular_spectrum_row_permutation_invariant_v1",
            "forbidden_features": "precomputed_candidate_reconstruction_error_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "stream": "rank_two_cohort_then_rank_four_successor_cohort",
        },
        "training": training,
        "trained_scores": trained_scores,
        "fresh_scores": fresh_scores,
        "forward": {
            key: value for key, value in forward.items() if key != "path"
        },
        "reversed": {
            key: value
            for key, value in reversed_stream.items()
            if key != "path"
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": args.policy_updates + (INITIAL_RECORDS + NEW_RECORDS) * 4,
            "unique_logical_lifetimes": args.policy_updates + (INITIAL_RECORDS + NEW_RECORDS) * 2,
            "optimizer_updates": args.policy_updates,
            "live_compression_transactions": 4,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_shared_basis_structure_growth"
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
    parser.add_argument("--policy-updates", type=int, default=50_000)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
