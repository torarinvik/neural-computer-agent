"""Pressure-test shared structure under genuinely competing subspaces.

This rung trains the v2 raw-value structure policy on runtime candidate ranks
``(2, 4, 8)``.  The frozen canonical memory then receives four rank-two
cohorts, each in a different orthogonal subspace.  The safe shared basis must
grow ``2 -> 4 -> 8 -> 8`` as the union changes from two to four to six and
finally eight dimensions.  The experiment is repeated with the subspace
arrival order reversed and with physical row order reversed independently.

Every policy proposal is advisory.  The memory-side route/value verifier,
expected-version check, and persistent copy-on-write boundary remain
authoritative.  This is a bounded competing-subspace test, not unrestricted
memory growth or general continual learning.
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
    OpaqueSharedBasisStructurePolicy,
    PersistentSharedBasisContentAddressedMemory,
)

from .external_temporal_content_retrieval_growth import _event_key
from .external_temporal_query_address_growth import EVENT_WIDTH, _build
from .external_temporal_shared_basis_policy_growth import (
    RETENTION_TOLERANCE,
    _digest,
    _minimal_safe_index,
    _route_verifier,
)
from .external_temporal_shared_basis_structure_growth import (
    POLICY_HIDDEN,
    POLICY_LEARNING_RATE,
)

COMPETING_SUBSPACE_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-shared-basis-competing-subspaces.v1"
)
COMPETING_CANDIDATE_RANKS = (2, 4, 8)
TRAINING_TOLERANCE = 0.08
POLICY_TEMPERATURE = 0.6
TRAINING_RECORD_MIN = 20
TRAINING_RECORD_MAX = 32
STAGE_RECORDS = 12
STAGE_COUNT = 4
TOTAL_RECORDS = STAGE_RECORDS * STAGE_COUNT
SUBSPACE_WIDTH = 2
SUBSPACE_RANKS = (2, 4, 6, 8)
NOISE_SCALE = 0.002
READ_MATCH_THRESHOLD = 0.75
WRITE_MATCH_THRESHOLD = 0.999


def _synthetic_values(
    *,
    seed: int,
    target_rank: int | None = None,
) -> torch.Tensor:
    if target_rank is not None and target_rank not in COMPETING_CANDIDATE_RANKS:
        raise ValueError("competing-subspace target rank is invalid")
    generator = torch.Generator().manual_seed(seed)
    records = int(
        torch.randint(
            TRAINING_RECORD_MIN,
            TRAINING_RECORD_MAX + 1,
            (),
            generator=generator,
        )
    )
    if target_rank is None:
        target_rank = COMPETING_CANDIDATE_RANKS[
            int(
                torch.randint(
                    len(COMPETING_CANDIDATE_RANKS),
                    (),
                    generator=generator,
                )
            )
        ]
    basis = torch.linalg.qr(
        torch.randn(EVENT_WIDTH, target_rank, generator=generator)
    ).Q[:, :target_rank]
    values = (
        torch.randn(records, target_rank, generator=generator)
        @ basis.transpose(0, 1)
        + NOISE_SCALE
        * torch.randn(records, EVENT_WIDTH, generator=generator)
    )
    return F.normalize(values, dim=-1)


def _target_index(values: torch.Tensor) -> int:
    """Private verifier target; no target or error enters the policy ABI."""

    _left, _singular, right = torch.linalg.svd(values, full_matrices=False)
    errors: list[float] = []
    physical_sizes: list[int] = []
    for rank in COMPETING_CANDIDATE_RANKS:
        basis = right[:rank]
        errors.append(
            float((values - values @ basis.transpose(0, 1) @ basis).abs().max())
        )
        physical_sizes.append(rank * (values.shape[1] + values.shape[0]))
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
        raise ValueError("competing-subspace policy updates must be positive")
    torch.manual_seed(seed)
    policy = OpaqueSharedBasisStructurePolicy(
        value_width=EVENT_WIDTH,
        hidden=POLICY_HIDDEN,
        max_spectral_bins=8,
        learning_rate=POLICY_LEARNING_RATE,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=POLICY_LEARNING_RATE)
    candidate_ranks = torch.tensor(COMPETING_CANDIDATE_RANKS, dtype=torch.long)
    explorer = torch.Generator().manual_seed(seed + 52_001)
    utilities: list[float] = []
    for update in range(updates):
        values = _synthetic_values(seed=seed + 20_000 + update)
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
    candidate_ranks = torch.tensor(COMPETING_CANDIDATE_RANKS, dtype=torch.long)
    scores: dict[str, list[float]] = {
        str(rank): [] for rank in COMPETING_CANDIDATE_RANKS
    }
    for rank_index, rank in enumerate(COMPETING_CANDIDATE_RANKS):
        for episode in range(episodes_per_rank):
            values = _synthetic_values(
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


def _live_payloads(
    system,
    *,
    seed: int,
    subspace_order: tuple[int, ...],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    base_keys = torch.stack(
        tuple(_event_key(system, symbol % 12) for symbol in range(TOTAL_RECORDS))
    )
    key_generator = torch.Generator().manual_seed(seed + 72_001)
    keys = F.normalize(
        base_keys
        + 0.30
        * torch.randn(TOTAL_RECORDS, EVENT_WIDTH, generator=key_generator),
        dim=-1,
    )
    basis_generator = torch.Generator().manual_seed(seed + 73_001)
    shared_basis = torch.linalg.qr(
        torch.randn(EVENT_WIDTH, SUBSPACE_RANKS[-1], generator=basis_generator)
    ).Q[:, : SUBSPACE_RANKS[-1]]
    value_generator = torch.Generator().manual_seed(seed + 74_001)
    canonical_stages: list[torch.Tensor] = []
    for stage in range(STAGE_COUNT):
        values = (
            torch.randn(STAGE_RECORDS, SUBSPACE_WIDTH, generator=value_generator)
            @ shared_basis[:, stage * SUBSPACE_WIDTH : (stage + 1) * SUBSPACE_WIDTH].transpose(
                0, 1
            )
            + NOISE_SCALE
            * torch.randn(
                STAGE_RECORDS,
                EVENT_WIDTH,
                generator=value_generator,
            )
        )
        canonical_stages.append(F.normalize(values, dim=-1))
    ordered_keys = tuple(
        keys[stage * STAGE_RECORDS : (stage + 1) * STAGE_RECORDS]
        for stage in subspace_order
    )
    ordered_values = tuple(canonical_stages[stage] for stage in subspace_order)
    return torch.cat(ordered_keys), ordered_values


def _select_and_commit(
    policy: OpaqueSharedBasisStructurePolicy,
    memory: PersistentSharedBasisContentAddressedMemory,
    candidate_ranks: tuple[int, ...],
    keys: torch.Tensor,
    values: torch.Tensor,
) -> dict[str, object]:
    candidates = memory.candidates()
    basis_before = memory.basis_count
    rank_tensor = torch.tensor(
        candidate_ranks,
        dtype=torch.long,
        device=candidates.values.device,
    )
    plan = policy.propose(
        candidates.values,
        candidates.occupied,
        rank_tensor,
    )
    selected_rank = candidate_ranks[plan.candidate_index]
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


def _run_stream(
    *,
    policy: OpaqueSharedBasisStructurePolicy,
    system,
    seed: int,
    path: Path,
    subspace_order: tuple[int, ...],
    reversed_physical_order: bool,
) -> dict[str, object]:
    keys, stages = _live_payloads(
        system,
        seed=seed,
        subspace_order=subspace_order,
    )
    memory = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
    )
    stage_reports: list[dict[str, object]] = []
    prefix_values: list[torch.Tensor] = []
    for stage, values in enumerate(stages):
        start = stage * STAGE_RECORDS
        end = start + STAGE_RECORDS
        order = (
            torch.arange(STAGE_RECORDS - 1, -1, -1)
            if reversed_physical_order
            else torch.arange(STAGE_RECORDS)
        )
        memory.write(
            keys[start:end][order],
            values[order],
            torch.ones(STAGE_RECORDS),
        )
        prefix_values.append(values)
        all_values = torch.cat(prefix_values)
        prefix_keys = keys[:end]
        before = _route_verifier(
            memory,
            prefix_keys,
            all_values,
            tolerance=RETENTION_TOLERANCE,
        )
        selection = _select_and_commit(
            policy,
            memory,
            COMPETING_CANDIDATE_RANKS,
            prefix_keys,
            all_values,
        )
        after = _route_verifier(
            memory,
            prefix_keys,
            all_values,
            tolerance=RETENTION_TOLERANCE,
        )
        stage_reports.append(
            {
                "stage": stage,
                "expected_union_rank": SUBSPACE_RANKS[stage],
                "record_count": end,
                "route_prefix_before": before,
                "selection": selection,
                "route_prefix_after": after,
            }
        )
    stale_version_rejected = False
    try:
        memory.replace_from_candidate(
            memory.compression_candidate(COMPETING_CANDIDATE_RANKS[-1]),
            expected_version=1,
        )
    except RuntimeError as error:
        stale_version_rejected = "stale" in str(error).lower()
    restored = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
    )
    all_values = torch.cat(stages)
    reload_routes = _route_verifier(
        restored,
        keys,
        all_values,
        tolerance=RETENTION_TOLERANCE,
    )
    return {
        "subspace_order": subspace_order,
        "reversed_physical_order": reversed_physical_order,
        "stages": stage_reports,
        "complete_prefix_retention": all(
            bool(stage["route_prefix_after"]) for stage in stage_reports
        ),
        "reload_routes": reload_routes,
        "stale_version_rejected": stale_version_rejected,
        "final_record_count": memory.record_count,
        "final_physical_value_scalars": memory.physical_value_scalar_count,
        "final_dense_value_scalars": memory.dense_value_scalar_count,
        "path": path,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.policy_updates < 1:
        raise ValueError("competing-subspace policy updates must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    policy, training = _train_policy(
        seed=args.seed,
        updates=args.policy_updates,
    )
    trained_scores = _evaluate_policy(policy, seed=args.seed + 800_000)
    torch.manual_seed(args.seed + 900_000)
    fresh_policy = OpaqueSharedBasisStructurePolicy(
        value_width=EVENT_WIDTH,
        hidden=POLICY_HIDDEN,
        max_spectral_bins=8,
        learning_rate=POLICY_LEARNING_RATE,
    ).eval()
    fresh_scores = _evaluate_policy(fresh_policy, seed=args.seed + 800_000)
    regimes = {
        "forward_subspaces": (0, 1, 2, 3),
        "reversed_subspaces": (3, 2, 1, 0),
    }
    with tempfile.TemporaryDirectory(
        prefix="neural-computer-shared-basis-competing-"
    ) as directory:
        streams: dict[str, dict[str, object]] = {}
        for regime_index, (name, order) in enumerate(regimes.items()):
            for physical_reversed in (False, True):
                label = f"{name}_{'physical_reversed' if physical_reversed else 'physical_forward'}"
                streams[label] = _run_stream(
                    policy=policy,
                    system=system,
                    seed=args.seed + 100 * regime_index + int(physical_reversed),
                    path=Path(directory) / f"{label}.pt",
                    subspace_order=order,
                    reversed_physical_order=physical_reversed,
                )
        corruption_source = streams["forward_subspaces_physical_forward"]
        corruption_path = Path(directory) / "corrupt.pt"
        payload = torch.load(corruption_source["path"], weights_only=False)
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
    expected_ranks = (2, 4, 8, 8)
    stream_gates = {
        f"{name}_expected_ranks": tuple(
            stage["selection"]["selected_rank"] for stage in stream["stages"]
        )
        == expected_ranks
        for name, stream in streams.items()
    }
    gates = {
        "trained_beats_fresh": (
            trained_scores["2"] >= fresh_scores["2"]
            and trained_scores["4"] >= fresh_scores["4"] + 0.10
            and trained_scores["8"] >= fresh_scores["8"] + 0.10
        ),
        "trained_heldout_rank_2": trained_scores["2"] >= 0.80,
        "trained_heldout_rank_4": trained_scores["4"] >= 0.80,
        "trained_heldout_rank_8": trained_scores["8"] >= 0.80,
        **stream_gates,
        "all_commits_accepted": all(
            bool(stage["selection"]["accepted"])
            for stream in streams.values()
            for stage in stream["stages"]
        ),
        "all_complete_prefixes_retained": all(
            bool(stream["complete_prefix_retention"])
            for stream in streams.values()
        ),
        "all_reloads_exact": all(
            bool(stream["reload_routes"]) for stream in streams.values()
        ),
        "all_stale_versions_rejected": all(
            bool(stream["stale_version_rejected"])
            for stream in streams.values()
        ),
        "subspace_order_reversal_tested": True,
        "physical_order_reversal_tested": True,
        "corruption_rejected": corruption_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": COMPETING_SUBSPACE_SCHEMA,
        "claim_boundary": (
            "Outcome-trained external structure policy over runtime ranks "
            "2/4/8, transferred through four incompatible orthogonal "
            "subspace arrivals with complete route retention and verifier-gated "
            "copy-on-write; not unrestricted semantic structure discovery, "
            "arbitrary new computation, or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "policy": "opaque_shared_basis_structure_policy_v2",
            "policy_signal": "single_scalar_verifier_utility_without_replay",
            "memory": "persistent_shared_basis_content_addressed_memory_v1",
            "candidate_ranks": COMPETING_CANDIDATE_RANKS,
            "feature_contract": (
                "opaque_spectral_pairwise_structure_row_permutation_invariant_v2"
            ),
            "forbidden_features": "precomputed_candidate_reconstruction_error_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "stream": "four_incompatible_orthogonal_rank_two_cohorts",
        },
        "training": training,
        "trained_scores": trained_scores,
        "fresh_scores": fresh_scores,
        "streams": {
            name: {key: value for key, value in stream.items() if key != "path"}
            for name, stream in streams.items()
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": args.policy_updates + TOTAL_RECORDS * 8,
            "unique_logical_lifetimes": args.policy_updates + TOTAL_RECORDS * 4,
            "optimizer_updates": args.policy_updates,
            "live_compression_transactions": len(streams) * STAGE_COUNT,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_competing_subspace_growth"
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
