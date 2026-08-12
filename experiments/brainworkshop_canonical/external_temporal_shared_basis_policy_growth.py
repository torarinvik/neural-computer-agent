"""Learn and transfer a generic shared-basis compression preference.

The previous shared-basis audit proved that a deterministic candidate can be
verified and committed safely.  This experiment adds an external policy that
learns, from one scalar utility per generic candidate bank, which compression
candidate offers the smallest safe representation.  It then transfers that
preference to a frozen canonical event boundary across a nonstationary stream:
an old value cohort is compressed, a new cohort grows the shared basis, and a
second candidate is proposed without replaying the old values.

The policy sees only generic candidate statistics.  It does not receive task
families, semantic labels, correct actions, or verifier-private targets.  The
memory's route/value verifier remains authoritative and can reject any policy
proposal before mutation.
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
    MemoryQuery,
    OpaqueSharedBasisCompressionPolicy,
    PersistentSharedBasisContentAddressedMemory,
    SharedBasisContentAddressedMemory,
)

from .external_temporal_content_retrieval_growth import _event_key
from .external_temporal_query_address_growth import EVENT_WIDTH, _build

SHARED_BASIS_POLICY_GROWTH_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-shared-basis-policy-growth.v1"
)
FEATURE_WIDTH = 8
CANDIDATE_RANKS = (1, 2, 4)
TRAINING_TOLERANCE = 0.08
RETENTION_TOLERANCE = 0.04
INITIAL_RECORDS = 6
NEW_RECORDS = 6
TOTAL_RECORDS = INITIAL_RECORDS + NEW_RECORDS
NOISE_SCALE = 0.003
READ_MATCH_THRESHOLD = 0.75
WRITE_MATCH_THRESHOLD = 0.999


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _synthetic_values(
    *,
    seed: int,
    target_rank: int | None = None,
    width: int = EVENT_WIDTH,
) -> tuple[torch.Tensor, int]:
    """Generate generic opaque low-rank values for policy learning."""

    if target_rank is not None and target_rank not in CANDIDATE_RANKS:
        raise ValueError("synthetic target rank is outside the candidate set")
    generator = torch.Generator().manual_seed(seed)
    records = int(torch.randint(6, 13, (), generator=generator))
    if target_rank is None:
        target_rank = CANDIDATE_RANKS[
            int(torch.randint(len(CANDIDATE_RANKS), (), generator=generator))
        ]
    basis = torch.linalg.qr(
        torch.randn(width, target_rank, generator=generator)
    ).Q[:, :target_rank]
    values = (
        torch.randn(records, target_rank, generator=generator) @ basis.transpose(0, 1)
        + NOISE_SCALE * torch.randn(records, width, generator=generator)
    )
    return F.normalize(values, dim=-1), target_rank


def _value_features(
    values: torch.Tensor,
    *,
    candidate_ranks: tuple[int, ...] = CANDIDATE_RANKS,
) -> tuple[torch.Tensor, tuple[float, ...], tuple[int, ...]]:
    """Build generic candidate statistics and return errors/storage sizes."""

    if values.ndim != 2 or values.shape[1] != EVENT_WIDTH:
        raise ValueError("shared-basis policy values have the wrong shape")
    if not candidate_ranks or any(rank < 1 for rank in candidate_ranks):
        raise ValueError("shared-basis candidate ranks are invalid")
    records, width = values.shape
    _left, _singular, right = torch.linalg.svd(values, full_matrices=False)
    dense_scalars = records * width
    max_rank = max(candidate_ranks)
    features: list[list[float]] = []
    errors: list[float] = []
    physical_sizes: list[int] = []
    for rank in candidate_ranks:
        basis = right[:rank]
        approximation = values @ basis.transpose(0, 1) @ basis
        error = float((values - approximation).abs().max())
        physical = rank * width + records * rank
        errors.append(error)
        physical_sizes.append(physical)
        storage_fraction = physical / max(1, dense_scalars)
        features.append(
            [
                rank / max_rank,
                error,
                storage_fraction,
                1.0 - storage_fraction,
                rank / width,
                records / 16.0,
                1.0,
                error / TRAINING_TOLERANCE,
            ]
        )
    return torch.tensor(features, dtype=values.dtype), tuple(errors), tuple(physical_sizes)


def _minimal_safe_index(
    errors: tuple[float, ...],
    physical_sizes: tuple[int, ...],
    *,
    dense_scalars: int,
) -> int:
    for index, (error, physical) in enumerate(zip(errors, physical_sizes, strict=True)):
        if error <= TRAINING_TOLERANCE and physical < dense_scalars:
            return index
    return len(errors) - 1


def _train_policy(
    *,
    seed: int,
    updates: int,
) -> tuple[OpaqueSharedBasisCompressionPolicy, dict[str, float | int]]:
    if updates < 1:
        raise ValueError("shared-basis policy updates must be positive")
    torch.manual_seed(seed)
    policy = OpaqueSharedBasisCompressionPolicy(
        feature_width=FEATURE_WIDTH,
        hidden=64,
        learning_rate=0.005,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.005)
    explorer = torch.Generator().manual_seed(seed + 50_001)
    utilities: list[float] = []
    for update in range(updates):
        values, _target_rank = _synthetic_values(seed=seed + 10_000 + update)
        features, errors, physical_sizes = _value_features(values)
        dense_scalars = values.shape[0] * values.shape[1]
        target_index = _minimal_safe_index(
            errors,
            physical_sizes,
            dense_scalars=dense_scalars,
        )
        plan = policy.propose(
            features.unsqueeze(0),
            explore=True,
            temperature=0.8,
            generator=explorer,
        )
        utility = float(plan.candidate_index == target_index)
        policy.adaptation_step(
            features.unsqueeze(0),
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
    policy: OpaqueSharedBasisCompressionPolicy,
    *,
    seed: int,
    episodes_per_rank: int = 64,
) -> dict[str, float]:
    scores: dict[str, list[float]] = {str(rank): [] for rank in CANDIDATE_RANKS}
    for rank_index, rank in enumerate(CANDIDATE_RANKS):
        for episode in range(episodes_per_rank):
            values, _ = _synthetic_values(
                seed=seed + rank_index * 10_000 + episode,
                target_rank=rank,
            )
            features, errors, physical_sizes = _value_features(values)
            target_index = _minimal_safe_index(
                errors,
                physical_sizes,
                dense_scalars=values.shape[0] * values.shape[1],
            )
            predicted = policy.propose(features.unsqueeze(0)).candidate_index
            scores[str(rank)].append(float(predicted == target_index))
    return {
        rank: sum(values) / len(values) for rank, values in scores.items()
    }


def _live_payloads(system, *, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    keys = torch.stack(
        tuple(_event_key(system, symbol) for symbol in range(TOTAL_RECORDS))
    )
    anchors = torch.stack(
        tuple(_event_key(system, symbol) for symbol in range(4))
    )
    shared_basis = torch.linalg.qr(anchors.transpose(0, 1)).Q[:, :4]
    generator = torch.Generator().manual_seed(seed + 70_001)
    initial = (
        torch.randn(INITIAL_RECORDS, 2, generator=generator)
        @ shared_basis[:, :2].transpose(0, 1)
        + NOISE_SCALE * torch.randn(
            INITIAL_RECORDS, EVENT_WIDTH, generator=generator
        )
    )
    successor = (
        torch.randn(NEW_RECORDS, 4, generator=generator)
        @ shared_basis.transpose(0, 1)
        + NOISE_SCALE * torch.randn(NEW_RECORDS, EVENT_WIDTH, generator=generator)
    )
    values = F.normalize(torch.cat((initial, successor)), dim=-1)
    return keys, values[:INITIAL_RECORDS], values[INITIAL_RECORDS:]


def _route_verifier(
    memory: SharedBasisContentAddressedMemory,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    tolerance: float,
) -> bool:
    indices: list[int] = []
    for key, expected in zip(keys, values, strict=True):
        read = memory.read(MemoryQuery(key.reshape(1, -1), top_k=1))
        if not bool(read.hit.item()):
            return False
        indices.append(int(read.indices[0, 0]))
        if not torch.allclose(read.value[0], expected, atol=tolerance, rtol=0.0):
            return False
    return len(indices) == len(set(indices))


def _candidate_views(
    memory: SharedBasisContentAddressedMemory,
) -> tuple[torch.Tensor, tuple[SharedBasisContentAddressedMemory, ...]]:
    candidates = tuple(
        memory.compression_candidate(rank) for rank in CANDIDATE_RANKS
    )
    dense = max(1, memory.dense_value_scalar_count)
    features: list[list[float]] = []
    for rank, candidate in zip(CANDIDATE_RANKS, candidates, strict=True):
        error = memory.max_value_error(candidate)
        storage_fraction = candidate.physical_value_scalar_count / dense
        features.append(
            [
                rank / max(CANDIDATE_RANKS),
                error,
                storage_fraction,
                1.0 - storage_fraction,
                rank / memory.width,
                memory.record_count / 16.0,
                memory.basis_count / memory.width,
                error / TRAINING_TOLERANCE,
            ]
        )
    return torch.tensor(features, dtype=memory.keys.dtype), candidates


def _select_and_commit(
    policy: OpaqueSharedBasisCompressionPolicy,
    memory: PersistentSharedBasisContentAddressedMemory,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> dict[str, object]:
    features, candidates = _candidate_views(memory)
    plan = policy.propose(features.unsqueeze(0))
    selected_rank = CANDIDATE_RANKS[plan.candidate_index]
    selected_candidate = candidates[plan.candidate_index]
    candidate_error = memory.max_value_error(selected_candidate)
    expected_version = int(memory.store_version.item())
    receipt = memory.replace_from_candidate(
        selected_candidate,
        expected_version=expected_version,
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
        "basis_rows_before": int(features[-1, 6].item() * memory.width),
        "basis_rows_after": memory.basis_count,
        "physical_value_scalars_after": memory.physical_value_scalar_count,
        "version": int(memory.store_version.item()),
    }


def _run_live_stream(
    *,
    policy: OpaqueSharedBasisCompressionPolicy,
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
        memory,
        keys[:INITIAL_RECORDS],
        initial_values,
        tolerance=1e-6,
    )
    initial_selection = _select_and_commit(
        policy,
        memory,
        keys[:INITIAL_RECORDS],
        initial_values,
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
    all_keys = keys
    all_values = torch.cat((initial_values, successor_values))
    successor_selection = _select_and_commit(
        policy,
        memory,
        all_keys,
        all_values,
    )
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
        all_keys,
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
        raise ValueError("shared-basis policy updates must be positive")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    policy, training = _train_policy(seed=args.seed, updates=args.policy_updates)
    trained_scores = _evaluate_policy(policy, seed=args.seed + 800_000)
    torch.manual_seed(args.seed + 900_000)
    fresh_policy = OpaqueSharedBasisCompressionPolicy(
        feature_width=FEATURE_WIDTH,
        hidden=64,
        learning_rate=0.005,
    ).eval()
    fresh_scores = _evaluate_policy(fresh_policy, seed=args.seed + 800_000)
    with tempfile.TemporaryDirectory(
        prefix="neural-computer-shared-basis-policy-"
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
        "schema": SHARED_BASIS_POLICY_GROWTH_SCHEMA,
        "claim_boundary": (
            "Outcome-trained external preference for generic shared-basis "
            "compression transferred across a nonstationary frozen canonical "
            "memory stream with verifier-gated copy-on-write; not arbitrary "
            "semantic structure discovery, unrestricted memory growth, new "
            "computation, or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "policy": "opaque_shared_basis_compression_policy_v1",
            "policy_signal": "single_scalar_verifier_utility_without_replay",
            "memory": "persistent_shared_basis_content_addressed_memory_v1",
            "candidate_ranks": CANDIDATE_RANKS,
            "feature_contract": "opaque_rank_error_storage_occupancy_statistics_v1",
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
            "unique_verifier_bits": args.policy_updates + TOTAL_RECORDS * 4,
            "unique_logical_lifetimes": args.policy_updates + TOTAL_RECORDS * 2,
            "optimizer_updates": args.policy_updates,
            "live_compression_transactions": 4,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_shared_basis_policy_growth"
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
    parser.add_argument("--policy-updates", type=int, default=3_000)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
