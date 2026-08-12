"""Pressure-test repeated external shared-structure growth without replay.

The v2 raw-value structure policy is trained only from scalar verifier utility
on fresh opaque banks.  A frozen canonical event boundary then receives four
successive cohorts.  The first cohort has rank-two structure; three successor
cohorts use the rank-four structure.  Every candidate is advisory and the
persistent memory verifier must preserve the complete route prefix before a
copy-on-write replacement is committed.

The same stream is run with forward and reversed physical insertion order.
This is a repeated bounded growth test, not unrestricted memory growth or
general continual learning.
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
    CANDIDATE_RANKS,
    INITIAL_RECORDS,
    RETENTION_TOLERANCE,
    _route_verifier,
)
from .external_temporal_shared_basis_structure_growth import (
    POLICY_HIDDEN,
    POLICY_LEARNING_RATE,
    _digest,
    _evaluate_policy,
    _select_and_commit,
    _train_policy,
)

REPEATED_SHARED_BASIS_GROWTH_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-shared-basis-repeated-growth.v1"
)
STAGE_RANKS = (2, 4, 4, 4)
STAGE_RECORDS = INITIAL_RECORDS
TOTAL_RECORDS = STAGE_RECORDS * len(STAGE_RANKS)
NOISE_SCALE = 0.003
READ_MATCH_THRESHOLD = 0.75
WRITE_MATCH_THRESHOLD = 0.999


def _repeated_live_payloads(
    system,
    *,
    seed: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    base_keys = torch.stack(
        tuple(_event_key(system, symbol % 12) for symbol in range(TOTAL_RECORDS))
    )
    key_generator = torch.Generator().manual_seed(seed + 70_001)
    keys = F.normalize(
        base_keys
        + 0.25
        * torch.randn(
            TOTAL_RECORDS,
            EVENT_WIDTH,
            generator=key_generator,
        ),
        dim=-1,
    )
    anchors = keys[:4]
    shared_basis = torch.linalg.qr(anchors.transpose(0, 1)).Q[:, :4]
    generator = torch.Generator().manual_seed(seed + 71_001)
    stages: list[torch.Tensor] = []
    for rank in STAGE_RANKS:
        values = (
            torch.randn(STAGE_RECORDS, rank, generator=generator)
            @ shared_basis[:, :rank].transpose(0, 1)
            + NOISE_SCALE
            * torch.randn(STAGE_RECORDS, EVENT_WIDTH, generator=generator)
        )
        stages.append(F.normalize(values, dim=-1))
    return keys, tuple(stages)


def _run_stream(
    *,
    policy: OpaqueSharedBasisStructurePolicy,
    system,
    seed: int,
    path: Path,
    reversed_order: bool,
) -> dict[str, object]:
    keys, stages = _repeated_live_payloads(system, seed=seed)
    memory = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
    )
    stage_reports: list[dict[str, object]] = []
    current_values: list[torch.Tensor] = []
    for stage_index, (expected_rank, values) in enumerate(
        zip(STAGE_RANKS, stages, strict=True)
    ):
        start = stage_index * STAGE_RECORDS
        end = start + STAGE_RECORDS
        order = (
            torch.arange(STAGE_RECORDS - 1, -1, -1)
            if reversed_order
            else torch.arange(STAGE_RECORDS)
        )
        memory.write(
            keys[start:end][order],
            values[order],
            torch.ones(STAGE_RECORDS),
        )
        current_values.append(values)
        prefix_values = torch.cat(current_values)
        prefix_keys = keys[:end]
        before = _route_verifier(
            memory,
            prefix_keys,
            prefix_values,
            tolerance=RETENTION_TOLERANCE,
        )
        selection = _select_and_commit(
            policy,
            memory,
            prefix_keys,
            prefix_values,
        )
        after = _route_verifier(
            memory,
            prefix_keys,
            prefix_values,
            tolerance=RETENTION_TOLERANCE,
        )
        stage_reports.append(
            {
                "stage": stage_index,
                "expected_rank": expected_rank,
                "record_count": end,
                "route_prefix_before": before,
                "selection": selection,
                "route_prefix_after": after,
            }
        )

    stale_version_rejected = False
    try:
        memory.replace_from_candidate(
            memory.compression_candidate(CANDIDATE_RANKS[-1]),
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
        "physical_order": "reversed" if reversed_order else "forward",
        "stages": stage_reports,
        "all_routes_after_growth": all(
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
        raise ValueError("repeated shared-basis policy updates must be positive")
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

    with tempfile.TemporaryDirectory(
        prefix="neural-computer-shared-basis-repeated-"
    ) as directory:
        forward = _run_stream(
            policy=policy,
            system=system,
            seed=args.seed,
            path=Path(directory) / "forward.pt",
            reversed_order=False,
        )
        reversed_stream = _run_stream(
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
    expected_rank_gates = {
        "forward_expected_ranks": tuple(
            stage["selection"]["selected_rank"] for stage in forward["stages"]
        )
        == STAGE_RANKS,
        "reversed_expected_ranks": tuple(
            stage["selection"]["selected_rank"]
            for stage in reversed_stream["stages"]
        )
        == STAGE_RANKS,
    }
    gates = {
        "trained_beats_fresh": (
            trained_scores["1"] >= fresh_scores["1"] + 0.10
            and trained_scores["2"] >= fresh_scores["2"]
            and trained_scores["4"] >= fresh_scores["4"]
        ),
        "trained_heldout_rank_1": trained_scores["1"] >= 0.80,
        "trained_heldout_rank_2": trained_scores["2"] >= 0.80,
        "trained_heldout_rank_4": trained_scores["4"] >= 0.80,
        **expected_rank_gates,
        "forward_each_commit_accepted": all(
            bool(stage["selection"]["accepted"]) for stage in forward["stages"]
        ),
        "forward_complete_prefix_retention": bool(
            forward["all_routes_after_growth"]
        ),
        "reversed_each_commit_accepted": all(
            bool(stage["selection"]["accepted"])
            for stage in reversed_stream["stages"]
        ),
        "reversed_complete_prefix_retention": bool(
            reversed_stream["all_routes_after_growth"]
        ),
        "forward_reload_routes": bool(forward["reload_routes"]),
        "reversed_reload_routes": bool(reversed_stream["reload_routes"]),
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
        "schema": REPEATED_SHARED_BASIS_GROWTH_SCHEMA,
        "claim_boundary": (
            "Outcome-trained permutation-invariant external structure policy "
            "from opaque value rows and scalar verifier utility, transferred "
            "through three successive frozen-memory growth transitions with "
            "verifier-gated commits; not unrestricted semantic structure "
            "discovery, arbitrary new computation, or general continual "
            "learning."
        ),
        "seed": args.seed,
        "architecture": {
            "policy": "opaque_shared_basis_structure_policy_v2",
            "policy_signal": "single_scalar_verifier_utility_without_replay",
            "memory": "persistent_shared_basis_content_addressed_memory_v1",
            "candidate_ranks": CANDIDATE_RANKS,
            "feature_contract": (
                "opaque_spectral_pairwise_structure_row_permutation_invariant_v2"
            ),
            "forbidden_features": "precomputed_candidate_reconstruction_error_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "stream": "rank_two_then_three_rank_four_successor_cohorts",
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
            "live_compression_transactions": len(STAGE_RANKS) * 2,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_repeated_shared_basis_growth"
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
    parser.add_argument("--policy-updates", type=int, default=20_000)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
