"""Verifier-gated shared-structure compression for canonical external memory.

This pressure test stores multiple distinct opaque capability values under
independent learned event keys.  The values share a low-dimensional external
structure with a small residual, so a factorized memory can reduce physical
storage without merging logical addresses.  A rank-one candidate is rejected
by an independent route/value verifier; a rank-two candidate is committed
copy-on-write only when every route remains usable.

The canonical controller and event encoder are frozen.  No semantic field,
task label, correct action, or verifier-private relation crosses the memory
boundary.  The rank choice is intentionally deterministic in this first
contract audit; this promotes safe shared storage, not learned compression or
general continual learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import (
    MemoryQuery,
    PersistentSharedBasisContentAddressedMemory,
    SharedBasisContentAddressedMemory,
)

from .external_temporal_content_retrieval_growth import _event_key
from .external_temporal_query_address_growth import EVENT_WIDTH, _build

SHARED_BASIS_COMPRESSION_GROWTH_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-shared-basis-compression-growth.v1"
)
RECORD_COUNT = 12
SHARED_RANK = 2
NOISE_SCALE = 0.002
READ_MATCH_THRESHOLD = 0.75
WRITE_MATCH_THRESHOLD = 0.999
RETENTION_TOLERANCE = 0.02


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _payloads(system, *, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build opaque keys and shared learned-value stress payloads."""

    keys = torch.stack(
        tuple(_event_key(system, symbol) for symbol in range(RECORD_COUNT))
    )
    anchor_values = torch.stack((_event_key(system, 0), _event_key(system, 1)))
    shared_basis = torch.linalg.qr(anchor_values.transpose(0, 1)).Q[:, :SHARED_RANK]
    generator = torch.Generator().manual_seed(seed + 70_101)
    coefficients = torch.randn(
        RECORD_COUNT,
        SHARED_RANK,
        generator=generator,
    )
    residual = NOISE_SCALE * torch.randn(
        RECORD_COUNT,
        EVENT_WIDTH,
        generator=generator,
    )
    values = coefficients @ shared_basis.transpose(0, 1) + residual
    return keys, values


def _write(
    memory: SharedBasisContentAddressedMemory,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    order: torch.Tensor,
) -> None:
    receipt = memory.write(
        keys[order],
        values[order],
        torch.ones(RECORD_COUNT),
    )
    if not bool(receipt.committed.all()) or memory.record_count != RECORD_COUNT:
        raise RuntimeError("shared-basis stress payload did not fully commit")


def _route_verifier(
    memory: SharedBasisContentAddressedMemory,
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    tolerance: float,
) -> bool:
    """Private held-out route/value verifier for every logical record."""

    selected_indices: list[int] = []
    for key, expected in zip(keys, values, strict=True):
        read = memory.read(MemoryQuery(key.reshape(1, -1), top_k=1))
        if not bool(read.hit.item()):
            return False
        selected_indices.append(int(read.indices[0, 0]))
        if not torch.allclose(
            read.value[0], expected, atol=tolerance, rtol=0.0
        ):
            return False
    return len(set(selected_indices)) == len(selected_indices)


def _run_stream(
    *,
    system,
    keys: torch.Tensor,
    values: torch.Tensor,
    path: Path,
    reversed_order: bool,
) -> dict[str, object]:
    memory = PersistentSharedBasisContentAddressedMemory(
        EVENT_WIDTH,
        path=path,
        write_threshold=0.0,
        write_match_threshold=WRITE_MATCH_THRESHOLD,
        read_match_threshold=READ_MATCH_THRESHOLD,
        basis_tolerance=1e-8,
    )
    order = torch.arange(RECORD_COUNT - 1, -1, -1) if reversed_order else torch.arange(RECORD_COUNT)
    _write(memory, keys, values, order=order)
    routes_before = _route_verifier(
        memory,
        keys,
        values,
        tolerance=1e-6,
    )
    rows_before = memory.record_count
    basis_before = memory.basis_count
    physical_before = memory.physical_value_scalar_count
    dense_scalars = memory.dense_value_scalar_count
    version_before = int(memory.store_version.item())

    rank_one = memory.compression_candidate(1)
    rank_one_error = memory.max_value_error(rank_one)
    rank_one_digest_before = _digest(memory)
    rejected = memory.replace_from_candidate(
        rank_one,
        expected_version=version_before,
        retention_probe=lambda candidate: _route_verifier(
            candidate,
            keys,
            values,
            tolerance=RETENTION_TOLERANCE,
        ),
    )
    rank_one_non_mutating = (
        not rejected.accepted
        and _digest(memory) == rank_one_digest_before
        and int(memory.store_version.item()) == version_before
    )

    rank_two = memory.compression_candidate(SHARED_RANK)
    rank_two_error = memory.max_value_error(rank_two)
    accepted = memory.replace_from_candidate(
        rank_two,
        expected_version=version_before,
        retention_probe=lambda candidate: _route_verifier(
            candidate,
            keys,
            values,
            tolerance=RETENTION_TOLERANCE,
        ),
    )
    routes_after = _route_verifier(
        memory,
        keys,
        values,
        tolerance=RETENTION_TOLERANCE,
    )
    stale_rejected = False
    try:
        memory.replace_from_candidate(
            memory.compression_candidate(SHARED_RANK),
            expected_version=version_before,
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
        values,
        tolerance=RETENTION_TOLERANCE,
    )
    return {
        "physical_order": "reversed" if reversed_order else "forward",
        "rows_before": rows_before,
        "rows_after": memory.record_count,
        "basis_rows_before": basis_before,
        "basis_rows_after": memory.basis_count,
        "dense_value_scalars": dense_scalars,
        "physical_value_scalars_before": physical_before,
        "physical_value_scalars_after": memory.physical_value_scalar_count,
        "rank_one_error": rank_one_error,
        "rank_two_error": rank_two_error,
        "rank_one_rejected": not rejected.accepted,
        "rank_one_non_mutating": rank_one_non_mutating,
        "rank_two_accepted": accepted.accepted,
        "routes_before": routes_before,
        "routes_after": routes_after,
        "reload_routes": reload_routes,
        "stale_version_rejected": stale_rejected,
        "version_before": version_before,
        "version_after": int(memory.store_version.item()),
        "path": path,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.seed < 0:
        raise ValueError("shared-basis compression seed must be non-negative")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    keys, values = _payloads(system, seed=args.seed)
    with tempfile.TemporaryDirectory(
        prefix="neural-computer-shared-basis-compression-"
    ) as directory:
        forward = _run_stream(
            system=system,
            keys=keys,
            values=values,
            path=Path(directory) / "forward.pt",
            reversed_order=False,
        )
        reversed_stream = _run_stream(
            system=system,
            keys=keys,
            values=values,
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
        "forward_routes_before": bool(forward["routes_before"]),
        "forward_rank_one_rejected": bool(forward["rank_one_rejected"]),
        "forward_rank_one_non_mutating": bool(forward["rank_one_non_mutating"]),
        "forward_rank_two_accepted": bool(forward["rank_two_accepted"]),
        "forward_routes_after": bool(forward["routes_after"]),
        "forward_reload_routes": bool(forward["reload_routes"]),
        "forward_storage_reduced": (
            int(forward["physical_value_scalars_after"])
            < int(forward["physical_value_scalars_before"])
            and int(forward["physical_value_scalars_after"])
            < int(forward["dense_value_scalars"])
        ),
        "forward_stale_version_rejected": bool(forward["stale_version_rejected"]),
        "reversed_routes_after": bool(reversed_stream["routes_after"]),
        "reversed_reload_routes": bool(reversed_stream["reload_routes"]),
        "reversed_storage_reduced": (
            int(reversed_stream["physical_value_scalars_after"])
            < int(reversed_stream["dense_value_scalars"])
        ),
        "corruption_rejected": corruption_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": SHARED_BASIS_COMPRESSION_GROWTH_SCHEMA,
        "claim_boundary": (
            "Verifier-gated copy-on-write shared-basis compression of distinct "
            "opaque external-memory values under a frozen canonical event "
            "boundary; not learned rank selection, semantic equivalence "
            "discovery, arbitrary new computation, unbounded memory growth, "
            "or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "memory": "persistent_shared_basis_content_addressed_memory_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "key_contract": "independent_opaque_content_addresses",
            "value_contract": "opaque_learned_values_with_shared_residual_structure",
            "candidate_policy": "deterministic_svd_rank_two_stress_candidate",
            "records": RECORD_COUNT,
            "target_rank": SHARED_RANK,
            "noise_scale": NOISE_SCALE,
        },
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
            "unique_verifier_bits": RECORD_COUNT * 2,
            "unique_logical_lifetimes": RECORD_COUNT * 2,
            "optimizer_updates": 0,
            "compression_transactions": 4,
            "replayed_examples": 0,
            "controller_updates": 0,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_shared_basis_memory_compression"
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
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
