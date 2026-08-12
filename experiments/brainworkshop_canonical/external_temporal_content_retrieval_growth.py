"""Retrieve learned temporal capabilities from related content keys.

The query-address promotion proved that one external memory can bind several
offsets under one rendered cue.  This follow-up stores those learned opaque
capability addresses in the canonical append-only content-addressed backend
and probes them with nearby learned-event keys.  The controller, event
encoder, capability file, and route table remain frozen during retrieval.

The memory returns only an opaque capability address and an explicit hit bit;
physical rows, query perturbations, verifier rules, and target actions remain
outside the deployed controller boundary.
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
    AppendOnlyContentAddressedMemory,
    MemoryQuery,
    PersistentAppendOnlyContentAddressedMemory,
    PersistentOpaqueContextRouteEvidence,
)

from .external_temporal_offset_growth import ExternalTemporalCapabilityFile
from .external_temporal_query_address_growth import (
    EVENT_WIDTH,
    MAX_OFFSET,
    SOURCE_DEPTH,
    SOURCE_QUERY,
    TARGET_DEPTH,
    TARGET_QUERY,
    UNKNOWN_QUERY,
    _build,
    _calibrate_source_route,
    _evaluate,
    _train_source,
    _train_target_route,
)

CONTENT_RETRIEVAL_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-content-retrieval-growth.v1"
)
MASTERY_THRESHOLD = 0.80
READ_MATCH_THRESHOLD = 0.75
NOISE_SCALE = 0.20


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _event_key(system, symbol: int) -> torch.Tensor:
    encoder = system.agent.runtime.encoders["stimulus"]
    return encoder(torch.tensor([symbol], dtype=torch.long))[0].detach()


def _address_basis(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 8_101)
    return F.normalize(
        torch.randn(MAX_OFFSET, EVENT_WIDTH, generator=generator), dim=-1
    )


def _noisy_key(key: torch.Tensor, *, seed: int, scale: float = NOISE_SCALE) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 8_307)
    noise = torch.randn(key.shape, generator=generator) * scale
    return F.normalize(key + noise, dim=0)


def _read_address(
    memory: AppendOnlyContentAddressedMemory,
    key: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, object]:
    read = memory.read(MemoryQuery(key.unsqueeze(0), top_k=1))
    hit = bool(read.hit[0])
    if not hit:
        return {
            "hit": False,
            "score": float(read.scores[0, 0])
            if read.scores.shape[1]
            else float("-inf"),
            "resolved_offset": None,
        }
    value = F.normalize(read.value[0], dim=0)
    scores = basis @ value
    return {
        "hit": True,
        "score": float(read.scores[0, 0]),
        "resolved_offset": int(scores.argmax()) + 1,
    }


def _probe(
    system,
    file: ExternalTemporalCapabilityFile,
    evidence: PersistentOpaqueContextRouteEvidence,
    memory: AppendOnlyContentAddressedMemory,
    basis: torch.Tensor,
    *,
    label: str,
    key: torch.Tensor,
    query_symbol: int,
    depth: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    lifetimes: int,
) -> dict[str, object]:
    route = _read_address(memory, key, basis)
    offset = route["resolved_offset"]
    if offset is None:
        return {
            "label": label,
            "route": route,
            "accuracy": 0.5,
            "lifetimes": [],
        }
    rows = _evaluate(
        system,
        file,
        evidence,
        query_symbol=query_symbol,
        depth=depth,
        batch_size=batch_size,
        data_steps=data_steps,
        seed=seed,
        lifetimes=lifetimes,
        forced_offset=int(offset),
    )
    return {
        "label": label,
        "route": route,
        "accuracy": sum(float(row["accuracy"]) for row in rows) / len(rows),
        "lifetimes": rows,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.target_updates,
        args.route_calibration_lifetimes,
        args.batch_size,
        args.data_steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("content retrieval budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("content retrieval optimization parameters are invalid")
    if args.data_steps <= TARGET_DEPTH:
        raise ValueError("data steps must include n-back-5 target trials")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    file = ExternalTemporalCapabilityFile()
    evidence = PersistentOpaqueContextRouteEvidence(
        EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=MASTERY_THRESHOLD,
        min_mastery_observations=8,
    )
    for _ in range(MAX_OFFSET):
        evidence.append_slot()
    source_history = _train_source(
        system,
        file,
        updates=args.source_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 10_000,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
    )
    source_route_history = _calibrate_source_route(
        system,
        file,
        evidence,
        lifetimes=args.route_calibration_lifetimes,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 15_000,
    )
    source_before = _evaluate(
        system,
        file,
        evidence,
        query_symbol=SOURCE_QUERY,
        depth=SOURCE_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 20_000,
        lifetimes=args.retention_lifetimes,
    )
    file_digest_before_target = file.digest()
    for parameter in file.parameters():
        parameter.requires_grad_(False)
    target_history = _train_target_route(
        system,
        file,
        evidence,
        updates=args.target_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 30_000,
    )
    source_after = _evaluate(
        system,
        file,
        evidence,
        query_symbol=SOURCE_QUERY,
        depth=SOURCE_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 20_000,
        lifetimes=args.retention_lifetimes,
    )
    target_after = _evaluate(
        system,
        file,
        evidence,
        query_symbol=TARGET_QUERY,
        depth=TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 40_000,
        lifetimes=args.retention_lifetimes,
    )
    key_source = _event_key(system, SOURCE_QUERY)
    key_target = _event_key(system, TARGET_QUERY)
    key_unknown = _event_key(system, UNKNOWN_QUERY)
    basis = _address_basis(args.seed)
    source_offset = int(evidence.preferred_order(key_source)[0]) + 1
    target_offset = int(evidence.preferred_order(key_target)[0]) + 1
    keys = torch.stack((key_source, key_target))
    values = torch.stack((basis[source_offset - 1], basis[target_offset - 1]))
    strength = torch.ones(2)
    noisy_source = _noisy_key(key_source, seed=args.seed + 1)
    noisy_target = _noisy_key(key_target, seed=args.seed + 2)
    with tempfile.TemporaryDirectory(prefix="neural-computer-content-memory-") as directory:
        path = Path(directory) / "content-memory.pt"
        memory = PersistentAppendOnlyContentAddressedMemory(
            EVENT_WIDTH,
            path=path,
            write_threshold=0.0,
            write_match_threshold=0.999,
            read_match_threshold=READ_MATCH_THRESHOLD,
        )
        receipt = memory.write(keys, values, strength)
        exact_source = _probe(
            system,
            file,
            evidence,
            memory,
            basis,
            label="exact_source",
            key=key_source,
            query_symbol=SOURCE_QUERY,
            depth=SOURCE_DEPTH,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 50_000,
            lifetimes=args.retention_lifetimes,
        )
        exact_target = _probe(
            system,
            file,
            evidence,
            memory,
            basis,
            label="exact_target",
            key=key_target,
            query_symbol=TARGET_QUERY,
            depth=TARGET_DEPTH,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 60_000,
            lifetimes=args.retention_lifetimes,
        )
        noisy_source_result = _probe(
            system,
            file,
            evidence,
            memory,
            basis,
            label="noisy_source",
            key=noisy_source,
            query_symbol=SOURCE_QUERY,
            depth=SOURCE_DEPTH,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 70_000,
            lifetimes=args.retention_lifetimes,
        )
        noisy_target_result = _probe(
            system,
            file,
            evidence,
            memory,
            basis,
            label="noisy_target",
            key=noisy_target,
            query_symbol=TARGET_QUERY,
            depth=TARGET_DEPTH,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 80_000,
            lifetimes=args.retention_lifetimes,
        )
        unknown_route = _read_address(memory, key_unknown, basis)
        restored = PersistentAppendOnlyContentAddressedMemory(
            EVENT_WIDTH,
            path=path,
            write_threshold=0.0,
            write_match_threshold=0.999,
            read_match_threshold=READ_MATCH_THRESHOLD,
        )
        restored_noisy_source = _read_address(restored, noisy_source, basis)
        restored_noisy_target = _read_address(restored, noisy_target, basis)
        memory_record_count_before_clear = memory.record_count
        corrupted = False
        corrupt_path = Path(directory) / "corrupt-content-memory.pt"
        payload = torch.load(path, weights_only=False)
        payload["state_dict"]["values"][0, 0] += 0.25
        torch.save(payload, corrupt_path)
        try:
            PersistentAppendOnlyContentAddressedMemory(
                EVENT_WIDTH,
                path=corrupt_path,
                write_threshold=0.0,
                write_match_threshold=0.999,
                read_match_threshold=READ_MATCH_THRESHOLD,
            )
        except ValueError as error:
            corrupted = "checksum" in str(error).lower()
        memory.clear()
        cleared = _read_address(memory, key_source, basis)
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    source_file_digest = file.digest()
    gates = {
        "source_mastered_before_growth": min(
            float(row["accuracy"]) for row in source_before
        )
        >= MASTERY_THRESHOLD,
        "target_mastered_after_growth": min(
            float(row["accuracy"]) for row in target_after
        )
        >= MASTERY_THRESHOLD,
        "source_retained_after_growth": min(
            float(row["accuracy"]) for row in source_after
        )
        >= MASTERY_THRESHOLD,
        "two_routes_written": bool(receipt.committed.all())
        and memory_record_count_before_clear == 2,
        "exact_source_retrieves_correct_route": exact_source["route"]["resolved_offset"]
        == source_offset
        and float(exact_source["accuracy"]) >= MASTERY_THRESHOLD,
        "exact_target_retrieves_correct_route": exact_target["route"]["resolved_offset"]
        == target_offset
        and float(exact_target["accuracy"]) >= MASTERY_THRESHOLD,
        "noisy_source_retrieves_correct_route": noisy_source_result["route"][
            "resolved_offset"
        ]
        == source_offset
        and float(noisy_source_result["accuracy"]) >= MASTERY_THRESHOLD,
        "noisy_target_retrieves_correct_route": noisy_target_result["route"][
            "resolved_offset"
        ]
        == target_offset
        and float(noisy_target_result["accuracy"]) >= MASTERY_THRESHOLD,
        "unknown_key_is_not_claimed": not bool(unknown_route["hit"]),
        "clear_memory_removes_hits": not bool(cleared["hit"]),
        "reload_preserves_noisy_routes": restored_noisy_source
        == noisy_source_result["route"]
        and restored_noisy_target == noisy_target_result["route"],
        "corruption_rejected": corrupted,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "external_file_frozen_after_target": file_digest_before_target == source_file_digest,
        "zero_replayed_examples": True,
    }
    source_bits = args.batch_size * args.source_updates * (
        args.data_steps - SOURCE_DEPTH
    )
    calibration_bits = args.batch_size * args.route_calibration_lifetimes * (
        args.data_steps - SOURCE_DEPTH
    )
    target_bits = args.batch_size * args.target_updates * (
        args.data_steps - TARGET_DEPTH
    )
    report = {
        "schema": CONTENT_RETRIEVAL_SCHEMA,
        "claim_boundary": (
            "Outcome-only temporal capability growth followed by related-key "
            "content-addressed retrieval with persistence and corruption gates; "
            "not learned compression, unrestricted memory growth, arbitrary "
            "new computation, or general continual learning."
        ),
        "architecture": {
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "capability": "external_temporal_capability_file_frozen_after_target",
            "history_transport": (
                "canonical_runtime_external_history_event_bridge_v1"
            ),
            "history_causality": "read_before_current_append",
            "bridge_offset_semantics": (
                "logical_lag_minus_one_for_pre_append_relative_read"
            ),
            "address_memory": "persistent_append_only_content_addressed_memory_v1",
            "query": "learned_event_tensor_plus_related_noise",
            "read_match_threshold": READ_MATCH_THRESHOLD,
            "noise_scale": NOISE_SCALE,
            "source_offset": source_offset,
            "target_offset": target_offset,
        },
        "seed": args.seed,
        "source_history_tail": source_history[-5:],
        "source_route_history_tail": source_route_history[-5:],
        "target_history_tail": target_history[-5:],
        "evaluation": {
            "source_before": source_before,
            "source_after": source_after,
            "target_after": target_after,
            "exact_source": exact_source,
            "exact_target": exact_target,
            "noisy_source": noisy_source_result,
            "noisy_target": noisy_target_result,
            "unknown_route": unknown_route,
            "cleared_route": cleared,
            "reloaded_noisy_source": restored_noisy_source,
            "reloaded_noisy_target": restored_noisy_target,
            "memory_record_count_before_clear": memory_record_count_before_clear,
            "route_payload": evidence.payload(),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": source_bits + calibration_bits + target_bits,
            "audit_verifier_bits": sum(
                int(row["unique_verifier_bits"])
                for result in (
                    source_before,
                    source_after,
                    target_after,
                    exact_source["lifetimes"],
                    exact_target["lifetimes"],
                    noisy_source_result["lifetimes"],
                    noisy_target_result["lifetimes"],
                )
                for row in result
            ),
            "unique_logical_lifetimes": args.batch_size
            * (
                args.source_updates
                + args.route_calibration_lifetimes
                + args.target_updates
            ),
            "optimizer_updates": args.source_updates,
            "route_memory_updates": args.route_calibration_lifetimes
            + args.target_updates,
            "content_memory_writes": 2,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": source_bits + calibration_bits + target_bits
            if all(gates.values())
            else None,
        },
        "status": "promoted_temporal_content_retrieval_growth"
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
    parser.add_argument("--source-updates", type=int, default=512)
    parser.add_argument("--target-updates", type=int, default=512)
    parser.add_argument("--route-calibration-lifetimes", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-steps", type=int, default=14)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
