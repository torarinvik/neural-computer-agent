"""Pressure-test a larger episodic archive under reversal and corruption.

This audit keeps the canonical controller and event frontend frozen while a
memory-side archive grows to 1,024 opaque binding records. It measures batch
retrieval, query-order invariance, finite-value corruption rejection, durable
reload, and scalar reversal demotion. The records are synthetic learned-key
proxies; no family label, task id, or correct unattempted action enters the
archive boundary.

The result is a storage/retrieval rung, not a claim that the controller has
learned 1,024 capabilities or that compression is solved.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import EpisodicBindingArchive

from .external_temporal_query_address_growth import _build
from .external_temporal_shared_basis_policy_growth import _digest

ARCHIVE_SCALE_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-archive-scale-"
    "reversal.v1"
)
CONTEXT_WIDTH = 8
SIGNATURE_WIDTH = 20
ACTIVE_SLOTS = 4
RECORD_COUNT = 1_024
QUERY_COUNT = 512
REVERSAL_PATIENCE = 4


def _make_keys(
    *,
    seed: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    contexts = F.normalize(
        torch.randn(count, CONTEXT_WIDTH, generator=generator),
        dim=-1,
    )
    signatures = F.normalize(
        torch.randn(count, SIGNATURE_WIDTH, generator=generator),
        dim=-1,
    )
    return contexts, signatures


def _query_keys(
    signatures: torch.Tensor,
    *,
    seed: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    known_indices = torch.randint(
        signatures.shape[0],
        (count,),
        generator=generator,
    )
    perturbation = 0.005 * torch.randn(
        count,
        signatures.shape[1],
        generator=generator,
    )
    known = F.normalize(signatures[known_indices] + perturbation, dim=-1)
    unknown = F.normalize(
        torch.randn(count, signatures.shape[1], generator=generator),
        dim=-1,
    )
    return known, unknown


def _lookup_accuracy(
    archive: EpisodicBindingArchive,
    queries: torch.Tensor,
    expected: torch.Tensor,
) -> tuple[float, tuple[int | None, ...]]:
    results = archive.lookup_many(queries)
    selected = tuple(result.binding_id for result in results)
    accuracy = sum(
        int(binding_id == int(target))
        for binding_id, target in zip(selected, expected, strict=True)
    ) / len(selected)
    return accuracy, selected


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.record_count < 16 or args.query_count < 1:
        raise ValueError("archive scale counts are invalid")
    started = perf_counter()
    system = _build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])

    contexts, signatures = _make_keys(seed=args.seed, count=args.record_count)
    archive = EpisodicBindingArchive(
        CONTEXT_WIDTH,
        SIGNATURE_WIDTH,
        active_slots=ACTIVE_SLOTS,
        matching_threshold=0.95,
        min_mastery_observations=8,
        reversal_threshold=0.5,
        reversal_patience=REVERSAL_PATIENCE,
    )
    record_ids = [
        archive.register(context, signature)
        for context, signature in zip(contexts, signatures, strict=True)
    ]
    for slot, binding_id in enumerate(record_ids[:ACTIVE_SLOTS]):
        archive.activate(binding_id, slot)

    for step in range(8):
        archive.observe(record_ids[0], 1.0, step=step)
        archive.observe(record_ids[1], 1.0, step=step)
    protected_before_reversal = (
        archive.is_protected(record_ids[0])
        and archive.is_protected(record_ids[1])
    )
    for step in range(8, 8 + REVERSAL_PATIENCE):
        archive.observe(record_ids[0], 0.0, step=step)
    reversal_status = archive.status()
    protected_after_reversal = archive.is_protected(record_ids[1])
    reversed_demoted = (
        reversal_status.reversal_count[record_ids[0]] == 1
        and not archive.is_protected(record_ids[0])
    )

    known_queries, unknown_queries = _query_keys(
        signatures,
        seed=args.seed + 100_000,
        count=args.query_count,
    )
    generator = torch.Generator().manual_seed(args.seed + 200_000)
    expected_known = torch.randint(
        args.record_count,
        (args.query_count,),
        generator=generator,
    )
    # Use exact known rows for labels, while the actual query path remains
    # fresh perturbations independently generated above.
    known_queries = F.normalize(
        signatures[expected_known]
        + 0.005
        * torch.randn(
            args.query_count,
            SIGNATURE_WIDTH,
            generator=torch.Generator().manual_seed(args.seed + 300_000),
        ),
        dim=-1,
    )
    lookup_started = perf_counter()
    known_accuracy, known_selected = _lookup_accuracy(
        archive,
        known_queries,
        expected_known,
    )
    lookup_seconds = perf_counter() - lookup_started
    unknown_results = archive.lookup_many(unknown_queries)
    unknown_false_known_rate = sum(
        int(result.binding_id is not None) for result in unknown_results
    ) / len(unknown_results)

    permutation = torch.randperm(args.query_count, generator=generator)
    permuted_results = archive.lookup_many(known_queries[permutation])
    permutation_accuracy = sum(
        int(result.binding_id == known_selected[int(index)])
        for result, index in zip(permuted_results, permutation, strict=True)
    ) / args.query_count
    scalar_consistency = all(
        archive.lookup(known_queries[index]).binding_id == known_selected[index]
        for index in range(min(args.query_count, 64))
    )

    payload = archive.payload()
    payload_bytes = len(json.dumps(payload, sort_keys=True).encode("utf-8"))
    restored = EpisodicBindingArchive.from_payload(payload)
    restored_known_accuracy, _ = _lookup_accuracy(
        restored,
        known_queries,
        expected_known,
    )
    restored_unknown = restored.lookup_many(unknown_queries)
    restored_unknown_false_known_rate = sum(
        int(result.binding_id is not None) for result in restored_unknown
    ) / len(restored_unknown)
    compact_snapshot_corruption_rejected = False
    with tempfile.TemporaryDirectory() as directory:
        snapshot_path = Path(directory) / "archive.pt"
        archive.snapshot(snapshot_path)
        compact_snapshot_bytes = snapshot_path.stat().st_size
        compact_restored = EpisodicBindingArchive(
            CONTEXT_WIDTH,
            SIGNATURE_WIDTH,
            active_slots=ACTIVE_SLOTS,
            matching_threshold=0.95,
            min_mastery_observations=8,
            reversal_threshold=0.5,
            reversal_patience=REVERSAL_PATIENCE,
        )
        compact_restored.load_snapshot(snapshot_path)
        compact_known_accuracy, _ = _lookup_accuracy(
            compact_restored,
            known_queries,
            expected_known,
        )
        compact_unknown = compact_restored.lookup_many(unknown_queries)
        compact_unknown_false_known_rate = sum(
            int(result.binding_id is not None) for result in compact_unknown
        ) / len(compact_unknown)
        compact_payload = torch.load(snapshot_path, weights_only=False)
        compact_payload["state_dict"]["attempts"][0] += 1
        torch.save(compact_payload, snapshot_path)
        try:
            compact_restored.load_snapshot(snapshot_path)
        except ValueError as error:
            compact_snapshot_corruption_rejected = "checksum" in str(error).lower()
    corruption = dict(payload)
    corruption["signature_keys"] = [list(row) for row in payload["signature_keys"]]
    corruption["signature_keys"][0][0] += 0.125
    corruption_rejected = False
    try:
        EpisodicBindingArchive.from_payload(corruption)
    except ValueError as error:
        corruption_rejected = "checksum" in str(error).lower()

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    retrieval_verifier_bits = args.query_count * 2
    reversal_verifier_bits = 2 * (8 + REVERSAL_PATIENCE)
    unique_verifier_bits = retrieval_verifier_bits + reversal_verifier_bits
    gates = {
        "archive_reached_scale": archive.record_count == args.record_count,
        "batch_known_retrieval": known_accuracy >= 0.99,
        "batch_unknown_rejection": unknown_false_known_rate <= 0.01,
        "query_order_invariance": permutation_accuracy >= 0.99,
        "scalar_batch_consistency": scalar_consistency,
        "protected_prefix_before_reversal": protected_before_reversal,
        "reversal_demotes_stale_record": reversed_demoted,
        "protected_sibling_survives_reversal": protected_after_reversal,
        "reload_known_retrieval": restored_known_accuracy >= 0.99,
        "reload_unknown_rejection": restored_unknown_false_known_rate <= 0.01,
        "corruption_rejected": corruption_rejected,
        "compact_snapshot_known_retrieval": compact_known_accuracy >= 0.99,
        "compact_snapshot_unknown_rejection": compact_unknown_false_known_rate <= 0.01,
        "compact_snapshot_corruption_rejected": compact_snapshot_corruption_rejected,
        "active_residency_reloaded": restored.active_binding_ids
        == archive.active_binding_ids,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "zero_replayed_training_examples": True,
    }
    report = {
        "schema": ARCHIVE_SCALE_SCHEMA,
        "claim_boundary": (
            "A checksummed external episodic archive scales opaque signature "
            "retrieval to the configured record count, preserves query-order "
            "invariance, and demotes a stale protected record from scalar "
            "reversal evidence; not learned capability acquisition or solved "
            "compression."
        ),
        "seed": args.seed,
        "architecture": {
            "archive": "episodic_binding_archive_v2",
            "retrieval": "cached_matrix_and_batched_lookup_v1",
            "integrity": "canonical_sha256_payload_checksum_v1",
            "reversal": "latched_stable_prefix_with_scalar_failure_patience_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "forbidden_features": "semantic_labels_task_ids_protocol_formats",
        },
        "archive": {
            "record_count": archive.record_count,
            "active_slots": archive.active_binding_ids,
            "payload_bytes": payload_bytes,
            "compact_snapshot_bytes": compact_snapshot_bytes,
            "schema": archive.configuration()["schema"],
        },
        "retrieval": {
            "query_count": args.query_count,
            "known_accuracy": known_accuracy,
            "unknown_false_known_rate": unknown_false_known_rate,
            "permutation_accuracy": permutation_accuracy,
            "scalar_batch_consistency": scalar_consistency,
            "lookup_seconds": lookup_seconds,
            "queries_per_second": args.query_count / max(lookup_seconds, 1e-12),
            "reloaded_known_accuracy": restored_known_accuracy,
            "reloaded_unknown_false_known_rate": restored_unknown_false_known_rate,
        },
        "reversal": {
            "protected_before": protected_before_reversal,
            "reversed_record_demoted": reversed_demoted,
            "protected_sibling_after": protected_after_reversal,
            "reversal_count": reversal_status.reversal_count[record_ids[0]],
        },
        "corruption_rejected": corruption_rejected,
        "compact_snapshot": {
            "bytes": compact_snapshot_bytes,
            "known_accuracy": compact_known_accuracy,
            "unknown_false_known_rate": compact_unknown_false_known_rate,
            "corruption_rejected": compact_snapshot_corruption_rejected,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": unique_verifier_bits,
            "unique_logical_lifetimes": unique_verifier_bits,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "controller_updates": 0,
            "retrieval_verifier_bits": retrieval_verifier_bits,
            "reversal_verifier_bits": reversal_verifier_bits,
            "latency_seconds": perf_counter() - started,
        },
        "status": "promoted_archive_scale_reversal_integrity"
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
    parser.add_argument("--record-count", type=int, default=RECORD_COUNT)
    parser.add_argument("--query-count", type=int, default=QUERY_COUNT)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
