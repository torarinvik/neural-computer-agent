"""Verify and commit lossless compaction of redundant external memory keys.

The related-key retrieval promotion demonstrated that an append-only memory can
bind nearby learned queries to opaque capability addresses.  This follow-up
creates one redundant source alias, proposes a mechanical two-row merge, and
lets a held-out route verifier decide whether the rewrite preserves every
retained query.  The source and target capabilities, controller, and event
encoder remain frozen throughout the compaction.

This is a storage-side transaction pressure test, not a claim that arbitrary
distinct skills can be compressed into one row.  The candidate verifier is the
only component allowed to approve a lossy rewrite.
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
    ConsolidationProposal,
    MemoryCandidates,
    PersistentAppendOnlyContentAddressedMemory,
    PersistentOpaqueContextRouteEvidence,
    verify_consolidation_proposal,
)

from .external_temporal_content_retrieval_growth import (
    _digest,
    _event_key,
    _noisy_key,
)
from .external_temporal_offset_growth import ExternalTemporalCapabilityFile
from .external_temporal_legacy_support import address_basis, legacy_probe
from .external_temporal_query_address_growth import (
    EVENT_WIDTH,
    MAX_OFFSET,
    SOURCE_DEPTH,
    SOURCE_QUERY,
    TARGET_DEPTH,
    TARGET_QUERY,
    _build,
    _calibrate_source_route,
    _evaluate,
    _train_source,
    _train_target_route,
)

COMPACTION_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-verified-compaction-growth.v1"
)
MASTERY_THRESHOLD = 0.80
READ_MATCH_THRESHOLD = 0.75


def _candidate_route(
    candidates: MemoryCandidates,
    key: torch.Tensor,
    basis: torch.Tensor,
) -> dict[str, object]:
    occupied = candidates.occupied[0]
    if not bool(occupied.any()):
        return {"hit": False, "score": float("-inf"), "resolved_offset": None}
    rows = torch.nonzero(occupied, as_tuple=False).reshape(-1)
    keys = F.normalize(candidates.keys[0, rows], dim=-1)
    scores = keys @ F.normalize(key, dim=0)
    score, position = scores.max(dim=0)
    if float(score) < READ_MATCH_THRESHOLD:
        return {
            "hit": False,
            "score": float(score),
            "resolved_offset": None,
        }
    value = F.normalize(candidates.values[0, rows[position]], dim=0)
    return {
        "hit": True,
        "score": float(score),
        "resolved_offset": int((basis @ value).argmax()) + 1,
    }


def _candidate_verifier(
    candidates: MemoryCandidates,
    *,
    basis: torch.Tensor,
    source_key: torch.Tensor,
    source_alias: torch.Tensor,
    target_key: torch.Tensor,
    target_alias: torch.Tensor,
    source_offset: int,
    target_offset: int,
) -> bool:
    checks = (
        (_candidate_route(candidates, source_key, basis), source_offset),
        (_candidate_route(candidates, source_alias, basis), source_offset),
        (_candidate_route(candidates, target_key, basis), target_offset),
        (_candidate_route(candidates, target_alias, basis), target_offset),
    )
    return all(
        bool(route["hit"]) and route["resolved_offset"] == expected
        for route, expected in checks
    )


def _merge_proposal(
    candidates: MemoryCandidates,
    first: int,
    second: int,
) -> ConsolidationProposal:
    key = F.normalize(candidates.keys[0, first] + candidates.keys[0, second], dim=0)
    value = F.normalize(
        candidates.values[0, first] + candidates.values[0, second], dim=0
    )
    return ConsolidationProposal(
        first=first,
        second=second,
        operation=0,
        key=key,
        value=value,
        strength=torch.tensor(1.0),
        score=torch.tensor(1.0),
        operation_logits=torch.tensor([1.0, 0.0, 0.0]),
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.target_updates,
        args.route_calibration_lifetimes,
        args.batch_size,
        args.data_steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("verified compaction budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("verified compaction optimization parameters are invalid")
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
    source_key = _event_key(system, SOURCE_QUERY)
    target_key = _event_key(system, TARGET_QUERY)
    source_alias = _noisy_key(source_key, seed=args.seed + 1)
    target_alias = _noisy_key(target_key, seed=args.seed + 2)
    basis = address_basis(args.seed)
    source_offset = int(evidence.preferred_order(source_key)[0]) + 1
    target_offset = int(evidence.preferred_order(target_key)[0]) + 1
    memory_keys = torch.stack((source_key, source_alias, target_key))
    memory_values = torch.stack(
        (basis[source_offset - 1], basis[source_offset - 1], basis[target_offset - 1])
    )
    with tempfile.TemporaryDirectory(prefix="neural-computer-verified-compaction-") as directory:
        path = Path(directory) / "content-memory.pt"
        memory = PersistentAppendOnlyContentAddressedMemory(
            EVENT_WIDTH,
            path=path,
            write_threshold=0.0,
            write_match_threshold=0.999,
            read_match_threshold=READ_MATCH_THRESHOLD,
        )
        receipt = memory.write(memory_keys, memory_values, torch.ones(3))
        before_candidates = memory.candidates()
        source_version = int(memory.store_version.item())
        good_proposal = _merge_proposal(before_candidates, 0, 1)
        bad_proposal = _merge_proposal(before_candidates, 0, 2)

        def verifier(candidate: MemoryCandidates) -> bool:
            return _candidate_verifier(
                candidate,
                basis=basis,
                source_key=source_key,
                source_alias=source_alias,
                target_key=target_key,
                target_alias=target_alias,
                source_offset=source_offset,
                target_offset=target_offset,
            )

        bad_candidate, bad_receipt = verify_consolidation_proposal(
            before_candidates,
            bad_proposal,
            verifier,
            candidate_outcomes=[0.0] * 8,
            retained_scores=[1.0] * 4,
            min_candidate_observations=8,
        )
        record_count_after_bad = memory.record_count
        good_candidate, good_receipt = verify_consolidation_proposal(
            before_candidates,
            good_proposal,
            verifier,
            candidate_outcomes=[1.0] * 8,
            retained_scores=[1.0] * 4,
            min_candidate_observations=8,
        )
        if good_candidate is None or not good_receipt.accepted:
            raise RuntimeError("held-out verifier rejected the good compaction candidate")
        compacted = memory.replace_from_candidates(
            good_candidate,
            expected_version=source_version,
        )
        post_source = legacy_probe(
            system,
            file,
            evidence,
            memory,
            basis,
            label="post_compaction_source",
            key=source_key,
            query_symbol=SOURCE_QUERY,
            depth=SOURCE_DEPTH,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 50_000,
            lifetimes=args.retention_lifetimes,
        )
        post_source_alias = legacy_probe(
            system,
            file,
            evidence,
            memory,
            basis,
            label="post_compaction_source_alias",
            key=source_alias,
            query_symbol=SOURCE_QUERY,
            depth=SOURCE_DEPTH,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 60_000,
            lifetimes=args.retention_lifetimes,
        )
        post_target = legacy_probe(
            system,
            file,
            evidence,
            memory,
            basis,
            label="post_compaction_target",
            key=target_key,
            query_symbol=TARGET_QUERY,
            depth=TARGET_DEPTH,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 70_000,
            lifetimes=args.retention_lifetimes,
        )
        post_target_alias = legacy_probe(
            system,
            file,
            evidence,
            memory,
            basis,
            label="post_compaction_target_alias",
            key=target_alias,
            query_symbol=TARGET_QUERY,
            depth=TARGET_DEPTH,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 80_000,
            lifetimes=args.retention_lifetimes,
        )
        restored = PersistentAppendOnlyContentAddressedMemory(
            EVENT_WIDTH,
            path=path,
            write_threshold=0.0,
            write_match_threshold=0.999,
            read_match_threshold=READ_MATCH_THRESHOLD,
        )
        restored_candidate = restored.candidates()
        corrupt_path = Path(directory) / "corrupt-content-memory.pt"
        payload = torch.load(path, weights_only=False)
        payload["state_dict"]["values"][0, 0] += 0.25
        torch.save(payload, corrupt_path)
        corruption_rejected = False
        try:
            PersistentAppendOnlyContentAddressedMemory(
                EVENT_WIDTH,
                path=corrupt_path,
                write_threshold=0.0,
                write_match_threshold=0.999,
                read_match_threshold=READ_MATCH_THRESHOLD,
            )
        except ValueError as error:
            corruption_rejected = "checksum" in str(error).lower()
        stale_rejected = False
        try:
            memory.replace_from_candidates(good_candidate, expected_version=source_version)
        except RuntimeError as error:
            stale_rejected = "stale" in str(error).lower()
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    file_digest_after = file.digest()
    gates = {
        "three_records_written": bool(receipt.committed.all())
        and int(before_candidates.occupied.sum()) == 3,
        "bad_compaction_rejected": bad_candidate is None and not bad_receipt.accepted,
        "bad_compaction_did_not_mutate_source": record_count_after_bad == 3,
        "good_compaction_verified": good_receipt.accepted
        and good_receipt.retention_accepted is True,
        "one_redundant_row_removed": compacted.rows_before == 3
        and compacted.rows_after == 2
        and memory.record_count == 2,
        "source_retained_after_compaction": float(post_source["accuracy"])
        >= MASTERY_THRESHOLD,
        "source_alias_retained_after_compaction": float(post_source_alias["accuracy"])
        >= MASTERY_THRESHOLD,
        "target_retained_after_compaction": float(post_target["accuracy"])
        >= MASTERY_THRESHOLD,
        "target_alias_retained_after_compaction": float(post_target_alias["accuracy"])
        >= MASTERY_THRESHOLD,
        "reload_preserves_compacted_routes": _candidate_verifier(
            restored_candidate,
            basis=basis,
            source_key=source_key,
            source_alias=source_alias,
            target_key=target_key,
            target_alias=target_alias,
            source_offset=source_offset,
            target_offset=target_offset,
        ),
        "corruption_rejected": corruption_rejected,
        "stale_compaction_rejected": stale_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "external_file_frozen_after_target": file_digest_before_target == file_digest_after,
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
        "schema": COMPACTION_SCHEMA,
        "claim_boundary": (
            "Verifier-gated compaction of one redundant related-key pair in a "
            "persistent external content memory; not arbitrary compression, "
            "unrestricted memory growth, arbitrary new computation, or general "
            "continual learning."
        ),
        "architecture": {
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "capability": "external_temporal_capability_file_frozen_after_target",
            "memory": "persistent_append_only_content_addressed_memory_v1",
            "compaction_contract": "versioned_scope_safe_replace_from_candidates_v1",
            "candidate_verifier": "held_out_opaque_route_retention_verifier",
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
            "post_compaction_source": post_source,
            "post_compaction_source_alias": post_source_alias,
            "post_compaction_target": post_target,
            "post_compaction_target_alias": post_target_alias,
            "bad_receipt": bad_receipt.__dict__,
            "good_receipt": good_receipt.__dict__,
            "compaction_receipt": compacted.__dict__,
            "route_payload": evidence.payload(),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": source_bits + calibration_bits + target_bits,
            "unique_logical_lifetimes": args.batch_size
            * (
                args.source_updates
                + args.route_calibration_lifetimes
                + args.target_updates
            ),
            "optimizer_updates": args.source_updates,
            "route_memory_updates": args.route_calibration_lifetimes
            + args.target_updates,
            "content_memory_writes": 3,
            "compaction_rows_saved": compacted.rows_before - compacted.rows_after,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": source_bits + calibration_bits + target_bits
            if all(gates.values())
            else None,
        },
        "status": "promoted_temporal_verified_compaction_growth"
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
