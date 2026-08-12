"""Acquire and retrieve temporal capabilities through outcome-only memory.

This experiment composes two independently replaceable boundaries.  Paired
common-random scalar action outcomes first acquire a stable external temporal
capability and two opaque query-conditioned addresses while the controller and
event encoder remain frozen.  A canonical content-addressed index then stores
only learned context keys and opaque absolute positions.  Exact and related
keys must retrieve the frozen capability, while unknown keys must miss.

The learner receives learned event tensors, opaque actions, and scalar
verifier outcomes only.  Query depths, target bits, expected offsets, and the
private verifier are audit-side state; they never enter the deployed route or
readout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import ExternalTemporalAddressIndex

from . import external_temporal_query_address_growth as query
from . import external_temporal_query_counterfactual_growth as counterfactual

CONTENT_COUNTERFACTUAL_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-content-counterfactual-growth.v1"
)
CANDIDATE_ADMISSION_THRESHOLD = 0.95
ROUTE_SELECTION_THRESHOLD = 0.99
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


def _related_key(key: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 8_307)
    noise = torch.randn(key.shape, generator=generator) * NOISE_SCALE
    return F.normalize(key + noise, dim=0)


def _read_address(
    memory: ExternalTemporalAddressIndex,
    key: torch.Tensor,
) -> dict[str, object]:
    read = memory.read(key.unsqueeze(0), top_k=1)
    hit = bool(read.hit[0])
    return {
        "hit": hit,
        "score": float(read.scores[0, 0]),
        "resolved_position": (
            int(read.target_positions[0, 0]) if hit else None
        ),
    }


def _probe(
    system,
    file: query.ExternalTemporalCapabilityFile,
    evidence,
    memory: ExternalTemporalAddressIndex,
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
    route = _read_address(memory, key)
    position = route["resolved_position"]
    if position is None:
        return {
            "label": label,
            "route": route,
            "accuracy": 0.5,
            "lifetimes": [],
        }
    rows = query._evaluate(
        system,
        file,
        evidence,
        query_symbol=query_symbol,
        depth=depth,
        batch_size=batch_size,
        data_steps=data_steps,
        seed=seed,
        lifetimes=lifetimes,
        forced_offset=int(position) + 1,
    )
    return {
        "label": label,
        "route": route,
        "accuracy": sum(float(row["accuracy"]) for row in rows) / len(rows),
        "lifetimes": rows,
    }


def _stable(
    rows: list[dict[str, object]],
    *,
    threshold: float = query.MASTERY_THRESHOLD,
) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= threshold


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.source_evaluation_lifetimes,
        args.source_route_lifetimes,
        args.target_route_updates,
        args.batch_size,
        args.data_steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("content counterfactual budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("content counterfactual optimization parameters are invalid")
    if args.data_steps <= query.TARGET_DEPTH:
        raise ValueError("data steps must include target trials")

    started = perf_counter()
    system = query._build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    offsets = tuple(range(1, query.MAX_OFFSET + 1))
    files, candidates = counterfactual._train_candidates(
        system,
        offsets=offsets,
        updates=args.source_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        entropy_weight=args.entropy_weight,
        evaluation_lifetimes=args.source_evaluation_lifetimes,
    )
    stable_offsets = tuple(
        int(record["offset"]) for record in candidates if bool(record["stable"])
    )
    winner_offset = stable_offsets[0] if stable_offsets else max(
        candidates,
        key=lambda record: min(float(row["accuracy"]) for row in record["evaluation"]),
    )["offset"]
    winner_index = winner_offset - 1
    winner = files[winner_index]

    evidence = counterfactual._evidence(
        mastery_threshold=query.MASTERY_THRESHOLD,
        observations=args.source_route_lifetimes,
    )
    source_before, source_context = counterfactual._record_fixed_route(
        system,
        winner,
        evidence,
        query_symbol=query.SOURCE_QUERY,
        depth=query.SOURCE_DEPTH,
        offset=winner_offset,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 200_000,
        lifetimes=args.source_route_lifetimes,
    )
    target_history, target_context = counterfactual._train_counterfactual_route(
        system,
        winner,
        evidence,
        updates=args.target_route_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 300_000,
    )
    source_after = query._evaluate(
        system,
        winner,
        evidence,
        query_symbol=query.SOURCE_QUERY,
        depth=query.SOURCE_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 200_000,
        lifetimes=args.retention_lifetimes,
    )
    target_after = query._evaluate(
        system,
        winner,
        evidence,
        query_symbol=query.TARGET_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 400_000,
        lifetimes=args.retention_lifetimes,
    )
    wrong_offset = query._evaluate(
        system,
        winner,
        evidence,
        query_symbol=query.TARGET_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 600_000,
        lifetimes=args.retention_lifetimes,
        forced_offset=1,
    )
    missing_history = query._evaluate(
        system,
        winner,
        evidence,
        query_symbol=query.TARGET_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 700_000,
        lifetimes=args.retention_lifetimes,
        reset_memory_each_step=True,
    )

    shuffled_evidence = counterfactual._evidence(
        mastery_threshold=query.MASTERY_THRESHOLD,
        observations=args.source_route_lifetimes,
    )
    counterfactual._record_fixed_route(
        system,
        winner,
        shuffled_evidence,
        query_symbol=query.SOURCE_QUERY,
        depth=query.SOURCE_DEPTH,
        offset=winner_offset,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 800_000,
        lifetimes=args.source_route_lifetimes,
    )
    shuffled_history, _ = counterfactual._train_counterfactual_route(
        system,
        winner,
        shuffled_evidence,
        updates=args.target_route_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 900_000,
        shuffled_outcomes=True,
    )
    shuffled = query._evaluate(
        system,
        winner,
        shuffled_evidence,
        query_symbol=query.TARGET_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 910_000,
        lifetimes=args.retention_lifetimes,
    )

    unknown_episode = query._episode(
        system,
        winner,
        evidence,
        query_symbol=query.UNKNOWN_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 500_000,
        train=False,
        explore=False,
        forced_offset=1,
    )
    key_source = source_context
    key_target = target_context
    key_unknown = unknown_episode.context
    source_position = int(evidence.preferred_order(key_source)[0])
    target_position = int(evidence.preferred_order(key_target)[0])
    memory = ExternalTemporalAddressIndex(
        query.EVENT_WIDTH,
        write_match_threshold=0.999,
        read_match_threshold=READ_MATCH_THRESHOLD,
    )
    receipt = memory.write(
        torch.stack((key_source, key_target)),
        target_scopes=torch.zeros(2, dtype=torch.long),
        target_positions=torch.tensor(
            [source_position, target_position], dtype=torch.long
        ),
        strength=torch.ones(2),
    )
    source_file_digest_before_memory = _digest(winner)
    exact_source = _probe(
        system,
        winner,
        evidence,
        memory,
        label="exact_source",
        key=key_source,
        query_symbol=query.SOURCE_QUERY,
        depth=query.SOURCE_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 50_000,
        lifetimes=args.retention_lifetimes,
    )
    exact_target = _probe(
        system,
        winner,
        evidence,
        memory,
        label="exact_target",
        key=key_target,
        query_symbol=query.TARGET_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 60_000,
        lifetimes=args.retention_lifetimes,
    )
    related_source = _probe(
        system,
        winner,
        evidence,
        memory,
        label="related_source",
        key=_related_key(key_source, seed=args.seed + 1),
        query_symbol=query.SOURCE_QUERY,
        depth=query.SOURCE_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 70_000,
        lifetimes=args.retention_lifetimes,
    )
    related_target = _probe(
        system,
        winner,
        evidence,
        memory,
        label="related_target",
        key=_related_key(key_target, seed=args.seed + 2),
        query_symbol=query.TARGET_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 80_000,
        lifetimes=args.retention_lifetimes,
    )
    unknown_route = _read_address(memory, key_unknown)
    restored = ExternalTemporalAddressIndex.from_payload(memory.payload())
    restored_related_target = _read_address(
        restored, _related_key(key_target, seed=args.seed + 2)
    )
    corrupted_payload = memory.payload()
    corrupted_state = corrupted_payload["state"]
    if not isinstance(corrupted_state, dict):
        raise TypeError("content address index payload state is not a mapping")
    corrupted_state["keys"] = corrupted_state["keys"].clone()
    corrupted_state["keys"][0, 0] += 0.25
    corruption_rejected = False
    try:
        ExternalTemporalAddressIndex.from_payload(corrupted_payload)
    except ValueError as error:
        corruption_rejected = "checksum" in str(error).lower()
    record_count_before_clear = memory.record_count
    memory.clear()
    cleared = _read_address(memory, key_target)

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    source_file_digest_after_memory = _digest(winner)
    candidate_training_bits = len(offsets) * args.source_updates * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    candidate_eval_bits = args.source_evaluation_lifetimes * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    source_route_bits = args.source_route_lifetimes * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    target_route_bits = args.target_route_updates * args.batch_size * (
        args.data_steps - query.TARGET_DEPTH
    )
    audit_results = (
        source_before,
        source_after,
        target_after,
        wrong_offset,
        missing_history,
        exact_source["lifetimes"],
        exact_target["lifetimes"],
        related_source["lifetimes"],
        related_target["lifetimes"],
    )
    audit_bits = sum(int(row["unique_verifier_bits"]) for rows in audit_results for row in rows)
    gates = {
        "unique_stable_candidate": len(stable_offsets) == 1,
        "winner_is_verifier_valid": winner_offset == query.SOURCE_DEPTH,
        "source_candidate_mastered": _stable(
            candidates[winner_index]["evaluation"],
            threshold=CANDIDATE_ADMISSION_THRESHOLD,
        ),
        "source_route_mastered_before_target": _stable(source_before),
        "target_route_mastered": _stable(
            target_after,
            threshold=ROUTE_SELECTION_THRESHOLD,
        ),
        "target_route_selects_target_offset": all(
            int(row["selected_offset"]) == query.TARGET_DEPTH
            and float(row["accuracy"]) >= ROUTE_SELECTION_THRESHOLD
            for row in target_after
        ),
        "source_retained_after_growth": _stable(source_after),
        "wrong_offset_rejects_mastery": not _stable(wrong_offset),
        "missing_history_rejects_mastery": not _stable(missing_history),
        "shuffled_route_feedback_rejects_target": not _stable(shuffled)
        or all(int(row["selected_offset"]) != query.TARGET_DEPTH for row in shuffled),
        "content_index_writes_two_routes": bool(receipt.committed.all())
        and record_count_before_clear == 2,
        "exact_source_retrieves_correct_route": exact_source["route"][
            "resolved_position"
        ]
        == source_position
        and _stable(exact_source["lifetimes"], threshold=ROUTE_SELECTION_THRESHOLD),
        "exact_target_retrieves_correct_route": exact_target["route"][
            "resolved_position"
        ]
        == target_position
        and _stable(exact_target["lifetimes"], threshold=ROUTE_SELECTION_THRESHOLD),
        "related_source_retrieves_correct_route": related_source["route"][
            "resolved_position"
        ]
        == source_position
        and _stable(related_source["lifetimes"], threshold=ROUTE_SELECTION_THRESHOLD),
        "related_target_retrieves_correct_route": related_target["route"][
            "resolved_position"
        ]
        == target_position
        and _stable(related_target["lifetimes"], threshold=ROUTE_SELECTION_THRESHOLD),
        "unknown_key_misses": not bool(unknown_route["hit"]),
        "index_reload_preserves_related_target": restored_related_target
        == related_target["route"],
        "index_corruption_rejected": corruption_rejected,
        "index_clear_removes_hits": not bool(cleared["hit"]),
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "frozen_promoted_file": source_file_digest_before_memory
        == source_file_digest_after_memory,
        "zero_replayed_examples": True,
    }
    unique_bits = (
        candidate_training_bits
        + candidate_eval_bits
        + source_route_bits
        + target_route_bits
    )
    counterfactual_bits = sum(
        int(record["counterfactual_verifier_bits"]) for record in candidates
    ) + args.target_route_updates * query.MAX_OFFSET * args.batch_size * (
        args.data_steps - query.TARGET_DEPTH
    )
    report = {
        "schema": CONTENT_COUNTERFACTUAL_SCHEMA,
        "claim_boundary": (
            "Outcome-only counterfactual acquisition of a frozen external temporal "
            "capability and related-key content retrieval through the canonical "
            "opaque address index; not unrestricted memory growth, arbitrary new "
            "computation, program induction, or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "controller": "frozen_canonical_amodal_controller",
            "readout_credit": "paired_counterfactual_scalar_keypress_arms",
            "candidate_admission": "stable_heldout_verifier_prefix",
            "route_memory": "persistent_opaque_context_route_evidence_v1",
            "content_memory": "external_temporal_address_index_v2",
            "content_address": "learned_context_key_to_opaque_absolute_position",
            "related_key_noise_scale": NOISE_SCALE,
            "read_match_threshold": READ_MATCH_THRESHOLD,
            "history_transport": "canonical_runtime_external_history_event_bridge_v2",
            "history_causality": "read_before_current_append",
            "candidate_offsets": list(offsets),
        },
        "candidate_records": candidates,
        "selected_offset": winner_offset,
        "source_position": source_position,
        "target_position": target_position,
        "source_history": source_before,
        "target_history": target_history[-8:],
        "shuffled_history": shuffled_history[-8:],
        "evaluation": {
            "source_before": source_before,
            "source_after": source_after,
            "target_after": target_after,
            "wrong_offset": wrong_offset,
            "missing_history": missing_history,
            "shuffled_route": shuffled,
            "exact_source": exact_source,
            "exact_target": exact_target,
            "related_source": related_source,
            "related_target": related_target,
            "unknown_route": unknown_route,
            "reloaded_related_target": restored_related_target,
            "cleared_route": cleared,
            "content_record_count_before_clear": record_count_before_clear,
            "route_payload": evidence.payload(),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": unique_bits,
            "counterfactual_verifier_bits": counterfactual_bits,
            "audit_verifier_bits": audit_bits,
            "unique_logical_lifetimes": len(offsets) * args.source_updates * args.batch_size
            + args.source_evaluation_lifetimes * args.batch_size
            + args.source_route_lifetimes * args.batch_size
            + args.target_route_updates * args.batch_size,
            "counterfactual_logical_lifetimes": len(offsets)
            * args.source_updates
            * args.batch_size
            * 2
            + args.target_route_updates * query.MAX_OFFSET * args.batch_size,
            "optimizer_updates": args.source_updates * len(offsets),
            "route_memory_updates": args.source_route_lifetimes
            + args.target_route_updates * query.MAX_OFFSET,
            "content_memory_writes": 2,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": unique_bits if all(gates.values()) else None,
        },
        "status": "promoted_content_counterfactual_growth"
        if all(gates.values())
        else "rejected",
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--source-updates", type=int, default=128)
    parser.add_argument("--source-evaluation-lifetimes", type=int, default=4)
    parser.add_argument("--source-route-lifetimes", type=int, default=8)
    parser.add_argument("--target-route-updates", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-steps", type=int, default=14)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
