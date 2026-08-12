"""Acquire a temporal address through outcome-only counterfactual search.

The readout and address are deliberately separated. Fresh external candidate
files each train against one opaque relative-address arm using paired scalar
keypress outcomes. Only a candidate that clears a stable held-out verifier
prefix is promoted. The promoted file is then frozen while a context-keyed
route ledger acquires a second address by evaluating all opaque offset arms on
fresh common-random verifier episodes.

No n-back depth, correct action, target bit, or semantic query label enters the
learner. Counterfactual arms and their pairing metadata are trainer-only; the
deployed controller sees ordinary learned events, opaque actions, and scalar
feedback only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from neural_computer import PersistentOpaqueContextRouteEvidence

from . import external_temporal_query_address_growth as query

COUNTERFACTUAL_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-query-counterfactual-growth.v1"
)
CANDIDATE_ADMISSION_THRESHOLD = 0.95
ROUTE_SELECTION_THRESHOLD = 0.99


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _evidence(*, mastery_threshold: float, observations: int) -> PersistentOpaqueContextRouteEvidence:
    evidence = PersistentOpaqueContextRouteEvidence(
        query.EVENT_WIDTH,
        matching_tolerance=1e-5,
        mastery_threshold=mastery_threshold,
        min_mastery_observations=observations,
    )
    for _ in range(query.MAX_OFFSET):
        evidence.append_slot()
    return evidence


def _train_candidates(
    system,
    *,
    offsets: tuple[int, ...],
    updates: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    learning_rate: float,
    entropy_weight: float,
    evaluation_lifetimes: int,
) -> tuple[tuple[query.ExternalTemporalCapabilityFile, ...], list[dict[str, object]]]:
    files: list[query.ExternalTemporalCapabilityFile] = []
    records: list[dict[str, object]] = []
    for offset in offsets:
        file = query.ExternalTemporalCapabilityFile()
        history = query._train_source(
            system,
            file,
            updates=updates,
            batch_size=batch_size,
            data_steps=data_steps,
            seed=seed + offset * 10_007,
            learning_rate=learning_rate,
            entropy_weight=entropy_weight,
            credit_mode="paired_counterfactual",
            forced_offset=offset,
        )
        for parameter in file.parameters():
            parameter.requires_grad_(False)
        evaluation = query._evaluate(
            system,
            file,
            _evidence(mastery_threshold=query.MASTERY_THRESHOLD, observations=1),
            query_symbol=query.SOURCE_QUERY,
            depth=query.SOURCE_DEPTH,
            batch_size=batch_size,
            data_steps=data_steps,
            seed=seed + 100_000,
            lifetimes=evaluation_lifetimes,
            forced_offset=offset,
        )
        records.append(
            {
                "offset": offset,
                "history_tail": history[-5:],
                "counterfactual_verifier_bits": sum(
                    int(row["counterfactual_verifier_bits"]) for row in history
                ),
                "evaluation": evaluation,
                "stable": bool(evaluation)
                and min(float(row["accuracy"]) for row in evaluation)
                >= CANDIDATE_ADMISSION_THRESHOLD,
                "digest": _digest(file),
            }
        )
        files.append(file)
    return tuple(files), records


def _record_fixed_route(
    system,
    file: query.ExternalTemporalCapabilityFile,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    query_symbol: int,
    depth: int,
    offset: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    lifetimes: int,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    rows: list[dict[str, object]] = []
    context: torch.Tensor | None = None
    for lifetime in range(lifetimes):
        episode = query._episode(
            system,
            file,
            evidence,
            query_symbol=query_symbol,
            depth=depth,
            batch_size=batch_size,
            data_steps=data_steps,
            seed=seed + lifetime,
            train=False,
            explore=False,
            forced_offset=offset,
        )
        context = episode.context
        evidence.observe(context, offset - 1, episode.accuracy)
        rows.append(
            {
                "lifetime": lifetime + 1,
                "accuracy": float(episode.accuracy),
                "selected_offset": offset,
                "unique_verifier_bits": episode.eligible_bits,
                "replayed_examples": 0,
            }
        )
    if context is None:
        raise RuntimeError("fixed route did not expose a learned query context")
    return rows, context


def _train_counterfactual_route(
    system,
    file: query.ExternalTemporalCapabilityFile,
    evidence: PersistentOpaqueContextRouteEvidence,
    *,
    updates: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    shuffled_outcomes: bool = False,
) -> tuple[list[dict[str, object]], torch.Tensor]:
    history: list[dict[str, object]] = []
    context: torch.Tensor | None = None
    offsets = torch.arange(1, query.MAX_OFFSET + 1, dtype=torch.long)
    for update in range(updates):
        outcomes: list[float] = []
        for offset in offsets.tolist():
            episode = query._episode(
                system,
                file,
                evidence,
                query_symbol=query.TARGET_QUERY,
                depth=query.TARGET_DEPTH,
                batch_size=batch_size,
                data_steps=data_steps,
                seed=seed + update,
                train=False,
                explore=False,
                forced_offset=offset,
            )
            context = episode.context
            outcomes.append(float(episode.accuracy))
        utility = torch.tensor(outcomes, dtype=torch.float32)
        observed = utility
        if shuffled_outcomes:
            generator = torch.Generator().manual_seed(seed + 88_001 + update)
            permutation = torch.randperm(query.MAX_OFFSET, generator=generator)
            observed = utility[permutation]
        if context is None:
            raise RuntimeError("counterfactual route did not expose a context")
        evidence.observe_batch(
            context.expand(query.MAX_OFFSET, -1),
            torch.arange(query.MAX_OFFSET, dtype=torch.long),
            observed,
        )
        history.append(
            {
                "update": update + 1,
                "best_observed_offset": int(observed.argmax()) + 1,
                "best_true_offset": int(utility.argmax()) + 1,
                "best_observed_accuracy": float(observed.max()),
                "best_true_accuracy": float(utility.max()),
                "unique_verifier_bits": batch_size
                * (data_steps - query.TARGET_DEPTH),
                "counterfactual_verifier_bits": query.MAX_OFFSET
                * batch_size
                * (data_steps - query.TARGET_DEPTH),
                "replayed_examples": 0,
            }
        )
    if context is None:
        raise RuntimeError("counterfactual route produced no episodes")
    return history, context


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
        raise ValueError("counterfactual address budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("counterfactual address optimization parameters are invalid")
    if args.data_steps <= query.TARGET_DEPTH:
        raise ValueError("data steps must include target trials")
    started = perf_counter()
    system = query._build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    offsets = tuple(range(1, query.MAX_OFFSET + 1))
    files, candidates = _train_candidates(
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
    source_evidence = _evidence(
        mastery_threshold=query.MASTERY_THRESHOLD,
        observations=args.source_route_lifetimes,
    )
    source_before, source_context = _record_fixed_route(
        system,
        winner,
        source_evidence,
        query_symbol=query.SOURCE_QUERY,
        depth=query.SOURCE_DEPTH,
        offset=winner_offset,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 200_000,
        lifetimes=args.source_route_lifetimes,
    )
    target_history, target_context = _train_counterfactual_route(
        system,
        winner,
        source_evidence,
        updates=args.target_route_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 300_000,
    )
    source_file_digest = _digest(winner)
    source_after = query._evaluate(
        system,
        winner,
        source_evidence,
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
        source_evidence,
        query_symbol=query.TARGET_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 400_000,
        lifetimes=args.retention_lifetimes,
    )
    unknown = query._evaluate(
        system,
        winner,
        source_evidence,
        query_symbol=query.UNKNOWN_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 500_000,
        lifetimes=args.retention_lifetimes,
    )
    wrong_offset = query._evaluate(
        system,
        winner,
        source_evidence,
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
        source_evidence,
        query_symbol=query.TARGET_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 700_000,
        lifetimes=args.retention_lifetimes,
        reset_memory_each_step=True,
    )
    shuffled_evidence = _evidence(
        mastery_threshold=query.MASTERY_THRESHOLD,
        observations=args.source_route_lifetimes,
    )
    _record_fixed_route(
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
    shuffled_history, _ = _train_counterfactual_route(
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
    restored = PersistentOpaqueContextRouteEvidence.from_payload(
        source_evidence.payload()
    )
    restored_target = query._evaluate(
        system,
        winner,
        restored,
        query_symbol=query.TARGET_QUERY,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 400_000,
        lifetimes=args.retention_lifetimes,
    )
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    winner_after = _digest(winner)
    candidate_training_bits = len(offsets) * args.source_updates * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    source_eval_bits = args.source_evaluation_lifetimes * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    target_bits = args.target_route_updates * args.batch_size * (
        args.data_steps - query.TARGET_DEPTH
    )
    audit_bits = sum(
        int(row["unique_verifier_bits"])
        for rows in (source_before, source_after, target_after, unknown, wrong_offset, missing_history, shuffled, restored_target)
        for row in rows
    )
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
        "source_retained_after_target": _stable(source_after),
        "wrong_offset_rejects_mastery": not _stable(wrong_offset),
        "missing_history_rejects_mastery": not _stable(missing_history),
        "unknown_query_does_not_claim_mastery": not _stable(unknown),
        "shuffled_route_feedback_rejects_target": not _stable(shuffled)
        or all(int(row["selected_offset"]) != query.TARGET_DEPTH for row in shuffled),
        "route_reload_exact": target_after == restored_target,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "frozen_promoted_file": source_file_digest == winner_after,
        "counterfactual_credit_recorded": sum(
            int(record["counterfactual_verifier_bits"]) for record in candidates
        ) > 0,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": COUNTERFACTUAL_SCHEMA,
        "claim_boundary": (
            "Outcome-only candidate search promotes one stable external temporal "
            "readout/address file, then acquires a second query-conditioned "
            "address with fresh counterfactual scalar probes; not arbitrary "
            "program induction, unrestricted memory growth, or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "controller": "frozen_canonical_amodal_controller",
            "readout_credit": "paired_counterfactual_scalar_keypress_arms",
            "candidate_admission": "stable_heldout_verifier_prefix",
            "address_acquisition": "all_opaque_offsets_common_random_scalar_probe",
            "route_memory": "persistent_opaque_context_route_evidence_v1",
            "history_transport": "canonical_runtime_external_history_event_bridge_v2",
            "history_causality": "read_before_current_append",
            "candidate_offsets": list(offsets),
        },
        "candidate_records": candidates,
        "selected_offset": winner_offset,
        "source_context": source_context.tolist(),
        "target_context": target_context.tolist(),
        "source_history": source_before,
        "target_history": target_history[-8:],
        "shuffled_history": shuffled_history[-8:],
        "evaluation": {
            "source_before": source_before,
            "source_after": source_after,
            "target_after": target_after,
            "unknown_query": unknown,
            "wrong_offset": wrong_offset,
            "missing_history": missing_history,
            "shuffled_route": shuffled,
            "reloaded_target": restored_target,
            "route_payload": source_evidence.payload(),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": (
                candidate_training_bits + source_eval_bits + target_bits
            ),
            "counterfactual_verifier_bits": sum(
                int(record["counterfactual_verifier_bits"]) for record in candidates
            )
            + args.target_route_updates
            * query.MAX_OFFSET
            * args.batch_size
            * (args.data_steps - query.TARGET_DEPTH),
            "audit_verifier_bits": audit_bits,
            "unique_logical_lifetimes": args.source_updates
            * len(offsets)
            * args.batch_size
            + args.source_evaluation_lifetimes * args.batch_size
            + args.source_route_lifetimes * args.batch_size
            + args.target_route_updates * args.batch_size,
            "counterfactual_logical_lifetimes": args.source_updates
            * len(offsets)
            * args.batch_size * 2
            + args.target_route_updates
            * query.MAX_OFFSET
            * args.batch_size,
            "optimizer_updates": args.source_updates * len(offsets),
            "route_memory_updates": args.source_route_lifetimes
            + args.target_route_updates * query.MAX_OFFSET,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": (
                candidate_training_bits + source_eval_bits + target_bits
                if all(gates.values())
                else None
            ),
        },
        "status": "promoted_counterfactual_query_address_growth"
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
