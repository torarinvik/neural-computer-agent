"""Grow a replay-free external temporal address bank through four queries.

The source readout is acquired once with paired common-random scalar action
arms and then frozen. Four new opaque query/address capabilities are acquired
sequentially from fresh scalar probes for n-back depths 5, 6, 7, and 8. After
each addition every earlier capability is re-evaluated without replay. The
learned route positions are appended to the variable-capacity canonical
content index, which must retrieve exact and related learned keys while
unknown keys miss.

Query symbols, n-back depths, target bits, and expected offsets are private
verifier state. The learner sees only rendered event tensors, opaque actions,
and scalar outcomes. The claim is deliberately bounded repeated external
address growth, not unrestricted memory growth or general continual learning.
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

OPEN_QUERY_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-open-query-growth.v1"
)
CANDIDATE_ADMISSION_THRESHOLD = 0.95
ROUTE_SELECTION_THRESHOLD = 0.99
READ_MATCH_THRESHOLD = 0.75
NOISE_SCALE = 0.20
QUERY_STAGES = ((1, 5), (2, 6), (3, 7), (4, 8))


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _stable(
    rows: list[dict[str, object]],
    *,
    threshold: float = query.MASTERY_THRESHOLD,
) -> bool:
    return bool(rows) and min(float(row["accuracy"]) for row in rows) >= threshold


def _related_key(key: torch.Tensor, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 8_307)
    return F.normalize(
        key + NOISE_SCALE * torch.randn(key.shape, generator=generator), dim=0
    )


def _record_fixed_route(
    system,
    file: query.ExternalTemporalCapabilityFile,
    evidence,
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
        raise RuntimeError("fixed route did not expose a query context")
    return rows, context


def _train_query_route(
    system,
    file: query.ExternalTemporalCapabilityFile,
    evidence,
    *,
    query_symbol: int,
    depth: int,
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
                query_symbol=query_symbol,
                depth=depth,
                batch_size=batch_size,
                data_steps=data_steps,
                seed=seed + update,
                train=False,
                explore=False,
                forced_offset=offset,
            )
            context = episode.context
            outcomes.append(float(episode.accuracy))
        if context is None:
            raise RuntimeError("query route did not expose a context")
        utility = torch.tensor(outcomes, dtype=torch.float32)
        observed = utility
        if shuffled_outcomes:
            generator = torch.Generator().manual_seed(seed + 88_001 + update)
            observed = utility[
                torch.randperm(query.MAX_OFFSET, generator=generator)
            ]
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
                "unique_verifier_bits": batch_size * (data_steps - depth),
                "counterfactual_verifier_bits": query.MAX_OFFSET
                * batch_size
                * (data_steps - depth),
                "replayed_examples": 0,
            }
        )
    if context is None:
        raise RuntimeError("query route produced no context")
    return history, context


def _read(
    index: ExternalTemporalAddressIndex, key: torch.Tensor
) -> dict[str, object]:
    result = index.read(key.unsqueeze(0), top_k=1)
    hit = bool(result.hit[0])
    return {
        "hit": hit,
        "score": float(result.scores[0, 0]),
        "resolved_position": int(result.target_positions[0, 0]) if hit else None,
    }


def _evaluate_via_index(
    system,
    file: query.ExternalTemporalCapabilityFile,
    evidence,
    index: ExternalTemporalAddressIndex,
    *,
    key: torch.Tensor,
    query_symbol: int,
    depth: int,
    expected_position: int,
    batch_size: int,
    data_steps: int,
    seed: int,
    lifetimes: int,
) -> dict[str, object]:
    route = _read(index, key)
    position = route["resolved_position"]
    if position is None:
        return {"route": route, "accuracy": 0.5, "lifetimes": []}
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
        "route": route,
        "accuracy": sum(float(row["accuracy"]) for row in rows) / len(rows),
        "position_correct": int(position) == expected_position,
        "lifetimes": rows,
    }


def _stage_evaluation(
    system,
    file: query.ExternalTemporalCapabilityFile,
    evidence,
    index: ExternalTemporalAddressIndex,
    *,
    routes: tuple[dict[str, object], ...],
    batch_size: int,
    data_steps: int,
    seed: int,
    lifetimes: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for route in routes:
        result = _evaluate_via_index(
            system,
            file,
            evidence,
            index,
            key=route["key"],
            query_symbol=int(route["query_symbol"]),
            depth=int(route["depth"]),
            expected_position=int(route["position"]),
            batch_size=batch_size,
            data_steps=data_steps,
            seed=seed + int(route["query_symbol"]) * 10_000,
            lifetimes=lifetimes,
        )
        results.append(
            {
                "query_symbol": route["query_symbol"],
                "depth": route["depth"],
                **result,
            }
        )
    return results


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
        raise ValueError("open query-growth budgets must be positive")
    if args.learning_rate <= 0.0 or args.entropy_weight < 0.0:
        raise ValueError("open query-growth optimization parameters are invalid")
    if args.data_steps <= max(depth for _, depth in QUERY_STAGES):
        raise ValueError("data steps must include the deepest query trials")

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
    winner = files[winner_offset - 1]
    winner_index = winner_offset - 1
    winner_digest_before_growth = _digest(winner)
    evidence = counterfactual._evidence(
        mastery_threshold=query.MASTERY_THRESHOLD,
        observations=args.source_route_lifetimes,
    )
    source_before, source_context = _record_fixed_route(
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
    source_position = int(evidence.preferred_order(source_context)[0])
    index = ExternalTemporalAddressIndex(
        query.EVENT_WIDTH,
        write_match_threshold=0.999,
        read_match_threshold=READ_MATCH_THRESHOLD,
    )
    index.write(
        source_context.unsqueeze(0),
        target_scopes=torch.zeros(1, dtype=torch.long),
        target_positions=torch.tensor([source_position], dtype=torch.long),
        strength=torch.ones(1),
    )
    routes: list[dict[str, object]] = [
        {
            "query_symbol": query.SOURCE_QUERY,
            "depth": query.SOURCE_DEPTH,
            "key": source_context,
            "position": source_position,
            "history": source_before,
        }
    ]
    stage_results: list[dict[str, object]] = []
    stage_histories: list[dict[str, object]] = []
    for stage_index, (query_symbol, depth) in enumerate(QUERY_STAGES):
        history, context = _train_query_route(
            system,
            winner,
            evidence,
            query_symbol=query_symbol,
            depth=depth,
            updates=args.target_route_updates,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 300_000 + stage_index * 10_000,
        )
        position = int(evidence.preferred_order(context)[0])
        index.write(
            context.unsqueeze(0),
            target_scopes=torch.zeros(1, dtype=torch.long),
            target_positions=torch.tensor([position], dtype=torch.long),
            strength=torch.ones(1),
        )
        routes.append(
            {
                "query_symbol": query_symbol,
                "depth": depth,
                "key": context,
                "position": position,
                "history": history,
            }
        )
        evaluated = _stage_evaluation(
            system,
            winner,
            evidence,
            index,
            routes=tuple(routes),
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 500_000 + stage_index * 10_000,
            lifetimes=args.retention_lifetimes,
        )
        stage_results.append(
            {
                "stage": stage_index + 1,
                "new_query_symbol": query_symbol,
                "new_depth": depth,
                "new_history_tail": history[-8:],
                "route_count": len(routes),
                "evaluations": evaluated,
                "record_count": index.record_count,
            }
        )
        stage_histories.append(
            {
                "query_symbol": query_symbol,
                "depth": depth,
                "history": history,
            }
        )

    shuffled_evidence = counterfactual._evidence(
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
    shuffled_history, _ = _train_query_route(
        system,
        winner,
        shuffled_evidence,
        query_symbol=QUERY_STAGES[-1][0],
        depth=QUERY_STAGES[-1][1],
        updates=args.target_route_updates,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 900_000,
        shuffled_outcomes=True,
    )
    unknown_episode = query._episode(
        system,
        winner,
        evidence,
        query_symbol=12,
        depth=QUERY_STAGES[-1][1],
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 1_000_000,
        train=False,
        explore=False,
        forced_offset=1,
    )
    unknown_route = _read(index, unknown_episode.context)
    related_results = _stage_evaluation(
        system,
        winner,
        evidence,
        index,
        routes=tuple(routes),
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 1_100_000,
        lifetimes=args.retention_lifetimes,
    )
    for result, route in zip(related_results, routes, strict=True):
        related = _evaluate_via_index(
            system,
            winner,
            evidence,
            index,
            key=_related_key(route["key"], seed=args.seed + 2_000 + int(route["query_symbol"])),
            query_symbol=int(route["query_symbol"]),
            depth=int(route["depth"]),
            expected_position=int(route["position"]),
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 1_200_000 + int(route["query_symbol"]) * 10_000,
            lifetimes=args.retention_lifetimes,
        )
        result["related"] = related

    restored = ExternalTemporalAddressIndex.from_payload(index.payload())
    restored_results = _stage_evaluation(
        system,
        winner,
        evidence,
        restored,
        routes=tuple(routes),
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 1_300_000,
        lifetimes=args.retention_lifetimes,
    )
    corrupted_payload = index.payload()
    corrupted_state = corrupted_payload["state"]
    if not isinstance(corrupted_state, dict):
        raise TypeError("open query index payload state is not a mapping")
    corrupted_state["keys"] = corrupted_state["keys"].clone()
    corrupted_state["keys"][0, 0] += 0.25
    corruption_rejected = False
    try:
        ExternalTemporalAddressIndex.from_payload(corrupted_payload)
    except ValueError as error:
        corruption_rejected = "checksum" in str(error).lower()
    record_count_before_clear = index.record_count
    index.clear()
    cleared_route = _read(index, routes[-1]["key"])

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    candidate_training_bits = len(offsets) * args.source_updates * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    candidate_eval_bits = args.source_evaluation_lifetimes * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    source_route_bits = args.source_route_lifetimes * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    target_route_bits = sum(
        args.target_route_updates * args.batch_size * (args.data_steps - depth)
        for _, depth in QUERY_STAGES
    )
    target_cf_bits = sum(
        args.target_route_updates
        * query.MAX_OFFSET
        * args.batch_size
        * (args.data_steps - depth)
        for _, depth in QUERY_STAGES
    )
    all_stage_evaluations = [
        evaluation for stage in stage_results for evaluation in stage["evaluations"]
    ]
    all_mastered = all(
        _stable(evaluation["lifetimes"], threshold=ROUTE_SELECTION_THRESHOLD)
        and bool(evaluation.get("position_correct"))
        for evaluation in all_stage_evaluations
    )
    all_related_mastered = all(
        _stable(evaluation["related"]["lifetimes"], threshold=ROUTE_SELECTION_THRESHOLD)
        and bool(evaluation["related"].get("position_correct"))
        for evaluation in related_results
    )
    reload_exact = all(
        original["route"] == restored_result["route"]
        for original, restored_result in zip(
            related_results, restored_results, strict=True
        )
    )
    gates = {
        "unique_stable_candidate": len(stable_offsets) == 1,
        "winner_is_verifier_valid": winner_offset == query.SOURCE_DEPTH,
        "source_candidate_mastered": _stable(
            candidates[winner_index]["evaluation"],
            threshold=CANDIDATE_ADMISSION_THRESHOLD,
        ),
        "source_retained_through_all_stages": all(
            _stable(
                evaluation["lifetimes"],
                threshold=ROUTE_SELECTION_THRESHOLD,
            )
            for evaluation in all_stage_evaluations
            if int(evaluation["query_symbol"]) == query.SOURCE_QUERY
        ),
        "all_new_queries_mastered": all_mastered,
        "all_related_keys_mastered": all_related_mastered,
        "route_count_grew_without_replay": record_count_before_clear
        == 1 + len(QUERY_STAGES),
        "unknown_key_misses": not bool(unknown_route["hit"]),
        "shuffled_new_query_rejects_mastery": not _stable(
            query._evaluate(
                system,
                winner,
                shuffled_evidence,
                query_symbol=QUERY_STAGES[-1][0],
                depth=QUERY_STAGES[-1][1],
                batch_size=args.batch_size,
                data_steps=args.data_steps,
                seed=args.seed + 1_400_000,
                lifetimes=args.retention_lifetimes,
            ),
            threshold=ROUTE_SELECTION_THRESHOLD,
        ),
        "index_reload_preserves_all_routes": reload_exact,
        "index_corruption_rejected": corruption_rejected,
        "index_clear_removes_hits": not bool(cleared_route["hit"]),
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "frozen_promoted_file": winner_digest_before_growth == _digest(winner),
        "zero_replayed_examples": True,
    }
    unique_bits = (
        candidate_training_bits
        + candidate_eval_bits
        + source_route_bits
        + target_route_bits
    )
    report = {
        "schema": OPEN_QUERY_SCHEMA,
        "claim_boundary": (
            "Replay-free sequential growth of five opaque temporal query/address "
            "routes through a variable-capacity external index with frozen "
            "controller and readout; not unrestricted memory growth, arbitrary "
            "new computation, program induction, or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "controller": "frozen_canonical_amodal_controller",
            "readout_credit": "paired_counterfactual_scalar_keypress_arms",
            "candidate_admission": "stable_heldout_verifier_prefix",
            "route_memory": "persistent_opaque_context_route_evidence_v1",
            "content_memory": "external_temporal_address_index_v2",
            "route_growth": "sequential_new_contexts_without_old_replay",
            "query_stages": [list(stage) for stage in QUERY_STAGES],
            "related_key_noise_scale": NOISE_SCALE,
            "read_match_threshold": READ_MATCH_THRESHOLD,
            "history_transport": "canonical_runtime_external_history_event_bridge_v2",
            "history_causality": "read_before_current_append",
        },
        "candidate_records": candidates,
        "selected_offset": winner_offset,
        "route_records": [
            {
                key: value.tolist() if isinstance(value, torch.Tensor) else value
                for key, value in route.items()
                if key != "history"
            }
            | {"history_tail": route["history"][-8:]}
            for route in routes
        ],
        "stage_results": stage_results,
        "related_results": related_results,
        "restored_results": restored_results,
        "shuffled_history": shuffled_history[-8:],
        "evaluation": {
            "unknown_route": unknown_route,
            "cleared_route": cleared_route,
            "record_count_before_clear": record_count_before_clear,
            "route_payload": evidence.payload(),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": unique_bits,
            "counterfactual_verifier_bits": sum(
                int(record["counterfactual_verifier_bits"]) for record in candidates
            )
            + target_cf_bits,
            "audit_verifier_bits": sum(
                int(row["unique_verifier_bits"])
                for evaluation in all_stage_evaluations
                for row in evaluation["lifetimes"]
            ),
            "unique_logical_lifetimes": len(offsets)
            * args.source_updates
            * args.batch_size
            + args.source_evaluation_lifetimes * args.batch_size
            + args.source_route_lifetimes * args.batch_size
            + sum(
                args.target_route_updates * args.batch_size for _ in QUERY_STAGES
            ),
            "counterfactual_logical_lifetimes": len(offsets)
            * args.source_updates
            * args.batch_size
            * 2
            + sum(
                args.target_route_updates
                * query.MAX_OFFSET
                * args.batch_size
                for _ in QUERY_STAGES
            ),
            "optimizer_updates": args.source_updates * len(offsets),
            "route_memory_updates": args.source_route_lifetimes
            + args.target_route_updates * len(QUERY_STAGES) * query.MAX_OFFSET,
            "content_memory_writes": record_count_before_clear,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": unique_bits if all(gates.values()) else None,
        },
        "status": "promoted_open_query_growth" if all(gates.values()) else "rejected",
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
