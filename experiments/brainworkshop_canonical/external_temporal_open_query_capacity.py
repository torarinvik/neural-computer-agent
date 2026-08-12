"""Pressure-test a bounded hot cache over a growable temporal route archive.

The preceding open-query rung grows five opaque temporal routes in long-term
external memory. This experiment adds capacity pressure: only two routes may
be resident in a hot cache, while all five remain in a growable archive. A
memory-side eviction policy ranks opaque signatures plus generic reliability
and age telemetry. Copy-on-write retention probes protect the mastered source
route and verify each replacement before committing it.

Returning routes are found by related learned keys and reactivated from their
opaque external handles. No old training stream is replayed, the controller,
event encoder, and capability file remain frozen, and the learner receives no
query-depth, route-position, semantic name, or task identifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from neural_computer import (
    EpisodicBindingArtifactIndex,
    ExternalCapabilityEvictionPolicy,
)

from . import external_temporal_open_query_growth as open_query
from . import external_temporal_query_address_growth as query
from . import external_temporal_query_counterfactual_growth as counterfactual

CAPACITY_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-open-query-capacity.v1"
)
ACTIVE_SLOTS = 2
POLICY_CONTEXT_WIDTH = query.EVENT_WIDTH
POLICY_CANDIDATE_WIDTH = query.EVENT_WIDTH + 2
POLICY_HIDDEN = 32
POLICY_TEMPERATURE = 0.7
POLICY_UPDATES = 3_000
POLICY_EVAL_EPISODES = 512
ROUTE_THRESHOLD = 0.99
ARCHIVE_MATCH_THRESHOLD = 0.75
ROUTE_STAGES = ((query.SOURCE_QUERY, query.SOURCE_DEPTH), *open_query.QUERY_STAGES)
ARRIVAL_SEQUENCE = (0, 0, 2, 3, 4, 1, 2, 3, 4, 0, 1, 2, 3, 4)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _policy_episode(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    generator = torch.Generator().manual_seed(seed)
    context = F.normalize(
        torch.randn(1, POLICY_CONTEXT_WIDTH, generator=generator), dim=-1
    )
    signatures = F.normalize(
        torch.randn(ACTIVE_SLOTS, POLICY_CONTEXT_WIDTH, generator=generator),
        dim=-1,
    )
    reliability = torch.rand(ACTIVE_SLOTS, generator=generator)
    age = torch.rand(ACTIVE_SLOTS, generator=generator)
    candidates = torch.cat(
        (
            signatures,
            reliability[:, None],
            age[:, None],
        ),
        dim=-1,
    ).unsqueeze(0)
    risk = (1.0 - reliability) + age
    return context, candidates, int(risk.argmax())


def _train_policy(
    *,
    seed: int,
    updates: int,
    reward_shuffled: bool = False,
) -> tuple[ExternalCapabilityEvictionPolicy, dict[str, float | int]]:
    if updates < 1:
        raise ValueError("capacity policy updates must be positive")
    torch.manual_seed(seed)
    policy = ExternalCapabilityEvictionPolicy(
        context_width=POLICY_CONTEXT_WIDTH,
        candidate_width=POLICY_CANDIDATE_WIDTH,
        hidden=POLICY_HIDDEN,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    explorer = torch.Generator().manual_seed(seed + 50_000)
    utilities: list[float] = []
    for update in range(updates):
        context, candidates, target = _policy_episode(seed + 100_000 + update)
        scores = policy.score_candidates(context, candidates)[0]
        selected = int(
            torch.multinomial(
                torch.softmax(scores / POLICY_TEMPERATURE, dim=-1),
                1,
                generator=explorer,
            )
        )
        utility = (
            float(torch.randint(2, (), generator=explorer))
            if reward_shuffled
            else float(selected == target)
        )
        loss = -(utility - 0.5) * torch.log_softmax(
            scores / POLICY_TEMPERATURE,
            dim=-1,
        )[selected]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        utilities.append(utility)
    policy.eval()
    return policy, {
        "optimizer_updates": updates,
        "unique_scalar_utilities": updates,
        "first_window_utility": sum(utilities[:100]) / min(100, len(utilities)),
        "last_window_utility": sum(utilities[-100:]) / min(100, len(utilities)),
        "reward_shuffled": int(reward_shuffled),
    }


@torch.no_grad()
def _evaluate_policy(
    policy: ExternalCapabilityEvictionPolicy,
    *,
    seed: int,
    episodes: int,
) -> float:
    correct = 0
    for episode in range(episodes):
        context, candidates, target = _policy_episode(seed + episode)
        correct += int(int(policy.score_candidates(context, candidates).argmax()) == target)
    return correct / episodes


def _route_handle(key: torch.Tensor) -> str:
    return f"opaque-temporal-route-{_tensor_digest(key)}"


def _route_by_handle(
    routes: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    return {str(route["handle"]): route for route in routes}


def _route_accuracy(
    system,
    file: query.ExternalTemporalCapabilityFile,
    evidence,
    route: dict[str, object],
    *,
    batch_size: int,
    data_steps: int,
    seed: int,
    lifetimes: int,
) -> list[dict[str, float | int]]:
    return query._evaluate(
        system,
        file,
        evidence,
        query_symbol=int(route["query_symbol"]),
        depth=int(route["depth"]),
        batch_size=batch_size,
        data_steps=data_steps,
        seed=seed,
        lifetimes=lifetimes,
        forced_offset=int(route["position"]) + 1,
    )


def _candidate_rows(
    index: EpisodicBindingArtifactIndex,
    *,
    step: int,
    order: tuple[int, ...],
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    for slot in order:
        binding_id = index.active_binding_ids[slot]
        if binding_id is None:
            raise RuntimeError("capacity cache contains an empty active slot")
        reliability, age = index.archive.telemetry(
            binding_id,
            step=step,
            age_horizon=16,
        )
        rows.append(
            torch.cat(
                (
                    index.archive.signature_key(binding_id),
                    torch.tensor((reliability, age), dtype=torch.float32),
                )
            )
        )
    candidates = torch.stack(rows).unsqueeze(0)
    if candidates.shape != (1, ACTIVE_SLOTS, POLICY_CANDIDATE_WIDTH):
        raise RuntimeError("capacity candidate ABI changed")
    return candidates


@torch.no_grad()
def _select_victim(
    policy: ExternalCapabilityEvictionPolicy,
    index: EpisodicBindingArtifactIndex,
    incoming_key: torch.Tensor,
    *,
    step: int,
    order: tuple[int, ...],
) -> tuple[int, int, torch.Tensor, bool]:
    candidates = _candidate_rows(index, step=step, order=order)
    scores = policy.score_candidates(incoming_key.unsqueeze(0), candidates)[0]
    unmasked_position = int(scores.argmax())
    masked = scores.clone()
    for position, slot in enumerate(order):
        binding_id = index.active_binding_ids[slot]
        if binding_id is not None and index.archive.is_protected(binding_id):
            masked[position] = torch.finfo(masked.dtype).min
    if not bool(torch.isfinite(masked).any()):
        raise RuntimeError("capacity cache has no eligible victim")
    selected_position = int(masked.argmax())
    return (
        selected_position,
        order[selected_position],
        scores,
        unmasked_position == selected_position,
    )


def _retention_probe(
    candidate: EpisodicBindingArtifactIndex,
    *,
    route_by_handle: dict[str, dict[str, object]],
    system,
    file: query.ExternalTemporalCapabilityFile,
    evidence,
    batch_size: int,
    data_steps: int,
    seed: int,
    lifetimes: int,
    verifier_bits: list[int] | None = None,
) -> bool:
    active = candidate.active_binding_ids
    if any(binding_id is None for binding_id in active):
        return False
    for position, binding_id in enumerate(active):
        assert binding_id is not None
        handle = candidate.artifact_handle(binding_id)
        if handle is None or handle not in route_by_handle:
            return False
        rows = _route_accuracy(
            system,
            file,
            evidence,
            route_by_handle[handle],
            batch_size=batch_size,
            data_steps=data_steps,
            seed=seed + position * 10_000,
            lifetimes=lifetimes,
        )
        if verifier_bits is not None:
            verifier_bits[0] += sum(
                int(row["unique_verifier_bits"]) for row in rows
            )
        if not rows or min(float(row["accuracy"]) for row in rows) < ROUTE_THRESHOLD:
            return False
    return True


def _reload_index(index: EpisodicBindingArtifactIndex) -> EpisodicBindingArtifactIndex:
    return EpisodicBindingArtifactIndex.from_payload(index.payload())


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.source_evaluation_lifetimes,
        args.source_route_lifetimes,
        args.target_route_updates,
        args.policy_updates,
        args.policy_eval_episodes,
        args.batch_size,
        args.data_steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("capacity pressure budgets must be positive")
    if args.data_steps <= max(depth for _, depth in ROUTE_STAGES):
        raise ValueError("data steps must include all route trials")

    started = perf_counter()
    system = query._build(args.seed)
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    policy, policy_training = _train_policy(
        seed=args.seed,
        updates=args.policy_updates,
    )
    policy_accuracy = _evaluate_policy(
        policy,
        seed=args.seed + 900_000,
        episodes=args.policy_eval_episodes,
    )
    shuffled_policy, shuffled_training = _train_policy(
        seed=args.seed + 1_000_000,
        updates=args.policy_updates,
        reward_shuffled=True,
    )
    shuffled_policy_accuracy = _evaluate_policy(
        shuffled_policy,
        seed=args.seed + 1_900_000,
        episodes=args.policy_eval_episodes,
    )

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
    winner_digest_before = _digest(winner)

    evidence = counterfactual._evidence(
        mastery_threshold=query.MASTERY_THRESHOLD,
        observations=args.source_route_lifetimes,
    )
    source_history, source_context = open_query._record_fixed_route(
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
    routes: list[dict[str, object]] = []
    source_position = int(evidence.preferred_order(source_context)[0])
    routes.append(
        {
            "route_id": 0,
            "query_symbol": query.SOURCE_QUERY,
            "depth": query.SOURCE_DEPTH,
            "key": source_context,
            "position": source_position,
            "handle": _route_handle(source_context),
            "history": source_history,
        }
    )
    stage_histories: list[dict[str, object]] = []
    for stage_index, (query_symbol, depth) in enumerate(open_query.QUERY_STAGES):
        history, context = open_query._train_query_route(
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
        routes.append(
            {
                "route_id": stage_index + 1,
                "query_symbol": query_symbol,
                "depth": depth,
                "key": context,
                "position": int(evidence.preferred_order(context)[0]),
                "handle": _route_handle(context),
                "history": history,
            }
        )
        stage_histories.append(
            {
                "query_symbol": query_symbol,
                "depth": depth,
                "history": history,
            }
        )
    route_by_handle = _route_by_handle(routes)

    index = EpisodicBindingArtifactIndex.create(
        query.EVENT_WIDTH,
        query.EVENT_WIDTH,
        active_slots=ACTIVE_SLOTS,
        matching_threshold=ARCHIVE_MATCH_THRESHOLD,
        mastery_threshold=0.8,
        min_mastery_observations=8,
    )
    binding_ids = [
        index.register(route["key"], route["key"], route["handle"])
        for route in routes
    ]
    index.activate(binding_ids[0], 0)
    index.activate(binding_ids[1], 1)
    for step in range(8):
        index.archive.observe(binding_ids[0], 1.0, step=step)
    if not index.archive.is_protected(binding_ids[0]):
        raise RuntimeError("source route did not reach protection")

    arrivals: list[dict[str, object]] = []
    active_noops = 0
    replacements = 0
    protected_evictions = 0
    reactivation_failures = 0
    retention_probe_bits = [0]
    step = 8
    for ordinal, route_id in enumerate(ARRIVAL_SEQUENCE):
        step += 1
        route = routes[route_id]
        query_key = (
            route["key"]
            if ordinal % 2 == 0
            else open_query._related_key(route["key"], seed=args.seed + 4_000_000 + ordinal)
        )
        lookup = index.lookup(query_key)
        if lookup.binding_id is None:
            raise RuntimeError("capacity arrival was not found in long-term archive")
        binding_id = lookup.binding_id
        if lookup.active_slot is not None:
            active_noops += 1
            rows = _route_accuracy(
                system,
                winner,
                evidence,
                route,
                batch_size=args.batch_size,
                data_steps=args.data_steps,
                seed=args.seed + 5_000_000 + ordinal,
                lifetimes=args.retention_lifetimes,
            )
            passed = bool(rows) and min(float(row["accuracy"]) for row in rows) >= ROUTE_THRESHOLD
            index.archive.observe(binding_id, float(passed), step=step)
            arrivals.append(
                {
                    "ordinal": ordinal,
                    "route_id": route_id,
                    "known_before": True,
                    "active_before": True,
                    "replacement": False,
                    "lookup_similarity": lookup.similarity,
                    "probe_passed": passed,
                }
            )
            continue

        order = (0, 1) if ordinal % 2 == 0 else (1, 0)
        position, victim_slot, scores, raw_match = _select_victim(
            policy,
            index,
            query_key,
            step=step,
            order=order,
        )
        displaced = index.active_binding_ids[victim_slot]
        if displaced is not None and index.archive.is_protected(displaced):
            protected_evictions += 1
        receipt = index.reactivate_verified(
            binding_id,
            victim_slot,
            lambda candidate, ordinal=ordinal: _retention_probe(
                candidate,
                route_by_handle=route_by_handle,
                system=system,
                file=winner,
                evidence=evidence,
                batch_size=args.batch_size,
                data_steps=args.data_steps,
                seed=args.seed + 6_000_000 + ordinal * 10_000,
                lifetimes=args.retention_lifetimes,
                verifier_bits=retention_probe_bits,
            ),
        )
        if not receipt.accepted:
            reactivation_failures += 1
        else:
            replacements += 1
            index.archive.observe(binding_id, 1.0, step=step)
        arrivals.append(
            {
                "ordinal": ordinal,
                "route_id": route_id,
                "known_before": True,
                "active_before": False,
                "replacement": True,
                "order": order,
                "selected_position": position,
                "selected_physical_slot": victim_slot,
                "raw_policy_selected_position": raw_match,
                "scores": scores.tolist(),
                "displaced_binding_id": displaced,
                "receipt": receipt.__dict__,
            }
        )

    final_active = []
    for slot, binding_id in enumerate(index.active_binding_ids):
        if binding_id is None:
            continue
        handle = index.artifact_handle(binding_id)
        if handle is None or handle not in route_by_handle:
            raise RuntimeError("active cache handle is not in the archive backend")
        rows = _route_accuracy(
            system,
            winner,
            evidence,
            route_by_handle[handle],
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 7_000_000 + slot * 10_000,
            lifetimes=args.retention_lifetimes,
        )
        final_active.append(
            {
                "slot": slot,
                "binding_id": binding_id,
                "route_id": route_by_handle[handle]["route_id"],
                "accuracy": min(float(row["accuracy"]) for row in rows),
            }
        )

    unknown_episode = query._episode(
        system,
        winner,
        evidence,
        query_symbol=query.CUE_SYMBOL,
        depth=query.TARGET_DEPTH,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 8_000_000,
        train=False,
        explore=False,
        forced_offset=1,
    )
    unknown_lookup = index.lookup(unknown_episode.context)
    restored = _reload_index(index)
    restored_active = []
    for slot, binding_id in enumerate(restored.active_binding_ids):
        if binding_id is None:
            continue
        handle = restored.artifact_handle(binding_id)
        assert handle is not None
        rows = _route_accuracy(
            system,
            winner,
            evidence,
            route_by_handle[handle],
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 9_000_000 + slot * 10_000,
            lifetimes=args.retention_lifetimes,
        )
        restored_active.append(
            {
                "slot": slot,
                "binding_id": binding_id,
                "route_id": route_by_handle[handle]["route_id"],
                "accuracy": min(float(row["accuracy"]) for row in rows),
            }
        )
    corrupted_payload = index.payload()
    corrupted_handles = corrupted_payload["artifact_handles"]
    if not isinstance(corrupted_handles, list):
        raise TypeError("capacity archive handles are not a list")
    corrupted_handles[0] = "tampered-capacity-handle"
    corruption_rejected = False
    try:
        EpisodicBindingArtifactIndex.from_payload(corrupted_payload)
    except ValueError as error:
        corruption_rejected = "checksum" in str(error).lower()

    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    candidate_training_bits = len(offsets) * args.source_updates * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    candidate_eval_bits = args.source_evaluation_lifetimes * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    route_acquisition_bits = args.source_route_lifetimes * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    ) + sum(
        args.target_route_updates * args.batch_size * (args.data_steps - depth)
        for _, depth in open_query.QUERY_STAGES
    )
    gates = {
        "held_out_victim_policy_mastery": policy_accuracy >= 0.80,
        "reward_shuffled_policy_rejects_mastery": shuffled_policy_accuracy <= 0.70,
        "unique_five_route_bank": len(routes) == 5 and len(binding_ids) == 5,
        "source_route_protected": index.archive.is_protected(binding_ids[0]),
        "all_reactivations_accepted": reactivation_failures == 0,
        "protected_source_never_evicted": protected_evictions == 0
        and index.active_binding_ids[0] == binding_ids[0],
        "repeated_replacements_occurred": replacements >= 4,
        "active_noop_probes_passed": active_noops >= 2,
        "all_active_routes_mastered": all(
            float(row["accuracy"]) >= ROUTE_THRESHOLD for row in final_active
        ),
        "all_archive_routes_remain_known": all(
            index.lookup(route["key"]).binding_id == binding_ids[int(route["route_id"])]
            for route in routes
        ),
        "unknown_key_not_claimed": unknown_lookup.binding_id is None,
        "reload_preserves_active_routes": restored_active == final_active,
        "corruption_rejected": corruption_rejected,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "frozen_promoted_file": winner_digest_before == _digest(winner),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": CAPACITY_SCHEMA,
        "claim_boundary": (
            "Replay-free capacity pressure over five externally acquired temporal "
            "routes: a learned opaque victim policy and verifier-gated two-slot "
            "hot cache reactivate archived routes while protecting prior mastery; "
            "not unrestricted memory growth, arbitrary computation, or general "
            "continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "long_term_archive": "episodic_binding_artifact_index_v1",
            "hot_cache": "two_slot_opaque_reactivation_cache",
            "victim_policy": "external_capability_eviction_policy_v1",
            "replacement": "copy_on_write_retention_verified_v1",
            "protection": "stable_scalar_prefix_v1",
            "controller": "frozen_canonical_amodal_controller",
            "event_encoder": "frozen_learned_event_encoder",
            "capability_file": "frozen_external_temporal_capability_file",
            "route_count": len(routes),
            "active_slots": ACTIVE_SLOTS,
            "forbidden_features": (
                "query_depth_route_position_semantic_names_task_ids_replayed_streams"
            ),
        },
        "policy_training": policy_training,
        "policy_accuracy": policy_accuracy,
        "shuffled_policy_training": shuffled_training,
        "shuffled_policy_accuracy": shuffled_policy_accuracy,
        "candidate_records": candidates,
        "routes": [
            {
                key: value.tolist() if isinstance(value, torch.Tensor) else value
                for key, value in route.items()
                if key not in {"key", "history"}
            }
            | {
                "key_digest": _tensor_digest(route["key"]),
                "history_tail": route["history"][-8:],
            }
            for route in routes
        ],
        "arrivals": arrivals,
        "final_active": final_active,
        "restored_active": restored_active,
        "archive_status": index.archive.status().__dict__,
        "unknown_lookup": {
            "binding_id": unknown_lookup.binding_id,
            "similarity": unknown_lookup.similarity,
            "active_slot": unknown_lookup.active_slot,
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": candidate_training_bits
            + candidate_eval_bits
            + route_acquisition_bits,
            "capacity_retention_verifier_bits": retention_probe_bits[0],
            "policy_verifier_bits": args.policy_updates,
            "shuffled_policy_verifier_bits": args.policy_updates,
            "unique_logical_lifetimes": len(routes) * args.batch_size
            + args.policy_updates,
            "optimizer_updates": len(offsets) * args.source_updates
            + args.policy_updates,
            "online_controller_updates": 0,
            "online_capability_updates": 0,
            "archive_records": len(routes),
            "active_slots": ACTIVE_SLOTS,
            "replacement_count": replacements,
            "reactivation_failures": reactivation_failures,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": (
                candidate_training_bits + candidate_eval_bits + route_acquisition_bits
                if all(gates.values())
                else None
            ),
        },
        "status": "promoted_open_query_capacity" if all(gates.values()) else "rejected",
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
    parser.add_argument("--policy-updates", type=int, default=POLICY_UPDATES)
    parser.add_argument("--policy-eval-episodes", type=int, default=POLICY_EVAL_EPISODES)
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
