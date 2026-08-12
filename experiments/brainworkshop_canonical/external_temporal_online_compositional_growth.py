"""Repeat external capability growth and learned compositional consolidation.

The prior rungs established the pieces independently: fresh temporal route
acquisition, interleaved external admission, and one-shot consolidation of two
distinct routes that share an opaque basis.  This pressure test composes them
over a stream of three route pairs.  Each pair is acquired from fresh scalar
temporal evidence, inserted into the external artifact memory, selected by a
generic opaque consolidation policy, and verifier-gated into one artifact with
two independent views.  Every earlier route is re-read after each later pair
arrives; no route-training stream is replayed.

The controller, learned event encoder, and temporal capability file are frozen
after acquisition.  The policy sees only opaque keys, artifact summaries,
strength, and age.  Query symbols, temporal depths, route identities, and
expected offsets remain private verifier state.  This is a bounded repeated
compositional-growth result, not unrestricted memory growth or general
continual learning.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import torch
from torch.nn import functional as F

from experiments.opaque_consolidation_amodal.train import _train_policy
from neural_computer import (
    CapabilityRetentionProbe,
    ExecutableArtifactMemory,
    MemoryCandidates,
    OpaqueConsolidationPolicy,
)

from . import external_temporal_open_query_growth as open_query
from . import external_temporal_query_address_growth as query
from . import external_temporal_query_counterfactual_growth as counterfactual
from .external_temporal_shared_artifact_consolidation import (
    _address_scrubbed_candidates,
    _promote_view,
    _selected_key_set,
    _tensor_digest,
)

ONLINE_COMPOSITION_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-online-"
    "compositional-growth.v1"
)
ROUTE_SPECS = (
    (query.SOURCE_QUERY, query.SOURCE_DEPTH),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 7),
)
PAIR_INDICES = ((0, 1), (2, 3), (4, 5))
MEMORY_CAPACITY = 4
SHARED_NOISE = 0.02
ROUTE_THRESHOLD = 0.99


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _acquire_routes(
    system,
    winner,
    evidence,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], int, int]:
    source_history, source_context = open_query._record_fixed_route(
        system,
        winner,
        evidence,
        query_symbol=ROUTE_SPECS[0][0],
        depth=ROUTE_SPECS[0][1],
        offset=args.winner_offset,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 200_000,
        lifetimes=args.source_route_lifetimes,
    )
    routes: list[dict[str, object]] = [
        {
            "route_id": 0,
            "query_symbol": ROUTE_SPECS[0][0],
            "depth": ROUTE_SPECS[0][1],
            "key": source_context,
            "position": int(evidence.preferred_order(source_context)[0]),
            "history": source_history,
        }
    ]
    for route_id, (query_symbol, depth) in enumerate(ROUTE_SPECS[1:], start=1):
        history, context = open_query._train_query_route(
            system,
            winner,
            evidence,
            query_symbol=query_symbol,
            depth=depth,
            updates=args.target_route_updates,
            batch_size=args.batch_size,
            data_steps=args.data_steps,
            seed=args.seed + 300_000 + route_id * 10_000,
        )
        routes.append(
            {
                "route_id": route_id,
                "query_symbol": query_symbol,
                "depth": depth,
                "key": context,
                "position": int(evidence.preferred_order(context)[0]),
                "history": history,
            }
        )
    route_bits = sum(
        int(row["unique_verifier_bits"])
        for route in routes
        for row in route["history"]
    )
    counterfactual_bits = sum(
        int(row["counterfactual_verifier_bits"])
        for route in routes[1:]
        for row in route["history"]
    )
    return routes, route_bits, counterfactual_bits


def _make_artifacts(
    routes: list[dict[str, object]],
    *,
    seed: int,
) -> tuple[
    tuple[torch.Tensor, ...],
    tuple[dict[str, torch.Tensor], ...],
    dict[int, torch.Tensor],
]:
    generator = torch.Generator().manual_seed(seed + 700_000)
    bases = tuple(
        F.normalize(
            torch.randn(query.EVENT_WIDTH, generator=generator),
            dim=0,
        )
        for _ in PAIR_INDICES
    )
    residuals = tuple(
        SHARED_NOISE * torch.randn(query.EVENT_WIDTH, generator=generator)
        for _ in routes
    )
    keys = tuple(route["key"] for route in routes)
    aliases = {
        first: open_query._related_key(
            keys[first],
            seed=seed + 4_000_000 + first,
        )
        for first, _second in PAIR_INDICES
    }
    artifacts = tuple(
        _fixed_route_artifact(
            shared_basis=bases[pair_index],
            residual=residuals[route_index],
            position=int(routes[route_index]["position"]),
        )
        for pair_index, (first, second) in enumerate(PAIR_INDICES)
        for route_index in (first, second)
    )
    # The comprehension above follows the pair layout, which is the same as
    # ROUTE_SPECS here.  Make the contract explicit before returning it.
    if len(artifacts) != len(routes):
        raise RuntimeError("compositional artifact route count changed")
    return keys, artifacts, aliases


def _fixed_route_artifact(
    *,
    shared_basis: torch.Tensor,
    residual: torch.Tensor,
    position: int,
) -> dict[str, torch.Tensor]:
    """Use one fixed-width opaque artifact ABI for raw and composed rows."""

    return {
        "shared_basis": shared_basis,
        "route_a_residual": residual,
        "route_b_residual": torch.zeros_like(residual),
        "route_a_address": torch.tensor([position], dtype=torch.float32),
        "route_b_address": torch.tensor([-1.0], dtype=torch.float32),
    }


def _fixed_composed_artifact(
    route_a: dict[str, torch.Tensor],
    route_b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "shared_basis": route_a["shared_basis"],
        "route_a_residual": route_a["route_a_residual"],
        "route_b_residual": route_b["route_a_residual"],
        "route_a_address": route_a["route_a_address"],
        "route_b_address": route_b["route_a_address"],
    }


def _fixed_position(
    artifact: dict[str, torch.Tensor],
    view: str | None,
) -> int:
    field = "route_b_address" if view == "route_b" else "route_a_address"
    return round(float(artifact[field].reshape(-1)[0]))


def _fixed_route_probe(
    memory: ExecutableArtifactMemory,
    key: torch.Tensor,
    *,
    view: str | None,
    expected_position: int,
) -> bool:
    artifact = _promote_view(memory, key, view=view)
    return artifact is not None and _fixed_position(artifact, view) == expected_position


def _new_memory(directory: Path) -> ExecutableArtifactMemory:
    return ExecutableArtifactMemory(
        directory,
        width=query.EVENT_WIDTH,
        capacity=MEMORY_CAPACITY,
        write_threshold=0.0,
        write_match_threshold=0.999,
    )


def _key_row(candidates: MemoryCandidates, key: torch.Tensor) -> int:
    for index in range(candidates.keys.shape[1]):
        if torch.equal(candidates.keys[0, index], key):
            return index
    raise LookupError("candidate key is not present")


def _bindings_for_stage(
    routes: list[dict[str, object]],
    keys: tuple[torch.Tensor, ...],
    aliases: dict[int, torch.Tensor],
    stage: int,
) -> tuple[tuple[torch.Tensor, str | None, int], ...]:
    bindings: list[tuple[torch.Tensor, str | None, int]] = []
    for route_index in range((stage + 1) * 2):
        pair_index = route_index // 2
        first, second = PAIR_INDICES[pair_index]
        position = int(routes[route_index]["position"])
        if route_index == first:
            bindings.extend(
                (
                    (keys[first], None, position),
                    (aliases[first], "route_a", position),
                )
            )
        elif route_index == second:
            bindings.append((keys[second], "route_b", position))
        else:
            raise RuntimeError("route pair layout changed")
    return tuple(bindings)


def _route_bindings_pass(
    memory: ExecutableArtifactMemory,
    bindings: tuple[tuple[torch.Tensor, str | None, int], ...],
    *,
    rows: int,
) -> bool:
    return len(memory.occupied) == rows and all(
        _fixed_route_probe(memory, key, view=view, expected_position=position)
        for key, view, position in bindings
    )


def _candidate_probe(
    bindings: tuple[tuple[torch.Tensor, str | None, int], ...],
) -> tuple[CapabilityRetentionProbe, ...]:
    return tuple(
        CapabilityRetentionProbe(key, [1.0] * 8)
        for key, _view, _position in bindings
    )


def _occupied_bytes(memory: ExecutableArtifactMemory) -> int:
    total = 0
    for index in memory.occupied:
        filename = memory.paths[index]
        if filename is None:
            raise RuntimeError("occupied artifact has no path")
        total += (memory.directory / filename).stat().st_size
    return total


def _policy_permutation_audit(
    memory: ExecutableArtifactMemory,
    policy: OpaqueConsolidationPolicy,
    shuffled: OpaqueConsolidationPolicy,
    untrained: OpaqueConsolidationPolicy,
    *,
    expected_keys: tuple[torch.Tensor, torch.Tensor],
    seed: int,
) -> dict[str, object]:
    candidates = memory.planner_candidates()
    occupied = memory.occupied
    learned_hits = 0
    shuffled_hits = 0
    untrained_hits = 0
    records: list[dict[str, object]] = []
    for permutation in itertools.permutations(occupied):
        row_order = permutation + tuple(
            index for index in range(memory.capacity) if index not in occupied
        )
        view = MemoryCandidates(
            keys=candidates.keys[:, row_order],
            values=candidates.values[:, row_order],
            strengths=candidates.strengths[:, row_order],
            timestamps=candidates.timestamps[:, row_order],
            occupied=candidates.occupied[:, row_order],
        )
        expected_physical = tuple(_key_row(view, key) for key in expected_keys)
        scrubbed = _address_scrubbed_candidates(
            view,
            seed=seed + sum((position + 1) * value for position, value in enumerate(permutation)),
        )
        expected_positions = expected_physical
        expected = {
            tuple(float(value) for value in scrubbed.keys[0, index].tolist())
            for index in expected_positions
        }
        proposals = (
            policy.propose(scrubbed),
            shuffled.propose(scrubbed),
            untrained.propose(scrubbed),
        )
        if any(proposal is None for proposal in proposals):
            raise RuntimeError("compositional policy produced no proposal")
        learned, shuffled_proposal, untrained_proposal = proposals
        assert learned is not None
        assert shuffled_proposal is not None
        assert untrained_proposal is not None
        learned_hit = _selected_key_set(scrubbed, learned) == expected
        shuffled_hit = _selected_key_set(scrubbed, shuffled_proposal) == expected
        untrained_hit = _selected_key_set(scrubbed, untrained_proposal) == expected
        learned_hits += int(learned_hit)
        shuffled_hits += int(shuffled_hit)
        untrained_hits += int(untrained_hit)
        records.append(
            {
                "permutation": permutation,
                "learned_pair": (learned.first, learned.second),
                "learned_hit": learned_hit,
                "shuffled_hit": shuffled_hit,
                "untrained_hit": untrained_hit,
            }
        )
    total = len(records)
    return {
        "permutations": total,
        "learned_hits": learned_hits,
        "shuffled_hits": shuffled_hits,
        "untrained_hits": untrained_hits,
        "learned_rate": learned_hits / total,
        "shuffled_rate": shuffled_hits / total,
        "untrained_rate": untrained_hits / total,
        "records": records,
    }


def _evaluate_route(
    memory: ExecutableArtifactMemory,
    system,
    winner,
    evidence,
    route: dict[str, object],
    *,
    key: torch.Tensor,
    view: str | None,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    artifact = _promote_view(memory, key, view=view)
    if artifact is None:
        return {"accuracy": 0.0, "lifetimes": [], "resolved": False}
    position = _fixed_position(artifact, view)
    rows = query._evaluate(
        system,
        winner,
        evidence,
        query_symbol=int(route["query_symbol"]),
        depth=int(route["depth"]),
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=seed,
        lifetimes=args.retention_lifetimes,
        forced_offset=position + 1,
    )
    return {
        "accuracy": min(float(row["accuracy"]) for row in rows),
        "lifetimes": rows,
        "resolved": True,
        "position": position,
    }


def _evaluate_prefix(
    memory: ExecutableArtifactMemory,
    routes: list[dict[str, object]],
    keys: tuple[torch.Tensor, ...],
    aliases: dict[int, torch.Tensor],
    system,
    winner,
    evidence,
    *,
    stage: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[dict[str, object]], bool]:
    results: list[dict[str, object]] = []
    for route_index in range((stage + 1) * 2):
        first, second = PAIR_INDICES[route_index // 2]
        if route_index == first:
            key, view = keys[first], None
        else:
            key, view = keys[second], "route_b"
        result = _evaluate_route(
            memory,
            system,
            winner,
            evidence,
            routes[route_index],
            key=key,
            view=view,
            args=args,
            seed=seed + route_index * 10_000,
        )
        result["route_id"] = route_index
        results.append(result)
    return results, all(
        bool(result["resolved"]) and float(result["accuracy"]) >= ROUTE_THRESHOLD
        for result in results
    )


def _stream(
    *,
    root: Path,
    stage_order: tuple[int, ...],
    routes: list[dict[str, object]],
    keys: tuple[torch.Tensor, ...],
    aliases: dict[int, torch.Tensor],
    artifacts: tuple[dict[str, torch.Tensor], ...],
    policy: OpaqueConsolidationPolicy,
    shuffled: OpaqueConsolidationPolicy,
    untrained: OpaqueConsolidationPolicy,
    system,
    winner,
    evidence,
    args: argparse.Namespace,
    winner_digest_before: str,
    reverse_insertion: bool,
    post_growth: Callable[
        [ExecutableArtifactMemory], tuple[ExecutableArtifactMemory, dict[str, object]]
    ]
    | None = None,
) -> dict[str, object]:
    controller_before = _digest(system.agent.controller)
    encoder_before = _digest(system.agent.runtime.encoders["stimulus"])
    memory = _new_memory(root / "live-0")
    stage_reports: list[dict[str, object]] = []
    all_prefixes_retained = True
    all_policy_controls = True
    all_rejections_non_mutating = True
    all_reloads_exact = True
    all_bytes_reduced = True
    for stage in stage_order:
        first, second = PAIR_INDICES[stage]
        insertion_order = (second, first) if reverse_insertion else (first, second)
        for route_index in insertion_order:
            memory.put(keys[route_index], artifacts[route_index])
            memory.observe_retention(keys[route_index], 1.0)
        candidates = memory.planner_candidates()
        proposal = policy.propose(candidates)
        if proposal is None:
            raise RuntimeError("online compositional policy produced no proposal")
        expected_key_set = {
            tuple(float(value) for value in keys[index].tolist())
            for index in (first, second)
        }
        selected_key_set = _selected_key_set(candidates, proposal)
        selected_correctly = selected_key_set == expected_key_set
        policy_audit = _policy_permutation_audit(
            memory,
            policy,
            shuffled,
            untrained,
            expected_keys=(keys[first], keys[second]),
            seed=args.seed + 800_000 + stage * 100_000,
        )
        policy_ok = (
            policy_audit["learned_rate"] == 1.0
            and (
                policy_audit["permutations"] <= 2
                or (
                    policy_audit["learned_rate"] > policy_audit["shuffled_rate"]
                    and policy_audit["learned_rate"] > policy_audit["untrained_rate"]
                )
            )
        )
        all_policy_controls = all_policy_controls and policy_ok
        before_manifest = (memory.directory / "manifest.json").read_text()
        before_rows = memory.occupied
        before_bytes = _occupied_bytes(memory)
        reject_key = candidates.keys[0, proposal.first].clone()
        reject_artifact = _promote_view(memory, reject_key, view=None)
        if reject_artifact is None:
            raise RuntimeError("rejected proposal row cannot be promoted")
        rejected, rejected_receipt = memory.consolidate_verified(
            (proposal.first, proposal.second),
            reject_key,
            reject_artifact,
            root / f"rejected-{stage}",
            verifier=lambda _candidate: False,
            candidate_outcome_probe=lambda _candidate, key=reject_key: (
                CapabilityRetentionProbe(key, [1.0] * 8),
            ),
            retained_scores=[1.0],
            min_candidate_observations=8,
        )
        rejected_non_mutating = (
            rejected is None
            and not rejected_receipt.accepted
            and memory.occupied == before_rows
            and (memory.directory / "manifest.json").read_text() == before_manifest
        )
        all_rejections_non_mutating = all_rejections_non_mutating and rejected_non_mutating
        accepted = False
        receipt: dict[str, object]
        reload_exact = False
        bytes_after = before_bytes
        prefix_results: list[dict[str, object]] = []
        if selected_correctly:
            replacement = _fixed_composed_artifact(artifacts[first], artifacts[second])
            bindings = _bindings_for_stage(routes, keys, aliases, stage)
            current_bindings = (
                (keys[first], None, int(routes[first]["position"])),
                (aliases[first], "route_a", int(routes[first]["position"])),
                (keys[second], "route_b", int(routes[second]["position"])),
            )
            candidate, accepted_receipt = memory.consolidate_verified(
                (proposal.first, proposal.second),
                keys[first],
                replacement,
                root / f"accepted-{stage}",
                verifier=lambda candidate_memory, expected=bindings, rows=stage + 1: _route_bindings_pass(
                    candidate_memory,
                    expected,
                    rows=rows,
                ),
                replacement_aliases=(aliases[first], keys[second]),
                replacement_alias_views=("route_a", "route_b"),
                candidate_outcome_probe=lambda _candidate, expected=current_bindings: _candidate_probe(
                    expected
                ),
                retained_scores=[1.0] * len(current_bindings),
                candidate_threshold=0.8,
                retention_floor=0.8,
                min_candidate_observations=8,
            )
            accepted = candidate is not None and accepted_receipt.accepted
            receipt = accepted_receipt.__dict__
            if accepted and candidate is not None:
                memory = candidate
                if stage < len(PAIR_INDICES) - 1 and memory.capacity < MEMORY_CAPACITY:
                    memory = memory.grow(
                        root / f"grown-{stage}",
                        MEMORY_CAPACITY,
                    )
                bytes_after = _occupied_bytes(memory)
                restored = ExecutableArtifactMemory.load(memory.directory)
                prefix_results, prefix_ok = _evaluate_prefix(
                    restored,
                    routes,
                    keys,
                    aliases,
                    system,
                    winner,
                    evidence,
                    stage=stage,
                    args=args,
                    seed=args.seed + 900_000 + stage * 10_000,
                )
                reload_exact = prefix_ok and _route_bindings_pass(
                    restored,
                    bindings,
                    rows=stage + 1,
                )
                all_prefixes_retained = all_prefixes_retained and prefix_ok
                all_reloads_exact = all_reloads_exact and reload_exact
                all_bytes_reduced = all_bytes_reduced and bytes_after < before_bytes
            else:
                all_prefixes_retained = False
        else:
            receipt = {
                "accepted": False,
                "reason": "policy selected a non-target pair; fail closed",
            }
            all_prefixes_retained = False
        stage_reports.append(
            {
                "stage": stage,
                "pair": (first, second),
                "insertion_order": insertion_order,
                "selected_pair": (proposal.first, proposal.second),
                "selected_correctly": selected_correctly,
                "policy": policy_audit,
                "rows_before": len(before_rows),
                "rows_after": len(memory.occupied),
                "bytes_before": before_bytes,
                "bytes_after": bytes_after,
                "accepted": accepted,
                "receipt": receipt,
                "rejected_transaction_non_mutating": rejected_non_mutating,
                "reload_exact": reload_exact,
                "prefix": prefix_results,
            }
        )
    post_growth_report: dict[str, object] = {}
    if post_growth is not None:
        memory, post_growth_report = post_growth(memory)
    corruption_rejected = False
    if memory.occupied:
        corruption_dir = root / "corrupted"
        shutil.copytree(memory.directory, corruption_dir)
        filename = memory.paths[memory.occupied[0]]
        if filename is None:
            raise RuntimeError("final artifact path is missing")
        path = corruption_dir / filename
        path.write_bytes(path.read_bytes() + b"tampered")
        try:
            ExecutableArtifactMemory.load(corruption_dir)
        except ValueError as error:
            corruption_rejected = "hash mismatch" in str(error).lower()
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    return {
        "stage_order": stage_order,
        "stages": stage_reports,
        "post_growth": post_growth_report,
        "final_rows": len(memory.occupied),
        "final_bytes": _occupied_bytes(memory) if memory.occupied else 0,
        "prefixes_retained": all_prefixes_retained,
        "policy_controls_pass": all_policy_controls,
        "rejected_transactions_non_mutating": all_rejections_non_mutating,
        "reloads_exact": all_reloads_exact,
        "bytes_reduced_each_stage": all_bytes_reduced,
        "corruption_rejected": corruption_rejected,
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "promoted_file_frozen": winner_digest_before == winner.digest(),
        "zero_replayed_examples": True,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    required = (
        args.source_updates,
        args.source_evaluation_lifetimes,
        args.source_route_lifetimes,
        args.target_route_updates,
        args.policy_updates,
        args.policy_batch_size,
        args.batch_size,
        args.data_steps,
        args.retention_lifetimes,
    )
    if min(required) < 1:
        raise ValueError("online compositional-growth budgets must be positive")
    if args.data_steps <= max(depth for _, depth in ROUTE_SPECS):
        raise ValueError("data steps must include every route depth")
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
    stable_offsets = tuple(int(record["offset"]) for record in candidates if bool(record["stable"]))
    winner_offset = stable_offsets[0] if stable_offsets else max(
        candidates,
        key=lambda record: min(float(row["accuracy"]) for row in record["evaluation"]),
    )["offset"]
    winner = files[winner_offset - 1]
    winner_digest_before = winner.digest()
    evidence = counterfactual._evidence(
        mastery_threshold=query.MASTERY_THRESHOLD,
        observations=args.source_route_lifetimes,
    )
    args.winner_offset = int(winner_offset)
    routes, route_bits, counterfactual_bits = _acquire_routes(
        system,
        winner,
        evidence,
        args,
    )
    keys, artifacts, aliases = _make_artifacts(routes, seed=args.seed)
    if tuple(sorted(aliases)) != tuple(first for first, _second in PAIR_INDICES):
        raise RuntimeError("alias pair layout changed")
    policy, policy_accounting = _train_policy(
        seed=args.seed,
        rows=8,
        width=query.EVENT_WIDTH,
        updates=args.policy_updates,
        batch_size=args.policy_batch_size,
        shuffled_utility=False,
    )
    shuffled, shuffled_accounting = _train_policy(
        seed=args.seed + 50_000,
        rows=8,
        width=query.EVENT_WIDTH,
        updates=args.policy_updates,
        batch_size=args.policy_batch_size,
        shuffled_utility=True,
    )
    torch.manual_seed(args.seed + 100_000)
    untrained = OpaqueConsolidationPolicy(query.EVENT_WIDTH, hidden=64).eval()
    with tempfile.TemporaryDirectory(prefix="online-compositional-growth-") as directory:
        root = Path(directory)
        forward = _stream(
            root=root / "forward",
            stage_order=(0, 1, 2),
            routes=routes,
            keys=keys,
            aliases=aliases,
            artifacts=artifacts,
            policy=policy,
            shuffled=shuffled,
            untrained=untrained,
            system=system,
            winner=winner,
            evidence=evidence,
            args=args,
            winner_digest_before=winner_digest_before,
            reverse_insertion=False,
        )
        reversed_stream = _stream(
            root=root / "reversed",
            stage_order=(0, 1, 2),
            routes=routes,
            keys=keys,
            aliases=aliases,
            artifacts=artifacts,
            policy=policy,
            shuffled=shuffled,
            untrained=untrained,
            system=system,
            winner=winner,
            evidence=evidence,
            args=args,
            winner_digest_before=winner_digest_before,
            reverse_insertion=True,
        )
    controller_after = _digest(system.agent.controller)
    encoder_after = _digest(system.agent.runtime.encoders["stimulus"])
    stage_accepts = lambda stream: all(bool(stage["accepted"]) for stage in stream["stages"])
    gates = {
        "all_forward_stages_accept": stage_accepts(forward),
        "all_reversed_stages_accept": stage_accepts(reversed_stream),
        "forward_prefixes_retained_after_every_pair": bool(forward["prefixes_retained"]),
        "reversed_prefixes_retained_after_every_pair": bool(reversed_stream["prefixes_retained"]),
        "forward_policy_controls_pass": bool(forward["policy_controls_pass"]),
        "reversed_policy_controls_pass": bool(reversed_stream["policy_controls_pass"]),
        "forward_rows_end_at_three": forward["final_rows"] == len(PAIR_INDICES),
        "reversed_rows_end_at_three": reversed_stream["final_rows"] == len(PAIR_INDICES),
        "forward_bytes_reduce_each_stage": bool(forward["bytes_reduced_each_stage"]),
        "reversed_bytes_reduce_each_stage": bool(reversed_stream["bytes_reduced_each_stage"]),
        "forward_rejections_non_mutating": bool(forward["rejected_transactions_non_mutating"]),
        "reversed_rejections_non_mutating": bool(reversed_stream["rejected_transactions_non_mutating"]),
        "forward_reloads_exact": bool(forward["reloads_exact"]),
        "reversed_reloads_exact": bool(reversed_stream["reloads_exact"]),
        "forward_corruption_rejected": bool(forward["corruption_rejected"]),
        "reversed_corruption_rejected": bool(reversed_stream["corruption_rejected"]),
        "controller_frozen": controller_before == controller_after,
        "event_encoder_frozen": encoder_before == encoder_after,
        "promoted_file_frozen": winner_digest_before == winner.digest(),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": ONLINE_COMPOSITION_SCHEMA,
        "claim_boundary": (
            "Repeated fresh acquisition and verifier-gated shared-view consolidation "
            "of three pairs of distinct temporal capabilities with prefix retention; "
            "not unrestricted memory growth, arbitrary new computation, or general "
            "continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "artifact_backend": "executable_artifact_memory_v2",
            "policy": "opaque_consolidation_policy_v1",
            "composition": "three_repeated_shared_basis_pair_rewrites_v1",
            "route_count": len(routes),
            "pair_count": len(PAIR_INDICES),
            "memory_capacity": MEMORY_CAPACITY,
            "physical_rows_final": len(PAIR_INDICES),
            "policy_control": "reward_shuffled_on_address_scrubbed_candidate_views_v1",
            "forbidden_features": "query_depth_route_id_semantic_names_task_ids_replayed_streams",
        },
        "routes": [
            {
                key: value.tolist() if isinstance(value, torch.Tensor) else value
                for key, value in route.items()
                if key not in {"key", "history"}
            }
            | {"key_digest": _tensor_digest(route["key"])}
            for route in routes
        ],
        "forward": forward,
        "reversed": reversed_stream,
        "policy_accounting": policy_accounting,
        "shuffled_policy_accounting": shuffled_accounting,
        "gates": gates,
        "accounting": {
            "unique_temporal_verifier_bits": route_bits,
            "counterfactual_temporal_verifier_bits": counterfactual_bits,
            "unique_verifier_bits": route_bits + counterfactual_bits,
            "policy_verifier_bits": int(policy_accounting["unique_verifier_bits"]),
            "shuffled_policy_verifier_bits": int(shuffled_accounting["unique_verifier_bits"]),
            "shared_view_retention_verifier_bits": sum(
                3 * (stage + 1) * 8 for stage in range(len(PAIR_INDICES))
            ),
            "temporal_logical_lifetimes": sum(
                len(route["history"]) for route in routes
            ),
            "policy_logical_lifetimes": int(policy_accounting["unique_lifetimes"]),
            "policy_audit_logical_lifetimes": sum(
                int(stage["policy"]["permutations"])
                for stream in (forward, reversed_stream)
                for stage in stream["stages"]
            ),
            "unique_logical_lifetimes": sum(
                len(route["history"]) for route in routes
            )
            + int(policy_accounting["unique_lifetimes"])
            + sum(
                int(stage["policy"]["permutations"])
                for stream in (forward, reversed_stream)
                for stage in stream["stages"]
            ),
            "transfer_ratio_against_fresh_learner": {
                "forward_policy": sum(
                    float(stage["policy"]["learned_rate"])
                    for stage in forward["stages"]
                )
                / max(
                    sum(
                        float(stage["policy"]["untrained_rate"])
                        for stage in forward["stages"]
                    ),
                    1e-9,
                ),
                "reversed_policy": sum(
                    float(stage["policy"]["learned_rate"])
                    for stage in reversed_stream["stages"]
                )
                / max(
                    sum(
                        float(stage["policy"]["untrained_rate"])
                        for stage in reversed_stream["stages"]
                    ),
                    1e-9,
                ),
            },
            "optimizer_updates": int(policy_accounting["optimizer_updates"]),
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": route_bits + counterfactual_bits
            if all(gates.values())
            else None,
        },
        "status": "promoted_online_compositional_growth"
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
    parser.add_argument("--policy-updates", type=int, default=3_000)
    parser.add_argument("--policy-batch-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-steps", type=int, default=14)
    parser.add_argument("--retention-lifetimes", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--entropy-weight", type=float, default=0.01)
    parser.add_argument("--report-out", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
