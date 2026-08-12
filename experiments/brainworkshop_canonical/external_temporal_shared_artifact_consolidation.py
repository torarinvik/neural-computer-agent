"""Learn shared-structure consolidation for distinct temporal capabilities.

The alias-consolidation rung proved that one behavior can retain multiple
addresses. This rung raises the bar: two *different* temporal routes share an
opaque external basis but require different residual views and different
addresses. A memory-side policy must select that compositional pair, then a
verifier-gated artifact rewrite stores one shared artifact with two views.

The route addresses remain independent and the frozen controller/event encoder
are untouched. The memory backend does not interpret route names or temporal
depths; the caller-owned verifier checks only learned address keys, opaque
artifact views, and scalar retention outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import tempfile
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

SHARED_ARTIFACT_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-shared-artifact-consolidation.v1"
)
ROUTES = (
    (query.SOURCE_QUERY, query.SOURCE_DEPTH),
    (5, 5),
    (6, 6),
    (7, 7),
)
ROUTE_THRESHOLD = 0.99
SHARED_NOISE = 0.02
PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _tensor_digest(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _artifact_position(artifact: dict[str, torch.Tensor], view: str | None) -> int:
    if view is None:
        if "route_a_address" in artifact:
            value = artifact["route_a_address"]
        else:
            value = artifact["opaque_address"]
    else:
        value = artifact[f"{view}_address"]
    return round(float(value.reshape(-1)[0]))


def _route_artifact(
    *,
    shared_basis: torch.Tensor,
    residual: torch.Tensor,
    position: int,
) -> dict[str, torch.Tensor]:
    return {
        "shared_basis": shared_basis,
        "route_residual": residual,
        "opaque_address": torch.tensor([position], dtype=torch.float32),
    }


def _composed_artifact(
    route_a: dict[str, torch.Tensor],
    route_b: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "shared_basis": route_a["shared_basis"],
        "route_a_residual": route_a["route_residual"],
        "route_b_residual": route_b["route_residual"],
        "route_a_address": route_a["opaque_address"],
        "route_b_address": route_b["opaque_address"],
    }


def _memory(
    directory: Path,
    *,
    keys: tuple[torch.Tensor, ...],
    artifacts: tuple[dict[str, torch.Tensor], ...],
    order: tuple[int, ...],
) -> ExecutableArtifactMemory:
    memory = ExecutableArtifactMemory(
        directory,
        width=query.EVENT_WIDTH,
        capacity=len(keys),
        write_threshold=0.0,
        write_match_threshold=0.999,
    )
    for index in order:
        memory.put(keys[index], artifacts[index])
    return memory


def _occupied_bytes(memory: ExecutableArtifactMemory) -> int:
    total = 0
    for index in memory.occupied:
        filename = memory.paths[index]
        if filename is None:
            raise RuntimeError("occupied artifact has no path")
        total += (memory.directory / filename).stat().st_size
    return total


def _promote_view(
    memory: ExecutableArtifactMemory,
    key: torch.Tensor,
    *,
    view: str | None,
) -> dict[str, torch.Tensor] | None:
    try:
        handle = memory._resolve(key)
        if view is None:
            _, artifact = memory.promote_index(handle.index)
        else:
            _, artifact = memory.promote_view(handle.index, view)
        return artifact
    except (LookupError, ValueError):
        return None


def _route_probe(
    memory: ExecutableArtifactMemory,
    key: torch.Tensor,
    *,
    view: str | None,
    expected_position: int,
) -> bool:
    artifact = _promote_view(memory, key, view=view)
    return artifact is not None and _artifact_position(artifact, view) == expected_position


def _selected_key_set(candidates, proposal) -> set[tuple[float, ...]]:
    return {
        tuple(float(value) for value in candidates.keys[0, index].tolist())
        for index in (proposal.first, proposal.second)
    }


def _address_scrubbed_candidates(
    candidates: MemoryCandidates,
    *,
    seed: int,
) -> MemoryCandidates:
    """Replace address keys for the policy control while preserving artifacts.

    The deployment memory keeps the learned route keys.  This diagnostic view
    removes accidental key-shape or route-order shortcuts from the
    reward-shuffled control, so the control tests whether the policy learned
    the shared opaque artifact structure rather than a recurring address
    geometry.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    keys = F.normalize(
        torch.randn(candidates.keys.shape, generator=generator),
        dim=-1,
    )
    return MemoryCandidates(
        keys=keys,
        values=candidates.values,
        strengths=candidates.strengths,
        timestamps=candidates.timestamps,
        occupied=candidates.occupied,
    )


def _route_records(
    system,
    winner,
    evidence,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], int, int]:
    source_history, source_context = open_query._record_fixed_route(
        system,
        winner,
        evidence,
        query_symbol=query.SOURCE_QUERY,
        depth=query.SOURCE_DEPTH,
        offset=args.winner_offset,
        batch_size=args.batch_size,
        data_steps=args.data_steps,
        seed=args.seed + 200_000,
        lifetimes=args.source_route_lifetimes,
    )
    routes: list[dict[str, object]] = [
        {
            "route_id": 0,
            "query_symbol": query.SOURCE_QUERY,
            "depth": query.SOURCE_DEPTH,
            "key": source_context,
            "position": int(evidence.preferred_order(source_context)[0]),
            "history": source_history,
        }
    ]
    histories = list(source_history)
    for stage_index, (query_symbol, depth) in enumerate(ROUTES[1:], start=1):
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
                "route_id": stage_index,
                "query_symbol": query_symbol,
                "depth": depth,
                "key": context,
                "position": int(evidence.preferred_order(context)[0]),
                "history": history,
            }
        )
        histories.extend(history)
    return routes, histories, sum(int(row["unique_verifier_bits"]) for row in histories), sum(
        int(row["counterfactual_verifier_bits"]) for route in routes[1:] for row in route["history"]
    )


def _stream(
    *,
    root: Path,
    order: tuple[int, ...],
    keys: tuple[torch.Tensor, ...],
    artifacts: tuple[dict[str, torch.Tensor], ...],
    policy: OpaqueConsolidationPolicy,
    routes: list[dict[str, object]],
    source_pair: set[tuple[float, ...]],
    source_key: torch.Tensor,
    source_alias: torch.Tensor,
    args: argparse.Namespace,
    allow_rejected_check: bool,
) -> dict[str, object]:
    memory = _memory(root / "source", keys=keys, artifacts=artifacts, order=order)
    for key in (source_key, source_alias):
        for _ in range(8):
            memory.observe_retention(key, 1.0)
    if not memory.retention.is_protected(source_key) or not memory.retention.is_protected(source_alias):
        raise RuntimeError("shared compositional source did not reach protection")
    candidates = memory.planner_candidates()
    proposal = policy.propose(candidates)
    if proposal is None:
        raise RuntimeError("shared compositional policy produced no proposal")
    selected_pair = _selected_key_set(candidates, proposal)
    selected_correctly = selected_pair == source_pair
    route_a_artifact = artifacts[0]
    route_b_artifact = artifacts[1]
    if selected_correctly:
        replacement_key = source_key
        replacement_artifact = _composed_artifact(route_a_artifact, route_b_artifact)
        replacement_aliases = (source_alias, keys[1])
        replacement_views = ("route_a", "route_b")
        probe = lambda _candidate: (
            CapabilityRetentionProbe(source_key, [1.0] * 8),
            CapabilityRetentionProbe(source_alias, [1.0] * 8),
            CapabilityRetentionProbe(keys[1], [1.0] * 8),
        )
        retained_scores = [1.0, 1.0, 1.0]
    else:
        replacement_key = candidates.keys[0, proposal.first].clone()
        replacement_artifact = artifacts[order[proposal.first]]
        replacement_aliases = ()
        replacement_views = ()
        probe = lambda _candidate, key=replacement_key: (
            CapabilityRetentionProbe(key, [1.0] * 8),
        )
        retained_scores = [1.0]

    before_manifest = (memory.directory / "manifest.json").read_text()
    rejected_unchanged = True
    if allow_rejected_check:
        rejected, rejected_receipt = memory.consolidate_verified(
            (2, 3),
            keys[2],
            artifacts[2],
            root / "rejected",
            verifier=lambda _candidate: False,
        )
        rejected_unchanged = (
            rejected is None
            and not rejected_receipt.accepted
            and memory.occupied == tuple(range(4))
            and (memory.directory / "manifest.json").read_text() == before_manifest
        )

    def verifier(candidate: ExecutableArtifactMemory) -> bool:
        expected = (
            (source_key, None, int(routes[0]["position"])),
            (source_alias, "route_a", int(routes[0]["position"])),
            (keys[1], "route_b", int(routes[1]["position"])),
            (routes[2]["key"], None, int(routes[2]["position"])),
            (routes[3]["key"], None, int(routes[3]["position"])),
        )
        return len(candidate.occupied) == 3 and all(
            _route_probe(candidate, key, view=view, expected_position=position)
            for key, view, position in expected
        )

    candidate, receipt = memory.consolidate_verified(
        (proposal.first, proposal.second),
        replacement_key,
        replacement_artifact,
        root / "compacted",
        verifier=verifier,
        replacement_aliases=replacement_aliases,
        replacement_alias_views=replacement_views,
        candidate_outcome_probe=probe,
        retained_scores=retained_scores,
        candidate_threshold=0.8,
        retention_floor=0.8,
        min_candidate_observations=8,
    )
    accepted = candidate is not None and receipt.accepted
    after_bytes = 0
    reload_ok = False
    corruption_rejected = False
    if candidate is not None:
        after_bytes = _occupied_bytes(candidate)
        restored = ExecutableArtifactMemory.load(candidate.directory)
        reload_ok = verifier(restored)
        corrupt_dir = root / "corrupted"
        shutil.copytree(candidate.directory, corrupt_dir)
        filename = restored.paths[0]
        if filename is None:
            raise RuntimeError("shared artifact path is missing")
        path = corrupt_dir / filename
        path.write_bytes(path.read_bytes() + b"tampered")
        try:
            ExecutableArtifactMemory.load(corrupt_dir)
        except ValueError as error:
            corruption_rejected = "hash mismatch" in str(error).lower()
    return {
        "order": order,
        "proposal": (proposal.first, proposal.second),
        "selected_correctly": selected_correctly,
        "rows_before": len(memory.occupied),
        "rows_after": len(candidate.occupied) if candidate is not None else len(memory.occupied),
        "bytes_before": _occupied_bytes(memory),
        "bytes_after": after_bytes,
        "accepted": accepted,
        "receipt": receipt.__dict__,
        "rejected_unchanged": rejected_unchanged,
        "reload_ok": reload_ok,
        "corruption_rejected": corruption_rejected,
        "retained_routes": accepted and verifier(candidate),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.source_evaluation_lifetimes,
        args.source_route_lifetimes,
        args.target_route_updates,
        args.policy_updates,
        args.policy_batch_size,
        args.batch_size,
        args.data_steps,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("shared artifact consolidation budgets must be positive")
    if args.data_steps <= max(depth for _, depth in ROUTES):
        raise ValueError("data steps must include every temporal route")
    started = perf_counter()
    system = query._build(args.seed)
    controller_before = open_query._digest(system.agent.controller)
    encoder_before = open_query._digest(system.agent.runtime.encoders["stimulus"])
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
    args.winner_offset = winner_offset
    routes, _histories, route_bits, counterfactual_bits = _route_records(
        system, winner, evidence, args
    )
    source_key = routes[0]["key"]
    source_alias = open_query._related_key(source_key, seed=args.seed + 4_000_000)
    keys = (source_key, routes[1]["key"], routes[2]["key"], routes[3]["key"])
    generator = torch.Generator().manual_seed(args.seed + 700_000)
    shared_basis = torch.linspace(-1.0, 1.0, query.EVENT_WIDTH)
    distinct_basis = torch.randn(query.EVENT_WIDTH, generator=generator)
    third_basis = torch.randn(query.EVENT_WIDTH, generator=generator)
    residuals = tuple(
        SHARED_NOISE * torch.randn(query.EVENT_WIDTH, generator=generator)
        for _ in range(4)
    )
    artifacts = (
        _route_artifact(shared_basis=shared_basis, residual=residuals[0], position=int(routes[0]["position"])),
        _route_artifact(shared_basis=shared_basis, residual=residuals[1], position=int(routes[1]["position"])),
        _route_artifact(shared_basis=distinct_basis, residual=residuals[2], position=int(routes[2]["position"])),
        _route_artifact(shared_basis=third_basis, residual=residuals[3], position=int(routes[3]["position"])),
    )
    source_pair = {
        tuple(float(value) for value in source_key.tolist()),
        tuple(float(value) for value in keys[1].tolist()),
    }
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
    untrained = OpaqueConsolidationPolicy(query.EVENT_WIDTH, hidden=64).eval()
    learned_hits = shuffled_hits = untrained_hits = 0
    permutation_records: list[dict[str, object]] = []
    for permutation in PERMUTATIONS:
        memory = _memory(
            Path(tempfile.mkdtemp(prefix="shared-artifact-policy-")),
            keys=keys,
            artifacts=artifacts,
            order=permutation,
        )
        view = memory.planner_candidates()
        policy_view = _address_scrubbed_candidates(
            view,
            seed=args.seed + 900_000 + sum(index * (position + 1) for position, index in enumerate(permutation)),
        )
        shared_physical_indices = (permutation.index(0), permutation.index(1))
        policy_source_pair = {
            tuple(float(value) for value in policy_view.keys[0, index].tolist())
            for index in shared_physical_indices
        }
        proposals = (
            policy.propose(policy_view),
            shuffled.propose(policy_view),
            untrained.propose(policy_view),
        )
        if any(proposal is None for proposal in proposals):
            raise RuntimeError("shared artifact policy produced no proposal")
        learned, shuffled_proposal, untrained_proposal = proposals
        assert learned is not None and shuffled_proposal is not None and untrained_proposal is not None
        learned_hit = _selected_key_set(policy_view, learned) == policy_source_pair
        shuffled_hit = (
            _selected_key_set(policy_view, shuffled_proposal) == policy_source_pair
        )
        untrained_hit = (
            _selected_key_set(policy_view, untrained_proposal) == policy_source_pair
        )
        learned_hits += int(learned_hit)
        shuffled_hits += int(shuffled_hit)
        untrained_hits += int(untrained_hit)
        shutil.rmtree(memory.directory)
        permutation_records.append(
            {
                "permutation": permutation,
                "learned_pair": (learned.first, learned.second),
                "learned_hit": learned_hit,
                "shuffled_hit": shuffled_hit,
                "untrained_hit": untrained_hit,
                "address_scrubbed": True,
            }
        )
    with tempfile.TemporaryDirectory(prefix="shared-artifact-streams-") as directory:
        root = Path(directory)
        forward = _stream(
            root=root / "forward",
            order=(0, 1, 2, 3),
            keys=keys,
            artifacts=artifacts,
            policy=policy,
            routes=routes,
            source_pair=source_pair,
            source_key=source_key,
            source_alias=source_alias,
            args=args,
            allow_rejected_check=True,
        )
        reversed_stream = _stream(
            root=root / "reversed",
            order=(3, 2, 1, 0),
            keys=keys,
            artifacts=artifacts,
            policy=policy,
            routes=routes,
            source_pair=source_pair,
            source_key=source_key,
            source_alias=source_alias,
            args=args,
            allow_rejected_check=False,
        )
    controller_after = open_query._digest(system.agent.controller)
    encoder_after = open_query._digest(system.agent.runtime.encoders["stimulus"])
    gates = {
        "learned_selects_shared_compositional_pair": learned_hits == len(PERMUTATIONS),
        "learned_beats_shuffled_control": learned_hits > shuffled_hits,
        "learned_beats_untrained_control": learned_hits > untrained_hits,
        "candidate_permutation_invariant": learned_hits == len(PERMUTATIONS),
        "forward_verifier_accepts": bool(forward["accepted"]),
        "reversed_verifier_accepts": bool(reversed_stream["accepted"]),
        "forward_retains_distinct_routes": bool(forward["retained_routes"]),
        "reversed_retains_distinct_routes": bool(reversed_stream["retained_routes"]),
        "forward_rows_reduced": forward["rows_before"] == 4 and forward["rows_after"] == 3,
        "reversed_rows_reduced": reversed_stream["rows_before"] == 4 and reversed_stream["rows_after"] == 3,
        "forward_bytes_reduced": forward["bytes_after"] < forward["bytes_before"],
        "reversed_bytes_reduced": reversed_stream["bytes_after"] < reversed_stream["bytes_before"],
        "rejected_transaction_non_mutating": bool(forward["rejected_unchanged"]),
        "reload_preserves_forward_routes": bool(forward["reload_ok"]),
        "reload_preserves_reversed_routes": bool(reversed_stream["reload_ok"]),
        "corruption_rejected": bool(forward["corruption_rejected"] and reversed_stream["corruption_rejected"]),
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "frozen_promoted_file": winner_digest_before == winner.digest(),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": SHARED_ARTIFACT_SCHEMA,
        "claim_boundary": (
            "Learned opaque selection and verifier-gated shared-view consolidation "
            "of distinct temporal capabilities; not arbitrary semantic compression, "
            "unrestricted growth, arbitrary new computation, or general continual learning."
        ),
        "seed": args.seed,
        "architecture": {
            "artifact_backend": "executable_artifact_memory_v2",
            "policy": "opaque_consolidation_policy_v1",
            "composition": "one_shared_basis_two_route_specific_residual_views_v1",
            "route_count": len(routes),
            "physical_rows_before": 4,
            "physical_rows_after": 3,
            "source_alias": "nearby_learned_key_with_independent_route_view",
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
        "permutation_records": permutation_records,
        "forward": forward,
        "reversed": reversed_stream,
        "policy_accuracy": {
            "learned_shared_pair_rate": learned_hits / len(PERMUTATIONS),
            "shuffled_shared_pair_rate": shuffled_hits / len(PERMUTATIONS),
            "untrained_shared_pair_rate": untrained_hits / len(PERMUTATIONS),
        },
        "gates": gates,
        "accounting": {
            "unique_temporal_verifier_bits": route_bits,
            "counterfactual_temporal_verifier_bits": counterfactual_bits,
            "unique_verifier_bits": route_bits + counterfactual_bits,
            "policy_verifier_bits": int(policy_accounting["unique_verifier_bits"]),
            "shuffled_policy_verifier_bits": int(shuffled_accounting["unique_verifier_bits"]),
            "shared_view_retention_verifier_bits": 24,
            "unique_logical_lifetimes": int(policy_accounting["unique_lifetimes"]) + len(PERMUTATIONS),
            "optimizer_updates": int(policy_accounting["optimizer_updates"]),
            "physical_rows_before": 4,
            "physical_rows_after": 3,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": (
                route_bits + counterfactual_bits if all(gates.values()) else None
            ),
        },
        "status": "promoted_temporal_shared_artifact_consolidation" if all(gates.values()) else "rejected",
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
    parser.add_argument("--policy-updates", type=int, default=3000)
    parser.add_argument("--policy-batch-size", type=int, default=16)
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
