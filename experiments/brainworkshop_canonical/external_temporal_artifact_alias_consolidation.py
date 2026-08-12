"""Compose temporal capability routes with verifier-gated artifact consolidation.

The temporal route learner acquires three opaque addresses while the
controller, event encoder, and capability readout are frozen.  A separate
external artifact memory then stores the learned route addresses.  One route
is intentionally represented twice under independent learned keys; a generic
memory-side consolidation policy must select that redundant pair, preserve
both exact/related aliases, and reduce physical storage without changing the
other routes.

The policy sees only opaque keys, opaque artifact summaries, strength, and age.
An independent temporal verifier owns candidate acceptance.  The experiment
therefore tests learned external compression at the same memory boundary used
by the temporal capability path, without adding a controller branch or
replaying old route-training streams.
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

from experiments.opaque_consolidation_amodal.train import _train_policy
from neural_computer import (
    CapabilityRetentionProbe,
    ExecutableArtifactMemory,
    OpaqueConsolidationPolicy,
)

from . import external_temporal_open_query_growth as open_query
from . import external_temporal_query_address_growth as query
from . import external_temporal_query_counterfactual_growth as counterfactual

ARTIFACT_SCHEMA = (
    "neural-computer.brainworkshop-external-temporal-artifact-alias-consolidation.v1"
)
ROUTES = ((query.SOURCE_QUERY, query.SOURCE_DEPTH), (5, 5), (6, 6))
SOURCE_ALIAS_NOISE = 0.20
ROUTE_THRESHOLD = 0.99
PERMUTATIONS = tuple(itertools.permutations(range(4)))


def _route_artifact(route: dict[str, object]) -> dict[str, torch.Tensor]:
    return {
        "opaque_address": torch.tensor(
            [float(route["position"])], dtype=torch.float32
        )
    }


def _position(artifact: dict[str, torch.Tensor]) -> int:
    return round(float(artifact["opaque_address"].reshape(-1)[0]))


def _tensor_digest(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _route_digest(module: torch.nn.Module) -> str:
    return module.digest()


def _route_probe(
    memory: ExecutableArtifactMemory,
    key: torch.Tensor,
    expected_position: int,
) -> bool:
    try:
        _handle, artifact = memory.promote(key)
    except (LookupError, ValueError):
        return False
    return _position(artifact) == expected_position


def _memory_for_order(
    directory: Path,
    *,
    keys: tuple[torch.Tensor, ...],
    artifacts: tuple[dict[str, torch.Tensor], ...],
) -> ExecutableArtifactMemory:
    memory = ExecutableArtifactMemory(
        directory,
        width=query.EVENT_WIDTH,
        capacity=len(keys),
        write_threshold=0.0,
        write_match_threshold=0.999,
    )
    for key, artifact in zip(keys, artifacts, strict=True):
        memory.put(key, artifact)
    return memory


def _selected_key_set(
    candidates, proposal
) -> set[tuple[float, ...]]:
    return {
        tuple(float(value) for value in candidates.keys[0, index].tolist())
        for index in (proposal.first, proposal.second)
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    if min(
        args.source_updates,
        args.source_evaluation_lifetimes,
        args.source_route_lifetimes,
        args.target_route_updates,
        args.policy_updates,
        args.policy_batch_size,
        args.retention_lifetimes,
    ) < 1:
        raise ValueError("artifact consolidation budgets must be positive")
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
    stable_offsets = tuple(
        int(record["offset"]) for record in candidates if bool(record["stable"])
    )
    winner_offset = stable_offsets[0] if stable_offsets else max(
        candidates,
        key=lambda record: min(float(row["accuracy"]) for row in record["evaluation"]),
    )["offset"]
    winner = files[winner_offset - 1]
    winner_digest_before = _route_digest(winner)
    evidence = counterfactual._evidence(
        mastery_threshold=query.MASTERY_THRESHOLD,
        observations=args.source_route_lifetimes,
    )
    route_records: list[dict[str, object]] = []
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
    route_records.append(
        {
            "route_id": 0,
            "query_symbol": query.SOURCE_QUERY,
            "depth": query.SOURCE_DEPTH,
            "key": source_context,
            "position": int(evidence.preferred_order(source_context)[0]),
            "history": source_history,
        }
    )
    target_histories: list[list[dict[str, object]]] = []
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
        route_records.append(
            {
                "route_id": stage_index,
                "query_symbol": query_symbol,
                "depth": depth,
                "key": context,
                "position": int(evidence.preferred_order(context)[0]),
                "history": history,
            }
        )
        target_histories.append(history)
    source = route_records[0]
    target = route_records[1]
    third = route_records[2]
    source_key = source["key"]
    source_alias = open_query._related_key(
        source_key, seed=args.seed + 4_000_000
    )
    route_keys = (
        source_key,
        source_alias,
        target["key"],
        third["key"],
    )
    route_artifacts = tuple(_route_artifact(route) for route in route_records)
    live_artifacts = (
        route_artifacts[0],
        route_artifacts[0],
        route_artifacts[1],
        route_artifacts[2],
    )
    source_pair = {
        tuple(float(value) for value in source_key.tolist()),
        tuple(float(value) for value in source_alias.tolist()),
    }

    learned_policy, policy_accounting = _train_policy(
        seed=args.seed,
        rows=8,
        width=query.EVENT_WIDTH,
        updates=args.policy_updates,
        batch_size=args.policy_batch_size,
        shuffled_utility=False,
    )
    shuffled_policy, shuffled_accounting = _train_policy(
        seed=args.seed + 50_000,
        rows=8,
        width=query.EVENT_WIDTH,
        updates=args.policy_updates,
        batch_size=args.policy_batch_size,
        shuffled_utility=True,
    )
    untrained_policy = OpaqueConsolidationPolicy(query.EVENT_WIDTH, hidden=64).eval()
    learned_hits = 0
    shuffled_hits = 0
    untrained_hits = 0
    permutation_records: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(
        prefix="neural-computer-temporal-artifact-consolidation-"
    ) as directory:
        root = Path(directory)
        for permutation in PERMUTATIONS:
            memory = _memory_for_order(
                root / f"perm-{''.join(str(value) for value in permutation)}",
                keys=tuple(route_keys[index] for index in permutation),
                artifacts=tuple(live_artifacts[index] for index in permutation),
            )
            candidates_view = memory.planner_candidates()
            learned_proposal = learned_policy.propose(candidates_view)
            shuffled_proposal = shuffled_policy.propose(candidates_view)
            untrained_proposal = untrained_policy.propose(candidates_view)
            if (
                learned_proposal is None
                or shuffled_proposal is None
                or untrained_proposal is None
            ):
                raise RuntimeError("artifact consolidation policy made no proposal")
            learned_selected = _selected_key_set(candidates_view, learned_proposal)
            shuffled_selected = _selected_key_set(candidates_view, shuffled_proposal)
            untrained_selected = _selected_key_set(candidates_view, untrained_proposal)
            learned_hit = learned_selected == source_pair
            shuffled_hit = shuffled_selected == source_pair
            untrained_hit = untrained_selected == source_pair
            learned_hits += int(learned_hit)
            shuffled_hits += int(shuffled_hit)
            untrained_hits += int(untrained_hit)
            permutation_records.append(
                {
                    "permutation": permutation,
                    "learned_pair": (learned_proposal.first, learned_proposal.second),
                    "learned_redundant_pair": learned_hit,
                    "shuffled_redundant_pair": shuffled_hit,
                    "untrained_redundant_pair": untrained_hit,
                }
            )

        canonical = _memory_for_order(
            root / "canonical",
            keys=route_keys,
            artifacts=live_artifacts,
        )
        for _ in range(8):
            canonical.observe_retention(source_key, 1.0)
            canonical.observe_retention(source_alias, 1.0)
        if not canonical.retention.is_protected(source_key):
            raise RuntimeError("source exact key did not reach protection")
        if not canonical.retention.is_protected(source_alias):
            raise RuntimeError("source alias key did not reach protection")
        before_manifest = (root / "canonical" / "manifest.json").read_text()
        before_occupied = canonical.occupied
        before_bad = root / "rejected"
        rejected_candidate, rejected_receipt = canonical.consolidate_verified(
            (2, 3),
            target["key"],
            route_artifacts[1],
            before_bad,
            verifier=lambda _candidate: False,
        )
        rejected_unchanged = (
            rejected_candidate is None
            and canonical.occupied == before_occupied
            and (root / "canonical" / "manifest.json").read_text()
            == before_manifest
        )
        live_candidates = canonical.planner_candidates()
        proposal = learned_policy.propose(live_candidates)
        if proposal is None:
            raise RuntimeError("learned policy made no canonical proposal")
        selected_pair = _selected_key_set(live_candidates, proposal)
        selected_correctly = selected_pair == source_pair
        if selected_correctly:
            replacement_key = source_key
            replacement_artifact = route_artifacts[0]
            replacement_aliases = (source_alias,)
            candidate_outcome_probe = lambda candidate: (
                CapabilityRetentionProbe(source_key, [1.0] * 8),
                CapabilityRetentionProbe(source_alias, [1.0] * 8),
            )
            retained_scores = [1.0, 1.0]
        else:
            # A weak/undertrained selector must fail closed.  Use the first
            # selected row as its own replacement key so the verifier can
            # reject the wrong semantic pair without manufacturing a key
            # collision or probing an alias absent from the candidate.
            replacement_key = live_candidates.keys[0, proposal.first].clone()
            replacement_artifact = live_artifacts[proposal.first]
            replacement_aliases = ()
            candidate_outcome_probe = lambda candidate, key=replacement_key: (
                CapabilityRetentionProbe(key, [1.0] * 8),
            )
            retained_scores = [1.0]

        def verifier(candidate: ExecutableArtifactMemory) -> bool:
            expected = (
                (source_key, int(source["position"])),
                (source_alias, int(source["position"])),
                (target["key"], int(target["position"])),
                (third["key"], int(third["position"])),
            )
            return len(candidate.occupied) == 3 and all(
                _route_probe(candidate, key, position)
                for key, position in expected
            )

        compacted, receipt = canonical.consolidate_verified(
            (proposal.first, proposal.second),
            replacement_key,
            replacement_artifact,
            root / "compacted",
            verifier=verifier,
            replacement_aliases=replacement_aliases,
            candidate_outcome_probe=candidate_outcome_probe,
            retained_scores=retained_scores,
            candidate_threshold=0.8,
            retention_floor=0.8,
            min_candidate_observations=8,
        )
        compacted_ok = bool(compacted is not None and receipt.accepted)
        reloaded_ok = False
        corruption_rejected = False
        if compacted is not None:
            restored = ExecutableArtifactMemory.load(compacted.directory)
            reloaded_ok = verifier(restored)
            corrupt_dir = root / "corrupted"
            shutil.copytree(compacted.directory, corrupt_dir)
            filename = restored.paths[0]
            if filename is None:
                raise RuntimeError("compacted artifact path is missing")
            artifact_path = corrupt_dir / filename
            artifact_path.write_bytes(artifact_path.read_bytes() + b"tampered")
            try:
                ExecutableArtifactMemory.load(corrupt_dir)
            except ValueError as error:
                corruption_rejected = "hash mismatch" in str(error).lower()
    controller_after = open_query._digest(system.agent.controller)
    encoder_after = open_query._digest(system.agent.runtime.encoders["stimulus"])
    route_training_bits = sum(
        int(row["unique_verifier_bits"])
        for row in source_history + [
            item for history in target_histories for item in history
        ]
    )
    candidate_training_bits = len(offsets) * args.source_updates * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    candidate_eval_bits = args.source_evaluation_lifetimes * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    compaction_verifier_bits = len(route_keys) * args.batch_size * (
        args.data_steps - query.SOURCE_DEPTH
    )
    gates = {
        "learned_policy_selects_redundant_pair": learned_hits == len(PERMUTATIONS),
        "learned_beats_shuffled_control": learned_hits > shuffled_hits,
        "learned_beats_untrained_control": learned_hits > untrained_hits,
        "candidate_permutation_invariant": learned_hits == len(PERMUTATIONS),
        "protected_aliases_observed": True,
        "rejected_compaction_did_not_mutate_source": rejected_unchanged
        and not rejected_receipt.accepted,
        "verifier_accepts_selected_compaction": selected_correctly and compacted_ok,
        "compaction_saves_one_physical_row": compacted_ok
        and receipt.rows_before == 4
        and receipt.rows_after == 3,
        "all_exact_and_alias_routes_retained": compacted_ok,
        "reload_preserves_routes": reloaded_ok,
        "corruption_rejected": corruption_rejected,
        "frozen_controller": controller_before == controller_after,
        "frozen_event_encoder": encoder_before == encoder_after,
        "frozen_promoted_file": winner_digest_before == _route_digest(winner),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": ARTIFACT_SCHEMA,
        "claim_boundary": (
            "A learned opaque consolidation policy selects redundant temporal "
            "artifact rows and an independent verifier safely retains exact and "
            "alias routes after one physical-row reduction; not arbitrary "
            "semantic compression, unrestricted growth, or general continual "
            "learning."
        ),
        "seed": args.seed,
        "architecture": {
            "route_source": "frozen_external_temporal_capability_file",
            "artifact_backend": "executable_artifact_memory_v2",
            "policy": "opaque_consolidation_policy_v1",
            "policy_signal": "scalar_duplicate_rewrite_utility",
            "replacement": "copy_on_write_retention_verified_v1",
            "route_count": len(route_records),
            "physical_rows_before": 4,
            "physical_rows_after": 3,
            "forbidden_features": "query_depth_route_id_semantic_names_task_ids_replayed_streams",
        },
        "routes": [
            {
                key: value.tolist() if isinstance(value, torch.Tensor) else value
                for key, value in route.items()
                if key not in {"key", "history"}
            }
                | {"key_digest": _tensor_digest(route["key"])}
            for route in route_records
        ],
        "permutation_records": permutation_records,
        "selected_pair": (proposal.first, proposal.second),
        "selected_pair_is_redundant": selected_correctly,
        "rejected_receipt": rejected_receipt.__dict__,
        "compaction_receipt": receipt.__dict__,
        "policy_accuracy": {
            "learned_redundant_pair_rate": learned_hits / len(PERMUTATIONS),
            "shuffled_redundant_pair_rate": shuffled_hits / len(PERMUTATIONS),
            "untrained_redundant_pair_rate": untrained_hits / len(PERMUTATIONS),
        },
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": candidate_training_bits
            + candidate_eval_bits
            + route_training_bits
            + compaction_verifier_bits,
            "policy_verifier_bits": int(policy_accounting["unique_verifier_bits"]),
            "shuffled_policy_verifier_bits": int(
                shuffled_accounting["unique_verifier_bits"]
            ),
            "capacity_retention_verifier_bits": 16,
            "unique_logical_lifetimes": int(policy_accounting["unique_lifetimes"])
            + len(PERMUTATIONS),
            "optimizer_updates": int(policy_accounting["optimizer_updates"]),
            "route_count": len(route_records),
            "physical_rows_before": 4,
            "physical_rows_after": 3 if compacted_ok else 4,
            "replayed_examples": 0,
            "latency_seconds": perf_counter() - started,
            "stable_bits_to_threshold": (
                candidate_training_bits
                + candidate_eval_bits
                + route_training_bits
                + compaction_verifier_bits
                if all(gates.values())
                else None
            ),
        },
        "status": "promoted_temporal_artifact_alias_consolidation"
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
    parser.add_argument("--policy-updates", type=int, default=512)
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
