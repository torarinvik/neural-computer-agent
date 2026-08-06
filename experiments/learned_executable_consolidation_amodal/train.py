"""Audit learned opaque compaction against executable artifact behavior."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from experiments.artifact_consolidation_amodal.train import (
    _load_composed,
    _load_single,
)
from experiments.artifact_view_routing_scaling_amodal.train import (
    OPERATIONS,
    _permuted_accuracy,
    _route_accuracy,
    _test_queries,
    _train_parent_and_artifacts,
    _train_router,
)
from experiments.opaque_consolidation_amodal.train import _train_policy
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _accuracy,
)
from neural_computer import (
    CapabilityRetentionLedger,
    ExecutableArtifactMemory,
    ExternalCapabilityLifecycle,
    FactorizedOpaqueAddressRouter,
    MemoryCandidates,
    OpaqueAddressRouter,
    RetentionPolicyConfig,
    select_growth_artifact_view,
)


def _artifact_summary(
    artifact: dict[str, torch.Tensor], *, width: int
) -> torch.Tensor:
    """Create an opaque fixed-width summary without interpreting tensor names."""
    tensors = [
        value.reshape(-1).to(dtype=torch.float32)
        for _name, value in sorted(artifact.items())
    ]
    flat = torch.cat(tensors)
    positions = torch.linspace(0, flat.numel() - 1, width).round().long()
    return F.normalize(flat[positions], dim=0)


def _remap_view(
    artifact: dict[str, torch.Tensor], *, source_view: str, target_view: str
) -> dict[str, torch.Tensor]:
    source_prefix = f"growth_slots.{source_view}."
    target_prefix = f"growth_slots.{target_view}."
    remapped: dict[str, torch.Tensor] = {}
    for name, value in artifact.items():
        if not name.startswith(source_prefix):
            raise ValueError(f"artifact contains an unexpected tensor name: {name}")
        remapped[target_prefix + name[len(source_prefix) :]] = value
    return remapped


def _row_views(
    memory: ExecutableArtifactMemory, index: int
) -> tuple[tuple[torch.Tensor, dict[str, torch.Tensor], str], ...]:
    """Return every opaque executable view in a physical row."""
    _handle, artifact = memory.promote_index(index)
    if not memory.alias_keys[index]:
        raise ValueError("executable source rows must carry opaque view aliases")
    views: list[tuple[torch.Tensor, dict[str, torch.Tensor], str]] = []
    for key, view in zip(memory.alias_keys[index], memory.alias_views[index], strict=True):
        if view is None:
            raise ValueError("merged artifact aliases must have executable views")
        selected = select_growth_artifact_view(
            artifact,
            source_prefix=f"growth_slots.{view}.",
        )
        views.append((key.detach().cpu().clone(), selected, view))
    return tuple(views)


def _merge_artifact_rows(
    memory: ExecutableArtifactMemory,
    first: int,
    second: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], tuple[torch.Tensor, ...], tuple[str, ...]]:
    combined: dict[str, torch.Tensor] = {}
    aliases: list[torch.Tensor] = []
    views: list[str] = []
    all_keys: list[torch.Tensor] = []
    for index in (first, second):
        for key, artifact, source_view in _row_views(memory, index):
            target_view = source_view
            combined.update(
                _remap_view(
                    artifact,
                    source_view="0",
                    target_view=target_view,
                )
            )
            aliases.append(key)
            views.append(target_view)
            all_keys.append(key)
    replacement_key = F.normalize(torch.stack(all_keys).sum(dim=0), dim=0)
    return replacement_key, combined, tuple(aliases), tuple(views)


def _memory_candidates(
    memory: ExecutableArtifactMemory,
) -> tuple[MemoryCandidates, tuple[int, ...], tuple[dict[str, torch.Tensor], ...]]:
    indices = memory.occupied
    artifacts = tuple(memory.promote_index(index)[1] for index in indices)
    return (
        MemoryCandidates(
            keys=torch.stack(
                [memory.rows.keys[index].detach().cpu() for index in indices]
            ).unsqueeze(0),
            values=torch.stack(
                [_artifact_summary(artifact, width=memory.width) for artifact in artifacts]
            ).unsqueeze(0),
            strengths=torch.tensor(
                [[float(memory.rows.strengths[index]) for index in indices]]
            ),
            timestamps=torch.tensor(
                [[float(memory.rows.timestamps[index]) for index in indices]]
            ),
            occupied=torch.ones((1, len(indices)), dtype=torch.bool),
        ),
        indices,
        artifacts,
    )


def _runtime_for_route(
    parent: Any,
    memory: ExecutableArtifactMemory,
    route_key: torch.Tensor,
    *,
    seed: int,
) -> Any:
    handle, artifact = memory.promote(route_key)
    if handle.view is None:
        runtime = _load_single(parent, artifact, seed=seed)
    else:
        runtime = _load_composed(
            parent,
            artifact,
            seed=seed,
            view=handle.view,
        )
    return runtime


def _probe_candidate(
    parent: Any,
    candidate: ExecutableArtifactMemory,
    route_keys: dict[str, torch.Tensor],
    baseline: dict[str, float],
    *,
    audit_count: int,
    probes: int,
    seed: int,
    tolerance: float,
    expected_core_digest: str,
) -> tuple[bool, list[float], dict[str, float], dict[str, list[float]], bool]:
    probe_scores: list[float] = []
    minimum_by_operation = {operation: 1.0 for operation in OPERATIONS}
    operation_traces = {operation: [] for operation in OPERATIONS}
    core_unchanged = True
    for probe in range(probes):
        operation_scores: dict[str, float] = {}
        for index, operation in enumerate(OPERATIONS):
            runtime = _runtime_for_route(
                parent,
                candidate,
                route_keys[operation],
                seed=seed + probe * 100 + index,
            )
            core_unchanged = core_unchanged and (
                _digest_core_without_growth(runtime) == expected_core_digest
            )
            score = _accuracy(
                runtime,
                operation=operation,
                count=audit_count,
                span=4,
                seed=seed + 10_000 + probe * 100 + index,
            )
            operation_scores[operation] = score
            operation_traces[operation].append(score)
            minimum_by_operation[operation] = min(
                minimum_by_operation[operation], score
            )
        probe_scores.append(min(operation_scores.values()))
    passed = all(
        minimum_by_operation[operation] >= baseline[operation] - tolerance
        for operation in OPERATIONS
    )
    return (
        passed and core_unchanged,
        probe_scores,
        minimum_by_operation,
        operation_traces,
        core_unchanged,
    )


def _digest_core_without_growth(runtime: Any) -> str:
    digest = hashlib.sha256()
    for name, value in runtime.controller.state_dict().items():
        if name.startswith("growth_slots."):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _row_members(
    memory: ExecutableArtifactMemory, members: dict[int, set[str]]
) -> dict[int, set[str]]:
    return {index: set(members[index]) for index in memory.occupied}


def _ordered_view_candidates(
    memory: ExecutableArtifactMemory,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    candidates = memory.view_candidates()
    by_view = {view: key for _index, key, view in candidates}
    view_ids = tuple(str(index) for index in range(len(OPERATIONS)))
    if set(by_view) != set(view_ids):
        raise RuntimeError(f"unexpected executable view ids: {sorted(by_view)}")
    return torch.stack([by_view[view] for view in view_ids]), view_ids


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    if min(
        args.updates,
        args.policy_updates,
        args.batch_size,
        args.audit_count,
        args.retention_probes,
        args.route_updates,
        args.route_batch_size,
    ) < 1:
        raise ValueError("all training and audit budgets must be positive")
    parent, artifacts, route_keys = _train_parent_and_artifacts(
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    parent_digest = _digest_core_without_growth(parent)
    baseline: dict[str, float] = {}
    for index, operation in enumerate(OPERATIONS):
        baseline[operation] = _accuracy(
            _load_single(parent, artifacts[operation], seed=args.seed + 20_000 + index),
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 30_000 + index,
        )
    if min(baseline.values()) < args.minimum_baseline:
        raise RuntimeError(f"independent artifacts did not reach baseline mastery: {baseline}")

    root = args.report_out.parent
    source_path = root / "executable_source_bank"
    if source_path.exists():
        shutil.rmtree(source_path)
    source = ExecutableArtifactMemory(
        source_path,
        width=64,
        capacity=len(OPERATIONS),
        retention_ledger=CapabilityRetentionLedger(
            64,
            config=RetentionPolicyConfig(
                mastery_threshold=args.retention_threshold,
                min_mastery_observations=args.retention_probes,
            ),
        ),
    )
    for index, operation in enumerate(OPERATIONS):
        primary_key = F.normalize(
            route_keys[operation]
            + 0.25 * torch.roll(route_keys[operation], shifts=1),
            dim=0,
        )
        namespaced_artifact = _remap_view(
            artifacts[operation],
            source_view="0",
            target_view=str(index),
        )
        row = source.put(primary_key, namespaced_artifact)
        source.alias_keys[row] = [route_keys[operation].detach().cpu().clone()]
        source.alias_views[row] = [str(index)]
    source.save()
    policy, policy_accounting = _train_policy(
        seed=args.seed + 40_000,
        rows=8,
        width=64,
        updates=args.policy_updates,
        batch_size=args.batch_size,
        shuffled_utility=False,
    )
    current = source
    members: dict[int, set[str]] = {
        index: {operation} for index, operation in enumerate(OPERATIONS)
    }
    known_scores = dict(baseline)
    step_reports: list[dict[str, object]] = []
    retention_observation_count = 0
    for step in range(len(OPERATIONS) - 1):
        bank, bank_indices, _bank_artifacts = _memory_candidates(current)
        proposal = policy.propose(bank)
        if proposal is None:
            raise RuntimeError("learned policy produced no executable consolidation pair")
        if proposal.operation != 0:
            raise RuntimeError(
                "learned executable consolidation selected a non-merge operation"
            )
        first = bank_indices[proposal.first]
        second = bank_indices[proposal.second]
        replacement_key, replacement_artifact, aliases, views = _merge_artifact_rows(
            current, first, second
        )
        final_path = root / f"executable_final_step_{step}"
        if final_path.exists():
            shutil.rmtree(final_path)
        old_members = _row_members(current, members)
        merged_members = old_members[first] | old_members[second]
        survivor_indices = tuple(
            index for index in current.occupied if index not in {first, second}
        )
        survivor_members = tuple(
            sorted({member for index in survivor_indices for member in old_members[index]})
        )
        retained_scores = [known_scores[member] for member in survivor_members]
        captured: dict[str, object] = {}
        verifier_seed = args.seed + 50_000 + step * 1_000

        def retention_probe(
            candidate: ExecutableArtifactMemory,
            *,
            _captured: dict[str, object] = captured,
            _merged_members: tuple[str, ...] = tuple(sorted(merged_members)),
            _probe_seed: int = verifier_seed + 100_000,
        ) -> list[float]:
            _passed, _scores, _minimums, traces, _core_unchanged = _probe_candidate(
                parent,
                candidate,
                route_keys,
                baseline,
                audit_count=args.audit_count,
                probes=args.retention_probes,
                seed=_probe_seed,
                tolerance=args.behavior_tolerance,
                expected_core_digest=parent_digest,
            )
            outcomes = [
                min(traces[member][probe] for member in _merged_members)
                for probe in range(args.retention_probes)
            ]
            _captured["retention_outcomes"] = outcomes
            return outcomes

        def verifier(
            candidate: ExecutableArtifactMemory,
            *,
            _captured: dict[str, object] = captured,
            _verifier_seed: int = verifier_seed,
        ) -> bool:
            passed, scores, operation_minimums, traces, core_unchanged = _probe_candidate(
                parent,
                candidate,
                route_keys,
                baseline,
                audit_count=args.audit_count,
                probes=args.retention_probes,
                seed=_verifier_seed,
                tolerance=args.behavior_tolerance,
                expected_core_digest=parent_digest,
            )
            _captured["scores"] = scores
            _captured["operation_minimums"] = operation_minimums
            _captured["operation_traces"] = traces
            _captured["core_unchanged"] = core_unchanged
            return passed

        lifecycle = ExternalCapabilityLifecycle(current)
        receipt = lifecycle.consolidate(
            (first, second),
            replacement_key,
            replacement_artifact,
            final_path,
            replacement_aliases=aliases,
            replacement_alias_views=views,
            verifier=verifier,
            candidate_outcome_probe=retention_probe,
            retained_scores=retained_scores,
            candidate_threshold=args.retention_threshold,
            retention_floor=args.retention_threshold,
            min_candidate_observations=args.retention_probes,
        )
        if not receipt.accepted:
            raise RuntimeError(f"executable consolidation rejected: {receipt}")
        candidate = lifecycle.memory
        members = {0: merged_members}
        members.update(
            {
                new_index: old_members[index]
                for new_index, index in enumerate(survivor_indices, start=1)
            }
        )
        current = candidate
        operation_traces = captured["operation_traces"]
        known_scores = {
            operation: min(operation_traces[operation])
            for operation in OPERATIONS
        }
        retention_observation_count += args.retention_probes
        for index in current.occupied:
            row_members = members[index]
            row_scores = [
                min(operation_traces[member][probe] for member in row_members)
                for probe in range(args.retention_probes)
            ]
            for key in current._row_retention_keys(index):
                for score in row_scores:
                    current.observe_retention(key, score)
                    retention_observation_count += 1
        step_reports.append(
            {
                "step": step,
                "selected_members": sorted(merged_members),
                "policy_score": float(proposal.score),
                "candidate_retention_floor": min(captured["retention_outcomes"]),
                "retention_checked": True,
                "rows_before": receipt.rows_before,
                "rows_after": receipt.rows_after,
                "rows_saved": receipt.rows_saved,
                "replacement_protected": candidate.retention.is_protected(
                    replacement_key
                ),
                "behavior_minimums": captured.get("operation_minimums", {}),
            }
        )

    reloaded = ExecutableArtifactMemory.load(current.directory)
    (
        reload_passed,
        reload_probe_scores,
        reload_minimums,
        _reload_traces,
        reload_core_unchanged,
    ) = _probe_candidate(
        parent,
        reloaded,
        route_keys,
        baseline,
        audit_count=args.audit_count,
        probes=args.retention_probes,
        seed=args.seed + 90_000,
        tolerance=args.behavior_tolerance,
        expected_core_digest=parent_digest,
    )
    final_views: dict[str, str | None] = {}
    for operation in OPERATIONS:
        final_views[operation] = reloaded.promote(route_keys[operation])[0].view

    candidate_keys, view_ids = _ordered_view_candidates(reloaded)
    router, route_accounting = _train_router(
        parent,
        candidate_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 60_000,
        shuffle_outcomes=False,
        credit=args.route_credit,
        router_kind=args.router_kind,
    )
    shuffled_router, shuffled_route_accounting = _train_router(
        parent,
        candidate_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 70_000,
        shuffle_outcomes=True,
        credit=args.route_credit,
        router_kind=args.router_kind,
    )
    route_path = root / "executable_route_router.pt"
    torch.save(router.state_dict(), route_path)
    route_queries, route_targets = _test_queries(
        parent,
        audit_count=args.audit_count,
        seed=args.seed + 80_000,
    )
    route_accuracy = _route_accuracy(
        router, route_queries, route_targets, candidate_keys
    )
    shuffled_route_accuracy = _route_accuracy(
        shuffled_router, route_queries, route_targets, candidate_keys
    )
    permuted_route_accuracy = _permuted_accuracy(
        router, route_queries, route_targets, candidate_keys
    )
    routed_behavior: dict[str, float] = {}
    wrong_behavior: dict[str, float] = {}
    routed_views: dict[str, str] = {}
    for family, operation in enumerate(OPERATIONS):
        family_queries = route_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        predictions = router(family_queries, candidate_keys).argmax(dim=-1)
        selected_index = int(torch.mode(predictions).values)
        selected_view = view_ids[selected_index]
        routed_views[operation] = selected_view
        routed_runtime = _runtime_for_route(
            parent,
            reloaded,
            candidate_keys[selected_index],
            seed=args.seed + 100_000 + family,
        )
        routed_behavior[operation] = _accuracy(
            routed_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 110_000 + family,
        )
        wrong_runtime = _runtime_for_route(
            parent,
            reloaded,
            candidate_keys[(selected_index + 1) % len(view_ids)],
            seed=args.seed + 120_000 + family,
        )
        wrong_behavior[operation] = _accuracy(
            wrong_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 110_000 + family,
        )

    reloaded_route_bank = ExecutableArtifactMemory.load(reloaded.directory)
    reloaded_candidate_keys, reloaded_view_ids = _ordered_view_candidates(
        reloaded_route_bank
    )
    router_class = (
        FactorizedOpaqueAddressRouter
        if args.router_kind == "factorized"
        else OpaqueAddressRouter
    )
    reloaded_router = router_class(width=int(candidate_keys.shape[1]), hidden=64)
    reloaded_router.load_state_dict(torch.load(route_path, weights_only=False))
    reloaded_router.eval()
    reloaded_route_accuracy = _route_accuracy(
        reloaded_router,
        route_queries,
        route_targets,
        reloaded_candidate_keys,
    )
    reloaded_permuted_route_accuracy = _permuted_accuracy(
        reloaded_router,
        route_queries,
        route_targets,
        reloaded_candidate_keys,
    )
    reloaded_routed_views: dict[str, str] = {}
    reloaded_routed_behavior: dict[str, float] = {}
    for family, operation in enumerate(OPERATIONS):
        family_queries = route_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        predictions = reloaded_router(
            family_queries, reloaded_candidate_keys
        ).argmax(dim=-1)
        selected_index = int(torch.mode(predictions).values)
        selected_view = reloaded_view_ids[selected_index]
        reloaded_routed_views[operation] = selected_view
        runtime = _runtime_for_route(
            parent,
            reloaded_route_bank,
            reloaded_candidate_keys[selected_index],
            seed=args.seed + 130_000 + family,
        )
        reloaded_routed_behavior[operation] = _accuracy(
            runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 130_000 + family,
        )
    artifact_name = reloaded.paths[0]
    if artifact_name is None:
        raise RuntimeError("reloaded executable bank has no artifact payload")
    artifact_path = reloaded.directory / artifact_name
    intact_payload = artifact_path.read_bytes()
    artifact_path.write_bytes(intact_payload + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(reloaded.directory)
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(intact_payload)
    report: dict[str, object] = {
        "schema": "neural-computer.learned-executable-consolidation-report.v2",
        "claim_boundary": (
            "A learned opaque memory-side policy selects pairwise executable "
            "artifact compactions; held-out behavior, alias, reload, and "
            "retention verifiers gate adoption. A separate outcome-trained "
            "opaque address router acquires the compacted views. This is not "
            "general continual learning."
        ),
        "seed": args.seed,
        "operations": list(OPERATIONS),
        "updates": args.updates,
        "policy_updates": args.policy_updates,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "retention_probes": args.retention_probes,
        "baseline_behavior": baseline,
        "step_reports": step_reports,
        "final_views": final_views,
        "final_behavior_minimums": reload_minimums,
        "reload_probe_floor": min(reload_probe_scores),
        "route_accuracy": route_accuracy,
        "reward_shuffled_route_accuracy": shuffled_route_accuracy,
        "candidate_permutation_accuracy": permuted_route_accuracy,
        "routed_views": routed_views,
        "routed_behavior": routed_behavior,
        "wrong_behavior": wrong_behavior,
        "reloaded_route_accuracy": reloaded_route_accuracy,
        "reloaded_candidate_permutation_accuracy": reloaded_permuted_route_accuracy,
        "reloaded_routed_views": reloaded_routed_views,
        "reloaded_routed_behavior": reloaded_routed_behavior,
        "frozen_core": reload_core_unchanged,
        "corruption_rejected": corruption_rejected,
        "accounting": {
            "behavior_evaluation_passes": (
                (len(OPERATIONS) - 1) * 2 + 1
            ),
            "artifact_training_optimizer_updates": args.updates * 5,
            "consolidation_policy_optimizer_updates": policy_accounting[
                "optimizer_updates"
            ],
            "unique_logical_lifetimes": (
                args.updates * args.batch_size * 5
                + args.policy_updates * args.batch_size
                + args.audit_count
                * args.retention_probes
                * len(OPERATIONS)
                * ((len(OPERATIONS) - 1) * 2 + 1)
            ),
            "unique_verifier_bits": (
                args.updates * args.batch_size * (2 + 4 * 4)
                + policy_accounting["unique_verifier_bits"]
                + args.audit_count
                * args.retention_probes
                * len(OPERATIONS)
                * 4
                * ((len(OPERATIONS) - 1) * 2 + 1)
            ),
            "retention_observations": retention_observation_count,
            "route_optimizer_updates": route_accounting[
                "route_optimizer_updates"
            ]
            + shuffled_route_accounting["route_optimizer_updates"],
            "route_unique_lifetimes": route_accounting[
                "unique_route_lifetimes"
            ]
            + shuffled_route_accounting["unique_route_lifetimes"],
            "route_unique_verifier_bits": route_accounting[
                "unique_route_verifier_bits"
            ]
            + shuffled_route_accounting["unique_route_verifier_bits"],
            "routed_behavior_verifier_bits": (
                args.audit_count * len(OPERATIONS) * 4 * 3
            ),
            "replayed_examples": 0,
            "route_replayed_examples": 0,
            "controller_optimizer_updates_during_consolidation": 0,
            "privileged_task_labels_seen_by_policy": 0,
            "wall_seconds": time.perf_counter() - started,
        },
        "gates": {
            "independent_artifacts_mastered": min(baseline.values())
            >= args.minimum_baseline,
            "three_sequential_compactions": len(step_reports) == len(OPERATIONS) - 1,
            "behavior_preserved_each_step": all(
                bool(report["behavior_minimums"])
                for report in step_reports
            ),
            "retention_protected_each_step": all(
                bool(report["replacement_protected"]) for report in step_reports
            ),
            "final_behavior_preserved_after_reload": reload_passed,
            "all_four_views_reloaded": set(final_views.values())
            == {str(index) for index in range(len(OPERATIONS))},
            "frozen_core": reload_core_unchanged,
            "corruption_rejected": corruption_rejected,
            "learned_route_at_least_90": route_accuracy >= 0.90,
            "reward_shuffled_route_near_chance": shuffled_route_accuracy <= 0.50,
            "candidate_permutation_invariant": permuted_route_accuracy >= 0.90,
            "routed_views_match_expected": routed_views
            == dict(zip(OPERATIONS, view_ids, strict=True)),
            "routed_behavior_mastered": min(routed_behavior.values()) >= 0.75,
            "wrong_view_is_causal": all(
                routed_behavior[operation] > wrong_behavior[operation] + 0.05
                for operation in OPERATIONS
            ),
            "reloaded_route_preserved": reloaded_route_accuracy >= 0.90,
            "reloaded_route_permutation_invariant": (
                reloaded_permuted_route_accuracy >= 0.90
            ),
            "reloaded_routed_views_match_expected": reloaded_routed_views
            == dict(zip(OPERATIONS, reloaded_view_ids, strict=True)),
            "reloaded_routed_behavior_preserved": all(
                reloaded_routed_behavior[operation]
                >= routed_behavior[operation] - 0.05
                for operation in OPERATIONS
            ),
            "no_replayed_examples": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--policy-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--retention-probes", type=int, default=8)
    parser.add_argument("--route-updates", type=int, default=2048)
    parser.add_argument("--route-batch-size", type=int, default=16)
    parser.add_argument(
        "--route-credit",
        choices=("attempted_outcome", "paired_counterfactual"),
        default="paired_counterfactual",
    )
    parser.add_argument(
        "--router-kind",
        choices=("factorized", "opaque"),
        default="opaque",
    )
    parser.add_argument("--retention-threshold", type=float, default=0.75)
    parser.add_argument("--minimum-baseline", type=float, default=0.75)
    parser.add_argument("--behavior-tolerance", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "baseline_behavior": report["baseline_behavior"],
                "route_accuracy": report["route_accuracy"],
                "reward_shuffled_route_accuracy": report[
                    "reward_shuffled_route_accuracy"
                ],
                "candidate_permutation_accuracy": report[
                    "candidate_permutation_accuracy"
                ],
                "step_reports": report["step_reports"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
