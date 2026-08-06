"""Audit three sequential external capability additions without replay."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections.abc import Callable, Sequence
from itertools import combinations
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.artifact_consolidation_amodal.train import (
    _load_composed,
    _load_single,
)
from experiments.artifact_view_routing_scaling_amodal.train import (
    OPERATIONS as OLD_OPERATIONS,
)
from experiments.artifact_view_routing_scaling_amodal.train import (
    _compact_bank,
    _fresh_queries,
    _test_queries,
    _train_parent_and_artifacts,
    _train_router,
)
from experiments.multistep_view_growth_amodal.train import _next_view_predictions
from experiments.online_view_growth_amodal.train import (
    _extend_bank,
    _train_new_artifact,
    _train_route_extension,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _accuracy,
    _digest_core,
)
from neural_computer import (
    CapabilityRetentionProbe,
    ExecutableArtifactMemory,
    OpaqueAddressRouter,
    OpaqueViewRouteExtension,
    RetentionPolicyConfig,
    compress_growth_artifact,
    stable_prefix_minimum,
)

NEW_OPERATIONS = ("rotate", "complement_rotate", "adjacent_xor")
ALL_OPERATIONS = (*OLD_OPERATIONS, *NEW_OPERATIONS)
OLD_VIEWS = tuple(str(index) for index in range(len(OLD_OPERATIONS)))
ALL_VIEWS = tuple(str(index) for index in range(len(ALL_OPERATIONS)))


def _load_view(
    parent,
    bank,
    view: str,
    *,
    seed: int,
    allow_dtype_cast: bool = False,
    decompress_artifact: bool = False,
):
    handle, artifact = bank.promote_view(0, view)
    if handle.view != view:
        raise RuntimeError("memory returned the wrong opaque view")
    return _load_composed(
        parent,
        artifact,
        seed=seed,
        view=view,
        allow_dtype_cast=allow_dtype_cast,
        decompress_artifact=decompress_artifact,
    )


def _global_extension_predictions(
    base_router: OpaqueAddressRouter,
    extension: OpaqueViewRouteExtension,
    queries: torch.Tensor,
    old_keys: torch.Tensor,
    global_view: int,
) -> torch.Tensor:
    local = _next_view_predictions(base_router, extension, queries, old_keys)
    return torch.where(
        local == len(OLD_VIEWS),
        torch.full_like(local, global_view),
        local,
    )


def _permuted_accuracy(
    base_router: OpaqueAddressRouter,
    extensions: list[OpaqueViewRouteExtension],
    old_keys: torch.Tensor,
    old_queries: torch.Tensor,
    old_targets: torch.Tensor,
    new_queries: list[torch.Tensor],
) -> float:
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted_keys = old_keys[permutation]
    old_predictions = base_router(old_queries, permuted_keys).argmax(dim=-1)
    old_predictions = permutation[old_predictions]
    predictions = [old_predictions]
    targets = [old_targets]
    for index, (extension, queries) in enumerate(zip(extensions, new_queries)):
        predictions.append(
            _global_extension_predictions(
                base_router,
                extension,
                queries,
                permuted_keys,
                len(OLD_VIEWS) + index,
            )
        )
        targets.append(
            torch.full(
                (queries.shape[0],),
                len(OLD_VIEWS) + index,
                dtype=torch.long,
            )
        )
    return float(
        (torch.cat(predictions) == torch.cat(targets)).float().mean()
    )


def _mode_view(predictions: torch.Tensor) -> str:
    return str(int(torch.mode(predictions).values))


def _payload_bytes(artifact: dict[str, torch.Tensor]) -> int:
    return sum(value.numel() * value.element_size() for value in artifact.values())


def _retention_probe_floor(
    probes: Sequence[CapabilityRetentionProbe],
    *,
    min_observations: int,
) -> float:
    return min(
        stable_prefix_minimum(
            probe.outcomes,
            min_observations=min_observations,
        )
        for probe in probes
    )


def _retention_probe_report(
    probes: Sequence[CapabilityRetentionProbe],
) -> list[list[float]]:
    return [
        torch.as_tensor(probe.outcomes, dtype=torch.float64)
        .reshape(-1)
        .tolist()
        for probe in probes
    ]


def _direct_capability_outcomes(
    parent,
    artifact: dict[str, torch.Tensor],
    *,
    views: tuple[str, ...],
    operations: tuple[str, ...],
    audit_count: int,
    retention_probes: int,
    seed: int,
) -> list[list[float]]:
    """Measure a full-precision artifact on the exact compression probe set."""

    outcomes: list[list[float]] = [[] for _ in operations]
    for probe in range(retention_probes):
        for index, operation in enumerate(operations):
            runtime = _load_composed(
                parent,
                artifact,
                seed=seed + 30_000 + probe * 100 + index,
                view=views[index],
            )
            outcomes[index].append(
                _accuracy(
                    runtime,
                    operation=operation,
                    count=audit_count,
                    span=4,
                    seed=seed + 40_000 + probe * 100 + index,
                )
            )
    return outcomes


def _failed_capability_index(reason: str) -> int | None:
    """Extract the opaque capability index from a structured gate failure."""

    match = re.search(r"capability (\d+):", reason)
    return int(match.group(1)) if match is not None else None


def _adaptive_mixed_precision_artifact(
    parent,
    artifact: dict[str, torch.Tensor],
    *,
    dtype: torch.dtype | str,
    failed_view: str,
    audit_count: int,
    retention_probes: int,
    retention_floor: float,
    seed: int,
    max_payload_ratio: float,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...], list[dict[str, object]]]:
    """Choose a loss-aware mixed-precision repair from fresh verifier probes.

    The initial codec remains uniform and is always attempted first.  If it
    fails, this helper searches only the tensors under the failed opaque view,
    greedily preserving the most useful tensor(s) losslessly while keeping the
    candidate within the caller's storage budget.  The final memory
    consolidation still reruns its full per-capability retention and behavior
    gates; these probes are only a bounded search heuristic, never an
    admission decision.
    """

    prefix = f"growth_slots.{failed_view}."
    names = tuple(name for name in artifact if name.startswith(prefix))
    if not names:
        raise ValueError(f"artifact has no tensors for failed view {failed_view!r}")
    source_payload = _payload_bytes(artifact)
    selected: list[str] = []
    remaining = list(names)
    best_artifact = compress_growth_artifact(artifact, dtype=dtype)
    diagnostics: list[dict[str, object]] = []

    while remaining:
        round_candidates: list[
            tuple[float, int, str, dict[str, torch.Tensor], list[float], float]
        ] = []
        for order, name in enumerate(remaining):
            preserved = (*selected, name)
            candidate = compress_growth_artifact(
                artifact,
                dtype=dtype,
                preserve_names=preserved,
            )
            payload_ratio = _payload_bytes(candidate) / source_payload
            if payload_ratio > max_payload_ratio + 1e-9:
                diagnostics.append(
                    {
                        "preserved_names": list(preserved),
                        "payload_byte_ratio": payload_ratio,
                        "candidate_outcomes": [],
                        "candidate_outcome_floor": None,
                        "within_payload_budget": False,
                        "dtype_override": None,
                        "dtype_override_names": [],
                    }
                )
                continue
            outcomes: list[float] = []
            for probe in range(retention_probes):
                runtime = _load_composed(
                    parent,
                    candidate,
                    seed=seed + 30_000 + probe * 100 + int(failed_view),
                    view=failed_view,
                    allow_dtype_cast=True,
                )
                outcomes.append(
                    _accuracy(
                        runtime,
                        operation=ALL_OPERATIONS[int(failed_view)],
                        count=audit_count,
                        span=4,
                        seed=seed + 40_000 + probe * 100 + int(failed_view),
                    )
                )
            floor = stable_prefix_minimum(
                outcomes,
                min_observations=retention_probes,
            )
            diagnostics.append(
                {
                    "preserved_names": list(preserved),
                    "payload_byte_ratio": payload_ratio,
                    "candidate_outcomes": outcomes,
                    "candidate_outcome_floor": floor,
                    "within_payload_budget": True,
                    "dtype_override": None,
                    "dtype_override_names": [],
                }
            )
            round_candidates.append(
                (floor, -order, name, candidate, outcomes, payload_ratio)
            )
        if not round_candidates:
            break
        floor, _order, name, candidate, _outcomes, _payload_ratio = max(
            round_candidates
        )
        selected.append(name)
        remaining.remove(name)
        best_artifact = candidate
        if floor >= retention_floor:
            break

    # A large critical tensor may barely miss the payload budget when it is
    # the only lossless entry.  Pair it with one opaque, independently
    # replaceable view encoded more compactly.  The candidate behavior gate
    # below decides whether that cross-view trade is actually safe.
    oversized_names = tuple(
        name
        for name in names
        if _payload_bytes(
            compress_growth_artifact(artifact, preserve_names=(name,))
        )
        / source_payload
        > max_payload_ratio
    )
    preserve_sets = tuple(
        preserved
        for size in (*range(1, min(2, len(names)) + 1), len(names))
        for preserved in combinations(names, size)
        if any(name in oversized_names for name in preserved)
    )
    for preserved in preserve_sets:
        for compressed_view in ALL_VIEWS:
            if compressed_view == failed_view:
                continue
            override_names = tuple(
                name
                for name in artifact
                if name.startswith(f"growth_slots.{compressed_view}.")
            )
            if not override_names:
                continue
            for override_dtype in (torch.int8, "int4"):
                candidate = compress_growth_artifact(
                    artifact,
                    preserve_names=preserved,
                    dtype_overrides={
                        name: override_dtype for name in override_names
                    },
                )
                payload_ratio = _payload_bytes(candidate) / source_payload
                if payload_ratio > max_payload_ratio + 1e-9:
                    continue
                outcomes: list[float] = []
                for probe in range(retention_probes):
                    runtime = _load_composed(
                        parent,
                        candidate,
                        seed=seed + 30_000 + probe * 100 + int(failed_view),
                        view=failed_view,
                        allow_dtype_cast=True,
                    )
                    outcomes.append(
                        _accuracy(
                            runtime,
                            operation=ALL_OPERATIONS[int(failed_view)],
                            count=audit_count,
                            span=4,
                            seed=seed
                            + 40_000
                            + probe * 100
                            + int(failed_view),
                        )
                    )
                floor = stable_prefix_minimum(
                    outcomes,
                    min_observations=retention_probes,
                )
                diagnostics.append(
                    {
                        "preserved_names": list(preserved),
                        "payload_byte_ratio": payload_ratio,
                        "candidate_outcomes": outcomes,
                        "candidate_outcome_floor": floor,
                        "within_payload_budget": True,
                        "dtype_override": (
                            "int8" if override_dtype == torch.int8 else "int4"
                        ),
                        "dtype_override_names": list(override_names),
                    }
                )
                if floor >= retention_floor:
                    best_artifact = candidate

    return best_artifact, tuple(selected), diagnostics


def _make_retention_callbacks(
    parent,
    artifact: dict[str, torch.Tensor],
    *,
    operation: str,
    new_view: str,
    candidate_views: tuple[str, ...],
    prior_operations: tuple[str, ...],
    previous_new_operations: tuple[str, ...],
    prior_views: tuple[str, ...],
    index: int,
    old_retention_scores: dict[str, float],
    retention_steps: list[dict[str, object]],
    audit_count: int,
    retention_probes: int,
    behavior_tolerance: float,
    seed: int,
) -> tuple[
    Callable[[ExecutableArtifactMemory], Sequence[float]],
    Callable[[ExecutableArtifactMemory], bool],
    list[float],
    dict[str, float],
    dict[str, float],
]:
    candidate_outcomes: list[float] = []
    behavior_baselines: dict[str, float] = {}
    behavior_scores: dict[str, float] = {}

    def candidate_outcome_probe(
        candidate: ExecutableArtifactMemory,
    ) -> Sequence[float]:
        observations: list[float] = []
        for probe in range(retention_probes):
            runtime = _load_view(
                parent,
                candidate,
                new_view,
                seed=seed + 40_000 + index * 1_000 + probe,
            )
            observations.append(
                _accuracy(
                    runtime,
                    operation=operation,
                    count=audit_count,
                    span=4,
                    seed=seed + 50_000 + index * 1_000 + probe,
                )
            )
        candidate_outcomes[:] = observations
        return observations

    def extension_verifier(candidate: ExecutableArtifactMemory) -> bool:
        if [
            candidate.promote_view(0, view)[0].view for view in candidate_views
        ] != list(candidate_views):
            return False
        baselines = {
            **old_retention_scores,
            **{
                prior_operation: retention_steps[prior_index][
                    "candidate_outcome_floor"
                ]
                for prior_index, prior_operation in enumerate(
                    previous_new_operations
                )
            },
            operation: _accuracy(
                _load_single(parent, artifact, seed=seed + 60_000 + index),
                operation=operation,
                count=audit_count,
                span=4,
                seed=seed + 61_000 + index,
            ),
        }
        behavior_baselines.update(baselines)
        for family, prior_operation in enumerate(prior_operations):
            runtime = _load_view(
                parent,
                candidate,
                prior_views[family],
                seed=seed + 62_000 + index * 100 + family,
            )
            behavior_scores[prior_operation] = _accuracy(
                runtime,
                operation=prior_operation,
                count=audit_count,
                span=4,
                seed=seed + 63_000 + index * 100 + family,
            )
        return all(
            behavior_scores[prior_operation]
            >= baselines[prior_operation] - behavior_tolerance
            for prior_operation in prior_operations
        )

    return (
        candidate_outcome_probe,
        extension_verifier,
        candidate_outcomes,
        behavior_baselines,
        behavior_scores,
    )


def _make_compression_callbacks(
    parent,
    *,
    views: tuple[str, ...],
    capability_keys: tuple[torch.Tensor, ...],
    selected_behavior: dict[str, float],
    audit_count: int,
    retention_probes: int,
    behavior_tolerance: float,
    seed: int,
    allow_dtype_cast: bool,
    decompress_artifact: bool,
) -> tuple[
    Callable[[ExecutableArtifactMemory], bool],
    Callable[[ExecutableArtifactMemory], Sequence[CapabilityRetentionProbe]],
    list[CapabilityRetentionProbe],
    dict[str, float],
]:
    candidate_outcomes: list[CapabilityRetentionProbe] = []
    behavior_scores: dict[str, float] = {}

    def verifier(candidate: ExecutableArtifactMemory) -> bool:
        if [view for _, _, view in candidate.view_candidates()] != list(views):
            return False
        for index, operation in enumerate(ALL_OPERATIONS):
            runtime = _load_view(
                parent,
                candidate,
                views[index],
                seed=seed + 10_000 + index,
                allow_dtype_cast=allow_dtype_cast,
                decompress_artifact=decompress_artifact,
            )
            behavior_scores[operation] = _accuracy(
                runtime,
                operation=operation,
                count=audit_count,
                span=4,
                seed=seed + 20_000 + index,
            )
        return all(
            behavior_scores[operation]
            >= selected_behavior[operation] - behavior_tolerance
            for operation in ALL_OPERATIONS
        )

    def candidate_outcome_probe(
        candidate: ExecutableArtifactMemory,
    ) -> Sequence[CapabilityRetentionProbe]:
        observations = [[] for _ in ALL_OPERATIONS]
        for probe in range(retention_probes):
            for index, operation in enumerate(ALL_OPERATIONS):
                runtime = _load_view(
                    parent,
                    candidate,
                    views[index],
                    seed=seed + 30_000 + probe * 100 + index,
                    allow_dtype_cast=allow_dtype_cast,
                    decompress_artifact=decompress_artifact,
                )
                observations[index].append(
                    _accuracy(
                        runtime,
                        operation=operation,
                        count=audit_count,
                        span=4,
                        seed=seed + 40_000 + probe * 100 + index,
                    )
                )
        candidate_outcomes[:] = [
            CapabilityRetentionProbe(key, values)
            for key, values in zip(capability_keys, observations)
        ]
        return candidate_outcomes

    return verifier, candidate_outcome_probe, candidate_outcomes, behavior_scores


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.updates,
        args.extension_artifact_updates,
        args.route_updates,
        args.extension_updates,
        args.batch_size,
        args.route_batch_size,
        args.audit_count,
        args.retention_probes,
    ) < 1:
        raise ValueError("updates and batch sizes must be positive")
    if any(
        value % 2
        for value in (args.batch_size, args.route_batch_size, args.audit_count)
    ):
        raise ValueError("all batch sizes and audit counts must be even")

    parent, old_artifacts, old_route_keys = _train_parent_and_artifacts(
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    parent_digest = _digest_core(parent, ("growth_slots.0.", "growth_slots.1."))
    root = args.report_out.parent
    old_bank = _compact_bank(
        old_artifacts,
        old_route_keys,
        directory=root / "old_view_bank",
    )
    old_bank.retention.config = RetentionPolicyConfig(
        mastery_threshold=args.retention_threshold,
        min_mastery_observations=args.retention_probes,
    )
    old_candidates = old_bank.view_candidates()
    old_retention_scores: dict[str, float] = {}
    old_retention_observations: dict[str, list[float]] = {}
    for family, operation in enumerate(OLD_OPERATIONS):
        key = old_candidates[family][1]
        observations: list[float] = []
        for probe in range(args.retention_probes):
            runtime = _load_view(
                parent,
                old_bank,
                OLD_VIEWS[family],
                seed=args.seed + 20_000 + family * 100 + probe,
            )
            outcome = _accuracy(
                runtime,
                operation=operation,
                count=args.audit_count,
                span=4,
                seed=args.seed + 30_000 + family * 100 + probe,
            )
            observations.append(outcome)
            old_bank.observe_retention(key, outcome)
        old_retention_observations[operation] = observations
        old_retention_scores[operation] = stable_prefix_minimum(
            observations,
            min_observations=args.retention_probes,
        )
    old_bank.save()
    old_protected_before_growth = [
        old_bank.retention.is_protected(key) for _, key, _ in old_candidates
    ]

    key_generator = torch.Generator(device="cpu").manual_seed(args.seed + 90_000)
    bank = old_bank
    expected_views = OLD_VIEWS
    new_artifacts: list[dict[str, torch.Tensor]] = []
    retention_steps: list[dict[str, object]] = []
    for index, operation in enumerate(NEW_OPERATIONS):
        artifact = _train_new_artifact(
            parent,
            seed=args.seed + (index + 1) * 1_000,
            updates=args.extension_artifact_updates,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            operation=operation,
        )
        new_artifacts.append(artifact)
        new_view = str(len(OLD_VIEWS) + index)
        key = F.normalize(torch.randn(64, generator=key_generator), dim=0)
        candidate_views = (*expected_views, new_view)
        prior_operations = (*OLD_OPERATIONS, *NEW_OPERATIONS[:index])
        candidate_operations = (*prior_operations, operation)
        retained_scores = [
            *(old_retention_scores[old_operation] for old_operation in OLD_OPERATIONS),
            *(step["candidate_outcome_floor"] for step in retention_steps),
        ]
        (
            candidate_outcome_probe,
            extension_verifier,
            candidate_outcomes,
            behavior_baselines,
            behavior_scores,
        ) = _make_retention_callbacks(
            parent,
            artifact,
            operation=operation,
            new_view=new_view,
            candidate_views=candidate_views,
            prior_operations=candidate_operations,
            previous_new_operations=NEW_OPERATIONS[:index],
            prior_views=candidate_views,
            index=index,
            old_retention_scores=old_retention_scores,
            retention_steps=retention_steps,
            audit_count=args.audit_count,
            retention_probes=args.retention_probes,
            behavior_tolerance=args.behavior_tolerance,
            seed=args.seed,
        )

        bank = _extend_bank(
            bank,
            artifact,
            key,
            directory=root / f"view_bank_{new_view}",
            existing_views=expected_views,
            new_view=new_view,
            target_slot=len(OLD_VIEWS) + index,
            verifier=extension_verifier,
            candidate_outcome_probe=candidate_outcome_probe,
            retained_scores=retained_scores,
            retention_threshold=args.retention_threshold,
            retention_probes=args.retention_probes,
        )
        bank.retention.config = RetentionPolicyConfig(
            mastery_threshold=args.retention_threshold,
            min_mastery_observations=args.retention_probes,
        )
        retention_steps.append(
            {
                "operation": operation,
                "candidate_outcomes": candidate_outcomes,
                "candidate_outcome_minimum": min(candidate_outcomes),
                "candidate_outcome_floor": stable_prefix_minimum(
                    candidate_outcomes,
                    min_observations=args.retention_probes,
                ),
                "behavior_baselines": behavior_baselines,
                "behavior_scores": behavior_scores,
                "replacement_protected": bank._row_is_protected(0),
            }
        )
        expected_views = candidate_views

    candidates = bank.view_candidates()
    candidate_keys = torch.stack([key for _, key, _ in candidates])
    if [view for _, _, view in candidates] != list(ALL_VIEWS):
        raise RuntimeError("three-step extension produced the wrong view order")
    old_keys = candidate_keys[: len(OLD_VIEWS)]

    base_router, base_accounting = _train_router(
        parent,
        old_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 60_000,
        shuffle_outcomes=False,
        credit="paired_counterfactual",
        router_kind="opaque",
    )
    base_router.eval()
    extensions: list[OpaqueViewRouteExtension] = []
    shuffled_extensions: list[OpaqueViewRouteExtension] = []
    extension_accounting: list[dict[str, int | float]] = []
    shuffled_accounting: list[dict[str, int | float]] = []
    extension_snapshots: list[dict[str, torch.Tensor]] = []
    for index, operation in enumerate(NEW_OPERATIONS):
        extension, accounting = _train_route_extension(
            parent,
            base_router,
            old_keys,
            updates=args.extension_updates,
            batch_size=args.route_batch_size,
            seed=args.seed + 70_000 + index * 20_000,
            shuffle_outcomes=False,
            operation=operation,
        )
        for parameter in extension.parameters():
            parameter.requires_grad_(False)
        extensions.append(extension)
        extension_accounting.append(accounting)
        extension_snapshots.append(
            {
                name: value.detach().cpu().clone()
                for name, value in extension.state_dict().items()
            }
        )
        shuffled, shuffled_stats = _train_route_extension(
            parent,
            base_router,
            old_keys,
            updates=args.extension_updates,
            batch_size=args.route_batch_size,
            seed=args.seed + 80_000 + index * 20_000,
            shuffle_outcomes=True,
            operation=operation,
        )
        shuffled_extensions.append(shuffled)
        shuffled_accounting.append(shuffled_stats)

    extensions_frozen = all(
        all(
            torch.equal(snapshot[name], value.detach().cpu())
            for name, value in extension.state_dict().items()
        )
        for extension, snapshot in zip(extensions, extension_snapshots)
    )

    old_queries, old_targets = _test_queries(
        parent,
        audit_count=args.audit_count,
        seed=args.seed + 110_000,
    )
    new_queries = [
        _fresh_queries(
            parent,
            operation=operation,
            count=args.audit_count * len(OLD_VIEWS),
            seed=args.seed + 120_000 + index * 10_000,
        )
        for index, operation in enumerate(NEW_OPERATIONS)
    ]
    old_predictions = base_router(old_queries, old_keys).argmax(dim=-1)
    new_predictions = [
        _global_extension_predictions(
            base_router,
            extension,
            queries,
            old_keys,
            len(OLD_VIEWS) + index,
        )
        for index, (extension, queries) in enumerate(zip(extensions, new_queries))
    ]
    target_blocks = [old_targets] + [
        torch.full(
            (queries.shape[0],),
            len(OLD_VIEWS) + index,
            dtype=torch.long,
        )
        for index, queries in enumerate(new_queries)
    ]
    combined_predictions = torch.cat([old_predictions, *new_predictions])
    combined_targets = torch.cat(target_blocks)
    base_old_route = float((old_predictions == old_targets).float().mean())
    new_route = [
        float((prediction == target).float().mean())
        for prediction, target in zip(new_predictions, target_blocks[1:])
    ]
    combined_route = float(
        (combined_predictions == combined_targets).float().mean()
    )
    permuted_route = _permuted_accuracy(
        base_router,
        extensions,
        old_keys,
        old_queries,
        old_targets,
        new_queries,
    )

    prior_extension_attempts: dict[str, float] = {}
    for later_index in range(1, len(NEW_OPERATIONS)):
        for earlier_index in range(later_index):
            local = _next_view_predictions(
                base_router,
                extensions[earlier_index],
                new_queries[later_index],
                old_keys,
            )
            prior_extension_attempts[
                f"{NEW_OPERATIONS[earlier_index]}_on_{NEW_OPERATIONS[later_index]}"
            ] = float((local == len(OLD_VIEWS)).float().mean())

    shuffled_selection = []
    for index, (extension, queries) in enumerate(
        zip(shuffled_extensions, new_queries)
    ):
        predictions = _global_extension_predictions(
            base_router,
            extension,
            queries,
            old_keys,
            len(OLD_VIEWS) + index,
        )
        shuffled_selection.append(
            float(
                (predictions == len(OLD_VIEWS) + index).float().mean()
            )
        )

    selected_views = {
        operation: _mode_view(
            old_predictions[index * args.audit_count : (index + 1) * args.audit_count]
        )
        for index, operation in enumerate(OLD_OPERATIONS)
    }
    selected_views.update(
        {
            operation: _mode_view(predictions)
            for operation, predictions in zip(NEW_OPERATIONS, new_predictions)
        }
    )
    selected_behavior: dict[str, float] = {}
    wrong_behavior: dict[str, float] = {}
    for index, operation in enumerate(ALL_OPERATIONS):
        view = selected_views[operation]
        selected_runtime = _load_view(
            parent,
            bank,
            view,
            seed=args.seed + 140_000 + index,
        )
        selected_behavior[operation] = _accuracy(
            selected_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 150_000 + index,
        )
        wrong_view = ALL_VIEWS[(int(view) + 1) % len(ALL_VIEWS)]
        wrong_runtime = _load_view(
            parent,
            bank,
            wrong_view,
            seed=args.seed + 160_000 + index,
        )
        wrong_behavior[operation] = _accuracy(
            wrong_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 150_000 + index,
        )

    _, uncompressed_artifact = bank.promote_view(0, ALL_VIEWS[0])
    uncompressed_compression_outcomes = _direct_capability_outcomes(
        parent,
        uncompressed_artifact,
        views=ALL_VIEWS,
        operations=ALL_OPERATIONS,
        audit_count=args.audit_count,
        retention_probes=args.retention_probes,
        seed=args.seed + 200_000,
    )
    uncompressed_compression_floors = [
        stable_prefix_minimum(
            outcomes,
            min_observations=args.retention_probes,
        )
        for outcomes in uncompressed_compression_outcomes
    ]
    compressed_artifact = compress_growth_artifact(uncompressed_artifact)
    compressed_root = root / "compressed_view_bank"
    aliases = tuple(key for _, key, _ in candidates)
    views = tuple(view for _, _, view in candidates)
    retained_scores = [
        old_retention_scores[operation] for operation in OLD_OPERATIONS
    ] + [
        float(step["candidate_outcome_floor"])
        for step in retention_steps
    ]
    (
        compressed_verifier,
        compressed_candidate_outcome_probe,
        compression_candidate_outcomes,
        compression_behavior_scores,
    ) = _make_compression_callbacks(
        parent,
        views=views,
        capability_keys=aliases,
        selected_behavior=selected_behavior,
        audit_count=args.audit_count,
        retention_probes=args.retention_probes,
        behavior_tolerance=args.behavior_tolerance,
        seed=args.seed + 200_000,
        allow_dtype_cast=True,
        decompress_artifact=False,
    )

    def consolidate_compressed_candidate(
        artifact: dict[str, torch.Tensor],
        *,
        callbacks=None,
    ):
        if compressed_root.exists():
            shutil.rmtree(compressed_root)
        if callbacks is None:
            verifier = compressed_verifier
            candidate_outcome_probe = compressed_candidate_outcome_probe
        else:
            verifier, candidate_outcome_probe = callbacks
        return bank.consolidate_verified(
            (0,),
            bank.rows.keys[0].detach().cpu(),
            artifact,
            compressed_root,
            replacement_aliases=aliases,
            replacement_alias_views=views,
            verifier=verifier,
            candidate_outcome_probe=candidate_outcome_probe,
            retained_scores=retained_scores,
            candidate_threshold=args.retention_threshold,
            retention_floor=args.retention_threshold,
            min_candidate_observations=args.retention_probes,
        )

    compression_adaptive = False
    compression_preserved_names: tuple[str, ...] = ()
    compression_dtype_overrides: dict[str, str] = {}
    compression_decompress_artifact = False
    compression_adaptation_diagnostics: list[dict[str, object]] = []
    compressed_bank, compression_receipt = consolidate_compressed_candidate(
        compressed_artifact
    )
    if not compression_receipt.accepted or compressed_bank is None:
        failed_index = _failed_capability_index(compression_receipt.reason)
        if failed_index is None or failed_index >= len(ALL_VIEWS):
            raise RuntimeError(
                "compressed artifact consolidation was rejected: "
                f"{compression_receipt}; candidate outcomes="
                f"{_retention_probe_report(compression_candidate_outcomes)}"
            )
        (
            compressed_artifact,
            compression_preserved_names,
            compression_adaptation_diagnostics,
        ) = _adaptive_mixed_precision_artifact(
            parent,
            uncompressed_artifact,
            dtype=torch.float16,
            failed_view=ALL_VIEWS[failed_index],
            audit_count=args.audit_count,
            retention_probes=args.retention_probes,
            retention_floor=float(retained_scores[failed_index]),
            seed=args.seed + 200_000,
            max_payload_ratio=0.55,
        )
        failed_capability_floor = stable_prefix_minimum(
            compression_candidate_outcomes[failed_index].outcomes,
            min_observations=args.retention_probes,
        )
        source_failed_capability_floor = uncompressed_compression_floors[
            failed_index
        ]
        ranked_repairs = sorted(
            (
                record
                for record in compression_adaptation_diagnostics
                if bool(record["within_payload_budget"])
                and record["candidate_outcome_floor"] is not None
                and float(record["candidate_outcome_floor"])
                > failed_capability_floor + 1e-9
            ),
            key=lambda record: float(record["candidate_outcome_floor"]),
            reverse=True,
        )
        repair_attempts: list[dict[str, object]] = []
        (
            mixed_verifier,
            mixed_candidate_outcome_probe,
            mixed_candidate_outcomes,
            mixed_behavior_scores,
        ) = _make_compression_callbacks(
            parent,
            views=views,
            capability_keys=aliases,
            selected_behavior=selected_behavior,
            audit_count=args.audit_count,
            retention_probes=args.retention_probes,
            behavior_tolerance=args.behavior_tolerance,
            seed=args.seed + 200_000,
            allow_dtype_cast=True,
            decompress_artifact=True,
        )
        compressed_bank = None
        compression_receipt = None
        for repair in ranked_repairs:
            repair_names = tuple(str(name) for name in repair["preserved_names"])
            override_dtype = repair.get("dtype_override")
            override_names = tuple(
                str(name) for name in repair.get("dtype_override_names", [])
            )
            if override_dtype not in (None, "int8", "int4"):
                raise ValueError("adaptive repair reported an unknown codec")
            repair_artifact = compress_growth_artifact(
                uncompressed_artifact,
                preserve_names=repair_names,
                dtype_overrides=(
                    {
                        name: (
                            torch.int8
                            if override_dtype == "int8"
                            else "int4"
                        )
                        for name in override_names
                    }
                    if override_dtype is not None
                    else None
                ),
            )
            compressed_bank, compression_receipt = consolidate_compressed_candidate(
                repair_artifact,
                callbacks=(
                    (mixed_verifier, mixed_candidate_outcome_probe)
                    if override_dtype is not None
                    else None
                ),
            )
            repair_attempts.append(
                {
                    "preserved_names": list(repair_names),
                    "dtype_override": override_dtype,
                    "dtype_override_names": list(override_names),
                    "receipt": str(compression_receipt),
                }
            )
            if compression_receipt.accepted and compressed_bank is not None:
                compressed_artifact = repair_artifact
                compression_preserved_names = repair_names
                compression_dtype_overrides = {
                    name: str(override_dtype) for name in override_names
                }
                compression_decompress_artifact = override_dtype is not None
                if override_dtype is not None:
                    compression_candidate_outcomes = mixed_candidate_outcomes
                    compression_behavior_scores = mixed_behavior_scores
                break
        compression_adaptation_diagnostics.extend(
            {"transaction": attempt} for attempt in repair_attempts
        )
        if (
            compression_receipt is None
            or not compression_receipt.accepted
            or compressed_bank is None
        ):
            raise RuntimeError(
                "compressed artifact consolidation was rejected after "
                "adaptive mixed-precision search: "
                f"{compression_receipt}; candidate outcomes="
                f"{_retention_probe_report(compression_candidate_outcomes)}; "
                f"preserved={compression_preserved_names!r}; "
                f"failed_capability_floor={failed_capability_floor:.6f}; "
                f"source_capability_floor={source_failed_capability_floor:.6f}; "
                "source_below_protected_floor="
                f"{source_failed_capability_floor < retained_scores[failed_index]}; "
                f"improving_repairs_tried={len(repair_attempts)}; "
                f"adaptive_candidates_measured="
                f"{len(compression_adaptation_diagnostics)}"
            )
        compression_adaptive = True
    compressed_reloaded = ExecutableArtifactMemory.load(compressed_root)
    compressed_selected_behavior: dict[str, float] = {}
    compressed_wrong_behavior: dict[str, float] = {}
    for index, operation in enumerate(ALL_OPERATIONS):
        view = selected_views[operation]
        selected_runtime = _load_view(
            parent,
            compressed_reloaded,
            view,
            seed=args.seed + 180_000 + index,
            allow_dtype_cast=True,
            decompress_artifact=compression_decompress_artifact,
        )
        compressed_selected_behavior[operation] = _accuracy(
            selected_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 150_000 + index,
        )
        wrong_view = ALL_VIEWS[(int(view) + 1) % len(ALL_VIEWS)]
        wrong_runtime = _load_view(
            parent,
            compressed_reloaded,
            wrong_view,
            seed=args.seed + 190_000 + index,
            allow_dtype_cast=True,
            decompress_artifact=compression_decompress_artifact,
        )
        compressed_wrong_behavior[operation] = _accuracy(
            wrong_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 150_000 + index,
        )
    compressed_candidate_exact = (
        [view for _, _, view in compressed_reloaded.view_candidates()]
        == list(ALL_VIEWS)
        and all(
            torch.equal(key, candidate_keys[index])
            for index, (_, key, _) in enumerate(compressed_reloaded.view_candidates())
        )
    )
    compressed_artifact_name = compressed_reloaded.paths[0]
    if compressed_artifact_name is None:
        raise RuntimeError("compressed bank has no artifact path")
    compressed_artifact_path = compressed_root / compressed_artifact_name
    compressed_intact_payload = compressed_artifact_path.read_bytes()
    compressed_artifact_path.write_bytes(compressed_intact_payload + b"corruption")
    compressed_corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(compressed_root)
    except ValueError as error:
        compressed_corruption_rejected = "hash mismatch" in str(error)
    compressed_artifact_path.write_bytes(compressed_intact_payload)
    compressed_core_digest = _digest_core(
        _load_view(
            parent,
            compressed_reloaded,
            ALL_VIEWS[-1],
            seed=args.seed + 200_000,
            allow_dtype_cast=True,
            decompress_artifact=compression_decompress_artifact,
        ),
        ("growth_slots.0.", "growth_slots.1."),
    )
    uncompressed_artifact_name = bank.paths[0]
    if uncompressed_artifact_name is None:
        raise RuntimeError("uncompressed bank has no artifact path")
    uncompressed_file_bytes = (
        root / f"view_bank_{ALL_VIEWS[-1]}" / uncompressed_artifact_name
    ).stat().st_size
    compressed_file_bytes = compressed_artifact_path.stat().st_size
    compression_payload_bytes_before = _payload_bytes(uncompressed_artifact)
    compression_payload_bytes_after = _payload_bytes(compressed_artifact)
    compression_behavior_preserved = all(
        compressed_selected_behavior[operation]
        >= selected_behavior[operation] - 0.05
        for operation in ALL_OPERATIONS
    )

    quantized_artifact = compress_growth_artifact(
        uncompressed_artifact,
        dtype=torch.int8,
    )
    quantized_root = root / "quantized_view_bank"
    if quantized_root.exists():
        shutil.rmtree(quantized_root)

    (
        quantized_verifier,
        quantized_candidate_outcome_probe,
        quantized_candidate_outcomes,
        quantized_behavior_scores,
    ) = _make_compression_callbacks(
        parent,
        views=views,
        capability_keys=aliases,
        selected_behavior=selected_behavior,
        audit_count=args.audit_count,
        retention_probes=args.retention_probes,
        behavior_tolerance=args.behavior_tolerance,
        seed=args.seed + 300_000,
        allow_dtype_cast=False,
        decompress_artifact=True,
    )
    quantized_bank, quantization_receipt = bank.consolidate_verified(
        (0,),
        bank.rows.keys[0].detach().cpu(),
        quantized_artifact,
        quantized_root,
        replacement_aliases=aliases,
        replacement_alias_views=views,
        verifier=quantized_verifier,
        candidate_outcome_probe=quantized_candidate_outcome_probe,
        retained_scores=[
            old_retention_scores[operation] for operation in OLD_OPERATIONS
        ]
        + [
            float(step["candidate_outcome_floor"])
            for step in retention_steps
        ],
        candidate_threshold=args.retention_threshold,
        retention_floor=args.retention_threshold,
        min_candidate_observations=args.retention_probes,
    )
    if not quantization_receipt.accepted or quantized_bank is None:
        raise RuntimeError(
            "quantized artifact consolidation was rejected: "
            f"{quantization_receipt}; candidate outcomes="
            f"{_retention_probe_report(quantized_candidate_outcomes)}"
        )
    quantized_reloaded = ExecutableArtifactMemory.load(quantized_root)
    quantized_selected_behavior: dict[str, float] = {}
    quantized_wrong_behavior: dict[str, float] = {}
    for index, operation in enumerate(ALL_OPERATIONS):
        view = selected_views[operation]
        selected_runtime = _load_view(
            parent,
            quantized_reloaded,
            view,
            seed=args.seed + 210_000 + index,
            decompress_artifact=True,
        )
        quantized_selected_behavior[operation] = _accuracy(
            selected_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 150_000 + index,
        )
        wrong_view = ALL_VIEWS[(int(view) + 1) % len(ALL_VIEWS)]
        wrong_runtime = _load_view(
            parent,
            quantized_reloaded,
            wrong_view,
            seed=args.seed + 220_000 + index,
            decompress_artifact=True,
        )
        quantized_wrong_behavior[operation] = _accuracy(
            wrong_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 150_000 + index,
        )
    quantized_candidate_exact = (
        [view for _, _, view in quantized_reloaded.view_candidates()]
        == list(ALL_VIEWS)
        and all(
            torch.equal(key, candidate_keys[index])
            for index, (_, key, _) in enumerate(quantized_reloaded.view_candidates())
        )
    )
    quantized_artifact_name = quantized_reloaded.paths[0]
    if quantized_artifact_name is None:
        raise RuntimeError("quantized bank has no artifact path")
    quantized_artifact_path = quantized_root / quantized_artifact_name
    quantized_intact_payload = quantized_artifact_path.read_bytes()
    quantized_artifact_path.write_bytes(quantized_intact_payload + b"corruption")
    quantized_corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(quantized_root)
    except ValueError as error:
        quantized_corruption_rejected = "hash mismatch" in str(error)
    quantized_artifact_path.write_bytes(quantized_intact_payload)
    quantized_core_digest = _digest_core(
        _load_view(
            parent,
            quantized_reloaded,
            ALL_VIEWS[-1],
            seed=args.seed + 230_000,
            decompress_artifact=True,
        ),
        ("growth_slots.0.", "growth_slots.1."),
    )
    quantized_payload_bytes_before = _payload_bytes(uncompressed_artifact)
    quantized_payload_bytes_after = _payload_bytes(quantized_artifact)
    quantized_file_bytes = quantized_artifact_path.stat().st_size
    quantized_behavior_preserved = all(
        quantized_selected_behavior[operation]
        >= selected_behavior[operation] - 0.05
        for operation in ALL_OPERATIONS
    )

    packed_artifact = compress_growth_artifact(
        uncompressed_artifact,
        dtype="int4",
    )
    packed_root = root / "packed_int4_view_bank"
    if packed_root.exists():
        shutil.rmtree(packed_root)
    (
        packed_verifier,
        packed_candidate_outcome_probe,
        packed_candidate_outcomes,
        packed_behavior_scores,
    ) = _make_compression_callbacks(
        parent,
        views=views,
        capability_keys=aliases,
        selected_behavior=selected_behavior,
        audit_count=args.audit_count,
        retention_probes=args.retention_probes,
        behavior_tolerance=args.behavior_tolerance,
        seed=args.seed + 400_000,
        allow_dtype_cast=False,
        decompress_artifact=True,
    )
    packed_bank, packed_receipt = bank.consolidate_verified(
        (0,),
        bank.rows.keys[0].detach().cpu(),
        packed_artifact,
        packed_root,
        replacement_aliases=aliases,
        replacement_alias_views=views,
        verifier=packed_verifier,
        candidate_outcome_probe=packed_candidate_outcome_probe,
        retained_scores=[
            old_retention_scores[operation] for operation in OLD_OPERATIONS
        ]
        + [
            float(step["candidate_outcome_floor"])
            for step in retention_steps
        ],
        candidate_threshold=args.retention_threshold,
        retention_floor=args.retention_threshold,
        min_candidate_observations=args.retention_probes,
    )
    if not packed_receipt.accepted or packed_bank is None:
        raise RuntimeError(
            "packed int4 consolidation was rejected: "
            f"{packed_receipt}; candidate outcomes="
            f"{_retention_probe_report(packed_candidate_outcomes)}"
        )
    packed_reloaded = ExecutableArtifactMemory.load(packed_root)
    packed_selected_behavior: dict[str, float] = {}
    packed_wrong_behavior: dict[str, float] = {}
    for index, operation in enumerate(ALL_OPERATIONS):
        view = selected_views[operation]
        selected_runtime = _load_view(
            parent,
            packed_reloaded,
            view,
            seed=args.seed + 240_000 + index,
            decompress_artifact=True,
        )
        packed_selected_behavior[operation] = _accuracy(
            selected_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 150_000 + index,
        )
        wrong_view = ALL_VIEWS[(int(view) + 1) % len(ALL_VIEWS)]
        wrong_runtime = _load_view(
            parent,
            packed_reloaded,
            wrong_view,
            seed=args.seed + 250_000 + index,
            decompress_artifact=True,
        )
        packed_wrong_behavior[operation] = _accuracy(
            wrong_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 150_000 + index,
        )
    packed_candidate_exact = (
        [view for _, _, view in packed_reloaded.view_candidates()]
        == list(ALL_VIEWS)
        and all(
            torch.equal(key, candidate_keys[index])
            for index, (_, key, _) in enumerate(packed_reloaded.view_candidates())
        )
    )
    packed_artifact_name = packed_reloaded.paths[0]
    if packed_artifact_name is None:
        raise RuntimeError("packed int4 bank has no artifact path")
    packed_artifact_path = packed_root / packed_artifact_name
    packed_intact_payload = packed_artifact_path.read_bytes()
    packed_artifact_path.write_bytes(packed_intact_payload + b"corruption")
    packed_corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(packed_root)
    except ValueError as error:
        packed_corruption_rejected = "hash mismatch" in str(error)
    packed_artifact_path.write_bytes(packed_intact_payload)
    packed_core_digest = _digest_core(
        _load_view(
            parent,
            packed_reloaded,
            ALL_VIEWS[-1],
            seed=args.seed + 260_000,
            decompress_artifact=True,
        ),
        ("growth_slots.0.", "growth_slots.1."),
    )
    packed_payload_bytes_before = _payload_bytes(uncompressed_artifact)
    packed_payload_bytes_after = _payload_bytes(packed_artifact)
    packed_file_bytes = packed_artifact_path.stat().st_size
    packed_behavior_preserved = all(
        packed_selected_behavior[operation]
        >= selected_behavior[operation] - 0.05
        for operation in ALL_OPERATIONS
    )

    torch.save(base_router.state_dict(), root / "frozen_four_view_router.pt")
    for index, extension in enumerate(extensions):
        torch.save(
            extension.state_dict(),
            root / f"view_{len(OLD_VIEWS) + index}_extension.pt",
        )
    reloaded = ExecutableArtifactMemory.load(root / f"view_bank_{ALL_VIEWS[-1]}")
    reloaded_candidates = reloaded.view_candidates()
    reloaded_keys = torch.stack([key for _, key, _ in reloaded_candidates])
    reloaded_router = OpaqueAddressRouter(width=64, hidden=64)
    reloaded_router.load_state_dict(
        torch.load(root / "frozen_four_view_router.pt", weights_only=False)
    )
    reloaded_extensions = []
    for index in range(len(NEW_OPERATIONS)):
        extension = OpaqueViewRouteExtension(width=64, hidden=64)
        extension.load_state_dict(
            torch.load(
                root / f"view_{len(OLD_VIEWS) + index}_extension.pt",
                weights_only=False,
            )
        )
        reloaded_extensions.append(extension)
    reloaded_old = reloaded_router(old_queries, reloaded_keys[: len(OLD_VIEWS)]).argmax(
        dim=-1
    )
    reloaded_predictions = [reloaded_old]
    for index, (extension, queries) in enumerate(
        zip(reloaded_extensions, new_queries)
    ):
        reloaded_predictions.append(
            _global_extension_predictions(
                reloaded_router,
                extension,
                queries,
                reloaded_keys[: len(OLD_VIEWS)],
                len(OLD_VIEWS) + index,
            )
        )
    reloaded_route = float(
        (
            torch.cat(reloaded_predictions) == combined_targets
        ).float().mean()
    )
    reloaded_candidate_exact = (
        [view for _, _, view in reloaded_candidates] == list(ALL_VIEWS)
        and all(
            torch.equal(key, candidate_keys[index])
            for index, (_, key, _) in enumerate(reloaded_candidates)
        )
    )
    artifact_name = reloaded.paths[0]
    if artifact_name is None:
        raise RuntimeError("reloaded three-step bank has no artifact path")
    artifact_path = (root / f"view_bank_{ALL_VIEWS[-1]}") / artifact_name
    intact_payload = artifact_path.read_bytes()
    artifact_path.write_bytes(intact_payload + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(root / f"view_bank_{ALL_VIEWS[-1]}")
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(intact_payload)
    reloaded_core_digest = _digest_core(
        _load_view(
            parent,
            reloaded,
            ALL_VIEWS[-1],
            seed=args.seed + 170_000,
        ),
        ("growth_slots.0.", "growth_slots.1."),
    )

    report = {
        "schema": "neural-computer.three-step-view-growth-report.v1",
        "claim_boundary": (
            "Three new executable views are acquired sequentially through a "
            "failure-gated external chain while the controller, four-view "
            "router, and earlier extensions remain frozen; no prior route "
            "data is replayed. This is not unrestricted continual learning."
        ),
        "seed": args.seed,
        "old_operations": list(OLD_OPERATIONS),
        "new_operations": list(NEW_OPERATIONS),
        "view_ids": list(ALL_VIEWS),
        "updates": args.updates,
        "extension_artifact_updates": args.extension_artifact_updates,
        "route_updates": args.route_updates,
        "extension_updates": args.extension_updates,
        "batch_size": args.batch_size,
        "route_batch_size": args.route_batch_size,
        "audit_count": args.audit_count,
        "retention_threshold": args.retention_threshold,
        "retention_probes": args.retention_probes,
        "behavior_tolerance": args.behavior_tolerance,
        "physical_rows": len(bank.occupied),
        "behavior_floor": args.retention_threshold,
        "old_retention_scores": old_retention_scores,
        "old_retention_observations": old_retention_observations,
        "old_protected_before_growth": old_protected_before_growth,
        "retention_steps": retention_steps,
        "all_replacements_protected": all(
            bool(step["replacement_protected"]) for step in retention_steps
        ),
        "compression_candidate_outcomes": _retention_probe_report(
            compression_candidate_outcomes
        ),
        "compression_candidate_outcome_floor": _retention_probe_floor(
            compression_candidate_outcomes,
            min_observations=args.retention_probes,
        ),
        "uncompressed_compression_outcomes": uncompressed_compression_outcomes,
        "uncompressed_compression_floors": uncompressed_compression_floors,
        "compression_behavior_scores": compression_behavior_scores,
        "compression_adaptive_mixed_precision": compression_adaptive,
        "compression_preserved_names": list(compression_preserved_names),
        "compression_dtype_overrides": compression_dtype_overrides,
        "compression_adaptation_diagnostics": (
            compression_adaptation_diagnostics
        ),
        "quantized_candidate_outcomes": _retention_probe_report(
            quantized_candidate_outcomes
        ),
        "quantized_candidate_outcome_floor": _retention_probe_floor(
            quantized_candidate_outcomes,
            min_observations=args.retention_probes,
        ),
        "quantized_behavior_scores": quantized_behavior_scores,
        "packed_candidate_outcomes": _retention_probe_report(
            packed_candidate_outcomes
        ),
        "packed_candidate_outcome_floor": _retention_probe_floor(
            packed_candidate_outcomes,
            min_observations=args.retention_probes,
        ),
        "packed_behavior_scores": packed_behavior_scores,
        "base_old_route_accuracy": base_old_route,
        "new_route_accuracy": dict(zip(NEW_OPERATIONS, new_route)),
        "three_step_route_accuracy": combined_route,
        "candidate_permutation_accuracy": permuted_route,
        "prior_extension_attempt_rates": prior_extension_attempts,
        "reward_shuffled_new_selection_rates": dict(
            zip(NEW_OPERATIONS, shuffled_selection)
        ),
        "selected_views": selected_views,
        "selected_behavior": selected_behavior,
        "wrong_behavior": wrong_behavior,
        "compressed_selected_behavior": compressed_selected_behavior,
        "compressed_wrong_behavior": compressed_wrong_behavior,
        "compression_payload_bytes_before": compression_payload_bytes_before,
        "compression_payload_bytes_after": compression_payload_bytes_after,
        "compression_payload_byte_ratio": (
            compression_payload_bytes_after / compression_payload_bytes_before
        ),
        "uncompressed_file_bytes": uncompressed_file_bytes,
        "compressed_file_bytes": compressed_file_bytes,
        "compression_file_byte_ratio": compressed_file_bytes / uncompressed_file_bytes,
        "compressed_candidate_exact": compressed_candidate_exact,
        "compressed_corruption_rejected": compressed_corruption_rejected,
        "compressed_core_digest": compressed_core_digest,
        "compression_behavior_preserved": compression_behavior_preserved,
        "quantized_selected_behavior": quantized_selected_behavior,
        "quantized_wrong_behavior": quantized_wrong_behavior,
        "quantized_payload_bytes_before": quantized_payload_bytes_before,
        "quantized_payload_bytes_after": quantized_payload_bytes_after,
        "quantized_payload_byte_ratio": (
            quantized_payload_bytes_after / quantized_payload_bytes_before
        ),
        "quantized_file_bytes": quantized_file_bytes,
        "quantized_file_byte_ratio": quantized_file_bytes / uncompressed_file_bytes,
        "quantized_candidate_exact": quantized_candidate_exact,
        "quantized_corruption_rejected": quantized_corruption_rejected,
        "quantized_core_digest": quantized_core_digest,
        "quantized_behavior_preserved": quantized_behavior_preserved,
        "packed_selected_behavior": packed_selected_behavior,
        "packed_wrong_behavior": packed_wrong_behavior,
        "packed_payload_bytes_before": packed_payload_bytes_before,
        "packed_payload_bytes_after": packed_payload_bytes_after,
        "packed_payload_byte_ratio": (
            packed_payload_bytes_after / packed_payload_bytes_before
        ),
        "packed_file_bytes": packed_file_bytes,
        "packed_file_byte_ratio": packed_file_bytes / uncompressed_file_bytes,
        "packed_candidate_exact": packed_candidate_exact,
        "packed_corruption_rejected": packed_corruption_rejected,
        "packed_core_digest": packed_core_digest,
        "packed_behavior_preserved": packed_behavior_preserved,
        "reloaded_three_step_accuracy": reloaded_route,
        "reloaded_candidate_exact": reloaded_candidate_exact,
        "corruption_rejected": corruption_rejected,
        "parent_core_digest": parent_digest,
        "reloaded_core_digest": reloaded_core_digest,
        "all_extensions_frozen_during_later_training": extensions_frozen,
        "accounting": {
            "unique_logical_lifetimes": (
                args.updates * (len(OLD_OPERATIONS) + 1) * args.batch_size
                + len(NEW_OPERATIONS)
                * args.extension_artifact_updates
                * args.batch_size
            ),
            "unique_verifier_bits": (
                args.updates * (len(OLD_OPERATIONS) + 1) * args.batch_size
                + len(NEW_OPERATIONS)
                * args.extension_artifact_updates
                * args.batch_size
            )
            * 4,
            "optimizer_updates": args.updates * (len(OLD_OPERATIONS) + 1)
            + len(NEW_OPERATIONS) * args.extension_artifact_updates,
            "base_route_optimizer_updates": base_accounting[
                "route_optimizer_updates"
            ],
            "extension_route_optimizer_updates": sum(
                accounting["route_optimizer_updates"]
                for accounting in extension_accounting
            ),
            "route_unique_verifier_bits": base_accounting[
                "unique_route_verifier_bits"
            ]
            + sum(
                accounting["unique_route_verifier_bits"]
                for accounting in extension_accounting
            )
            + sum(
                accounting["unique_route_verifier_bits"]
                for accounting in shuffled_accounting
            ),
            "replayed_examples_after_each_extension": [0] * len(NEW_OPERATIONS),
            "retention_observations": (
                (len(OLD_OPERATIONS) + len(NEW_OPERATIONS))
                * args.retention_probes
            ),
            "compression_retention_observations": (
                len(ALL_OPERATIONS) * args.retention_probes
            ),
            "uncompressed_compression_retention_observations": (
                len(ALL_OPERATIONS) * args.retention_probes
            ),
            "quantization_retention_observations": (
                len(ALL_OPERATIONS) * args.retention_probes
            ),
            "packed_quantization_retention_observations": (
                len(ALL_OPERATIONS) * args.retention_probes
            ),
            "compression_optimizer_updates": 0,
            "replayed_examples_for_compression": 0,
            "compression_audit_lifetimes": args.audit_count * len(ALL_OPERATIONS),
            "compression_audit_verifier_bits": (
                args.audit_count * len(ALL_OPERATIONS) * 4
            ),
            "quantization_optimizer_updates": 0,
            "replayed_examples_for_quantization": 0,
            "quantization_audit_lifetimes": args.audit_count * len(ALL_OPERATIONS),
            "quantization_audit_verifier_bits": (
                args.audit_count * len(ALL_OPERATIONS) * 4
            ),
            "packed_quantization_optimizer_updates": 0,
            "replayed_examples_for_packed_quantization": 0,
            "packed_quantization_audit_lifetimes": args.audit_count * len(ALL_OPERATIONS),
            "packed_quantization_audit_verifier_bits": (
                args.audit_count * len(ALL_OPERATIONS) * 4
            ),
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "one_physical_row": len(bank.occupied) == 1,
            "seven_opaque_views": [view for _, _, view in candidates]
            == list(ALL_VIEWS),
            "base_old_route_mastered": base_old_route >= 0.90,
            "all_new_views_mastered": all(rate >= 0.75 for rate in new_route),
            "old_capabilities_protected_before_growth": all(
                old_protected_before_growth
            ),
            "all_new_candidate_retention_stable": all(
                float(step["candidate_outcome_floor"]) >= args.retention_threshold
                for step in retention_steps
            ),
            "all_replacements_protected_before_next_step": all(
                bool(step["replacement_protected"]) for step in retention_steps
            ),
            "retention_safe_three_step_growth": (
                all(old_protected_before_growth)
                and all(
                    float(step["candidate_outcome_floor"])
                    >= args.retention_threshold
                    for step in retention_steps
                )
                and all(bool(step["replacement_protected"]) for step in retention_steps)
            ),
            "compression_candidate_retention_stable": (
                _retention_probe_floor(
                    compression_candidate_outcomes,
                    min_observations=args.retention_probes,
                )
                >= args.retention_threshold
            ),
            "compression_behavior_verified_before_adoption": all(
                compression_behavior_scores[operation]
                >= selected_behavior[operation] - args.behavior_tolerance
                for operation in ALL_OPERATIONS
            ),
            "quantized_candidate_retention_stable": (
                _retention_probe_floor(
                    quantized_candidate_outcomes,
                    min_observations=args.retention_probes,
                )
                >= args.retention_threshold
            ),
            "quantized_behavior_verified_before_adoption": all(
                quantized_behavior_scores[operation]
                >= selected_behavior[operation] - args.behavior_tolerance
                for operation in ALL_OPERATIONS
            ),
            "packed_candidate_retention_stable": (
                _retention_probe_floor(
                    packed_candidate_outcomes,
                    min_observations=args.retention_probes,
                )
                >= args.retention_threshold
            ),
            "packed_behavior_verified_before_adoption": all(
                packed_behavior_scores[operation]
                >= selected_behavior[operation] - args.behavior_tolerance
                for operation in ALL_OPERATIONS
            ),
            "all_extension_behaviors_preserved": all(
                all(
                    step["behavior_scores"][operation]
                    >= step["behavior_baselines"][operation]
                    - args.behavior_tolerance
                    for operation in step["behavior_baselines"]
                )
                for step in retention_steps
            ),
            "three_step_chain_at_least_80": combined_route >= 0.80,
            "candidate_permutation_invariant": permuted_route >= 0.80,
            "all_prior_extensions_attempted": all(
                rate >= 0.75 for rate in prior_extension_attempts.values()
            ),
            "all_reward_shuffled_new_near_chance": all(
                rate <= 0.50 for rate in shuffled_selection
            ),
            "selected_views_match_expected": selected_views
            == dict(zip(ALL_OPERATIONS, ALL_VIEWS, strict=True)),
            "all_selected_behaviors_mastered": min(selected_behavior.values())
            >= 0.70,
            "all_wrong_views_causal": all(
                selected_behavior[operation]
                > wrong_behavior[operation] + 0.05
                for operation in ALL_OPERATIONS
            ),
            "compressed_payload_at_most_55_percent": (
                compression_payload_bytes_after / compression_payload_bytes_before
                <= 0.55
            ),
            "compressed_file_smaller": compressed_file_bytes < uncompressed_file_bytes,
            "compressed_behavior_preserved": compression_behavior_preserved
            and min(compressed_selected_behavior.values()) >= 0.70,
            "compressed_wrong_views_causal": all(
                compressed_selected_behavior[operation]
                > compressed_wrong_behavior[operation] + 0.05
                for operation in ALL_OPERATIONS
            ),
            "compressed_candidate_exact": compressed_candidate_exact,
            "compressed_corruption_rejected": compressed_corruption_rejected,
            "compressed_core_unchanged": parent_digest == compressed_core_digest,
            "quantized_payload_at_most_30_percent": (
                quantized_payload_bytes_after / quantized_payload_bytes_before
                <= 0.30
            ),
            "quantized_file_smaller": quantized_file_bytes < uncompressed_file_bytes,
            "quantized_behavior_preserved": quantized_behavior_preserved
            and min(quantized_selected_behavior.values()) >= 0.70,
            "quantized_wrong_views_causal": all(
                quantized_selected_behavior[operation]
                > quantized_wrong_behavior[operation] + 0.05
                for operation in ALL_OPERATIONS
            ),
            "quantized_candidate_exact": quantized_candidate_exact,
            "quantized_corruption_rejected": quantized_corruption_rejected,
            "quantized_core_unchanged": parent_digest == quantized_core_digest,
            "packed_payload_at_most_15_percent": (
                packed_payload_bytes_after / packed_payload_bytes_before <= 0.15
            ),
            "packed_file_smaller": packed_file_bytes < uncompressed_file_bytes,
            "packed_behavior_preserved": packed_behavior_preserved
            and min(packed_selected_behavior.values()) >= 0.70,
            "packed_wrong_views_causal": all(
                packed_selected_behavior[operation]
                > packed_wrong_behavior[operation] + 0.05
                for operation in ALL_OPERATIONS
            ),
            "packed_candidate_exact": packed_candidate_exact,
            "packed_corruption_rejected": packed_corruption_rejected,
            "packed_core_unchanged": parent_digest == packed_core_digest,
            "reloaded_route_preserved": reloaded_route >= combined_route - 0.05,
            "reloaded_candidate_exact": reloaded_candidate_exact,
            "frozen_controller_core": parent_digest == reloaded_core_digest,
            "all_extensions_frozen_during_later_training": extensions_frozen,
            "corruption_rejected": corruption_rejected,
            "no_replayed_examples_after_each_extension": True,
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
    parser.add_argument("--updates", type=int, default=64)
    parser.add_argument("--extension-artifact-updates", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--route-updates", type=int, default=256)
    parser.add_argument("--extension-updates", type=int, default=128)
    parser.add_argument("--route-batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=32)
    parser.add_argument("--retention-probes", type=int, default=8)
    parser.add_argument("--retention-threshold", type=float, default=0.70)
    parser.add_argument("--behavior-tolerance", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise ValueError("torch threads must be positive")
    torch.set_num_threads(args.torch_threads)
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "base_old_route_accuracy": report["base_old_route_accuracy"],
                "new_route_accuracy": report["new_route_accuracy"],
                "three_step_route_accuracy": report["three_step_route_accuracy"],
                "candidate_permutation_accuracy": report[
                    "candidate_permutation_accuracy"
                ],
                "prior_extension_attempt_rates": report[
                    "prior_extension_attempt_rates"
                ],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
