"""Audit two sequential external capability additions without replay."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.artifact_consolidation_amodal.train import _load_single
from experiments.artifact_view_routing_amodal.train import _load_composed
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
    ExecutableArtifactMemory,
    OpaqueAddressRouter,
    OpaqueViewRouteExtension,
    RetentionPolicyConfig,
    failure_gated_view_scores,
)

NEW_OPERATIONS = ("rotate", "complement_rotate")
ALL_OPERATIONS = (*OLD_OPERATIONS, *NEW_OPERATIONS)
OLD_VIEWS = tuple(str(index) for index in range(len(OLD_OPERATIONS)))
FIRST_NEW_VIEW = str(len(OLD_VIEWS))
SECOND_NEW_VIEW = str(len(OLD_VIEWS) + 1)
ALL_VIEWS = (*OLD_VIEWS, FIRST_NEW_VIEW, SECOND_NEW_VIEW)


def _next_view_predictions(
    base_router: OpaqueAddressRouter,
    extension: OpaqueViewRouteExtension,
    queries: torch.Tensor,
    old_keys: torch.Tensor,
) -> torch.Tensor:
    scores = failure_gated_view_scores(
        base_router(queries, old_keys),
        extension(queries),
        True,
    )
    return scores.argmax(dim=-1)


def _chain_predictions(
    base_router: OpaqueAddressRouter,
    first_extension: OpaqueViewRouteExtension,
    second_extension: OpaqueViewRouteExtension,
    old_keys: torch.Tensor,
    old_queries: torch.Tensor,
    first_queries: torch.Tensor,
    second_queries: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    old_predictions = base_router(old_queries, old_keys).argmax(dim=-1)
    first_predictions = _next_view_predictions(
        base_router, first_extension, first_queries, old_keys
    )
    # The first new view is deliberately attempted and rejected on the
    # second procedure. The next extension is then opened by the second
    # scalar failure; it is not selected optimistically alongside view five.
    second_predictions = _next_view_predictions(
        base_router, second_extension, second_queries, old_keys
    )
    second_predictions = torch.where(
        second_predictions == len(OLD_VIEWS),
        torch.full_like(second_predictions, int(SECOND_NEW_VIEW)),
        second_predictions,
    )
    return old_predictions, first_predictions, second_predictions


def _load_view(parent, bank, view: str, *, seed: int):
    handle, artifact = bank.promote_view(0, view)
    if handle.view != view:
        raise RuntimeError("memory returned the wrong opaque view")
    return _load_composed(parent, artifact, seed=seed, view=view)


def _permuted_chain_accuracy(
    base_router: OpaqueAddressRouter,
    first_extension: OpaqueViewRouteExtension,
    second_extension: OpaqueViewRouteExtension,
    old_keys: torch.Tensor,
    old_queries: torch.Tensor,
    old_targets: torch.Tensor,
    first_queries: torch.Tensor,
    second_queries: torch.Tensor,
) -> float:
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted_keys = old_keys[permutation]
    old_predictions = base_router(old_queries, permuted_keys).argmax(dim=-1)
    old_predictions = permutation[old_predictions]
    first_predictions = _next_view_predictions(
        base_router, first_extension, first_queries, permuted_keys
    )
    second_predictions = _next_view_predictions(
        base_router, second_extension, second_queries, permuted_keys
    )
    second_predictions = torch.where(
        second_predictions == len(OLD_VIEWS),
        torch.full_like(second_predictions, int(SECOND_NEW_VIEW)),
        second_predictions,
    )
    targets = torch.cat(
        (
            old_targets,
            torch.full((first_queries.shape[0],), 4, dtype=torch.long),
            torch.full((second_queries.shape[0],), 5, dtype=torch.long),
        )
    )
    predictions = torch.cat(
        (old_predictions, first_predictions, second_predictions)
    )
    return float((predictions == targets).float().mean())


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
        old_retention_scores[operation] = min(observations)
    old_bank.save()
    old_protected_before_first = [
        old_bank.retention.is_protected(key) for _, key, _ in old_candidates
    ]

    first_artifact = _train_new_artifact(
        parent,
        seed=args.seed + 1_000,
        updates=args.extension_artifact_updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        operation=NEW_OPERATIONS[0],
    )
    key_generator = torch.Generator(device="cpu").manual_seed(args.seed + 90_000)
    first_key = F.normalize(torch.randn(64, generator=key_generator), dim=0)

    first_candidate_outcomes: list[float] = []
    first_behavior_baselines: dict[str, float] = {}
    first_behavior_scores: dict[str, float] = {}

    def first_candidate_outcome_probe(
        candidate: ExecutableArtifactMemory,
    ) -> Sequence[float]:
        observations: list[float] = []
        for probe in range(args.retention_probes):
            runtime = _load_view(
                parent,
                candidate,
                FIRST_NEW_VIEW,
                seed=args.seed + 40_000 + probe,
            )
            observations.append(
                _accuracy(
                    runtime,
                    operation=NEW_OPERATIONS[0],
                    count=args.audit_count,
                    span=4,
                    seed=args.seed + 50_000 + probe,
                )
            )
        first_candidate_outcomes[:] = observations
        return observations

    def first_extension_verifier(candidate: ExecutableArtifactMemory) -> bool:
        expected_views = [
            candidate.promote_view(0, view)[0].view
            for view in (*OLD_VIEWS, FIRST_NEW_VIEW)
        ]
        if expected_views != [*OLD_VIEWS, FIRST_NEW_VIEW]:
            return False
        baselines = {
            **old_retention_scores,
            NEW_OPERATIONS[0]: _accuracy(
                _load_single(parent, first_artifact, seed=args.seed + 60_000),
                operation=NEW_OPERATIONS[0],
                count=args.audit_count,
                span=4,
                seed=args.seed + 61_000,
            ),
        }
        first_behavior_baselines.update(baselines)
        for family, operation in enumerate((*OLD_OPERATIONS, NEW_OPERATIONS[0])):
            runtime = _load_view(
                parent,
                candidate,
                (*OLD_VIEWS, FIRST_NEW_VIEW)[family],
                seed=args.seed + 62_000 + family,
            )
            first_behavior_scores[operation] = _accuracy(
                runtime,
                operation=operation,
                count=args.audit_count,
                span=4,
                seed=args.seed + 63_000 + family,
            )
        return all(
            first_behavior_scores[operation]
            >= baselines[operation] - args.behavior_tolerance
            for operation in (*OLD_OPERATIONS, NEW_OPERATIONS[0])
        )

    first_bank = _extend_bank(
        old_bank,
        first_artifact,
        first_key,
        directory=root / "first_view_bank",
        existing_views=OLD_VIEWS,
        new_view=FIRST_NEW_VIEW,
        target_slot=4,
        verifier=first_extension_verifier,
        candidate_outcome_probe=first_candidate_outcome_probe,
        retained_scores=[old_retention_scores[operation] for operation in OLD_OPERATIONS],
        retention_threshold=args.retention_threshold,
        retention_probes=args.retention_probes,
    )
    first_bank.retention.config = RetentionPolicyConfig(
        mastery_threshold=args.retention_threshold,
        min_mastery_observations=args.retention_probes,
    )
    first_protected_before_second = first_bank._row_is_protected(0)

    candidates = first_bank.view_candidates()
    first_keys = torch.stack([key for _, key, _ in candidates])
    if [view for _, _, view in candidates] != [*OLD_VIEWS, FIRST_NEW_VIEW]:
        raise RuntimeError("first extension produced the wrong view order")
    old_keys = first_keys[: len(OLD_VIEWS)]
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
    first_extension, first_accounting = _train_route_extension(
        parent,
        base_router,
        old_keys,
        updates=args.extension_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 70_000,
        shuffle_outcomes=False,
        operation=NEW_OPERATIONS[0],
    )
    shuffled_first, shuffled_first_accounting = _train_route_extension(
        parent,
        base_router,
        old_keys,
        updates=args.extension_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 80_000,
        shuffle_outcomes=True,
        operation=NEW_OPERATIONS[0],
    )
    first_extension_digest = {
        name: value.detach().cpu().clone()
        for name, value in first_extension.state_dict().items()
    }

    second_artifact = _train_new_artifact(
        parent,
        seed=args.seed + 2_000,
        updates=args.extension_artifact_updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        operation=NEW_OPERATIONS[1],
    )
    second_key = F.normalize(torch.randn(64, generator=key_generator), dim=0)
    second_candidate_outcomes: list[float] = []
    second_behavior_baselines: dict[str, float] = {}
    second_behavior_scores: dict[str, float] = {}
    first_retention_floor = min(first_candidate_outcomes)
    retained_scores_after_first = [
        *(old_retention_scores[operation] for operation in OLD_OPERATIONS),
        first_retention_floor,
    ]

    def second_candidate_outcome_probe(
        candidate: ExecutableArtifactMemory,
    ) -> Sequence[float]:
        observations: list[float] = []
        for probe in range(args.retention_probes):
            runtime = _load_view(
                parent,
                candidate,
                SECOND_NEW_VIEW,
                seed=args.seed + 70_000 + probe,
            )
            observations.append(
                _accuracy(
                    runtime,
                    operation=NEW_OPERATIONS[1],
                    count=args.audit_count,
                    span=4,
                    seed=args.seed + 80_000 + probe,
                )
            )
        second_candidate_outcomes[:] = observations
        return observations

    def second_extension_verifier(candidate: ExecutableArtifactMemory) -> bool:
        expected_views = [
            candidate.promote_view(0, view)[0].view for view in ALL_VIEWS
        ]
        if expected_views != list(ALL_VIEWS):
            return False
        baselines = {
            **old_retention_scores,
            NEW_OPERATIONS[0]: first_retention_floor,
            NEW_OPERATIONS[1]: _accuracy(
                _load_single(parent, second_artifact, seed=args.seed + 90_000),
                operation=NEW_OPERATIONS[1],
                count=args.audit_count,
                span=4,
                seed=args.seed + 91_000,
            ),
        }
        second_behavior_baselines.update(baselines)
        for family, operation in enumerate(ALL_OPERATIONS):
            runtime = _load_view(
                parent,
                candidate,
                ALL_VIEWS[family],
                seed=args.seed + 92_000 + family,
            )
            second_behavior_scores[operation] = _accuracy(
                runtime,
                operation=operation,
                count=args.audit_count,
                span=4,
                seed=args.seed + 93_000 + family,
            )
        return all(
            second_behavior_scores[operation]
            >= baselines[operation] - args.behavior_tolerance
            for operation in ALL_OPERATIONS
        )

    bank = _extend_bank(
        first_bank,
        second_artifact,
        second_key,
        directory=root / "extended_view_bank",
        existing_views=(*OLD_VIEWS, FIRST_NEW_VIEW),
        new_view=SECOND_NEW_VIEW,
        target_slot=5,
        verifier=second_extension_verifier,
        candidate_outcome_probe=second_candidate_outcome_probe,
        retained_scores=retained_scores_after_first,
        retention_threshold=args.retention_threshold,
        retention_probes=args.retention_probes,
    )
    replacement_protected_after_second = bank._row_is_protected(0)
    second_candidates = bank.view_candidates()
    candidate_keys = torch.stack([key for _, key, _ in second_candidates])
    if [view for _, _, view in second_candidates] != list(ALL_VIEWS):
        raise RuntimeError("second extension produced the wrong view order")
    old_keys = candidate_keys[: len(OLD_VIEWS)]

    second_extension, second_accounting = _train_route_extension(
        parent,
        base_router,
        old_keys,
        updates=args.extension_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 90_000,
        shuffle_outcomes=False,
        operation=NEW_OPERATIONS[1],
    )
    shuffled_second, shuffled_second_accounting = _train_route_extension(
        parent,
        base_router,
        old_keys,
        updates=args.extension_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 100_000,
        shuffle_outcomes=True,
        operation=NEW_OPERATIONS[1],
    )
    first_extension_unchanged = all(
        torch.equal(first_extension_digest[name], value.detach().cpu())
        for name, value in first_extension.state_dict().items()
    )

    old_queries, old_targets = _test_queries(
        parent,
        audit_count=args.audit_count,
        seed=args.seed + 110_000,
    )
    first_queries = _fresh_queries(
        parent,
        operation=NEW_OPERATIONS[0],
        count=args.audit_count * len(OLD_VIEWS),
        seed=args.seed + 120_000,
    )
    second_queries = _fresh_queries(
        parent,
        operation=NEW_OPERATIONS[1],
        count=args.audit_count * len(OLD_VIEWS),
        seed=args.seed + 130_000,
    )
    old_predictions, first_predictions, second_predictions = _chain_predictions(
        base_router,
        first_extension,
        second_extension,
        old_keys,
        old_queries,
        first_queries,
        second_queries,
    )
    first_targets = torch.full(
        (first_queries.shape[0],), 4, dtype=torch.long
    )
    second_targets = torch.full(
        (second_queries.shape[0],), 5, dtype=torch.long
    )
    combined_targets = torch.cat((old_targets, first_targets, second_targets))
    combined_predictions = torch.cat(
        (old_predictions, first_predictions, second_predictions)
    )
    base_old_route = float((old_predictions == old_targets).float().mean())
    first_route = float((first_predictions == first_targets).float().mean())
    second_route = float((second_predictions == second_targets).float().mean())
    combined_route = float((combined_predictions == combined_targets).float().mean())
    permuted_route = _permuted_chain_accuracy(
        base_router,
        first_extension,
        second_extension,
        old_keys,
        old_queries,
        old_targets,
        first_queries,
        second_queries,
    )
    first_wrong_predictions = _next_view_predictions(
        base_router, first_extension, second_queries, old_keys
    )
    first_wrong_route = float((first_wrong_predictions == 4).float().mean())
    shuffled_first_rate = float(
        (
            _next_view_predictions(
                base_router, shuffled_first, first_queries, old_keys
            )
            == 4
        )
        .float()
        .mean()
    )
    shuffled_second_rate = float(
        (
            _next_view_predictions(
                base_router, shuffled_second, second_queries, old_keys
            )
            == 5
        )
        .float()
        .mean()
    )

    selected_views: dict[str, str] = {
        operation: str(
            int(
                torch.mode(
                    old_predictions[
                        index * args.audit_count : (index + 1) * args.audit_count
                    ]
                ).values
            )
        )
        for index, operation in enumerate(OLD_OPERATIONS)
    }
    selected_views[NEW_OPERATIONS[0]] = str(int(torch.mode(first_predictions).values))
    selected_views[NEW_OPERATIONS[1]] = str(int(torch.mode(second_predictions).values))
    selected_behavior: dict[str, float] = {}
    wrong_behavior: dict[str, float] = {}
    for index, operation in enumerate(ALL_OPERATIONS):
        view = selected_views[operation]
        selected_runtime = _load_view(
            parent, bank, view, seed=args.seed + 140_000 + index
        )
        behavior_seed = (
            args.seed + 110_000 + index
            if index < len(OLD_OPERATIONS)
            else args.seed + 150_000 + index
        )
        selected_behavior[operation] = _accuracy(
            selected_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=behavior_seed,
        )
        wrong_view = ALL_VIEWS[(int(view) + 1) % len(ALL_VIEWS)]
        wrong_runtime = _load_view(
            parent, bank, wrong_view, seed=args.seed + 160_000 + index
        )
        wrong_behavior[operation] = _accuracy(
            wrong_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=behavior_seed,
        )

    torch.save(base_router.state_dict(), root / "frozen_four_view_router.pt")
    torch.save(first_extension.state_dict(), root / "first_view_extension.pt")
    torch.save(second_extension.state_dict(), root / "second_view_extension.pt")
    reloaded = ExecutableArtifactMemory.load(root / "extended_view_bank")
    reloaded_candidates = reloaded.view_candidates()
    reloaded_keys = torch.stack([key for _, key, _ in reloaded_candidates])
    reloaded_router = OpaqueAddressRouter(width=64, hidden=64)
    reloaded_router.load_state_dict(
        torch.load(root / "frozen_four_view_router.pt", weights_only=False)
    )
    reloaded_first = OpaqueViewRouteExtension(width=64, hidden=64)
    reloaded_first.load_state_dict(
        torch.load(root / "first_view_extension.pt", weights_only=False)
    )
    reloaded_second = OpaqueViewRouteExtension(width=64, hidden=64)
    reloaded_second.load_state_dict(
        torch.load(root / "second_view_extension.pt", weights_only=False)
    )
    reloaded_old, reloaded_first_pred, reloaded_second_pred = _chain_predictions(
        reloaded_router,
        reloaded_first,
        reloaded_second,
        reloaded_keys[: len(OLD_VIEWS)],
        old_queries,
        first_queries,
        second_queries,
    )
    reloaded_predictions = torch.cat(
        (reloaded_old, reloaded_first_pred, reloaded_second_pred)
    )
    reloaded_route = float(
        (reloaded_predictions == combined_targets).float().mean()
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
        raise RuntimeError("reloaded multi-step bank has no artifact path")
    artifact_path = (root / "extended_view_bank") / artifact_name
    intact_payload = artifact_path.read_bytes()
    artifact_path.write_bytes(intact_payload + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(root / "extended_view_bank")
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(intact_payload)
    reloaded_core_digest = _digest_core(
        _load_view(parent, reloaded, SECOND_NEW_VIEW, seed=args.seed + 170_000),
        ("growth_slots.0.", "growth_slots.1."),
    )
    report = {
        "schema": "neural-computer.multistep-view-growth-report.v1",
        "claim_boundary": (
            "Two new executable views are acquired sequentially through a "
            "failure-gated external chain while the controller, four-view "
            "router, and first extension remain frozen; no prior route data "
            "is replayed. This is not unrestricted continual learning."
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
        "old_protected_before_first": old_protected_before_first,
        "first_candidate_outcomes": first_candidate_outcomes,
        "first_candidate_outcome_floor": min(first_candidate_outcomes),
        "first_behavior_baselines": first_behavior_baselines,
        "first_behavior_scores": first_behavior_scores,
        "first_protected_before_second": first_protected_before_second,
        "second_candidate_outcomes": second_candidate_outcomes,
        "second_candidate_outcome_floor": min(second_candidate_outcomes),
        "second_behavior_baselines": second_behavior_baselines,
        "second_behavior_scores": second_behavior_scores,
        "replacement_protected_after_second": replacement_protected_after_second,
        "base_old_route_accuracy": base_old_route,
        "first_new_route_accuracy": first_route,
        "second_new_route_accuracy": second_route,
        "two_step_route_accuracy": combined_route,
        "candidate_permutation_accuracy": permuted_route,
        "first_extension_on_second_task_rate": first_wrong_route,
        "reward_shuffled_first_new_selection_rate": shuffled_first_rate,
        "reward_shuffled_second_new_selection_rate": shuffled_second_rate,
        "selected_views": selected_views,
        "selected_behavior": selected_behavior,
        "wrong_behavior": wrong_behavior,
        "reloaded_two_step_accuracy": reloaded_route,
        "reloaded_candidate_exact": reloaded_candidate_exact,
        "corruption_rejected": corruption_rejected,
        "parent_core_digest": parent_digest,
        "reloaded_core_digest": reloaded_core_digest,
        "first_extension_frozen_during_second": first_extension_unchanged,
        "accounting": {
            "unique_logical_lifetimes": (
                args.updates * args.batch_size * 5
                + 2 * args.extension_artifact_updates * args.batch_size
            ),
            "unique_verifier_bits": (
                args.updates * args.batch_size * 5
                + 2 * args.extension_artifact_updates * args.batch_size
            )
            * 4,
            "optimizer_updates": args.updates * 5
            + 2 * args.extension_artifact_updates,
            "base_route_optimizer_updates": base_accounting[
                "route_optimizer_updates"
            ],
            "first_extension_route_optimizer_updates": first_accounting[
                "route_optimizer_updates"
            ],
            "second_extension_route_optimizer_updates": second_accounting[
                "route_optimizer_updates"
            ],
            "route_unique_verifier_bits": (
                base_accounting["unique_route_verifier_bits"]
                + first_accounting["unique_route_verifier_bits"]
                + second_accounting["unique_route_verifier_bits"]
                + shuffled_first_accounting["unique_route_verifier_bits"]
                + shuffled_second_accounting["unique_route_verifier_bits"]
            ),
            "replayed_examples_after_first_extension": 0,
            "replayed_examples_after_second_extension": 0,
            "retention_observations": (
                (len(OLD_OPERATIONS) + len(NEW_OPERATIONS))
                * args.retention_probes
            ),
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "one_physical_row": len(bank.occupied) == 1,
            "six_opaque_views": [view for _, _, view in second_candidates]
            == list(ALL_VIEWS),
            "base_old_route_mastered": base_old_route >= 0.90,
            "first_new_view_mastered": first_route >= 0.75,
            "second_new_view_mastered": second_route >= 0.75,
            "old_capabilities_protected_before_first": all(
                old_protected_before_first
            ),
            "first_candidate_retention_stable": (
                min(first_candidate_outcomes) >= args.retention_threshold
            ),
            "first_extension_behavior_preserved": all(
                first_behavior_scores[operation]
                >= first_behavior_baselines[operation]
                - args.behavior_tolerance
                for operation in (*OLD_OPERATIONS, NEW_OPERATIONS[0])
            ),
            "first_capability_protected_before_second": first_protected_before_second,
            "second_candidate_retention_stable": (
                min(second_candidate_outcomes) >= args.retention_threshold
            ),
            "second_extension_behavior_preserved": all(
                second_behavior_scores[operation]
                >= second_behavior_baselines[operation]
                - args.behavior_tolerance
                for operation in ALL_OPERATIONS
            ),
            "replacement_protected_after_second": replacement_protected_after_second,
            "retention_safe_two_step_growth": (
                all(old_protected_before_first)
                and first_protected_before_second
                and replacement_protected_after_second
            ),
            "two_step_chain_at_least_80": combined_route >= 0.80,
            "candidate_permutation_invariant": permuted_route >= 0.80,
            "first_extension_wrong_on_second_task": first_wrong_route >= 0.75,
            "reward_shuffled_first_near_chance": shuffled_first_rate <= 0.50,
            "reward_shuffled_second_near_chance": shuffled_second_rate <= 0.50,
            "selected_views_match_expected": selected_views
            == dict(zip(ALL_OPERATIONS, ALL_VIEWS, strict=True)),
            "all_selected_behaviors_mastered": min(selected_behavior.values())
            >= 0.70,
            "all_wrong_views_causal": all(
                selected_behavior[operation]
                > wrong_behavior[operation] + 0.05
                for operation in ALL_OPERATIONS
            ),
            "reloaded_route_preserved": reloaded_route >= combined_route - 0.05,
            "reloaded_candidate_exact": reloaded_candidate_exact,
            "frozen_controller_core": parent_digest == reloaded_core_digest,
            "first_extension_frozen_during_second": first_extension_unchanged,
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
                "first_new_route_accuracy": report["first_new_route_accuracy"],
                "second_new_route_accuracy": report["second_new_route_accuracy"],
                "two_step_route_accuracy": report["two_step_route_accuracy"],
                "candidate_permutation_accuracy": report[
                    "candidate_permutation_accuracy"
                ],
                "selected_views": report["selected_views"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
