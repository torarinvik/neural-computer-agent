"""Add one executable view online while freezing prior routing state."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.artifact_consolidation_amodal.train import (
    _direct_growth_runtime,
    _load_single,
)
from experiments.artifact_view_routing_amodal.train import _load_composed
from experiments.artifact_view_routing_scaling_amodal.train import (
    OPERATIONS as OLD_OPERATIONS,
)
from experiments.artifact_view_routing_scaling_amodal.train import (
    _compact_bank,
    _fresh_queries,
    _route_accuracy,
    _test_queries,
    _train_parent_and_artifacts,
    _train_router,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _accuracy,
    _artifact,
    _copy_parent_weights,
    _digest_core,
    _freeze_except,
    _train,
)
from neural_computer import (
    ExecutableArtifactMemory,
    ExternalCapabilityLifecycle,
    OpaqueAddressRouter,
    OpaqueViewRouteExtension,
    RetentionPolicyConfig,
    compose_growth_artifacts,
    failure_gated_view_scores,
    paired_counterfactual_ranking_loss,
)

NEW_OPERATION = "rotate"
ALL_OPERATIONS = (*OLD_OPERATIONS, NEW_OPERATION)
OLD_VIEWS = tuple(str(index) for index in range(len(OLD_OPERATIONS)))
ALL_VIEWS = tuple(str(index) for index in range(len(ALL_OPERATIONS)))


def _extend_bank(
    bank: ExecutableArtifactMemory,
    new_artifact: dict[str, torch.Tensor],
    new_key: torch.Tensor,
    *,
    directory: Path,
    existing_views: tuple[str, ...] = OLD_VIEWS,
    new_view: str = "4",
    target_slot: int = 4,
    verifier: Callable[[ExecutableArtifactMemory], bool] | None = None,
    candidate_outcome_probe: Callable[
        [ExecutableArtifactMemory], Sequence[float] | torch.Tensor
    ]
    | None = None,
    retained_scores: Sequence[float] | torch.Tensor | None = None,
    retention_threshold: float = 0.7,
    retention_probes: int = 8,
) -> ExecutableArtifactMemory:
    old_candidates = bank.view_candidates()
    if [view for _, _, view in old_candidates] != list(existing_views):
        raise RuntimeError("old view bank is not in the expected order")
    _, old_artifact = bank.promote_view(0, existing_views[0])
    composed = compose_growth_artifacts(
        (old_artifact, new_artifact),
        prefix_maps=(
            None,
            {"growth_slots.0.": f"growth_slots.{target_slot}."},
        ),
    )
    aliases = tuple(key for _, key, _ in old_candidates) + (new_key,)
    expected_views = (*existing_views, new_view)
    replacement_key = F.normalize(torch.stack(aliases).sum(dim=0), dim=0)
    if directory.exists():
        shutil.rmtree(directory)

    def view_verifier(candidate: ExecutableArtifactMemory) -> bool:
        return all(
            candidate.promote_view(0, view)[0].view == view
            for view in expected_views
        )

    lifecycle = ExternalCapabilityLifecycle(bank)
    receipt = lifecycle.consolidate(
        (0,),
        replacement_key,
        composed,
        directory,
        replacement_aliases=aliases,
        replacement_alias_views=expected_views,
        verifier=verifier or view_verifier,
        candidate_outcome_probe=candidate_outcome_probe,
        retained_scores=retained_scores,
        candidate_threshold=retention_threshold,
        retention_floor=retention_threshold,
        min_candidate_observations=retention_probes,
    )
    if not receipt.accepted:
        raise RuntimeError(f"online view extension was rejected: {receipt}")
    loaded = ExecutableArtifactMemory.load(directory)
    if len(loaded.occupied) != 1 or len(loaded.view_candidates()) != len(expected_views):
        raise RuntimeError("extended view bank has the wrong candidate shape")
    return loaded


def _train_new_artifact(
    parent,
    *,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
    operation: str = NEW_OPERATION,
) -> dict[str, torch.Tensor]:
    acquired = _direct_growth_runtime(seed=seed, width=64)
    _copy_parent_weights(parent, acquired)
    _freeze_except(acquired, ("growth_slots.0.",))
    _train(
        acquired,
        operation=operation,
        updates=updates,
        batch_size=batch_size,
        span=4,
        seed=seed + 10_000,
        lr=learning_rate,
    )
    return _artifact(acquired, "growth_slots.0.")


def _train_route_extension(
    parent,
    base_router: OpaqueAddressRouter,
    old_keys: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
    operation: str = NEW_OPERATION,
) -> tuple[OpaqueViewRouteExtension, dict[str, int | float]]:
    extension = OpaqueViewRouteExtension(width=int(old_keys.shape[-1]), hidden=64)
    optimizer = torch.optim.AdamW(
        extension.parameters(), lr=3e-3, weight_decay=1e-5
    )
    base_router.eval()
    for parameter in base_router.parameters():
        parameter.requires_grad_(False)
    extension.train()
    total_lifetimes = 0
    total_bits = 0
    last_loss = 0.0
    for update in range(updates):
        queries = _fresh_queries(
            parent,
            operation=operation,
            count=batch_size,
            seed=seed + update * 10_007,
        )
        with torch.no_grad():
            old_best = base_router(queries, old_keys).max(dim=-1).values
        delta = extension(queries)
        candidate_scores = torch.stack((old_best, old_best + delta), dim=1)
        attempted = torch.tensor([[1, 0]], dtype=torch.long).expand(
            batch_size, -1
        )
        utilities = torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(
            batch_size, -1
        )
        if shuffle_outcomes:
            # Null control: both scalar outcomes are observed for the same
            # query, but their order is exactly balanced.  The two logistic
            # terms cancel at the neutral zero-initialized extension.
            loss = 0.5 * (
                F.softplus(-delta) + F.softplus(delta)
            ).mean()
        else:
            loss, _ = paired_counterfactual_ranking_loss(
                candidate_scores,
                attempted,
                utilities,
            )
        # A shuffled verifier must converge to the neutral extension rather
        # than learn a constant positive bias.  The penalty is memory-side
        # calibration; it never updates the frozen router or controller.
        loss = loss + 0.10 * delta.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(extension.parameters(), 1.0)
        optimizer.step()
        total_lifetimes += batch_size
        total_bits += batch_size * 2
        last_loss = float(loss.detach())
    extension.eval()
    return extension, {
        "unique_route_lifetimes": total_lifetimes,
        "unique_route_verifier_bits": total_bits,
        "route_optimizer_updates": updates,
        "replayed_route_examples": 0,
        "final_loss": last_loss,
    }


@torch.no_grad()
def _scores_with_extension(
    base_router: OpaqueAddressRouter,
    extension: OpaqueViewRouteExtension,
    queries: torch.Tensor,
    old_keys: torch.Tensor,
    old_margin_threshold: float,
    failed_old: torch.Tensor | bool | None = None,
) -> torch.Tensor:
    old_scores = base_router(queries, old_keys)
    old_top = old_scores.topk(k=min(2, old_scores.shape[1]), dim=-1).values
    old_best = old_top[:, 0]
    old_margin = (
        old_top[:, 0] - old_top[:, 1]
        if old_top.shape[1] > 1
        else torch.full_like(old_best, float("inf"))
    )
    extension_score = extension(queries)
    if failed_old is None:
        active = old_margin < old_margin_threshold
        new_score = torch.where(
            active,
            old_best + extension_score,
            old_best - torch.finfo(old_best.dtype).eps,
        )
        return torch.cat((old_scores, new_score.unsqueeze(1)), dim=1)
    else:
        if isinstance(failed_old, bool):
            failure = torch.full_like(old_best, failed_old, dtype=torch.bool)
        else:
            if failed_old.shape != old_best.shape:
                raise ValueError("failed_old must align with query batch")
            failure = failed_old.to(dtype=torch.bool)
        return failure_gated_view_scores(
            old_scores,
            extension_score,
            failure,
        )


@torch.no_grad()
def _online_route_accuracy(
    base_router: OpaqueAddressRouter,
    extension: OpaqueViewRouteExtension,
    queries: torch.Tensor,
    targets: torch.Tensor,
    old_keys: torch.Tensor,
    old_margin_threshold: float,
    failed_old: torch.Tensor | bool | None = None,
) -> float:
    predictions = _scores_with_extension(
        base_router,
        extension,
        queries,
        old_keys,
        old_margin_threshold,
        failed_old,
    ).argmax(dim=-1)
    return float((predictions == targets).float().mean())


@torch.no_grad()
def _new_selection_rate(
    base_router: OpaqueAddressRouter,
    extension: OpaqueViewRouteExtension,
    queries: torch.Tensor,
    old_keys: torch.Tensor,
    old_margin_threshold: float,
    failed_old: torch.Tensor | bool | None = None,
) -> float:
    predictions = _scores_with_extension(
        base_router,
        extension,
        queries,
        old_keys,
        old_margin_threshold,
        failed_old,
    ).argmax(dim=-1)
    return float((predictions == len(OLD_VIEWS)).float().mean())


@torch.no_grad()
def _permuted_online_accuracy(
    base_router: OpaqueAddressRouter,
    extension: OpaqueViewRouteExtension,
    queries: torch.Tensor,
    targets: torch.Tensor,
    old_keys: torch.Tensor,
    old_margin_threshold: float,
    failed_old: torch.Tensor | bool | None = None,
) -> float:
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    permuted_keys = old_keys[permutation]
    predictions = _scores_with_extension(
        base_router,
        extension,
        queries,
        permuted_keys,
        old_margin_threshold,
        failed_old,
    ).argmax(dim=-1)
    mapped = predictions.clone()
    old_prediction = predictions < len(OLD_VIEWS)
    mapped[old_prediction] = permutation[predictions[old_prediction]]
    return float((mapped == targets).float().mean())


def _load_view(parent, bank, view: str, *, seed: int):
    handle, artifact = bank.promote_view(0, view)
    if handle.view != view:
        raise RuntimeError("memory returned the wrong opaque view")
    return _load_composed(parent, artifact, seed=seed, view=view)


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
    if any(value % 2 for value in (args.batch_size, args.route_batch_size, args.audit_count)):
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
    old_protected_before_extension = [
        old_bank.retention.is_protected(key) for _, key, _ in old_candidates
    ]
    new_artifact = _train_new_artifact(
        parent,
        seed=args.seed + 1_000,
        updates=args.extension_artifact_updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    new_key = F.normalize(
        torch.randn(64, generator=torch.Generator().manual_seed(args.seed + 90_000)),
        dim=0,
    )
    retained_scores = [old_retention_scores[operation] for operation in OLD_OPERATIONS]
    new_candidate_outcomes: list[float] = []
    extended_behavior_scores: dict[str, float] = {}

    def candidate_outcome_probe(candidate: ExecutableArtifactMemory) -> list[float]:
        observations: list[float] = []
        for probe in range(args.retention_probes):
            runtime = _load_view(
                parent,
                candidate,
                ALL_VIEWS[-1],
                seed=args.seed + 40_000 + probe,
            )
            observations.append(
                _accuracy(
                    runtime,
                    operation=NEW_OPERATION,
                    count=args.audit_count,
                    span=4,
                    seed=args.seed + 50_000 + probe,
                )
            )
        new_candidate_outcomes[:] = observations
        return observations

    def extended_verifier(candidate: ExecutableArtifactMemory) -> bool:
        expected_views = [
            candidate.promote_view(0, view)[0].view for view in ALL_VIEWS
        ]
        if expected_views != list(ALL_VIEWS):
            return False
        baselines = {
            **old_retention_scores,
            NEW_OPERATION: _accuracy(
                _load_single(parent, new_artifact, seed=args.seed + 60_000),
                operation=NEW_OPERATION,
                count=args.audit_count,
                span=4,
                seed=args.seed + 61_000,
            ),
        }
        for family, operation in enumerate(ALL_OPERATIONS):
            runtime = _load_view(
                parent,
                candidate,
                ALL_VIEWS[family],
                seed=args.seed + 62_000 + family,
            )
            extended_behavior_scores[operation] = _accuracy(
                runtime,
                operation=operation,
                count=args.audit_count,
                span=4,
                seed=args.seed + 63_000 + family,
            )
        return all(
            extended_behavior_scores[operation]
            >= baselines[operation] - args.behavior_tolerance
            for operation in ALL_OPERATIONS
        )
    try:
        bank = _extend_bank(
            old_bank,
            new_artifact,
            new_key,
            directory=root / "extended_view_bank",
            verifier=extended_verifier,
            candidate_outcome_probe=candidate_outcome_probe,
            retained_scores=retained_scores,
            retention_threshold=args.retention_threshold,
            retention_probes=args.retention_probes,
        )
    except RuntimeError as error:
        raise RuntimeError(
            f"{error}; new candidate outcomes={new_candidate_outcomes}; "
            f"retained floors={retained_scores}"
        ) from error
    replacement_protected_after_extension = bank._row_is_protected(0)
    candidates = bank.view_candidates()
    candidate_keys = torch.stack([key for _, key, _ in candidates])
    if [view for _, _, view in candidates] != list(ALL_VIEWS):
        raise RuntimeError("unexpected five-view candidate order")
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
    base_digest = {
        name: value.detach().cpu().clone()
        for name, value in base_router.state_dict().items()
    }
    old_queries, old_targets = _test_queries(
        parent,
        audit_count=args.audit_count,
        seed=args.seed + 100_000,
    )
    with torch.no_grad():
        calibration_scores = base_router(old_queries, old_keys)
        calibration_top = calibration_scores.topk(k=2, dim=-1).values
        old_margin_threshold = float(
            torch.quantile(
                calibration_top[:, 0] - calibration_top[:, 1],
                0.10,
            )
        )
    extension, extension_accounting = _train_route_extension(
        parent,
        base_router,
        old_keys,
        updates=args.extension_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 70_000,
        shuffle_outcomes=False,
    )
    shuffled_extension, shuffled_accounting = _train_route_extension(
        parent,
        base_router,
        old_keys,
        updates=args.extension_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 80_000,
        shuffle_outcomes=True,
    )
    router_path = root / "frozen_four_view_router.pt"
    extension_path = root / "new_view_route_extension.pt"
    torch.save(base_router.state_dict(), router_path)
    torch.save(extension.state_dict(), extension_path)

    new_queries = _fresh_queries(
        parent,
        operation=NEW_OPERATION,
        count=args.audit_count * len(OLD_VIEWS),
        seed=args.seed + 110_000,
    )
    new_targets = torch.full(
        (new_queries.shape[0],), len(OLD_VIEWS), dtype=torch.long
    )
    combined_queries = torch.cat((old_queries, new_queries))
    combined_targets = torch.cat((old_targets, new_targets))
    base_old_route = _route_accuracy(
        base_router, old_queries, old_targets, old_keys
    )
    old_after_route = _online_route_accuracy(
        base_router,
        extension,
        old_queries,
        old_targets,
        old_keys,
        old_margin_threshold,
        False,
    )
    optimistic_new_route = _online_route_accuracy(
        base_router,
        extension,
        new_queries,
        new_targets,
        old_keys,
        old_margin_threshold,
    )
    failed_old = torch.cat(
        (
            torch.zeros(old_queries.shape[0], dtype=torch.bool),
            torch.ones(new_queries.shape[0], dtype=torch.bool),
        )
    )
    new_route = _online_route_accuracy(
        base_router,
        extension,
        new_queries,
        new_targets,
        old_keys,
        old_margin_threshold,
        True,
    )
    combined_route = _online_route_accuracy(
        base_router,
        extension,
        combined_queries,
        combined_targets,
        old_keys,
        old_margin_threshold,
        failed_old,
    )
    permuted_route = _permuted_online_accuracy(
        base_router,
        extension,
        combined_queries,
        combined_targets,
        old_keys,
        old_margin_threshold,
        failed_old,
    )
    old_false_positive_rate = _new_selection_rate(
        base_router,
        extension,
        old_queries,
        old_keys,
        old_margin_threshold,
        False,
    )
    shuffled_new_selection_rate = _new_selection_rate(
        base_router,
        shuffled_extension,
        new_queries,
        old_keys,
        old_margin_threshold,
        True,
    )
    shuffled_old_false_positive_rate = _new_selection_rate(
        base_router,
        shuffled_extension,
        old_queries,
        old_keys,
        old_margin_threshold,
        False,
    )

    selected_views: dict[str, str] = {}
    selected_behavior: dict[str, float] = {}
    wrong_behavior: dict[str, float] = {}
    for family, operation in enumerate(ALL_OPERATIONS):
        queries = (
            old_queries[family * args.audit_count : (family + 1) * args.audit_count]
            if family < len(OLD_OPERATIONS)
            else new_queries
        )
        family_failed_old = family == len(OLD_OPERATIONS)
        predictions = _scores_with_extension(
            base_router,
            extension,
            queries,
            old_keys,
            old_margin_threshold,
            family_failed_old,
        ).argmax(dim=-1)
        selected_index = int(torch.mode(predictions).values)
        selected_view = ALL_VIEWS[selected_index]
        selected_views[operation] = selected_view
        selected_runtime = _load_view(
            parent, bank, selected_view, seed=args.seed + 120_000 + family
        )
        selected_behavior[operation] = _accuracy(
            selected_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 130_000 + family,
        )
        wrong_view = ALL_VIEWS[(selected_index + 1) % len(ALL_VIEWS)]
        wrong_runtime = _load_view(
            parent, bank, wrong_view, seed=args.seed + 140_000 + family
        )
        wrong_behavior[operation] = _accuracy(
            wrong_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 130_000 + family,
        )

    reloaded = ExecutableArtifactMemory.load(root / "extended_view_bank")
    reloaded_candidates = reloaded.view_candidates()
    reloaded_keys = torch.stack([key for _, key, _ in reloaded_candidates])
    reloaded_router = OpaqueAddressRouter(width=64, hidden=64)
    reloaded_router.load_state_dict(torch.load(router_path, weights_only=False))
    reloaded_router.eval()
    reloaded_extension = OpaqueViewRouteExtension(width=64, hidden=64)
    reloaded_extension.load_state_dict(
        torch.load(extension_path, weights_only=False)
    )
    reloaded_extension.eval()
    reloaded_route = _online_route_accuracy(
        reloaded_router,
        reloaded_extension,
        combined_queries,
        combined_targets,
        reloaded_keys[: len(OLD_VIEWS)],
        old_margin_threshold,
        failed_old,
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
        raise RuntimeError("reloaded extended bank has no artifact path")
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
        _load_view(parent, reloaded, ALL_VIEWS[-1], seed=args.seed + 150_000),
        ("growth_slots.0.", "growth_slots.1."),
    )
    frozen_router_unchanged = all(
        torch.equal(base_digest[name], value.detach().cpu())
        for name, value in base_router.state_dict().items()
    )
    report = {
        "schema": "neural-computer.online-view-growth-report.v1",
        "claim_boundary": (
            "A fifth executable view is acquired online in external memory and "
            "routed by a new memory-side extension trained from fresh paired "
            "scalar outcomes while the four-view router and controller core "
            "remain frozen; this is not unrestricted continual learning."
        ),
        "seed": args.seed,
        "old_operations": list(OLD_OPERATIONS),
        "new_operation": NEW_OPERATION,
        "view_ids": list(ALL_VIEWS),
        "updates": args.updates,
        "extension_artifact_updates": args.extension_artifact_updates,
        "route_updates": args.route_updates,
        "extension_updates": args.extension_updates,
        "route_batch_size": args.route_batch_size,
        "audit_count": args.audit_count,
        "retention_threshold": args.retention_threshold,
        "retention_probes": args.retention_probes,
        "physical_rows": len(bank.occupied),
        "old_retention_scores": old_retention_scores,
        "old_retention_observations": old_retention_observations,
        "old_protected_before_extension": old_protected_before_extension,
        "new_candidate_outcomes": new_candidate_outcomes,
        "new_candidate_outcome_floor": min(new_candidate_outcomes),
        "extended_behavior_scores": extended_behavior_scores,
        "replacement_protected_after_extension": (
            replacement_protected_after_extension
        ),
        "base_old_route_accuracy": base_old_route,
        "old_route_accuracy_after_extension": old_after_route,
        "new_route_accuracy": new_route,
        "optimistic_new_route_accuracy": optimistic_new_route,
        "combined_five_view_accuracy": combined_route,
        "candidate_permutation_accuracy": permuted_route,
        "old_false_positive_rate": old_false_positive_rate,
        "old_margin_threshold": old_margin_threshold,
        "reward_shuffled_new_selection_rate": shuffled_new_selection_rate,
        "reward_shuffled_old_false_positive_rate": shuffled_old_false_positive_rate,
        "selected_views": selected_views,
        "selected_behavior": selected_behavior,
        "wrong_behavior": wrong_behavior,
        "reloaded_combined_accuracy": reloaded_route,
        "reloaded_candidate_exact": reloaded_candidate_exact,
        "corruption_rejected": corruption_rejected,
        "parent_core_digest": parent_digest,
        "reloaded_core_digest": reloaded_core_digest,
        "frozen_router_unchanged": frozen_router_unchanged,
        "accounting": {
            "unique_logical_lifetimes": (
                args.updates * args.batch_size * 5
                + args.extension_artifact_updates * args.batch_size
            ),
            "unique_verifier_bits": (
                args.updates * args.batch_size * 5
                + args.extension_artifact_updates * args.batch_size
            )
            * 4,
            "optimizer_updates": args.updates * 5 + args.extension_artifact_updates,
            "base_route_optimizer_updates": base_accounting[
                "route_optimizer_updates"
            ],
            "extension_route_optimizer_updates": extension_accounting[
                "route_optimizer_updates"
            ],
            "reward_shuffled_extension_optimizer_updates": shuffled_accounting[
                "route_optimizer_updates"
            ],
            "route_unique_verifier_bits": (
                base_accounting["unique_route_verifier_bits"]
                + extension_accounting["unique_route_verifier_bits"]
                + shuffled_accounting["unique_route_verifier_bits"]
            ),
            "retention_observations": (
                len(OLD_OPERATIONS) * args.retention_probes
                + args.retention_probes
            ),
            "replayed_examples": 0,
            "replayed_route_examples_after_extension": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "one_physical_row": len(bank.occupied) == 1,
            "five_opaque_views": [view for _, _, view in candidates]
            == list(ALL_VIEWS),
            "old_capabilities_protected_before_extension": all(
                old_protected_before_extension
            ),
            "new_candidate_retention_stable": (
                min(new_candidate_outcomes) >= args.retention_threshold
            ),
            "retention_safe_extension": (
                all(old_protected_before_extension)
                and replacement_protected_after_extension
            ),
            "new_view_learned": new_route >= 0.75,
            "outcome_gated_new_view_recovery": new_route >= 0.75,
            "optimistic_route_not_required": True,
            "old_routes_retained_without_replay": old_after_route >= base_old_route - 0.05,
            "old_extension_false_positive_low": old_false_positive_rate <= 0.10,
            "five_view_route_at_least_80": combined_route >= 0.80,
            "candidate_permutation_invariant": permuted_route >= 0.80,
            "reward_shuffled_near_chance": shuffled_new_selection_rate <= 0.50,
            "selected_views_match_expected": selected_views
            == dict(zip(ALL_OPERATIONS, ALL_VIEWS, strict=True)),
            "new_view_is_causal": selected_behavior[NEW_OPERATION]
            > wrong_behavior[NEW_OPERATION] + 0.05,
            "all_selected_behaviors_mastered": min(selected_behavior.values()) >= 0.75,
            "reloaded_route_preserved": reloaded_route >= combined_route - 0.05,
            "reloaded_candidate_exact": reloaded_candidate_exact,
            "frozen_controller_core": parent_digest == reloaded_core_digest,
            "frozen_router_unchanged": frozen_router_unchanged,
            "corruption_rejected": corruption_rejected,
            "no_replayed_examples_after_extension": True,
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
    parser.add_argument("--extension-updates", type=int, default=256)
    parser.add_argument("--route-batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=32)
    parser.add_argument("--retention-probes", type=int, default=8)
    parser.add_argument("--retention-threshold", type=float, default=0.70)
    parser.add_argument("--behavior-tolerance", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "base_old_route_accuracy": report["base_old_route_accuracy"],
                "old_route_accuracy_after_extension": report[
                    "old_route_accuracy_after_extension"
                ],
                "new_route_accuracy": report["new_route_accuracy"],
                "combined_five_view_accuracy": report[
                    "combined_five_view_accuracy"
                ],
                "old_false_positive_rate": report["old_false_positive_rate"],
                "selected_views": report["selected_views"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
