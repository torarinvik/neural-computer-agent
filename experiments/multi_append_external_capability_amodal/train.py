"""Audit two sequential protected appends of external learned programs."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    PROGRAMS,
    _artifact,
    _capability_accuracy,
    _load_artifact,
    _new_capability,
    _route_queries,
    _stable_bits,
    _test_queries,
    _train_capability,
)
from experiments.sequential_external_capability_amodal.train import (
    _digest_module,
    _retention_probe,
    _train_base_router,
    _train_route_extension,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
    _runtime,
)
from neural_computer import (
    ExecutableArtifactMemory,
    ExternalCapabilityLifecycle,
    OpaqueAddressRouter,
    OpaqueViewRouteExtension,
    PersistentOpaqueStateStore,
    RetentionPolicyConfig,
    failure_gated_view_scores,
    paired_counterfactual_ranking_loss,
)

FIRST_APPEND = ("rotate4", "rotate", 4)
SECOND_APPEND = ("adjacent_xor4", "adjacent_xor", 4)
ALL_PROGRAMS = (*PROGRAMS, FIRST_APPEND, SECOND_APPEND)
ROUTE_WIDTH = 48


def _train_second_extension(
    parent,
    base_scores: Callable[[torch.Tensor], torch.Tensor],
    *,
    operation: str,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
) -> tuple[OpaqueViewRouteExtension, dict[str, int | float]]:
    """Train one new route against a frozen cascade of established routes."""

    extension = OpaqueViewRouteExtension(width=ROUTE_WIDTH, hidden=64)
    optimizer = torch.optim.AdamW(extension.parameters(), lr=3e-3, weight_decay=1e-5)
    for update in range(updates):
        query_count = batch_size // 2 if shuffle_outcomes else batch_size
        queries = _route_queries(
            parent,
            operation=operation,
            span=4,
            count=query_count,
            seed=seed + update * 10_007,
        )
        if shuffle_outcomes:
            queries = queries.repeat_interleave(2, dim=0)
        with torch.no_grad():
            established = base_scores(queries)
            established_best = established.max(dim=-1).values
        delta = extension(queries)
        candidate_scores = torch.stack(
            (established_best, established_best + delta), dim=1
        )
        attempted = torch.tensor([[1, 0]], dtype=torch.long).expand(batch_size, -1)
        if shuffle_outcomes:
            utilities = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
            ).repeat(query_count, 1)
        else:
            utilities = torch.tensor([[1.0, 0.0]]).expand(batch_size, -1)
        loss, _ = paired_counterfactual_ranking_loss(
            candidate_scores, attempted, utilities
        )
        loss = loss + 0.10 * delta.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(extension.parameters(), 1.0)
        optimizer.step()
    extension.eval()
    return extension, {
        "unique_route_lifetimes": updates * batch_size,
        "unique_route_verifier_bits": updates * batch_size * 2,
        "route_optimizer_updates": updates,
        "replayed_route_examples": 0,
    }


def _two_append_scores(
    base_router: OpaqueAddressRouter,
    first_extension: OpaqueViewRouteExtension,
    second_extension: OpaqueViewRouteExtension,
    queries: torch.Tensor,
    old_keys: torch.Tensor,
    *,
    first_failed: bool,
    second_failed: bool,
) -> torch.Tensor:
    established = failure_gated_view_scores(
        base_router(queries, old_keys),
        first_extension(queries),
        first_failed,
    )
    return failure_gated_view_scores(
        established,
        second_extension(queries),
        second_failed,
    )


def _route_rates(
    base_router: OpaqueAddressRouter,
    first_extension: OpaqueViewRouteExtension,
    second_extension: OpaqueViewRouteExtension,
    parent,
    old_keys: torch.Tensor,
    *,
    audit_count: int,
    seed: int,
) -> dict[str, float]:
    old_queries, old_targets = _test_queries(parent, count=audit_count, seed=seed)
    first_queries = _route_queries(
        parent,
        operation=FIRST_APPEND[1],
        span=4,
        count=audit_count,
        seed=seed + 10_001,
    )
    second_queries = _route_queries(
        parent,
        operation=SECOND_APPEND[1],
        span=4,
        count=audit_count,
        seed=seed + 20_002,
    )
    old_scores = _two_append_scores(
        base_router,
        first_extension,
        second_extension,
        old_queries,
        old_keys,
        first_failed=False,
        second_failed=False,
    )
    first_scores = _two_append_scores(
        base_router,
        first_extension,
        second_extension,
        first_queries,
        old_keys,
        first_failed=True,
        second_failed=False,
    )
    second_scores = _two_append_scores(
        base_router,
        first_extension,
        second_extension,
        second_queries,
        old_keys,
        first_failed=True,
        second_failed=True,
    )
    old_predictions = old_scores.argmax(dim=-1)
    first_predictions = first_scores.argmax(dim=-1)
    second_predictions = second_scores.argmax(dim=-1)
    return {
        "old": float((old_predictions == old_targets).float().mean()),
        "first": float((first_predictions == len(PROGRAMS)).float().mean()),
        "second": float((second_predictions == len(PROGRAMS) + 1).float().mean()),
        "combined": float(
            torch.cat(
                (
                    (old_predictions == old_targets).float(),
                    (first_predictions == len(PROGRAMS)).float(),
                    (second_predictions == len(PROGRAMS) + 1).float(),
                )
            ).mean()
        ),
    }


def _permuted_combined_accuracy(
    base_router: OpaqueAddressRouter,
    first_extension: OpaqueViewRouteExtension,
    second_extension: OpaqueViewRouteExtension,
    parent,
    old_keys: torch.Tensor,
    *,
    audit_count: int,
    seed: int,
) -> float:
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted_keys = old_keys[permutation]
    old_queries, old_targets = _test_queries(parent, count=audit_count, seed=seed)
    old_predictions = _two_append_scores(
        base_router,
        first_extension,
        second_extension,
        old_queries,
        permuted_keys,
        first_failed=False,
        second_failed=False,
    ).argmax(dim=-1)
    old_predictions = permutation[old_predictions]
    first_queries = _route_queries(
        parent,
        operation=FIRST_APPEND[1],
        span=4,
        count=audit_count,
        seed=seed + 10_001,
    )
    second_queries = _route_queries(
        parent,
        operation=SECOND_APPEND[1],
        span=4,
        count=audit_count,
        seed=seed + 20_002,
    )
    first_predictions = _two_append_scores(
        base_router,
        first_extension,
        second_extension,
        first_queries,
        permuted_keys,
        first_failed=True,
        second_failed=False,
    ).argmax(dim=-1)
    second_predictions = _two_append_scores(
        base_router,
        first_extension,
        second_extension,
        second_queries,
        permuted_keys,
        first_failed=True,
        second_failed=True,
    ).argmax(dim=-1)
    correct = torch.cat(
        (
            old_predictions == old_targets,
            first_predictions == len(PROGRAMS),
            second_predictions == len(PROGRAMS) + 1,
        )
    )
    return float(correct.float().mean())


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if (
        min(
            args.parent_updates,
            args.updates,
            args.append_updates,
            args.route_updates,
            args.extension_updates,
            args.batch_size,
            args.route_batch_size,
            args.audit_count,
            args.retention_probes,
        )
        < 1
    ):
        raise ValueError("all update and audit counts must be positive")
    for name, value in (
        ("batch_size", args.batch_size),
        ("route_batch_size", args.route_batch_size),
        ("audit_count", args.audit_count),
    ):
        if value % 2:
            raise ValueError(f"{name} must be even")
    if args.route_batch_size < 4:
        raise ValueError("route_batch_size must be at least four for shuffled arms")

    root = args.report_out.parent
    root.mkdir(parents=True, exist_ok=True)
    parent = _runtime(seed=args.seed, growth=False)
    parent_history, parent_progress = _train_with_progress(
        parent,
        operation="forward",
        updates=args.parent_updates,
        batch_size=args.batch_size,
        span=2,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        credit_mode="sampled",
    )
    parent.eval()
    parent_digest_before = _digest_core(parent, ())
    parent_stable_bits = _stable_bits(
        parent_progress,
        threshold=0.75,
        bits_per_update=args.batch_size * 2,
    )

    bank_path = root / "base_bank"
    first_path = root / "first_grown_bank"
    second_path = root / "second_grown_bank"
    for path in (bank_path, first_path, second_path):
        if path.exists():
            shutil.rmtree(path)
    bank = ExecutableArtifactMemory(
        bank_path,
        width=ROUTE_WIDTH,
        capacity=len(PROGRAMS),
        write_match_threshold=0.99999,
    )
    bank.retention.config = RetentionPolicyConfig(
        mastery_threshold=args.retention_threshold,
        min_mastery_observations=args.retention_probes,
    )
    lifecycle = ExternalCapabilityLifecycle(bank)
    histories: dict[str, object] = {
        "parent": {"history": parent_history, "progress": parent_progress}
    }
    stable_bits: dict[str, int | None] = {}
    route_keys: list[torch.Tensor] = []
    for index, (label, operation, span) in enumerate(PROGRAMS):
        program, decoder = _new_capability(args.seed + index + 1)
        history, progress = _train_capability(
            parent,
            program,
            decoder,
            operation=operation,
            span=span,
            updates=args.updates,
            batch_size=args.batch_size,
            seed=args.seed + 200 * (index + 1),
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
        )
        key = F.normalize(
            _route_queries(
                parent,
                operation=operation,
                span=span,
                count=args.audit_count,
                seed=args.seed + 50_000 + index,
            ).mean(dim=0),
            dim=0,
        )
        receipt = lifecycle.admit(key, _artifact(program, decoder))
        if not receipt.accepted:
            raise RuntimeError(f"base admission failed: {receipt.reason}")
        route_keys.append(key)
        stable_bits[label] = _stable_bits(
            progress,
            threshold=0.75,
            bits_per_update=args.batch_size * span,
        )
        histories[label] = {
            "history": history,
            "progress": progress,
            "stable_bits_to_threshold": stable_bits[label],
        }

    for index, (_label, operation, _span) in enumerate(PROGRAMS):
        _retention_probe(
            parent,
            bank,
            route_keys[index],
            row=index,
            operation=operation,
            audit_count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 30_000 + index * 100,
        )
    bank.save()
    protected_base = all(bank.retention.is_protected(key) for key in route_keys)
    old_keys = torch.stack([key for _, key in bank.address_rows()])
    base_result = _train_base_router(
        parent,
        old_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 60_000,
    )
    base_router = base_result["router"]
    base_digest = _digest_module(base_router)

    append_specs = (FIRST_APPEND, SECOND_APPEND)
    extensions: list[OpaqueViewRouteExtension] = []
    extension_accounting: list[dict[str, int | float]] = []
    append_keys: list[torch.Tensor] = []
    for append_index, (label, operation, span) in enumerate(append_specs):
        program, decoder = _new_capability(args.seed + 10_001 + append_index)
        history, progress = _train_capability(
            parent,
            program,
            decoder,
            operation=operation,
            span=span,
            updates=args.append_updates,
            batch_size=args.batch_size,
            seed=args.seed + 70_000 + append_index * 1000,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
        )
        key = F.normalize(
            _route_queries(
                parent,
                operation=operation,
                span=span,
                count=args.audit_count,
                seed=args.seed + 80_000 + append_index,
            ).mean(dim=0),
            dim=0,
        )
        artifact = _artifact(program, decoder)
        plan = lifecycle.plan_admission(key, artifact)
        if plan.action != "grow" or not all(lifecycle.protection_mask().tolist()):
            raise RuntimeError("protected append did not select transactional growth")
        destination = first_path if append_index == 0 else second_path
        receipt = lifecycle.admit(
            key, artifact, plan=plan, grow_destination=destination
        )
        if not receipt.accepted:
            raise RuntimeError(f"append admission failed: {receipt.reason}")
        grown = lifecycle.memory
        grown.retention.config = RetentionPolicyConfig(
            mastery_threshold=args.retention_threshold,
            min_mastery_observations=args.retention_probes,
        )
        row = len(PROGRAMS) + append_index
        observations = _retention_probe(
            parent,
            grown,
            key,
            row=row,
            operation=operation,
            audit_count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 90_000 + append_index * 100,
        )
        grown.save()
        append_keys.append(key)
        stable_bits[label] = _stable_bits(
            progress,
            threshold=0.75,
            bits_per_update=args.batch_size * span,
        )
        histories[label] = {
            "history": history,
            "progress": progress,
            "stable_bits_to_threshold": stable_bits[label],
            "retention_observations": observations,
        }
        if append_index == 0:
            extension, accounting = _train_route_extension(
                parent,
                base_router,
                old_keys,
                operation=operation,
                updates=args.extension_updates,
                batch_size=args.route_batch_size,
                seed=args.seed + 100_000,
                shuffle_outcomes=False,
            )
            shuffled_extension, shuffled_accounting = _train_route_extension(
                parent,
                base_router,
                old_keys,
                operation=operation,
                updates=args.extension_updates,
                batch_size=args.route_batch_size,
                seed=args.seed + 110_000,
                shuffle_outcomes=True,
            )
            extensions.append(extension)
            extension_accounting.extend((accounting, shuffled_accounting))
            shuffled_first = shuffled_extension
        else:
            first_extension = extensions[0]

            def established_scores(
                queries: torch.Tensor,
                first=first_extension,
            ) -> torch.Tensor:
                return failure_gated_view_scores(
                    base_router(queries, old_keys),
                    first(queries),
                    True,
                )

            extension, accounting = _train_second_extension(
                parent,
                established_scores,
                operation=operation,
                updates=args.extension_updates,
                batch_size=args.route_batch_size,
                seed=args.seed + 120_000,
                shuffle_outcomes=False,
            )
            shuffled_second, shuffled_accounting = _train_second_extension(
                parent,
                established_scores,
                operation=operation,
                updates=args.extension_updates,
                batch_size=args.route_batch_size,
                seed=args.seed + 130_000,
                shuffle_outcomes=True,
            )
            extensions.append(extension)
            extension_accounting.extend((accounting, shuffled_accounting))
            shuffled_second_extension = shuffled_second

    first_extension, second_extension = extensions
    base_digest_after = _digest_module(base_router)
    rates = _route_rates(
        base_router,
        first_extension,
        second_extension,
        parent,
        old_keys,
        audit_count=args.audit_count,
        seed=args.seed + 140_000,
    )
    permuted_accuracy = _permuted_combined_accuracy(
        base_router,
        first_extension,
        second_extension,
        parent,
        old_keys,
        audit_count=args.audit_count,
        seed=args.seed + 140_000,
    )
    first_null = _route_rates(
        base_router,
        shuffled_first,
        second_extension,
        parent,
        old_keys,
        audit_count=args.audit_count,
        seed=args.seed + 150_000,
    )["first"]
    second_null = _route_rates(
        base_router,
        first_extension,
        shuffled_second_extension,
        parent,
        old_keys,
        audit_count=args.audit_count,
        seed=args.seed + 160_000,
    )["second"]

    selected_rows = {
        "rotate4": len(PROGRAMS),
        "adjacent_xor4": len(PROGRAMS) + 1,
    }
    old_queries, _old_targets = _test_queries(
        parent, count=args.audit_count, seed=args.seed + 170_000
    )
    for family, (label, _operation, _span) in enumerate(PROGRAMS):
        selected_rows[label] = int(
            base_router(
                old_queries[
                    family * args.audit_count : (family + 1) * args.audit_count
                ],
                old_keys,
            )
            .argmax(dim=-1)
            .mode()
            .values
        )
    selected_behavior: dict[str, float] = {}
    wrong_behavior: dict[str, float] = {}
    for row, (label, operation, span) in enumerate(ALL_PROGRAMS):
        _, artifact = lifecycle.memory.promote_index(selected_rows[label])
        program, decoder = _load_artifact(artifact)
        selected_behavior[label] = _capability_accuracy(
            parent,
            program,
            decoder,
            operation=operation,
            span=span,
            count=args.audit_count,
            seed=args.seed + 180_000 + row,
        )
        wrong_row = (selected_rows[label] + 1) % len(ALL_PROGRAMS)
        _, wrong_artifact = lifecycle.memory.promote_index(wrong_row)
        wrong_program, wrong_decoder = _load_artifact(wrong_artifact)
        wrong_behavior[label] = _capability_accuracy(
            parent,
            wrong_program,
            wrong_decoder,
            operation=operation,
            span=span,
            count=args.audit_count,
            seed=args.seed + 180_000 + row,
        )

    base_store = PersistentOpaqueStateStore(
        root / "base_router.pt",
        configuration={
            "component": "multi-append-base-router",
            "schema": "neural-computer.opaque-address-router.v1",
            "width": ROUTE_WIDTH,
            "hidden": 64,
            "candidate_count": len(PROGRAMS),
        },
    )
    first_store = PersistentOpaqueStateStore(
        root / "first_extension.pt",
        configuration={
            "component": "multi-append-first-extension",
            "schema": "neural-computer.opaque-view-route-extension.v1",
            "width": ROUTE_WIDTH,
            "hidden": 64,
        },
    )
    second_store = PersistentOpaqueStateStore(
        root / "second_extension.pt",
        configuration={
            "component": "multi-append-second-extension",
            "schema": "neural-computer.opaque-view-route-extension.v1",
            "width": ROUTE_WIDTH,
            "hidden": 64,
        },
    )
    base_store.save_module(base_router)
    first_store.save_module(first_extension)
    second_store.save_module(second_extension)
    reloaded_base = OpaqueAddressRouter(width=ROUTE_WIDTH, hidden=64)
    reloaded_first = OpaqueViewRouteExtension(width=ROUTE_WIDTH, hidden=64)
    reloaded_second = OpaqueViewRouteExtension(width=ROUTE_WIDTH, hidden=64)
    base_store.load_module(reloaded_base)
    first_store.load_module(reloaded_first)
    second_store.load_module(reloaded_second)
    reloaded_rates = _route_rates(
        reloaded_base,
        reloaded_first,
        reloaded_second,
        parent,
        old_keys,
        audit_count=args.audit_count,
        seed=args.seed + 140_000,
    )
    reloaded_behavior: dict[str, float] = {}
    for row, (label, operation, span) in enumerate(ALL_PROGRAMS):
        _, artifact = lifecycle.memory.promote_index(selected_rows[label])
        program, decoder = _load_artifact(artifact)
        reloaded_behavior[label] = _capability_accuracy(
            parent,
            program,
            decoder,
            operation=operation,
            span=span,
            count=args.audit_count,
            seed=args.seed + 180_000 + row,
        )

    artifact_name = lifecycle.memory.paths[0]
    if artifact_name is None:
        raise RuntimeError("grown bank has no artifact path")
    artifact_path = second_path / artifact_name
    intact = artifact_path.read_bytes()
    artifact_path.write_bytes(intact + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(second_path)
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(intact)
    parent_digest_after = _digest_core(parent, ())
    base_accounting = base_result["accounting"]
    base_route_lifetimes = int(base_accounting["unique_route_lifetimes"])
    base_route_bits = int(base_accounting["unique_route_verifier_bits"])
    extension_route_lifetimes = sum(
        int(item["unique_route_lifetimes"]) for item in extension_accounting
    )
    route_bits = sum(
        int(item["unique_route_verifier_bits"]) for item in extension_accounting
    )
    report = {
        "schema": "neural-computer.multi-append-external-capability-report.v1",
        "claim_boundary": (
            "Two sequential protected external capability appends use frozen "
            "parent and prior route state, with fresh scalar outcomes and no "
            "replay. This is bounded external growth, not general continual "
            "learning or arbitrary program induction."
        ),
        "seed": args.seed,
        "programs": [
            {"label": label, "operation": operation, "span": span}
            for label, operation, span in ALL_PROGRAMS
        ],
        "parent_stable_bits_to_threshold": parent_stable_bits,
        "stable_bits_to_threshold": stable_bits,
        "physical_rows": len(lifecycle.memory.occupied),
        "protected_base": protected_base,
        "route_rates": rates,
        "candidate_permutation_accuracy": permuted_accuracy,
        "reward_shuffled_first_selection": first_null,
        "reward_shuffled_second_selection": second_null,
        "reloaded_route_rates": reloaded_rates,
        "selected_behavior": selected_behavior,
        "wrong_behavior": wrong_behavior,
        "reloaded_behavior": reloaded_behavior,
        "parent_core_digest_before": parent_digest_before,
        "parent_core_digest_after": parent_digest_after,
        "base_router_digest_before": base_digest,
        "base_router_digest_after": base_digest_after,
        "corruption_rejected": corruption_rejected,
        "histories": histories,
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + args.updates
                * args.batch_size
                * sum(span + 2 for _, _, span in PROGRAMS)
                + 2 * args.append_updates * args.batch_size * 6
                + base_route_bits
                + route_bits
                + 5 * args.retention_probes * args.audit_count * 4
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.updates * args.batch_size * len(PROGRAMS) * 2
                + 2 * args.append_updates * args.batch_size * 2
                + base_route_lifetimes
                + extension_route_lifetimes
            ),
            "optimizer_updates": (
                args.parent_updates
                + args.updates * len(PROGRAMS)
                + 2 * args.append_updates
            ),
            "route_optimizer_updates": int(base_accounting["route_optimizer_updates"])
            + sum(
                int(item["route_optimizer_updates"]) for item in extension_accounting
            ),
            "route_unique_lifetimes": (
                base_route_lifetimes + extension_route_lifetimes
            ),
            "route_unique_verifier_bits": base_route_bits + route_bits,
            "replayed_examples": 0,
            "distribution_shifts": 2,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "parent_stable": parent_stable_bits is not None,
            "all_capabilities_stable": all(
                value is not None for value in stable_bits.values()
            ),
            "five_artifacts_present": len(lifecycle.memory.occupied)
            == len(ALL_PROGRAMS),
            "protected_base": protected_base,
            "protected_after_appends": all(lifecycle.protection_mask().tolist()),
            "old_route_retained": rates["old"] >= 0.8,
            "first_append_route_recovered": rates["first"] >= 0.8,
            "second_append_route_recovered": rates["second"] >= 0.8,
            "combined_route_recovered": rates["combined"] >= 0.8,
            "candidate_permutation_invariant": permuted_accuracy >= 0.8,
            "reward_shuffled_not_selected": first_null <= 0.5 and second_null <= 0.5,
            "all_selected_capabilities_mastered": all(
                value >= 0.75 for value in selected_behavior.values()
            ),
            "wrong_artifact_is_causal": all(
                selected_behavior[label] > wrong_behavior[label] + 0.05
                for label in selected_behavior
            ),
            "reloaded_route_preserved": reloaded_rates["combined"] >= 0.8,
            "reloaded_behavior_preserved": all(
                reloaded_behavior[label] >= selected_behavior[label] - 0.05
                for label in selected_behavior
            ),
            "parent_core_unchanged": parent_digest_before == parent_digest_after,
            "base_router_frozen": base_digest == base_digest_after,
            "corruption_rejected": corruption_rejected,
            "no_replayed_examples": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--append-updates", type=int, default=256)
    parser.add_argument("--route-updates", type=int, default=1024)
    parser.add_argument("--extension-updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--route-batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--retention-probes", type=int, default=8)
    parser.add_argument("--retention-threshold", type=float, default=0.75)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "route_rates": report["route_rates"],
                "reloaded_route_rates": report["reloaded_route_rates"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
