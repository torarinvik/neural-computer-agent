"""Audit sequential protected appends of external learned programs."""

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
    OpaqueAppendOnlyRouteChain,
    OpaqueViewRouteExtension,
    PersistentOpaqueStateStore,
    RetentionPolicyConfig,
    paired_counterfactual_ranking_loss,
)

FIRST_APPEND = ("rotate4", "rotate", 4)
SECOND_APPEND = ("adjacent_xor4", "adjacent_xor", 4)
THIRD_APPEND = ("complement_rotate4", "complement_rotate", 4)
APPEND_SPECS = (FIRST_APPEND, SECOND_APPEND, THIRD_APPEND)
ALL_PROGRAMS = (*PROGRAMS, *APPEND_SPECS)
ROUTE_WIDTH = 48


def _train_append_extension(
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


def _append_scores(
    base_router: OpaqueAddressRouter,
    extensions: tuple[OpaqueViewRouteExtension, ...],
    queries: torch.Tensor,
    old_keys: torch.Tensor,
    *,
    active_extension_count: int,
) -> torch.Tensor:
    if not 0 <= active_extension_count <= len(extensions):
        raise ValueError("active extension count is out of range")
    chain = OpaqueAppendOnlyRouteChain(
        base_router,
        width=ROUTE_WIDTH,
        extensions=extensions,
    )
    failures = torch.zeros(
        queries.shape[0],
        len(extensions),
        dtype=torch.bool,
        device=queries.device,
    )
    if active_extension_count:
        failures[:, :active_extension_count] = True
    return chain(queries, old_keys, failures)


def _route_rates(
    base_router: OpaqueAddressRouter,
    extensions: tuple[OpaqueViewRouteExtension, ...],
    parent,
    old_keys: torch.Tensor,
    *,
    audit_count: int,
    seed: int,
) -> dict[str, float]:
    old_queries, old_targets = _test_queries(parent, count=audit_count, seed=seed)
    scores = [
        _append_scores(
            base_router,
            extensions,
            old_queries,
            old_keys,
            active_extension_count=0,
        )
    ]
    for index, (_label, operation, span) in enumerate(APPEND_SPECS, start=1):
        scores.append(
            _append_scores(
                base_router,
                extensions,
                _route_queries(
                    parent,
                    operation=operation,
                    span=span,
                    count=audit_count,
                    seed=seed + index * 10_001,
                ),
                old_keys,
                active_extension_count=index,
            )
        )
    correct = [(scores[0].argmax(dim=-1) == old_targets).float()]
    correct.extend(
        (scores[index].argmax(dim=-1) == len(PROGRAMS) + index - 1).float()
        for index in range(1, len(scores))
    )
    return {
        "old": float(correct[0].mean()),
        **{
            label: float(correct[index + 1].mean())
            for index, (label, _operation, _span) in enumerate(APPEND_SPECS)
        },
        "combined": float(torch.cat(correct).mean()),
    }


def _permuted_combined_accuracy(
    base_router: OpaqueAddressRouter,
    extensions: tuple[OpaqueViewRouteExtension, ...],
    parent,
    old_keys: torch.Tensor,
    *,
    audit_count: int,
    seed: int,
) -> float:
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted_keys = old_keys[permutation]
    old_queries, old_targets = _test_queries(parent, count=audit_count, seed=seed)
    old_predictions = _append_scores(
        base_router,
        extensions,
        old_queries,
        permuted_keys,
        active_extension_count=0,
    ).argmax(dim=-1)
    old_predictions = permutation[old_predictions]
    correct = [old_predictions == old_targets]
    for index, (_label, operation, span) in enumerate(APPEND_SPECS, start=1):
        queries = _route_queries(
            parent,
            operation=operation,
            span=span,
            count=audit_count,
            seed=seed + index * 10_001,
        )
        predictions = _append_scores(
            base_router,
            extensions,
            queries,
            permuted_keys,
            active_extension_count=index,
        ).argmax(dim=-1)
        correct.append(predictions == len(PROGRAMS) + index - 1)
    return float(torch.cat(correct).float().mean())


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
    growth_paths = [
        root / f"grown_bank_{index + 1}" for index in range(len(APPEND_SPECS))
    ]
    for path in (bank_path, *growth_paths):
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

    extensions: list[OpaqueViewRouteExtension] = []
    shuffled_extensions: list[OpaqueViewRouteExtension] = []
    extension_accounting: list[dict[str, int | float]] = []
    append_keys: list[torch.Tensor] = []
    for append_index, (label, operation, span) in enumerate(APPEND_SPECS):
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
            raise RuntimeError(
                "protected append did not select transactional growth: "
                f"append_index={append_index}, action={plan.action}, "
                f"protection_mask={lifecycle.protection_mask().tolist()}"
            )
        destination = growth_paths[append_index]
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
        established_extensions = tuple(extensions)

        def established_scores(
            queries: torch.Tensor,
            prior=established_extensions,
            active_count=append_index,
        ) -> torch.Tensor:
            return _append_scores(
                base_router,
                prior,
                queries,
                old_keys,
                active_extension_count=active_count,
            )

        extension, accounting = _train_append_extension(
            parent,
            established_scores,
            operation=operation,
            updates=args.extension_updates,
            batch_size=args.route_batch_size,
            seed=args.seed + 100_000 + append_index * 20_000,
            shuffle_outcomes=False,
        )
        shuffled_extension, shuffled_accounting = _train_append_extension(
            parent,
            established_scores,
            operation=operation,
            updates=args.extension_updates,
            batch_size=args.route_batch_size,
            seed=args.seed + 110_000 + append_index * 20_000,
            shuffle_outcomes=True,
        )
        extensions.append(extension)
        shuffled_extensions.append(shuffled_extension)
        extension_accounting.extend((accounting, shuffled_accounting))

    base_digest_after = _digest_module(base_router)
    rates = _route_rates(
        base_router,
        tuple(extensions),
        parent,
        old_keys,
        audit_count=args.audit_count,
        seed=args.seed + 140_000,
    )
    permuted_accuracy = _permuted_combined_accuracy(
        base_router,
        tuple(extensions),
        parent,
        old_keys,
        audit_count=args.audit_count,
        seed=args.seed + 140_000,
    )
    shuffled_selection: dict[str, float] = {}
    for append_index, ((label, operation, span), shuffled) in enumerate(
        zip(APPEND_SPECS, shuffled_extensions, strict=True)
    ):
        variant = list(extensions)
        variant[append_index] = shuffled
        queries = _route_queries(
            parent,
            operation=operation,
            span=span,
            count=args.audit_count,
            seed=args.seed + 150_000 + append_index,
        )
        shuffled_selection[label] = float(
            (
                _append_scores(
                    base_router,
                    tuple(variant),
                    queries,
                    old_keys,
                    active_extension_count=append_index + 1,
                ).argmax(dim=-1)
                == len(PROGRAMS) + append_index
            )
            .float()
            .mean()
        )

    selected_rows = {
        label: len(PROGRAMS) + index
        for index, (label, _operation, _span) in enumerate(APPEND_SPECS)
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
    base_store.save_module(base_router)
    extension_stores = []
    for index, extension in enumerate(extensions):
        store = PersistentOpaqueStateStore(
            root / f"extension_{index + 1}.pt",
            configuration={
                "component": f"multi-append-extension-{index + 1}",
                "schema": "neural-computer.opaque-view-route-extension.v1",
                "width": ROUTE_WIDTH,
                "hidden": 64,
            },
        )
        store.save_module(extension)
        extension_stores.append(store)
    reloaded_base = OpaqueAddressRouter(width=ROUTE_WIDTH, hidden=64)
    base_store.load_module(reloaded_base)
    reloaded_extensions = []
    for store in extension_stores:
        extension = OpaqueViewRouteExtension(width=ROUTE_WIDTH, hidden=64)
        store.load_module(extension)
        reloaded_extensions.append(extension)
    reloaded_rates = _route_rates(
        reloaded_base,
        tuple(reloaded_extensions),
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
    artifact_path = growth_paths[-1] / artifact_name
    intact = artifact_path.read_bytes()
    artifact_path.write_bytes(intact + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(growth_paths[-1])
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
            f"{len(APPEND_SPECS)} sequential protected external capability "
            "appends use frozen parent and prior route state, with fresh "
            "scalar outcomes and no replay. This is bounded external growth, "
            "not general continual learning or arbitrary program induction."
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
        "reward_shuffled_selection": shuffled_selection,
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
                + args.append_updates
                * args.batch_size
                * sum(span + 2 for _, _, span in APPEND_SPECS)
                + base_route_bits
                + route_bits
                + len(ALL_PROGRAMS)
                * args.retention_probes
                * args.audit_count
                * 4
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.updates * args.batch_size * len(PROGRAMS) * 2
                + len(APPEND_SPECS) * args.append_updates * args.batch_size * 2
                + base_route_lifetimes
                + extension_route_lifetimes
            ),
            "optimizer_updates": (
                args.parent_updates
                + args.updates * len(PROGRAMS)
                + len(APPEND_SPECS) * args.append_updates
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
            "distribution_shifts": len(APPEND_SPECS),
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "parent_stable": parent_stable_bits is not None,
            "all_capabilities_stable": all(
                value is not None for value in stable_bits.values()
            ),
            "all_artifacts_present": len(lifecycle.memory.occupied)
            == len(ALL_PROGRAMS),
            "protected_base": protected_base,
            "protected_after_appends": all(lifecycle.protection_mask().tolist()),
            "old_route_retained": rates["old"] >= 0.8,
            "all_append_routes_recovered": all(
                rates[label] >= 0.8
                for label, _operation, _span in APPEND_SPECS
            ),
            "combined_route_recovered": rates["combined"] >= 0.8,
            "candidate_permutation_invariant": permuted_accuracy >= 0.8,
            "reward_shuffled_not_selected": all(
                value <= 0.5 for value in shuffled_selection.values()
            ),
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
