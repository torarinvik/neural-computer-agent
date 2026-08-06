"""Scale outcome-only executable-view routing from two to four views."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    generate_sequence_memory_batch,
)
from experiments.artifact_consolidation_amodal.train import _direct_growth_runtime
from experiments.artifact_view_routing_amodal.train import _load_composed
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _accuracy,
    _artifact,
    _copy_parent_weights,
    _digest_core,
    _freeze_except,
    _runtime,
    _train,
)
from neural_computer import (
    ControllerFeedback,
    ExecutableArtifactMemory,
    FactorizedOpaqueAddressRouter,
    OpaqueAddressRouter,
    attempted_outcome_loss,
    compose_growth_artifacts,
    paired_counterfactual_ranking_loss,
)

OPERATIONS = (
    "forward",
    "reverse",
    "complement",
    "complement_reverse",
)
VIEW_IDS = tuple(str(index) for index in range(len(OPERATIONS)))


def _fresh_queries(parent, *, operation: str, count: int, seed: int) -> torch.Tensor:
    batch = generate_sequence_memory_batch(
        count,
        span=4,
        distractors=1,
        seed=seed,
        operation=operation,
    )
    state = parent.initial_state(batch.batch_size, device=batch.input_frames.device)
    quiet = ControllerFeedback(
        torch.zeros(batch.batch_size, 2),
        torch.zeros(batch.batch_size),
        torch.ones(batch.batch_size),
        torch.zeros(batch.batch_size),
    )
    for frame in batch.input_frames.transpose(0, 1):
        _, state = parent.step_streams({"vision": frame}, state, quiet)
    for frame in batch.distractor_frames.transpose(0, 1):
        _, state = parent.step_streams({"vision": frame}, state, quiet)
    first_query, state = parent.step_streams(
        {"vision": batch.query_frames[:, 0]},
        state,
        quiet,
    )
    return F.normalize(
        torch.cat([first_query.controller.memory_query_key, state.hidden], dim=-1),
        dim=-1,
    )


def _train_parent_and_artifacts(
    *,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[object, dict[str, dict[str, torch.Tensor]], dict[str, torch.Tensor]]:
    parent = _runtime(seed=seed, growth=False)
    _train(
        parent,
        operation="forward",
        updates=updates,
        batch_size=batch_size,
        span=2,
        seed=seed + 100,
        lr=learning_rate,
    )
    artifacts: dict[str, dict[str, torch.Tensor]] = {}
    route_keys: dict[str, torch.Tensor] = {}
    for index, operation in enumerate(OPERATIONS):
        acquired = _direct_growth_runtime(seed=seed + index + 1, width=64)
        _copy_parent_weights(parent, acquired)
        _freeze_except(acquired, ("growth_slots.0.",))
        _train(
            acquired,
            operation=operation,
            updates=updates,
            batch_size=batch_size,
            span=4,
            seed=seed + 200 * (index + 1),
            lr=learning_rate,
        )
        artifacts[operation] = _artifact(acquired, "growth_slots.0.")
    generator = torch.Generator(device="cpu").manual_seed(seed + 50_000)
    random_keys = F.normalize(
        torch.randn(len(OPERATIONS), 64, generator=generator),
        dim=-1,
    )
    for index, operation in enumerate(OPERATIONS):
        route_keys[operation] = random_keys[index]
    return parent, artifacts, route_keys


def _compact_bank(
    artifacts: dict[str, dict[str, torch.Tensor]],
    route_keys: dict[str, torch.Tensor],
    *,
    directory: Path,
) -> ExecutableArtifactMemory:
    source_path = directory.parent / "source_bank"
    if source_path.exists():
        shutil.rmtree(source_path)
    source = ExecutableArtifactMemory(source_path, width=64, capacity=4)
    for operation in OPERATIONS:
        source.put(route_keys[operation], artifacts[operation])
    if directory.exists():
        shutil.rmtree(directory)
    composed = compose_growth_artifacts(
        tuple(artifacts[operation] for operation in OPERATIONS),
        prefix_maps=tuple(
            {"growth_slots.0.": f"growth_slots.{index}."}
            for index in range(len(OPERATIONS))
        ),
    )

    def verifier(candidate: ExecutableArtifactMemory) -> bool:
        return all(
            candidate.promote(route_keys[operation])[0].view == VIEW_IDS[index]
            for index, operation in enumerate(OPERATIONS)
        )

    candidate, receipt = source.consolidate_verified(
        tuple(range(len(OPERATIONS))),
        F.normalize(sum(route_keys.values()), dim=0),
        composed,
        directory,
        replacement_aliases=tuple(route_keys[operation] for operation in OPERATIONS),
        replacement_alias_views=VIEW_IDS,
        verifier=verifier,
    )
    if not receipt.accepted or candidate is None:
        raise RuntimeError(f"four-view compaction was rejected: {receipt}")
    loaded = ExecutableArtifactMemory.load(directory)
    if len(loaded.occupied) != 1 or len(loaded.view_candidates()) != len(OPERATIONS):
        raise RuntimeError("compacted four-view bank has the wrong candidate shape")
    return loaded


def _train_router(
    parent,
    candidate_keys: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
    credit: str,
    router_kind: str,
) -> tuple[torch.nn.Module, dict[str, int | float]]:
    router_class = (
        FactorizedOpaqueAddressRouter
        if router_kind == "factorized"
        else OpaqueAddressRouter
    )
    router = router_class(width=int(candidate_keys.shape[-1]), hidden=64)
    optimizer = torch.optim.AdamW(router.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    router.train()
    route_lifetimes = 0
    route_bits = 0
    last_loss = 0.0
    for update in range(updates):
        family = update % len(OPERATIONS)
        opponent = (family + 1 + (update // len(OPERATIONS)) % 3) % len(OPERATIONS)
        queries = _fresh_queries(
            parent,
            operation=OPERATIONS[family],
            count=batch_size,
            seed=seed + update * 10_007,
        )
        scores = router(queries, candidate_keys)
        if credit == "paired_counterfactual":
            attempted = torch.tensor([[family, opponent]], dtype=torch.long).expand(
                batch_size, -1
            )
            outcomes = torch.tensor([[1.0, 0.0]], dtype=torch.float32).expand(
                batch_size, -1
            )
            if shuffle_outcomes:
                outcomes = torch.randint(
                    0,
                    2,
                    outcomes.shape,
                    generator=generator,
                ).to(torch.float32)
            loss, _ = paired_counterfactual_ranking_loss(
                scores,
                attempted,
                outcomes,
            )
            route_bits += batch_size * 2
        elif credit == "attempted_outcome":
            attempted = torch.randint(
                len(OPERATIONS),
                (batch_size,),
                generator=generator,
            )
            outcomes = (attempted == family).to(torch.float32)
            if shuffle_outcomes:
                outcomes = torch.randint(
                    0,
                    2,
                    outcomes.shape,
                    generator=generator,
                ).to(torch.float32)
            loss = attempted_outcome_loss(scores, attempted, outcomes)
            route_bits += batch_size
        else:
            raise ValueError(f"unknown route credit {credit!r}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
        route_lifetimes += batch_size
    router.eval()
    return router, {
        "unique_route_lifetimes": route_lifetimes,
        "unique_route_verifier_bits": route_bits,
        "route_optimizer_updates": updates,
        "replayed_route_examples": 0,
        "final_loss": last_loss,
    }


@torch.no_grad()
def _test_queries(parent, *, audit_count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    queries = torch.cat(
        [
            _fresh_queries(
                parent,
                operation=operation,
                count=audit_count,
                seed=seed + index * 10_007,
            )
            for index, operation in enumerate(OPERATIONS)
        ]
    )
    targets = torch.arange(len(OPERATIONS)).repeat_interleave(audit_count)
    return queries, targets


@torch.no_grad()
def _route_accuracy(router, queries, targets, candidate_keys) -> float:
    predictions = router(queries, candidate_keys).argmax(dim=-1)
    return float((predictions == targets).float().mean())


@torch.no_grad()
def _permuted_accuracy(router, queries, targets, candidate_keys) -> float:
    permutation = torch.tensor([2, 0, 3, 1], dtype=torch.long)
    predictions = router(queries, candidate_keys[permutation]).argmax(dim=-1)
    return float((permutation[predictions] == targets).float().mean())


def _load_view(parent, bank, view: str, *, seed: int):
    handle, artifact = bank.promote_view(0, view)
    if handle.view != view:
        raise RuntimeError("memory returned the wrong opaque view")
    return _load_composed(parent, artifact, seed=seed, view=view)


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(args.updates, args.route_updates, args.batch_size, args.audit_count) < 1:
        raise ValueError("updates, route updates, and batch sizes must be positive")
    parent, artifacts, route_keys = _train_parent_and_artifacts(
        seed=args.seed,
        updates=args.updates,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    parent_digest = _digest_core(parent, ("growth_slots.0.", "growth_slots.1."))
    bank_path = args.report_out.parent / "view_bank"
    bank = _compact_bank(artifacts, route_keys, directory=bank_path)
    candidates = bank.view_candidates()
    candidate_keys = torch.stack([key for _, key, _ in candidates])
    view_ids = [view for _, _, view in candidates]
    if view_ids != list(VIEW_IDS):
        raise RuntimeError(f"unexpected view order: {view_ids}")

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
    shuffled_router, shuffled_accounting = _train_router(
        parent,
        candidate_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 70_000,
        shuffle_outcomes=True,
        credit=args.route_credit,
        router_kind=args.router_kind,
    )
    router_path = args.report_out.parent / "view_router.pt"
    torch.save(router.state_dict(), router_path)
    test_queries, test_targets = _test_queries(
        parent,
        audit_count=args.audit_count,
        seed=args.seed + 80_000,
    )
    route_started = perf_counter()
    normal_route = _route_accuracy(router, test_queries, test_targets, candidate_keys)
    route_latency_ms = 1000.0 * (perf_counter() - route_started)
    shuffled_route = _route_accuracy(
        shuffled_router, test_queries, test_targets, candidate_keys
    )
    permuted_route = _permuted_accuracy(
        router, test_queries, test_targets, candidate_keys
    )

    selected_behavior: dict[str, float] = {
        "2": _accuracy(
            parent,
            operation="forward",
            count=args.audit_count,
            span=2,
            seed=args.seed + 90_002,
        )
    }
    wrong_behavior: dict[str, float] = {}
    selected_views: dict[str, str] = {}
    for family, operation in enumerate(OPERATIONS):
        family_queries = test_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        predictions = router(family_queries, candidate_keys).argmax(dim=-1)
        selected_index = int(torch.mode(predictions).values)
        selected_view = view_ids[selected_index]
        selected_views[operation] = selected_view
        selected_runtime = _load_view(
            parent, bank, selected_view, seed=args.seed + 100_000 + family
        )
        selected_behavior[operation] = _accuracy(
            selected_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 110_000 + family,
        )
        wrong_view = view_ids[(selected_index + 1) % len(view_ids)]
        wrong_runtime = _load_view(
            parent, bank, wrong_view, seed=args.seed + 120_000 + family
        )
        wrong_behavior[operation] = _accuracy(
            wrong_runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 110_000 + family,
        )

    reloaded = ExecutableArtifactMemory.load(bank_path)
    reloaded_candidates = reloaded.view_candidates()
    reloaded_keys = torch.stack([key for _, key, _ in reloaded_candidates])
    router_class = (
        FactorizedOpaqueAddressRouter
        if args.router_kind == "factorized"
        else OpaqueAddressRouter
    )
    reloaded_router = router_class(
        width=int(reloaded_keys.shape[-1]),
        hidden=64,
    )
    reloaded_router.load_state_dict(torch.load(router_path, weights_only=False))
    reloaded_router.eval()
    reload_route_started = perf_counter()
    reloaded_route = _route_accuracy(
        reloaded_router,
        test_queries,
        test_targets,
        reloaded_keys,
    )
    reload_route_latency_ms = 1000.0 * (
        perf_counter() - reload_route_started
    )
    reloaded_candidate_exact = (
        [view for _, _, view in reloaded_candidates] == view_ids
        and all(
            torch.equal(key, candidate_keys[index])
            for index, (_, key, _) in enumerate(reloaded_candidates)
        )
    )
    reloaded_views: dict[str, str] = {}
    reloaded_behavior: dict[str, float] = {"2": selected_behavior["2"]}
    for family, operation in enumerate(OPERATIONS):
        family_queries = test_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        predictions = reloaded_router(family_queries, reloaded_keys).argmax(dim=-1)
        selected_index = int(torch.mode(predictions).values)
        view = reloaded_candidates[selected_index][2]
        reloaded_views[operation] = view
        runtime = _load_view(parent, reloaded, view, seed=args.seed + 130_000 + family)
        reloaded_behavior[operation] = _accuracy(
            runtime,
            operation=operation,
            count=args.audit_count,
            span=4,
            seed=args.seed + 110_000 + family,
        )

    artifact_name = reloaded.paths[0]
    if artifact_name is None:
        raise RuntimeError("reloaded view bank has no artifact path")
    artifact_path = bank_path / artifact_name
    intact_payload = artifact_path.read_bytes()
    artifact_path.write_bytes(intact_payload + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(bank_path)
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(intact_payload)
    reloaded_core_digest = _digest_core(
        _load_view(parent, reloaded, reloaded_views[OPERATIONS[-1]], seed=args.seed + 140_000),
        ("growth_slots.0.", "growth_slots.1."),
    )
    report = {
        "schema": "neural-computer.artifact-view-routing-scaling-report.v1",
        "claim_boundary": (
            "Four independently acquired equal-span procedures route through "
            "four opaque executable views in one physical artifact row using "
            "only controller queries, opaque keys, attempted outcomes, and "
            "scalar verifier outcomes; no general continual-learning claim."
        ),
        "seed": args.seed,
        "operations": list(OPERATIONS),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "route_updates": args.route_updates,
        "route_batch_size": args.route_batch_size,
        "route_credit": args.route_credit,
        "router_kind": args.router_kind,
        "audit_count": args.audit_count,
        "source_rows": len(OPERATIONS),
        "compacted_rows": len(bank.occupied),
        "view_ids": view_ids,
        "route_accuracy": normal_route,
        "route_latency_ms": route_latency_ms,
        "reward_shuffled_route_accuracy": shuffled_route,
        "candidate_permutation_accuracy": permuted_route,
        "selected_views": selected_views,
        "selected_behavior": selected_behavior,
        "wrong_behavior": wrong_behavior,
        "reloaded_route_accuracy": reloaded_route,
        "reloaded_route_latency_ms": reload_route_latency_ms,
        "reloaded_views": reloaded_views,
        "reloaded_behavior": reloaded_behavior,
        "reloaded_candidate_exact": reloaded_candidate_exact,
        "corruption_rejected": corruption_rejected,
        "parent_core_digest": parent_digest,
        "reloaded_core_digest": reloaded_core_digest,
        "accounting": {
            "unique_logical_lifetimes": args.updates * args.batch_size * 5,
            "unique_verifier_bits": args.updates * args.batch_size * (2 + 4 * 4),
            "optimizer_updates": args.updates * 5,
            "route_optimizer_updates": args.route_updates * 2,
            "route_unique_lifetimes": (
                route_accounting["unique_route_lifetimes"]
                + shuffled_accounting["unique_route_lifetimes"]
            ),
            "route_unique_verifier_bits": (
                route_accounting["unique_route_verifier_bits"]
                + shuffled_accounting["unique_route_verifier_bits"]
            ),
            "replayed_examples": 0,
            "route_replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "one_physical_row": len(bank.occupied) == 1,
            "four_opaque_views": view_ids == list(VIEW_IDS),
            "selected_views_match_expected": selected_views
            == dict(zip(OPERATIONS, VIEW_IDS, strict=True)),
            "learned_route_at_least_90": normal_route >= 0.90,
            "reward_shuffled_route_near_chance": shuffled_route <= 0.50,
            "candidate_permutation_invariant": permuted_route >= 0.90,
            "selected_parent_retained": selected_behavior["2"] >= 0.80,
            "all_procedures_mastered": min(
                selected_behavior[operation] for operation in OPERATIONS
            ) >= 0.75,
            "wrong_view_is_causal": all(
                selected_behavior[operation] > wrong_behavior[operation] + 0.05
                for operation in OPERATIONS
            ),
            "reloaded_route_preserved": reloaded_route >= 0.90,
            "reloaded_views_match_expected": reloaded_views
            == dict(zip(OPERATIONS, VIEW_IDS, strict=True)),
            "reloaded_candidate_exact": reloaded_candidate_exact,
            "reloaded_behavior_preserved": all(
                reloaded_behavior[operation]
                >= selected_behavior[operation] - 0.05
                for operation in ("2", *OPERATIONS)
            ),
            "frozen_core": parent_digest == reloaded_core_digest,
            "corruption_rejected": corruption_rejected,
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--route-updates", type=int, default=1024)
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
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "route_accuracy": report["route_accuracy"],
                "reward_shuffled_route_accuracy": report[
                    "reward_shuffled_route_accuracy"
                ],
                "candidate_permutation_accuracy": report[
                    "candidate_permutation_accuracy"
                ],
                "selected_behavior": report["selected_behavior"],
                "wrong_behavior": report["wrong_behavior"],
                "reloaded_route_accuracy": report["reloaded_route_accuracy"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
