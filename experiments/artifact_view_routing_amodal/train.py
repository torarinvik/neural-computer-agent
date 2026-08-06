"""Learn opaque executable-view routing from attempted scalar outcomes."""

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
from experiments.artifact_consolidation_amodal.train import (
    _load_composed,
    _train_parent_and_artifacts,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _accuracy,
    _digest_core,
)
from experiments.working_memory_continuous.canonical_no_replay_artifact_bank import (
    _context_keys,
)
from neural_computer import (
    ExecutableArtifactMemory,
    FactorizedOpaqueAddressRouter,
    compose_growth_artifacts,
    paired_counterfactual_ranking_loss,
)


def _fresh_queries(
    parent,
    *,
    span: int,
    count: int,
    seed: int,
) -> torch.Tensor:
    batch = generate_sequence_memory_batch(
        count,
        span=span,
        distractors=1,
        seed=seed,
        operation="forward",
    )
    return F.normalize(_context_keys(parent, batch, occupancy_scale=8.0), dim=-1)


def _train_view_router(
    parent,
    candidate_keys: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
) -> tuple[FactorizedOpaqueAddressRouter, dict[str, int | float]]:
    router = FactorizedOpaqueAddressRouter(
        width=int(candidate_keys.shape[-1]),
        hidden=64,
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    router.train()
    total_lifetimes = 0
    total_bits = 0
    losses: list[float] = []
    for update in range(updates):
        family = update % 2
        span = 3 + family
        queries = _fresh_queries(
            parent,
            span=span,
            count=batch_size,
            seed=seed + update * 10_007,
        )
        attempted = torch.tensor([[0, 1]], dtype=torch.long).expand(
            batch_size, -1
        )
        outcomes = (attempted == family).to(torch.float32)
        if shuffle_outcomes:
            outcomes = torch.randint(
                0,
                2,
                outcomes.shape,
                generator=generator,
            ).to(torch.float32)
        loss, _ = paired_counterfactual_ranking_loss(
            router(queries, candidate_keys),
            attempted,
            outcomes,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        total_lifetimes += batch_size
        total_bits += batch_size * 2
    router.eval()
    return router, {
        "unique_route_lifetimes": total_lifetimes,
        "unique_route_verifier_bits": total_bits,
        "route_optimizer_updates": updates,
        "replayed_route_examples": 0,
        "final_loss": losses[-1],
    }


@torch.no_grad()
def _test_queries(parent, *, audit_count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    queries = torch.cat(
        [
            _fresh_queries(
                parent,
                span=3 + family,
                count=audit_count,
                seed=seed + family * 10_007,
            )
            for family in range(2)
        ]
    )
    targets = torch.arange(2).repeat_interleave(audit_count)
    return queries, targets


@torch.no_grad()
def _route_accuracy(
    router: FactorizedOpaqueAddressRouter,
    queries: torch.Tensor,
    targets: torch.Tensor,
    candidate_keys: torch.Tensor,
) -> float:
    predictions = router(queries, candidate_keys).argmax(dim=-1)
    return float((predictions == targets).float().mean())


@torch.no_grad()
def _permuted_accuracy(
    router: FactorizedOpaqueAddressRouter,
    queries: torch.Tensor,
    targets: torch.Tensor,
    candidate_keys: torch.Tensor,
) -> float:
    permutation = torch.tensor([1, 0], dtype=torch.long)
    predictions = router(queries, candidate_keys[permutation]).argmax(dim=-1)
    return float((permutation[predictions] == targets).float().mean())


def _compact_bank(
    parent,
    artifacts: dict[str, dict[str, torch.Tensor]],
    route_keys: dict[str, torch.Tensor],
    *,
    directory: Path,
) -> ExecutableArtifactMemory:
    source_path = directory.parent / "source_bank"
    if source_path.exists():
        shutil.rmtree(source_path)
    source = ExecutableArtifactMemory(source_path, width=48, capacity=2)
    source.put(route_keys["3"], artifacts["3"])
    source.put(route_keys["4"], artifacts["4"])
    composed = compose_growth_artifacts(
        (artifacts["3"], artifacts["4"]),
        prefix_maps=(
            {"growth_slots.0.": "growth_slots.0."},
            {"growth_slots.0.": "growth_slots.1."},
        ),
    )
    if directory.exists():
        shutil.rmtree(directory)

    def verifier(candidate: ExecutableArtifactMemory) -> bool:
        return all(
            candidate.promote(route_keys[span])[0].view == view
            for span, view in (("3", "0"), ("4", "1"))
        )

    candidate, receipt = source.consolidate_verified(
        (0, 1),
        F.normalize(route_keys["3"] + route_keys["4"], dim=0),
        composed,
        directory,
        replacement_aliases=(route_keys["3"], route_keys["4"]),
        replacement_alias_views=("0", "1"),
        verifier=verifier,
    )
    if not receipt.accepted or candidate is None:
        raise RuntimeError(f"view bank compaction was rejected: {receipt}")
    loaded = ExecutableArtifactMemory.load(directory)
    if len(loaded.occupied) != 1 or len(loaded.view_candidates()) != 2:
        raise RuntimeError("compacted view bank has the wrong candidate shape")
    return loaded


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
    bank = _compact_bank(parent, artifacts, route_keys, directory=bank_path)
    candidates = bank.view_candidates()
    candidate_keys = torch.stack([key for _, key, _ in candidates])
    view_ids = [view for _, _, view in candidates]
    if view_ids != ["0", "1"]:
        raise RuntimeError(f"unexpected view order: {view_ids}")

    router, route_accounting = _train_view_router(
        parent,
        candidate_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 60_000,
        shuffle_outcomes=False,
    )
    shuffled_router, shuffled_accounting = _train_view_router(
        parent,
        candidate_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 70_000,
        shuffle_outcomes=True,
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
        shuffled_router,
        test_queries,
        test_targets,
        candidate_keys,
    )
    permuted_route = _permuted_accuracy(
        router,
        test_queries,
        test_targets,
        candidate_keys,
    )

    parent_behavior = _accuracy(
        parent,
        operation="forward",
        count=args.audit_count,
        span=2,
        seed=args.seed + 90_002,
    )
    selected_behavior: dict[str, float] = {"2": parent_behavior}
    wrong_behavior: dict[str, float] = {}
    selected_views: dict[str, str] = {}
    for family, span in enumerate((3, 4)):
        family_queries = test_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        predictions = router(family_queries, candidate_keys).argmax(dim=-1)
        selected_index = int(torch.mode(predictions).values)
        selected_view = view_ids[selected_index]
        selected_views[str(span)] = selected_view
        selected_runtime = _load_view(
            parent,
            bank,
            selected_view,
            seed=args.seed + 100_000 + family,
        )
        selected_behavior[str(span)] = _accuracy(
            selected_runtime,
            operation="forward",
            count=args.audit_count,
            span=span,
            seed=args.seed + 110_000 + family,
        )
        wrong_view = view_ids[1 - selected_index]
        wrong_runtime = _load_view(
            parent,
            bank,
            wrong_view,
            seed=args.seed + 120_000 + family,
        )
        wrong_behavior[str(span)] = _accuracy(
            wrong_runtime,
            operation="forward",
            count=args.audit_count,
            span=span,
            seed=args.seed + 110_000 + family,
        )

    reloaded = ExecutableArtifactMemory.load(bank_path)
    reloaded_router = FactorizedOpaqueAddressRouter(width=48, hidden=64)
    reloaded_router.load_state_dict(torch.load(router_path, weights_only=False))
    reloaded_router.eval()
    reloaded_candidates = reloaded.view_candidates()
    reloaded_keys = torch.stack([key for _, key, _ in reloaded_candidates])
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
    reload_behavior: dict[str, float] = {"2": parent_behavior}
    reload_views: dict[str, str] = {}
    for family, span in enumerate((3, 4)):
        family_queries = test_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        selected_index = int(
            torch.mode(reloaded_router(family_queries, reloaded_keys).argmax(dim=-1)).values
        )
        view = reloaded_candidates[selected_index][2]
        reload_views[str(span)] = view
        runtime = _load_view(parent, reloaded, view, seed=args.seed + 130_000 + family)
        reload_behavior[str(span)] = _accuracy(
            runtime,
            operation="forward",
            count=args.audit_count,
            span=span,
            seed=args.seed + 110_000 + family,
        )

    artifact_path = bank_path / reloaded.paths[0]
    if reloaded.paths[0] is None:
        raise RuntimeError("reloaded view bank has no artifact path")
    intact_payload = artifact_path.read_bytes()
    artifact_path.write_bytes(intact_payload + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(bank_path)
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(intact_payload)

    reloaded_core_digest = _digest_core(
        _load_view(parent, reloaded, reload_views["4"], seed=args.seed + 140_000),
        ("growth_slots.0.", "growth_slots.1."),
    )

    report = {
        "schema": "neural-computer.artifact-view-routing-report.v1",
        "claim_boundary": (
            "A frozen parent routes between two already-acquired executable "
            "views in one physical artifact row using only opaque queries, "
            "opaque candidate keys, attempted view outcomes, and scalar "
            "verifier outcomes; no general continual-learning claim."
        ),
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "route_updates": args.route_updates,
        "route_batch_size": args.route_batch_size,
        "audit_count": args.audit_count,
        "source_rows": 2,
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
        "reloaded_candidate_exact": reloaded_candidate_exact,
        "reloaded_views": reload_views,
        "reloaded_behavior": reload_behavior,
        "corruption_rejected": corruption_rejected,
        "parent_core_digest": parent_digest,
        "reloaded_core_digest": reloaded_core_digest,
        "accounting": {
            "unique_logical_lifetimes": args.updates * args.batch_size * 3,
            "unique_verifier_bits": args.updates * args.batch_size * (2 + 3 + 4),
            "optimizer_updates": args.updates * 3,
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
            "two_opaque_views": view_ids == ["0", "1"],
            "selected_views_match_expected": selected_views == {"3": "0", "4": "1"},
            "learned_route_at_least_90": normal_route >= 0.90,
            "reward_shuffled_route_near_chance": shuffled_route <= 0.50,
            "candidate_permutation_invariant": permuted_route >= 0.90,
            "selected_parent_retained": selected_behavior["2"] >= 0.80,
            "wrong_view_is_causal": all(
                selected_behavior[str(span)]
                > wrong_behavior[str(span)] + 0.05
                for span in (3, 4)
            ),
            "reloaded_route_preserved": reloaded_route >= 0.90,
            "reloaded_views_match_expected": reload_views == {"3": "0", "4": "1"},
            "reloaded_candidate_exact": reloaded_candidate_exact,
            "reloaded_behavior_preserved": all(
                reload_behavior[str(span)]
                >= selected_behavior[str(span)] - 0.05
                for span in (2, 3, 4)
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
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--route-updates", type=int, default=512)
    parser.add_argument("--route-batch-size", type=int, default=16)
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
