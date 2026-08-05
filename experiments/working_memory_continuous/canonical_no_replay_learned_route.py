"""No-replay retention with learned opaque address discovery.

This audit is the learned-routing successor to
``canonical_no_replay_artifact_bank``. Capability files are acquired exactly
as before, but their memory keys are random opaque vectors and the route is
learned by ``FactorizedOpaqueAddressRouter`` from fresh attempted-row outcomes. The
router sees only the frozen controller's opaque hidden query and candidate
keys; it receives no occupancy feature, span label, semantic task ID, or
correct unattempted row.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    SequenceMemoryBatch,
    generate_sequence_memory_batch,
)
from neural_computer import (
    AmodalControllerRuntime,
    ControllerFeedback,
    ExecutableArtifactMemory,
    FactorizedOpaqueAddressRouter,
    attempted_outcome_loss,
    paired_counterfactual_ranking_loss,
)

from .canonical_growth_pressure_test import (
    _accuracy,
    _artifact,
    _copy_parent_weights,
    _digest_core,
    _freeze_except,
    _rollout,
    _runtime,
)
from .canonical_no_replay_artifact_bank import (
    _growth_runtime,
    _load_selected,
    _stable_bits,
    _train_current_span,
)


def _quiet_feedback(batch_size: int) -> ControllerFeedback:
    zeros = torch.zeros(batch_size)
    return ControllerFeedback(
        torch.zeros(batch_size, 2), zeros, torch.ones(batch_size), zeros
    )


@torch.no_grad()
def _opaque_queries(
    runtime: AmodalControllerRuntime,
    batch: SequenceMemoryBatch,
) -> torch.Tensor:
    """Return only the controller's opaque latent context query."""
    state = runtime.initial_state(batch.batch_size, device=batch.input_frames.device)
    feedback = _quiet_feedback(batch.batch_size)
    for frame in batch.input_frames.transpose(0, 1):
        _, state = runtime.step_streams({"vision": frame}, state, feedback)
    for frame in batch.distractor_frames.transpose(0, 1):
        _, state = runtime.step_streams({"vision": frame}, state, feedback)
    return F.normalize(state.hidden, dim=-1)


def _fresh_route_batch(
    parent: AmodalControllerRuntime,
    *,
    family: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    batch = generate_sequence_memory_batch(
        batch_size,
        span=family + 2,
        distractors=1,
        seed=seed,
        operation="forward",
    )
    return _opaque_queries(parent, batch)


def _random_keys(*, rows: int, width: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return F.normalize(torch.randn(rows, width, generator=generator), dim=-1)


def _train_router(
    parent: AmodalControllerRuntime,
    keys: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
    query_cache_batch_size: int,
) -> tuple[FactorizedOpaqueAddressRouter, dict[str, int | float]]:
    router = FactorizedOpaqueAddressRouter(width=int(keys.shape[-1]), hidden=64)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=3e-3, weight_decay=1e-5
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    families = torch.randint(0, 3, (updates,), generator=generator).tolist()
    family_counts = [families.count(family) * batch_size for family in range(3)]
    query_cache: list[torch.Tensor] = []
    for family, count in enumerate(family_counts):
        chunks: list[torch.Tensor] = []
        remaining = count
        chunk_index = 0
        while remaining:
            chunk = min(remaining, query_cache_batch_size)
            if chunk % 2:
                chunk -= 1
            chunk = max(chunk, 2)
            chunks.append(
                _fresh_route_batch(
                    parent,
                    family=family,
                    batch_size=chunk,
                    seed=seed + family * 1_000_003 + chunk_index * 10_007,
                )
            )
            remaining -= chunk
            chunk_index += 1
        query_cache.append(torch.cat(chunks) if chunks else torch.empty(0, 32))
    family_offsets = [0, 0, 0]
    route_lifetimes = 0
    route_bits = 0
    losses: list[float] = []
    router.train()
    for update, family in enumerate(families):
        start = family_offsets[family]
        stop = start + batch_size
        queries = query_cache[family][start:stop]
        family_offsets[family] = stop
        attempted = torch.randint(
            keys.shape[0], (batch_size,), generator=generator
        )
        outcomes = (attempted == family).to(torch.float32)
        if shuffle_outcomes:
            outcomes = outcomes[torch.randperm(batch_size, generator=generator)]
        loss = attempted_outcome_loss(router(queries, keys), attempted, outcomes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        route_lifetimes += batch_size
        route_bits += batch_size
        if update == updates - 1:
            losses.append(float(loss.detach()))
    router.eval()
    return router, {
        "unique_route_lifetimes": route_lifetimes,
        "unique_route_verifier_bits": route_bits,
        "route_optimizer_updates": updates,
        "replayed_route_examples": 0,
        "final_loss": losses[-1] if losses else None,
        "query_cache_lifetimes": sum(family_counts),
    }


def _train_router_counterfactual(
    parent: AmodalControllerRuntime,
    keys: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
    query_cache_batch_size: int,
) -> tuple[FactorizedOpaqueAddressRouter, dict[str, int | float]]:
    """Train the route from paired common-random attempted-row outcomes.

    Each pair shares one fresh hidden query but attempts two distinct opaque
    rows. The verifier's outcome difference trains the score difference for
    those rows; no correct-row label is exposed to the router. This is the
    generic counterfactual credit path for memory-side decisions.
    """
    if keys.shape[0] < 2:
        raise ValueError("counterfactual routing needs at least two candidate rows")
    router = FactorizedOpaqueAddressRouter(width=int(keys.shape[-1]), hidden=64)
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=3e-3, weight_decay=1e-5
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    families = [update % 3 for update in range(updates)]
    pair_schedule = torch.tensor(
        [[0, 1], [0, 2], [1, 2]], dtype=torch.long
    )
    family_counts = [families.count(family) * batch_size for family in range(3)]
    query_cache: list[torch.Tensor] = []
    for family, count in enumerate(family_counts):
        chunks: list[torch.Tensor] = []
        remaining = count
        chunk_index = 0
        while remaining:
            chunk = min(remaining, query_cache_batch_size)
            if chunk % 2:
                chunk -= 1
            chunk = max(chunk, 2)
            chunks.append(
                _fresh_route_batch(
                    parent,
                    family=family,
                    batch_size=chunk,
                    seed=seed + family * 1_000_003 + chunk_index * 10_007,
                )
            )
            remaining -= chunk
            chunk_index += 1
        query_cache.append(torch.cat(chunks) if chunks else torch.empty(0, 32))

    family_offsets = [0, 0, 0]
    route_lifetimes = 0
    route_bits = 0
    losses: list[float] = []
    router.train()
    for update, family in enumerate(families):
        start = family_offsets[family]
        stop = start + batch_size
        queries = query_cache[family][start:stop]
        family_offsets[family] = stop
        attempted = pair_schedule[(update // 3) % len(pair_schedule)]
        attempted = attempted.unsqueeze(0).expand(batch_size, -1).clone()
        outcomes = (attempted == family).to(torch.float32)
        if shuffle_outcomes:
            outcomes = torch.randint(
                0, 2, outcomes.shape, generator=generator
            ).to(torch.float32)
        loss, _ = paired_counterfactual_ranking_loss(
            router(queries, keys),
            attempted,
            outcomes,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
        route_lifetimes += batch_size
        route_bits += batch_size * 2
        if update == updates - 1:
            losses.append(float(loss.detach()))
    router.eval()
    return router, {
        "unique_route_lifetimes": route_lifetimes,
        "unique_route_verifier_bits": route_bits,
        "route_optimizer_updates": updates,
        "replayed_route_examples": 0,
        "final_loss": losses[-1] if losses else None,
        "query_cache_lifetimes": sum(family_counts),
        "counterfactual_pairs": route_lifetimes,
    }


@torch.no_grad()
def _test_queries(
    parent: AmodalControllerRuntime,
    *,
    audit_count: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries: list[torch.Tensor] = []
    targets: list[int] = []
    for family in range(3):
        queries.append(
            _fresh_route_batch(
                parent,
                family=family,
                batch_size=audit_count,
                seed=seed + family * 10_007,
            )
        )
        targets.extend([family] * audit_count)
    return torch.cat(queries), torch.tensor(targets, dtype=torch.long)


@torch.no_grad()
def _route_accuracy(
    router: FactorizedOpaqueAddressRouter,
    queries: torch.Tensor,
    targets: torch.Tensor,
    keys: torch.Tensor,
) -> float:
    predictions = router(queries, keys).argmax(dim=-1)
    return float((predictions == targets).float().mean())


@torch.no_grad()
def _permuted_accuracy(
    router: FactorizedOpaqueAddressRouter,
    queries: torch.Tensor,
    targets: torch.Tensor,
    keys: torch.Tensor,
) -> float:
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    predictions = router(queries, keys[permutation]).argmax(dim=-1)
    return float((permutation[predictions] == targets).float().mean())


def _train_parent(
    parent: AmodalControllerRuntime,
    *,
    updates: int,
    batch_size: int,
    audit_count: int,
    seed: int,
    learning_rate: float,
    eval_every: int,
) -> tuple[list[dict[str, float | int]], list[dict[str, float | int]]]:
    parameters = list(parent.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-5)
    history: list[dict[str, float | int]] = []
    progress: list[dict[str, float | int]] = []
    parent.train()
    for update in range(1, updates + 1):
        batch = generate_sequence_memory_batch(
            batch_size,
            span=2,
            distractors=1,
            seed=seed + update * 10_007,
            operation="forward",
        )
        result = _rollout(parent, batch, train=True)
        optimizer.zero_grad(set_to_none=True)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        history.append(
            {
                "update": update,
                "training_accuracy": float(result["rewards"].mean()),
                "loss": float(result["loss"].detach()),
            }
        )
        if update == updates or (eval_every > 0 and update % eval_every == 0):
            parent.eval()
            progress.append(
                {
                    "update": update,
                    "heldout_accuracy": _accuracy(
                        parent,
                        operation="forward",
                        count=audit_count,
                        span=2,
                        seed=seed + 1_000_000 + update,
                    ),
                }
            )
            parent.train()
    parent.eval()
    return history, progress


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    stages = (2, 3, 4)
    if min(
        args.updates_per_stage,
        args.batch_size,
        args.audit_count,
        args.route_updates,
    ) < 1:
        raise ValueError("all update and batch sizes must be positive")
    if (
        args.batch_size % 2
        or args.audit_count % 2
        or args.route_batch_size % 2
        or args.route_cache_batch_size % 2
    ):
        raise ValueError("batch sizes must be even")

    parent = _runtime(seed=args.seed, growth=False)
    parent_history, parent_progress = _train_parent(
        parent,
        updates=args.updates_per_stage,
        batch_size=args.batch_size,
        audit_count=args.audit_count,
        seed=args.seed + 100,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
    )

    bank_path = args.report.parent / "artifact_memory"
    if bank_path.exists():
        shutil.rmtree(bank_path)
    keys = _random_keys(rows=3, width=32, seed=args.seed + 50_000)
    bank = ExecutableArtifactMemory(bank_path, width=32, capacity=3)
    zero_runtime = _growth_runtime(seed=args.seed, width=args.growth_width)
    _copy_parent_weights(parent, zero_runtime)
    bank.put(keys[0], _artifact(zero_runtime, "growth_slots.0."))

    stage_history: dict[str, dict[str, list[dict[str, float | int]]]] = {
        "2": {"training": parent_history, "progress": parent_progress}
    }
    for index, span in enumerate((3, 4), start=1):
        acquired = _growth_runtime(seed=args.seed + index, width=args.growth_width)
        _copy_parent_weights(parent, acquired)
        _freeze_except(acquired, ("growth_slots.0.",))
        history, progress = _train_current_span(
            acquired,
            span=span,
            updates=args.updates_per_stage,
            batch_size=args.batch_size,
            seed=args.seed + 200 * index,
            learning_rate=args.learning_rate,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
        )
        bank.put(keys[index], _artifact(acquired, "growth_slots.0."))
        stage_history[str(span)] = {"training": history, "progress": progress}
    bank = ExecutableArtifactMemory.load(bank_path)

    train_router = (
        _train_router_counterfactual
        if args.route_credit == "paired_counterfactual"
        else _train_router
    )
    router, route_accounting = train_router(
        parent,
        keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 60_000,
        shuffle_outcomes=False,
        query_cache_batch_size=args.route_cache_batch_size,
    )
    shuffled_router, shuffled_accounting = train_router(
        parent,
        keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 70_000,
        shuffle_outcomes=True,
        query_cache_batch_size=args.route_cache_batch_size,
    )
    router_path = args.report.parent / "opaque_address_router.pt"
    torch.save(router.state_dict(), router_path)

    test_queries, test_targets = _test_queries(
        parent,
        audit_count=args.audit_count,
        seed=args.seed + 80_000,
    )
    normal_route = _route_accuracy(router, test_queries, test_targets, keys)
    shuffled_route = _route_accuracy(
        shuffled_router, test_queries, test_targets, keys
    )
    permuted_route = _permuted_accuracy(
        router, test_queries, test_targets, keys
    )

    selected_rows: dict[str, int] = {}
    selected_behavior: dict[str, float] = {}
    wrong_behavior: dict[str, float] = {}
    for family, span in enumerate(stages):
        family_queries = test_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        predictions = router(family_queries, keys).argmax(dim=-1)
        row = int(torch.mode(predictions).values)
        selected_rows[str(span)] = row
        selected = _load_selected(
            parent,
            bank,
            row,
            seed=args.seed + 90_000 + family,
            growth_width=args.growth_width,
        )
        selected_behavior[str(span)] = _accuracy(
            selected,
            operation="forward",
            count=args.audit_count,
            span=span,
            seed=args.seed + 100_000 + family,
        )
        wrong = _load_selected(
            parent,
            bank,
            (row + 1) % 3,
            seed=args.seed + 110_000 + family,
            growth_width=args.growth_width,
        )
        wrong_behavior[str(span)] = _accuracy(
            wrong,
            operation="forward",
            count=args.audit_count,
            span=span,
            seed=args.seed + 100_000 + family,
        )

    final_runtime = _load_selected(
        parent,
        bank,
        selected_rows["4"],
        seed=args.seed + 120_000,
        growth_width=args.growth_width,
    )
    controls = {
        "blank_sequence": _accuracy(
            final_runtime,
            operation="forward",
            count=args.audit_count,
            span=4,
            seed=args.seed + 130_001,
            blank_sequence=True,
        ),
        "workspace_disabled": _accuracy(
            final_runtime,
            operation="forward",
            count=args.audit_count,
            span=4,
            seed=args.seed + 130_002,
            disable_workspace=True,
        ),
    }
    stable_bits = {
        str(span): _stable_bits(
            stage_history[str(span)]["progress"],
            threshold=args.mastery_threshold,
            bits_per_update=args.batch_size * span,
        )
        for span in stages
    }
    core_unchanged = _digest_core(
        parent, ("growth_slots.0.", "growth_slots.1.")
    ) == _digest_core(
        final_runtime, ("growth_slots.0.", "growth_slots.1.")
    )
    report = {
        "schema": "canonical-no-replay-learned-route-v1",
        "claim_boundary": (
            "A frozen canonical parent acquires spans 3 and 4 as independent "
            "executable artifacts. A permutation-equivariant opaque router "
            "discovers artifact addresses from fresh attempted-row outcomes. "
            "No occupancy feature or span label is supplied to the router."
        ),
        "seed": args.seed,
        "stages": list(stages),
        "updates_per_stage": args.updates_per_stage,
        "batch_size": args.batch_size,
        "audit_count": args.audit_count,
        "route_updates": args.route_updates,
        "route_batch_size": args.route_batch_size,
        "growth_width": args.growth_width,
        "router": "factorized_opaque_query_key_v1",
        "route_credit": args.route_credit,
        "route_input": "controller_hidden_only",
        "artifact_memory": str(bank_path),
        "router_checkpoint": str(router_path),
        "candidate_keys": keys.tolist(),
        "route_accuracy": normal_route,
        "reward_shuffled_route_accuracy": shuffled_route,
        "candidate_permutation_accuracy": permuted_route,
        "selected_rows": selected_rows,
        "selected_behavior": selected_behavior,
        "wrong_behavior": wrong_behavior,
        "controls": controls,
        "stable_bits_to_threshold": stable_bits,
        "history": stage_history,
        "accounting": {
            "unique_logical_lifetimes": args.updates_per_stage * args.batch_size * 3,
            "unique_verifier_bits": sum(
                args.updates_per_stage * args.batch_size * span for span in stages
            ),
            "optimizer_updates": args.updates_per_stage * 3 + args.route_updates * 2,
            "replayed_examples": 0,
            "route_optimizer_updates": args.route_updates * 2,
            "route_unique_lifetimes": (
                route_accounting["unique_route_lifetimes"]
                + shuffled_accounting["unique_route_lifetimes"]
            ),
            "route_unique_verifier_bits": (
                route_accounting["unique_route_verifier_bits"]
                + shuffled_accounting["unique_route_verifier_bits"]
            ),
            "route_counterfactual_pairs": (
                route_accounting.get("counterfactual_pairs", 0)
                + shuffled_accounting.get("counterfactual_pairs", 0)
            ),
            "route_replayed_examples": 0,
            "stable_bits_to_threshold": stable_bits,
            "diagnostic_lifetimes_charged_to_budget": args.audit_count * 3,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "no_replayed_examples": True,
            "learned_route_at_least_90": normal_route >= 0.90,
            "reward_shuffled_route_near_chance": shuffled_route <= 0.50,
            "candidate_permutation_invariant": permuted_route >= 0.90,
            "all_prior_spans_retained": (
                selected_behavior["2"] >= args.mastery_threshold
                and selected_behavior["3"] >= args.mastery_threshold
            ),
            "target_span_mastered": (
                selected_behavior["4"] >= args.mastery_threshold
                and stable_bits["4"] is not None
            ),
            "wrong_artifact_is_causal": (
                selected_behavior["3"] > wrong_behavior["3"] + 0.05
                and selected_behavior["4"] > wrong_behavior["4"] + 0.05
            ),
            "blank_sequence_near_chance": controls["blank_sequence"] <= 0.65,
            "workspace_ablation_is_informative": controls["workspace_disabled"] < selected_behavior["4"] - 0.05,
            "frozen_parent_core": core_unchanged,
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69401)
    parser.add_argument("--updates-per-stage", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=64)
    parser.add_argument("--route-updates", type=int, default=2048)
    parser.add_argument("--route-batch-size", type=int, default=16)
    parser.add_argument("--route-cache-batch-size", type=int, default=128)
    parser.add_argument(
        "--route-credit",
        choices=("single_outcome", "paired_counterfactual"),
        default="single_outcome",
    )
    parser.add_argument("--growth-width", type=int, default=64)
    parser.add_argument("--mastery-threshold", type=float, default=0.80)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "accepted_diagnostic": report["accepted_diagnostic"],
                "route_accuracy": report["route_accuracy"],
                "reward_shuffled_route_accuracy": report[
                    "reward_shuffled_route_accuracy"
                ],
                "candidate_permutation_accuracy": report[
                    "candidate_permutation_accuracy"
                ],
                "selected_behavior": report["selected_behavior"],
                "controls": report["controls"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
