"""Acquire generated compositions as isolated append-only artifacts.

Each composition is learned in a fresh external routed capability.  Once its
behavior is stable, the artifact is admitted to persistent opaque memory and
protected by fresh retention outcomes.  A separate permutation-equivariant
router learns which opaque row to select from the rendered event query.  Old
artifact weights are never updated while a new artifact is acquired.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F

from experiments.frozen_core_transfer_amodal.train import _train_with_progress
from experiments.parent_conditioned_artifact_bank_amodal.train import (
    _capability_accuracy,
    _new_capability,
    _route_queries,
    _stable_bits,
    _train_capability,
)
from experiments.working_memory_continuous.canonical_growth_pressure_test import (
    _digest_core,
    _runtime,
)
from neural_computer import (
    ExecutableArtifactMemory,
    ExternalCapabilityLifecycle,
    OpaqueAppendOnlyRouteChain,
    OpaqueAddressRouter,
    OpaqueViewRouteExtension,
    PersistentOpaqueStateStore,
    RetentionPolicyConfig,
    paired_counterfactual_ranking_loss,
)
from .train_pipeline import _new_stack


COMPOSITION_COUNT = 6
SPAN = 4
ROUTE_WIDTH = 48
THRESHOLD = 0.75
SHUFFLE_REPLICATES = 3


def _stack_artifact(stack, decoder) -> dict[str, torch.Tensor]:
    return {
        **{
            f"stack.{name}": value.detach().cpu().clone()
            for name, value in stack.state_dict().items()
        },
        **{
            f"decoder.{name}": value.detach().cpu().clone()
            for name, value in decoder.state_dict().items()
        },
    }


def _load_stack_artifact(
    artifact: dict[str, torch.Tensor],
) -> tuple[torch.nn.Module, torch.nn.Module]:
    stack = _new_stack(seed=0, program_count=2, stack="routed")
    decoder = _new_capability(seed=1)[1]
    stack_state = {
        name.removeprefix("stack."): value
        for name, value in artifact.items()
        if name.startswith("stack.")
    }
    decoder_state = {
        name.removeprefix("decoder."): value
        for name, value in artifact.items()
        if name.startswith("decoder.")
    }
    if not stack_state or not decoder_state:
        raise ValueError("composition artifact is missing stack or decoder state")
    stack.load_state_dict(stack_state, strict=True)
    decoder.load_state_dict(decoder_state, strict=True)
    stack.eval()
    decoder.eval()
    return stack, decoder


def _generated_key(parent, composition_id: int, *, count: int, seed: int) -> torch.Tensor:
    return F.normalize(
        _route_queries(
            parent,
            operation="generated_composition",
            span=SPAN,
            count=count,
            seed=seed,
            generated_composition_ids=(composition_id,),
        ).mean(dim=0),
        dim=0,
    )


def _train_router(
    parent,
    candidate_keys: torch.Tensor,
    composition_ids: tuple[int, ...],
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
) -> tuple[OpaqueAddressRouter, dict[str, int]]:
    router = OpaqueAddressRouter(width=int(candidate_keys.shape[-1]), hidden=64)
    optimizer = torch.optim.AdamW(router.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    row_count = len(composition_ids)
    router.train()
    for update in range(updates):
        target_index = update % row_count
        composition_id = composition_ids[target_index]
        queries = _route_queries(
            parent,
            operation="generated_composition",
            span=SPAN,
            count=batch_size,
            seed=seed + update * 10_007,
            generated_composition_ids=(composition_id,),
        )
        competitor = torch.randint(
            row_count - 1,
            (batch_size,),
            generator=generator,
        )
        competitor = competitor + (competitor >= target_index).to(torch.long)
        attempted = torch.stack(
            [
                torch.full((batch_size,), target_index, dtype=torch.long),
                competitor,
            ],
            dim=1,
        )
        if shuffle_outcomes:
            signs = torch.zeros(batch_size, dtype=torch.long)
            signs[: batch_size // 2] = 1
            signs = signs[torch.randperm(batch_size, generator=generator)]
            outcomes = torch.stack([signs, 1 - signs], dim=1).float()
        else:
            outcomes = (attempted == target_index).float()
        loss, _ = paired_counterfactual_ranking_loss(
            router(queries, candidate_keys), attempted, outcomes
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
        optimizer.step()
    router.eval()
    return router, {
        "unique_route_lifetimes": updates * batch_size,
        "unique_route_verifier_bits": updates * batch_size * 2,
        "route_optimizer_updates": updates,
    }


def _append_scores(
    base_router: OpaqueAddressRouter,
    extensions: tuple[OpaqueViewRouteExtension, ...],
    queries: torch.Tensor,
    base_keys: torch.Tensor,
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
        queries.shape[0], len(extensions), dtype=torch.bool
    )
    if active_extension_count:
        failures[:, :active_extension_count] = True
    return chain(queries, base_keys, failures)


def _train_append_extension(
    parent,
    established_scores,
    composition_id: int,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
) -> tuple[OpaqueViewRouteExtension, dict[str, int]]:
    """Train one route stage only from fresh outcomes for its new artifact."""

    extension = OpaqueViewRouteExtension(width=ROUTE_WIDTH, hidden=64)
    optimizer = torch.optim.AdamW(
        extension.parameters(), lr=3e-3, weight_decay=1e-5
    )
    extension.train()
    for update in range(updates):
        query_count = batch_size // 2 if shuffle_outcomes else batch_size
        queries = _route_queries(
            parent,
            operation="generated_composition",
            span=SPAN,
            count=query_count,
            seed=seed + update * 10_007,
            generated_composition_ids=(composition_id,),
        )
        if shuffle_outcomes:
            queries = queries.repeat_interleave(2, dim=0)
        with torch.no_grad():
            established = established_scores(queries)
            established_best = established.max(dim=-1).values
        delta = extension(queries)
        candidate_scores = torch.stack(
            (established_best, established_best + delta), dim=1
        )
        attempted = torch.tensor([[1, 0]], dtype=torch.long).expand(
            batch_size, -1
        )
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
    }


@torch.no_grad()
def _append_stage_accuracy(
    base_router: OpaqueAddressRouter,
    extensions: tuple[OpaqueViewRouteExtension, ...],
    parent,
    base_keys: torch.Tensor,
    composition_id: int,
    target_index: int,
    base_count: int,
    *,
    count: int,
    seed: int,
    permute_base_keys: bool = False,
) -> float:
    permutation = torch.arange(base_count - 1, -1, -1)
    keys = base_keys[permutation] if permute_base_keys else base_keys
    queries = _route_queries(
        parent,
        operation="generated_composition",
        span=SPAN,
        count=count,
        seed=seed,
        generated_composition_ids=(composition_id,),
    )
    active = max(0, target_index - base_count + 1)
    predictions = _append_scores(
        base_router,
        extensions,
        queries,
        keys,
        active_extension_count=active,
    ).argmax(dim=-1)
    if permute_base_keys:
        predictions = torch.where(
            predictions < base_count,
            permutation[predictions.clamp_max(base_count - 1)],
            predictions,
        )
    return float((predictions == target_index).float().mean())


@torch.no_grad()
def _append_route_accuracy(
    base_router: OpaqueAddressRouter,
    extensions: tuple[OpaqueViewRouteExtension, ...],
    parent,
    base_keys: torch.Tensor,
    composition_ids: tuple[int, ...],
    base_count: int,
    *,
    count: int,
    seed: int,
    permute_base_keys: bool = False,
) -> float:
    correct = [
        _append_stage_accuracy(
            base_router,
            extensions,
            parent,
            base_keys,
            composition_id,
            index,
            base_count,
            count=count,
            seed=seed + index * 10_007,
            permute_base_keys=permute_base_keys,
        )
        for index, composition_id in enumerate(composition_ids)
    ]
    return sum(correct) / len(correct)


def _run_append_only_routes(
    parent,
    candidate_keys: torch.Tensor,
    composition_ids: tuple[int, ...],
    *,
    base_count: int,
    updates: int,
    batch_size: int,
    route_audit_count: int,
    seed: int,
    root: Path,
) -> dict[str, object]:
    if not 1 <= base_count < len(composition_ids):
        raise ValueError("append-only routing needs at least one new artifact")
    base_keys = candidate_keys[:base_count]
    base_ids = composition_ids[:base_count]
    if base_count == 1:
        base_router = OpaqueAddressRouter(width=ROUTE_WIDTH, hidden=64)
        base_accounting = {
            "unique_route_lifetimes": 0,
            "unique_route_verifier_bits": 0,
            "route_optimizer_updates": 0,
        }
    else:
        base_router, base_accounting = _train_router(
            parent,
            base_keys,
            base_ids,
            updates=updates,
            batch_size=batch_size,
            seed=seed,
            shuffle_outcomes=False,
        )
    extensions: list[OpaqueViewRouteExtension] = []
    shuffled_extensions: list[OpaqueViewRouteExtension] = []
    append_accounting: list[dict[str, int]] = []
    shuffled_accounting: list[dict[str, int]] = []
    for append_index, composition_id in enumerate(composition_ids[base_count:]):
        established = tuple(extensions)

        def established_scores(
            queries: torch.Tensor,
            prior=established,
        ) -> torch.Tensor:
            return _append_scores(
                base_router,
                prior,
                queries,
                base_keys,
                active_extension_count=len(prior),
            )

        extension, accounting = _train_append_extension(
            parent,
            established_scores,
            composition_id,
            updates=updates,
            batch_size=batch_size,
            seed=seed + 20_000 + append_index * 1_000_003,
            shuffle_outcomes=False,
        )
        shuffled_extension, shuffled_item = _train_append_extension(
            parent,
            established_scores,
            composition_id,
            updates=updates,
            batch_size=batch_size,
            seed=seed + 30_000 + append_index * 1_000_003,
            shuffle_outcomes=True,
        )
        extensions.append(extension)
        shuffled_extensions.append(shuffled_extension)
        append_accounting.append(accounting)
        shuffled_accounting.append(shuffled_item)

    route_accuracy = _append_route_accuracy(
        base_router,
        tuple(extensions),
        parent,
        base_keys,
        composition_ids,
        base_count,
        count=route_audit_count,
        seed=seed + 50_000,
    )
    permuted_accuracy = _append_route_accuracy(
        base_router,
        tuple(extensions),
        parent,
        base_keys,
        composition_ids,
        base_count,
        count=route_audit_count,
        seed=seed + 50_000,
        permute_base_keys=True,
    )
    shuffled_route_accuracy_samples: list[float] = []
    for replicate in range(len(shuffled_extensions)):
        variant = list(extensions)
        variant[replicate] = shuffled_extensions[replicate]
        for audit_index in range(2):
            shuffled_route_accuracy_samples.append(
                _append_stage_accuracy(
                    base_router,
                    tuple(variant),
                    parent,
                    base_keys,
                    composition_ids[base_count + replicate],
                    base_count + replicate,
                    base_count,
                    count=route_audit_count,
                    seed=seed
                    + 60_000
                    + replicate * 1_000_003
                    + audit_index * 100_003,
                )
            )

    base_store = PersistentOpaqueStateStore(
        root / "append_only_base_router.pt",
        configuration={
            "component": "generated-composition-append-only-base-router",
            "schema": "neural-computer.opaque-address-router.v1",
            "width": ROUTE_WIDTH,
            "hidden": 64,
            "candidate_count": base_count,
        },
    )
    base_store.save_module(base_router)
    extension_stores: list[PersistentOpaqueStateStore] = []
    for index, extension in enumerate(extensions):
        store = PersistentOpaqueStateStore(
            root / f"append_only_extension_{index + 1}.pt",
            configuration={
                "component": f"generated-composition-append-only-extension-{index + 1}",
                "schema": "neural-computer.opaque-view-route-extension.v1",
                "width": ROUTE_WIDTH,
                "hidden": 64,
            },
        )
        store.save_module(extension)
        extension_stores.append(store)
    reloaded_base = OpaqueAddressRouter(width=ROUTE_WIDTH, hidden=64)
    base_store.load_module(reloaded_base)
    reloaded_extensions: list[OpaqueViewRouteExtension] = []
    for store in extension_stores:
        extension = OpaqueViewRouteExtension(width=ROUTE_WIDTH, hidden=64)
        store.load_module(extension)
        reloaded_extensions.append(extension)
    reloaded_route_accuracy = _append_route_accuracy(
        reloaded_base,
        tuple(reloaded_extensions),
        parent,
        base_keys,
        composition_ids,
        base_count,
        count=route_audit_count,
        seed=seed + 50_000,
    )
    cold_start_old_accuracy = _append_route_accuracy(
        base_router,
        tuple(extensions),
        parent,
        base_keys,
        base_ids,
        base_count,
        count=route_audit_count,
        seed=seed + 70_000,
    )
    route_accounting = {
        "unique_route_lifetimes": base_accounting["unique_route_lifetimes"]
        + sum(item["unique_route_lifetimes"] for item in append_accounting)
        + sum(item["unique_route_lifetimes"] for item in shuffled_accounting),
        "unique_route_verifier_bits": base_accounting["unique_route_verifier_bits"]
        + sum(item["unique_route_verifier_bits"] for item in append_accounting)
        + sum(item["unique_route_verifier_bits"] for item in shuffled_accounting),
        "route_optimizer_updates": base_accounting["route_optimizer_updates"]
        + sum(item["route_optimizer_updates"] for item in append_accounting)
        + sum(item["route_optimizer_updates"] for item in shuffled_accounting),
    }
    return {
        "route_mode": "append_only",
        "route_accuracy": route_accuracy,
        "permuted_route_accuracy": permuted_accuracy,
        "shuffled_route_accuracy": sum(shuffled_route_accuracy_samples)
        / len(shuffled_route_accuracy_samples),
        "shuffled_route_accuracy_samples": shuffled_route_accuracy_samples,
        "reloaded_route_accuracy": reloaded_route_accuracy,
        "cold_start_old_accuracy": cold_start_old_accuracy,
        "route_accounting": route_accounting,
        "base_count": base_count,
        "extension_count": len(extensions),
    }


@torch.no_grad()
def _route_accuracy(
    router,
    parent,
    candidate_keys: torch.Tensor,
    composition_ids: tuple[int, ...],
    *,
    count: int,
    seed: int,
    permute_keys: bool = False,
) -> float:
    permutation = torch.arange(len(composition_ids) - 1, -1, -1)
    keys = candidate_keys[permutation] if permute_keys else candidate_keys
    correct: list[torch.Tensor] = []
    for index, composition_id in enumerate(composition_ids):
        queries = _route_queries(
            parent,
            operation="generated_composition",
            span=SPAN,
            count=count,
            seed=seed + index * 10_007,
            generated_composition_ids=(composition_id,),
        )
        predictions = router(queries, keys).argmax(dim=-1)
        if permute_keys:
            predictions = permutation[predictions]
        correct.append(predictions == index)
    return float(torch.cat(correct).float().mean())


def _probe_artifact(
    parent,
    bank: ExecutableArtifactMemory,
    key: torch.Tensor,
    row: int,
    composition_id: int,
    *,
    count: int,
    probes: int,
    seed: int,
) -> list[float]:
    outcomes: list[float] = []
    for probe in range(probes):
        _, artifact = bank.promote_index(row)
        stack, decoder = _load_stack_artifact(artifact)
        outcome = _capability_accuracy(
            parent,
            stack,
            decoder,
            operation="generated_composition",
            span=SPAN,
            count=count,
            seed=seed + probe * 101,
            generated_composition_ids=(composition_id,),
        )
        outcomes.append(outcome)
        bank.observe_retention(key, outcome)
    return outcomes


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.parent_updates,
        args.artifact_updates,
        args.route_updates,
        args.batch_size,
        args.route_batch_size,
        args.audit_count,
        args.route_audit_count,
        args.retention_probes,
    ) < 1:
        raise ValueError("all update and audit counts must be positive")
    if (
        args.batch_size % 2
        or args.route_batch_size % 2
        or args.audit_count % 2
        or args.route_audit_count % 2
    ):
        raise ValueError("batch sizes and audit count must be even")
    composition_ids = tuple(args.composition_ids)
    if not composition_ids:
        raise ValueError("at least one composition ID is required")
    if len(set(composition_ids)) != len(composition_ids):
        raise ValueError("composition IDs must be unique")
    if any(composition_id < 0 or composition_id >= COMPOSITION_COUNT for composition_id in composition_ids):
        raise ValueError("composition ID is out of range")
    if args.route_mode == "append_only" and not 1 <= args.base_route_count < len(composition_ids):
        raise ValueError(
            "append-only routing needs at least one base row and one appended row"
        )

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

    bank_path = root / "artifact_bank"
    if bank_path.exists():
        import shutil

        shutil.rmtree(bank_path)
    bank = ExecutableArtifactMemory(
        bank_path,
        width=ROUTE_WIDTH,
        capacity=1,
        write_match_threshold=0.99999,
    )
    bank.retention.config = RetentionPolicyConfig(
        mastery_threshold=THRESHOLD,
        min_mastery_observations=args.retention_probes,
    )
    lifecycle = ExternalCapabilityLifecycle(bank)
    row_by_composition: dict[int, int] = {}
    key_by_composition: dict[int, torch.Tensor] = {}
    histories: dict[str, object] = {
        "parent": {"history": parent_history, "progress": parent_progress}
    }
    artifact_stable: dict[str, int | None] = {}
    artifact_behavior: dict[str, float] = {}
    retention: dict[str, list[float]] = {}
    growth_paths = [root / f"artifact_bank_growth_{index + 1}" for index in range(len(composition_ids))]

    for phase_index, composition_id in enumerate(composition_ids):
        stack = _new_stack(args.seed + 1_000 + phase_index, program_count=2, stack="routed")
        decoder = _new_capability(args.seed + 2_000 + phase_index)[1]
        history, progress = _train_capability(
            parent,
            stack,
            decoder,
            operation="generated_composition",
            span=SPAN,
            updates=args.artifact_updates,
            batch_size=args.batch_size,
            seed=args.seed + 20_000 + phase_index * 1_000_003,
            audit_count=args.audit_count,
            eval_every=args.eval_every,
            learning_rate=args.learning_rate,
            generated_composition_ids=(composition_id,),
        )
        key = _generated_key(
            parent,
            composition_id,
            count=args.audit_count,
            seed=args.seed + 40_000 + phase_index,
        )
        artifact = _stack_artifact(stack, decoder)
        if phase_index == 0:
            receipt = lifecycle.admit(key, artifact)
        else:
            plan = lifecycle.plan_admission(key, artifact)
            if plan.action != "grow" or not all(lifecycle.protection_mask().tolist()):
                raise RuntimeError(
                    "protected append did not select transactional growth: "
                    f"phase={phase_index}, action={plan.action}, "
                    f"protection_mask={lifecycle.protection_mask().tolist()}"
                )
            receipt = lifecycle.admit(
                key,
                artifact,
                plan=plan,
                grow_destination=growth_paths[phase_index],
            )
        if not receipt.accepted or receipt.index is None:
            raise RuntimeError(f"artifact admission failed: {receipt.reason}")
        bank = lifecycle.memory
        row_by_composition[composition_id] = receipt.index
        key_by_composition[composition_id] = key
        stable = _stable_bits(
            progress,
            threshold=THRESHOLD,
            bits_per_update=args.batch_size * SPAN,
        )
        artifact_stable[str(composition_id)] = stable
        artifact_behavior[str(composition_id)] = _capability_accuracy(
            parent,
            stack,
            decoder,
            operation="generated_composition",
            span=SPAN,
            count=args.audit_count,
            seed=args.seed + 50_000 + phase_index,
            generated_composition_ids=(composition_id,),
        )
        retention[str(composition_id)] = _probe_artifact(
            parent,
            bank,
            key,
            receipt.index,
            composition_id,
            count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 60_000 + phase_index * 1_000,
        )
        histories[str(composition_id)] = {
            "history": history,
            "progress": progress,
            "stable_bits_to_threshold": stable,
            "admission": {
                "action": receipt.action,
                "row": receipt.index,
                "source_capacity": receipt.source_capacity,
                "destination_capacity": receipt.destination_capacity,
            },
        }

    bank.save()
    candidate_keys = torch.stack([key_by_composition[composition_id] for composition_id in composition_ids])
    route_audit_count = max(args.audit_count, args.route_audit_count)
    if args.route_mode == "append_only":
        route_result = _run_append_only_routes(
            parent,
            candidate_keys,
            composition_ids,
            base_count=args.base_route_count,
            updates=args.route_updates,
            batch_size=args.route_batch_size,
            route_audit_count=route_audit_count,
            seed=args.seed + 70_000,
            root=root,
        )
        route_mode = route_result["route_mode"]
        route_accuracy = float(route_result["route_accuracy"])
        permuted_accuracy = float(route_result["permuted_route_accuracy"])
        shuffled_route_accuracy = float(route_result["shuffled_route_accuracy"])
        shuffled_route_accuracy_samples = list(
            route_result["shuffled_route_accuracy_samples"]
        )
        reloaded_route_accuracy = float(
            route_result["reloaded_route_accuracy"]
        )
        cold_start_old_accuracy = float(
            route_result["cold_start_old_accuracy"]
        )
        route_accounting = dict(route_result["route_accounting"])
        route_accounting["replayed_route_examples"] = 0
    else:
        router, route_accounting = _train_router(
            parent,
            candidate_keys,
            composition_ids,
            updates=args.route_updates,
            batch_size=args.route_batch_size,
            seed=args.seed + 70_000,
            shuffle_outcomes=False,
        )
        route_mode = "bank"
        route_accuracy = _route_accuracy(
            router,
            parent,
            candidate_keys,
            composition_ids,
            count=route_audit_count,
            seed=args.seed + 90_000,
        )
        permuted_accuracy = _route_accuracy(
            router,
            parent,
            candidate_keys,
            composition_ids,
            count=route_audit_count,
            seed=args.seed + 90_000,
            permute_keys=True,
        )
        shuffled_route_accuracy_samples = []
        shuffled_accounting: list[dict[str, int]] = []
        for replicate in range(SHUFFLE_REPLICATES):
            shuffled_router, accounting = _train_router(
                parent,
                candidate_keys,
                composition_ids,
                updates=args.route_updates,
                batch_size=args.route_batch_size,
                seed=args.seed + 80_000 + replicate * 1_000_003,
                shuffle_outcomes=True,
            )
            shuffled_accounting.append(accounting)
            shuffled_route_accuracy_samples.extend(
                _route_accuracy(
                    shuffled_router,
                    parent,
                    candidate_keys,
                    composition_ids,
                    count=route_audit_count,
                    seed=args.seed
                    + 90_000
                    + replicate * 1_000_003
                    + audit_index * 100_003,
                )
                for audit_index in range(2)
            )
        shuffled_route_accuracy = sum(shuffled_route_accuracy_samples) / len(
            shuffled_route_accuracy_samples
        )
        router_store = PersistentOpaqueStateStore(
            root / "router.pt",
            configuration={
                "component": "generated-composition-artifact-router",
                "schema": "neural-computer.opaque-address-router.v1",
                "width": ROUTE_WIDTH,
                "hidden": 64,
                "candidate_count": len(composition_ids),
            },
        )
        router_store.save_module(router)
        reloaded_router = OpaqueAddressRouter(width=ROUTE_WIDTH, hidden=64)
        router_store.load_module(reloaded_router)
        reloaded_route_accuracy = _route_accuracy(
            reloaded_router,
            parent,
            candidate_keys,
            composition_ids,
            count=route_audit_count,
            seed=args.seed + 90_000,
        )
        cold_start_old_accuracy = route_accuracy
        route_accounting = {
            "unique_route_lifetimes": route_accounting["unique_route_lifetimes"]
            + sum(item["unique_route_lifetimes"] for item in shuffled_accounting),
            "unique_route_verifier_bits": route_accounting["unique_route_verifier_bits"]
            + sum(item["unique_route_verifier_bits"] for item in shuffled_accounting),
            "route_optimizer_updates": route_accounting["route_optimizer_updates"]
            + sum(item["route_optimizer_updates"] for item in shuffled_accounting),
            "replayed_route_examples": 0,
        }

    corruption_rejected = False
    artifact_path = bank.directory / (bank.paths[0] or "")
    original_bytes = artifact_path.read_bytes()
    artifact_path.write_bytes(original_bytes + b"corruption")
    try:
        ExecutableArtifactMemory.load(bank.directory)
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(original_bytes)

    parent_digest_after = _digest_core(parent, ())
    report = {
        "schema": "neural-computer.generated-composition-artifact-bank-report.v1",
        "claim_boundary": (
            "Generated compositions were acquired as isolated external "
            "artifacts, protected by fresh retention outcomes, and selected "
            "through an opaque learned router. This tests bounded no-replay "
            "continual artifact growth, not unrestricted program induction or "
            "general continual learning."
        ),
        "seed": args.seed,
        "composition_ids": list(composition_ids),
        "route_mode": route_mode,
        "base_route_count": args.base_route_count,
        "parent_updates": args.parent_updates,
        "artifact_updates": args.artifact_updates,
        "route_updates": args.route_updates,
        "batch_size": args.batch_size,
        "route_batch_size": args.route_batch_size,
        "audit_count": args.audit_count,
        "route_audit_count": route_audit_count,
        "retention_probes": args.retention_probes,
        "artifact_stable_bits_to_threshold": artifact_stable,
        "artifact_behavior": artifact_behavior,
        "retention_outcomes": retention,
        "route_accuracy": route_accuracy,
        "permuted_route_accuracy": permuted_accuracy,
        "cold_start_old_accuracy": cold_start_old_accuracy,
        "shuffled_route_accuracy": shuffled_route_accuracy,
        "shuffled_route_accuracy_samples": shuffled_route_accuracy_samples,
        "reloaded_route_accuracy": reloaded_route_accuracy,
        "physical_rows": len(bank.occupied),
        "protected_rows": int(bank.protection_mask().sum()),
        "corruption_rejected": corruption_rejected,
        "parent_core_digest_before": parent_digest_before,
        "parent_core_digest_after": parent_digest_after,
        "core_unchanged": parent_digest_before == parent_digest_after,
        "histories": histories,
        "accounting": {
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + len(composition_ids)
                * args.artifact_updates
                * args.batch_size
                * (SPAN + 6)
                + len(composition_ids)
                * args.retention_probes
                * args.audit_count
                * SPAN
                + route_accounting["unique_route_verifier_bits"]
            ),
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + len(composition_ids)
                * args.artifact_updates
                * args.batch_size
                * 2
                + len(composition_ids)
                * args.retention_probes
                * args.audit_count
                + route_accounting["unique_route_lifetimes"]
            ),
            "optimizer_updates": args.parent_updates
            + len(composition_ids) * args.artifact_updates,
            "route_optimizer_updates": route_accounting["route_optimizer_updates"],
            "replayed_examples": 0,
            "replayed_route_examples": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "parent_stable": _stable_bits(
                parent_progress,
                threshold=THRESHOLD,
                bits_per_update=args.batch_size * 2,
            )
            is not None,
            "all_artifacts_stable": all(value is not None for value in artifact_stable.values()),
            "all_artifacts_mastered": bool(artifact_behavior)
            and min(artifact_behavior.values()) >= THRESHOLD,
            "all_rows_present": len(bank.occupied) == len(composition_ids),
            "all_rows_protected": int(bank.protection_mask().sum()) == len(composition_ids),
            "route_mastered": route_accuracy >= THRESHOLD,
            "cold_start_old_retained": cold_start_old_accuracy >= THRESHOLD,
            "candidate_permutation_invariant": permuted_accuracy >= THRESHOLD,
            "reward_shuffled_not_mastered": (
                sum(shuffled_route_accuracy_samples)
                / len(shuffled_route_accuracy_samples)
                < 0.65
                and sum(
                    value >= THRESHOLD for value in shuffled_route_accuracy_samples
                )
                <= len(shuffled_route_accuracy_samples) // 3
            ),
            "reloaded_route_preserved": reloaded_route_accuracy >= THRESHOLD,
            "corruption_rejected": corruption_rejected,
            "core_unchanged": parent_digest_before == parent_digest_after,
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
    parser.add_argument("--parent-updates", type=int, default=32)
    parser.add_argument("--artifact-updates", type=int, default=32)
    parser.add_argument("--route-updates", type=int, default=128)
    parser.add_argument("--route-mode", choices=("bank", "append_only"), default="bank")
    parser.add_argument("--base-route-count", type=int, default=2)
    parser.add_argument("--composition-ids", type=int, nargs="+", default=tuple(range(COMPOSITION_COUNT)))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--route-batch-size", type=int, default=8)
    parser.add_argument("--audit-count", type=int, default=16)
    parser.add_argument("--route-audit-count", type=int, default=256)
    parser.add_argument("--retention-probes", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                "promoted": report["promoted"],
                "route_accuracy": report["route_accuracy"],
                "artifact_behavior": report["artifact_behavior"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
