"""Audit one protected external-capability append without replay.

The audit starts with the promoted three-program external bank, fills its
capacity, and records fresh retention outcomes for every row.  A new
capability must then encounter an explicit protected-capacity failure.  The
bank is grown into a new directory, the new artifact is appended, and a
memory-side route extension is trained only on fresh queries for that
capability.  The parent controller and the old router are never updated after
the append point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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

APPENDED_PROGRAM = ("rotate4", "rotate", 4)
ALL_PROGRAMS = (*PROGRAMS, APPENDED_PROGRAM)
EVENT_WIDTH = 32
ACTION_WIDTH = 2
INTENTION_WIDTH = 16
ROUTE_WIDTH = 48
CAPABILITY_CONTEXT_HIDDEN = 64
CAPABILITY_CONTEXT_WIDTH = 32
CAPABILITY_ADAPTER_HIDDEN = 64
DECODER_HIDDEN = 16


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _new_external_capability(
    parent,
    *,
    operation: str,
    updates: int,
    batch_size: int,
    seed: int,
    audit_count: int,
    eval_every: int,
    learning_rate: float,
):
    program, decoder = _new_capability(seed)
    history, progress = _train_capability(
        parent,
        program,
        decoder,
        operation=operation,
        span=4,
        updates=updates,
        batch_size=batch_size,
        seed=seed + 20_000,
        audit_count=audit_count,
        eval_every=eval_every,
        learning_rate=learning_rate,
    )
    return program, decoder, history, progress


def _train_route_extension(
    parent,
    base_router: OpaqueAddressRouter,
    old_keys: torch.Tensor,
    *,
    operation: str,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
) -> tuple[OpaqueViewRouteExtension, dict[str, int | float]]:
    """Train one append-only route from paired fresh scalar outcomes."""

    extension = OpaqueViewRouteExtension(width=ROUTE_WIDTH, hidden=64)
    optimizer = torch.optim.AdamW(
        extension.parameters(), lr=3e-3, weight_decay=1e-5
    )
    base_router.eval()
    for parameter in base_router.parameters():
        parameter.requires_grad_(False)
    extension.train()
    last_loss = 0.0
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
            # Present both outcome orderings for the same fresh query.  The
            # shuffled control then has exactly zero expected route credit,
            # rather than allowing optimizer noise to make the new row win.
            queries = queries.repeat_interleave(2, dim=0)
        with torch.no_grad():
            old_best = base_router(queries, old_keys).max(dim=-1).values
        delta = extension(queries)
        candidate_scores = torch.stack((old_best, old_best + delta), dim=1)
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
            candidate_scores,
            attempted,
            utilities,
        )
        # Keep the null arm calibrated: an uninformative extension must not
        # preempt an established route merely because it was appended later.
        loss = loss + 0.10 * delta.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(extension.parameters(), 1.0)
        optimizer.step()
        last_loss = float(loss.detach())
    extension.eval()
    return extension, {
        "unique_route_lifetimes": updates * batch_size,
        "unique_route_verifier_bits": updates * batch_size * 2,
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
    *,
    failed_old: bool | torch.Tensor,
) -> torch.Tensor:
    return failure_gated_view_scores(
        base_router(queries, old_keys),
        extension(queries),
        failed_old,
    )


def _retention_probe(
    parent,
    bank: ExecutableArtifactMemory,
    key: torch.Tensor,
    *,
    row: int,
    operation: str,
    audit_count: int,
    probes: int,
    seed: int,
) -> list[float]:
    outcomes: list[float] = []
    for probe in range(probes):
        _, artifact = bank.promote_index(row)
        program, decoder = _load_artifact(artifact)
        outcomes.append(
            _capability_accuracy(
                parent,
                program,
                decoder,
                operation=operation,
                span=4,
                count=audit_count,
                seed=seed + probe * 101,
            )
        )
        bank.observe_retention(key, outcomes[-1])
    return outcomes


def _route_accuracy(
    base_router: OpaqueAddressRouter,
    extension: OpaqueViewRouteExtension,
    old_queries: torch.Tensor,
    old_targets: torch.Tensor,
    new_queries: torch.Tensor,
    *,
    old_keys: torch.Tensor,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    old_scores = base_router(old_queries, old_keys)
    old_predictions = old_scores.argmax(dim=-1)
    new_scores = _scores_with_extension(
        base_router,
        extension,
        new_queries,
        old_keys,
        failed_old=True,
    )
    new_predictions = new_scores.argmax(dim=-1)
    old_accuracy = float((old_predictions == old_targets).float().mean())
    new_accuracy = float(
        (new_predictions == len(PROGRAMS)).float().mean()
    )
    return old_accuracy, new_accuracy, old_predictions, new_predictions


@torch.no_grad()
def _permuted_route_accuracy(
    base_router: OpaqueAddressRouter,
    extension: OpaqueViewRouteExtension,
    old_queries: torch.Tensor,
    old_targets: torch.Tensor,
    new_queries: torch.Tensor,
    old_keys: torch.Tensor,
) -> float:
    permutation = torch.tensor([2, 0, 1], dtype=torch.long)
    permuted_keys = old_keys[permutation]
    old_predictions = base_router(old_queries, permuted_keys).argmax(dim=-1)
    old_predictions = permutation[old_predictions]
    new_predictions = _scores_with_extension(
        base_router,
        extension,
        new_queries,
        permuted_keys,
        failed_old=True,
    ).argmax(dim=-1)
    predictions = torch.cat((old_predictions, new_predictions))
    targets = torch.cat(
        (old_targets, torch.full_like(new_predictions, len(PROGRAMS)))
    )
    return float((predictions == targets).float().mean())


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if min(
        args.parent_updates,
        args.updates,
        args.route_updates,
        args.extension_updates,
        args.batch_size,
        args.route_batch_size,
        args.audit_count,
        args.retention_probes,
    ) < 1:
        raise ValueError("all update and audit counts must be positive")
    if args.batch_size % 2 or args.route_batch_size % 2 or args.audit_count % 2:
        raise ValueError("batch sizes and audit count must be even")

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

    bank_path = root / "full_bank"
    grown_path = root / "grown_bank"
    for path in (bank_path, grown_path):
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
    base_artifact_rows: list[int] = []
    for index, (label, operation, span) in enumerate(PROGRAMS):
        program, decoder, history, progress = _new_external_capability(
            parent,
            operation=operation,
            updates=args.updates,
            batch_size=args.batch_size,
            seed=args.seed + index + 1,
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
        if not receipt.accepted or receipt.index is None:
            raise RuntimeError(f"base capability admission failed: {receipt.reason}")
        base_artifact_rows.append(receipt.index)
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

    base_retention_observations: dict[str, list[float]] = {}
    for index, (_label, operation, _span) in enumerate(PROGRAMS):
        base_retention_observations[operation] = _retention_probe(
            parent,
            bank,
            route_keys[index],
            row=base_artifact_rows[index],
            operation=operation,
            audit_count=args.audit_count,
            probes=args.retention_probes,
            seed=args.seed + 30_000 + index * 100,
        )
    bank.save()
    protected_before_append = [
        bank.retention.is_protected(key) for key in route_keys
    ]

    old_candidates = bank.address_rows()
    old_keys = torch.stack([key for _, key in old_candidates])
    base_router_result = _train_base_router(
        parent,
        old_keys,
        updates=args.route_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 60_000,
    )
    base_router = base_router_result["router"]
    base_router_digest_before_append = _digest_module(base_router)

    new_program, new_decoder, new_history, new_progress = _new_external_capability(
        parent,
        operation=APPENDED_PROGRAM[1],
        updates=args.append_updates,
        batch_size=args.batch_size,
        seed=args.seed + 10_001,
        audit_count=args.audit_count,
        eval_every=args.eval_every,
        learning_rate=args.learning_rate,
    )
    new_artifact = _artifact(new_program, new_decoder)
    new_key = F.normalize(
        _route_queries(
            parent,
            operation=APPENDED_PROGRAM[1],
            span=APPENDED_PROGRAM[2],
            count=args.audit_count,
            seed=args.seed + 55_000,
        ).mean(dim=0),
        dim=0,
    )
    admission_plan = lifecycle.plan_admission(new_key, new_artifact)
    capacity_blocked = admission_plan.action == "grow" and all(
        lifecycle.protection_mask().tolist()
    )
    capacity_error = (
        "all occupied rows are protected; lifecycle selected growth"
        if capacity_blocked
        else "lifecycle did not select protected-capacity growth"
    )
    if not capacity_blocked:
        raise RuntimeError(capacity_error)

    growth_receipt = lifecycle.admit(
        new_key,
        new_artifact,
        plan=admission_plan,
        grow_destination=grown_path,
    )
    if not growth_receipt.accepted:
        raise RuntimeError(f"growth admission failed: {growth_receipt.reason}")
    grown = lifecycle.memory
    grown.retention.config = RetentionPolicyConfig(
        mastery_threshold=args.retention_threshold,
        min_mastery_observations=args.retention_probes,
    )
    new_row = len(PROGRAMS)
    new_retention_observations = _retention_probe(
        parent,
        grown,
        new_key,
        row=new_row,
        operation=APPENDED_PROGRAM[1],
        audit_count=args.audit_count,
        probes=args.retention_probes,
        seed=args.seed + 40_000,
    )
    grown.save()
    protected_after_growth = [
        grown.retention.is_protected(key) for key in route_keys
    ]
    new_protected_after_append = grown.retention.is_protected(new_key)

    extension, extension_accounting = _train_route_extension(
        parent,
        base_router,
        old_keys,
        operation=APPENDED_PROGRAM[1],
        updates=args.extension_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 70_000,
        shuffle_outcomes=False,
    )
    shuffled_extension, shuffled_accounting = _train_route_extension(
        parent,
        base_router,
        old_keys,
        operation=APPENDED_PROGRAM[1],
        updates=args.extension_updates,
        batch_size=args.route_batch_size,
        seed=args.seed + 80_000,
        shuffle_outcomes=True,
    )
    base_router_digest_after_append = _digest_module(base_router)
    extension_digest = _digest_module(extension)

    old_queries, old_targets = _test_queries(
        parent,
        count=args.audit_count,
        seed=args.seed + 110_000,
    )
    new_queries = _route_queries(
        parent,
        operation=APPENDED_PROGRAM[1],
        span=4,
        count=args.audit_count * len(PROGRAMS),
        seed=args.seed + 120_000,
    )
    old_route, new_route, old_predictions, new_predictions = _route_accuracy(
        base_router,
        extension,
        old_queries,
        old_targets,
        new_queries,
        old_keys=old_keys,
    )
    combined_route = float(
        torch.cat((
            (old_predictions == old_targets).float(),
            (new_predictions == len(PROGRAMS)).float(),
        )).mean()
    )
    permuted_route = _permuted_route_accuracy(
        base_router,
        extension,
        old_queries,
        old_targets,
        new_queries,
        old_keys,
    )
    shuffled_new_selection = float(
        (
            _scores_with_extension(
                base_router,
                shuffled_extension,
                new_queries,
                old_keys,
                failed_old=True,
            ).argmax(dim=-1)
            == len(PROGRAMS)
        )
        .float()
        .mean()
    )

    selected_rows: dict[str, int] = {}
    for family, (label, _operation, _span) in enumerate(PROGRAMS):
        block = old_queries[
            family * args.audit_count : (family + 1) * args.audit_count
        ]
        selected_rows[label] = int(
            torch.mode(base_router(block, old_keys).argmax(dim=-1)).values
        )
    selected_rows[APPENDED_PROGRAM[0]] = int(torch.mode(new_predictions).values)
    selected_behavior: dict[str, float] = {}
    wrong_behavior: dict[str, float] = {}
    for label, operation, _span in ALL_PROGRAMS:
        row = selected_rows[label]
        _, artifact = grown.promote_index(row)
        program, decoder = _load_artifact(artifact)
        seed = args.seed + 140_000 + row
        selected_behavior[label] = _capability_accuracy(
            parent,
            program,
            decoder,
            operation=operation,
            span=4,
            count=args.audit_count,
            seed=seed,
        )
        wrong_row = (row + 1) % len(ALL_PROGRAMS)
        _, wrong_artifact = grown.promote_index(wrong_row)
        wrong_program, wrong_decoder = _load_artifact(wrong_artifact)
        wrong_behavior[label] = _capability_accuracy(
            parent,
            wrong_program,
            wrong_decoder,
            operation=operation,
            span=4,
            count=args.audit_count,
            seed=seed,
        )

    router_path = root / "frozen_base_router.pt"
    extension_path = root / "appended_route_extension.pt"
    base_router_state = PersistentOpaqueStateStore(
        router_path,
        configuration={
            "component": "sequential-external-base-route",
            "schema": "neural-computer.opaque-address-router.v1",
            "width": ROUTE_WIDTH,
            "hidden": 64,
            "candidate_count": len(PROGRAMS),
        },
    )
    extension_state = PersistentOpaqueStateStore(
        extension_path,
        configuration={
            "component": "sequential-external-route-extension",
            "schema": "neural-computer.opaque-view-route-extension.v1",
            "width": ROUTE_WIDTH,
            "hidden": 64,
        },
    )
    base_router_state.save_module(base_router)
    extension_state.save_module(extension)
    reloaded_bank = ExecutableArtifactMemory.load(grown_path)
    reloaded_router = OpaqueAddressRouter(width=ROUTE_WIDTH, hidden=64)
    base_router_state.load_module(reloaded_router)
    reloaded_router.eval()
    reloaded_extension = OpaqueViewRouteExtension(width=ROUTE_WIDTH, hidden=64)
    extension_state.load_module(reloaded_extension)
    reloaded_extension.eval()
    reloaded_candidates = reloaded_bank.address_rows()
    reloaded_keys = torch.stack([key for _, key in reloaded_candidates])
    reloaded_old_route, reloaded_new_route, _, _ = _route_accuracy(
        reloaded_router,
        reloaded_extension,
        old_queries,
        old_targets,
        new_queries,
        old_keys=reloaded_keys[: len(PROGRAMS)],
    )
    reloaded_candidate_exact = all(
        torch.equal(reloaded_keys[index], key)
        for index, key in enumerate(torch.stack((*route_keys, new_key)))
    )
    reloaded_behavior: dict[str, float] = {}
    for label, operation, _span in ALL_PROGRAMS:
        row = selected_rows[label]
        _, artifact = reloaded_bank.promote_index(row)
        program, decoder = _load_artifact(artifact)
        reloaded_behavior[label] = _capability_accuracy(
            parent,
            program,
            decoder,
            operation=operation,
            span=4,
            count=args.audit_count,
            seed=args.seed + 140_000 + row,
        )

    artifact_name = reloaded_bank.paths[0]
    if artifact_name is None:
        raise RuntimeError("reloaded bank has no artifact path")
    artifact_path = grown_path / artifact_name
    intact_payload = artifact_path.read_bytes()
    artifact_path.write_bytes(intact_payload + b"corruption")
    corruption_rejected = False
    try:
        ExecutableArtifactMemory.load(grown_path)
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    artifact_path.write_bytes(intact_payload)

    parent_digest_after = _digest_core(parent, ())
    capability_bits = args.updates * args.batch_size * sum(
        span + 2 for _, _, span in PROGRAMS
    )
    append_bits = args.append_updates * args.batch_size * (APPENDED_PROGRAM[2] + 2)
    report = {
        "schema": "neural-computer.sequential-external-capability-report.v1",
        "claim_boundary": (
            "One fresh external capability is appended after a protected three-"
            "artifact bank reaches capacity. The bank grows transactionally; "
            "the parent controller, old router, and old artifacts remain frozen; "
            "the new route learns from fresh scalar outcomes only. This is not "
            "unrestricted continual learning."
        ),
        "seed": args.seed,
        "base_programs": [
            {"label": label, "operation": operation, "span": span}
            for label, operation, span in PROGRAMS
        ],
        "appended_program": {
            "label": APPENDED_PROGRAM[0],
            "operation": APPENDED_PROGRAM[1],
            "span": APPENDED_PROGRAM[2],
        },
        "parent_updates": args.parent_updates,
        "updates_per_base_artifact": args.updates,
        "append_updates": args.append_updates,
        "route_updates": args.route_updates,
        "extension_updates": args.extension_updates,
        "batch_size": args.batch_size,
        "route_batch_size": args.route_batch_size,
        "audit_count": args.audit_count,
        "retention_probes": args.retention_probes,
        "retention_threshold": args.retention_threshold,
        "parent_stable_bits_to_threshold": parent_stable_bits,
        "stable_bits_to_threshold": stable_bits,
        "new_stable_bits_to_threshold": _stable_bits(
            new_progress,
            threshold=0.75,
            bits_per_update=args.batch_size * APPENDED_PROGRAM[2],
        ),
        "base_retention_observations": base_retention_observations,
        "new_retention_observations": new_retention_observations,
        "protected_before_append": protected_before_append,
        "protected_after_growth": protected_after_growth,
        "new_protected_after_append": new_protected_after_append,
        "capacity_blocked": capacity_blocked,
        "capacity_error": capacity_error,
        "physical_rows_before_growth": len(bank.occupied),
        "physical_rows_after_growth": len(grown.occupied),
        "old_route_accuracy": old_route,
        "new_route_accuracy": new_route,
        "combined_route_accuracy": combined_route,
        "candidate_permutation_accuracy": permuted_route,
        "reward_shuffled_new_selection_rate": shuffled_new_selection,
        "selected_rows": selected_rows,
        "selected_behavior": selected_behavior,
        "wrong_behavior": wrong_behavior,
        "reloaded_old_route_accuracy": reloaded_old_route,
        "reloaded_new_route_accuracy": reloaded_new_route,
        "reloaded_candidate_exact": reloaded_candidate_exact,
        "reloaded_behavior": reloaded_behavior,
        "corruption_rejected": corruption_rejected,
        "parent_core_digest_before": parent_digest_before,
        "parent_core_digest_after": parent_digest_after,
        "base_router_digest_before_append": base_router_digest_before_append,
        "base_router_digest_after_append": base_router_digest_after_append,
        "extension_digest": extension_digest,
        "histories": {
            **histories,
            APPENDED_PROGRAM[0]: {
                "history": new_history,
                "progress": new_progress,
            },
        },
        "accounting": {
            "unique_logical_lifetimes": (
                args.parent_updates * args.batch_size
                + args.updates * args.batch_size * len(PROGRAMS) * 2
                + args.append_updates * args.batch_size * 2
            ),
            "unique_verifier_bits": (
                args.parent_updates * args.batch_size * 2
                + capability_bits
                + append_bits
            ),
            "optimizer_updates": (
                args.parent_updates
                + args.updates * len(PROGRAMS)
                + args.append_updates
            ),
            "base_route_optimizer_updates": base_router_result["accounting"][
                "route_optimizer_updates"
            ],
            "extension_route_optimizer_updates": extension_accounting[
                "route_optimizer_updates"
            ],
            "shuffled_extension_route_optimizer_updates": shuffled_accounting[
                "route_optimizer_updates"
            ],
            "route_unique_verifier_bits": (
                base_router_result["accounting"]["unique_route_verifier_bits"]
                + extension_accounting["unique_route_verifier_bits"]
                + shuffled_accounting["unique_route_verifier_bits"]
            ),
            "retention_verifier_bits": (
                (len(PROGRAMS) + 1)
                * args.retention_probes
                * args.audit_count
                * 4
            ),
            "replayed_examples": 0,
            "replayed_route_examples_after_append": 0,
            "wall_seconds": perf_counter() - started,
        },
        "gates": {
            "base_artifacts_present": len(bank.occupied) == len(PROGRAMS),
            "base_capabilities_stable": all(
                stable_bits[label] is not None for label, _, _ in PROGRAMS
            ),
            "all_base_rows_protected_before_append": all(
                protected_before_append
            ),
            "capacity_blocked_without_eviction": capacity_blocked,
            "growth_preserved_protected_rows": all(protected_after_growth),
            "four_artifacts_after_growth": len(grown.occupied) == len(ALL_PROGRAMS),
            "appended_capability_stable": (
                _stable_bits(
                    new_progress,
                    threshold=0.75,
                    bits_per_update=args.batch_size * APPENDED_PROGRAM[2],
                )
                is not None
            ),
            "appended_row_protected": new_protected_after_append,
            "old_route_mastered": old_route >= 0.90,
            "new_route_mastered": new_route >= 0.75,
            "combined_route_at_least_80": combined_route >= 0.80,
            "candidate_permutation_invariant": permuted_route >= 0.80,
            "reward_shuffled_extension_near_chance": shuffled_new_selection <= 0.50,
            "all_selected_capabilities_mastered": min(selected_behavior.values())
            >= 0.70,
            "wrong_artifact_is_causal": all(
                selected_behavior[label] > wrong_behavior[label] + 0.05
                for label, _, _ in ALL_PROGRAMS
            ),
            "reloaded_route_preserved": (
                reloaded_old_route >= old_route - 0.05
                and reloaded_new_route >= new_route - 0.05
            ),
            "reloaded_candidate_exact": reloaded_candidate_exact,
            "reloaded_behavior_preserved": all(
                reloaded_behavior[label] >= selected_behavior[label] - 0.05
                for label, _, _ in ALL_PROGRAMS
            ),
            "parent_core_unchanged": parent_digest_before == parent_digest_after,
            "base_router_frozen_during_append": (
                base_router_digest_before_append == base_router_digest_after_append
            ),
            "corruption_rejected": corruption_rejected,
            "no_replayed_examples_after_append": True,
        },
    }
    report["promoted"] = all(report["gates"].values())
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def _train_base_router(
    parent,
    candidate_keys: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    """Train the frozen three-row router through the canonical bank helper."""

    from experiments.parent_conditioned_artifact_bank_amodal.train import (
        _train_router,
    )

    return _train_router(
        parent,
        candidate_keys,
        updates=updates,
        batch_size=batch_size,
        seed=seed,
        shuffle_outcomes=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--parent-updates", type=int, default=128)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--append-updates", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--route-updates", type=int, default=1024)
    parser.add_argument("--extension-updates", type=int, default=512)
    parser.add_argument("--route-batch-size", type=int, default=16)
    parser.add_argument("--audit-count", type=int, default=64)
    parser.add_argument("--retention-probes", type=int, default=8)
    parser.add_argument("--retention-threshold", type=float, default=0.75)
    parser.add_argument("--eval-every", type=int, default=32)
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
                "capacity_blocked": report["capacity_blocked"],
                "old_route_accuracy": report["old_route_accuracy"],
                "new_route_accuracy": report["new_route_accuracy"],
                "combined_route_accuracy": report["combined_route_accuracy"],
                "candidate_permutation_accuracy": report[
                    "candidate_permutation_accuracy"
                ],
                "reward_shuffled_new_selection_rate": report[
                    "reward_shuffled_new_selection_rate"
                ],
                "selected_behavior": report["selected_behavior"],
                "wrong_behavior": report["wrong_behavior"],
                "gates": report["gates"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
