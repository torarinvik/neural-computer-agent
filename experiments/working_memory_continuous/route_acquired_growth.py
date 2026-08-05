"""Route multiple acquired working-memory artifacts through opaque addresses.

The frozen parent emits a learned query from public sensory history. A
memory-side router receives only that query, opaque candidate row keys, an
attempted row, and its scalar outcome during training. It then selects and
verifies a canonical external artifact before the generic growth loader
rehydrates it on the frozen parent.

Procedure/span identities are verifier-private. This experiment tests the
missing memory-mediated step after single-artifact acquisition: discovery and
selection of one of several independently acquired executable states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.archive.unified_cognitive_controller.environment import (
    NULL_ACTION,
)
from experiments.archive.unified_cognitive_controller.legacy_model import (
    UnifiedCognitiveController,
)
from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    evaluate_sequence_memory,
    generate_sequence_memory_batch,
)
from experiments.working_memory_continuous.acquire_frozen_growth import (
    _build_successor,
)
from neural_computer import (
    ExecutableArtifactMemory,
    OpaqueAddressRouter,
    attempted_outcome_loss,
    freeze_core,
    load_growth_artifact,
)


def _load(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _context_query(
    model: UnifiedCognitiveController,
    *,
    seed: int,
    count: int,
    span: int,
    distractors: int,
    device: torch.device,
) -> torch.Tensor:
    """Produce an opaque query after observing the public write history."""
    batch = generate_sequence_memory_batch(
        count,
        span=span,
        distractors=distractors,
        seed=seed,
        operation="mixed",
        heldout=True,
        device=device,
    )
    state = model.initial_state(count, device=device)
    null = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    for index in range(span):
        _, state = model.step(
            batch.input_frames[:, index], state, null, zeros, zeros
        )
    for index in range(distractors):
        _, state = model.step(
            batch.distractor_frames[:, index], state, null, zeros, zeros
        )
    return F.normalize(state.hidden.mean(dim=0), dim=0)


def _queries(
    model: UnifiedCognitiveController,
    spans: tuple[int, ...],
    *,
    per_family: int,
    seed: int,
    distractors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries: list[torch.Tensor] = []
    targets: list[int] = []
    for family, span in enumerate(spans):
        for offset in range(per_family):
            queries.append(
                _context_query(
                    model,
                    seed=seed + family * 10_000 + offset,
                    count=64,
                    span=span,
                    distractors=distractors,
                    device=device,
                )
            )
            targets.append(family)
    return torch.stack(queries), torch.tensor(targets, device=device)


def _random_keys(
    rows: int,
    width: int,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return F.normalize(torch.randn(rows, width, generator=generator), dim=-1).to(
        device
    )


def _train_router(
    keys: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
) -> OpaqueAddressRouter:
    router = OpaqueAddressRouter(width=int(keys.shape[-1]), hidden=64).to(
        keys.device
    )
    optimizer = torch.optim.AdamW(router.parameters(), lr=3e-3, weight_decay=1e-5)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(updates):
        indices = torch.randint(
            queries.shape[0], (batch_size,), generator=generator
        ).to(queries.device)
        attempted = torch.randint(
            keys.shape[0], (batch_size,), generator=generator
        ).to(queries.device)
        outcomes = (attempted == targets[indices]).to(torch.float32)
        if shuffle_outcomes:
            permutation = torch.randperm(
                outcomes.shape[0], generator=generator
            ).to(queries.device)
            outcomes = outcomes[permutation]
        loss = attempted_outcome_loss(
            router(queries[indices], keys), attempted, outcomes
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    router.eval()
    return router


@torch.no_grad()
def _route_accuracy(
    router: OpaqueAddressRouter,
    queries: torch.Tensor,
    targets: torch.Tensor,
    keys: torch.Tensor,
) -> float:
    prediction = router(queries, keys).argmax(dim=-1)
    return float((prediction == targets).float().mean())


@torch.no_grad()
def _permuted_accuracy(
    router: OpaqueAddressRouter,
    queries: torch.Tensor,
    targets: torch.Tensor,
    keys: torch.Tensor,
) -> float:
    permutation = torch.arange(keys.shape[0] - 1, -1, -1, device=keys.device)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(
        permutation.numel(), device=keys.device
    )
    return _route_accuracy(
        router, queries, inverse[targets], keys[permutation]
    )


@torch.no_grad()
def _cosine_accuracy(
    queries: torch.Tensor,
    targets: torch.Tensor,
    keys: torch.Tensor,
) -> float:
    prediction = (F.normalize(queries, dim=-1) @ F.normalize(keys, dim=-1).T).argmax(
        dim=-1
    )
    return float((prediction == targets).float().mean())


def _load_artifacts(
    paths: tuple[Path, ...],
    *,
    device: torch.device,
    destination: Path,
    keys: torch.Tensor,
) -> ExecutableArtifactMemory:
    memory = ExecutableArtifactMemory(
        destination,
        width=int(keys.shape[-1]),
        capacity=len(paths),
        device=device,
    )
    for key, path in zip(keys, paths):
        source = ExecutableArtifactMemory.load(path, device=device)
        rows = source.address_rows()
        if len(rows) != 1:
            raise ValueError("each source acquisition memory must contain one row")
        _, artifact = source.promote_index(rows[0][0])
        memory.put(key.detach().cpu(), artifact)
    memory.validate()
    return ExecutableArtifactMemory.load(destination, device=device)


def _rehydrate(
    parent_payload: dict[str, object],
    artifact: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> UnifiedCognitiveController:
    model, _, _, prefixes = _build_successor(
        parent_payload, device=device, slot_width=256
    )
    freeze_core(model, prefixes)
    receipt = load_growth_artifact(model, artifact, growth_prefixes=prefixes)
    if not receipt.core_unchanged:
        raise RuntimeError("routed growth artifact changed the frozen core")
    model.eval()
    return model


def _corruption_control(
    memory: ExecutableArtifactMemory,
    destination: Path,
    *,
    device: torch.device,
) -> bool:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(memory.directory, destination)
    filename = memory.paths[0]
    if filename is None:
        raise RuntimeError("memory row has no artifact path")
    artifact_path = destination / filename
    artifact_path.write_bytes(artifact_path.read_bytes() + b"corruption")
    try:
        ExecutableArtifactMemory.load(destination, device=device)
    except ValueError as error:
        return "hash mismatch" in str(error)
    return False


def run(args: argparse.Namespace) -> dict[str, object]:
    if (
        len(args.spans) != len(args.artifact_memories)
        or len(args.spans) != len(args.operations)
        or len(args.spans) < 2
    ):
        raise ValueError(
            "spans, operations, and artifact memories must have equal length >= 2"
        )
    if args.train_queries_per_family < 2 or args.test_queries_per_family < 2:
        raise ValueError("each family needs at least two query episodes")
    if min(args.updates, args.batch_size, args.behavior_count) < 1:
        raise ValueError("updates, batch size, and behavior count must be positive")
    if args.behavior_count % 2:
        raise ValueError("behavior count must be even")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    parent_payload = _load(args.parent, device)
    parent = UnifiedCognitiveController(
        **dict(parent_payload["model_configuration"])
    ).to(device)
    parent.load_state_dict(parent_payload["state_dict"], strict=True)
    parent.eval()
    digest_before = _digest(parent)
    for parameter in parent.parameters():
        parameter.requires_grad_(False)

    train_queries, train_targets = _queries(
        parent,
        args.spans,
        per_family=args.train_queries_per_family,
        seed=args.seed + 10_000,
        distractors=args.distractors,
        device=device,
    )
    test_queries, test_targets = _queries(
        parent,
        args.spans,
        per_family=args.test_queries_per_family,
        seed=args.seed + 20_000,
        distractors=args.distractors,
        device=device,
    )
    keys = _random_keys(
        len(args.spans),
        int(train_queries.shape[-1]),
        seed=args.seed + 30_000,
        device=device,
    )
    router = _train_router(
        keys,
        train_queries,
        train_targets,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed + 40_000,
        shuffle_outcomes=False,
    )
    shuffled_router = _train_router(
        keys,
        train_queries,
        train_targets,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed + 50_000,
        shuffle_outcomes=True,
    )

    memory = _load_artifacts(
        tuple(args.artifact_memories),
        device=device,
        destination=args.bank,
        keys=keys,
    )
    restored = ExecutableArtifactMemory.load(args.bank, device=device)
    reload_exact = all(
        torch.equal(left[1], right[1])
        for left, right in zip(memory.address_rows(), restored.address_rows())
    )

    predictions = router(test_queries, keys).argmax(dim=-1)
    shuffled_predictions = shuffled_router(test_queries, keys).argmax(dim=-1)
    selected_rows: list[int] = []
    behavior: dict[str, dict[str, object]] = {}
    for family, span in enumerate(args.spans):
        family_mask = test_targets == family
        family_predictions = predictions[family_mask]
        selected = int(torch.mode(family_predictions).values)
        selected_rows.append(selected)
        _, selected_artifact = restored.promote_index(selected)
        _, wrong_artifact = restored.promote_index((selected + 1) % len(args.spans))
        selected_model = _rehydrate(
            parent_payload, selected_artifact, device=device
        )
        wrong_model = _rehydrate(parent_payload, wrong_artifact, device=device)
        selected_audit = evaluate_sequence_memory(
            selected_model,
            count=args.behavior_count,
            span=span,
            distractors=args.distractors,
            seed=args.seed + 60_000 + family,
            operation=args.operations[family],
            device=device,
        )
        wrong_audit = evaluate_sequence_memory(
            wrong_model,
            count=args.behavior_count,
            span=span,
            distractors=args.distractors,
            seed=args.seed + 60_000 + family,
            operation="complement",
            device=device,
        )
        behavior[str(span)] = {
            "selected_row": selected,
            "expected_row": family,
            "selected_route_accuracy": float(
                (family_predictions == family).float().mean()
            ),
            "selected_artifact_accuracy": selected_audit["accuracy"],
            "wrong_artifact_accuracy": wrong_audit["accuracy"],
            "wrong_address_reduces_accuracy": (
                wrong_audit["accuracy"] < selected_audit["accuracy"] - 0.05
            ),
        }
    digest_after = _digest(parent)
    corruption_rejected = _corruption_control(
        restored,
        args.report.parent / "bank_corrupted",
        device=device,
    )
    report = {
        "schema": "opaque-address-routed-growth-audit-v1",
        "claim_boundary": (
            "The frozen parent emits opaque queries. A replaceable memory-side "
            "router sees candidate opaque addresses, attempted row indices, "
            "and scalar outcomes. Procedure identities and correct rows remain "
            "verifier-private. The controller receives no router branch."),
        "parent": str(args.parent),
        "artifact_memories": [str(path) for path in args.artifact_memories],
        "bank": str(args.bank),
        "spans_private_to_verifier": list(args.spans),
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "train_queries_per_family": args.train_queries_per_family,
        "test_queries_per_family": args.test_queries_per_family,
        "distractors": args.distractors,
        "verifier_bits": args.updates * args.batch_size,
        "address_width": int(keys.shape[-1]),
        "normal_route_accuracy": _route_accuracy(
            router, test_queries, test_targets, keys
        ),
        "reward_shuffled_route_accuracy": _route_accuracy(
            shuffled_router, test_queries, test_targets, keys
        ),
        "candidate_permutation_accuracy": _permuted_accuracy(
            router, test_queries, test_targets, keys
        ),
        "cosine_baseline_accuracy": _cosine_accuracy(
            test_queries, test_targets, keys
        ),
        "shuffled_route_predictions": shuffled_predictions.detach().cpu().tolist(),
        "selected_rows": selected_rows,
        "bank_reload_exact": reload_exact,
        "controller_weights_unchanged": digest_before == digest_after,
        "corruption_rejected": corruption_rejected,
        "behavior": behavior,
        "gates": {
            "normal_route_at_least_90": _route_accuracy(
                router, test_queries, test_targets, keys
            ) >= 0.90,
            "reward_shuffled_route_near_chance": _route_accuracy(
                shuffled_router, test_queries, test_targets, keys
            ) <= 0.75,
            "candidate_permutation_invariant": _permuted_accuracy(
                router, test_queries, test_targets, keys
            ) >= 0.90,
            "cosine_baseline_near_chance": _cosine_accuracy(
                test_queries, test_targets, keys
            ) <= 0.75,
            "both_rows_selected": selected_rows == list(range(len(args.spans))),
            "all_wrong_address_controls_causal": all(
                bool(item["wrong_address_reduces_accuracy"])
                for item in behavior.values()
            ),
            "bank_reload_exact": reload_exact,
            "controller_frozen": digest_before == digest_after,
            "corruption_rejected": corruption_rejected,
        },
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--artifact-memories", type=Path, nargs="+", required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=62001)
    parser.add_argument("--spans", type=int, nargs="+", default=(2, 3))
    parser.add_argument(
        "--operations",
        choices=(
            "mixed", "forward", "reverse", "complement",
            "complement_reverse", "complement_rotate", "adjacent_xor",
            "prefix_parity", "global_parity", "rotate", "undo_complement",
            "producer_global_parity",
        ),
        nargs="+",
        default=("complement", "rotate"),
    )
    parser.add_argument("--train-queries-per-family", type=int, default=16)
    parser.add_argument("--test-queries-per-family", type=int, default=32)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--behavior-count", type=int, default=64)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    args = parser.parse_args()
    report = run(args)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "normal_route_accuracy",
                    "reward_shuffled_route_accuracy",
                    "candidate_permutation_accuracy",
                    "cosine_baseline_accuracy",
                    "selected_rows",
                    "accepted_diagnostic",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
