"""Route two executable procedures with the same frozen parent and schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

from experiments.archive.unified_cognitive_controller.environment import NULL_ACTION
from experiments.archive.unified_cognitive_controller.legacy_model import (
    UnifiedCognitiveController,
)
from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    evaluate_sequence_memory,
    generate_sequence_memory_batch,
)
from experiments.working_memory_continuous.route_acquired_growth import (
    _cosine_accuracy,
    _permuted_accuracy,
    _random_keys,
    _route_accuracy,
)
from neural_computer import (
    ExecutableArtifactMemory,
    OpaqueAddressRouter,
    attempted_outcome_loss,
    freeze_core,
    load_growth_artifact,
)

QUERY_BATCH = 64


def _load(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _operation_query(
    model: UnifiedCognitiveController,
    *,
    seed: int,
    count: int,
    span: int,
    operation: str,
    distractors: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode public history plus its public operation cue as an opaque query."""
    batch = generate_sequence_memory_batch(
        count,
        span=span,
        distractors=distractors,
        seed=seed,
        operation=operation,
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
    _, state = model.step(
        batch.query_frames[:, 0], state, null, zeros, zeros
    )
    return torch.nn.functional.normalize(state.hidden.mean(dim=0), dim=0)


def _queries(
    model: UnifiedCognitiveController,
    operations: tuple[str, ...],
    *,
    per_family: int,
    seed: int,
    distractors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    queries: list[torch.Tensor] = []
    targets: list[int] = []
    for family, operation in enumerate(operations):
        for offset in range(per_family):
            queries.append(
                _operation_query(
                    model,
                    seed=seed + family * 10_000 + offset,
                    count=QUERY_BATCH,
                    span=10,
                    operation=operation,
                    distractors=distractors,
                    device=device,
                )
            )
            targets.append(family)
    return torch.stack(queries), torch.tensor(targets, device=device)


def _common_artifacts(
    parent: dict[str, object],
    candidate: dict[str, object],
) -> tuple[dict[str, object], tuple[dict[str, torch.Tensor], ...]]:
    configuration = dict(candidate["model_configuration"])
    common = UnifiedCognitiveController(**configuration)
    result = common.load_state_dict(parent["state_dict"], strict=False)
    growth_names = {
        name for name in common.state_dict() if name.startswith("skill_")
    }
    if set(result.missing_keys) != {name for name in growth_names if name not in parent["state_dict"]}:
        raise RuntimeError("common parent state does not match growth schema")
    old = {
        name: torch.zeros_like(value)
        for name, value in common.state_dict().items()
        if name in growth_names
    }
    new = {name: value.detach().cpu().clone() for name, value in old.items()}
    for name, value in parent["state_dict"].items():
        if name in old:
            old[name] = value.detach().cpu().clone()
    for name, value in candidate["state_dict"].items():
        if name in new:
            new[name] = value.detach().cpu().clone()
    return configuration, (old, new)


def _train_router_accounted(
    keys: torch.Tensor,
    queries: torch.Tensor,
    targets: torch.Tensor,
    *,
    updates: int,
    batch_size: int,
    seed: int,
    shuffle_outcomes: bool,
) -> tuple[OpaqueAddressRouter, dict[str, int]]:
    router = OpaqueAddressRouter(width=int(keys.shape[-1]), hidden=64).to(
        keys.device
    )
    optimizer = torch.optim.AdamW(
        router.parameters(), lr=3e-3, weight_decay=1e-5
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    unique_feedback: set[tuple[int, int]] = set()
    total_feedback = 0
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
        unique_feedback.update(
            zip(indices.tolist(), attempted.tolist())
        )
        total_feedback += batch_size
        loss = attempted_outcome_loss(
            router(queries[indices], keys), attempted, outcomes
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    router.eval()
    return router, {
        "verifier_bits": total_feedback,
        "unique_verifier_bits": len(unique_feedback),
        "replayed_examples": total_feedback - len(unique_feedback),
    }


def _rehydrate(
    parent: dict[str, object],
    configuration: dict[str, object],
    artifact: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> UnifiedCognitiveController:
    model = UnifiedCognitiveController(**configuration).to(device)
    result = model.load_state_dict(parent["state_dict"], strict=False)
    expected = {
        name for name in model.state_dict() if name.startswith("skill_")
    }
    if set(result.missing_keys) != {
        name for name in expected if name not in parent["state_dict"]
    }:
        raise RuntimeError("rehydration parent state mismatch")
    freeze_core(model, ("skill_",))
    receipt = load_growth_artifact(
        model,
        {name: value.to(device) for name, value in artifact.items()},
        growth_prefixes=("skill_",),
    )
    if not receipt.core_unchanged:
        raise RuntimeError("routed procedure changed the frozen core")
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
    path = destination / filename
    path.write_bytes(path.read_bytes() + b"corruption")
    try:
        ExecutableArtifactMemory.load(destination, device=device)
    except ValueError as error:
        return "hash mismatch" in str(error)
    return False


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.behavior_count < 2 or args.behavior_count % 2:
        raise ValueError("behavior-count must be positive and even")
    started = time.perf_counter()
    device = torch.device(args.device)
    parent = _load(args.parent, device)
    candidate = _load(args.candidate, device)
    configuration, artifacts = _common_artifacts(parent, candidate)
    parent_model = UnifiedCognitiveController(
        **dict(parent["model_configuration"])
    ).to(device)
    parent_model.load_state_dict(parent["state_dict"], strict=True)
    parent_model.eval()
    digest_before = _digest(parent_model)
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    operations = ("mixed", "complement")
    train_queries, train_targets = _queries(
        parent_model, operations, per_family=args.train_queries_per_family,
        seed=args.seed + 10_000, distractors=args.distractors, device=device
    )
    test_queries, test_targets = _queries(
        parent_model, operations, per_family=args.test_queries_per_family,
        seed=args.seed + 20_000, distractors=args.distractors, device=device
    )
    keys = _random_keys(2, int(train_queries.shape[-1]), seed=args.seed + 30_000, device=device)
    router, accounting = _train_router_accounted(
        keys, train_queries, train_targets, updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 40_000,
        shuffle_outcomes=False,
    )
    shuffled_router, shuffled_accounting = _train_router_accounted(
        keys, train_queries, train_targets, updates=args.updates,
        batch_size=args.batch_size, seed=args.seed + 50_000,
        shuffle_outcomes=True,
    )
    memory = ExecutableArtifactMemory(
        args.bank, width=int(keys.shape[-1]), capacity=2, device=device
    )
    for key, artifact in zip(keys, artifacts):
        memory.put(key.detach().cpu(), artifact)
    memory.validate()
    restored = ExecutableArtifactMemory.load(args.bank, device=device)
    reload_exact = all(
        torch.equal(left[1], right[1])
        for left, right in zip(memory.address_rows(), restored.address_rows())
    )
    prediction = router(test_queries, keys).argmax(dim=-1)
    selected_rows: list[int] = []
    behavior: dict[str, dict[str, object]] = {}
    for family, operation in enumerate(operations):
        family_prediction = prediction[test_targets == family]
        selected = int(torch.mode(family_prediction).values)
        selected_rows.append(selected)
        _, selected_artifact = restored.promote_index(selected)
        _, wrong_artifact = restored.promote_index(1 - selected)
        zero_artifact = {
            name: torch.zeros_like(value)
            for name, value in selected_artifact.items()
        }
        selected_model = _rehydrate(
            parent, configuration, selected_artifact, device=device
        )
        wrong_model = _rehydrate(
            parent, configuration, wrong_artifact, device=device
        )
        zero_model = _rehydrate(
            parent, configuration, zero_artifact, device=device
        )
        selected_audit = evaluate_sequence_memory(
            selected_model, count=args.behavior_count, span=10,
            distractors=args.distractors, seed=args.seed + 60_000 + family,
            operation=operation, device=device,
        )
        wrong_audit = evaluate_sequence_memory(
            wrong_model, count=args.behavior_count, span=10,
            distractors=args.distractors, seed=args.seed + 60_000 + family,
            operation=operation, device=device,
        )
        zero_audit = evaluate_sequence_memory(
            zero_model, count=args.behavior_count, span=10,
            distractors=args.distractors, seed=args.seed + 60_000 + family,
            operation=operation, device=device,
        )
        behavior[operation] = {
            "selected_row": selected,
            "expected_row": family,
            "selected_route_accuracy": float(
                (family_prediction == family).float().mean()
            ),
            "selected_artifact_accuracy": selected_audit["accuracy"],
            "wrong_artifact_accuracy": wrong_audit["accuracy"],
            "zero_artifact_accuracy": zero_audit["accuracy"],
            "wrong_real_address_reduces_accuracy": (
                wrong_audit["accuracy"] < selected_audit["accuracy"] - 0.05
            ),
            "selected_artifact_is_causal": (
                zero_audit["accuracy"] < selected_audit["accuracy"] - 0.05
            ),
        }
    digest_after = _digest(parent_model)
    corruption_rejected = _corruption_control(
        restored, args.report.parent / "bank_corrupted", device=device
    )
    normal_route = _route_accuracy(router, test_queries, test_targets, keys)
    shuffled_route = _route_accuracy(
        shuffled_router, test_queries, test_targets, keys
    )
    report = {
        "schema": "opaque-address-routed-mixed-procedure-audit-v1",
        "claim_boundary": (
            "The frozen parent emits opaque queries after public sensory and "
            "operation-cue events. A replaceable memory-side router sees only "
            "opaque row keys, attempted rows, and scalar outcomes. The two "
            "procedure identities and correct rows remain verifier-private."),
        "parent": str(args.parent),
        "candidate": str(args.candidate),
        "bank": str(args.bank),
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "verifier_bits": accounting["verifier_bits"],
        "unique_verifier_bits": accounting["unique_verifier_bits"],
        "unique_logical_lifetimes": int(
            train_queries.shape[0] * QUERY_BATCH
        ),
        "optimizer_updates": args.updates,
        "replayed_examples": accounting["replayed_examples"],
        "reward_shuffled_verifier_bits": shuffled_accounting[
            "verifier_bits"
        ],
        "wall_seconds": time.perf_counter() - started,
        "normal_route_accuracy": normal_route,
        "reward_shuffled_route_accuracy": shuffled_route,
        "candidate_permutation_accuracy": _permuted_accuracy(
            router, test_queries, test_targets, keys
        ),
        "cosine_baseline_accuracy": _cosine_accuracy(
            test_queries, test_targets, keys
        ),
        "selected_rows": selected_rows,
        "bank_reload_exact": reload_exact,
        "controller_weights_unchanged": digest_before == digest_after,
        "corruption_rejected": corruption_rejected,
        "behavior": behavior,
        "gates": {
            "normal_route_at_least_90": normal_route >= 0.90,
            "reward_shuffled_route_near_chance": shuffled_route <= 0.75,
            "candidate_permutation_invariant": _permuted_accuracy(
                router, test_queries, test_targets, keys
            ) >= 0.90,
            "cosine_baseline_near_chance": _cosine_accuracy(
                test_queries, test_targets, keys
            ) <= 0.75,
            "both_rows_selected": selected_rows == [0, 1],
            "all_selected_artifacts_causal": all(
                bool(item["selected_artifact_is_causal"])
                for item in behavior.values()
            ),
            "bank_reload_exact": reload_exact,
            "controller_frozen": digest_before == digest_after,
            "corruption_rejected": corruption_rejected,
        },
    }
    latency_started = time.perf_counter()
    with torch.no_grad():
        router(test_queries, keys)
    report["router_latency_ms_per_query"] = (
        (time.perf_counter() - latency_started)
        * 1000.0
        / max(int(test_queries.shape[0]), 1)
    )
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=64001)
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
