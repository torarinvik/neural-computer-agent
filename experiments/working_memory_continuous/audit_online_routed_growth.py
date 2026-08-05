"""Verify that online artifact append preserves old routing and adds new routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.archive.unified_cognitive_controller.legacy_model import (
    UnifiedCognitiveController,
)
from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    evaluate_sequence_memory,
)
from experiments.working_memory_continuous.audit_online_artifact_growth import (
    _load_single,
    _random_keys,
)
from experiments.working_memory_continuous.route_acquired_procedure_bank import (
    _rehydrate,
)
from experiments.working_memory_continuous.route_mixed_procedures import (
    QUERY_BATCH,
    _corruption_control,
    _queries,
    _train_router_accounted,
)
from neural_computer import ExecutableArtifactMemory


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    if len(args.sources) != 2:
        raise ValueError("exactly two acquired source memories are required")
    started = time.perf_counter()
    device = torch.device(args.device)
    parent_payload = torch.load(
        args.parent, map_location=device, weights_only=False
    )
    parent = UnifiedCognitiveController(
        **dict(parent_payload["model_configuration"])
    ).to(device)
    parent.load_state_dict(parent_payload["state_dict"], strict=True)
    parent.eval()
    for parameter in parent.parameters():
        parameter.requires_grad_(False)
    core_before = _digest(parent)

    source_records = [
        _load_single(path, device=device) for path in args.sources
    ]
    source_keys = [record[0] for record in source_records]
    artifacts = [record[1] for record in source_records]
    width = int(source_keys[0].numel())
    keys = _random_keys(2, width, args.seed).to(device)
    memory = ExecutableArtifactMemory(
        args.bank, width=width, capacity=2, device=device
    )
    first_row = memory.put(keys[0].cpu(), artifacts[0])
    first_cold = ExecutableArtifactMemory.load(args.bank, device=device)
    _, first_loaded_before_append = first_cold.promote_index(first_row)
    first_model_before = _rehydrate(
        parent_payload, first_loaded_before_append, device=device
    )
    first_before = evaluate_sequence_memory(
        first_model_before,
        count=args.behavior_count,
        span=10,
        distractors=args.distractors,
        seed=args.seed + 60_000,
        operation="complement",
        device=device,
    )["accuracy"]

    second_row = memory.put(keys[1].cpu(), artifacts[1])
    expanded = ExecutableArtifactMemory.load(args.bank, device=device)
    _, first_loaded_after_append = expanded.promote_index(first_row)
    first_model_after = _rehydrate(
        parent_payload, first_loaded_after_append, device=device
    )
    first_after = evaluate_sequence_memory(
        first_model_after,
        count=args.behavior_count,
        span=10,
        distractors=args.distractors,
        seed=args.seed + 60_000,
        operation="complement",
        device=device,
    )["accuracy"]

    operations = ("complement", "complement_reverse")
    train_queries, train_targets = _queries(
        parent,
        operations,
        per_family=args.train_queries_per_family,
        seed=args.seed + 10_000,
        distractors=args.distractors,
        device=device,
    )
    test_queries, test_targets = _queries(
        parent,
        operations,
        per_family=args.test_queries_per_family,
        seed=args.seed + 20_000,
        distractors=args.distractors,
        device=device,
    )
    router, accounting = _train_router_accounted(
        keys,
        train_queries,
        train_targets,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed + 30_000,
        shuffle_outcomes=False,
    )
    shuffled_router, _ = _train_router_accounted(
        keys,
        train_queries,
        train_targets,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed + 40_000,
        shuffle_outcomes=True,
    )
    prediction = router(test_queries, keys).argmax(dim=-1)
    behavior: dict[str, dict[str, object]] = {}
    for family, operation in enumerate(operations):
        selected = int(torch.mode(prediction[test_targets == family]).values)
        _, selected_artifact = expanded.promote_index(selected)
        _, wrong_artifact = expanded.promote_index(1 - selected)
        zero_artifact = {
            name: torch.zeros_like(value)
            for name, value in selected_artifact.items()
        }
        models = {
            "selected": _rehydrate(
                parent_payload, selected_artifact, device=device
            ),
            "wrong": _rehydrate(parent_payload, wrong_artifact, device=device),
            "zero": _rehydrate(parent_payload, zero_artifact, device=device),
        }
        accuracies = {
            name: evaluate_sequence_memory(
                model,
                count=args.behavior_count,
                span=10,
                distractors=args.distractors,
                seed=args.seed + 61_000 + family,
                operation=operation,
                device=device,
            )["accuracy"]
            for name, model in models.items()
        }
        behavior[operation] = {
            "selected_row": selected,
            "route_accuracy": float(
                (prediction[test_targets == family] == family).float().mean()
            ),
            "accuracies": accuracies,
            "wrong_address_discriminative": (
                accuracies["wrong"] < accuracies["selected"] - 0.05
            ),
            "selected_artifact_causal": (
                accuracies["zero"] < accuracies["selected"] - 0.05
            ),
        }
    core_after = _digest(parent)
    normal_route = float((prediction == test_targets).float().mean())
    shuffled_route = float(
        (shuffled_router(test_queries, keys).argmax(dim=-1) == test_targets)
        .float()
        .mean()
    )
    report = {
        "schema": "online-opaque-routed-growth-audit-v1",
        "claim_boundary": (
            "A new artifact is appended after an older artifact is already "
            "deployed. The expanded bank routes both procedures through a "
            "frozen controller using opaque outcome-trained addressing."
        ),
        "parent": str(args.parent),
        "sources": [str(path) for path in args.sources],
        "bank": str(args.bank),
        "seed": args.seed,
        "rows": {"first": first_row, "second": second_row},
        "unique_logical_lifetimes": int(train_queries.shape[0] * QUERY_BATCH),
        "unique_verifier_bits": accounting["unique_verifier_bits"],
        "verifier_bits": accounting["verifier_bits"],
        "optimizer_updates": args.updates,
        "replayed_examples": accounting["replayed_examples"],
        "wall_seconds": time.perf_counter() - started,
        "first_accuracy_before_append": first_before,
        "first_accuracy_after_append": first_after,
        "normal_route_accuracy": normal_route,
        "reward_shuffled_route_accuracy": shuffled_route,
        "behavior": behavior,
        "controller_weights_unchanged": core_before == core_after,
        "corruption_rejected": _corruption_control(
            expanded, args.report.parent / "bank_corrupted", device=device
        ),
    }
    report["gates"] = {
        "append_used_new_row": second_row == 1,
        "old_behavior_retained_after_append": abs(first_after - first_before) <= 0.02,
        "normal_route_at_least_90": normal_route >= 0.90,
        "reward_shuffled_route_near_chance": shuffled_route <= 0.75,
        "both_artifacts_selected": [
            int(torch.mode(prediction[test_targets == family]).values)
            for family in range(2)
        ] == [0, 1],
        "all_wrong_addresses_discriminative": all(
            bool(item["wrong_address_discriminative"])
            for item in behavior.values()
        ),
        "all_selected_artifacts_causal": all(
            bool(item["selected_artifact_causal"])
            for item in behavior.values()
        ),
        "controller_frozen": core_before == core_after,
        "corruption_rejected": report["corruption_rejected"],
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs=2, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=68001)
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
    print(json.dumps(report["gates"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
