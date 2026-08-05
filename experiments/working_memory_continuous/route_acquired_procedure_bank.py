"""Route independently acquired working-memory procedures by opaque address."""
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
from experiments.working_memory_continuous.route_acquired_growth import (
    _cosine_accuracy,
    _load_artifacts,
    _permuted_accuracy,
    _route_accuracy,
)
from experiments.working_memory_continuous.route_mixed_procedures import (
    QUERY_BATCH,
    _corruption_control,
    _queries,
    _random_keys,
    _train_router_accounted,
)
from neural_computer import ExecutableArtifactMemory


def _load(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _rehydrate(
    parent: dict[str, object],
    artifact: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> UnifiedCognitiveController:
    from experiments.working_memory_continuous.route_acquired_growth import (
        _rehydrate as rehydrate_growth,
    )

    return rehydrate_growth(parent, artifact, device=device)


def run(args: argparse.Namespace) -> dict[str, object]:
    if len(args.artifacts) < 2 or len(args.artifacts) != len(args.operations):
        raise ValueError(
            "artifacts and operations must have equal length >= 2"
        )
    if args.behavior_count < 2 or args.behavior_count % 2:
        raise ValueError("behavior-count must be positive and even")
    started = time.perf_counter()
    device = torch.device(args.device)
    parent_payload = _load(args.parent, device)
    parent_model = UnifiedCognitiveController(
        **dict(parent_payload["model_configuration"])
    ).to(device)
    parent_model.load_state_dict(parent_payload["state_dict"], strict=True)
    parent_model.eval()
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    digest_before = _digest(parent_model)

    operations = tuple(args.operations)
    train_queries, train_targets = _queries(
        parent_model,
        operations,
        per_family=args.train_queries_per_family,
        seed=args.seed + 10_000,
        distractors=args.distractors,
        device=device,
    )
    test_queries, test_targets = _queries(
        parent_model,
        operations,
        per_family=args.test_queries_per_family,
        seed=args.seed + 20_000,
        distractors=args.distractors,
        device=device,
    )
    keys = _random_keys(
        len(args.artifacts),
        int(train_queries.shape[-1]),
        seed=args.seed + 30_000,
        device=device,
    )
    router, accounting = _train_router_accounted(
        keys,
        train_queries,
        train_targets,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed + 40_000,
        shuffle_outcomes=False,
    )
    shuffled_router, shuffled_accounting = _train_router_accounted(
        keys,
        train_queries,
        train_targets,
        updates=args.updates,
        batch_size=args.batch_size,
        seed=args.seed + 50_000,
        shuffle_outcomes=True,
    )
    memory = _load_artifacts(
        tuple(args.artifacts),
        device=device,
        destination=args.bank,
        keys=keys,
    )
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
        zero_artifact = {
            name: torch.zeros_like(value)
            for name, value in selected_artifact.items()
        }
        selected_model = _rehydrate(
            parent_payload, selected_artifact, device=device
        )
        zero_model = _rehydrate(parent_payload, zero_artifact, device=device)
        seed = args.seed + 60_000 + family
        selected_audit = evaluate_sequence_memory(
            selected_model,
            count=args.behavior_count,
            span=10,
            distractors=args.distractors,
            seed=seed,
            operation=operation,
            device=device,
        )
        wrong_artifact_accuracies: dict[str, float] = {}
        wrong_address_discriminative: dict[str, bool] = {}
        for wrong_index in range(len(args.artifacts)):
            if wrong_index == selected:
                continue
            _, wrong_artifact = restored.promote_index(wrong_index)
            wrong_model = _rehydrate(
                parent_payload, wrong_artifact, device=device
            )
            wrong_audit = evaluate_sequence_memory(
                wrong_model,
                count=args.behavior_count,
                span=10,
                distractors=args.distractors,
                seed=seed,
                operation=operation,
                device=device,
            )
            wrong_accuracy = float(wrong_audit["accuracy"])
            wrong_artifact_accuracies[str(wrong_index)] = wrong_accuracy
            wrong_address_discriminative[str(wrong_index)] = (
                wrong_accuracy < selected_audit["accuracy"] - 0.05
            )
        zero_audit = evaluate_sequence_memory(
            zero_model,
            count=args.behavior_count,
            span=10,
            distractors=args.distractors,
            seed=seed,
            operation=operation,
            device=device,
        )
        all_artifact_accuracies = {
            **wrong_artifact_accuracies,
            str(selected): float(selected_audit["accuracy"]),
        }
        zero_accuracy = float(zero_audit["accuracy"])
        causal_artifact_rows = [
            index
            for index, accuracy in all_artifact_accuracies.items()
            if accuracy > zero_accuracy + 0.05
        ]
        behavior[operation] = {
            "selected_row": selected,
            "expected_row": family,
            "selected_route_accuracy": float(
                (family_prediction == family).float().mean()
            ),
            "selected_artifact_accuracy": selected_audit["accuracy"],
            "all_artifact_accuracies": all_artifact_accuracies,
            "wrong_artifact_accuracies": wrong_artifact_accuracies,
            "zero_artifact_accuracy": zero_accuracy,
            "wrong_address_discriminative": wrong_address_discriminative,
            "causal_artifact_rows": causal_artifact_rows,
            "selected_within_five_points_of_best": (
                float(selected_audit["accuracy"])
                >= max(all_artifact_accuracies.values()) - 0.05
            ),
            "selected_artifact_is_causal": (
                zero_audit["accuracy"] < selected_audit["accuracy"] - 0.05
            ),
        }
    digest_after = _digest(parent_model)
    normal_route = _route_accuracy(router, test_queries, test_targets, keys)
    shuffled_route = _route_accuracy(
        shuffled_router, test_queries, test_targets, keys
    )
    permutation_route = _permuted_accuracy(
        router, test_queries, test_targets, keys
    )
    cosine_route = _cosine_accuracy(test_queries, test_targets, keys)
    report: dict[str, object] = {
        "schema": "opaque-address-routed-acquired-procedure-bank-v2",
        "claim_boundary": (
            "Multiple independently acquired, same-schema working-memory "
            "growth artifacts are routed by a replaceable memory-side router. "
            "The router sees opaque keys, attempted rows, and scalar outcomes; "
            "procedure identities remain verifier-private."
        ),
        "parent": str(args.parent),
        "artifacts": [str(path) for path in args.artifacts],
        "bank": str(args.bank),
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": args.batch_size,
        "unique_logical_lifetimes": int(
            train_queries.shape[0] * QUERY_BATCH
        ),
        "unique_verifier_bits": accounting["unique_verifier_bits"],
        "verifier_bits": accounting["verifier_bits"],
        "optimizer_updates": accounting["verifier_bits"] // args.batch_size,
        "replayed_examples": accounting["replayed_examples"],
        "reward_shuffled_verifier_bits": shuffled_accounting[
            "verifier_bits"
        ],
        "wall_seconds": time.perf_counter() - started,
        "normal_route_accuracy": normal_route,
        "reward_shuffled_route_accuracy": shuffled_route,
        "candidate_permutation_accuracy": permutation_route,
        "cosine_baseline_accuracy": cosine_route,
        "selected_rows": selected_rows,
        "bank_reload_exact": reload_exact,
        "controller_weights_unchanged": digest_before == digest_after,
        "corruption_rejected": _corruption_control(
            restored, args.report.parent / "bank_corrupted", device=device
        ),
        "behavior": behavior,
    }
    latency_started = time.perf_counter()
    with torch.no_grad():
        router(test_queries, keys)
    report["router_latency_ms_per_query"] = (
        (time.perf_counter() - latency_started)
        * 1000.0
        / max(int(test_queries.shape[0]), 1)
    )
    report["gates"] = {
        "normal_route_at_least_90": normal_route >= 0.90,
        "reward_shuffled_route_near_chance": shuffled_route <= 0.75,
        "candidate_permutation_invariant": permutation_route >= 0.90,
        "cosine_baseline_near_chance": cosine_route <= 0.75,
        "all_rows_selected": selected_rows == list(range(len(args.artifacts))),
        "all_wrong_addresses_discriminative": all(
            all(bool(value) for value in item["wrong_address_discriminative"].values())
            for item in behavior.values()
        ),
        "all_selected_artifacts_causal": all(
            bool(item["selected_artifact_is_causal"])
            for item in behavior.values()
        ),
        "all_tasks_have_a_causal_artifact": all(
            bool(item["causal_artifact_rows"])
            for item in behavior.values()
        ),
        "selected_within_five_points_of_best": all(
            bool(item["selected_within_five_points_of_best"])
            for item in behavior.values()
        ),
        "beneficial_off_diagonal_transfer_observed": any(
            len(item["causal_artifact_rows"]) > 1
            for item in behavior.values()
        ),
        "bank_reload_exact": reload_exact,
        "controller_frozen": digest_before == digest_after,
        "corruption_rejected": report["corruption_rejected"],
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    compositional_gates = {
        key: value
        for key, value in report["gates"].items()
        if key != "all_wrong_addresses_discriminative"
    }
    report["accepted_compositional_diagnostic"] = all(
        compositional_gates.values()
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--operations",
        nargs="+",
        default=("complement", "complement_reverse"),
    )
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=66001)
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
