"""Route real learned span artifacts through the canonical memory boundary.

The historical span-nine and span-ten artifacts were learned independently
against the same parent but used different numbers of successor slots. This
audit lifts both into one common two-slot growth schema, padding the unused
slot in span nine with zeros, then tests random opaque-address routing through
the production artifact store. The lift is control-plane schema adaptation;
the memory payload remains tensor-only and the controller remains frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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
    _permuted_accuracy,
    _queries,
    _random_keys,
    _route_accuracy,
    _train_router,
)
from neural_computer import (
    ExecutableArtifactMemory,
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


def _common_growth_artifacts(
    parent: dict[str, object],
    source_bank: Path,
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    span9_payload = _load(source_bank / "span9.pt", device)
    span10_payload = _load(source_bank / "span10.pt", device)
    common_configuration = dict(span10_payload["child_model_configuration"])
    common = UnifiedCognitiveController(**common_configuration).to(device)
    result = common.load_state_dict(parent["state_dict"], strict=False)
    growth_names = {
        name for name in common.state_dict() if name.startswith("skill_")
    }
    if set(result.missing_keys) != growth_names or result.unexpected_keys:
        raise RuntimeError(
            "common growth schema does not match parent: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )

    def padded(payload: dict[str, object]) -> dict[str, torch.Tensor]:
        state = {
            name: torch.zeros_like(value)
            for name, value in common.state_dict().items()
            if name in growth_names
        }
        source = payload.get("skill_state")
        if not isinstance(source, dict):
            raise TypeError("historical artifact has no skill_state mapping")
        for name, value in source.items():
            if name not in state:
                raise ValueError(f"historical growth key is outside common schema: {name}")
            if value.shape != state[name].shape:
                raise ValueError(f"historical growth shape mismatch for {name}")
            state[name] = value.detach().cpu().clone()
        return state

    return common_configuration, padded(span9_payload), padded(span10_payload)


def _rehydrate(
    parent: dict[str, object],
    configuration: dict[str, object],
    artifact: dict[str, torch.Tensor],
    *,
    device: torch.device,
) -> UnifiedCognitiveController:
    model = UnifiedCognitiveController(**configuration).to(device)
    result = model.load_state_dict(parent["state_dict"], strict=False)
    growth_names = {
        name for name in model.state_dict() if name.startswith("skill_")
    }
    if set(result.missing_keys) != growth_names or result.unexpected_keys:
        raise RuntimeError("parent/common schema load mismatch")
    freeze_core(model, ("skill_",))
    full_artifact = {
        name: value.to(device)
        for name, value in artifact.items()
    }
    receipt = load_growth_artifact(
        model, full_artifact, growth_prefixes=("skill_",)
    )
    if not receipt.core_unchanged:
        raise RuntimeError("historical artifact changed the frozen core")
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
    device = torch.device(args.device)
    if args.behavior_count < 2 or args.behavior_count % 2:
        raise ValueError("behavior-count must be positive and even")
    parent = _load(args.parent, device)
    parent_model = UnifiedCognitiveController(
        **dict(parent["model_configuration"])
    ).to(device)
    parent_model.load_state_dict(parent["state_dict"], strict=True)
    parent_model.eval()
    digest_before = _digest(parent_model)
    for parameter in parent_model.parameters():
        parameter.requires_grad_(False)
    configuration, span9_artifact, span10_artifact = _common_growth_artifacts(
        parent, args.source_bank, device=device
    )
    artifacts = (span9_artifact, span10_artifact)
    spans = (9, 10)
    train_queries, train_targets = _queries(
        parent_model,
        spans,
        per_family=args.train_queries_per_family,
        seed=args.seed + 10_000,
        distractors=args.distractors,
        device=device,
    )
    test_queries, test_targets = _queries(
        parent_model,
        spans,
        per_family=args.test_queries_per_family,
        seed=args.seed + 20_000,
        distractors=args.distractors,
        device=device,
    )
    keys = _random_keys(2, int(train_queries.shape[-1]), seed=args.seed + 30_000, device=device)
    router = _train_router(
        keys, train_queries, train_targets,
        updates=args.updates, batch_size=args.batch_size,
        seed=args.seed + 40_000, shuffle_outcomes=False,
    )
    shuffled_router = _train_router(
        keys, train_queries, train_targets,
        updates=args.updates, batch_size=args.batch_size,
        seed=args.seed + 50_000, shuffle_outcomes=True,
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
    for family, span in enumerate(spans):
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
            selected_model, count=args.behavior_count, span=span,
            distractors=args.distractors, seed=args.seed + 60_000 + family,
            operation="mixed", device=device,
        )
        wrong_audit = evaluate_sequence_memory(
            wrong_model, count=args.behavior_count, span=span,
            distractors=args.distractors, seed=args.seed + 60_000 + family,
            operation="mixed", device=device,
        )
        zero_audit = evaluate_sequence_memory(
            zero_model, count=args.behavior_count, span=span,
            distractors=args.distractors, seed=args.seed + 60_000 + family,
            operation="mixed", device=device,
        )
        behavior[str(span)] = {
            "selected_row": selected,
            "expected_row": family,
            "selected_route_accuracy": float(
                (family_prediction == family).float().mean()
            ),
            "selected_artifact_accuracy": selected_audit["accuracy"],
            "wrong_artifact_accuracy": wrong_audit["accuracy"],
            "wrong_address_reduces_accuracy": (
                wrong_audit["accuracy"] < selected_audit["accuracy"] - 0.05
            ),
            "zero_artifact_accuracy": zero_audit["accuracy"],
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
        "schema": "opaque-address-routed-historical-growth-audit-v1",
        "claim_boundary": (
            "Real learned span-nine and span-ten tensor artifacts are lifted "
            "into one common generic growth schema. A frozen controller emits "
            "queries; a replaceable memory-side router sees only opaque row "
            "keys, attempted rows, and scalar outcomes. Span identities and "
            "correct rows remain verifier-private."),
        "parent": str(args.parent),
        "source_bank": str(args.source_bank),
        "bank": str(args.bank),
        "seed": args.seed,
        "spans_private_to_verifier": list(spans),
        "updates": args.updates,
        "batch_size": args.batch_size,
        "verifier_bits": args.updates * args.batch_size,
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
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=63001)
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
