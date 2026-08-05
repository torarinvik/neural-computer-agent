"""Execute two independently acquired factors in one frozen controller.

This audit is deliberately narrower than arbitrary program composition.  It
tests the missing boundary between verified top-k artifact promotion and
execution: two same-schema growth artifacts are remapped into disjoint slots,
loaded together, and evaluated on their two private procedures.  The memory
backend still sees only opaque keys and tensor payloads; procedure names stay
in the verifier-side experiment harness.
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

from experiments.archive.unified_cognitive_controller.legacy_model import (
    UnifiedCognitiveController,
)
from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    evaluate_sequence_memory,
    generate_sequence_memory_batch,
    rollout_sequence_memory,
)
from experiments.working_memory_continuous.acquire_frozen_growth import (
    _slot_prefixes,
)
from experiments.working_memory_continuous.route_acquired_growth import (
    _context_query,
)
from neural_computer import (
    ExecutableArtifactMemory,
    compose_growth_artifacts,
    freeze_core,
    load_growth_artifact,
)


def _load(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def _digest(
    module: torch.nn.Module,
    *,
    excluded_prefixes: tuple[str, ...] = (),
) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        if name.startswith(excluded_prefixes):
            continue
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _slot_map(source_slot: int, target_slot: int) -> dict[str, str]:
    return dict(zip(
        _slot_prefixes(source_slot),
        _slot_prefixes(target_slot),
        strict=True,
    ))


def _load_source_artifact(
    directory: Path,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    memory = ExecutableArtifactMemory.load(directory, device=device)
    rows = memory.address_rows()
    if len(rows) != 1:
        raise ValueError(f"expected one acquired artifact in {directory}")
    _, artifact = memory.promote_index(rows[0][0])
    return artifact


def _two_slot_model(
    payload: dict[str, object],
    *,
    device: torch.device,
) -> tuple[UnifiedCognitiveController, tuple[tuple[str, ...], ...]]:
    configuration = dict(payload["model_configuration"])
    configuration["skill_adapter_widths"] = (256, 256)
    model = UnifiedCognitiveController(**configuration).to(device)
    prefixes = (_slot_prefixes(0), _slot_prefixes(1))
    missing, unexpected = model.load_state_dict(
        payload["state_dict"], strict=False
    )
    expected_missing = {
        name
        for name in model.state_dict()
        if name.startswith(prefixes[0] + prefixes[1])
    }
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            "two-slot checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    freeze_core(model, prefixes[0] + prefixes[1])
    model.eval()
    return model, prefixes


def _rehydrate(
    payload: dict[str, object],
    artifacts: tuple[dict[str, torch.Tensor], ...],
    *,
    targets: tuple[int, ...],
    device: torch.device,
) -> tuple[UnifiedCognitiveController, dict[str, object]]:
    if len(artifacts) != len(targets):
        raise ValueError("artifacts and target slots must align")
    model, prefixes = _two_slot_model(payload, device=device)
    if not artifacts:
        return model, {
            "loaded_keys": 0,
            "core_unchanged": True,
            "namespaces": [],
        }
    maps = tuple(_slot_map(0, target) for target in targets)
    composed = compose_growth_artifacts(artifacts, prefix_maps=maps)
    receipt = load_growth_artifact(
        model,
        composed,
        growth_prefixes=prefixes[0] + prefixes[1],
    )
    if not receipt.core_unchanged:
        raise RuntimeError("growth composition changed frozen controller state")
    return model, {
        "loaded_keys": len(receipt.loaded_keys),
        "core_unchanged": receipt.core_unchanged,
        "namespaces": sorted(
            "slot0" if name.startswith(prefixes[0]) else "slot1"
            for name in composed
        ),
    }


@torch.no_grad()
def _insertion_is_exact(
    parent: UnifiedCognitiveController,
    blank: UnifiedCognitiveController,
    *,
    count: int,
    span: int,
    distractors: int,
    seed: int,
    device: torch.device,
) -> bool:
    batch = generate_sequence_memory_batch(
        count,
        span=span,
        distractors=distractors,
        seed=seed,
        operation="mixed",
        heldout=True,
        device=device,
    )
    parent_result = rollout_sequence_memory(
        parent, batch, sample_actions=False
    )
    blank_result = rollout_sequence_memory(
        blank, batch, sample_actions=False
    )
    return bool(
        torch.equal(parent_result["logits"], blank_result["logits"])
        and torch.equal(
            parent_result["final_workspace"], blank_result["final_workspace"]
        )
        and torch.equal(
            parent_result["final_hidden"], blank_result["final_hidden"]
        )
    )


def _accuracy(
    model: UnifiedCognitiveController,
    *,
    operation: str,
    count: int,
    span: int,
    distractors: int,
    seed: int,
    device: torch.device,
) -> float:
    return float(
        evaluate_sequence_memory(
            model,
            count=count,
            span=span,
            distractors=distractors,
            seed=seed,
            operation=operation,
            device=device,
        )["accuracy"]
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    started = perf_counter()
    if len(args.artifacts) != 2:
        raise ValueError("composition audit requires exactly two artifacts")
    if args.count < 1 or args.span < 1 or args.distractors < 0:
        raise ValueError("count, span, and distractors are invalid")
    device = torch.device(args.device)
    parent_payload = _load(args.parent, device)
    parent = UnifiedCognitiveController(
        **dict(parent_payload["model_configuration"])
    ).to(device)
    parent.load_state_dict(parent_payload["state_dict"], strict=True)
    parent.eval()
    parent_digest = _digest(parent)

    source_artifacts = tuple(
        _load_source_artifact(path, device=device) for path in args.artifacts
    )
    query = _context_query(
        parent,
        seed=args.seed,
        count=64,
        span=args.span,
        distractors=args.distractors,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    candidate_keys = tuple(
        F.normalize(
            query + 0.05 * torch.randn(
                query.shape, generator=generator, device=device
            ),
            dim=0,
        ).cpu()
        for _ in source_artifacts
    )
    if args.bank.exists():
        shutil.rmtree(args.bank)
    bank = ExecutableArtifactMemory(
        args.bank,
        width=int(query.numel()),
        capacity=len(source_artifacts),
        device=device,
    )
    for key, artifact in zip(candidate_keys, source_artifacts, strict=True):
        bank.put(key, artifact)
    bank = ExecutableArtifactMemory.load(args.bank, device=device)
    handles, candidates = bank.promote_candidates(query, top_k=2)
    candidates_by_row = {
        handle.index: artifact
        for handle, artifact in zip(handles, candidates, strict=True)
    }
    if set(candidates_by_row) != {0, 1}:
        raise RuntimeError("top-k promotion did not return both artifact rows")
    ordered_candidates = tuple(candidates_by_row[index] for index in (0, 1))

    blank, blank_receipt = _rehydrate(
        parent_payload, (), targets=(), device=device
    )
    factor_a, factor_a_receipt = _rehydrate(
        parent_payload, (ordered_candidates[0],), targets=(0,), device=device
    )
    factor_b, factor_b_receipt = _rehydrate(
        parent_payload, (ordered_candidates[1],), targets=(1,), device=device
    )
    composed, composed_receipt = _rehydrate(
        parent_payload,
        ordered_candidates,
        targets=(0, 1),
        device=device,
    )

    operations = ("complement", "complement_reverse")
    models = {
        "parent": parent,
        "factor_a": factor_a,
        "factor_b": factor_b,
        "composed": composed,
    }
    behavior: dict[str, dict[str, float]] = {}
    for operation_index, operation in enumerate(operations):
        seed = args.seed + 10_000 + operation_index
        behavior[operation] = {
            model_name: _accuracy(
                model,
                operation=operation,
                count=args.count,
                span=args.span,
                distractors=args.distractors,
                seed=seed,
                device=device,
            )
            for model_name, model in models.items()
        }

    frozen = all(
        _digest(model, excluded_prefixes=("skill_adapters.", "skill_adapter_"))
        == parent_digest
        for model in (factor_a, factor_b, composed)
    )
    composition_names = set(
        compose_growth_artifacts(
            ordered_candidates,
            prefix_maps=(_slot_map(0, 0), _slot_map(0, 1)),
        )
    )
    expected_names_count = sum(len(artifact) for artifact in ordered_candidates)
    names_are_slot_scoped = all(
        name.startswith(_slot_prefixes(0) + _slot_prefixes(1))
        for name in composition_names
    )
    report = {
        "schema": "multi-artifact-growth-composition-audit-v1",
        "claim_boundary": (
            "Two verified opaque artifacts can be promoted together, remapped "
            "into disjoint generic growth slots, and executed by one frozen "
            "controller on two verifier-private procedures. This does not "
            "qualify arbitrary program synthesis or sequential factor algebra."
        ),
        "parent": str(args.parent),
        "artifacts": [str(path) for path in args.artifacts],
        "bank": str(args.bank),
        "seed": args.seed,
        "count": args.count,
        "span": args.span,
        "distractors": args.distractors,
        "accounting": {
            "unique_logical_lifetimes": 2 * args.count + 64,
            "unique_verifier_bits": 2 * args.count * args.span,
            "verifier_bits": 2 * args.count * args.span,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": perf_counter() - started,
            "stable_bits_to_threshold": None,
            "retention_on_mastered_primitives": None,
            "transfer_ratio_against_fresh_learner": None,
        },
        "top_k_handles": [
            {
                "index": handle.index,
                "confidence": handle.confidence,
                "margin": handle.margin,
            }
            for handle in handles
        ],
        "behavior": behavior,
        "receipts": {
            "blank": blank_receipt,
            "factor_a": factor_a_receipt,
            "factor_b": factor_b_receipt,
            "composed": composed_receipt,
        },
        "controller_core_unchanged": frozen,
        "top_k_retrieved_two": len(handles) == 2,
        "namespaces_disjoint": (
            len(composition_names) == expected_names_count
            and names_are_slot_scoped
            and any(name.startswith(_slot_prefixes(0)) for name in composition_names)
            and any(name.startswith(_slot_prefixes(1)) for name in composition_names)
        ),
        "blank_insertion_exact": _insertion_is_exact(
            parent,
            blank,
            count=min(args.count, 32),
            span=args.span,
            distractors=args.distractors,
            seed=args.seed + 2_000,
            device=device,
        ),
    }
    report["gates"] = {
        "top_k_retrieved_two": report["top_k_retrieved_two"],
        "namespaces_disjoint": report["namespaces_disjoint"],
        "blank_insertion_exact": report["blank_insertion_exact"],
        "controller_core_unchanged": report["controller_core_unchanged"],
        "factor_a_beats_parent_on_first": (
            behavior["complement"]["factor_a"]
            > behavior["complement"]["parent"] + 0.05
        ),
        "factor_b_beats_parent_on_second": (
            behavior["complement_reverse"]["factor_b"]
            > behavior["complement_reverse"]["parent"] + 0.05
        ),
        "composed_retains_first_factor": (
            behavior["complement"]["composed"]
            >= behavior["complement"]["factor_a"] - 0.05
        ),
        "composed_retains_second_factor": (
            behavior["complement_reverse"]["composed"]
            >= behavior["complement_reverse"]["factor_b"] - 0.05
        ),
    }
    report["accepted_diagnostic"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, nargs=2, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=69001)
    parser.add_argument("--count", type=int, default=128)
    parser.add_argument("--span", type=int, default=10)
    parser.add_argument("--distractors", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
