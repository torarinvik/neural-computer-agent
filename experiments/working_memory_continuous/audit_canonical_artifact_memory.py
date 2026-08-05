"""Bridge the canonical artifact store to real working-memory procedures.

This audit uses already learned, verifier-qualified Brain Workshop successor
artifacts only as a compatibility fixture. The new canonical store performs
addressing, persistence, and integrity verification; the legacy procedure
runner remains outside production and is used only to measure whether the
rehydrated learned state still changes behavior causally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import torch

from experiments.archive.unified_cognitive_controller.legacy_model import (
    UnifiedCognitiveController,
)
from experiments.archive.unified_cognitive_controller.train_sequence_working_memory import (
    evaluate_sequence_memory,
)
from neural_computer import ExecutableArtifactMemory, load_growth_artifact


def _load(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def _resolve_checkpoint(value: object) -> Path:
    path = Path(str(value))
    if path.exists():
        return path
    fallback = Path("artifacts/checkpoints") / path.name
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"checkpoint is unavailable: {value}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_digest(model: UnifiedCognitiveController) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _span(filename: str) -> int:
    match = re.search(r"span(\d+)", filename)
    if match is None:
        raise ValueError(f"artifact filename has no span number: {filename}")
    return int(match.group(1))


def _load_canonical_bank(
    source_bank: Path,
    canonical_bank: Path,
    device: torch.device,
) -> tuple[
    ExecutableArtifactMemory,
    list[dict[str, object]],
    list[str],
    torch.Tensor,
]:
    source_manifest = json.loads((source_bank / "manifest.json").read_text())
    paths = source_manifest.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError("source bank has no artifact paths")
    source_rows = _load(source_bank / "rows.pt", device)
    keys = source_rows.get("keys")
    if not isinstance(keys, torch.Tensor) or keys.ndim != 2:
        raise ValueError("source bank has no learned address rows")
    bank = ExecutableArtifactMemory(
        canonical_bank,
        width=int(keys.shape[1]),
        capacity=len(paths),
        device=device,
    )
    payloads: list[dict[str, object]] = []
    source_names: list[str] = []
    for index, filename in enumerate(paths):
        if not isinstance(filename, str):
            raise TypeError("source bank path is not a string")
        payload = _load(source_bank / filename, device)
        state = payload.get("skill_state")
        if not isinstance(state, dict):
            raise TypeError("source artifact has no tensor skill_state mapping")
        bank.put(keys[index].detach().cpu(), state)
        payloads.append(payload)
        source_names.append(filename)
    return (
        ExecutableArtifactMemory.load(canonical_bank, device=device),
        payloads,
        source_names,
        keys,
    )


def _rehydrated_model(
    parent: dict[str, object],
    payload: dict[str, object],
    state: dict[str, torch.Tensor],
    device: torch.device,
    *,
    zero_skill: bool = False,
) -> tuple[UnifiedCognitiveController, bool]:
    model = UnifiedCognitiveController(
        **dict(payload["child_model_configuration"])
    ).to(device)
    result = model.load_state_dict(parent["state_dict"], strict=False)
    growth_keys = set(state)
    if set(result.missing_keys) != growth_keys or result.unexpected_keys:
        raise RuntimeError(
            "canonical growth rehydration mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    load_state = (
        {name: torch.zeros_like(value) for name, value in state.items()}
        if zero_skill
        else state
    )
    receipt = load_growth_artifact(
        model,
        load_state,
        growth_prefixes=("skill_",),
    )
    model.eval()
    return model, receipt.core_unchanged


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    source_bank = args.source_bank
    canonical_bank, payloads, source_names, keys = _load_canonical_bank(
        source_bank, args.canonical_bank, device
    )
    canonical_bank.validate()
    reloaded_bank = ExecutableArtifactMemory.load(args.canonical_bank, device=device)

    parent_path = _resolve_checkpoint(payloads[0]["parent_checkpoint"])
    parent = _load(parent_path, device)
    parent_model = UnifiedCognitiveController(
        **dict(parent["model_configuration"])
    ).to(device)
    parent_model.load_state_dict(parent["state_dict"], strict=True)
    parent_digest_before = _state_digest(parent_model)

    rows: list[dict[str, object]] = []
    for index, payload in enumerate(payloads):
        filename = source_names[index]
        span = _span(filename)
        child = _load(_resolve_checkpoint(payload["child_checkpoint"]), device)
        _, loaded_state = reloaded_bank.promote(keys[index])
        direct = UnifiedCognitiveController(
            **dict(child["model_configuration"])
        ).to(device)
        direct.load_state_dict(child["state_dict"], strict=True)
        direct.eval()
        rehydrated, rehydrated_core_unchanged = _rehydrated_model(
            parent, payload, loaded_state, device
        )
        corrupted, corrupted_core_unchanged = _rehydrated_model(
            parent,
            payload,
            {name: torch.zeros_like(value) for name, value in loaded_state.items()},
            device,
        )
        direct_audit = evaluate_sequence_memory(
            direct, count=args.count, span=span, distractors=args.distractors,
            seed=args.seed + index, operation="mixed", device=device,
        )
        rehydrated_audit = evaluate_sequence_memory(
            rehydrated, count=args.count, span=span, distractors=args.distractors,
            seed=args.seed + index, operation="mixed", device=device,
        )
        corrupted_audit = evaluate_sequence_memory(
            corrupted, count=args.count, span=span, distractors=args.distractors,
            seed=args.seed + index, operation="mixed", device=device,
        )
        retention: dict[str, dict[str, float]] = {}
        for offset, retention_span in enumerate(args.retention_spans):
            retention_seed = args.seed + 100_000 + index * 10_000 + offset
            parent_audit = evaluate_sequence_memory(
                parent_model,
                count=args.retention_count,
                span=retention_span,
                distractors=args.distractors,
                seed=retention_seed,
                operation="mixed",
                device=device,
            )
            rehydrated_retention = evaluate_sequence_memory(
                rehydrated,
                count=args.retention_count,
                span=retention_span,
                distractors=args.distractors,
                seed=retention_seed,
                operation="mixed",
                device=device,
            )
            retention[str(retention_span)] = {
                "parent_accuracy": float(parent_audit["accuracy"]),
                "rehydrated_accuracy": float(rehydrated_retention["accuracy"]),
                "change": float(
                    rehydrated_retention["accuracy"] - parent_audit["accuracy"]
                ),
            }
        rows.append({
            "artifact": filename,
            "span": span,
            "route_index": index,
            "route_confidence": float(reloaded_bank.promote(keys[index])[0].confidence),
            "direct": direct_audit,
            "rehydrated": rehydrated_audit,
            "corrupted": corrupted_audit,
            "rehydrated_core_unchanged": rehydrated_core_unchanged,
            "corrupted_core_unchanged": corrupted_core_unchanged,
            "retention": retention,
            "retention_within_gate": all(
                values["change"] >= -args.retention_tolerance
                for values in retention.values()
            ),
            "rehydrated_matches_direct": (
                rehydrated_audit["accuracy"] == direct_audit["accuracy"]
            ),
            "corruption_reduces_accuracy": (
                corrupted_audit["accuracy"] < rehydrated_audit["accuracy"] - 0.05
            ),
        })

    corruption_dir = args.report.parent / "canonical_bank_corrupted"
    if corruption_dir.exists():
        shutil.rmtree(corruption_dir)
    shutil.copytree(args.canonical_bank, corruption_dir)
    corrupted_filename = reloaded_bank.paths[0]
    if corrupted_filename is None:
        raise RuntimeError("first canonical artifact row is empty")
    corrupted_path = corruption_dir / corrupted_filename
    corrupted_path.write_bytes(corrupted_path.read_bytes() + b"corruption")
    try:
        ExecutableArtifactMemory.load(corruption_dir, device=device)
    except ValueError as error:
        corruption_rejected = "hash mismatch" in str(error)
    else:
        corruption_rejected = False

    parent_digest_after = _state_digest(parent_model)
    report = {
        "schema": "canonical-artifact-memory-working-memory-audit-v1",
        "claim_boundary": (
            "Canonical artifact memory routes and verifies previously learned "
            "controller-native growth state; it does not claim cold-start "
            "procedure discovery."
        ),
        "source_bank": str(source_bank),
        "canonical_bank": str(args.canonical_bank),
        "canonical_bank_sha256": _sha256(args.canonical_bank / "rows.pt"),
        "count": args.count,
        "distractors": args.distractors,
        "retention_spans": list(args.retention_spans),
        "retention_count": args.retention_count,
        "retention_tolerance": args.retention_tolerance,
        "seed": args.seed,
        "controller_weights_unchanged": parent_digest_before == parent_digest_after,
        "rows": rows,
        "corruption_rejected": corruption_rejected,
        "all_routes_exact": all(bool(row["rehydrated_matches_direct"]) for row in rows),
        "all_corruption_controls_causal": all(
            bool(row["corruption_reduces_accuracy"]) for row in rows
        ),
        "all_retention_gates_pass": all(
            bool(row["retention_within_gate"]) for row in rows
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--canonical-bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--retention-count", type=int, default=32)
    parser.add_argument("--retention-tolerance", type=float, default=0.02)
    parser.add_argument("--retention-spans", default="2,3,4,5,6,7,8")
    parser.add_argument("--seed", type=int, default=49011)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.count < 2 or args.retention_count < 2 or args.distractors < 0:
        raise ValueError("count, retention_count, and distractors are invalid")
    args.retention_spans = tuple(
        int(value) for value in args.retention_spans.split(",")
        if value.strip()
    )
    if any(value < 1 for value in args.retention_spans):
        raise ValueError("retention spans must be positive")
    if args.retention_tolerance < 0.0:
        raise ValueError("retention tolerance cannot be negative")
    report = run(args)
    print(json.dumps({
        "controller_weights_unchanged": report["controller_weights_unchanged"],
        "all_routes_exact": report["all_routes_exact"],
        "all_corruption_controls_causal": report["all_corruption_controls_causal"],
        "all_retention_gates_pass": report["all_retention_gates_pass"],
        "corruption_rejected": report["corruption_rejected"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
