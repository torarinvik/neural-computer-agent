"""Audit external serialization and causal corruption of a learned skill slot.

The parent controller remains the frozen computation core.  Only the learned
successor-slot parameters are written to a separate skill-memory artifact and
then rehydrated on top of the parent.  No verifier labels or correct actions
are stored in the artifact.  A zeroed artifact is the causal corruption
control: it should remove the new skill while leaving the parent intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train_sequence_working_memory import evaluate_sequence_memory


SKILL_PREFIXES = (
    "skill_adapters.",
    "skill_adapter_gates.",
    "skill_adapter_gate_refiners.",
    "skill_adapter_gate_extensions.",
    "skill_adapter_read_projections.",
    "skill_adapter_intention_interactions.",
    "skill_adapter_outer_event_projections.",
    "skill_adapter_outer_intention_projections.",
    "skill_adapter_critics.",
    "skill_adapter_critic_scales.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, device: torch.device) -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)


def _is_skill_key(key: str) -> bool:
    return key.startswith(SKILL_PREFIXES)


def _build_skill_memory(
        parent: dict[str, object], child: dict[str, object],
        *, parent_path: Path, child_path: Path,
        ) -> dict[str, object]:
    parent_config = dict(parent["model_configuration"])
    child_config = dict(child["model_configuration"])
    parent_state = parent["state_dict"]
    child_state = child["state_dict"]
    if not isinstance(parent_state, dict) or not isinstance(child_state, dict):
        raise ValueError("checkpoint state_dict must be mappings")
    if parent_config != {
            key: value for key, value in child_config.items()
            if key not in {
                "skill_adapter_widths", "skill_adapter_gate_mode",
                "skill_adapter_gate_hidden",
                "skill_adapter_legacy_read_from",
                "skill_adapter_reads_intention_from",
                "skill_adapter_reads_workspace_from",
                "skill_adapter_reads_workspace_usage_from",
                "skill_adapter_reads_event_age_from",
                "event_age", "skill_adapter_read_bottleneck",
                "skill_adapter_critic_width",
            }}:
        raise ValueError("parent and child base configurations differ")
    skill_state = {
        key: value.detach().cpu().clone()
        for key, value in child_state.items() if _is_skill_key(key)}
    if not skill_state:
        raise ValueError("child checkpoint contains no skill-slot state")
    for key, value in parent_state.items():
        if key in child_state and not _is_skill_key(key):
            if not torch.equal(value, child_state[key]):
                raise ValueError(f"frozen parent parameter changed: {key}")
    return {
        "schema": "sequence-skill-memory-v1",
        "claim_boundary": (
            "Only learned successor-slot parameters are stored. No verifier "
            "correct action, task label, or semantic rule is present."),
        "parent_checkpoint": str(parent_path),
        "child_checkpoint": str(child_path),
        "parent_sha256": _sha256(parent_path),
        "child_sha256": _sha256(child_path),
        "child_model_configuration": child_config,
        "skill_state": skill_state,
    }


def _rehydrate(
        parent: dict[str, object], skill_memory: dict[str, object],
        device: torch.device, *, zero_skill: bool = False,
        ) -> UnifiedCognitiveController:
    config = dict(skill_memory["child_model_configuration"])
    model = UnifiedCognitiveController(**config).to(device)
    parent_state = parent["state_dict"]
    result = model.load_state_dict(parent_state, strict=False)
    allowed = {key for key in model.state_dict() if _is_skill_key(key)}
    if set(result.missing_keys) != allowed or result.unexpected_keys:
        raise RuntimeError(
            "parent rehydration mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    skill_state = skill_memory["skill_state"]
    if not isinstance(skill_state, dict):
        raise ValueError("skill memory state must be a mapping")
    if zero_skill:
        skill_state = {
            key: torch.zeros_like(value) for key, value in skill_state.items()}
    result = model.load_state_dict(skill_state, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != (
            set(model.state_dict()) - set(skill_state)):
        raise RuntimeError(
            "skill rehydration mismatch: "
            f"missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--skill-memory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, default=4096)
    parser.add_argument("--span", type=int, default=9)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2:
        raise ValueError("count must be positive and even")
    device = torch.device(args.device)
    parent = _load(args.parent, device)
    child = _load(args.child, device)
    skill_memory = _build_skill_memory(
        parent, child, parent_path=args.parent, child_path=args.child)
    args.skill_memory.parent.mkdir(parents=True, exist_ok=True)
    torch.save(skill_memory, args.skill_memory)
    # Reload the serialized artifact as a separate object before evaluating.
    reloaded = torch.load(
        args.skill_memory, map_location=device, weights_only=False)
    normal = _rehydrate(parent, reloaded, device=device)
    corrupted = _rehydrate(
        parent, reloaded, device=device, zero_skill=True)
    normal_audit = evaluate_sequence_memory(
        normal, count=args.count, span=args.span,
        distractors=args.distractors, seed=args.seed,
        operation="mixed", device=device)
    corrupted_audit = evaluate_sequence_memory(
        corrupted, count=args.count, span=args.span,
        distractors=args.distractors, seed=args.seed,
        operation="mixed", device=device)
    child_model = UnifiedCognitiveController(
        **dict(child["model_configuration"])).to(device)
    child_model.load_state_dict(child["state_dict"], strict=True)
    child_model.eval()
    direct_audit = evaluate_sequence_memory(
        child_model, count=args.count, span=args.span,
        distractors=args.distractors, seed=args.seed,
        operation="mixed", device=device)
    report = {
        "schema": "sequence-skill-memory-audit-v1",
        "skill_memory": str(args.skill_memory),
        "skill_memory_sha256": _sha256(args.skill_memory),
        "skill_parameter_count": sum(
            int(value.numel()) for value in reloaded["skill_state"].values()),
        "count": args.count,
        "span": args.span,
        "distractors": args.distractors,
        "seed": args.seed,
        "direct_child_audit": direct_audit,
        "rehydrated_audit": normal_audit,
        "corrupted_skill_audit": corrupted_audit,
        "rehydrated_matches_direct_accuracy": (
            normal_audit["accuracy"] == direct_audit["accuracy"]),
        "corruption_reduces_accuracy": (
            corrupted_audit["accuracy"] < normal_audit["accuracy"] - 0.05),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "direct": direct_audit["accuracy"],
        "rehydrated": normal_audit["accuracy"],
        "corrupted": corrupted_audit["accuracy"],
        "corruption_reduces_accuracy": report[
            "corruption_reduces_accuracy"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
