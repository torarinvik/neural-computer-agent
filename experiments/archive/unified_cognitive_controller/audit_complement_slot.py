"""Audit an appended complement-operation skill slot.

This audit is intentionally verifier-side.  The learner only saw latent
features, attempted opaque actions, and scalar outcomes; this script uses the
task generator solely to test the resulting checkpoint.  It compares the
parent, the trained child, and a child with its appended slot zeroed, then
optionally evaluates an outcome-shuffled child as an adversarial control.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .legacy_model import UnifiedCognitiveController
from .train_sequence_working_memory import evaluate_sequence_memory


def _load_model(path: Path, device: torch.device) -> UnifiedCognitiveController:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    compatibility = model.load_state_dict(
        payload["state_dict"], strict=False)
    if compatibility.unexpected_keys:
        raise RuntimeError(
            f"unexpected checkpoint keys in {path}: "
            f"{compatibility.unexpected_keys}")
    model.eval()
    return model


def _zero_last_slot(model: UnifiedCognitiveController) -> None:
    """Remove the appended slot's effect without changing inherited slots."""
    slot_count = len(model.skill_adapters)
    if not slot_count:
        raise ValueError("candidate has no skill slot to zero")
    index = slot_count - 1
    collections = (
        model.skill_adapters,
        model.skill_adapter_gates,
        model.skill_adapter_gate_refiners,
        model.skill_adapter_gate_extensions,
        model.skill_adapter_critics,
    )
    with torch.no_grad():
        for collection in collections:
            if index >= len(collection):
                continue
            for parameter in collection[index].parameters():
                parameter.zero_()


def _finite(value: Any) -> Any:
    if isinstance(value, float) and not torch.isfinite(
            torch.tensor(value)):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite(item) for item in value]
    return value


@torch.no_grad()
def _audit_model(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        span: int, distractors: int, device: torch.device) -> dict[str, Any]:
    return {
        "complement": evaluate_sequence_memory(
            model, count=count, span=span, distractors=distractors,
            seed=seed, operation="complement", device=device),
        "span9": evaluate_sequence_memory(
            model, count=count, span=9, distractors=distractors,
            seed=seed + 1, operation="mixed", device=device),
        "span10": evaluate_sequence_memory(
            model, count=count, span=10, distractors=distractors,
            seed=seed + 2, operation="mixed", device=device),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--shuffled-candidate", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=294800)
    parser.add_argument("--span", type=int, default=10)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps"))
    args = parser.parse_args()
    if args.count < 1 or args.span < 1 or args.distractors < 0:
        raise ValueError("count/span must be positive and distractors nonnegative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device)

    parent = _load_model(args.parent, device)
    candidate = _load_model(args.candidate, device)
    zeroed = _load_model(args.candidate, device)
    _zero_last_slot(zeroed)
    result: dict[str, Any] = {
        "schema": "complement-slot-audit-v1",
        "device": str(device),
        "count": args.count,
        "seed": args.seed,
        "span": args.span,
        "distractors": args.distractors,
        "parent": str(args.parent),
        "candidate": str(args.candidate),
        "parent_audit": _audit_model(
            parent, count=args.count, seed=args.seed, span=args.span,
            distractors=args.distractors, device=device),
        "candidate_audit": _audit_model(
            candidate, count=args.count, seed=args.seed, span=args.span,
            distractors=args.distractors, device=device),
        "zeroed_slot_audit": _audit_model(
            zeroed, count=args.count, seed=args.seed, span=args.span,
            distractors=args.distractors, device=device),
    }
    if args.shuffled_candidate is not None:
        shuffled = _load_model(args.shuffled_candidate, device)
        result["shuffled_candidate"] = str(args.shuffled_candidate)
        result["shuffled_audit"] = _audit_model(
            shuffled, count=args.count, seed=args.seed, span=args.span,
            distractors=args.distractors, device=device)
    child = result["candidate_audit"]["complement"]["accuracy"]
    zero = result["zeroed_slot_audit"]["complement"]["accuracy"]
    result["causal_gain_points"] = 100.0 * (child - zero)
    result["promotion_bars"] = {
        "causal_gain_points_at_least": 5.0,
        "old_retention_drop_points_at_most": 2.0,
        "reset_near_chance": [0.45, 0.55],
    }
    result = _finite(result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "device": str(device),
        "candidate_complement": child,
        "zeroed_complement": zero,
        "causal_gain_points": result["causal_gain_points"],
        "report": str(args.report),
    }, indent=2))


if __name__ == "__main__":
    main()
