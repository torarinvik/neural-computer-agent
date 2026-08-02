"""Audit bounded hot/cold routing for an external sequence skill artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .audit_sequence_skill_memory import (
    _build_skill_memory,
    _load,
    _rehydrate,
)
from .model import UnifiedCognitiveController
from .skill_memory_bank import SkillArtifactBank
from .train_sequence_working_memory import (
    evaluate_sequence_memory,
    generate_sequence_memory_batch,
)
from .environment import NULL_ACTION


@torch.no_grad()
def _context_key(
        model: UnifiedCognitiveController, *, seed: int, count: int,
        span: int, distractors: int, device: torch.device,
        ) -> torch.Tensor:
    """Make a generic address from the pre-query controller state."""
    batch = generate_sequence_memory_batch(
        count, span=span, distractors=distractors, seed=seed,
        operation="mixed", heldout=True, device=device)
    state = model.initial_state(count, device=device)
    null = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    for index in range(span):
        _, state = model.step(
            batch.input_frames[:, index], state, null, zeros, zeros)
    for index in range(distractors):
        _, state = model.step(
            batch.distractor_frames[:, index], state, null, zeros, zeros)
    return F.normalize(state.hidden.mean(0), dim=0)


def _audit(
        model: UnifiedCognitiveController, *, count: int, span: int,
        distractors: int, seed: int, device: torch.device,
        ) -> dict[str, object]:
    return evaluate_sequence_memory(
        model, count=count, span=span, distractors=distractors,
        seed=seed, operation="mixed", device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
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
    real = _build_skill_memory(
        parent, child, parent_path=args.parent, child_path=args.child)
    decoy = dict(real)
    decoy["skill_state"] = {
        key: torch.zeros_like(value) for key, value in real["skill_state"].items()}
    # The key is controller-produced, not a task name or verifier label.
    parent_model = UnifiedCognitiveController(
        **dict(parent["model_configuration"])).to(device)
    parent_model.load_state_dict(parent["state_dict"], strict=True)
    parent_model.eval()
    real_key = _context_key(
        parent_model, seed=args.seed, count=256, span=args.span,
        distractors=args.distractors, device=device).cpu()
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 991)
    decoy_key = F.normalize(torch.randn(
        real_key.shape, generator=generator), dim=0)
    args.bank.mkdir(parents=True, exist_ok=True)
    bank = SkillArtifactBank(
        args.bank, width=real_key.numel(), capacity=2, device="cpu")
    real_index = bank.put(real_key, real, name="real-skill.pt")
    decoy_index = bank.put(decoy_key, decoy, name="decoy-skill.pt")
    bank.save()
    cold_rows = {
        "keys": bank.memory.store.keys.clone(),
        "usage": bank.memory.store.usage.clone(),
        "valid": bank.memory.store.valid.clone(),
    }
    restored_bank = SkillArtifactBank.load(args.bank, device="cpu")
    cold_reload_exact = all(torch.equal(
        cold_rows[name], getattr(restored_bank.memory.store, name))
        for name in cold_rows)
    query_key = _context_key(
        parent_model, seed=args.seed + 1, count=256, span=args.span,
        distractors=args.distractors, device=device).cpu()
    selected_before, confidence_before, _ = bank.promote(query_key)
    selected_after, confidence_after, reloaded_artifact = (
        restored_bank.promote(query_key))
    # Promotion should resolve the real artifact.  The decoy remains a
    # physically valid cold row and supplies the corruption control below.
    if selected_before != real_index or selected_after != real_index:
        raise AssertionError(
            "generic controller context did not address the real skill row")
    normal = _rehydrate(parent, reloaded_artifact, device=device)
    parent_for_eviction = parent_model
    corrupt = _rehydrate(
        parent, torch.load(
            args.bank / "decoy-skill.pt", map_location="cpu",
            weights_only=False), device=device)
    normal_audit = _audit(
        normal, count=args.count, span=args.span,
        distractors=args.distractors, seed=args.seed + 90_000,
        device=device)
    evicted_audit = _audit(
        parent_for_eviction, count=args.count, span=args.span,
        distractors=args.distractors, seed=args.seed + 90_000,
        device=device)
    corrupt_audit = _audit(
        corrupt, count=args.count, span=args.span,
        distractors=args.distractors, seed=args.seed + 90_000,
        device=device)
    restored_bank.evict_hot(selected_after)
    report = {
        "schema": "sequence-skill-bank-audit-v1",
        "bank": str(args.bank),
        "real_index": real_index,
        "decoy_index": decoy_index,
        "selected_before_reload": selected_before,
        "selected_after_reload": selected_after,
        "confidence_before_reload": confidence_before,
        "confidence_after_reload": confidence_after,
        "cold_reload_exact": cold_reload_exact,
        "count": args.count,
        "span": args.span,
        "distractors": args.distractors,
        "seed": args.seed,
        "normal_promoted_audit": normal_audit,
        "hot_evicted_parent_audit": evicted_audit,
        "decoy_artifact_audit": corrupt_audit,
        "promotion_survives_reload": (
            selected_before == selected_after == real_index
            and cold_reload_exact),
        "eviction_reduces_accuracy": (
            evicted_audit["accuracy"] < normal_audit["accuracy"] - 0.05),
        "decoy_reduces_accuracy": (
            corrupt_audit["accuracy"] < normal_audit["accuracy"] - 0.05),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "normal": normal_audit["accuracy"],
        "evicted": evicted_audit["accuracy"],
        "decoy": corrupt_audit["accuracy"],
        "selected_after_reload": selected_after,
        "cold_reload_exact": cold_reload_exact,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
