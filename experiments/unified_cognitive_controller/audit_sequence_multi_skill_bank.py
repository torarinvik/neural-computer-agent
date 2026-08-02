"""Audit routing between two cold, opaque sequence-skill artifacts.

Each skill is learned in a separate controller checkpoint and then stored as
an opaque artifact behind the generic hot/cold bank.  At evaluation time the
bank sees only a controller-produced context key; it never receives a span,
operation, answer, or task name.  The audit checks that reloading the bank
selects the right artifact and that routing a span-nine episode to its own
artifact avoids the forgetting introduced when a span-ten residual is left
always-on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .audit_sequence_skill_bank import _context_key
from .audit_sequence_skill_memory import _build_skill_memory, _load, _rehydrate
from .model import UnifiedCognitiveController
from .skill_memory_bank import SkillArtifactBank
from .train_sequence_working_memory import evaluate_sequence_memory


def _model(payload: dict[str, object], device: torch.device) -> UnifiedCognitiveController:
    model = UnifiedCognitiveController(
        **dict(payload["model_configuration"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _audit(
        model: UnifiedCognitiveController, *, span: int, count: int,
        seed: int, distractors: int, device: torch.device,
        ) -> dict[str, float]:
    return evaluate_sequence_memory(
        model, count=count, span=span, distractors=distractors,
        seed=seed, operation="mixed", device=device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--span9-child", type=Path, required=True)
    parser.add_argument("--span10-child", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=49001)
    parser.add_argument("--count", type=int, default=2048)
    parser.add_argument("--key-count", type=int, default=256)
    parser.add_argument("--distractors", type=int, default=2)
    parser.add_argument(
        "--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.count < 2 or args.count % 2 or args.key_count < 2:
        raise ValueError("count and key-count must be positive; count must be even")
    device = torch.device(args.device)
    base_payload = _load(args.base, device)
    span9_payload = _load(args.span9_child, device)
    span10_payload = _load(args.span10_child, device)
    base_model = _model(base_payload, device)

    span9_memory = _build_skill_memory(
        base_payload, span9_payload,
        parent_path=args.base, child_path=args.span9_child)
    span10_memory = _build_skill_memory(
        base_payload, span10_payload,
        parent_path=args.base, child_path=args.span10_child)
    key9 = _context_key(
        base_model, seed=args.seed, count=args.key_count, span=9,
        distractors=args.distractors, device=device).cpu()
    key10 = _context_key(
        base_model, seed=args.seed + 1, count=args.key_count, span=10,
        distractors=args.distractors, device=device).cpu()
    # The keys are addresses produced by the frozen controller, not labels.
    if float(F.cosine_similarity(key9, key10, dim=0)) > 0.999:
        raise AssertionError("skill keys collapsed to the same address")

    args.bank.mkdir(parents=True, exist_ok=True)
    bank = SkillArtifactBank(
        args.bank, width=key9.numel(), capacity=2, device="cpu")
    index9 = bank.put(key9, span9_memory, name="span9.pt")
    index10 = bank.put(key10, span10_memory, name="span10.pt")
    bank.save()
    restored = SkillArtifactBank.load(args.bank, device="cpu")
    cold_reload_exact = all(torch.equal(
        getattr(bank.memory.store, name),
        getattr(restored.memory.store, name))
        for name in ("keys", "usage", "valid", "age"))
    if not cold_reload_exact:
        raise AssertionError("multi-skill bank rows changed across reload")

    selections: list[dict[str, object]] = []
    for span, index, offset in ((9, index9, 0), (10, index10, 1)):
        for repetition in range(4):
            query = _context_key(
                base_model,
                seed=args.seed + 100 + offset * 10 + repetition,
                count=args.key_count, span=span,
                distractors=args.distractors, device=device).cpu()
            selected, confidence, _ = restored.promote(query)
            selections.append({
                "span": span,
                "expected_index": index,
                "selected_index": selected,
                "confidence": confidence,
            })
    if any(row["selected_index"] != row["expected_index"] for row in selections):
        raise AssertionError("reloaded bank selected the wrong skill row")

    routed9 = _rehydrate(
        base_payload, span9_memory, device=device)
    routed10 = _rehydrate(
        base_payload, span10_memory, device=device)
    direct10 = _model(span10_payload, device)
    parent9 = _audit(
        routed9, span=9, count=args.count, seed=args.seed + 20_000,
        distractors=args.distractors, device=device)
    routed10_audit = _audit(
        routed10, span=10, count=args.count, seed=args.seed + 21_000,
        distractors=args.distractors, device=device)
    direct10_audit = _audit(
        direct10, span=10, count=args.count, seed=args.seed + 21_000,
        distractors=args.distractors, device=device)
    routed9_on10 = _audit(
        routed9, span=10, count=args.count, seed=args.seed + 22_000,
        distractors=args.distractors, device=device)
    routed10_on9 = _audit(
        routed10, span=9, count=args.count, seed=args.seed + 23_000,
        distractors=args.distractors, device=device)

    report = {
        "schema": "sequence-multi-skill-bank-audit-v1",
        "claim_boundary": (
            "The bank stores controller-produced keys and opaque learned slot "
            "parameters; it receives no span, task name, operation, or answer."),
        "base": str(args.base),
        "span9_child": str(args.span9_child),
        "span10_child": str(args.span10_child),
        "bank": str(args.bank),
        "seed": args.seed,
        "count": args.count,
        "key_count": args.key_count,
        "key_cosine_span9_span10": float(
            F.cosine_similarity(key9, key10, dim=0)),
        "selections": selections,
        "cold_reload_exact": cold_reload_exact,
        "cold_reload_routing_pass": True,
        "span9_routed_audit": parent9,
        "span10_routed_audit": routed10_audit,
        "span10_direct_child_audit": direct10_audit,
        "span9_skill_on_span10_audit": routed9_on10,
        "span10_skill_on_span9_audit": routed10_on9,
        "rehydrated_matches_direct_span10": (
            routed10_audit["accuracy"] == direct10_audit["accuracy"]),
        "routed_span10_beats_span9_parent": (
            routed10_audit["accuracy"] > routed9_on10["accuracy"] + 0.02),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "selected": [row["selected_index"] for row in selections],
        "expected": [row["expected_index"] for row in selections],
        "key_cosine": report["key_cosine_span9_span10"],
        "span9": parent9["accuracy"],
        "span10": routed10_audit["accuracy"],
        "span10_direct": direct10_audit["accuracy"],
        "span9_on_span10": routed9_on10["accuracy"],
        "span10_on_span9": routed10_on9["accuracy"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
