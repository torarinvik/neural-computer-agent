"""Localize four-identity transfer failure at the mapping-memory boundary.

This is a frozen, sensory-only diagnostic. It compares the controller's
existing two-identity mapping behavior with each two-identity card inside the
four-identity compositional lifetime, both alone and after both cards are
stored. No verifier metadata enters the model; private fields select scored
query subsets only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.syllogimous_neural_computer.training_memory import (
    DifferentiableBatchMemory)

from .environment import (
    generate_compositional_temporal_attention_lifetime,
    generate_temporal_attention_lifetime)
from .probe_temporal_rule_memory import _load
from .train import _append, _forward, seed_everything


@torch.no_grad()
def _accuracy(model, episodes, memory, device):
    output, targets = _forward(model, episodes, memory, device)
    return float((output.answer_logits[:, -1].argmax(-1) == targets).float().mean())


@torch.no_grad()
def _append_study(model, items, memory, index, device):
    memory, _, _ = _append(
        model, [item.studies[index] for item in items], memory, device)
    return memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--transfer-strength", type=float, default=.01)
    parser.add_argument("--lifetimes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=563)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    transfers = (
        str(args.pairwise_transfer_checkpoint),
        str(args.projection_transfer_checkpoint))
    model, _ = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device,
        transfer_paths=transfers, transfer_strength=args.transfer_strength)
    start = 53_000_000 + args.seed * 10_000
    atom = [
        generate_temporal_attention_lifetime(
            start + index, heldout=True, query_count=1,
            feedback_mode="color-button")
        for index in range(args.lifetimes)]
    composed = [
        generate_compositional_temporal_attention_lifetime(
            start + 1_000_000 + index, heldout=True, query_count=1,
            feedback_mode="color-button")
        for index in range(args.lifetimes)]

    atom_memory = DifferentiableBatchMemory(
        args.lifetimes, model.hidden, device=device)
    atom_memory = _append_study(model, atom, atom_memory, 0, device)
    atom_scores = [
        _accuracy(
            model, [item.old_audit_queries[color] for item in atom],
            atom_memory, device)
        for color in range(2)]

    empty = DifferentiableBatchMemory(args.lifetimes, model.hidden, device=device)
    first_only = _append_study(model, composed, empty, 0, device)
    second_only = _append_study(model, composed, empty, 1, device)
    both = _append_study(model, composed, first_only, 1, device)

    def score(memory):
        return [
            _accuracy(
                model, [item.old_audit_queries[color] for item in composed],
                memory, device)
            for color in range(4)]

    result = {
        "schema": "mapping-capacity-probe-v1",
        "weights_frozen": True,
        "sensory_only": True,
        "lifetimes": args.lifetimes,
        "two_identity_one_card_per_color": atom_scores,
        "two_identity_one_card_mean": sum(atom_scores) / len(atom_scores),
        "four_identity_first_card_only_per_color": score(first_only),
        "four_identity_second_card_only_per_color": score(second_only),
        "four_identity_both_cards_per_color": score(both),
        "chance_action_accuracy": 1 / 8,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
