"""Tiny cached diagnostic: can the consolidator preserve the transferred rule?"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_neural_computer.training_memory import DifferentiableBatchMemory
from .consolidator import LatentConsolidator
from .environment import SHOTS, generate_temporal_attention_lifetime
from .probe_temporal_rule_memory import _load
from .train import _append, seed_everything
from .train_consolidator import _initial_memory


def _cache(model, consolidator, *, start, lifetimes, batch_size, device):
    cached = {shot: [] for shot in SHOTS}
    labels = []
    for offset in range(0, lifetimes, batch_size):
        count = min(batch_size, lifetimes - offset)
        items = [generate_temporal_attention_lifetime(
            start + offset + i, heldout=start >= 42_000_000,
            feedback_mode="color-button") for i in range(count)]
        y = torch.tensor([item.rule for item in items], device=device)
        labels.append(y)
        memory = _initial_memory(model, items, device)
        cursor = 0
        for shot in SHOTS:
            while cursor < shot:
                with torch.no_grad():
                    appended, _, _ = _append(
                        model, [item.supports[cursor] for item in items], memory, device)
                    memory = consolidator(appended)
                cursor += 1
            # The input to this consolidation step is the memory before it;
            # replaying it is enough to test preservation without rerunning pixels.
            if shot == 0:
                pre = _initial_memory(model, items, device)
            else:
                # Reconstruct the pre-consolidation memory for this shot.
                pre = _initial_memory(model, items, device)
                for index in range(shot):
                    with torch.no_grad():
                        appended, _, _ = _append(
                            model, [item.supports[index] for item in items], pre, device)
                    pre = appended if index == shot - 1 else consolidator(appended)
            cached[shot].append((pre.keys.detach(), pre.values.detach(),
                                 pre.strengths.detach(), pre.admissions.detach()))
    merged = {}
    for shot in SHOTS:
        parts = list(zip(*cached[shot]))
        merged[shot] = tuple(torch.cat(part, dim=0) for part in parts)
    return merged, torch.cat(labels)


def _fit(consolidator, cached, labels, *, steps, seed, device):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.LayerNorm(320), nn.Linear(320, 2)).to(device)
    opt = torch.optim.AdamW((*consolidator.parameters(), *head.parameters()), lr=3e-4)
    for _ in range(steps):
        shot = SHOTS[torch.randint(len(SHOTS), ()).item()]
        keys, values, strengths, admissions = cached[shot]
        idx = torch.randint(labels.numel(), (min(64, labels.numel()),), device=device)
        mem = DifferentiableBatchMemory(len(idx), 160, device=device,
                                        keys=keys[idx], values=values[idx],
                                        strengths=strengths[idx], admissions=admissions[idx])
        compact = consolidator(mem)
        logits = head(torch.cat((compact.keys[:, 0], compact.values[:, 0]), dim=-1))
        loss = nn.functional.cross_entropy(logits, labels[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    return head


@torch.no_grad()
def _accuracy(consolidator, head, cached, labels, device):
    out = {}
    for shot in SHOTS:
        keys, values, strengths, admissions = cached[shot]
        mem = DifferentiableBatchMemory(labels.numel(), 160, device=device,
                                        keys=keys, values=values,
                                        strengths=strengths, admissions=admissions)
        compact = consolidator(mem)
        pred = head(torch.cat((compact.keys[:, 0], compact.values[:, 0]), dim=-1)).argmax(-1)
        out[str(shot)] = float((pred == labels).float().mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controller-checkpoint", type=Path, required=True)
    p.add_argument("--consolidator-checkpoint", type=Path, required=True)
    p.add_argument("--pairwise-transfer-checkpoint", type=Path, required=True)
    p.add_argument("--projection-transfer-checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--train-lifetimes", type=int, default=128)
    p.add_argument("--test-lifetimes", type=int, default=128)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", default="cuda")
    args = p.parse_args(); seed_everything(83); device = torch.device(args.device)
    model, base_consolidator = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device,
        transfer_paths=(str(args.pairwise_transfer_checkpoint),
                        str(args.projection_transfer_checkpoint)),
        transfer_strength=0.01)
    train_cache, train_y = _cache(model, base_consolidator, start=40_000_000,
                                  lifetimes=args.train_lifetimes, batch_size=64, device=device)
    test_cache, test_y = _cache(model, base_consolidator, start=42_000_000,
                                lifetimes=args.test_lifetimes, batch_size=64, device=device)
    candidate = LatentConsolidator(model.hidden, heads=5).to(device)
    candidate.load_state_dict(base_consolidator.state_dict()); candidate.train()
    head = _fit(candidate, train_cache, train_y, steps=args.steps, seed=83, device=device)
    result = {"train": _accuracy(candidate, head, train_cache, train_y, device),
              "test": _accuracy(candidate, head, test_cache, test_y, device)}
    shuffled = train_y[torch.randperm(train_y.numel())]
    shuffled_candidate = LatentConsolidator(model.hidden, heads=5).to(device)
    shuffled_candidate.load_state_dict(base_consolidator.state_dict()); shuffled_candidate.train()
    shuffled_head = _fit(shuffled_candidate, train_cache, shuffled, steps=args.steps,
                          seed=84, device=device)
    result["shuffled_test"] = _accuracy(shuffled_candidate, shuffled_head,
                                         test_cache, test_y, device)
    report = {"schema": "cached-consolidator-auxiliary-v1", "steps": args.steps,
              "train_lifetimes": args.train_lifetimes, "test_lifetimes": args.test_lifetimes,
              "transfer_strength": 0.01, **result}
    args.report.write_text(json.dumps(report, indent=2) + "\n"); print(json.dumps(report))


if __name__ == "__main__": main()
