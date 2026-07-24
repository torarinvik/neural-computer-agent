"""Tiny capacity control for the exact in-agent event binder on frozen snapshots."""
from __future__ import annotations

import argparse
import json
import torch
from torch import nn

from experiments.syllogimous_neural_computer.model import EventSnapshotWriteBinder
from .probe_temporal_event_snapshot_binder import _extract, _load
from .train import seed_everything


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--controller-checkpoint", required=True)
    p.add_argument("--consolidator-checkpoint", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--train-lifetimes", type=int, default=1024)
    p.add_argument("--test-lifetimes", type=int, default=1024)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--open-gate", action="store_true")
    p.add_argument("--open-relation", action="store_true")
    p.add_argument("--direct-bound", action="store_true",
                   help="bypass write-source gate and classify the relation latent directly")
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--device", default="cuda")
    p.add_argument("--train-start", type=int, default=29_000_000)
    p.add_argument("--test-start", type=int, default=31_000_000)
    a = p.parse_args(); seed_everything(a.seed)
    device = torch.device(a.device)
    model, _ = _load(a.controller_checkpoint, a.consolidator_checkpoint, device)
    train_x, train_y, _ = _extract(model, start=a.train_start, lifetimes=a.train_lifetimes,
                                   batch_size=256, heldout=False, feedback_mode="color-button",
                                   render_variants=1, device=device)
    test_x, test_y, _ = _extract(model, start=a.test_start, lifetimes=a.test_lifetimes,
                                  batch_size=256, heldout=True, feedback_mode="color-button",
                                  render_variants=1, device=device)
    binder = EventSnapshotWriteBinder(model.hidden, width=64).to(device)
    if a.open_relation:
        nn.init.kaiming_uniform_(binder.relation[-1].weight, a=5 ** 0.5)
        nn.init.zeros_(binder.relation[-1].bias)
    if a.open_gate:
        nn.init.constant_(binder.gate.bias, 0.0)
    head = nn.Sequential(nn.LayerNorm(model.hidden), nn.Linear(model.hidden, 64),
                         nn.GELU(), nn.Linear(64, 2)).to(device)
    opt = torch.optim.AdamW((*binder.parameters(), *head.parameters()), lr=1e-3)
    def latent(source, snaps, mask):
        if not a.direct_bound:
            return binder(source, snaps, mask)
        events, _ = binder.recent_events(snaps, mask)
        events = binder.project(events) + binder.positions
        rel = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            rel.extend((events[:, left] * events[:, right],
                        (events[:, left] - events[:, right]).abs()))
        return binder.relation(torch.cat((*events.unbind(dim=1), *rel), dim=-1))
    source_train = torch.zeros(train_x.shape[0], model.hidden, device=device)
    for _ in range(a.steps):
        idx = torch.randint(train_x.shape[0], (min(128, train_x.shape[0]),), device=device)
        cpu_idx = idx.cpu()
        out = latent(source_train[idx], train_x[cpu_idx].to(device),
                     torch.ones(idx.shape[0], 3, dtype=torch.bool, device=device))
        loss = nn.functional.cross_entropy(head(out), train_y[cpu_idx].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    @torch.no_grad()
    def acc(x, y):
        out = latent(torch.zeros(x.shape[0], model.hidden, device=device), x.to(device),
                     torch.ones(x.shape[0], 3, dtype=torch.bool, device=device))
        return float((head(out).argmax(-1).cpu() == y).float().mean())
    result = {"train_accuracy": acc(train_x, train_y), "test_accuracy": acc(test_x, test_y),
              "baseline": float(test_y.float().mean()), "steps": a.steps,
              "schema": "integrated-binder-capacity-v1"}
    open(a.report, "w").write(json.dumps(result, indent=2) + "\n"); print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
