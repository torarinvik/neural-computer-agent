"""Sub-minute audit for the frozen pairwise relation -> writer adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.syllogimous_neural_computer.model import FixedPairwiseTransfer
from .probe_temporal_event_snapshot_binder import _extract
from .probe_temporal_rule_memory import _load
from .train import seed_everything


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--controller-checkpoint", type=Path, required=True)
    p.add_argument("--consolidator-checkpoint", type=Path, required=True)
    p.add_argument("--pairwise-checkpoint", type=Path, required=True)
    p.add_argument("--projection-checkpoint", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--lifetimes", type=int, default=128)
    p.add_argument("--start", type=int, default=33_000_000)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=29)
    args = p.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    controller, _ = _load(args.controller_checkpoint, args.consolidator_checkpoint, device)
    forward = _extract(controller, start=args.start, lifetimes=args.lifetimes,
                       batch_size=min(128, args.lifetimes), heldout=True,
                       feedback_mode="color-button", render_variants=1,
                       device=device)
    reverse = _extract(controller, start=args.start, lifetimes=args.lifetimes,
                       batch_size=min(128, args.lifetimes), heldout=True,
                       feedback_mode="color-button", render_variants=1,
                       reverse_events=True, device=device)
    transfer = FixedPairwiseTransfer(str(args.pairwise_checkpoint),
                                      str(args.projection_checkpoint),
                                      hidden=forward[0].shape[-1]).to(device).eval()
    payload = torch.load(args.projection_checkpoint, map_location=device, weights_only=False)
    head = torch.nn.Sequential(torch.nn.LayerNorm(160), torch.nn.Linear(160, 2)).to(device)
    head.load_state_dict(payload["head"])
    def classify(x):
        source = torch.zeros(x.shape[0], x.shape[-1], device=device)
        mask = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=device)
        with torch.no_grad():
            transfer.strength.fill_(1.0)
            delta = transfer(source, x.to(device), mask)
            return head(delta).argmax(-1).cpu()
    pred = classify(forward[0]); rev_pred = classify(reverse[0])
    y, rev_y = forward[1], reverse[1]
    report = {
        "normal_accuracy": float((pred == y).float().mean()),
        "reversed_accuracy": float((rev_pred == rev_y).float().mean()),
        "flip_rate": float((pred != rev_pred).float().mean()),
        "stale_label_accuracy": float((rev_pred == y).float().mean()),
        "zero_strength_is_exact_noop": None,
        "schema": "fixed-pairwise-transfer-audit-v1",
    }
    source = torch.randn(8, forward[0].shape[-1], device=device)
    x = forward[0][:8].to(device)
    mask = torch.ones(8, x.shape[1], dtype=torch.bool, device=device)
    with torch.no_grad():
        transfer.strength.zero_()
        baseline = transfer(source, x, mask)
        report["zero_strength_is_exact_noop"] = bool(torch.equal(baseline, source))
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
