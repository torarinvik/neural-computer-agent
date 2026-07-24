"""Cheap adversarial audit for an apparent temporal-feature decoding success.

The audit is deliberately diagnostic-only: it never changes the controller.  A
feature is only considered credible if its normal held-out score survives
label-shuffling, feature-row permutation, and zero-feature controls.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .probe_temporal_rule_memory import _extract, _load
from .probe_temporal_order import _fit_probe
from .train import seed_everything


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--controller-checkpoint", type=Path, required=True)
    p.add_argument("--consolidator-checkpoint", type=Path, required=True)
    p.add_argument("--pairwise-transfer-checkpoint", type=Path)
    p.add_argument("--projection-transfer-checkpoint", type=Path)
    p.add_argument("--transfer-strength", type=float, default=0.0)
    p.add_argument("--tap", default="latest_row_feature")
    p.add_argument("--shots", type=int, default=2)
    p.add_argument("--train-lifetimes", type=int, default=128)
    p.add_argument("--test-lifetimes", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--feedback-mode", choices=("white-button", "color-button"), default="color-button")
    p.add_argument("--seed", type=int, default=115)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    seed_everything(a.seed)
    device = torch.device(a.device)
    paths = None
    if a.pairwise_transfer_checkpoint or a.projection_transfer_checkpoint:
        if not (a.pairwise_transfer_checkpoint and a.projection_transfer_checkpoint):
            raise ValueError("both transfer checkpoints are required")
        paths = (str(a.pairwise_transfer_checkpoint), str(a.projection_transfer_checkpoint))
    model, consolidator = _load(a.controller_checkpoint, a.consolidator_checkpoint, device,
                                transfer_paths=paths, transfer_strength=a.transfer_strength)
    tx, ty = _extract(model, consolidator, start=11_000_000, lifetimes=a.train_lifetimes,
                      batch_size=a.batch_size, heldout=False, feedback_mode=a.feedback_mode,
                      device=device, preserve_raw_write=True)
    vx, vy = _extract(model, consolidator, start=13_000_000, lifetimes=a.test_lifetimes,
                      batch_size=a.batch_size, heldout=True, feedback_mode=a.feedback_mode,
                      device=device, preserve_raw_write=True)
    x, z = tx[a.shots][a.tap], vx[a.shots][a.tap]
    def fit(labels, test_labels, xx=x, zz=z):
        return _fit_probe(xx, labels, zz, test_labels, nonlinear=True, device=device, seed=a.seed)
    g = torch.Generator().manual_seed(a.seed + 991)
    shuffled = ty[torch.randperm(ty.numel(), generator=g)]
    row_perm = torch.randperm(x.shape[0], generator=g)
    test_perm = torch.randperm(z.shape[0], generator=g)
    zero_x, zero_z = torch.zeros_like(x), torch.zeros_like(z)
    out = {
        "schema": "temporal-feature-adversarial-audit-v1",
        "tap": a.tap, "shots": a.shots,
        "normal": fit(ty, vy),
        "shuffled_labels": fit(shuffled, vy),
        "permuted_rows": fit(ty, vy, x[row_perm], z[test_perm]),
        "zero_feature": fit(ty, vy, zero_x, zero_z),
        "train_examples": int(ty.numel()), "test_examples": int(vy.numel()),
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
