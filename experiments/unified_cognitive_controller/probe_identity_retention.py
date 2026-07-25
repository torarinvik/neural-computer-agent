"""Disposable probe for visual-identity retention in frozen checkpoints.

Verifier-private identity labels train only a throwaway diagnostic head.  No
probe weights enter the agent or any capability claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .environment import generate_lifetimes
from .model import UnifiedCognitiveController
from .train import seed_everything


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _features(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        heldout: bool, device: torch.device) -> tuple[
            torch.Tensor, torch.Tensor]:
    batch = generate_lifetimes(
        count, 3, seed=seed, heldout=heldout,
        task="visible_identity", support_trials=1, device=device)
    frames = batch.frames.reshape(-1, 3, 32, 32)
    labels = batch.stimulus_identities.reshape(-1)
    return model.vision(frames), labels


def _fit_probe(
        train_x: torch.Tensor, train_y: torch.Tensor,
        test_x: torch.Tensor, test_y: torch.Tensor, *,
        seed: int, shuffled: bool) -> dict[str, float]:
    generator = torch.Generator(device=train_x.device).manual_seed(seed)
    labels = train_y
    if shuffled:
        labels = labels[torch.randperm(
            labels.numel(), generator=generator, device=labels.device)]
    probe = nn.Sequential(
        nn.LayerNorm(train_x.shape[-1]),
        nn.Linear(train_x.shape[-1], 64),
        nn.GELU(),
        nn.Linear(64, 2),
    ).to(train_x.device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(400):
        indices = torch.randint(
            0, train_x.shape[0], (512,), generator=generator,
            device=train_x.device)
        loss = nn.functional.cross_entropy(
            probe(train_x[indices]), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = (
            probe(train_x).argmax(-1) == labels).float().mean()
        test_accuracy = (
            probe(test_x).argmax(-1) == test_y).float().mean()
    return {
        "train_accuracy": float(train_accuracy),
        "heldout_accuracy": float(test_accuracy),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3501)
    parser.add_argument("--train-lifetimes", type=int, default=2048)
    parser.add_argument("--test-lifetimes", type=int, default=1024)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "unified-cognitive-controller-v1":
        raise ValueError("unsupported controller checkpoint")
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_x, train_y = _features(
        model, count=args.train_lifetimes, seed=args.seed,
        heldout=False, device=device)
    test_x, test_y = _features(
        model, count=args.test_lifetimes, seed=args.seed + 1,
        heldout=True, device=device)
    report = {
        "schema": "unified-controller-disposable-identity-probe-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "diagnostic_only": True,
        "probe_weights_discarded": True,
        "agent_weights_changed": False,
        "verifier_private_labels_used_by_probe": True,
        "normal_labels": _fit_probe(
            train_x, train_y, test_x, test_y,
            seed=args.seed + 2, shuffled=False),
        "shuffled_labels": _fit_probe(
            train_x, train_y, test_x, test_y,
            seed=args.seed + 3, shuffled=True),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
