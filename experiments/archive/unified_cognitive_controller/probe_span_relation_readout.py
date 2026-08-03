"""Disposable probe for a frozen controller's multi-item relation state.

The probe is strictly diagnostic.  It receives a state only after the normal
pixel-only rollout has processed the fourth-item query.  The verifier's
correct action is used only to determine whether that frozen state contains
the relation; probe weights are never written back to the controller.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .legacy_model import UnifiedCognitiveController
from .train import configure_compute, seed_everything
from .train_procedural_shape_span import (
    generate_procedural_shape_batch,
    nuisance_from_level,
    rollout_procedural_shape_span,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _features(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        heldout: bool, span: int, anchor_focus: int,
        query_thought_steps: int, device: torch.device) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = generate_procedural_shape_batch(
        count, span=span, vocabulary=2, seed=seed,
        nuisance=nuisance_from_level(0.135), heldout=heldout,
        objective="recognition", query_count=1, next_query_stage=2,
        next_query_anchor_focus=anchor_focus, next_query_target_aligned=True,
        device=device)
    rollout = rollout_procedural_shape_span(
        model, batch, sample_actions=False,
        query_thought_steps=query_thought_steps)
    return (
        rollout["final_hidden"],
        rollout["final_workspace"].flatten(1),
        batch.correct_actions[:, 0])


def _fit_probe(
        train_x: torch.Tensor, train_y: torch.Tensor,
        test_x: torch.Tensor, test_y: torch.Tensor, *,
        seed: int, shuffled: bool, linear: bool) -> dict[str, float]:
    generator = torch.Generator(device=train_x.device).manual_seed(seed)
    labels = train_y
    if shuffled:
        labels = labels[torch.randperm(
            labels.numel(), generator=generator, device=labels.device)]
    probe = (
        nn.Sequential(
            nn.LayerNorm(train_x.shape[1]), nn.Linear(train_x.shape[1], 2))
        if linear else nn.Sequential(
            nn.LayerNorm(train_x.shape[1]),
            nn.Linear(train_x.shape[1], 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )).to(train_x.device)
    optimizer = torch.optim.AdamW(
        probe.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(300):
        indices = torch.randint(
            0, train_x.shape[0], (256,), generator=generator,
            device=train_x.device)
        loss = nn.functional.cross_entropy(probe(train_x[indices]), labels[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        return {
            "train_accuracy": float(
                (probe(train_x).argmax(-1) == labels).float().mean()),
            "heldout_accuracy": float(
                (probe(test_x).argmax(-1) == test_y).float().mean()),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=44850)
    parser.add_argument("--train-lifetimes", type=int, default=1024)
    parser.add_argument("--test-lifetimes", type=int, default=1024)
    parser.add_argument("--span", type=int, default=4)
    parser.add_argument("--next-query-anchor-focus", type=int, default=-1)
    parser.add_argument("--query-thought-steps", type=int, default=0)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--cpu-threads", type=int, default=0)
    args = parser.parse_args()
    logical_patterns = (2 * 2) ** args.span
    if (args.train_lifetimes % logical_patterns
            or args.test_lifetimes % logical_patterns):
        raise ValueError(
            f"probe lifetime counts must be multiples of {logical_patterns}")
    anchor_focus = args.next_query_anchor_focus
    if anchor_focus < 0:
        anchor_focus = args.span - 2
    if not 0 <= anchor_focus < args.span - 1:
        raise ValueError("anchor focus must identify a non-final item")
    compute = configure_compute(args.cpu_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if payload.get("schema") != "unified-cognitive-controller-v1":
        raise ValueError("unsupported controller checkpoint")
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    train_hidden, train_workspace, train_y = _features(
        model, count=args.train_lifetimes, seed=args.seed, heldout=False,
        span=args.span, anchor_focus=anchor_focus,
        query_thought_steps=args.query_thought_steps, device=device)
    test_hidden, test_workspace, test_y = _features(
        model, count=args.test_lifetimes, seed=args.seed + 1, heldout=True,
        span=args.span, anchor_focus=anchor_focus,
        query_thought_steps=args.query_thought_steps, device=device)
    features = {
        "hidden": (train_hidden, test_hidden),
        "workspace": (train_workspace, test_workspace),
        "combined": (
            torch.cat((train_hidden, train_workspace), dim=1),
            torch.cat((test_hidden, test_workspace), dim=1)),
    }
    report = {
        "schema": "procedural-shape-relation-readout-probe-v2",
        "span": args.span,
        "next_query_anchor_focus": anchor_focus,
        "query_thought_steps": args.query_thought_steps,
        "diagnostic_only": True,
        "probe_weights_discarded": True,
        "agent_weights_changed": False,
        "verifier_private_labels_used_by_probe": True,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "compute": compute,
        "normal_labels": {
            family: {
                name: _fit_probe(
                    train_x, train_y, test_x, test_y,
                    seed=args.seed + 2 + index, shuffled=False,
                    linear=(family == "linear"))
                for index, (name, (train_x, test_x)) in enumerate(features.items())}
            for family in ("linear", "mlp")},
        "shuffled_labels": {
            family: {
                name: _fit_probe(
                    train_x, train_y, test_x, test_y,
                    seed=args.seed + 8 + index, shuffled=True,
                    linear=(family == "linear"))
                for index, (name, (train_x, test_x)) in enumerate(features.items())}
            for family in ("linear", "mlp")},
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
