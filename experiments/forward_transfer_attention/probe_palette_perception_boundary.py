"""Probe identity and relation information before recurrent event mixing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from experiments.syllogimous_latent_agent.data import collate_episodes

from .environment import generate_temporal_attention_lifetime
from .probe_answer_fusion_input import _fit
from .probe_temporal_event_snapshot_binder import _reverse_episode_events
from .probe_palette_sample_efficiency import (
    TEST_START, TRAIN_START, _balanced_specs, _parse_pairs)
from .probe_temporal_rule_memory import _load
from .train import seed_everything


@torch.no_grad()
def _collect(model, *, start, lifetimes, palettes, heldout, batch_size, device,
             feedback_mode, reverse_events=False):
    vision_parts, fused_parts, rule_parts, identity_parts = [], [], [], []
    specs = _balanced_specs(start, lifetimes, palettes, heldout)
    for offset in range(0, len(specs), batch_size):
        batch_specs = specs[offset:offset + batch_size]
        items = [
            generate_temporal_attention_lifetime(
                seed, heldout=heldout, feedback_mode=feedback_mode,
                color_ids=palette)
            for seed, palette in batch_specs
        ]
        episodes = [item.supports[0] for item in items]
        if reverse_events:
            episodes = [_reverse_episode_events(episode)
                        for episode in episodes]
        batch = collate_episodes(episodes)
        frames = batch["frames"].to(device)
        pcm = batch["pcm"].to(device)
        flat_vision = model.vision(
            frames.reshape(-1, *frames.shape[2:]))
        vision_parts.append(
            flat_vision.reshape(frames.shape[0], frames.shape[1], -1).cpu())
        fused_parts.append(model._encode(frames, pcm).cpu())
        rule_parts.append(torch.tensor(
            [1 - item.rule if reverse_events else item.rule for item in items],
            dtype=torch.long))
        identity_parts.append(torch.tensor([
            (
                palette[item.support_features[0][
                    1 if reverse_events else 0]],
                palette[item.support_features[0][
                    0 if reverse_events else 1]],
                palette[item.support_features[0][item.rule]],
            )
            for item, (_seed, palette) in zip(items, batch_specs)
        ], dtype=torch.long))
    return (torch.cat(vision_parts), torch.cat(fused_parts),
            torch.cat(rule_parts), torch.cat(identity_parts))


def _mlp_reversal_audit(train_x, train_y, test_x, test_y, reversed_x,
                        reversed_y, *, seed, device):
    seed_everything(seed)
    mean = train_x.mean(0, keepdim=True)
    scale = train_x.std(0, keepdim=True).clamp_min(1e-5)
    train_x = ((train_x - mean) / scale).to(device)
    test_x = ((test_x - mean) / scale).to(device)
    reversed_x = ((reversed_x - mean) / scale).to(device)
    model = nn.Sequential(
        nn.Linear(train_x.shape[1], 64), nn.GELU(), nn.Linear(64, 2)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3, weight_decay=1e-3)
    train_y_device = train_y.to(device)
    for _ in range(300):
        loss = nn.functional.cross_entropy(model(train_x), train_y_device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        normal = model(test_x).argmax(-1).cpu()
        reversed_predictions = model(reversed_x).argmax(-1).cpu()
    return {
        "normal_accuracy": float((normal == test_y).float().mean()),
        "reversed_relabeled_accuracy": float(
            (reversed_predictions == reversed_y).float().mean()),
        "reversed_stale_label_accuracy": float(
            (reversed_predictions == test_y).float().mean()),
        "prediction_flip_rate": float(
            (normal != reversed_predictions).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller-checkpoint", type=Path, required=True)
    parser.add_argument("--consolidator-checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-palettes", default="0,1;1,2;2,3")
    parser.add_argument("--test-palettes", default="0,2;0,3;1,3")
    parser.add_argument("--train-lifetimes", type=int, default=120)
    parser.add_argument("--test-lifetimes", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument(
        "--feedback-mode", choices=("color-button", "color-object"),
        default="color-button")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    train_palettes = _parse_pairs(args.train_palettes)
    test_palettes = _parse_pairs(args.test_palettes)
    model, _ = _load(
        args.controller_checkpoint, args.consolidator_checkpoint, device)
    train = _collect(
        model, start=TRAIN_START, lifetimes=args.train_lifetimes,
        palettes=train_palettes, heldout=False,
        batch_size=args.batch_size, device=device,
        feedback_mode=args.feedback_mode)
    test = _collect(
        model, start=TEST_START + 400_000, lifetimes=args.test_lifetimes,
        palettes=test_palettes, heldout=True,
        batch_size=args.batch_size, device=device,
        feedback_mode=args.feedback_mode)
    reversed_test = _collect(
        model, start=TEST_START + 400_000, lifetimes=args.test_lifetimes,
        palettes=test_palettes, heldout=True,
        batch_size=args.batch_size, device=device,
        feedback_mode=args.feedback_mode, reverse_events=True)
    train_vision, train_fused, train_rule, train_ids = train
    test_vision, test_fused, test_rule, test_ids = test
    reversed_vision, reversed_fused, reversed_rule, reversed_ids = reversed_test
    results = {}
    for boundary, train_x, test_x in (
            ("vision", train_vision, test_vision),
            ("fused_event", train_fused, test_fused)):
        boundary_results = {}
        for position, name in enumerate(
                ("first_identity", "second_identity", "rewarded_identity")):
            boundary_results[name] = {
                "linear": _fit(
                    train_x[:, position], train_ids[:, position],
                    test_x[:, position], test_ids[:, position],
                    nonlinear=False, seed=args.seed, device=device),
                "mlp": _fit(
                    train_x[:, position], train_ids[:, position],
                    test_x[:, position], test_ids[:, position],
                    nonlinear=True, seed=args.seed, device=device),
            }
        boundary_results["rule_from_joint_events"] = {
            "linear": _fit(
                train_x.flatten(1), train_rule, test_x.flatten(1), test_rule,
                nonlinear=False, seed=args.seed, device=device),
            "mlp": _fit(
                train_x.flatten(1), train_rule, test_x.flatten(1), test_rule,
                nonlinear=True, seed=args.seed, device=device),
        }
        results[boundary] = boundary_results
    shuffled = train_rule[torch.randperm(
        train_rule.numel(),
        generator=torch.Generator().manual_seed(args.seed + 77))]
    results["fused_rule_shuffled_labels"] = _fit(
        train_fused.flatten(1), shuffled, test_fused.flatten(1), test_rule,
        nonlinear=True, seed=args.seed, device=device)
    results["fused_mlp_reversal_audit"] = _mlp_reversal_audit(
        train_fused.flatten(1), train_rule,
        test_fused.flatten(1), test_rule,
        reversed_fused.flatten(1), reversed_rule,
        seed=args.seed, device=device)
    results["vision_mlp_reversal_audit"] = _mlp_reversal_audit(
        train_vision.flatten(1), train_rule,
        test_vision.flatten(1), test_rule,
        reversed_vision.flatten(1), reversed_rule,
        seed=args.seed, device=device)
    results["reversal_metadata_check"] = {
        "first_second_swapped": bool(torch.equal(
            test_ids[:, :2], reversed_ids[:, (1, 0)])),
        "rewarded_identity_unchanged": bool(torch.equal(
            test_ids[:, 2], reversed_ids[:, 2])),
        "all_rule_labels_flipped": bool(torch.equal(
            1 - test_rule, reversed_rule)),
    }
    report = {
        "schema": "palette-perception-boundary-probe-v1",
        "controller_frozen": True,
        "disposable_supervised_diagnostic": True,
        "train_palettes": train_palettes,
        "heldout_palette_pairs": test_palettes,
        "train_lifetimes": args.train_lifetimes,
        "test_lifetimes": args.test_lifetimes,
        "feedback_mode": args.feedback_mode,
        "identity_majorities": {
            name: float(
                torch.bincount(values, minlength=4).max() / values.numel())
            for name, values in {
                "first": test_ids[:, 0],
                "second": test_ids[:, 1],
                "rewarded": test_ids[:, 2],
            }.items()
        },
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
