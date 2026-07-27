"""Disposable capacity check for the action-residual integration interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from .environment import NULL_ACTION, generate_lifetimes
from .model import UnifiedCognitiveController
from .train import evaluate


def _load(path: Path, device: torch.device):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(**payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model


def _logits(model: UnifiedCognitiveController, frames: torch.Tensor) -> torch.Tensor:
    count = frames.shape[0]
    state = model.initial_state(count, device=frames.device)
    null = torch.full((count,), NULL_ACTION, dtype=torch.long, device=frames.device)
    zeros = torch.zeros(count, device=frames.device)
    output, _ = model.step(frames, state, null, zeros, zeros)
    return output.logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-lifetimes", type=int, default=2048)
    parser.add_argument("--test-lifetimes", type=int, default=512)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--shuffle-target", action="store_true")
    parser.add_argument("--gated", action="store_true")
    parser.add_argument("--retain-binary", action="store_true")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    parent_payload, parent = _load(args.parent, device)
    _, specialist = _load(args.specialist, device)
    config = dict(parent_payload["model_configuration"])
    config["action_adapter_width"] = args.width
    config["action_adapter_gated"] = args.gated
    student = UnifiedCognitiveController(**config).to(device)
    student.load_state_dict(parent.state_dict(), strict=False)
    assert student.action_adapter is not None
    train = generate_lifetimes(
        args.train_lifetimes, 3, seed=args.seed, task="visible_context", device=device)
    test = generate_lifetimes(
        args.test_lifetimes, 3, seed=args.seed + 1, heldout=True,
        task="visible_context", device=device)
    with torch.no_grad():
        target = _logits(specialist, train.frames[:, 0]).argmax(-1)
        test_target = _logits(specialist, test.frames[:, 0]).argmax(-1)
        if args.shuffle_target:
            target = target.roll(1)
        binary = generate_lifetimes(
            args.train_lifetimes, 3, seed=args.seed + 2,
            task="binary_mapping", device=device)
        binary_target = _logits(parent, binary.frames[:, 0]).argmax(-1)
    trainable = list(student.action_adapter.parameters())
    if student.action_adapter_gate is not None:
        trainable += list(student.action_adapter_gate.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=3e-3)
    for _ in range(args.steps):
        loss = nn.functional.cross_entropy(_logits(student, train.frames[:, 0]), target)
        if args.retain_binary:
            loss = loss + nn.functional.cross_entropy(
                _logits(student, binary.frames[:, 0]), binary_target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        train_accuracy = float((_logits(student, train.frames[:, 0]).argmax(-1) == target).float().mean())
        heldout_accuracy = float((_logits(student, test.frames[:, 0]).argmax(-1) == test_target).float().mean())
    behavior = evaluate(
        student, count=args.test_lifetimes, trials=6, seed=args.seed + 9_000_000,
        device=device, task="visible_context", feedback_trials=1)
    binary_retention = evaluate(
        student, count=args.test_lifetimes, trials=6, seed=args.seed + 9_000_001,
        device=device, task="binary_mapping", feedback_trials=1)
    report = {
        "schema": "action-adapter-capacity-probe-v1",
        "claim_boundary": "Teacher-behavior capacity diagnostic; probe weights are disposable.",
        "train_accuracy": train_accuracy,
        "heldout_first_frame_accuracy": heldout_accuracy,
        "behavior": behavior,
        "binary_retention": binary_retention,
        "shuffle_target": args.shuffle_target,
        "gated": args.gated,
        "retain_binary": args.retain_binary,
        "interface_viable": heldout_accuracy >= 0.90,
    }
    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": config,
            "state_dict": student.state_dict(),
            "source_report": str(args.report),
            "admission_status": "diagnostic_capacity_candidate",
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"train": train_accuracy, "heldout": heldout_accuracy,
                      "gate": behavior["gate"], "viable": report["interface_viable"]}))


if __name__ == "__main__":
    main()
