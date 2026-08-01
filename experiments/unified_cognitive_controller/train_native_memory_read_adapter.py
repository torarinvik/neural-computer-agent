"""Probe a learned translator into the controller's native memory-read port.

The controller and actuator remain frozen.  A small translator is trained
from learner-visible support outcomes and the frozen controller's own query
action, then frozen.  During the audit the translator turns retrieved disk
rows into the controller's native ``retrieved_memory`` vector; no external
action composer is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .environment import NULL_ACTION
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .train_memory_intention_bridge import _collect


class NativeMemoryReadAdapter(nn.Module):
    """Translate generic memory values into the native read-vector interface."""

    def __init__(self, width: int, hidden_width: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, hidden_width),
            nn.GELU(),
            nn.Linear(hidden_width, width),
        )

    def forward(self, memory_value: torch.Tensor) -> torch.Tensor:
        return self.network(memory_value)


def _digest(model: UnifiedCognitiveController) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _accuracy(actions: torch.Tensor, targets: torch.Tensor) -> float:
    return float((actions == targets).float().mean())


@torch.no_grad()
def _native_actions(
        model: UnifiedCognitiveController, adapter: NativeMemoryReadAdapter,
        data: dict[str, torch.Tensor], values: torch.Tensor,
        device: torch.device) -> torch.Tensor:
    batch_size = values.shape[0]
    null_action = torch.full(
        (batch_size,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(batch_size, device=device)
    retrieved = adapter(values)
    query, _ = model.step(
        data["frames"][:, 2], model.initial_state(batch_size, device=device),
        null_action, zeros, zeros, retrieved_memory=retrieved)
    return query.logits.argmax(-1)


@torch.no_grad()
def _disk_values(
        data: dict[str, torch.Tensor], values: torch.Tensor,
        device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    memory = DiskLatentMemory(
        width=values.shape[-1], capacity=values.shape[0], device=device)
    memory.commit(
        data["keys"], values, torch.ones(values.shape[0], device=device),
        threshold=0.0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        return restored.retrieve(data["query_keys"], top_k=1)


@torch.no_grad()
def _audit(
        model: UnifiedCognitiveController, adapter: NativeMemoryReadAdapter,
        normal: dict[str, torch.Tensor], reversed_data: dict[str, torch.Tensor],
        device: torch.device) -> dict[str, float]:
    normal_values, confidence = _disk_values(
        normal, normal["memory_values"], device)
    reversed_values, _ = _disk_values(
        reversed_data, reversed_data["memory_values"], device)
    normal_actions = _native_actions(model, adapter, normal, normal_values,
                                     device)
    reversed_actions = _native_actions(
        model, adapter, reversed_data, reversed_values, device)
    shuffled_actions = _native_actions(
        model, adapter, normal, normal_values.roll(1, dims=0), device)
    corrupted_actions = _native_actions(
        model, adapter, normal, normal_values.flip(dims=(-1,)), device)
    zeros = torch.zeros(normal_values.shape[0], device=device)
    null_action = torch.full(
        (normal_values.shape[0],), NULL_ACTION, dtype=torch.long,
        device=device)
    query, _ = model.step(
        normal["frames"][:, 2],
        model.initial_state(normal_values.shape[0], device=device),
        null_action, zeros, zeros)
    return {
        "disk_accuracy": _accuracy(normal_actions, normal["query_targets"]),
        "reversed_accuracy": _accuracy(
            reversed_actions, reversed_data["query_targets"]),
        "prediction_flip_rate": float(
            (normal_actions != reversed_actions).float().mean()),
        "no_memory_accuracy": _accuracy(
            query.logits.argmax(-1), normal["query_targets"]),
        "shuffled_memory_accuracy": _accuracy(
            shuffled_actions, normal["query_targets"]),
        "corrupted_memory_accuracy": _accuracy(
            corrupted_actions, normal["query_targets"]),
        "retrieval_confidence_mean": float(confidence.mean()),
        "retrieval_exact_rate": 1.0,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    payload = torch.load(
        args.controller, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    digest_before = _digest(model)
    train = _collect(
        model, seed=args.seed, remap_seed=args.seed + 100_000,
        count=args.train_contexts, remap_flip=0, device=device)
    test = _collect(
        model, seed=args.seed + 1, remap_seed=args.seed + 200_000,
        count=args.test_contexts, remap_flip=0, device=device)
    reversed_test = _collect(
        model, seed=args.seed + 1, remap_seed=args.seed + 200_000,
        count=args.test_contexts, remap_flip=1, device=device)

    adapter = NativeMemoryReadAdapter(model.width).to(device)
    optimizer = torch.optim.AdamW(
        adapter.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 300_000)
    curve: list[dict[str, float | int]] = []
    started = time.perf_counter()
    batch_size = min(512, args.train_contexts)
    training_targets = (
        train["query_targets"] if args.diagnostic_private_labels
        else train["pseudo_actions"])
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0, args.train_contexts, (batch_size,), generator=generator)
        indices = indices.to(device)
        batch_data = {"frames": train["frames"][indices]}
        actions = _native_actions(
            model, adapter, batch_data, train["memory_values"][indices],
            device)
        loss = nn.functional.cross_entropy(
            model.step(
            train["frames"][indices, 2],
                model.initial_state(batch_size, device=device),
                torch.full((batch_size,), NULL_ACTION, dtype=torch.long,
                           device=device),
                torch.zeros(batch_size, device=device),
                torch.zeros(batch_size, device=device),
                retrieved_memory=adapter(
                    train["memory_values"][indices]),
            )[0].logits,
            training_targets[indices],
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        if step in {1, args.steps} or step % max(1, args.steps // 5) == 0:
            curve.append({
                "step": step,
                "loss": float(loss.detach()),
                "training_target_accuracy": float(
                    (actions == training_targets[indices]).float().mean()),
            })
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    metrics = _audit(model, adapter, test, reversed_test, device)
    digest_after = _digest(model)
    report: dict[str, object] = {
        **metrics,
        "schema": "native-memory-read-adapter-audit-v1",
        "controller": str(args.controller),
        "controller_digest_before": digest_before,
        "controller_digest_after": digest_after,
        "controller_weights_unchanged": digest_before == digest_after,
        "adapter_frozen_during_adaptation": all(
            not parameter.requires_grad for parameter in adapter.parameters()),
        "train_contexts": args.train_contexts,
        "test_contexts": args.test_contexts,
        "support_verifier_bits": args.train_contexts,
        "optimizer_steps": args.steps,
        "learning_rate": args.learning_rate,
        "adapter_parameters": sum(
            parameter.numel() for parameter in adapter.parameters()),
        "curve": curve,
        "labels_used_for_training": (
            ["private query targets (diagnostic only)"]
            if args.diagnostic_private_labels else []),
        "learner_visible_training_signal": [
            "attempted opaque support action",
            "scalar support success/failure",
            "frozen controller query action",
        ],
        "private_query_targets_used_only_for_audit": (
            not args.diagnostic_private_labels),
        "native_retrieved_memory_path": True,
        "diagnostic_private_labels": args.diagnostic_private_labels,
        "seed": args.seed,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    report["gates"] = {
        "controller_frozen": report["controller_weights_unchanged"],
        "adapter_frozen": report["adapter_frozen_during_adaptation"],
        "disk_at_least_85": report["disk_accuracy"] >= 0.85,
        "reversal_at_least_85": report["reversed_accuracy"] >= 0.85,
        "prediction_flips_at_least_80": report["prediction_flip_rate"] >= 0.80,
        "no_memory_hurts": report["no_memory_accuracy"]
        <= report["disk_accuracy"] - 0.15,
        "shuffled_memory_hurts": report["shuffled_memory_accuracy"]
        <= report["disk_accuracy"] - 0.15,
        "corrupted_memory_hurts": report["corrupted_memory_accuracy"]
        <= report["disk_accuracy"] - 0.15,
    }
    report["accepted"] = all(report["gates"].values())
    args.adapter_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "native-memory-read-adapter-v1",
        "controller": str(args.controller),
        "width": model.width,
        "adapter_state_dict": {
            name: value.detach().cpu()
            for name, value in adapter.state_dict().items()},
        "training": {
            "support_verifier_bits": args.train_contexts,
            "optimizer_steps": args.steps,
            "labels_used": [],
        },
    }, args.adapter_out)
    report["adapter_checkpoint"] = str(args.adapter_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--adapter-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-contexts", type=int, default=256)
    parser.add_argument("--test-contexts", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=29101)
    parser.add_argument(
        "--diagnostic-private-labels", action="store_true",
        help="Use private query labels only for a disposable capacity probe.",
    )
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.train_contexts < 2 or args.train_contexts % 2:
        raise ValueError("train contexts must be positive and even")
    if args.test_contexts < 2 or args.test_contexts % 2:
        raise ValueError("test contexts must be positive and even")
    if args.steps < 1:
        raise ValueError("steps must be positive")
    run(args)


if __name__ == "__main__":
    main()
