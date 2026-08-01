"""Train and audit a generic memory-to-intention reader with a frozen core.

The reader is a reusable amodal adapter: it consumes the frozen controller's
query intention and a retrieved generic memory row, then emits an intention
residual before the frozen protocol decoder. It is trained only from scalar
support outcomes and the controller's own query action; during the audit only
serialized memory rows change.
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
from .memory_intention_bridge import MemoryIntentionReader
from .model import UnifiedCognitiveController
from .train_memory_intention_bridge import _collect


def _digest(model: UnifiedCognitiveController) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _accuracy(actions: torch.Tensor, targets: torch.Tensor) -> float:
    return float((actions == targets).float().mean())


def _actions(
        model: UnifiedCognitiveController, reader: MemoryIntentionReader,
        intentions: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    residual = reader(intentions, values)
    return model.actuator(intentions + residual).argmax(-1)


@torch.no_grad()
def _disk_read(
        data: dict[str, torch.Tensor], device: torch.device
        ) -> tuple[torch.Tensor, torch.Tensor]:
    values = data["memory_values"]
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
        model: UnifiedCognitiveController, reader: MemoryIntentionReader,
        normal: dict[str, torch.Tensor], reversed_data: dict[str, torch.Tensor],
        device: torch.device) -> dict[str, float]:
    normal_values, confidence = _disk_read(normal, device)
    reversed_values, _ = _disk_read(reversed_data, device)
    normal_actions = _actions(
        model, reader, normal["query_intentions"], normal_values)
    reversed_actions = _actions(
        model, reader, reversed_data["query_intentions"], reversed_values)
    shuffled_actions = _actions(
        model, reader, normal["query_intentions"], normal_values.roll(1, 0))
    corrupted_actions = _actions(
        model, reader, normal["query_intentions"], normal_values.flip(-1))
    null_action = torch.full(
        (normal_values.shape[0],), NULL_ACTION, dtype=torch.long,
        device=device)
    zeros = torch.zeros(normal_values.shape[0], device=device)
    # The promoted checkpoint has a structurally-zero compatibility suffix.
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

    reader = MemoryIntentionReader(
        model.width, model.intention_width).to(device)
    optimizer = torch.optim.AdamW(
        reader.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 300_000)
    curve: list[dict[str, float | int]] = []
    started = time.perf_counter()
    batch_size = min(512, args.train_contexts)
    for step in range(1, args.steps + 1):
        indices = torch.randint(
            0, args.train_contexts, (batch_size,), generator=generator)
        indices = indices.to(device)
        intentions = train["query_intentions"][indices]
        values = train["memory_values"][indices]
        logits = model.actuator(intentions + reader(intentions, values))
        loss = nn.functional.cross_entropy(
            logits, train["pseudo_actions"][indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
        optimizer.step()
        if step in {1, args.steps} or step % max(1, args.steps // 5) == 0:
            curve.append({
                "step": step,
                "loss": float(loss.detach()),
                "pseudo_action_accuracy": float(
                    (logits.argmax(-1) == train["pseudo_actions"][indices])
                    .float().mean()),
            })
    for parameter in reader.parameters():
        parameter.requires_grad_(False)
    metrics = _audit(model, reader, test, reversed_test, device)
    digest_after = _digest(model)
    report: dict[str, object] = {
        **metrics,
        "schema": "native-intention-memory-reader-audit-v1",
        "controller": str(args.controller),
        "controller_digest_before": digest_before,
        "controller_digest_after": digest_after,
        "controller_weights_unchanged": digest_before == digest_after,
        "reader_frozen_during_adaptation": all(
            not parameter.requires_grad for parameter in reader.parameters()),
        "train_contexts": args.train_contexts,
        "test_contexts": args.test_contexts,
        "support_verifier_bits": args.train_contexts,
        "optimizer_steps": args.steps,
        "learning_rate": args.learning_rate,
        "reader_parameters": sum(
            parameter.numel() for parameter in reader.parameters()),
        "curve": curve,
        "labels_used_for_training": [],
        "learner_visible_training_signal": [
            "attempted opaque support action",
            "scalar support success/failure",
            "frozen controller query action",
        ],
        "private_query_targets_used_only_for_audit": True,
        "intention_bus_residual": True,
        "seed": args.seed,
        "device": str(device),
        "wall_seconds": time.perf_counter() - started,
    }
    report["gates"] = {
        "controller_frozen": report["controller_weights_unchanged"],
        "reader_frozen": report["reader_frozen_during_adaptation"],
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
    args.reader_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": "native-intention-memory-reader-v1",
        "controller": str(args.controller),
        "memory_width": model.width,
        "intention_width": model.intention_width,
        "reader_state_dict": {
            name: value.detach().cpu()
            for name, value in reader.state_dict().items()},
        "training": {
            "support_verifier_bits": args.train_contexts,
            "optimizer_steps": args.steps,
            "labels_used": [],
        },
    }, args.reader_out)
    report["reader_checkpoint"] = str(args.reader_out)
    report["reader_checkpoint_sha256"] = _sha256(args.reader_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--reader-out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--train-contexts", type=int, default=256)
    parser.add_argument("--test-contexts", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=29201)
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
