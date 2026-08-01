"""Audit learning through disk memory with a completely frozen controller.

This is the smallest honest test of the memory-only claim.  The controller
produces support keys and values, all parameters remain immutable, process
state is reset before the query, and the value is recovered only through a
serialized :class:`DiskLatentMemory`.  The task is deliberately a mastered
binary mapping: this script proves episodic acquisition through external
memory, not discovery of a new primitive.  The adjacent-rule Gate 1 audit
remains separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from .environment import NULL_ACTION
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .train_persistent_memory import persistent_rollout


def _state_digest(model: UnifiedCognitiveController) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def _disk_read(
        model: UnifiedCognitiveController,
        result: dict[str, torch.Tensor], *, capacity: int,
        device: torch.device,
        ) -> torch.Tensor:
    """Write and reload each physical bank, then perform hard disk reads."""
    count = result["query_keys"].shape[0]
    if count % capacity:
        raise ValueError("contexts must be divisible by memory capacity")
    reads: list[torch.Tensor] = []
    for start in range(0, count, capacity):
        bank = DiskLatentMemory(
            width=model.width, capacity=capacity, device=device)
        stop = start + capacity
        bank.commit(
            result["store_keys"][start:stop],
            result["store_values"][start:stop],
            torch.ones(capacity, device=device), threshold=0.0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.pt"
            bank.save(path)
            restored = DiskLatentMemory.load(path, device=device)
            read, _ = restored.retrieve(
                result["query_keys"][start:stop], top_k=1)
        reads.append(read)
    return torch.cat(reads, dim=0)


@torch.no_grad()
def _actions(
        model: UnifiedCognitiveController,
        result: dict[str, torch.Tensor],
        retrieved_memory: torch.Tensor | None,
        device: torch.device,
        ) -> torch.Tensor:
    count = result["query_frames"].shape[0]
    null_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    output, _ = model.step(
        result["query_frames"], model.initial_state(count, device=device),
        null_action, zeros, zeros, retrieved_memory=retrieved_memory)
    return output.logits.argmax(-1)


def _accuracy(
        actions: torch.Tensor, correct: torch.Tensor) -> float:
    return float((actions == correct).float().mean())


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = _state_digest(model)

    normal = persistent_rollout(
        model, count=args.contexts, capacity=args.memory_capacity,
        seed=args.seed, device=device, sample_actions=False,
        memory_mode="hard")
    reversed_result = persistent_rollout(
        model, count=args.contexts, capacity=args.memory_capacity,
        seed=args.seed, device=device, sample_actions=False,
        memory_mode="hard", reverse_rules=True)

    disk_value = _disk_read(
        model, normal, capacity=args.memory_capacity, device=device)
    reversed_disk_value = _disk_read(
        model, reversed_result, capacity=args.memory_capacity, device=device)
    disk_actions = _actions(model, normal, disk_value, device)
    reversed_disk_actions = _actions(
        model, reversed_result, reversed_disk_value, device)
    no_memory_actions = _actions(model, normal, None, device)
    shuffled_actions = _actions(
        model, normal, disk_value.roll(1, dims=0), device)
    corrupted_actions = _actions(
        model, normal, disk_value.flip(dims=(-1,)), device)
    oracle_actions = _actions(
        model, normal, normal["store_values"], device)
    after = _state_digest(model)

    normal_accuracy = _accuracy(disk_actions, normal["correct_actions"])
    reversed_accuracy = _accuracy(
        reversed_disk_actions, reversed_result["correct_actions"])
    no_memory_accuracy = _accuracy(
        no_memory_actions, normal["correct_actions"])
    shuffled_accuracy = _accuracy(
        shuffled_actions, normal["correct_actions"])
    corrupted_accuracy = _accuracy(
        corrupted_actions, normal["correct_actions"])
    report: dict[str, object] = {
        "schema": "frozen-external-memory-audit-v1",
        "checkpoint": str(args.checkpoint),
        "contexts": args.contexts,
        "memory_capacity": args.memory_capacity,
        "device": str(device),
        "controller_frozen": all(
            not parameter.requires_grad for parameter in model.parameters()),
        "state_digest_before": before,
        "state_digest_after": after,
        "weights_unchanged": before == after,
        "disk_memory_accuracy": normal_accuracy,
        "reversed_disk_memory_accuracy": reversed_accuracy,
        "paired_prediction_flip_rate": float(
            (disk_actions != reversed_disk_actions).float().mean()),
        "no_memory_accuracy": no_memory_accuracy,
        "shuffled_memory_accuracy": shuffled_accuracy,
        "corrupted_memory_accuracy": corrupted_accuracy,
        "oracle_same_context_accuracy": _accuracy(
            oracle_actions, normal["correct_actions"]),
        "claim_scope": (
            "frozen-weight episodic adaptation through serialized external "
            "memory; not adjacent-primitive discovery"),
    }
    gates = {
        "controller_frozen": bool(report["controller_frozen"]),
        "weights_unchanged": bool(report["weights_unchanged"]),
        "disk_memory_at_least_85": normal_accuracy >= 0.85,
        "reversed_at_least_85": reversed_accuracy >= 0.85,
        "prediction_flips_at_least_80":
            report["paired_prediction_flip_rate"] >= 0.80,
        "no_memory_hurts": no_memory_accuracy <= normal_accuracy - 0.15,
        "shuffled_memory_hurts":
            shuffled_accuracy <= normal_accuracy - 0.15,
        "corrupted_memory_hurts":
            corrupted_accuracy <= normal_accuracy - 0.15,
    }
    report["gates"] = gates
    report["accepted"] = all(gates.values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=256)
    parser.add_argument("--memory-capacity", type=int, default=8)
    parser.add_argument("--seed", type=int, default=18001)
    parser.add_argument("--device", default=_device_name())
    args = parser.parse_args()
    if args.contexts < args.memory_capacity:
        raise ValueError("contexts must be at least one memory bank")
    if args.contexts % args.memory_capacity:
        raise ValueError("contexts must be divisible by memory capacity")
    report = run(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
