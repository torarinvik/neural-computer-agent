"""Audit learned replacement through bounded physical disk save/reload."""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from .audit_selective_disk import _gated_retrieve
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .train_adaptive_memory_read import _outcomes
from .train_memory_replacement import (
    _bank_reward,
    replacement_batch,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def _physical_policy(
        model: UnifiedCognitiveController, data: dict[str, object],
        actions: torch.Tensor, directory: Path, *,
        device: torch.device) -> dict[str, object]:
    reads = []
    before_rows = 0
    after_rows = 0
    capacity_growth = 0
    for bank in range(actions.shape[0]):
        memory = DiskLatentMemory(
            model.width, capacity=data["bank_keys"].shape[1],
            device=device)
        memory.commit(
            data["bank_keys"][bank],
            data["bank_values"][bank],
            data["bank_strengths"][bank],
            threshold=0.0)
        # These are insertion timestamps from the valid episode history. The
        # physical slots were permuted specifically to defeat slot shortcuts.
        memory.store.age[:memory.count].copy_(
            data["bank_ages"][bank].to(torch.long))
        memory.store.clock = int(data["bank_ages"][bank].max())
        before_rows += memory.count
        original_capacity = memory.store.capacity
        action = int(actions[bank])
        if action:
            memory.replace(
                action - 1,
                data["candidate_key"][bank],
                data["candidate_value"][bank],
                data["candidate_strength"][bank])
        capacity_growth += memory.store.capacity - original_capacity
        path = directory / f"bank-{bank:04d}.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        after_rows += restored.count
        reads.append(_gated_retrieve(
            model, restored, data["future_queries"][bank],
            read_threshold=None))
    memory = torch.cat(reads)
    outcomes = _outcomes(
        model, data["future_batch"], memory, device=device)
    return {
        "accuracy": float(outcomes.mean()),
        "before_rows": before_rows,
        "after_rows": after_rows,
        "capacity_growth": capacity_growth,
        "disk_bytes": sum(
            path.stat().st_size for path in directory.glob("*.pt")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--banks", type=int, default=512)
    parser.add_argument("--bank-capacity", type=int, default=4)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    data = replacement_batch(
        model, banks=args.banks, capacity=args.bank_capacity,
        seed=args.seed, device=device,
        write_threshold=args.write_threshold)
    learned = model.memory_replacement_scores(
        data["option_features"]).argmax(-1)
    shuffled_features = data["option_features"].clone()
    shuffled_features[:, 1:, 0] = (
        shuffled_features[:, 1:, 0].roll(1, dims=1))
    shuffled = model.memory_replacement_scores(
        shuffled_features).argmax(-1)
    target = data["target_action"]
    tensor_accuracy = float(_bank_reward(
        model, data, learned, device=device).mean())
    shuffled_tensor_accuracy = float(_bank_reward(
        model, data, shuffled, device=device).mean())
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        normal_directory = root / "normal"
        corrupted_directory = root / "age-corrupted"
        normal_directory.mkdir()
        corrupted_directory.mkdir()
        normal = _physical_policy(
            model, data, learned, normal_directory,
            device=device)
        corrupted = _physical_policy(
            model, data, shuffled, corrupted_directory,
            device=device)
    report = {
        "schema": "unified-controller-memory-replacement-disk-audit-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "seed": args.seed,
        "banks": args.banks,
        "bank_capacity": args.bank_capacity,
        "generated_contexts": data["generated_contexts"],
        "weights_changed": False,
        "semantic_or_utility_labels_used_for_training": False,
        "target_eviction_rate": float(
            (learned == target).float().mean()),
        "tensor_accuracy": tensor_accuracy,
        "age_corrupted_tensor_accuracy": shuffled_tensor_accuracy,
        "physical": normal,
        "age_corrupted_physical": corrupted,
    }
    report["gate"] = {
        "physical_accuracy_at_least_94":
            normal["accuracy"] >= 0.94,
        "tensor_disk_gap_at_most_2":
            abs(normal["accuracy"] - tensor_accuracy) <= 0.02,
        "target_eviction_at_least_80":
            report["target_eviction_rate"] >= 0.80,
        "bounded_row_count_preserved":
            normal["before_rows"] == normal["after_rows"]
            == args.banks * args.bank_capacity,
        "physical_capacity_never_grew":
            normal["capacity_growth"] == 0,
        "age_signal_is_physically_causal":
            corrupted["accuracy"] <= normal["accuracy"] - 0.10,
    }
    report["gate"]["accepted"] = all(report["gate"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
