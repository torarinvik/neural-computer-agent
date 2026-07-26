"""Audit frequency-recency replacement through real bounded disk memory."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import torch

from .audit_selective_disk import _gated_retrieve
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .train_adaptive_memory_read import _outcomes
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_memory_replacement import _gather_rows, _select_batch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wrap_store(store) -> DiskLatentMemory:
    memory = DiskLatentMemory.__new__(DiskLatentMemory)
    memory.store = store
    return memory


@torch.no_grad()
def _materialize_histories(
        model: UnifiedCognitiveController, data: dict[str, object],
        directory: Path, *, device: torch.device
        ) -> tuple[
            list[DiskLatentMemory], torch.Tensor, int, int]:
    memories = []
    realized_counts = []
    persisted_exact = 0
    requested_exact = 0
    for bank in range(data["bank_keys"].shape[0]):
        memory = DiskLatentMemory(
            model.width, capacity=data["bank_keys"].shape[1],
            device=device)
        memory.commit(
            data["bank_keys"][bank],
            data["bank_values"][bank],
            data["bank_strengths"][bank],
            threshold=0.0)
        memory.store.age[:memory.count].copy_(
            data["bank_ages"][bank].to(torch.long))
        memory.store.clock = int(data["bank_ages"][bank].max())
        for slot, count in enumerate(
                data["bank_access_counts"][bank].tolist()):
            if count:
                memory.retrieve(
                    data["bank_keys"][bank, slot:slot + 1].repeat(
                        count, 1),
                    top_k=1, confidence_mode="cosine",
                    record_access=True)
        requested_exact += int(torch.equal(
            memory.store.access_count,
            data["bank_access_counts"][bank]))
        path = directory / f"history-{bank:04d}.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        persisted_exact += int(torch.equal(
            restored.store.access_count,
            memory.store.access_count))
        realized_counts.append(restored.store.access_count.clone())
        memories.append(restored)
    return (
        memories, torch.stack(realized_counts),
        persisted_exact, requested_exact)


def _retarget_future(
        data: dict[str, object], target_slot: torch.Tensor
        ) -> tuple[object, torch.Tensor]:
    banks, capacity = data["bank_ages"].shape
    target_logical = torch.gather(
        data["slot_to_logical"], 1,
        target_slot.unsqueeze(1)).squeeze(1)
    logical = torch.arange(
        capacity, device=target_slot.device).expand(banks, -1)
    retained = logical[
        logical != target_logical.unsqueeze(1)].reshape(banks, capacity - 1)
    future_logical = torch.cat((
        retained,
        torch.full(
            (banks, 1), capacity, device=target_slot.device,
            dtype=torch.long),
    ), dim=1)
    base = (
        torch.arange(banks, device=target_slot.device).unsqueeze(1)
        * (capacity + 1))
    future_batch = _select_batch(
        data["source_batch"], (base + future_logical).reshape(-1))
    future_queries = _gather_rows(
        data["query_group"], future_logical)
    return future_batch, future_queries


@torch.no_grad()
def _physical_policy(
        model: UnifiedCognitiveController,
        memories: list[DiskLatentMemory],
        data: dict[str, object], actions: torch.Tensor,
        future_batch, future_queries: torch.Tensor,
        directory: Path, *, device: torch.device) -> dict[str, object]:
    reads = []
    before_rows = 0
    after_rows = 0
    capacity_growth = 0
    for bank, source in enumerate(memories):
        memory = _wrap_store(source.store.clone())
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
        path = directory / f"decision-{bank:04d}.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        after_rows += restored.count
        reads.append(_gated_retrieve(
            model, restored, future_queries[bank],
            read_threshold=None))
    memory_reads = torch.cat(reads)
    outcomes = _outcomes(
        model, future_batch, memory_reads, device=device)
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
    parser.add_argument("--banks", type=int, default=256)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--noise-scale", type=float, default=0.04)
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
    data = frequency_recency_batch(
        model, banks=args.banks, capacity=args.bank_capacity,
        seed=args.seed, device=device,
        write_threshold=args.write_threshold,
        noise_scale=args.noise_scale)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        history_directory = root / "histories"
        history_directory.mkdir()
        memories, counts, persisted_exact, requested_exact = (
            _materialize_histories(
                model, data, history_directory, device=device))
        normalized_access = (
            torch.log1p(counts.to(data["bank_ages"].dtype))
            / math.log(10.0))
        visible_utility = (
            0.5 * data["bank_ages"] / args.bank_capacity
            + 0.5 * normalized_access)
        realized_utility = visible_utility + data["utility_noise"]
        target_slot = realized_utility.argmin(-1)
        target = target_slot + 1
        visible_oracle = visible_utility.argmin(-1) + 1
        future_batch, future_queries = _retarget_future(
            data, target_slot)

        features = data["option_features"].clone()
        features[:, 1:, 5] = normalized_access - 0.5
        learned = model.memory_replacement_scores(features).argmax(-1)
        recency = data["bank_ages"].argmin(-1) + 1
        frequency = counts.argmin(-1) + 1
        age_shuffled_features = features.clone()
        age_shuffled_features[:, 1:, 0] = (
            age_shuffled_features[:, 1:, 0].roll(1, dims=1))
        age_shuffled = model.memory_replacement_scores(
            age_shuffled_features).argmax(-1)
        frequency_shuffled_features = features.clone()
        frequency_shuffled_features[:, 1:, 5] = (
            frequency_shuffled_features[:, 1:, 5].roll(1, dims=1))
        frequency_shuffled = model.memory_replacement_scores(
            frequency_shuffled_features).argmax(-1)

        policies = {
            "learned": learned,
            "visible_oracle": visible_oracle,
            "oracle": target,
            "recency": recency,
            "frequency": frequency,
            "age_shuffled": age_shuffled,
            "frequency_shuffled": frequency_shuffled,
        }
        physical = {}
        for name, actions in policies.items():
            directory = root / name
            directory.mkdir()
            physical[name] = _physical_policy(
                model, memories, data, actions,
                future_batch, future_queries, directory,
                device=device)
        target_rates = {
            name: float((actions == target).float().mean())
            for name, actions in policies.items()}

    learned_accuracy = physical["learned"]["accuracy"]
    strongest_single = max(
        physical["recency"]["accuracy"],
        physical["frequency"]["accuracy"])
    available_composition_gap = max(
        0.0, physical["visible_oracle"]["accuracy"] - strongest_single)
    captured_composition_gap = learned_accuracy - strongest_single
    report = {
        "schema":
            "unified-controller-frequency-recency-disk-audit-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "seed": args.seed,
        "banks": args.banks,
        "bank_capacity": args.bank_capacity,
        "generated_contexts": data["generated_contexts"],
        "weights_changed": False,
        "semantic_or_utility_labels_used_for_training": False,
        "history_reads_were_ordinary_content_addressed_retrievals": True,
        "requested_histories_reproduced_exactly": requested_exact,
        "access_histories_survived_save_reload_exactly": persisted_exact,
        "policy_target_eviction_rates": target_rates,
        "physical": physical,
        "strongest_single_feature_accuracy": strongest_single,
        "available_composition_gap": available_composition_gap,
        "captured_composition_gap": captured_composition_gap,
    }
    report["gate"] = {
        "physical_accuracy_at_least_94":
            learned_accuracy >= 0.94,
        "learned_target_at_least_75":
            target_rates["learned"] >= 0.75,
        "within_3_points_of_visible_oracle":
            learned_accuracy
            >= physical["visible_oracle"]["accuracy"] - 0.03,
        "beats_each_single_feature_control":
            learned_accuracy > physical["recency"]["accuracy"]
            and learned_accuracy > physical["frequency"]["accuracy"],
        "captures_75_percent_of_composition_gap":
            captured_composition_gap + 1e-6
            >= 0.75 * available_composition_gap,
        "age_corruption_hurts_by_4_points":
            physical["age_shuffled"]["accuracy"]
            <= learned_accuracy - 0.04,
        "frequency_corruption_hurts_by_4_points":
            physical["frequency_shuffled"]["accuracy"]
            <= learned_accuracy - 0.04,
        "all_access_histories_persisted":
            persisted_exact == args.banks,
        "bounded_row_count_preserved":
            physical["learned"]["before_rows"]
            == physical["learned"]["after_rows"]
            == args.banks * args.bank_capacity,
        "physical_capacity_never_grew":
            physical["learned"]["capacity_growth"] == 0,
    }
    report["gate"]["accepted"] = all(report["gate"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
