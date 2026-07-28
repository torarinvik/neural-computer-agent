"""Audit learned volatility use through real bounded disk-backed memories."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch

from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_persistent_physical_stream import _gated_retrieve
from .train_adaptive_memory_read import _outcomes
from .train_controller_memory_volatility import (
    expand_with_volatility,
    volatility_batch,
)


def _model_from_checkpoint(
        path: Path, device: torch.device) -> UnifiedCognitiveController:
    payload = torch.load(path, map_location=device, weights_only=False)
    configuration = dict(payload["model_configuration"])
    if int(configuration["adaptive_memory_replace_features"]) == 8:
        model = UnifiedCognitiveController(**configuration).to(device)
        model.load_state_dict(payload["state_dict"])
    else:
        model, _ = expand_with_volatility(payload, device=device)
    model.eval()
    return model


@torch.no_grad()
def audit(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float,
        intervention: str = "none") -> dict[str, float | int | bool]:
    if intervention not in {
            "none", "shuffle_volatility", "constant_volatility",
            "reverse_histories"}:
        raise ValueError("unsupported intervention")
    data = volatility_batch(
        model, banks=banks, capacity=capacity, seed=seed, device=device,
        write_threshold=write_threshold,
        reverse_histories=intervention == "reverse_histories")
    memories: list[DiskLatentMemory] = []
    exact_histories = 0
    generator = torch.Generator().manual_seed(seed + 83_000_000)
    with tempfile.TemporaryDirectory(prefix="controller-volatility-audit-") as root:
        directory = Path(root)
        for bank in range(banks):
            memory = DiskLatentMemory(
                width=model.width, capacity=capacity, device=device)
            # Equal admission strength isolates the volatility mechanism and
            # makes each exact content query address its own row. The broader
            # unequal-prior interaction is a separate retrieval-credit rung.
            memory.commit(
                data["bank_keys"][bank], data["bank_values"][bank],
                torch.ones_like(data["bank_strengths"][bank]), threshold=0.0)
            keys = memory.store.keys[:capacity]
            stable = data["stable_mask"][bank]
            stable_history = torch.tensor(
                [0.0] * 5 + [1.0] * 5, device=device)
            decoy_history = 1.0 - stable_history
            if intervention == "reverse_histories":
                stable_history, decoy_history = (
                    decoy_history, stable_history)
            for event in range(10):
                memory.retrieve(
                    keys, top_k=1, confidence_mode="cosine",
                    record_access=True)
                outcomes = torch.where(
                    stable, stable_history[event], decoy_history[event])
                memory.store.record_outcomes(
                    keys, outcomes, update_volatility=True,
                    success_protection_rate=0.2,
                    failure_thaw_rate=0.25, stale_thaw_rate=0.0)
            if intervention == "shuffle_volatility":
                order = torch.randperm(capacity, generator=generator).to(device)
                memory.store.volatility[:capacity] = (
                    memory.store.volatility[order].clone())
            elif intervention == "constant_volatility":
                memory.store.volatility[:capacity].fill_(
                    float(memory.store.volatility[:capacity].mean()))
            path = directory / f"source-{bank:04d}.pt"
            before = memory.store.clone()
            memory.save(path)
            memory = DiskLatentMemory.load(path, device=device)
            exact_histories += int(
                torch.equal(before.keys, memory.store.keys)
                and torch.equal(before.values, memory.store.values)
                and torch.equal(before.access_count, memory.store.access_count)
                and torch.equal(before.success_count, memory.store.success_count)
                and torch.equal(before.failure_count, memory.store.failure_count)
                and torch.equal(before.volatility, memory.store.volatility))
            memories.append(memory)

        option_features = data["option_features"].clone()
        for bank, memory in enumerate(memories):
            age = memory.store.age[:capacity].to(
                option_features.dtype) / capacity
            access = (
                torch.log1p(memory.store.access_count[:capacity].to(
                    option_features.dtype)) / math.log(10.0))
            reliability = (
                (memory.store.success_count[:capacity].to(
                    option_features.dtype) + 1.0)
                / (
                    memory.store.success_count[:capacity]
                    + memory.store.failure_count[:capacity] + 2
                ).to(option_features.dtype))
            option_features[bank, 1:, 0] = age
            option_features[bank, 1:, 1] = (
                memory.store.usage[:capacity])
            option_features[bank, 1:, 5] = access - 0.5
            option_features[bank, 1:, 6] = reliability - 0.5
            option_features[bank, 1:, 7] = (
                memory.store.volatility[:capacity])
        actions = model.memory_replacement_scores(option_features).argmax(-1)
        reads = []
        persisted_after = 0
        selected_stable = torch.zeros(banks, dtype=torch.bool, device=device)
        for bank, source in enumerate(memories):
            memory = DiskLatentMemory.__new__(DiskLatentMemory)
            memory.store = source.store.clone()
            action = int(actions[bank])
            if action:
                selected_stable[bank] = data["stable_mask"][
                    bank, action - 1]
                memory.elastic_replace(
                    action - 1, data["candidate_key"][bank],
                    data["candidate_value"][bank],
                    data["candidate_strength"][bank])
            path = directory / f"updated-{bank:04d}.pt"
            memory.save(path)
            restored = DiskLatentMemory.load(path, device=device)
            persisted_after += int(
                restored.count == capacity
                and restored.store.capacity == capacity)
            reads.append(_gated_retrieve(
                model, restored, data["future_queries"][bank],
                read_threshold=None))
        outcomes = _outcomes(
            model, data["future_batch"], torch.cat(reads), device=device)
        accuracy = outcomes.reshape(banks, -1).float().mean(-1)
    return {
        "accuracy": float(accuracy.mean()),
        "replace_rate": float((actions > 0).float().mean()),
        "stable_eviction_rate": float(selected_stable.float().mean()),
        "valid_replacement_rate": float(
            ((actions > 0) & ~selected_stable).float().mean()),
        "exact_histories_after_reload": exact_histories,
        "bounded_banks_after_replacement": persisted_after,
        "banks": banks,
        "all_history_fields_persist_exactly": exact_histories == banks,
        "all_banks_remain_bounded": persisted_after == banks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=17200)
    parser.add_argument("--banks", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device(args.device)
    model = _model_from_checkpoint(args.checkpoint, device)
    normal = audit(
        model, banks=args.banks, capacity=args.capacity, seed=args.seed,
        device=device, write_threshold=args.write_threshold)
    shuffled = audit(
        model, banks=args.banks, capacity=args.capacity, seed=args.seed,
        device=device, write_threshold=args.write_threshold,
        intervention="shuffle_volatility")
    constant = audit(
        model, banks=args.banks, capacity=args.capacity, seed=args.seed,
        device=device, write_threshold=args.write_threshold,
        intervention="constant_volatility")
    reversed_histories = audit(
        model, banks=args.banks, capacity=args.capacity, seed=args.seed,
        device=device, write_threshold=args.write_threshold,
        intervention="reverse_histories")
    report = {
        "schema": "unified-controller-physical-memory-volatility-audit-v1",
        "normal": normal,
        "volatility_shuffled": shuffled,
        "volatility_constant": constant,
        "outcome_histories_reversed": reversed_histories,
        "gates": {
            "normal_valid_replacement_at_least_90_percent":
                normal["valid_replacement_rate"] >= 0.90,
            "normal_accuracy_at_least_90_percent":
                normal["accuracy"] >= 0.90,
            "shuffle_costs_30_points_valid_replacement":
                normal["valid_replacement_rate"]
                >= shuffled["valid_replacement_rate"] + 0.30,
            "constant_costs_30_points_valid_replacement":
                normal["valid_replacement_rate"]
                >= constant["valid_replacement_rate"] + 0.30,
            "reversal_flips_at_least_90_percent":
                reversed_histories["stable_eviction_rate"] >= 0.90,
            "histories_persist_exactly":
                normal["all_history_fields_persist_exactly"],
            "capacity_remains_bounded":
                normal["all_banks_remain_bounded"],
        },
    }
    report["gates"]["accepted"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
