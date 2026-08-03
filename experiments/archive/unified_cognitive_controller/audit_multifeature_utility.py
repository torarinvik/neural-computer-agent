"""Audit recency-frequency-reliability replacement on physical disk memory."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import torch

from .audit_frequency_recency_replacement import (
    _physical_policy,
    _retarget_future,
    _sha256,
)
from .memory import DiskLatentMemory
from .legacy_model import UnifiedCognitiveController
from .train_frequency_recency_replacement import frequency_recency_batch


@torch.no_grad()
def _materialize_histories(
        model: UnifiedCognitiveController, data: dict[str, object],
        directory: Path, *, device: torch.device
        ) -> tuple[
            list[DiskLatentMemory], torch.Tensor, torch.Tensor, torch.Tensor,
            int, int]:
    memories = []
    realized_access = []
    realized_success = []
    realized_failure = []
    persisted_exact = 0
    requested_exact = 0
    for bank in range(data["bank_keys"].shape[0]):
        memory = DiskLatentMemory(
            model.width, capacity=data["bank_keys"].shape[1],
            device=device)
        memory.commit(
            data["bank_keys"][bank], data["bank_values"][bank],
            data["bank_strengths"][bank], threshold=0.0)
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
        for slot, (successes, failures) in enumerate(zip(
                data["bank_success_counts"][bank].tolist(),
                data["bank_failure_counts"][bank].tolist())):
            count = successes + failures
            if not count:
                continue
            outcomes = torch.cat((
                torch.ones(successes, device=device),
                torch.zeros(failures, device=device)))
            memory.store.record_outcomes(
                data["bank_keys"][bank, slot:slot + 1].repeat(
                    count, 1),
                outcomes)
        requested_exact += int(
            torch.equal(
                memory.store.access_count,
                data["bank_access_counts"][bank])
            and torch.equal(
                memory.store.success_count,
                data["bank_success_counts"][bank])
            and torch.equal(
                memory.store.failure_count,
                data["bank_failure_counts"][bank]))
        path = directory / f"history-{bank:04d}.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        persisted_exact += int(
            torch.equal(
                restored.store.access_count,
                memory.store.access_count)
            and torch.equal(
                restored.store.success_count,
                memory.store.success_count)
            and torch.equal(
                restored.store.failure_count,
                memory.store.failure_count))
        realized_access.append(restored.store.access_count.clone())
        realized_success.append(restored.store.success_count.clone())
        realized_failure.append(restored.store.failure_count.clone())
        memories.append(restored)
    return (
        memories,
        torch.stack(realized_access),
        torch.stack(realized_success),
        torch.stack(realized_failure),
        persisted_exact,
        requested_exact,
    )


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
    if model.adaptive_memory_replace_features < 7:
        raise ValueError("checkpoint has no reliability utility feature")
    data = frequency_recency_batch(
        model, banks=args.banks, capacity=args.bank_capacity,
        seed=args.seed, device=device,
        write_threshold=args.write_threshold,
        noise_scale=args.noise_scale,
        recency_weight=1 / 3, frequency_weight=1 / 3,
        reliability_weight=1 / 3)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        histories = root / "histories"
        histories.mkdir()
        (
            memories, access, successes, failures,
            persisted_exact, requested_exact,
        ) = _materialize_histories(
            model, data, histories, device=device)
        normalized_access = (
            torch.log1p(access.to(data["bank_ages"].dtype))
            / math.log(10.0))
        reliability = (
            (successes.to(data["bank_ages"].dtype) + 1.0)
            / (successes + failures + 2).to(data["bank_ages"].dtype))
        visible_utility = (
            data["bank_ages"] / args.bank_capacity
            + normalized_access
            + reliability) / 3.0
        realized_utility = visible_utility + data["utility_noise"]
        target_slot = realized_utility.argmin(-1)
        target = target_slot + 1
        visible_oracle = visible_utility.argmin(-1) + 1
        future_batch, future_queries = _retarget_future(
            data, target_slot)

        features = data["option_features"].clone()
        features[:, 1:, 5] = normalized_access - 0.5
        features[:, 1:, 6] = reliability - 0.5
        learned = model.memory_replacement_scores(features).argmax(-1)
        recency = data["bank_ages"].argmin(-1) + 1
        frequency = access.argmin(-1) + 1
        reliable = reliability.argmin(-1) + 1
        shuffled = {}
        for name, feature in (("age", 0), ("frequency", 5),
                              ("reliability", 6)):
            corrupt = features.clone()
            corrupt[:, 1:, feature] = (
                corrupt[:, 1:, feature].roll(1, dims=1))
            shuffled[name] = model.memory_replacement_scores(
                corrupt).argmax(-1)
        policies = {
            "learned": learned,
            "visible_oracle": visible_oracle,
            "oracle": target,
            "recency": recency,
            "frequency": frequency,
            "reliability": reliable,
            "age_shuffled": shuffled["age"],
            "frequency_shuffled": shuffled["frequency"],
            "reliability_shuffled": shuffled["reliability"],
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
        physical[name]["accuracy"]
        for name in ("recency", "frequency", "reliability"))
    available_gap = max(
        0.0, physical["visible_oracle"]["accuracy"] - strongest_single)
    captured_gap = learned_accuracy - strongest_single
    minimum_accuracy_drop = 0.15 / args.bank_capacity

    def feature_is_causal(name: str) -> bool:
        corrupted = f"{name}_shuffled"
        return (
            target_rates[corrupted]
            <= target_rates["learned"] - 0.20
            and physical[corrupted]["accuracy"]
            <= learned_accuracy - minimum_accuracy_drop)

    report = {
        "schema": "unified-controller-multifeature-disk-audit-v1",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "seed": args.seed,
        "banks": args.banks,
        "bank_capacity": args.bank_capacity,
        "generated_contexts": data["generated_contexts"],
        "weights_changed": False,
        "semantic_or_utility_labels_used_for_training": False,
        "history_reads_and_outcomes_used_content_addressing": True,
        "requested_histories_reproduced_exactly": requested_exact,
        "histories_survived_save_reload_exactly": persisted_exact,
        "policy_target_eviction_rates": target_rates,
        "physical": physical,
        "strongest_single_feature_accuracy": strongest_single,
        "available_composition_gap": available_gap,
        "captured_composition_gap": captured_gap,
        "capacity_aware_minimum_accuracy_drop": minimum_accuracy_drop,
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
            all(
                learned_accuracy > physical[name]["accuracy"]
                for name in ("recency", "frequency", "reliability")),
        "captures_75_percent_of_composition_gap":
            captured_gap + 1e-6 >= 0.75 * available_gap,
        "age_corruption_changes_20_percent_of_targets_and_15_percent_of_slot":
            feature_is_causal("age"),
        "frequency_corruption_changes_20_percent_of_targets_and_15_percent_of_slot":
            feature_is_causal("frequency"),
        "reliability_corruption_changes_20_percent_of_targets_and_15_percent_of_slot":
            feature_is_causal("reliability"),
        "all_histories_persisted":
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
