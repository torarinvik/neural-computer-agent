"""Audit diversity-preserving consolidation under unseen appearance shifts."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .audit_selective_disk import _support
from .environment import generate_lifetimes
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import evaluate, seed_everything
from .train_adaptive_memory_read import _outcomes
from .train_equivalence_consolidation import (
    consolidate,
    natural_memory_streams,
)
from .train_memory_replacement import _select_batch


def _load(
        path: Path, device: torch.device,
        ) -> tuple[dict[str, object], UnifiedCognitiveController]:
    payload = torch.load(path, map_location=device, weights_only=False)
    configuration = payload.get("model_configuration")
    if not isinstance(configuration, dict):
        raise ValueError("checkpoint lacks model configuration")
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model


@torch.no_grad()
def appearance_query_batch(
        model: UnifiedCognitiveController, *, streams: int, seed: int,
        appearance: str, device: torch.device,
        ):
    pool = _add_context_signatures(
        generate_lifetimes(
            streams * 2, 3, seed=seed, heldout=True,
            task="binary_mapping", appearance=appearance,
            support_trials=1, device=device),
        seed=seed + 10_000_000)
    rule_zero = torch.where(pool.rule_bits == 0)[0]
    rule_one = torch.where(pool.rule_bits == 1)[0]
    if rule_zero.numel() != streams or rule_one.numel() != streams:
        raise RuntimeError("balanced generator did not produce equal rules")
    batch = _select_batch(pool, torch.cat((rule_zero, rule_one)))
    _, probes, _ = _support(model, batch, device=device)
    return batch, probes


@torch.no_grad()
def appearance_behavior(
        model: UnifiedCognitiveController,
        bank: dict[str, torch.Tensor], *, streams: int, seed: int,
        appearance: str, device: torch.device,
        ) -> dict[str, float]:
    batch, probes = appearance_query_batch(
        model, streams=streams, seed=seed,
        appearance=appearance, device=device)
    values = bank["values"].repeat(2, 1, 1)
    valid = bank["valid"].repeat(2, 1)
    scores = model.calibrated_memory_equivalence_logits(
        probes, values).masked_fill(~valid, float("-inf"))
    selected = scores.argmax(-1)
    retrieved = values[
        torch.arange(values.shape[0], device=device), selected]
    outcomes = _outcomes(model, batch, retrieved, device=device)
    coverage = (
        (bank["rule_bits"] == 0).any(-1)
        & (bank["rule_bits"] == 1).any(-1))
    return {
        "visual_accuracy": float(outcomes.float().mean()),
        "rule_zero_accuracy": float(outcomes[:streams].float().mean()),
        "rule_one_accuracy": float(outcomes[streams:].float().mean()),
        "both_behaviors_retained": float(coverage.float().mean()),
        "mean_rows": float(valid[:streams].sum(-1).float().mean()),
    }


@torch.no_grad()
def evaluate_appearances(
        model: UnifiedCognitiveController, *, streams: int, length: int,
        seed: int, device: torch.device,
        ) -> tuple[dict[str, object], dict[str, dict[str, torch.Tensor]]]:
    data = natural_memory_streams(
        model, streams=streams, length=length, seed=seed,
        device=device, heldout=True)
    banks = {
        "one_representative": consolidate(
            model, data, capacity=2, representatives_per_class=1),
        "two_representatives": consolidate(
            model, data, capacity=4, representatives_per_class=2),
        "three_representatives": consolidate(
            model, data, capacity=6, representatives_per_class=3),
        "first_four": consolidate(
            model, data, capacity=4, policy="first"),
        "uncompressed": {
            "keys": data["keys"],
            "values": data["values"],
            "usage": torch.ones_like(data["strengths"]),
            "valid": torch.ones_like(
                data["strengths"], dtype=torch.bool),
            "rule_bits": data["rule_bits"],
        },
    }
    corrupted = {
        name: value.clone()
        for name, value in banks["two_representatives"].items()
    }
    corrupted["values"] = torch.zeros_like(corrupted["values"])
    banks["corrupted_two_representatives"] = corrupted
    appearances = ("bars", "diamonds", "dot_pairs")
    report = {
        appearance: {
            name: appearance_behavior(
                model, bank, streams=streams,
                seed=seed + 10_000_000 + appearance_index,
                appearance=appearance, device=device)
            for name, bank in banks.items()
        }
        for appearance_index, appearance in enumerate(appearances)
    }
    return report, banks


@torch.no_grad()
def physical_audit(
        model: UnifiedCognitiveController, *, streams: int, length: int,
        seed: int, device: torch.device,
        ) -> dict[str, object]:
    data = natural_memory_streams(
        model, streams=streams, length=length, seed=seed,
        device=device, heldout=True)
    bank = consolidate(
        model, data, capacity=4, representatives_per_class=2)
    restored_values = []
    restored_valid = []
    exact = compressed_bytes = full_bytes = 0
    with tempfile.TemporaryDirectory(
            prefix="diversity-consolidation-") as root:
        directory = Path(root)
        for index in range(streams):
            compact = DiskLatentMemory(
                model.width, capacity=4, device=device)
            compact.commit(
                bank["keys"][index], bank["values"][index],
                bank["usage"][index], threshold=0.0)
            compact_path = directory / f"compact-{index:04d}.pt"
            compact.save(compact_path)
            restored = DiskLatentMemory.load(
                compact_path, device=device)
            exact += int(
                torch.equal(restored.store.keys, compact.store.keys)
                and torch.equal(restored.store.values, compact.store.values)
                and torch.equal(restored.store.usage, compact.store.usage))
            restored_values.append(restored.store.values[:4])
            restored_valid.append(restored.store.valid[:4])
            compressed_bytes += compact_path.stat().st_size

            full = DiskLatentMemory(
                model.width, capacity=length, device=device)
            full.commit(
                data["keys"][index], data["values"][index],
                torch.ones(length, device=device), threshold=0.0)
            full_path = directory / f"full-{index:04d}.pt"
            full.save(full_path)
            full_bytes += full_path.stat().st_size
    restored_bank = {
        "values": torch.stack(restored_values),
        "valid": torch.stack(restored_valid),
        "rule_bits": bank["rule_bits"],
    }
    behavior = appearance_behavior(
        model, restored_bank, streams=streams,
        seed=seed + 10_000_000, appearance="dot_pairs",
        device=device)
    return {
        **behavior,
        "banks": streams,
        "exact_reload_count": exact,
        "all_banks_reload_exactly": exact == streams,
        "compressed_bytes": compressed_bytes,
        "uncompressed_bytes": full_bytes,
        "byte_ratio": compressed_bytes / full_bytes,
        "logical_row_ratio": 4 / length,
    }


@torch.no_grad()
def cross_appearance_counterfactual(
        model: UnifiedCognitiveController,
        bank: dict[str, torch.Tensor], *, count: int, seed: int,
        device: torch.device,
        ) -> dict[str, object]:
    ordinary = _add_context_signatures(
        generate_lifetimes(
            count, 3, seed=seed, heldout=True,
            task="binary_mapping", appearance="dot_pairs",
            support_trials=1, device=device),
        seed=seed + 10_000_000)
    reversed_batch = _add_context_signatures(
        generate_lifetimes(
            count, 3, seed=seed, heldout=True, reverse_rules=True,
            task="binary_mapping", appearance="dot_pairs",
            support_trials=1, device=device),
        seed=seed + 10_000_000)
    if not torch.equal(ordinary.frames, reversed_batch.frames):
        raise RuntimeError("counterfactual changed the visual stream")
    _, ordinary_probe, _ = _support(
        model, ordinary, device=device)
    _, reversed_probe, _ = _support(
        model, reversed_batch, device=device)
    values = bank["values"][:count]
    valid = bank["valid"][:count]

    def retrieve(probe: torch.Tensor):
        scores = model.calibrated_memory_equivalence_logits(
            probe, values).masked_fill(~valid, float("-inf"))
        selected = scores.argmax(-1)
        retrieved = values[
            torch.arange(count, device=device), selected]
        return selected, retrieved

    ordinary_row, ordinary_value = retrieve(ordinary_probe)
    reversed_row, reversed_value = retrieve(reversed_probe)
    ordinary_outcome = _outcomes(
        model, ordinary, ordinary_value, device=device)
    reversed_outcome = _outcomes(
        model, reversed_batch, reversed_value, device=device)
    return {
        "frames_identical": True,
        "bank_tensors_identical": True,
        "ordinary_accuracy": float(ordinary_outcome.float().mean()),
        "reversed_accuracy": float(reversed_outcome.float().mean()),
        "selection_flip_rate": float(
            (ordinary_row != reversed_row).float().mean()),
        "probe_change_rate": float(
            (ordinary_probe != reversed_probe).any(-1).float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20601)
    parser.add_argument("--streams", type=int, default=2048)
    parser.add_argument("--stream-length", type=int, default=16)
    parser.add_argument("--physical-streams", type=int, default=256)
    parser.add_argument("--retention-count", type=int, default=256)
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if (
            args.streams < 32 or args.stream_length < 8
            or args.physical_streams < 16
            or args.retention_count < 32):
        raise ValueError("audit budgets are too small")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload, model = _load(args.checkpoint_in, device)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    started = time.perf_counter()
    appearances, banks = evaluate_appearances(
        model, streams=args.streams, length=args.stream_length,
        seed=args.seed + 20_000_000, device=device)
    physical = physical_audit(
        model, streams=args.physical_streams,
        length=args.stream_length, seed=args.seed + 30_000_000,
        device=device)
    counterfactual = cross_appearance_counterfactual(
        model, banks["two_representatives"],
        count=args.streams,
        seed=args.seed + 40_000_000, device=device)
    binary = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 50_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 60_000_000, device=device,
        task="four_rule", feedback_trials=2)
    changed = sorted(
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu()))
    dot = appearances["dot_pairs"]
    gates = {
        "two_representatives_dot_pairs_at_least_98":
            dot["two_representatives"]["visual_accuracy"] >= 0.98,
        "two_representatives_retain_both_behaviors":
            dot["two_representatives"][
                "both_behaviors_retained"] >= 0.995,
        "two_beats_one_by_at_least_one_point":
            dot["two_representatives"]["visual_accuracy"]
            >= dot["one_representative"]["visual_accuracy"] + 0.01,
        "two_beats_first_four_by_at_least_four_points":
            dot["two_representatives"]["visual_accuracy"]
            >= dot["first_four"]["visual_accuracy"] + 0.04,
        "two_within_two_points_of_uncompressed":
            dot["two_representatives"]["visual_accuracy"]
            >= dot["uncompressed"]["visual_accuracy"] - 0.02,
        "bars_at_least_99_5":
            appearances["bars"]["two_representatives"][
                "visual_accuracy"] >= 0.995,
        "diamonds_at_least_99_5":
            appearances["diamonds"]["two_representatives"][
                "visual_accuracy"] >= 0.995,
        "corruption_costs_at_least_20_points":
            dot["two_representatives"]["visual_accuracy"]
            >= dot["corrupted_two_representatives"][
                "visual_accuracy"] + 0.20,
        "physical_dot_pairs_at_least_98":
            physical["visual_accuracy"] >= 0.98,
        "physical_retains_both_behaviors":
            physical["both_behaviors_retained"] >= 0.995,
        "physical_reload_exact":
            physical["all_banks_reload_exactly"],
        "physical_rows_compressed_4x":
            physical["logical_row_ratio"] <= 0.25,
        "counterfactual_accuracy_at_least_98":
            counterfactual["ordinary_accuracy"] >= 0.98
            and counterfactual["reversed_accuracy"] >= 0.98,
        "counterfactual_selection_flips_at_least_98":
            counterfactual["selection_flip_rate"] >= 0.98,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "no_parameters_changed": not changed,
    }
    gates["accepted"] = all(gates.values())
    report = {
        "schema": "diversity-preserving-consolidation-v1",
        "claim_boundary": (
            "Zero-shot cross-appearance audit of a learned equivalence "
            "relation. No new verifier outcome or parameter update is used."),
        "configuration": {
            **vars(args),
            "checkpoint_in": str(args.checkpoint_in),
            "report": str(args.report),
        },
        "learner_visible": [
            "controller_created_memory_values",
            "learned_pairwise_equivalence_scores",
        ],
        "hidden_from_policy": [
            "rule_bits",
            "appearance_name",
            "correct_representative_count",
        ],
        "semantic_labels_used_for_policy": False,
        "accounting": {
            "new_training_verifier_bits": 0,
            "new_optimizer_updates": 0,
            "evaluation_streams": args.streams,
            "stream_length": args.stream_length,
        },
        "appearances": appearances,
        "physical": physical,
        "counterfactual": counterfactual,
        "retention": {
            "binary_mapping": binary,
            "four_rule": four_rule,
        },
        "changed_parameters": changed,
        "gates": gates,
        "wall_seconds": time.perf_counter() - started,
        "source_schema": payload.get("schema"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "accepted": gates["accepted"],
        "dot_one": dot["one_representative"]["visual_accuracy"],
        "dot_two": dot["two_representatives"]["visual_accuracy"],
        "dot_three": dot["three_representatives"]["visual_accuracy"],
        "dot_first_four": dot["first_four"]["visual_accuracy"],
        "dot_full": dot["uncompressed"]["visual_accuracy"],
        "physical_dot": physical["visual_accuracy"],
        "counterfactual": counterfactual,
        "wall_seconds": report["wall_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
