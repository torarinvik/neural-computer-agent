"""Test whether verified access history can safely prune physical memory rows."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from .audit_selective_disk import _support
from .legacy_model import UnifiedCognitiveController
from .train import seed_everything
from .train_adaptive_memory_read import _outcomes
from .train_adaptive_representative_read import (
    APPEARANCES,
    _query_batch,
)
from .train_equivalence_consolidation import (
    consolidate,
    natural_memory_streams,
)


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
    if model.representative_read_critic is None:
        raise ValueError("checkpoint lacks adaptive read critic")
    return payload, model


@torch.no_grad()
def query_event(
        model: UnifiedCognitiveController,
        bank: dict[str, torch.Tensor], *,
        seed: int, appearance: str, physical_mask: torch.Tensor,
        device: torch.device, reverse_rules: bool = False,
        force_deep: bool | None = None,
        ) -> dict[str, torch.Tensor]:
    """Replay one balanced future event per rule for every physical bank."""
    streams, capacity, _ = bank["values"].shape
    batch = _query_batch(
        count=streams * 2, seed=seed, appearance=appearance,
        reverse_rules=reverse_rules, device=device)
    _, probes, _ = _support(model, batch, device=device)
    values = bank["values"].repeat(2, 1, 1)
    valid = bank["valid"].repeat(2, 1)
    ranks = bank["representative_ranks"].repeat(2, 1)
    physical = physical_mask.repeat(2, 1) & valid
    row = torch.arange(streams * 2, device=device)
    scores = model.calibrated_memory_equivalence_logits(probes, values)

    shallow_mask = physical & (ranks == 0)
    shallow_scores = scores.masked_fill(~shallow_mask, float("-inf"))
    top, indices = shallow_scores.topk(2, dim=-1)
    finite = torch.isfinite(top)
    safe_top = torch.where(
        finite, top, torch.full_like(top, -20.0))
    first_value = values[row, indices[:, 0]]
    second_value = values[row, indices[:, 1]]
    second_value = torch.where(
        finite[:, 1, None], second_value,
        torch.zeros_like(second_value))
    features = torch.cat((
        probes,
        first_value,
        second_value,
        (probes - first_value).abs(),
        probes * first_value,
        safe_top,
        finite.to(probes.dtype),
    ), dim=-1)
    deep_requested = (
        model.representative_deep_read_probability(features)
        >= model.adaptive_representative_read_threshold)
    if force_deep is not None:
        deep_requested.fill_(force_deep)
    shallow_row = shallow_scores.argmax(-1)
    deep_scores = scores.masked_fill(~physical, float("-inf"))
    deep_row = deep_scores.argmax(-1)
    selected_row = torch.where(
        deep_requested, deep_row, shallow_row)
    shallow_value = values[row, shallow_row]
    deep_value = values[row, deep_row]
    selected_value = values[row, selected_row]
    shallow_correct = _outcomes(
        model, batch, shallow_value, device=device)
    deep_correct = _outcomes(
        model, batch, deep_value, device=device)
    selected_correct = _outcomes(
        model, batch, selected_value, device=device)
    comparisons = torch.where(
        deep_requested, physical.sum(-1), shallow_mask.sum(-1))
    return {
        "deep_requested": deep_requested,
        "deep_row": deep_row,
        "shallow_correct": shallow_correct,
        "deep_correct": deep_correct,
        "selected_correct": selected_correct,
        "comparisons": comparisons,
    }


@torch.no_grad()
def history_scores(
        model: UnifiedCognitiveController,
        bank: dict[str, torch.Tensor], *, rounds: int, seed: int,
        device: torch.device,
        ) -> dict[str, torch.Tensor]:
    """Attach task-agnostic use/protection scalars to physical rows."""
    streams, capacity, _ = bank["values"].shape
    full = bank["valid"]
    requested = torch.zeros(
        streams, capacity, device=device, dtype=torch.float32)
    correct_use = torch.zeros_like(requested)
    causal_rescue = torch.zeros_like(requested)
    row = torch.arange(streams * 2, device=device)
    bank_index = row % streams
    for round_index in range(rounds):
        for appearance_index, appearance in enumerate(APPEARANCES):
            event = query_event(
                model, bank,
                seed=seed + round_index * 100 + appearance_index,
                appearance=appearance, physical_mask=full, device=device)
            selected = event["deep_row"]
            used = event["deep_requested"]
            deep_correct = event["deep_correct"].to(torch.bool)
            shallow_correct = event["shallow_correct"].to(torch.bool)
            rescued = used & deep_correct & ~shallow_correct
            correct = used & deep_correct
            requested.index_put_(
                (bank_index[used], selected[used]),
                torch.ones_like(selected[used], dtype=torch.float32),
                accumulate=True)
            correct_use.index_put_(
                (bank_index[correct], selected[correct]),
                torch.ones_like(selected[correct], dtype=torch.float32),
                accumulate=True)
            causal_rescue.index_put_(
                (bank_index[rescued], selected[rescued]),
                torch.ones_like(selected[rescued], dtype=torch.float32),
                accumulate=True)
    return {
        "requested": requested,
        "correct_use": correct_use,
        "causal_rescue": causal_rescue,
    }


def survival_mask(
        bank: dict[str, torch.Tensor], scores: torch.Tensor,
        *, threshold: float,
        ) -> torch.Tensor:
    """Always preserve core representatives; extras must earn survival."""
    core = bank["valid"] & (bank["representative_ranks"] < 2)
    extra = (
        bank["valid"] & (bank["representative_ranks"] >= 2)
        & (scores >= threshold))
    return core | extra


def bankwise_survival_mask(
        bank: dict[str, torch.Tensor], scores: torch.Tensor,
        *, threshold: float,
        ) -> torch.Tensor:
    """Protect a bank's diversity reserve after any extra row proves useful."""
    core = bank["valid"] & (bank["representative_ranks"] < 2)
    candidates = bank["valid"] & (bank["representative_ranks"] >= 2)
    protect = ((scores >= threshold) & candidates).any(-1)
    return core | (candidates & protect[:, None])


def shuffled_matched_mask(
        bank: dict[str, torch.Tensor], mask: torch.Tensor, *,
        seed: int,
        ) -> torch.Tensor:
    """Keep the same extras per bank, but destroy which extra earned survival."""
    result = bank["valid"] & (bank["representative_ranks"] < 2)
    generator = torch.Generator(
        device=mask.device).manual_seed(seed)
    for index in range(mask.shape[0]):
        candidates = torch.where(
            bank["valid"][index]
            & (bank["representative_ranks"][index] >= 2))[0]
        count = int((
            mask[index]
            & (bank["representative_ranks"][index] >= 2)).sum())
        if count:
            order = torch.randperm(
                candidates.numel(), generator=generator,
                device=mask.device)
            result[index, candidates[order[:count]]] = True
    return result


def shuffled_bankwise_mask(
        bank: dict[str, torch.Tensor], mask: torch.Tensor, *,
        seed: int,
        ) -> torch.Tensor:
    """Preserve the number of protected banks while shuffling their identities."""
    core = bank["valid"] & (bank["representative_ranks"] < 2)
    candidates = bank["valid"] & (bank["representative_ranks"] >= 2)
    protected = (mask & candidates).any(-1)
    generator = torch.Generator(
        device=mask.device).manual_seed(seed)
    shuffled = torch.zeros_like(protected)
    # Shuffle only among banks with the same reserve size. This preserves both
    # the number of protected banks and the exact physical-row budget.
    candidate_count = candidates.sum(-1)
    for count in candidate_count.unique():
        group = torch.where(candidate_count == count)[0]
        permutation = torch.randperm(
            group.numel(), generator=generator, device=mask.device)
        shuffled[group] = protected[group[permutation]]
    return core | (candidates & shuffled[:, None])


@torch.no_grad()
def future_metrics(
        model: UnifiedCognitiveController,
        bank: dict[str, torch.Tensor], mask: torch.Tensor, *,
        rounds: int, seed: int, device: torch.device,
        reverse_rules: bool = False,
        ) -> dict[str, float]:
    correct = comparisons = deep = events = 0.0
    for round_index in range(rounds):
        for appearance_index, appearance in enumerate(APPEARANCES):
            event = query_event(
                model, bank,
                seed=seed + round_index * 100 + appearance_index,
                appearance=appearance, physical_mask=mask,
                device=device, reverse_rules=reverse_rules)
            correct += float(event["selected_correct"].float().sum())
            comparisons += float(event["comparisons"].float().sum())
            deep += float(event["deep_requested"].float().sum())
            events += event["selected_correct"].numel()
    return {
        "accuracy": correct / events,
        "mean_comparisons": comparisons / events,
        "deep_read_rate": deep / events,
        "mean_physical_rows": float(mask.float().sum(-1).mean()),
        "physical_row_ratio_vs_six": float(
            mask.float().sum() / bank["valid"].float().sum()),
    }


@torch.no_grad()
def physical_disk_audit(
        model: UnifiedCognitiveController,
        bank: dict[str, torch.Tensor], mask: torch.Tensor,
        protection_scores: torch.Tensor, *, banks: int,
        device: torch.device,
        ) -> dict[str, object]:
    """Actually remove rows, persist them, and verify every history field."""
    from .memory import DiskLatentMemory

    exact = compact_bytes = full_bytes = 0
    compact_rows = full_rows = 0
    banks = min(banks, bank["values"].shape[0])
    with tempfile.TemporaryDirectory(
            prefix="adaptive-physical-pruning-") as root:
        directory = Path(root)
        for index in range(banks):
            valid = bank["valid"][index]
            source = DiskLatentMemory(
                model.width, capacity=int(valid.sum()), device=device)
            source.commit(
                bank["keys"][index, valid],
                bank["values"][index, valid],
                bank["usage"][index, valid], threshold=0.0)
            score = protection_scores[index, valid]
            source.store.success_count[:score.numel()] = score.to(
                torch.long)
            source.store.volatility[:score.numel()] = torch.where(
                score > 0, torch.full_like(score, 0.1),
                torch.ones_like(score))
            full_path = directory / f"full-{index:04d}.pt"
            source.save(full_path)

            selected = torch.where(mask[index, valid])[0]
            compact = source.compact(selected)
            compact_path = directory / f"compact-{index:04d}.pt"
            compact.save(compact_path)
            restored = DiskLatentMemory.load(
                compact_path, device=device)
            exact += int(
                restored.store.capacity == compact.store.capacity
                and torch.equal(
                    restored.store.keys, compact.store.keys)
                and torch.equal(
                    restored.store.values, compact.store.values)
                and torch.equal(
                    restored.store.success_count,
                    compact.store.success_count)
                and torch.equal(
                    restored.store.volatility,
                    compact.store.volatility))
            compact_bytes += compact_path.stat().st_size
            full_bytes += full_path.stat().st_size
            compact_rows += compact.count
            full_rows += source.count
    return {
        "banks": banks,
        "exact_reload_count": exact,
        "all_compacted_banks_reload_exactly": exact == banks,
        "compact_rows": compact_rows,
        "full_rows": full_rows,
        "logical_row_ratio": compact_rows / full_rows,
        "compact_bytes": compact_bytes,
        "full_bytes": full_bytes,
        "physical_byte_ratio": compact_bytes / full_bytes,
    }


@torch.no_grad()
def audit(
        checkpoint: Path, *, streams: int, history_rounds: int,
        future_rounds: int, seed: int, threshold: float,
        device: torch.device, disk_banks: int = 128,
        ) -> dict[str, object]:
    payload, model = _load(checkpoint, device)
    data = natural_memory_streams(
        model, streams=streams, length=16, seed=seed,
        device=device, heldout=True)
    bank = consolidate(
        model, data, capacity=6, representatives_per_class=3)
    full = bank["valid"].clone()
    fixed_four = bank["valid"] & (bank["representative_ranks"] < 2)
    histories = history_scores(
        model, bank, rounds=history_rounds,
        seed=seed + 1_000_000, device=device)
    policies: dict[str, torch.Tensor] = {
        "fixed_four": fixed_four,
        "full_six": full,
    }
    for name, scores in histories.items():
        learned = survival_mask(bank, scores, threshold=threshold)
        policies[name] = learned
        policies[f"{name}_shuffled_matched"] = shuffled_matched_mask(
            bank, learned, seed=seed + 2_000_000)
        bankwise = bankwise_survival_mask(
            bank, scores, threshold=threshold)
        policies[f"{name}_bankwise"] = bankwise
        policies[f"{name}_bankwise_shuffled"] = shuffled_bankwise_mask(
            bank, bankwise, seed=seed + 2_100_000)
    results = {
        name: future_metrics(
            model, bank, mask, rounds=future_rounds,
            seed=seed + 3_000_000, device=device)
        for name, mask in policies.items()
    }
    chosen = policies["causal_rescue_bankwise"]
    reversed_metrics = future_metrics(
        model, bank, chosen, rounds=future_rounds,
        seed=seed + 3_000_000, device=device, reverse_rules=True)
    reversed_full = future_metrics(
        model, bank, full, rounds=future_rounds,
        seed=seed + 3_000_000, device=device, reverse_rules=True)
    corrupted_bank = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in bank.items()
    }
    corrupted_bank["values"].zero_()
    corrupted_metrics = future_metrics(
        model, corrupted_bank, chosen, rounds=future_rounds,
        seed=seed + 3_000_000, device=device)
    disk = physical_disk_audit(
        model, bank, chosen, histories["causal_rescue"],
        banks=disk_banks, device=device)
    full = results["full_six"]
    fixed = results["fixed_four"]
    learned = results["causal_rescue_bankwise"]
    shuffled = results["causal_rescue_bankwise_shuffled"]
    gates = {
        "accuracy_within_0_075_points_of_full":
            learned["accuracy"] >= full["accuracy"] - 0.00075,
        "mean_rows_at_most_4_5":
            learned["mean_physical_rows"] <= 4.5,
        "beats_fixed_four_by_0_05_points":
            learned["accuracy"] >= fixed["accuracy"] + 0.0005,
        "beats_matched_shuffle_by_0_03_points":
            learned["accuracy"] >= shuffled["accuracy"] + 0.0003,
        "reversed_task_within_0_075_points_of_full":
            reversed_metrics["accuracy"]
            >= reversed_full["accuracy"] - 0.00075,
        "corrupted_memory_at_most_60_percent":
            corrupted_metrics["accuracy"] <= 0.60,
        "disk_reload_exact": disk[
            "all_compacted_banks_reload_exactly"],
        "physical_file_bytes_reduced":
            disk["physical_byte_ratio"] < 1.0,
    }
    gates["accepted"] = all(gates.values())
    return {
        "schema": "adaptive-physical-pruning-audit-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_training_examples":
            payload.get("representative_read_training_examples"),
        "streams": streams,
        "history_rounds": history_rounds,
        "future_rounds": future_rounds,
        "threshold": threshold,
        "verifier_history_events":
            streams * 2 * len(APPEARANCES) * history_rounds,
        "verifier_future_events":
            streams * 2 * len(APPEARANCES) * future_rounds,
        "results": results,
        "reversed_rules": {
            "adaptive_pruned": reversed_metrics,
            "full_six": reversed_full,
        },
        "corrupted_memory": corrupted_metrics,
        "physical_disk": disk,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=2048)
    parser.add_argument("--history-rounds", type=int, default=4)
    parser.add_argument("--future-rounds", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=41_001)
    parser.add_argument("--disk-banks", type=int, default=128)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    seed_everything(args.seed)
    report = audit(
        args.checkpoint, streams=args.streams,
        history_rounds=args.history_rounds,
        future_rounds=args.future_rounds, seed=args.seed,
        threshold=args.threshold, device=torch.device(args.device),
        disk_banks=args.disk_banks)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
