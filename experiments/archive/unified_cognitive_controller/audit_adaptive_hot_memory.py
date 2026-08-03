"""Audit a lossless cold disk archive with a verified adaptive hot working set."""
from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from .audit_adaptive_physical_pruning import (
    _load,
    query_event,
)
from .memory import DiskLatentMemory, TieredLatentMemory
from .train import seed_everything
from .train_equivalence_consolidation import (
    consolidate,
    natural_memory_streams,
)


@dataclass(frozen=True)
class Phase:
    name: str
    appearance: str
    rounds: int


PHASES = (
    Phase("hard_initial", "dot_pairs", 16),
    Phase("easy_interlude", "bars", 24),
    Phase("hard_return", "dot_pairs", 12),
)


def _bank_repeat(values: torch.Tensor) -> torch.Tensor:
    return values.repeat(2)


def _stratified_shuffle(
        signal: torch.Tensor, candidate_count: torch.Tensor, *,
        generator: torch.Generator,
        ) -> torch.Tensor:
    """Shuffle bank evidence without changing its count or row budget."""
    shuffled = torch.zeros_like(signal)
    for count in candidate_count.unique():
        group = torch.where(candidate_count == count)[0]
        order = torch.randperm(
            group.numel(), generator=generator, device=signal.device)
        shuffled[group] = signal[group[order]]
    return shuffled


@torch.no_grad()
def _events(
        model, bank: dict[str, torch.Tensor], *, seed: int,
        appearance: str, device: torch.device,
        ) -> dict[str, dict[str, torch.Tensor]]:
    core = bank["valid"] & (bank["representative_ranks"] < 2)
    full = bank["valid"]
    return {
        "core": query_event(
            model, bank, seed=seed, appearance=appearance,
            physical_mask=core, device=device),
        "full": query_event(
            model, bank, seed=seed, appearance=appearance,
            physical_mask=full, device=device),
        "cold_retry": query_event(
            model, bank, seed=seed, appearance=appearance,
            physical_mask=full, device=device, force_deep=True),
    }


def _metric_row(
        *, correct: torch.Tensor, recovered: torch.Tensor,
        comparisons: torch.Tensor, retries: torch.Tensor,
        hot_rows: torch.Tensor, protected: torch.Tensor,
        ) -> dict[str, float]:
    return {
        "first_attempt_accuracy": float(correct.float().mean()),
        "accuracy_after_cold_retry": float(recovered.float().mean()),
        "mean_hot_rows": float(hot_rows.float().mean()),
        "mean_comparisons": float(comparisons.float().mean()),
        "cold_retry_rate": float(retries.float().mean()),
        "protected_bank_rate": float(protected.float().mean()),
    }


@torch.no_grad()
def simulate(
        model, bank: dict[str, torch.Tensor], *, decay: float,
        threshold: float, seed: int, device: torch.device,
        ) -> dict[str, object]:
    """Run cumulative, decaying, and shuffled-evidence policies together."""
    if not 0.0 <= decay <= 1.0:
        raise ValueError("decay must be between zero and one")
    streams = bank["values"].shape[0]
    full_mask = bank["valid"]
    core_mask = full_mask & (bank["representative_ranks"] < 2)
    extra_mask = full_mask & ~core_mask
    core_rows = core_mask.sum(-1)
    full_rows = full_mask.sum(-1)
    candidate_count = extra_mask.sum(-1)
    rank = bank["representative_ranks"].repeat(2, 1)
    generator = torch.Generator(device=device).manual_seed(seed + 90_000_000)
    scores = {
        "cumulative": torch.zeros(streams, device=device),
        "decaying": torch.zeros(streams, device=device),
        "shuffled_decay": torch.zeros(streams, device=device),
    }
    phase_reports: dict[str, object] = {}
    traces: dict[str, list[dict[str, float]]] = {
        "cumulative": [], "decaying": [], "shuffled_decay": [],
        "fixed_core": [], "fixed_full": [],
    }

    event_seed = seed + 1_000_000
    for phase_index, phase in enumerate(PHASES):
        aggregates = {
            policy: {
                key: [] for key in (
                    "correct", "recovered", "comparisons",
                    "retries", "hot_rows", "protected")
            }
            for policy in scores
        }
        fixed_core_correct = []
        fixed_full_correct = []
        fixed_core_rows = []
        fixed_full_rows = []
        for round_index in range(phase.rounds):
            event = _events(
                model, bank,
                seed=event_seed + phase_index * 10_000 + round_index,
                appearance=phase.appearance, device=device)
            core_correct = event["core"]["selected_correct"].to(torch.bool)
            full_correct = event["full"]["selected_correct"].to(torch.bool)
            cold_correct = event["cold_retry"][
                "selected_correct"].to(torch.bool)
            cold_selected = event["cold_retry"]["deep_row"]
            cold_extra = rank.gather(
                1, cold_selected[:, None]).squeeze(1) >= 2
            fixed_core_correct.append(core_correct)
            fixed_full_correct.append(full_correct)
            fixed_core_rows.append(_bank_repeat(core_rows))
            fixed_full_rows.append(_bank_repeat(full_rows))
            traces["fixed_core"].append({
                "first_attempt_accuracy": float(
                    core_correct.float().mean()),
                "mean_hot_rows": float(core_rows.float().mean()),
            })
            traces["fixed_full"].append({
                "first_attempt_accuracy": float(
                    full_correct.float().mean()),
                "mean_hot_rows": float(full_rows.float().mean()),
            })

            raw_rescue_by_policy: dict[str, torch.Tensor] = {}
            for policy, score in scores.items():
                protected = score >= threshold
                protected_event = _bank_repeat(protected)
                hot_correct = torch.where(
                    protected_event, full_correct, core_correct)
                hot_comparisons = torch.where(
                    protected_event,
                    event["full"]["comparisons"],
                    event["core"]["comparisons"])
                retry = ~hot_correct
                recovered = hot_correct | (retry & cold_correct)
                rescue_event = retry & cold_correct & cold_extra
                rescue = rescue_event.reshape(2, streams).any(0)
                raw_rescue_by_policy[policy] = rescue
                hot_rows = torch.where(
                    protected, full_rows, core_rows)
                row = _metric_row(
                    correct=hot_correct, recovered=recovered,
                    comparisons=(
                        hot_comparisons
                        + retry.to(hot_comparisons.dtype)
                        * event["cold_retry"]["comparisons"]),
                    retries=retry, hot_rows=hot_rows,
                    protected=protected)
                traces[policy].append(row)
                for key, value in (
                        ("correct", hot_correct),
                        ("recovered", recovered),
                        ("comparisons", (
                            hot_comparisons
                            + retry.to(hot_comparisons.dtype)
                            * event["cold_retry"]["comparisons"])),
                        ("retries", retry),
                        ("hot_rows", hot_rows),
                        ("protected", protected)):
                    aggregates[policy][key].append(value)

            scores["cumulative"].add_(
                raw_rescue_by_policy["cumulative"].float())
            scores["decaying"].mul_(decay).add_(
                raw_rescue_by_policy["decaying"].float())
            shuffled = _stratified_shuffle(
                raw_rescue_by_policy["decaying"],
                candidate_count, generator=generator)
            scores["shuffled_decay"].mul_(decay).add_(shuffled.float())

        phase_report: dict[str, object] = {}
        for policy, values in aggregates.items():
            phase_report[policy] = _metric_row(
                correct=torch.cat(values["correct"]),
                recovered=torch.cat(values["recovered"]),
                comparisons=torch.cat(values["comparisons"]),
                retries=torch.cat(values["retries"]),
                hot_rows=torch.cat(values["hot_rows"]),
                protected=torch.cat(values["protected"]))
        phase_report["fixed_core"] = {
            "first_attempt_accuracy": float(
                torch.cat(fixed_core_correct).float().mean()),
            "mean_hot_rows": float(
                torch.cat(fixed_core_rows).float().mean()),
        }
        phase_report["fixed_full"] = {
            "first_attempt_accuracy": float(
                torch.cat(fixed_full_correct).float().mean()),
            "mean_hot_rows": float(
                torch.cat(fixed_full_rows).float().mean()),
        }
        phase_reports[phase.name] = phase_report

    return {
        "phases": phase_reports,
        "round_traces": traces,
        "final_protection": {
            name: {
                "mean": float(score.mean()),
                "protected_bank_rate": float(
                    (score >= threshold).float().mean()),
            }
            for name, score in scores.items()
        },
    }


@torch.no_grad()
def disk_hot_cold_audit(
        model, bank: dict[str, torch.Tensor], *,
        banks: int, device: torch.device,
        ) -> dict[str, object]:
    """Verify cold archives and compact hot stores coexist losslessly."""
    banks = min(banks, bank["values"].shape[0])
    exact_cold = exact_hot = 0
    exact_tier_metadata = promotion_thaw_cycles = 0
    cold_bytes = hot_bytes = 0
    with tempfile.TemporaryDirectory(prefix="adaptive-hot-cold-") as root:
        directory = Path(root)
        for index in range(banks):
            valid = bank["valid"][index]
            cold = DiskLatentMemory(
                model.width, capacity=int(valid.sum()), device=device)
            cold.commit(
                bank["keys"][index, valid],
                bank["values"][index, valid],
                bank["usage"][index, valid], threshold=0.0)
            cold.store.volatility[:cold.count] = torch.linspace(
                0.1, 0.9, cold.count, device=device)
            ranks = bank["representative_ranks"][index, valid]
            archive = TieredLatentMemory(
                cold, ranks, protection=0.0, threshold=0.5)
            archive_path = directory / f"archive-{index:04d}"
            archive.save(archive_path)
            restored_archive = TieredLatentMemory.load(
                archive_path, device=device)
            restored_cold = restored_archive.cold
            exact_cold += int(
                torch.equal(restored_cold.store.keys, cold.store.keys)
                and torch.equal(
                    restored_cold.store.values, cold.store.values)
                and torch.equal(
                    restored_cold.store.volatility,
                    cold.store.volatility))
            exact_tier_metadata += int(
                torch.equal(
                    restored_archive.representative_ranks, ranks)
                and restored_archive.protection == 0.0
                and restored_archive.threshold == 0.5)
            hot = restored_archive.hot()
            hot_path = directory / f"hot-{index:04d}.pt"
            hot.save(hot_path)
            restored_hot = DiskLatentMemory.load(hot_path, device=device)
            exact_hot += int(
                torch.equal(restored_hot.store.keys, hot.store.keys)
                and torch.equal(
                    restored_hot.store.values, hot.store.values)
                and torch.equal(
                    restored_hot.store.volatility,
                    hot.store.volatility))
            core_count = hot.count
            restored_archive.observe_verified_rescue(True, decay=0.9)
            promoted_count = restored_archive.hot().count
            for _ in range(7):
                restored_archive.observe_verified_rescue(False, decay=0.9)
            thawed_count = restored_archive.hot().count
            if core_count < cold.count:
                cycle_valid = (
                    core_count < promoted_count
                    and thawed_count == core_count)
            else:
                # A bank with no reserve is already maximally compact.
                cycle_valid = (
                    promoted_count == core_count
                    and thawed_count == core_count)
            promotion_thaw_cycles += int(
                cycle_valid
                and restored_archive.cold.count == cold.count)
            cold_bytes += sum(
                path.stat().st_size
                for path in archive_path.iterdir())
            hot_bytes += hot_path.stat().st_size
    return {
        "banks": banks,
        "exact_cold_reloads": exact_cold,
        "exact_hot_reloads": exact_hot,
        "exact_tier_metadata_reloads": exact_tier_metadata,
        "successful_promotion_thaw_cycles": promotion_thaw_cycles,
        "all_cold_and_hot_reload_exactly":
            exact_cold == banks
            and exact_hot == banks
            and exact_tier_metadata == banks,
        "all_archives_promote_and_thaw_without_cold_loss":
            promotion_thaw_cycles == banks,
        "cold_bytes": cold_bytes,
        "hot_bytes": hot_bytes,
        "hot_to_cold_byte_ratio": hot_bytes / cold_bytes,
    }


@torch.no_grad()
def audit(
        checkpoint: Path, *, streams: int, seed: int,
        decay: float, threshold: float, disk_banks: int,
        device: torch.device,
        ) -> dict[str, object]:
    payload, model = _load(checkpoint, device)
    inherited_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    data = natural_memory_streams(
        model, streams=streams, length=16, seed=seed,
        device=device, heldout=True)
    bank = consolidate(
        model, data, capacity=6, representatives_per_class=3)
    simulation = simulate(
        model, bank, decay=decay, threshold=threshold,
        seed=seed, device=device)
    disk = disk_hot_cold_audit(
        model, bank, banks=disk_banks, device=device)
    corrupted_bank = {
        name: value.clone() if isinstance(value, torch.Tensor) else value
        for name, value in bank.items()
    }
    corrupted_bank["values"].zero_()
    corrupted = query_event(
        model, corrupted_bank, seed=seed + 99_000_000,
        appearance="dot_pairs",
        physical_mask=corrupted_bank["valid"],
        device=device, force_deep=True)
    corrupted_accuracy = float(
        corrupted["selected_correct"].float().mean())
    inherited_bit_identical = all(
        torch.equal(inherited_state[name], value)
        for name, value in model.state_dict().items())
    phases = simulation["phases"]
    assert isinstance(phases, dict)
    initial = phases["hard_initial"]
    easy = phases["easy_interlude"]
    returned = phases["hard_return"]
    assert isinstance(initial, dict)
    assert isinstance(easy, dict)
    assert isinstance(returned, dict)
    traces = simulation["round_traces"]
    assert isinstance(traces, dict)
    return_start = PHASES[0].rounds + PHASES[1].rounds
    fixed_trace = traces["fixed_core"]
    decaying_trace = traces["decaying"]
    shuffled_trace = traces["shuffled_decay"]

    def paired_gain(
            candidate: list[dict[str, float]],
            baseline: list[dict[str, float]], start: int, stop: int,
            ) -> float:
        return sum(
            candidate[index]["first_attempt_accuracy"]
            - baseline[index]["first_attempt_accuracy"]
            for index in range(start, stop)) / (stop - start)

    early_start = return_start
    early_stop = return_start + 4
    late_start = return_start + PHASES[2].rounds - 4
    late_stop = return_start + PHASES[2].rounds
    reactivation = {
        "first_four_gain_over_fixed_core": paired_gain(
            decaying_trace, fixed_trace, early_start, early_stop),
        "last_four_gain_over_fixed_core": paired_gain(
            decaying_trace, fixed_trace, late_start, late_stop),
        "last_four_gain_over_shuffled_evidence": paired_gain(
            decaying_trace, shuffled_trace, late_start, late_stop),
    }
    gates = {
        "initial_hard_beats_fixed_core_by_0_05_points":
            initial["decaying"]["first_attempt_accuracy"]
            >= initial["fixed_core"]["first_attempt_accuracy"] + 0.0005,
        "easy_accuracy_at_least_99_9_percent":
            easy["decaying"]["first_attempt_accuracy"] >= 0.999,
        "easy_thaws_at_least_0_10_rows_vs_cumulative":
            easy["decaying"]["mean_hot_rows"]
            <= easy["cumulative"]["mean_hot_rows"] - 0.10,
        "returned_hard_beats_fixed_core_by_0_05_points":
            returned["decaying"]["first_attempt_accuracy"]
            >= returned["fixed_core"]["first_attempt_accuracy"] + 0.0005,
        "returned_hard_beats_shuffled_by_0_05_points":
            returned["decaying"]["first_attempt_accuracy"]
            >= returned["shuffled_decay"]["first_attempt_accuracy"] + 0.0005,
        "cold_archive_recovers_at_least_0_2_points":
            returned["decaying"]["accuracy_after_cold_retry"]
            >= returned["decaying"]["first_attempt_accuracy"] + 0.002,
        "reactivation_gain_grows_by_0_05_points":
            reactivation["last_four_gain_over_fixed_core"]
            >= reactivation["first_four_gain_over_fixed_core"] + 0.0005,
        "late_reactivation_beats_shuffle_by_0_08_points":
            reactivation["last_four_gain_over_shuffled_evidence"] >= 0.0008,
        "cold_and_hot_disk_reload_exact":
            disk["all_cold_and_hot_reload_exactly"],
        "all_physical_archives_promote_and_thaw":
            disk["all_archives_promote_and_thaw_without_cold_loss"],
        "hot_serialization_smaller_than_cold":
            disk["hot_to_cold_byte_ratio"] < 1.0,
        "corrupted_cold_archive_at_most_60_percent":
            corrupted_accuracy <= 0.60,
        "inherited_controller_bit_identical":
            inherited_bit_identical,
    }
    gates["accepted"] = all(gates.values())
    total_rounds = sum(phase.rounds for phase in PHASES)
    executed_cold_retries = round(
        sum(
            row["cold_retry_rate"]
            for row in decaying_trace)
        * streams * 2)
    return {
        "schema": "adaptive-hot-cold-memory-audit-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_training_examples":
            payload.get("representative_read_training_examples"),
        "streams": streams,
        "seed": seed,
        "decay": decay,
        "threshold": threshold,
        "phases": [
            {"name": phase.name, "appearance": phase.appearance,
             "rounds": phase.rounds}
            for phase in PHASES
        ],
        **simulation,
        "physical_disk": disk,
        "corrupted_cold_archive_accuracy": corrupted_accuracy,
        "inherited_controller_bit_identical":
            inherited_bit_identical,
        "reactivation": reactivation,
        "accounting": {
            "hot_events": streams * 2 * total_rounds,
            "policy_executed_cold_retries": executed_cold_retries,
            "audit_computed_cold_counterfactuals":
                streams * 2 * total_rounds,
            "fixed_baseline_events": streams * 4 * total_rounds,
            "controller_optimizer_updates": 0,
            "semantic_labels_visible": False,
        },
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--streams", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=47_001)
    parser.add_argument("--decay", type=float, default=0.90)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--disk-banks", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    seed_everything(args.seed)
    report = audit(
        args.checkpoint, streams=args.streams, seed=args.seed,
        decay=args.decay, threshold=args.threshold,
        disk_banks=args.disk_banks, device=torch.device(args.device))
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
