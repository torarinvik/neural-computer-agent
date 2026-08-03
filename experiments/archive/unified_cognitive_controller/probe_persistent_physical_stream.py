"""Probe a genuinely persistent bounded physical-memory decision stream.

Unlike the online adaptation pilot, each bank is materialized only once.  The
winning decision mutates that same serialized bank, and later decisions derive
their age, access, and verifier-outcome features from the reloaded store.
"""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch

from .audit_multifeature_utility import _materialize_histories
from .audit_selective_disk import _gated_retrieve
from .environment import CognitiveLifetimeBatch
from .memory import DiskLatentMemory
from .legacy_model import UnifiedCognitiveController
from .train_adaptive_memory_read import _outcomes
from .train_frequency_recency_replacement import frequency_recency_batch
from .train_memory_replacement import (
    _gather_rows,
    _hard_bank_read,
    _select_batch,
)


def _replace_batch_rows(
        base: CognitiveLifetimeBatch, indices: torch.Tensor,
        source: CognitiveLifetimeBatch) -> CognitiveLifetimeBatch:
    fields = {}
    for name in (
            "frames", "correct_actions", "stimulus_identities",
            "rule_bits", "seeds"):
        value = getattr(base, name).clone()
        value[indices] = getattr(source, name)
        fields[name] = value
    return replace(base, **fields)


def _candidate_batch(
        data: dict[str, object], *, banks: int,
        capacity: int, device: torch.device) -> CognitiveLifetimeBatch:
    indices = (
        torch.arange(banks, device=device) * (capacity + 1) + capacity)
    return _select_batch(data["source_batch"], indices)


def _initial_rows(
        data: dict[str, object], *, banks: int,
        capacity: int, device: torch.device
        ) -> tuple[CognitiveLifetimeBatch, torch.Tensor]:
    base = (
        torch.arange(banks, device=device).unsqueeze(1) * (capacity + 1))
    indices = (base + data["slot_to_logical"]).reshape(-1)
    batch = _select_batch(data["source_batch"], indices)
    queries = _gather_rows(
        data["query_group"][:, :capacity],
        data["slot_to_logical"])
    return batch, queries


def _ranked_age(memory: DiskLatentMemory) -> torch.Tensor:
    """Map persistent insertion clocks back to the controller's [1/C, 1] scale."""
    ages = memory.store.age[:memory.count]
    order = ages.argsort()
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(
        1, memory.count + 1, device=ages.device)
    return ranks.to(memory.store.keys.dtype) / memory.count


def _stream_features(
        model: UnifiedCognitiveController,
        memories: list[DiskLatentMemory],
        candidate_keys: torch.Tensor,
        candidate_strengths: torch.Tensor,
        *, weights: tuple[float, float, float],
        noise: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    targets = []
    for bank, memory in enumerate(memories):
        capacity = memory.count
        keys = memory.store.keys[:capacity]
        age = _ranked_age(memory)
        access = (
            torch.log1p(
                memory.store.access_count[:capacity].to(keys.dtype))
            / math.log(10.0))
        reliability = (
            (memory.store.success_count[:capacity].to(keys.dtype) + 1.0)
            / (
                memory.store.success_count[:capacity]
                + memory.store.failure_count[:capacity] + 2
            ).to(keys.dtype))
        similarity = (
            torch.nn.functional.normalize(
                candidate_keys[bank], dim=-1)
            @ torch.nn.functional.normalize(keys, dim=-1).T)
        row = torch.stack((
            age,
            memory.store.usage[:capacity],
            similarity,
            candidate_strengths[bank].expand(capacity),
            torch.zeros_like(age),
            access - 0.5,
            reliability - 0.5,
        ), dim=-1)
        skip = torch.zeros(
            1, model.adaptive_memory_replace_features,
            device=keys.device, dtype=keys.dtype)
        skip[0, 3] = candidate_strengths[bank]
        skip[0, 4] = 1.0
        if model.adaptive_memory_replace_features > 7:
            row = torch.cat((
                row,
                row.new_zeros(
                    capacity,
                    model.adaptive_memory_replace_features - 7),
            ), dim=-1)
        rows.append(torch.cat((skip, row)))
        visible = (
            weights[0] * age
            + weights[1] * access
            + weights[2] * reliability)
        targets.append((visible + noise[bank]).argmin() + 1)
    return torch.stack(rows), torch.stack(targets)


def _future_for_actions(
        row_batch: CognitiveLifetimeBatch, row_queries: torch.Tensor,
        candidate_batch: CognitiveLifetimeBatch,
        candidate_queries: torch.Tensor, actions: torch.Tensor,
        *, capacity: int,
        ) -> tuple[CognitiveLifetimeBatch, torch.Tensor]:
    future_batch = row_batch
    future_queries = row_queries.clone()
    replacing = (actions > 0).nonzero(as_tuple=False).squeeze(1)
    if replacing.numel():
        flat = replacing * capacity + actions[replacing] - 1
        future_batch = _replace_batch_rows(
            future_batch, flat,
            _select_batch(candidate_batch, replacing))
        future_queries[replacing, actions[replacing] - 1] = (
            candidate_queries[replacing])
    return future_batch, future_queries


@torch.no_grad()
def _physical_rewards(
        model: UnifiedCognitiveController,
        memories: list[DiskLatentMemory],
        data: dict[str, object],
        row_batch: CognitiveLifetimeBatch,
        row_queries: torch.Tensor,
        candidate_batch: CognitiveLifetimeBatch,
        candidate_queries: torch.Tensor,
        actions: torch.Tensor, desired_actions: torch.Tensor,
        directory: Path, *, device: torch.device,
        ) -> tuple[torch.Tensor, int]:
    capacity = row_queries.shape[1]
    future_batch, future_queries = _future_for_actions(
        row_batch, row_queries, candidate_batch, candidate_queries,
        desired_actions, capacity=capacity)
    reads = []
    persisted = 0
    for bank, source in enumerate(memories):
        memory = DiskLatentMemory.__new__(DiskLatentMemory)
        memory.store = source.store.clone()
        action = int(actions[bank])
        if action:
            memory.replace(
                action - 1, data["candidate_key"][bank],
                data["candidate_value"][bank],
                data["candidate_strength"][bank])
        path = directory / f"candidate-{bank:04d}.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        persisted += int(
            restored.count == capacity
            and restored.store.capacity == capacity)
        reads.append(_gated_retrieve(
            model, restored, future_queries[bank],
            read_threshold=None))
    outcomes = _outcomes(
        model, future_batch, torch.cat(reads), device=device)
    return outcomes.reshape(len(memories), capacity).mean(-1), persisted


@torch.no_grad()
def _tensor_rewards(
        model: UnifiedCognitiveController,
        memories: list[DiskLatentMemory],
        data: dict[str, object],
        row_batch: CognitiveLifetimeBatch,
        row_queries: torch.Tensor,
        candidate_batch: CognitiveLifetimeBatch,
        candidate_queries: torch.Tensor,
        actions: torch.Tensor, desired_actions: torch.Tensor, *,
        device: torch.device,
        ) -> torch.Tensor:
    capacity = row_queries.shape[1]
    keys = torch.stack([
        memory.store.keys[:capacity] for memory in memories])
    values = torch.stack([
        memory.store.values[:capacity] for memory in memories])
    strengths = torch.stack([
        memory.store.usage[:capacity] for memory in memories])
    replacing = (actions > 0).nonzero(as_tuple=False).squeeze(1)
    if replacing.numel():
        slots = actions[replacing] - 1
        keys[replacing, slots] = data["candidate_key"][replacing]
        values[replacing, slots] = data["candidate_value"][replacing]
        strengths[replacing, slots] = data["candidate_strength"][replacing]
    future_batch, future_queries = _future_for_actions(
        row_batch, row_queries, candidate_batch, candidate_queries,
        desired_actions, capacity=capacity)
    reads = _hard_bank_read(
        model, keys, values, strengths, future_queries)
    outcomes = _outcomes(
        model, future_batch, reads.reshape(-1, reads.shape[-1]),
        device=device)
    return outcomes.reshape(len(memories), capacity).mean(-1)


@torch.no_grad()
def _apply_winner(
        model: UnifiedCognitiveController,
        memories: list[DiskLatentMemory],
        data: dict[str, object],
        row_batch: CognitiveLifetimeBatch,
        row_queries: torch.Tensor,
        candidate_batch: CognitiveLifetimeBatch,
        candidate_queries: torch.Tensor,
        actions: torch.Tensor, desired_actions: torch.Tensor,
        directory: Path, *, device: torch.device,
        ) -> tuple[
            list[DiskLatentMemory], CognitiveLifetimeBatch, torch.Tensor,
            int, int, bool]:
    capacity = row_queries.shape[1]
    evaluation_batch, evaluation_queries = _future_for_actions(
        row_batch, row_queries, candidate_batch, candidate_queries,
        desired_actions, capacity=capacity)
    updated_batch, updated_queries = _future_for_actions(
        row_batch, row_queries, candidate_batch, candidate_queries,
        actions, capacity=capacity)
    restored_memories = []
    exact = 0
    transition_exact = True
    for bank, memory in enumerate(memories):
        action = int(actions[bank])
        before_access = int(memory.store.access_count.sum())
        before_outcomes = int(
            memory.store.success_count.sum()
            + memory.store.failure_count.sum())
        discarded_access = 0
        discarded_outcomes = 0
        if action:
            discarded_access = int(
                memory.store.access_count[action - 1])
            discarded_outcomes = int(
                memory.store.success_count[action - 1]
                + memory.store.failure_count[action - 1])
            memory.replace(
                action - 1, data["candidate_key"][bank],
                data["candidate_value"][bank],
                data["candidate_strength"][bank])
        reads, _ = memory.retrieve(
            evaluation_queries[bank], top_k=1,
            confidence_mode="cosine", record_access=True)
        outcomes = _outcomes(
            model,
            _select_batch(
                evaluation_batch,
                torch.arange(
                    bank * capacity, (bank + 1) * capacity,
                    device=device)),
            reads, device=device)
        memory.store.record_outcomes(evaluation_queries[bank], outcomes)
        transition_exact = transition_exact and (
            int(memory.store.access_count.sum())
            == before_access - discarded_access + capacity
            and int(
                memory.store.success_count.sum()
                + memory.store.failure_count.sum())
            == before_outcomes - discarded_outcomes + capacity)
        path = directory / f"bank-{bank:04d}.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        exact += int(
            torch.equal(
                restored.store.access_count,
                memory.store.access_count)
            and torch.equal(
                restored.store.success_count,
                memory.store.success_count)
            and torch.equal(
                restored.store.failure_count,
                memory.store.failure_count)
            and restored.count == capacity)
        restored_memories.append(restored)
    return (
        restored_memories, updated_batch, updated_queries, exact,
        int((actions > 0).sum()), transition_exact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7021)
    parser.add_argument("--banks", type=int, default=8)
    parser.add_argument("--bank-capacity", type=int, default=6)
    parser.add_argument("--rounds-per-phase", type=int, default=2)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if args.banks < 1 or args.rounds_per_phase < 1:
        raise ValueError("banks and rounds per phase must be positive")
    seed_everything = torch.manual_seed
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    phases = [
        ("old_equal", (0.5, 0.5, 0.0)),
        ("reliability_dominant", (0.3, 0.3, 0.4)),
        ("old_return", (0.5, 0.5, 0.0)),
    ]
    started = time.perf_counter()
    initial = frequency_recency_batch(
        model, banks=args.banks, capacity=args.bank_capacity,
        seed=args.seed * 10_000, device=device,
        write_threshold=0.5, noise_scale=0.04,
        recency_weight=0.5, frequency_weight=0.5)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        initial_directory = root / "initial"
        initial_directory.mkdir()
        memories, _, _, _, initial_exact, requested_exact = (
            _materialize_histories(
                model, initial, initial_directory, device=device))
        row_batch, row_queries = _initial_rows(
            initial, banks=args.banks, capacity=args.bank_capacity,
            device=device)
        trace = []
        maximum_parity_difference = 0.0
        maximum_cross_choice_regret = 0.0
        persisted_exact = initial_exact
        candidate_persisted = 0
        total_replacements = 0
        every_history_transition_exact = True
        initial_access = sum(
            int(memory.store.access_count.sum()) for memory in memories)
        initial_outcomes = sum(
            int(
                memory.store.success_count.sum()
                + memory.store.failure_count.sum())
            for memory in memories)
        rounds = 0
        for phase_index, (phase, weights) in enumerate(phases):
            for round_index in range(args.rounds_per_phase):
                rounds += 1
                seed = (
                    args.seed * 10_000_000
                    + phase_index * 1_000_000 + round_index + 1)
                data = frequency_recency_batch(
                    model, banks=args.banks,
                    capacity=args.bank_capacity, seed=seed,
                    device=device, write_threshold=0.5,
                    noise_scale=0.04,
                    recency_weight=weights[0],
                    frequency_weight=weights[1],
                    reliability_weight=weights[2])
                candidate_batch = _candidate_batch(
                    data, banks=args.banks,
                    capacity=args.bank_capacity, device=device)
                candidate_queries = data["query_group"][:, -1]
                generator = torch.Generator(
                    device=device).manual_seed(seed + 70_000_000)
                noise = (
                    torch.rand(
                        args.banks, args.bank_capacity,
                        generator=generator, device=device) * 2 - 1
                ) * 0.04
                features, target = _stream_features(
                    model, memories, data["candidate_key"],
                    data["candidate_strength"], weights=weights,
                    noise=noise)
                direction = torch.randint(
                    0, 2, (2,), generator=generator,
                    device=device).float() * 2 - 1
                current = (
                    model.memory_replacement_extra_gate.weight
                    .detach().clone())
                signs = (1.0, 0.0, -1.0)
                physical_means = []
                tensor_means = []
                actions_by_candidate = []
                for candidate_index, sign in enumerate(signs):
                    model.memory_replacement_extra_gate.weight.copy_(
                        current + sign * 3.0 * direction.unsqueeze(0))
                    actions = model.memory_replacement_scores(
                        features).argmax(-1)
                    actions_by_candidate.append(actions)
                    candidate_directory = (
                        root / f"round-{rounds:03d}-candidate-"
                        f"{candidate_index}")
                    candidate_directory.mkdir()
                    physical, exact = _physical_rewards(
                        model, memories, data, row_batch, row_queries,
                        candidate_batch, candidate_queries, actions, target,
                        candidate_directory, device=device)
                    tensor = _tensor_rewards(
                        model, memories, data, row_batch, row_queries,
                        candidate_batch, candidate_queries, actions, target,
                        device=device)
                    candidate_persisted += exact
                    physical_means.append(float(physical.mean()))
                    tensor_means.append(float(tensor.mean()))
                physical_winner = max(
                    range(3), key=physical_means.__getitem__)
                tensor_winner = max(
                    range(3), key=tensor_means.__getitem__)
                cross_regret = max(
                    max(tensor_means) - tensor_means[physical_winner],
                    max(physical_means) - physical_means[tensor_winner])
                maximum_cross_choice_regret = max(
                    maximum_cross_choice_regret, cross_regret)
                maximum_parity_difference = max(
                    maximum_parity_difference,
                    max(abs(a - b) for a, b in zip(
                        physical_means, tensor_means)))
                winner_actions = actions_by_candidate[physical_winner]
                model.memory_replacement_extra_gate.weight.copy_(current)
                state_directory = root / f"round-{rounds:03d}-state"
                state_directory.mkdir()
                (
                    memories, row_batch, row_queries, exact, replacements,
                    transition_exact,
                ) = _apply_winner(
                    model, memories, data, row_batch, row_queries,
                    candidate_batch, candidate_queries, winner_actions,
                    target,
                    state_directory, device=device)
                persisted_exact += exact
                total_replacements += replacements
                every_history_transition_exact = (
                    every_history_transition_exact and transition_exact)
                trace.append({
                    "phase": phase,
                    "round": round_index + 1,
                    "physical_rewards": physical_means,
                    "tensor_rewards": tensor_means,
                    "physical_winner": physical_winner,
                    "tensor_winner": tensor_winner,
                    "cross_choice_regret": cross_regret,
                    "target_eviction_rate": float(
                        (winner_actions == target).float().mean()),
                    "replacements": replacements,
                    "total_access_count": sum(
                        int(memory.store.access_count.sum())
                        for memory in memories),
                    "total_outcome_count": sum(
                        int(
                            memory.store.success_count.sum()
                            + memory.store.failure_count.sum())
                        for memory in memories),
                })
        final_access = sum(
            int(memory.store.access_count.sum()) for memory in memories)
        final_outcomes = sum(
            int(
                memory.store.success_count.sum()
                + memory.store.failure_count.sum())
            for memory in memories)
        audit_seed = args.seed * 10_000_000 + 99_000_001
        audit_data = frequency_recency_batch(
            model, banks=args.banks, capacity=args.bank_capacity,
            seed=audit_seed, device=device, write_threshold=0.5,
            noise_scale=0.04, recency_weight=1 / 3,
            frequency_weight=1 / 3, reliability_weight=1 / 3)
        audit_generator = torch.Generator(
            device=device).manual_seed(audit_seed + 70_000_000)
        audit_noise = (
            torch.rand(
                args.banks, args.bank_capacity,
                generator=audit_generator, device=device) * 2 - 1
        ) * 0.04
        normal_features, _ = _stream_features(
            model, memories, audit_data["candidate_key"],
            audit_data["candidate_strength"],
            weights=(1 / 3, 1 / 3, 1 / 3), noise=audit_noise)
        corrupted_memories = []
        corruption_exact = 0
        corruption_directory = root / "corrupted-history-audit"
        corruption_directory.mkdir()
        for bank, source in enumerate(memories):
            corrupted = DiskLatentMemory.__new__(DiskLatentMemory)
            corrupted.store = source.store.clone()
            for field in (
                    "access_count", "success_count", "failure_count"):
                value = getattr(corrupted.store, field)
                value[:args.bank_capacity] = value[
                    :args.bank_capacity].roll(1)
            path = corruption_directory / f"bank-{bank:04d}.pt"
            corrupted.save(path)
            restored = DiskLatentMemory.load(path, device=device)
            corruption_exact += int(
                torch.equal(
                    restored.store.access_count,
                    corrupted.store.access_count)
                and torch.equal(
                    restored.store.success_count,
                    corrupted.store.success_count)
                and torch.equal(
                    restored.store.failure_count,
                    corrupted.store.failure_count))
            corrupted_memories.append(restored)
        corrupted_features, _ = _stream_features(
            model, corrupted_memories, audit_data["candidate_key"],
            audit_data["candidate_strength"],
            weights=(1 / 3, 1 / 3, 1 / 3), noise=audit_noise)
        normal_actions = model.memory_replacement_scores(
            normal_features).argmax(-1)
        corrupted_actions = model.memory_replacement_scores(
            corrupted_features).argmax(-1)
        corruption_action_flip_rate = float(
            (normal_actions != corrupted_actions).float().mean())
        expected_state_saves = args.banks * (rounds + 1)
        expected_candidate_saves = args.banks * rounds * 3
        gate = {
            "every_state_save_reload_exact":
                persisted_exact == expected_state_saves,
            "every_candidate_remained_bounded":
                candidate_persisted == expected_candidate_saves,
            "physical_tensor_rewards_match_within_1e_6":
                maximum_parity_difference <= 1e-6,
            "physical_tensor_choices_equivalent_within_1e_6":
                maximum_cross_choice_regret <= 1e-6,
            "every_read_and_outcome_transition_accounted_exactly":
                every_history_transition_exact,
            "persistent_access_history_nonempty": final_access > 0,
            "persistent_outcome_history_nonempty": final_outcomes > 0,
            "corrupted_histories_persisted_exactly":
                corruption_exact == args.banks,
            "history_corruption_changes_at_least_one_decision":
                corruption_action_flip_rate > 0.0,
            "at_least_one_replacement_persisted":
                total_replacements > 0,
            "all_banks_remained_at_capacity": all(
                memory.count == args.bank_capacity
                and memory.store.capacity == args.bank_capacity
                for memory in memories),
        }
        gate["accepted"] = all(gate.values())
    report = {
        "schema": "unified-controller-persistent-physical-stream-v1",
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "banks": args.banks,
        "bank_capacity": args.bank_capacity,
        "rounds_per_phase": args.rounds_per_phase,
        "phases": [
            {"name": name, "weights": weights}
            for name, weights in phases],
        "trace": trace,
        "accounting": {
            "persistent_banks": args.banks,
            "physical_rounds": rounds,
            "state_save_reloads": expected_state_saves,
            "candidate_save_reloads": expected_candidate_saves,
            "total_replacements": total_replacements,
            "initial_access_count": initial_access,
            "final_access_count": final_access,
            "initial_outcome_count": initial_outcomes,
            "final_outcome_count": final_outcomes,
            "requested_initial_histories_reproduced_exactly":
                requested_exact,
        },
        "maximum_physical_tensor_reward_difference":
            maximum_parity_difference,
        "maximum_cross_choice_regret": maximum_cross_choice_regret,
        "history_corruption_action_flip_rate":
            corruption_action_flip_rate,
        "gate": gate,
        "total_seconds": time.perf_counter() - started,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "gate": gate,
        "total_seconds": report["total_seconds"],
        "final_access_count": final_access,
        "final_outcome_count": final_outcomes,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
