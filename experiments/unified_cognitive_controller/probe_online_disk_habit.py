"""Probe the controller-to-disk-memory habit loop end to end.

This is deliberately a small integration audit, not a training run.  A frozen
controller emits the keys, values, and admission strengths.  Those tensors are
committed to real :class:`DiskLatentMemory` rows, a physical receipt identifies
the row used by each read, and only the verifier's binary action outcome is
allowed to update row-local volatility.  The audit also runs a row-corruption
control so successful and failed receipts coexist in one bank.

No task labels, rule bits, or memory-slot labels are passed to the controller
or to the memory mechanism.  Labels are used only after the fact to score the
verifier-side audit.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .environment import NULL_ACTION, generate_lifetimes
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures


def _controller_stream(
        model: UnifiedCognitiveController, count: int, *, seed: int,
        device: torch.device) -> tuple[object, dict[str, torch.Tensor]]:
    """Generate a support/query stream using only raw controller outputs."""
    batch = _add_context_signatures(
        generate_lifetimes(
            count, 3, seed=seed, heldout=True, task="binary_mapping",
            support_trials=1, device=device),
        seed=seed + 10_000_000)
    state = model.initial_state(count, device=device)
    null = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    support0, state = model.step(
        batch.frames[:, 0], state, null, zeros, zeros)
    support_action = support0.logits.argmax(-1)
    support_outcome = (
        support_action == batch.correct_actions[:, 0]).to(torch.float32)
    support1, _ = model.step(
        batch.frames[:, 1], state, support_action, support_outcome,
        torch.ones_like(support_outcome))
    fresh = model.initial_state(count, device=device)
    query0, _ = model.step(
        batch.frames[:, 2], fresh, null, zeros, zeros)
    return batch, {
        "store_keys": support0.memory_key.detach(),
        "store_values": support1.memory_value.detach(),
        "write_strengths": support1.memory_write_strength.detach(),
        "query_keys": query0.memory_key.detach(),
        "query_frames": batch.frames[:, 2],
        "correct_actions": batch.correct_actions[:, 2],
    }


def _make_memories(
        stream: dict[str, torch.Tensor], *, banks: int, capacity: int,
        width: int, device: torch.device) -> list[DiskLatentMemory]:
    memories: list[DiskLatentMemory] = []
    for bank in range(banks):
        start = bank * capacity
        end = start + capacity
        memory = DiskLatentMemory(width, capacity=capacity, device=device)
        # A controller may choose a weak write.  The integration audit must
        # still observe the physical row, so thresholding is disabled here;
        # strengths remain part of the stored controller output.
        memory.commit(
            stream["store_keys"][start:end],
            stream["store_values"][start:end],
            stream["write_strengths"][start:end].clamp_min(1e-4),
            threshold=0.0)
        memories.append(memory)
    return memories


@torch.no_grad()
def _physical_decision(
        model: UnifiedCognitiveController,
        memories: list[DiskLatentMemory],
        stream: dict[str, torch.Tensor], *,
        capacity: int, device: torch.device,
        mutate: str = "normal",
        query_roll: bool = False,
        record_access: bool = False,
        ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Read real rows, decode an action, and return outcome/receipt vectors."""
    if mutate not in {"normal", "shuffle", "corrupt_first"}:
        raise ValueError("unsupported physical-memory control")
    count = stream["query_keys"].shape[0]
    banks = count // capacity
    keys = stream["query_keys"].reshape(banks, capacity, -1)
    frames = stream["query_frames"]
    correct = stream["correct_actions"]
    reads: list[torch.Tensor] = []
    receipts: list[torch.Tensor] = []
    for bank in range(banks):
        memory = memories[bank].clone()
        if mutate == "shuffle":
            memory.store.values[:capacity] = memory.store.values[
                torch.arange(capacity - 1, -1, -1, device=device)]
        elif mutate == "corrupt_first":
            memory.store.values[0].mul_(-1.0)
        query = keys[bank]
        if query_roll:
            query = query.roll(1, dims=0)
        read, _, receipt = memory.retrieve_with_receipt(
            query, top_k=1, confidence_mode="cosine",
            record_access=record_access, usage_prior_scale=0.0)
        reads.append(read)
        receipts.append(receipt)
    recalled = torch.cat(reads, dim=0)
    state = model.initial_state(count, device=device)
    null = torch.full((count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)
    query, _ = model.step(
        frames, state, null, zeros, zeros, retrieved_memory=recalled)
    actions = query.logits.argmax(-1)
    outcomes = (actions == correct).to(torch.float32)
    return outcomes, actions, receipts


def _record_receipts(
        memories: list[DiskLatentMemory], receipts: list[torch.Tensor],
        outcomes: torch.Tensor, *, rounds: int,
        shuffled: bool = False) -> None:
    capacity = outcomes.shape[0] // len(memories)
    for _ in range(rounds):
        for bank, memory in enumerate(memories):
            local = receipts[bank]
            if shuffled:
                local = local.roll(1)
            memory.record_outcomes_from_receipts(
                local, outcomes[bank * capacity:(bank + 1) * capacity],
                update_volatility=True, success_protection_rate=0.20,
                failure_thaw_rate=0.25, stale_thaw_rate=0.0)


def run_probe(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, habit_rounds: int = 8,
        ) -> dict[str, object]:
    if banks < 1 or capacity < 2 or capacity % 2 or banks * capacity % 2:
        raise ValueError("banks must be positive and capacity must be even")
    started = time.perf_counter()
    batch, stream = _controller_stream(
        model, banks * capacity, seed=seed, device=device)
    memories = _make_memories(
        stream, banks=banks, capacity=capacity, width=model.width,
        device=device)
    normal, normal_actions, normal_receipts = _physical_decision(
        model, memories, stream, capacity=capacity, device=device,
        record_access=True)
    no_memory = stream["query_frames"].new_zeros(
        banks * capacity, model.width)
    state = model.initial_state(banks * capacity, device=device)
    null = torch.full(
        (banks * capacity,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(banks * capacity, device=device)
    no_memory_output, _ = model.step(
        stream["query_frames"], state, null, zeros, zeros,
        retrieved_memory=no_memory)
    no_memory_outcomes = (
        no_memory_output.logits.argmax(-1)
        == stream["correct_actions"]).to(torch.float32)
    shuffled, _, _ = _physical_decision(
        model, memories, stream, capacity=capacity, device=device,
        mutate="shuffle")
    # Corrupting just one controller-produced row creates a verifier-side
    # success/failure mixture without supplying a semantic row label.
    corrupted_memories = [memory.clone() for memory in memories]
    corrupt_outcomes, _, corrupt_receipts = _physical_decision(
        model, corrupted_memories, stream, capacity=capacity,
        device=device, mutate="corrupt_first")
    successes = int(corrupt_outcomes.sum())
    failures = int(corrupt_outcomes.numel() - successes)

    receipt_memories = [memory.clone() for memory in corrupted_memories]
    shuffled_receipt_memories = [memory.clone() for memory in corrupted_memories]
    _record_receipts(
        receipt_memories, corrupt_receipts, corrupt_outcomes,
        rounds=habit_rounds)
    _record_receipts(
        shuffled_receipt_memories, corrupt_receipts, corrupt_outcomes,
        rounds=habit_rounds, shuffled=True)
    receipt_volatility = torch.stack([
        memory.store.volatility[:capacity] for memory in receipt_memories])
    shuffled_volatility = torch.stack([
        memory.store.volatility[:capacity]
        for memory in shuffled_receipt_memories])
    receipt_success_rows = torch.zeros(
        banks, capacity, dtype=torch.bool, device=device)
    receipt_failure_rows = torch.zeros_like(receipt_success_rows)
    for bank, local_receipts in enumerate(corrupt_receipts):
        local_outcomes = corrupt_outcomes[
            bank * capacity:(bank + 1) * capacity]
        receipt_success_rows[bank].scatter_(
            0, local_receipts[local_outcomes > 0.5], True)
        receipt_failure_rows[bank].scatter_(
            0, local_receipts[local_outcomes <= 0.5], True)
    # Rows that saw both outcomes are excluded from the separation statistic;
    # a causal receipt can attribute them correctly, but no one-sided habit
    # claim should be made about a contradictory verifier history.
    contradictory = receipt_success_rows & receipt_failure_rows
    receipt_success_rows &= ~contradictory
    receipt_failure_rows &= ~contradictory
    successful_volatility = receipt_volatility[receipt_success_rows]
    failed_volatility = receipt_volatility[receipt_failure_rows]
    correct_gap = (
        float(failed_volatility.mean() - successful_volatility.mean())
        if successful_volatility.numel() and failed_volatility.numel()
        else 0.0)
    shuffled_gap = (
        float(
            shuffled_volatility[receipt_failure_rows].mean()
            - shuffled_volatility[receipt_success_rows].mean())
        if successful_volatility.numel() and failed_volatility.numel()
        else 0.0)

    # A row-local habit gate should make a high-volatility failed row easier to
    # replace than a low-volatility successful row.  The candidate is another
    # controller output, not a task-specific payload.
    candidate_count = max(2, banks + (banks % 2))
    _, candidate_stream = _controller_stream(
        model, candidate_count, seed=seed + 77_777, device=device)
    replacement_rewrites: list[float] = []
    for bank, memory in enumerate(receipt_memories):
        baseline = memory.clone()
        high = int(memory.store.volatility[:capacity].argmax())
        low = int(memory.store.volatility[:capacity].argmin())
        replacement_rewrites.append(memory.elastic_replace(
            high, candidate_stream["store_keys"][bank],
            candidate_stream["store_values"][bank],
            candidate_stream["write_strengths"][bank],
            minimum_rewrite=0.0))
        # The low-volatility row is tested on a separate clone so this check
        # cannot be affected by replacing the high-volatility row.
        low_clone = baseline
        low_rewrite = low_clone.elastic_replace(
            low, candidate_stream["store_keys"][bank],
            candidate_stream["store_values"][bank],
            candidate_stream["write_strengths"][bank],
            minimum_rewrite=0.0)
        replacement_rewrites.append(low_rewrite)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "online-habit.pt"
        receipt_memories[0].save(path)
        restored = DiskLatentMemory.load(path, device=device)
        disk_roundtrip = bool(torch.equal(
            restored.store.keys, receipt_memories[0].store.keys)
            and torch.equal(
                restored.store.volatility, receipt_memories[0].store.volatility))

    total_seconds = time.perf_counter() - started
    report: dict[str, object] = {
        "schema": "unified-controller-online-disk-habit-probe-v1",
        "seed": seed,
        "configuration": {
            "banks": banks, "capacity": capacity,
            "habit_rounds": habit_rounds, "device": str(device),
        },
        "checkpoint_controller_outputs_used": True,
        "semantic_or_task_labels_used_for_controller_or_memory": False,
        "learner_visible": [
            "controller latent keys", "controller latent values",
            "controller write strengths", "physical disk read receipts",
            "verifier binary outcome", "row-local volatility",
        ],
        "accuracy": {
            "physical_normal": float(normal.mean()),
            "physical_shuffled_values": float(shuffled.mean()),
            "no_memory": float(no_memory_outcomes.mean()),
            "corrupted_first_row": float(corrupt_outcomes.mean()),
        },
        "habit": {
            "corrupted_successes": successes,
            "corrupted_failures": failures,
            "successful_row_volatility": float(
                successful_volatility.mean())
            if successful_volatility.numel() else None,
            "failed_row_volatility": float(failed_volatility.mean())
            if failed_volatility.numel() else None,
            "receipt_success_minus_failure_gap": correct_gap,
            "shuffled_receipt_success_minus_failure_gap": shuffled_gap,
            "replacement_rewrites_high_then_low": replacement_rewrites,
            "disk_roundtrip_exact": disk_roundtrip,
        },
        "accounting": {
            "controller_writes": banks * capacity,
            "physical_reads": banks * capacity * 3,
            "verifier_outcome_updates": banks * capacity * habit_rounds,
            "total_seconds": total_seconds,
        },
    }
    gates = {
        "normal_at_least_85": report["accuracy"]["physical_normal"] >= 0.85,
        "memory_matters": (
            report["accuracy"]["no_memory"]
            <= report["accuracy"]["physical_normal"] - 0.15),
        "shuffled_memory_hurts": (
            report["accuracy"]["physical_shuffled_values"]
            <= report["accuracy"]["physical_normal"] - 0.15),
        "both_verifier_outcomes_present": successes > 0 and failures > 0,
        "receipt_protects_successes": correct_gap >= 0.10,
        "shuffled_receipts_weaken_protection": shuffled_gap < correct_gap,
        "high_volatility_rewrites_more": any(
            high > low + 0.10
            for high, low in zip(
                replacement_rewrites[::2], replacement_rewrites[1::2])),
        "disk_roundtrip_exact": disk_roundtrip,
        "under_one_minute": total_seconds <= 60.0,
    }
    report["gates"] = gates | {"accepted": all(gates.values())}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--banks", type=int, default=16)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--habit-rounds", type=int, default=8)
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
    report = run_probe(
        model, banks=args.banks, capacity=args.capacity,
        seed=args.seed, device=device, habit_rounds=args.habit_rounds)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
