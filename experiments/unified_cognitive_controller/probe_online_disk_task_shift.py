"""Probe bounded task-shift acquisition using controller-created disk rows.

The old bank is filled by a frozen controller.  One row is intentionally made
an unsuccessful decoy and receives failed verifier outcomes through physical
read receipts.  A fresh controller-produced row then competes for the same
bounded capacity.  The habit policy replaces the most volatile row; controls
replace a least-volatile or shuffled-volatility row.  The controller sees only
latent rows, receipts, and scalar outcomes--the decoy/new-row distinction is
verifier-side audit state.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_online_disk_habit import (
    _controller_stream,
    _make_memories,
)


@torch.no_grad()
def _read_actions(
        model: UnifiedCognitiveController,
        memories: list[DiskLatentMemory],
        stream: dict[str, torch.Tensor], *,
        keys: torch.Tensor, frames: torch.Tensor,
        correct_actions: torch.Tensor, capacity: int,
        device: torch.device, record_access: bool = False,
        ) -> tuple[torch.Tensor, list[torch.Tensor]]:
    banks = len(memories)
    reads = []
    receipts = []
    for bank, memory in enumerate(memories):
        query = keys[bank * capacity:(bank + 1) * capacity]
        read, _, receipt = memory.retrieve_with_receipt(
            query, top_k=1, confidence_mode="cosine",
            record_access=record_access, usage_prior_scale=0.0)
        reads.append(read)
        receipts.append(receipt)
    recalled = torch.cat(reads)
    state = model.initial_state(frames.shape[0], device=device)
    null = torch.full(
        (frames.shape[0],), 2, dtype=torch.long, device=device)
    zeros = torch.zeros(frames.shape[0], device=device)
    output, _ = model.step(
        frames, state, null, zeros, zeros, retrieved_memory=recalled)
    outcomes = (
        output.logits.argmax(-1) == correct_actions).to(torch.float32)
    return outcomes, receipts


@torch.no_grad()
def _evaluate_replacement(
        model: UnifiedCognitiveController,
        memories: list[DiskLatentMemory],
        old_stream: dict[str, torch.Tensor],
        candidate_stream: dict[str, torch.Tensor],
        actions: torch.Tensor, *, capacity: int,
        device: torch.device, directory: Path,
        ) -> dict[str, object]:
    banks = len(memories)
    old_outcomes = []
    candidate_outcomes = []
    all_outcomes = []
    bounded = True
    roundtrip_exact = True
    rewrites = []
    for bank, source in enumerate(memories):
        memory = source.clone()
        before_capacity = memory.store.capacity
        slot = int(actions[bank])
        rewrite = memory.elastic_replace(
            slot, candidate_stream["store_keys"][bank],
            candidate_stream["store_values"][bank],
            candidate_stream["write_strengths"][bank],
            minimum_rewrite=0.0)
        rewrites.append(rewrite)
        path = directory / f"bank-{bank:04d}.pt"
        memory.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        bounded &= (
            restored.count == capacity
            and restored.store.capacity == before_capacity == capacity)
        roundtrip_exact &= bool(
            torch.equal(restored.store.keys, memory.store.keys)
            and torch.equal(
                restored.store.volatility, memory.store.volatility))
        old_keys = old_stream["query_keys"][
            bank * capacity:(bank + 1) * capacity]
        old_frames = old_stream["query_frames"][
            bank * capacity:(bank + 1) * capacity]
        old_correct = old_stream["correct_actions"][
            bank * capacity:(bank + 1) * capacity]
        new_keys = candidate_stream["query_keys"][bank:bank + 1]
        new_frames = candidate_stream["query_frames"][bank:bank + 1]
        new_correct = candidate_stream["correct_actions"][bank:bank + 1]
        keys = torch.cat((old_keys, new_keys))
        frames = torch.cat((old_frames, new_frames))
        correct = torch.cat((old_correct, new_correct))
        reads = []
        for query in keys.split(capacity):
            read, _, _ = restored.retrieve_with_receipt(
                query, top_k=1, confidence_mode="cosine",
                usage_prior_scale=0.0)
            reads.append(read)
        recalled = torch.cat(reads)
        state = model.initial_state(capacity + 1, device=device)
        null = torch.full(
            (capacity + 1,), 2, dtype=torch.long, device=device)
        zeros = torch.zeros(capacity + 1, device=device)
        output, _ = model.step(
            frames, state, null, zeros, zeros, retrieved_memory=recalled)
        outcomes = (
            output.logits.argmax(-1) == correct).to(torch.float32)
        old_outcomes.append(outcomes[:capacity])
        candidate_outcomes.append(outcomes[capacity:])
        all_outcomes.append(outcomes)
    all_old = torch.cat(old_outcomes)
    all_candidate = torch.cat(candidate_outcomes)
    all_values = torch.cat(all_outcomes)
    return {
        "old_accuracy": float(all_old.mean()),
        "new_accuracy": float(all_candidate.mean()),
        "composite_accuracy": float(all_values.mean()),
        "replacement_rewrites": rewrites,
        "bounded": bounded,
        "disk_roundtrip_exact": roundtrip_exact,
    }


@torch.no_grad()
def _ensure_failed_decoys(
        model: UnifiedCognitiveController,
        memories: list[DiskLatentMemory],
        stream: dict[str, torch.Tensor], *, capacity: int,
        device: torch.device) -> list[str]:
    """Make the verifier-side decoy genuinely fail without changing keys."""
    choices = ("negate", "zero", "flip", "roll")
    selected: list[str] = []
    for bank, memory in enumerate(memories):
        start = bank * capacity
        stop = start + capacity
        base = memory.store.values[0].clone()
        chosen = "none"
        for name in choices:
            candidate = base.clone()
            if name == "negate":
                candidate.mul_(-1.0)
            elif name == "zero":
                candidate.zero_()
            elif name == "flip":
                candidate = candidate.flip(0)
            else:
                candidate = candidate.roll(1, dims=0)
            probe = memory.clone()
            probe.store.values[0].copy_(candidate)
            outcomes, receipts = _read_actions(
                model, [probe], stream,
                keys=stream["query_keys"][start:stop],
                frames=stream["query_frames"][start:stop],
                correct_actions=stream["correct_actions"][start:stop],
                capacity=capacity, device=device)
            if (
                    outcomes[0] <= 0.5
                    and int(receipts[0][0]) == 0):
                memory.store.values[0].copy_(candidate)
                chosen = name
                break
        selected.append(chosen)
    return selected


def run_probe(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, habit_rounds: int = 8,
        ) -> dict[str, object]:
    if banks < 2 or banks % 2 or capacity < 2 or capacity % 2:
        raise ValueError("banks and capacity must be positive even values")
    started = time.perf_counter()
    _, old_stream = _controller_stream(
        model, banks * capacity, seed=seed, device=device)
    _, candidate_stream = _controller_stream(
        model, banks, seed=seed + 77_777, device=device,
        query_from_support=True)
    base_memories = _make_memories(
        old_stream, banks=banks, capacity=capacity,
        width=model.width, device=device)
    shift_memories = [memory.clone() for memory in base_memories]
    # Make the first row a genuine failed decoy in the physical store.  This
    # is a renderer-independent adversarial control, not a learner-visible
    # semantic label.  The transform is selected only by the verifier's
    # observed outcome, so every bank has a real failure rather than a merely
    # perturbed row.
    decoy_transforms = _ensure_failed_decoys(
        model, shift_memories, old_stream, capacity=capacity, device=device)
    old_keys = old_stream["query_keys"]
    old_frames = old_stream["query_frames"]
    old_correct = old_stream["correct_actions"]
    decoy_outcomes, decoy_receipts = _read_actions(
        model, shift_memories, old_stream, keys=old_keys,
        frames=old_frames, correct_actions=old_correct,
        capacity=capacity, device=device, record_access=True)
    for _ in range(habit_rounds):
        for bank, memory in enumerate(shift_memories):
            start = bank * capacity
            memory.record_outcomes_from_receipts(
                decoy_receipts[bank], decoy_outcomes[start:start + capacity],
                update_volatility=True, success_protection_rate=0.20,
                failure_thaw_rate=0.25, stale_thaw_rate=0.0)
    volatility = torch.stack([
        memory.store.volatility[:capacity] for memory in shift_memories])
    high_actions = volatility.argmax(-1)
    low_actions = volatility.argmin(-1)
    shuffled_actions = volatility.roll(1, dims=1).argmax(-1)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        high = _evaluate_replacement(
            model, shift_memories, old_stream, candidate_stream,
            high_actions, capacity=capacity, device=device,
            directory=root / "high")
        low = _evaluate_replacement(
            model, shift_memories, old_stream, candidate_stream,
            low_actions, capacity=capacity, device=device,
            directory=root / "low")
        shuffled = _evaluate_replacement(
            model, shift_memories, old_stream, candidate_stream,
            shuffled_actions, capacity=capacity, device=device,
            directory=root / "shuffled")
    decoy_failures = int((decoy_outcomes <= 0.5).sum())
    decoy_successes = int((decoy_outcomes > 0.5).sum())
    report: dict[str, object] = {
        "schema": "unified-controller-online-disk-task-shift-v1",
        "configuration": {
            "banks": banks, "capacity": capacity,
            "habit_rounds": habit_rounds, "seed": seed,
            "device": str(device),
        },
        "semantic_or_task_labels_used_for_controller_or_memory": False,
        "learner_visible": [
            "controller latent keys and values", "write strengths",
            "physical receipts", "binary verifier outcomes", "volatility",
        ],
        "decoy_outcomes": {
            "successes": decoy_successes, "failures": decoy_failures,
            "transforms": decoy_transforms,
        },
        "volatility": {
            "high_action_rows": high_actions.tolist(),
            "low_action_rows": low_actions.tolist(),
            "high_action_is_first_decoy_rate": float(
                (high_actions == 0).float().mean()),
            "mean": float(volatility.mean()),
        },
        "policies": {"high_volatility": high, "low_volatility": low,
                     "shuffled_volatility": shuffled},
        "accounting": {
            "controller_writes": banks * capacity + banks,
            "physical_reads": banks * capacity * habit_rounds
            + banks * (capacity + 1) * 3,
            "verifier_outcome_updates": banks * capacity * habit_rounds,
            "total_seconds": time.perf_counter() - started,
        },
    }
    gates = {
        "decoy_has_failures": decoy_failures > 0,
        "decoy_has_successes": decoy_successes > 0,
        "habit_selects_decoy": (
            report["volatility"]["high_action_is_first_decoy_rate"] >= 0.75),
        "new_row_acquired": high["new_accuracy"] >= 0.85,
        "old_rows_retained": high["old_accuracy"] >= 0.70,
        "high_beats_low_composite": (
            high["composite_accuracy"]
            >= low["composite_accuracy"] + 0.10),
        "shuffled_volatility_costs_five_points": (
            high["composite_accuracy"]
            >= shuffled["composite_accuracy"] + 0.05),
        "all_policies_bounded": (
            high["bounded"] and low["bounded"] and shuffled["bounded"]),
        "all_disk_roundtrips_exact": (
            high["disk_roundtrip_exact"]
            and low["disk_roundtrip_exact"]
            and shuffled["disk_roundtrip_exact"]),
        "under_one_minute": report["accounting"]["total_seconds"] <= 60.0,
    }
    report["gates"] = gates | {"accepted": all(gates.values())}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7351)
    parser.add_argument("--banks", type=int, default=8)
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
