"""Reward-train recurring-context recall through external latent memory."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .environment import NULL_ACTION, generate_lifetimes
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import (
    attempted_success_loss, evaluate, rollout, seed_everything)


def _choose(
        logits: torch.Tensor, *, sample: bool,
        exploration: float) -> torch.Tensor:
    if not sample:
        return logits.argmax(-1)
    probabilities = torch.softmax(logits, dim=-1)
    behavior = probabilities * (1.0 - exploration) + exploration / 2
    return torch.multinomial(behavior, 1).squeeze(1)


def _grouped_read(
        queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, *,
        capacity: int, mode: str, temperature: float = 10.0
        ) -> tuple[torch.Tensor, torch.Tensor]:
    groups = queries.shape[0] // capacity
    query_group = nn.functional.normalize(
        queries.reshape(groups, capacity, -1), dim=-1)
    key_group = nn.functional.normalize(
        keys.reshape(groups, capacity, -1), dim=-1)
    value_group = values.reshape(groups, capacity, -1)
    similarity = torch.einsum(
        "gcw,gkw->gck", query_group, key_group)
    top1 = similarity.argmax(-1)
    if mode == "none":
        read = torch.zeros_like(query_group)
    elif mode == "shuffled":
        read = value_group.roll(1, dims=1)
    elif mode == "corrupted":
        read = value_group.flip(dims=(-1,))
    elif mode == "hard":
        read = torch.gather(
            value_group, 1,
            top1.unsqueeze(-1).expand(-1, -1, value_group.shape[-1]))
    elif mode == "soft":
        weights = torch.softmax(similarity * temperature, dim=-1)
        read = torch.einsum("gck,gkw->gcw", weights, value_group)
    else:
        raise ValueError("unsupported memory mode")
    return read.reshape_as(queries), top1.reshape(-1)


def persistent_rollout(
        model: UnifiedCognitiveController, *, count: int, capacity: int,
        seed: int, device: torch.device, sample_actions: bool,
        exploration: float = 0.10, memory_mode: str = "soft",
        reverse_rules: bool = False,
        memory_temperature: float = 10.0) -> dict[str, torch.Tensor]:
    if count % capacity:
        raise ValueError("count must be divisible by memory capacity")
    batch = _add_context_signatures(
        generate_lifetimes(
            count, 3, seed=seed, heldout=True,
            reverse_rules=reverse_rules, task="binary_mapping",
            support_trials=1, device=device),
        seed=seed + 10_000_000)
    state = model.initial_state(count, device=device)
    null_action = torch.full(
        (count,), NULL_ACTION, dtype=torch.long, device=device)
    zeros = torch.zeros(count, device=device)

    support0, state = model.step(
        batch.frames[:, 0], state, null_action, zeros, zeros)
    support_action = _choose(
        support0.logits, sample=sample_actions, exploration=exploration)
    support_outcome = (
        support_action == batch.correct_actions[:, 0]).to(torch.float32)
    support1, _ = model.step(
        batch.frames[:, 1], state, support_action, support_outcome,
        torch.ones_like(support_outcome))

    fresh = model.initial_state(count, device=device)
    query0, _ = model.step(
        batch.frames[:, 2], fresh, null_action, zeros, zeros)
    recalled, selected = _grouped_read(
        query0.memory_key, support0.memory_key,
        support1.memory_value, capacity=capacity, mode=memory_mode,
        temperature=memory_temperature)
    query, _ = model.step(
        batch.frames[:, 2], fresh, null_action, zeros, zeros,
        retrieved_memory=recalled)
    action = _choose(
        query.logits, sample=sample_actions, exploration=exploration)
    outcome = (
        action == batch.correct_actions[:, 2]).to(torch.float32)
    target = torch.arange(count, device=device) % capacity
    return {
        "logits": query.logits,
        "actions": action,
        "outcomes": outcome,
        "correct_actions": batch.correct_actions[:, 2],
        "retrieval_top1": (selected == target).to(torch.float32),
        "store_keys": support0.memory_key,
        "store_values": support1.memory_value,
        "query_keys": query0.memory_key,
        "write_strengths": support1.memory_write_strength,
        "query_frames": batch.frames[:, 2],
    }


@torch.no_grad()
def evaluate_persistent(
        model: UnifiedCognitiveController, *, count: int, capacity: int,
        seed: int, device: torch.device) -> dict[str, object]:
    model.eval()
    normal = persistent_rollout(
        model, count=count, capacity=capacity, seed=seed, device=device,
        sample_actions=False, memory_mode="hard")
    reversed_result = persistent_rollout(
        model, count=count, capacity=capacity, seed=seed, device=device,
        sample_actions=False, memory_mode="hard", reverse_rules=True)
    no_memory = persistent_rollout(
        model, count=count, capacity=capacity, seed=seed, device=device,
        sample_actions=False, memory_mode="none")
    shuffled = persistent_rollout(
        model, count=count, capacity=capacity, seed=seed, device=device,
        sample_actions=False, memory_mode="shuffled")
    corrupted = persistent_rollout(
        model, count=count, capacity=capacity, seed=seed, device=device,
        sample_actions=False, memory_mode="corrupted")
    normal_accuracy = float(normal["outcomes"].mean())
    reversed_accuracy = float(reversed_result["outcomes"].mean())
    no_memory_accuracy = float(no_memory["outcomes"].mean())
    shuffled_accuracy = float(shuffled["outcomes"].mean())
    corrupted_accuracy = float(corrupted["outcomes"].mean())
    flip_rate = float(
        (normal["actions"] != reversed_result["actions"]).float().mean())
    retrieval_accuracy = float(normal["retrieval_top1"].mean())
    report = {
        "normal_accuracy": normal_accuracy,
        "reversed_rule_accuracy": reversed_accuracy,
        "paired_prediction_flip_rate": flip_rate,
        "retrieval_top1": retrieval_accuracy,
        "no_memory_accuracy": no_memory_accuracy,
        "shuffled_memory_accuracy": shuffled_accuracy,
        "corrupted_memory_accuracy": corrupted_accuracy,
        "write_strength_mean": float(
            normal["write_strengths"].mean()),
    }
    report["gate"] = {
        "normal_at_least_85": normal_accuracy >= 0.85,
        "reversed_at_least_85": reversed_accuracy >= 0.85,
        "prediction_flips_at_least_80": flip_rate >= 0.80,
        "no_memory_hurts": no_memory_accuracy <= normal_accuracy - 0.15,
        "shuffled_memory_hurts":
            shuffled_accuracy <= normal_accuracy - 0.15,
        "corrupted_memory_hurts":
            corrupted_accuracy <= normal_accuracy - 0.15,
    }
    report["gate"]["accepted"] = all(report["gate"].values())

    # Exercise the actual serializable disk implementation on one memory bank.
    disk = DiskLatentMemory(
        width=model.width, capacity=capacity, device=device)
    disk.commit(
        normal["store_keys"][:capacity],
        normal["store_values"][:capacity],
        torch.ones(capacity, device=device), threshold=0.0)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "memory.pt"
        disk.save(path)
        restored = DiskLatentMemory.load(path, device=device)
        disk_read, _ = restored.retrieve(
            normal["query_keys"][:capacity], top_k=1)
    hard_read, _ = _grouped_read(
        normal["query_keys"][:capacity],
        normal["store_keys"][:capacity],
        normal["store_values"][:capacity],
        capacity=capacity, mode="hard")
    report["disk_roundtrip"] = {
        "rows": restored.count,
        "read_matches_hard_memory": bool(
            torch.allclose(disk_read, hard_read)),
    }
    return report


def _rehearsal_loss(
        model: UnifiedCognitiveController, *, task: str,
        feedback_trials: int, count: int, seed: int,
        device: torch.device, exploration: float,
        appearance: str = "bars") -> torch.Tensor:
    batch = generate_lifetimes(
        count, 6, seed=seed, task=task,
        support_trials=feedback_trials, appearance=appearance,
        device=device)
    result = rollout(
        model, batch, sample_actions=True, exploration=exploration,
        feedback_trials=feedback_trials)
    losses = []
    for trial in range(batch.trials):
        trial_loss = attempted_success_loss(
            result["logits"][:, trial], result["actions"][:, trial],
            result["rewards"][:, trial])
        losses.append(trial_loss * (0.2 if trial == 0 else 1.0))
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=4001)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--memory-capacity", type=int, default=8)
    parser.add_argument("--test-contexts", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--exploration", type=float, default=0.10)
    parser.add_argument(
        "--memory-temperature", type=float, default=10.0,
        help="soft-memory similarity temperature used during training")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--persistent-updates-per-cycle", type=int, default=1,
        help=(
            "new-memory updates followed by one binary and one four-rule "
            "rehearsal update"))
    parser.add_argument(
        "--relation-rehearsal", action="store_true",
        help=(
            "add one pair-relation rehearsal update per cycle and require "
            "bars/diamonds/dot-pairs retention before saving"))
    args = parser.parse_args()
    if args.batch_size % args.memory_capacity:
        raise ValueError("batch size must divide into complete memory banks")
    if args.test_contexts % args.memory_capacity:
        raise ValueError("test contexts must divide into memory banks")
    if args.persistent_updates_per_cycle < 1:
        raise ValueError("persistent updates per cycle must be positive")
    if args.memory_temperature <= 0:
        raise ValueError("memory temperature must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model = UnifiedCognitiveController(
        **payload["model_configuration"]).to(device)
    model.load_state_dict(payload["state_dict"])
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    history = []
    persistent_steps = 0
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        data_seed = args.seed * 1_000_000 + step
        cycle = args.persistent_updates_per_cycle + 2 + int(
            args.relation_rehearsal)
        slot = (step - 1) % cycle
        if slot < args.persistent_updates_per_cycle:
            task = "persistent_recall"
            result = persistent_rollout(
                model, count=args.batch_size,
                capacity=args.memory_capacity, seed=data_seed,
                device=device, sample_actions=True,
                exploration=args.exploration, memory_mode="soft",
                memory_temperature=args.memory_temperature)
            loss = attempted_success_loss(
                result["logits"], result["actions"], result["outcomes"])
            accuracy = float(result["outcomes"].mean())
            retrieval = float(result["retrieval_top1"].mean())
            persistent_steps += 1
        elif slot == args.persistent_updates_per_cycle:
            task = "binary_mapping_rehearsal"
            loss = _rehearsal_loss(
                model, task="binary_mapping", feedback_trials=1,
                count=args.batch_size, seed=data_seed, device=device,
                exploration=args.exploration)
            accuracy = None
            retrieval = None
        elif slot == args.persistent_updates_per_cycle + 1:
            task = "four_rule_rehearsal"
            loss = _rehearsal_loss(
                model, task="four_rule", feedback_trials=2,
                count=args.batch_size, seed=data_seed, device=device,
                exploration=args.exploration)
            accuracy = None
            retrieval = None
        else:
            task = "pair_relation_rehearsal"
            # Alternate appearances without a learner-visible task or renderer
            # ID.  This preserves the abstract relation rather than one contour.
            appearance = ("bars", "diamonds", "dot_pairs")[step % 3]
            loss = _rehearsal_loss(
                model, task="pair_relation", feedback_trials=1,
                count=args.batch_size, seed=data_seed, device=device,
                exploration=args.exploration, appearance=appearance)
            accuracy = None
            retrieval = None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            entry = {
                "step": step,
                "task": task,
                "loss": float(loss.detach()),
                "persistent_accuracy": accuracy,
                "retrieval_top1": retrieval,
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(entry)
            print(json.dumps(entry, sort_keys=True), flush=True)

    persistent_evaluation = evaluate_persistent(
        model, count=args.test_contexts, capacity=args.memory_capacity,
        seed=args.seed + 90_000_000, device=device)
    binary_retention = evaluate(
        model, count=1024, trials=6, seed=args.seed + 91_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule_retention = evaluate(
        model, count=1024, trials=6, seed=args.seed + 92_000_000,
        device=device, task="four_rule", feedback_trials=2)
    relation_retention = {}
    if args.relation_rehearsal:
        for index, appearance in enumerate(("bars", "diamonds", "dot_pairs")):
            relation_retention[appearance] = evaluate(
                model, count=1024, trials=6,
                seed=args.seed + 93_000_000 + index * 10_000,
                device=device, task="pair_relation", feedback_trials=1,
                appearance=appearance)
    relation_retained = (
        not args.relation_rehearsal
        or all(
            result["gate"]["accepted"]
            for result in relation_retention.values()))
    admitted = (
        persistent_evaluation["gate"]["accepted"]
        and binary_retention["gate"]["accepted"]
        and four_rule_retention["gate"]["accepted"]
        and relation_retained
        and persistent_evaluation["disk_roundtrip"][
            "read_matches_hard_memory"])
    report = {
        "schema": "unified-controller-persistent-recall-v1",
        "learner_visible": [
            "rendered_rgb_frame", "own_previous_opaque_action",
            "scalar_verified_outcome", "own_latent_active_state",
            "content_addressed_external_latent_read",
        ],
        "semantic_labels_used_for_training": False,
        "unattempted_action_labels_used_for_training": False,
        "within_lifetime_weight_updates": False,
        "memory_group_ids_visible_to_learner": False,
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "unique_persistent_contexts_seen":
            persistent_steps * args.batch_size,
        "history": history,
        "persistent_evaluation": persistent_evaluation,
        "binary_retention": binary_retention,
        "four_rule_retention": four_rule_retention,
        "relation_retention": relation_retention,
        "all_admission_gates_passed": admitted,
        "weights_changed": any(
            not torch.equal(initial[name], value.detach().cpu())
            for name, value in model.state_dict().items()),
        "total_seconds": time.perf_counter() - started,
    }
    if admitted and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": payload["model_configuration"],
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(
        report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "checkpoint_saved": report["checkpoint_saved"],
        "persistent_evaluation": persistent_evaluation,
        "binary_retained": binary_retention["gate"]["accepted"],
        "four_rule_retained": four_rule_retention["gate"]["accepted"],
        "relation_retained": relation_retained,
        "total_seconds": report["total_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
