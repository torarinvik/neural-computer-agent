"""Train one zero-initialized volatility gate on physical task-shift reward.

This is the smallest trainable extension after the frozen task-shift control.
The controller body, memory keys/values, receipt attribution, and disk stores
remain fixed.  Only the third coefficient of the expanded replacement gate is
updated.  Each update is a symmetric ``alpha + delta`` versus ``alpha -
delta`` horse race scored by physical old-plus-new verifier outcomes; no task
or row labels are supplied to the learner.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch

from .legacy_model import UnifiedCognitiveController
from .probe_online_disk_habit import _controller_stream, _make_memories
from .probe_online_disk_task_shift import (
    _ensure_failed_decoys,
    _evaluate_replacement,
    _read_actions,
)
from .train import evaluate, seed_everything
from .train_controller_memory_volatility import expand_with_volatility


def _reward_direction(
        plus_reward: float, minus_reward: float,
        tie_margin: float = 1e-8) -> float:
    """Return a direction only when the verifier gives a real comparison."""
    if tie_margin < 0.0:
        raise ValueError("tie margin must be non-negative")
    difference = plus_reward - minus_reward
    if abs(difference) <= tie_margin:
        return 0.0
    return 1.0 if difference > 0.0 else -1.0


def _task_shift_batch(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, habit_rounds: int,
        shuffle_receipts: bool = False,
        ) -> dict[str, object]:
    _, old_stream = _controller_stream(
        model, banks * capacity, seed=seed, device=device)
    _, candidate_stream = _controller_stream(
        model, banks, seed=seed + 77_777, device=device,
        query_from_support=True)
    memories = _make_memories(
        old_stream, banks=banks, capacity=capacity,
        width=model.width, device=device)
    # Preserve a generic age feature as the cold/reset baseline, but randomize
    # insertion order so the zero-volatility policy cannot win by always
    # selecting the intentionally failed row at slot zero.
    age_generator = torch.Generator(device=device).manual_seed(seed + 8_100_000)
    for memory in memories:
        memory.store.age[:capacity] = (
            torch.randperm(
                capacity, generator=age_generator, device=device,
                dtype=torch.long) + 1)
    transforms = _ensure_failed_decoys(
        model, memories, old_stream, capacity=capacity, device=device)
    outcomes, receipts = _read_actions(
        model, memories, old_stream,
        keys=old_stream["query_keys"],
        frames=old_stream["query_frames"],
        correct_actions=old_stream["correct_actions"],
        capacity=capacity, device=device, record_access=True)
    for _ in range(habit_rounds):
        for bank, memory in enumerate(memories):
            local_receipts = receipts[bank]
            if shuffle_receipts:
                local_receipts = local_receipts.roll(1)
            start = bank * capacity
            memory.record_outcomes_from_receipts(
                local_receipts, outcomes[start:start + capacity],
                update_volatility=True, success_protection_rate=0.20,
                failure_thaw_rate=0.25, stale_thaw_rate=0.0)
    volatility = torch.stack([
        memory.store.volatility[:capacity] for memory in memories])
    options = torch.zeros(
        banks, capacity + 1, model.adaptive_memory_replace_features,
        device=device)
    options[:, 1:, 0] = torch.stack([
        memory.store.age[:capacity].to(torch.float32) / capacity
        for memory in memories])
    options[:, 1:, 7] = volatility
    return {
        "memories": memories,
        "old_stream": old_stream,
        "candidate_stream": candidate_stream,
        "volatility": volatility,
        "options": options,
        "decoy_transforms": transforms,
        "decoy_outcomes": outcomes,
    }


@torch.no_grad()
def _gate_actions(
        model: UnifiedCognitiveController, options: torch.Tensor,
        alpha: float) -> torch.Tensor:
    extra = model.memory_replacement_extra_gate.weight
    extra[:, 2].fill_(alpha)
    # This rung requires admission; compare only bounded replacement rows.
    # The skip decision remains a separate later gate, avoiding a confounded
    # two-decision experiment.
    return model.memory_replacement_scores(options)[:, 1:].argmax(-1)


@torch.no_grad()
def _score(
        model: UnifiedCognitiveController, data: dict[str, object],
        alpha: float, *, device: torch.device, directory: Path,
        ) -> dict[str, object]:
    actions = _gate_actions(model, data["options"], alpha)
    result = _evaluate_replacement(
        model, data["memories"], data["old_stream"],
        data["candidate_stream"], actions,
        capacity=data["options"].shape[1] - 1,
        device=device, directory=directory)
    result["actions"] = actions.detach().cpu().tolist()
    return result


def _train(
        model: UnifiedCognitiveController, *, steps: int, banks: int,
        capacity: int, seed: int, device: torch.device,
        habit_rounds: int, learning_delta: float, step_size: float,
        shuffle_rewards: bool, shuffle_receipts: bool, tie_margin: float,
        ) -> tuple[float, list[dict[str, object]], int, float]:
    extra = model.memory_replacement_extra_gate.weight
    alpha = 0.0
    history = []
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(seed + 91_000_000)
    for step in range(1, steps + 1):
        data = _task_shift_batch(
            model, banks=banks, capacity=capacity,
            seed=seed * 1_000_000 + step, device=device,
            habit_rounds=habit_rounds,
            shuffle_receipts=shuffle_receipts)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plus = _score(
                model, data, alpha + learning_delta,
                device=device, directory=root / "plus")
            minus = _score(
                model, data, alpha - learning_delta,
                device=device, directory=root / "minus")
        plus_reward = float(plus["composite_accuracy"])
        minus_reward = float(minus["composite_accuracy"])
        if shuffle_rewards:
            if bool(torch.randint(
                    0, 2, (), generator=generator, device=device)):
                plus_reward, minus_reward = minus_reward, plus_reward
        # A tied verifier score contains no directional information.  The
        # helper keeps ties neutral for shuffled-reward controls and prevents
        # a deterministic argmax plateau from masquerading as learning.
        direction = _reward_direction(
            plus_reward, minus_reward, tie_margin)
        alpha += step_size * direction
        with torch.no_grad():
            extra[:, 2].fill_(alpha)
        history.append({
            "step": step,
            "alpha": alpha,
            "plus_reward": plus_reward,
            "minus_reward": minus_reward,
            "direction": direction,
            "decoy_failures": int(
                (data["decoy_outcomes"] <= 0.5).sum()),
            "elapsed_seconds": time.perf_counter() - started,
        })
    return alpha, history, steps * banks * capacity, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=18401)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-banks", type=int, default=8)
    parser.add_argument("--test-banks", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument("--habit-rounds", type=int, default=8)
    parser.add_argument("--learning-delta", type=float, default=1.0)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument(
        "--tie-margin", type=float, default=1e-8,
        help="leave the coefficient unchanged when the two verifier scores tie")
    parser.add_argument("--shuffle-rewards", action="store_true")
    parser.add_argument("--shuffle-receipts", action="store_true")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_banks < 2 or args.batch_banks % 2:
        raise ValueError("steps positive and batch banks must be even >= 2")
    if args.test_banks < 2 or args.test_banks % 2:
        raise ValueError("test banks must be even >= 2")
    if args.capacity < 2 or args.capacity % 2:
        raise ValueError("capacity must be even >= 2")
    seed_everything(args.seed)
    device = torch.device(args.device)
    started = time.perf_counter()
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model, configuration = expand_with_volatility(payload, device=device)
    model.eval()
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    extra = model.memory_replacement_extra_gate.weight
    inherited_columns = extra[:, :2].detach().clone()
    alpha, history, verifier_bits, training_seconds = _train(
        model, steps=args.steps, banks=args.batch_banks,
        capacity=args.capacity, seed=args.seed, device=device,
        habit_rounds=args.habit_rounds, learning_delta=args.learning_delta,
        step_size=args.step_size, shuffle_rewards=args.shuffle_rewards,
        shuffle_receipts=args.shuffle_receipts, tie_margin=args.tie_margin)
    with torch.no_grad():
        extra[:, 2].fill_(alpha)
    def evaluate_arm(seed: int, *, cross_context: bool = False):
        data = _task_shift_batch(
            model, banks=args.test_banks, capacity=args.capacity,
            seed=seed, device=device, habit_rounds=args.habit_rounds,
            shuffle_receipts=False)
        if cross_context:
            _, candidate = _controller_stream(
                model, args.test_banks, seed=seed + 77_777,
                device=device, query_from_support=False)
            data["candidate_stream"] = candidate
        with tempfile.TemporaryDirectory() as directory:
            learned = _score(
                model, data, alpha, device=device,
                directory=Path(directory) / "learned")
            reset = _score(
                model, data, 0.0, device=device,
                directory=Path(directory) / "reset")
        return learned, reset
    learned, reset = evaluate_arm(args.seed + 90_000_000)
    cross_context, cross_reset = evaluate_arm(
        args.seed + 91_000_000, cross_context=True)
    cross_shuffled_data = _task_shift_batch(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 95_000_000, device=device,
        habit_rounds=args.habit_rounds, shuffle_receipts=True)
    _, cross_candidate = _controller_stream(
        model, args.test_banks, seed=args.seed + 95_000_000 + 77_777,
        device=device, query_from_support=False)
    cross_shuffled_data["candidate_stream"] = cross_candidate
    with tempfile.TemporaryDirectory() as directory:
        cross_shuffled = _score(
            model, cross_shuffled_data, alpha, device=device,
            directory=Path(directory) / "cross-shuffled")
    shuffled_data = _task_shift_batch(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 92_000_000, device=device,
        habit_rounds=args.habit_rounds, shuffle_receipts=True)
    with tempfile.TemporaryDirectory() as directory:
        shuffled = _score(
            model, shuffled_data, alpha, device=device,
            directory=Path(directory) / "shuffled")
    binary = evaluate(
        model, count=128, trials=6, seed=args.seed + 93_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=128, trials=6, seed=args.seed + 94_000_000,
        device=device, task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu())]
    first_two_unchanged = torch.equal(
        extra[:, :2].detach().cpu(), inherited_columns.cpu())
    report = {
        "schema": "unified-controller-online-task-shift-gate-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out else None),
            "device": str(device),
        },
        "semantic_or_task_labels_used_for_training": False,
        "training_signal": "physical old-plus-new verifier reward",
        "history": history,
        "learned_alpha": alpha,
        "adjacent": {"learned": learned, "reset": reset},
        "cross_context": {"learned": cross_context, "reset": cross_reset},
        "cross_context_shuffled_receipt": cross_shuffled,
        "shuffled_receipt": shuffled,
        "retention": {"binary_mapping": binary, "four_rule": four_rule},
        "accounting": {
            "physical_verifier_bits": verifier_bits,
            "optimizer_updates": args.steps,
            "training_seconds": training_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "changed_parameters": changed,
        "gates": {
            "alpha_moves_positive": alpha > 0.0,
            "adjacent_new_row_at_least_85":
                learned["new_accuracy"] >= 0.85,
            "adjacent_old_retained_at_least_70":
                learned["old_accuracy"] >= 0.70,
            "learned_beats_reset_by_10_points":
                learned["composite_accuracy"]
                >= reset["composite_accuracy"] + 0.10,
            "receipt_shuffle_costs_five_points":
                learned["composite_accuracy"]
                >= shuffled["composite_accuracy"] + 0.05,
            "cross_context_reported": True,
            "cross_context_new_row_at_least_85": (
                cross_context["new_accuracy"] >= 0.85),
            "cross_context_beats_reset_by_five_points": (
                cross_context["composite_accuracy"]
                >= cross_reset["composite_accuracy"] + 0.05),
            "cross_context_receipt_shuffle_costs_five_points": (
                cross_context["composite_accuracy"]
                >= cross_shuffled["composite_accuracy"] + 0.05),
            "cross_context_bounded_and_exact": (
                cross_context["bounded"]
                and cross_context["disk_roundtrip_exact"]),
            "binary_retained": binary["gate"]["accepted"],
            "four_rule_retained": four_rule["gate"]["accepted"],
            "only_volatility_column_changed": (
                changed == ["memory_replacement_extra_gate.weight"]
                and first_two_unchanged),
            "under_five_minutes": training_seconds <= 300.0,
        },
    }
    report["gates"]["accepted"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if report["gates"]["accepted"] and args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
    print(json.dumps({
        "history": history,
        "learned_alpha": alpha,
        "adjacent": report["adjacent"],
        "cross_context": report["cross_context"],
        "cross_context_shuffled_receipt": report[
            "cross_context_shuffled_receipt"],
        "shuffled_receipt": shuffled,
        "gates": report["gates"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
