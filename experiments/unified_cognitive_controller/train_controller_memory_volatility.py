"""Train the unified visual controller to use verified-use row volatility.

Stable and decoy rows have the same access count and the same aggregate
success/failure counts. Only the temporal order of scalar verifier outcomes
differs, so the inherited recency/frequency/reliability features cannot solve
the task. The eighth generic replacement feature is zero-initialized and is
the only trainable controller coefficient.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .model import UnifiedCognitiveController
from .train import evaluate, seed_everything
from .train_frequency_recency_replacement import (
    evaluate_frequency_recency,
    frequency_recency_batch,
)
from .train_memory_replacement import (
    _bank_reward,
    _gather_rows,
    _select_batch,
)


def outcome_order_volatility(
        outcomes: torch.Tensor, *, success_rate: float = 0.2,
        failure_rate: float = 0.25) -> torch.Tensor:
    """Apply the same generic verified-outcome update used by physical rows."""
    volatility = torch.ones(
        outcomes.shape[:-1], device=outcomes.device, dtype=torch.float32)
    for event in outcomes.unbind(-1):
        protected = volatility * (1.0 - success_rate)
        thawed = 1.0 - (1.0 - volatility) * (1.0 - failure_rate)
        volatility = torch.where(event > 0.5, protected, thawed)
    return volatility


def expand_with_volatility(
        payload: dict[str, object], *, device: torch.device
        ) -> tuple[UnifiedCognitiveController, dict[str, object]]:
    configuration = dict(payload["model_configuration"])
    old_features = int(configuration["adaptive_memory_replace_features"])
    if old_features != 7:
        raise ValueError("volatility integration requires a seven-feature parent")
    configuration["adaptive_memory_replace_features"] = 8
    model = UnifiedCognitiveController(**configuration).to(device)
    state = {
        name: value.to(device)
        for name, value in payload["state_dict"].items()
        if name != "memory_replacement_extra_gate.weight"
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing != ["memory_replacement_extra_gate.weight"] or unexpected:
        raise ValueError(
            f"unexpected expanded-controller mismatch: {missing=}, "
            f"{unexpected=}")
    old = payload["state_dict"]["memory_replacement_extra_gate.weight"].to(
        device)
    with torch.no_grad():
        model.memory_replacement_extra_gate.weight.zero_()
        model.memory_replacement_extra_gate.weight[:, :old.shape[1]].copy_(old)
    return model, configuration


@torch.no_grad()
def volatility_batch(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float,
        shuffle_volatility: bool = False,
        reverse_histories: bool = False) -> dict[str, object]:
    if capacity < 4 or capacity % 2:
        raise ValueError("capacity must be even and at least four")
    if model.adaptive_memory_replace_features != 8:
        raise ValueError("volatility task requires exactly eight features")
    data = frequency_recency_batch(
        model, banks=banks, capacity=capacity, seed=seed, device=device,
        write_threshold=write_threshold, noise_scale=0.0,
        recency_weight=0.5, frequency_weight=0.5,
        reliability_weight=0.0)
    generator = torch.Generator().manual_seed(seed + 81_000_000)
    order = torch.stack([
        torch.randperm(capacity, generator=generator)
        for _ in range(banks)]).to(device)
    stable_slots = order[:, :capacity // 2]
    decoy_slots = order[:, capacity // 2:]
    stable_mask = torch.zeros(
        banks, capacity, device=device, dtype=torch.bool)
    stable_mask.scatter_(1, stable_slots, True)

    failures_then_successes = torch.tensor(
        [0.0] * 5 + [1.0] * 5, device=device)
    successes_then_failures = 1.0 - failures_then_successes
    if reverse_histories:
        failures_then_successes, successes_then_failures = (
            successes_then_failures, failures_then_successes)
    stable_volatility = outcome_order_volatility(
        failures_then_successes.unsqueeze(0))[0]
    decoy_volatility = outcome_order_volatility(
        successes_then_failures.unsqueeze(0))[0]
    volatility = torch.where(
        stable_mask, stable_volatility, decoy_volatility)
    if shuffle_volatility:
        volatility = torch.gather(volatility, 1, order)

    option_features = data["option_features"].clone()
    # Equal aggregate reliability: five successes and five failures per row.
    option_features[:, 1:, 6] = 0.0
    option_features[:, 1:, 7] = volatility
    option_features[:, 0, 7] = 0.0

    slot_to_logical = data["slot_to_logical"]
    stable_logical = torch.gather(slot_to_logical, 1, stable_slots)
    candidate_logical = torch.full(
        (banks, 1), capacity, device=device, dtype=torch.long)
    future_logical = torch.cat((stable_logical, candidate_logical), dim=1)
    base = (
        torch.arange(banks, device=device).unsqueeze(1)
        * (capacity + 1))
    future_indices = (base + future_logical).reshape(-1)
    result = dict(data)
    result.update({
        "option_features": option_features,
        "future_batch": _select_batch(data["source_batch"], future_indices),
        "future_queries": _gather_rows(data["query_group"], future_logical),
        "stable_mask": stable_mask,
        "row_volatility": volatility,
        "stable_volatility": float(stable_volatility),
        "decoy_volatility": float(decoy_volatility),
        "generated_contexts": int(data["generated_contexts"]),
    })
    return result


@torch.no_grad()
def evaluate_volatility(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device, write_threshold: float,
        shuffle_volatility: bool = False,
        reverse_histories: bool = False) -> dict[str, float]:
    data = volatility_batch(
        model, banks=banks, capacity=capacity, seed=seed, device=device,
        write_threshold=write_threshold,
        shuffle_volatility=shuffle_volatility,
        reverse_histories=reverse_histories)
    actions = model.memory_replacement_scores(
        data["option_features"]).argmax(-1)
    reward = _bank_reward(model, data, actions, device=device)
    oracle_actions = (~data["stable_mask"]).to(
        torch.float32).argmax(-1) + 1
    oracle_reward = _bank_reward(
        model, data, oracle_actions, device=device)
    replacing = actions > 0
    selected_stable = torch.zeros_like(replacing)
    selected_stable[replacing] = data["stable_mask"][
        replacing, actions[replacing] - 1]
    target = replacing & ~selected_stable
    return {
        "accuracy": float(reward.mean()),
        "oracle_accuracy": float(oracle_reward.mean()),
        "valid_replacement_rate": float(target.float().mean()),
        "replace_rate": float(replacing.float().mean()),
        "stable_eviction_rate": float(selected_stable.float().mean()),
        "stable_volatility": float(data["stable_volatility"]),
        "decoy_volatility": float(data["decoy_volatility"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=17100)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-banks", type=int, default=128)
    parser.add_argument("--test-banks", type=int, default=1024)
    parser.add_argument("--retention-count", type=int, default=1024)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--shuffle-rewards", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model, configuration = expand_with_volatility(payload, device=device)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    extra = model.memory_replacement_extra_gate.weight
    extra.requires_grad_(True)
    inherited_columns = extra[:, :2].detach().clone()
    optimizer = torch.optim.Adam([extra], lr=args.learning_rate)

    preflight = evaluate_volatility(
        model, banks=min(args.test_banks, 512), capacity=args.capacity,
        seed=args.seed + 90_000_000, device=device,
        write_threshold=args.write_threshold)
    started = time.perf_counter()
    history = []
    generated_contexts = verifier_bits = 0
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 82_000_000)
    for step in range(1, args.steps + 1):
        data = volatility_batch(
            model, banks=args.batch_banks, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=args.write_threshold)
        generated_contexts += int(data["generated_contexts"])
        logits = model.memory_replacement_scores(data["option_features"])
        distribution = torch.distributions.Categorical(
            logits=logits / args.temperature)
        actions = distribution.sample()
        reward = _bank_reward(model, data, actions, device=device)
        verifier_bits += reward.numel() * (args.capacity // 2 + 1)
        if args.shuffle_rewards:
            reward = reward[torch.randperm(
                reward.numel(), generator=reward_generator, device=device)]
        advantage = reward - reward.mean()
        loss = -(advantage.detach() * distribution.log_prob(actions)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert extra.grad is not None
        extra.grad[:, :2] = 0.0
        optimizer.step()
        with torch.no_grad():
            extra[:, :2].copy_(inherited_columns)
        if step == 1 or step % 8 == 0 or step == args.steps:
            prefix = evaluate_volatility(
                model, banks=min(64, args.test_banks),
                capacity=args.capacity,
                seed=args.seed + 89_000_000, device=device,
                write_threshold=args.write_threshold)
            history.append({
                "step": step,
                "reward": float(reward.mean()),
                "volatility_weight": float(extra[0, 2].detach()),
                "held_out_accuracy": prefix["accuracy"],
                "held_out_valid_replacement_rate":
                    prefix["valid_replacement_rate"],
                "elapsed_seconds": time.perf_counter() - started,
            })

    training_loop_seconds = time.perf_counter() - started
    held_out = evaluate_volatility(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 91_000_000, device=device,
        write_threshold=args.write_threshold)
    shuffled = evaluate_volatility(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 91_000_000, device=device,
        write_threshold=args.write_threshold, shuffle_volatility=True)
    reversed_histories = evaluate_volatility(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 91_000_000, device=device,
        write_threshold=args.write_threshold, reverse_histories=True)
    old_utility = evaluate_frequency_recency(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 92_000_000, device=device,
        write_threshold=args.write_threshold, noise_scale=0.04,
        recency_weight=0.3, frequency_weight=0.3,
        reliability_weight=0.4)
    binary = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 93_000_000,
        device=device, task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 94_000_000,
        device=device, task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu())]
    stable_threshold_update = None
    for index, entry in enumerate(history):
        if (
                entry["held_out_valid_replacement_rate"] >= 0.95
                and all(
                    later["held_out_valid_replacement_rate"] >= 0.95
                    for later in history[index:])):
            stable_threshold_update = entry["step"]
            break
    total_seconds = time.perf_counter() - started
    report = {
        "schema": "unified-controller-memory-volatility-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out) if args.checkpoint_out else None),
            "report": str(args.report),
        },
        "model_configuration": configuration,
        "preflight": preflight,
        "history": history,
        "held_out": held_out,
        "volatility_shuffled": shuffled,
        "outcome_histories_reversed": reversed_histories,
        "retention": {
            "old_utility": old_utility,
            "binary_mapping": binary,
            "four_rule": four_rule,
        },
        "accounting": {
            "unique_logical_contexts": generated_contexts,
            "unique_verifier_bits": verifier_bits,
            "optimizer_updates": args.steps,
            "replayed_examples": 0,
            "training_loop_seconds": training_loop_seconds,
            "total_experiment_seconds": total_seconds,
            "stable_updates_to_95_percent_valid_replacement":
                stable_threshold_update,
            "stable_unique_verifier_bits_to_threshold": (
                stable_threshold_update * args.batch_banks
                * (args.capacity // 2 + 1)
                if stable_threshold_update is not None else None),
        },
        "changed_parameters": changed,
        "gates": {
            "held_out_within_2_points_of_oracle":
                held_out["accuracy"] >= held_out["oracle_accuracy"] - 0.02,
            "valid_replacement_at_least_95_percent":
                held_out["valid_replacement_rate"] >= 0.95,
            "shuffle_costs_at_least_6_points":
                held_out["accuracy"] >= shuffled["accuracy"] + 0.06,
            "shuffle_reduces_valid_replacement_by_30_points":
                held_out["valid_replacement_rate"]
                >= shuffled["valid_replacement_rate"] + 0.30,
            "reversal_flips_behavior":
                reversed_histories["stable_eviction_rate"] >= 0.90,
            "old_utility_retained":
                old_utility["accuracy"] >= 0.90,
            "binary_retained": binary["gate"]["accepted"],
            "four_rule_retained": four_rule["gate"]["accepted"],
            "only_extra_replacement_weight_changed":
                changed == ["memory_replacement_extra_gate.weight"],
            "under_five_minute_cap":
                total_seconds <= 300.0,
        },
    }
    report["gates"]["accepted"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
    print(json.dumps({
        "preflight": preflight,
        "history": history,
        "held_out": held_out,
        "volatility_shuffled": shuffled,
        "outcome_histories_reversed": reversed_histories,
        "accounting": report["accounting"],
        "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
