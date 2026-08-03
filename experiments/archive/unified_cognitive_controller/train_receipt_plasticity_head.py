"""Train one fresh plasticity coefficient from causal read receipts.

The controller body and all inherited replacement features stay frozen.  A
single zero-initialized coefficient learns how receipt-attributed volatility
should affect replacement decisions.  The verifier owns row identities and
outcomes; the learner sees only generic latent/memory features.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .audit_receipt_volatility_controller import (
    _evaluate_policy,
    _physical_volatility,
)
from .legacy_model import UnifiedCognitiveController
from .train import evaluate, seed_everything
from .train_controller_memory_volatility import expand_with_volatility, volatility_batch
from .train_memory_replacement import _bank_reward


@torch.no_grad()
def receipt_batch(
        model: UnifiedCognitiveController, *, banks: int, capacity: int,
        seed: int, device: torch.device,
        ) -> dict[str, object]:
    """Generate a training batch whose volatility came from physical receipts."""
    data = volatility_batch(
        model, banks=banks, capacity=capacity, seed=seed, device=device,
        write_threshold=0.5)
    volatility = _physical_volatility(
        model, data, policy="receipt", device=device)
    options = data["option_features"].clone()
    options[:, 1:, 7] = volatility
    data["option_features"] = options
    data["receipt_volatility"] = volatility
    return data


@torch.no_grad()
def _evaluate_fixed(
        model: UnifiedCognitiveController, data: dict[str, object], *,
        value: float, device: torch.device) -> dict[str, float]:
    options = data["option_features"].clone()
    options[:, 1:, 7] = value
    actions = model.memory_replacement_scores(options).argmax(-1)
    reward = _bank_reward(model, data, actions, device=device)
    replacing = actions > 0
    stable = torch.zeros_like(replacing)
    stable[replacing] = data["stable_mask"][
        replacing, actions[replacing] - 1]
    return {
        "accuracy": float(reward.mean()),
        "stable_eviction_rate": float(stable.float().mean()),
        "replace_rate": float(replacing.float().mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=18100)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-banks", type=int, default=16)
    parser.add_argument("--test-banks", type=int, default=128)
    parser.add_argument("--retention-count", type=int, default=128)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--shuffle-rewards", action="store_true")
    args = parser.parse_args()
    if args.capacity < 4 or args.capacity % 2:
        raise ValueError("capacity must be even and at least four")
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
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 82_000_000)
    started = time.perf_counter()
    history = []
    generated_contexts = verifier_bits = 0
    for step in range(1, args.steps + 1):
        data = receipt_batch(
            model, banks=args.batch_banks, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device)
        generated_contexts += int(data["generated_contexts"])
        verifier_bits += args.batch_banks * args.capacity * 10
        logits = model.memory_replacement_scores(data["option_features"])
        distribution = torch.distributions.Categorical(
            logits=logits / args.temperature)
        actions = distribution.sample()
        reward = _bank_reward(model, data, actions, device=device)
        if args.shuffle_rewards:
            reward = reward[torch.randperm(
                reward.numel(), generator=reward_generator, device=device)]
        advantage = reward - reward.mean()
        loss = -(advantage.detach() * distribution.log_prob(actions)).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert extra.grad is not None
        # The first two extra features are inherited from the seven-feature
        # parent. Only the new volatility column may change.
        extra.grad[:, :2] = 0.0
        optimizer.step()
        with torch.no_grad():
            extra[:, :2].copy_(inherited_columns)
        if step == 1 or step % 4 == 0 or step == args.steps:
            prefix_data = receipt_batch(
                model, banks=min(32, args.test_banks),
                capacity=args.capacity, seed=args.seed + 89_000_000 + step,
                device=device)
            prefix = _evaluate_policy(
                model, prefix_data, policy="receipt", device=device)
            history.append({
                "step": step,
                "loss": float(loss.detach()),
                "reward": float(reward.mean()),
                "volatility_weight": float(extra[0, 2].detach()),
                "receipt_accuracy": prefix["accuracy"],
                "receipt_stable_eviction_rate":
                    prefix["stable_eviction_rate"],
                "elapsed_seconds": time.perf_counter() - started,
            })

    heldout_data = receipt_batch(
        model, banks=args.test_banks, capacity=args.capacity,
        seed=args.seed + 91_000_000, device=device)
    receipt = _evaluate_policy(
        model, heldout_data, policy="receipt", device=device)
    ordinary = _evaluate_policy(
        model, heldout_data, policy="ordinary", device=device)
    shuffled = _evaluate_policy(
        model, heldout_data, policy="shuffled_receipt", device=device)
    fixed_low = _evaluate_fixed(
        model, heldout_data, value=0.0, device=device)
    fixed_high = _evaluate_fixed(
        model, heldout_data, value=1.0, device=device)
    binary = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 93_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 94_000_000, device=device,
        task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu())]
    total_seconds = time.perf_counter() - started
    report = {
        "schema": "unified-controller-receipt-plasticity-head-v1",
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
            "device": str(device),
        },
        "model_configuration": configuration,
        "semantic_or_task_labels_used_for_training": False,
        "training_signal": "verified future outcomes after receipt-attributed histories",
        "history": history,
        "policies": {
            "receipt": receipt,
            "ordinary": ordinary,
            "shuffled_receipt": shuffled,
            "fixed_low": fixed_low,
            "fixed_high": fixed_high,
        },
        "retention": {"binary_mapping": binary, "four_rule": four_rule},
        "accounting": {
            "unique_logical_contexts": generated_contexts,
            "physical_verifier_bits": verifier_bits,
            "optimizer_updates": args.steps,
            "training_seconds": total_seconds,
        },
        "changed_parameters": changed,
        "gates": {
            "receipt_accuracy_at_least_95": receipt["accuracy"] >= 0.95,
            "receipt_stable_eviction_at_most_10_percent":
                receipt["stable_eviction_rate"] <= 0.10,
            "receipt_beats_ordinary":
                receipt["accuracy"] >= ordinary["accuracy"],
            "shuffled_receipt_costs_at_least_6_points":
                receipt["accuracy"] >= shuffled["accuracy"] + 0.06,
            "receipt_beats_fixed_low_or_high": receipt["accuracy"] >= max(
                fixed_low["accuracy"], fixed_high["accuracy"]),
            "binary_retained": binary["gate"]["accepted"],
            "four_rule_retained": four_rule["gate"]["accepted"],
            "only_new_volatility_column_changed":
                changed == ["memory_replacement_extra_gate.weight"],
            "under_five_minute_cap": total_seconds <= 300.0,
        },
    }
    report["gates"]["accepted"] = all(report["gates"].values())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if report["gates"]["accepted"] and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "history": history,
        "policies": report["policies"],
        "accounting": report["accounting"],
        "gates": report["gates"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
