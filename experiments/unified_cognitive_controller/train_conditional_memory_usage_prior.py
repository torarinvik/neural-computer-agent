"""Learn per-query content-versus-usage retrieval from visual reward."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import torch

from .audit_selective_disk import (
    _add_context_signatures,
    _query_keys,
    _support,
)
from .environment import generate_lifetimes
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController, full_memory_usage_features
from .train import evaluate, seed_everything
from .train_adaptive_memory_read import _outcomes
from .train_memory_replacement import _select_batch


def expand_with_conditional_usage_prior(
        payload: dict[str, object], *, hidden: int,
        device: torch.device,
        ) -> tuple[UnifiedCognitiveController, dict[str, object]]:
    configuration = dict(payload["model_configuration"])
    if not bool(configuration.get("adaptive_memory_usage_prior")):
        raise ValueError("conditional policy requires the adaptive-prior parent")
    configuration["adaptive_memory_usage_prior_hidden"] = hidden
    model = UnifiedCognitiveController(**configuration).to(device)
    missing, unexpected = model.load_state_dict(
        payload["state_dict"], strict=False)
    expected = {
        "memory_usage_prior_policy.0.weight",
        "memory_usage_prior_policy.0.bias",
        "memory_usage_prior_policy.2.weight",
        "memory_usage_prior_policy.2.bias",
    }
    if set(missing) != expected or unexpected:
        raise ValueError(
            f"unexpected conditional-prior mismatch: {missing=}, "
            f"{unexpected=}")
    return model, configuration


def _near_keys(
        queries: torch.Tensor, cosine: float,
        generator: torch.Generator) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(queries, dim=-1)
    noise = torch.randn(
        normalized.shape, generator=generator,
        device="cpu", dtype=normalized.dtype).to(normalized.device)
    orthogonal = noise - (
        noise * normalized).sum(-1, keepdim=True) * normalized
    orthogonal = torch.nn.functional.normalize(orthogonal, dim=-1)
    return (
        cosine * normalized
        + math.sqrt(1.0 - cosine * cosine) * orthogonal)


@torch.no_grad()
def conditional_batch(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, heldout: bool,
        shuffle_features: bool = False,
        corrupt_values: bool = False,
        ) -> dict[str, object]:
    full = _add_context_signatures(
        generate_lifetimes(
            count * 2, 3, seed=seed, heldout=heldout,
            task="binary_mapping", support_trials=1, device=device),
        seed=seed + 10_000_000)
    target_indices = torch.arange(count, device=device)
    distractor_indices = torch.arange(count, count * 2, device=device)
    target = _select_batch(full, target_indices)
    distractor = _select_batch(full, distractor_indices)
    _, correct_values, _ = _support(model, target, device=device)
    _, wrong_values, _ = _support(model, distractor, device=device)
    queries = _query_keys(model, target, device=device)
    generator = torch.Generator().manual_seed(seed + 85_000_000)
    arm = torch.arange(count, device=device) % 2
    arm = arm[torch.randperm(count, generator=generator).to(device)]
    exact = arm == 0
    ambiguous = ~exact

    exact_wrong = _near_keys(queries, 0.80, generator)
    ambiguous_correct = _near_keys(queries, 0.99, generator)
    keys = torch.empty(count, 2, model.width, device=device)
    values = torch.empty_like(keys)
    usage = torch.empty(count, 2, device=device)
    # Exact arm: content identifies the low-usage correct row; scale 0 wins.
    keys[exact, 0] = torch.nn.functional.normalize(queries[exact], dim=-1)
    keys[exact, 1] = exact_wrong[exact]
    values[exact, 0] = correct_values[exact]
    values[exact, 1] = wrong_values[exact]
    usage[exact] = torch.tensor([0.5, 1.0], device=device)
    # Ambiguous arm: content slightly favors the wrong low-usage row, while
    # verified usage identifies the nearly identical correct row; scale 1 wins.
    keys[ambiguous, 0] = torch.nn.functional.normalize(
        queries[ambiguous], dim=-1)
    keys[ambiguous, 1] = ambiguous_correct[ambiguous]
    values[ambiguous, 0] = wrong_values[ambiguous]
    values[ambiguous, 1] = correct_values[ambiguous]
    usage[ambiguous] = torch.tensor([0.5, 1.0], device=device)
    if corrupt_values:
        values = values.roll(1, dims=1)

    normalized_queries = torch.nn.functional.normalize(queries, dim=-1)
    cosine = torch.einsum("bw,bkw->bk", normalized_queries, keys)
    scores, order = cosine.topk(2, dim=-1)
    top_usage = torch.gather(usage, 1, order[:, :1]).squeeze(1)
    features = torch.stack((
        scores[:, 0],
        scores[:, 0] - scores[:, 1],
        top_usage,
        torch.ones_like(top_usage),
    ), dim=-1)
    policy_features = full_memory_usage_features(
        features, queries, keys, usage)
    if shuffle_features:
        features = features.roll(1, dims=0)
        policy_features = policy_features.roll(1, dims=0)
    return {
        "target_batch": target,
        "queries": queries,
        "keys": keys,
        "values": values,
        "usage": usage,
        "features": features,
        "policy_features": policy_features,
        "target_scale": arm.to(torch.float32),
        "arm": arm,
        "generated_contexts": count * 2,
    }


@torch.no_grad()
def retrieve_for_scales(
        data: dict[str, object], scales: torch.Tensor) -> torch.Tensor:
    queries = torch.nn.functional.normalize(data["queries"], dim=-1)
    keys = torch.nn.functional.normalize(data["keys"], dim=-1)
    cosine = torch.einsum("bw,bkw->bk", queries, keys)
    ranked = cosine + (
        scales.unsqueeze(-1)
        * data["usage"].clamp_min(1e-6).log())
    selected = ranked.argmax(-1)
    return torch.gather(
        data["values"], 1,
        selected[:, None, None].expand(
            -1, 1, data["values"].shape[-1])).squeeze(1)


@torch.no_grad()
def evaluate_conditional(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, shuffle_features: bool = False,
        corrupt_values: bool = False) -> dict[str, object]:
    data = conditional_batch(
        model, count=count, seed=seed, device=device, heldout=True,
        shuffle_features=shuffle_features, corrupt_values=corrupt_values)
    probability = model.memory_usage_prior_probability(
        data.get("policy_features", data["features"]))
    learned_scale = (probability >= 0.5).to(torch.float32)
    policies = {
        "learned": learned_scale,
        "fixed_zero": torch.zeros_like(learned_scale),
        "fixed_one": torch.ones_like(learned_scale),
    }
    report: dict[str, object] = {
        "mean_probability": float(probability.mean()),
        "scale_action_accuracy": float(
            (learned_scale == data["target_scale"]).float().mean()),
    }
    for name, scales in policies.items():
        memory = retrieve_for_scales(data, scales)
        outcomes = _outcomes(
            model, data["target_batch"], memory, device=device).float()
        report[name] = {
            "accuracy": float(outcomes.mean()),
            "exact_accuracy": float(outcomes[data["arm"] == 0].mean()),
            "ambiguous_accuracy": float(outcomes[data["arm"] == 1].mean()),
        }
    return report


@torch.no_grad()
def physical_audit(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device) -> dict[str, object]:
    data = conditional_batch(
        model, count=count, seed=seed, device=device, heldout=True)
    probability = model.memory_usage_prior_probability(
        data.get("policy_features", data["features"]))
    scales = (probability >= 0.5).to(torch.float32)
    reads = []
    exact = 0
    with tempfile.TemporaryDirectory(
            prefix="conditional-usage-prior-") as root:
        directory = Path(root)
        for index in range(count):
            memory = DiskLatentMemory(
                model.width, capacity=2, device=device)
            memory.commit(
                data["keys"][index], data["values"][index],
                data["usage"][index], threshold=0.0)
            path = directory / f"bank-{index:04d}.pt"
            memory.save(path)
            restored = DiskLatentMemory.load(path, device=device)
            exact += int(
                torch.equal(restored.store.keys, memory.store.keys)
                and torch.equal(restored.store.values, memory.store.values)
                and torch.equal(restored.store.usage, memory.store.usage))
            read, _ = restored.retrieve(
                data["queries"][index:index + 1], top_k=1,
                confidence_mode="cosine",
                usage_prior_scale=scales[index:index + 1])
            reads.append(read)
    outcomes = _outcomes(
        model, data["target_batch"], torch.cat(reads),
        device=device).float()
    return {
        "accuracy": float(outcomes.mean()),
        "exact_accuracy": float(outcomes[data["arm"] == 0].mean()),
        "ambiguous_accuracy": float(outcomes[data["arm"] == 1].mean()),
        "scale_action_accuracy": float(
            (scales == data["target_scale"]).float().mean()),
        "exact_reload_count": exact,
        "banks": count,
        "all_banks_reload_exactly": exact == count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=17600)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=1024)
    parser.add_argument("--physical-count", type=int, default=256)
    parser.add_argument("--retention-count", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--entropy-cost", type=float, default=0.002)
    parser.add_argument("--shuffle-rewards", action="store_true")
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model, configuration = expand_with_conditional_usage_prior(
        payload, hidden=args.hidden, device=device)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.memory_usage_prior_policy is not None
    for parameter in model.memory_usage_prior_policy.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        model.memory_usage_prior_policy.parameters(),
        lr=args.learning_rate)
    preflight = evaluate_conditional(
        model, count=min(256, args.test_count),
        seed=args.seed + 90_000_000, device=device)
    started = time.perf_counter()
    history = []
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 86_000_000)
    stable_threshold = None
    generated_contexts = verifier_bits = 0
    for step in range(1, args.steps + 1):
        data = conditional_batch(
            model, count=args.batch_size,
            seed=args.seed * 1_000_000 + step,
            device=device, heldout=False)
        generated_contexts += int(data["generated_contexts"])
        probability = model.memory_usage_prior_probability(
            data.get("policy_features", data["features"]))
        distribution = torch.distributions.Bernoulli(probs=probability)
        scales = distribution.sample()
        with torch.no_grad():
            memory = retrieve_for_scales(data, scales)
            reward = _outcomes(
                model, data["target_batch"], memory,
                device=device).float()
        verifier_bits += reward.numel()
        if args.shuffle_rewards:
            reward = reward[torch.randperm(
                reward.numel(), generator=reward_generator, device=device)]
        advantage = reward - reward.mean()
        loss = -(advantage.detach() * distribution.log_prob(scales)).mean()
        loss -= args.entropy_cost * distribution.entropy().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.memory_usage_prior_policy.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 8 == 0 or step == args.steps:
            prefix = evaluate_conditional(
                model, count=min(256, args.test_count),
                seed=args.seed + 90_000_000, device=device)
            action_accuracy = float(prefix["scale_action_accuracy"])
            history.append({
                "step": step,
                "reward": float(reward.mean()),
                "scale_action_accuracy": action_accuracy,
                "held_out_accuracy":
                    prefix["learned"]["accuracy"],
                "elapsed_seconds": time.perf_counter() - started,
            })
    for index, entry in enumerate(history):
        if (
                entry["scale_action_accuracy"] >= 0.95
                and all(
                    later["scale_action_accuracy"] >= 0.95
                    for later in history[index:])):
            stable_threshold = entry["step"]
            break
    training_seconds = time.perf_counter() - started
    held_out = evaluate_conditional(
        model, count=args.test_count,
        seed=args.seed + 91_000_000, device=device)
    feature_shuffled = evaluate_conditional(
        model, count=args.test_count,
        seed=args.seed + 91_000_000, device=device,
        shuffle_features=True)
    corrupted = evaluate_conditional(
        model, count=args.test_count,
        seed=args.seed + 91_000_000, device=device,
        corrupt_values=True)
    physical = physical_audit(
        model, count=args.physical_count,
        seed=args.seed + 92_000_000, device=device)
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
        "schema": "unified-controller-conditional-usage-prior-v1",
        "configuration": {
            **vars(args),
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out) if args.checkpoint_out else None),
            "report": str(args.report),
        },
        "model_configuration": configuration,
        "preflight": preflight,
        "history": history,
        "held_out": held_out,
        "feature_shuffled": feature_shuffled,
        "value_corrupted": corrupted,
        "physical": physical,
        "retention": {
            "binary_mapping": binary,
            "four_rule": four_rule,
        },
        "changed_parameters": changed,
        "accounting": {
            "unique_logical_contexts": generated_contexts,
            "unique_verifier_bits": verifier_bits,
            "optimizer_updates": args.steps,
            "replayed_examples": 0,
            "training_seconds": training_seconds,
            "total_seconds": total_seconds,
            "stable_updates_to_95_percent": stable_threshold,
            "stable_verifier_bits_to_95_percent": (
                stable_threshold * args.batch_size
                if stable_threshold is not None else None),
        },
        "gates": {
            "learned_accuracy_at_least_90":
                held_out["learned"]["accuracy"] >= 0.90,
            "both_arms_at_least_88":
                min(
                    held_out["learned"]["exact_accuracy"],
                    held_out["learned"]["ambiguous_accuracy"]) >= 0.88,
            "beats_both_fixed_scales_by_8_points":
                held_out["learned"]["accuracy"] >= max(
                    held_out["fixed_zero"]["accuracy"],
                    held_out["fixed_one"]["accuracy"]) + 0.08,
            "scale_action_accuracy_at_least_95":
                held_out["scale_action_accuracy"] >= 0.95,
            "feature_shuffle_costs_20_points":
                held_out["scale_action_accuracy"]
                >= feature_shuffled["scale_action_accuracy"] + 0.20,
            "values_are_causal":
                corrupted["learned"]["accuracy"]
                <= held_out["learned"]["accuracy"] - 0.15,
            "physical_accuracy_at_least_88":
                physical["accuracy"] >= 0.88,
            "physical_reload_exact":
                physical["all_banks_reload_exactly"],
            "binary_retained": binary["gate"]["accepted"],
            "four_rule_retained": four_rule["gate"]["accepted"],
            "only_conditional_policy_changed":
                all(
                    name.startswith("memory_usage_prior_policy.")
                    for name in changed),
            "under_five_minutes": total_seconds <= 300.0,
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
        "feature_shuffled": feature_shuffled,
        "value_corrupted": corrupted,
        "physical": physical,
        "accounting": report["accounting"],
        "gates": report["gates"],
    }, indent=2))


if __name__ == "__main__":
    main()
