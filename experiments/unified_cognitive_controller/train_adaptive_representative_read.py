"""Learn when extra within-class memory representatives are worth reading."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from .audit_selective_disk import _support
from .environment import generate_lifetimes
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import evaluate, seed_everything
from .train_adaptive_memory_read import _outcomes
from .train_equivalence_consolidation import (
    consolidate,
    natural_memory_streams,
)


APPEARANCES = ("bars", "diamonds", "dot_pairs")


def _load_expanded(
        path: Path, *, hidden: int, threshold: float,
        critic_seed: int, device: torch.device,
        ) -> tuple[dict[str, object], UnifiedCognitiveController]:
    payload = torch.load(path, map_location=device, weights_only=False)
    configuration = dict(payload["model_configuration"])
    if configuration.get("adaptive_representative_read_hidden", 0):
        raise ValueError("checkpoint already contains a representative critic")
    configuration["adaptive_representative_read_hidden"] = hidden
    configuration["adaptive_representative_read_threshold"] = threshold
    model = UnifiedCognitiveController(**configuration).to(device)
    missing, unexpected = model.load_state_dict(
        payload["state_dict"], strict=False)
    expected = {
        name for name in model.state_dict()
        if name.startswith("representative_read_critic.")
        or name.startswith("representative_read_feature_")
    }
    if set(missing) != expected or unexpected:
        raise RuntimeError(
            f"unexpected critic insertion: {missing=}, {unexpected=}")
    assert model.representative_read_critic is not None
    with torch.random.fork_rng(
            devices=[device.index or 0] if device.type == "cuda" else []):
        torch.manual_seed(120_000 + critic_seed)
        for module in model.representative_read_critic.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    return payload, model


@torch.no_grad()
def _query_batch(
        *, count: int, seed: int, appearance: str,
        reverse_rules: bool, device: torch.device,
        ):
    return _add_context_signatures(
        generate_lifetimes(
            count, 3, seed=seed, heldout=True,
            reverse_rules=reverse_rules, task="binary_mapping",
            appearance=appearance, support_trials=1, device=device),
        seed=seed + 10_000_000)


@torch.no_grad()
def _examples_from_bank(
        model: UnifiedCognitiveController,
        bank: dict[str, torch.Tensor], *, streams: int, seed: int,
        reverse_rules: bool, device: torch.device,
        ) -> dict[str, torch.Tensor]:
    features = []
    outcomes = []
    comparisons = []
    appearance_ids = []
    values = bank["values"].repeat(2, 1, 1)
    valid = bank["valid"].repeat(2, 1)
    ranks = bank["representative_ranks"].repeat(2, 1)
    row = torch.arange(values.shape[0], device=device)
    for appearance_index, appearance in enumerate(APPEARANCES):
        batch = _query_batch(
            count=streams * 2,
            seed=seed + appearance_index,
            appearance=appearance, reverse_rules=reverse_rules,
            device=device)
        _, probes, _ = _support(model, batch, device=device)
        scores = model.calibrated_memory_equivalence_logits(
            probes, values)
        first_mask = valid & (ranks == 0)
        first_scores = scores.masked_fill(
            ~first_mask, float("-inf"))
        top, indices = first_scores.topk(2, dim=-1)
        finite = torch.isfinite(top)
        safe_top = torch.where(
            finite, top, torch.full_like(top, -20.0))
        first_value = values[row, indices[:, 0]]
        second_value = values[row, indices[:, 1]]
        second_value = torch.where(
            finite[:, 1, None], second_value,
            torch.zeros_like(second_value))
        features.append(torch.cat((
            probes,
            first_value,
            second_value,
            (probes - first_value).abs(),
            probes * first_value,
            safe_top,
            finite.to(probes.dtype),
        ), dim=-1))
        budget_outcomes = []
        budget_comparisons = []
        for representative_budget in (1, 3):
            mask = valid & (ranks < representative_budget)
            selected = scores.masked_fill(
                ~mask, float("-inf")).argmax(-1)
            retrieved = values[row, selected]
            budget_outcomes.append(
                _outcomes(
                    model, batch, retrieved,
                    device=device).to(probes.dtype))
            budget_comparisons.append(
                mask.sum(-1).to(probes.dtype))
        outcomes.append(torch.stack(budget_outcomes, dim=-1))
        comparisons.append(torch.stack(
            budget_comparisons, dim=-1))
        appearance_ids.append(torch.full(
            (streams * 2,), appearance_index,
            device=device, dtype=torch.long))
    return {
        "features": torch.cat(features).detach(),
        "outcomes": torch.cat(outcomes),
        "comparisons": torch.cat(comparisons),
        "appearance_ids": torch.cat(appearance_ids),
    }


@torch.no_grad()
def representative_read_examples(
        model: UnifiedCognitiveController, *, examples: int, seed: int,
        reverse_rules: bool, corrupt_memory: bool,
        device: torch.device,
        ) -> dict[str, torch.Tensor]:
    divisor = len(APPEARANCES) * 2
    if examples < divisor or examples % divisor:
        raise ValueError(
            f"examples must be divisible by {divisor}")
    streams = examples // divisor
    data = natural_memory_streams(
        model, streams=streams, length=16, seed=seed,
        device=device, heldout=True)
    bank = consolidate(
        model, data, capacity=6, representatives_per_class=3)
    if corrupt_memory:
        bank = {
            name: value.clone()
            for name, value in bank.items()
        }
        bank["values"].zero_()
    result = _examples_from_bank(
        model, bank, streams=streams, seed=seed + 1_000_000,
        reverse_rules=reverse_rules, device=device)
    generator = torch.Generator(device=device).manual_seed(
        seed + 9_000_000)
    permutation = torch.randperm(
        examples, generator=generator, device=device)
    return {
        name: value[permutation]
        for name, value in result.items()
    }


@torch.no_grad()
def policy_metrics(
        model: UnifiedCognitiveController,
        data: dict[str, torch.Tensor], *, read_cost: float,
        shuffle_features: bool = False,
        ) -> dict[str, object]:
    features = data["features"]
    if shuffle_features:
        features = features.roll(1, dims=0)
    probability = model.representative_deep_read_probability(features)
    deep = probability >= model.adaptive_representative_read_threshold
    outcomes = data["outcomes"]
    comparisons = data["comparisons"]
    selected_outcome = torch.where(
        deep, outcomes[:, 1], outcomes[:, 0])
    selected_comparisons = torch.where(
        deep, comparisons[:, 1], comparisons[:, 0])
    utility = selected_outcome - read_cost * selected_comparisons

    def fixed(index: int) -> dict[str, float]:
        return {
            "accuracy": float(outcomes[:, index].mean()),
            "mean_comparisons": float(
                comparisons[:, index].mean()),
            "verified_utility": float(
                (outcomes[:, index]
                 - read_cost * comparisons[:, index]).mean()),
        }

    by_appearance = {}
    for appearance_index, appearance in enumerate(APPEARANCES):
        selected = data["appearance_ids"] == appearance_index
        by_appearance[appearance] = {
            "accuracy": float(
                selected_outcome[selected].mean()),
            "mean_comparisons": float(
                selected_comparisons[selected].mean()),
            "deep_read_rate": float(
                deep[selected].to(torch.float32).mean()),
        }
    return {
        "adaptive": {
            "accuracy": float(selected_outcome.mean()),
            "mean_comparisons": float(
                selected_comparisons.mean()),
            "deep_read_rate": float(
                deep.to(torch.float32).mean()),
            "verified_utility": float(utility.mean()),
        },
        "fixed_shallow": fixed(0),
        "fixed_deep": fixed(1),
        "by_appearance": by_appearance,
    }


@torch.no_grad()
def physical_audit(
        model: UnifiedCognitiveController, *, streams: int, seed: int,
        device: torch.device, read_cost: float,
        ) -> dict[str, object]:
    data = natural_memory_streams(
        model, streams=streams, length=16, seed=seed,
        device=device, heldout=True)
    bank = consolidate(
        model, data, capacity=6, representatives_per_class=3)
    restored: dict[str, list[torch.Tensor]] = {
        "values": [], "valid": [],
        "representative_ranks": [],
    }
    exact = 0
    with tempfile.TemporaryDirectory(
            prefix="adaptive-representative-read-") as root:
        directory = Path(root)
        for index in range(streams):
            memory = DiskLatentMemory(
                model.width, capacity=6, device=device)
            memory.commit(
                bank["keys"][index], bank["values"][index],
                bank["usage"][index], threshold=0.0)
            path = directory / f"bank-{index:04d}.pt"
            memory.save(path)
            loaded = DiskLatentMemory.load(path, device=device)
            exact += int(
                torch.equal(loaded.store.keys, memory.store.keys)
                and torch.equal(
                    loaded.store.values, memory.store.values)
                and torch.equal(
                    loaded.store.usage, memory.store.usage))
            restored["values"].append(loaded.store.values[:6])
            restored["valid"].append(loaded.store.valid[:6])
            restored["representative_ranks"].append(
                bank["representative_ranks"][index])
    restored_bank = {
        name: torch.stack(values)
        for name, values in restored.items()
    }
    examples = _examples_from_bank(
        model, restored_bank, streams=streams,
        seed=seed + 1_000_000, reverse_rules=False,
        device=device)
    return {
        **policy_metrics(
            model, examples, read_cost=read_cost),
        "banks": streams,
        "exact_reload_count": exact,
        "all_banks_reload_exactly": exact == streams,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20701)
    parser.add_argument("--critic-seed", type=int, default=2)
    parser.add_argument("--critic-hidden", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--training-examples", type=int, default=4098)
    parser.add_argument("--test-examples", type=int, default=24576)
    parser.add_argument("--optimizer-updates", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--read-cost", type=float, default=0.00025)
    parser.add_argument("--physical-streams", type=int, default=256)
    parser.add_argument("--retention-count", type=int, default=512)
    parser.add_argument(
        "--population-search-verifier-bits", type=int, default=0)
    parser.add_argument(
        "--shuffle-verifier-outcomes", action="store_true")
    parser.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if (
            args.training_examples < 6
            or args.test_examples < 6
            or args.optimizer_updates < 1
            or args.critic_hidden < 4
            or args.read_cost <= 0.0
            or args.population_search_verifier_bits < 0):
        raise ValueError("training and evaluation budgets are invalid")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload, model = _load_expanded(
        args.checkpoint_in, hidden=args.critic_hidden,
        threshold=args.threshold, critic_seed=args.critic_seed,
        device=device)
    inherited = {
        name: value.detach().cpu().clone()
        for name, value in payload["state_dict"].items()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.representative_read_critic is not None
    for parameter in model.representative_read_critic.parameters():
        parameter.requires_grad_(True)

    started = time.perf_counter()
    train = representative_read_examples(
        model, examples=args.training_examples,
        seed=args.seed + 10_000_000, reverse_rules=False,
        corrupt_memory=False, device=device)
    assert model.representative_read_feature_mean is not None
    assert model.representative_read_feature_scale is not None
    model.representative_read_feature_mean.copy_(
        train["features"].mean(0))
    model.representative_read_feature_scale.copy_(
        train["features"].std(0).clamp_min(1e-4))
    need_deep = (
        train["outcomes"][:, 1] > train["outcomes"][:, 0]
    ).to(torch.float32)
    if args.shuffle_verifier_outcomes:
        generator = torch.Generator(device=device).manual_seed(
            args.seed + 11_000_000)
        need_deep = need_deep[
            torch.randperm(
                need_deep.numel(), generator=generator,
                device=device)]
    positive_rate = float(need_deep.mean())
    positive_weight = torch.tensor(
        (1.0 - positive_rate) / max(positive_rate, 1e-4),
        device=device)
    optimizer = torch.optim.AdamW(
        model.representative_read_critic.parameters(),
        lr=args.learning_rate, weight_decay=1e-4)
    history = []
    for update in range(1, args.optimizer_updates + 1):
        probability = model.representative_deep_read_probability(
            train["features"])
        loss = torch.nn.functional.binary_cross_entropy(
            probability, need_deep,
            weight=torch.where(
                need_deep > 0, positive_weight,
                torch.ones_like(need_deep)))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            model.representative_read_critic.parameters(), 1.0)
        optimizer.step()
        if update in (1, args.optimizer_updates):
            history.append({
                "update": update,
                "loss": float(loss.detach()),
                "mean_probability": float(
                    probability.detach().mean()),
            })

    test = representative_read_examples(
        model, examples=args.test_examples,
        seed=args.seed + 20_000_000, reverse_rules=False,
        corrupt_memory=False, device=device)
    reversed_test = representative_read_examples(
        model, examples=args.test_examples,
        seed=args.seed + 20_000_000, reverse_rules=True,
        corrupt_memory=False, device=device)
    corrupted = representative_read_examples(
        model, examples=min(args.test_examples, 4098),
        seed=args.seed + 30_000_000, reverse_rules=False,
        corrupt_memory=True, device=device)
    metrics = policy_metrics(
        model, test, read_cost=args.read_cost)
    reversed_metrics = policy_metrics(
        model, reversed_test, read_cost=args.read_cost)
    shuffled_features = policy_metrics(
        model, test, read_cost=args.read_cost,
        shuffle_features=True)
    corrupted_metrics = policy_metrics(
        model, corrupted, read_cost=args.read_cost)
    physical = physical_audit(
        model, streams=args.physical_streams,
        seed=args.seed + 40_000_000, device=device,
        read_cost=args.read_cost)
    binary = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 50_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 60_000_000, device=device,
        task="four_rule", feedback_trials=2)
    inherited_identical = all(
        torch.equal(
            model.state_dict()[name].detach().cpu(), value)
        for name, value in inherited.items())
    adaptive = metrics["adaptive"]
    shallow = metrics["fixed_shallow"]
    deep = metrics["fixed_deep"]
    gates = {
        "adaptive_within_0_15_points_of_deep":
            adaptive["accuracy"] >= deep["accuracy"] - 0.0015,
        "adaptive_beats_shallow_by_0_7_points":
            adaptive["accuracy"] >= shallow["accuracy"] + 0.007,
        "adaptive_utility_beats_fixed_deep":
            adaptive["verified_utility"] > deep["verified_utility"],
        "adaptive_uses_at_most_2_15_comparisons":
            adaptive["mean_comparisons"] <= 2.15,
        "reverse_accuracy_retained":
            reversed_metrics["adaptive"]["accuracy"]
            >= reversed_metrics["fixed_deep"]["accuracy"] - 0.0015,
        "feature_alignment_causal":
            adaptive["verified_utility"]
            >= shuffled_features["adaptive"][
                "verified_utility"] + 0.002,
        "memory_values_causal":
            corrupted_metrics["adaptive"]["accuracy"] <= 0.60,
        "physical_reload_exact":
            physical["all_banks_reload_exactly"],
        "physical_adaptive_within_0_5_points_of_deep":
            physical["adaptive"]["accuracy"]
            >= physical["fixed_deep"]["accuracy"] - 0.005,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "inherited_tensors_bit_identical": inherited_identical,
        "under_three_minutes":
            time.perf_counter() - started < 180.0,
    }
    gates["accepted"] = all(gates.values())
    configuration = dict(payload["model_configuration"])
    configuration.update({
        "adaptive_representative_read_hidden":
            args.critic_hidden,
        "adaptive_representative_read_threshold":
            args.threshold,
    })
    report = {
        "schema": "adaptive-representative-read-v1",
        "claim_boundary": (
            "The critic predicts marginal success from latent query/memory "
            "state using only executed shallow/deep scalar outcomes. "
            "Accuracy is primary; comparison cost is secondary."),
        "configuration": {
            **vars(args),
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out)
                if args.checkpoint_out is not None else None),
            "report": str(args.report),
        },
        "semantic_labels_used_for_training": False,
        "learner_visible": [
            "fresh_controller_latent",
            "stored_controller_latents",
            "shallow_attempt_scalar_outcome",
            "deep_attempt_scalar_outcome",
        ],
        "hidden_from_learner": [
            "appearance_name",
            "rule_bit",
            "correct_read_budget",
        ],
        "accounting": {
            "unique_training_examples": args.training_examples,
            "unique_training_verifier_bits":
                args.training_examples * 2,
            "optimizer_updates": args.optimizer_updates,
            "population_search_critic_initializations": 12,
            "population_search_verifier_data_reused": True,
            "population_search_unique_verifier_bits":
                args.population_search_verifier_bits,
        },
        "training_need_deep_rate": positive_rate,
        "history": history,
        "held_out": metrics,
        "reversed_rule": reversed_metrics,
        "shuffled_feature_control": shuffled_features,
        "corrupted_memory_control": corrupted_metrics,
        "physical": physical,
        "retention": {
            "binary_mapping": binary,
            "four_rule": four_rule,
        },
        "inherited_tensors_bit_identical": inherited_identical,
        "gates": gates,
        "wall_seconds": time.perf_counter() - started,
    }
    if gates["accepted"] and args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(
            parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
            "admission_status":
                "adaptive_representative_read",
        }, args.checkpoint_out)
        report["checkpoint_saved"] = True
    else:
        report["checkpoint_saved"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "accepted": gates["accepted"],
        "adaptive": adaptive,
        "fixed_shallow": shallow,
        "fixed_deep": deep,
        "reversed_adaptive":
            reversed_metrics["adaptive"],
        "shuffled_features":
            shuffled_features["adaptive"],
        "physical_adaptive": physical["adaptive"],
        "gates": gates,
        "wall_seconds": report["wall_seconds"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
