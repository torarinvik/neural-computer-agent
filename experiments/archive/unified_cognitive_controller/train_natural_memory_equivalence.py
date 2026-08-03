"""Retrieve naturally acquired memories by learned latent equivalence.

The learner receives a fresh feedback-bearing memory value and four stored
values.  Stored values come from independent visual lifetimes; two values are
called equivalent only when executing them earns the same scalar verifier
outcome.  Rule bits are never used by this experiment.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import tempfile
import time
from pathlib import Path

import torch

from .audit_selective_disk import _support
from .environment import generate_lifetimes
from .memory import DiskLatentMemory
from .legacy_model import UnifiedCognitiveController
from .probe_persistent_interface import _add_context_signatures
from .train import evaluate, seed_everything
from .train_adaptive_memory_read import _outcomes
from .train_conditional_memory_usage_prior import evaluate_conditional
from .train_continuous_memory_usage_prior import (
    evaluate_policy as evaluate_parent_continuous,
)
from .train_four_target_memory_retrieval import (
    four_target_batch,
    policy_features,
    select_rows,
)
from .train_memory_replacement import _select_batch


@torch.no_grad()
def natural_equivalence_batch(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, heldout: bool, exact_fraction: float,
        source_multiplier: int = 2,
        crossing_jitter_range: tuple[float, float] = (0.0, 0.0),
        slope_jitter_range: tuple[float, float] = (0.0, 0.0),
        shuffle_probe: bool = False,
        shuffle_relation_values: bool = False,
        corrupt_retrieved_values: bool = False,
        permute_rows: bool = True,
        reverse_target_rule: bool = False,
        ) -> dict[str, object]:
    """Build a bank with natural equivalences and conflicts.

    The easy curriculum arm replaces one independent stored value with the
    exact fresh value.  The hard arm uses four independently acquired values.
    In both cases examples are retained only when scalar verification shows at
    least one equivalent and one conflicting memory.  This private filtering
    is counted as experience and exposes no label to the learner.
    """
    if not 0.0 <= exact_fraction <= 1.0:
        raise ValueError("exact fraction must lie in [0, 1]")
    if source_multiplier < 2:
        raise ValueError("source multiplier must be at least two")
    source_count = count * source_multiplier
    base = four_target_batch(
        model, count=source_count, seed=seed, device=device,
        heldout=heldout, permute_rows=False,
        crossing_jitter_range=crossing_jitter_range,
        slope_jitter_range=slope_jitter_range)
    source_batches = base.pop("_source_batches")
    target_batch = source_batches[0]
    if reverse_target_rule:
        target_batch = replace(
            target_batch,
            correct_actions=1 - target_batch.correct_actions,
            rule_bits=1 - target_batch.rule_bits)
    _, probe_values, _ = _support(model, target_batch, device=device)
    candidate_values = []
    for batch in source_batches[1:]:
        _, values, _ = _support(model, batch, device=device)
        candidate_values.append(values)
    extra = _add_context_signatures(
        generate_lifetimes(
            source_count, 3, seed=seed + 70_000_000,
            heldout=heldout, task="binary_mapping", support_trials=1,
            device=device),
        seed=seed + 80_000_000)
    _, extra_values, _ = _support(model, extra, device=device)
    candidate_values.append(extra_values)
    values = torch.stack(candidate_values, dim=1)
    outcomes = torch.stack([
        _outcomes(
            model, target_batch, values[:, row], device=device).float()
        for row in range(4)
    ], dim=-1)

    generator = torch.Generator().manual_seed(seed + 81_000_000)
    exact_mask = (
        torch.rand(source_count, generator=generator) < exact_fraction
    ).to(device)
    values[exact_mask, 0] = probe_values[exact_mask]
    outcomes[exact_mask, 0] = 1.0
    valid = (outcomes.sum(-1) > 0) & (outcomes.sum(-1) < 4)
    valid_indices = torch.where(valid)[0]
    if valid_indices.numel() < count:
        raise RuntimeError(
            f"only {valid_indices.numel()} mixed natural banks for {count}")
    indices = valid_indices[:count]

    def take(value):
        return value[indices] if isinstance(value, torch.Tensor) else value

    queries = take(base["queries"])
    keys = take(base["keys"])
    usage = take(base["usage"])
    values = values[indices]
    outcomes = outcomes[indices]
    probe_values = probe_values[indices]
    selected_batch = _select_batch(target_batch, indices)
    if permute_rows:
        random = torch.rand(count, 4, generator=generator).to(device)
        permutation = random.argsort(dim=-1)
    else:
        permutation = torch.arange(
            4, device=device).expand(count, -1)
    gather_width = permutation.unsqueeze(-1).expand(-1, -1, model.width)
    keys = torch.gather(keys, 1, gather_width)
    usage = torch.gather(usage, 1, permutation)
    values = torch.gather(values, 1, gather_width)
    outcomes = torch.gather(outcomes, 1, permutation)
    relation_values = values
    if corrupt_retrieved_values:
        values = values.roll(1, dims=1)

    normalized_queries = torch.nn.functional.normalize(queries, dim=-1)
    normalized_keys = torch.nn.functional.normalize(keys, dim=-1)
    content = torch.einsum(
        "bw,bkw->bk", normalized_queries, normalized_keys)
    content_order = content.argsort(dim=-1, descending=True)
    sorted_values = torch.gather(
        relation_values, 1, content_order.unsqueeze(-1).expand(
            -1, -1, model.width))
    sorted_outcomes = torch.gather(outcomes, 1, content_order)
    if shuffle_probe:
        probe_values = probe_values.roll(1, dims=0)
    if shuffle_relation_values:
        sorted_values = sorted_values.roll(1, dims=0)

    selected_base = {
        key: take(value)
        for key, value in base.items()
        if isinstance(value, torch.Tensor)
        and value.shape[:1] == (source_count,)
    }
    selected_base.update({
        "target_batch": selected_batch,
        "queries": queries,
        "keys": keys,
        "values": values,
        "usage": usage,
        "probe_values": probe_values,
        "sorted_values": sorted_values,
        "row_outcomes": outcomes,
        "sorted_outcomes": sorted_outcomes,
        "duplicate_count": sorted_outcomes.sum(-1).to(torch.long),
        "source_examples": source_count,
        "mining_verifier_bits": source_count * 4,
        "generated_contexts": source_count * 5,
        "exact_examples": int(exact_mask[indices].sum()),
    })
    return selected_base


@torch.no_grad()
def evaluate_equivalence(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, exact_fraction: float = 0.0,
        crossing_jitter_range: tuple[float, float] = (0.0, 0.0),
        slope_jitter_range: tuple[float, float] = (0.0, 0.0),
        shuffle_probe: bool = False,
        shuffle_relation_values: bool = False,
        corrupt_retrieved_values: bool = False,
        permute_rows: bool = True,
        reverse_target_rule: bool = False,
        ) -> dict[str, object]:
    data = natural_equivalence_batch(
        model, count=count, seed=seed, device=device, heldout=True,
        exact_fraction=exact_fraction,
        crossing_jitter_range=crossing_jitter_range,
        slope_jitter_range=slope_jitter_range,
        shuffle_probe=shuffle_probe,
        shuffle_relation_values=shuffle_relation_values,
        corrupt_retrieved_values=corrupt_retrieved_values,
        permute_rows=permute_rows,
        reverse_target_rule=reverse_target_rule)
    features = policy_features(data)
    learned = model.memory_equivalence_probability(
        features, data["probe_values"], data["sorted_values"])
    relation_logits = model.memory_equivalence_logits(
        data["probe_values"], data["sorted_values"])
    relation_selected = relation_logits.argmax(-1)
    relation_success = data["sorted_outcomes"].gather(
        1, relation_selected.unsqueeze(-1)).squeeze(-1)
    parent = model.memory_usage_prior_probability(features)
    candidates = model.memory_usage_prior_candidates(features)
    policies = {"learned": learned, "parent": parent}
    policies.update({
        f"fixed_rank_{rank}": candidates[:, rank]
        for rank in range(4)
    })
    report: dict[str, object] = {}
    for name, scales in policies.items():
        selected, selected_values = select_rows(data, scales)
        success = _outcomes(
            model, data["target_batch"], selected_values,
            device=device).float()
        report[name] = {
            "accuracy": float(success.mean()),
            "mean_scale": float(scales.mean()),
            "by_duplicate_count": {
                str(duplicates): float(
                    success[data["duplicate_count"] == duplicates].mean())
                for duplicates in (1, 2, 3)
                if bool((data["duplicate_count"] == duplicates).any())
            },
        }
    report["examples"] = count
    report["mean_equivalent_rows"] = float(
        data["duplicate_count"].float().mean())
    report["relation_selector_accuracy"] = float(relation_success.mean())
    return report


@torch.no_grad()
def counterfactual_reversal_audit(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device) -> dict[str, object]:
    """Reverse the verifier rule through a valid episode replay."""
    common = {
        "model": model,
        "count": count,
        "seed": seed,
        "device": device,
        "heldout": True,
        "exact_fraction": 0.0,
        "crossing_jitter_range": (0.06, 0.06),
        "slope_jitter_range": (0.10, 0.10),
    }
    ordinary = natural_equivalence_batch(**common)
    reversed_data = natural_equivalence_batch(
        **common, reverse_target_rule=True)
    banks_identical = all(torch.equal(
        ordinary[name], reversed_data[name])
        for name in ("keys", "values", "usage", "queries"))
    selections = []
    successes = []
    for data in (ordinary, reversed_data):
        features = policy_features(data)
        scales = model.memory_equivalence_probability(
            features, data["probe_values"], data["sorted_values"])
        selected, selected_values = select_rows(data, scales)
        selections.append(selected)
        successes.append(_outcomes(
            model, data["target_batch"], selected_values,
            device=device).float())
    return {
        "banks_identical": banks_identical,
        "ordinary_accuracy": float(successes[0].mean()),
        "reversed_accuracy": float(successes[1].mean()),
        "selection_flip_rate": float(
            (selections[0] != selections[1]).float().mean()),
        "probe_change_rate": float(
            (ordinary["probe_values"] != reversed_data["probe_values"])
            .any(-1).float().mean()),
    }


@torch.no_grad()
def physical_equivalence_audit(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device) -> dict[str, object]:
    """Exercise the learned relation through real disk-backed memory banks."""
    data = natural_equivalence_batch(
        model, count=count, seed=seed, device=device, heldout=True,
        exact_fraction=0.0, crossing_jitter_range=(0.06, 0.06),
        slope_jitter_range=(0.10, 0.10))
    features = policy_features(data)
    scales = model.memory_equivalence_probability(
        features, data["probe_values"], data["sorted_values"])
    reads = []
    exact_reloads = 0
    with tempfile.TemporaryDirectory(
            prefix="natural-equivalence-") as root:
        directory = Path(root)
        for index in range(count):
            memory = DiskLatentMemory(model.width, capacity=4, device=device)
            memory.commit(
                data["keys"][index], data["values"][index],
                data["usage"][index], threshold=0.0)
            path = directory / f"bank-{index:04d}.pt"
            memory.save(path)
            restored = DiskLatentMemory.load(path, device=device)
            exact_reloads += int(
                torch.equal(restored.store.keys, memory.store.keys)
                and torch.equal(restored.store.values, memory.store.values)
                and torch.equal(restored.store.usage, memory.store.usage))
            read, _ = restored.retrieve(
                data["queries"][index:index + 1], top_k=1,
                confidence_mode="cosine",
                usage_prior_scale=scales[index:index + 1])
            reads.append(read)
    outcomes = _outcomes(
        model, data["target_batch"], torch.cat(reads), device=device).float()
    return {
        "banks": count,
        "visual_accuracy": float(outcomes.mean()),
        "exact_reload_count": exact_reloads,
        "all_banks_reload_exactly": exact_reloads == count,
    }


def nearest_verified_candidate_loss(
        predicted: torch.Tensor, candidates: torch.Tensor,
        successful: torch.Tensor) -> torch.Tensor:
    """Move to the nearest verified mode without bridging failed intervals."""
    distance = (predicted.unsqueeze(-1) - candidates).square()
    distance = distance.masked_fill(~successful, float("inf"))
    return distance.min(dim=-1).values.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=20201)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-count", type=int, default=512)
    parser.add_argument("--physical-count", type=int, default=128)
    parser.add_argument("--retention-count", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--replay-updates", type=int, default=16)
    parser.add_argument("--shuffle-rewards", action="store_true")
    parser.add_argument("--exact-only", action="store_true")
    parser.add_argument("--training-crossing-jitter", type=float, default=0.04)
    parser.add_argument("--training-slope-jitter", type=float, default=0.08)
    parser.add_argument("--heldout-crossing-jitter", type=float, default=0.06)
    parser.add_argument("--heldout-slope-jitter", type=float, default=0.10)
    args = parser.parse_args()
    if args.steps < 3 or args.batch_size < 8 or args.replay_updates < 1:
        parser.error("steps, batch size, and replay updates are too small")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    configuration = dict(payload["model_configuration"])
    if int(configuration.get("adaptive_memory_equivalence_hidden", 0)) == 0:
        configuration["adaptive_memory_equivalence_hidden"] = args.hidden
        model = UnifiedCognitiveController(**configuration).to(device)
        missing, unexpected = model.load_state_dict(
            payload["state_dict"], strict=False)
        expected = {
            "memory_equivalence_opening",
            "memory_equivalence_selector.0.weight",
            "memory_equivalence_selector.0.bias",
            "memory_equivalence_selector.2.weight",
            "memory_equivalence_selector.2.bias",
        }
        if set(missing) != expected or unexpected:
            raise ValueError(
                f"unexpected equivalence expansion: {missing=}, {unexpected=}")
    else:
        model = UnifiedCognitiveController(**configuration).to(device)
        model.load_state_dict(payload["state_dict"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert model.memory_equivalence_selector is not None
    assert model.memory_equivalence_opening is not None
    trainable = list(model.memory_equivalence_selector.parameters()) + [
        model.memory_equivalence_opening]
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=1e-4)
    initial = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()}
    preflight = evaluate_equivalence(
        model, count=min(args.test_count, 256),
        seed=args.seed + 90_000_000, device=device)
    history = []
    verifier_bits = logical_lifetimes = replayed_examples = 0
    optimizer_updates = 0
    started = time.perf_counter()
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 91_000_000)
    for step in range(1, args.steps + 1):
        if args.exact_only:
            exact_fraction = 1.0
        elif step <= args.steps // 3:
            exact_fraction = 1.0
        elif step <= 2 * args.steps // 3:
            exact_fraction = 0.5
        else:
            exact_fraction = 0.0
        data = natural_equivalence_batch(
            model, count=args.batch_size,
            seed=args.seed * 1_000_000 + step,
            device=device, heldout=False,
            exact_fraction=exact_fraction,
            crossing_jitter_range=(
                -args.training_crossing_jitter,
                args.training_crossing_jitter),
            slope_jitter_range=(
                -args.training_slope_jitter,
                args.training_slope_jitter))
        verifier_bits += int(data["mining_verifier_bits"])
        logical_lifetimes += int(data["generated_contexts"])
        features = policy_features(data).detach()
        probe = data["probe_values"].detach()
        row_values = data["sorted_values"].detach()
        successful = data["sorted_outcomes"].bool().detach()
        if args.shuffle_rewards:
            successful = successful.flatten()[
                torch.randperm(
                    successful.numel(), generator=reward_generator,
                    device=device)
            ].reshape_as(successful)
            # A shuffled row may create an all-failure sample. It carries no
            # positive target and is omitted rather than silently relabeled.
            valid = successful.any(-1)
            features, probe, row_values, successful = (
                features[valid], probe[valid], row_values[valid],
                successful[valid])
        for _ in range(args.replay_updates):
            logits = model.memory_equivalence_logits(probe, row_values)
            log_mass = torch.logsumexp(
                logits.masked_fill(~successful, float("-inf")), dim=-1)
            selector_loss = (
                torch.logsumexp(logits, dim=-1) - log_mass).mean()
            prediction = model.memory_equivalence_probability(
                features, probe, row_values)
            candidates = model.memory_usage_prior_candidates(features)
            action_loss = nearest_verified_candidate_loss(
                prediction, candidates, successful)
            loss = selector_loss + action_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_updates += 1
            replayed_examples += int(features.shape[0])
        if step in {1, args.steps // 3, 2 * args.steps // 3, args.steps}:
            hard = evaluate_equivalence(
                model, count=min(args.test_count, 256),
                seed=args.seed + 92_000_000 + step, device=device,
                crossing_jitter_range=(
                    args.heldout_crossing_jitter,
                    args.heldout_crossing_jitter),
                slope_jitter_range=(
                    args.heldout_slope_jitter,
                    args.heldout_slope_jitter))
            history.append({
                "step": step,
                "verifier_bits": verifier_bits,
                "exact_fraction": exact_fraction,
                "hard_accuracy": hard["learned"]["accuracy"],
                "opening": float(model.memory_equivalence_opening.detach()),
            })
    evaluations = {
        "independent": evaluate_equivalence(
            model, count=args.test_count, seed=args.seed + 93_000_000,
            device=device, exact_fraction=0.0,
            crossing_jitter_range=(
                args.heldout_crossing_jitter,
                args.heldout_crossing_jitter),
            slope_jitter_range=(
                args.heldout_slope_jitter,
                args.heldout_slope_jitter)),
        "exact": evaluate_equivalence(
            model, count=args.test_count, seed=args.seed + 94_000_000,
            device=device, exact_fraction=1.0),
        "probe_shuffle": evaluate_equivalence(
            model, count=args.test_count, seed=args.seed + 93_000_000,
            device=device, exact_fraction=0.0, shuffle_probe=True,
            crossing_jitter_range=(
                args.heldout_crossing_jitter,
                args.heldout_crossing_jitter),
            slope_jitter_range=(
                args.heldout_slope_jitter,
                args.heldout_slope_jitter)),
        "relation_value_shuffle": evaluate_equivalence(
            model, count=args.test_count, seed=args.seed + 93_000_000,
            device=device, exact_fraction=0.0,
            shuffle_relation_values=True,
            crossing_jitter_range=(
                args.heldout_crossing_jitter,
                args.heldout_crossing_jitter),
            slope_jitter_range=(
                args.heldout_slope_jitter,
                args.heldout_slope_jitter)),
        "value_corruption": evaluate_equivalence(
            model, count=args.test_count, seed=args.seed + 93_000_000,
            device=device, exact_fraction=0.0,
            corrupt_retrieved_values=True,
            crossing_jitter_range=(
                args.heldout_crossing_jitter,
                args.heldout_crossing_jitter),
            slope_jitter_range=(
                args.heldout_slope_jitter,
                args.heldout_slope_jitter)),
        "unpermuted_rows": evaluate_equivalence(
            model, count=args.test_count, seed=args.seed + 93_000_000,
            device=device, exact_fraction=0.0, permute_rows=False,
            crossing_jitter_range=(
                args.heldout_crossing_jitter,
                args.heldout_crossing_jitter),
            slope_jitter_range=(
                args.heldout_slope_jitter,
                args.heldout_slope_jitter)),
    }
    reversal = counterfactual_reversal_audit(
        model, count=args.test_count, seed=args.seed + 94_500_000,
        device=device)
    physical = physical_equivalence_audit(
        model, count=args.physical_count, seed=args.seed + 95_000_000,
        device=device)
    parent_continuous = evaluate_parent_continuous(
        model, count=args.test_count, rows=4,
        seed=args.seed + 96_000_000, device=device,
        scale_cost=0.30, difficulty="separated")
    parent_conditional = evaluate_conditional(
        model, count=args.test_count, seed=args.seed + 97_000_000,
        device=device)
    binary = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 98_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 99_000_000, device=device,
        task="four_rule", feedback_trials=2)
    allowed_changes = {
        "memory_equivalence_opening",
        "memory_equivalence_selector.0.weight",
        "memory_equivalence_selector.0.bias",
        "memory_equivalence_selector.2.weight",
        "memory_equivalence_selector.2.bias",
    }
    changed = sorted(
        name for name, value in model.state_dict().items()
        if not torch.equal(value.detach().cpu(), initial[name]))
    independent = evaluations["independent"]
    learned = independent["learned"]
    minimum_duplicate_accuracy = min(
        learned["by_duplicate_count"].values())
    gates = {
        "independent_accuracy_at_least_95":
            learned["accuracy"] >= 0.95,
        "each_duplicate_count_at_least_90":
            minimum_duplicate_accuracy >= 0.90,
        "relation_selector_at_least_98":
            independent["relation_selector_accuracy"] >= 0.98,
        "beats_parent_by_25_points":
            learned["accuracy"]
            >= independent["parent"]["accuracy"] + 0.25,
        "probe_shuffle_costs_30_points":
            learned["accuracy"]
            >= evaluations["probe_shuffle"]["learned"]["accuracy"] + 0.30,
        "relation_value_shuffle_costs_30_points":
            learned["accuracy"]
            >= evaluations[
                "relation_value_shuffle"]["learned"]["accuracy"] + 0.30,
        "retrieved_values_are_causal":
            learned["accuracy"]
            >= evaluations["value_corruption"]["learned"]["accuracy"] + 0.20,
        "physical_row_permutation_invariant":
            evaluations["unpermuted_rows"]["learned"]["accuracy"] >= 0.95,
        "counterfactual_banks_identical": reversal["banks_identical"],
        "counterfactual_ordinary_at_least_95":
            reversal["ordinary_accuracy"] >= 0.95,
        "counterfactual_reversed_at_least_95":
            reversal["reversed_accuracy"] >= 0.95,
        "counterfactual_selection_flips_at_least_95":
            reversal["selection_flip_rate"] >= 0.95,
        "physical_accuracy_at_least_95":
            physical["visual_accuracy"] >= 0.95,
        "physical_reload_exact": physical["all_banks_reload_exactly"],
        "parent_continuous_retained":
            parent_continuous["continuous"]["row_accuracy"] >= 0.95,
        "parent_conditional_retained":
            parent_conditional["learned"]["accuracy"] >= 0.95,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "only_equivalence_module_changed":
            set(changed) <= allowed_changes,
        "under_three_minutes":
            time.perf_counter() - started <= 180.0,
    }
    gates["accepted"] = all(gates.values())
    stable_verifier_bits = None
    for index, entry in enumerate(history):
        if (
                entry["hard_accuracy"] >= 0.95
                and all(
                    later["hard_accuracy"] >= 0.95
                    for later in history[index:])):
            stable_verifier_bits = entry["verifier_bits"]
            break
    report = {
        "schema": "natural-memory-equivalence-v1",
        "seed": args.seed,
        "configuration": vars(args) | {
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out) if args.checkpoint_out else None),
            "report": str(args.report),
        },
        "preflight": preflight,
        "history": history,
        "evaluations": evaluations,
        "physical": physical,
        "counterfactual_reversal": reversal,
        "retention": {
            "parent_continuous": parent_continuous,
            "parent_conditional": parent_conditional,
            "binary_mapping": binary,
            "four_rule": four_rule,
        },
        "accounting": {
            "unique_verifier_bits": verifier_bits,
            "unique_logical_lifetimes": logical_lifetimes,
            "optimizer_updates": optimizer_updates,
            "replayed_examples": replayed_examples,
            "wall_seconds": time.perf_counter() - started,
        },
        "changed_parameters": changed,
        "only_equivalence_module_changed": set(changed) <= allowed_changes,
        "stable_verifier_bits_to_95": stable_verifier_bits,
        "gates": gates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-controller-natural-equivalence-v1",
            "model_configuration": configuration,
            "state_dict": model.state_dict(),
            "source_checkpoint": str(args.checkpoint_in),
            "report": report,
        }, args.checkpoint_out)
    print(json.dumps({
        "independent": evaluations["independent"]["learned"],
        "parent": evaluations["independent"]["parent"],
        "probe_shuffle": evaluations["probe_shuffle"]["learned"],
        "value_shuffle": evaluations[
            "relation_value_shuffle"]["learned"],
        "value_corruption": evaluations["value_corruption"]["learned"],
        "physical": physical,
        "counterfactual_reversal": reversal,
        "gates": gates,
        "accounting": report["accounting"],
        "opening": float(model.memory_equivalence_opening.detach()),
        "changed": changed,
    }, indent=2))


if __name__ == "__main__":
    main()
