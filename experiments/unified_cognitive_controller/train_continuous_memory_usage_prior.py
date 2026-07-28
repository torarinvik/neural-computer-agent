"""Learn the least verified-usage influence needed for each visual query."""
from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import torch

from .audit_selective_disk import _add_context_signatures, _query_keys, _support
from .environment import generate_lifetimes
from .memory import DiskLatentMemory
from .model import UnifiedCognitiveController, full_memory_usage_features
from .train import evaluate, seed_everything
from .train_adaptive_memory_read import _outcomes
from .train_conditional_memory_usage_prior import (
    _near_keys,
    evaluate_conditional,
)
from .train_memory_replacement import _select_batch


def load_conditional_controller(
        payload: dict[str, object],
        device: torch.device) -> UnifiedCognitiveController:
    configuration = dict(payload["model_configuration"])
    if int(configuration.get("adaptive_memory_usage_prior_hidden", 0)) < 1:
        raise ValueError("continuous policy requires the conditional parent")
    model = UnifiedCognitiveController(**configuration).to(device)
    model.load_state_dict(payload["state_dict"])
    return model


@torch.no_grad()
def continuous_batch(
        model: UnifiedCognitiveController, *, count: int, rows: int,
        seed: int, device: torch.device, heldout: bool,
        difficulty: str = "familiar",
        shuffle_features: bool = False,
        corrupt_values: bool = False) -> dict[str, object]:
    if rows < 2:
        raise ValueError("continuous retrieval needs at least two rows")
    full = _add_context_signatures(
        generate_lifetimes(
            count * rows, 3, seed=seed, heldout=heldout,
            task="binary_mapping", support_trials=1, device=device),
        seed=seed + 10_000_000)
    batches = [
        _select_batch(
            full,
            torch.arange(
                index * count, (index + 1) * count, device=device))
        for index in range(rows)
    ]
    _, correct_values, _ = _support(model, batches[0], device=device)
    wrong_values = [
        _support(model, batch, device=device)[1] for batch in batches[1:]]
    queries = _query_keys(model, batches[0], device=device)
    normalized_queries = torch.nn.functional.normalize(queries, dim=-1)
    generator = torch.Generator().manual_seed(seed + 85_000_000)

    arm = torch.arange(count, device=device) % 2
    arm = arm[torch.randperm(count, generator=generator).to(device)]
    exact = arm == 0
    ambiguous = ~exact
    if difficulty not in {"familiar", "separated", "broad"}:
        raise ValueError(
            "difficulty must be familiar, separated, or broad")
    top_usage = torch.full((count,), 0.50, device=device)
    boundary = torch.empty(count, device=device)
    if difficulty == "familiar":
        # This is one gradual step beyond the binary parent's distribution:
        # the same two separable query regimes, but a small range of valid
        # continuous boundaries rather than endpoint actions.
        boundary[exact] = (
            0.25 + 0.10 * torch.rand(
                int(exact.sum()), generator=generator)).to(device)
        boundary[ambiguous] = (
            0.015 + 0.065 * torch.rand(
                int(ambiguous.sum()), generator=generator)).to(device)
    elif difficulty == "separated":
        boundary[exact] = (
            0.12 + 0.06 * torch.rand(
                int(exact.sum()), generator=generator)).to(device)
        boundary[ambiguous] = (
            0.35 + 0.20 * torch.rand(
                int(ambiguous.sum()), generator=generator)).to(device)
        margin = torch.empty(count, device=device)
        margin[exact] = (
            0.18 + 0.04 * torch.rand(
                int(exact.sum()), generator=generator)).to(device)
        margin[ambiguous] = (
            0.008 + 0.007 * torch.rand(
                int(ambiguous.sum()), generator=generator)).to(device)
        top_usage = torch.exp(-margin / boundary)
    else:
        top_usage = (
            0.30 + 0.30 * torch.rand(
                count, generator=generator)).to(device)
        boundary[exact] = (
            0.60 + 0.25 * torch.rand(
                int(exact.sum()), generator=generator)).to(device)
        boundary[ambiguous] = (
            0.12 + 0.43 * torch.rand(
                int(ambiguous.sum()), generator=generator)).to(device)
        margin = boundary * (1.0 / top_usage).log()
    if difficulty == "familiar":
        margin = boundary * (1.0 / top_usage).log()
    second_cosine = (1.0 - margin).clamp_min(0.10)

    keys = torch.empty(count, rows, model.width, device=device)
    values = torch.empty_like(keys)
    usage = torch.empty(count, rows, device=device)
    target_index = torch.empty(count, dtype=torch.long, device=device)

    keys[:, 0] = normalized_queries
    keys[:, 1] = _near_keys(queries, 0.50, generator)
    # _near_keys accepts one cosine for the batch; construct the variable
    # second-row cosine explicitly using one shared orthogonal direction.
    orthogonal = keys[:, 1] - (
        keys[:, 1] * normalized_queries).sum(-1, keepdim=True) \
        * normalized_queries
    orthogonal = torch.nn.functional.normalize(orthogonal, dim=-1)
    keys[:, 1] = (
        second_cosine.unsqueeze(-1) * normalized_queries
        + torch.sqrt(1.0 - second_cosine.square()).unsqueeze(-1)
        * orthogonal)
    values[exact, 0] = correct_values[exact]
    values[exact, 1] = wrong_values[0][exact]
    values[ambiguous, 0] = wrong_values[0][ambiguous]
    values[ambiguous, 1] = correct_values[ambiguous]
    usage[:, 0] = top_usage
    usage[:, 1] = 1.0
    target_index[exact] = 0
    target_index[ambiguous] = 1

    for index in range(2, rows):
        cosine = (second_cosine - 0.06 * (index - 1)).clamp_min(0.02)
        noise_key = _near_keys(queries, 0.40, generator)
        extra_orthogonal = noise_key - (
            noise_key * normalized_queries).sum(-1, keepdim=True) \
            * normalized_queries
        extra_orthogonal = torch.nn.functional.normalize(
            extra_orthogonal, dim=-1)
        keys[:, index] = (
            cosine.unsqueeze(-1) * normalized_queries
            + torch.sqrt(1.0 - cosine.square()).unsqueeze(-1)
            * extra_orthogonal)
        values[:, index] = wrong_values[index - 1]
        usage[:, index] = 0.90 - 0.05 * (index - 2)

    if corrupt_values:
        values = values.roll(1, dims=1)
    cosine = torch.einsum(
        "bw,bkw->bk", normalized_queries,
        torch.nn.functional.normalize(keys, dim=-1))
    scores, order = cosine.topk(2, dim=-1)
    content_usage = torch.gather(usage, 1, order[:, :1]).squeeze(1)
    features = torch.stack((
        scores[:, 0],
        scores[:, 0] - scores[:, 1],
        content_usage,
        torch.ones_like(content_usage),
    ), dim=-1)
    policy_features = full_memory_usage_features(
        features, queries, keys, usage)
    if shuffle_features:
        features = features.roll(1, dims=0)
        policy_features = policy_features.roll(1, dims=0)
    return {
        "target_batch": batches[0],
        "queries": queries,
        "keys": keys,
        "values": values,
        "usage": usage,
        "features": features,
        "policy_features": policy_features,
        "arm": arm,
        "target_index": target_index,
        "decision_boundary": boundary,
        "generated_contexts": count * rows,
    }


@torch.no_grad()
def select_rows(
        data: dict[str, object],
        scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    queries = torch.nn.functional.normalize(data["queries"], dim=-1)
    keys = torch.nn.functional.normalize(data["keys"], dim=-1)
    ranked = (
        torch.einsum("bw,bkw->bk", queries, keys)
        + scales.unsqueeze(-1) * data["usage"].clamp_min(1e-6).log())
    selected = ranked.argmax(-1)
    values = torch.gather(
        data["values"], 1,
        selected[:, None, None].expand(
            -1, 1, data["values"].shape[-1])).squeeze(1)
    return selected, values


@torch.no_grad()
def evaluate_policy(
        model: UnifiedCognitiveController, *, count: int, rows: int,
        seed: int, device: torch.device, scale_cost: float,
        difficulty: str = "familiar",
        shuffle_features: bool = False,
        corrupt_values: bool = False) -> dict[str, object]:
    data = continuous_batch(
        model, count=count, rows=rows, seed=seed, device=device,
        heldout=True, difficulty=difficulty,
        shuffle_features=shuffle_features,
        corrupt_values=corrupt_values)
    continuous = model.memory_usage_prior_probability(
        data.get("policy_features", data["features"]))
    policies = {
        "continuous": continuous,
        "thresholded": (continuous >= 0.5).to(torch.float32),
        "fixed_zero": torch.zeros_like(continuous),
        "fixed_one": torch.ones_like(continuous),
    }
    report: dict[str, object] = {}
    for name, scales in policies.items():
        selected, values = select_rows(data, scales)
        outcomes = _outcomes(
            model, data["target_batch"], values, device=device).float()
        correct_rows = selected == data["target_index"]
        report[name] = {
            "visual_accuracy": float(outcomes.mean()),
            "row_accuracy": float(correct_rows.float().mean()),
            "exact_row_accuracy": float(
                correct_rows[data["arm"] == 0].float().mean()),
            "ambiguous_row_accuracy": float(
                correct_rows[data["arm"] == 1].float().mean()),
            "mean_scale": float(scales.mean()),
            "ambiguous_mean_scale": float(
                scales[data["arm"] == 1].mean()),
            "verified_utility": float(
                outcomes.mean() - scale_cost * scales.mean()),
        }
    report["ambiguous_minimum_required_scale"] = {
        "mean": float(data["decision_boundary"][data["arm"] == 1].mean()),
        "maximum": float(data["decision_boundary"][data["arm"] == 1].max()),
    }
    report["exact_maximum_allowed_scale"] = {
        "mean": float(data["decision_boundary"][data["arm"] == 0].mean()),
        "minimum": float(data["decision_boundary"][data["arm"] == 0].min()),
    }
    return report


@torch.no_grad()
def physical_audit(
        model: UnifiedCognitiveController, *, count: int, rows: int,
        seed: int, device: torch.device,
        difficulty: str = "familiar") -> dict[str, object]:
    data = continuous_batch(
        model, count=count, rows=rows, seed=seed, device=device,
        heldout=True, difficulty=difficulty)
    scales = model.memory_usage_prior_probability(
        data.get("policy_features", data["features"]))
    correct_rows = []
    reads = []
    exact_reloads = 0
    with tempfile.TemporaryDirectory(
            prefix="continuous-usage-prior-") as root:
        directory = Path(root)
        for index in range(count):
            memory = DiskLatentMemory(
                model.width, capacity=rows, device=device)
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
            correct_rows.append(torch.equal(
                read.squeeze(0), data["values"][
                    index, data["target_index"][index]]))
    outcomes = _outcomes(
        model, data["target_batch"], torch.cat(reads),
        device=device).float()
    return {
        "rows": rows,
        "banks": count,
        "visual_accuracy": float(outcomes.mean()),
        "row_accuracy": sum(correct_rows) / count,
        "mean_scale": float(scales.mean()),
        "exact_reload_count": exact_reloads,
        "all_banks_reload_exactly": exact_reloads == count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-in", type=Path, required=True)
    parser.add_argument("--checkpoint-out", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=17700)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=1024)
    parser.add_argument("--physical-count", type=int, default=128)
    parser.add_argument("--retention-count", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--exploration-std", type=float, default=1.5)
    parser.add_argument("--scale-cost", type=float, default=0.08)
    parser.add_argument(
        "--difficulty", choices=("familiar", "separated", "broad"),
        default="separated")
    parser.add_argument("--shuffle-rewards", action="store_true")
    parser.add_argument("--cost-only", action="store_true")
    parser.add_argument("--reset-policy", action="store_true")
    args = parser.parse_args()
    if args.cost_only and args.shuffle_rewards:
        parser.error("--cost-only and --shuffle-rewards are mutually exclusive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model = load_conditional_controller(payload, device)
    if args.reset_policy:
        assert model.memory_usage_prior_policy is not None
        for module in model.memory_usage_prior_policy:
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
        output = model.memory_usage_prior_policy[-1]
        assert isinstance(output, torch.nn.Linear)
        torch.nn.init.zeros_(output.weight)
        torch.nn.init.constant_(output.bias, -2.0)
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
    preflight = evaluate_policy(
        model, count=min(256, args.test_count), rows=2,
        seed=args.seed + 90_000_000, device=device,
        scale_cost=args.scale_cost, difficulty=args.difficulty)
    started = time.perf_counter()
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 86_000_000)
    history = []
    verifier_bits = generated_contexts = 0
    for step in range(1, args.steps + 1):
        data = continuous_batch(
            model, count=args.batch_size, rows=2,
            seed=args.seed * 1_000_000 + step,
            device=device, heldout=False, difficulty=args.difficulty)
        generated_contexts += int(data["generated_contexts"])
        assert model.memory_usage_prior_policy is not None
        mean = model.memory_usage_prior_policy(
            data["features"]).squeeze(-1)
        distribution = torch.distributions.Normal(
            mean, torch.full_like(mean, args.exploration_std))
        latent_action = distribution.rsample()
        scales = torch.sigmoid(latent_action)
        if args.cost_only:
            # The learner sees only the generic cost of its own chosen scale.
            # Visual verifier outcomes are reserved for disjoint audits.
            loss = args.scale_cost * scales.mean()
        else:
            with torch.no_grad():
                _, values = select_rows(data, scales)
                outcomes = _outcomes(
                    model, data["target_batch"], values,
                    device=device).float()
            verifier_bits += outcomes.numel()
            training_outcomes = outcomes
            if args.shuffle_rewards:
                training_outcomes = outcomes[torch.randperm(
                    outcomes.numel(), generator=reward_generator,
                    device=device)]
            advantage = training_outcomes - training_outcomes.mean()
            policy_loss = -(
                advantage.detach()
                * distribution.log_prob(latent_action.detach())).mean()
            loss = policy_loss + args.scale_cost * scales.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.memory_usage_prior_policy.parameters(), 1.0)
        optimizer.step()
        if (
                args.steps <= 16
                or step == 1
                or step % 8 == 0
                or step == args.steps):
            prefix = evaluate_policy(
                model, count=min(256, args.test_count), rows=2,
                seed=args.seed + 90_000_000, device=device,
                scale_cost=args.scale_cost, difficulty=args.difficulty)
            history.append({
                "step": step,
                "visual_accuracy":
                    prefix["continuous"]["visual_accuracy"],
                "row_accuracy": prefix["continuous"]["row_accuracy"],
                "mean_scale": prefix["continuous"]["mean_scale"],
                "utility": prefix["continuous"]["verified_utility"],
                "elapsed_seconds": time.perf_counter() - started,
            })
    training_seconds = time.perf_counter() - started
    evaluations = {
        str(rows): evaluate_policy(
            model, count=args.test_count, rows=rows,
            seed=args.seed + 91_000_000 + rows, device=device,
            scale_cost=args.scale_cost, difficulty=args.difficulty)
        for rows in (2, 3, 4)
    }
    feature_shuffled = evaluate_policy(
        model, count=args.test_count, rows=4,
        seed=args.seed + 91_000_004, device=device,
        scale_cost=args.scale_cost, difficulty=args.difficulty,
        shuffle_features=True)
    value_corrupted = evaluate_policy(
        model, count=args.test_count, rows=4,
        seed=args.seed + 91_000_004, device=device,
        scale_cost=args.scale_cost, difficulty=args.difficulty,
        corrupt_values=True)
    physical = {
        str(rows): physical_audit(
            model, count=args.physical_count, rows=rows,
            seed=args.seed + 92_000_000 + rows, device=device,
            difficulty=args.difficulty)
        for rows in (2, 3, 4)
    }
    parent_conditional = evaluate_conditional(
        model, count=args.test_count,
        seed=args.seed + 95_000_000, device=device)
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
    continuous_two = evaluations["2"]["continuous"]
    thresholded_two = evaluations["2"]["thresholded"]
    inherited_binary = preflight["thresholded"]
    continuous_four = evaluations["4"]["continuous"]
    feature_four = feature_shuffled["continuous"]
    corrupted_four = value_corrupted["continuous"]
    gates = {
        "two_row_accuracy_at_least_95":
            continuous_two["row_accuracy"] >= 0.95,
        "continuous_saves_15_percent_vs_inherited_binary":
            continuous_two["mean_scale"]
            <= inherited_binary["mean_scale"] - 0.15,
        "continuous_utility_beats_inherited_binary":
            continuous_two["verified_utility"]
            >= inherited_binary["verified_utility"] + 0.01,
        "zero_shot_four_row_accuracy_at_least_93":
            continuous_four["row_accuracy"] >= 0.93,
        "feature_shuffle_costs_20_points":
            continuous_four["row_accuracy"]
            >= feature_four["row_accuracy"] + 0.20,
        "values_are_causal":
            corrupted_four["visual_accuracy"]
            <= continuous_four["visual_accuracy"] - 0.15,
        "physical_all_rows_at_least_93":
            min(item["row_accuracy"] for item in physical.values()) >= 0.93,
        "physical_reload_exact":
            all(
                item["all_banks_reload_exactly"]
                for item in physical.values()),
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "parent_conditional_retrieval_retained":
            parent_conditional["learned"]["accuracy"] >= 0.95
            and parent_conditional["scale_action_accuracy"] >= 0.95,
        "only_continuous_policy_changed":
            all(
                name.startswith("memory_usage_prior_policy.")
                for name in changed),
        "under_five_minutes": total_seconds <= 300.0,
    }
    gates["accepted"] = all(gates.values())
    stable_threshold = None
    for index, entry in enumerate(history):
        if (
                entry["row_accuracy"] >= 0.95
                and entry["mean_scale"]
                <= inherited_binary["mean_scale"] - 0.15
                and entry["utility"]
                >= inherited_binary["verified_utility"] + 0.01
                and all(
                    (
                        later["row_accuracy"] >= 0.95
                        and later["mean_scale"]
                        <= inherited_binary["mean_scale"] - 0.15
                        and later["utility"]
                        >= inherited_binary["verified_utility"] + 0.01
                    )
                    for later in history[index:])):
            stable_threshold = entry["step"]
            break
    report = {
        "schema": "unified-controller-continuous-usage-prior-v1",
        "configuration": {
            **vars(args),
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out) if args.checkpoint_out else None),
            "report": str(args.report),
        },
        "model_configuration": payload["model_configuration"],
        "preflight": preflight,
        "history": history,
        "evaluations": evaluations,
        "feature_shuffled_four_rows": feature_shuffled,
        "value_corrupted_four_rows": value_corrupted,
        "physical": physical,
        "parent_conditional_retention": parent_conditional,
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
                if stable_threshold is not None and not args.cost_only
                else 0 if stable_threshold is not None else None),
        },
        "gates": gates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": payload["model_configuration"],
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
    print(json.dumps({
        "preflight": preflight,
        "history": history,
        "evaluations": evaluations,
        "feature_shuffled_four_rows": feature_shuffled,
        "value_corrupted_four_rows": value_corrupted,
        "physical": physical,
        "parent_conditional_retention": parent_conditional,
        "accounting": report["accounting"],
        "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
