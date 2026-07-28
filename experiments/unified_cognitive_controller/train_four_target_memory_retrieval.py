"""Learn continuous retrieval when any of four physical rows may be correct."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
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
    conditional_batch,
    _near_keys,
    evaluate_conditional,
)
from .train_continuous_memory_usage_prior import (
    continuous_batch,
    evaluate_policy as evaluate_parent_continuous,
    load_conditional_controller,
)
from .train_memory_replacement import _select_batch


_CROSSINGS = torch.tensor([
    [0.25, 0.55, 0.80],
    [0.30, 0.40, 0.72],
    # The first four-way curriculum rung keeps a small robust interval before
    # the final regime. Later curricula may narrow 0.58 toward 0.55 after the
    # shared head has learned all four selections without forgetting.
    [0.10, 0.45, 0.58],
    [0.10, 0.30, 0.60],
])
_TOP_USAGE = torch.tensor([0.25, 0.45, 0.70, 0.90])
_ALTERNATING_CROSSING_PATTERN = (
    (-1, 1, -1), (1, -1, 1),
    (-1, 1, -1), (1, -1, 1))
_GROUPED_CROSSING_PATTERN = (
    (-1, -1, -1), (-1, -1, 1),
    (1, 1, 1), (1, 1, 1))


@torch.no_grad()
def four_target_batch(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, heldout: bool,
        target_classes: tuple[int, ...] = (0, 1, 2, 3),
        boundary_shift_range: tuple[float, float] = (0.0, 0.0),
        crossing_jitter_range: tuple[float, float] = (0.0, 0.0),
        crossing_jitter_pattern: tuple[float, float, float] | None = None,
        slope_jitter_range: tuple[float, float] = (0.0, 0.0),
        slope_jitter_pattern: tuple[float, float] | None = None,
        shuffle_features: bool = False,
        corrupt_values: bool = False,
        permute_rows: bool = True) -> dict[str, object]:
    full = _add_context_signatures(
        generate_lifetimes(
            count * 4, 3, seed=seed, heldout=heldout,
            task="binary_mapping", support_trials=1, device=device),
        seed=seed + 10_000_000)
    batches = [
        _select_batch(
            full,
            torch.arange(
                index * count, (index + 1) * count, device=device))
        for index in range(4)
    ]
    _, correct_values, _ = _support(model, batches[0], device=device)
    # A row called "wrong" must also be wrong according to the only judge the
    # learner may observe: its scalar outcome. Independent binary lifetimes
    # collide half the time because they can encode the same mapping. Replaying
    # this same visible support with the opposite verifier outcome produces the
    # behaviorally opposite latent rule without exposing that rule, an action
    # label, or a row identity to the learner.
    opposite_actions = batches[0].correct_actions.clone()
    opposite_actions[:, 0] = 1 - opposite_actions[:, 0]
    counterfactual = replace(
        batches[0], correct_actions=opposite_actions)
    _, wrong_values, _ = _support(
        model, counterfactual, device=device)
    queries = _query_keys(model, batches[0], device=device)
    normalized_queries = torch.nn.functional.normalize(queries, dim=-1)
    generator = torch.Generator().manual_seed(seed + 85_000_000)
    if (
            not target_classes
            or any(index < 0 or index > 3 for index in target_classes)
            or len(set(target_classes)) != len(target_classes)):
        raise ValueError(
            "target_classes must contain unique values from zero to three")
    class_values = torch.tensor(target_classes, device=device)
    target_class = class_values[
        torch.arange(count, device=device) % len(target_classes)]
    target_class = target_class[
        torch.randperm(count, generator=generator).to(device)]

    shift_min, shift_max = boundary_shift_range
    if shift_min > shift_max:
        raise ValueError("boundary shift minimum exceeds maximum")
    boundary_shift = (
        shift_min
        + (shift_max - shift_min)
        * torch.rand(count, generator=generator).to(device))
    crossing_min, crossing_max = crossing_jitter_range
    if crossing_min > crossing_max:
        raise ValueError("crossing jitter minimum exceeds maximum")
    crossing_jitter = (
        crossing_min
        + (crossing_max - crossing_min)
        * torch.rand(count, 3, generator=generator).to(device))
    if crossing_jitter_pattern is not None:
        crossing_pattern = torch.tensor(
            crossing_jitter_pattern, device=device)
        if crossing_pattern.ndim == 2:
            if crossing_pattern.shape != (4, 3):
                raise ValueError(
                    "class-specific crossing pattern must have shape [4, 3]")
            crossing_pattern = crossing_pattern[target_class]
        elif crossing_pattern.shape != (3,):
            raise ValueError("crossing pattern must have shape [3]")
        crossing_jitter = crossing_jitter * crossing_pattern
    crossings = (
        _CROSSINGS.to(device)[target_class]
        + boundary_shift.unsqueeze(-1)
        + crossing_jitter).sort(dim=-1).values
    if bool(
            (crossings <= 0.0).any()
            or (crossings >= 1.0).any()
            or (crossings[:, 1:] <= crossings[:, :-1]).any()):
        raise ValueError(
            "boundary shift produces invalid or unordered crossings")
    top_usage = _TOP_USAGE.to(device)[target_class]
    top_slope = top_usage.log()
    slope_min, slope_max = slope_jitter_range
    if slope_min > slope_max:
        raise ValueError("slope jitter minimum exceeds maximum")
    slope_jitter = (
        slope_min
        + (slope_max - slope_min)
        * torch.rand(count, 2, generator=generator).to(device))
    if slope_jitter_pattern is not None:
        slope_pattern = torch.tensor(
            slope_jitter_pattern, device=device)
        if slope_pattern.ndim == 2:
            if slope_pattern.shape != (4, 2):
                raise ValueError(
                    "class-specific slope pattern must have shape [4, 2]")
            slope_pattern = slope_pattern[target_class]
        elif slope_pattern.shape != (2,):
            raise ValueError("slope pattern must have shape [2]")
        slope_jitter = slope_jitter * slope_pattern
    middle_ratio = 2.0 / 3.0 + slope_jitter[:, 0]
    lower_ratio = 1.0 / 3.0 + slope_jitter[:, 1]
    if bool(
            (middle_ratio >= 1.0).any()
            or (lower_ratio <= 0.0).any()
            or (middle_ratio <= lower_ratio).any()):
        raise ValueError("slope jitter produces unordered line slopes")
    slopes = torch.stack((
        top_slope,
        top_slope * middle_ratio,
        top_slope * lower_ratio,
        torch.zeros_like(top_slope),
    ), dim=-1)
    cosine = torch.ones(count, 4, device=device)
    for index in range(3):
        cosine[:, index + 1] = (
            cosine[:, index]
            + (slopes[:, index] - slopes[:, index + 1])
            * crossings[:, index])
    usage = slopes.exp()

    keys = torch.empty(count, 4, model.width, device=device)
    keys[:, 0] = normalized_queries
    for index in range(1, 4):
        seed_key = _near_keys(queries, 0.40, generator)
        orthogonal = seed_key - (
            seed_key * normalized_queries).sum(-1, keepdim=True) \
            * normalized_queries
        orthogonal = torch.nn.functional.normalize(orthogonal, dim=-1)
        keys[:, index] = (
            cosine[:, index:index + 1] * normalized_queries
            + torch.sqrt(
                1.0 - cosine[:, index:index + 1].square())
            * orthogonal)

    values = torch.empty_like(keys)
    for row in range(4):
        correct = target_class == row
        values[correct, row] = correct_values[correct]
        for other_row in range(4):
            if other_row == row:
                continue
            mask = target_class == other_row
            values[mask, row] = wrong_values[mask]
    if corrupt_values:
        values = values.roll(1, dims=1)

    if permute_rows:
        random = torch.rand(count, 4, generator=generator).to(device)
        permutation = random.argsort(dim=-1)
    else:
        permutation = torch.arange(
            4, device=device).expand(count, -1)
    gather_width = permutation.unsqueeze(-1).expand(-1, -1, model.width)
    keys = torch.gather(keys, 1, gather_width)
    values = torch.gather(values, 1, gather_width)
    usage = torch.gather(usage, 1, permutation)
    target_slot = (
        permutation == target_class.unsqueeze(-1)).to(torch.long).argmax(-1)

    normalized_keys = torch.nn.functional.normalize(keys, dim=-1)
    content = torch.einsum(
        "bw,bkw->bk", normalized_queries, normalized_keys)
    scores, order = content.topk(2, dim=-1)
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
        "target_class": target_class,
        "target_slot": target_slot,
        "boundary_shift": boundary_shift,
        "crossings": crossings,
        "crossing_jitter": crossing_jitter,
        "slope_jitter": slope_jitter,
        "generated_contexts": count * 4,
    }


@torch.no_grad()
def behavioral_row_outcomes(
        model: UnifiedCognitiveController,
        data: dict[str, object], *,
        device: torch.device) -> torch.Tensor:
    """Score every physical row using only the environment verifier."""
    outcomes = []
    for row in range(4):
        outcomes.append(_outcomes(
            model, data["target_batch"], data["values"][:, row],
            device=device).float())
    return torch.stack(outcomes, dim=-1)


def policy_features(data: dict[str, object]) -> torch.Tensor:
    return data.get("policy_features", data["features"])


def scale_interval_loss(
        predicted: torch.Tensor, allowed: torch.Tensor,
        candidate_scales: torch.Tensor) -> torch.Tensor:
    """Penalize only scales outside an empirically allowed action region."""
    if allowed.ndim != 2 or allowed.shape[0] != predicted.shape[0]:
        raise ValueError("allowed scales must have shape [batch, candidates]")
    if allowed.shape[1] != candidate_scales.numel():
        raise ValueError("candidate scale count does not match allowed mask")
    valid = allowed.any(-1)
    if not bool(valid.any()):
        raise RuntimeError("no allowed scale was observed")
    expanded = candidate_scales.unsqueeze(0).expand_as(allowed)
    lower = expanded.masked_fill(~allowed, float("inf")).min(-1).values
    upper = expanded.masked_fill(~allowed, float("-inf")).max(-1).values
    below = torch.relu(lower[valid] - predicted[valid])
    above = torch.relu(predicted[valid] - upper[valid])
    return (below.square() + above.square()).mean()


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
def evaluate_four_target(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device, scale_cost: float,
        boundary_shift_range: tuple[float, float] = (0.0, 0.0),
        crossing_jitter_range: tuple[float, float] = (0.0, 0.0),
        crossing_jitter_pattern: tuple[float, float, float] | None = None,
        slope_jitter_range: tuple[float, float] = (0.0, 0.0),
        slope_jitter_pattern: tuple[float, float] | None = None,
        shuffle_features: bool = False,
        corrupt_values: bool = False,
        permute_rows: bool = True) -> dict[str, object]:
    data = four_target_batch(
        model, count=count, seed=seed, device=device, heldout=True,
        boundary_shift_range=boundary_shift_range,
        crossing_jitter_range=crossing_jitter_range,
        crossing_jitter_pattern=crossing_jitter_pattern,
        slope_jitter_range=slope_jitter_range,
        slope_jitter_pattern=slope_jitter_pattern,
        shuffle_features=shuffle_features,
        corrupt_values=corrupt_values, permute_rows=permute_rows)
    learned = model.memory_usage_prior_probability(policy_features(data))
    grid = torch.linspace(0.0, 1.0, 9, device=device)
    policies = {"learned": learned}
    policies.update({
        f"fixed_{float(scale):.3f}":
            torch.full_like(learned, float(scale))
        for scale in grid
    })
    report: dict[str, object] = {}
    for name, scales in policies.items():
        selected, values = select_rows(data, scales)
        outcomes = _outcomes(
            model, data["target_batch"], values, device=device).float()
        correct = selected == data["target_slot"]
        report[name] = {
            "visual_accuracy": float(outcomes.mean()),
            "row_accuracy": float(correct.float().mean()),
            "class_row_accuracy": [
                float(correct[data["target_class"] == index].float().mean())
                for index in range(4)
            ],
            "mean_scale": float(scales.mean()),
            "verified_utility": float(
                outcomes.mean() - scale_cost * scales.mean()),
        }
    fixed = [
        item["row_accuracy"] for name, item in report.items()
        if name.startswith("fixed_")]
    report["best_fixed_row_accuracy"] = max(fixed)
    return report


@torch.no_grad()
def physical_audit(
        model: UnifiedCognitiveController, *, count: int, seed: int,
        device: torch.device,
        boundary_shift_range: tuple[float, float] = (0.0, 0.0),
        crossing_jitter_range: tuple[float, float] = (0.0, 0.0),
        crossing_jitter_pattern: tuple[float, float, float] | None = None,
        slope_jitter_range: tuple[float, float] = (0.0, 0.0),
        slope_jitter_pattern: tuple[float, float] | None = None,
        ) -> dict[str, object]:
    data = four_target_batch(
        model, count=count, seed=seed, device=device, heldout=True,
        boundary_shift_range=boundary_shift_range,
        crossing_jitter_range=crossing_jitter_range,
        crossing_jitter_pattern=crossing_jitter_pattern,
        slope_jitter_range=slope_jitter_range,
        slope_jitter_pattern=slope_jitter_pattern)
    scales = model.memory_usage_prior_probability(policy_features(data))
    reads = []
    correct = 0
    exact_reloads = 0
    with tempfile.TemporaryDirectory(
            prefix="four-target-retrieval-") as root:
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
            correct += int(torch.equal(
                read.squeeze(0), data["values"][
                    index, data["target_slot"][index]]))
    outcomes = _outcomes(
        model, data["target_batch"], torch.cat(reads),
        device=device).float()
    return {
        "banks": count,
        "visual_accuracy": float(outcomes.mean()),
        "row_accuracy": correct / count,
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
    parser.add_argument("--seed", type=int, default=17800)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-count", type=int, default=1024)
    parser.add_argument("--physical-count", type=int, default=128)
    parser.add_argument("--retention-count", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--exploration-std", type=float, default=0.5)
    parser.add_argument("--scale-cost", type=float, default=0.02)
    parser.add_argument("--critic-hidden", type=int, default=0)
    parser.add_argument("--critic-learning-rate", type=float, default=0.01)
    parser.add_argument("--critic-warmup", type=int, default=8)
    parser.add_argument(
        "--paired-delta", type=float, default=0.0,
        help="positive latent offset enables a verified low/high horse race")
    parser.add_argument(
        "--es-delta", type=float, default=0.0,
        help="positive latent offset enables a five-candidate evolution step")
    parser.add_argument(
        "--verified-imitation-candidates", type=int, default=0,
        help=(
            "explore this many uniformly spaced scales and imitate the "
            "center of scales whose retrieved value earns verifier reward"))
    parser.add_argument(
        "--imitation-target", choices=("interval", "center"),
        default="interval",
        help=(
            "stop anywhere in the verified successful interval, or regress "
            "to its outcome-derived center"))
    parser.add_argument(
        "--imitation-replay-updates", type=int, default=1,
        help=(
            "optimizer steps taken from each verified-imitation batch; "
            "extra passes consume compute and count as replay, not experience"))
    parser.add_argument(
        "--accumulate-imitation-replay", action="store_true",
        help=(
            "retain verified feature/interval pairs from earlier environment "
            "steps and optimize their union"))
    parser.add_argument(
        "--parent-distillation-weight", type=float, default=0.0,
        help=(
            "functional-rehearsal weight for preserving the inherited "
            "continuous and conditional retrieval policies"))
    parser.add_argument("--parent-distillation-count", type=int, default=128)
    parser.add_argument("--shuffle-rewards", action="store_true")
    parser.add_argument("--reset-policy", action="store_true")
    parser.add_argument(
        "--training-classes", default="0,1,2,3",
        help="comma-separated private generator classes used in training")
    parser.add_argument("--training-shift-min", type=float, default=0.0)
    parser.add_argument("--training-shift-max", type=float, default=0.0)
    parser.add_argument("--training-crossing-jitter-min", type=float, default=0.0)
    parser.add_argument("--training-crossing-jitter-max", type=float, default=0.0)
    parser.add_argument("--training-slope-jitter-min", type=float, default=0.0)
    parser.add_argument("--training-slope-jitter-max", type=float, default=0.0)
    parser.add_argument(
        "--structured-shape-curriculum", action="store_true",
        help=(
            "cycle random deformations with two hard class-specific sign "
            "patterns, always inside the configured training magnitudes"))
    parser.add_argument("--transfer-negative-min", type=float)
    parser.add_argument("--transfer-negative-max", type=float)
    parser.add_argument("--transfer-positive-min", type=float)
    parser.add_argument("--transfer-positive-max", type=float)
    parser.add_argument("--transfer-shape-crossing-min", type=float)
    parser.add_argument("--transfer-shape-crossing-max", type=float)
    parser.add_argument("--transfer-shape-slope-min", type=float)
    parser.add_argument("--transfer-shape-slope-max", type=float)
    parser.add_argument(
        "--expand-usage-residual-hidden", type=int, default=0,
        help=(
            "add a zero-output residual branch of this width and train it "
            "while freezing the inherited usage-prior policy"))
    parser.add_argument(
        "--usage-residual-features", type=int, default=4,
        choices=(4, 12),
        help="legacy four features or full sorted four-row evidence")
    parser.add_argument(
        "--expand-usage-proposer-hidden", type=int, default=0,
        help=(
            "add a zero-effect relational rank-exchange proposer of this "
            "width and train it while freezing the inherited policy"))
    parser.add_argument(
        "--ablate-proposer-credit-loss", action="store_true",
        help=(
            "diagnostic control: withhold verified candidate credit from "
            "the selector while leaving architecture and experience fixed"))
    args = parser.parse_args()
    if sum((
            args.paired_delta > 0.0,
            args.es_delta > 0.0,
            args.critic_hidden > 0,
            args.verified_imitation_candidates > 0)) > 1:
        parser.error(
            "paired, evolution, critic and verified-imitation modes "
            "are exclusive")
    if (
            args.verified_imitation_candidates == 1
            or args.verified_imitation_candidates < 0):
        parser.error("--verified-imitation-candidates must be zero or >= 2")
    if args.imitation_replay_updates < 1:
        parser.error("--imitation-replay-updates must be positive")
    if args.parent_distillation_weight < 0.0:
        parser.error("--parent-distillation-weight must be nonnegative")
    if args.parent_distillation_count < 2:
        parser.error("--parent-distillation-count must be at least two")
    if args.expand_usage_residual_hidden < 0:
        parser.error("--expand-usage-residual-hidden must be nonnegative")
    if args.expand_usage_proposer_hidden < 0:
        parser.error("--expand-usage-proposer-hidden must be nonnegative")
    if (
            args.expand_usage_residual_hidden > 0
            and args.expand_usage_proposer_hidden > 0):
        parser.error("residual and proposer expansion are exclusive")
    if args.reset_policy and args.expand_usage_residual_hidden > 0:
        parser.error("reset-policy and residual expansion are exclusive")
    if args.reset_policy and args.expand_usage_proposer_hidden > 0:
        parser.error("reset-policy and proposer expansion are exclusive")
    if (
            args.expand_usage_proposer_hidden > 0
            and args.imitation_target != "center"):
        parser.error(
            "the relational proposer requires center imitation")
    if (
            args.ablate_proposer_credit_loss
            and args.expand_usage_proposer_hidden <= 0):
        parser.error(
            "proposer-credit ablation requires proposer expansion")
    transfer_values = (
        args.transfer_negative_min, args.transfer_negative_max,
        args.transfer_positive_min, args.transfer_positive_max)
    if any(value is not None for value in transfer_values):
        if any(value is None for value in transfer_values):
            parser.error("all four transfer boundaries are required")
        assert args.transfer_negative_min is not None
        assert args.transfer_negative_max is not None
        assert args.transfer_positive_min is not None
        assert args.transfer_positive_max is not None
        if not (
                args.transfer_negative_min
                <= args.transfer_negative_max
                < args.training_shift_min
                <= args.training_shift_max
                < args.transfer_positive_min
                <= args.transfer_positive_max):
            parser.error(
                "transfer bands must be ordered and disjoint from training")
        transfer_ranges = {
            "negative": (
                args.transfer_negative_min, args.transfer_negative_max),
            "positive": (
                args.transfer_positive_min, args.transfer_positive_max),
        }
    else:
        transfer_ranges = None
    shape_transfer_values = (
        args.transfer_shape_crossing_min,
        args.transfer_shape_crossing_max,
        args.transfer_shape_slope_min,
        args.transfer_shape_slope_max,
    )
    if any(value is not None for value in shape_transfer_values):
        if any(value is None for value in shape_transfer_values):
            parser.error("all four shape-transfer boundaries are required")
        if transfer_ranges is not None:
            parser.error(
                "boundary and shape transfer audits are mutually exclusive")
        assert args.transfer_shape_crossing_min is not None
        assert args.transfer_shape_crossing_max is not None
        assert args.transfer_shape_slope_min is not None
        assert args.transfer_shape_slope_max is not None
        if not (
                max(
                    abs(args.training_crossing_jitter_min),
                    abs(args.training_crossing_jitter_max))
                < args.transfer_shape_crossing_min
                <= args.transfer_shape_crossing_max
                and max(
                    abs(args.training_slope_jitter_min),
                    abs(args.training_slope_jitter_max))
                < args.transfer_shape_slope_min
                <= args.transfer_shape_slope_max):
            parser.error(
                "shape-transfer magnitudes must be positive, ordered, and "
                "strictly outside both training envelopes")
        transfer_variants = {
            "alternating": {
                "crossing_jitter_range": (
                    args.transfer_shape_crossing_min,
                    args.transfer_shape_crossing_max),
                "crossing_jitter_pattern": (
                    _ALTERNATING_CROSSING_PATTERN),
                "slope_jitter_range": (
                    args.transfer_shape_slope_min,
                    args.transfer_shape_slope_max),
                "slope_jitter_pattern": (1, -1),
            },
            "grouped": {
                "crossing_jitter_range": (
                    args.transfer_shape_crossing_min,
                    args.transfer_shape_crossing_max),
                "crossing_jitter_pattern": (
                    _GROUPED_CROSSING_PATTERN),
                "slope_jitter_range": (
                    args.transfer_shape_slope_min,
                    args.transfer_shape_slope_max),
                "slope_jitter_pattern": (-1, 1),
            },
        }
    elif transfer_ranges is not None:
        transfer_variants = {
            name: {"boundary_shift_range": shift_range}
            for name, shift_range in transfer_ranges.items()}
    else:
        transfer_variants = None
    if (
            args.verified_imitation_candidates <= 0
            and args.imitation_replay_updates != 1):
        parser.error(
            "--imitation-replay-updates requires verified imitation")
    training_classes = tuple(
        int(value) for value in args.training_classes.split(","))
    seed_everything(args.seed)
    device = torch.device(args.device)
    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False)
    model_configuration = dict(payload["model_configuration"])
    inherited_residual_hidden = int(
        model_configuration.get(
            "adaptive_memory_usage_prior_residual_hidden", 0))
    inherited_proposer_hidden = int(
        model_configuration.get(
            "adaptive_memory_usage_prior_proposer_hidden", 0))
    if (
            args.expand_usage_residual_hidden > 0
            and inherited_residual_hidden == 0):
        model_configuration[
            "adaptive_memory_usage_prior_residual_hidden"
        ] = args.expand_usage_residual_hidden
        model_configuration[
            "adaptive_memory_usage_prior_residual_features"
        ] = args.usage_residual_features
        model = UnifiedCognitiveController(**model_configuration).to(device)
        missing, unexpected = model.load_state_dict(
            payload["state_dict"], strict=False)
        expected = {
            "memory_usage_prior_residual.0.weight",
            "memory_usage_prior_residual.0.bias",
            "memory_usage_prior_residual.2.weight",
            "memory_usage_prior_residual.2.bias",
        }
        if set(missing) != expected or unexpected:
            raise ValueError(
                f"unexpected usage-residual mismatch: "
                f"{missing=}, {unexpected=}")
    elif (
            args.expand_usage_proposer_hidden > 0
            and inherited_proposer_hidden == 0):
        model_configuration[
            "adaptive_memory_usage_prior_proposer_hidden"
        ] = args.expand_usage_proposer_hidden
        model = UnifiedCognitiveController(**model_configuration).to(device)
        missing, unexpected = model.load_state_dict(
            payload["state_dict"], strict=False)
        expected = {
            "memory_usage_prior_proposer.0.weight",
            "memory_usage_prior_proposer.0.bias",
            "memory_usage_prior_proposer.2.weight",
            "memory_usage_prior_proposer.2.bias",
        }
        if set(missing) != expected or unexpected:
            raise ValueError(
                f"unexpected usage-proposer mismatch: "
                f"{missing=}, {unexpected=}")
    else:
        if (
                args.expand_usage_residual_hidden > 0
                and inherited_residual_hidden
                != args.expand_usage_residual_hidden):
            raise ValueError(
                "requested residual width differs from checkpoint")
        if (
                args.expand_usage_proposer_hidden > 0
                and inherited_proposer_hidden
                != args.expand_usage_proposer_hidden):
            raise ValueError(
                "requested proposer width differs from checkpoint")
        model = UnifiedCognitiveController(**model_configuration).to(device)
        model.load_state_dict(payload["state_dict"])
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
    if args.expand_usage_residual_hidden > 0:
        assert model.memory_usage_prior_residual is not None
        trainable_parameters = list(
            model.memory_usage_prior_residual.parameters())
    elif args.expand_usage_proposer_hidden > 0:
        if args.verified_imitation_candidates <= 0:
            parser.error(
                "the relational proposer currently requires "
                "verified-imitation training")
        assert model.memory_usage_prior_proposer is not None
        trainable_parameters = list(
            model.memory_usage_prior_proposer.parameters())
    else:
        trainable_parameters = list(
            model.memory_usage_prior_policy.parameters())
    for parameter in trainable_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=args.learning_rate)
    use_relational_candidates = (
        args.expand_usage_proposer_hidden > 0)
    parent_features = parent_allowed = parent_candidate_scales = None
    rehearsal_contexts = 0
    if args.parent_distillation_weight > 0.0:
        # Preserve the inherited *retrieval decision*, not its incidental
        # numeric scale. Every scale that selects the same row remains legal,
        # leaving plasticity inside the old skill's behavioral equivalence
        # class.
        parent_continuous_data = continuous_batch(
            model, count=args.parent_distillation_count, rows=4,
            seed=args.seed + 87_000_000, device=device, heldout=False,
            difficulty="separated")
        parent_conditional_data = conditional_batch(
            model, count=args.parent_distillation_count,
            seed=args.seed + 88_000_000, device=device, heldout=False)
        parent_features = torch.cat((
            policy_features(parent_continuous_data),
            policy_features(parent_conditional_data)), dim=0)
        with torch.no_grad():
            continuous_teacher_scales = (
                model.memory_usage_prior_probability(
                    policy_features(parent_continuous_data)))
            continuous_teacher_selected, _ = select_rows(
                parent_continuous_data, continuous_teacher_scales)
            conditional_teacher_scales = (
                model.memory_usage_prior_probability(
                    policy_features(parent_conditional_data)))
            conditional_teacher_actions = (
                conditional_teacher_scales >= 0.5)
            parent_candidate_scales = torch.linspace(
                0.0, 1.0, 101, device=device)
            continuous_allowed_columns = []
            conditional_allowed_columns = []
            for scale in parent_candidate_scales:
                selected, _ = select_rows(
                    parent_continuous_data,
                    torch.full_like(continuous_teacher_scales, scale))
                continuous_allowed_columns.append(
                    selected == continuous_teacher_selected)
                conditional_allowed_columns.append(
                    torch.where(
                        conditional_teacher_actions,
                        torch.full_like(
                            conditional_teacher_actions, scale >= 0.51),
                        torch.full_like(
                            conditional_teacher_actions, scale <= 0.49)))
            parent_allowed = torch.cat((
                torch.stack(continuous_allowed_columns, dim=-1),
                torch.stack(conditional_allowed_columns, dim=-1)), dim=0)
        rehearsal_contexts = (
            int(parent_continuous_data["generated_contexts"])
            + int(parent_conditional_data["generated_contexts"]))
    critic = None
    critic_optimizer = None
    if args.critic_hidden > 0:
        critic = torch.nn.Sequential(
            torch.nn.Linear(5, args.critic_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(args.critic_hidden, 1),
        ).to(device)
        critic_optimizer = torch.optim.Adam(
            critic.parameters(), lr=args.critic_learning_rate)
    preflight = evaluate_four_target(
        model, count=min(args.test_count, 256),
        seed=args.seed + 90_000_000, device=device,
        scale_cost=args.scale_cost)
    started = time.perf_counter()
    reward_generator = torch.Generator(device=device).manual_seed(
        args.seed + 86_000_000)
    history = []
    contexts = verifier_bits = 0
    optimizer_updates = replayed_examples = 0
    imitation_feature_replay: list[torch.Tensor] = []
    imitation_allowed_replay: list[torch.Tensor] = []
    for step in range(1, args.steps + 1):
        crossing_jitter_range = (
            args.training_crossing_jitter_min,
            args.training_crossing_jitter_max)
        slope_jitter_range = (
            args.training_slope_jitter_min,
            args.training_slope_jitter_max)
        crossing_jitter_pattern = None
        slope_jitter_pattern = None
        if args.structured_shape_curriculum and step % 3 != 1:
            # Keep the magnitudes strictly inside the declared training
            # envelope while ensuring narrow, decision-changing intervals are
            # not left to chance in a small sample-efficiency run.
            crossing_magnitude = max(
                abs(args.training_crossing_jitter_min),
                abs(args.training_crossing_jitter_max))
            slope_magnitude = max(
                abs(args.training_slope_jitter_min),
                abs(args.training_slope_jitter_max))
            crossing_jitter_range = (0.0, crossing_magnitude)
            slope_jitter_range = (0.0, slope_magnitude)
            if step % 3 == 2:
                crossing_jitter_pattern = (
                    _ALTERNATING_CROSSING_PATTERN)
                slope_jitter_pattern = (1, -1)
            else:
                crossing_jitter_pattern = _GROUPED_CROSSING_PATTERN
                slope_jitter_pattern = (-1, 1)
        data = four_target_batch(
            model, count=args.batch_size,
            seed=args.seed * 1_000_000 + step,
            device=device, heldout=False,
            target_classes=training_classes,
            boundary_shift_range=(
                args.training_shift_min, args.training_shift_max),
            crossing_jitter_range=crossing_jitter_range,
            crossing_jitter_pattern=crossing_jitter_pattern,
            slope_jitter_range=slope_jitter_range,
            slope_jitter_pattern=slope_jitter_pattern)
        contexts += int(data["generated_contexts"])
        mean = model.memory_usage_prior_logits(policy_features(data))
        critic_loss_value = None
        if args.verified_imitation_candidates > 0:
            # Search is expressed entirely in the controller's generic scalar
            # action space. Each distinct retrieved value is verified once;
            # candidate success is then reconstructed from those outcome bits.
            # No target row, private scale interval, or correct action label
            # enters the loss. The controller is penalized only while its
            # prediction lies outside the empirically successful region.
            if use_relational_candidates:
                candidate_scales = model.memory_usage_prior_candidates(
                    policy_features(data))
            else:
                candidate_scales = torch.linspace(
                    0.0, 1.0, args.verified_imitation_candidates,
                    device=device)
            row_outcomes = behavioral_row_outcomes(
                model, data, device=device)
            verifier_bits += row_outcomes.numel()
            if args.shuffle_rewards:
                row_outcomes = row_outcomes.flatten()[
                    torch.randperm(
                        row_outcomes.numel(),
                        generator=reward_generator, device=device)
                ].reshape_as(row_outcomes)
            candidate_outcomes = []
            candidate_count = (
                candidate_scales.shape[1]
                if candidate_scales.ndim == 2
                else candidate_scales.numel())
            for candidate_index in range(candidate_count):
                scale = (
                    candidate_scales[:, candidate_index]
                    if candidate_scales.ndim == 2
                    else candidate_scales[candidate_index])
                selected, _ = select_rows(
                    data, torch.full_like(mean, scale)
                    if scale.ndim == 0 else scale)
                candidate_outcomes.append(
                    row_outcomes.gather(
                        1, selected.unsqueeze(-1)).squeeze(-1))
            candidate_outcomes = torch.stack(
                candidate_outcomes, dim=-1)
            successful = candidate_outcomes > 0.5
            successful_count = successful.sum(-1)
            valid = successful_count > 0
            if not bool(valid.any()):
                raise RuntimeError(
                    "verified imitation found no successful action")
            if args.accumulate_imitation_replay:
                imitation_feature_replay.append(
                    policy_features(data)[valid].detach())
                imitation_allowed_replay.append(
                    successful[valid].detach())
                replay_features = torch.cat(
                    imitation_feature_replay, dim=0)
                replay_allowed = torch.cat(
                    imitation_allowed_replay, dim=0)
            else:
                replay_features = policy_features(data)[valid]
                replay_allowed = successful[valid]
            current_examples = int(valid.sum())
            old_examples = replay_features.shape[0] - current_examples
            for replay_update in range(args.imitation_replay_updates):
                replay_mean = model.memory_usage_prior_logits(
                    replay_features)
                predicted_scale = torch.sigmoid(replay_mean)
                if use_relational_candidates:
                    # The learned no-op gate is part of the actual action
                    # policy, so optimize the final probability rather than
                    # the inherited logits.
                    predicted_scale = (
                        model.memory_usage_prior_probability(replay_features))
                    replay_candidate_scales = (
                        model.memory_usage_prior_candidates(replay_features))
                else:
                    replay_candidate_scales = candidate_scales
                if args.imitation_target == "center":
                    replay_target = (
                        replay_allowed.to(predicted_scale.dtype)
                        * (
                            replay_candidate_scales
                            if replay_candidate_scales.ndim == 2
                            else replay_candidate_scales.unsqueeze(0))
                    ).sum(-1) / replay_allowed.sum(-1)
                    loss = torch.nn.functional.mse_loss(
                        predicted_scale, replay_target.detach())
                    if (
                            use_relational_candidates
                            and not args.ablate_proposer_credit_loss):
                        assert model.memory_usage_prior_proposer is not None
                        proposal_features, _ = (
                            model.memory_usage_prior_proposal_features(
                                replay_features))
                        proposal_logits = (
                            model.memory_usage_prior_proposer(
                                proposal_features)[:, :4])
                        verified_distribution = (
                            replay_allowed.to(proposal_logits.dtype)
                            / replay_allowed.sum(
                                -1, keepdim=True).clamp_min(1))
                        # Candidate success comes exclusively from scalar
                        # verifier outcomes.  This trains the selector even
                        # while its exact no-op opening is still closed,
                        # breaking the otherwise circular credit path.
                        loss = loss - (
                            verified_distribution.detach()
                            * proposal_logits.log_softmax(-1)
                        ).sum(-1).mean()
                else:
                    loss = scale_interval_loss(
                        predicted_scale, replay_allowed, candidate_scales)
                if parent_features is not None:
                    assert parent_allowed is not None
                    assert parent_candidate_scales is not None
                    parent_prediction = (
                        model.memory_usage_prior_probability(parent_features))
                    loss = loss + args.parent_distillation_weight * (
                        scale_interval_loss(
                            parent_prediction, parent_allowed,
                            parent_candidate_scales))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, 1.0)
                optimizer.step()
                optimizer_updates += 1
                if replay_update > 0:
                    replayed_examples += replay_features.shape[0]
                else:
                    replayed_examples += old_examples
                if parent_features is not None:
                    replayed_examples += parent_features.shape[0]
        elif args.es_delta > 0.0:
            offsets = mean.new_tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
            offsets = offsets * args.es_delta
            candidate_scales = torch.sigmoid(
                mean.unsqueeze(-1) + offsets.unsqueeze(0))
            candidate_outcomes = []
            with torch.no_grad():
                for candidate in range(offsets.numel()):
                    _, values = select_rows(
                        data, candidate_scales[:, candidate])
                    candidate_outcomes.append(_outcomes(
                        model, data["target_batch"], values,
                        device=device).float())
            candidate_outcomes = torch.stack(candidate_outcomes, dim=-1)
            verifier_bits += candidate_outcomes.numel()
            if args.shuffle_rewards:
                candidate_outcomes = candidate_outcomes.flatten()[
                    torch.randperm(
                        candidate_outcomes.numel(),
                        generator=reward_generator, device=device)
                ].reshape_as(candidate_outcomes)
            candidate_utility = (
                candidate_outcomes
                - args.scale_cost * candidate_scales.detach())
            centered = (
                candidate_utility
                - candidate_utility.mean(dim=-1, keepdim=True))
            verified_direction = (
                centered * offsets.unsqueeze(0)).mean(dim=-1)
            verified_direction = verified_direction / offsets.square().mean()
            loss = -(verified_direction.detach() * mean).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, 1.0)
            optimizer.step()
            optimizer_updates += 1
        elif args.paired_delta > 0.0:
            low_scale = torch.sigmoid(mean - args.paired_delta)
            high_scale = torch.sigmoid(mean + args.paired_delta)
            with torch.no_grad():
                _, low_values = select_rows(data, low_scale)
                _, high_values = select_rows(data, high_scale)
                low_outcomes = _outcomes(
                    model, data["target_batch"], low_values,
                    device=device).float()
                high_outcomes = _outcomes(
                    model, data["target_batch"], high_values,
                    device=device).float()
            verifier_bits += low_outcomes.numel() + high_outcomes.numel()
            paired_outcomes = torch.stack((
                low_outcomes, high_outcomes), dim=-1)
            if args.shuffle_rewards:
                paired_outcomes = paired_outcomes.flatten()[
                    torch.randperm(
                        paired_outcomes.numel(),
                        generator=reward_generator, device=device)
                ].reshape_as(paired_outcomes)
            verified_direction = (
                paired_outcomes[:, 1] - paired_outcomes[:, 0])
            loss = (
                -(verified_direction.detach() * mean).mean()
                + args.scale_cost * torch.sigmoid(mean).mean())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, 1.0)
            optimizer.step()
            optimizer_updates += 1
        else:
            distribution = torch.distributions.Normal(
                mean, torch.full_like(mean, args.exploration_std))
            latent_action = distribution.rsample()
            scales = torch.sigmoid(latent_action)
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
        if (
                args.verified_imitation_candidates <= 0
                and
                args.es_delta <= 0.0
                and args.paired_delta <= 0.0
                and critic is None):
            advantage = training_outcomes - training_outcomes.mean()
            policy_loss = -(
                advantage.detach()
                * distribution.log_prob(latent_action.detach())).mean()
            loss = policy_loss + args.scale_cost * scales.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, 1.0)
            optimizer.step()
            optimizer_updates += 1
        elif (
                args.verified_imitation_candidates <= 0
                and args.es_delta <= 0.0
                and args.paired_delta <= 0.0):
            assert critic_optimizer is not None
            critic_input = torch.cat((
                data["features"], scales.detach().unsqueeze(-1)), dim=-1)
            critic_logits = critic(critic_input).squeeze(-1)
            critic_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                critic_logits, training_outcomes)
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            critic_optimizer.step()
            critic_loss_value = float(critic_loss.detach())
            if step > args.critic_warmup:
                for parameter in critic.parameters():
                    parameter.requires_grad_(False)
                deterministic_scale = torch.sigmoid(mean)
                predicted_success = torch.sigmoid(critic(torch.cat((
                    data["features"],
                    deterministic_scale.unsqueeze(-1)), dim=-1))).squeeze(-1)
                loss = (
                    -predicted_success.mean()
                    + args.scale_cost * deterministic_scale.mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, 1.0)
                optimizer.step()
                optimizer_updates += 1
                for parameter in critic.parameters():
                    parameter.requires_grad_(True)
        if (
                args.steps <= 16
                or step == 1
                or step % 8 == 0
                or step == args.steps):
            prefix = evaluate_four_target(
                model, count=min(args.test_count, 256),
                seed=args.seed + 90_000_000, device=device,
                scale_cost=args.scale_cost)
            entry = {
                "step": step,
                "row_accuracy": prefix["learned"]["row_accuracy"],
                "minimum_class_accuracy": min(
                    prefix["learned"]["class_row_accuracy"]),
                "visual_accuracy": prefix["learned"]["visual_accuracy"],
                "mean_scale": prefix["learned"]["mean_scale"],
                "critic_loss": critic_loss_value,
                "elapsed_seconds": time.perf_counter() - started,
            }
            if transfer_variants is not None:
                transfer_prefixes = [
                    evaluate_four_target(
                        model, count=min(args.test_count, 256),
                        seed=args.seed + 90_100_000 + index,
                        device=device, scale_cost=args.scale_cost,
                        **variant)
                    for index, variant
                    in enumerate(transfer_variants.values())
                ]
                entry["transfer_row_accuracy"] = min(
                    report["learned"]["row_accuracy"]
                    for report in transfer_prefixes)
                entry["transfer_minimum_class_accuracy"] = min(
                    min(report["learned"]["class_row_accuracy"])
                    for report in transfer_prefixes)
            history.append(entry)
    training_seconds = time.perf_counter() - started
    held_out = evaluate_four_target(
        model, count=args.test_count, seed=args.seed + 91_000_000,
        device=device, scale_cost=args.scale_cost)
    feature_shuffled = evaluate_four_target(
        model, count=args.test_count, seed=args.seed + 91_000_000,
        device=device, scale_cost=args.scale_cost,
        shuffle_features=True)
    value_corrupted = evaluate_four_target(
        model, count=args.test_count, seed=args.seed + 91_000_000,
        device=device, scale_cost=args.scale_cost,
        corrupt_values=True)
    unpermuted = evaluate_four_target(
        model, count=args.test_count, seed=args.seed + 91_000_000,
        device=device, scale_cost=args.scale_cost,
        permute_rows=False)
    physical = physical_audit(
        model, count=args.physical_count,
        seed=args.seed + 92_000_000, device=device)
    transfer_evaluation = None
    if transfer_variants is not None:
        transfer_evaluation = {}
        for index, (name, variant) in enumerate(
                transfer_variants.items()):
            seed = args.seed + 97_000_000 + index * 10_000
            transfer_evaluation[name] = {
                "evaluation_parameters": variant,
                "held_out": evaluate_four_target(
                    model, count=args.test_count, seed=seed,
                    device=device, scale_cost=args.scale_cost,
                    **variant),
                "feature_shuffled": evaluate_four_target(
                    model, count=args.test_count, seed=seed,
                    device=device, scale_cost=args.scale_cost,
                    **variant,
                    shuffle_features=True),
                "value_corrupted": evaluate_four_target(
                    model, count=args.test_count, seed=seed,
                    device=device, scale_cost=args.scale_cost,
                    **variant,
                    corrupt_values=True),
                "unpermuted_rows": evaluate_four_target(
                    model, count=args.test_count, seed=seed,
                    device=device, scale_cost=args.scale_cost,
                    **variant,
                    permute_rows=False),
                "physical": physical_audit(
                    model, count=args.physical_count,
                    seed=seed + 1_000, device=device,
                    **variant),
            }
    parent_continuous = evaluate_parent_continuous(
        model, count=args.test_count, rows=4,
        seed=args.seed + 93_000_000, device=device,
        scale_cost=0.30, difficulty="separated")
    parent_conditional = evaluate_conditional(
        model, count=args.test_count,
        seed=args.seed + 94_000_000, device=device)
    binary = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 95_000_000, device=device,
        task="binary_mapping", feedback_trials=1)
    four_rule = evaluate(
        model, count=args.retention_count, trials=6,
        seed=args.seed + 96_000_000, device=device,
        task="four_rule", feedback_trials=2)
    changed = [
        name for name, value in model.state_dict().items()
        if not torch.equal(initial[name], value.detach().cpu())]
    total_seconds = time.perf_counter() - started
    learned = held_out["learned"]
    shuffled = feature_shuffled["learned"]
    corrupted = value_corrupted["learned"]
    gates = {
        "row_accuracy_at_least_90":
            learned["row_accuracy"] >= 0.90,
        "every_class_at_least_85":
            min(learned["class_row_accuracy"]) >= 0.85,
        "best_fixed_scale_at_most_35":
            held_out["best_fixed_row_accuracy"] <= 0.35,
        "feature_shuffle_costs_20_points":
            learned["row_accuracy"] >= shuffled["row_accuracy"] + 0.20,
        "values_are_causal":
            learned["visual_accuracy"]
            >= corrupted["visual_accuracy"] + 0.15,
        "row_permutation_invariant":
            unpermuted["learned"]["row_accuracy"] >= 0.90,
        "physical_accuracy_at_least_90":
            physical["row_accuracy"] >= 0.90,
        "physical_reload_exact":
            physical["all_banks_reload_exactly"],
        "parent_continuous_retained":
            parent_continuous["continuous"]["row_accuracy"] >= 0.95,
        "parent_conditional_retained":
            parent_conditional["learned"]["accuracy"] >= 0.95,
        "binary_retained": binary["gate"]["accepted"],
        "four_rule_retained": four_rule["gate"]["accepted"],
        "only_policy_changed":
            all(
                name.startswith((
                    "memory_usage_prior_policy.",
                    "memory_usage_prior_residual.",
                    "memory_usage_prior_proposer."))
                for name in changed),
        "under_five_minutes": total_seconds <= 300.0,
    }
    if transfer_evaluation is not None:
        transfer_held = [
            report["held_out"]["learned"]
            for report in transfer_evaluation.values()]
        transfer_shuffled = [
            report["feature_shuffled"]["learned"]
            for report in transfer_evaluation.values()]
        transfer_corrupted = [
            report["value_corrupted"]["learned"]
            for report in transfer_evaluation.values()]
        gates.update({
            "transfer_each_band_at_least_90":
                min(item["row_accuracy"] for item in transfer_held) >= 0.90,
            "transfer_every_class_at_least_85":
                min(
                    min(item["class_row_accuracy"])
                    for item in transfer_held) >= 0.85,
            "transfer_best_fixed_at_most_35":
                max(
                    report["held_out"]["best_fixed_row_accuracy"]
                    for report in transfer_evaluation.values()) <= 0.35,
            "transfer_feature_shuffle_costs_20_points":
                min(
                    held["row_accuracy"] - shuffled["row_accuracy"]
                    for held, shuffled
                    in zip(transfer_held, transfer_shuffled)) >= 0.20,
            "transfer_values_are_causal":
                min(
                    held["visual_accuracy"] - corrupted["visual_accuracy"]
                    for held, corrupted
                    in zip(transfer_held, transfer_corrupted)) >= 0.15,
            "transfer_row_permutation_invariant":
                min(
                    report["unpermuted_rows"]["learned"]["row_accuracy"]
                    for report in transfer_evaluation.values()) >= 0.90,
            "transfer_physical_at_least_90":
                min(
                    report["physical"]["row_accuracy"]
                    for report in transfer_evaluation.values()) >= 0.90,
            "transfer_physical_reload_exact":
                all(
                    report["physical"]["all_banks_reload_exactly"]
                    for report in transfer_evaluation.values()),
        })
    gates["accepted"] = all(gates.values())
    stable_threshold = None
    for index, entry in enumerate(history):
        if (
                (
                    entry["transfer_row_accuracy"] >= 0.90
                    and entry["transfer_minimum_class_accuracy"] >= 0.85
                    if transfer_variants is not None
                    else (
                        entry["row_accuracy"] >= 0.90
                        and entry["minimum_class_accuracy"] >= 0.85))
                and all(
                    (
                        later["transfer_row_accuracy"] >= 0.90
                        and later["transfer_minimum_class_accuracy"] >= 0.85
                        if transfer_variants is not None
                        else (
                            later["row_accuracy"] >= 0.90
                            and later["minimum_class_accuracy"] >= 0.85))
                    for later in history[index:])):
            stable_threshold = entry["step"]
            break
    report = {
        "schema": (
            (
                "unified-controller-four-target-shape-transfer-v1"
                if any(value is not None for value in shape_transfer_values)
                else "unified-controller-four-target-boundary-transfer-v1")
            if transfer_variants is not None
            else "unified-controller-four-target-retrieval-v1"),
        "configuration": {
            **vars(args),
            "checkpoint_in": str(args.checkpoint_in),
            "checkpoint_out": (
                str(args.checkpoint_out) if args.checkpoint_out else None),
            "report": str(args.report),
        },
        "model_configuration": model_configuration,
        "preflight": preflight,
        "history": history,
        "held_out": held_out,
        "feature_shuffled": feature_shuffled,
        "value_corrupted": value_corrupted,
        "unpermuted_rows": unpermuted,
        "physical": physical,
        "transfer_evaluation": transfer_evaluation,
        "retention": {
            "parent_continuous": parent_continuous,
            "parent_conditional": parent_conditional,
            "binary_mapping": binary,
            "four_rule": four_rule,
        },
        "changed_parameters": changed,
        "accounting": {
            "unique_logical_contexts": contexts,
            "unique_verifier_bits": verifier_bits,
            "parent_rehearsal_contexts": rehearsal_contexts,
            "environment_steps": args.steps,
            "optimizer_updates": optimizer_updates,
            "replayed_examples": replayed_examples,
            "training_seconds": training_seconds,
            "total_seconds": total_seconds,
            "stable_updates_to_90_percent": stable_threshold,
            "stable_verifier_bits_to_90_percent": (
                stable_threshold * args.batch_size
                * (
                    4 if args.verified_imitation_candidates > 0
                    else 5 if args.es_delta > 0.0
                    else 2 if args.paired_delta > 0.0
                    else 1)
                if stable_threshold is not None else None),
        },
        "gates": gates,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.checkpoint_out:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "schema": "unified-cognitive-controller-v1",
            "model_configuration": model_configuration,
            "state_dict": model.state_dict(),
            "source_report": str(args.report),
        }, args.checkpoint_out)
    print(json.dumps({
        "preflight": preflight,
        "history": history,
        "held_out": held_out,
        "feature_shuffled": feature_shuffled,
        "value_corrupted": value_corrupted,
        "unpermuted_rows": unpermuted,
        "physical": physical,
        "accounting": report["accounting"],
        "gates": gates,
    }, indent=2))


if __name__ == "__main__":
    main()
