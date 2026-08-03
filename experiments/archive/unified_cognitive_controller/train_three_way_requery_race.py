"""Horse race: reuse two-action values to learn a third ranked memory read."""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
from torch import nn

from .probe_requery_operation import ranked_requery_batch
from .train import seed_everything
from .train_redundancy_transfer import build_transfer_arms
from .train_safe_requery_adaptation import (
    ActionValueHead,
    _load_head,
    head_from_skill_payload,
)
from .train_shadow_compute_advantage import ComputeAdvantageHead
from .verified_skill_store import VerifiedSkillStore


class RankedActionValueHead(nn.Module):
    """Action values for primary, second-ranked, and third-ranked reads."""

    def __init__(self, hidden: int = 64, input_width: int = 4) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q_values(features).argmax(-1)


class FrozenChampionThirdValueHead(nn.Module):
    """Keep the learned binary primitive fixed; learn only the new operation."""

    def __init__(self, source: ComputeAdvantageHead) -> None:
        super().__init__()
        self.encoder = copy.deepcopy(source.network[:3])
        self.old_advantage = copy.deepcopy(source.network[-1])
        hidden = source.network[1].out_features
        self.baseline_value = nn.Linear(hidden, 1)
        self.third_value = nn.Linear(hidden, 1)
        self.log_advantage_scale = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.baseline_value.weight)
        nn.init.constant_(self.baseline_value.bias, 0.5)
        nn.init.zeros_(self.third_value.weight)
        nn.init.constant_(self.third_value.bias, 0.5)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.old_advantage.parameters():
            parameter.requires_grad_(False)

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(features)
        advantage = self.old_advantage(encoded).squeeze(-1)
        baseline = self.baseline_value(encoded).squeeze(-1)
        scaled_advantage = self.log_advantage_scale.exp() * advantage
        third = self.third_value(encoded).squeeze(-1)
        return torch.stack(
            (baseline, baseline + scaled_advantage, third), dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q_values(features).argmax(-1)


class StoredCoreThirdValueHead(nn.Module):
    """Freeze calibrated old Q values; learn a residual for only action three."""

    def __init__(
            self, source: ActionValueHead, *,
            input_width: int = 7, hidden: int = 32) -> None:
        super().__init__()
        self.old_values = copy.deepcopy(source)
        for parameter in self.old_values.parameters():
            parameter.requires_grad_(False)
        self.third_residual = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.third_residual[-1].weight)
        nn.init.zeros_(self.third_residual[-1].bias)

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        old = self.old_values.q_values(features[:, :4])
        third = (
            old.mean(1, keepdim=True)
            + self.third_residual(features))
        return torch.cat((old, third), dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q_values(features).argmax(-1)


class ComposedChampionThirdValueHead(nn.Module):
    """Compose calibrated old values with the stronger champion ordering."""

    def __init__(
            self, values: ActionValueHead,
            champion: ComputeAdvantageHead, *,
            input_width: int = 7, hidden: int = 32) -> None:
        super().__init__()
        self.old_values = copy.deepcopy(values)
        self.champion = copy.deepcopy(champion)
        for module in (self.old_values, self.champion):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.third_residual = nn.Sequential(
            nn.LayerNorm(input_width),
            nn.Linear(input_width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.third_residual[-1].weight)
        nn.init.zeros_(self.third_residual[-1].bias)

    def q_values(self, features: torch.Tensor) -> torch.Tensor:
        old_features = features[:, :4]
        stored = self.old_values.q_values(old_features)
        center = stored.mean(1, keepdim=True)
        magnitude = (
            (stored[:, 1] - stored[:, 0]).abs().unsqueeze(1) / 2)
        direction = torch.where(
            self.champion(old_features).unsqueeze(1) > 0,
            torch.ones_like(magnitude), -torch.ones_like(magnitude))
        old = torch.cat((
            center - direction * magnitude,
            center + direction * magnitude,
        ), dim=1)
        third = center + self.third_residual(features)
        return torch.cat((old, third), dim=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.q_values(features).argmax(-1)


@torch.no_grad()
def expand_action_value_head(source: ActionValueHead) -> RankedActionValueHead:
    """Preserve known action values; initialize the new action symmetrically."""
    hidden = source.network[1].out_features
    expanded = RankedActionValueHead(hidden).to(
        next(source.parameters()).device)
    expanded.network[0].load_state_dict(source.network[0].state_dict())
    expanded.network[1].load_state_dict(source.network[1].state_dict())
    source_output = source.network[-1]
    target_output = expanded.network[-1]
    target_output.weight[:2].copy_(source_output.weight)
    target_output.bias[:2].copy_(source_output.bias)
    target_output.weight[2].copy_(source_output.weight.mean(0))
    target_output.bias[2].copy_(source_output.bias.mean())
    return expanded


@torch.no_grad()
def expand_advantage_head(
        source: ComputeAdvantageHead) -> RankedActionValueHead:
    """Preserve the champion's old decision while adding a neutral third Q."""
    hidden = source.network[1].out_features
    expanded = RankedActionValueHead(hidden).to(
        next(source.parameters()).device)
    expanded.network[0].load_state_dict(source.network[0].state_dict())
    expanded.network[1].load_state_dict(source.network[1].state_dict())
    source_output = source.network[-1]
    target_output = expanded.network[-1]
    # Q0 and Q2 share a neutral reward-scale baseline. Q1-Q0 is exactly the
    # old advantage, so argmax over actions 0/1 preserves every old decision.
    target_output.weight.zero_()
    target_output.bias.fill_(0.5)
    target_output.weight[1].copy_(source_output.weight.squeeze(0))
    target_output.bias[1].add_(source_output.bias.squeeze(0))
    return expanded


def paired_multiaction_improvement(
        incumbent_actions: torch.Tensor,
        challenger_actions: torch.Tensor,
        attempted_actions: torch.Tensor,
        attempted_utilities: torch.Tensor,
        *, action_count: int = 3, z: float = 1.96,
        ) -> dict[str, float]:
    """Paired IPS evidence for uniformly randomized discrete operations."""
    if action_count < 2:
        raise ValueError("action_count must be at least two")
    if not (
            incumbent_actions.shape == challenger_actions.shape
            == attempted_actions.shape == attempted_utilities.shape):
        raise ValueError("logged tensors must have matching shapes")
    centered = attempted_utilities - attempted_utilities.mean()
    paired = action_count * centered * (
        (attempted_actions == challenger_actions).to(centered.dtype)
        - (attempted_actions == incumbent_actions).to(centered.dtype))
    mean = float(paired.mean())
    se = (
        float(paired.std(unbiased=True)) / math.sqrt(paired.numel())
        if paired.numel() > 1 else float("inf"))
    return {
        "estimated_improvement": mean,
        "standard_error": se,
        "lower_95": mean - z * se,
        "upper_95": mean + z * se,
        "records": paired.numel(),
    }


@torch.no_grad()
def evaluate_head(
        head: nn.Module, features: torch.Tensor, utilities: torch.Tensor,
        ) -> dict[str, float]:
    actions = head(features)
    chosen = utilities.gather(1, actions[:, None]).squeeze(1)
    oracle_actions = utilities.argmax(1)
    fixed = utilities.mean(0)
    oracle = utilities.max(1).values.mean()
    return {
        "verified_utility": float(chosen.mean()),
        "oracle_action_accuracy":
            float((actions == oracle_actions).float().mean()),
        "third_read_rate": float((actions == 2).float().mean()),
        "strongest_fixed_utility": float(fixed.max()),
        "oracle_utility": float(oracle),
        "available_oracle_gap": float(oracle - fixed.max()),
    }


class BinaryIncumbentAdapter(nn.Module):
    """Expose an immutable two-operation champion as actions zero/one."""

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return (self.head(features[:, :4]) > 0).long()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--selected-prefix", type=Path, required=True)
    parser.add_argument("--champion-head", type=Path, required=True)
    parser.add_argument("--skill-store", type=Path, required=True)
    parser.add_argument("--skill-id", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=8042)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--test-contexts", type=int, default=2040)
    parser.add_argument("--capacity", type=int, default=6)
    parser.add_argument("--write-threshold", type=float, default=0.5)
    parser.add_argument("--second-cost", type=float, default=0.01)
    parser.add_argument("--third-cost", type=float, default=0.02)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--active-pool-multiplier", type=int, default=4)
    parser.add_argument("--evidence-every", type=int, default=4)
    parser.add_argument("--practical-gain", type=float, default=0.02)
    args = parser.parse_args()
    if args.test_contexts % args.capacity:
        raise ValueError("test-contexts must divide evenly by capacity")
    if (args.batch_size * args.active_pool_multiplier) % args.capacity:
        raise ValueError("candidate pool must divide evenly by capacity")

    seed_everything(args.seed)
    device = torch.device(args.device)
    parent = torch.load(
        args.parent_checkpoint, map_location=device, weights_only=False)
    selected = torch.load(
        args.selected_prefix, map_location=device, weights_only=False)
    controller = build_transfer_arms(
        parent, selected, device=device,
        fresh_seed=args.seed + 1)["selected_experience"]
    champion_source = _load_head(args.champion_head, device)
    champion = BinaryIncumbentAdapter(champion_source).to(device)
    stored = VerifiedSkillStore(args.skill_store).load(
        args.skill_id, device=device)
    source = head_from_skill_payload(stored["payload"], device)
    if not isinstance(source, ActionValueHead):
        raise ValueError("three-way transfer requires action-value source")
    stored_transferred = StoredCoreThirdValueHead(source).to(device)
    composed_transfer = ComposedChampionThirdValueHead(
        source, champion_source).to(device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(args.seed + 90_000_000)
        stored_reset = RankedActionValueHead(
            source.network[1].out_features, input_width=7).to(device)
    arms = {
        "stored_transferred": stored_transferred,
        "composed_transfer": composed_transfer,
        "stored_reset": stored_reset,
    }
    optimizers = {
        name: torch.optim.AdamW(
            head.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        for name, head in arms.items()
    }
    logs: dict[str, list[tuple[torch.Tensor, ...]]] = {
        name: [] for name in arms}
    histories = {name: [] for name in arms}
    evidence = {name: [] for name in arms}
    selections = {name: [] for name in arms}
    costs = torch.tensor(
        [0.0, args.second_cost, args.third_cost], device=device)

    test_features, test_outcomes, _ = ranked_requery_batch(
        controller, count=args.test_contexts, capacity=args.capacity,
        seed=args.seed + 80_000_000, device=device,
        write_threshold=args.write_threshold, candidate_count=3,
        include_rank_features=True)
    test_utilities = test_outcomes - costs
    champion_metrics = evaluate_head(
        champion, test_features, test_utilities)
    target_utility = (
        champion_metrics["verified_utility"] + args.practical_gain)

    def record(step: int) -> None:
        for name, head in arms.items():
            row = evaluate_head(head, test_features, test_utilities)
            row.update({
                "step": step,
                "verifier_bits": step * args.batch_size,
                "reaches_practical_target":
                    row["verified_utility"] >= target_utility,
            })
            histories[name].append(row)

    record(0)
    action_generators = {
        name: torch.Generator(device=device).manual_seed(
            args.seed + 70_000_000)
        for name in arms}
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        pool_count = args.batch_size * args.active_pool_multiplier
        features_pool, outcomes_pool, _ = ranked_requery_batch(
            controller, count=pool_count, capacity=args.capacity,
            seed=args.seed * 1_000_000 + step, device=device,
            write_threshold=args.write_threshold, candidate_count=3,
            include_rank_features=True)
        utilities_pool = outcomes_pool - costs
        for name, head in arms.items():
            with torch.no_grad():
                incumbent_actions = champion(features_pool)
                candidate_actions = head(features_pool)
                disagreement = candidate_actions != incumbent_actions
                disagree = disagreement.nonzero(
                    as_tuple=False).squeeze(1)
                agree = (~disagreement).nonzero(
                    as_tuple=False).squeeze(1)
                indices = torch.cat((disagree, agree))[:args.batch_size]
            features = features_pool[indices]
            utilities = utilities_pool[indices]
            attempted = torch.randint(
                0, 3, (args.batch_size,),
                generator=action_generators[name], device=device)
            observed = utilities.gather(
                1, attempted[:, None]).squeeze(1)
            predicted = head.q_values(features).gather(
                1, attempted[:, None]).squeeze(1)
            loss = nn.functional.smooth_l1_loss(predicted, observed)
            optimizers[name].zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizers[name].step()
            logs[name].append((
                features.detach(), attempted.detach(), observed.detach()))
            selections[name].append({
                "step": step,
                "pool_contexts": pool_count,
                "verified_contexts": args.batch_size,
                "pool_disagreement_fraction":
                    float(disagreement.float().mean()),
                "selected_disagreements":
                    int(disagreement[indices].sum()),
            })
        if step % 2 == 0 or step == args.steps:
            record(step)
        if step % args.evidence_every == 0:
            for name, head in arms.items():
                features = torch.cat([row[0] for row in logs[name]])
                attempted = torch.cat([row[1] for row in logs[name]])
                observed = torch.cat([row[2] for row in logs[name]])
                with torch.no_grad():
                    incumbent_actions = champion(features)
                    challenger_actions = head(features)
                evidence[name].append({
                    "step": step,
                    **paired_multiaction_improvement(
                        incumbent_actions, challenger_actions,
                        attempted, observed),
                })

    first_target = {}
    for name, rows in histories.items():
        matches = [
            row for row in rows if row["reaches_practical_target"]]
        first_target[name] = (
            matches[0]["verifier_bits"] if matches else None)
    transferred_bits = first_target["composed_transfer"]
    reset_bits = first_target["stored_reset"]
    gate = {
        "viable_oracle_headroom": (
            champion_metrics["oracle_utility"]
            >= target_utility + 0.02),
        "transferred_reaches_target":
            transferred_bits is not None,
        "transferred_reaches_target_before_reset": (
            transferred_bits is not None
            and (reset_bits is None or transferred_bits < reset_bits)),
        "transferred_final_beats_champion": (
            histories["composed_transfer"][-1]["verified_utility"]
            >= target_utility),
    }
    gate["accepted_for_replication"] = all(gate.values())
    report = {
        "schema": "three-way-requery-race-v1",
        "configuration": {
            **vars(args),
            "parent_checkpoint": str(args.parent_checkpoint),
            "selected_prefix": str(args.selected_prefix),
            "champion_head": str(args.champion_head),
            "skill_store": str(args.skill_store),
            "report": str(args.report),
        },
        "learner_visible": [
            "four_generic_memory_statistics", "attempted_action_0_1_2",
            "attempted_action_scalar_outcome", "logging_propensity_1_3",
        ],
        "hidden_from_learner": [
            "unattempted_outcomes", "oracle_action", "correct_answer",
            "semantic_task_identity", "private_test_metrics",
        ],
        "champion": champion_metrics,
        "target_utility": target_utility,
        "first_target_bits": first_target,
        "arms": {
            name: {
                "history": histories[name],
                "promotion_evidence": evidence[name],
                "selection": selections[name],
            }
            for name in arms
        },
        "gate": gate,
        "accounting": {
            "verifier_bits_per_arm": args.steps * args.batch_size,
            "unlabeled_candidate_contexts_per_arm":
                args.steps * args.batch_size
                * args.active_pool_multiplier,
            "optimizer_updates_per_arm": args.steps,
            "private_test_both_action_bits":
                args.test_contexts * 3,
            "wall_seconds": time.perf_counter() - started,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
