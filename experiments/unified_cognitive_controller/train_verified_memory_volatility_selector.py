"""Learn to use row volatility from scalar verifier reward alone."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn

from .probe_verified_memory_volatility import _is_recalled, _prepare_trial


class ReplacementSelector(nn.Module):
    """Small task-agnostic policy over generic physical-memory features."""

    def __init__(self) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.score(features).squeeze(-1)


def _cached_environments(
        first_seed: int, count: int, policy: str
        ) -> tuple[torch.Tensor, torch.Tensor]:
    features, stable_masks = [], []
    for seed in range(first_seed, first_seed + count):
        memory, _, _, stable, _, _, _ = _prepare_trial(seed, policy)
        access = memory.store.access_count.float()
        age = memory.store.age.float() / max(1, memory.store.clock)
        usage = memory.store.usage.float()
        features.append(torch.stack((
            memory.store.volatility, torch.log1p(access) / math.log(25.0),
            age, usage), dim=-1))
        stable_mask = torch.zeros(8, dtype=torch.bool)
        stable_mask[stable] = True
        stable_masks.append(stable_mask)
    return torch.stack(features), torch.stack(stable_masks)


def _sample_episode(
        model: ReplacementSelector, features: torch.Tensor,
        stable: torch.Tensor, *, greedy: bool
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = features.shape[0]
    available = torch.ones(
        batch, features.shape[1], dtype=torch.bool, device=features.device)
    log_probability = features.new_zeros(batch)
    entropy = features.new_zeros(batch)
    selected_stable = features.new_zeros(batch)
    for _ in range(4):
        logits = model(features).masked_fill(~available, -1e9)
        distribution = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else distribution.sample()
        log_probability += distribution.log_prob(action)
        entropy += distribution.entropy()
        selected_stable += stable.gather(1, action[:, None]).squeeze(1)
        available.scatter_(1, action[:, None], False)
    # The private verifier counts preserved stable skills plus newly installed
    # skills. Selecting a stable row sacrifices one verified capability.
    reward = 1.0 - selected_stable / 7.0
    return reward, log_probability, entropy


@torch.no_grad()
def _evaluate(
        model: ReplacementSelector, first_seed: int, count: int,
        policy: str) -> dict[str, float]:
    features, stable = _cached_environments(first_seed, count, policy)
    reward, _, _ = _sample_episode(
        model, features, stable.float(), greedy=True)
    return {
        "mean_verified_score": float(reward.mean()),
        "perfect_episode_rate": float((reward == 1.0).float().mean()),
        "mean_stable_rows_rewritten": float((1.0 - reward).mean() * 7.0),
    }


@torch.no_grad()
def _evaluate_physical(
        model: ReplacementSelector, first_seed: int, count: int,
        policy: str) -> dict[str, float]:
    scores, stable_rates, new_rates = [], [], []
    for seed in range(first_seed, first_seed + count):
        memory, rows, values, stable, _, _, _ = _prepare_trial(seed, policy)
        access = memory.store.access_count.float()
        features = torch.stack((
            memory.store.volatility,
            torch.log1p(access) / math.log(25.0),
            memory.store.age.float() / max(1, memory.store.clock),
            memory.store.usage.float()), dim=-1)
        available = torch.ones(8, dtype=torch.bool)
        for offset in range(4):
            logits = model(features).masked_fill(~available, -1e9)
            chosen = int(logits.argmax())
            memory.elastic_replace(
                chosen, rows[8 + offset], values[8 + offset], 1.0)
            available[chosen] = False
        stable_retained = sum(
            _is_recalled(memory, rows[index], values[index])
            for index in stable.tolist())
        new_acquired = sum(
            _is_recalled(memory, rows[8 + offset], values[8 + offset])
            for offset in range(4))
        stable_rates.append(stable_retained / 3)
        new_rates.append(new_acquired / 4)
        scores.append((stable_retained + new_acquired) / 7)
    return {
        "mean_verified_score": sum(scores) / len(scores),
        "stable_retention": sum(stable_rates) / len(stable_rates),
        "new_acquisition": sum(new_rates) / len(new_rates),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--updates", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--train-environments", type=int, default=1024)
    parser.add_argument("--eval-environments", type=int, default=512)
    parser.add_argument("--seed", type=int, default=12000)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_features, train_stable = _cached_environments(
        args.seed, args.train_environments, "verified")
    model = ReplacementSelector()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    baseline = 0.0
    trace = []
    for update in range(1, args.updates + 1):
        indices = torch.randint(
            args.train_environments, (args.batch_size,))
        reward, log_probability, entropy = _sample_episode(
            model, train_features[indices],
            train_stable[indices].float(), greedy=False)
        baseline = 0.95 * baseline + 0.05 * float(reward.mean())
        advantage = reward - baseline
        loss = -(advantage.detach() * log_probability).mean()
        loss -= 0.002 * entropy.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update == 1 or update % 16 == 0:
            trace.append({
                "update": update,
                "reward": float(reward.mean()),
                "perfect_rate": float((reward == 1.0).float().mean()),
            })

    eval_seed = args.seed + args.train_environments + 10_000
    intact = _evaluate(
        model, eval_seed, args.eval_environments, "verified")
    access = _evaluate(
        model, eval_seed, args.eval_environments, "access")
    shuffled = _evaluate(
        model, eval_seed, args.eval_environments, "shuffled_verified")
    uniform = _evaluate(
        model, eval_seed, args.eval_environments, "uniform")
    physical = _evaluate_physical(
        model, eval_seed, min(128, args.eval_environments), "verified")
    physical_shuffled = _evaluate_physical(
        model, eval_seed, min(128, args.eval_environments),
        "shuffled_verified")
    report = {
        "schema": "verified-memory-volatility-selector-v1",
        "training": {
            "signal": "scalar final verifier reward only",
            "semantic_task_ids": False,
            "correct_replacement_labels": False,
            "unique_environments": args.train_environments,
            "updates": args.updates,
            "trace": trace,
        },
        "held_out": {
            "intact_verified_volatility": intact,
            "access_only": access,
            "shuffled_row_correspondence": shuffled,
            "uniform": uniform,
            "physical_memory": physical,
            "physical_memory_shuffled": physical_shuffled,
        },
        "gates": {
            "held_out_perfect_rate_at_least_95_percent":
                intact["perfect_episode_rate"] >= 0.95,
            "causal_shuffle_costs_at_least_15_points":
                intact["mean_verified_score"]
                >= shuffled["mean_verified_score"] + 0.15,
            "verified_signal_beats_access_only_by_12_points":
                intact["mean_verified_score"]
                >= access["mean_verified_score"] + 0.12,
            "physical_memory_retains_and_acquires_at_least_95_percent":
                min(physical["stable_retention"], physical["new_acquisition"])
                >= 0.95,
            "physical_shuffle_costs_at_least_15_points":
                physical["mean_verified_score"]
                >= physical_shuffled["mean_verified_score"] + 0.15,
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["training"]["trace"], indent=2))
    print(json.dumps(report["held_out"], indent=2))
    print(json.dumps(report["gates"], indent=2))


if __name__ == "__main__":
    main()
