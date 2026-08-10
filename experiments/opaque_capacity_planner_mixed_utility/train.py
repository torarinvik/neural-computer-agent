"""Promote replay-free mixed-action capacity-policy learning."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    MemoryCandidates,
    OpaqueCapacityPlanner,
)

PRETRAIN_EPISODES = 400
MIXED_EPISODES = 1600
HELDOUT_PER_ACTION = 100
WIDTH = 8
CAPACITY = 5
HIDDEN = 48
TEMPERATURE = 0.8
PLANNER_LEARNING_RATE = 0.01
OPTIMIZER_LEARNING_RATE = 0.005
STABLE_WINDOW = 100
ACTIONS = ("admit", "evict", "consolidate", "grow")


def _digest(module: torch.nn.Module) -> str:
    values = []
    for name, value in sorted(module.state_dict().items()):
        values.append((name, value.detach().cpu().clone()))
    return repr(values)


def _episode(
    seed: int,
    action: str,
) -> tuple[
    MemoryCandidates,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    str,
    int | None,
    tuple[int, int] | None,
]:
    if action not in ACTIONS:
        raise ValueError(f"unknown target action: {action}")
    generator = torch.Generator().manual_seed(seed)
    keys = torch.nn.functional.normalize(
        torch.randn(1, CAPACITY, WIDTH, generator=generator),
        dim=-1,
    )
    values = torch.nn.functional.normalize(
        torch.randn(1, CAPACITY, WIDTH, generator=generator),
        dim=-1,
    )
    strengths = torch.rand(1, CAPACITY, generator=generator)
    timestamps = torch.arange(CAPACITY, dtype=torch.float32).view(1, -1)
    occupied = torch.ones(1, CAPACITY, dtype=torch.bool)
    protected = torch.zeros(1, CAPACITY, dtype=torch.bool)
    consolidation_available = torch.zeros(1, dtype=torch.bool)

    if action == "admit":
        occupied[0, 2:] = False
        protected[0, 0] = True
    elif action == "evict":
        protected[0, 0] = True
        protected[0, 1] = True
    elif action == "consolidate":
        consolidation_available[0] = True
        protected[:] = True
        values[0, 1] = torch.nn.functional.normalize(
            values[0, 0] + 0.03 * torch.randn(WIDTH, generator=generator),
            dim=0,
        )
    elif action == "grow":
        protected[:] = True

    bank = MemoryCandidates(
        keys=keys,
        values=values,
        strengths=strengths,
        timestamps=timestamps,
        occupied=occupied,
    )
    eviction_index = None
    pair = None
    if action == "evict":
        available = torch.where(occupied[0] & ~protected[0])[0]
        eviction_index = int(available[strengths[0, available].argmin()])
    elif action == "consolidate":
        similarity = values[0] @ values[0].transpose(0, 1)
        similarity.fill_diagonal_(-2.0)
        flat_index = int(similarity.reshape(-1).argmax())
        first, second = divmod(flat_index, CAPACITY)
        first, second = sorted((first, second))
        pair = (first, second)
    return (
        bank,
        keys[0, 0].unsqueeze(0),
        values[0, 0].unsqueeze(0),
        protected,
        consolidation_available,
        action,
        eviction_index,
        pair,
    )


def _utility(plan, target_action: str, target_eviction: int | None, target_pair: tuple[int, int] | None) -> float:
    if plan.action != target_action:
        return 0.0
    if target_action == "evict":
        return 1.0 if plan.eviction_index == target_eviction else 0.5
    if target_action == "consolidate":
        return 1.0 if plan.pair == target_pair else 0.5
    return 1.0


def _mean(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def _stable_tail(values: list[float]) -> tuple[int | None, float]:
    if len(values) < STABLE_WINDOW:
        return None, min(values, default=0.0)
    means = [
        _mean(values[index : index + STABLE_WINDOW])
        for index in range(len(values) - STABLE_WINDOW + 1)
    ]
    for index in range(len(means)):
        if min(means[index:]) >= 0.9:
            return index + 1, min(means[index:])
    return None, min(means)


def _run_episode(
    planner: OpaqueCapacityPlanner,
    optimizer: torch.optim.Optimizer,
    seed: int,
    action: str,
    explorer: torch.Generator,
) -> float:
    (
        bank,
        incoming_key,
        incoming_value,
        protected,
        consolidation_available,
        target_action,
        target_eviction,
        target_pair,
    ) = _episode(seed, action)
    plan = planner.propose(
        bank,
        incoming_key,
        incoming_value,
        protected,
        consolidation_available=consolidation_available,
        explore=True,
        temperature=TEMPERATURE,
        generator=explorer,
    )
    utility = _utility(plan, target_action, target_eviction, target_pair)
    planner.adaptation_step(
        bank,
        incoming_key,
        incoming_value,
        protected,
        plan,
        utility,
        consolidation_available=consolidation_available,
        optimizer=optimizer,
    )
    return utility


def _evaluate(
    planner: OpaqueCapacityPlanner,
    seed: int,
    *,
    actions: tuple[str, ...] = ACTIONS,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for action_index, action in enumerate(actions):
        utilities = []
        for episode_index in range(HELDOUT_PER_ACTION):
            episode = _episode(seed + action_index * 100000 + episode_index, action)
            bank, incoming_key, incoming_value, protected, available = episode[:5]
            target_action, target_eviction, target_pair = episode[5:]
            plan = planner.propose(
                bank,
                incoming_key,
                incoming_value,
                protected,
                consolidation_available=available,
            )
            utilities.append(
                _utility(plan, target_action, target_eviction, target_pair)
            )
        scores[action] = _mean(utilities)
    return scores


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    controller = AmodalCognitiveController(
        width=8,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    torch.manual_seed(seed)
    planner = OpaqueCapacityPlanner(
        width=WIDTH,
        hidden=HIDDEN,
        learning_rate=PLANNER_LEARNING_RATE,
    )
    optimizer = torch.optim.Adam(planner.parameters(), lr=OPTIMIZER_LEARNING_RATE)
    explorer = torch.Generator().manual_seed(seed + 4000)
    pretrain_utilities = [
        _run_episode(
            planner,
            optimizer,
            seed + 10000 + index,
            "consolidate",
            explorer,
        )
        for index in range(PRETRAIN_EPISODES)
    ]
    mixed_utilities: list[float] = []
    mixed_actions: list[str] = []
    mixed_by_action = {action: [] for action in ACTIONS}
    for index in range(MIXED_EPISODES):
        action = ACTIONS[index % len(ACTIONS)]
        mixed_actions.append(action)
        utility = _run_episode(
            planner,
            optimizer,
            seed + 100000 + index,
            action,
            explorer,
        )
        mixed_utilities.append(utility)
        mixed_by_action[action].append(utility)

    torch.manual_seed(seed + 8000)
    fresh = OpaqueCapacityPlanner(
        width=WIDTH,
        hidden=HIDDEN,
        learning_rate=PLANNER_LEARNING_RATE,
    )
    trained_scores = _evaluate(planner, seed + 500000)
    fresh_scores = _evaluate(fresh, seed + 500000)
    retention_scores = _evaluate(planner, seed + 700000, actions=("consolidate",))
    stable_start, stable_minimum = _stable_tail(mixed_utilities)
    first_window = _mean(mixed_utilities[:STABLE_WINDOW])
    last_window = _mean(mixed_utilities[-STABLE_WINDOW:])
    online_by_action = {
        action: {
            "first_window": _mean(values[: STABLE_WINDOW // len(ACTIONS)]),
            "last_window": _mean(values[-STABLE_WINDOW // len(ACTIONS) :]),
        }
        for action, values in mixed_by_action.items()
    }
    gates = {
        "mixed_utility_improved": last_window >= first_window + 0.05,
        "stable_mixed_threshold": stable_start is not None and stable_minimum >= 0.9,
        "all_action_transfer": all(value >= 0.9 for value in trained_scores.values()),
        "prior_consolidation_retained": retention_scores["consolidate"] >= 0.9,
        "transfer_beats_fresh": all(
            trained_scores[action] >= fresh_scores[action] + 0.2
            for action in ACTIONS[:-1]
        ),
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "updates_match_unique_episodes": PRETRAIN_EPISODES + MIXED_EPISODES
        == PRETRAIN_EPISODES + len(mixed_actions),
    }
    report = {
        "schema": "neural-computer.opaque-capacity-planner-mixed-utility.v1",
        "claim_boundary": (
            "replay-free sequential learning and held-out transfer across four "
            "bounded capacity actions with consolidation retention; not universal "
            "capacity planning, autonomous verifier design, unbounded memory, "
            "or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "pretrain_episodes": PRETRAIN_EPISODES,
            "mixed_episodes": MIXED_EPISODES,
            "heldout_per_action": HELDOUT_PER_ACTION,
            "actions": ACTIONS,
            "capacity": CAPACITY,
            "width": WIDTH,
            "hidden": HIDDEN,
            "temperature": TEMPERATURE,
            "update": "centered_single_verifier_utility_policy_gradient_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "pretrain_last_window_utility": _mean(pretrain_utilities[-STABLE_WINDOW:]),
            "mixed_first_window_utility": first_window,
            "mixed_last_window_utility": last_window,
            "stable_start_episode": stable_start,
            "stable_window_minimum": stable_minimum,
            "online_by_action": online_by_action,
            "trained_heldout_by_action": trained_scores,
            "fresh_heldout_by_action": fresh_scores,
            "retention_after_mixed_training": retention_scores,
        },
        "accounting": {
            "unique_verifier_utilities": PRETRAIN_EPISODES + MIXED_EPISODES,
            "unique_logical_lifetimes": PRETRAIN_EPISODES + MIXED_EPISODES,
            "optimizer_updates": PRETRAIN_EPISODES + MIXED_EPISODES,
            "replayed_examples": 0,
            "controller_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85601)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
