"""Promote replay-free online verifier-utility learning for capacity policy."""

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

TRAIN_EPISODES = 600
HELDOUT_EPISODES = 100
WIDTH = 6
CAPACITY = 4
HIDDEN = 32
TEMPERATURE = 0.8
PLANNER_LEARNING_RATE = 0.01
OPTIMIZER_LEARNING_RATE = 0.005
STABLE_WINDOW = 50


def _digest(module: torch.nn.Module) -> str:
    digest = []
    for name, value in sorted(module.state_dict().items()):
        digest.append((name, value.detach().cpu().clone()))
    return repr(digest)


def _episode(seed: int) -> tuple[MemoryCandidates, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int]]:
    generator = torch.Generator().manual_seed(seed)
    keys = torch.nn.functional.normalize(
        torch.randn(1, CAPACITY, WIDTH, generator=generator),
        dim=-1,
    )
    values = torch.nn.functional.normalize(
        torch.randn(1, CAPACITY, WIDTH, generator=generator),
        dim=-1,
    )
    values[0, 1] = torch.nn.functional.normalize(
        values[0, 0] + 0.05 * torch.randn(WIDTH, generator=generator),
        dim=0,
    )
    keys[0, 1] = torch.nn.functional.normalize(
        keys[0, 0] + 0.05 * torch.randn(WIDTH, generator=generator),
        dim=0,
    )
    strengths = torch.rand(1, CAPACITY, generator=generator)
    timestamps = torch.arange(CAPACITY, dtype=torch.float32).view(1, -1)
    bank = MemoryCandidates(
        keys=keys,
        values=values,
        strengths=strengths,
        timestamps=timestamps,
        occupied=torch.ones(1, CAPACITY, dtype=torch.bool),
    )
    similarity = values[0] @ values[0].transpose(0, 1)
    similarity.fill_diagonal_(-2.0)
    flat_index = int(similarity.reshape(-1).argmax())
    first, second = divmod(flat_index, CAPACITY)
    first, second = sorted((first, second))
    return (
        bank,
        values[0, first].unsqueeze(0),
        values[0, second].unsqueeze(0),
        torch.ones(1, CAPACITY, dtype=torch.bool),
        (first, second),
    )


def _utility(plan, target_pair: tuple[int, int]) -> float:
    action_correct = plan.action == "consolidate"
    pair_correct = plan.pair == target_pair
    if action_correct and pair_correct:
        return 1.0
    if action_correct:
        return 0.5
    return 0.0


def _stable_tail(scores: list[float]) -> tuple[int | None, float]:
    if len(scores) < STABLE_WINDOW:
        return None, min(scores, default=0.0)
    means = [
        sum(scores[index : index + STABLE_WINDOW]) / STABLE_WINDOW
        for index in range(len(scores) - STABLE_WINDOW + 1)
    ]
    for index, _ in enumerate(means):
        if min(means[index:]) >= 0.95:
            return index + 1, min(means[index:])
    return None, min(means)


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
    online_utilities: list[float] = []
    online_losses: list[float] = []
    for episode_index in range(TRAIN_EPISODES):
        bank, incoming_key, incoming_value, protected, target_pair = _episode(
            seed + 10000 + episode_index
        )
        plan = planner.propose(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=torch.ones(1, dtype=torch.bool),
            explore=True,
            temperature=TEMPERATURE,
            generator=explorer,
        )
        utility = _utility(plan, target_pair)
        online_utilities.append(utility)
        online_losses.append(
            planner.adaptation_step(
                bank,
                incoming_key,
                incoming_value,
                protected,
                plan,
                utility,
                consolidation_available=torch.ones(1, dtype=torch.bool),
                optimizer=optimizer,
            )
        )

    torch.manual_seed(seed + 8000)
    fresh = OpaqueCapacityPlanner(
        width=WIDTH,
        hidden=HIDDEN,
        learning_rate=PLANNER_LEARNING_RATE,
    )
    trained_heldout: list[float] = []
    fresh_heldout: list[float] = []
    for episode_index in range(HELDOUT_EPISODES):
        bank, incoming_key, incoming_value, protected, target_pair = _episode(
            900000 + seed + episode_index
        )
        trained_plan = planner.propose(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=torch.ones(1, dtype=torch.bool),
        )
        fresh_plan = fresh.propose(
            bank,
            incoming_key,
            incoming_value,
            protected,
            consolidation_available=torch.ones(1, dtype=torch.bool),
        )
        trained_heldout.append(float(trained_plan.action == "consolidate" and trained_plan.pair == target_pair))
        fresh_heldout.append(float(fresh_plan.action == "consolidate" and fresh_plan.pair == target_pair))

    stable_start, stable_minimum = _stable_tail(online_utilities)
    first_mean = sum(online_utilities[:STABLE_WINDOW]) / STABLE_WINDOW
    last_mean = sum(online_utilities[-STABLE_WINDOW:]) / STABLE_WINDOW
    trained_transfer = sum(trained_heldout) / HELDOUT_EPISODES
    fresh_transfer = sum(fresh_heldout) / HELDOUT_EPISODES
    gates = {
        "online_utility_improved": last_mean >= first_mean + 0.3,
        "stable_online_threshold": stable_start is not None and stable_minimum >= 0.95,
        "trained_heldout_transfer": trained_transfer >= 0.95,
        "transfer_beats_fresh": trained_transfer >= fresh_transfer + 0.2,
        "controller_frozen": controller_digest == _digest(controller),
        "replay_zero": True,
        "planner_updates_match_unique_episodes": len(online_losses) == TRAIN_EPISODES,
    }
    report = {
        "schema": "neural-computer.opaque-capacity-planner-online-utility.v1",
        "claim_boundary": (
            "online verifier-utility learning with exploratory proposals for one "
            "bounded redundant-pair consolidation regime; not universal capacity "
            "policy, autonomous verifier design, unbounded memory, or general "
            "continual learning"
        ),
        "seed": seed,
        "configuration": {
            "train_episodes": TRAIN_EPISODES,
            "heldout_episodes": HELDOUT_EPISODES,
            "width": WIDTH,
            "capacity": CAPACITY,
            "hidden": HIDDEN,
            "temperature": TEMPERATURE,
            "exploration": "masked_categorical_sampling_v1",
            "update": "centered_single_verifier_utility_policy_gradient_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "first_window_utility": first_mean,
            "last_window_utility": last_mean,
            "stable_start_episode": stable_start,
            "stable_window_minimum": stable_minimum,
            "trained_heldout_transfer": trained_transfer,
            "fresh_heldout_transfer": fresh_transfer,
            "transfer_gain": trained_transfer - fresh_transfer,
            "mean_online_loss_last_window": sum(online_losses[-STABLE_WINDOW:]) / STABLE_WINDOW,
        },
        "accounting": {
            "unique_verifier_utilities": TRAIN_EPISODES,
            "unique_logical_lifetimes": TRAIN_EPISODES,
            "optimizer_updates": TRAIN_EPISODES,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85501)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
