"""Explore a world, model it, plan through it -- and check it is not a fluke.

The claim is narrow and testable: in a world where the agent's actions decide
what it sees next, and reward arrives only on a place nobody named, building a
model and searching it beats guessing policies with the same experience.

The controls are what make that a claim rather than a demonstration.

- **random** -- the policy the exploration itself used. Anything that does not
  beat this has learned nothing.
- **model-free** -- the best of many reactive policies drawn at random and
  *scored on the same number of episodes the modeller spent exploring*. This is
  the one that matters. Model-based learning is only worth its complexity if
  the same experience spent guessing does worse.
- **shuffled rewards** -- the goal's identity destroyed while its frequency is
  kept. Planning still runs; it just has no target worth reaching.
- **held-out starts** -- the agent is dropped at places it never began an
  episode from. A memorised trajectory cannot do this; a model can.

The last is the sharpest, because it separates the two ways of appearing to
have learned. Everything is read against `optimal_return`, which the verifier
knows and the agent never sees.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer.promotion import sha256_file

from .controller_pretraining import load_temporal_controller_artifact
from .counter_state_programs import nearest_cluster
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .navigation_environment import (
    NavigationTask,
    NavigationVerifier,
    sample_navigation_task,
)
from .prototype_templates import cluster_events, estimated_tolerance
from .world_model import WorldModel, plan_to, policy_from_model

EXPERIMENT_ID = "brainworkshop-navigation-transfer-2026-08-15"
NAVIGATION_TRANSFER_SCHEMA = "neural-computer.navigation-transfer.v1"
DEVELOPMENT_SEED = 41
EPISODE_STEPS = 24
EXPLORE_EPISODES = 12
EVALUATION_EPISODES = 4
MODEL_FREE_CANDIDATES = 400


def discover_places(
    encoders,
    task: NavigationTask,
    *,
    episodes: int,
    steps: int,
    seed: int,
    frame_noise: float = 0.0,
):
    """Cluster the places from a stream the agent actually walks through.

    A first version rendered each place once and clustered those eight frames.
    The tolerance estimator refused, and it was right to: with one observation
    per place there is no within-place mode, so there is no gap between
    "same place" and "different place" to find. That is not a quirk of the
    estimator, it is the absence of the evidence the estimator needs.

    So the alphabet is discovered the way every other task discovers it -- from
    a stream of observations, here produced by wandering. The reward is ignored
    and the episodes are counted, because wandering to see what a world looks
    like is not free even though it teaches nothing about what pays.
    """

    events = []
    for index in range(episodes):
        verifier = NavigationVerifier(
            task, steps=steps, seed=seed + index, frame_noise=frame_noise
        )
        policy = random_policy(task.action_count, seed + 600 + index)
        while not verifier.done:
            frame = verifier.observation()
            with torch.no_grad():
                events.append(encoders.vision(frame.unsqueeze(0)))
            verifier.score(
                torch.tensor([policy(0)], dtype=torch.long)
            )
    stacked = torch.cat(events)
    tolerance = estimated_tolerance(stacked)
    if tolerance is None:
        raise ValueError("the places do not separate into an alphabet")
    return cluster_events(stacked, tolerance=tolerance, maximum_clusters=32)


def cluster_of_place(encoders, clusters, *, place_count: int, frame_size: int = 36):
    """Which cluster each place renders to. **Scoring only.**

    The agent lives entirely in cluster indices: it observes a cluster, plans
    over clusters, and acts. The verifier lives in place indices. The two are
    the same set under a permutation nobody needs to know -- except a *record*,
    which has to ask "did it find the actual goal" and "was that the shortest
    actual route", and cannot ask either without the map.

    A first version of this record compared the two namespaces directly. It
    reported the goal as never found while the agent was reliably standing on
    it, and reported one-step plans for two-step journeys.
    """

    from .rendered_environment import render_position

    frames = torch.stack(
        [render_position(place, size=frame_size) for place in range(place_count)]
    )
    with torch.no_grad():
        events = encoders.vision(frames)
    return tuple(int(index) for index in nearest_cluster(events, clusters))


def _observe(encoders, verifier, clusters) -> int:
    frame = verifier.observation()
    with torch.no_grad():
        event = encoders.vision(frame.unsqueeze(0))
    return int(nearest_cluster(event, clusters).item())


def run_episode(
    encoders,
    task: NavigationTask,
    clusters,
    policy,
    *,
    steps: int,
    seed: int,
    start: int | None = None,
    model: WorldModel | None = None,
    frame_noise: float = 0.0,
) -> dict[str, Any]:
    """One episode under a policy, optionally teaching a model as it goes."""

    if start is not None:
        task = NavigationTask(
            transitions=task.transitions,
            goal=task.goal,
            start=int(start),
            place_count=task.place_count,
        ).validate()
    verifier = NavigationVerifier(
        task, steps=steps, seed=seed, frame_noise=frame_noise
    )
    total = 0.0
    scored = 0
    while not verifier.done:
        place = _observe(encoders, verifier, clusters)
        action = int(policy(place))
        step = verifier.score(torch.tensor([action], dtype=torch.long))
        reward = int(step.reward.item())
        total += reward
        scored += 1
        if model is not None:
            following = (
                _observe(encoders, verifier, clusters) if not verifier.done else None
            )
            if following is not None:
                model.observe(place, action, following, reward)
    return {"mean_reward": total / scored if scored else 0.0, "steps": scored}


def random_policy(action_count: int, seed: int):
    generator = torch.Generator().manual_seed(int(seed))

    def act(place: int) -> int:
        del place
        return int(
            torch.randint(0, int(action_count), (1,), generator=generator).item()
        )

    return act


def table_policy(table):
    def act(place: int) -> int:
        return int(table[place])

    return act


def explore(
    encoders,
    task: NavigationTask,
    clusters,
    *,
    episodes: int,
    steps: int,
    seed: int,
    frame_noise: float = 0.0,
) -> tuple[WorldModel, float]:
    """Wander at random; keep what the wandering revealed."""

    model = WorldModel(
        place_count=int(clusters.shape[0]), action_count=task.action_count
    )
    total = 0.0
    for index in range(episodes):
        outcome = run_episode(
            encoders,
            task,
            clusters,
            random_policy(task.action_count, seed + 300 + index),
            steps=steps,
            seed=seed + index,
            model=model,
            frame_noise=frame_noise,
        )
        total += outcome["mean_reward"]
    return model, total / episodes if episodes else 0.0


def best_random_policy(
    encoders,
    task: NavigationTask,
    clusters,
    *,
    budget: int,
    steps: int,
    seed: int,
    candidates: int = MODEL_FREE_CANDIDATES,
    frame_noise: float = 0.0,
) -> tuple[tuple[int, ...], float]:
    """The model-free control, given the same experience as the modeller.

    `budget` episodes are spent in total, spread over as many random reactive
    policies as that allows. Sampling policies and keeping the best is the
    fairest cheap stand-in for policy search: it uses no gradients, no credit
    assignment and no model, which is exactly the comparison being drawn.
    """

    generator = torch.Generator().manual_seed(int(seed))
    places = int(clusters.shape[0])
    tried = min(candidates, max(1, budget))
    best_table: tuple[int, ...] = tuple([0] * places)
    best_score = -1.0
    for index in range(tried):
        table = tuple(
            int(value)
            for value in torch.randint(
                0, task.action_count, (places,), generator=generator
            )
        )
        outcome = run_episode(
            encoders,
            task,
            clusters,
            table_policy(table),
            steps=steps,
            seed=seed + index,
            frame_noise=frame_noise,
        )
        if outcome["mean_reward"] > best_score:
            best_score, best_table = outcome["mean_reward"], table
    return best_table, best_score


def evaluate(
    encoders,
    task: NavigationTask,
    clusters,
    policy,
    *,
    steps: int,
    seed: int,
    episodes: int,
    starts=None,
    frame_noise: float = 0.0,
) -> float:
    """Mean return over evaluation episodes, optionally from chosen starts."""

    places = starts if starts is not None else [task.start] * episodes
    total = 0.0
    for index, start in enumerate(places):
        total += run_episode(
            encoders,
            task,
            clusters,
            policy,
            steps=steps,
            seed=seed + 900 + index,
            start=int(start),
            frame_noise=frame_noise,
        )["mean_reward"]
    return total / len(places) if places else 0.0


def solve_navigation(
    encoders,
    task: NavigationTask,
    clusters,
    *,
    seed: int,
    steps: int = EPISODE_STEPS,
    explore_episodes: int = EXPLORE_EPISODES,
    evaluation_episodes: int = EVALUATION_EPISODES,
    frame_noise: float = 0.0,
) -> dict[str, Any]:
    """Explore, model, plan, and be measured against everything cheaper."""

    model, explore_return = explore(
        encoders,
        task,
        clusters,
        episodes=explore_episodes,
        steps=steps,
        seed=seed,
        frame_noise=frame_noise,
    )
    optimal = task.optimal_return(steps)
    planned = policy_from_model(model)

    # Where a memorised route and a model come apart.
    held_out = [
        place for place in range(task.place_count) if place != task.start
    ]
    reachable = [
        place
        for place in held_out
        if task.distances()[place] <= steps
    ]

    model_free_table, _ = best_random_policy(
        encoders,
        task,
        clusters,
        budget=explore_episodes,
        steps=steps,
        seed=seed + 4000,
        frame_noise=frame_noise,
    )
    # The same control handed ten times the experience. Random policy search
    # is a weak searcher, and a weak searcher losing on equal budget proves
    # less than one losing on a budget it should not need.
    rich_table, _ = best_random_policy(
        encoders,
        task,
        clusters,
        budget=10 * explore_episodes,
        steps=steps,
        seed=seed + 4000,
        frame_noise=frame_noise,
    )

    # Scoring-side: ask the model for a route from the *cluster* the start
    # place renders to, so the reported plan is over the graph the agent
    # actually planned on rather than over a permutation of it.
    place_to_cluster = cluster_of_place(
        encoders, clusters, place_count=task.place_count
    )
    route = plan_to(model, place_to_cluster[task.start], model.goals())
    truth = task.distances()[task.start]
    return {
        "optimal_return": optimal,
        "explore_return": explore_return,
        "coverage": model.coverage,
        "goals_found": len(model.goals()),
        "found_the_goal": place_to_cluster[task.goal] in set(model.goals()),
        "goals_are_only_the_goal": set(model.goals())
        == {place_to_cluster[task.goal]},
        "plan_length": None if route is None else route.length,
        "true_distance": truth,
        "plan_is_optimal": route is not None and route.length == truth,
        "planned_return": evaluate(
            encoders,
            task,
            clusters,
            planned,
            steps=steps,
            seed=seed,
            episodes=evaluation_episodes,
            frame_noise=frame_noise,
        ),
        "random_return": evaluate(
            encoders,
            task,
            clusters,
            random_policy(task.action_count, seed + 8000),
            steps=steps,
            seed=seed,
            episodes=evaluation_episodes,
            frame_noise=frame_noise,
        ),
        "model_free_return": evaluate(
            encoders,
            task,
            clusters,
            table_policy(model_free_table),
            steps=steps,
            seed=seed,
            episodes=evaluation_episodes,
            frame_noise=frame_noise,
        ),
        "model_free_10x_return": evaluate(
            encoders,
            task,
            clusters,
            table_policy(rich_table),
            steps=steps,
            seed=seed,
            episodes=evaluation_episodes,
            frame_noise=frame_noise,
        ),
        "held_out_return": evaluate(
            encoders,
            task,
            clusters,
            planned,
            steps=steps,
            seed=seed,
            episodes=len(reachable),
            starts=reachable,
            frame_noise=frame_noise,
        ),
        "held_out_optimal": (
            sum(
                min(1.0, (steps - task.distances()[place] + 1) / steps)
                for place in reachable
            )
            / len(reachable)
            if reachable
            else 0.0
        ),
        "held_out_starts": len(reachable),
    }


def shuffled_reward_model(model: WorldModel, *, seed: int) -> WorldModel:
    """Keep how often reward arrived; destroy where it arrived.

    The planner still has somewhere to go. It is just not the goal.
    """

    generator = torch.Generator().manual_seed(int(seed))
    total = sum(model.rewarded.values())
    scrambled = WorldModel(
        place_count=model.place_count,
        action_count=model.action_count,
        counts=model.counts,
        visited=model.visited,
    )
    for _ in range(total):
        place = int(
            torch.randint(0, model.place_count, (1,), generator=generator).item()
        )
        scrambled.rewarded[place] = scrambled.rewarded.get(place, 0) + 1
    return scrambled


def run_navigation(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    tasks: int = 12,
    explore_episodes: int = EXPLORE_EPISODES,
    frame_noise: float = 0.0,
) -> dict[str, Any]:
    """Many sampled worlds, each solved from scratch and controlled against."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    # One alphabet, established once by wandering in the first sampled world,
    # and spoken by every task after it. The places are the same renderings
    # everywhere, so rediscovering them per task would only invite the
    # first-come cluster-index bug the integrated agent already paid for.
    anchor = sample_navigation_task(seed=9000)
    if anchor is None:
        raise RuntimeError("could not sample a world to look at")
    clusters = discover_places(
        encoders,
        anchor,
        episodes=4,
        steps=EPISODE_STEPS,
        seed=seed,
        frame_noise=frame_noise,
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    shuffled_rows: list[dict[str, Any]] = []
    for index in range(tasks):
        task = sample_navigation_task(seed=9000 + 37 * index)
        if task is None:
            continue
        row = solve_navigation(
            encoders,
            task,
            clusters,
            seed=seed + 1000 * index,
            explore_episodes=explore_episodes,
            frame_noise=frame_noise,
        )
        row["task"] = index
        rows.append(row)

        # The same experience, with the goal's identity destroyed.
        model, _ = explore(
            encoders,
            task,
            clusters,
            episodes=explore_episodes,
            steps=EPISODE_STEPS,
            seed=seed + 1000 * index,
            frame_noise=frame_noise,
        )
        scrambled = shuffled_reward_model(model, seed=seed + index)
        shuffled_rows.append(
            {
                "task": index,
                "return": evaluate(
                    encoders,
                    task,
                    clusters,
                    policy_from_model(scrambled),
                    steps=EPISODE_STEPS,
                    seed=seed + 1000 * index,
                    episodes=EVALUATION_EPISODES,
                    frame_noise=frame_noise,
                ),
                "optimal_return": row["optimal_return"],
            }
        )

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the navigation run mutated AgentBrain.bank")

    def mean(key: str, source=None) -> float:
        items = source if source is not None else rows
        return sum(float(item[key]) for item in items) / len(items) if items else 0.0

    solved = [
        row for row in rows if row["planned_return"] >= 0.9 * row["optimal_return"]
    ]
    report = {
        "schema": NAVIGATION_TRANSFER_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "tasks": len(rows),
        "explore_episodes": explore_episodes,
        "episode_steps": EPISODE_STEPS,
        "frame_noise": frame_noise,
        "found_the_goal": sum(1 for row in rows if row["found_the_goal"]),
        "near_optimal": len(solved),
        "plans_optimal": sum(1 for row in rows if row["plan_is_optimal"]),
        "mean_coverage": mean("coverage"),
        "mean_optimal_return": mean("optimal_return"),
        "mean_planned_return": mean("planned_return"),
        "mean_model_free_return": mean("model_free_return"),
        "mean_model_free_10x_return": mean("model_free_10x_return"),
        "mean_random_return": mean("random_return"),
        "mean_explore_return": mean("explore_return"),
        "mean_held_out_return": mean("held_out_return"),
        "mean_held_out_optimal": mean("held_out_optimal"),
        "mean_shuffled_return": mean("return", shuffled_rows),
        "rows": rows,
        "shuffled": shuffled_rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "navigation_transfer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=repository / "artifacts/checkpoints/AgentBrain.bank",
    )
    parser.add_argument(
        "--frontend",
        type=Path,
        default=repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_navigation_transfer_2026-08-15"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--explore-episodes", type=int, default=EXPLORE_EPISODES)
    parser.add_argument("--frame-noise", type=float, default=0.0)
    arguments = parser.parse_args()
    report = run_navigation(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
        tasks=arguments.tasks,
        explore_episodes=arguments.explore_episodes,
        frame_noise=arguments.frame_noise,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
