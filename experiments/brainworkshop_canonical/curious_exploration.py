"""Exploring on purpose, and the television that stops it working.

The probe policy has been uniform since navigation began and every record has
said so. This replaces it, using nothing that was not already built.

**Novelty is a task, and successor features make a changing task cheap.** The
occupancies stored for "reach each place" do not depend on what the agent
currently wants. Wanting to see the least-visited place is a weight vector, and
it changes every single step -- so re-aiming costs one dot product per stored
policy rather than a replan. That is the argument for having built the
successor features first, and this is the first thing that actually exercises
it.

**Optimism is the cheap baseline and has to be beaten.** An action never tried
here is worth trying, and a policy that does only that -- try everything at
this place, then wander -- is most of the way to full coverage on a small
world. It is included as its own arm precisely so that curiosity has to earn
the rest. Without it, the curious arms would be credited with a result that a
much simpler rule already gets.

**The noisy television is not hypothetical here.** Prediction-error curiosity
is famously captured by an inexhaustible source of surprise. The distractor
from the identity work is one: with something else moving in the frame, no
whole-scene reading ever repeats, so an agent that counts novelty over
readings sees everything as equally new and the signal carries no direction at
all. The gated arm counts novelty only over the part of the scene it was
measured to control. Both are run, in both conditions, because "gating is
necessary" is a claim about a failure that has to be shown.

Identification is handed over throughout, by the declared place-to-cluster
oracle. That is deliberate: the identity record already measures what working
out which marker you are costs, and mixing it in here would leave two axes
moving at once.
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
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .navigation_environment import NavigationTask, sample_navigation_task
from .novelty import NoveltyCounts, SlidingWindowUCB
from .object_identity import DistractorVerifier, _model_accuracy
from .object_scene import PLACE_COUNT, render_markers
from .successor_features import (
    DEFAULT_DISCOUNT,
    SuccessorFeatureLibrary,
    SuccessorFeatureRecord,
    generalised_policy_improvement,
    greedy_policy,
    reach,
    successor_features,
)
from .successor_transfer import (
    SlotReader,
    best_weighted_return,
    build_slot_reader,
    gpi_chooser,
    random_chooser,
    run_weighted_episode,
    weights_for,
    worst_weighted_return,
)
from .world_model import WorldModel

EXPERIMENT_ID = "brainworkshop-curious-exploration-2026-08-16"
CURIOUS_EXPLORATION_SCHEMA = "neural-computer.curious-exploration.v1"
DEVELOPMENT_SEED = 41
# Worlds are drawn from here. Defaulting to the development value keeps
# every recorded diagnostic reproducing exactly; a holdout run passes a seed
# from an unused block so the *worlds* are unseen and not merely the
# exploration randomness.
DEVELOPMENT_WORLD_SEED = 9000
WORLD_SEED_STRIDE = 37
FRAME_SIZE = 36
EXPLORE_EPISODES = 6
EXPLORE_STEPS = 10
EVALUATION_STEPS = 20
TRAINING_GOALS = 4
# The family the meta-controller chooses among. A short horizon takes the
# nearest unseen place; a long one will cross the world for a better one. Which
# is right changes during the run, which is the reason for choosing repeatedly
# rather than once.
DISCOUNT_FAMILY = (0.5, 0.8, 0.95, 0.99)
# The family's extremes are run as their own arms, so the meta-controller is
# compared against the best fixed horizon rather than against nothing.
ARMS = (
    "uniform",
    "optimistic",
    "curious_g50",
    "curious",
    "curious_g99",
    "curious_ungated",
    "curious_bandit",
)
FIXED_DISCOUNT = {"curious_g50": 0.5, "curious_g99": 0.99}
CONDITIONS = ("none", "random_walk")


def base_policies(model: WorldModel, *, discount: float):
    """One stored policy per place: "go there, by the best route you know".

    The Option Keyboard's basis, at this scale. Any weight vector over places
    -- including a novelty vector that changes every step -- is then answered
    by generalised policy improvement over these, with no further search.
    """

    return [
        successor_features(
            model,
            greedy_policy(model, reach(model.place_count, place), discount=discount),
            discount=discount,
        )
        for place in range(model.place_count)
    ]


def untried_actions(model: WorldModel, place: int) -> list[int]:
    """Actions never taken here. Optimism, in the only form the model allows.

    Load-bearing, and not an optimisation. An untried cell is treated as a
    self-loop by everything downstream, so it looks like *staying put* -- which
    novelty scores as worthless, because the agent has just been here. Pure
    novelty-seeking would therefore never try a new action at all. Untried
    cells have to be preferred outright, before any value is consulted.
    """

    return [
        action
        for action in range(model.action_count)
        if not model.counts[action][int(place)]
    ]


def explore(
    reader: SlotReader,
    task: NavigationTask,
    *,
    arm: str,
    condition: str,
    episodes: int,
    steps: int,
    seed: int,
    cluster_of_place,
) -> dict[str, Any]:
    """One exploration budget, spent however this arm chooses to spend it."""

    model = WorldModel(reader.alphabet, task.action_count)
    counts = NoveltyCounts(reader.alphabet)
    generator = torch.Generator().manual_seed(int(seed))
    bandit = (
        SlidingWindowUCB(DISCOUNT_FAMILY, window=8, seed=seed)
        if arm == "curious_bandit"
        else None
    )
    curve: list[float] = []
    chosen: list[int] = []

    for episode in range(episodes):
        counts.start_episode()
        arm_index = bandit.select() if bandit is not None else None
        discount = (
            DISCOUNT_FAMILY[arm_index]
            if arm_index is not None
            else FIXED_DISCOUNT.get(arm, DEFAULT_DISCOUNT)
        )
        if arm_index is not None:
            chosen.append(arm_index)
        # Refreshed once per episode rather than once per step: the stored
        # occupancies only go stale when the *model* changes, and the task
        # changing every step costs nothing.
        psis = (
            base_policies(model, discount=discount)
            if arm.startswith("curious")
            else None
        )
        start = int(torch.randint(0, PLACE_COUNT, (1,), generator=generator).item())
        goal = int(torch.randint(0, PLACE_COUNT, (1,), generator=generator).item())
        verifier = DistractorVerifier(
            task,
            start=start,
            goal=goal,
            steps=steps,
            condition=condition,
            seed=seed + 17 * episode,
        )
        before = model.known_cells
        while not verifier.done:
            reading = [symbol for _, symbol in reader.read(verifier.observation())]
            # Declared oracle: identification is measured elsewhere, and
            # letting it vary here would move two axes at once.
            place = int(cluster_of_place[verifier.place])
            counts.observe(place, reading)
            fresh = untried_actions(model, place)
            if arm == "uniform":
                action = int(
                    torch.randint(
                        0, task.action_count, (1,), generator=generator
                    ).item()
                )
            elif fresh:
                action = fresh[episode % len(fresh)]
            elif psis is None:
                action = int(
                    torch.randint(
                        0, task.action_count, (1,), generator=generator
                    ).item()
                )
            else:
                weights = counts.weights(gated=arm != "curious_ungated")
                action = generalised_policy_improvement(psis, place, weights)
            verifier.score(torch.tensor([action], dtype=torch.long))
            following = int(cluster_of_place[verifier.place])
            model.observe(place, action, following, 0)
        if bandit is not None and arm_index is not None:
            bandit.update(arm_index, float(model.known_cells - before))
        curve.append(model.coverage)

    return {
        "model": model,
        "coverage": model.coverage,
        "curve": curve,
        "model_accuracy": _model_accuracy(model, task, cluster_of_place),
        "distinct_readings": len(counts.readings),
        "bandit_counts": list(bandit.counts()) if bandit is not None else [],
        "bandit_means": list(bandit.means()) if bandit is not None else [],
        "chosen": chosen,
    }


def steps_to_coverage(curve, *, threshold: float, steps: int) -> int | None:
    """How much experience this arm needed, rather than where it ended up.

    Final coverage saturates on a world this small, so the arms separate on
    speed or not at all.
    """

    for episode, value in enumerate(curve):
        if value >= threshold:
            return (episode + 1) * steps
    return None


def blind_baseline(
    reader: SlotReader,
    task: NavigationTask,
    *,
    seed: int,
    cluster_of_place,
    steps: int = EVALUATION_STEPS,
) -> float:
    """Acting without looking, on the held-out goals.

    Computed once per world rather than once per arm: it does not consult a
    model, so recomputing it for every arm measures nothing and costs the
    largest single block of frontend reads in the run.
    """

    blind = random_chooser(seed + 5, task.action_count)
    scored = []
    for goal in range(TRAINING_GOALS, PLACE_COUNT):
        weights = weights_for(PLACE_COUNT, (goal,), None)
        for start in range(PLACE_COUNT):
            scored.append(
                run_weighted_episode(
                    reader,
                    task,
                    start=start,
                    shown=(goal,),
                    weights_by_place=weights,
                    given=None,
                    steps=steps,
                    chooser=blind,
                    told=cluster_of_place,
                ).score
            )
    return sum(scored) / len(scored) if scored else 0.0


def downstream(
    reader: SlotReader,
    task: NavigationTask,
    model: WorldModel,
    *,
    seed: int,
    cluster_of_place,
    steps: int = EVALUATION_STEPS,
    discount: float = DEFAULT_DISCOUNT,
) -> dict[str, float]:
    """What the model is worth afterwards, on goals it was never aimed at.

    No distractor and no further experience: the only thing that differs
    between arms here is the model exploration left behind.
    """

    library = SuccessorFeatureLibrary(
        place_count=reader.alphabet,
        action_count=task.action_count,
        cumulant_dimension=reader.alphabet,
    )
    for goal in range(TRAINING_GOALS):
        policy = greedy_policy(
            model, reach(reader.alphabet, cluster_of_place[goal]), discount=discount
        )
        if library.duplicate_of(policy) is not None:
            continue
        library.append(
            SuccessorFeatureRecord(
                policy=policy,
                psi=successor_features(model, policy, discount=discount),
                discount=discount,
            )
        )
    if library.record_count == 0:
        return {"gpi": 0.0, "optimal": 0.0, "fraction": 0.0}

    chooser = gpi_chooser(library)
    scored, floors, ceilings = [], [], []
    for goal in range(TRAINING_GOALS, PLACE_COUNT):
        weights = weights_for(PLACE_COUNT, (goal,), None)
        for start in range(PLACE_COUNT):
            ceilings.append(best_weighted_return(task, start, weights, steps))
            floors.append(worst_weighted_return(task, start, weights, steps))
            scored.append(
                run_weighted_episode(
                    reader,
                    task,
                    start=start,
                    shown=(goal,),
                    weights_by_place=weights,
                    given=None,
                    steps=steps,
                    chooser=chooser,
                    told=cluster_of_place,
                ).score
            )
    fractions = [
        (value - floor) / (ceiling - floor)
        for value, floor, ceiling in zip(scored, floors, ceilings)
        if ceiling - floor > 1e-9
    ]
    return {
        "gpi": sum(scored) / len(scored),
        "optimal": sum(ceilings) / len(ceilings),
        "fraction": sum(fractions) / len(fractions) if fractions else 0.0,
    }


def run_curious_exploration(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    world_seed: int = DEVELOPMENT_WORLD_SEED,
    tasks: int = 6,
    episodes: int = EXPLORE_EPISODES,
    steps: int = EXPLORE_STEPS,
    threshold: float = 0.9,
) -> dict[str, Any]:
    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    reader = build_slot_reader(encoders)
    cluster_of_place = tuple(
        reader.read(render_markers((place, place), size=FRAME_SIZE))[0][1]
        for place in range(PLACE_COUNT)
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index in range(tasks):
        task = sample_navigation_task(seed=world_seed + WORLD_SEED_STRIDE * index)
        if task is None:
            continue
        blind = blind_baseline(
            reader,
            task,
            seed=seed + 100 * index,
            cluster_of_place=cluster_of_place,
        )
        for condition in CONDITIONS:
            for arm in ARMS:
                explored = explore(
                    reader,
                    task,
                    arm=arm,
                    condition=condition,
                    episodes=episodes,
                    steps=steps,
                    seed=seed + 100 * index,
                    cluster_of_place=cluster_of_place,
                )
                after = downstream(
                    reader,
                    task,
                    explored["model"],
                    seed=seed + 100 * index,
                    cluster_of_place=cluster_of_place,
                )
                rows.append(
                    {
                        "task": index,
                        "condition": condition,
                        "arm": arm,
                        "coverage": explored["coverage"],
                        "model_accuracy": explored["model_accuracy"],
                        "distinct_readings": explored["distinct_readings"],
                        "steps_to_threshold": steps_to_coverage(
                            explored["curve"], threshold=threshold, steps=steps
                        ),
                        "curve": explored["curve"],
                        "bandit_counts": explored["bandit_counts"],
                        "bandit_means": explored["bandit_means"],
                        "downstream": after,
                        "downstream_random": blind,
                    }
                )

    tail = sha256_file(bank_path)
    if tail != before:
        raise RuntimeError("the curious exploration run mutated AgentBrain.bank")

    summary: dict[str, Any] = {}
    budget = episodes * steps
    for condition in CONDITIONS:
        block: dict[str, Any] = {}
        for arm in ARMS:
            chosen = [
                row
                for row in rows
                if row["condition"] == condition and row["arm"] == arm
            ]
            if not chosen:
                continue
            reached = [
                row["steps_to_threshold"]
                for row in chosen
                if row["steps_to_threshold"] is not None
            ]
            block[arm] = {
                "tasks": len(chosen),
                "coverage": sum(row["coverage"] for row in chosen) / len(chosen),
                "model_accuracy": (
                    sum(row["model_accuracy"] for row in chosen) / len(chosen)
                ),
                "distinct_readings": (
                    sum(row["distinct_readings"] for row in chosen) / len(chosen)
                ),
                "reached_threshold": len(reached),
                # Unreached runs are charged the whole budget rather than
                # dropped, so an arm cannot look fast by failing often.
                "steps_to_threshold": (
                    sum(
                        row["steps_to_threshold"]
                        if row["steps_to_threshold"] is not None
                        else budget
                        for row in chosen
                    )
                    / len(chosen)
                ),
                "downstream_fraction": (
                    sum(row["downstream"]["fraction"] for row in chosen) / len(chosen)
                ),
                "downstream_random": (
                    sum(row["downstream_random"] for row in chosen) / len(chosen)
                ),
            }
        summary[condition] = block

    report = {
        "schema": CURIOUS_EXPLORATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "world_seed": world_seed,
        "tasks": tasks,
        "episodes": episodes,
        "episode_steps": steps,
        "budget_steps": budget,
        "threshold": threshold,
        "discount_family": list(DISCOUNT_FAMILY),
        "conditions": summary,
        "rows": rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": tail == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "curious_exploration.json").write_text(
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
        "--bank", type=Path, default=repository / "artifacts/checkpoints/AgentBrain.bank"
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
            / "brainworkshop_curious_exploration_2026-08-16"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument(
        "--world-seed", type=int, default=DEVELOPMENT_WORLD_SEED
    )
    parser.add_argument("--tasks", type=int, default=6)
    parser.add_argument("--episodes", type=int, default=EXPLORE_EPISODES)
    arguments = parser.parse_args()
    report = run_curious_exploration(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
        world_seed=arguments.world_seed,
        tasks=arguments.tasks,
        episodes=arguments.episodes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
