"""Being told where to go, and reading the scene as things rather than a picture.

Two questions, and they turn out to be one experiment.

**Can a goal be given rather than stumbled upon?** The navigation agent sought
whatever had paid during exploration. It could not be *asked* for anything, and
one model served exactly one goal -- which is most of what a model is for.

**Can an observation have parts?** Every observation so far has been atomic, so
"what is in the scene" and "which scene is this" were the same question.

Showing the goal *in the scene* answers both at once. The scene holds two
markers, the agent's place and the requested place, and it is handed to two
agents that differ in one respect only:

- the **object** agent receives one event per marker and has to work out which
  one it is, given that slot order is by position and carries no identity;
- the **scene** agent receives one event for the whole picture, which is what
  every agent in this repository has received until now.

They see the same pixels through the same frozen encoder. What separates them
is that sixty-four scenes decompose into eight places -- so the object agent's
model of the dynamics is goal-independent and should transfer to goals it was
never trained on, while the scene agent's is a graph over pictures and a new
goal is a part of the graph it has never visited.

The held-out goals are the measurement. Everything else is a control.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neural_computer.promotion import sha256_file

from .controller_pretraining import load_temporal_controller_artifact
from .counter_state_programs import nearest_cluster
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .navigation_environment import NavigationTask, sample_navigation_task
from .object_scene import PLACE_COUNT, encode_scene, encode_slots, render_scene
from .prototype_templates import cluster_events, estimated_tolerance
from .world_model import WorldModel, plan_to

EXPERIMENT_ID = "brainworkshop-object-navigation-2026-08-15"
OBJECT_NAVIGATION_SCHEMA = "neural-computer.object-navigation.v1"
DEVELOPMENT_SEED = 41
EPISODE_STEPS = 20
EXPLORE_EPISODES = 24
FRAME_SIZE = 36


class SceneVerifier:
    """Renders the agent and its instruction together; scores arrival."""

    def __init__(
        self,
        task: NavigationTask,
        *,
        start: int,
        goal: int,
        steps: int,
        frame_size: int = FRAME_SIZE,
    ) -> None:
        self.task = task.validate()
        self.goal = int(goal)
        self.steps = int(steps)
        self.frame_size = int(frame_size)
        self._place = int(start)
        self._position = 0

    @property
    def done(self) -> bool:
        return self._position >= self.steps

    def observation(self) -> torch.Tensor:
        if self.done:
            raise RuntimeError("scene episode is complete")
        return render_scene(self._place, self.goal, size=self.frame_size)

    def score(self, action: torch.Tensor) -> float:
        if self.done:
            raise RuntimeError("scene episode is complete")
        chosen = int(action.item())
        if not 0 <= chosen < self.task.action_count:
            raise ValueError("scene action is outside the protocol")
        self._place = int(self.task.transitions[chosen][self._place])
        self._position += 1
        return float(self._place == self.goal)


def shortest(task: NavigationTask, start: int, goal: int) -> int:
    """Verifier-side: moves needed. Used to score, never to plan."""

    from collections import deque

    if start == goal:
        return 0
    distance = {start: 0}
    frontier = deque([start])
    while frontier:
        place = frontier.popleft()
        for action in range(task.action_count):
            following = int(task.transitions[action][place])
            if following in distance:
                continue
            distance[following] = distance[place] + 1
            if following == goal:
                return distance[following]
            frontier.append(following)
    return task.place_count + 1


def best_return(task: NavigationTask, start: int, goal: int, steps: int) -> float:
    """The most reward obtainable, by exact finite-horizon dynamic programming.

    Reaching the goal quickly is not the whole story: an agent can only *keep*
    scoring if some action holds it there, and the sampler guarantees that only
    for the task's own goal, not for every place that might be asked for. A
    first version assumed arrival-then-stay and produced optima the agent could
    not reach at all, which read as the agent failing when it was the ceiling
    that was wrong.
    """

    horizon = int(steps)
    value = [0.0] * task.place_count
    for _ in range(horizon):
        following = [
            max(
                (1.0 if int(task.transitions[action][place]) == goal else 0.0)
                + value[int(task.transitions[action][place])]
                for action in range(task.action_count)
            )
            for place in range(task.place_count)
        ]
        value = following
    return value[start] / horizon if horizon else 0.0


# --- the two frontends -----------------------------------------------------


@dataclass
class SlotFrontend:
    """One event per marker. Slot order is positional and means nothing."""

    encoders: Any
    clusters: torch.Tensor

    def read(self, frame: torch.Tensor) -> tuple[int, ...]:
        events = encode_slots(self.encoders, frame)
        return tuple(int(index) for index in nearest_cluster(events, self.clusters))


@dataclass
class SceneFrontend:
    """One event for the whole picture, as every earlier agent received."""

    encoders: Any
    clusters: torch.Tensor

    def read(self, frame: torch.Tensor) -> int:
        event = encode_scene(self.encoders, frame)
        return int(nearest_cluster(event, self.clusters).item())


def build_frontends(encoders, *, task: NavigationTask, steps: int):
    """Discover both alphabets from scenes the agent could actually meet.

    Both are estimated the same way, from a stream with repeats -- a catalogue
    of one look per scene has no within-scene mode and the estimator refuses
    it, which is the correct answer and was measured before relying on it.
    """

    slot_events = []
    scene_events = []
    for place in range(task.place_count):
        for goal in range(task.place_count):
            frame = render_scene(place, goal, size=FRAME_SIZE)
            slot_events.append(encode_slots(encoders, frame))
            # Twice, so a scene has a within-scene distance of zero and the
            # bimodal structure the estimator needs actually exists.
            scene_events.append(encode_scene(encoders, frame))
            scene_events.append(encode_scene(encoders, frame))
    slots = torch.cat(slot_events)
    scenes = torch.cat(scene_events)
    slot_tolerance = estimated_tolerance(slots)
    scene_tolerance = estimated_tolerance(scenes)
    if slot_tolerance is None or scene_tolerance is None:
        raise ValueError("an alphabet could not be estimated")
    return (
        SlotFrontend(
            encoders,
            cluster_events(slots, tolerance=slot_tolerance, maximum_clusters=256),
        ),
        SceneFrontend(
            encoders,
            cluster_events(scenes, tolerance=scene_tolerance, maximum_clusters=256),
        ),
    )


# --- the object agent ------------------------------------------------------


def identify_goal(observations) -> int | None:
    """Which of the things in the scene is the one that never moved.

    The agent is not told which slot it is. Across a wandering episode its own
    marker visits many places and the instruction stays where it is, so the
    place present in *every* observation is the goal. When the agent is
    standing on it there is only one marker, which is the same evidence.
    """

    common: set[int] | None = None
    for places in observations:
        current = set(places)
        common = current if common is None else (common & current)
        if not common:
            return None
    if common is None or len(common) != 1:
        return None
    return next(iter(common))


def explore_objects(
    front: SlotFrontend,
    task: NavigationTask,
    pairs,
    *,
    steps: int,
    seed: int,
) -> tuple[WorldModel, int]:
    """Wander; keep what the actions did to *me*, not to the picture."""

    model = WorldModel(
        place_count=int(front.clusters.shape[0]), action_count=task.action_count
    )
    generator = torch.Generator().manual_seed(int(seed))
    identified = 0
    for start, goal in pairs:
        verifier = SceneVerifier(task, start=start, goal=goal, steps=steps)
        seen = []
        history = []
        while not verifier.done:
            places = front.read(verifier.observation())
            seen.append(places)
            action = int(
                torch.randint(0, task.action_count, (1,), generator=generator).item()
            )
            reward = verifier.score(torch.tensor([action], dtype=torch.long))
            following = (
                front.read(verifier.observation()) if not verifier.done else None
            )
            history.append((places, action, following, reward))
        goal_place = identify_goal(seen)
        if goal_place is None:
            continue
        identified += 1
        for places, action, following, reward in history:
            if following is None:
                continue
            here = [place for place in places if place != goal_place]
            there = [place for place in following if place != goal_place]
            # Standing on the goal shows one marker; that is where I am.
            mine = here[0] if here else goal_place
            next_mine = there[0] if there else goal_place
            model.observe(mine, action, next_mine, int(reward))
    return model, identified


def object_policy(model: WorldModel, front: SlotFrontend):
    """Plan to the place the scene is *asking* for, using no reward at all."""

    def act(frame: torch.Tensor, remembered_goal: int | None) -> tuple[int, int | None]:
        places = front.read(frame)
        goal = remembered_goal
        if goal is None and len(places) == 1:
            goal = places[0]
        if goal is None:
            return 0, goal
        here = [place for place in places if place != goal]
        mine = here[0] if here else goal
        if mine == goal:
            holding = model.holding_action(mine)
            return (holding if holding is not None else 0), goal
        route = plan_to(model, mine, (goal,))
        if route is None or not route.actions:
            return 0, goal
        return int(route.actions[0]), goal

    return act


def run_random_episode(
    task: NavigationTask, *, start: int, goal: int, steps: int, seed: int
) -> float:
    """The floor: act without looking."""

    verifier = SceneVerifier(task, start=start, goal=goal, steps=steps)
    generator = torch.Generator().manual_seed(int(seed))
    total = 0.0
    scored = 0
    while not verifier.done:
        action = int(
            torch.randint(0, task.action_count, (1,), generator=generator).item()
        )
        total += verifier.score(torch.tensor([action], dtype=torch.long))
        scored += 1
    return total / scored if scored else 0.0


def run_object_episode(
    front: SlotFrontend,
    model: WorldModel,
    task: NavigationTask,
    *,
    start: int,
    goal: int,
    steps: int,
    told: int | None = None,
) -> float:
    """Work out which marker is the instruction, then go to it.

    `told` hands the instruction over instead, which is not how the agent is
    meant to work -- it is how the cost of working it out is separated from the
    quality of the plan that follows.

    Identification runs *online* and every step is scored. A first version
    spent a probing move outside the accounting and gave up entirely when one
    probe was inconclusive -- which happens whenever the probing action leaves
    the agent where it was. Here the intersection of everything seen so far is
    narrowed while acting, and the actions tried are cycled so a self-loop does
    not stall it forever.
    """

    verifier = SceneVerifier(task, start=start, goal=goal, steps=steps)
    policy = object_policy(model, front)
    common: set[int] | None = None
    remembered: int | None = None if told is None else int(told)
    total = 0.0
    scored = 0
    probes = 0
    while not verifier.done:
        frame = verifier.observation()
        places = set(front.read(frame))
        common = places if common is None else (common & places)
        if remembered is None and common is not None and len(common) == 1:
            remembered = next(iter(common))
        if remembered is None:
            # Not yet sure which marker is the instruction. Try a different
            # action each time, so an action that holds still cannot trap this.
            action = probes % task.action_count
            probes += 1
        else:
            action, remembered = policy(frame, remembered)
        total += verifier.score(torch.tensor([action], dtype=torch.long))
        scored += 1
    return total / scored if scored else 0.0


# --- the scene agent, which is the control ---------------------------------


def explore_scenes(
    front: SceneFrontend,
    task: NavigationTask,
    pairs,
    *,
    steps: int,
    seed: int,
) -> WorldModel:
    """The same wandering, read without decomposing."""

    model = WorldModel(
        place_count=int(front.clusters.shape[0]), action_count=task.action_count
    )
    generator = torch.Generator().manual_seed(int(seed))
    for start, goal in pairs:
        verifier = SceneVerifier(task, start=start, goal=goal, steps=steps)
        while not verifier.done:
            scene = front.read(verifier.observation())
            action = int(
                torch.randint(0, task.action_count, (1,), generator=generator).item()
            )
            reward = verifier.score(torch.tensor([action], dtype=torch.long))
            if verifier.done:
                break
            model.observe(
                scene, action, front.read(verifier.observation()), int(reward)
            )
    return model


def run_scene_episode(
    front: SceneFrontend,
    model: WorldModel,
    task: NavigationTask,
    *,
    start: int,
    goal: int,
    steps: int,
) -> float:
    """Plan over the graph of pictures towards a picture that once paid."""

    verifier = SceneVerifier(task, start=start, goal=goal, steps=steps)
    goals = model.goals()
    total = 0.0
    scored = 0
    while not verifier.done:
        scene = front.read(verifier.observation())
        action = 0
        if goals:
            if scene in set(goals):
                holding = model.holding_action(scene)
                action = holding if holding is not None else 0
            else:
                route = plan_to(model, scene, goals)
                if route is not None and route.actions:
                    action = int(route.actions[0])
        total += verifier.score(torch.tensor([action], dtype=torch.long))
        scored += 1
    return total / scored if scored else 0.0


# --- the experiment --------------------------------------------------------


def measure_goals(
    goals,
    *,
    task,
    steps,
    slot_front,
    scene_front,
    model,
    scene_model,
    goal_cluster,
    seed,
    index,
):
    """Score every agent and every control on one set of goals.

    A module-level function rather than a closure over the task loop: the
    closure captured loop variables and was only correct because it happened
    to be called before the next iteration, which is the kind of thing that
    stays correct until someone moves a line.
    """

    object_scores, scene_scores, optimal = [], [], []
    told_scores, wrong_scores, random_scores = [], [], []
    chooser = torch.Generator().manual_seed(seed + 77 * index)
    for goal in goals:
        for start in range(PLACE_COUNT):
            if shortest(task, start, goal) > steps:
                continue
            optimal.append(best_return(task, start, goal, steps))
            object_scores.append(
                run_object_episode(
                    slot_front,
                    model,
                    task,
                    start=start,
                    goal=goal,
                    steps=steps,
                )
            )
            scene_scores.append(
                run_scene_episode(
                    scene_front,
                    scene_model,
                    task,
                    start=start,
                    goal=goal,
                    steps=steps,
                )
            )
            # The planner's ceiling: identification handed over, so
            # what remains is the model and the search alone.
            told_scores.append(
                run_object_episode(
                    slot_front,
                    model,
                    task,
                    start=start,
                    goal=goal,
                    steps=steps,
                    told=goal_cluster[goal],
                )
            )
            # The control that checks the shown goal is being used at
            # all: plan to some other place instead.
            other = int(
                torch.randint(
                    0, PLACE_COUNT, (1,), generator=chooser
                ).item()
            )
            wrong_scores.append(
                run_object_episode(
                    slot_front,
                    model,
                    task,
                    start=start,
                    goal=goal,
                    steps=steps,
                    told=goal_cluster[other],
                )
            )
            random_scores.append(
                run_random_episode(
                    task,
                    start=start,
                    goal=goal,
                    steps=steps,
                    seed=seed + 5000 + start,
                )
            )
    count = max(1, len(optimal))
    return {
        "object": sum(object_scores) / count,
        "scene": sum(scene_scores) / count,
        "told": sum(told_scores) / count,
        "wrong_goal": sum(wrong_scores) / count,
        "random": sum(random_scores) / count,
        "optimal": sum(optimal) / count,
        "episodes": len(optimal),
    }


def run_object_navigation(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    tasks: int = 8,
    steps: int = EPISODE_STEPS,
    explore_episodes: int = EXPLORE_EPISODES,
    training_goals: int = 4,
) -> dict[str, Any]:
    """Train on some goals; be asked for others."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index in range(tasks):
        task = sample_navigation_task(seed=9000 + 37 * index)
        if task is None:
            continue
        slot_front, scene_front = build_frontends(encoders, task=task, steps=steps)

        generator = torch.Generator().manual_seed(seed + index)
        seen_goals = list(range(training_goals))
        held_out = list(range(training_goals, PLACE_COUNT))
        pairs = []
        for episode in range(explore_episodes):
            start = int(
                torch.randint(0, PLACE_COUNT, (1,), generator=generator).item()
            )
            pairs.append((start, seen_goals[episode % len(seen_goals)]))

        model, identified = explore_objects(
            slot_front, task, pairs, steps=steps, seed=seed + 100 * index
        )
        scene_model = explore_scenes(
            scene_front, task, pairs, steps=steps, seed=seed + 100 * index
        )

        # Scoring-side: which slot cluster each place renders to, so a *told*
        # goal can be expressed in the agent's own alphabet. Used only to hand
        # over an instruction the agent would otherwise have had to work out.
        goal_cluster = tuple(
            slot_front.read(render_scene(place, place, size=FRAME_SIZE))[0]
            for place in range(PLACE_COUNT)
        )
        trained = measure_goals(
            seen_goals,
            task=task,
            steps=steps,
            slot_front=slot_front,
            scene_front=scene_front,
            model=model,
            scene_model=scene_model,
            goal_cluster=goal_cluster,
            seed=seed,
            index=index,
        )
        new = measure_goals(
            held_out,
            task=task,
            steps=steps,
            slot_front=slot_front,
            scene_front=scene_front,
            model=model,
            scene_model=scene_model,
            goal_cluster=goal_cluster,
            seed=seed,
            index=index,
        )
        rows.append(
            {
                "task": index,
                "goals_identified": identified,
                "explore_episodes": len(pairs),
                "slot_alphabet": int(slot_front.clusters.shape[0]),
                "scene_alphabet": int(scene_front.clusters.shape[0]),
                "model_coverage": model.coverage,
                "trained_goals": trained,
                "held_out_goals": new,
            }
        )

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the object navigation run mutated AgentBrain.bank")

    def mean(group: str, key: str) -> float:
        return (
            sum(float(row[group][key]) for row in rows) / len(rows) if rows else 0.0
        )

    report = {
        "schema": OBJECT_NAVIGATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "tasks": len(rows),
        "episode_steps": steps,
        "explore_episodes": explore_episodes,
        "training_goals": training_goals,
        "held_out_goals": PLACE_COUNT - training_goals,
        "mean_slot_alphabet": (
            sum(int(row["slot_alphabet"]) for row in rows) / len(rows) if rows else 0
        ),
        "mean_scene_alphabet": (
            sum(int(row["scene_alphabet"]) for row in rows) / len(rows) if rows else 0
        ),
        "mean_goals_identified": (
            sum(int(row["goals_identified"]) for row in rows) / len(rows)
            if rows
            else 0
        ),
        "trained_object": mean("trained_goals", "object"),
        "trained_scene": mean("trained_goals", "scene"),
        "trained_told": mean("trained_goals", "told"),
        "trained_wrong_goal": mean("trained_goals", "wrong_goal"),
        "trained_random": mean("trained_goals", "random"),
        "trained_optimal": mean("trained_goals", "optimal"),
        "held_out_object": mean("held_out_goals", "object"),
        "held_out_scene": mean("held_out_goals", "scene"),
        "held_out_told": mean("held_out_goals", "told"),
        "held_out_wrong_goal": mean("held_out_goals", "wrong_goal"),
        "held_out_random": mean("held_out_goals", "random"),
        "held_out_optimal": mean("held_out_goals", "optimal"),
        "rows": rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "object_navigation.json").write_text(
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
            / "brainworkshop_object_navigation_2026-08-15"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--explore-episodes", type=int, default=EXPLORE_EPISODES)
    arguments = parser.parse_args()
    report = run_object_navigation(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
        tasks=arguments.tasks,
        explore_episodes=arguments.explore_episodes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
