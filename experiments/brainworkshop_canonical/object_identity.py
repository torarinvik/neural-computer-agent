"""Which one am I, when something else is moving too.

The object-navigation agent found itself by elimination. The goal is the place
present in every observation; whatever else is in the scene is me. That is
correct exactly while the scene holds two things, and the record said so:
*identity is worked out from persistence, which works because the goal does not
move.*

Adding a third marker that moves on its own separates the two halves of that
rule and only one of them survives.

**Finding the goal still works.** It is still the only place in every frame, so
intersection still returns it. That half was never the fragile one.

**Finding myself does not.** Elimination now leaves two candidates and the rule
picks whichever sorted first, which is a coin flip on every frame. The agent
then builds its model of the dynamics out of somebody else's movements.

What replaces it is the discrimination DeepMind's AlignNet work sets up but
does not itself make: alignment says a track is the *same thing* over time, and
what says *which* thing is the only evidence an agent can manufacture rather
than wait for -- whether the action mattered.

That distinction is finer than it first looks, and the `cycling` condition is
here to insist on it. A distractor walking a fixed circuit is **perfectly
predictable**: its next place follows from its current one with no error at
all. What it is not is **responsive**. So an agent that picks itself by asking
"which track can I predict?" chooses the distractor about as often as itself,
and an agent that asks "which track does knowing my action help me predict?"
does not. Both are measured, because the second only earns its complexity if
the first actually fails.

Conditions change one thing at a time, from the previous record's setting:

- `none` -- two markers, exactly the earlier experiment, where persistence must
  still work or something has been broken in passing;
- `random_walk` -- a third marker stepping to a random place each tick;
- `cycling` -- a third marker on a fixed circuit, which is the adversarial one.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neural_computer.promotion import sha256_file

from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .navigation_environment import NavigationTask, sample_navigation_task
from .object_scene import PLACE_COUNT, render_markers
from .slot_alignment import Tracker, TrackHistory, beam_track, identify_roles
from .successor_transfer import SlotReader, build_slot_reader
from .world_model import WorldModel

EXPERIMENT_ID = "brainworkshop-object-identity-2026-08-16"
OBJECT_IDENTITY_SCHEMA = "neural-computer.object-identity.v1"
DEVELOPMENT_SEED = 41
# Worlds are drawn from here. Defaulting to the development value keeps
# every recorded diagnostic reproducing exactly; a holdout run passes a seed
# from an unused block so the *worlds* are unseen and not merely the
# exploration randomness.
DEVELOPMENT_WORLD_SEED = 9000
WORLD_SEED_STRIDE = 37
EPISODE_STEPS = 20
EXPLORE_EPISODES = 16
FRAME_SIZE = 36
CONDITIONS = ("none", "random_walk", "cycling")
ARMS = ("hybrid", "search", "alignment", "predictability", "persistence")


class DistractorVerifier:
    """The navigation scene, plus something else that moves for its own reasons.

    The distractor is not an obstacle and not a second goal. It changes nothing
    about the dynamics or the reward -- it is there purely so that "the thing
    that moved" stops naming one object.
    """

    def __init__(
        self,
        task: NavigationTask,
        *,
        start: int,
        goal: int,
        steps: int,
        condition: str,
        seed: int,
        frame_size: int = FRAME_SIZE,
    ) -> None:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown distractor condition: {condition}")
        self.task = task.validate()
        self.goal = int(goal)
        self.steps = int(steps)
        self.condition = condition
        self.frame_size = int(frame_size)
        self._place = int(start)
        self._position = 0
        self._generator = torch.Generator().manual_seed(int(seed))
        self._distractor = int(
            torch.randint(0, PLACE_COUNT, (1,), generator=self._generator).item()
        )
        # A fixed circuit through every place, so the distractor is completely
        # predictable from its own position and completely deaf to the agent.
        order = torch.randperm(PLACE_COUNT, generator=self._generator).tolist()
        self._circuit = {
            int(order[index]): int(order[(index + 1) % PLACE_COUNT])
            for index in range(PLACE_COUNT)
        }

    @property
    def done(self) -> bool:
        return self._position >= self.steps

    @property
    def place(self) -> int:
        """Scoring-side truth, read only by the accounting."""

        return self._place

    @property
    def distractor(self) -> int | None:
        return None if self.condition == "none" else self._distractor

    def observation(self) -> torch.Tensor:
        if self.done:
            raise RuntimeError("distractor episode is complete")
        markers = [self.goal, self._place]
        if self.condition != "none":
            markers.insert(1, self._distractor)
        return render_markers(markers, size=self.frame_size)

    def _advance_distractor(self) -> None:
        if self.condition == "random_walk":
            self._distractor = int(
                torch.randint(0, PLACE_COUNT, (1,), generator=self._generator).item()
            )
        elif self.condition == "cycling":
            self._distractor = self._circuit[self._distractor]

    def score(self, action: torch.Tensor) -> float:
        if self.done:
            raise RuntimeError("distractor episode is complete")
        chosen = int(action.item())
        if not 0 <= chosen < self.task.action_count:
            raise ValueError("distractor action is outside the protocol")
        self._place = int(self.task.transitions[chosen][self._place])
        self._advance_distractor()
        self._position += 1
        return float(self._place == self.goal)


# --- the two ways of working out who you are -------------------------------


@dataclass(frozen=True)
class Identification:
    """What an arm concluded, and what was actually true."""

    own: int | None
    goal: int | None
    true_own: int
    true_goal: int

    @property
    def own_correct(self) -> bool:
        return self.own is not None and self.own == self.true_own

    @property
    def goal_correct(self) -> bool:
        return self.goal is not None and self.goal == self.true_goal


def persistence_identify(observations, final) -> tuple[int | None, int | None]:
    """The previous record's rule: the goal never moved, and I am the rest.

    `final` is the last observation, from which "me" is read by elimination.
    With two markers that is exact. With three it picks one of two arbitrarily,
    which is the failure this experiment is about.
    """

    common: set[int] | None = None
    for places in observations:
        current = set(places)
        common = current if common is None else (common & current)
        if not common:
            return None, None
    if common is None or len(common) != 1:
        return None, None
    goal = next(iter(common))
    others = [place for place in final if place != goal]
    return (others[0] if others else goal), goal


def explore_episode(
    reader: SlotReader,
    verifier: DistractorVerifier,
    *,
    action_count: int,
    generator: torch.Generator,
):
    """Act at random and watch. The only experience anything here gets.

    Returns the raw readings, the actions, and the true places, so that every
    arm is scored on exactly the same episode and differs only in how it reads
    it.

    Random rather than cycling, and that is not a detail. Controllability asks
    whether knowing the action improves the prediction, which is a question
    with no answer unless the same place is left by different actions. Under a
    deterministic probe policy the action is a function of the step, the agent
    looks exactly as unresponsive as a distractor on a circuit, and the whole
    discrimination collapses -- measured, it scored below chance.
    """

    readings = []
    actions: list[int] = []
    truth: list[int] = []
    while not verifier.done:
        truth.append(verifier.place)
        readings.append(reader.read(verifier.observation()))
        action = int(
            torch.randint(0, int(action_count), (1,), generator=generator).item()
        )
        actions.append(action)
        verifier.score(torch.tensor([action], dtype=torch.long))
    return readings, actions, truth


def traces_for(arm: str, readings, actions, *, alphabet: int):
    """The per-step track symbols this arm believes in."""

    if arm in ("search", "hybrid"):
        return list(beam_track(readings, actions[:-1], alphabet=alphabet))
    tracker = Tracker.started(readings[0])
    traces = [tuple(tracker.reading())]
    for step, reading in enumerate(readings[1:]):
        tracker.update(reading, action=actions[step])
        traces.append(tuple(tracker.reading()))
    return traces


def histories_from(traces, actions) -> list[TrackHistory]:
    width = max((len(row) for row in traces), default=0)
    histories = [TrackHistory() for _ in range(width)]
    for step in range(len(traces) - 1):
        before, after = traces[step], traces[step + 1]
        for track in range(min(len(before), len(after))):
            histories[track].observe(before[track], actions[step], after[track])
    return histories


def identify(
    arm: str,
    histories: list[TrackHistory],
    final: Sequence[int],
    seen: list[tuple[int, ...]],
    *,
    alphabet: int,
) -> tuple[int | None, int | None]:
    """Own symbol and goal symbol, by whichever rule this arm uses."""

    if arm == "persistence":
        return persistence_identify(seen, seen[-1] if seen else ())
    if not histories:
        return None, None
    if arm == "hybrid":
        # Each half by the mechanism that suits it. The goal is static, so the
        # place present in every frame names it without any correspondence
        # being established at all -- that rule was never the broken one. Which
        # marker is *me* needs correspondence, and gets it from the search.
        _, found = persistence_identify(seen, seen[-1] if seen else ())
        roles = identify_roles(histories, alphabet=alphabet)
        return (
            None if roles.own is None else _symbol(final, roles.own),
            found,
        )
    if arm == "predictability":
        # The plausible wrong answer: pick the track you can predict best.
        evidence = [history.evidence(alphabet=alphabet) for history in histories]
        own = max(range(len(evidence)), key=lambda index: evidence[index].with_action)
        still = [
            index
            for index in range(len(evidence))
            if evidence[index].moved <= 0.05 and index != own
        ]
        goal = still[0] if len(still) == 1 else None
        return (
            _symbol(final, own),
            None if goal is None else _symbol(final, goal),
        )
    roles = identify_roles(histories, alphabet=alphabet)
    return (
        None if roles.own is None else _symbol(final, roles.own),
        None if roles.target is None else _symbol(final, roles.target),
    )


def _symbol(final: Sequence[int], track: int) -> int | None:
    return int(final[track]) if 0 <= track < len(final) else None


def run_condition(
    reader: SlotReader,
    task: NavigationTask,
    *,
    condition: str,
    steps: int,
    explore_episodes: int,
    seed: int,
    cluster_of_place,
) -> dict[str, Any]:
    """Explore once; let every arm read the same experience its own way."""

    generator = torch.Generator().manual_seed(int(seed))
    models = {arm: WorldModel(reader.alphabet, task.action_count) for arm in ARMS}
    correct = {arm: {"own": 0, "goal": 0, "follow": 0, "frames": 0} for arm in ARMS}
    attempted = 0

    for episode in range(explore_episodes):
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
        readings, actions, truth = explore_episode(
            reader,
            verifier,
            action_count=task.action_count,
            generator=generator,
        )
        attempted += 1
        seen = [tuple(symbol for _, symbol in frame) for frame in readings]
        for arm in ARMS:
            traces = traces_for(arm, readings, actions, alphabet=reader.alphabet)
            histories = histories_from(traces, actions)
            own, found = identify(
                arm, histories, traces[-1], seen, alphabet=reader.alphabet
            )
            if own == cluster_of_place[truth[-1]]:
                correct[arm]["own"] += 1
            if found == cluster_of_place[goal]:
                correct[arm]["goal"] += 1
            # How faithfully the arm's best track follows the agent, whichever
            # track that turns out to be: tracking quality with the
            # identification question set aside.
            width = min(len(row) for row in traces)
            if width:
                best = max(
                    range(width),
                    key=lambda track: sum(
                        1
                        for step, row in enumerate(traces)
                        if row[track] == cluster_of_place[truth[step]]
                    ),
                )
                correct[arm]["follow"] += sum(
                    1
                    for step, row in enumerate(traces)
                    if row[best] == cluster_of_place[truth[step]]
                )
                correct[arm]["frames"] += len(traces)
            if own is None:
                continue
            track = _track_of_symbol(traces[-1], own)
            if track is None:
                continue
            # The model is built from whichever track the arm believes is the
            # agent, which is exactly how a wrong identification does its
            # damage: not by being wrong once, but by poisoning the model.
            for symbol, action, following in histories[track].steps:
                models[arm].observe(symbol, action, following, 0)

    scored: dict[str, Any] = {
        "condition": condition,
        "explore_episodes": attempted,
    }
    for arm in ARMS:
        scored[f"{arm}_own_accuracy"] = (
            correct[arm]["own"] / attempted if attempted else 0.0
        )
        scored[f"{arm}_goal_accuracy"] = (
            correct[arm]["goal"] / attempted if attempted else 0.0
        )
        scored[f"{arm}_track_fidelity"] = (
            correct[arm]["follow"] / correct[arm]["frames"]
            if correct[arm]["frames"]
            else 0.0
        )
        scored[f"{arm}_coverage"] = models[arm].coverage
        scored[f"{arm}_model_accuracy"] = _model_accuracy(
            models[arm], task, cluster_of_place
        )
    return scored


def _track_of_symbol(final: Sequence[int], symbol: int) -> int | None:
    for index, value in enumerate(final):
        if int(value) == int(symbol):
            return index
    return None


def _model_accuracy(model: WorldModel, task: NavigationTask, cluster_of_place) -> float:
    """How much of what the model believes about the dynamics is true.

    Scoring-side. The identification arms differ in what they *learn*, and this
    is where that difference becomes visible before any behaviour does.
    """

    place_of_cluster = {int(cluster): place for place, cluster in enumerate(cluster_of_place)}
    known = 0
    right = 0
    for action in range(task.action_count):
        for cluster in range(model.place_count):
            believed = model.successor(cluster, action)
            if believed is None or cluster not in place_of_cluster:
                continue
            known += 1
            place = place_of_cluster[cluster]
            truth = int(task.transitions[action][place])
            if believed == int(cluster_of_place[truth]):
                right += 1
    return right / known if known else 0.0


def run_object_identity(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    world_seed: int = DEVELOPMENT_WORLD_SEED,
    tasks: int = 4,
    steps: int = EPISODE_STEPS,
    explore_episodes: int = EXPLORE_EPISODES,
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
        for condition in CONDITIONS:
            scored = run_condition(
                reader,
                task,
                condition=condition,
                steps=steps,
                explore_episodes=explore_episodes,
                seed=seed + 100 * index,
                cluster_of_place=cluster_of_place,
            )
            scored["task"] = index
            rows.append(scored)

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the object identity run mutated AgentBrain.bank")

    summary: dict[str, Any] = {}
    for condition in CONDITIONS:
        block = [row for row in rows if row["condition"] == condition]
        if not block:
            continue
        entry: dict[str, Any] = {"tasks": len(block)}
        for arm in ARMS:
            for key in (
                "own_accuracy",
                "goal_accuracy",
                "track_fidelity",
                "coverage",
                "model_accuracy",
            ):
                entry[f"{arm}_{key}"] = sum(
                    float(row[f"{arm}_{key}"]) for row in block
                ) / len(block)
        summary[condition] = entry

    report = {
        "schema": OBJECT_IDENTITY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "world_seed": world_seed,
        "tasks": tasks,
        "episode_steps": steps,
        "explore_episodes": explore_episodes,
        "conditions": summary,
        "rows": rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "object_identity.json").write_text(
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
            repository / "session_records" / "brainworkshop_object_identity_2026-08-16"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument(
        "--world-seed", type=int, default=DEVELOPMENT_WORLD_SEED
    )
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--explore-episodes", type=int, default=EXPLORE_EPISODES)
    arguments = parser.parse_args()
    report = run_object_identity(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
        world_seed=arguments.world_seed,
        tasks=arguments.tasks,
        explore_episodes=arguments.explore_episodes,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
