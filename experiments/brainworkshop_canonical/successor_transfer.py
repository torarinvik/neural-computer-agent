"""A task becomes a vector, and the model stops being thrown away.

The object-navigation agent could be shown where to go. Two things it could
not do: be asked for anything other than a place, and keep what it learned. Its
model served one goal and was discarded, so "go to five" and "go to six" were
two problems rather than one problem twice.

Successor features answer both, and answer them with the same object. A policy
stores what it goes on to *see* rather than what that is worth, a task is a
weight vector over those features, and the value of any stored policy under any
new task is one dot product. Generalised policy improvement then acts greedily
across everything stored.

So the measurement has two halves and they are deliberately separated.

**In the model, with nothing else moving.** Given four policies induced for the
training goals, how close does generalised policy improvement get to the
optimal policy for a task it has never seen -- without planning, without a
sweep, without a step in the world? And how does that close as the library
grows? That is the accumulation claim, and it is measured against the re-solved
optimum rather than against the stored policies, because beating a policy built
for a different goal is not an achievement.

**In the world, through pixels.** The same agent, reading a rendered scene
through the frozen encoder, tracking its own marker, forming `w` from what the
scene shows it, and acting. Three task families:

- `single` -- one target marker, which is the previous experiment restated as a
  one-hot vector and is therefore the check that this generalises rather than
  replaces;
- `disjunction` -- two target markers, "reach either", which the old goal
  language could not represent at all;
- `avoid` -- one target and a negative weight on a hazard, which no stored
  goal-reaching policy is good for, because each of them is happy to walk
  straight through it.

The hazard is **given**, not shown. One colour cannot say "not here", and
inventing a second one would undo the reason the markers share a colour in the
first place. It is declared where it is used, and the `single` and
`disjunction` families are shown in the scene exactly as before.
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
from .object_scene import PLACE_COUNT, encode_slots, render_markers, scene_parts
from .prototype_templates import cluster_events, estimated_tolerance
from .slot_alignment import Tracker, TrackHistory, identify_roles
from .successor_features import (
    DEFAULT_DISCOUNT,
    SuccessorFeatureLibrary,
    SuccessorFeatureRecord,
    generalised_policy_improvement,
    gpi_policy,
    greedy_policy,
    policy_values,
    reach,
    successor_features,
)
from .world_model import WorldModel

EXPERIMENT_ID = "brainworkshop-successor-transfer-2026-08-16"
SUCCESSOR_TRANSFER_SCHEMA = "neural-computer.successor-transfer.v1"
DEVELOPMENT_SEED = 41
# Worlds are drawn from here. Defaulting to the development value keeps
# every recorded diagnostic reproducing exactly; a holdout run passes a seed
# from an unused block so the *worlds* are unseen and not merely the
# exploration randomness.
DEVELOPMENT_WORLD_SEED = 9000
WORLD_SEED_STRIDE = 37
EPISODE_STEPS = 20
EXPLORE_EPISODES = 24
FRAME_SIZE = 36
TRAINING_GOALS = 4
HAZARD_PENALTY = 1.0


# --- an environment that pays a weight vector ------------------------------


class WeightedVerifier:
    """Shows the agent and its targets; pays whatever the task weights say.

    The weights it scores against are indexed by *place*. The agent never sees
    them: it sees markers, and forms its own vector over its own alphabet. The
    two only have to agree for the measurement to mean anything, and that
    agreement is what is being measured.
    """

    def __init__(
        self,
        task: NavigationTask,
        *,
        start: int,
        shown: tuple[int, ...],
        weights_by_place: torch.Tensor,
        steps: int,
        frame_size: int = FRAME_SIZE,
    ) -> None:
        self.task = task.validate()
        self.shown = tuple(int(place) for place in shown)
        self.weights = weights_by_place.double()
        if int(self.weights.shape[0]) != task.place_count:
            raise ValueError("task weights must be defined at every place")
        self.steps = int(steps)
        self.frame_size = int(frame_size)
        self._place = int(start)
        self._position = 0

    @property
    def done(self) -> bool:
        return self._position >= self.steps

    @property
    def place(self) -> int:
        """Where the agent actually is. Scoring-side, and an oracle.

        Read by exactly one thing: the `told` arms, which exist to separate the
        cost of working out which marker you are from the quality of the plan
        that follows. Any other reader of this would be cheating.
        """

        return self._place

    def observation(self) -> torch.Tensor:
        if self.done:
            raise RuntimeError("weighted episode is complete")
        return render_markers((*self.shown, self._place), size=self.frame_size)

    def score(self, action: torch.Tensor) -> float:
        if self.done:
            raise RuntimeError("weighted episode is complete")
        chosen = int(action.item())
        if not 0 <= chosen < self.task.action_count:
            raise ValueError("weighted action is outside the protocol")
        self._place = int(self.task.transitions[chosen][self._place])
        self._position += 1
        return float(self.weights[self._place])


def best_weighted_return(
    task: NavigationTask, start: int, weights: torch.Tensor, steps: int
) -> float:
    """The most this task can pay, by exact finite-horizon dynamic programming.

    Scoring-side. Written as a ceiling rather than as a plan: with negative
    weights a shortest path is no longer the answer, so nothing shorter than
    the full backward induction is correct here.
    """

    horizon = int(steps)
    payout = [float(value) for value in weights]
    value = [0.0] * task.place_count
    for _ in range(horizon):
        value = [
            max(
                payout[int(task.transitions[action][place])]
                + value[int(task.transitions[action][place])]
                for action in range(task.action_count)
            )
            for place in range(task.place_count)
        ]
    return value[int(start)] / horizon if horizon else 0.0


def worst_weighted_return(
    task: NavigationTask, start: int, weights: torch.Tensor, steps: int
) -> float:
    """The same, minimised. The floor a task can be normalised against."""

    horizon = int(steps)
    payout = [float(value) for value in weights]
    value = [0.0] * task.place_count
    for _ in range(horizon):
        value = [
            min(
                payout[int(task.transitions[action][place])]
                + value[int(task.transitions[action][place])]
                for action in range(task.action_count)
            )
            for place in range(task.place_count)
        ]
    return value[int(start)] / horizon if horizon else 0.0


# --- reading the scene as tracks -------------------------------------------


@dataclass
class SlotReader:
    """Parts of a scene, each as a centroid and a place symbol."""

    encoders: Any
    clusters: torch.Tensor

    def read_with_events(self, frame: torch.Tensor):
        """Read slots and retain the learned event vectors at the same seam.

        The ordinary ``read`` API intentionally exposes only the small symbol
        used by the development navigation models.  The external identity
        artifact, however, needs the learned event tensors themselves.  Doing
        both operations here avoids encoding the same rendered frame twice in
        the live loop and keeps that artifact outside the controller.
        """

        parts = scene_parts(frame)
        events = encode_slots(self.encoders, frame)
        symbols = [int(index) for index in nearest_cluster(events, self.clusters)]
        return tuple(
            (centroid, symbol)
            for (centroid, _), symbol in zip(parts, symbols, strict=True)
        ), events

    def read(self, frame: torch.Tensor):
        parts, _ = self.read_with_events(frame)
        return parts

    @property
    def alphabet(self) -> int:
        return int(self.clusters.shape[0])


def build_slot_reader(encoders) -> SlotReader:
    """Discover the place alphabet from every scene the agent could meet."""

    events = torch.cat(
        [
            encode_slots(encoders, render_markers((goal, agent), size=FRAME_SIZE))
            for agent in range(PLACE_COUNT)
            for goal in range(PLACE_COUNT)
        ]
    )
    tolerance = estimated_tolerance(events)
    if tolerance is None:
        raise ValueError("the slot alphabet could not be estimated")
    return SlotReader(
        encoders, cluster_events(events, tolerance=tolerance, maximum_clusters=256)
    )


# --- exploration, which is the only experience anything gets ---------------


def explore(
    reader: SlotReader,
    task: NavigationTask,
    pairs,
    *,
    steps: int,
    seed: int,
) -> tuple[WorldModel, int]:
    """Wander under single shown goals; keep only what happened to *me*."""

    model = WorldModel(place_count=reader.alphabet, action_count=task.action_count)
    generator = torch.Generator().manual_seed(int(seed))
    identified = 0
    for start, goal in pairs:
        verifier = WeightedVerifier(
            task,
            start=start,
            shown=(goal,),
            weights_by_place=torch.zeros(task.place_count, dtype=torch.float64),
            steps=steps,
        )
        tracker: Tracker | None = None
        histories: list[TrackHistory] = []
        pending: list[tuple[int, int]] = []
        last_action: int | None = None
        while not verifier.done:
            parts = reader.read(verifier.observation())
            if tracker is None:
                tracker = Tracker.started(parts)
                histories = [TrackHistory() for _ in range(tracker.count)]
            else:
                tracker.update(parts, action=last_action)
                while len(histories) < tracker.count:
                    histories.append(TrackHistory())
                for track, (symbol, action) in enumerate(pending):
                    if track < tracker.count:
                        histories[track].observe(symbol, action, tracker.symbols[track])
            before = tracker.reading()
            action = int(
                torch.randint(0, task.action_count, (1,), generator=generator).item()
            )
            pending = [(symbol, action) for symbol in before]
            last_action = action
            verifier.score(torch.tensor([action], dtype=torch.long))
        roles = identify_roles(histories)
        if roles.own is None:
            continue
        identified += 1
        for symbol, action, following in histories[roles.own].steps:
            model.observe(symbol, action, following, 0)
    return model, identified


# --- the arms ---------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeOutcome:
    """What the episode paid, and how much of it went on finding out who I am."""

    score: float
    probes: int
    steps: int


def run_weighted_episode(
    reader: SlotReader,
    task: NavigationTask,
    *,
    start: int,
    shown: tuple[int, ...],
    weights_by_place: torch.Tensor,
    given: torch.Tensor | None,
    steps: int,
    chooser,
    told: tuple[int, ...] | None = None,
) -> EpisodeOutcome:
    """One episode, with `chooser(place, weights) -> action` doing the work.

    Identification runs online and every step is scored, as in the object
    navigation record: until the agent knows which track it is, it cycles
    actions, and those steps count against it. `given` is the part of the task
    the scene cannot show -- the hazard weights -- handed over in the agent's
    own alphabet.

    `told` is the declared oracle: the place-to-cluster table, which turns the
    agent's own position into a symbol without it having to work anything out.
    It measures what identification costs, and is never the headline arm.
    """

    verifier = WeightedVerifier(
        task,
        start=start,
        shown=shown,
        weights_by_place=weights_by_place,
        steps=steps,
    )
    tracker: Tracker | None = None
    histories: list[TrackHistory] = []
    pending: list[tuple[int, int]] = []
    last_action: int | None = None
    total = 0.0
    scored = 0
    probes = 0
    while not verifier.done:
        parts = reader.read(verifier.observation())
        if tracker is None:
            tracker = Tracker.started(parts)
            histories = [TrackHistory() for _ in range(tracker.count)]
        else:
            tracker.update(parts, action=last_action)
            while len(histories) < tracker.count:
                histories.append(TrackHistory())
            for track, (symbol, action) in enumerate(pending):
                if track < tracker.count:
                    histories[track].observe(symbol, action, tracker.symbols[track])
        if told is not None:
            own_symbol: int | None = int(told[verifier.place])
            target_symbols: tuple[int, ...] = tuple(
                int(told[place]) for place in shown
            )
        else:
            roles = identify_roles(histories)
            own_symbol = (
                None if roles.own is None else int(tracker.symbols[roles.own])
            )
            target_symbols = tuple(
                int(tracker.symbols[target]) for target in roles.targets
            )
        if own_symbol is None or not target_symbols:
            action = probes % task.action_count
            probes += 1
        else:
            weights = torch.zeros(reader.alphabet, dtype=torch.float64)
            for symbol in target_symbols:
                weights[symbol] = 1.0
            if given is not None:
                weights = weights + given
            action = chooser(own_symbol, weights)
        pending = [(symbol, action) for symbol in tracker.reading()]
        last_action = action
        total += verifier.score(torch.tensor([action], dtype=torch.long))
        scored += 1
    return EpisodeOutcome(
        score=total / scored if scored else 0.0, probes=probes, steps=scored
    )


def gpi_chooser(library: SuccessorFeatureLibrary):
    """No planning at all: dot products against what is already stored."""

    psis = library.psis()

    def choose(place: int, weights: torch.Tensor) -> int:
        return generalised_policy_improvement(psis, place, weights)

    return choose


def replan_chooser(model: WorldModel, *, discount: float = DEFAULT_DISCOUNT):
    """The control that re-solves the whole task every time it is asked.

    Cached per weight vector, because the point of the comparison is the
    quality of the answer, not how slowly this arm is implemented.
    """

    cache: dict[tuple, tuple[int, ...]] = {}

    def choose(place: int, weights: torch.Tensor) -> int:
        key = tuple(round(float(value), 6) for value in weights)
        if key not in cache:
            cache[key] = greedy_policy(model, weights, discount=discount)
        return int(cache[key][int(place)])

    return choose


def single_chooser(library: SuccessorFeatureLibrary, *, discount: float):
    """Follow whichever one stored policy is best for this task, and only it.

    The floor generalised policy improvement is guaranteed to clear. Included
    because clearing it is not the interesting claim, and the size of the gap
    to the re-solved optimum is.
    """

    records = library.records()

    def choose(place: int, weights: torch.Tensor) -> int:
        best = max(
            records,
            key=lambda record: policy_values(record.psi, record.policy, weights)[
                int(place)
            ],
        )
        return int(best.policy[int(place)])

    return choose


def random_chooser(seed: int, action_count: int):
    """The floor: act without reading anything."""

    generator = torch.Generator().manual_seed(int(seed))

    def choose(place: int, weights: torch.Tensor) -> int:
        return int(
            torch.randint(0, int(action_count), (1,), generator=generator).item()
        )

    return choose


# --- the experiment ---------------------------------------------------------


def task_families(place_count: int, held_out, generator) -> list[dict[str, Any]]:
    """The three goal languages, in places rather than clusters."""

    held = list(held_out)
    families: list[dict[str, Any]] = []
    for place in held:
        families.append({"family": "single", "shown": (place,), "hazard": None})
    for index in range(len(held)):
        pair = (held[index], held[(index + 1) % len(held)])
        if pair[0] != pair[1]:
            families.append({"family": "disjunction", "shown": pair, "hazard": None})
    for place in held:
        hazard = int(torch.randint(0, place_count, (1,), generator=generator).item())
        if hazard == place:
            continue
        families.append({"family": "avoid", "shown": (place,), "hazard": hazard})
    return families


def weights_for(place_count: int, shown, hazard) -> torch.Tensor:
    weights = torch.zeros(int(place_count), dtype=torch.float64)
    for place in shown:
        weights[int(place)] = 1.0
    if hazard is not None:
        weights[int(hazard)] -= HAZARD_PENALTY
    return weights


def in_model_accumulation(
    model: WorldModel,
    policies,
    held_out,
    *,
    discount: float,
) -> list[dict[str, Any]]:
    """How close stitching gets to optimal as the library grows, offline.

    No environment, no frontend, no episode: purely the question of whether
    storing more policies makes a novel task cheaper, which is the claim the
    accumulation machinery has always been about and has never been asked here.
    """

    rows: list[dict[str, Any]] = []
    places = model.place_count
    tasks = [reach(places, place) for place in held_out]
    for size in range(1, len(policies) + 1):
        subset = policies[:size]
        psis = [successor_features(model, policy, discount=discount) for policy in subset]
        ratios = []
        for weights in tasks:
            stitched = gpi_policy(psis, places, weights)
            stitched_value = policy_values(
                successor_features(model, stitched, discount=discount),
                stitched,
                weights,
            )
            best = greedy_policy(model, weights, discount=discount)
            best_value = policy_values(
                successor_features(model, best, discount=discount), best, weights
            )
            single = [
                max(
                    policy_values(psi, policy, weights)[place]
                    for psi, policy in zip(psis, subset)
                )
                for place in range(places)
            ]
            for place in range(places):
                if best_value[place] > 1e-9:
                    ratios.append(
                        {
                            "gpi": stitched_value[place] / best_value[place],
                            "single": single[place] / best_value[place],
                        }
                    )
        rows.append(
            {
                "library_size": size,
                "gpi_over_optimal": (
                    sum(row["gpi"] for row in ratios) / len(ratios) if ratios else 0.0
                ),
                "single_over_optimal": (
                    sum(row["single"] for row in ratios) / len(ratios)
                    if ratios
                    else 0.0
                ),
            }
        )
    return rows


def run_successor_transfer(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    world_seed: int = DEVELOPMENT_WORLD_SEED,
    tasks: int = 8,
    steps: int = EPISODE_STEPS,
    explore_episodes: int = EXPLORE_EPISODES,
    discount: float = DEFAULT_DISCOUNT,
    library_directory: Path | None = None,
) -> dict[str, Any]:
    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    reader = build_slot_reader(encoders)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    accumulation: list[list[dict[str, Any]]] = []
    for index in range(tasks):
        task = sample_navigation_task(seed=world_seed + WORLD_SEED_STRIDE * index)
        if task is None:
            continue

        generator = torch.Generator().manual_seed(seed + index)
        trained = list(range(TRAINING_GOALS))
        held_out = list(range(TRAINING_GOALS, PLACE_COUNT))
        pairs = [
            (
                int(torch.randint(0, PLACE_COUNT, (1,), generator=generator).item()),
                trained[episode % len(trained)],
            )
            for episode in range(explore_episodes)
        ]
        model, identified = explore(
            reader, task, pairs, steps=steps, seed=seed + 100 * index
        )

        # Scoring-side only: which cluster each place renders to, so that a
        # hazard the scene cannot show can be handed over in the agent's own
        # alphabet, and so the ceiling can be computed in place space.
        cluster_of_place = tuple(
            reader.read(render_markers((place, place), size=FRAME_SIZE))[0][1]
            for place in range(PLACE_COUNT)
        )

        library = SuccessorFeatureLibrary(
            place_count=reader.alphabet,
            action_count=task.action_count,
            cumulant_dimension=reader.alphabet,
        )
        policies = []
        for goal in trained:
            policy = greedy_policy(
                model,
                reach(reader.alphabet, cluster_of_place[goal]),
                discount=discount,
            )
            if library.duplicate_of(policy) is not None:
                continue
            policies.append(policy)
            library.append(
                SuccessorFeatureRecord(
                    policy=policy,
                    psi=successor_features(model, policy, discount=discount),
                    discount=discount,
                    provenance={"trained_goal_cluster": int(cluster_of_place[goal])},
                )
            )
        if library.record_count == 0:
            continue
        if library_directory is not None:
            library_directory.mkdir(parents=True, exist_ok=True)
            library.save(library_directory / f"task{index}.successors")

        accumulation.append(
            in_model_accumulation(
                model,
                policies,
                [cluster_of_place[place] for place in held_out],
                discount=discount,
            )
        )

        arms = {
            "gpi": (gpi_chooser(library), False),
            "best_single": (single_chooser(library, discount=discount), False),
            "replan": (replan_chooser(model, discount=discount), False),
            "random": (random_chooser(seed + 991 * index, task.action_count), False),
            # Declared oracles, to split the shortfall into its causes.
            "gpi_told": (gpi_chooser(library), True),
            "replan_told": (replan_chooser(model, discount=discount), True),
        }
        scored: dict[str, list[float]] = {name: [] for name in arms}
        scored["optimal"] = []
        scored["floor"] = []
        probe_steps: list[int] = []
        family_of: list[str] = []

        for specification in task_families(PLACE_COUNT, held_out, generator):
            shown = specification["shown"]
            hazard = specification["hazard"]
            weights_by_place = weights_for(PLACE_COUNT, shown, hazard)
            given = None
            if hazard is not None:
                given = torch.zeros(reader.alphabet, dtype=torch.float64)
                given[cluster_of_place[hazard]] -= HAZARD_PENALTY
            for start in range(PLACE_COUNT):
                family_of.append(specification["family"])
                scored["optimal"].append(
                    best_weighted_return(task, start, weights_by_place, steps)
                )
                scored["floor"].append(
                    worst_weighted_return(task, start, weights_by_place, steps)
                )
                for name, (chooser, oracle) in arms.items():
                    outcome = run_weighted_episode(
                        reader,
                        task,
                        start=start,
                        shown=shown,
                        weights_by_place=weights_by_place,
                        given=given,
                        steps=steps,
                        chooser=chooser,
                        told=cluster_of_place if oracle else None,
                    )
                    scored[name].append(outcome.score)
                    if name == "gpi":
                        probe_steps.append(outcome.probes)

        summary: dict[str, Any] = {
            "task": index,
            "own_identified": identified,
            "explore_episodes": len(pairs),
            "alphabet": reader.alphabet,
            "model_coverage": model.coverage,
            "library_size": library.record_count,
            "library_digest": library.digest(),
            "mean_probe_steps": (
                sum(probe_steps) / len(probe_steps) if probe_steps else 0.0
            ),
        }
        for family in ("single", "disjunction", "avoid"):
            keep = [i for i, name in enumerate(family_of) if name == family]
            if not keep:
                continue
            span = [
                scored["optimal"][i] - scored["floor"][i]
                for i in keep
                if scored["optimal"][i] - scored["floor"][i] > 1e-9
            ]
            block: dict[str, Any] = {"episodes": len(keep)}
            for name in (*arms, "optimal", "floor"):
                block[name] = sum(scored[name][i] for i in keep) / len(keep)
            # Normalised so that the floor is zero and the ceiling is one,
            # which is the only comparison that survives negative weights.
            for name in arms:
                fractions = [
                    (scored[name][i] - scored["floor"][i])
                    / (scored["optimal"][i] - scored["floor"][i])
                    for i in keep
                    if scored["optimal"][i] - scored["floor"][i] > 1e-9
                ]
                block[f"{name}_fraction"] = (
                    sum(fractions) / len(fractions) if fractions else 0.0
                )
            block["scored_spans"] = len(span)
            summary[family] = block
        rows.append(summary)

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the successor transfer run mutated AgentBrain.bank")

    def mean(family: str, key: str) -> float:
        values = [float(row[family][key]) for row in rows if family in row]
        return sum(values) / len(values) if values else 0.0

    curve = []
    if accumulation:
        width = min(len(row) for row in accumulation)
        for size in range(width):
            curve.append(
                {
                    "library_size": accumulation[0][size]["library_size"],
                    "gpi_over_optimal": sum(
                        row[size]["gpi_over_optimal"] for row in accumulation
                    )
                    / len(accumulation),
                    "single_over_optimal": sum(
                        row[size]["single_over_optimal"] for row in accumulation
                    )
                    / len(accumulation),
                }
            )

    report: dict[str, Any] = {
        "schema": SUCCESSOR_TRANSFER_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "world_seed": world_seed,
        "tasks": len(rows),
        "episode_steps": steps,
        "explore_episodes": explore_episodes,
        "discount": discount,
        "training_goals": TRAINING_GOALS,
        "hazard_penalty": HAZARD_PENALTY,
        "mean_alphabet": (
            sum(int(row["alphabet"]) for row in rows) / len(rows) if rows else 0
        ),
        "mean_own_identified": (
            sum(int(row["own_identified"]) for row in rows) / len(rows) if rows else 0
        ),
        "mean_library_size": (
            sum(int(row["library_size"]) for row in rows) / len(rows) if rows else 0
        ),
        "mean_probe_steps": (
            sum(float(row["mean_probe_steps"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "accumulation": curve,
        "rows": rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    for family in ("single", "disjunction", "avoid"):
        for key in (
            "gpi",
            "best_single",
            "replan",
            "random",
            "gpi_told",
            "replan_told",
            "optimal",
            "floor",
            "gpi_fraction",
            "best_single_fraction",
            "replan_fraction",
            "random_fraction",
            "gpi_told_fraction",
            "replan_told_fraction",
        ):
            report[f"{family}_{key}"] = mean(family, key)

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "successor_transfer.json").write_text(
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
            repository / "session_records" / "brainworkshop_successor_transfer_2026-08-16"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument(
        "--world-seed", type=int, default=DEVELOPMENT_WORLD_SEED
    )
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--explore-episodes", type=int, default=EXPLORE_EPISODES)
    arguments = parser.parse_args()
    report = run_successor_transfer(
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
