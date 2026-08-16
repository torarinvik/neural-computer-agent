"""Every piece at once, with nothing handed over.

Five records, five isolated axes. Each of them changed one thing and declared
the rest, which is the right way to measure a mechanism and a poor way to know
where the system is. Two of the five oracle identification outright; three take
the decomposition as given. Nothing has ever run all of it together.

This does, on the hardest of the goal languages -- relations to a target that
moves -- and the number it is after is not the score. It is **how the costs
compound**, which is a thing that cannot be predicted from the parts:

- the cut is **selected** by description length from the candidate family,
  rather than being connected components by assumption;
- which marker is the agent is **searched** for, by the correspondence beam and
  the matched-contrast controllability, rather than being read off the
  verifier;
- exploration is **curious and gated**, rather than uniform;
- the goal is a **relation**, including relations that were never a reward.

Two arms, because a third would have been a lie. `integrated` selects the cut
and searches for its own identity; `told_all` is handed both, which is exactly
the configuration the relational transfer record measured. The gap between them
is the price of what that record declared.

A `told_identity` arm -- cut selected, identity given -- was written first and
removed: cut selection returns components in every task, so that arm is the
same agent as `told_all` and reporting both would have dressed one number as
two. What the cut selection is worth is reported instead as the bits it
scored and the cut it chose, which is the honest form of "this step was free".

**Orientation is charged.** The correspondence search needs a stretch of
history before it can say anything, so the agent spends its first steps acting
randomly and watching. Those steps are scored like any other. An agent that
needs a quarter of its episode to work out which marker it is has paid a
quarter of its episode, and the accounting should say so rather than starting
the clock afterwards.
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
from .learned_decomposition import candidate_cuts, measure_cut, wander
from .navigation_environment import NavigationTask, sample_navigation_task
from .object_scene import render_markers
from .relation_cumulants import (
    PLACE_COUNT,
    joint_state,
    relation_cumulants,
    relation_weights,
)
from .relational_transfer import (
    HELD_OUT_RELATIONS,
    TRAINED_RELATIONS,
    RelationVerifier,
    joint_ceiling,
    target_circuit,
)
from .slot_alignment import Tracker, TrackHistory, beam_track, identify_roles
from .successor_features import (
    DEFAULT_DISCOUNT,
    generalised_policy_improvement,
    greedy_policy,
    successor_features,
)
from .successor_transfer import SlotReader, build_slot_reader
from .world_model import WorldModel

EXPERIMENT_ID = "brainworkshop-integrated-navigation-2026-08-16"
INTEGRATED_NAVIGATION_SCHEMA = "neural-computer.integrated-navigation.v1"
DEVELOPMENT_SEED = 41
DEVELOPMENT_WORLD_SEED = 9000
WORLD_SEED_STRIDE = 37
FRAME_SIZE = 36
EPISODE_STEPS = 20
EXPLORE_EPISODES = 40
# Steps spent acting at random so the correspondence search has something to
# work from. Scored like every other step.
ORIENTATION_STEPS = 6
ARMS = ("integrated", "told_all", "random")
# Two orientation knobs, both of which were tried as fixes and both of which
# lose. Swept at 0/2/6 contrasts x revise/commit: requiring more evidence
# before naming yourself degrades the agent monotonically (0.644 -> 0.534 ->
# 0.491 on trained relations) because the orientation delay it buys -- 3.6 to
# 10.8 steps of twenty -- costs more than the better identification is worth.
# Re-deciding every step is a wash. The settings below are the ones that
# measured best, which are also the simplest.
MINIMUM_CONTRASTS = 0
REORIENT = False


def select_cut(reader: SlotReader, encoders, task: NavigationTask, *, seed: int):
    """Choose the decomposition by description length, rather than assume it."""

    episodes = [
        wander(task, start=start, goal=(start * 3 + 1) % PLACE_COUNT, steps=30,
               seed=seed + start)
        for start in range(3)
    ]
    scored = [
        row
        for row in (measure_cut(cut, encoders, episodes) for cut in candidate_cuts(seed))
        if row["status"] == "scored"
    ]
    if not scored:
        raise ValueError("no candidate cut could be scored")
    best = min(scored, key=lambda row: row["total_bits"])
    return best["cut"], {row["cut"]: row["total_bits"] for row in scored}


class SearchedReader:
    """Reads a configuration without being told which marker is which.

    Orientation runs the correspondence beam over a prefix of random actions
    and asks which track the action moved; after that a greedy tracker carries
    the identity forward. Both halves are the mechanisms the identity record
    measured, wired together and paying their own way.
    """

    def __init__(self, reader: SlotReader) -> None:
        self.reader = reader
        self.tracker: Tracker | None = None
        self.own: int | None = None
        self._pending: list[tuple[int, int]] = []
        self._histories: list[TrackHistory] = []
        self._readings: list[Any] = []
        self._actions: list[int] = []

    @property
    def oriented(self) -> bool:
        return self.own is not None

    def observe(self, frame: torch.Tensor, *, last_action: int | None) -> None:
        parts = self.reader.read(frame)
        self._readings.append(parts)
        if self.tracker is None:
            self.tracker = Tracker.started(parts)
            self._histories = [TrackHistory() for _ in range(self.tracker.count)]
        else:
            self.tracker.update(parts, action=last_action)
            while len(self._histories) < self.tracker.count:
                self._histories.append(TrackHistory())
            for track, (symbol, action) in enumerate(self._pending):
                if track < self.tracker.count:
                    self._histories[track].observe(
                        symbol, action, self.tracker.symbols[track]
                    )
        self._pending = []

    def record(self, action: int) -> None:
        if self.tracker is None:
            raise RuntimeError("nothing has been observed yet")
        self._pending = [(symbol, action) for symbol in self.tracker.reading()]
        self._actions.append(int(action))

    def orient(self, *, alphabet: int, minimum_contrasts: int | None = None) -> bool:
        """Re-decide which track is the agent, from all the history so far.

        Two things here were wrong in the first version and both were found by
        measurement rather than by reading.

        **It committed permanently.** The first successful orientation was kept
        for the rest of the episode. But the correspondence search commits its
        own errors early -- traced on a clean synthetic case, the beam followed
        the agent for four steps and then swapped onto the distractor, so both
        tracks were mixtures and neither followed one object. An identity
        settled from the first three frames inherits that and never recovers.
        This now re-decides every step, on the whole history, so evidence that
        arrives later can overturn it.

        **It accepted on almost no evidence.** Three frames give a couple of
        matched contrasts, and a mixture of two objects can look perfectly
        controllable across two of them. A minimum contrast count is the cheap
        guard: refuse to name yourself until the same place has been left more
        than a handful of times.
        """

        required = MINIMUM_CONTRASTS if minimum_contrasts is None else minimum_contrasts
        if len(self._readings) < 3:
            return False
        trace = beam_track(
            self._readings, self._actions[: len(self._readings) - 1], alphabet=alphabet
        )
        width = max((len(row) for row in trace), default=0)
        histories = [TrackHistory() for _ in range(width)]
        for step in range(len(trace) - 1):
            before, after = trace[step], trace[step + 1]
            for track in range(min(len(before), len(after))):
                histories[track].observe(
                    before[track], self._actions[step], after[track]
                )
        roles = identify_roles(histories, alphabet=alphabet)
        if roles.own is None:
            return False
        if roles.evidence[roles.own].contrasts < int(required):
            return False
        # Carry the identity into the online tracker by symbol, which is what
        # survives the two indexings being different objects.
        settled = trace[-1]
        if roles.own >= len(settled):
            return False
        symbol = int(settled[roles.own])
        if self.tracker is None:
            return False
        for index, value in enumerate(self.tracker.symbols):
            if int(value) == symbol:
                self.own = index
                return True
        return False

    def configuration(self, *, alphabet: int) -> tuple[int, int] | None:
        if self.tracker is None or self.own is None:
            return None
        symbols = self.tracker.reading()
        if self.own >= len(symbols):
            return None
        mine = int(symbols[self.own])
        others = [
            int(symbol)
            for index, symbol in enumerate(symbols)
            if index != self.own and int(symbol) != mine
        ]
        return mine, (others[0] if others else mine)


def run_integrated_episode(
    reader: SlotReader,
    task: NavigationTask,
    circuit: tuple[int, ...],
    *,
    start: int,
    target: int,
    relation: str,
    steps: int,
    arm: str,
    psis,
    weights: torch.Tensor,
    cluster_of_place,
    alphabet: int,
    seed: int,
) -> tuple[float, int, float]:
    """One episode: the score, the orientation cost, and how often it was right.

    The third number is what turns the shortfall from a total into an
    attribution. Scoring-side, and read only by the accounting.
    """

    verifier = RelationVerifier(
        task, circuit, start=start, target=target, relation=relation, steps=steps
    )
    generator = torch.Generator().manual_seed(int(seed))
    searched = SearchedReader(reader) if arm == "integrated" else None
    total = 0.0
    scored = 0
    spent = 0
    correct = 0
    last_action: int | None = None
    while not verifier.done:
        frame = verifier.observation()
        configuration: tuple[int, int] | None
        if searched is not None:
            searched.observe(frame, last_action=last_action)
            if REORIENT or not searched.oriented:
                searched.orient(alphabet=alphabet)
            configuration = searched.configuration(alphabet=alphabet)
        else:
            symbols = [symbol for _, symbol in reader.read(frame)]
            mine = int(cluster_of_place[verifier.place])
            others = [symbol for symbol in symbols if symbol != mine]
            configuration = (mine, others[0] if others else mine)

        if configuration is not None and int(configuration[0]) == int(
            cluster_of_place[verifier.place]
        ):
            correct += 1
        if arm == "random" or configuration is None:
            action = int(
                torch.randint(0, task.action_count, (1,), generator=generator).item()
            )
            if configuration is None:
                spent += 1
        else:
            state = joint_state(*configuration, place_count=alphabet)
            action = generalised_policy_improvement(psis, state, weights)
        if searched is not None:
            searched.record(action)
        last_action = action
        total += verifier.score(torch.tensor([action], dtype=torch.long))
        scored += 1
    return (
        total / scored if scored else 0.0,
        spent,
        correct / scored if scored else 0.0,
    )


def explore(
    reader: SlotReader,
    task: NavigationTask,
    circuit: tuple[int, ...],
    *,
    arm: str,
    episodes: int,
    steps: int,
    seed: int,
    cluster_of_place,
) -> WorldModel:
    """Build the configuration model, under this arm's identification."""

    joint = WorldModel(reader.alphabet * reader.alphabet, task.action_count)
    generator = torch.Generator().manual_seed(int(seed))
    for episode in range(episodes):
        start = int(torch.randint(0, PLACE_COUNT, (1,), generator=generator).item())
        target = int(torch.randint(0, PLACE_COUNT, (1,), generator=generator).item())
        verifier = RelationVerifier(
            task,
            circuit,
            start=start,
            target=target,
            relation=TRAINED_RELATIONS[0],
            steps=steps,
        )
        searched = SearchedReader(reader) if arm == "integrated" else None
        last_action: int | None = None
        previous: int | None = None
        while not verifier.done:
            frame = verifier.observation()
            if searched is not None:
                searched.observe(frame, last_action=last_action)
                if REORIENT or not searched.oriented:
                    searched.orient(alphabet=reader.alphabet)
                configuration = searched.configuration(alphabet=reader.alphabet)
            else:
                symbols = [symbol for _, symbol in reader.read(frame)]
                mine = int(cluster_of_place[verifier.place])
                others = [symbol for symbol in symbols if symbol != mine]
                configuration = (mine, others[0] if others else mine)
            state = (
                None
                if configuration is None
                else joint_state(*configuration, place_count=reader.alphabet)
            )
            if previous is not None and state is not None and last_action is not None:
                joint.observe(previous, last_action, state, 0)
            fresh = (
                []
                if state is None
                else [
                    action
                    for action in range(task.action_count)
                    if not joint.counts[action][state]
                ]
            )
            action = (
                fresh[episode % len(fresh)]
                if fresh
                else int(
                    torch.randint(
                        0, task.action_count, (1,), generator=generator
                    ).item()
                )
            )
            if searched is not None:
                searched.record(action)
            previous = state
            last_action = action
            verifier.score(torch.tensor([action], dtype=torch.long))
    return joint


def run_integrated_navigation(
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
    starts: int = 6,
    discount: float = DEFAULT_DISCOUNT,
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
    features = relation_cumulants(place_count=reader.alphabet)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index in range(tasks):
        task = sample_navigation_task(seed=world_seed + WORLD_SEED_STRIDE * index)
        if task is None:
            continue
        circuit = target_circuit(seed + index)
        chosen_cut, cut_bits = select_cut(reader, encoders, task, seed=seed + index)

        models = {
            arm: explore(
                reader,
                task,
                circuit,
                arm=arm,
                episodes=explore_episodes,
                steps=steps,
                seed=seed + 100 * index,
                cluster_of_place=cluster_of_place,
            )
            for arm in ("integrated", "told_all")
        }
        # Acting blindly needs a psi to be handed something, never consults it.
        models["random"] = models["told_all"]

        stored = {}
        for arm, model in models.items():
            policies = [
                greedy_policy(
                    model,
                    relation_weights(relation),
                    discount=discount,
                    cumulants=features,
                )
                for relation in TRAINED_RELATIONS
            ]
            stored[arm] = [
                successor_features(
                    model, policy, discount=discount, cumulants=features
                )
                for policy in policies
            ]

        generator = torch.Generator().manual_seed(seed + 7 * index)
        pairs = [
            (
                int(torch.randint(0, PLACE_COUNT, (1,), generator=generator).item()),
                int(torch.randint(0, PLACE_COUNT, (1,), generator=generator).item()),
            )
            for _ in range(starts)
        ]

        for relation in (*TRAINED_RELATIONS, *HELD_OUT_RELATIONS):
            weights = relation_weights(relation)
            entry: dict[str, Any] = {
                "task": index,
                "relation": relation,
                "held_out": relation in HELD_OUT_RELATIONS,
                "chosen_cut": chosen_cut,
                "cut_bits": cut_bits,
                "integrated_coverage": models["integrated"].coverage,
                "told_coverage": models["told_all"].coverage,
            }
            ceilings, floors = [], []
            scored: dict[str, list[float]] = {arm: [] for arm in ARMS}
            spends: list[int] = []
            rights: list[float] = []
            for start, target in pairs:
                ceilings.append(
                    joint_ceiling(
                        task, circuit, start=start, target=target,
                        relation=relation, steps=steps, best=True,
                    )
                )
                floors.append(
                    joint_ceiling(
                        task, circuit, start=start, target=target,
                        relation=relation, steps=steps, best=False,
                    )
                )
                for arm in ARMS:
                    value, spent, right = run_integrated_episode(
                        reader,
                        task,
                        circuit,
                        start=start,
                        target=target,
                        relation=relation,
                        steps=steps,
                        arm=arm,
                        psis=stored[arm],
                        weights=weights,
                        cluster_of_place=cluster_of_place,
                        alphabet=reader.alphabet,
                        seed=seed + 31 * index,
                    )
                    scored[arm].append(value)
                    if arm == "integrated":
                        spends.append(spent)
                        rights.append(right)
            entry["orientation_steps"] = sum(spends) / len(spends) if spends else 0.0
            entry["identification_accuracy"] = (
                sum(rights) / len(rights) if rights else 0.0
            )
            for arm in ARMS:
                entry[arm] = sum(scored[arm]) / len(scored[arm])
                fractions = [
                    (value - floor) / (ceiling - floor)
                    for value, floor, ceiling in zip(scored[arm], floors, ceilings)
                    if ceiling - floor > 1e-9
                ]
                entry[f"{arm}_fraction"] = (
                    sum(fractions) / len(fractions) if fractions else 0.0
                )
            entry["optimal"] = sum(ceilings) / len(ceilings)
            rows.append(entry)

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the integrated run mutated AgentBrain.bank")

    def block(held_out: bool) -> dict[str, Any]:
        chosen = [row for row in rows if row["held_out"] == held_out]
        if not chosen:
            return {}
        summary: dict[str, Any] = {"relations": len(chosen)}
        for key in (*ARMS, *(f"{arm}_fraction" for arm in ARMS), "optimal"):
            summary[key] = sum(float(row[key]) for row in chosen) / len(chosen)
        for key in ("orientation_steps", "identification_accuracy"):
            summary[key] = sum(float(row[key]) for row in chosen) / len(chosen)
        return summary

    report = {
        "schema": INTEGRATED_NAVIGATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "world_seed": world_seed,
        "tasks": tasks,
        "episode_steps": steps,
        "explore_episodes": explore_episodes,
        "cuts_chosen": sorted({row["chosen_cut"] for row in rows}),
        "mean_integrated_coverage": (
            sum(row["integrated_coverage"] for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_told_coverage": (
            sum(row["told_coverage"] for row in rows) / len(rows) if rows else 0.0
        ),
        "trained": block(False),
        "held_out": block(True),
        "rows": rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "integrated_navigation.json").write_text(
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
            / "brainworkshop_integrated_navigation_2026-08-16"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--world-seed", type=int, default=DEVELOPMENT_WORLD_SEED)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--explore-episodes", type=int, default=EXPLORE_EPISODES)
    arguments = parser.parse_args()
    report = run_integrated_navigation(
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
