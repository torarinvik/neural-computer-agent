"""Being asked for a relation nobody ever paid you for.

The successor transfer record showed that a task can be a weight vector, and
left the goal language as the open item. This is the escalation, and it turns
on one correction to how that caveat was written.

**A vector over places was never only "go to place p".** Any subset of places
is a vector, so disjunction and avoidance were already expressible -- both were
measured. Even "be in the same row as place four" is a place-vector, because
the satisfying set is fixed once the other place is.

**What a place-vector cannot express is a goal whose satisfying set moves.**
"Stay next to that marker" is not a set of places. It is a set of
*configurations*, and it changes every time the marker does. So the target here
moves on its own -- a fixed circuit, deterministic, uncontrollable -- and the
state becomes the pair.

The escalation costs a cumulant matrix and nothing else. `phi(mine, theirs)`
over six relations, a world model over configurations rather than places, and
every existing piece -- occupancies, generalised policy improvement, the
library -- runs unmodified, because all of it was written against a cumulant
matrix from the start.

## The regime, which is the point

PGM (Barrett et al., 2018) is the warning: networks that interpolate happily
collapse when a *held-out attribute* appears at test, and our previous splits
were all interpolation -- new goals drawn from the same family as the trained
ones. So the split here is by **relation**. Policies are induced for three
relations; the agent is then asked for three it was never rewarded for, with no
further experience. `same_column`, `diagonal` and `opposite` are never a reward
signal during learning; they are only ever *observed* as occupancies.

The relation set is filtered before any of this: three earlier candidates were
dropped because one fixed place satisfies them 0.625 of the time, so an agent
ignoring the marker scored 1.000 and the comparison measured nothing.

## The control that decides whether any of this was necessary

`place_gpi` is the previous system: occupancies over place cumulants, and the
best weight vector over places that the goal admits -- for each place, how
often the relation would hold if the marker were anywhere. That is not a
strawman but the fairest reading of the old representation, and if it keeps up
then pairs bought nothing.

Identification is handed over by the declared oracle, as in the exploration
record. Two markers that both move is the hard case for correspondence, the
identity record measures exactly what it costs, and letting it vary here would
move two axes at once. The target's *place* still comes from pixels; only which
slot is which is given.
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
from .object_scene import render_markers
from .relation_cumulants import (
    CONSTANT_PLACE_LIMIT,
    PLACE_COUNT,
    RELATIONS,
    _holds,
    constant_place_rate,
    joint_state,
    marginal_place_weights,
    relation_cumulants,
    relation_weights,
)
from .successor_features import (
    DEFAULT_DISCOUNT,
    generalised_policy_improvement,
    greedy_policy,
    policy_values,
    successor_features,
)
from .successor_transfer import SlotReader, build_slot_reader
from .world_model import WorldModel

EXPERIMENT_ID = "brainworkshop-relational-transfer-2026-08-16"
RELATIONAL_TRANSFER_SCHEMA = "neural-computer.relational-transfer.v1"
DEVELOPMENT_SEED = 41
# Worlds are drawn from here. Defaulting to the development value keeps
# every recorded diagnostic reproducing exactly; a holdout run passes a seed
# from an unused block so the *worlds* are unseen and not merely the
# exploration randomness.
DEVELOPMENT_WORLD_SEED = 9000
WORLD_SEED_STRIDE = 37
FRAME_SIZE = 36
EPISODE_STEPS = 20
EXPLORE_EPISODES = 40
TRAINED_RELATIONS = ("same", "same_row", "adjacent")
HELD_OUT_RELATIONS = ("same_column", "diagonal", "opposite")
ARMS = ("gpi", "best_single", "replan", "place_gpi", "random")


def target_circuit(seed: int, *, place_count: int = PLACE_COUNT) -> tuple[int, ...]:
    """Where the other marker goes next, whatever the agent does.

    A permutation cycle rather than a random walk, so the joint dynamics stay
    deterministic and the ceiling can be exact backward induction rather than
    an expectation. It is also the `cycling` distractor from the identity
    record, which is the case that is predictable and still uncontrollable.
    """

    generator = torch.Generator().manual_seed(int(seed))
    order = torch.randperm(int(place_count), generator=generator).tolist()
    circuit = [0] * int(place_count)
    for index, place in enumerate(order):
        circuit[int(place)] = int(order[(index + 1) % int(place_count)])
    return tuple(circuit)


class RelationVerifier:
    """Two markers, one of which the agent does not drive; pays a relation."""

    def __init__(
        self,
        task: NavigationTask,
        circuit: tuple[int, ...],
        *,
        start: int,
        target: int,
        relation: str,
        steps: int,
        frame_size: int = FRAME_SIZE,
    ) -> None:
        if relation not in RELATIONS:
            raise ValueError(f"unknown relation: {relation}")
        self.task = task.validate()
        self.circuit = circuit
        self.relation = relation
        self.steps = int(steps)
        self.frame_size = int(frame_size)
        self._place = int(start)
        self._target = int(target)
        self._position = 0

    @property
    def done(self) -> bool:
        return self._position >= self.steps

    @property
    def place(self) -> int:
        """Scoring-side truth, read only by the declared identification oracle."""

        return self._place

    @property
    def target(self) -> int:
        return self._target

    def observation(self) -> torch.Tensor:
        if self.done:
            raise RuntimeError("relation episode is complete")
        return render_markers((self._target, self._place), size=self.frame_size)

    def score(self, action: torch.Tensor) -> float:
        if self.done:
            raise RuntimeError("relation episode is complete")
        chosen = int(action.item())
        if not 0 <= chosen < self.task.action_count:
            raise ValueError("relation action is outside the protocol")
        self._place = int(self.task.transitions[chosen][self._place])
        self._target = int(self.circuit[self._target])
        self._position += 1
        return float(_holds(self.relation, self._place, self._target))


def joint_ceiling(
    task: NavigationTask,
    circuit: tuple[int, ...],
    *,
    start: int,
    target: int,
    relation: str,
    steps: int,
    best: bool = True,
) -> float:
    """Exact backward induction over configurations. Scoring-side."""

    places = task.place_count
    value = [0.0] * (places * places)
    choose = max if best else min
    for _ in range(int(steps)):
        following = [0.0] * (places * places)
        for mine in range(places):
            for theirs in range(places):
                options = []
                for action in range(task.action_count):
                    landed = int(task.transitions[action][mine])
                    moved = int(circuit[theirs])
                    options.append(
                        float(_holds(relation, landed, moved))
                        + value[landed * places + moved]
                    )
                following[mine * places + theirs] = choose(options)
        value = following
    return value[int(start) * places + int(target)] / int(steps) if steps else 0.0


def explore(
    reader: SlotReader,
    task: NavigationTask,
    circuit: tuple[int, ...],
    *,
    episodes: int,
    steps: int,
    seed: int,
    cluster_of_place,
) -> tuple[WorldModel, WorldModel]:
    """Wander the configuration space; keep a joint model and a place model.

    Both are built from exactly the same experience, so the comparison later is
    about what each representation can *hold*, not about what either was shown.
    Untried-first optimism, which the exploration record measured as most of
    what directed exploration buys.
    """

    joint = WorldModel(reader.alphabet * reader.alphabet, task.action_count)
    marginal = WorldModel(reader.alphabet, task.action_count)
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
        while not verifier.done:
            mine, theirs = read_configuration(
                reader, verifier, cluster_of_place=cluster_of_place
            )
            state = joint_state(mine, theirs, place_count=reader.alphabet)
            fresh = [
                action
                for action in range(task.action_count)
                if not joint.counts[action][state]
            ]
            action = (
                fresh[episode % len(fresh)]
                if fresh
                else int(
                    torch.randint(
                        0, task.action_count, (1,), generator=generator
                    ).item()
                )
            )
            verifier.score(torch.tensor([action], dtype=torch.long))
            if verifier.done:
                # The step happened but there is no frame left to read it
                # from, so it is dropped rather than filled in from the
                # verifier's own state.
                break
            after_mine, after_theirs = read_configuration(
                reader, verifier, cluster_of_place=cluster_of_place
            )
            joint.observe(
                state,
                action,
                joint_state(after_mine, after_theirs, place_count=reader.alphabet),
                0,
            )
            marginal.observe(mine, action, after_mine, 0)
    return joint, marginal


def read_configuration(
    reader: SlotReader, verifier: RelationVerifier, *, cluster_of_place
) -> tuple[int, int]:
    """What the scene says, with only the assignment handed over.

    The symbols come from pixels through the frozen encoder. The oracle says
    which of them is the agent -- nothing more -- because two markers that both
    move is the correspondence problem the identity record measures separately.
    """

    symbols = [symbol for _, symbol in reader.read(verifier.observation())]
    mine = int(cluster_of_place[verifier.place])
    others = [symbol for symbol in symbols if symbol != mine]
    return mine, (others[0] if others else mine)


def run_relational_transfer(
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
    place_of_cluster = {
        int(cluster): place for place, cluster in enumerate(cluster_of_place)
    }
    features = relation_cumulants(place_count=reader.alphabet)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for index in range(tasks):
        task = sample_navigation_task(seed=world_seed + WORLD_SEED_STRIDE * index)
        if task is None:
            continue
        circuit = target_circuit(seed + index)
        joint, marginal = explore(
            reader,
            task,
            circuit,
            episodes=explore_episodes,
            steps=steps,
            seed=seed + 100 * index,
            cluster_of_place=cluster_of_place,
        )

        # Stored for the relations that were rewarded. The held-out relations
        # are never a reward here -- only ever an occupancy that nobody asked
        # about at the time.
        policies = [
            greedy_policy(
                joint,
                relation_weights(relation),
                discount=discount,
                cumulants=features,
            )
            for relation in TRAINED_RELATIONS
        ]
        psis = [
            successor_features(joint, policy, discount=discount, cumulants=features)
            for policy in policies
        ]
        place_policies = [
            greedy_policy(
                marginal,
                marginal_place_weights(relation, place_count=reader.alphabet),
                discount=discount,
            )
            for relation in TRAINED_RELATIONS
        ]
        place_psis = [
            successor_features(marginal, policy, discount=discount)
            for policy in place_policies
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
            place_weights = marginal_place_weights(
                relation, place_count=reader.alphabet
            )
            replan_policy = greedy_policy(
                joint, weights, discount=discount, cumulants=features
            )
            scored: dict[str, list[float]] = {arm: [] for arm in ARMS}
            ceilings: list[float] = []
            floors: list[float] = []
            for start, target in pairs:
                ceilings.append(
                    joint_ceiling(
                        task,
                        circuit,
                        start=start,
                        target=target,
                        relation=relation,
                        steps=steps,
                        best=True,
                    )
                )
                floors.append(
                    joint_ceiling(
                        task,
                        circuit,
                        start=start,
                        target=target,
                        relation=relation,
                        steps=steps,
                        best=False,
                    )
                )
                for arm in ARMS:
                    scored[arm].append(
                        run_relation_episode(
                            reader,
                            task,
                            circuit,
                            start=start,
                            target=target,
                            relation=relation,
                            steps=steps,
                            arm=arm,
                            psis=psis,
                            policies=policies,
                            place_psis=place_psis,
                            weights=weights,
                            place_weights=place_weights,
                            replan_policy=replan_policy,
                            cluster_of_place=cluster_of_place,
                            alphabet=reader.alphabet,
                            seed=seed + 31 * index,
                        )
                    )
            entry: dict[str, Any] = {
                "task": index,
                "relation": relation,
                "held_out": relation in HELD_OUT_RELATIONS,
                "optimal": sum(ceilings) / len(ceilings),
                "floor": sum(floors) / len(floors),
                "joint_coverage": joint.coverage,
                "place_coverage": marginal.coverage,
            }
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
            rows.append(entry)

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the relational transfer run mutated AgentBrain.bank")

    def block(held_out: bool) -> dict[str, Any]:
        chosen = [row for row in rows if row["held_out"] == held_out]
        if not chosen:
            return {}
        summary = {"relations": len(chosen)}
        for key in (
            *ARMS,
            *(f"{arm}_fraction" for arm in ARMS),
            "optimal",
            "floor",
        ):
            summary[key] = sum(float(row[key]) for row in chosen) / len(chosen)
        return summary

    report = {
        "schema": RELATIONAL_TRANSFER_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "world_seed": world_seed,
        "tasks": tasks,
        "episode_steps": steps,
        "explore_episodes": explore_episodes,
        "starts": starts,
        "trained_relations": list(TRAINED_RELATIONS),
        "held_out_relations": list(HELD_OUT_RELATIONS),
        "cumulant_rank": int(torch.linalg.matrix_rank(features)),
        "constant_place_rate": {
            relation: constant_place_rate(relation) for relation in RELATIONS
        },
        "constant_place_limit": CONSTANT_PLACE_LIMIT,
        "configurations": int(features.shape[0]),
        "mean_joint_coverage": (
            sum(row["joint_coverage"] for row in rows) / len(rows) if rows else 0.0
        ),
        "mean_place_coverage": (
            sum(row["place_coverage"] for row in rows) / len(rows) if rows else 0.0
        ),
        "trained": block(False),
        "held_out": block(True),
        "rows": rows,
        "place_of_cluster": {str(k): v for k, v in place_of_cluster.items()},
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "relational_transfer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def run_relation_episode(
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
    policies,
    place_psis,
    weights: torch.Tensor,
    place_weights: torch.Tensor,
    replan_policy,
    cluster_of_place,
    alphabet: int,
    seed: int,
) -> float:
    """One episode under one arm. Every arm reads the same scene."""

    verifier = RelationVerifier(
        task,
        circuit,
        start=start,
        target=target,
        relation=relation,
        steps=steps,
    )
    generator = torch.Generator().manual_seed(int(seed))
    total = 0.0
    scored = 0
    while not verifier.done:
        mine, theirs = read_configuration(
            reader, verifier, cluster_of_place=cluster_of_place
        )
        state = joint_state(mine, theirs, place_count=alphabet)
        if arm == "random":
            action = int(
                torch.randint(0, task.action_count, (1,), generator=generator).item()
            )
        elif arm == "gpi":
            action = generalised_policy_improvement(psis, state, weights)
        elif arm == "replan":
            action = int(replan_policy[state])
        elif arm == "best_single":
            best = max(
                range(len(psis)),
                key=lambda index: policy_values(psis[index], policies[index], weights)[
                    state
                ],
            )
            action = int(policies[best][state])
        elif arm == "place_gpi":
            # The old representation, at its best: it cannot see where the
            # marker is, so it plays the place vector that is right on average.
            action = generalised_policy_improvement(place_psis, mine, place_weights)
        else:  # pragma: no cover - guarded by the ARMS tuple
            raise ValueError(f"unknown arm: {arm}")
        total += verifier.score(torch.tensor([action], dtype=torch.long))
        scored += 1
    return total / scored if scored else 0.0


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
            / "brainworkshop_relational_transfer_2026-08-16"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument(
        "--world-seed", type=int, default=DEVELOPMENT_WORLD_SEED
    )
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--explore-episodes", type=int, default=EXPLORE_EPISODES)
    arguments = parser.parse_args()
    report = run_relational_transfer(
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
