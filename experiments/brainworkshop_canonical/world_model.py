"""Learn what the actions do, then search for what to do.

Delayed reward is usually attacked head-on: try policies, see what pays,
propagate credit backwards. That is expensive, noisy, and it throws away the
thing this world hands over for free.

Because the agent sees where it *ended up*, the dynamics are fully observed.
`(place, action) -> next place` is supervision, not a bandit signal -- so the
hard part of the earlier k-action work does not recur here, and the delayed
reward stops being a credit-assignment problem and becomes a **search** problem
over a model the agent can build directly.

That is the claim, and it is the one the experiment has to earn against a
model-free control rather than by assertion. Three things follow from it.

**The model is a table, and its gaps are visible.** A cell is known once the
agent has tried that action in that place. What it has not tried, it knows it
has not tried -- so coverage is a measurable quantity rather than a hope, and a
plan that would route through an unknown cell can be refused instead of
guessed.

**The goal is discovered, not given.** The verifier says only whether a step
paid. A place is a goal because standing on it paid, which means the agent must
stumble on it during exploration before any planning is possible at all.

**Planning is breadth-first and exact.** Over a known deterministic model the
shortest route is not an approximation, so any shortfall against the optimal
path is a shortfall in the *model* -- which is what makes plan length a clean
readout of what was learned rather than of how well the planner searched.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

WORLD_MODEL_SCHEMA = "neural-computer.world-model.v1"


@dataclass
class WorldModel:
    """What the agent believes the actions do, and where reward was found."""

    place_count: int
    action_count: int
    # counts[action][place][next place]: how often that was seen. Counts rather
    # than a single value because a noisy frontend can mis-cluster a place, and
    # a model that overwrote itself on one bad reading would be worse than one
    # that outvoted it.
    counts: list[list[dict[int, int]]] = field(default_factory=list)
    rewarded: dict[int, int] = field(default_factory=dict)
    visited: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.place_count < 2 or self.action_count < 1:
            raise ValueError("a world model needs places and actions")
        if not self.counts:
            self.counts = [
                [{} for _ in range(self.place_count)]
                for _ in range(self.action_count)
            ]

    def observe(self, place: int, action: int, following: int, reward: int) -> None:
        """One transition, and whether arriving there paid."""

        if not 0 <= place < self.place_count or not 0 <= following < self.place_count:
            raise ValueError("a transition leaves the place set")
        if not 0 <= action < self.action_count:
            raise ValueError("a transition uses an action outside the protocol")
        cell = self.counts[action][place]
        cell[following] = cell.get(following, 0) + 1
        self.visited[place] = self.visited.get(place, 0) + 1
        if reward:
            self.rewarded[following] = self.rewarded.get(following, 0) + 1

    def successor(self, place: int, action: int) -> int | None:
        """Where this action leads, or nothing if it was never tried here."""

        cell = self.counts[action][place]
        if not cell:
            return None
        return max(cell.items(), key=lambda item: (item[1], -item[0]))[0]

    @property
    def known_cells(self) -> int:
        return sum(
            1
            for action in range(self.action_count)
            for place in range(self.place_count)
            if self.counts[action][place]
        )

    @property
    def coverage(self) -> float:
        """The share of the world the agent has actually tried."""

        total = self.action_count * self.place_count
        return self.known_cells / total if total else 0.0

    def goals(self) -> tuple[int, ...]:
        """Places that paid when arrived at, most often first."""

        return tuple(
            place
            for place, _ in sorted(
                self.rewarded.items(), key=lambda item: (-item[1], item[0])
            )
        )

    def holding_action(self, place: int) -> int | None:
        """An action believed to leave the agent where it is."""

        for action in range(self.action_count):
            if self.successor(place, action) == place:
                return action
        return None

    def payload(self) -> dict[str, Any]:
        return {
            "schema": WORLD_MODEL_SCHEMA,
            "place_count": self.place_count,
            "action_count": self.action_count,
            "known_cells": self.known_cells,
            "coverage": self.coverage,
            "goals": list(self.goals()),
            "visited_places": len(self.visited),
        }


@dataclass(frozen=True)
class Plan:
    """A route the model believes reaches a goal, and what it is made of."""

    actions: tuple[int, ...]
    places: tuple[int, ...]
    goal: int

    @property
    def length(self) -> int:
        return len(self.actions)


def plan_to(model: WorldModel, start: int, goals) -> Plan | None:
    """Shortest believed route from `start` to any of `goals`.

    Breadth-first over known cells only. An action never tried in a place is
    not treated as a self-loop or as anything else -- it simply is not an edge,
    so the planner cannot route through a part of the world the agent has no
    evidence about.
    """

    targets = {int(goal) for goal in goals}
    if not targets:
        return None
    if start in targets:
        return Plan(actions=(), places=(start,), goal=start)
    previous: dict[int, tuple[int, int]] = {}
    seen = {start}
    frontier = deque([start])
    while frontier:
        place = frontier.popleft()
        for action in range(model.action_count):
            following = model.successor(place, action)
            if following is None or following in seen:
                continue
            seen.add(following)
            previous[following] = (place, action)
            if following in targets:
                actions: list[int] = []
                places: list[int] = [following]
                cursor = following
                while cursor != start:
                    source, taken = previous[cursor]
                    actions.append(taken)
                    places.append(source)
                    cursor = source
                return Plan(
                    actions=tuple(reversed(actions)),
                    places=tuple(reversed(places)),
                    goal=following,
                )
            frontier.append(following)
    return None


def policy_from_model(model: WorldModel, *, fallback: int = 0):
    """A closed-loop policy: replan from wherever the agent actually is.

    Following a plan open-loop would be enough in a deterministic world that
    the model has right. Replanning each tick is what makes a *wrong* model
    recoverable -- the agent notices it is somewhere the plan did not expect
    and asks for a route from there instead of walking the rest of a route
    that no longer applies.
    """

    goals = model.goals()

    def act(place: int) -> int:
        if not goals:
            return fallback
        if place in set(goals):
            holding = model.holding_action(place)
            if holding is not None:
                return holding
        route = plan_to(model, place, goals)
        if route is None or not route.actions:
            return fallback
        return int(route.actions[0])

    return act
