"""Goals about how two things stand to each other, not about where one is.

The successor transfer record left this open:

> **Cumulants are hand-chosen.** `phi` is the place indicator [...] no
> property, no relation, no conjunction over anything but places.

That statement needs one correction before it can be acted on, and the
correction is the whole reason this module exists.

**A weight vector over places is already more than "go to place p".** Any
*subset* of places is a `w`, so "reach either of these two" and "reach this and
avoid that" were expressible all along -- the successor transfer record
measured both. Even "be in the same row as place four" is a place-vector,
because the satisfying set is fixed once the other place is fixed.

So the ceiling is not relations. The ceiling is that **the satisfying set has
to be constant**, and it stops being constant the moment the thing the goal
refers to moves. "Stay next to that marker" is not a set of places; it is a set
of *configurations*, and it changes under the agent's feet every time the
marker does.

That is what needs pair features, and it is where Relation Networks (Santoro et
al., 2017) point: relational generalisation comes from one shared function
applied to every pair of objects. Their function is a network. At this scale a
pair is an index and the shared function is a table, so the whole escalation is
a cumulant matrix -- `phi(mine, theirs)` over the six relations below -- and
not a line of new machinery. `successor_features` already accepts cumulants;
the pair state is just a bigger world model.

The relations are deliberately overlapping rather than a partition. Standing on
the marker is also standing in its row, so `same` implies `same_row`, and a
goal is satisfied by a set of configurations rather than by one. Nothing here
requires them to be independent, and their rank is measured rather than
assumed.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .rendered_environment import _GRID_POSITIONS

RELATION_SCHEMA = "neural-computer.relation-cumulants.v1"
PLACE_COUNT = len(_GRID_POSITIONS)

# Read as "mine RELATION theirs". Every one is a predicate on the pair, and
# none of them mentions an absolute place -- which is the point: a goal written
# in these survives the other object moving.
#
# The set is filtered, not chosen for variety. `above`, `left_of` and `far`
# were in it first and were removed by measurement: each is satisfied 0.625 of
# the time by standing in one fixed place forever, so an agent that ignores the
# marker entirely scores 1.000 on them and the comparison measures nothing.
# That is the same constant-answer trap the rule sampler already guards, and
# `constant_place_rate` below is the guard.
RELATIONS: tuple[str, ...] = (
    "same",
    "same_row",
    "same_column",
    "adjacent",
    "diagonal",
    "opposite",
)
# A relation solvable this often by a fixed place is not a test of anything
# relational. Measured rates for the set above: 0.125, 0.375, 0.375, 0.500,
# 0.250, 0.125.
CONSTANT_PLACE_LIMIT = 0.5


def _holds(relation: str, mine: int, theirs: int) -> bool:
    row, column = _GRID_POSITIONS[mine]
    other_row, other_column = _GRID_POSITIONS[theirs]
    if relation == "same":
        return mine == theirs
    if relation == "same_row":
        return row == other_row
    if relation == "same_column":
        return column == other_column
    if relation == "adjacent":
        return max(abs(row - other_row), abs(column - other_column)) == 1
    if relation == "diagonal":
        return mine != theirs and abs(row - other_row) == abs(column - other_column)
    if relation == "opposite":
        return (other_row, other_column) == (2 - row, 2 - column)
    raise ValueError(f"unknown relation: {relation}")


def constant_place_rate(relation: str, *, place_count: int = PLACE_COUNT) -> float:
    """How often one fixed place satisfies this, whatever the marker does.

    The guard. A relation with a high rate is answered by an agent that never
    looks at the other object, so it cannot separate a relational
    representation from a positional one -- it only measures whether either can
    find a good place to stand.
    """

    return max(
        sum(float(_holds(relation, mine, theirs)) for theirs in range(int(place_count)))
        / int(place_count)
        for mine in range(int(place_count))
    )


def joint_state(mine: int, theirs: int, *, place_count: int = PLACE_COUNT) -> int:
    """One index for the configuration, because that is what the state is now."""

    if not 0 <= mine < place_count or not 0 <= theirs < place_count:
        raise ValueError("a configuration leaves the place set")
    return int(mine) * int(place_count) + int(theirs)


def split_state(state: int, *, place_count: int = PLACE_COUNT) -> tuple[int, int]:
    return divmod(int(state), int(place_count))


def relation_cumulants(
    *, place_count: int = PLACE_COUNT, relations: Sequence[str] = RELATIONS
) -> torch.Tensor:
    """`phi[configuration]` -- which relations hold, as indicators.

    Rows are configurations rather than places, which is the entire change.
    Everything downstream -- occupancies, generalised policy improvement,
    the library -- takes this and needs no modification, because it was
    always written against a cumulant matrix rather than against places.
    """

    features = torch.zeros(
        (int(place_count) * int(place_count), len(relations)), dtype=torch.float64
    )
    for mine in range(int(place_count)):
        for theirs in range(int(place_count)):
            state = joint_state(mine, theirs, place_count=place_count)
            for index, relation in enumerate(relations):
                features[state, index] = float(_holds(relation, mine, theirs))
    return features


def relation_weights(relation: str, *, relations: Sequence[str] = RELATIONS):
    """A task, as one relation being worth having."""

    if relation not in relations:
        raise ValueError(f"unknown relation: {relation}")
    weights = torch.zeros(len(relations), dtype=torch.float64)
    weights[list(relations).index(relation)] = 1.0
    return weights


def marginal_place_weights(
    relation: str, *, place_count: int = PLACE_COUNT
) -> torch.Tensor:
    """The best a place-only agent can do: ignore the marker and average.

    This is what a `w` over places amounts to when the goal refers to
    something that moves -- for each place, how often the relation would hold
    if the other object were anywhere. It is the fairest version of the old
    representation rather than a strawman: nothing over places can condition on
    where the marker currently is, so the most that representation supports is
    standing where the relation holds most often.
    """

    weights = torch.zeros(int(place_count), dtype=torch.float64)
    for mine in range(int(place_count)):
        weights[mine] = sum(
            float(_holds(relation, mine, theirs)) for theirs in range(int(place_count))
        ) / int(place_count)
    return weights


def satisfying_places(relation: str, theirs: int, *, place_count: int = PLACE_COUNT):
    """Where the relation holds, given where the other object is right now.

    Scoring-side, and the reason the place representation is not merely worse
    but structurally unable: this set is a different set at every step.
    """

    return tuple(
        mine
        for mine in range(int(place_count))
        if _holds(relation, mine, int(theirs))
    )


__all__ = [
    "CONSTANT_PLACE_LIMIT",
    "PLACE_COUNT",
    "RELATIONS",
    "RELATION_SCHEMA",
    "constant_place_rate",
    "joint_state",
    "marginal_place_weights",
    "relation_cumulants",
    "relation_weights",
    "satisfying_places",
    "split_state",
]
