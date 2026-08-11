"""A small domain-neutral recipe basis and fail-closed expressibility probe.

This module is an architectural instrument, not a task solver.  It models a
fixed-width abstract register whose slots are opaque values.  The basis holds
generic local effects (increment, decrement, conditional update, copy, and
swap); no task, modality, or semantic label is represented.

The important distinction is between a recipe that is hard to find and an
atomic transition that the current basis cannot express.  Search should be
allowed to return an explicit ``inexpressible`` result instead of silently
exhausting a budget and minting a bad external file.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from itertools import product
from typing import Literal

RECIPE_BASIS_SCHEMA = "neural-computer.recipe-basis.v1"
RecipeOp = Literal[
    "noop",
    "inc",
    "dec",
    "cinc",
    "cdec",
    "copy",
    "swap",
    "parallel",
]


@dataclass(frozen=True)
class RecipeInstruction:
    """One opaque generic instruction over abstract register slots.

    ``parallel`` is the only structural extension.  It applies two disjoint
    local instructions to the same pre-step state, then commits both effects
    atomically.  It is deliberately not named after the benchmark operation
    it can express.
    """

    op: RecipeOp
    first: int | None = None
    second: int | None = None
    children: tuple[RecipeInstruction, RecipeInstruction] | None = None

    def validate(self, *, slot_count: int, allow_parallel: bool = True) -> None:
        if self.op == "parallel":
            if not allow_parallel:
                raise ValueError("parallel instruction is outside this basis")
            if self.children is None or len(self.children) != 2:
                raise ValueError("parallel instruction needs two children")
            for child in self.children:
                child.validate(slot_count=slot_count, allow_parallel=False)
            first = self.children[0].written_slots()
            second = self.children[1].written_slots()
            if first & second:
                raise ValueError("parallel children must write disjoint slots")
            return
        if self.children is not None:
            raise ValueError("atomic instruction cannot contain children")
        if self.op == "noop":
            if self.first is not None or self.second is not None:
                raise ValueError("noop cannot have slot arguments")
            return
        if self.first is None or not 0 <= self.first < slot_count:
            raise ValueError("instruction first slot is out of range")
        if self.op in {"cinc", "cdec", "copy", "swap"}:
            if self.second is None or not 0 <= self.second < slot_count:
                raise ValueError("instruction second slot is out of range")
            if self.op == "swap" and self.first == self.second:
                raise ValueError("swap needs two distinct slots")
        elif self.second is not None:
            raise ValueError("unary instruction cannot have a second slot")

    def read_slots(self) -> frozenset[int]:
        if self.op == "parallel":
            assert self.children is not None
            return self.children[0].read_slots() | self.children[1].read_slots()
        if self.op in {"cinc", "cdec", "copy", "swap"}:
            assert self.first is not None and self.second is not None
            return frozenset((self.first, self.second))
        return frozenset() if self.op == "noop" else frozenset((self.first,))

    def written_slots(self) -> frozenset[int]:
        if self.op == "parallel":
            assert self.children is not None
            return self.children[0].written_slots() | self.children[1].written_slots()
        if self.op in {"noop", "inc", "dec", "cinc", "cdec"}:
            return frozenset() if self.op == "noop" else frozenset((self.first,))
        if self.op == "copy":
            assert self.first is not None
            return frozenset((self.first,))
        if self.op == "swap":
            assert self.first is not None and self.second is not None
            return frozenset((self.first, self.second))
        raise ValueError(f"unsupported recipe operation: {self.op!r}")

    def apply(self, state: tuple[int, ...], *, values: int) -> tuple[int, ...]:
        """Apply one instruction without exposing semantic meanings."""

        self.validate(slot_count=len(state), allow_parallel=True)
        if values < 2 or any(value < 0 or value >= values for value in state):
            raise ValueError("recipe state values are outside the configured domain")
        if self.op == "parallel":
            assert self.children is not None
            first = self.children[0].apply(state, values=values)
            second = self.children[1].apply(state, values=values)
            merged = list(state)
            for slot in self.children[0].written_slots():
                merged[slot] = first[slot]
            for slot in self.children[1].written_slots():
                merged[slot] = second[slot]
            return tuple(merged)

        if self.op == "noop":
            return state
        assert self.first is not None
        result = list(state)
        if self.op == "inc":
            result[self.first] = (result[self.first] + 1) % values
        elif self.op == "dec":
            result[self.first] = (result[self.first] - 1) % values
        elif self.op == "cinc":
            assert self.second is not None
            if state[self.second] != 0:
                result[self.first] = (result[self.first] + 1) % values
        elif self.op == "cdec":
            assert self.second is not None
            if state[self.second] != 0:
                result[self.first] = (result[self.first] - 1) % values
        elif self.op == "copy":
            assert self.second is not None
            result[self.first] = state[self.second]
        elif self.op == "swap":
            assert self.second is not None
            result[self.first], result[self.second] = (
                state[self.second],
                state[self.first],
            )
        else:
            raise ValueError(f"unsupported recipe operation: {self.op!r}")
        return tuple(result)


@dataclass(frozen=True)
class ExpressibilityResult:
    """Fail-closed result for one atomic transition probe."""

    status: Literal["expressible", "inexpressible", "invalid"]
    instruction: RecipeInstruction | None
    checked_candidates: int
    reason: str


class RecipeBasis:
    """Versioned abstract instruction basis with optional parallel composition."""

    _ATOMIC_OPS: tuple[RecipeOp, ...] = (
        "noop",
        "inc",
        "dec",
        "cinc",
        "cdec",
        "copy",
        "swap",
    )

    def __init__(
        self,
        slot_count: int = 6,
        values: int = 8,
        *,
        allow_parallel: bool = False,
    ) -> None:
        if slot_count < 1 or values < 2:
            raise ValueError("recipe basis dimensions must be positive")
        self.slot_count = int(slot_count)
        self.values = int(values)
        self.allow_parallel = bool(allow_parallel)

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": RECIPE_BASIS_SCHEMA,
            "slot_count": self.slot_count,
            "values": self.values,
            "allow_parallel": self.allow_parallel,
            "atomicity": "one_instruction_one_verifier_step_v1",
        }

    def atomic_candidates(self) -> tuple[RecipeInstruction, ...]:
        candidates: list[RecipeInstruction] = [RecipeInstruction("noop")]
        for slot in range(self.slot_count):
            candidates.extend(
                (RecipeInstruction("inc", slot), RecipeInstruction("dec", slot))
            )
        for first in range(self.slot_count):
            for second in range(self.slot_count):
                if first == second:
                    continue
                candidates.extend(
                    (
                        RecipeInstruction("cinc", first, second),
                        RecipeInstruction("cdec", first, second),
                        RecipeInstruction("copy", first, second),
                    )
                )
            for second in range(first + 1, self.slot_count):
                candidates.append(RecipeInstruction("swap", first, second))
        if self.allow_parallel:
            atomic = tuple(candidates)
            for left_index, left in enumerate(atomic):
                for right in atomic[left_index + 1 :]:
                    if not left.written_slots() or not right.written_slots():
                        continue
                    if left.written_slots() & right.written_slots():
                        continue
                    candidates.append(
                        RecipeInstruction("parallel", children=(left, right))
                    )
        return tuple(candidates)

    def expressibility_probe(
        self,
        target: Callable[[tuple[int, ...]], tuple[int, ...]],
        *,
        states: Iterable[tuple[int, ...]] | None = None,
    ) -> ExpressibilityResult:
        """Determine whether ``target`` is an expressible atomic transition.

        The target is checked over the supplied states.  If omitted, all
        states are enumerated; this is exact for the configured finite domain.
        A mismatch is an explicit ``inexpressible`` result, not a search
        timeout or a fabricated candidate.
        """

        probe_states = tuple(
            product(range(self.values), repeat=self.slot_count)
            if states is None
            else states
        )
        if not probe_states:
            return ExpressibilityResult(
                "invalid", None, 0, "expressibility probe needs at least one state"
            )
        expected: list[tuple[int, ...]] = []
        for state in probe_states:
            if len(state) != self.slot_count:
                return ExpressibilityResult(
                    "invalid", None, 0, "probe state has the wrong slot count"
                )
            try:
                result = target(tuple(state))
            except (AssertionError, IndexError, KeyError, TypeError, ValueError) as error:
                return ExpressibilityResult(
                    "invalid", None, 0, f"target transition raised {error!r}"
                )
            if len(result) != self.slot_count or any(
                value < 0 or value >= self.values for value in result
            ):
                return ExpressibilityResult(
                    "invalid", None, 0, "target returns an invalid register state"
                )
            expected.append(tuple(result))

        candidates = self.atomic_candidates()
        for index, candidate in enumerate(candidates, start=1):
            if all(
                candidate.apply(tuple(state), values=self.values) == want
                for state, want in zip(probe_states, expected, strict=True)
            ):
                return ExpressibilityResult(
                    "expressible",
                    candidate,
                    index,
                    "target has an exact atomic representation in this basis",
                )
        return ExpressibilityResult(
            "inexpressible",
            None,
            len(candidates),
            "target is outside the atomic instruction basis",
        )


def paired_increment_target(
    first: int,
    second: int,
    *,
    values: int = 8,
) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
    """Return a generic two-slot simultaneous increment effect.

    On a two-valued subdomain, increment is the usual bit flip.  The
    experiment uses this representative because the structural question is
    whether one local effect can apply to a group of slots, not whether the
    basis should contain a benchmark-named ``flip`` primitive.
    """

    if first == second:
        raise ValueError("paired target needs two distinct slots")
    if values < 2:
        raise ValueError("paired target needs at least two values")

    def target(state: tuple[int, ...]) -> tuple[int, ...]:
        result = list(state)
        result[first] = (result[first] + 1) % values
        result[second] = (result[second] + 1) % values
        return tuple(result)

    return target


__all__ = [
    "RECIPE_BASIS_SCHEMA",
    "ExpressibilityResult",
    "RecipeBasis",
    "RecipeInstruction",
    "paired_increment_target",
]
