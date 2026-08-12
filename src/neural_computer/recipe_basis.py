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

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Literal

RECIPE_BASIS_SCHEMA = "neural-computer.recipe-basis.v2"
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
SlotValues = int | Sequence[int]


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
    modulus: int | None = None
    children: tuple[RecipeInstruction, RecipeInstruction] | None = None

    def validate(self, *, slot_count: int, allow_parallel: bool = True) -> None:
        if self.op == "parallel":
            if self.modulus is not None:
                raise ValueError("parallel instruction cannot have a modulus")
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
        arithmetic = {"inc", "dec", "cinc", "cdec"}
        if self.op in arithmetic:
            if self.modulus is None or self.modulus < 2:
                raise ValueError("arithmetic instruction needs modulus >= 2")
        elif self.modulus is not None:
            raise ValueError("only arithmetic instructions can have a modulus")
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

    def apply(self, state: tuple[int, ...], *, values: SlotValues) -> tuple[int, ...]:
        """Apply one instruction using explicit per-slot value domains."""

        self.validate(slot_count=len(state), allow_parallel=True)
        slot_values = _normalize_slot_values(values, slot_count=len(state))
        if any(
            value < 0 or value >= slot_values[index]
            for index, value in enumerate(state)
        ):
            raise ValueError("recipe state values are outside the configured domain")
        if self.op == "parallel":
            assert self.children is not None
            first = self.children[0].apply(state, values=slot_values)
            second = self.children[1].apply(state, values=slot_values)
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
        if self.op in {"inc", "dec", "cinc", "cdec"}:
            assert self.modulus is not None
            if self.modulus != slot_values[self.first]:
                raise ValueError(
                    "arithmetic modulus must match the target slot value domain"
                )
        if self.op == "inc":
            result[self.first] = (result[self.first] + 1) % self.modulus
        elif self.op == "dec":
            result[self.first] = (result[self.first] - 1) % self.modulus
        elif self.op == "cinc":
            assert self.second is not None
            if state[self.second] != 0:
                result[self.first] = (result[self.first] + 1) % self.modulus
        elif self.op == "cdec":
            assert self.second is not None
            if state[self.second] != 0:
                result[self.first] = (result[self.first] - 1) % self.modulus
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
        if any(
            value < 0 or value >= slot_values[index]
            for index, value in enumerate(result)
        ):
            raise ValueError("instruction produced an invalid slot value")
        return tuple(result)


def _normalize_slot_values(values: SlotValues, *, slot_count: int) -> tuple[int, ...]:
    if isinstance(values, int):
        slot_values = (values,) * slot_count
    else:
        slot_values = tuple(int(value) for value in values)
        if len(slot_values) != slot_count:
            raise ValueError("slot value domains must match the state width")
    if any(value < 2 for value in slot_values):
        raise ValueError("slot value domains must be at least two")
    return slot_values


@dataclass(frozen=True)
class ExpressibilityResult:
    """Fail-closed result for one atomic transition probe."""

    status: Literal["expressible", "inexpressible", "invalid"]
    instruction: RecipeInstruction | None
    checked_candidates: int
    reason: str


@dataclass(frozen=True)
class SequenceExpressibilityResult:
    """Fail-closed result for a bounded instruction-sequence probe.

    ``inexpressible`` means that every effect reachable up to ``max_length``
    was exhaustively checked.  ``budget_exhausted`` is deliberately separate:
    the search stopped before that bounded space was certified.  This keeps a
    difficult-to-find reusable program distinct from a program outside the
    current instruction basis.
    """

    status: Literal["expressible", "inexpressible", "budget_exhausted", "invalid"]
    instructions: tuple[RecipeInstruction, ...] | None
    checked_candidates: int
    visited_effects: int
    max_length: int
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
        slot_values: Sequence[int] | None = None,
        allow_parallel: bool = False,
    ) -> None:
        if slot_count < 1 or values < 2:
            raise ValueError("recipe basis dimensions must be positive")
        self.slot_count = int(slot_count)
        self.slot_values = _normalize_slot_values(
            values if slot_values is None else slot_values,
            slot_count=self.slot_count,
        )
        self.values = max(self.slot_values)
        self.allow_parallel = bool(allow_parallel)

    def configuration(self) -> dict[str, object]:
        return {
            "schema": RECIPE_BASIS_SCHEMA,
            "slot_count": self.slot_count,
            "values": self.values,
            "slot_values": self.slot_values,
            "allow_parallel": self.allow_parallel,
            "atomicity": "one_instruction_one_verifier_step_v1",
        }

    def atomic_candidates(self) -> tuple[RecipeInstruction, ...]:
        candidates: list[RecipeInstruction] = [RecipeInstruction("noop")]
        for slot in range(self.slot_count):
            candidates.extend(
                (
                    RecipeInstruction(
                        "inc", slot, modulus=self.slot_values[slot]
                    ),
                    RecipeInstruction(
                        "dec", slot, modulus=self.slot_values[slot]
                    ),
                )
            )
        for first in range(self.slot_count):
            for second in range(self.slot_count):
                if first == second:
                    continue
                candidates.extend(
                    (
                        RecipeInstruction(
                            "cinc",
                            first,
                            second,
                            modulus=self.slot_values[first],
                        ),
                        RecipeInstruction(
                            "cdec",
                            first,
                            second,
                            modulus=self.slot_values[first],
                        ),
                        *(
                            (RecipeInstruction("copy", first, second),)
                            if self.slot_values[second] <= self.slot_values[first]
                            else ()
                        ),
                    )
                )
            for second in range(first + 1, self.slot_count):
                if self.slot_values[first] == self.slot_values[second]:
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
            product(*(range(value) for value in self.slot_values))
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
                value < 0 or value >= self.slot_values[index]
                for index, value in enumerate(result)
            ):
                return ExpressibilityResult(
                    "invalid", None, 0, "target returns an invalid register state"
                )
            expected.append(tuple(result))

        candidates = self.atomic_candidates()
        for index, candidate in enumerate(candidates, start=1):
            if all(
                candidate.apply(tuple(state), values=self.slot_values) == want
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

    def sequence_probe(
        self,
        target: Callable[[tuple[int, ...]], tuple[int, ...]],
        *,
        max_length: int,
        states: Iterable[tuple[int, ...]] | None = None,
        max_expansions: int | None = None,
    ) -> SequenceExpressibilityResult:
        """Search for a bounded reusable instruction sequence exactly.

        The search is breadth-first over *observable register effects*, not
        over task names or hand-written semantic programs.  Equivalent
        prefixes are merged, so the returned sequence is shortest within the
        supplied finite probe state set.  If ``max_expansions`` interrupts an
        otherwise complete search, the result is ``budget_exhausted`` rather
        than the stronger ``inexpressible`` claim.
        """

        if (
            not isinstance(max_length, int)
            or isinstance(max_length, bool)
            or max_length < 0
        ):
            return SequenceExpressibilityResult(
                "invalid", None, 0, 0,
                max_length if isinstance(max_length, int) else -1,
                "sequence probe max_length must be a non-negative integer",
            )
        if max_expansions is not None and (
            not isinstance(max_expansions, int)
            or isinstance(max_expansions, bool)
            or max_expansions < 1
        ):
            return SequenceExpressibilityResult(
                "invalid", None, 0, 0, max_length,
                "sequence probe max_expansions must be a positive integer",
            )
        probe_states = tuple(
            product(*(range(value) for value in self.slot_values))
            if states is None
            else tuple(tuple(state) for state in states)
        )
        if not probe_states:
            return SequenceExpressibilityResult(
                "invalid", None, 0, 0, max_length,
                "sequence probe needs at least one state",
            )
        expected: list[tuple[int, ...]] = []
        for state in probe_states:
            if len(state) != self.slot_count or any(
                value < 0 or value >= self.slot_values[index]
                for index, value in enumerate(state)
            ):
                return SequenceExpressibilityResult(
                    "invalid", None, 0, 0, max_length,
                    "probe state is outside the configured register domain",
                )
            try:
                result = tuple(target(state))
            except (AssertionError, IndexError, KeyError, TypeError, ValueError) as error:
                return SequenceExpressibilityResult(
                    "invalid", None, 0, 0, max_length,
                    f"target transition raised {error!r}",
                )
            if len(result) != self.slot_count or any(
                value < 0 or value >= self.slot_values[index]
                for index, value in enumerate(result)
            ):
                return SequenceExpressibilityResult(
                    "invalid", None, 0, 0, max_length,
                    "target returns an invalid register state",
                )
            expected.append(result)

        target_effect = tuple(expected)
        identity_effect = tuple(probe_states)
        if target_effect == identity_effect:
            return SequenceExpressibilityResult(
                "expressible", (), 0, 1, max_length,
                "target is the empty instruction sequence",
            )

        candidates = self.atomic_candidates()
        frontier: list[tuple[tuple[tuple[int, ...], ...], tuple[RecipeInstruction, ...]]] = [
            (identity_effect, ())
        ]
        visited = {identity_effect}
        checked = 0
        for _depth in range(1, max_length + 1):
            next_frontier: list[
                tuple[tuple[tuple[int, ...], ...], tuple[RecipeInstruction, ...]]
            ] = []
            for effect, prefix in frontier:
                for candidate in candidates:
                    if max_expansions is not None and checked >= max_expansions:
                        return SequenceExpressibilityResult(
                            "budget_exhausted", None, checked, len(visited),
                            max_length,
                            "search budget ended before the bounded space was certified",
                        )
                    next_effect = tuple(
                        candidate.apply(state, values=self.slot_values)
                        for state in effect
                    )
                    checked += 1
                    sequence = (*prefix, candidate)
                    if next_effect == target_effect:
                        return SequenceExpressibilityResult(
                            "expressible", sequence, checked, len(visited),
                            max_length,
                            "target has an exact bounded sequence representation",
                        )
                    if next_effect not in visited:
                        visited.add(next_effect)
                        next_frontier.append((next_effect, sequence))
            if not next_frontier:
                break
            frontier = next_frontier
        return SequenceExpressibilityResult(
            "inexpressible", None, checked, len(visited), max_length,
            "target is not reachable within the certified sequence bound",
        )


def paired_increment_target(
    first: int,
    second: int,
    *,
    modulus: int,
) -> Callable[[tuple[int, ...]], tuple[int, ...]]:
    """Return a generic two-slot simultaneous increment effect.

    On a two-valued subdomain, increment is the usual bit flip.  The modulus
    is part of the instruction's data contract; it is not inherited from one
    global register width.
    """

    if first == second:
        raise ValueError("paired target needs two distinct slots")
    if modulus < 2:
        raise ValueError("paired target needs modulus >= 2")

    def target(state: tuple[int, ...]) -> tuple[int, ...]:
        result = list(state)
        result[first] = (result[first] + 1) % modulus
        result[second] = (result[second] + 1) % modulus
        return tuple(result)

    return target


def apply_sequence(
    instructions: Iterable[RecipeInstruction],
    state: tuple[int, ...],
    *,
    values: SlotValues,
) -> tuple[int, ...]:
    """Apply an opaque instruction sequence with one explicit value contract."""

    current = state
    for instruction in instructions:
        current = instruction.apply(current, values=values)
    return current


__all__ = [
    "RECIPE_BASIS_SCHEMA",
    "ExpressibilityResult",
    "RecipeBasis",
    "RecipeInstruction",
    "SequenceExpressibilityResult",
    "apply_sequence",
    "paired_increment_target",
]
