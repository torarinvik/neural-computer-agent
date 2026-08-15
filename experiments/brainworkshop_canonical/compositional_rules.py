"""A task distribution where accumulation is possible at all.

Every arm of the accumulation curve reported the same thing: zero composes,
zero inverts of a learned file, zero ANDs over a learned file. That was read
three times as a fact about the architecture. It is not.

`rule_automata.sample_rule_population` draws each rule independently. Over the
eighteen sampled rules the largest pairwise agreement between any two, or its
complement, is 0.808 -- and that is between a one-state rule and a four-state
one, which is coincidence rather than structure. Independent samples share
nothing, so no library can ever help with the next one except by being an
exact duplicate. That is precisely the reuse the curve measured, and it is the
only reuse the distribution permits.

Testing whether capability accumulates therefore needs a distribution in which
it *could*. This module builds one, and builds it mechanically rather than by
hand -- the standing constraint on this work is that the task class must not be
a pile of written-out predicates chosen to flatter the agent.

The construction is the product of Mealy machines, which is the standard way
these compose and needs no new theory. Two primitives are sampled, run in
parallel over the same symbol stream, and their outputs merged by a boolean
combiner. The result is a Mealy machine on the product state space, so a
composite of two three-state primitives has up to nine states and is a
genuinely harder task -- but a *decomposable* one, which is the whole point.

Primitives are drawn once into a shared pool and reused across composites, so
a library that has learned a primitive from one task has something real to
offer the next. Whether it can use it is the question these tasks exist to
ask, and the answer is no longer settled in advance by the sampler.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .rule_automata import RuleAutomaton, minimize, positive_rate, sample_rule

COMBINERS: dict[str, callable] = {
    "and": lambda left, right: int(bool(left) and bool(right)),
    "or": lambda left, right: int(bool(left) or bool(right)),
    "xor": lambda left, right: int(bool(left) != bool(right)),
}


@dataclass(frozen=True)
class CompositeRule:
    """A product task, with the primitives it was built from recorded.

    The parts are kept for *scoring* an experiment -- to ask whether a library
    holding a primitive made the composite cheaper -- and are never handed to
    a learner. `automaton` is all the environment ever sees.
    """

    automaton: RuleAutomaton
    parts: tuple[RuleAutomaton, ...]
    combiner: str

    @property
    def state_count(self) -> int:
        return self.automaton.state_count

    def digest(self) -> str:
        return self.automaton.digest()

    def part_digests(self) -> tuple[str, ...]:
        return tuple(part.digest() for part in self.parts)


def product_rule(
    left: RuleAutomaton, right: RuleAutomaton, combiner: str
) -> RuleAutomaton:
    """Run both machines on the same stream; merge their outputs.

    States are pairs, so this is a Mealy machine again and everything already
    written for the class applies to it unchanged. Minimising afterwards
    matters: a product is frequently smaller than the pair count suggests, and
    the honest complexity of the task is what survives minimisation.
    """

    if left.symbol_count != right.symbol_count:
        raise ValueError("composed rules must share an alphabet")
    if combiner not in COMBINERS:
        raise ValueError(f"unknown combiner: {combiner}")
    merge = COMBINERS[combiner]
    symbols = left.symbol_count
    pairs = [
        (first, second)
        for first in range(left.state_count)
        for second in range(right.state_count)
    ]
    index = {pair: position for position, pair in enumerate(pairs)}
    transitions = []
    outputs = []
    for first, second in pairs:
        row = []
        emission = []
        for symbol in range(symbols):
            row.append(
                index[
                    (
                        int(left.transitions[first][symbol]),
                        int(right.transitions[second][symbol]),
                    )
                ]
            )
            emission.append(
                merge(
                    int(left.outputs[first][symbol]),
                    int(right.outputs[second][symbol]),
                )
            )
        transitions.append(tuple(row))
        outputs.append(tuple(emission))
    return minimize(
        RuleAutomaton(
            symbol_count=symbols,
            transitions=tuple(transitions),
            outputs=tuple(outputs),
        )
    )


def sample_primitive_pool(
    *,
    symbol_count: int = 4,
    count: int = 4,
    state_counts: tuple[int, ...] = (2, 3),
    seed: int = 8000,
) -> tuple[RuleAutomaton, ...]:
    """A small shared vocabulary of parts, sampled rather than written."""

    pool: list[RuleAutomaton] = []
    digests: set[str] = set()
    attempt = 0
    while len(pool) < count and attempt < count * 200:
        states = state_counts[len(pool) % len(state_counts)]
        rule = sample_rule(
            symbol_count=symbol_count, state_count=states, seed=seed + attempt
        )
        attempt += 1
        if rule is None or rule.digest() in digests:
            continue
        digests.add(rule.digest())
        pool.append(rule)
    if len(pool) < count:
        raise ValueError("could not sample a distinct primitive pool")
    return tuple(pool)


def sample_composite_population(
    pool: tuple[RuleAutomaton, ...],
    *,
    combiners: tuple[str, ...] = ("and", "or", "xor"),
    minimum_positive_rate: float = 0.15,
    maximum_positive_rate: float = 0.85,
    seed: int = 21,
) -> tuple[CompositeRule, ...]:
    """Every distinct pair of primitives under every combiner, once.

    A composite whose press rate sits near zero or one cannot separate a
    learner from a constant policy at any episode length, so it is dropped on
    the same grounds the primitive sampler uses.
    """

    composites: list[CompositeRule] = []
    seen: set[str] = set()
    for left, right in combinations(range(len(pool)), 2):
        for combiner in combiners:
            automaton = product_rule(pool[left], pool[right], combiner)
            if automaton.digest() in seen:
                continue
            rate = positive_rate(automaton, seed=seed)
            if not minimum_positive_rate <= rate <= maximum_positive_rate:
                continue
            seen.add(automaton.digest())
            composites.append(
                CompositeRule(
                    automaton=automaton,
                    parts=(pool[left], pool[right]),
                    combiner=combiner,
                )
            )
    if not composites:
        raise ValueError("no composite cleared the press-rate window")
    return tuple(composites)


def shares_a_part(first: CompositeRule, second: CompositeRule) -> bool:
    """Whether two composites were built from an overlapping vocabulary."""

    return bool(set(first.part_digests()) & set(second.part_digests()))
