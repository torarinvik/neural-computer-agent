"""Try to break the induction stack before believing any of it.

Everything built on top of feedback inversion -- identification, factorisation,
fitted output tables -- rests on two assumptions that were never tested because
the benchmark never violated them.

**Labels are exact.** `infer_machine` backtracks on any contradiction and
`fit_output_table` returns `None` on a single conflicting cell. One flipped
reward in a thousand is a contradiction, so the honest prediction is that the
whole stack is not merely degraded by noise but destroyed by it.

**The target is finite-state.** The task sampler draws Mealy machines, the
inference searches Mealy machines, the decomposition is Hartmanis-Stearns over
Mealy machines, and the compiler emits a state machine. If the environment ever
asks for something that counts, nothing in the chain can represent it -- and a
procedure that cannot represent the answer will confidently return a wrong one
rather than abstain, which is worse.

This module builds probes for both, plus three lesser ones that plausibly
matter: a larger alphabet than the four the frontend was tuned on, a rule that
changes partway through, and eligibility that hides part of the evidence.

The probes are deliberately *not* run through the rendered environment. Every
one of them isolates the inference stack on synthetic traces, so a failure
points at the inference rather than at the frontend, the clustering, or the
verifier. What survives here earns an environment run; what dies here does not.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from .identification_ceiling import Trace

PROBE_SCHEMA = "neural-computer.adversarial-probe.v1"


@dataclass(frozen=True)
class Probe:
    """A named target behaviour over a symbol alphabet."""

    name: str
    symbol_count: int
    predicate: Callable[[list[int]], list[int]]
    finite_state: bool
    note: str


def _stream(count: int, symbol_count: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, symbol_count, (count,), generator=generator).tolist()


def count_threshold(symbol: int, threshold: int) -> Callable[[list[int]], list[int]]:
    """Press once at least `threshold` copies of a symbol have been seen.

    Finite-state for a fixed threshold -- it needs `threshold + 1` states -- so
    this is the *inside*-the-class control for the counting probes, and it
    should be identified. If it is not, the counting failures below say nothing
    about class membership.
    """

    def rule(symbols: list[int]) -> list[int]:
        seen = 0
        produced = []
        for value in symbols:
            seen += int(value == symbol)
            produced.append(int(seen >= threshold))
        return produced

    return rule


def running_majority(first: int, second: int) -> Callable[[list[int]], list[int]]:
    """Press while one symbol has occurred strictly more often than another.

    Not finite-state: the difference between two counts is unbounded, so no
    fixed number of states tracks it over an arbitrary stream. This is the
    probe that asks whether the whole stack is a Mealy-machine fitter wearing
    a general-agent label.
    """

    def rule(symbols: list[int]) -> list[int]:
        balance = 0
        produced = []
        for value in symbols:
            balance += int(value == first) - int(value == second)
            produced.append(int(balance > 0))
        return produced

    return rule


def count_parity(symbol: int) -> Callable[[list[int]], list[int]]:
    """Press on an odd count. Finite-state at two states, however long.

    Included because it *looks* like counting and is not, which is exactly the
    distinction a general learner has to get right without being told.
    """

    def rule(symbols: list[int]) -> list[int]:
        seen = 0
        produced = []
        for value in symbols:
            seen += int(value == symbol)
            produced.append(seen % 2)
        return produced

    return rule


def switching(
    before: Callable[[list[int]], list[int]],
    after: Callable[[list[int]], list[int]],
    at: int,
) -> Callable[[list[int]], list[int]]:
    """One rule, then another. Stationarity is an assumption, so test it."""

    def rule(symbols: list[int]) -> list[int]:
        head = before(symbols)
        tail = after(symbols)
        return [head[index] if index < at else tail[index] for index in range(len(symbols))]

    return rule


def probes() -> tuple[Probe, ...]:
    return (
        Probe(
            "threshold-3",
            4,
            count_threshold(0, 3),
            True,
            "finite-state control for the counting probes",
        ),
        Probe(
            "parity",
            4,
            count_parity(0),
            True,
            "two states however long the stream",
        ),
        Probe(
            "majority",
            4,
            running_majority(0, 1),
            False,
            "unbounded counter difference; outside the class",
        ),
        Probe(
            "wide-alphabet",
            8,
            count_parity(5),
            True,
            "twice the alphabet the frontend was tuned on",
        ),
        Probe(
            "switching",
            4,
            switching(count_parity(0), count_parity(1), 224),
            False,
            "rule changes partway; stationarity violated",
        ),
    )


def trace_for(
    probe: Probe,
    *,
    seed: int,
    length: int,
    noise: float = 0.0,
    eligible_fraction: float = 1.0,
) -> Trace:
    """One episode of a probe, optionally corrupted.

    Noise flips the *label*, which is what a mis-scored press looks like after
    the reward is inverted. Eligibility hides steps entirely, which is what a
    verifier that does not score every tick looks like.
    """

    symbols = _stream(length, probe.symbol_count, seed)
    outputs = probe.predicate(symbols)
    generator = torch.Generator().manual_seed(seed + 77)
    if noise > 0.0:
        flips = torch.rand(length, generator=generator) < noise
        outputs = [
            value ^ int(flip) for value, flip in zip(outputs, flips.tolist())
        ]
    if eligible_fraction >= 1.0:
        eligible = [True] * length
    else:
        keep = torch.rand(length, generator=generator) < eligible_fraction
        eligible = [bool(value) for value in keep.tolist()]
        if not any(eligible):
            eligible[0] = True
    return Trace(
        symbols=tuple(symbols),
        outputs=tuple(outputs),
        eligible=tuple(eligible),
        symbol_count=probe.symbol_count,
    )


def clean_accuracy(machine, probe: Probe, *, seed: int, length: int) -> float:
    """Score a hypothesis against the probe's *uncorrupted* behaviour.

    Scoring against a noisy held-out trace would let a hypothesis be right for
    the wrong reason, or wrong for no reason. The target is what the rule
    actually does.
    """

    truth = trace_for(probe, seed=seed, length=length)
    predicted = machine.expected(list(truth.symbols))
    hits = sum(
        1
        for index, flag in enumerate(truth.eligible)
        if flag and predicted[index] == truth.outputs[index]
    )
    trials = sum(1 for flag in truth.eligible if flag)
    return hits / trials if trials else 0.0
