"""Fit the machine that disagrees least, instead of one that never disagrees.

Three attempts at noise tolerance failed here, and each failure said something
that this one uses.

*Statistical merging* compared output proportions with a Hoeffding bound. At
the counts a short-episode prefix tree produces the bound is vacuous, and a
four-state rule came back as one state at zero noise.

*A violation budget in the exact search* let a disagreement be spent instead of
being fatal. A violation can be spent anywhere, so the branching multiplied and
the search stopped finishing at all.

*Prefix-level majority voting* outvoted the noise where evidence was thick and
dropped the rest. It dropped 84% of it, and with only shallow constraints left
the search identified nothing even at zero noise -- threshold-3 fell to 0.02.
The lesson is the useful one: the unit that recurs often enough to vote on is
not the prefix, it is the **state**. A four-state machine over 1792 labelled
steps visits each of its sixteen cells about a hundred times. Prefixes at that
depth are visited once.

But states are what noise stops us from learning, so voting per state needs a
machine and the machine needs the vote. The way out of that circle is to stop
treating the machine as something to be *derived* and start treating it as
something to be *scored*: pick the machine that disagrees with the fewest
labels. That objective is defined for every machine, needs no consistency, and
degrades smoothly instead of failing.

Search is local. Start from a machine, change one cell -- one transition target
or one output bit -- keep the change if disagreements fall, repeat to a local
minimum, restart. Each evaluation is one pass over the evidence, and the
neighbourhood is `states x symbols x (states + 1)` moves, which is small at the
sizes that matter.

Model size is chosen by description length, because "fewest disagreements"
alone always prefers the largest machine on offer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .robust_induction import description_bits, error_bits
from .rule_automata import RuleAutomaton, minimize

NOISE_TOLERANT_SCHEMA = "neural-computer.noise-tolerant-induction.v1"
MAX_STATES = 8
RESTARTS = 6
MAX_SWEEPS = 40


@dataclass(frozen=True)
class Fit:
    """A machine, how much it disagrees with the evidence, and what it costs."""

    machine: RuleAutomaton
    disagreements: int
    trials: int
    description_bits: float
    error_bits: float

    @property
    def error_rate(self) -> float:
        return self.disagreements / self.trials if self.trials else 1.0

    @property
    def total_bits(self) -> float:
        return self.description_bits + self.error_bits

    def payload(self) -> dict[str, object]:
        return {
            "schema": NOISE_TOLERANT_SCHEMA,
            "states": self.machine.state_count,
            "disagreements": self.disagreements,
            "trials": self.trials,
            "error_rate": self.error_rate,
            "total_bits": self.total_bits,
        }


def _flatten(traces):
    """Episodes as flat arrays plus the index each one starts at."""

    symbols: list[int] = []
    outputs: list[int] = []
    eligible: list[bool] = []
    starts: list[int] = []
    for trace in traces:
        starts.append(len(symbols))
        symbols.extend(int(value) for value in trace.symbols)
        outputs.extend(int(value) for value in trace.outputs)
        eligible.extend(bool(value) for value in trace.eligible)
    return symbols, outputs, eligible, set(starts)


def _best_outputs(
    transitions, symbols, outputs, eligible, starts, states, symbol_count
):
    """Given transitions, the output table follows by majority -- no search needed.

    Every eligible step votes in exactly one cell, so the table that minimises
    disagreements is the majority of each cell. This is what makes the local
    search cheap: only the transitions are searched over.
    """

    ones = [[0] * symbol_count for _ in range(states)]
    totals = [[0] * symbol_count for _ in range(states)]
    state = 0
    for index, symbol in enumerate(symbols):
        if index in starts:
            state = 0
        if eligible[index]:
            totals[state][symbol] += 1
            ones[state][symbol] += outputs[index]
        state = transitions[state][symbol]
    table = [
        [
            1 if totals[s][y] and ones[s][y] * 2 > totals[s][y] else 0
            for y in range(symbol_count)
        ]
        for s in range(states)
    ]
    wrong = sum(
        (totals[s][y] - ones[s][y]) if table[s][y] else ones[s][y]
        for s in range(states)
        for y in range(symbol_count)
    )
    return table, wrong


def _climb(transitions, symbols, outputs, eligible, starts, states, symbol_count):
    """Change one transition at a time while it helps."""

    table, best = _best_outputs(
        transitions, symbols, outputs, eligible, starts, states, symbol_count
    )
    for _ in range(MAX_SWEEPS):
        improved = False
        for state in range(states):
            for symbol in range(symbol_count):
                current = transitions[state][symbol]
                for target in range(states):
                    if target == current:
                        continue
                    transitions[state][symbol] = target
                    candidate, score = _best_outputs(
                        transitions,
                        symbols,
                        outputs,
                        eligible,
                        starts,
                        states,
                        symbol_count,
                    )
                    if score < best:
                        best, table, current, improved = score, candidate, target, True
                    else:
                        transitions[state][symbol] = current
        if not improved:
            break
    return table, best


def induce_noise_tolerant(
    traces,
    *,
    max_states: int = MAX_STATES,
    restarts: int = RESTARTS,
    seed: int = 0,
) -> Fit | None:
    """Smallest-description machine that disagrees least with the evidence."""

    episodes = tuple(traces)
    if not episodes:
        return None
    symbols, outputs, eligible, starts = _flatten(episodes)
    trials = sum(1 for flag in eligible if flag)
    if trials < 2:
        return None
    symbol_count = max(trace.symbol_count for trace in episodes)
    generator = torch.Generator().manual_seed(int(seed))
    best: Fit | None = None
    for states in range(1, max_states + 1):
        best_here: tuple[int, list, list] | None = None
        for restart in range(restarts):
            if restart == 0:
                # A machine that ignores its input is a neutral start.
                transitions = [[0] * symbol_count for _ in range(states)]
            else:
                transitions = [
                    [
                        int(value)
                        for value in torch.randint(
                            0, states, (symbol_count,), generator=generator
                        )
                    ]
                    for _ in range(states)
                ]
            table, score = _climb(
                transitions, symbols, outputs, eligible, starts, states, symbol_count
            )
            if best_here is None or score < best_here[0]:
                best_here = (score, [row[:] for row in transitions], table)
            if score == 0:
                break
        if best_here is None:
            continue
        score, transitions, table = best_here
        machine = minimize(
            RuleAutomaton(
                symbol_count=symbol_count,
                transitions=tuple(tuple(row) for row in transitions),
                outputs=tuple(tuple(row) for row in table),
            )
        )
        candidate = Fit(
            machine=machine,
            disagreements=score,
            trials=trials,
            description_bits=description_bits(machine),
            error_bits=error_bits(score, trials),
        )
        if best is None or candidate.total_bits < best.total_bits:
            best = candidate
    return best
