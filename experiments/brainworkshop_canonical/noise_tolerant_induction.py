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


def class_weights(outputs, eligible, *, balanced: bool) -> tuple[float, float]:
    """How much a mistake on each label costs.

    Unweighted, a rule that presses rarely is beaten by a machine that never
    presses at all: the noise-tolerance record measured a threshold-6 rule
    collapsing to one state at an error the size of its own positive rate.
    "Fewest disagreements" is the wrong objective under class imbalance,
    because the majority class buys the argument.

    Balanced, each label's mistakes are priced by the inverse of how often that
    label occurs, so the two classes contribute equally however skewed the
    stream is. Equal frequencies give equal weights, which is why turning this
    on changes nothing on a balanced task.
    """

    if not balanced:
        return 1.0, 1.0
    ones = sum(
        1 for value, flag in zip(outputs, eligible, strict=True) if flag and value
    )
    total = sum(1 for flag in eligible if flag)
    zeros = total - ones
    if not ones or not zeros:
        # One class never occurs; there is no imbalance to correct, and
        # inverting a zero frequency would be an infinite price.
        return 1.0, 1.0
    return 0.5 * total / zeros, 0.5 * total / ones


def _best_outputs(
    transitions,
    symbols,
    outputs,
    eligible,
    starts,
    states,
    symbol_count,
    weights=(1.0, 1.0),
):
    """Given transitions, the output table follows by majority -- no search needed.

    Every eligible step votes in exactly one cell, so the table that minimises
    the weighted disagreements is the weighted majority of each cell. This is
    what makes the local search cheap: only the transitions are searched over.

    Two scores come back. The weighted one is what the search and the model
    selection minimise. The raw count is what the fit reports as its error
    rate, and it stays unweighted on purpose: that number is used elsewhere as
    a calibrated estimate of the label noise, and a reweighted version of it
    would no longer estimate anything.
    """

    zero_weight, one_weight = weights
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
            1
            if totals[s][y]
            and ones[s][y] * one_weight > (totals[s][y] - ones[s][y]) * zero_weight
            else 0
            for y in range(symbol_count)
        ]
        for s in range(states)
    ]
    score = 0.0
    wrong = 0
    for s in range(states):
        for y in range(symbol_count):
            zeros = totals[s][y] - ones[s][y]
            if table[s][y]:
                score += zeros * zero_weight
                wrong += zeros
            else:
                score += ones[s][y] * one_weight
                wrong += ones[s][y]
    return table, score, wrong


def _climb(
    transitions,
    symbols,
    outputs,
    eligible,
    starts,
    states,
    symbol_count,
    weights=(1.0, 1.0),
):
    """Change one transition at a time while it helps."""

    table, best, raw = _best_outputs(
        transitions, symbols, outputs, eligible, starts, states, symbol_count, weights
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
                    candidate, score, candidate_raw = _best_outputs(
                        transitions,
                        symbols,
                        outputs,
                        eligible,
                        starts,
                        states,
                        symbol_count,
                        weights,
                    )
                    if score < best:
                        best, table, current, improved = score, candidate, target, True
                        raw = candidate_raw
                    else:
                        transitions[state][symbol] = current
        if not improved:
            break
    return table, best, raw


def induce_noise_tolerant(
    traces,
    *,
    max_states: int = MAX_STATES,
    restarts: int = RESTARTS,
    seed: int = 0,
    balanced: bool = False,
) -> Fit | None:
    """Smallest-description machine that disagrees least with the evidence.

    `balanced` prices each label's mistakes by the inverse of how often it
    occurs, which is what a rule with rare positives needs and what a balanced
    one is unaffected by.
    """

    episodes = tuple(traces)
    if not episodes:
        return None
    symbols, outputs, eligible, starts = _flatten(episodes)
    trials = sum(1 for flag in eligible if flag)
    if trials < 2:
        return None
    symbol_count = max(trace.symbol_count for trace in episodes)
    weights = class_weights(outputs, eligible, balanced=balanced)
    generator = torch.Generator().manual_seed(int(seed))
    best: Fit | None = None
    for states in range(1, max_states + 1):
        best_here: tuple[float, list, list, int] | None = None
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
            table, score, raw = _climb(
                transitions,
                symbols,
                outputs,
                eligible,
                starts,
                states,
                symbol_count,
                weights,
            )
            if best_here is None or score < best_here[0]:
                best_here = (score, [row[:] for row in transitions], table, raw)
            if score == 0:
                break
        if best_here is None:
            continue
        score, transitions, table, raw = best_here
        machine = minimize(
            RuleAutomaton(
                symbol_count=symbol_count,
                transitions=tuple(tuple(row) for row in transitions),
                outputs=tuple(tuple(row) for row in table),
            )
        )
        candidate = Fit(
            machine=machine,
            disagreements=raw,
            trials=trials,
            description_bits=description_bits(machine),
            # Selection is on the weighted score; the reported error rate is
            # the raw one, so `error_rate` keeps estimating the label noise.
            error_bits=error_bits(round(score), trials),
        )
        if best is None or candidate.total_bits < best.total_bits:
            best = candidate
    return best


def balanced_accuracy(machine: RuleAutomaton, traces) -> float:
    """Mean of the two per-class accuracies, on evidence in hand.

    The measure the plain objective is blind to. A machine that never presses
    scores its own negative rate under accuracy -- 0.978 on a stream with 2%
    positives -- and exactly 0.5 here, which is chance, which is what it is.
    """

    hits = [0, 0]
    totals = [0, 0]
    for trace in traces:
        predicted = machine.expected(list(trace.symbols))
        for index, flag in enumerate(trace.eligible):
            if not flag:
                continue
            label = int(trace.outputs[index])
            totals[label] += 1
            hits[label] += int(predicted[index] == label)
    present = [label for label in (0, 1) if totals[label]]
    if not present:
        return 0.0
    return sum(hits[label] / totals[label] for label in present) / len(present)


def induce_validated(fit_traces, validation_traces, **kwargs) -> Fit | None:
    """Fit both objectives, and let held-out evidence pick.

    Neither objective is right on its own, which the measurements say plainly.

    *Plain* -- fewest disagreements -- is beaten by the majority class whenever
    positives are rare: at a positive rate of 0.022 it returns the one-state
    machine that never presses, with a recall of zero and an accuracy of 0.963.

    *Balanced* -- mistakes priced by inverse class frequency -- recovers that
    rule at a recall of 0.833, and costs nothing at all on balanced tasks. But
    it buys structure in skewed noise: given random labels that are 5% ones it
    returns an eight-state machine where the plain objective honestly returns
    one.

    So neither is a prior worth committing to, and the choice is made the way
    every other choice in this repository is made -- on evidence the candidate
    did not fit to. Balanced accuracy is the judge because it is the measure
    that is not fooled by either failure: chance is 0.5 for the never-press
    machine and for the noise-fitted one alike.
    """

    validation = tuple(validation_traces)
    if not validation:
        raise ValueError("choosing between fits needs held-out evidence")
    candidates = [
        induce_noise_tolerant(fit_traces, balanced=balanced, **kwargs)
        for balanced in (False, True)
    ]
    scored = [
        (balanced_accuracy(fit.machine, validation), fit)
        for fit in candidates
        if fit is not None
    ]
    if not scored:
        return None
    # Ties go to the smaller machine: same evidence, shorter description.
    best = max(scored, key=lambda item: (item[0], -item[1].machine.state_count))
    return best[1]
