"""Learning when a wrong answer only tells you what the answer is not.

Every result in this session rests on one line:

    target[t] = action[t] if reward[t] else 1 - action[t]

Feedback inversion is exact, and it is exact *because there are two actions*.
With two, being wrong names the right answer as surely as being right does, so
a scalar reward is secretly a full supervision signal and everything
downstream -- noise tolerance, the library, composition -- is doing supervised
learning wearing a disguise.

With `k` actions that disguise comes off. A success still names the target. A
failure rules out one of `k` and leaves `k-1` standing. The evidence is
**partial**, most of it is negative, and the amount of it a wrong guess buys
falls as the action set grows. This module is what learning looks like on that
evidence, and it is the first thing in this repository that could not have been
done with a classifier.

Two consequences shape the design.

**The learner must choose what to try.** Under two actions the probe policy is
irrelevant: whatever it plays, the reward names the target. Under `k` it is
not. A policy that always plays the same action learns only which cells that
action is wrong in, and never which action is right. So the probe is uniform
over the action set, and that choice is a real one with a measurable cost.

**The objective generalises, and stays search-free.** Given transitions, each
cell of the output table is still filled by counting rather than searching: a
candidate action `a` disagrees with a success that chose something else, and
with a failure that chose `a`. Minimising over `a` is `argmin(neg[a] - pos[a])`
per cell, which is one pass. At two actions this is exactly the majority rule
the binary fitter already used, so nothing about the earlier results is
disturbed.

A correct machine has **zero** disagreements under this objective, which is
what makes it the right one: it never charges the learner for a failure it
correctly predicted would fail.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .noise_tolerant_induction import MAX_SWEEPS, RESTARTS
from .robust_induction import description_bits, error_bits
from .rule_automata import RuleAutomaton, minimize

CHOICE_SCHEMA = "neural-computer.choice-induction.v1"
MAX_STATES = 8


@dataclass(frozen=True)
class ChoiceTrace:
    """One episode as the agent actually experienced it.

    Not `(symbol, label)`. The agent never sees a label: it sees what it chose
    and whether that was right, and those are different objects the moment
    there is more than one way to be wrong.
    """

    symbols: tuple[int, ...]
    actions: tuple[int, ...]
    rewards: tuple[int, ...]
    eligible: tuple[bool, ...]
    symbol_count: int
    action_count: int

    def validate(self) -> ChoiceTrace:
        length = len(self.symbols)
        if not (
            len(self.actions) == len(self.rewards) == len(self.eligible) == length
        ):
            raise ValueError("a choice trace must be square")
        if self.action_count < 2 or self.symbol_count < 2:
            raise ValueError("a choice trace needs at least two of each")
        if any(not 0 <= s < self.symbol_count for s in self.symbols):
            raise ValueError("a symbol is outside the alphabet")
        if any(not 0 <= a < self.action_count for a in self.actions):
            raise ValueError("an action is outside the protocol")
        if any(r not in (0, 1) for r in self.rewards):
            raise ValueError("a reward is not a scalar outcome")
        return self

    @property
    def resolved(self) -> int:
        """Steps where the target is known outright rather than constrained."""

        return sum(
            1
            for reward, flag in zip(self.rewards, self.eligible, strict=True)
            if flag and reward
        )


@dataclass(frozen=True)
class ChoiceFit:
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
            "schema": CHOICE_SCHEMA,
            "states": self.machine.state_count,
            "actions": self.machine.action_count,
            "disagreements": self.disagreements,
            "trials": self.trials,
            "error_rate": self.error_rate,
            "total_bits": self.total_bits,
        }


def _flatten(traces):
    symbols: list[int] = []
    actions: list[int] = []
    rewards: list[int] = []
    eligible: list[bool] = []
    starts: set[int] = set()
    for trace in traces:
        starts.add(len(symbols))
        symbols.extend(int(v) for v in trace.symbols)
        actions.extend(int(v) for v in trace.actions)
        rewards.extend(int(v) for v in trace.rewards)
        eligible.extend(bool(v) for v in trace.eligible)
    return symbols, actions, rewards, eligible, starts


def _best_outputs(
    transitions, symbols, actions, rewards, eligible, starts, states, symbol_count, action_count
):
    """Fill each cell by counting, not by searching.

    A candidate action `a` in a cell disagrees with a *success* that chose
    something other than `a`, and with a *failure* that chose `a`. Everything
    else is silent about `a`, which is the whole difference from the binary
    case: most of the evidence constrains rather than determines.
    """

    positives = [[[0] * action_count for _ in range(symbol_count)] for _ in range(states)]
    negatives = [[[0] * action_count for _ in range(symbol_count)] for _ in range(states)]
    totals_positive = [[0] * symbol_count for _ in range(states)]
    state = 0
    for index, symbol in enumerate(symbols):
        if index in starts:
            state = 0
        if eligible[index]:
            action = actions[index]
            if rewards[index]:
                positives[state][symbol][action] += 1
                totals_positive[state][symbol] += 1
            else:
                negatives[state][symbol][action] += 1
        state = transitions[state][symbol]

    table: list[list[int]] = []
    disagreements = 0
    for s in range(states):
        row: list[int] = []
        for y in range(symbol_count):
            best = 0
            best_score = None
            for a in range(action_count):
                score = (
                    totals_positive[s][y] - positives[s][y][a]
                ) + negatives[s][y][a]
                if best_score is None or score < best_score:
                    best, best_score = a, score
            row.append(best)
            disagreements += best_score or 0
        table.append(row)
    return table, disagreements


def _climb(
    transitions, symbols, actions, rewards, eligible, starts, states, symbol_count, action_count
):
    table, best = _best_outputs(
        transitions, symbols, actions, rewards, eligible, starts, states, symbol_count, action_count
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
                        actions,
                        rewards,
                        eligible,
                        starts,
                        states,
                        symbol_count,
                        action_count,
                    )
                    if score < best:
                        best, table, current, improved = score, candidate, target, True
                    else:
                        transitions[state][symbol] = current
        if not improved:
            break
    return table, best


def induce_from_choices(
    traces,
    *,
    max_states: int = MAX_STATES,
    restarts: int = RESTARTS,
    seed: int = 0,
) -> ChoiceFit | None:
    """The smallest machine that contradicts the fewest outcomes."""

    episodes = tuple(trace.validate() for trace in traces)
    if not episodes:
        return None
    symbols, actions, rewards, eligible, starts = _flatten(episodes)
    trials = sum(1 for flag in eligible if flag)
    if trials < 2:
        return None
    symbol_count = max(trace.symbol_count for trace in episodes)
    action_count = max(trace.action_count for trace in episodes)
    generator = torch.Generator().manual_seed(int(seed))

    best: ChoiceFit | None = None
    for states in range(1, max_states + 1):
        best_here = None
        for restart in range(restarts):
            if restart == 0:
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
                transitions,
                symbols,
                actions,
                rewards,
                eligible,
                starts,
                states,
                symbol_count,
                action_count,
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
                action_count=action_count,
            )
        )
        candidate = ChoiceFit(
            machine=machine,
            disagreements=score,
            trials=trials,
            description_bits=description_bits(machine),
            error_bits=error_bits(score, trials),
        )
        if best is None or candidate.total_bits < best.total_bits:
            best = candidate
    return best


def agreement(machine: RuleAutomaton, traces) -> tuple[int, int]:
    """Outcomes a machine is consistent with, out of those it is scored on.

    Consistency, not accuracy: a failure the machine also predicted would fail
    is evidence *for* it, and a scorer that ignored that would penalise a
    correct machine for the probe policy's mistakes.
    """

    consistent = trials = 0
    for trace in traces:
        predicted = machine.expected(list(trace.symbols))
        for index, flag in enumerate(trace.eligible):
            if not flag:
                continue
            trials += 1
            chose = int(trace.actions[index])
            if trace.rewards[index]:
                consistent += int(predicted[index] == chose)
            else:
                consistent += int(predicted[index] != chose)
    return consistent, trials


def implied_accuracy(
    consistent: int, trials: int, action_count: int
) -> tuple[float, int]:
    """Turn a consistency rate into the accuracy it implies, and its weight.

    Consistency cannot be tested against a fixed gate, because its floor rises
    with the action set: a machine answering at chance is consistent with 0.500
    of outcomes at two actions and **0.625** at four, so a 0.8 bar means very
    different things at the two ends and would wave through a coin flip.

    Under a uniform probe the relation is exact. A success is informative only
    when the machine agrees; a failure is consistent whenever the machine did
    not name the action that failed, which a *correct* machine never does and a
    wrong one avoids with probability `(k-2)/(k-1)`. Collecting terms,

        c = a + (1 - a)(k - 2) / k        =>        a = (c k - (k - 2)) / 2

    which returns chance at chance for every `k`, and is the identity at two
    actions.

    Inverting multiplies the noise by `k/2`, so the second return value is the
    trial count that carries the same information as this estimate does -- and
    it is what any test of the estimate has to be given, rather than the raw
    count, which would overstate the evidence fourfold at four actions.
    """

    if trials <= 0:
        return 0.0, 0
    if action_count < 2:
        raise ValueError("an implied accuracy needs at least two actions")
    rate = consistent / trials
    accuracy = (rate * action_count - (action_count - 2)) / 2.0
    accuracy = min(1.0, max(0.0, accuracy))
    inflation = (action_count / 2.0) ** 2
    return accuracy, max(1, int(trials / inflation))
