"""Notice that the hypothesis class is too small, then leave it.

The adversarial audit left two flaws open, and one of them is worse than it
looks. Given a running-majority rule -- press while one symbol has occurred
more often than another -- the inducer returns a twelve-state machine that
scores 0.518 on held-out evidence. It does not abstain, it does not warn, it
returns a confident wrong answer. Majority is not finite-state at any size, so
no amount of search or data inside that class will ever help, and nothing in
the stack can tell.

That is the corner-painting failure in its purest form: an agent whose
representation cannot express the answer, with no signal that this is the
problem. Everything else in this session -- better proposers, factorisation,
segmentation -- is search *within* a fixed class. None of it helps here.

Two pieces fix it, and the first is the one that has to be trusted.

**A competence verdict from the learning curve.** Fit at each rung of an
evidence ladder and watch two trajectories: the size of the hypothesis, and
its error on episodes it was not fitted on. Inside the class those settle --
size stops growing, error goes to zero. Outside it they diverge: every extra
episode buys another state and the error stalls. That difference is visible
without knowing the answer, which is what makes it usable by an agent rather
than by an experimenter.

**An escalation when the verdict says so.** The class is widened by exactly
one construct -- a single integer counter, incremented by -1, 0 or +1 per
transition, whose *sign* the output may read. This is deliberately the
smallest step that leaves finite-state behind rather than a leap to a
universal machine, because a class that can express everything can also fit
anything, and the whole point of the verdict is to stop that.

The verdict is calibrated adversarially in `test_class_escalation.py`, and the
one number that matters is how often it says IDENTIFIED when it is wrong.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise

from .identification_ceiling import infer_machine
from .rule_automata import RuleAutomaton

ESCALATION_SCHEMA = "neural-computer.class-escalation.v1"
LADDER = (7, 14, 28, 56, 112)
COUNTER_STATES = 2
COUNTER_NODE_BUDGET = 400_000


class Verdict(str, Enum):
    """What the learner believes about its own hypothesis."""

    IDENTIFIED = "identified"
    NEED_MORE_DATA = "need_more_data"
    CLASS_INADEQUATE = "class_inadequate"


@dataclass(frozen=True)
class Competence:
    """A verdict and the trajectories it was read from."""

    verdict: Verdict
    sizes: tuple[int | None, ...]
    errors: tuple[float | None, ...]
    rungs: tuple[int, ...]
    machine: RuleAutomaton | None

    def payload(self) -> dict[str, object]:
        return {
            "schema": ESCALATION_SCHEMA,
            "verdict": self.verdict.value,
            "rungs": list(self.rungs),
            "sizes": list(self.sizes),
            "errors": list(self.errors),
        }


def _error(machine: RuleAutomaton, traces) -> float:
    wrong = 0
    trials = 0
    for trace in traces:
        predicted = machine.expected(list(trace.symbols))
        for position, flag in enumerate(trace.eligible):
            if not flag:
                continue
            trials += 1
            wrong += int(predicted[position] != trace.outputs[position])
    return wrong / trials if trials else 1.0


def assess(
    fit_traces,
    validation_traces,
    *,
    ladder: tuple[int, ...] = LADDER,
    tolerance: float = 0.0,
) -> Competence:
    """Climb the ladder and read the two trajectories.

    `tolerance` is the error rate below which a fit counts as explaining the
    evidence. It is zero for clean feedback; a caller that knows its labels are
    noisy should raise it rather than have this pretend the noise is structure.
    """

    episodes = tuple(fit_traces)
    checks = tuple(validation_traces)
    sizes: list[int | None] = []
    errors: list[float | None] = []
    used: list[int] = []
    best: RuleAutomaton | None = None
    for rung in ladder:
        if rung > len(episodes):
            break
        used.append(rung)
        machine = infer_machine(episodes[:rung])
        if machine is None:
            sizes.append(None)
            errors.append(None)
            continue
        sizes.append(machine.state_count)
        errors.append(_error(machine, checks))
        best = machine
    return Competence(
        verdict=_verdict(tuple(sizes), tuple(errors), tolerance),
        sizes=tuple(sizes),
        errors=tuple(errors),
        rungs=tuple(used),
        machine=best,
    )


def _verdict(
    sizes: tuple[int | None, ...],
    errors: tuple[float | None, ...],
    tolerance: float,
) -> Verdict:
    """Read a verdict off the trajectories, using no knowledge of the answer.

    The rules are deliberately blunt, because a subtle rule tuned on the
    probes it is validated against measures nothing.

    - Settled and right: the last fit explains held-out evidence. IDENTIFIED.
    - Growing and still wrong: the hypothesis needed more states at every rung
      and never explained the evidence, which is what a target outside the
      class looks like from inside it. CLASS_INADEQUATE.
    - Anything else: the ladder has not settled. NEED_MORE_DATA.
    """

    seen = [
        (size, error)
        for size, error in zip(sizes, errors)
        if size is not None and error is not None
    ]
    if not seen:
        return Verdict.NEED_MORE_DATA
    if seen[-1][1] <= tolerance:
        return Verdict.IDENTIFIED
    if len(seen) < 3:
        return Verdict.NEED_MORE_DATA
    growing = all(
        later[0] > earlier[0] for earlier, later in pairwise(seen)
    )
    never_explained = all(error > tolerance for _, error in seen)
    if growing and never_explained:
        return Verdict.CLASS_INADEQUATE
    return Verdict.NEED_MORE_DATA


@dataclass(frozen=True)
class CounterMachine:
    """Finite control plus one integer counter whose sign the output reads.

    The smallest useful step past finite state. `increments` moves the counter
    on each transition and `outputs` is indexed by the counter's sign *after*
    the move, so a rule like "press while more As than Bs" is expressible with
    a single control state.
    """

    symbol_count: int
    transitions: tuple[tuple[int, ...], ...]
    increments: tuple[tuple[int, ...], ...]
    outputs: tuple[tuple[tuple[int, int, int], ...], ...]

    @property
    def state_count(self) -> int:
        return len(self.transitions)

    def expected(self, symbols) -> list[int]:
        state = 0
        counter = 0
        produced: list[int] = []
        for symbol in symbols:
            index = int(symbol)
            counter += int(self.increments[state][index])
            sign = 0 if counter == 0 else (1 if counter > 0 else 2)
            produced.append(int(self.outputs[state][index][sign]))
            state = int(self.transitions[state][index])
        return produced


def induce_counter_machine(
    traces,
    *,
    states: int = COUNTER_STATES,
    node_budget: int = COUNTER_NODE_BUDGET,
) -> CounterMachine | None:
    """Smallest counter machine consistent with the traces, or None.

    Same depth-first assignment as the finite-state search, with two extra
    cells per transition: how the counter moves, and what to emit for each
    sign it can land on. Increments are tried in the order 0, +1, -1 so a
    hypothesis that ignores the counter is preferred to one that uses it --
    the class contains finite-state machines and should not reach past them
    without cause.
    """

    episodes = tuple(traces)
    if not episodes:
        return None
    steps = [
        (index, position)
        for index, trace in enumerate(episodes)
        for position in range(len(trace.symbols))
    ]
    if not steps:
        return None
    symbol_count = max(trace.symbol_count for trace in episodes)
    for size in range(1, states + 1):
        found = _search_counter(episodes, steps, size, symbol_count, node_budget)
        if found is not None:
            return found
    return None


def _search_counter(episodes, steps, states, symbol_count, node_budget):
    transitions: dict[tuple[int, int], int] = {}
    increments: dict[tuple[int, int], int] = {}
    outputs: dict[tuple[int, int, int], int] = {}
    used = [1]
    visited = [0]

    def walk(step: int, state: int, counter: int) -> bool:
        if step == len(steps):
            return True
        visited[0] += 1
        if visited[0] > node_budget:
            raise _Budget
        index, position = steps[step]
        trace = episodes[index]
        symbol = int(trace.symbols[position])
        want = trace.outputs[position] if trace.eligible[position] else None
        restart = step + 1 < len(steps) and steps[step + 1][1] == 0
        moves = (
            [increments[(state, symbol)]]
            if (state, symbol) in increments
            else [0, 1, -1]
        )
        for move in moves:
            fresh_move = (state, symbol) not in increments
            increments[(state, symbol)] = move
            value = counter + move
            sign = 0 if value == 0 else (1 if value > 0 else 2)
            cell = (state, symbol, sign)
            recorded = outputs.get(cell)
            if want is not None:
                if recorded is None:
                    outputs[cell] = want
                elif recorded != want:
                    if fresh_move:
                        del increments[(state, symbol)]
                    continue
            fresh_cell = want is not None and recorded is None
            targets = (
                [transitions[(state, symbol)]]
                if (state, symbol) in transitions
                else list(range(used[0])) + ([used[0]] if used[0] < states else [])
            )
            for target in targets:
                fresh_target = (state, symbol) not in transitions
                transitions[(state, symbol)] = target
                grew = fresh_target and target == used[0]
                if grew:
                    used[0] += 1
                if walk(step + 1, 0 if restart else target, 0 if restart else value):
                    return True
                if grew:
                    used[0] -= 1
                if fresh_target:
                    del transitions[(state, symbol)]
            if fresh_cell:
                del outputs[cell]
            if fresh_move:
                del increments[(state, symbol)]
        return False

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, len(steps) * 2 + 1000))
    try:
        if walk(0, 0, 0):
            return _counter_from_cells(transitions, increments, outputs, used[0], symbol_count)
    except _Budget:
        return None
    finally:
        sys.setrecursionlimit(limit)
    return None


class _Budget(Exception):
    """Search for this size cost more than it is worth."""


def _counter_from_cells(transitions, increments, outputs, used, symbol_count):
    return CounterMachine(
        symbol_count=symbol_count,
        transitions=tuple(
            tuple(
                int(transitions.get((state, symbol), state))
                for symbol in range(symbol_count)
            )
            for state in range(used)
        ),
        increments=tuple(
            tuple(
                int(increments.get((state, symbol), 0))
                for symbol in range(symbol_count)
            )
            for state in range(used)
        ),
        outputs=tuple(
            tuple(
                tuple(
                    int(outputs.get((state, symbol, sign), 0)) for sign in range(3)
                )
                for symbol in range(symbol_count)
            )
            for state in range(used)
        ),
    )


def escalate(fit_traces, validation_traces, *, tolerance: float = 0.0):
    """Assess the finite-state class; widen it only if the verdict says to.

    Returns the competence report and, when the class was found wanting and a
    counter machine explains what it could not, that machine. The order
    matters: escalating first would let the larger class fit anything, and
    escalating never is the flaw this module exists to fix.
    """

    report = assess(fit_traces, validation_traces, tolerance=tolerance)
    if report.verdict is Verdict.IDENTIFIED:
        return report, None
    # `CLASS_INADEQUATE` is the confident signal, and on its own it is far too
    # rare to be useful. Measured over four running-majority rules, it fires
    # at sixteen-step episodes and never at forty-eight or a hundred and
    # twenty-eight, because the finite-state search stops returning anything
    # at all and a missing trajectory cannot be read. Abstention and
    # inadequacy are genuinely indistinguishable from the inside.
    #
    # So the trigger is the weaker, honest one: the finite-state class did not
    # explain the evidence, whatever the reason. What keeps that safe is not
    # the trigger but the gate below -- a wider hypothesis is accepted only if
    # it predicts episodes it was never fitted on.
    machine = induce_counter_machine(fit_traces)
    if machine is None:
        return report, None
    wrong = 0
    trials = 0
    for trace in validation_traces:
        predicted = machine.expected(list(trace.symbols))
        for position, flag in enumerate(trace.eligible):
            if not flag:
                continue
            trials += 1
            wrong += int(predicted[position] != trace.outputs[position])
    if trials and wrong / trials <= tolerance:
        return report, machine
    return report, None
