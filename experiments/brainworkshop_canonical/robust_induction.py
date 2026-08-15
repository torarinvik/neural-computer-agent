"""Induction that survives a wrong label, and says how wrong it still is.

The exact inducer is brittle in the worst way. `infer_machine` backtracks on
any contradiction, `fit_output_table` returns `None` on one conflicting cell,
and `build_tree` raises outright. Measured against a probe at 0.5% label noise
-- one mis-scored press in two hundred -- the stack does not degrade, it throws
`ValueError: the same prefix produced two different outputs`. An agent reading
rewards from a real verifier would crash on its first bad tick.

That brittleness is not incidental to the method, it *is* the method: exact
consistency is what makes the minimal-machine search sound. So the fix is a
different criterion rather than a wider try block.

Two changes make it robust.

**Cells hold counts, not a label.** A prefix-tree cell records how often each
output followed, so a single flip is a minority rather than a contradiction.

**States merge on a statistical test, not on equality.** Two states are
compatible when their output proportions agree within a Hoeffding bound, which
is the criterion Carrasco and Oncina's ALERGIA (1994) uses for stochastic
automata, transplanted onto Mealy outputs. The bound loosens as counts shrink,
so thin evidence does not force a split and a genuine difference still does.

Selection is by **description length**, and that matters more than the merge
rule. Under noise, a machine that fits every label exists and is enormous;
scoring `bits(machine) + bits(the labels it gets wrong)` prices that machine
out and stops the search at the true one. It is also what lets this report an
*honest* error rate instead of a confident wrong answer, which is the failure
mode that matters when the target is not finite-state at all -- a machine that
cannot represent counting will now say so with a residual error that does not
fall as evidence grows, rather than returning a machine that looks fine.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2, sqrt

from .rule_automata import RuleAutomaton, minimize

ROBUST_SCHEMA = "neural-computer.robust-induction.v1"
DEFAULT_ALPHA = 0.05
MAX_STATES = 24


@dataclass(frozen=True)
class RobustHypothesis:
    """A machine, what it costs to describe, and what it still gets wrong."""

    machine: RuleAutomaton
    error_rate: float
    description_bits: float
    error_bits: float
    trials: int

    @property
    def total_bits(self) -> float:
        return self.description_bits + self.error_bits


def _entropy(rate: float) -> float:
    if rate <= 0.0 or rate >= 1.0:
        return 0.0
    return -(rate * log2(rate) + (1.0 - rate) * log2(1.0 - rate))


def description_bits(machine: RuleAutomaton) -> float:
    """Bits to write the machine down: a target and an output per cell."""

    cells = machine.state_count * machine.symbol_count
    return cells * (log2(max(machine.state_count, 2)) + 1.0)


def error_bits(errors: int, trials: int) -> float:
    """Bits to list the exceptions, priced by their own entropy.

    A machine that fits noise exactly pays for every extra state; a machine
    that ignores a real pattern pays for every mistake. Adding the two is what
    stops both.
    """

    if trials <= 0:
        return 0.0
    return trials * _entropy(errors / trials)


def _hoeffding_compatible(
    left_ones: int, left_total: int, right_ones: int, right_total: int, alpha: float
) -> bool:
    """Whether two output proportions differ by less than sampling noise."""

    if left_total == 0 or right_total == 0:
        return True
    difference = abs(left_ones / left_total - right_ones / right_total)
    bound = sqrt(0.5 * log2(2.0 / alpha)) * (
        1.0 / sqrt(left_total) + 1.0 / sqrt(right_total)
    )
    return difference <= bound


@dataclass
class _Counts:
    """Prefix tree whose cells count outputs instead of asserting one."""

    child: dict[tuple[int, int], int]
    ones: dict[tuple[int, int], int]
    total: dict[tuple[int, int], int]
    nodes: int
    symbol_count: int


def _build(traces) -> _Counts:
    child: dict[tuple[int, int], int] = {}
    ones: dict[tuple[int, int], int] = {}
    total: dict[tuple[int, int], int] = {}
    nodes = 1
    symbol_count = 0
    for trace in traces:
        symbol_count = max(symbol_count, trace.symbol_count)
        node = 0
        for position, symbol in enumerate(trace.symbols):
            key = (node, int(symbol))
            if key not in child:
                child[key] = nodes
                nodes += 1
            if trace.eligible[position]:
                total[key] = total.get(key, 0) + 1
                ones[key] = ones.get(key, 0) + int(trace.outputs[position])
            node = child[key]
    return _Counts(child, ones, total, nodes, symbol_count)


def _quotient_machine(
    counts: _Counts, parent: list[int]
) -> tuple[RuleAutomaton, dict[int, int]]:
    """Collapse the tree by the current partition; outputs by majority."""

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    roots: list[int] = []
    for node in range(counts.nodes):
        root = find(node)
        if root not in roots:
            roots.append(root)
    start = find(0)
    order = [start] + [root for root in roots if root != start]
    index = {root: position for position, root in enumerate(order)}
    merged_ones: dict[tuple[int, int], int] = {}
    merged_total: dict[tuple[int, int], int] = {}
    merged_child: dict[tuple[int, int], int] = {}
    for (node, symbol), target in counts.child.items():
        key = (find(node), symbol)
        merged_child.setdefault(key, find(target))
        merged_ones[key] = merged_ones.get(key, 0) + counts.ones.get((node, symbol), 0)
        merged_total[key] = merged_total.get(key, 0) + counts.total.get(
            (node, symbol), 0
        )
    transitions = []
    outputs = []
    for root in order:
        row = []
        emission = []
        for symbol in range(counts.symbol_count):
            target = merged_child.get((root, symbol))
            row.append(index[target] if target in index else index[root])
            seen = merged_total.get((root, symbol), 0)
            positive = merged_ones.get((root, symbol), 0)
            emission.append(int(seen > 0 and positive * 2 > seen))
        transitions.append(tuple(row))
        outputs.append(tuple(emission))
    return (
        RuleAutomaton(
            symbol_count=counts.symbol_count,
            transitions=tuple(transitions),
            outputs=tuple(outputs),
        ),
        index,
    )


def _errors(machine: RuleAutomaton, traces) -> tuple[int, int]:
    """Labels the machine disagrees with, and labels it was shown."""

    wrong = 0
    trials = 0
    for trace in traces:
        predicted = machine.expected(list(trace.symbols))
        for position, flag in enumerate(trace.eligible):
            if not flag:
                continue
            trials += 1
            wrong += int(predicted[position] != trace.outputs[position])
    return wrong, trials


@dataclass
class _State:
    """Class-level counts, so a comparison sees a class not a representative."""

    parent: list[int]
    child: dict[tuple[int, int], int]
    ones: dict[tuple[int, int], int]
    total: dict[tuple[int, int], int]

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def copy(self) -> _State:
        return _State(
            list(self.parent), dict(self.child), dict(self.ones), dict(self.total)
        )


def _fold(
    state: _State, red: int, blue: int, symbol_count: int, alpha: float
) -> int | None:
    """Fold `blue` into `red`, propagating; evidence, or None if incompatible.

    The propagation is what the first version of this lacked. Merging two
    states forces their successors together, and without that the compatibility
    test only ever compared one step of behaviour -- which merged everything
    into two states and scored worse at zero noise than the exact method it was
    meant to replace.
    """

    evidence = 0
    stack = [(red, blue)]
    while stack:
        first, second = stack.pop()
        first, second = state.find(first), state.find(second)
        if first == second:
            continue
        for symbol in range(symbol_count):
            left_key = (first, symbol)
            right_key = (second, symbol)
            if not _hoeffding_compatible(
                state.ones.get(left_key, 0),
                state.total.get(left_key, 0),
                state.ones.get(right_key, 0),
                state.total.get(right_key, 0),
                alpha,
            ):
                return None
        state.parent[second] = first
        for symbol in range(symbol_count):
            left_key = (first, symbol)
            right_key = (second, symbol)
            moved_total = state.total.pop(right_key, 0)
            moved_ones = state.ones.pop(right_key, 0)
            if moved_total and state.total.get(left_key, 0):
                evidence += min(moved_total, state.total[left_key])
            if moved_total:
                state.total[left_key] = state.total.get(left_key, 0) + moved_total
                state.ones[left_key] = state.ones.get(left_key, 0) + moved_ones
            absorbed = state.child.pop(right_key, None)
            if absorbed is None:
                continue
            surviving = state.child.get(left_key)
            if surviving is None:
                state.child[left_key] = absorbed
                continue
            stack.append((surviving, absorbed))
    return evidence


def _machine_of(state: _State, counts: _Counts) -> RuleAutomaton:
    """Read the current partition out as a machine; outputs by majority."""

    roots: list[int] = []
    for node in range(counts.nodes):
        root = state.find(node)
        if root not in roots:
            roots.append(root)
    start = state.find(0)
    order = [start] + [root for root in roots if root != start]
    index = {root: position for position, root in enumerate(order)}
    transitions = []
    outputs = []
    for root in order:
        row = []
        emission = []
        for symbol in range(counts.symbol_count):
            target = state.child.get((root, symbol))
            resolved = None if target is None else state.find(target)
            row.append(index.get(resolved, index[root]))
            seen = state.total.get((root, symbol), 0)
            positive = state.ones.get((root, symbol), 0)
            emission.append(int(seen > 0 and positive * 2 > seen))
        transitions.append(tuple(row))
        outputs.append(tuple(emission))
    return RuleAutomaton(
        symbol_count=counts.symbol_count,
        transitions=tuple(transitions),
        outputs=tuple(outputs),
    )


def induce_robust(
    traces,
    *,
    alpha: float = DEFAULT_ALPHA,
    max_states: int = MAX_STATES,
    node_budget: int = 200_000,
) -> RobustHypothesis | None:
    """Smallest machine that explains the labels *apart from a few*.

    Statistical state merging was tried first and lost badly: on a four-state
    threshold rule at zero noise it returned one state, because the Hoeffding
    bound is vacuous at the counts a short-episode prefix tree produces. That
    version is kept below as `induce_by_merging` and measured, rather than
    quietly deleted.

    What works is the exact search with a *violation budget*. A disagreement
    stops being a contradiction and becomes something the hypothesis is allowed
    to spend, so one mis-scored reward costs a label instead of the whole
    machine. For each state count the budget is grown until a machine appears,
    which makes the recorded violation count close to minimal for that size,
    and description length then chooses across sizes.

    Description length is what stops both failure modes. A machine that fits
    every noisy label exists and is enormous, and pays for its states; a machine
    that ignores a real pattern pays for its mistakes. Adding the two prices
    finds the knee.

    Outputs are normalised before scoring. A machine and its inverse have the
    same description length, and the error term is symmetric, so without this
    a hypothesis that is wrong about *everything* scores zero error bits and
    wins -- which is exactly what the first version did on the parity probe.
    """

    from .identification_ceiling import _consistent_machine

    episodes = tuple(traces)
    if not episodes:
        return None
    trials = sum(1 for trace in episodes for flag in trace.eligible if flag)
    if trials < 2:
        return None
    ceiling = max(1, trials // 4)
    best: RobustHypothesis | None = None
    for states in range(1, max_states + 1):
        budget = 0
        found = None
        while budget <= ceiling:
            found = _consistent_machine(
                episodes, states, node_budget, violation_budget=budget
            )
            if found is not None:
                break
            budget = max(1, budget * 2)
            if budget > ceiling:
                break
        if found is None:
            continue
        candidate = _score(found, episodes, trials)
        if best is None or candidate.total_bits < best.total_bits:
            best = candidate
        if candidate.error_rate == 0.0:
            # Nothing larger can beat a machine that explains everything.
            break
    return best


def _score(machine: RuleAutomaton, episodes, trials: int) -> RobustHypothesis:
    """Price a hypothesis, taking the better of it and its inverse."""

    reduced = minimize(machine)
    wrong, _ = _errors(reduced, episodes)
    if wrong * 2 > trials:
        reduced = minimize(
            RuleAutomaton(
                symbol_count=reduced.symbol_count,
                transitions=reduced.transitions,
                outputs=tuple(
                    tuple(1 - int(value) for value in row) for row in reduced.outputs
                ),
            )
        )
        wrong, _ = _errors(reduced, episodes)
    return RobustHypothesis(
        machine=reduced,
        error_rate=wrong / trials if trials else 1.0,
        description_bits=description_bits(reduced),
        error_bits=error_bits(wrong, trials),
        trials=trials,
    )


def induce_by_merging(

    traces,
    *,
    alpha: float = DEFAULT_ALPHA,
    max_states: int = MAX_STATES,
) -> RobustHypothesis | None:
    """Merge on a statistical test; keep whichever partition costs fewest bits.

    Every partition along the merge path is a candidate, not just the last one.
    Description length picks the point where the machine stops explaining and
    starts memorising, which is what makes this stop at the true machine under
    noise instead of growing to fit it.
    """

    episodes = tuple(traces)
    if not episodes:
        return None
    counts = _build(episodes)
    if not counts.child:
        return None
    state = _State(
        parent=list(range(counts.nodes)),
        child=dict(counts.child),
        ones=dict(counts.ones),
        total=dict(counts.total),
    )
    best: RobustHypothesis | None = None

    def consider() -> None:
        nonlocal best
        reduced = minimize(_machine_of(state, counts))
        if reduced.state_count > max_states:
            return
        wrong, trials = _errors(reduced, episodes)
        candidate = _score(reduced, episodes, trials)
        del wrong
        if best is None or candidate.total_bits < best.total_bits:
            best = candidate

    red = [0]
    while True:
        settled = {state.find(item) for item in red}
        blue: list[int] = []
        for item in settled:
            for symbol in range(counts.symbol_count):
                target = state.child.get((item, symbol))
                if target is None:
                    continue
                root = state.find(target)
                if root not in settled and root not in blue:
                    blue.append(root)
        if not blue:
            break
        chosen: tuple[int, _State] | None = None
        unmergeable: list[int] = []
        for candidate in blue:
            merged_anywhere = False
            for item in sorted(settled):
                trial = state.copy()
                evidence = _fold(trial, item, candidate, counts.symbol_count, alpha)
                if evidence is None:
                    continue
                merged_anywhere = True
                if chosen is None or evidence > chosen[0]:
                    chosen = (evidence, trial)
            if not merged_anywhere:
                unmergeable.append(candidate)
        if unmergeable:
            red.append(unmergeable[0])
            if len(red) > max_states * 4:
                break
            consider()
            continue
        if chosen is None:
            break
        state = chosen[1]
        consider()
    consider()
    return best
