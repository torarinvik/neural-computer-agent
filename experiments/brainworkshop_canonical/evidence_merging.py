"""Evidence-driven state merging, because the agent cannot ask questions.

The induced-program record ended by pointing at Angluin's L*: stop watching,
start asking. Checking the environment first says that is not available here.
`RenderedBrainWorkshopVerifier` generates its whole symbol stream from a seed
in its constructor, and `score` reads the stream at the current position
without ever consulting the action. The agent chooses presses, not stimuli, so
there is no membership query to make -- and since inverting the reward reveals
the target label at every eligible step whatever the agent pressed, its actions
carry no information-gathering value either. Active learning has nothing to
grip on in this environment.

That leaves the other explanation for why identification stopped at four
states. The exact search in `identification_ceiling` gave up under budget, and
a five-state rule over four symbols has twenty transition cells visited about
twenty times each in a single 448-step episode: the machine is heavily
over-determined by the data. So the failure looks computational rather than
informational, and that is a falsifiable claim -- a better solver on the same
evidence should identify rules the exact search could not.

This module is that solver. EDSM (Lang, Pearlmutter and Price, 1998; the
Abbadingo winner) differs from the RPNI already tried in one decisive way.
RPNI takes the *first* consistent merge, which on a near-chain is an
evidence-free guess that poisons everything downstream -- measured at 31 states
for a 2-state rule. EDSM scores every candidate merge by how much agreement the
determinizing cascade actually produced, and takes the *best* one, so a merge
is made when the data supports it rather than when nothing yet contradicts it.

The red-blue frontier keeps that honest: red states are settled, blue states
are their unmerged children, and a blue state with no consistent merge is
promoted to red rather than forced into one.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rule_automata import RuleAutomaton, minimize

EVIDENCE_SCHEMA = "neural-computer.evidence-merging.v1"


@dataclass
class _Tree:
    """A prefix tree over episodes: transitions, outputs, and node count."""

    child: dict[tuple[int, int], int]
    output: dict[tuple[int, int], int]
    nodes: int
    symbol_count: int


def build_tree(traces) -> _Tree:
    """Prefix tree acceptor over one or more episodes.

    Every episode starts at the root because every episode restarts the
    machine in its initial state, so episodes share their short prefixes and
    the tree branches where the data branches.
    """

    child: dict[tuple[int, int], int] = {}
    output: dict[tuple[int, int], int] = {}
    nodes = 1
    symbol_count = 0
    for trace in traces:
        symbol_count = max(symbol_count, trace.symbol_count)
        node = 0
        for position, symbol in enumerate(trace.symbols):
            key = (node, symbol)
            if key not in child:
                child[key] = nodes
                nodes += 1
            if trace.eligible[position]:
                recorded = output.get(key)
                if recorded is None:
                    output[key] = trace.outputs[position]
                elif recorded != trace.outputs[position]:
                    raise ValueError(
                        "the same prefix produced two different outputs; the "
                        "target is not a deterministic function of history"
                    )
            node = child[key]
    if not child:
        raise ValueError("an empty prefix tree cannot be merged")
    return _Tree(child=child, output=output, nodes=nodes, symbol_count=symbol_count)


@dataclass
class _Partition:
    """The current quotient: class-level transitions and outputs."""

    parent: list[int]
    child: dict[tuple[int, int], int]
    output: dict[tuple[int, int], int]

    def find(self, node: int) -> int:
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def copy(self) -> _Partition:
        return _Partition(list(self.parent), dict(self.child), dict(self.output))


def _fold(partition: _Partition, red: int, blue: int, symbol_count: int) -> int | None:
    """Fold `blue` into `red` in place; evidence, or None if inconsistent.

    Every entry belonging to the absorbed class is moved onto the surviving
    one, so a later comparison sees the *class's* transitions rather than one
    representative's. Getting this wrong is what made the first attempt merge
    everything into a single state: conflicts were invisible because only the
    two representatives were consulted.

    Evidence counts transitions where both classes already had a recorded
    output and those outputs agreed -- the support the data actually offers
    for this merge, as opposed to the absence of a contradiction.
    """

    evidence = 0
    stack = [(red, blue)]
    while stack:
        first, second = stack.pop()
        first, second = partition.find(first), partition.find(second)
        if first == second:
            continue
        partition.parent[second] = first
        for symbol in range(symbol_count):
            absorbed_out = partition.output.pop((second, symbol), None)
            surviving_out = partition.output.get((first, symbol))
            if absorbed_out is not None:
                if surviving_out is None:
                    partition.output[(first, symbol)] = absorbed_out
                elif surviving_out != absorbed_out:
                    return None
                else:
                    evidence += 1
            absorbed = partition.child.pop((second, symbol), None)
            surviving = partition.child.get((first, symbol))
            if absorbed is None:
                continue
            if surviving is None:
                partition.child[(first, symbol)] = absorbed
                continue
            stack.append((surviving, absorbed))
    return evidence


def infer_by_merging(traces, *, max_states: int = 64) -> RuleAutomaton | None:
    """Evidence-driven state merging over a prefix tree of episodes."""

    episodes = tuple(traces)
    if not episodes:
        return None
    tree = build_tree(episodes)
    partition = _Partition(
        parent=list(range(tree.nodes)),
        child=dict(tree.child),
        output=dict(tree.output),
    )
    red = [0]
    while True:
        settled = {partition.find(item) for item in red}
        blue: list[int] = []
        for state in settled:
            for symbol in range(tree.symbol_count):
                target = partition.child.get((state, symbol))
                if target is None:
                    continue
                root = partition.find(target)
                if root not in settled and root not in blue:
                    blue.append(root)
        if not blue:
            break
        best: tuple[int, _Partition] | None = None
        unmergeable: list[int] = []
        for candidate in blue:
            merged_anywhere = False
            for state in sorted(settled):
                trial = partition.copy()
                evidence = _fold(trial, state, candidate, tree.symbol_count)
                if evidence is None:
                    continue
                merged_anywhere = True
                if best is None or evidence > best[0]:
                    best = (evidence, trial)
            if not merged_anywhere:
                unmergeable.append(candidate)
        if unmergeable:
            # A blue state that fits nowhere is a genuine new state. Promoting
            # it is the whole reason for the red-blue frontier: it is never
            # forced into a merge the evidence rejects.
            red.append(unmergeable[0])
            if len(red) > max_states:
                return None
            continue
        if best is None:
            break
        partition = best[1]
    return _to_automaton(tree, partition, red)


def _to_automaton(
    tree: _Tree, partition: _Partition, red: list[int]
) -> RuleAutomaton | None:
    """Number the surviving classes and complete unvisited cells.

    A cell no episode exercised is unconstrained; the least-committal
    completion is to stay put and emit nothing.
    """

    roots: list[int] = []
    for state in red:
        root = partition.find(state)
        if root not in roots:
            roots.append(root)
    start = partition.find(0)
    if start not in roots:
        roots.insert(0, start)
    order = [start] + [root for root in roots if root != start]
    index = {root: position for position, root in enumerate(order)}
    transitions = []
    outputs = []
    for root in order:
        row = []
        emission = []
        for symbol in range(tree.symbol_count):
            target = partition.child.get((root, symbol))
            recorded = partition.output.get((root, symbol))
            resolved = None if target is None else partition.find(target)
            row.append(index.get(resolved, index[root]))
            emission.append(0 if recorded is None else int(recorded))
        transitions.append(tuple(row))
        outputs.append(tuple(emission))
    return minimize(
        RuleAutomaton(
            symbol_count=tree.symbol_count,
            transitions=tuple(transitions),
            outputs=tuple(outputs),
        )
    )
