"""Find a machine's parts, instead of enumerating products of whole solutions.

The composition record got a 6x saving by trying every product of two library
files under three combiners. That works at nine files and is `L^2 x combiners`
at scale, the combiners are an experimenter's list, and the library stores
whole solutions rather than anything reusable. It is a demonstration that
composition pays, not a mechanism for finding it.

There is an exact one, and it is old. Hartmanis and Stearns showed in 1966
that a sequential machine decomposes into parallel components precisely when
its state set admits *closed partitions*: a partition is closed when knowing
which block the machine is in is enough to know which block it moves to, for
every input. Two closed partitions whose blocks jointly pin down the state --
their meet is the identity -- are a parallel decomposition. No search over
candidate products is involved; the partitions are computed from the
transition function.

That changes what a library can hold. Today it holds solved tasks and finds
accidental factorisations among them. Factoring first means it holds *parts*,
and a later task that shares a part reuses it directly.

Two details matter for using this on Mealy machines rather than the state
machines the theory is usually stated for.

The components carry transitions only. A closed partition guarantees the next
*block* is determined, and says nothing about outputs, so a factor is a
state-computer and the output is recovered from the pair of blocks and the
symbol. That is more general than the boolean combiners the previous record
enumerated, which are one particular output table out of many.

And the output table is *fitted*, not searched. Given two factors, every
observation says what one cell of the table must be, so filling it is linear
in the evidence rather than exponential in the library. That is the whole
saving, and `reconstruct` checks it by rebuilding the original machine and
comparing behaviour rather than trusting the algebra.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from .rule_automata import RuleAutomaton, minimize

FACTORIZATION_SCHEMA = "neural-computer.machine-factorization.v1"


@dataclass(frozen=True)
class Partition:
    """A partition of a machine's states, as a block index per state."""

    blocks: tuple[int, ...]

    @property
    def block_count(self) -> int:
        return len(set(self.blocks))

    @property
    def is_identity(self) -> bool:
        """Every state alone: the partition that distinguishes everything."""

        return self.block_count == len(self.blocks)

    @property
    def is_trivial(self) -> bool:
        """One block: the partition that distinguishes nothing."""

        return self.block_count == 1

    def meet(self, other: Partition) -> Partition:
        """States together only where both partitions put them together."""

        if len(self.blocks) != len(other.blocks):
            raise ValueError("partitions must cover the same state set")
        seen: dict[tuple[int, int], int] = {}
        merged: list[int] = []
        for pair in zip(self.blocks, other.blocks):
            if pair not in seen:
                seen[pair] = len(seen)
            merged.append(seen[pair])
        return Partition(tuple(merged))

    def canonical(self) -> Partition:
        """Renumber blocks by first appearance so equal partitions compare."""

        seen: dict[int, int] = {}
        renumbered: list[int] = []
        for block in self.blocks:
            if block not in seen:
                seen[block] = len(seen)
            renumbered.append(seen[block])
        return Partition(tuple(renumbered))


def _closure(machine: RuleAutomaton, parent: list[int]) -> list[int]:
    """Merge states until the partition is closed under every transition."""

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> bool:
        left, right = find(left), find(right)
        if left == right:
            return False
        parent[max(left, right)] = min(left, right)
        return True

    changed = True
    while changed:
        changed = False
        groups: dict[int, list[int]] = {}
        for state in range(machine.state_count):
            groups.setdefault(find(state), []).append(state)
        for members in groups.values():
            if len(members) < 2:
                continue
            for symbol in range(machine.symbol_count):
                targets = [
                    int(machine.transitions[state][symbol]) for state in members
                ]
                first = targets[0]
                for target in targets[1:]:
                    if union(first, target):
                        changed = True
    return [find(state) for state in range(machine.state_count)]


def smallest_closed_partition(
    machine: RuleAutomaton, left: int, right: int
) -> Partition:
    """The coarsest-information closed partition that identifies two states.

    Merging two states forces their successors together, and so on until it
    settles. Every closed partition is a join of these, which is why they are
    enough to generate the lattice.
    """

    parent = list(range(machine.state_count))
    parent[max(left, right)] = min(left, right)
    return Partition(tuple(_closure(machine, parent))).canonical()


LATTICE_STATE_LIMIT = 12


def closed_partitions(
    machine: RuleAutomaton, *, state_limit: int = LATTICE_STATE_LIMIT
) -> tuple[Partition, ...]:
    """Every closed partition reachable by joining the generating ones.

    Closing the generating set under join costs a transition closure per pair,
    which is fine for the machine sizes this work produces and quadratic in a
    set that grows with the state count. Past `state_limit` the generators are
    returned without the join closure: still sound, since every one of them is
    closed, and merely less complete.
    """

    machine.validate()
    generated: dict[tuple[int, ...], Partition] = {}
    for left, right in combinations(range(machine.state_count), 2):
        partition = smallest_closed_partition(machine, left, right)
        generated[partition.blocks] = partition
    if machine.state_count > state_limit:
        return tuple(generated.values())
    # Joins of closed partitions are closed; iterate to a fixed point.
    frontier = list(generated.values())
    while frontier:
        current = frontier.pop()
        for other in list(generated.values()):
            parent = list(range(machine.state_count))

            def union(a: int, b: int, table: list[int] = parent) -> None:
                while table[a] != a:
                    a = table[a]
                while table[b] != b:
                    b = table[b]
                table[max(a, b)] = min(a, b)

            for state in range(machine.state_count):
                for peer in range(state + 1, machine.state_count):
                    if (
                        current.blocks[state] == current.blocks[peer]
                        or other.blocks[state] == other.blocks[peer]
                    ):
                        union(state, peer)
            joined = Partition(tuple(_closure(machine, parent))).canonical()
            if joined.blocks not in generated:
                generated[joined.blocks] = joined
                frontier.append(joined)
    return tuple(generated.values())


@dataclass(frozen=True)
class Factorization:
    """Two state-computers and the output table that reads their pair."""

    left: RuleAutomaton
    right: RuleAutomaton
    left_blocks: tuple[int, ...]
    right_blocks: tuple[int, ...]
    outputs: dict[tuple[int, int, int], int]

    def predict(self, symbols) -> list[int]:
        """Run both components and read the table. Missing cells emit zero."""

        first = 0
        second = 0
        produced: list[int] = []
        for symbol in symbols:
            produced.append(self.outputs.get((first, second, int(symbol)), 0))
            first = int(self.left.transitions[first][int(symbol)])
            second = int(self.right.transitions[second][int(symbol)])
        return produced

    @property
    def part_sizes(self) -> tuple[int, int]:
        return self.left.state_count, self.right.state_count


def _quotient(machine: RuleAutomaton, partition: Partition) -> RuleAutomaton:
    """The component that tracks only which block the machine is in.

    Outputs are zeroed: a component computes state, and the output lives in
    the table over the pair. Closure is what makes the transitions
    well-defined, so any representative of a block gives the same answer.
    """

    blocks = sorted(set(partition.blocks))
    index = {block: position for position, block in enumerate(blocks)}
    representative: dict[int, int] = {}
    for state, block in enumerate(partition.blocks):
        representative.setdefault(block, state)
    transitions = []
    for block in blocks:
        state = representative[block]
        transitions.append(
            tuple(
                index[partition.blocks[int(machine.transitions[state][symbol])]]
                for symbol in range(machine.symbol_count)
            )
        )
    return RuleAutomaton(
        symbol_count=machine.symbol_count,
        transitions=tuple(transitions),
        outputs=tuple((0,) * machine.symbol_count for _ in blocks),
    ).validate()


def factorize(machine: RuleAutomaton) -> tuple[Factorization, ...]:
    """Every parallel decomposition into two non-trivial components.

    Ordered smallest-largest-component first, so the most balanced split -- the
    one that reuses best -- comes out in front.
    """

    machine = minimize(machine)
    if machine.state_count < 4:
        return ()
    candidates = [
        partition
        for partition in closed_partitions(machine)
        if not partition.is_trivial and not partition.is_identity
    ]
    found: list[Factorization] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for first, second in combinations(candidates, 2):
        if not first.meet(second).is_identity:
            continue
        key = tuple(sorted((first.blocks, second.blocks)))
        if key in seen:
            continue
        seen.add(key)
        left = _quotient(machine, first)
        right = _quotient(machine, second)
        outputs: dict[tuple[int, int, int], int] = {}
        left_index = {
            block: position
            for position, block in enumerate(sorted(set(first.blocks)))
        }
        right_index = {
            block: position
            for position, block in enumerate(sorted(set(second.blocks)))
        }
        for state in range(machine.state_count):
            for symbol in range(machine.symbol_count):
                key_cell = (
                    left_index[first.blocks[state]],
                    right_index[second.blocks[state]],
                    symbol,
                )
                outputs[key_cell] = int(machine.outputs[state][symbol])
        found.append(
            Factorization(
                left=left,
                right=right,
                left_blocks=first.blocks,
                right_blocks=second.blocks,
                outputs=outputs,
            )
        )
    found.sort(key=lambda item: max(item.part_sizes))
    return tuple(found)


def reconstruct(factorization: Factorization) -> RuleAutomaton:
    """Rebuild one machine from the two components and the table."""

    left = factorization.left
    right = factorization.right
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
        transitions.append(
            tuple(
                index[
                    (
                        int(left.transitions[first][symbol]),
                        int(right.transitions[second][symbol]),
                    )
                ]
                for symbol in range(symbols)
            )
        )
        outputs.append(
            tuple(
                factorization.outputs.get((first, second, symbol), 0)
                for symbol in range(symbols)
            )
        )
    return minimize(
        RuleAutomaton(
            symbol_count=symbols,
            transitions=tuple(transitions),
            outputs=tuple(outputs),
        )
    )
