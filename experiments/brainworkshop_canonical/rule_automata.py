"""Sampled sequence rules, so the task distribution is not hand-written.

Four hand-written rules cannot falsify a generality claim: a solver that
covers all four is indistinguishable from one that fits four special cases,
and holding out *seeds* only re-samples episodes of a rule the system has
already seen. This module replaces the rule list with the general class of
finite-state rules over the symbol stream.

A rule is a Mealy machine: it consumes one symbol per tick, updates state, and
emits the press/no-press the verifier expects. Nothing about the class is
chosen to suit the controller, which is the point — a task distribution
defined by the agent's own program space could never falsify the agent.

Three properties make this usable as a held-out distribution:

- **canonical identity.** Minimise, then relabel by breadth-first order, then
  digest. Two rules are the same rule exactly when their digests match, so a
  held-out family is provably unseen rather than merely differently seeded.
- **a complexity axis.** The minimal state count is the rule's description
  length, and it is ground truth rather than an estimate, so acquisition cost
  can be plotted against difficulty instead of against a hand-made ladder.
- **known embeddings.** The four hand-written rules are instances of this
  class (`known_rule`), so every result recorded before this module remains a
  measurable point inside the new distribution.

The learner never sees any of this. It sees rendered symbols and scalar
outcomes; the machine lives in the verifier.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

RULE_AUTOMATON_SCHEMA = "neural-computer.rule-automaton.v1"


@dataclass(frozen=True)
class RuleAutomaton:
    """A Mealy machine over symbols: press is a function of state and symbol."""

    symbol_count: int
    transitions: tuple[tuple[int, ...], ...]
    outputs: tuple[tuple[int, ...], ...]
    # How many actions the output alphabet ranges over. Two is press-or-not,
    # which is what every rule in this repository was until now and what this
    # still defaults to, so nothing binary changes. Above two, an output is a
    # *choice*, and the feedback the agent gets from a wrong one stops being
    # equivalent to the right one -- which is the whole point.
    action_count: int = 2
    schema: str = RULE_AUTOMATON_SCHEMA

    @property
    def state_count(self) -> int:
        return len(self.transitions)

    def validate(self) -> RuleAutomaton:
        if self.schema != RULE_AUTOMATON_SCHEMA:
            raise ValueError("unsupported rule automaton schema")
        if self.symbol_count < 2:
            raise ValueError("a rule automaton needs at least two symbols")
        if self.action_count < 2:
            raise ValueError("a rule automaton needs at least two actions")
        if not self.transitions or len(self.transitions) != len(self.outputs):
            raise ValueError("rule automaton rows are inconsistent")
        for row, out in zip(self.transitions, self.outputs, strict=True):
            if len(row) != self.symbol_count or len(out) != self.symbol_count:
                raise ValueError("rule automaton rows must cover every symbol")
            if any(not 0 <= state < self.state_count for state in row):
                raise ValueError("rule automaton transition leaves the state set")
            if any(not 0 <= int(bit) < self.action_count for bit in out):
                raise ValueError("rule automaton output is outside the action set")
        return self

    def expected(self, symbols: torch.Tensor | list[int]) -> list[int]:
        """The press the verifier expects at every position of one episode."""

        self.validate()
        state = 0
        presses: list[int] = []
        for symbol in symbols:
            index = int(symbol)
            if not 0 <= index < self.symbol_count:
                raise ValueError("episode symbol is outside the rule alphabet")
            presses.append(int(self.outputs[state][index]))
            state = int(self.transitions[state][index])
        return presses

    def payload(self) -> dict[str, object]:
        self.validate()
        payload = {
            "schema": self.schema,
            "symbol_count": self.symbol_count,
            "transitions": [list(row) for row in self.transitions],
            "outputs": [list(row) for row in self.outputs],
        }
        # Written only when it is not the binary default, so every digest
        # recorded before actions were a choice still means what it meant.
        if self.action_count != 2:
            payload["action_count"] = self.action_count
        return payload

    def digest(self) -> str:
        """Canonical identity: minimise, relabel, then hash."""

        canonical = canonicalize(self)
        encoded = json.dumps(canonical.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


def _reachable(automaton: RuleAutomaton) -> list[int]:
    seen = [0]
    frontier = [0]
    while frontier:
        state = frontier.pop()
        for symbol in range(automaton.symbol_count):
            target = int(automaton.transitions[state][symbol])
            if target not in seen:
                seen.append(target)
                frontier.append(target)
    return sorted(seen)


def minimize(automaton: RuleAutomaton) -> RuleAutomaton:
    """Drop unreachable states, then merge behaviourally identical ones."""

    automaton.validate()
    reachable = _reachable(automaton)
    index = {state: position for position, state in enumerate(reachable)}
    transitions = [
        tuple(index[int(automaton.transitions[state][symbol])] for symbol in range(automaton.symbol_count))
        for state in reachable
    ]
    outputs = [tuple(int(bit) for bit in automaton.outputs[state]) for state in reachable]
    # Partition refinement: start from output behaviour, split on where each
    # symbol sends a state, until no block splits.
    block = {state: outputs[state] for state in range(len(reachable))}
    labels = {signature: number for number, signature in enumerate(sorted(set(block.values())))}
    partition = [labels[block[state]] for state in range(len(reachable))]
    while True:
        signatures = [
            (partition[state], tuple(partition[transitions[state][symbol]] for symbol in range(automaton.symbol_count)))
            for state in range(len(reachable))
        ]
        labels = {signature: number for number, signature in enumerate(sorted(set(signatures)))}
        refined = [labels[signature] for signature in signatures]
        if refined == partition:
            break
        partition = refined
    representative: dict[int, int] = {}
    for state, group in enumerate(partition):
        representative.setdefault(group, state)
    # The block containing the start state must be numbered zero. Partition
    # labels come from sorting signatures, which has no reason to put the
    # start state first, and `expected` always begins at state 0 -- so
    # numbering by label alone silently returns a machine with different
    # behaviour. A symmetric two-state parity rule minimises to the same
    # transition table with its outputs swapped, which is the inverse of the
    # machine that went in.
    start_group = partition[0]
    groups = [start_group] + sorted(
        group for group in representative if group != start_group
    )
    order = {group: position for position, group in enumerate(groups)}
    merged_transitions = tuple(
        tuple(
            order[partition[transitions[representative[group]][symbol]]]
            for symbol in range(automaton.symbol_count)
        )
        for group in groups
    )
    merged_outputs = tuple(
        tuple(int(bit) for bit in outputs[representative[group]]) for group in groups
    )
    return RuleAutomaton(
        symbol_count=automaton.symbol_count,
        transitions=merged_transitions,
        outputs=merged_outputs,
        action_count=automaton.action_count,
    ).validate()


def canonicalize(automaton: RuleAutomaton) -> RuleAutomaton:
    """Minimise, then relabel states in breadth-first order from the start."""

    reduced = minimize(automaton)
    # The start state must survive minimisation as the block containing 0.
    order = [0]
    position = 0
    while position < len(order):
        state = order[position]
        position += 1
        for symbol in range(reduced.symbol_count):
            target = int(reduced.transitions[state][symbol])
            if target not in order:
                order.append(target)
    index = {state: number for number, state in enumerate(order)}
    return RuleAutomaton(
        action_count=reduced.action_count,
        symbol_count=reduced.symbol_count,
        transitions=tuple(
            tuple(index[int(reduced.transitions[state][symbol])] for symbol in range(reduced.symbol_count))
            for state in order
        ),
        outputs=tuple(
            tuple(int(bit) for bit in reduced.outputs[state]) for state in order
        ),
    ).validate()


def positive_rate(
    automaton: RuleAutomaton, *, seed: int, steps: int = 512, episodes: int = 8
) -> float:
    """How often the rule asks for a press, on uniform symbol streams."""

    generator = torch.Generator().manual_seed(int(seed))
    presses = 0
    total = 0
    for _ in range(episodes):
        symbols = torch.randint(
            0, automaton.symbol_count, (steps,), generator=generator
        )
        expected = automaton.expected(symbols)
        presses += sum(expected)
        total += len(expected)
    return presses / total if total else 0.0


def best_constant_rate(
    automaton: RuleAutomaton, *, seed: int, steps: int = 512, episodes: int = 8
) -> float:
    """How well the best single fixed action does, on uniform symbol streams.

    `positive_rate` answers this for two actions only, and answers it
    obliquely: a press rate of 0.9 means "always press" scores 0.9. With more
    than two actions the majority action is what a learner has to beat, and it
    is no longer read off a single number. This is the baseline every
    multi-action result has to be reported against.
    """

    generator = torch.Generator().manual_seed(int(seed))
    counts = [0] * automaton.action_count
    total = 0
    for _ in range(episodes):
        symbols = torch.randint(
            0, automaton.symbol_count, (steps,), generator=generator
        )
        for action in automaton.expected(symbols):
            counts[int(action)] += 1
            total += 1
    return max(counts) / total if total else 0.0


def sample_rule(
    *,
    symbol_count: int,
    state_count: int,
    seed: int,
    action_count: int = 2,
    maximum_constant_rate: float | None = None,
    minimum_positive_rate: float = 0.15,
    maximum_positive_rate: float = 0.85,
    attempts: int = 256,
) -> RuleAutomaton | None:
    """Draw one non-degenerate rule of exactly `state_count` states.

    Rejection is on measurability, not on content: a rule that almost never
    asks for a press, or almost always does, cannot separate a learner from a
    constant policy at any episode length. Rules that minimise to fewer states
    than asked for are rejected too, so the state count stays ground truth.
    """

    if state_count < 1:
        raise ValueError("a rule needs at least one state")
    if action_count < 2:
        raise ValueError("a rule needs at least two actions")
    generator = torch.Generator().manual_seed(int(seed))
    for attempt in range(attempts):
        transitions = torch.randint(
            0, state_count, (state_count, symbol_count), generator=generator
        )
        outputs = torch.randint(
            0, action_count, (state_count, symbol_count), generator=generator
        )
        candidate = RuleAutomaton(
            symbol_count=symbol_count,
            transitions=tuple(tuple(int(v) for v in row) for row in transitions),
            outputs=tuple(tuple(int(v) for v in row) for row in outputs),
            action_count=action_count,
        )
        reduced = canonicalize(candidate)
        if reduced.state_count != state_count:
            continue
        if maximum_constant_rate is None:
            # The historical window, kept exactly so every rule sampled before
            # actions were a choice is sampled the same way now.
            rate = positive_rate(reduced, seed=seed + attempt)
            if not minimum_positive_rate <= rate <= maximum_positive_rate:
                continue
        else:
            # The measurable quantity at any action count: what a learner has
            # to beat is the best single fixed answer. Applied at two actions
            # as well -- a first version applied it only above two, which let
            # binary rules into a comparison at constant rates up to 0.835
            # while three-action rules were held to 0.6, and made the binary
            # column incomparable with the rest.
            if best_constant_rate(reduced, seed=seed + attempt) > maximum_constant_rate:
                continue
        return reduced
    return None


def sample_rule_population(
    *,
    symbol_count: int,
    state_counts: tuple[int, ...],
    count: int,
    seed: int,
    exclude_digests: frozenset[str] = frozenset(),
) -> tuple[RuleAutomaton, ...]:
    """Draw distinct rules, spread over the requested complexity levels."""

    drawn: list[RuleAutomaton] = []
    seen = set(exclude_digests)
    attempt = 0
    while len(drawn) < count and attempt < count * 512:
        states = state_counts[len(drawn) % len(state_counts)]
        candidate = sample_rule(
            symbol_count=symbol_count, state_count=states, seed=seed + attempt
        )
        attempt += 1
        if candidate is None:
            continue
        digest = candidate.digest()
        if digest in seen:
            continue
        seen.add(digest)
        drawn.append(candidate)
    if len(drawn) < count:
        raise RuntimeError("rule sampling budget exhausted before the population filled")
    return tuple(drawn)


def known_rule(name: str, *, symbol_count: int, target_symbol: int = 0, n_back: int = 1) -> RuleAutomaton:
    """The hand-written rules as instances of the sampled class.

    Their minimal state counts place the existing records on this module's
    complexity axis: `current_symbol` needs one state, `changed` and `onset`
    need one per symbol, and `n_back` needs one per remembered history.
    """

    if name == "current_symbol":
        return canonicalize(
            RuleAutomaton(
                symbol_count=symbol_count,
                transitions=((0,) * symbol_count,),
                outputs=(tuple(int(s == target_symbol) for s in range(symbol_count)),),
            )
        )
    if name in ("changed", "onset"):
        # State s means "the previous symbol was s".
        transitions = tuple(
            tuple(symbol for symbol in range(symbol_count)) for _ in range(symbol_count)
        )
        if name == "changed":
            outputs = tuple(
                tuple(int(symbol != previous) for symbol in range(symbol_count))
                for previous in range(symbol_count)
            )
        else:
            outputs = tuple(
                tuple(
                    int(symbol == target_symbol and symbol != previous)
                    for symbol in range(symbol_count)
                )
                for previous in range(symbol_count)
            )
        return canonicalize(
            RuleAutomaton(
                symbol_count=symbol_count, transitions=transitions, outputs=outputs
            )
        )
    if name == "n_back":
        if n_back < 1:
            raise ValueError("n-back needs a positive depth")
        # State is the last `n_back` symbols, most recent last.
        histories = [()]
        for _ in range(n_back):
            histories = [(*history, symbol) for history in histories for symbol in range(symbol_count)]
        index = {history: number for number, history in enumerate(histories)}
        transitions = tuple(
            tuple(index[(*history[1:], symbol)] for symbol in range(symbol_count))
            for history in histories
        )
        outputs = tuple(
            tuple(int(symbol == history[0]) for symbol in range(symbol_count))
            for history in histories
        )
        return canonicalize(
            RuleAutomaton(
                symbol_count=symbol_count, transitions=transitions, outputs=outputs
            )
        )
    raise KeyError(f"unknown hand-written rule: {name}")


def held_out_split(
    rules: tuple[RuleAutomaton, ...], *, holdout_fraction: float = 0.5
) -> tuple[tuple[RuleAutomaton, ...], tuple[RuleAutomaton, ...]]:
    """Split by canonical digest, so the holdout is unseen rules, not seeds.

    The split is a deterministic function of rule identity: adding rules to a
    population never moves an existing rule across the boundary, so a holdout
    cannot drift into training as the study grows.
    """

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout fraction must lie strictly between zero and one")
    boundary = int(holdout_fraction * (1 << 32))
    train: list[RuleAutomaton] = []
    holdout: list[RuleAutomaton] = []
    for rule in rules:
        bucket = int(rule.digest()[:8], 16)
        (holdout if bucket < boundary else train).append(rule)
    return tuple(train), tuple(holdout)
