"""Make the library pay on a task it has never seen.

The integrated agent's library pays 2.79x, and every bit of that comes from
*exact repeats*. A store that only helps when the same task comes back is a
cache. The claim a library is supposed to support is different and much
stronger: capability N+1 is cheaper because it is **built out of** capabilities
1..N, even though it has never been met.

That is the claim this module tests, and the task distribution makes it
falsifiable rather than assumed. Composites are products of primitives drawn
from a shared pool, so a composite the agent has never seen is genuinely
decomposable into parts it may already hold -- and `compositional_rules` builds
them mechanically, not by hand.

Three things make this practical.

**Scoring a combination costs nothing.** Each stored program is executed once
against the evidence in hand, producing a press vector. Every pair under every
combiner is then an elementwise boolean operation on two cached vectors. A
library of thirty programs offers 1305 pair hypotheses and not one of them
costs an episode, a program execution, or a search.

**Which raises the question of what to do about the statistics**, and the
answer is not the obvious one. Testing 1305 hypotheses at a single-hypothesis
alpha is a different act from testing one, so the first version of this module
divided alpha by the number examined. Measured, that correction turns out to
be the wrong trade:

- over 800 unrelated targets against a 24-record library, the naive threshold
  adopted a wrong composite **5 times** and the corrected one **never**;
- but **all 5 were refused by confirmation**, whose true held-out accuracies
  were 0.775 to 0.798 against a gate of 0.8;
- and the correction cost **six of eight** composable tasks at 10% label
  noise, where the naive threshold recovered seven.

Confirmation is not optional and runs on every candidate anyway. Tightening the
free gate until it can do the expensive gate's job buys a redundancy that is
already there and pays for it in noise tolerance, so the correction is off by
default and available for measurement. What keeps the search honest is that a
candidate must clear an accuracy floor, that a correct candidate outranks a
wrong one when both clear it, and that nothing is admitted before it has been
run in the environment.

**A found combination has to become a program.** Behavioural agreement is not
enough to execute: the composite is materialised as the product Mealy machine
of its parts and compiled like any induced hypothesis, so what gets confirmed
and admitted is a real artifact rather than a promise about two other files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neural_computer.induced_library import (
    InducedProgramLibrary,
    InducedProgramRecord,
)

from .compositional_rules import COMBINERS, product_rule
from .counter_state_programs import predict_symbols
from .current_symbol_acquire import THRESHOLD
from .lease_discrimination import DISCRIMINATION_ALPHA, binomial_upper_tail
from .rule_automata import RuleAutomaton

COMPOSITION_SCHEMA = "neural-computer.compositional-recognition.v1"
# Combiners the search will try. Deliberately the same three the task sampler
# builds composites from: a search over a *different* vocabulary than the world
# uses would measure the mismatch rather than the mechanism.
SEARCH_COMBINERS = ("and", "or", "xor")


@dataclass(frozen=True)
class Candidate:
    """One hypothesis the library can offer, and how well it did."""

    kind: str                      # "single" or "pair"
    slots: tuple[int, ...]
    combiner: str | None
    hits: int
    trials: int

    @property
    def rate(self) -> float:
        return self.hits / self.trials if self.trials else 0.0

    def label(self) -> str:
        if self.kind == "single":
            return f"slot {self.slots[0]}"
        return f"slot {self.slots[0]} {self.combiner} slot {self.slots[1]}"

    def payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "slots": list(self.slots),
            "combiner": self.combiner,
            "hits": self.hits,
            "trials": self.trials,
            "rate": self.rate,
            "label": self.label(),
        }


def flatten_targets(traces) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Every scored label, and where each one came from.

    Flattening once is what makes combination free: after this a hypothesis is
    a bit vector of the same length, and merging two of them is elementwise.
    """

    targets: list[int] = []
    positions: list[tuple[int, int]] = []
    for index, trace in enumerate(traces):
        for step, flag in enumerate(trace.eligible):
            if flag:
                targets.append(int(trace.outputs[step]))
                positions.append((index, step))
    return tuple(targets), tuple(positions)


def record_predictions(
    record: InducedProgramRecord, traces, positions
) -> tuple[int, ...]:
    """What one stored program would have pressed, at the scored steps."""

    by_trace: dict[int, tuple[int, ...]] = {}
    for index, trace in enumerate(traces):
        presses, _ = predict_symbols(
            record.program,
            trace.symbols,
            cluster_count=record.alphabet,
            initial_counters=record.initial_counters,
        )
        by_trace[index] = presses
    return tuple(by_trace[index][step] for index, step in positions)


def _merge(left: tuple[int, ...], right: tuple[int, ...], combiner: str):
    merge = COMBINERS[combiner]
    return tuple(merge(a, b) for a, b in zip(left, right, strict=True))


def _hits(predicted: tuple[int, ...], targets: tuple[int, ...]) -> int:
    return sum(1 for a, b in zip(predicted, targets, strict=True) if a == b)


def search_compositions(
    library: InducedProgramLibrary,
    traces,
    *,
    threshold: float = THRESHOLD,
    alpha: float = DISCRIMINATION_ALPHA,
    exclude: frozenset[int] = frozenset(),
    correct_for_multiplicity: bool = False,
    combiners: tuple[str, ...] = SEARCH_COMBINERS,
) -> tuple[Candidate | None, dict[str, Any]]:
    """The best thing the library can build that the evidence supports.

    Returns the winner and an accounting of what was examined, because the
    number examined is part of the test rather than a detail of it.

    `correct_for_multiplicity` divides alpha by the number of hypotheses
    examined. It is off by default because the measurements say so rather than
    because it is unprincipled: it prevents about five false adoptions in eight
    hundred unrelated targets, every one of which confirmation refuses anyway,
    and it costs six of eight composable tasks at 10% label noise.
    """

    episodes = tuple(traces)
    targets, positions = flatten_targets(episodes)
    trials = len(targets)
    report: dict[str, Any] = {
        "schema": COMPOSITION_SCHEMA,
        "library_size": library.record_count,
        "trials": trials,
        "singles_examined": 0,
        "pairs_examined": 0,
        "hypotheses": 0,
        "alpha": alpha,
        "effective_alpha": alpha,
        "winner": None,
    }
    if not trials or not library.record_count:
        return None, report

    usable = [slot for slot in range(library.record_count) if slot not in exclude]
    predictions = {
        slot: record_predictions(library.record(slot), episodes, positions)
        for slot in usable
    }

    candidates: list[Candidate] = []
    for slot in usable:
        candidates.append(
            Candidate("single", (slot,), None, _hits(predictions[slot], targets), trials)
        )
    report["singles_examined"] = len(candidates)
    for position, left in enumerate(usable):
        for right in usable[position + 1 :]:
            for combiner in combiners:
                merged = _merge(predictions[left], predictions[right], combiner)
                candidates.append(
                    Candidate(
                        "pair",
                        (left, right),
                        combiner,
                        _hits(merged, targets),
                        trials,
                    )
                )
    report["pairs_examined"] = len(candidates) - report["singles_examined"]
    report["hypotheses"] = len(candidates)

    # Every candidate examined is a chance to be fooled, so the bar rises with
    # how many were examined. Bonferroni needs no assumption about how the
    # hypotheses are correlated, which matters here because they are heavily
    # correlated -- pairs sharing a slot are anything but independent.
    effective = alpha / len(candidates) if correct_for_multiplicity else alpha
    report["effective_alpha"] = effective

    surviving = [
        candidate
        for candidate in candidates
        if candidate.rate >= threshold
        and binomial_upper_tail(trials, threshold, candidate.rate) <= effective
    ]
    report["survivors"] = len(surviving)
    if not surviving:
        return None, report
    # Ties to the simpler hypothesis: a single record before a pair, and a
    # smaller slot before a larger one, so the answer does not depend on the
    # order the library happens to be in.
    winner = max(
        surviving,
        key=lambda item: (item.rate, item.kind == "single", -max(item.slots)),
    )
    report["winner"] = winner.payload()
    return winner, report


def machine_of(record: InducedProgramRecord) -> RuleAutomaton | None:
    """The hypothesis a record was compiled from, if it kept one.

    The library stores programs and treats provenance as opaque, which is the
    right boundary for a store. Composition is the one thing that needs to look
    inside: two press vectors can be merged, but the *result* has to become an
    executable artifact, and for that the parts must be machines again.
    """

    payload = record.provenance.get("machine")
    if not isinstance(payload, dict):
        return None
    try:
        return RuleAutomaton(
            symbol_count=int(payload["symbol_count"]),
            transitions=tuple(tuple(int(v) for v in row) for row in payload["transitions"]),
            outputs=tuple(tuple(int(v) for v in row) for row in payload["outputs"]),
        ).validate()
    except (KeyError, TypeError, ValueError):
        return None


def composed_machine(
    library: InducedProgramLibrary, candidate: Candidate
) -> RuleAutomaton | None:
    """Materialise a candidate as one Mealy machine, or admit it cannot be."""

    machines = [machine_of(library.record(slot)) for slot in candidate.slots]
    if any(machine is None for machine in machines):
        return None
    if candidate.kind == "single":
        return machines[0]
    if candidate.combiner not in COMBINERS:
        return None
    return product_rule(machines[0], machines[1], candidate.combiner)
