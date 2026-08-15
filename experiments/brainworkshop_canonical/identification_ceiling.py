"""How much of the task does one episode of feedback already determine?

Three quantities decide whether a learner is any good, and this repository has
measured two of them.

*Expressiveness* -- what the program family can represent -- is the counter
bridge at 18/18 against the temporal family's 7/18. *Search cost* -- what the
proposer spends -- is the proposer record: 1432 episodes down to 125.

The third is *identifiability*: how much evidence it takes to pin the task
down at all. Without it there is no way to tell whether 125 episodes is close
to optimal or absurd, and no way to say whether the eleven unsolved rules fail
because the evidence was thin or because nothing in the family could have used
it.

This module measures it, and deliberately does so without the program family
in the way. The learner-side inputs are exactly what the agent already has:

- symbols come from clustering its own frontend's events, so the alphabet is
  discovered rather than supplied (`prototype_templates.cluster_events`);
- outputs come from inverting one scored episode's per-step reward
  (`feedback_proposer.probe_target`).

From that trace it infers the smallest Mealy machine consistent with what it
saw. Nothing reads the rule, the automaton, or the verifier's internals.

Two things about the inference are worth stating up front, because both were
found by trying the obvious thing first.

Greedy state merging -- RPNI (Oncina and Garcia, 1992), the standard tool --
does not work on a single episode. An episode is a *chain*: every node has one
outgoing edge, so the earliest merges have no evidence against them, get
accepted, and poison every later one. `_rpni` is kept here and reports 31
states for a 2-state rule, which is what that failure looks like.

Exact minimal inference works, and then stops working. Depth-first assignment
with immediate checking finds the true minimal machine for rules up to four
states in well under a second, and does not finish at five within any budget
worth spending -- including with eight episodes instead of one, which does not
help. That is Gold's 1978 result showing minimal automaton inference from a
passive sample is NP-hard, met in practice rather than in a footnote, and it
is the reason Angluin's L* buys polynomial guarantees with *active* queries
instead. The boundary is reported rather than hidden.

Identification is then scored the honest way, by prediction on a *different*
episode rather than by comparing to the generating machine. Cluster indices
are arbitrary, so a digest comparison would need the right relabelling and
would flatter or punish the result for the wrong reason; predicting a held-out
episode does not care how the states or symbols were named.

What this is and is not: it is a ceiling, in the same sense as the enumeration
ceiling. An experimenter's inference procedure, run to establish what the
evidence supports. It is not the agent, nothing here is admitted, and no
program it produces is given to the searcher.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file

from .accumulation_curve import _config, curriculum_rules
from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .feedback_proposer import probe_target
from .program_search import install_proposal, propose_from_bank
from .prototype_templates import cluster_events, observe_events
from .rendered_environment import RenderedBrainWorkshopConfig
from .rendered_live import run_rendered_live_lifetime
from .rule_automata import RuleAutomaton, minimize

EXPERIMENT_ID = "brainworkshop-identification-ceiling-2026-08-15"
IDENTIFICATION_SCHEMA = "neural-computer.identification-ceiling.v1"
DEVELOPMENT_SEED = 41
STEPS = 448


@dataclass(frozen=True)
class Trace:
    """One episode as the learner sees it: clustered symbols, scored outputs."""

    symbols: tuple[int, ...]
    outputs: tuple[int, ...]
    eligible: tuple[bool, ...]
    symbol_count: int

    def __post_init__(self) -> None:
        if not (len(self.symbols) == len(self.outputs) == len(self.eligible)):
            raise ValueError("a trace must align symbols, outputs, and eligibility")


def _assign_symbols(events: torch.Tensor, clusters: torch.Tensor) -> tuple[int, ...]:
    """Nearest discovered cluster per event. The alphabet is not supplied."""

    distances = torch.cdist(events, clusters)
    return tuple(int(index) for index in distances.argmin(dim=1))


MAX_STATES = 12
NODE_BUDGET = 500_000


def infer_machine(
    traces: Trace | tuple[Trace, ...],
    *,
    max_states: int = MAX_STATES,
    node_budget: int = NODE_BUDGET,
) -> RuleAutomaton | None:
    """The *smallest* Mealy machine consistent with the trace, or None.

    Greedy state merging is the usual tool and it does not work on a single
    episode. An episode is a chain -- every node has exactly one outgoing edge
    -- so the first few merges have no evidence against them, get accepted,
    and poison every later one. `_rpni` below is kept because it is what a
    reader expects to see tried, and it reports fifty states for a two-state
    rule.

    Instead this searches state counts in ascending order and, for each,
    assigns states to positions by depth-first search with immediate
    checking: following the trace fixes transition and output cells as it
    goes, and a contradiction backtracks at once. The first count that admits
    a consistent assignment is minimal by construction, which is a stronger
    claim than a merge heuristic can make and is what a ceiling needs.

    Symmetry is broken the standard way -- a newly introduced state always
    takes the next index -- so the same machine is not searched under every
    relabelling.
    """

    episodes = (traces,) if isinstance(traces, Trace) else tuple(traces)
    if not episodes or sum(len(item.symbols) for item in episodes) < 2:
        return None
    for states in range(1, max_states + 1):
        machine = _consistent_machine(episodes, states, node_budget)
        if machine is not None:
            return machine
    return None


class _BudgetExhausted(Exception):
    """Raised when a state count costs more search than it is worth."""


def _consistent_machine(
    traces: tuple[Trace, ...], states: int, node_budget: int
) -> RuleAutomaton | None:
    """Depth-first assignment of `states` states to every trace, or None.

    Walking a trace fills two tables as it goes -- where each (state, symbol)
    leads and what it emits -- and any disagreement with a later visit
    backtracks at once.

    Episodes are kept separate rather than concatenated, because each one
    restarts the machine in its initial state. That is not bookkeeping: a
    single episode is one long string and gives the weakest possible sample,
    while several independent runs from the start state are exactly the
    multi-string evidence that makes contradictions surface early. It is the
    difference between this search finishing and not.
    """

    steps = [(index, position) for index, trace in enumerate(traces)
             for position in range(len(trace.symbols))]
    transitions: dict[tuple[int, int], int] = {}
    outputs: dict[tuple[int, int], int | None] = {}
    used = [1]
    visited = [0]

    def walk(step: int, state: int) -> bool:
        if step == len(steps):
            return True
        visited[0] += 1
        if visited[0] > node_budget:
            raise _BudgetExhausted
        index, position = steps[step]
        trace = traces[index]
        symbol = trace.symbols[position]
        key = (state, symbol)
        want = trace.outputs[position] if trace.eligible[position] else None
        # The next step restarts at the initial state when this episode ends.
        following = (
            0
            if step + 1 < len(steps) and steps[step + 1][1] == 0
            else None
        )
        if key in transitions:
            successor = transitions[key]
            recorded = outputs[key]
            if want is not None and recorded is None:
                outputs[key] = want
                if walk(step + 1, following if following is not None else successor):
                    return True
                outputs[key] = None
                return False
            if want is not None and recorded != want:
                return False
            return walk(step + 1, following if following is not None else successor)
        options = list(range(used[0]))
        if used[0] < states:
            options.append(used[0])
        for target in options:
            transitions[key] = target
            outputs[key] = want
            grew = target == used[0]
            if grew:
                used[0] += 1
            if walk(step + 1, following if following is not None else target):
                return True
            if grew:
                used[0] -= 1
            del transitions[key]
            del outputs[key]
        return False

    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(limit, len(steps) * 2 + 1000))
    try:
        if walk(0, 0):
            return _machine_from_cells(traces[0], transitions, outputs, used[0])
    except _BudgetExhausted:
        return None
    finally:
        sys.setrecursionlimit(limit)
    return None


def _machine_from_cells(
    trace: Trace,
    transitions: dict[tuple[int, int], int],
    outputs: dict[tuple[int, int], int | None],
    used: int,
) -> RuleAutomaton:
    """Fill unvisited cells with a silent self-loop and minimise.

    A cell the trace never exercised is genuinely unconstrained. The
    least-committal completion is to stay put and emit nothing; the held-out
    score is what says whether that guess mattered.
    """

    rows = []
    emissions = []
    for state in range(used):
        row = []
        emission = []
        for symbol in range(trace.symbol_count):
            target = transitions.get((state, symbol))
            row.append(state if target is None else int(target))
            recorded = outputs.get((state, symbol))
            emission.append(0 if recorded is None else int(recorded))
        rows.append(tuple(row))
        emissions.append(tuple(emission))
    return minimize(
        RuleAutomaton(
            symbol_count=trace.symbol_count,
            transitions=tuple(rows),
            outputs=tuple(emissions),
        )
    )


def _rpni(trace: Trace) -> RuleAutomaton | None:
    """State merging over the observed chain, gated on the whole trace.

    A single episode is a *chain*: every node has exactly one outgoing edge,
    so a merge is almost always locally consistent and local determinization
    accepts nearly everything. The first version of this did exactly that,
    over-merged in the first few positions, and then reported fifty states for
    a two-state rule because every later position conflicted with the ruined
    class.

    So each tentative merge is checked against the entire trace instead:
    build the quotient transition table, and reject if any (class, symbol)
    pair is seen with two different outputs. Disagreeing successors are not a
    contradiction -- they are a further merge -- so those are applied and the
    check repeats until nothing changes.
    """

    length = len(trace.symbols)
    # Position t is a node; the edge out of t reads symbols[t], emits
    # outputs[t], and lands on node t + 1.
    parent = list(range(length + 1))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(left: int, right: int) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)

    def quotient() -> dict[tuple[int, int], tuple[int, int]] | None:
        """The merged transition table, or None if two outputs collide."""

        table: dict[tuple[int, int], tuple[int, int]] = {}
        while True:
            table = {}
            pending: tuple[int, int] | None = None
            for position in range(length):
                if not trace.eligible[position]:
                    continue
                key = (find(position), trace.symbols[position])
                value = (find(position + 1), trace.outputs[position])
                seen = table.get(key)
                if seen is None:
                    table[key] = value
                    continue
                if seen[1] != value[1]:
                    return None
                if seen[0] != value[0]:
                    pending = (seen[0], value[0])
                    break
            if pending is None:
                return table
            union(*pending)

    def try_merge(left: int, right: int) -> bool:
        snapshot = list(parent)
        union(left, right)
        if quotient() is None:
            parent[:] = snapshot
            return False
        return True

    established: list[int] = []
    for position in range(length + 1):
        if find(position) != position:
            continue
        if any(find(candidate) == position for candidate in established):
            continue
        if not any(try_merge(candidate, position) for candidate in established):
            established.append(position)
    table = quotient()
    if table is None:
        return None
    return _to_automaton(table, trace, find)


def _to_automaton(
    table: dict[tuple[int, int], tuple[int, int]],
    trace: Trace,
    find,
) -> RuleAutomaton | None:
    """Number the merged states and fill unobserved cells with a self-loop.

    An unobserved (state, symbol) cell is genuinely unconstrained by the
    trace. Filling it with a non-emitting self-loop is the least-committal
    completion, and the held-out score is what says whether the guess mattered.
    """

    roots = sorted({key[0] for key in table} | {value[0] for value in table.values()})
    start = find(0)
    if start not in roots:
        return None
    # Relabel so the start state is 0, which `canonicalize` expects anyway.
    order = [start] + [root for root in roots if root != start]
    index = {root: position for position, root in enumerate(order)}
    transitions = []
    outputs = []
    for root in order:
        row_target = []
        row_output = []
        for symbol in range(trace.symbol_count):
            cell = table.get((root, symbol))
            if cell is None or cell[0] not in index:
                row_target.append(index[root])
                row_output.append(0)
                continue
            row_target.append(index[cell[0]])
            row_output.append(int(cell[1]))
        transitions.append(tuple(row_target))
        outputs.append(tuple(row_output))
    machine = RuleAutomaton(
        symbol_count=trace.symbol_count,
        transitions=tuple(transitions),
        outputs=tuple(outputs),
    )
    return minimize(machine)


def held_out_accuracy(machine: RuleAutomaton, trace: Trace) -> float:
    """How well an inferred machine predicts an episode it never saw."""

    predicted = machine.expected(list(trace.symbols))
    hits = sum(
        1
        for step, flag in enumerate(trace.eligible)
        if flag and predicted[step] == trace.outputs[step]
    )
    trials = sum(1 for flag in trace.eligible if flag)
    return hits / trials if trials else 0.0


def episode_trace(
    payload: dict[str, object],
    encoders,
    bank,
    config: RenderedBrainWorkshopConfig,
    clusters: torch.Tensor,
    *,
    seed: int,
) -> Trace:
    """One scored episode, turned into what the learner can actually read."""

    machine = _machine(payload, learn=False)
    proposals = propose_from_bank(bank)
    install_proposal(machine, bank, proposals[0])
    lifetime = run_rendered_live_lifetime(
        machine, encoders, config, seed=seed, learn=False, sample=False
    )
    probe = probe_target(lifetime)
    events = observe_events(encoders, config, seed=seed)
    return Trace(
        symbols=_assign_symbols(events, clusters),
        outputs=probe.target,
        eligible=probe.eligible,
        symbol_count=int(clusters.shape[0]),
    )


def identification_report(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    steps: int = STEPS,
    node_budget: int = NODE_BUDGET,
) -> dict[str, Any]:
    """For each sampled rule: is it identifiable from one episode's feedback?"""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for rule in curriculum_rules():
        config = _config(rule, steps)
        clusters = cluster_events(observe_events(encoders, config, seed=seed))
        trace = episode_trace(
            payload, encoders, bank, config, clusters, seed=seed
        )
        held_out = episode_trace(
            payload, encoders, bank, config, clusters, seed=seed + 1
        )
        began = time.perf_counter()
        machine = infer_machine(trace, node_budget=node_budget)
        rows.append(
            {
                "rule_digest": rule.digest(),
                "true_state_count": rule.state_count,
                "clusters_found": int(clusters.shape[0]),
                "identified": machine is not None,
                "inferred_state_count": (
                    None if machine is None else machine.state_count
                ),
                "held_out_accuracy": (
                    None if machine is None else held_out_accuracy(machine, held_out)
                ),
                "episodes_of_feedback": 1,
                "inference_seconds": time.perf_counter() - began,
            }
        )
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("identification ceiling mutated AgentBrain.bank")
    identified = [row for row in rows if row["identified"]]
    exact = [row for row in identified if row["held_out_accuracy"] == 1.0]
    report = {
        "schema": IDENTIFICATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic",
        "note": (
            "experimenter inference procedure run to establish what the "
            "feedback supports; nothing is admitted and no inferred machine "
            "is given to the searcher"
        ),
        "bank_sha256": before,
        "bank_unchanged": after == before,
        "seed": seed,
        "steps": steps,
        "node_budget": node_budget,
        "rules": rows,
        "identified": len(identified),
        "exactly_predicting_held_out": len(exact),
        "of": len(rows),
        "identified_by_state_count": {
            str(states): sum(
                1
                for row in rows
                if row["true_state_count"] == states and row["identified"]
            )
            for states in sorted({row["true_state_count"] for row in rows})
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "identification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    (output_directory / "checksums.sha256").write_text(
        "\n".join(
            f"{sha256_file(path)}  {path.name}"
            for path in sorted(output_directory.glob("*.json"))
        )
        + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument(
        "--bank",
        type=Path,
        default=repository / "artifacts/checkpoints/AgentBrain.bank",
    )
    parser.add_argument(
        "--frontend",
        type=Path,
        default=repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_identification_ceiling_2026-08-15"
        ),
    )
    parser.add_argument("--node-budget", type=int, default=NODE_BUDGET)
    arguments = parser.parse_args()
    report = identification_report(
        arguments.controller_artifact,
        arguments.bank,
        arguments.output_dir,
        frontend_path=arguments.frontend,
        node_budget=arguments.node_budget,
    )
    print(
        json.dumps(
            {
                "bank_unchanged": report["bank_unchanged"],
                "identified": f"{report['identified']}/{report['of']}",
                "exactly_predicting_held_out": report[
                    "exactly_predicting_held_out"
                ],
                "identified_by_state_count": report["identified_by_state_count"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
