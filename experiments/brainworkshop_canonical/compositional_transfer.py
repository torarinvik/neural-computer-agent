"""Does the library help with a task it has never seen?

The integrated agent's 2.79x is entirely from exact repeats. That makes the
store a cache. The claim a library has to support is that capability N+1 is
cheaper because it is *built from* capabilities 1..N -- on a task that has
never occurred.

The stream is built to make that answerable and to make it refutable. The agent
meets four primitives first, then a run of **composites it has never seen**,
each one a product of two of those primitives under a boolean combiner. If the
library is only a cache, the composites cost exactly what a fresh agent pays.

Six arms, and the last three exist to try to break the first.

- **composing** -- library persists, and it is asked what it can *build*.
- **recognising** -- library persists, single records only. Isolates
  composition from plain repeat-recognition, which is the comparison the
  headline number has to be against. Against the no-library control it would
  only re-measure what the integrated agent already showed.
- **control** -- library discarded before every task.
- **disjoint** -- composing, but the composites are products of a *different*
  primitive pool than the one the agent learned. Composition must not fire.
  If it does, the mechanism is finding structure in coincidence.
- **shuffled** -- composing, with each probe's labels permuted.
- **uncorrected** -- composing with the naive threshold, to show what testing
  a thousand hypotheses at a single-hypothesis alpha actually does.

The last is not a straw man. Scoring a combination is free, so the search is
enormous by construction, and an agent that treats a thousand chances to be
fooled like one chance is going to be fooled. What that costs is measured here
rather than asserted.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer import ExternalTemporalProgramBank
from neural_computer.induced_library import InducedProgramLibrary
from neural_computer.promotion import sha256_file

from .accumulation_curve import _config
from .compositional_recognition import machine_of
from .compositional_rules import sample_composite_population, sample_primitive_pool
from .controller_pretraining import load_temporal_controller_artifact
from .counter_state_programs import cluster_symbol_map
from .current_symbol_acquire import FRONTEND_SEED, THRESHOLD, _machine, curated_frontend
from .integrated_agent import (
    CONFIRMATION_STEPS,
    TASK_SEED_STRIDE,
    admit,
    discover_alphabet,
    noisy_feedback,
    shuffled_feedback,
    solve_task,
)
from .rule_automata import RuleAutomaton, positive_rate


def in_cluster_symbols(
    machine: RuleAutomaton, channel_of_symbol: tuple[int, ...]
) -> RuleAutomaton:
    """Rewrite a verifier-side rule in the frontend's own symbol names.

    **Scoring only.** An induced machine speaks cluster indices, because that
    is the alphabet the agent discovered; the sampled primitive speaks the
    verifier's. The two are the same rule under a permutation, so comparing
    their canonical digests directly reports a mismatch on every single case --
    which is exactly what a first version of this check did, while the slots it
    was doubting were correct.

    The permutation comes from `cluster_symbol_map`, which is an oracle. It is
    used here to *check* an answer, never to produce one.
    """

    machine.validate()
    if sorted(channel_of_symbol) != list(range(machine.symbol_count)):
        raise ValueError("the symbol map is not a permutation of the alphabet")
    transitions = [[0] * machine.symbol_count for _ in range(machine.state_count)]
    outputs = [[0] * machine.symbol_count for _ in range(machine.state_count)]
    for state in range(machine.state_count):
        for symbol in range(machine.symbol_count):
            channel = int(channel_of_symbol[symbol])
            transitions[state][channel] = int(machine.transitions[state][symbol])
            outputs[state][channel] = int(machine.outputs[state][symbol])
    return RuleAutomaton(
        symbol_count=machine.symbol_count,
        transitions=tuple(tuple(row) for row in transitions),
        outputs=tuple(tuple(row) for row in outputs),
    ).validate()

EXPERIMENT_ID = "brainworkshop-compositional-transfer-2026-08-15"
TRANSFER_SCHEMA = "neural-computer.compositional-transfer.v1"
DEVELOPMENT_SEED = 41
# More primitives means more pairs, and the number of hypotheses is the thing
# under test: six primitives put the search past a hundred candidates per task
# before any composite has been admitted.
PRIMITIVE_COUNT = 6
COMPOSITES_PER_RUN = 8
# Primitives are repeated so the library has them firmly before the composites
# arrive; a library that failed to hold its parts would make the composite
# question unanswerable rather than answered in the negative.
PRIMITIVE_REPEATS = 2


def build_stream(
    *,
    seed: int,
    symbol_count: int = 4,
    primitive_count: int = PRIMITIVE_COUNT,
    composites: int = COMPOSITES_PER_RUN,
    disjoint: bool = False,
):
    """Primitives first, then composites the agent has never met.

    With `disjoint`, the composites are built from a second, unrelated pool.
    The agent still learns the first pool, so its library is the same size and
    the same shape -- and has nothing to offer. That is the control that
    separates composition from coincidence.
    """

    # The pool moves with the seed, so a replicate is a different world rather
    # than the same world seen through different episodes.
    learned = sample_primitive_pool(
        symbol_count=symbol_count, count=primitive_count, seed=8000 + 37 * seed
    )
    source = (
        sample_primitive_pool(
            symbol_count=symbol_count, count=primitive_count, seed=900_000 + 37 * seed
        )
        if disjoint
        else learned
    )
    # Tighter than the sampler's default. At a press rate of 0.16 a machine
    # that never presses scores 0.84, which clears the 0.8 gate outright -- so
    # such a task cannot separate composition from doing nothing, and the first
    # run had two of them.
    population = sample_composite_population(
        source,
        seed=seed,
        minimum_positive_rate=0.25,
        maximum_positive_rate=0.75,
    )
    chosen = population[:composites]

    stream = []
    position = 0
    for repeat in range(PRIMITIVE_REPEATS):
        for primitive in learned:
            stream.append(
                (position, primitive, repeat, "primitive", None, (primitive.digest(),))
            )
            position += 1
    for composite in chosen:
        stream.append(
            (
                position,
                composite.automaton,
                0,
                "composite",
                composite.combiner,
                composite.part_digests(),
            )
        )
        position += 1
    return tuple(stream), learned, chosen


def run_arm(
    payload: dict[str, object],
    encoders,
    bank: ExternalTemporalProgramBank,
    stream,
    clusters: torch.Tensor,
    *,
    grow: bool,
    compose: bool,
    seed: int,
    frontend_digest: str,
    corrupt=None,
    correct_for_multiplicity: bool = False,
    part_digests_in_cluster_symbols: dict[str, str] | None = None,
    library_path: Path | None = None,
) -> dict[str, Any]:
    """One pass over the stream, reporting primitives and composites apart."""

    alphabet = int(clusters.shape[0])
    library = InducedProgramLibrary(
        alphabet=alphabet, frontend_digest=frontend_digest
    )
    rows: list[dict[str, Any]] = []
    for position, rule, repeat, kind, combiner, part_digests in stream:
        if not grow:
            library = InducedProgramLibrary(
                alphabet=alphabet, frontend_digest=frontend_digest
            )
        outcome, record = solve_task(
            payload,
            encoders,
            bank,
            library,
            rule,
            clusters,
            seed=seed + TASK_SEED_STRIDE * position,
            repeat_index=repeat,
            corrupt=corrupt,
            compose=compose,
            correct_for_multiplicity=correct_for_multiplicity,
        )
        if record is not None:
            admit(library, record, outcome, library_path=library_path)
        row = outcome.payload()
        row["kind"] = kind
        # Verifier-side annotations, for reading the table only.
        row["true_combiner"] = combiner
        row["true_parts"] = list(part_digests)
        # Did it reuse the parts the task was actually built from? Comparing
        # digests is the only way to tell composition from a lucky pair, and
        # the machines are read out of the library rather than assumed.
        if row["source"] == "composed" and row["composed_from"]:
            used = []
            for slot in row["composed_from"]:
                machine = machine_of(library.record(int(slot)))
                used.append(None if machine is None else machine.digest())
            row["used_parts"] = used
            expected = [
                (part_digests_in_cluster_symbols or {}).get(digest)
                for digest in part_digests
            ]
            row["parts_recovered"] = (
                all(expected)
                and sorted(digest for digest in used if digest) == sorted(expected)
            )
        else:
            row["used_parts"] = None
            row["parts_recovered"] = None
        rate = positive_rate(rule, seed=seed + TASK_SEED_STRIDE * position)
        row["positive_rate"] = rate
        row["trivial"] = max(rate, 1.0 - rate) >= THRESHOLD
        rows.append(row)

    def summarise(kind: str) -> dict[str, Any]:
        subset = [row for row in rows if row["kind"] == kind]
        hard = [row for row in subset if not row["trivial"]]
        return {
            "tasks": len(subset),
            "trivial": len(subset) - len(hard),
            "solved": sum(1 for row in subset if row["solved"]),
            "solved_nontrivial": sum(1 for row in hard if row["solved"]),
            "recognised": sum(1 for row in subset if row["source"] == "recognised"),
            "composed": sum(1 for row in subset if row["source"] == "composed"),
            "induced": sum(1 for row in subset if row["source"] == "induced"),
            "unsolved": sum(1 for row in subset if row["source"] == "unsolved"),
            "false_recognitions": sum(int(row["false_recognitions"]) for row in subset),
            "combiner_recovered": sum(
                1
                for row in subset
                if row["source"] == "composed"
                and row["combiner"] == row["true_combiner"]
            ),
            "parts_recovered": sum(
                1 for row in subset if row.get("parts_recovered")
            ),
            "acquisition_steps": sum(int(row["acquisition_steps"]) for row in subset),
            "verifier_steps": sum(int(row["verifier_steps"]) for row in subset),
            "max_hypotheses": max(
                (int(row["composition_hypotheses"]) for row in subset), default=0
            ),
        }

    return {
        "grew_the_library": grow,
        "composed": compose,
        "final_library_size": library.record_count,
        "primitives": summarise("primitive"),
        "composites": summarise("composite"),
        "rows": rows,
    }


def run_transfer(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    composites: int = COMPOSITES_PER_RUN,
    library_path: Path | None = None,
) -> dict[str, Any]:
    """Six arms over one matched stream of primitives and novel composites."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    bank = ExternalTemporalProgramBank.load_bank(bank_path)

    stream, learned, chosen = build_stream(seed=seed, composites=composites)
    apart, _, _ = build_stream(seed=seed, composites=composites, disjoint=True)
    clusters = discover_alphabet(
        encoders, _config(stream[0][1], CONFIRMATION_STEPS), seed=seed
    )
    digest = encoders.digest()

    # Scoring-side only: what each primitive's digest becomes once written in
    # the frontend's own symbol names, so "did it reuse the right part" can be
    # asked of machines that live in different alphabets.
    channel_of_symbol = cluster_symbol_map(
        encoders, clusters, symbol_count=learned[0].symbol_count
    )
    part_map = {
        primitive.digest(): in_cluster_symbols(primitive, channel_of_symbol).digest()
        for primitive in learned
    }

    def arm(name, source, *, grow, compose, corrupt=None, **kwargs):
        started = time.perf_counter()
        result = run_arm(
            payload,
            encoders,
            bank,
            source,
            clusters,
            grow=grow,
            compose=compose,
            seed=seed,
            frontend_digest=digest,
            corrupt=corrupt,
            part_digests_in_cluster_symbols=part_map,
            **kwargs,
        )
        result["arm"] = name
        result["seconds"] = time.perf_counter() - started
        return result

    started = time.perf_counter()
    arms = {
        "composing": arm(
            "composing",
            stream,
            grow=True,
            compose=True,
            library_path=library_path,
        ),
        "recognising": arm("recognising", stream, grow=True, compose=False),
        "control": arm("control", stream, grow=False, compose=False),
        "disjoint": arm("disjoint", apart, grow=True, compose=True),
        # The disjoint arm runs a *different* stream, so the control it has to
        # be read against is a control on that stream rather than on this one.
        "disjoint_control": arm(
            "disjoint_control", apart, grow=False, compose=False
        ),
        "shuffled": arm(
            "shuffled", stream, grow=True, compose=True, corrupt=shuffled_feedback(seed)
        ),
        # The default is uncorrected, so this arm is the *corrected* one.
        "corrected": arm(
            "corrected",
            stream,
            grow=True,
            compose=True,
            correct_for_multiplicity=True,
        ),
        # Composition on an unreliable verifier rather than a destroyed one.
        # Induction survives one label in five; composition is scored by a test
        # that does not move as the evidence gets dirtier, so it has to be
        # asked separately.
        "composing_noisy": arm(
            "composing_noisy",
            stream,
            grow=True,
            compose=True,
            corrupt=noisy_feedback(0.10, seed),
        ),
        "recognising_noisy": arm(
            "recognising_noisy",
            stream,
            grow=True,
            compose=False,
            corrupt=noisy_feedback(0.10, seed),
        ),
        # The place an uncorrected search should actually break: a library
        # that cannot build the task, still offering hundreds of hypotheses.
        # On a stream it *can* build, the naive threshold looks fine, which is
        # exactly why testing it only there would prove nothing.
        "disjoint_corrected": arm(
            "disjoint_corrected",
            apart,
            grow=True,
            compose=True,
            correct_for_multiplicity=True,
        ),
    }
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the compositional transfer run mutated AgentBrain.bank")

    def composite_cost(name: str) -> int:
        return int(arms[name]["composites"]["acquisition_steps"])

    report = {
        "schema": TRANSFER_SCHEMA,
        "noisy_composition_ratio": (
            composite_cost("composing_noisy") / composite_cost("recognising_noisy")
            if composite_cost("recognising_noisy")
            else None
        ),
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "primitive_count": len(learned),
        "composite_count": len(chosen),
        "composite_state_counts": [rule.automaton.state_count for rule in chosen],
        "composite_combiners": [rule.combiner for rule in chosen],
        "arms": arms,
        "composite_acquisition": {
            name: composite_cost(name) for name in arms
        },
        "composition_ratio_against_recognition": (
            composite_cost("composing") / composite_cost("recognising")
            if composite_cost("recognising")
            else None
        ),
        "composition_ratio_against_control": (
            composite_cost("composing") / composite_cost("control")
            if composite_cost("control")
            else None
        ),
        "disjoint_ratio_against_its_control": (
            composite_cost("disjoint") / composite_cost("disjoint_control")
            if composite_cost("disjoint_control")
            else None
        ),
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "compositional_transfer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--controller",
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
        "--output",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_compositional_transfer_2026-08-15"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--composites", type=int, default=COMPOSITES_PER_RUN)
    arguments = parser.parse_args()
    report = run_transfer(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
        composites=arguments.composites,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
