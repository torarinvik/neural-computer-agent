"""Try to make compositional recognition adopt something wrong.

The agent-level runs say composition pays and never once adopted a false
composite. Those runs are dozens of tasks. A search that scores a thousand
hypotheses for free needs to be attacked at a scale dozens of tasks cannot
reach, and it needs to be attacked where it is weakest rather than where it was
convenient to measure.

Four attacks, all offline. The search reads labelled traces, and after feedback
inversion that is exactly what the agent hands it, so nothing is lost by
skipping the environment -- and thousands of trials become affordable.

**Scale.** Grow the library with records the task has nothing to do with. The
hypothesis count is quadratic, so a library of twenty offers over six hundred
chances to be fooled per task. Does the false-adoption rate stay at zero?

**Noise.** Every composition measured so far ran on perfect feedback. Under
label noise the true composite no longer explains the evidence exactly, and the
competence test it has to clear does not move. Does composition degrade or
collapse?

**A vocabulary the world does not share.** Compose the task with an operator
the search does not have. The right answer is to find nothing. Finding
something means the pair space is rich enough to imitate an operator it lacks,
and every positive result would be suspect.

**The correction.** Corrected against uncorrected, at every scale and noise
level, counting adoptions that are wrong. The agent-level runs could not
separate them; this is built to.

An adoption is wrong when the thing adopted does not actually solve the task --
judged on clean held-out episodes the search never saw, not on the evidence it
was chosen with.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer.induced_library import (
    InducedProgramLibrary,
    InducedProgramRecord,
    canonical_signature_stream,
)

from .compositional_recognition import composed_machine, search_compositions
from .compositional_rules import COMBINERS, product_rule, sample_primitive_pool
from .counter_state_programs import compile_rule, initial_counters, predict_symbols
from .current_symbol_acquire import THRESHOLD
from .identification_ceiling import Trace
from .rule_automata import RuleAutomaton, sample_rule

EXPERIMENT_ID = "brainworkshop-composition-adversarial-2026-08-15"
ADVERSARIAL_SCHEMA = "neural-computer.composition-adversarial.v1"
ALPHABET = 4
PROBE_STEPS = 16
# The evidence budget the agent's ladder reaches before it would induce.
PROBE_EPISODES = 8
HELD_OUT_EPISODES = 20
HELD_OUT_STEPS = 64


def _record(machine: RuleAutomaton) -> InducedProgramRecord:
    program = compile_rule(
        machine, channel_of_symbol=tuple(range(ALPHABET)), cluster_count=ALPHABET
    )
    start = initial_counters(
        program, cluster_count=ALPHABET, states=machine.state_count
    )
    signature, _ = predict_symbols(
        program,
        canonical_signature_stream(ALPHABET),
        cluster_count=ALPHABET,
        initial_counters=start,
    )
    return InducedProgramRecord(
        program=program,
        initial_counters=start,
        alphabet=ALPHABET,
        signature=signature,
        provenance={"machine": machine.payload()},
    ).validate()


def traces_for(
    machine: RuleAutomaton,
    *,
    seed: int,
    episodes: int = PROBE_EPISODES,
    steps: int = PROBE_STEPS,
    noise: float = 0.0,
) -> tuple[Trace, ...]:
    """Labelled evidence of the kind feedback inversion produces."""

    generator = torch.Generator().manual_seed(int(seed))
    produced = []
    for _ in range(episodes):
        stream = torch.randint(0, ALPHABET, (steps,), generator=generator).tolist()
        labels = list(machine.expected(stream))
        if noise > 0:
            flips = (torch.rand(steps, generator=generator) < noise).tolist()
            labels = [value ^ int(flip) for value, flip in zip(labels, flips)]
        produced.append(
            Trace(
                symbols=tuple(stream),
                outputs=tuple(labels),
                eligible=tuple([True] * steps),
                symbol_count=ALPHABET,
            )
        )
    return tuple(produced)


def solves(candidate: RuleAutomaton, target: RuleAutomaton, *, seed: int) -> bool:
    """Judged on clean episodes the search never saw."""

    hits = trials = 0
    for trace in traces_for(
        target, seed=seed, episodes=HELD_OUT_EPISODES, steps=HELD_OUT_STEPS
    ):
        predicted = candidate.expected(list(trace.symbols))
        for index, label in enumerate(trace.outputs):
            trials += 1
            hits += int(predicted[index] == label)
    return trials > 0 and hits / trials >= THRESHOLD


def padded_library(
    pool: tuple[RuleAutomaton, ...], *, size: int, seed: int
) -> InducedProgramLibrary:
    """The pool, plus unrelated records, up to a size.

    Padding is what makes the search large. The extra records cannot help with
    anything, so every hypothesis they add is purely a chance to be wrong.
    """

    library = InducedProgramLibrary(alphabet=ALPHABET)
    for machine in pool:
        library.append(_record(machine))
    attempt = 0
    while library.record_count < size and attempt < size * 200:
        machine = sample_rule(
            symbol_count=ALPHABET,
            state_count=2 + (attempt % 3),
            seed=seed + attempt,
        )
        attempt += 1
        if machine is None:
            continue
        if library.duplicate_of(_record(machine).signature) is not None:
            continue
        library.append(_record(machine))
    return library


def trial(
    library: InducedProgramLibrary,
    target: RuleAutomaton,
    *,
    seed: int,
    noise: float,
    corrected: bool,
) -> dict[str, Any]:
    """One search, and whether what it returned actually solves the task."""

    evidence = traces_for(target, seed=seed, noise=noise)
    found, report = search_compositions(
        library, evidence, correct_for_multiplicity=corrected
    )
    if found is None:
        return {"adopted": False, "correct": False, "hypotheses": report["hypotheses"]}
    built = composed_machine(library, found)
    good = built is not None and solves(built, target, seed=seed + 7717)
    return {
        "adopted": True,
        "correct": bool(good),
        "kind": found.kind,
        "hypotheses": report["hypotheses"],
    }


def run_adversarial(
    output_directory: Path,
    *,
    seed: int = 41,
    library_sizes: tuple[int, ...] = (4, 8, 16, 24),
    noise_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20),
    targets_per_cell: int = 12,
) -> dict[str, Any]:
    """Every attack, corrected and uncorrected, over the whole grid."""

    pool = sample_primitive_pool(symbol_count=ALPHABET, count=4, seed=8000)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []

    for size in library_sizes:
        library = padded_library(pool, size=size, seed=60_000 + size)
        for noise in noise_levels:
            for corrected in (True, False):
                buildable = {"adopted": 0, "correct": 0, "wrong": 0, "missed": 0}
                unbuildable = {"adopted": 0, "wrong": 0, "refused": 0}
                foreign = {"adopted": 0, "wrong": 0, "refused": 0}
                hypotheses = 0
                for index in range(targets_per_cell):
                    left = pool[index % len(pool)]
                    right = pool[(index + 1) % len(pool)]
                    combiner = ("and", "or", "xor")[index % 3]

                    # 1. A composite the library can build.
                    target = product_rule(left, right, combiner)
                    outcome = trial(
                        library,
                        target,
                        seed=seed + 31 * index,
                        noise=noise,
                        corrected=corrected,
                    )
                    hypotheses = max(hypotheses, int(outcome["hypotheses"]))
                    if outcome["adopted"]:
                        buildable["adopted"] += 1
                        buildable["correct" if outcome["correct"] else "wrong"] += 1
                    else:
                        buildable["missed"] += 1

                    # 2. A rule with nothing to do with the library.
                    stranger = sample_rule(
                        symbol_count=ALPHABET,
                        state_count=3,
                        seed=400_000 + 13 * index,
                    )
                    if stranger is not None:
                        outcome = trial(
                            library,
                            stranger,
                            seed=seed + 53 * index,
                            noise=noise,
                            corrected=corrected,
                        )
                        if outcome["adopted"]:
                            unbuildable["adopted"] += 1
                            if not outcome["correct"]:
                                unbuildable["wrong"] += 1
                        else:
                            unbuildable["refused"] += 1

                    # 3. Composed with an operator the search does not have.
                    implied = _implication(left, right)
                    outcome = trial(
                        library,
                        implied,
                        seed=seed + 71 * index,
                        noise=noise,
                        corrected=corrected,
                    )
                    if outcome["adopted"]:
                        foreign["adopted"] += 1
                        if not outcome["correct"]:
                            foreign["wrong"] += 1
                    else:
                        foreign["refused"] += 1

                rows.append(
                    {
                        "library_size": size,
                        "noise": noise,
                        "corrected": corrected,
                        "max_hypotheses": hypotheses,
                        "buildable": buildable,
                        "unbuildable": unbuildable,
                        "foreign_operator": foreign,
                    }
                )

    def total(corrected: bool, key: str, field: str) -> int:
        return sum(
            int(row[key][field]) for row in rows if row["corrected"] is corrected
        )

    report = {
        "schema": ADVERSARIAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "library_sizes": list(library_sizes),
        "noise_levels": list(noise_levels),
        "targets_per_cell": targets_per_cell,
        "max_hypotheses": max(int(row["max_hypotheses"]) for row in rows),
        "corrected": {
            "buildable_correct": total(True, "buildable", "correct"),
            "buildable_wrong": total(True, "buildable", "wrong"),
            "buildable_missed": total(True, "buildable", "missed"),
            "unbuildable_wrong": total(True, "unbuildable", "wrong"),
            "foreign_wrong": total(True, "foreign_operator", "wrong"),
        },
        "uncorrected": {
            "buildable_correct": total(False, "buildable", "correct"),
            "buildable_wrong": total(False, "buildable", "wrong"),
            "buildable_missed": total(False, "buildable", "missed"),
            "unbuildable_wrong": total(False, "unbuildable", "wrong"),
            "foreign_wrong": total(False, "foreign_operator", "wrong"),
        },
        "rows": rows,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "composition_adversarial.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _implication(left: RuleAutomaton, right: RuleAutomaton) -> RuleAutomaton:
    """`left implies right`: a combiner the search deliberately lacks."""

    original = dict(COMBINERS)
    COMBINERS["implies"] = lambda a, b: int((not a) or bool(b))
    try:
        return product_rule(left, right, "implies")
    finally:
        COMBINERS.clear()
        COMBINERS.update(original)


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            repository
            / "session_records"
            / "brainworkshop_compositional_transfer_2026-08-15"
        ),
    )
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--targets", type=int, default=12)
    arguments = parser.parse_args()
    report = run_adversarial(
        arguments.output, seed=arguments.seed, targets_per_cell=arguments.targets
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
