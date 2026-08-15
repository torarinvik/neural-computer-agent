"""Does capability N+1 get cheaper when the parts are already known?

The accumulation curve asked this three times and got the same answer: no
composes, no reuse beyond exact duplicates. Every reading of that treated it
as a fact about the architecture. It was a fact about the sampler. Rules drawn
independently share nothing, so a library can only ever help by already
containing the answer, which is the reuse that was measured.

`compositional_rules` builds a distribution where the question has a chance:
primitives sampled once into a shared pool, tasks formed as products of pairs
under a boolean combiner. A composite of two three-state primitives is a
harder task than either -- up to nine states -- and a decomposable one.

The curriculum presents the primitives first and then the composites, which is
what a library is *for*. Both arms use the same evidence ladder: 7 short
episodes, then 14, 28, 56, 112, stopping as soon as the task is identified and
verified on a held-out episode. They differ in one thing.

- **control** induces from scratch at every rung.
- **growing** first checks whether anything it already knows explains the
  evidence -- a library machine outright, or a product of two library machines
  under a combiner -- and only induces if nothing does.

Checking the library is free. A candidate machine's predictions on an already
observed trace cost no verifier evidence at all, which is the same argument
that made behavioural dedup free in the proposer record. So any difference
between the arms is evidence the library actually carried, not accounting.

Cost is labelled steps, not episodes, because the two arms may stop at
different rungs and an episode is not a fixed amount of evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file

from .compositional_rules import (
    COMBINERS,
    product_rule,
    sample_composite_population,
    sample_primitive_pool,
)
from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .identification_ceiling import (
    NODE_BUDGET,
    episode_trace,
    held_out_accuracy,
    infer_machine,
)
from .machine_factorization import factorize
from .prototype_templates import cluster_events, observe_events
from .rendered_environment import RenderedBrainWorkshopConfig
from .rule_automata import RuleAutomaton

EXPERIMENT_ID = "brainworkshop-composition-accumulation-2026-08-15"
COMPOSITION_SCHEMA = "neural-computer.composition-accumulation.v1"
DEVELOPMENT_SEED = 41
EPISODE_STEPS = 16
LADDER = (7, 14, 28, 56, 112)
EVALUATION_STEPS = 448


def _config(rule: RuleAutomaton, steps: int) -> RenderedBrainWorkshopConfig:
    return RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps,
        streams=("vision",),
        symbol_count=rule.symbol_count,
        match_rule="automaton",
        rule=rule,
    )


def library_candidates(
    library: tuple[RuleAutomaton, ...],
) -> tuple[tuple[str, RuleAutomaton], ...]:
    """Everything the library can offer: its files, and their products.

    Enumerated rather than searched, because the library is small and each
    candidate is checked offline. The label records which parts were used, so
    a win can be attributed instead of assumed.
    """

    candidates: list[tuple[str, RuleAutomaton]] = []
    for index, machine in enumerate(library):
        candidates.append((f"retrieve:{index}", machine))
    for left, right in combinations(range(len(library)), 2):
        for combiner in COMBINERS:
            try:
                merged = product_rule(library[left], library[right], combiner)
            except ValueError:
                continue
            candidates.append((f"{combiner}:{left}+{right}", merged))
    return tuple(candidates)


TRIVIAL_PART_CACHE: dict[int, RuleAutomaton] = {}


def trivial_part(symbol_count: int) -> RuleAutomaton:
    """A one-state component, so a single part can be used on its own."""

    if symbol_count not in TRIVIAL_PART_CACHE:
        TRIVIAL_PART_CACHE[symbol_count] = RuleAutomaton(
            symbol_count=symbol_count,
            transitions=((0,) * symbol_count,),
            outputs=((0,) * symbol_count,),
        ).validate()
    return TRIVIAL_PART_CACHE[symbol_count]


def fit_output_table(
    left: RuleAutomaton, right: RuleAutomaton, traces
) -> dict[tuple[int, int, int], int] | None:
    """Fill the output table two components imply, or None on a conflict.

    This is the step that replaces enumerating combiners. Every observation
    names one cell, so the table is *read off* the evidence in linear time
    instead of being guessed from a list an experimenter wrote. It can express
    any output function of the two blocks and the symbol, of which `and`, `or`
    and `xor` are three.
    """

    table: dict[tuple[int, int, int], int] = {}
    for trace in traces:
        first = 0
        second = 0
        for position, symbol in enumerate(trace.symbols):
            if trace.eligible[position]:
                cell = (first, second, int(symbol))
                want = int(trace.outputs[position])
                seen = table.get(cell)
                if seen is None:
                    table[cell] = want
                elif seen != want:
                    return None
            first = int(left.transitions[first][int(symbol)])
            second = int(right.transitions[second][int(symbol)])
    return table


def table_accuracy(
    left: RuleAutomaton,
    right: RuleAutomaton,
    table: dict[tuple[int, int, int], int],
    trace,
) -> float:
    """How well a fitted pair predicts an episode it was not fitted on."""

    first = 0
    second = 0
    hits = 0
    trials = 0
    for position, symbol in enumerate(trace.symbols):
        if trace.eligible[position]:
            trials += 1
            if table.get((first, second, int(symbol)), 0) == int(
                trace.outputs[position]
            ):
                hits += 1
        first = int(left.transitions[first][int(symbol)])
        second = int(right.transitions[second][int(symbol)])
    return hits / trials if trials else 0.0


def parts_of(machine: RuleAutomaton) -> tuple[RuleAutomaton, ...]:
    """The components a machine decomposes into, or the machine itself."""

    found = factorize(machine)
    if not found:
        return (machine,)
    best = found[0]
    return (best.left, best.right)


def _fits(machine: RuleAutomaton, traces) -> bool:
    """Whether a candidate reproduces every labelled step it has been shown."""

    return all(held_out_accuracy(machine, trace) == 1.0 for trace in traces)


def learn_one_task(
    payload: dict[str, object],
    encoders,
    bank: ExternalTemporalProgramBank,
    rule: RuleAutomaton,
    *,
    seed: int,
    library: tuple[RuleAutomaton, ...],
    use_library: str,
    node_budget: int = NODE_BUDGET,
) -> dict[str, Any]:
    """Climb the evidence ladder until the task is identified, or give up."""

    evaluation = _config(rule, EVALUATION_STEPS)
    clusters = cluster_events(observe_events(encoders, evaluation, seed=seed))
    held_out = episode_trace(
        payload, encoders, bank, evaluation, clusters, seed=seed + 1
    )
    short = replace(evaluation, steps=EPISODE_STEPS).validate()
    candidates = (
        library_candidates(library) if use_library == "products" else ()
    )
    traces: list[Any] = []
    collected = 0
    for rung in LADDER:
        while collected < rung:
            traces.append(
                episode_trace(
                    payload,
                    encoders,
                    bank,
                    short,
                    clusters,
                    seed=seed + 1000 + collected,
                )
            )
            collected += 1
        spent = collected * EPISODE_STEPS
        for label, candidate in candidates:
            if not _fits(candidate, traces):
                continue
            accuracy = held_out_accuracy(candidate, held_out)
            if accuracy == 1.0:
                return {
                    "identified": True,
                    "source": "library",
                    "label": label,
                    "labelled_steps": spent,
                    "episodes": collected,
                    "held_out_accuracy": accuracy,
                    "state_count": candidate.state_count,
                    "machine": candidate,
                }
        if use_library == "factored" and library:
            pool = (trivial_part(rule.symbol_count), *library)
            for left in range(len(pool)):
                for right in range(left, len(pool)):
                    table = fit_output_table(pool[left], pool[right], traces)
                    if table is None:
                        continue
                    accuracy = table_accuracy(
                        pool[left], pool[right], table, held_out
                    )
                    if accuracy == 1.0:
                        return {
                            "identified": True,
                            "source": "library",
                            "label": f"fit:{left - 1}+{right - 1}",
                            "labelled_steps": spent,
                            "episodes": collected,
                            "held_out_accuracy": accuracy,
                            "state_count": (
                                pool[left].state_count * pool[right].state_count
                            ),
                            "machine": None,
                        }
        machine = infer_machine(tuple(traces), node_budget=node_budget)
        if machine is not None:
            accuracy = held_out_accuracy(machine, held_out)
            if accuracy == 1.0:
                return {
                    "identified": True,
                    "source": "induced",
                    "label": "induce",
                    "labelled_steps": spent,
                    "episodes": collected,
                    "held_out_accuracy": accuracy,
                    "state_count": machine.state_count,
                    "machine": machine,
                }
    return {
        "identified": False,
        "source": None,
        "label": None,
        "labelled_steps": collected * EPISODE_STEPS,
        "episodes": collected,
        "held_out_accuracy": None,
        "state_count": None,
        "machine": None,
    }


def run_arm(
    payload: dict[str, object],
    encoders,
    bank: ExternalTemporalProgramBank,
    curriculum,
    *,
    use_library: str,
    seed: int,
    node_budget: int = NODE_BUDGET,
) -> dict[str, Any]:
    """One pass over the curriculum, with or without a growing library."""

    library: list[RuleAutomaton] = []
    rows: list[dict[str, Any]] = []
    for kind, rule, parts in curriculum:
        result = learn_one_task(
            payload,
            encoders,
            bank,
            rule,
            seed=seed,
            library=tuple(library),
            use_library=use_library,
            node_budget=node_budget,
        )
        machine = result.pop("machine")
        result["kind"] = kind
        result["true_state_count"] = rule.state_count
        result["parts_in_library"] = sum(
            1
            for part in parts
            if any(item.digest() == part for item in library)
        )
        rows.append(result)
        if use_library == "products" and machine is not None:
            if all(item.digest() != machine.digest() for item in library):
                library.append(machine)
        elif use_library == "factored" and machine is not None:
            # Store the solution *and* its parts. Storing only the parts was
            # tried first and lost: nine whole machines collapse to three
            # distinct components, and three components cannot span the task
            # space. Factoring is meant to extend the vocabulary, not replace
            # it, so the library keeps both and the pool strictly contains
            # what the products arm has.
            for item in (machine, *parts_of(machine)):
                if all(held.digest() != item.digest() for held in library):
                    library.append(item)
    return {
        "used_library": use_library,
        "mode": use_library,
        "library_size": len(library),
        "identified": sum(1 for row in rows if row["identified"]),
        "of": len(rows),
        "labelled_steps": sum(int(row["labelled_steps"]) for row in rows),
        "from_library": sum(1 for row in rows if row["source"] == "library"),
        "tasks": rows,
    }


def run_composition_accumulation(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    node_budget: int = NODE_BUDGET,
) -> dict[str, Any]:
    """Both arms over a curriculum of primitives followed by composites."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    pool = sample_primitive_pool()
    composites = sample_composite_population(pool)
    curriculum = [("primitive", item, ()) for item in pool]
    curriculum += [
        ("composite", item.automaton, item.part_digests()) for item in composites
    ]
    started = time.perf_counter()
    growing = run_arm(
        payload, encoders, bank, curriculum,
        use_library="products", seed=seed, node_budget=node_budget,
    )
    factored = run_arm(
        payload, encoders, bank, curriculum,
        use_library="factored", seed=seed, node_budget=node_budget,
    )
    control = run_arm(
        payload, encoders, bank, curriculum,
        use_library="none", seed=seed, node_budget=node_budget,
    )
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("composition accumulation mutated AgentBrain.bank")
    composite_rows = [
        (grown, fixed)
        for grown, fixed in zip(growing["tasks"], control["tasks"])
        if grown["kind"] == "composite"
    ]
    grown_steps = sum(int(row["labelled_steps"]) for row, _ in composite_rows)
    fixed_steps = sum(int(row["labelled_steps"]) for _, row in composite_rows)
    report = {
        "schema": COMPOSITION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic",
        "note": (
            "tasks are products of sampled primitives, so a library has "
            "something to offer; nothing is admitted and no rule is read by "
            "the learner"
        ),
        "bank_sha256": before,
        "bank_unchanged": after == before,
        "seed": seed,
        "episode_steps": EPISODE_STEPS,
        "ladder": list(LADDER),
        "primitives": len(pool),
        "composites": len(composites),
        "growing": growing,
        "factored": factored,
        "control": control,
        "composite_labelled_steps_factored": sum(
            int(row["labelled_steps"])
            for row in factored["tasks"]
            if row["kind"] == "composite"
        ),
        "factored_cost_ratio": (
            sum(
                int(row["labelled_steps"])
                for row in factored["tasks"]
                if row["kind"] == "composite"
            )
            / fixed_steps
            if fixed_steps
            else None
        ),
        "composites_solved_from_parts": sum(
            1
            for row in factored["tasks"]
            if row["kind"] == "composite" and row["source"] == "library"
        ),
        "composite_labelled_steps_growing": grown_steps,
        "composite_labelled_steps_control": fixed_steps,
        "composite_cost_ratio": (
            grown_steps / fixed_steps if fixed_steps else None
        ),
        "composites_solved_from_library": sum(
            1 for row, _ in composite_rows if row["source"] == "library"
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "composition.json").write_text(
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
            / "brainworkshop_composition_accumulation_2026-08-15"
        ),
    )
    parser.add_argument("--node-budget", type=int, default=NODE_BUDGET)
    arguments = parser.parse_args()
    report = run_composition_accumulation(
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
                "growing": f"{report['growing']['identified']}/{report['growing']['of']}",
                "factored": f"{report['factored']['identified']}/{report['factored']['of']}",
                "control": f"{report['control']['identified']}/{report['control']['of']}",
                "composite_steps_factored": report[
                    "composite_labelled_steps_factored"
                ],
                "factored_cost_ratio": report["factored_cost_ratio"],
                "composites_solved_from_parts": report[
                    "composites_solved_from_parts"
                ],
                "composite_steps_growing": report["composite_labelled_steps_growing"],
                "composite_steps_control": report["composite_labelled_steps_control"],
                "composite_cost_ratio": report["composite_cost_ratio"],
                "composites_solved_from_library": report[
                    "composites_solved_from_library"
                ],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
