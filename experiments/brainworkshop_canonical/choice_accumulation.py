"""Does the library still pay when the answer is a choice?

`choice_ceiling` shows a `k`-action rule can be *learned*. That is a fitter
result: the hypothesis was stepped against the verifier and thrown away. An
agent has to compile it, keep it, and find it again -- and every one of those
steps was built for a world with two answers.

Three things had to widen, and each is additive so that nothing binary moves:

- the **counter ABI**, which reserved counter zero for a press and now uses one
  counter per action with the largest read as the answer;
- the **library record**, which now carries an action count, written to disk
  only when it is not two, so all six libraries already committed load and
  digest byte-identically;
- **recognition**, which cannot simply test a consistency rate, because that
  rate's floor climbs with the action set -- a machine answering at chance is
  consistent with 0.500 of outcomes at two actions and 0.625 at four. The rate
  is inverted into the accuracy it implies before it is tested, and the trial
  count is discounted for the noise that inversion adds.

The measurement is the same one the binary library was held to: a stream with
repeats, a growing arm against a matched control, and the cost counted in
episodes bought rather than in anything cheaper.
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
from neural_computer.promotion import sha256_file

from .accumulation_curve import _config
from .choice_agent import probe_episode, run_choice_program_episode
from .choice_induction import (
    agreement,
    implied_accuracy,
    induce_from_choices,
)
from .choice_programs import (
    choice_initial_counters,
    compile_choice_rule,
    predict_choice_symbols,
)
from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import FRONTEND_SEED, THRESHOLD, _machine, curated_frontend
from .integrated_agent import (
    TASK_SEED_STRIDE,
    discover_alphabet,
    proves_competence,
)
from .rule_automata import best_constant_rate, sample_rule

EXPERIMENT_ID = "brainworkshop-choice-accumulation-2026-08-15"
CHOICE_ACCUMULATION_SCHEMA = "neural-computer.choice-accumulation.v1"
DEVELOPMENT_SEED = 41
PROBE_STEPS = 16
CONFIRMATION_STEPS = 448
CONFIRMATION_EPISODES = 2
PROBE_LADDER = (2, 4, 8, 16, 28, 56)
INDUCTION_LADDER_INDEX = 2
MAX_CONSTANT_RATE = 0.6


def record_for(machine, *, alphabet: int, provenance: dict[str, Any]):
    """Compile a `k`-action hypothesis into a storable program file."""

    program = compile_choice_rule(machine, cluster_count=alphabet)
    start = choice_initial_counters(machine, cluster_count=alphabet)
    signature, statuses = predict_choice_symbols(
        program,
        canonical_signature_stream(alphabet),
        action_count=machine.action_count,
        cluster_count=alphabet,
        initial_counters=start,
    )
    if statuses != ("halted",):
        raise RuntimeError(f"compiled program did not halt cleanly: {statuses}")
    return InducedProgramRecord(
        program=program,
        initial_counters=start,
        alphabet=int(alphabet),
        signature=signature,
        provenance=dict(provenance),
        action_count=int(machine.action_count),
    ).validate()


def record_answers(record: InducedProgramRecord, traces):
    """What a stored program would have answered, at the scored steps."""

    answered = []
    for trace in traces:
        answers, _ = predict_choice_symbols(
            record.program,
            trace.symbols,
            action_count=record.action_count,
            cluster_count=record.alphabet,
            initial_counters=record.initial_counters,
        )
        answered.append(answers)
    return answered


def recognise_choice(
    library: InducedProgramLibrary,
    traces,
    *,
    threshold: float = THRESHOLD,
    exclude: frozenset[int] = frozenset(),
) -> tuple[int, float] | None:
    """The stored program the evidence positively supports, if any.

    Consistency is inverted into implied accuracy before it is tested, and the
    trials are discounted for the noise that inversion adds -- otherwise a
    coin flip at four actions walks in at 0.625.
    """

    best: tuple[int, float] | None = None
    for slot in range(library.record_count):
        if slot in exclude:
            continue
        record = library.record(slot)
        answered = record_answers(record, traces)
        consistent = trials = 0
        for answers, trace in zip(answered, traces, strict=True):
            for index, flag in enumerate(trace.eligible):
                if not flag:
                    continue
                trials += 1
                chose = int(trace.actions[index])
                if trace.rewards[index]:
                    consistent += int(answers[index] == chose)
                else:
                    consistent += int(answers[index] != chose)
        accuracy, effective = implied_accuracy(
            consistent, trials, record.action_count
        )
        if not proves_competence(
            round(accuracy * effective), effective, threshold=threshold
        ):
            continue
        if best is None or accuracy > best[1]:
            best = (slot, accuracy)
    return best


def solve_choice_task(
    encoders,
    library: InducedProgramLibrary,
    rule,
    clusters: torch.Tensor,
    *,
    seed: int,
    ladder: tuple[int, ...] = PROBE_LADDER,
    induction_from: int = INDUCTION_LADDER_INDEX,
) -> dict[str, Any]:
    """Buy evidence a rung at a time; recognise, else induce; confirm; keep."""

    alphabet = int(clusters.shape[0])
    action_count = rule.action_count
    probe_config = _config(rule, PROBE_STEPS)
    full = _config(rule, CONFIRMATION_STEPS)
    outcome: dict[str, Any] = {
        "rule_digest": rule.digest(),
        "state_count": rule.state_count,
        "action_count": action_count,
        "library_size_before": library.record_count,
        "probe_episodes": 0,
        "confirmation_episodes": 0,
        "source": "unsolved",
        "library_slot": None,
        "solved": False,
        "admitted": False,
        "false_recognitions": 0,
        "pooled_accuracy": None,
        "best_constant": best_constant_rate(rule, seed=seed),
    }

    traces = []
    refused: set[int] = set()
    attempt = 0
    for rung, budget in enumerate(ladder):
        while len(traces) < budget:
            traces.append(
                probe_episode(
                    encoders,
                    probe_config,
                    clusters,
                    seed=seed + 1000 + len(traces),
                    policy_seed=seed + 7000 + len(traces),
                    action_count=action_count,
                )
            )
        outcome["probe_episodes"] = len(traces)

        candidate = None
        source = ""
        slot = None
        found = recognise_choice(library, traces, exclude=frozenset(refused))
        if found is not None:
            slot, _ = found
            candidate = library.record(slot)
            source = "recognised"
        elif rung >= induction_from:
            fit = induce_from_choices(tuple(traces))
            if fit is not None:
                consistent, trials = agreement(fit.machine, traces)
                implied, _ = implied_accuracy(consistent, trials, action_count)
                outcome["fit_implied_accuracy"] = implied
                outcome["fit_states"] = fit.machine.state_count
                if implied >= THRESHOLD or budget == ladder[-1]:
                    candidate = record_for(
                        fit.machine,
                        alphabet=alphabet,
                        provenance={
                            "source": "induced",
                            "probe_episodes": len(traces),
                            "states": fit.machine.state_count,
                            "actions": action_count,
                            "machine": fit.machine.payload(),
                        },
                    )
                    source = "induced"
        if candidate is None:
            continue

        accuracies = []
        for offset in range(CONFIRMATION_EPISODES):
            executed = run_choice_program_episode(
                candidate, encoders, full, clusters, seed=seed + 100 + 10 * attempt + offset
            )
            accuracies.append(float(executed["accuracy"]))
        attempt += 1
        outcome["confirmation_episodes"] += CONFIRMATION_EPISODES
        pooled = sum(accuracies) / len(accuracies)
        trials = CONFIRMATION_EPISODES * CONFIRMATION_STEPS
        outcome["pooled_accuracy"] = pooled
        if proves_competence(round(pooled * trials), trials, threshold=THRESHOLD):
            outcome["source"] = source
            outcome["library_slot"] = slot
            outcome["solved"] = True
            if source == "induced":
                duplicate = library.duplicate_of(candidate.signature)
                if duplicate is None:
                    outcome["library_slot"] = library.append(candidate)
                    outcome["admitted"] = True
                else:
                    outcome["library_slot"] = duplicate
            return outcome
        if source == "recognised" and slot is not None:
            refused.add(slot)
            outcome["false_recognitions"] += 1

    return outcome


def run_choice_accumulation(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    action_counts: tuple[int, ...] = (2, 3, 4),
    pool_size: int = 3,
    stream_length: int = 9,
) -> dict[str, Any]:
    """A stream with repeats, at each action count, growing against control."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    anchor = sample_rule(symbol_count=4, state_count=2, seed=1)
    clusters = discover_alphabet(
        encoders, _config(anchor, CONFIRMATION_STEPS), seed=seed
    )
    alphabet = int(clusters.shape[0])

    started = time.perf_counter()
    cells: list[dict[str, Any]] = []
    for action_count in action_counts:
        pool = []
        for index in range(pool_size):
            rule = sample_rule(
                symbol_count=4,
                state_count=1 + index,
                seed=4400 + 10 * action_count + index,
                action_count=action_count,
                maximum_constant_rate=MAX_CONSTANT_RATE,
            )
            if rule is not None:
                pool.append(rule)
        generator = torch.Generator().manual_seed(seed + action_count)
        draws = torch.randint(
            0, len(pool), (stream_length,), generator=generator
        ).tolist()

        arms: dict[str, Any] = {}
        for grow in (True, False):
            library = InducedProgramLibrary(alphabet=alphabet)
            rows = []
            for position, choice in enumerate(draws):
                if not grow:
                    library = InducedProgramLibrary(alphabet=alphabet)
                rows.append(
                    solve_choice_task(
                        encoders,
                        library,
                        pool[choice],
                        clusters,
                        seed=seed + TASK_SEED_STRIDE * position,
                    )
                )
            arms["growing" if grow else "control"] = {
                "tasks": len(rows),
                "solved": sum(1 for row in rows if row["solved"]),
                "recognised": sum(1 for row in rows if row["source"] == "recognised"),
                "induced": sum(1 for row in rows if row["source"] == "induced"),
                "admitted": sum(1 for row in rows if row["admitted"]),
                "false_recognitions": sum(
                    int(row["false_recognitions"]) for row in rows
                ),
                "probe_episodes": sum(int(row["probe_episodes"]) for row in rows),
                "acquisition_steps": sum(
                    int(row["probe_episodes"]) * PROBE_STEPS for row in rows
                ),
                "final_library_size": library.record_count,
                "rows": rows,
            }
        cells.append(
            {
                "action_count": action_count,
                "distinct_tasks": len({rule.digest() for rule in pool}),
                "acquisition_ratio": (
                    arms["growing"]["acquisition_steps"]
                    / arms["control"]["acquisition_steps"]
                    if arms["control"]["acquisition_steps"]
                    else None
                ),
                "arms": arms,
            }
        )

    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the choice accumulation run mutated AgentBrain.bank")

    report = {
        "schema": CHOICE_ACCUMULATION_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "alphabet": alphabet,
        "cells": cells,
        "false_recognitions": sum(
            int(cell["arms"]["growing"]["false_recognitions"]) for cell in cells
        ),
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "choice_accumulation.json").write_text(
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
            repository / "session_records" / "brainworkshop_choice_ceiling_2026-08-15"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    arguments = parser.parse_args()
    report = run_choice_accumulation(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
