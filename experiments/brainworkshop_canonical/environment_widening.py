"""Change the world, not the agent, and find out what was an artifact.

Everything measured in this session ran in one environment: four positions on a
grid, rendered identically every time they appear, over a single visual stream.
Four well-separated stimuli is a generous world. Clustering them is trivial --
the largest within-cluster distance is 0.001 and the smallest between-cluster
distance is 4.64 -- and every later stage inherits that.

So the honest question is not whether the agent can do more. It is which of the
results so far were about the agent and which were about the room. Two axes are
moved here, one at a time, with the agent untouched:

**Alphabet.** Four symbols to eight. This is the axis the machinery is most
obviously tuned to: a machine's table is states times symbols, the counter
layout allocates one input channel per cluster, and the induced program's
instruction count grows with both.

**Stimulus noise.** Pixel noise drawn fresh at every observation, so the same
symbol never looks the same twice. This attacks the assumption the rest of the
stack rests on rather than any individual stage: the alphabet is *discovered*
by greedy distance clustering, and if that discovery returns the wrong number
of letters, nothing downstream can recover -- the traces are in a language the
programs do not speak.

The failure mode to watch for is therefore not lower accuracy. It is the
alphabet coming out the wrong size, which is reported directly rather than
being left to show up as an unexplained collapse.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from neural_computer import ExternalTemporalProgramBank
from neural_computer.promotion import sha256_file

from .accumulation_curve import _config
from .controller_pretraining import load_temporal_controller_artifact
from .current_symbol_acquire import FRONTEND_SEED, _machine, curated_frontend
from .integrated_agent import (
    CONFIRMATION_STEPS,
    discover_alphabet,
    run_arm,
    task_stream,
)
from .rule_automata import sample_rule

EXPERIMENT_ID = "brainworkshop-environment-widening-2026-08-15"
WIDENING_SCHEMA = "neural-computer.environment-widening.v1"
DEVELOPMENT_SEED = 41
ALPHABETS = (4, 6, 8)
# Pushed past where it works on purpose. A sweep that stops at the last level
# that succeeds reports a floor rather than a boundary.
NOISE_LEVELS = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
STATE_COUNTS = (1, 2, 3, 4, 5, 6)


def widened_pool(alphabet: int, *, pool_size: int) -> tuple:
    """One rule per state count, at this alphabet size."""

    rules = []
    for index in range(pool_size):
        states = STATE_COUNTS[index % len(STATE_COUNTS)]
        rule = sample_rule(
            symbol_count=alphabet,
            state_count=states,
            seed=8000 + 100 * alphabet + 10 * states + index,
        )
        if rule is not None:
            rules.append(rule)
    return tuple(rules)


def config_factory(noise: float):
    """The same task, rendered in a harder room."""

    def build(rule, steps: int):
        return replace(_config(rule, steps), frame_noise=float(noise)).validate()

    return build


def run_cell(
    payload: dict[str, object],
    encoders,
    bank: ExternalTemporalProgramBank,
    *,
    alphabet: int,
    noise: float,
    seed: int,
    stream_length: int,
    pool_size: int,
) -> dict[str, Any]:
    """One (alphabet, noise) point of the grid, both arms."""

    rules = widened_pool(alphabet, pool_size=pool_size)
    config_for = config_factory(noise)
    stream = task_stream(
        rules, length=stream_length, pool_size=len(rules), seed=seed
    )
    try:
        clusters = discover_alphabet(
            encoders, config_for(rules[0], CONFIRMATION_STEPS), seed=seed
        )
    except ValueError:
        # The estimator refused to name an alphabet. That is a result.
        return {
            "alphabet": alphabet,
            "noise": noise,
            "discovered_alphabet": None,
            "alphabet_recovered": False,
            "solved": None,
            "acquisition_ratio": None,
            "failure": "the stimuli do not separate into an alphabet",
        }
    discovered = int(clusters.shape[0])
    row: dict[str, Any] = {
        "alphabet": alphabet,
        "noise": noise,
        "discovered_alphabet": discovered,
        "alphabet_recovered": discovered == alphabet,
        "distinct_tasks": len({rule.digest() for _, rule, _ in stream}),
    }
    if discovered != alphabet:
        # Nothing downstream can be meaningful: the traces are written in an
        # alphabet the programs do not share. Say so rather than reporting the
        # collapse as an accuracy.
        row["solved"] = None
        row["acquisition_ratio"] = None
        row["failure"] = "the frontend did not recover the alphabet"
        return row

    growing = run_arm(
        payload,
        encoders,
        bank,
        stream,
        clusters,
        grow=True,
        seed=seed,
        frontend_digest=encoders.digest(),
        config_for=config_for,
    )
    control = run_arm(
        payload,
        encoders,
        bank,
        stream,
        clusters,
        grow=False,
        seed=seed,
        frontend_digest=encoders.digest(),
        config_for=config_for,
    )
    row.update(
        {
            "tasks": growing["tasks"],
            "trivial_tasks": growing["trivial_tasks"],
            "solved": growing["solved"],
            "solved_nontrivial": growing["solved_nontrivial"],
            "nontrivial_tasks": growing["nontrivial_tasks"],
            "control_solved": control["solved"],
            "recognised": growing["recognised"],
            "induced": growing["induced"],
            "admitted": growing["admitted"],
            "false_recognitions": growing["false_recognitions"],
            "growing_acquisition": growing["total_acquisition_steps"],
            "control_acquisition": control["total_acquisition_steps"],
            "acquisition_ratio": (
                growing["total_acquisition_steps"]
                / control["total_acquisition_steps"]
                if control["total_acquisition_steps"]
                else None
            ),
            "failure": None,
        }
    )
    return row


def run_widening(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    stream_length: int = 12,
    pool_size: int = 4,
    alphabets: tuple[int, ...] = ALPHABETS,
    noise_levels: tuple[float, ...] = NOISE_LEVELS,
) -> dict[str, Any]:
    """The grid, one axis at a time, with the agent held fixed."""

    before = sha256_file(bank_path)
    payload = load_temporal_controller_artifact(controller_path)
    encoders = curated_frontend(
        _machine(payload, learn=False), seed=FRONTEND_SEED, path=frontend_path
    )
    bank = ExternalTemporalProgramBank.load_bank(bank_path)

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for alphabet in alphabets:
        for noise in noise_levels:
            rows.append(
                run_cell(
                    payload,
                    encoders,
                    bank,
                    alphabet=alphabet,
                    noise=noise,
                    seed=seed,
                    stream_length=stream_length,
                    pool_size=pool_size,
                )
            )
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("the widening sweep mutated AgentBrain.bank")

    usable = [row for row in rows if row["failure"] is None]
    report = {
        "schema": WIDENING_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "stream_length": stream_length,
        "pool_size": pool_size,
        "cells": len(rows),
        "cells_with_a_recovered_alphabet": len(usable),
        "cells_fully_solved": sum(
            1 for row in usable if row["solved"] == row["tasks"]
        ),
        "false_recognitions": sum(int(row["false_recognitions"]) for row in usable),
        "worst_acquisition_ratio": (
            max(row["acquisition_ratio"] for row in usable) if usable else None
        ),
        "rows": rows,
        "agent_bank_sha256": before,
        "agent_bank_unchanged": after == before,
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "environment_widening.json").write_text(
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
            / "brainworkshop_environment_widening_2026-08-15"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--stream-length", type=int, default=12)
    parser.add_argument("--pool-size", type=int, default=4)
    arguments = parser.parse_args()
    report = run_widening(
        arguments.controller,
        arguments.bank,
        arguments.output,
        frontend_path=arguments.frontend,
        seed=arguments.seed,
        stream_length=arguments.stream_length,
        pool_size=arguments.pool_size,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
