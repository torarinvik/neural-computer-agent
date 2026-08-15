"""Close the loop: feedback to hypothesis to program, with no oracle.

Two results sat next to each other without touching. The counter bridge showed
a program family that expresses **18/18** of the sampled rules, and it was
dismissed as a ceiling because an experimenter's compiler read the rule and
emitted the answer. The identification ceiling showed that **one episode of
feedback determines the task exactly** for nine of eighteen rules, and it was
recorded as a ceiling because the inferred machine was never given to anything.

Put them together and neither is a ceiling any more. The compiler in
`counter_state_programs` is provenance-neutral code: it turns *a Mealy machine*
into a counter program and does not care where the machine came from. Feeding
it a machine inferred from the agent's own reward is not an oracle. It is
learning.

The pipeline reads nothing it should not:

1. cluster the frontend's own events, so the alphabet is discovered;
2. run one episode and invert its per-step reward into the target behaviour;
3. infer the smallest Mealy machine consistent with that trace;
4. compile the machine to a counter program;
5. score it on a *different* episode.

Step 4 needs a map from the hypothesis's symbols to the executor's input
channels, and there is nothing to map: the hypothesis was inferred over
cluster indices, and the executor's channels are cluster indices. The identity
map is not a convenience here, it is the reason no oracle is required.
`cluster_symbol_map` -- which renders a canonical frame per symbol to find out
what each symbol looks like -- is exactly the oracle step this replaces, and
it is not imported.

What this is: still a diagnostic, on an already-consumed development seed.
Nothing is admitted and `AgentBrain.bank` is untouched. What it is not: a
compilation of the answer. `tests/test_induced_counter_program.py` asserts
this module never reads `config.rule`, in the style of the agent-path boundary
test, because "nobody imported the oracle" is a weaker guarantee than
"the code cannot see it".
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

from .accumulation_curve import _config, curriculum_rules
from .controller_pretraining import load_temporal_controller_artifact
from .counter_state_programs import (
    compile_rule as compile_automaton,
)
from .counter_state_programs import (
    initial_counters,
    run_counter_program,
)
from .current_symbol_acquire import FRONTEND_SEED, THRESHOLD, _machine, curated_frontend
from .identification_ceiling import (
    NODE_BUDGET,
    episode_trace,
    infer_machine,
)
from .prototype_templates import cluster_events, observe_events

EXPERIMENT_ID = "brainworkshop-induced-counter-program-2026-08-15"
INDUCED_SCHEMA = "neural-computer.induced-counter-program.v1"
DEVELOPMENT_SEED = 41
STEPS = 448
# One observation pass, one probe episode, one held-out evaluation.
EPISODES_PER_RULE = 2


def induce_program(
    payload: dict[str, object],
    encoders,
    bank: ExternalTemporalProgramBank,
    config,
    *,
    seed: int,
    node_budget: int = NODE_BUDGET,
    learning_episodes: int = 1,
    learning_steps: int | None = None,
) -> dict[str, Any]:
    """Feedback in, executed counter program out. Reads no rule.

    `learning_episodes` and `learning_steps` set how the feedback budget is
    *segmented*. That turns out to matter more than how large it is: the same
    number of labelled steps identifies every rule when it arrives as many
    short episodes and less than half of them when it arrives as one long one.
    """

    clusters = cluster_events(observe_events(encoders, config, seed=seed))
    if learning_steps is None:
        traces = (episode_trace(payload, encoders, bank, config, clusters, seed=seed),)
        spent = 1
    else:
        short = replace(config, steps=int(learning_steps)).validate()
        traces = tuple(
            episode_trace(
                payload, encoders, bank, short, clusters, seed=seed + 1000 + index
            )
            for index in range(int(learning_episodes))
        )
        spent = int(learning_episodes)
    machine = infer_machine(traces, node_budget=node_budget)
    if machine is None:
        return {
            "identified": False,
            "episodes_spent": spent,
            "accuracy": None,
            "solved": False,
            "reason": "no machine consistent with the trace within budget",
        }
    # The hypothesis speaks in cluster indices and so does the executor, so
    # there is nothing to translate and nothing to look up.
    channel_of_symbol = tuple(range(int(clusters.shape[0])))
    program = compile_automaton(
        machine,
        channel_of_symbol=channel_of_symbol,
        cluster_count=int(clusters.shape[0]),
    )
    executed = run_counter_program(
        program,
        encoders,
        config,
        clusters,
        seed=seed + 1,
        initial_counters=initial_counters(
            program,
            cluster_count=int(clusters.shape[0]),
            states=machine.state_count,
        ),
    )
    accuracy = float(executed["accuracy"])
    return {
        "identified": True,
        "episodes_spent": spent + 1,
        "learning_episodes": spent,
        "learning_steps": learning_steps,
        "feedback_steps": sum(len(item.symbols) for item in traces),
        "inferred_state_count": machine.state_count,
        "instructions": len(program.instructions),
        "counters": program.counter_count,
        "accuracy": accuracy,
        "executor_statuses": executed["statuses"],
        "scored": int(executed["scored"]),
        "solved": accuracy >= THRESHOLD,
        "reason": None,
    }


def run_induction(
    controller_path: Path,
    bank_path: Path,
    output_directory: Path,
    *,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    steps: int = STEPS,
    node_budget: int = NODE_BUDGET,
    learning_episodes: int = 1,
    learning_steps: int | None = None,
) -> dict[str, Any]:
    """Every sampled rule, learned rather than compiled from the answer."""

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
        row = induce_program(
            payload,
            encoders,
            bank,
            config,
            seed=seed,
            node_budget=node_budget,
            learning_episodes=learning_episodes,
            learning_steps=learning_steps,
        )
        # Recorded for reading the table, never consulted by the pipeline.
        row["true_state_count"] = rule.state_count
        row["rule_digest"] = rule.digest()
        rows.append(row)
    after = sha256_file(bank_path)
    if after != before:
        raise RuntimeError("induced counter program mutated AgentBrain.bank")
    solved = [row for row in rows if row["solved"]]
    perfect = [row for row in rows if row["accuracy"] == 1.0]
    report = {
        "schema": INDUCED_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "status": "diagnostic",
        "note": (
            "programs are induced from the agent's own per-step reward and "
            "its own event clusters; no rule is read and nothing is admitted"
        ),
        "bank_sha256": before,
        "bank_unchanged": after == before,
        "seed": seed,
        "steps": steps,
        "threshold": THRESHOLD,
        "node_budget": node_budget,
        "learning_episodes": learning_episodes,
        "learning_steps": learning_steps,
        "feedback_steps": sum(int(row.get("feedback_steps", 0)) for row in rows),
        "rules": rows,
        "solved": len(solved),
        "exactly_correct": len(perfect),
        "of": len(rows),
        "episodes_spent": sum(int(row["episodes_spent"]) for row in rows),
        "solved_by_state_count": {
            str(states): {
                "solved": sum(
                    1
                    for row in rows
                    if row["true_state_count"] == states and row["solved"]
                ),
                "of": sum(1 for row in rows if row["true_state_count"] == states),
            }
            for states in sorted({row["true_state_count"] for row in rows})
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "induction.json").write_text(
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
            / "brainworkshop_induced_counter_program_2026-08-15"
        ),
    )
    parser.add_argument("--node-budget", type=int, default=NODE_BUDGET)
    parser.add_argument("--learning-episodes", type=int, default=1)
    parser.add_argument("--learning-steps", type=int, default=None)
    arguments = parser.parse_args()
    report = run_induction(
        arguments.controller_artifact,
        arguments.bank,
        arguments.output_dir,
        frontend_path=arguments.frontend,
        node_budget=arguments.node_budget,
        learning_episodes=arguments.learning_episodes,
        learning_steps=arguments.learning_steps,
    )
    print(
        json.dumps(
            {
                "bank_unchanged": report["bank_unchanged"],
                "solved": f"{report['solved']}/{report['of']}",
                "exactly_correct": report["exactly_correct"],
                "episodes_spent": report["episodes_spent"],
                "solved_by_state_count": report["solved_by_state_count"],
                "elapsed_seconds": report["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
