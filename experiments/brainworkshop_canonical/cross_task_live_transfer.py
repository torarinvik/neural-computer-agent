"""Run one canonical agent through live Neural Workshop and then a maze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .canonical_live_machine import run_canonical_neural_workshop_live_lifetime
from .maze_environment import sample_maze_task
from .maze_transfer import (
    ACTION_COUNT,
    EVENT_WIDTH,
    _run_cross_task_maze,
    build_event_dictionary,
    SharedAmodalMazeAgent,
)
from .neural_workshop_live import (
    NeuralWorkshopLiveConfig,
    build_neural_workshop_environment,
)
from .operator_world_transfer import verified_bundle
from .rendered_environment import RenderedBrainWorkshopEncoders
from .runner import CanonicalBrainWorkshopAgent

LIVE_CROSS_TASK_SCHEMA = "neural-computer.live-cross-task-transfer.v1"
LIVE_CROSS_TASK_EXPERIMENT_ID = "brainworkshop-live-cross-task-2026-08-16"
DEVELOPMENT_SEED = 83


def _new_agent(seed: int, event_width: int) -> CanonicalBrainWorkshopAgent:
    return CanonicalBrainWorkshopAgent(
        symbol_count=4,
        event_width=event_width,
        intention_width=ACTION_COUNT,
        feedback_width=8,
        n_back=1,
        reader_kind="context",
        seed=seed,
    )


def run_live_cross_task_transfer(
    neural_workshop: Path,
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 1,
    trials: int = 12,
    maze_training_episodes: int = 4,
    maze_evaluation_episodes: int = 2,
    maze_steps: int = 20,
) -> dict[str, Any]:
    """Run rendered Workshop → maze on one core per replicate.

    The Workshop environment is opened through its public RGBA/action/receipt
    surface.  The maze stage receives only a task-local rendered frontend and
    scalar feedback; its facts never enter the controller.
    """

    if min(
        replicates,
        trials,
        maze_training_episodes,
        maze_evaluation_episodes,
        maze_steps,
    ) < 1:
        raise ValueError("live cross-task budgets must be positive")
    rows: list[dict[str, Any]] = []
    for replicate in range(int(replicates)):
        config = NeuralWorkshopLiveConfig(
            active_cells=2,
            trials=trials,
            event_width=EVENT_WIDTH,
            action_ports=1,
            visible=False,
        )
        agent = _new_agent(seed + replicate * 1_000, config.event_width)
        environment, verifier = build_neural_workshop_environment(
            neural_workshop,
            config,
            seed=seed + 40_000 + replicate,
        )
        live_report = run_canonical_neural_workshop_live_lifetime(
            agent,
            config,
            seed=seed + 50_000 + replicate,
            environment=environment,
            verifier=verifier,
            sample=False,
        )
        maze = sample_maze_task(
            seed=seed + 20_000 + replicate,
            grid_size=7,
            minimum_distance=5,
        )
        if maze is None:
            raise RuntimeError("live cross-task maze sampler failed")
        encoders = RenderedBrainWorkshopEncoders.seeded(
            EVENT_WIDTH,
            source_key_width=4,
            seed=seed + 30_000 + replicate,
        )
        dictionary = build_event_dictionary(maze, encoders)
        maze_agent = SharedAmodalMazeAgent(
            agent,
            encoders,
            dictionary,
            mode="workshop_warm",
            operator=verified_bundle(world_seed=seed + 10_000 + replicate),
        )
        controller_before_maze = maze_agent.controller_digest
        maze_report = _run_cross_task_maze(
            maze_agent,
            maze,
            seed=seed + replicate * 1_000,
            training_episodes=maze_training_episodes,
            evaluation_episodes=maze_evaluation_episodes,
            steps=maze_steps,
            initial_verifier_bits=live_report.unique_verifier_bits,
        )
        environment_after_maze, verifier_after_maze = build_neural_workshop_environment(
            neural_workshop,
            config,
            seed=seed + 60_000 + replicate,
        )
        live_after_maze = run_canonical_neural_workshop_live_lifetime(
            agent,
            config,
            seed=seed + 70_000 + replicate,
            environment=environment_after_maze,
            verifier=verifier_after_maze,
            sample=False,
        )
        controller_after_maze = maze_agent.controller_digest
        rows.append(
            {
                "replicate": replicate,
                "live_workshop": live_report.as_dict(),
                "maze": maze_report,
                "live_workshop_after_maze": live_after_maze.as_dict(),
                "same_core_instance": maze_agent.core is agent,
                "controller_digest_after_live_workshop": controller_before_maze,
                "controller_digest_after_maze": controller_after_maze,
                "controller_digest_after_live_workshop_again": (
                    live_after_maze.controller_digest_after
                ),
                "controller_unchanged": (
                    live_report.controller_frozen
                    and controller_before_maze == controller_after_maze
                    and controller_after_maze == live_after_maze.controller_digest_after
                ),
                "intention_records_after_live_workshop": (
                    agent.intention_repertoire.record_count
                ),
            }
        )
    report = {
        "schema": LIVE_CROSS_TASK_SCHEMA,
        "experiment_id": LIVE_CROSS_TASK_EXPERIMENT_ID,
        "seed": seed,
        "replicate_count": len(rows),
        "replicates": rows,
        "shared_agent_boundary": {
            "one_controller_across_live_workshop_and_maze": all(
                bool(row["same_core_instance"]) for row in rows
            ),
            "one_amodal_event_bus": True,
            "one_intention_bus": True,
            "controller_receives": "learned_events_and_opaque_feedback_only",
            "maze_facts_task_local": True,
            "live_verifier_state_task_local": True,
        },
        "controller_unchanged_for_all_replicates": all(
            bool(row["controller_unchanged"]) for row in rows
        ),
        "live_workshop_survives_maze_for_all_replicates": all(
            row["live_workshop_after_maze"]["controller_frozen"]
            for row in rows
        ),
        "claim_status": "development_diagnostic",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "live_cross_task_transfer.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural-workshop", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "session_records" / "brainworkshop_live_cross_task_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--maze-training-episodes", type=int, default=4)
    parser.add_argument("--maze-evaluation-episodes", type=int, default=2)
    parser.add_argument("--maze-steps", type=int, default=20)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_live_cross_task_transfer(
                arguments.neural_workshop,
                arguments.output,
                seed=arguments.seed,
                replicates=arguments.replicates,
                trials=arguments.trials,
                maze_training_episodes=arguments.maze_training_episodes,
                maze_evaluation_episodes=arguments.maze_evaluation_episodes,
                maze_steps=arguments.maze_steps,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
