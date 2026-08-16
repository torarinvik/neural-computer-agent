"""Stage a reusable planning operator inside the rendered live task loop.

This development diagnostic keeps the causal boundary explicit: one canonical
agent runs a real Neural Workshop lifetime, a rendered source maze, a rendered
target maze, and then Workshop again.  The source maze only verifies an opaque
operator candidate; its world model is not carried into the target maze.  The
target wrapper receives the candidate only after a stable-prefix and retention
gate pass.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from .canonical_live_machine import run_canonical_neural_workshop_live_lifetime
from .cross_task_live_transfer import _new_agent
from .maze_environment import sample_maze_task
from .maze_transfer import (
    EVENT_WIDTH,
    SharedAmodalMazeAgent,
    _run_cross_task_maze,
    build_event_dictionary,
)
from .neural_workshop_live import (
    NeuralWorkshopLiveConfig,
    build_neural_workshop_environment,
)
from .operator_world_transfer import (
    VerifiedPlanningOperatorStager,
    verified_bundle,
)
from .rendered_environment import RenderedBrainWorkshopEncoders

LIVE_OPERATOR_TRANSFER_SCHEMA = "neural-computer.live-operator-transfer.v1"
LIVE_OPERATOR_TRANSFER_EXPERIMENT_ID = "brainworkshop-live-operator-transfer-2026-08-16"
DEVELOPMENT_SEED = 97


def _maze_stage(
    agent,
    encoders,
    task,
    *,
    operator,
    mode: str,
    seed: int,
    training_episodes: int,
    evaluation_episodes: int,
    steps: int,
    initial_verifier_bits: int,
    evaluation_seed: int | None = None,
    watch_label: str | None = None,
) -> dict[str, Any]:
    dictionary = build_event_dictionary(task, encoders)
    maze_agent = SharedAmodalMazeAgent(
        agent,
        encoders,
        dictionary,
        mode=mode,
        operator=operator,
    )
    report = _run_cross_task_maze(
        maze_agent,
        task,
        seed=seed,
        training_episodes=training_episodes,
        evaluation_episodes=evaluation_episodes,
        steps=steps,
        initial_verifier_bits=initial_verifier_bits,
        evaluation_seed=evaluation_seed,
        watch_label=watch_label,
    )
    return {
        "report": report,
        "dictionary_digest": dictionary.digest,
        "operator_digest": None if operator is None else operator.digest,
        "controller_digest_after": maze_agent.controller_digest,
        "same_core_instance": maze_agent.core is agent,
    }


def run_live_operator_transfer(
    neural_workshop: Path,
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 1,
    trials: int = 12,
    source_maze_training_episodes: int = 40,
    target_maze_training_episodes: int = 40,
    maze_evaluation_episodes: int = 2,
    maze_steps: int = 20,
    visible: bool = False,
    watch: bool = False,
) -> dict[str, Any]:
    """Run live Workshop -> source maze -> target maze -> Workshop."""

    if min(
        replicates,
        trials,
        source_maze_training_episodes,
        target_maze_training_episodes,
        maze_evaluation_episodes,
        maze_steps,
    ) < 1:
        raise ValueError("live operator transfer budgets must be positive")
    rows: list[dict[str, Any]] = []
    for replicate in range(int(replicates)):
        config = NeuralWorkshopLiveConfig(
            active_cells=2,
            trials=trials,
            event_width=EVENT_WIDTH,
            action_ports=1,
            visible=visible,
        )
        matched_config = replace(config, visible=False)
        base_seed = seed + replicate * 1_000
        agent = _new_agent(base_seed, config.event_width)
        environment, verifier = build_neural_workshop_environment(
            neural_workshop,
            config,
            seed=seed + 40_000 + replicate,
        )
        live_before = run_canonical_neural_workshop_live_lifetime(
            agent,
            config,
            seed=seed + 50_000 + replicate,
            environment=environment,
            verifier=verifier,
            sample=False,
        )
        if watch:
            print(f"[replicate {replicate}] Workshop before maze complete")
        source_seed = seed + 10_000 + replicate
        target_seed = seed + 20_000 + replicate
        source = sample_maze_task(
            seed=source_seed,
            grid_size=7,
            minimum_distance=5,
        )
        target = sample_maze_task(
            seed=target_seed,
            grid_size=7,
            minimum_distance=5,
        )
        if source is None or target is None:
            raise RuntimeError("live operator transfer maze sampler failed")
        if source.transitions == target.transitions:
            raise AssertionError("source and target mazes must differ")
        encoders = RenderedBrainWorkshopEncoders.seeded(
            EVENT_WIDTH,
            source_key_width=4,
            seed=seed + 30_000 + replicate,
        )
        candidate = verified_bundle(world_seed=source_seed)
        source_stage = _maze_stage(
            agent,
            encoders,
            source,
            operator=candidate,
            mode="workshop_warm",
            seed=base_seed,
            training_episodes=source_maze_training_episodes,
            evaluation_episodes=maze_evaluation_episodes,
            steps=maze_steps,
            initial_verifier_bits=live_before.unique_verifier_bits,
            evaluation_seed=base_seed + 20_000,
            watch_label=(f"replicate {replicate} source maze" if watch else None),
        )
        stager = VerifiedPlanningOperatorStager(
            threshold=0.70,
            min_observations=2,
            min_stable_observations=2,
        )
        for checkpoint in source_stage["report"]["curve"]:
            for outcome in checkpoint["evaluation_returns"]:
                stager.observe(candidate, outcome)
        admission = stager.admit_verified(
            candidate,
            lambda retained, digest=candidate.digest: retained.digest == digest,
        )
        target_stage = _maze_stage(
            agent,
            encoders,
            target,
            operator=candidate if admission.accepted else None,
            mode="workshop_warm",
            seed=base_seed + 1_000,
            training_episodes=target_maze_training_episodes,
            evaluation_episodes=maze_evaluation_episodes,
            steps=maze_steps,
            initial_verifier_bits=source_stage["report"]["unique_verifier_bits"],
            evaluation_seed=base_seed + 30_000,
            watch_label=(f"replicate {replicate} target maze" if watch else None),
        )
        environment_after, verifier_after = build_neural_workshop_environment(
            neural_workshop,
            matched_config,
            seed=seed + 60_000 + replicate,
        )
        live_after = run_canonical_neural_workshop_live_lifetime(
            agent,
            config,
            seed=seed + 70_000 + replicate,
            environment=environment_after,
            verifier=verifier_after,
            sample=False,
        )
        # Matched control: identical public seeds, Workshop warm-up, source
        # maze evidence, and target budget, but no operator is supplied after
        # the source stage.  This isolates the causal value of the admitted
        # artifact from simply spending more maze experience.
        matched_agent = _new_agent(base_seed, config.event_width)
        matched_environment, matched_verifier = build_neural_workshop_environment(
            neural_workshop,
            matched_config,
            seed=seed + 40_000 + replicate,
        )
        matched_live_before = run_canonical_neural_workshop_live_lifetime(
            matched_agent,
            matched_config,
            seed=seed + 50_000 + replicate,
            environment=matched_environment,
            verifier=matched_verifier,
            sample=False,
        )
        matched_source = _maze_stage(
            matched_agent,
            encoders,
            source,
            operator=candidate,
            mode="workshop_warm",
            seed=base_seed,
            training_episodes=source_maze_training_episodes,
            evaluation_episodes=maze_evaluation_episodes,
            steps=maze_steps,
            initial_verifier_bits=matched_live_before.unique_verifier_bits,
            evaluation_seed=base_seed + 20_000,
        )
        matched_target = _maze_stage(
            matched_agent,
            encoders,
            target,
            operator=None,
            mode="fresh",
            seed=base_seed + 1_000,
            training_episodes=target_maze_training_episodes,
            evaluation_episodes=maze_evaluation_episodes,
            steps=maze_steps,
            initial_verifier_bits=matched_source["report"]["unique_verifier_bits"],
            evaluation_seed=base_seed + 30_000,
        )
        matched_environment_after, matched_verifier_after = (
            build_neural_workshop_environment(
                neural_workshop,
                matched_config,
                seed=seed + 60_000 + replicate,
            )
        )
        matched_live_after = run_canonical_neural_workshop_live_lifetime(
            matched_agent,
            matched_config,
            seed=seed + 70_000 + replicate,
            environment=matched_environment_after,
            verifier=matched_verifier_after,
            sample=False,
        )
        if watch:
            print(f"[replicate {replicate}] Workshop after maze complete")
        rows.append(
            {
                "replicate": replicate,
                "source_task": source.payload(),
                "target_task": target.payload(),
                "live_workshop_before": live_before.as_dict(),
                "source_maze": source_stage,
                "target_maze": target_stage,
                "live_workshop_after": live_after.as_dict(),
                "matched_control": {
                    "live_workshop_before": matched_live_before.as_dict(),
                    "source_maze": matched_source,
                    "target_maze": matched_target,
                    "live_workshop_after": matched_live_after.as_dict(),
                    "controller_unchanged": (
                        matched_live_before.controller_digest_after
                        == matched_source["controller_digest_after"]
                        == matched_target["controller_digest_after"]
                        == matched_live_after.controller_digest_after
                    ),
                },
                "operator_admission": admission.__dict__,
                "same_core_instance": (
                    source_stage["same_core_instance"]
                    and target_stage["same_core_instance"]
                ),
                "controller_unchanged": (
                    live_before.controller_digest_after
                    == source_stage["controller_digest_after"]
                    == target_stage["controller_digest_after"]
                    == live_after.controller_digest_after
                ),
            }
        )
    final_advantages: list[float | None] = []
    for row in rows:
        if not row["operator_admission"]["accepted"]:
            final_advantages.append(None)
            continue
        admitted_return = float(
            row["target_maze"]["report"]["curve"][-1]["normalized_return"]
        )
        control_return = float(
            row["matched_control"]["target_maze"]["report"]["curve"][-1][
                "normalized_return"
            ]
        )
        final_advantages.append(admitted_return - control_return)
    admitted_advantages = [
        advantage for advantage in final_advantages if advantage is not None
    ]
    report = {
        "schema": LIVE_OPERATOR_TRANSFER_SCHEMA,
        "experiment_id": LIVE_OPERATOR_TRANSFER_EXPERIMENT_ID,
        "seed": seed,
        "replicate_count": len(rows),
        "replicates": rows,
        "shared_agent_boundary": {
            "one_controller_across_workshop_source_maze_target_maze_workshop": all(
                bool(row["same_core_instance"]) for row in rows
            ),
            "one_amodal_event_bus": True,
            "one_intention_bus": True,
            "controller_receives": "learned_events_and_opaque_feedback_only",
            "maze_facts_task_local": True,
            "workshop_verifier_state_task_local": True,
        },
        "all_candidates_admitted": all(
            bool(row["operator_admission"]["accepted"]) for row in rows
        ),
        "controller_unchanged_for_all_replicates": all(
            bool(row["controller_unchanged"]) for row in rows
        ),
        "matched_control_controller_unchanged_for_all_replicates": all(
            bool(row["matched_control"]["controller_unchanged"]) for row in rows
        ),
        "admitted_replicate_count": sum(
            bool(row["operator_admission"]["accepted"]) for row in rows
        ),
        "target_final_return_advantage_against_no_operator": final_advantages,
        "positive_transfer_for_all_admitted_replicates": bool(admitted_advantages)
        and all(advantage > 0.0 for advantage in admitted_advantages),
        "claim_status": "development_diagnostic",
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "live_operator_transfer.json").write_text(
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
        default=repository / "session_records" / "brainworkshop_live_operator_transfer_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--source-maze-training-episodes", type=int, default=40)
    parser.add_argument("--target-maze-training-episodes", type=int, default=40)
    parser.add_argument("--maze-evaluation-episodes", type=int, default=2)
    parser.add_argument("--maze-steps", type=int, default=20)
    parser.add_argument(
        "--visible",
        action="store_true",
        help="show the primary Neural Workshop window (control stays headless)",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="print the primary source/target maze as an external ASCII trace",
    )
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_live_operator_transfer(
                arguments.neural_workshop,
                arguments.output,
                seed=arguments.seed,
                replicates=arguments.replicates,
                trials=arguments.trials,
                source_maze_training_episodes=arguments.source_maze_training_episodes,
                target_maze_training_episodes=arguments.target_maze_training_episodes,
                maze_evaluation_episodes=arguments.maze_evaluation_episodes,
                maze_steps=arguments.maze_steps,
                visible=arguments.visible,
                watch=arguments.watch,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
