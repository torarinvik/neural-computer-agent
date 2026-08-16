"""Development composition audit for persistent causal identity v3.

This is the first composition test after the bounded v3 fixture.  It runs the
existing rendered relational-navigation loop and adds the external v3 identity
artifact at the learned-event/tracking seam.  The old episode-local scorer,
an oracle, a blind arm, a fresh v3 model, and one deliberately stale model are
measured on the same task draws.  No holdout seed is consumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from neural_computer import PersistentCausalIdentityV3
from neural_computer.promotion import sha256_file

from .integrated_navigation import (
    DEVELOPMENT_SEED,
    DEVELOPMENT_WORLD_SEED,
    EPISODE_STEPS,
    EXPLORE_EPISODES,
    run_integrated_navigation,
)

EXPERIMENT_ID = "brainworkshop-integrated-navigation-persistent-identity-v3-2026-08-16"
EXPERIMENT_SCHEMA = "neural-computer.integrated-navigation-persistent-identity-v3.v1"


def run_integrated_navigation_v3(
    output_directory: Path,
    *,
    controller_path: Path | None = None,
    bank_path: Path | None = None,
    frontend_path: Path | None = None,
    seed: int = DEVELOPMENT_SEED,
    world_seed: int = DEVELOPMENT_WORLD_SEED,
    tasks: int = 4,
    steps: int = EPISODE_STEPS,
    explore_episodes: int = EXPLORE_EPISODES,
    starts: int = 6,
) -> dict[str, Any]:
    """Run the composed development loop and write an auditable report."""

    repository = Path(__file__).parents[2]
    controller = controller_path or (
        repository
        / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
    )
    bank = bank_path or repository / "artifacts/checkpoints/AgentBrain.bank"
    frontend = frontend_path or (
        repository / "artifacts/checkpoints/rendered_frontend_seed1001.pt"
    )
    before = sha256_file(bank)
    # The fresh arm is reset per navigation world; the stale arm deliberately
    # carries one graph across worlds so changed dynamics have to be detected.
    stale = PersistentCausalIdentityV3()
    navigation = run_integrated_navigation(
        controller,
        bank,
        output_directory,
        frontend_path=frontend,
        seed=seed,
        world_seed=world_seed,
        tasks=tasks,
        steps=steps,
        explore_episodes=explore_episodes,
        starts=starts,
        include_episode_local=True,
        persistent_identity_factories={
            "persistent_v3": PersistentCausalIdentityV3,
            "stale_v3": lambda: stale,
        },
    )
    after = sha256_file(bank)
    trained = navigation["trained"]
    held_out = navigation["held_out"]
    report = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "world_seed": world_seed,
        "tasks": tasks,
        "episode_steps": steps,
        "explore_episodes": explore_episodes,
        "starts": starts,
        "matched_arms": (
            "random",
            "episode_local",
            "persistent_v3",
            "stale_v3",
            "told_all",
        ),
        "trained": trained,
        "held_out": held_out,
        "persistent_advantage_over_episode_local": {
            "trained": trained["persistent_v3"] - trained["episode_local"],
            "held_out": held_out["persistent_v3"] - held_out["episode_local"],
        },
        "stale_model": {
            "trained_confident_wrong_rate": trained["stale_v3_confident_wrong_rate"],
            "held_out_confident_wrong_rate": held_out["stale_v3_confident_wrong_rate"],
            "trained_quarantine_count": trained["stale_v3_quarantine_count"],
            "held_out_quarantine_count": held_out["stale_v3_quarantine_count"],
        },
        "agent_bank_sha256_before": before,
        "agent_bank_sha256_after": after,
        "agent_bank_unchanged": before == after,
        "navigation_report": "integrated_navigation.json",
        "claim_status": "development_composition_diagnostic_not_promoted",
        "claim_boundary": (
            "This is a development composition result on the consumed navigation "
            "world block. It demonstrates that the v3 artifact can be called in "
            "the real rendered learned-event/tracking/policy/decoder loop, but it "
            "does not establish safe persistence, transfer, or holdout benefit."
        ),
        "required_next_gates": (
            "fresh pixel rerenders with true crossing and occlusion",
            "action-shuffled, missing-evidence, corrupted-memory, and reversal controls",
            "fresh-learner replication and stable-prefix accounting",
            "pre-registered comparison against episode-local scoring before any holdout",
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "persistent_identity_v3_navigation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "session_records"
        / "brainworkshop_integrated_navigation_v3_2026-08-16",
    )
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--explore-episodes", type=int, default=EXPLORE_EPISODES)
    parser.add_argument("--starts", type=int, default=6)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_integrated_navigation_v3(
                arguments.output,
                tasks=arguments.tasks,
                explore_episodes=arguments.explore_episodes,
                starts=arguments.starts,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
