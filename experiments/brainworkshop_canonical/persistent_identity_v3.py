"""Development-only closed-loop audit for persistent causal identity v3.

V3 keeps the production live contract from the v2 diagnostic, but changes one
development fixture detail: the two episodes use the same learned state region
with the controlled track entering from opposite ends.  This lets the
state-conditioned transition graph test both slot rebinding and an unannounced
dynamics reversal without granting coordinates or lifetime labels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from neural_computer import PersistentCausalIdentityV3
from neural_computer.promotion import sha256_file

from . import persistent_identity_v2 as base

EXPERIMENT_ID = "brainworkshop-persistent-identity-v3-2026-08-16"
EXPERIMENT_SCHEMA = "neural-computer.persistent-identity-v3-closed-loop.v1"
DEVELOPMENT_SEED = 47
EPISODE_STEPS = 8
EPISODES = 3

_PROBE_PATTERN = (
    torch.tensor([1.0, 0.0]),
    torch.tensor([-1.0, 0.0]),
    torch.tensor([1.0, 0.0]),
    torch.tensor([1.0, 0.0]),
    torch.tensor([1.0, 0.0]),
    torch.tensor([-1.0, 0.0]),
    torch.tensor([-1.0, 0.0]),
    torch.tensor([-1.0, 0.0]),
)


def _world_for_v3(episode: int, *, reverse: bool = False) -> base._MarkerWorld:
    """Same learned state region, opposite slot order between episodes."""

    if episode % 2 == 0:
        return base._MarkerWorld(1, 3, 1, 0, reverse=reverse)
    return base._MarkerWorld(1, 3, 3, 4, reverse=reverse)


def run_persistent_identity_v3(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    steps: int = EPISODE_STEPS,
    episodes: int = EPISODES,
) -> dict[str, Any]:
    if steps < 4 or episodes < 2:
        raise ValueError("persistent identity v3 needs at least two four-step episodes")
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    encoders = base.RenderedBrainWorkshopEncoders.seeded(
        base.EVENT_WIDTH,
        source_key_width=4,
        seed=base.FRONTEND_SEED,
    )
    frontend_digest_before = encoders.digest()
    repository = Path(__file__).parents[2]
    bank_path = repository / "artifacts/checkpoints" / "AgentBrain.bank"
    bank_digest_before = sha256_file(bank_path) if bank_path.is_file() else None
    previous_pattern = base._CyclingDecoder._pattern
    base._CyclingDecoder._pattern = _PROBE_PATTERN
    try:
        arms = {}
        for arm in ("no_persistent", "episode_local"):
            result = base._run_arm(
                arm,
                encoders,
                episodes=episodes,
                steps=steps,
                world_factory=_world_for_v3,
            )
            result["arm"] = arm
            arms[arm] = result
        persistent = base._run_arm(
            "persistent_v2",
            encoders,
            episodes=episodes,
            steps=steps,
            persistent_factory=PersistentCausalIdentityV3,
            world_factory=_world_for_v3,
        )
        persistent["arm"] = "persistent_v3"
        arms["persistent_v3"] = persistent
        stale = base._run_arm(
            "persistent_v2",
            encoders,
            episodes=episodes,
            steps=steps,
            reverse_episode=1,
            persistent_factory=PersistentCausalIdentityV3,
            world_factory=_world_for_v3,
        )
        stale["arm"] = "stale_persistent_v3"
        controls = base._control_report(
            encoders,
            model_factory=PersistentCausalIdentityV3,
            world_factory=_world_for_v3,
            controlled_slot=1,
        )
    finally:
        base._CyclingDecoder._pattern = previous_pattern
    frontend_digest_after = encoders.digest()
    bank_digest_after = sha256_file(bank_path) if bank_path.is_file() else None
    report = {
        "schema": EXPERIMENT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "steps": steps,
        "episodes": episodes,
        "artifact": PersistentCausalIdentityV3().configuration(),
        "arms": arms,
        "persistent_advantage_over_episode_local": (
            arms["persistent_v3"]["integrated_return"]
            - arms["episode_local"]["integrated_return"]
        ),
        "stale_persistent_arm": stale,
        "controls": controls,
        "frontend_digest_before": frontend_digest_before,
        "frontend_digest_after": frontend_digest_after,
        "frontend_unchanged": frontend_digest_before == frontend_digest_after,
        "bank_digest_before": bank_digest_before,
        "bank_digest_after": bank_digest_after,
        "bank_unchanged": bank_digest_before == bank_digest_after,
        "unique_verifier_bits": int(steps * episodes * 4),
        "unique_logical_lifetimes": int(episodes * 4),
        "optimizer_updates": 0,
        "replayed_examples": int(steps * episodes * 4),
        "wall_seconds": time.perf_counter() - started,
        "claim_status": "development_closed_loop_diagnostic_not_promoted",
        "claim_boundary": (
            "V3 exercises state-conditioned action-labelled transition graphs "
            "through rendered learned events, tracking, the policy-free live "
            "runtime, opaque decoding, and receipt-linked feedback. The fixture "
            "is bounded, the event state region is shared between episodes, and "
            "no curated-bank or persistent-self admission is claimed."
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "persistent_identity_v3.json").write_text(
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
        / "brainworkshop_persistent_identity_v3_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--steps", type=int, default=EPISODE_STEPS)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_persistent_identity_v3(
                arguments.output,
                seed=arguments.seed,
                steps=arguments.steps,
                episodes=arguments.episodes,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
