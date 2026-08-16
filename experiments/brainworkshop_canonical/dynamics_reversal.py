"""Measure model invalidation and recovery after an unannounced reversal.

This is the first environment escalation after the cross-world operator audit.
Perceptual complexity is held fixed: only the transition table changes halfway
through a lifetime stream.  The reusable update operator clears a model when a
known state/action cell contradicts fresh evidence; the control keeps the
contradictory counts and therefore carries a mixed world forward.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .operator_world_transfer import (
    ACTION_COUNT,
    EPISODE_STEPS,
    PLACE_COUNT,
    SOURCE_WORLD_SEED,
    STABLE_THRESHOLD,
    WORLD_SEED_STRIDE,
    VerifiedOperatorBundle,
    _episode,
    _evaluate,
    _random_tasks,
    sample_ring_world,
)
from .world_model import WorldModel

EXPERIMENT_ID = "brainworkshop-dynamics-reversal-2026-08-16"
DYNAMICS_REVERSAL_SCHEMA = "neural-computer.dynamics-reversal.v1"
DEVELOPMENT_SEED = 41
PRE_CHANGE_EPISODES = 8
POST_CHANGE_EPISODES = 12
EVALUATION_EPISODES = 8
EVALUATION_PREFIXES = (1, 2, 4, 8, 12)


def reversed_world(world):
    """Swap the two controllable directions while preserving observations."""

    return type(world)(
        seed=world.seed + 1_000_000,
        transitions=(world.transitions[1], world.transitions[0], world.transitions[2]),
    )


def _noninvalidating_bundle(*, world_seed: int) -> VerifiedOperatorBundle:
    return VerifiedOperatorBundle(
        invalidation="ignore_mismatch",
        verified_world_seed=int(world_seed),
    ).validate()


def _stable_bits(curve: list[dict[str, Any]]) -> int | None:
    for index, row in enumerate(curve):
        if all(float(later["normalized_return"]) >= STABLE_THRESHOLD for later in curve[index:]):
            return int(row["unique_verifier_bits"])
    return None


def run_reversal_arm(
    source,
    changed,
    *,
    bundle: VerifiedOperatorBundle,
    seed: int,
    pre_change_episodes: int = PRE_CHANGE_EPISODES,
    post_change_episodes: int = POST_CHANGE_EPISODES,
    evaluation_episodes: int = EVALUATION_EPISODES,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = WorldModel(PLACE_COUNT, ACTION_COUNT)
    pre_tasks = _random_tasks(source, seed=seed + 10, count=pre_change_episodes)
    post_tasks = _random_tasks(changed, seed=seed + 20, count=post_change_episodes)
    for index, task in enumerate(pre_tasks):
        _episode(
            source,
            model,
            task,
            mode="reusable",
            bundle=bundle,
            artifact=None,
            seed=seed + index,
        )
    curve = []
    verifier_bits = pre_change_episodes * EPISODE_STEPS
    latencies = []
    for prefix in range(1, post_change_episodes + 1):
        _episode(
            changed,
            model,
            post_tasks[prefix - 1],
            mode="reusable",
            bundle=bundle,
            artifact=None,
            seed=seed + 1_000 + prefix,
        )
        if prefix not in EVALUATION_PREFIXES:
            continue
        tasks = _random_tasks(
            changed,
            seed=seed + 100_000 + prefix * 1_000,
            count=evaluation_episodes,
        )
        score, latency = _evaluate(
            changed,
            model,
            tasks,
            mode="reusable",
            bundle=bundle,
            artifact=None,
            seed=seed + 200_000 + prefix * 1_000,
        )
        verifier_bits += evaluation_episodes * EPISODE_STEPS
        latencies.append(latency)
        curve.append(
            {
                "post_change_episodes": prefix,
                "normalized_return": score,
                "coverage": model.coverage,
                "unique_verifier_bits": verifier_bits,
            }
        )
    return {
        "bundle_digest": bundle.digest,
        "curve": curve,
        "stable_bits_to_threshold": _stable_bits(curve),
        "final_model_coverage": model.coverage,
        "unique_verifier_bits": verifier_bits,
        "unique_logical_lifetimes": (
            pre_change_episodes
            + post_change_episodes
            + len(curve) * evaluation_episodes
        ),
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "wall_seconds": time.perf_counter() - started,
        "latency_ms_per_decision": sum(latencies) / len(latencies) if latencies else 0.0,
        "retention_on_mastered_primitive": "not_claimed",
    }


def run_reversal(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    replicates: int = 3,
) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for replicate in range(int(replicates)):
        source = sample_ring_world(SOURCE_WORLD_SEED + WORLD_SEED_STRIDE * replicate)
        changed = reversed_world(source)
        recovery = run_reversal_arm(
            source,
            changed,
            bundle=VerifiedOperatorBundle(
                verified_world_seed=source.seed,
                invalidation="rebuild_on_mismatch",
            ).validate(),
            seed=seed + 1_000 * replicate,
        )
        mixed = run_reversal_arm(
            source,
            changed,
            bundle=_noninvalidating_bundle(world_seed=source.seed),
            seed=seed + 1_000 * replicate,
        )
        rows.append(
            {
                "replicate": replicate,
                "source_world_seed": source.seed,
                "source_world_digest": source.digest,
                "changed_world_digest": changed.digest,
                "recovery": recovery,
                "mixed_model_control": mixed,
            }
        )
    recovery_bits = [row["recovery"]["stable_bits_to_threshold"] for row in rows]
    mixed_bits = [row["mixed_model_control"]["stable_bits_to_threshold"] for row in rows]
    accounting = {
        arm: {
            "unique_verifier_bits": sum(
                int(row[arm]["unique_verifier_bits"])
                for row in rows
            ),
            "unique_logical_lifetimes": sum(
                int(row[arm]["unique_logical_lifetimes"])
                for row in rows
            ),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": sum(float(row[arm]["wall_seconds"]) for row in rows),
            "latency_ms_per_decision": sum(
                float(row[arm]["latency_ms_per_decision"]) for row in rows
            )
            / len(rows),
            "stable_bits_to_threshold": [row[arm]["stable_bits_to_threshold"] for row in rows],
            "retention_on_mastered_primitive": "not_claimed",
        }
        for arm in ("recovery", "mixed_model_control")
    }
    report = {
        "schema": DYNAMICS_REVERSAL_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "replicates": rows,
        "replicate_count": len(rows),
        "stable_threshold": STABLE_THRESHOLD,
        "recovery_stable_bits": recovery_bits,
        "mixed_model_stable_bits": mixed_bits,
        "accounting": accounting,
        "claim_status": "development_diagnostic",
        "seconds": time.perf_counter() - started,
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "dynamics_reversal.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository / "session_records" / "brainworkshop_dynamics_reversal_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--replicates", type=int, default=3)
    arguments = parser.parse_args()
    print(json.dumps(run_reversal(arguments.output, seed=arguments.seed, replicates=arguments.replicates), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
