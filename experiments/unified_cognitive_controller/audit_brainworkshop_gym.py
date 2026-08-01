"""Sub-minute deterministic smoke audit for the Brain Workshop gym."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .brainworkshop_gym import (
    ALL_MATCHES,
    BrainWorkshopConfig,
    BrainWorkshopEventEncoders,
    generate_brainworkshop_episode,
)


def _score(episode, actions: list[int], latency_ms: float) -> dict[str, float]:
    targets = episode.verifier_targets()
    rewards = [
        episode.score_action(index, action, latency_ms)
        for index, action in enumerate(actions)
    ]
    return {
        "accuracy": sum(
            action == target for action, target in zip(actions, targets)
        ) / len(targets),
        "mean_reward": sum(rewards) / len(rewards),
        "total_reward": sum(rewards),
    }


def _episode_digest(episode) -> str:
    digest = hashlib.sha256()
    for stimulus, target in zip(
            episode.stimuli, episode.verifier_targets()):
        digest.update(bytes((stimulus.position, stimulus.audio, target)))
    return digest.hexdigest()


def run(seed: int = 44010) -> dict[str, object]:
    config = BrainWorkshopConfig(n_back=2, trials=24)
    episode = generate_brainworkshop_episode(config, seed=seed)
    targets = list(episode.verifier_targets())
    rng = random.Random(seed + 1)
    random_actions = [rng.randrange(ALL_MATCHES + 1) for _ in targets]
    no_op = [0] * len(targets)
    inverted = [target ^ ALL_MATCHES for target in targets]
    encoders = BrainWorkshopEventEncoders(event_width=16)
    collection = encoders.encode(episode.observations[0])
    report = {
        "schema": "brainworkshop-gym-smoke-v1",
        "seed": seed,
        "config": {
            "n_back": config.n_back,
            "trials": config.trials,
            "modalities": config.modalities,
            "trial_ms": config.trial_ms,
        },
        "episode_digest": _episode_digest(episode),
        "target_distribution": {
            str(action): targets.count(action)
            for action in range(ALL_MATCHES + 1)
        },
        "oracle_fast": _score(episode, targets, latency_ms=0),
        "oracle_deadline": _score(
            episode, targets, latency_ms=config.trial_ms),
        "random": _score(episode, random_actions, latency_ms=0),
        "no_op": _score(episode, no_op, latency_ms=0),
        "inverted": _score(episode, inverted, latency_ms=0),
        "event_collection_shape": list(collection.payload.shape),
        "event_count": int(collection.payload.shape[1]),
        "observation_exposes_targets": any(
            hasattr(observation, "target")
            for observation in episode.observations),
    }
    report["gates"] = {
        "deterministic_episode": report["episode_digest"]
        == _episode_digest(generate_brainworkshop_episode(config, seed=seed)),
        "oracle_perfect": report["oracle_fast"]["accuracy"] == 1.0,
        "latency_bonus_small": (
            report["oracle_fast"]["mean_reward"]
            > report["oracle_deadline"]["mean_reward"]
            and report["oracle_fast"]["mean_reward"] <= 1.05),
        "random_below_oracle": report["random"]["accuracy"] < 0.75,
        "counterfactual_inversion_hurts": (
            report["inverted"]["accuracy"] < 0.25),
        "two_independent_events": report["event_count"] == 2,
        "targets_private": not report["observation_exposes_targets"],
    }
    report["accepted"] = all(report["gates"].values())
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=44010)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = run(args.seed)
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text)
    print(text, end="")
    if not report["accepted"]:
        raise SystemExit("Brain Workshop gym smoke audit failed")


if __name__ == "__main__":
    main()
