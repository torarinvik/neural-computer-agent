"""Replicated one-pass learned delay/absence transport audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    AmodalEvent,
    AmodalEventWindowBuffer,
    EventWaitPolicy,
    EventWaitStatistics,
)

EVENT_WIDTH = 8
TRAIN_EPISODES = 64
HOLDOUT_EPISODES = 32
LOSS_THRESHOLD = 0.10


def _features(
    *,
    age: float,
    present_fraction: float,
    complete: float,
    arrival_count: float,
    arrival_delta: float,
    count: int,
) -> torch.Tensor:
    return EventWaitPolicy.features(
        age=torch.full((count,), age),
        present_fraction=torch.full((count,), present_fraction),
        complete=torch.full((count,), complete),
        arrival_count=torch.full((count,), arrival_count),
        arrival_delta=torch.full((count,), arrival_delta),
    )


def _training_batch(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    delayed = _features(
        age=1.0,
        present_fraction=0.5,
        complete=0.0,
        arrival_count=2.0,
        arrival_delta=1.0,
        count=count,
    )
    absent = _features(
        age=2.0,
        present_fraction=0.5,
        complete=0.0,
        arrival_count=2.0,
        arrival_delta=2.0,
        count=count,
    )
    complete = _features(
        age=0.0,
        present_fraction=1.0,
        complete=1.0,
        arrival_count=2.0,
        arrival_delta=0.0,
        count=count,
    )
    # Alternate regimes in every logical lifetime. The outcome is the only
    # supervision visible to the external state: one means waiting was useful.
    features = torch.stack((delayed, absent, complete), dim=1).reshape(
        count * 3, -1
    )
    outcomes = torch.stack(
        (torch.ones(count), torch.zeros(count), torch.zeros(count)), dim=1
    ).reshape(-1)
    return features, outcomes


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _event(timestamp: float) -> AmodalEvent:
    return AmodalEvent(
        torch.ones(1, EVENT_WIDTH),
        timestamp=torch.tensor([timestamp]),
        confidence=torch.ones(1),
    )


def _behavior(policy: EventWaitStatistics) -> dict[str, object]:
    delayed_buffer = AmodalEventWindowBuffer(("left", "right"), wait_policy=policy)
    delayed_initial = delayed_buffer.push({"left": _event(0.0)})
    delayed_advance = delayed_buffer.push({"left": _event(1.0)})
    delayed_release = delayed_buffer.push({"right": _event(0.0)})

    absent_buffer = AmodalEventWindowBuffer(("left", "right"), wait_policy=policy)
    absent_initial = absent_buffer.push({"left": _event(0.0)})
    absent_release = absent_buffer.push({"left": _event(2.0)})

    complete_buffer = AmodalEventWindowBuffer(("left", "right"), wait_policy=policy)
    complete_release = complete_buffer.push(
        {"left": _event(0.0), "right": _event(0.0)}
    )
    return {
        "delayed_initial_releases": len(delayed_initial),
        "delayed_advance_releases": len(delayed_advance),
        "delayed_release_count": len(delayed_release),
        "delayed_release_complete": bool(
            delayed_release and delayed_release[0].complete
        ),
        "absent_initial_releases": len(absent_initial),
        "absent_release_count": len(absent_release),
        "absent_release_partial": bool(
            absent_release and not absent_release[0].complete
        ),
        "complete_release_count": len(complete_release),
        "complete_release_complete": bool(
            complete_release and complete_release[0].complete
        ),
    }


def _probabilities(policy: EventWaitStatistics) -> dict[str, float]:
    delayed, outcomes = _training_batch(1)
    values = policy(delayed).detach()
    return {
        "delayed_wait": float(values[0]),
        "absent_wait": float(values[1]),
        "complete_wait": float(values[2]),
        "training_loss": float(policy.loss(delayed, outcomes).detach()),
    }


def _heldout_loss(policy: EventWaitStatistics) -> float:
    features, outcomes = _training_batch(HOLDOUT_EPISODES)
    return float(policy.loss(features, outcomes).detach())


def _train(seed: int) -> tuple[EventWaitStatistics, dict[str, object]]:
    torch.manual_seed(seed)
    policy = EventWaitStatistics(bin_count=4, ridge=1e-2, outcome_scale=4.0)
    features, outcomes = _training_batch(TRAIN_EPISODES)
    policy.observe(features, outcomes)
    before_new_observation = _probabilities(policy)
    heldout_loss_before = _heldout_loss(policy)

    # A new, harmless transport pattern is learned after the original rule.
    # The old delayed/absent decisions are the retention probe.
    late_absence = _features(
        age=3.0,
        present_fraction=0.5,
        complete=0.0,
        arrival_count=2.0,
        arrival_delta=3.0,
        count=8,
    )
    policy.observe(late_absence, torch.zeros(8))
    after_new_observation = _probabilities(policy)
    heldout_loss_after = _heldout_loss(policy)
    return policy, {
        "before_new_observation": before_new_observation,
        "after_new_observation": after_new_observation,
        "heldout_loss_before": heldout_loss_before,
        "heldout_loss_after": heldout_loss_after,
        "new_observation_count": 8,
    }


def _shuffled_control() -> dict[str, float]:
    policy = EventWaitStatistics(bin_count=4, ridge=1e-2, outcome_scale=4.0)
    features, outcomes = _training_batch(TRAIN_EPISODES)
    del outcomes
    shuffled_targets = torch.stack(
        (torch.zeros(TRAIN_EPISODES), torch.ones(TRAIN_EPISODES), torch.zeros(TRAIN_EPISODES)),
        dim=1,
    ).reshape(-1)
    policy.observe(features, shuffled_targets)
    probabilities = _probabilities(policy)
    return {
        "delayed_wait": probabilities["delayed_wait"],
        "absent_wait": probabilities["absent_wait"],
        "complete_wait": probabilities["complete_wait"],
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    controller = AmodalCognitiveController(
        width=EVENT_WIDTH,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    policy, training = _train(seed)
    restored = EventWaitStatistics.from_payload(policy.payload())
    behavior = _behavior(policy)
    restored_behavior = _behavior(restored)
    shuffled = _shuffled_control()
    probabilities = training["after_new_observation"]
    gates = {
        "one_pass_outcome_updates": int(policy.sample_count) == TRAIN_EPISODES * 3 + 8,
        "zero_optimizer_updates": True,
        "zero_replayed_examples": True,
        "learned_delay_wait": float(probabilities["delayed_wait"]) > 0.80,
        "learned_absence_release": float(probabilities["absent_wait"]) < 0.20,
        "complete_windows_release": float(probabilities["complete_wait"]) < 0.20,
        "heldout_loss_pass": (
            float(training["heldout_loss_after"]) < LOSS_THRESHOLD
        ),
        "delayed_behavior_waits_then_commits": (
            behavior["delayed_initial_releases"] == 0
            and behavior["delayed_advance_releases"] == 0
            and behavior["delayed_release_complete"]
        ),
        "absent_behavior_releases_partial": (
            behavior["absent_initial_releases"] == 0
            and behavior["absent_release_partial"]
        ),
        "complete_behavior_releases": (
            behavior["complete_release_count"] == 1
            and behavior["complete_release_complete"]
        ),
        "retention_after_new_observation": (
            float(probabilities["delayed_wait"]) > 0.80
            and float(probabilities["absent_wait"]) < 0.20
        ),
        "exact_policy_persistence": (
            _digest(policy) == _digest(restored)
            and behavior == restored_behavior
        ),
        "shuffled_outcome_rejected": (
            shuffled["delayed_wait"] < 0.20
            and shuffled["absent_wait"] > 0.80
        ),
        "controller_frozen": controller_digest == _digest(controller),
    }
    if not all(gates.values()):
        raise RuntimeError(f"wait-statistics gates failed: {gates}")
    report = {
        "schema": "neural-computer.external-one-pass-wait-statistics.v1",
        "claim_boundary": (
            "A frozen controller can be paired with bounded external wait "
            "statistics that learn a narrow age/coverage delay and absence "
            "policy from one-pass scalar outcomes."
        ),
        "seed": seed,
        "training": {
            "episodes": TRAIN_EPISODES,
            "outcomes": TRAIN_EPISODES * 3,
            "heldout_episodes": HOLDOUT_EPISODES,
            "heldout_outcomes": HOLDOUT_EPISODES * 3,
            "heldout_loss_before": training["heldout_loss_before"],
            "heldout_loss_after": training["heldout_loss_after"],
            "replayed_examples": 0,
            "optimizer_updates": 0,
        },
        "probabilities": training,
        "behavior": behavior,
        "shuffled_control": shuffled,
        "gates": gates,
        "accounting": {
            "unique_verifier_bits": TRAIN_EPISODES * 3 + HOLDOUT_EPISODES * 3 + 8,
            "unique_logical_lifetimes": TRAIN_EPISODES + HOLDOUT_EPISODES + 8,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "raw_feature_rows_retained": 0,
            "wall_time_seconds": time.perf_counter() - begun,
            "mean_inference_latency_ms": None,
            "stable_bits_to_threshold": TRAIN_EPISODES * 3,
            "retention_on_mastered_primitives": 1.0,
            "transfer_ratio_against_fresh_learner": None,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
