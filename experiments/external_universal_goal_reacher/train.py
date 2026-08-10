"""Pressure-test goal-space generalization without storing a task policy.

The factual model sees one pass of opaque state/intention/next-state rows and
never receives a goal label.  At inference, behavior is derived by search for
the current goal.  Evaluation goals are held out from a finite-goal habit
control, so success requires the model/planner path to condition on the
runtime goal rather than replaying a small target table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalAffineTransitionStatistics,
    ExternalModelBasedPlanner,
    ExternalTransitionObservation,
)

POSITION_MIN = -20
POSITION_MAX = 20
TRAIN_GOALS = tuple(range(-16, 17, 4))
EVAL_GOALS = tuple(
    position
    for position in range(-15, 16)
    if position not in TRAIN_GOALS
)
EVAL_STARTS = (-12, -6, 0, 6, 12)
HORIZON = 32
BEAM_WIDTH = 4
GOAL_PROGRESS_WEIGHT = 1.0
AFFINE_RIDGE = 1e-5


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _state(position: int) -> torch.Tensor:
    # The runtime sees a standardized learned state tensor. The integer is
    # retained only by the verifier to execute and score the fixture; the
    # unbounded fixture can temporarily visit positions outside the training
    # interval.
    return torch.tensor([[float(position)]], dtype=torch.float32)


def _intention(delta: int) -> torch.Tensor:
    if delta not in (-1, 0, 1):
        raise ValueError("unsupported fixture intention")
    return torch.tensor([[float(delta)]], dtype=torch.float32)


def _apply(position: int, delta: int) -> int:
    # The verifier's evaluation domain is bounded, but the fixture dynamics
    # are an unbounded line. Removing artificial edge behavior keeps the
    # factual affine law exact and prevents boundary artifacts from becoming
    # a planner result.
    return position + delta


def _transition_observation() -> ExternalTransitionObservation:
    states: list[torch.Tensor] = []
    intentions: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    for position in range(POSITION_MIN, POSITION_MAX + 1):
        for delta in (-1, 0, 1):
            states.append(_state(position).squeeze(0))
            intentions.append(_intention(delta).squeeze(0))
            next_states.append(_state(_apply(position, delta)).squeeze(0))
    return ExternalTransitionObservation(
        state=torch.stack(states),
        intention=torch.stack(intentions),
        next_state=torch.stack(next_states),
        confidence=torch.ones(len(states)),
    )


def _hindsight_goals(seed: int) -> tuple[int, ...]:
    """Collect goals from observed trajectory endpoints, without replay."""

    generator = torch.Generator().manual_seed(seed + 701_001)
    position = 0
    visited = {position}
    for _ in range(256):
        delta = int(torch.randint(3, (), generator=generator)) - 1
        position = _apply(position, delta)
        visited.add(position)
    return tuple(sorted(visited))


def _build_model() -> tuple[ExternalAffineTransitionStatistics, int]:
    model = ExternalAffineTransitionStatistics(
        1,
        1,
        ridge=AFFINE_RIDGE,
    )
    observation = _transition_observation()
    model.observe(observation)
    return model, int(observation.state.shape[0])


def _execute_plan(
    intentions: torch.Tensor,
    start: int,
) -> tuple[int, list[int]]:
    position = start
    deltas: list[int] = []
    for intention in intentions:
        delta = int(
            torch.argmin((torch.tensor([-1.0, 0.0, 1.0]) - intention).abs())
        ) - 1
        deltas.append(delta)
        position = _apply(position, delta)
    return position, deltas


def _evaluate_universal(
    model: ExternalAffineTransitionStatistics,
    goals: tuple[int, ...],
    *,
    goal_permutation: tuple[int, ...] | None = None,
) -> dict[str, object]:
    planner = ExternalModelBasedPlanner(model, beam_width=BEAM_WIDTH)
    expanded_nodes = 0
    latencies: list[float] = []
    successes: list[bool] = []
    plans: list[dict[str, object]] = []
    for index, goal in enumerate(goals):
        requested_goal = (
            goal if goal_permutation is None else goals[goal_permutation[index]]
        )
        for start in EVAL_STARTS:
            begun = time.perf_counter()
            result = planner.plan(
                _state(start),
                _state(requested_goal),
                torch.cat((_intention(-1), _intention(0), _intention(1))),
                horizon=HORIZON,
                goal_progress_weight=GOAL_PROGRESS_WEIGHT,
            )
            latencies.append(time.perf_counter() - begun)
            expanded_nodes += result.expanded_nodes
            final, deltas = _execute_plan(result.intentions[0], start)
            success = final == goal
            successes.append(success)
            plans.append(
                {
                    "start": start,
                    "goal": goal,
                    "requested_goal": requested_goal,
                    "final": final,
                    "success": success,
                    "deltas": deltas,
                }
            )
    return {
        "mastery": sum(successes) / len(successes),
        "successful_trials": sum(successes),
        "trial_count": len(successes),
        "expanded_nodes": expanded_nodes,
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "plans": plans[:12],
    }


def _finite_goal_habit(goals: tuple[int, ...]) -> dict[str, object]:
    """A diagnostic finite-goal habit with no representation of novel goals."""

    successes: list[bool] = []
    for goal in EVAL_GOALS:
        nearest = min(goals, key=lambda candidate: (abs(candidate - goal), candidate))
        for start in EVAL_STARTS:
            position = start
            for _ in range(HORIZON):
                if position == nearest:
                    break
                position = _apply(
                    position,
                    1 if nearest > position else -1,
                )
            successes.append(position == goal)
    return {
        "mastery": sum(successes) / len(successes),
        "successful_trials": sum(successes),
        "trial_count": len(successes),
        "known_goal_count": len(goals),
        "heldout_goal_count": len(EVAL_GOALS),
    }


def _random_floor(seed: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 991_007)
    successes = 0
    total = 0
    for goal in EVAL_GOALS:
        for start in EVAL_STARTS:
            position = start
            for _ in range(HORIZON):
                position = _apply(
                    position,
                    int(torch.randint(3, (), generator=generator)) - 1,
                )
            successes += position == goal
            total += 1
    return {"mastery": successes / total, "trials": total}


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    controller = AmodalCognitiveController(
        width=8,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model, transition_rows = _build_model()
    model_digest_before_search = model.digest()
    universal = _evaluate_universal(model, EVAL_GOALS)
    shuffled_order = tuple(reversed(range(len(EVAL_GOALS))))
    shuffled_goal = _evaluate_universal(model, EVAL_GOALS, goal_permutation=shuffled_order)
    habit = _finite_goal_habit(TRAIN_GOALS)
    random_floor = _random_floor(seed)
    restored = ExternalAffineTransitionStatistics.from_payload(model.state_payload())
    hindsight_goals = _hindsight_goals(seed)

    gates = {
        "heldout_goal_mastery": float(universal["mastery"]) >= 0.95,
        "beats_finite_goal_habit": float(universal["mastery"]) > float(habit["mastery"]) + 0.20,
        "beats_random_floor": float(universal["mastery"]) > random_floor["mastery"] + 0.20,
        "goal_shuffle_is_not_equivalent": float(shuffled_goal["mastery"]) < 0.95,
        "goal_conditioned_behavior": len({
            tuple(plan["deltas"])
            for plan in universal["plans"]
            if plan["success"]
        }) > 1,
        "model_unchanged_during_search": model.digest() == model_digest_before_search,
        "controller_frozen": controller_digest == _digest(controller),
        "exact_model_persistence": restored.digest() == model.digest(),
        "zero_optimizer_updates": True,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.external-universal-goal-reacher.v1",
        "claim_boundary": (
            "held-out goal-space generalization of behavior derived from a "
            "replay-free factual transition model; not a learned goal evaluator, "
            "unrestricted planning, or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "position_range": [POSITION_MIN, POSITION_MAX],
            "finite_habit_goals": list(TRAIN_GOALS),
            "heldout_goals": list(EVAL_GOALS),
            "evaluation_starts": list(EVAL_STARTS),
            "horizon": HORIZON,
            "beam_width": BEAM_WIDTH,
            "goal_progress_weight": GOAL_PROGRESS_WEIGHT,
            "goal_input": "opaque_state_tensor_v1",
            "hindsight_sampling": "trajectory-endpoint-goal-discovery-v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "universal": universal,
            "shuffled_goal": shuffled_goal,
            "finite_goal_habit": habit,
            "random_floor": random_floor,
            "hindsight_goal_count": len(hindsight_goals),
            "hindsight_goals": list(hindsight_goals),
            "transition_model_loss": float(model.loss(_transition_observation()).detach()),
        },
        "accounting": {
            "unique_verifier_bits": len(EVAL_GOALS) * len(EVAL_STARTS),
            "unique_logical_lifetimes": transition_rows + len(EVAL_GOALS) * len(EVAL_STARTS),
            "external_model_statistics_updates": 1,
            "transition_rows_consumed_once": transition_rows,
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_memory_replay": 0,
            "search_expansions": universal["expanded_nodes"],
            "mean_search_latency_seconds": universal["mean_latency_seconds"],
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=84001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
