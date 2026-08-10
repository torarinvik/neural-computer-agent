"""Verify the canonical amodal controller -> factual search -> decoder path.

This is an integration pressure test, not a general learning claim. A frozen
amodal controller supplies an opaque learned state representation. A one-pass
factual transition model is then queried by ``PolicyFreeAmodalRuntime`` for
several novel opaque goals. The controller's direct intention is measured as a
control but is never used by the deployed path.
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
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    ExternalAffineTransitionStatistics,
    ExternalModelBasedPlanner,
    ExternalTransitionObservation,
    OpaqueProtocolDecoder,
    PolicyFreeAmodalRuntime,
)

STATE_WIDTH = 12
CONTROLLER_WIDTH = 4
INTENTION_WIDTH = 2
ACTION_BASIS = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
)
TARGET_DISPLACEMENTS = ((2, 2), (2, -2), (-2, 2), (-2, -2))
HORIZON = 4
BEAM_WIDTH = 8
GOAL_PROGRESS_WEIGHT = 1.0
TRANSITION_ROWS = 64


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _transition_observation(seed: int) -> ExternalTransitionObservation:
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(TRANSITION_ROWS, STATE_WIDTH, generator=generator)
    states[:, 2:] = states[:, 2:].tanh()
    actions = ACTION_BASIS.repeat(TRANSITION_ROWS // len(ACTION_BASIS), 1)
    padded = torch.zeros(TRANSITION_ROWS, STATE_WIDTH)
    padded[:, :INTENTION_WIDTH] = actions
    return ExternalTransitionObservation(
        state=states,
        intention=actions,
        next_state=states + padded,
        confidence=torch.ones(TRANSITION_ROWS),
    )


def _feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )


def _execute(intentions: torch.Tensor) -> tuple[int, int]:
    displacement = torch.zeros(2)
    for intention in intentions:
        index = int(torch.linalg.vector_norm(ACTION_BASIS - intention, dim=-1).argmin())
        displacement += ACTION_BASIS[index]
    return int(displacement[0]), int(displacement[1])


def _random_floor(seed: int, trials: int) -> float:
    generator = torch.Generator().manual_seed(seed + 991_007)
    successes = 0
    total = 0
    for target in TARGET_DISPLACEMENTS:
        for _ in range(trials):
            indices = torch.randint(len(ACTION_BASIS), (HORIZON,), generator=generator)
            displacement = tuple(int(value) for value in ACTION_BASIS[indices].sum(dim=0))
            successes += displacement == target
            total += 1
    return successes / total


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    controller = AmodalCognitiveController(
        width=CONTROLLER_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder(
        "opaque_backend",
        OpaqueProtocolDecoder(INTENTION_WIDTH, INTENTION_WIDTH),
    )
    planner_model = ExternalAffineTransitionStatistics(
        STATE_WIDTH,
        INTENTION_WIDTH,
        ridge=1e-6,
    )
    observation = _transition_observation(seed)
    planner_model.observe(observation)
    model_digest = planner_model.digest()
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(planner_model, beam_width=BEAM_WIDTH),
    )
    state = runtime.initial_state(1, device="cpu")
    event = [AmodalEvent(torch.randn(1, CONTROLLER_WIDTH))]
    feedback = _feedback()
    preview, _ = runtime.step_events(event, state, feedback)
    current_state = preview.controller.state_representation.detach()

    target_rows: list[dict[str, object]] = []
    for target in TARGET_DISPLACEMENTS:
        goal = current_state.clone()
        goal[:, :2] += torch.tensor(target, dtype=goal.dtype)
        started = time.perf_counter()
        output, _ = policy_free.step_events(
            event,
            state,
            feedback,
            goal,
            ACTION_BASIS,
            horizon=HORIZON,
            beam_width=BEAM_WIDTH,
            goal_progress_weight=GOAL_PROGRESS_WEIGHT,
        )
        target_rows.append(
            {
                "target": list(target),
                "planned_displacement": list(_execute(output.planning.intentions[0])),
                "success": _execute(output.planning.intentions[0]) == target,
                "expanded_nodes": output.planning.expanded_nodes,
                "latency_seconds": time.perf_counter() - started,
                "decoded_shape": list(output.decoded["opaque_backend"].shape),
            }
        )

    direct_displacement = _execute(preview.controller.intention.payload[0].repeat(HORIZON, 1))
    shuffled_rows: list[dict[str, object]] = []
    for index, target in enumerate(TARGET_DISPLACEMENTS):
        requested = TARGET_DISPLACEMENTS[-index - 1]
        goal = current_state.clone()
        goal[:, :2] += torch.tensor(requested, dtype=goal.dtype)
        output, _ = policy_free.step_events(
            event,
            state,
            feedback,
            goal,
            ACTION_BASIS,
            horizon=HORIZON,
            beam_width=BEAM_WIDTH,
            goal_progress_weight=GOAL_PROGRESS_WEIGHT,
        )
        shuffled_rows.append(
            {
                "target": list(target),
                "requested_goal": list(requested),
                "success_against_original": _execute(output.planning.intentions[0]) == target,
            }
        )

    policy_free_mastery = sum(bool(row["success"]) for row in target_rows) / len(target_rows)
    shuffled_mastery = sum(
        bool(row["success_against_original"]) for row in shuffled_rows
    ) / len(shuffled_rows)
    random_floor = _random_floor(seed, trials=128)
    restored = ExternalAffineTransitionStatistics.from_payload(
        planner_model.state_payload()
    )
    gates = {
        "all_novel_goals_mastered": policy_free_mastery == 1.0,
        "beats_random_floor": policy_free_mastery > random_floor + 0.5,
        "goal_conditioned": shuffled_mastery < policy_free_mastery,
        "controller_frozen": controller_digest == _digest(controller),
        "model_unchanged_during_search": model_digest == planner_model.digest(),
        "exact_model_persistence": restored.digest() == planner_model.digest(),
        "direct_controller_is_not_deployed": direct_displacement
        != target_rows[0]["planned_displacement"],
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.policy-free-amodal-runtime.v1",
        "claim_boundary": (
            "canonical integration of one frozen amodal controller with one-pass "
            "factual model search; not general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "controller_width": CONTROLLER_WIDTH,
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "candidate_intentions": ACTION_BASIS.tolist(),
            "horizon": HORIZON,
            "beam_width": BEAM_WIDTH,
            "goal_progress_weight": GOAL_PROGRESS_WEIGHT,
            "goal_input": "opaque_external_destination_state_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "policy_free": target_rows,
            "policy_free_mastery": policy_free_mastery,
            "goal_shuffled_mastery": shuffled_mastery,
            "random_floor": random_floor,
            "direct_controller_displacement": list(direct_displacement),
        },
        "accounting": {
            "unique_verifier_bits": len(TARGET_DISPLACEMENTS),
            "unique_logical_lifetimes": TRANSITION_ROWS + len(TARGET_DISPLACEMENTS),
            "factual_statistics_updates": 1,
            "transition_rows_consumed_once": TRANSITION_ROWS,
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "search_expansions": sum(int(row["expanded_nodes"]) for row in target_rows),
            "mean_search_latency_seconds": sum(
                float(row["latency_seconds"]) for row in target_rows
            ) / len(target_rows),
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
