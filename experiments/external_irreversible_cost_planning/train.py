"""Causal audit of replay-free planning with an absorbing failure state."""

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
    ExternalRandomFeatureTransitionStatistics,
    ExternalTransitionObservation,
)

STATE_WIDTH = 1
INTENTION_WIDTH = 4
FEATURE_WIDTH = 256
HORIZON = 2
BEAM_WIDTH = 5
START_STATE = 0.0
GOAL_STATE = 2.0
TRAP_STATE = -1.0

# Candidate rows are opaque to the deployed planner. Only this verifier-side
# fixture knows that row 2 is the irreversible trap and row 1 is direct.
INTENTIONS = torch.eye(INTENTION_WIDTH, dtype=torch.float32)
COSTS = torch.tensor([5.0, 1.0, 0.0, 4.0])
SHUFFLED_COSTS = torch.tensor([1.0, 5.0, 4.0, 0.0])


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture_observations() -> ExternalTransitionObservation:
    states = torch.tensor([[-1.0], [0.0], [1.0], [2.0]]).repeat_interleave(
        INTENTION_WIDTH,
        dim=0,
    )
    intentions = INTENTIONS.repeat(4, 1)
    scalar_state = states[:, 0]
    action = intentions.argmax(dim=-1)
    safe_step = torch.clamp(scalar_state + 1.0, max=GOAL_STATE)
    non_trap_next = torch.where(
        action == 0,
        safe_step,
        torch.where(
            action == 1,
            torch.full_like(scalar_state, GOAL_STATE),
            torch.where(
                action == 2,
                torch.full_like(scalar_state, TRAP_STATE),
                scalar_state,
            ),
        ),
    )
    next_state = torch.where(
        scalar_state < 0.0,
        torch.full_like(scalar_state, TRAP_STATE),
        non_trap_next,
    ).unsqueeze(-1)
    return ExternalTransitionObservation(
        state=states,
        intention=intentions,
        next_state=next_state,
        confidence=torch.ones(states.shape[0]),
    )


def _learn_models(
    seed: int,
) -> tuple[
    ExternalRandomFeatureTransitionStatistics,
    ExternalAffineTransitionStatistics,
    ExternalTransitionObservation,
    ExternalTransitionObservation,
]:
    observation = _fixture_observations()
    factual_model = ExternalRandomFeatureTransitionStatistics(
        STATE_WIDTH,
        INTENTION_WIDTH,
        feature_width=FEATURE_WIDTH,
        ridge=1e-5,
        seed=seed,
    )
    factual_model.observe(observation)
    cost_model = ExternalAffineTransitionStatistics(
        STATE_WIDTH,
        INTENTION_WIDTH,
        ridge=1e-8,
    )
    scalar_costs = COSTS[observation.intention.argmax(dim=-1)].unsqueeze(-1)
    cost_observation = ExternalTransitionObservation(
        state=observation.state,
        intention=observation.intention,
        next_state=scalar_costs,
        confidence=observation.confidence,
    )
    cost_model.observe(cost_observation)
    return factual_model, cost_model, observation, cost_observation


def _fixture_step(state: float, action: int) -> float:
    if state < 0.0:
        return TRAP_STATE
    if action == 0:
        return min(state + 1.0, GOAL_STATE)
    if action == 1:
        return GOAL_STATE
    if action == 2:
        return TRAP_STATE
    return state


def _execute(intentions: torch.Tensor) -> tuple[float, float, bool]:
    state = START_STATE
    cost = 0.0
    visited_trap = False
    for intention in intentions:
        action = int(intention.argmax().item())
        state = _fixture_step(state, action)
        cost += float(COSTS[action])
        visited_trap = visited_trap or state == TRAP_STATE
    return state, cost, visited_trap


def _plan(
    planner: ExternalModelBasedPlanner,
    *,
    intention_costs: torch.Tensor | None = None,
) -> dict[str, object]:
    result = planner.plan(
        torch.tensor([[START_STATE]]),
        torch.tensor([[GOAL_STATE]]),
        INTENTIONS,
        horizon=HORIZON,
        beam_width=BEAM_WIDTH,
        intention_costs=intention_costs,
        step_cost_weight=0.0 if intention_costs is None else 1.0,
    )
    final_state, actual_cost, visited_trap = _execute(result.intentions[0])
    return {
        "actions": result.intentions[0].argmax(dim=-1).tolist(),
        "intentions": result.intentions[0].tolist(),
        "predicted_states": result.predicted_states[0].flatten().tolist(),
        "final_state": final_state,
        "actual_cost": actual_cost,
        "visited_trap": visited_trap,
        "reaches_goal": final_state == GOAL_STATE,
        "score": float(result.scores.item()),
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    factual_model, cost_model, observation, cost_observation = _learn_models(seed)
    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=1,
        intention_width=INTENTION_WIDTH,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_before = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    factual_before = factual_model.digest()
    cost_before = cost_model.digest()
    predicted_costs = cost_model(
        torch.zeros(INTENTION_WIDTH, STATE_WIDTH),
        INTENTIONS,
    ).flatten()
    planner = ExternalModelBasedPlanner(factual_model, beam_width=BEAM_WIDTH)
    terminal_only = _plan(planner)
    cost_aware = _plan(planner, intention_costs=predicted_costs)
    shuffled_cost = _plan(planner, intention_costs=SHUFFLED_COSTS)
    restored_factual = ExternalRandomFeatureTransitionStatistics.from_payload(
        factual_model.state_payload()
    )
    restored_cost = ExternalAffineTransitionStatistics.from_payload(
        cost_model.state_payload()
    )
    factual_error = float(factual_model.loss(observation))
    cost_error = float(cost_model.loss(cost_observation))
    report = {
        "schema": "neural-computer.external-irreversible-cost-planning.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "feature_width": FEATURE_WIDTH,
            "horizon": HORIZON,
            "beam_width": BEAM_WIDTH,
            "objective": "terminal_goal_plus_learned_opaque_cost_v1",
            "controller_frozen": True,
            "trap_state_is_absorbing": True,
        },
        "metrics": {
            "terminal_only": terminal_only,
            "cost_aware": cost_aware,
            "shuffled_cost": shuffled_cost,
            "predicted_costs": predicted_costs.tolist(),
            "factual_loss": factual_error,
            "cost_loss": cost_error,
            "cost_saving": terminal_only["actual_cost"] - cost_aware["actual_cost"],
        },
        "gates": {
            "terminal_only_reaches_goal": terminal_only["reaches_goal"],
            "cost_aware_reaches_goal": cost_aware["reaches_goal"],
            "cost_aware_avoids_irreversible_trap": not cost_aware["visited_trap"],
            "cost_aware_is_strictly_cheaper": (
                cost_aware["actual_cost"] < terminal_only["actual_cost"]
            ),
            "shuffled_cost_changes_behavior": (
                shuffled_cost["actions"] != cost_aware["actions"]
            ),
            "shuffled_cost_fails_goal": not shuffled_cost["reaches_goal"],
            "factual_model_quality": factual_error < 1e-4,
            "cost_model_quality": cost_error < 1e-6,
            "controller_unchanged": controller_before == _digest(controller),
            "factual_model_unchanged_during_search": (
                factual_before == factual_model.digest()
            ),
            "cost_model_unchanged_during_search": cost_before == cost_model.digest(),
            "persistence_exact": (
                restored_factual.digest() == factual_model.digest()
                and restored_cost.digest() == cost_model.digest()
            ),
        },
        "accounting": {
            "unique_transition_rows_consumed_once": int(observation.state.shape[0]),
            "unique_scalar_cost_outcomes_consumed_once": int(
                cost_observation.state.shape[0]
            ),
            "factual_model_optimizer_updates": 0,
            "cost_model_optimizer_updates": 0,
            "planner_search_optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_regime_replay": 0,
        },
        "model_digests": {
            "factual": factual_model.digest(),
            "cost": cost_model.digest(),
        },
        "promoted": False,
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report["promoted"] = all(report["gates"].values())
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=83311)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
