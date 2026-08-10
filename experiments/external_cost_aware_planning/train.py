"""Causal audit of optional opaque step-cost planning."""

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

STATE_WIDTH = 1
INTENTION_WIDTH = 3
RIDGE = 1e-8
HORIZON = 2
BEAM_WIDTH = 3
START_STATE = 0.0
GOAL_STATE = 2.0

# The candidates are deliberately opaque one-hot vectors. Their physical
# interpretation belongs only to this verifier-side fixture.
INTENTIONS = torch.eye(INTENTION_WIDTH, dtype=torch.float32)
DELTAS = torch.tensor([2.0, 1.0, 0.0])
COSTS = torch.tensor([1.0, 5.0, 0.0])
SHUFFLED_COSTS = torch.tensor([5.0, 1.0, 0.0])


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _learn_factual_model() -> tuple[
    ExternalAffineTransitionStatistics,
    ExternalTransitionObservation,
]:
    states = torch.tensor([[-1.0], [0.0], [1.0], [2.0], [3.0]]).repeat_interleave(
        INTENTION_WIDTH,
        dim=0,
    )
    intentions = INTENTIONS.repeat(5, 1)
    next_states = states + (intentions * DELTAS).sum(dim=-1, keepdim=True)
    observation = ExternalTransitionObservation(
        state=states,
        intention=intentions,
        next_state=next_states,
        confidence=torch.ones(states.shape[0]),
    )
    model = ExternalAffineTransitionStatistics(
        STATE_WIDTH,
        INTENTION_WIDTH,
        ridge=RIDGE,
    )
    model.observe(observation)
    return model, observation


def _action_index(intention: torch.Tensor) -> int:
    return int(intention.argmax().item())


def _execute(intentions: torch.Tensor) -> tuple[float, float]:
    state = START_STATE
    cost = 0.0
    for intention in intentions:
        index = _action_index(intention)
        state += float(DELTAS[index])
        cost += float(COSTS[index])
    return state, cost


def _run_plan(
    planner: ExternalModelBasedPlanner,
    *,
    intention_costs: torch.Tensor | None = None,
    step_cost_weight: float = 0.0,
) -> dict[str, object]:
    result = planner.plan(
        torch.tensor([[START_STATE]]),
        torch.tensor([[GOAL_STATE]]),
        INTENTIONS,
        horizon=HORIZON,
        beam_width=BEAM_WIDTH,
        intention_costs=intention_costs,
        step_cost_weight=step_cost_weight,
    )
    final_state, actual_cost = _execute(result.intentions[0])
    return {
        "intentions": result.intentions[0].tolist(),
        "predicted_states": result.predicted_states[0].flatten().tolist(),
        "final_state": final_state,
        "actual_cost": actual_cost,
        "score": float(result.scores.item()),
        "reaches_goal": abs(final_state - GOAL_STATE) < 1e-5,
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    model, observation = _learn_factual_model()
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
    model_before_search = model.digest()
    planner = ExternalModelBasedPlanner(model, beam_width=BEAM_WIDTH)
    terminal_only = _run_plan(planner)
    cost_aware = _run_plan(
        planner,
        intention_costs=COSTS,
        step_cost_weight=1.0,
    )
    shuffled_cost = _run_plan(
        planner,
        intention_costs=SHUFFLED_COSTS,
        step_cost_weight=1.0,
    )
    restored = ExternalAffineTransitionStatistics.from_payload(model.state_payload())
    report = {
        "schema": "neural-computer.external-cost-aware-planning.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "horizon": HORIZON,
            "beam_width": BEAM_WIDTH,
            "objective": "terminal_goal_plus_opaque_step_cost_v1",
            "controller_frozen": True,
        },
        "metrics": {
            "terminal_only": terminal_only,
            "cost_aware": cost_aware,
            "shuffled_cost": shuffled_cost,
            "cost_saving": float(
                terminal_only["actual_cost"] - cost_aware["actual_cost"]
            ),
        },
        "gates": {
            "terminal_only_reaches_goal": terminal_only["reaches_goal"],
            "cost_aware_reaches_goal": cost_aware["reaches_goal"],
            "cost_aware_is_strictly_cheaper": (
                cost_aware["actual_cost"] < terminal_only["actual_cost"]
            ),
            "shuffled_cost_changes_behavior": (
                shuffled_cost["intentions"] != cost_aware["intentions"]
            ),
            "controller_unchanged": controller_before == _digest(controller),
            "model_unchanged_during_search": model_before_search == model.digest(),
            "persistence_exact": restored.digest() == model.digest(),
            "factual_loss_is_small": float(model.loss(observation)) < 1e-6,
        },
        "accounting": {
            "unique_transition_rows_consumed_once": int(observation.state.shape[0]),
            "unique_verifier_cost_scalars": INTENTION_WIDTH,
            "planner_search_optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_regime_replay": 0,
        },
        "model_digest": model.digest(),
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
    parser.add_argument("--seed", type=int, default=83301)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
