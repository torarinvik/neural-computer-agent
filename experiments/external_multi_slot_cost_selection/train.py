"""Causal audit of cost-aware selection across a growing factual bank."""

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
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 1
INTENTION_WIDTH = 3
CONTEXT_WIDTH = 3
SLOT_SCALES = (0.7, 1.4, 2.3)
TARGETS = (2.1, 4.2, 6.9)
HORIZON = 3
BEAM_WIDTH = 8
COST_WEIGHT = 1e-3
INTENTIONS = torch.eye(INTENTION_WIDTH, dtype=torch.float32)
ACTION_DELTAS = torch.tensor([2.0, 1.0, 0.0])
ACTION_COSTS = torch.tensor([5.0, 1.0, 0.0])
TRAINING_STATES = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _learn_bank(
    seed: int,
) -> tuple[
    ExternalTransitionModelBank,
    ExternalAffineTransitionStatistics,
    list[ExternalTransitionObservation],
    ExternalTransitionObservation,
]:
    generator = torch.Generator().manual_seed(seed)
    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-5,
        capacity=len(SLOT_SCALES),
    )
    transition_observations: list[ExternalTransitionObservation] = []
    cost_states: list[torch.Tensor] = []
    cost_intentions: list[torch.Tensor] = []
    cost_targets: list[torch.Tensor] = []
    for slot_index, scale in enumerate(SLOT_SCALES):
        context = torch.eye(CONTEXT_WIDTH)[slot_index]
        slot = bank.ensure_context(context)
        states = TRAINING_STATES.repeat_interleave(INTENTION_WIDTH, dim=0)
        intentions = INTENTIONS.repeat(TRAINING_STATES.shape[0], 1)
        permutation = torch.randperm(states.shape[0], generator=generator)
        states = states[permutation]
        intentions = intentions[permutation]
        next_states = states + scale * (intentions * ACTION_DELTAS).sum(
            dim=-1,
            keepdim=True,
        )
        observation = ExternalTransitionObservation(
            state=states,
            intention=intentions,
            next_state=next_states,
            confidence=torch.ones(states.shape[0]),
        )
        bank.adaptation_step(
            observation,
            bank.context_at(slot).unsqueeze(0).expand(states.shape[0], -1),
            None,
        )
        transition_observations.append(observation)
        cost_states.append(states)
        cost_intentions.append(intentions)
        cost_targets.append(intentions @ ACTION_COSTS.unsqueeze(-1))

    cost_model = ExternalAffineTransitionStatistics(
        STATE_WIDTH,
        INTENTION_WIDTH,
        ridge=1e-5,
    )
    cost_observation = ExternalTransitionObservation(
        state=torch.cat(cost_states),
        intention=torch.cat(cost_intentions),
        next_state=torch.cat(cost_targets),
        confidence=torch.ones(len(cost_states) * TRAINING_STATES.shape[0] * 3),
    )
    cost_model.observe(cost_observation)
    return bank, cost_model, transition_observations, cost_observation


def _execute(intentions: torch.Tensor, scale: float) -> tuple[float, float]:
    state = 0.0
    cost = 0.0
    for intention in intentions:
        action = int(intention.argmax().item())
        state += scale * float(ACTION_DELTAS[action])
        cost += float(ACTION_COSTS[action])
    return state, cost


def _select(
    planner: ExternalModelBasedPlanner,
    bank: ExternalTransitionModelBank,
    cost_model: ExternalAffineTransitionStatistics,
    *,
    use_cost: bool,
) -> list[dict[str, object]]:
    predicted_costs = cost_model(
        torch.zeros(INTENTION_WIDTH, STATE_WIDTH),
        INTENTIONS,
    ).flatten()
    selections: list[dict[str, object]] = []
    for target in TARGETS:
        selection = planner.select_bank_model(
            bank,
            torch.zeros(1, STATE_WIDTH),
            torch.tensor([[target]]),
            INTENTIONS,
            horizon=HORIZON,
            beam_width=BEAM_WIDTH,
            intention_costs=predicted_costs if use_cost else None,
            step_cost_weight=COST_WEIGHT if use_cost else 0.0,
        )
        slot_index = bank.physical_index_for_slot_id(selection.selected_slot_id)
        final_state, actual_cost = _execute(
            selection.planning.intentions[0],
            SLOT_SCALES[slot_index],
        )
        selections.append(
            {
                "target": target,
                "selected_slot_id": selection.selected_slot_id,
                "candidate_slot_ids": list(selection.candidate_slot_ids),
                "intentions": selection.planning.intentions[0].tolist(),
                "actions": selection.planning.intentions[0].argmax(dim=-1).tolist(),
                "final_state": final_state,
                "actual_cost": actual_cost,
                "reaches_goal": abs(final_state - target) < 0.05,
                "score": float(selection.scores[selection.selected_slot_id]),
                "all_model_scores": selection.scores.tolist(),
            }
        )
    return selections


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    bank, cost_model, transition_observations, cost_observation = _learn_bank(seed)
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
    bank_before = bank.digest()
    cost_before = cost_model.digest()
    planner = ExternalModelBasedPlanner(bank, beam_width=BEAM_WIDTH)
    terminal_only = _select(planner, bank, cost_model, use_cost=False)
    cost_aware = _select(planner, bank, cost_model, use_cost=True)
    restored_bank = ExternalTransitionModelBank.from_payload(bank.payload())
    restored_cost = ExternalAffineTransitionStatistics.from_payload(
        cost_model.state_payload()
    )
    terminal_cost = sum(float(row["actual_cost"]) for row in terminal_only)
    cost_aware_cost = sum(float(row["actual_cost"]) for row in cost_aware)
    model_losses = [
        float(bank.models[index].loss(observation))
        for index, observation in enumerate(transition_observations)
    ]
    predicted_costs = cost_model(
        torch.zeros(INTENTION_WIDTH, STATE_WIDTH),
        INTENTIONS,
    ).flatten()
    report = {
        "schema": "neural-computer.external-multi-slot-cost-selection.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "slot_scales": list(SLOT_SCALES),
            "targets": list(TARGETS),
            "horizon": HORIZON,
            "beam_width": BEAM_WIDTH,
            "cost_weight": COST_WEIGHT,
            "selection": "goal_conditioned_external_bank_search_v1",
            "context_label_to_planner": False,
            "controller_frozen": True,
        },
        "metrics": {
            "terminal_only": terminal_only,
            "cost_aware": cost_aware,
            "terminal_total_cost": terminal_cost,
            "cost_aware_total_cost": cost_aware_cost,
            "cost_saving": terminal_cost - cost_aware_cost,
            "cost_predictions": predicted_costs.tolist(),
            "model_losses": model_losses,
            "cost_model_loss": float(cost_model.loss(cost_observation)),
        },
        "gates": {
            "terminal_only_reaches_all_goals": all(
                row["reaches_goal"] for row in terminal_only
            ),
            "cost_aware_reaches_all_goals": all(
                row["reaches_goal"] for row in cost_aware
            ),
            "cost_aware_selects_expected_slots": [
                row["selected_slot_id"] for row in cost_aware
            ]
            == [0, 1, 2],
            "cost_aware_beats_terminal_total_cost": cost_aware_cost < terminal_cost,
            "stable_slot_ids_are_preserved": all(
                row["candidate_slot_ids"] == [0, 1, 2] for row in cost_aware
            ),
            "factual_model_quality": max(model_losses) < 1e-5,
            "cost_model_quality": float(cost_model.loss(cost_observation)) < 1e-6,
            "controller_unchanged": controller_before == _digest(controller),
            "bank_unchanged_during_search": bank_before == bank.digest(),
            "cost_model_unchanged_during_search": cost_before == cost_model.digest(),
            "persistence_exact": (
                restored_bank.digest() == bank.digest()
                and restored_cost.digest() == cost_model.digest()
            ),
        },
        "accounting": {
            "unique_transition_rows_consumed_once": sum(
                int(observation.state.shape[0]) for observation in transition_observations
            ),
            "unique_scalar_cost_rows_consumed_once": int(
                cost_observation.state.shape[0]
            ),
            "factual_model_optimizer_updates": 0,
            "cost_model_optimizer_updates": 0,
            "planner_search_optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_regime_replay": 0,
        },
        "bank_digest": bank.digest(),
        "cost_model_digest": cost_model.digest(),
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
    parser.add_argument("--seed", type=int, default=83321)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
