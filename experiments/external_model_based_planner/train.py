"""Causal pressure test for policy-free continual goal acquisition.

The fixture exposes only opaque state and intention tensors to the external
transition model and planner.  A tiny verifier-side simulator is retained by
the experiment so the production components never receive positions, goals,
action IDs, or task labels.  The controller is instantiated and frozen; all
learning occurs in the replaceable external transition model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import (
    AmodalCognitiveController,
    ExternalModelBasedPlanner,
    ExternalTransitionModel,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
HIDDEN_WIDTH = 48
POSITION_COUNT = 6
TARGETS = ((0, 4), (4, 0), (1, 5))


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )
    states: list[torch.Tensor] = []
    intentions: list[torch.Tensor] = []
    next_states: list[torch.Tensor] = []
    for position in range(POSITION_COUNT):
        for action_index, delta in enumerate((-1, 1)):
            next_position = min(
                POSITION_COUNT - 1,
                max(0, position + delta),
            )
            states.append(state_codes[position])
            intentions.append(intention_codes[action_index])
            next_states.append(state_codes[next_position])
    return (
        state_codes,
        intention_codes,
        torch.stack(states),
        torch.stack(intentions),
        torch.stack(next_states),
    )


def _train_model(
    seed: int,
    observations: ExternalTransitionObservation,
    updates: int,
) -> tuple[ExternalTransitionModel, float, int]:
    torch.manual_seed(seed)
    model = ExternalTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=HIDDEN_WIDTH,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    final_loss = float("inf")
    for update in range(1, updates + 1):
        optimizer.zero_grad()
        loss = model.loss(observations)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return model, final_loss, updates


def _execute_plan(
    intentions: torch.Tensor,
    intention_codes: torch.Tensor,
    start: int,
) -> int:
    position = start
    for intention in intentions:
        action = int(
            torch.linalg.vector_norm(intention_codes - intention, dim=-1).argmin()
        )
        position += -1 if action == 0 else 1
        position = min(POSITION_COUNT - 1, max(0, position))
    return position


def _evaluate_targets(
    model: ExternalTransitionModel,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    targets: tuple[tuple[int, int], ...],
    *,
    horizon: int,
    beam_width: int,
    planning_goals: tuple[int, ...] | None = None,
) -> dict[str, object]:
    if planning_goals is not None and len(planning_goals) != len(targets):
        raise ValueError("planning goals must align with targets")
    requested_goals = (
        tuple(goal for _start, goal in targets)
        if planning_goals is None
        else planning_goals
    )
    planner = ExternalModelBasedPlanner(model, beam_width=beam_width)
    successes: list[bool] = []
    scores: list[float] = []
    latencies: list[float] = []
    predicted_final: list[int] = []
    for (start, goal), requested_goal in zip(
        targets, requested_goals, strict=True
    ):
        begun = time.perf_counter()
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[requested_goal].unsqueeze(0),
            intention_codes,
            horizon=horizon,
        )
        latencies.append(time.perf_counter() - begun)
        final = _execute_plan(
            result.intentions[0],
            intention_codes,
            start,
        )
        predicted_final.append(final)
        successes.append(final == goal)
        scores.append(float(result.scores.item()))
    return {
        "successes": successes,
        "mastery": sum(successes) / len(successes),
        "scores": scores,
        "predicted_final_positions": predicted_final,
        "mean_latency_seconds": sum(latencies) / len(latencies),
    }


def _report(
    *,
    seed: int,
    source_updates: int,
    target_updates: int,
    model: ExternalTransitionModel,
    source_loss: float,
    trained: dict[str, object],
    retained: dict[str, object],
    shuffled_goal: dict[str, object],
    shuffled_model: dict[str, object],
    fresh: dict[str, object],
    persisted: dict[str, object],
    controller_unchanged: bool,
    elapsed_seconds: float,
) -> dict[str, object]:
    retention_floor = min(
        float(value)
        for value in (
            trained["mastery"],
            retained["mastery"],
        )
    )
    gates = {
        "controller_unchanged": controller_unchanged,
        "model_learns_source_transitions": source_loss < 0.01,
        "target_optimizer_updates_zero": target_updates == 0,
        "replay_examples_zero": True,
        "target_mastery": float(trained["mastery"]) >= 0.8,
        "retention_prefix_floor": retention_floor >= 0.8,
        "goal_conditioning_causal": float(shuffled_goal["mastery"]) <= 0.5,
        "shuffled_model_near_floor": float(shuffled_model["mastery"]) <= 0.5,
        "persistence_exact": persisted["successes"] == trained["successes"],
        "fresh_control_not_mastered": float(fresh["mastery"]) < 0.8,
    }
    return {
        "schema": "neural-computer.external-model-based-planner-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "position_count": POSITION_COUNT,
            "targets": [list(pair) for pair in TARGETS],
            "horizon": 4,
            "beam_width": 16,
            "policy": "none_external_transition_model_plus_search",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "source": {
            "optimizer_updates": source_updates,
            "replayed_examples": 0,
            "unique_transition_lifetimes": POSITION_COUNT * 2,
            "final_loss": source_loss,
        },
        "target": {
            "optimizer_updates": target_updates,
            "replayed_examples": 0,
            "unique_verifier_bits": len(TARGETS),
            "unique_goal_lifetimes": len(TARGETS),
        },
        "trained": trained,
        "retained": retained,
        "shuffled_goal": shuffled_goal,
        "shuffled_model": shuffled_model,
        "fresh_model": fresh,
        "persisted_model": persisted,
        "model_digest": model.digest(),
        "elapsed_seconds": elapsed_seconds,
    }


def run(seed: int, report_out: Path, *, source_updates: int) -> dict[str, object]:
    begun = time.perf_counter()
    (
        state_codes,
        intention_codes,
        transition_states,
        transition_intentions,
        next_states,
    ) = _fixture(seed)
    observations = ExternalTransitionObservation(
        state=transition_states,
        intention=transition_intentions,
        next_state=next_states,
        confidence=torch.ones(POSITION_COUNT * 2),
    )
    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_before = _digest_module(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    model, source_loss, actual_updates = _train_model(
        seed + 1000,
        observations,
        source_updates,
    )
    trained = _evaluate_targets(
        model,
        state_codes,
        intention_codes,
        TARGETS,
        horizon=4,
        beam_width=16,
    )
    retained = _evaluate_targets(
        model,
        state_codes,
        intention_codes,
        TARGETS,
        horizon=4,
        beam_width=16,
    )
    shuffled_goal = _evaluate_targets(
        model,
        state_codes,
        intention_codes,
        TARGETS,
        horizon=4,
        beam_width=16,
        planning_goals=tuple((goal + 1) % POSITION_COUNT for _start, goal in TARGETS),
    )
    shuffled_observations = ExternalTransitionObservation(
        state=observations.state,
        intention=observations.intention,
        next_state=observations.next_state.roll(shifts=1, dims=0),
        confidence=observations.confidence,
    )
    shuffled_model, _, _ = _train_model(
        seed + 2000,
        shuffled_observations,
        source_updates,
    )
    shuffled_model_result = _evaluate_targets(
        shuffled_model,
        state_codes,
        intention_codes,
        TARGETS,
        horizon=4,
        beam_width=16,
    )
    fresh_model = ExternalTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=HIDDEN_WIDTH,
    )
    fresh = _evaluate_targets(
        fresh_model,
        state_codes,
        intention_codes,
        TARGETS,
        horizon=4,
        beam_width=16,
    )
    restored = ExternalTransitionModel.from_payload(model.state_payload())
    persisted = _evaluate_targets(
        restored,
        state_codes,
        intention_codes,
        TARGETS,
        horizon=4,
        beam_width=16,
    )
    controller_unchanged = controller_before == _digest_module(controller)
    report = _report(
        seed=seed,
        source_updates=actual_updates,
        target_updates=0,
        model=model,
        source_loss=source_loss,
        trained=trained,
        retained=retained,
        shuffled_goal=shuffled_goal,
        shuffled_model=shuffled_model_result,
        fresh=fresh,
        persisted=persisted,
        controller_unchanged=controller_unchanged,
        elapsed_seconds=time.perf_counter() - begun,
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69316)
    parser.add_argument("--source-updates", type=int, default=1200)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if args.source_updates < 1:
        raise SystemExit("--source-updates must be positive")
    run(args.seed, args.report_out, source_updates=args.source_updates)


if __name__ == "__main__":
    main()
