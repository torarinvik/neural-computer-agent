"""Pressure test append-only factual memory across disjoint dynamics.

The production-facing components see only opaque state, intention, context,
and learned verifier tensors.  The position simulator exists solely to score
the experiment and never enters the controller, memory, or planner.
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
    ExternalGoalEvaluator,
    ExternalModelBasedPlanner,
    ExternalTransitionMemory,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 6
HIDDEN_WIDTH = 48
POSITION_COUNT = 6
TARGETS = ((0, 4), (4, 0), (1, 5))
SOURCE_DELTAS = (-1, 1)
TARGET_DELTAS = (-2, 2)


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
    ExternalTransitionObservation,
    ExternalTransitionObservation,
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
    context_codes = F.normalize(
        torch.randn(2, CONTEXT_WIDTH, generator=generator), dim=-1
    )

    def observations(deltas: tuple[int, int]) -> ExternalTransitionObservation:
        states: list[torch.Tensor] = []
        intentions: list[torch.Tensor] = []
        next_states: list[torch.Tensor] = []
        for position in range(POSITION_COUNT):
            for action_index, delta in enumerate(deltas):
                next_position = min(
                    POSITION_COUNT - 1, max(0, position + delta)
                )
                states.append(state_codes[position])
                intentions.append(intention_codes[action_index])
                next_states.append(state_codes[next_position])
        return ExternalTransitionObservation(
            state=torch.stack(states),
            intention=torch.stack(intentions),
            next_state=torch.stack(next_states),
            confidence=torch.ones(POSITION_COUNT * 2),
        )

    return (
        state_codes,
        intention_codes,
        context_codes[0],
        context_codes[1],
        observations(SOURCE_DELTAS),
        observations(TARGET_DELTAS),
        context_codes,
        torch.arange(POSITION_COUNT),
    )


def _train_goal_evaluator(
    seed: int, state_codes: torch.Tensor, updates: int = 1200
) -> tuple[ExternalGoalEvaluator, float, int]:
    torch.manual_seed(seed)
    evaluator = ExternalGoalEvaluator(STATE_WIDTH, hidden_width=HIDDEN_WIDTH)
    state = state_codes.repeat_interleave(POSITION_COUNT, dim=0)
    goals = state_codes.repeat(POSITION_COUNT, 1)
    outcome = torch.tensor(
        [float(left == right) for left in range(POSITION_COUNT) for right in range(POSITION_COUNT)]
    )
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=0.01)
    final_loss = float("inf")
    for _update in range(1, updates + 1):
        optimizer.zero_grad()
        loss = evaluator.loss(state, goals, outcome)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return evaluator, final_loss, updates


def _execute_plan(
    intentions: torch.Tensor,
    intention_codes: torch.Tensor,
    start: int,
    deltas: tuple[int, int],
) -> int:
    position = start
    for intention in intentions:
        action = int(
            torch.linalg.vector_norm(intention_codes - intention, dim=-1).argmin()
        )
        position = min(POSITION_COUNT - 1, max(0, position + deltas[action]))
    return position


def _evaluate(
    memory: ExternalTransitionMemory,
    evaluator: ExternalGoalEvaluator,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    context: torch.Tensor,
    targets: tuple[tuple[int, int], ...],
    deltas: tuple[int, int],
    *,
    horizon: int,
    planning_goals: tuple[int, ...] | None = None,
) -> dict[str, object]:
    requested_goals = (
        tuple(goal for _start, goal in targets)
        if planning_goals is None
        else planning_goals
    )
    planner = ExternalModelBasedPlanner(
        memory,
        beam_width=16,
        goal_evaluator=evaluator,
    )
    successes: list[bool] = []
    predicted_final: list[int] = []
    latencies: list[float] = []
    hit_rates: list[float] = []
    for (start, goal), requested_goal in zip(targets, requested_goals, strict=True):
        begun = time.perf_counter()
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[requested_goal].unsqueeze(0),
            intention_codes,
            horizon=horizon,
            transition_context=context.unsqueeze(0),
        )
        latencies.append(time.perf_counter() - begun)
        final = _execute_plan(result.intentions[0], intention_codes, start, deltas)
        predicted_final.append(final)
        successes.append(final == goal)
        hit_rates.append(float((result.predicted_states[0].abs().sum(dim=-1) > 0).float().mean()))
    return {
        "successes": successes,
        "mastery": sum(successes) / len(successes),
        "predicted_final_positions": predicted_final,
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "transition_hit_rate": sum(hit_rates) / len(hit_rates),
    }


def _fresh_memory() -> ExternalTransitionMemory:
    return ExternalTransitionMemory(
        STATE_WIDTH,
        INTENTION_WIDTH,
        context_width=CONTEXT_WIDTH,
    )


def run(seed: int, report_out: Path, *, evaluator_updates: int) -> dict[str, object]:
    begun = time.perf_counter()
    (
        state_codes,
        intention_codes,
        source_context,
        target_context,
        source_observations,
        target_observations,
        _context_codes,
        _positions,
    ) = _fixture(seed)
    source_context_rows = source_context.unsqueeze(0).expand(POSITION_COUNT * 2, -1)
    target_context_rows = target_context.unsqueeze(0).expand(POSITION_COUNT * 2, -1)

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

    evaluator, evaluator_loss, evaluator_updates_actual = _train_goal_evaluator(
        seed + 1000, state_codes, evaluator_updates
    )
    memory = _fresh_memory()
    source_receipt = memory.write(source_observations, context=source_context_rows)
    source_before = _evaluate(
        memory,
        evaluator,
        state_codes,
        intention_codes,
        source_context,
        TARGETS,
        SOURCE_DELTAS,
        horizon=4,
    )
    source_digest_before_target = memory.digest()

    # The target phase appends only new target observations. No old source rows
    # are replayed and no optimizer updates touch the controller or evaluator.
    target_receipt = memory.write(target_observations, context=target_context_rows)
    trained = _evaluate(
        memory,
        evaluator,
        state_codes,
        intention_codes,
        target_context,
        TARGETS,
        TARGET_DELTAS,
        horizon=2,
    )
    retained = _evaluate(
        memory,
        evaluator,
        state_codes,
        intention_codes,
        source_context,
        TARGETS,
        SOURCE_DELTAS,
        horizon=4,
    )
    shuffled_goal = _evaluate(
        memory,
        evaluator,
        state_codes,
        intention_codes,
        target_context,
        TARGETS,
        TARGET_DELTAS,
        horizon=2,
        planning_goals=tuple((goal + 1) % POSITION_COUNT for _start, goal in TARGETS),
    )
    shuffled_context = _evaluate(
        memory,
        evaluator,
        state_codes,
        intention_codes,
        source_context,
        TARGETS,
        TARGET_DELTAS,
        horizon=2,
    )

    corrupted = _fresh_memory()
    corrupted.write(
        ExternalTransitionObservation(
            state=target_observations.state,
            intention=target_observations.intention,
            next_state=target_observations.next_state.roll(1, 0),
            confidence=target_observations.confidence,
        ),
        context=target_context_rows,
    )
    corrupted_result = _evaluate(
        corrupted,
        evaluator,
        state_codes,
        intention_codes,
        target_context,
        TARGETS,
        TARGET_DELTAS,
        horizon=2,
    )
    fresh_result = _evaluate(
        _fresh_memory(),
        evaluator,
        state_codes,
        intention_codes,
        target_context,
        TARGETS,
        TARGET_DELTAS,
        horizon=2,
    )

    restored = _fresh_memory()
    restored.store.load_state_dict(memory.store.state_dict())
    persisted = _evaluate(
        restored,
        evaluator,
        state_codes,
        intention_codes,
        target_context,
        TARGETS,
        TARGET_DELTAS,
        horizon=2,
    )
    controller_unchanged = controller_before == _digest_module(controller)
    gates = {
        "controller_unchanged": controller_unchanged,
        "evaluator_learns_verifier": evaluator_loss < 0.01,
        "source_memory_commits": bool(source_receipt.committed.all()),
        "target_memory_appends": bool(target_receipt.committed.all()),
        "target_mastery": float(trained["mastery"]) >= 0.8,
        "source_retention_after_target_append": float(retained["mastery"]) >= 0.8,
        "goal_conditioning_causal": float(shuffled_goal["mastery"]) <= 0.5,
        "context_conditioning_causal": float(shuffled_context["mastery"]) <= 0.5,
        "corruption_control_not_mastered": float(corrupted_result["mastery"]) < 0.8,
        "fresh_control_not_mastered": float(fresh_result["mastery"]) < 0.8,
        "persistence_exact": persisted["successes"] == trained["successes"],
    }
    report = {
        "schema": "neural-computer.external-transition-memory-transfer-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "position_count": POSITION_COUNT,
            "source_deltas": list(SOURCE_DELTAS),
            "target_deltas": list(TARGET_DELTAS),
            "targets": [list(pair) for pair in TARGETS],
            "policy": "none_append_only_transition_memory_plus_learned_verifier_search",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "source": {
            "unique_transition_lifetimes": POSITION_COUNT * 2,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "before_target_append": source_before,
            "memory_digest": source_digest_before_target,
        },
        "target": {
            "unique_transition_lifetimes": POSITION_COUNT * 2,
            "unique_verifier_bits": len(TARGETS),
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "memory_records_after_append": memory.record_count,
            "after_target_append": trained,
        },
        "retained": retained,
        "shuffled_goal": shuffled_goal,
        "shuffled_context": shuffled_context,
        "corrupted_memory": corrupted_result,
        "fresh_memory": fresh_result,
        "persisted_memory": persisted,
        "evaluator": {
            "optimizer_updates": evaluator_updates_actual,
            "replayed_examples": 0,
            "final_loss": evaluator_loss,
            "digest": evaluator.digest(),
        },
        "controller_digest": controller_before,
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69401)
    parser.add_argument("--evaluator-updates", type=int, default=1200)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if args.evaluator_updates < 1:
        raise SystemExit("--evaluator-updates must be positive")
    run(args.seed, args.report_out, evaluator_updates=args.evaluator_updates)


if __name__ == "__main__":
    main()
