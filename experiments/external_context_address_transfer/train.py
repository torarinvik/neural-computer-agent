"""Pressure test learned regime admission without supplied context labels.

The resolver receives only opaque transition observations and verified
next-state tensors.  Regime names and integer positions exist only in this
fixture's verifier-side scoring code.
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
    ExternalContextAddressResolver,
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
REGIME_DELTAS = ((-1, 1), (-2, 2), (1, -1), (-1, 1))
TARGETS = ((0, 4), (4, 0), (1, 5))
HORIZONS = (4, 2, 4, 4)


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(seed: int) -> tuple[torch.Tensor, torch.Tensor, tuple[ExternalTransitionObservation, ...]]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )

    regimes: list[ExternalTransitionObservation] = []
    for deltas in REGIME_DELTAS:
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
        regimes.append(
            ExternalTransitionObservation(
                state=torch.stack(states),
                intention=torch.stack(intentions),
                next_state=torch.stack(next_states),
                confidence=torch.ones(POSITION_COUNT * 2),
            )
        )
    return state_codes, intention_codes, tuple(regimes)


def _train_goal_evaluator(
    seed: int, state_codes: torch.Tensor, updates: int
) -> tuple[ExternalGoalEvaluator, float]:
    torch.manual_seed(seed)
    evaluator = ExternalGoalEvaluator(STATE_WIDTH, hidden_width=HIDDEN_WIDTH)
    state = state_codes.repeat_interleave(POSITION_COUNT, dim=0)
    goals = state_codes.repeat(POSITION_COUNT, 1)
    outcome = torch.tensor(
        [
            float(left == right)
            for left in range(POSITION_COUNT)
            for right in range(POSITION_COUNT)
        ]
    )
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=0.01)
    final_loss = float("inf")
    for _update in range(updates):
        optimizer.zero_grad()
        loss = evaluator.loss(state, goals, outcome)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return evaluator, final_loss


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


def _evaluate_regime(
    memory: ExternalTransitionMemory,
    evaluator: ExternalGoalEvaluator,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    context: torch.Tensor,
    deltas: tuple[int, int],
    horizon: int,
) -> dict[str, object]:
    planner = ExternalModelBasedPlanner(
        memory,
        beam_width=16,
        goal_evaluator=evaluator,
    )
    successes: list[bool] = []
    predicted: list[int] = []
    latencies: list[float] = []
    for start, goal in TARGETS:
        begun = time.perf_counter()
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[goal].unsqueeze(0),
            intention_codes,
            horizon=horizon,
            transition_context=context.unsqueeze(0),
        )
        latencies.append(time.perf_counter() - begun)
        final = _execute_plan(result.intentions[0], intention_codes, start, deltas)
        predicted.append(final)
        successes.append(final == goal)
    return {
        "successes": successes,
        "mastery": sum(successes) / len(successes),
        "predicted_final_positions": predicted,
        "mean_latency_seconds": sum(latencies) / len(latencies),
    }


def _new_memory() -> ExternalTransitionMemory:
    return ExternalTransitionMemory(
        STATE_WIDTH,
        INTENTION_WIDTH,
        context_width=CONTEXT_WIDTH,
    )


def _expand_context(context: torch.Tensor) -> torch.Tensor:
    return context.unsqueeze(0).expand(POSITION_COUNT * 2, -1)


def _transition_diagnostic(
    memory: ExternalTransitionMemory,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> dict[str, float]:
    prediction, hits = memory.predict_with_hit(
        observation.state,
        observation.intention,
        context=_expand_context(context),
    )
    return {
        "hit_rate": float(hits.float().mean()),
        "next_state_mse": float(
            (prediction - observation.next_state).square().mean()
        ),
    }


def run(seed: int, report_out: Path, *, evaluator_updates: int) -> dict[str, object]:
    begun = time.perf_counter()
    state_codes, intention_codes, regimes = _fixture(seed)
    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_digest = _digest_module(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    evaluator, evaluator_loss = _train_goal_evaluator(
        seed + 1000, state_codes, evaluator_updates
    )
    memory = _new_memory()
    resolver = ExternalContextAddressResolver(
        CONTEXT_WIDTH,
        address_seed=seed + 2000,
    )
    admissions: list[dict[str, object]] = []
    contexts: list[torch.Tensor] = []
    for index, observation in enumerate(regimes):
        resolution = resolver.resolve(observation, memory)
        contexts.append(resolution.context)
        receipt = memory.write(
            observation,
            context=_expand_context(resolution.context),
        )
        admissions.append(
            {
                "regime_index": index,
                "reused": resolution.reused,
                "matched_observations": resolution.matched_observations,
                "mean_error": resolution.mean_error,
                "committed": bool(receipt.committed.all()),
                "record_count": memory.record_count,
            }
        )

    rediscovered = [
        resolver.resolve(observation, memory) for observation in regimes
    ]
    retained = [
        _evaluate_regime(
            memory,
            evaluator,
            state_codes,
            intention_codes,
            contexts[index],
            REGIME_DELTAS[index],
            HORIZONS[index],
        )
        for index in range(len(regimes))
    ]

    shuffled_context = [
        _evaluate_regime(
            memory,
            evaluator,
            state_codes,
            intention_codes,
            contexts[(index + 1) % len(contexts)],
            REGIME_DELTAS[index],
            HORIZONS[index],
        )
        for index in range(len(regimes) - 1)
    ]
    shuffled_context_diagnostics = [
        _transition_diagnostic(
            memory,
            regimes[index],
            contexts[(index + 1) % len(contexts)],
        )
        for index in range(len(regimes) - 1)
    ]

    corrupted = _new_memory()
    for index, observation in enumerate(regimes):
        corrupted.write(
            ExternalTransitionObservation(
                state=observation.state,
                intention=observation.intention,
                next_state=observation.next_state.roll(1, 0),
                confidence=observation.confidence,
            ),
            context=_expand_context(contexts[index]),
        )
    corrupted_results = [
        _evaluate_regime(
            corrupted,
            evaluator,
            state_codes,
            intention_codes,
            contexts[index],
            REGIME_DELTAS[index],
            HORIZONS[index],
        )
        for index in range(len(regimes) - 1)
    ]
    corrupted_diagnostics = [
        _transition_diagnostic(corrupted, regimes[index], contexts[index])
        for index in range(len(regimes) - 1)
    ]
    fresh_results = [
        _evaluate_regime(
            _new_memory(),
            evaluator,
            state_codes,
            intention_codes,
            contexts[index],
            REGIME_DELTAS[index],
            HORIZONS[index],
        )
        for index in range(len(regimes) - 1)
    ]
    fresh_memory = _new_memory()
    fresh_diagnostics = [
        _transition_diagnostic(fresh_memory, regimes[index], contexts[index])
        for index in range(len(regimes) - 1)
    ]

    restored_memory = _new_memory()
    restored_memory.store.load_state_dict(memory.store.state_dict())
    restored_resolver = ExternalContextAddressResolver.from_payload(resolver.payload())
    persisted = [
        _evaluate_regime(
            restored_memory,
            evaluator,
            state_codes,
            intention_codes,
            restored_resolver.resolve(regimes[index], restored_memory).context,
            REGIME_DELTAS[index],
            HORIZONS[index],
        )
        for index in range(len(regimes) - 1)
    ]
    controller_unchanged = controller_digest == _digest_module(controller)
    first_admissions = [item["reused"] for item in admissions]
    rediscovery_reused = [item.reused for item in rediscovered]
    gates = {
        "controller_unchanged": controller_unchanged,
        "goal_evaluator_learns": evaluator_loss < 0.01,
        "new_regimes_allocate": first_admissions[:3] == [False, False, False],
        "duplicate_regime_reuses": first_admissions[3] is True,
        "all_regimes_rediscover": rediscovery_reused == [True, True, True, True],
        "address_count_matches_unique_dynamics": resolver.context_count == 3,
        "all_regimes_mastered": all(
            float(result["mastery"]) >= 0.8 for result in retained[:3]
        ),
        "reversal_retention": float(retained[2]["mastery"]) >= 0.8,
        "shuffled_context_factual_control": min(
            diagnostic["next_state_mse"]
            for diagnostic in shuffled_context_diagnostics
        ) > 0.1,
        "corruption_control": max(
            float(result["mastery"]) for result in corrupted_results
        ) < 0.8,
        "corruption_factual_control": min(
            diagnostic["next_state_mse"] for diagnostic in corrupted_diagnostics
        ) > 0.1,
        "fresh_factual_control": max(
            diagnostic["hit_rate"] for diagnostic in fresh_diagnostics
        ) == 0.0,
        "persistence_exact": [
            result["successes"] for result in persisted
        ] == [result["successes"] for result in retained[:3]],
    }
    report = {
        "schema": "neural-computer.external-context-address-transfer-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "regime_count": len(regimes),
            "unique_dynamics_count": 3,
            "regime_deltas": [list(deltas) for deltas in REGIME_DELTAS],
            "targets": [list(pair) for pair in TARGETS],
            "policy": "learned_fact_consistency_address_admission_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "admissions": admissions,
        "rediscovery_reused": rediscovery_reused,
        "retained": retained,
        "shuffled_context": shuffled_context,
        "shuffled_context_diagnostics": shuffled_context_diagnostics,
        "corrupted_memory": corrupted_results,
        "corrupted_diagnostics": corrupted_diagnostics,
        "fresh_memory": fresh_results,
        "fresh_diagnostics": fresh_diagnostics,
        "persisted": persisted,
        "accounting": {
            "unique_transition_lifetimes": len(regimes) * POSITION_COUNT * 2,
            "unique_verifier_bits": len(regimes) * len(TARGETS),
            "target_optimizer_updates": 0,
            "evaluator_optimizer_updates": evaluator_updates,
            "replayed_examples": 0,
            "memory_records": memory.record_count,
            "learned_context_addresses": resolver.context_count,
        },
        "evaluator": {
            "final_loss": evaluator_loss,
            "digest": evaluator.digest(),
        },
        "controller_digest": controller_digest,
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69501)
    parser.add_argument("--evaluator-updates", type=int, default=1200)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if args.evaluator_updates < 1:
        raise SystemExit("--evaluator-updates must be positive")
    run(args.seed, args.report_out, evaluator_updates=args.evaluator_updates)


if __name__ == "__main__":
    main()
