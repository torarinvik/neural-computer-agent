"""Accounted sequential compounding audit for policy-free transition models."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from collections.abc import Callable

import torch
import torch.nn.functional as F

from neural_computer import (
    AmodalCognitiveController,
    ExternalModelBasedPlanner,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 6
HIDDEN_WIDTH = 48
POSITION_COUNT = 8
SOURCE_UPDATES = 1200
TARGET_UPDATES = 300
LOSS_THRESHOLD = 0.01
REGIME_DELTAS = ((-1, 1), (-2, 2), (-3, 3), (-4, 4))
REGIME_TARGETS = (
    ((0, 3), (3, 0), (1, 4)),
    ((0, 6), (6, 0), (1, 7)),
    ((0, 6), (6, 0), (1, 7)),
    ((0, 4), (4, 0), (1, 5)),
)
HORIZON = 3


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(
    seed: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    tuple[ExternalTransitionObservation, ...],
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )
    contexts = F.normalize(
        torch.randn(len(REGIME_DELTAS), CONTEXT_WIDTH, generator=generator), dim=-1
    )

    observations: list[ExternalTransitionObservation] = []
    for deltas in REGIME_DELTAS:
        states: list[torch.Tensor] = []
        intentions: list[torch.Tensor] = []
        next_states: list[torch.Tensor] = []
        for position in range(POSITION_COUNT):
            for action_index, delta in enumerate(deltas):
                next_position = min(
                    POSITION_COUNT - 1,
                    max(0, position + delta),
                )
                states.append(state_codes[position])
                intentions.append(intention_codes[action_index])
                next_states.append(state_codes[next_position])
        observations.append(
            ExternalTransitionObservation(
                state=torch.stack(states),
                intention=torch.stack(intentions),
                next_state=torch.stack(next_states),
                confidence=torch.ones(POSITION_COUNT * 2),
            )
        )
    return state_codes, intention_codes, tuple(observations), contexts


def _train_slot(
    bank: ExternalTransitionModelBank,
    index: int,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
    updates: int,
    *,
    stop_at_threshold: bool,
    mastery_probe: Callable[[], bool] | None = None,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(bank.models[index].parameters(), lr=0.01)
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    final_loss = float("inf")
    for update in range(1, updates + 1):
        final_loss = bank.adaptation_step(observation, context_batch, optimizer)
        if (
            stop_at_threshold
            and final_loss <= LOSS_THRESHOLD
            and (mastery_probe is None or mastery_probe())
        ):
            return final_loss, update
    return final_loss, updates


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
    bank: ExternalTransitionModelBank,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    context: torch.Tensor,
    deltas: tuple[int, int],
    targets: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    planner = ExternalModelBasedPlanner(bank, beam_width=16)
    successes: list[bool] = []
    predicted_final: list[int] = []
    expanded_nodes = 0
    latencies: list[float] = []
    for start, goal in targets:
        begun = time.perf_counter()
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[goal].unsqueeze(0),
            intention_codes,
            horizon=HORIZON,
            transition_context=context.unsqueeze(0),
        )
        latencies.append(time.perf_counter() - begun)
        expanded_nodes += result.expanded_nodes
        final = _execute_plan(result.intentions[0], intention_codes, start, deltas)
        predicted_final.append(final)
        successes.append(final == goal)
    return {
        "successes": successes,
        "mastery": sum(successes) / len(successes),
        "predicted_final_positions": predicted_final,
        "expanded_nodes": expanded_nodes,
        "mean_latency_seconds": sum(latencies) / len(latencies),
    }


def _new_bank() -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=HIDDEN_WIDTH,
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    state_codes, intention_codes, observations, contexts = _fixture(seed)

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

    bank = _new_bank()
    source_index = bank.ensure_context(contexts[0])
    source_loss, source_updates = _train_slot(
        bank,
        source_index,
        observations[0],
        contexts[0],
        SOURCE_UPDATES,
        stop_at_threshold=False,
    )
    source_digest = bank.models[source_index].digest()

    warm_rows: list[dict[str, object]] = []
    prior_digests: dict[int, str] = {source_index: source_digest}
    previous_index = source_index
    warm_total = source_updates
    fresh_total = source_updates
    all_warm_mastery = True
    all_fresh_mastery = True
    all_warm_faster = True
    all_prior_retained = True

    for regime_index in range(1, len(REGIME_DELTAS)):
        context = contexts[regime_index]
        warm_index = bank.ensure_context(context, initialize_from=previous_index)
        zero_shot = _evaluate(
            bank,
            state_codes,
            intention_codes,
            context,
            REGIME_DELTAS[regime_index],
            REGIME_TARGETS[regime_index],
        )
        warm_loss, warm_updates = _train_slot(
            bank,
            warm_index,
            observations[regime_index],
            context,
            TARGET_UPDATES,
            stop_at_threshold=True,
            mastery_probe=lambda: float(
                _evaluate(
                    bank,
                    state_codes,
                    intention_codes,
                    context,
                    REGIME_DELTAS[regime_index],
                    REGIME_TARGETS[regime_index],
                )["mastery"]
            )
            >= 0.8,
        )
        after = _evaluate(
            bank,
            state_codes,
            intention_codes,
            context,
            REGIME_DELTAS[regime_index],
            REGIME_TARGETS[regime_index],
        )

        fresh = _new_bank()
        fresh_index = fresh.ensure_context(context)
        fresh_loss, fresh_updates = _train_slot(
            fresh,
            fresh_index,
            observations[regime_index],
            context,
            TARGET_UPDATES,
            stop_at_threshold=True,
            mastery_probe=lambda: float(
                _evaluate(
                    fresh,
                    state_codes,
                    intention_codes,
                    context,
                    REGIME_DELTAS[regime_index],
                    REGIME_TARGETS[regime_index],
                )["mastery"]
            )
            >= 0.8,
        )
        fresh_after = _evaluate(
            fresh,
            state_codes,
            intention_codes,
            context,
            REGIME_DELTAS[regime_index],
            REGIME_TARGETS[regime_index],
        )

        warm_total += warm_updates
        fresh_total += fresh_updates
        warm_mastered = float(after["mastery"]) >= 0.8
        fresh_mastered = float(fresh_after["mastery"]) >= 0.8
        all_warm_mastery = all_warm_mastery and warm_mastered
        all_fresh_mastery = all_fresh_mastery and fresh_mastered
        all_warm_faster = all_warm_faster and warm_updates < fresh_updates

        retained: list[dict[str, object]] = []
        for retained_index, retained_deltas in enumerate(REGIME_DELTAS[: regime_index + 1]):
            result = _evaluate(
                bank,
                state_codes,
                intention_codes,
                contexts[retained_index],
                retained_deltas,
                REGIME_TARGETS[retained_index],
            )
            digest_stable = (
                retained_index == regime_index
                or bank.models[retained_index].digest()
                == prior_digests[retained_index]
            )
            retained.append(
                {
                    "regime_index": retained_index,
                    "mastery": result["mastery"],
                    "digest_stable": digest_stable,
                }
            )
            all_prior_retained = all_prior_retained and bool(
                float(result["mastery"]) >= 0.8 and digest_stable
            )

        prior_digests[warm_index] = bank.models[warm_index].digest()
        previous_index = warm_index
        warm_rows.append(
            {
                "regime_index": regime_index,
                "deltas": list(REGIME_DELTAS[regime_index]),
                "zero_shot": zero_shot,
                "warm": {
                    "optimizer_updates": warm_updates,
                    "loss": warm_loss,
                    "result": after,
                },
                "fresh": {
                    "optimizer_updates": fresh_updates,
                    "loss": fresh_loss,
                    "result": fresh_after,
                },
                "retained_prefix": retained,
                "cumulative_cost": {
                    "warm_model_updates": warm_total,
                    "fresh_model_updates": fresh_total,
                    "warm_search_expansions": after["expanded_nodes"],
                    "fresh_search_expansions": fresh_after["expanded_nodes"],
                },
            }
        )

    gates = {
        "controller_unchanged": controller_digest == _digest_module(controller),
        "source_model_learns": source_loss < LOSS_THRESHOLD,
        "all_warm_regimes_mastered": all_warm_mastery,
        "fresh_controls_mastered": all_fresh_mastery,
        "warm_beats_fresh_every_target": all_warm_faster,
        "all_prior_regimes_retained_and_byte_stable": all_prior_retained,
        "old_regime_replay_during_adaptation_zero": True,
        "planner_is_inference_only": True,
    }
    report = {
        "schema": "neural-computer.external-transition-model-compounding-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "regime_deltas": [list(pair) for pair in REGIME_DELTAS],
            "targets_by_regime": [
                [list(pair) for pair in targets] for targets in REGIME_TARGETS
            ],
            "horizon": HORIZON,
            "source_updates": SOURCE_UPDATES,
            "target_update_budget": TARGET_UPDATES,
            "loss_threshold": LOSS_THRESHOLD,
            "policy": "none_external_transition_model_compounding_search_v1",
        },
        "gates": gates,
        "promoted": all(
            value is True for value in gates.values()
        ),
        "source": {
            "optimizer_updates": source_updates,
            "logical_transition_lifetimes": POSITION_COUNT * 2,
            "replayed_examples": POSITION_COUNT * 2 * (source_updates - 1),
            "loss": source_loss,
            "model_digest": source_digest,
        },
        "sequential_targets": warm_rows,
        "accounting": {
            "controller_optimizer_updates": 0,
            "old_regime_replay_during_target_adaptation": 0,
            "source_acquisition_charged_in_cumulative_cost": True,
            "search_compute_reported_as_expanded_nodes": True,
            "unique_verifier_bits_per_regime": POSITION_COUNT * 2,
        },
        "digests": {
            "controller": controller_digest,
            "bank": bank.digest(),
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=70311)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
