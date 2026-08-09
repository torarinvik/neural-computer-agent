"""Pressure test policy-free continual acquisition on disjoint dynamics."""

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
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 6
HIDDEN_WIDTH = 48
POSITION_COUNT = 6
SOURCE_UPDATES = 1200
TARGET_UPDATES = 125
TARGET_LOSS_THRESHOLD = 0.01
SOURCE_DELTAS = (-1, 1)
TARGET_DELTAS = (-2, 2)
TARGETS = ((0, 4), (4, 0), (1, 5))


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, ExternalTransitionObservation, ExternalTransitionObservation, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )
    contexts = F.normalize(
        torch.randn(2, CONTEXT_WIDTH, generator=generator), dim=-1
    )

    def observations(
        deltas: tuple[int, int],
    ) -> ExternalTransitionObservation:
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
        return ExternalTransitionObservation(
            state=torch.stack(states),
            intention=torch.stack(intentions),
            next_state=torch.stack(next_states),
            confidence=torch.ones(POSITION_COUNT * 2),
        )

    return (
        state_codes,
        intention_codes,
        observations(SOURCE_DELTAS),
        observations(TARGET_DELTAS),
        contexts,
    )


def _train_slot(
    bank: ExternalTransitionModelBank,
    index: int,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
    updates: int,
    learning_rate: float,
    stop_at_threshold: bool = True,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(
        bank.models[index].parameters(),
        lr=learning_rate,
    )
    final_loss = float("inf")
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    for update in range(1, updates + 1):
        final_loss = bank.adaptation_step(observation, context_batch, optimizer)
        if stop_at_threshold and final_loss <= TARGET_LOSS_THRESHOLD:
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
        position = min(
            POSITION_COUNT - 1,
            max(0, position + deltas[action]),
        )
    return position


def _evaluate(
    bank: ExternalTransitionModelBank,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    context: torch.Tensor,
    deltas: tuple[int, int],
    *,
    horizon: int,
    targets: tuple[tuple[int, int], ...] = TARGETS,
) -> dict[str, object]:
    planner = ExternalModelBasedPlanner(bank, beam_width=16)
    successes: list[bool] = []
    predicted_final: list[int] = []
    latencies: list[float] = []
    for start, goal in targets:
        begun = time.perf_counter()
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[goal].unsqueeze(0),
            intention_codes,
            horizon=horizon,
            transition_context=context.unsqueeze(0),
        )
        latencies.append(time.perf_counter() - begun)
        final = _execute_plan(
            result.intentions[0],
            intention_codes,
            start,
            deltas,
        )
        predicted_final.append(final)
        successes.append(final == goal)
    return {
        "successes": successes,
        "mastery": sum(successes) / len(successes),
        "predicted_final_positions": predicted_final,
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
    state_codes, intention_codes, source, target, contexts = _fixture(seed)
    source_context, target_context = contexts

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
    source_index = bank.ensure_context(source_context)
    source_loss, source_updates_actual = _train_slot(
        bank,
        source_index,
        source,
        source_context,
        SOURCE_UPDATES,
        0.01,
        stop_at_threshold=False,
    )
    source_model_digest = bank.models[source_index].digest()
    source_before = _evaluate(
        bank,
        state_codes,
        intention_codes,
        source_context,
        SOURCE_DELTAS,
        horizon=4,
    )

    target_index = bank.ensure_context(
        target_context,
        initialize_from=source_index,
    )
    target_before = _evaluate(
        bank,
        state_codes,
        intention_codes,
        target_context,
        TARGET_DELTAS,
        horizon=2,
    )
    target_loss, target_updates_actual = _train_slot(
        bank,
        target_index,
        target,
        target_context,
        TARGET_UPDATES,
        0.01,
    )
    target_after = _evaluate(
        bank,
        state_codes,
        intention_codes,
        target_context,
        TARGET_DELTAS,
        horizon=2,
    )
    source_after = _evaluate(
        bank,
        state_codes,
        intention_codes,
        source_context,
        SOURCE_DELTAS,
        horizon=4,
    )
    wrong_context = _evaluate(
        bank,
        state_codes,
        intention_codes,
        source_context,
        TARGET_DELTAS,
        horizon=2,
    )

    fresh = _new_bank()
    fresh_index = fresh.ensure_context(target_context)
    fresh_loss, fresh_updates_actual = _train_slot(
        fresh,
        fresh_index,
        target,
        target_context,
        TARGET_UPDATES,
        0.01,
    )
    fresh_result = _evaluate(
        fresh,
        state_codes,
        intention_codes,
        target_context,
        TARGET_DELTAS,
        horizon=2,
    )

    corrupted = _new_bank()
    corrupted_index = corrupted.ensure_context(target_context)
    corrupted_target = ExternalTransitionObservation(
        state=target.state,
        intention=target.intention,
        next_state=target.next_state.roll(1, dims=0),
        confidence=target.confidence,
    )
    corrupted_loss, corrupted_updates_actual = _train_slot(
        corrupted,
        corrupted_index,
        corrupted_target,
        target_context,
        TARGET_UPDATES,
        0.01,
    )
    corrupted_result = _evaluate(
        corrupted,
        state_codes,
        intention_codes,
        target_context,
        TARGET_DELTAS,
        horizon=2,
    )

    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    persisted = _evaluate(
        restored,
        state_codes,
        intention_codes,
        target_context,
        TARGET_DELTAS,
        horizon=2,
    )
    gates = {
        "controller_unchanged": controller_digest == _digest_module(controller),
        "source_model_learns": source_loss < 0.01,
        "target_model_adapts": float(target_after["mastery"]) >= 0.8,
        "target_learns_faster_than_fresh": target_updates_actual < fresh_updates_actual,
        "source_retained": float(source_after["mastery"]) >= 0.8,
        "source_slot_byte_stable": (
            source_model_digest == bank.models[source_index].digest()
        ),
        "wrong_context_control": float(wrong_context["mastery"]) < 0.8,
        "corruption_control": float(corrupted_result["mastery"]) < 0.8,
        "fresh_control_is_matched": fresh_result["mastery"] < 0.8
        or target_after["mastery"] >= fresh_result["mastery"],
        "persistence_exact": persisted["successes"] == target_after["successes"],
    }
    report = {
        "schema": "neural-computer.external-transition-model-bank-continual-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "source_deltas": list(SOURCE_DELTAS),
            "target_deltas": list(TARGET_DELTAS),
            "targets": [list(pair) for pair in TARGETS],
            "source_updates": SOURCE_UPDATES,
            "target_updates": TARGET_UPDATES,
            "target_loss_threshold": TARGET_LOSS_THRESHOLD,
            "policy": "none_external_contextual_transition_model_bank_search_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "source": {
            "optimizer_updates": source_updates_actual,
            "logical_transition_lifetimes": POSITION_COUNT * 2,
            "replayed_examples": POSITION_COUNT * 2 * (source_updates_actual - 1),
            "loss": source_loss,
            "before_target": source_before,
            "after_target": source_after,
        },
        "target": {
            "optimizer_updates": target_updates_actual,
            "logical_transition_lifetimes": POSITION_COUNT * 2,
            "unique_verifier_bits": POSITION_COUNT * 2,
            "replayed_current_target_examples": POSITION_COUNT * 2 * (target_updates_actual - 1),
            "replayed_old_source_examples": 0,
            "loss": target_loss,
            "before_adaptation": target_before,
            "after_adaptation": target_after,
        },
        "fresh_target": {
            "optimizer_updates": fresh_updates_actual,
            "loss": fresh_loss,
            "replayed_examples": POSITION_COUNT * 2 * (fresh_updates_actual - 1),
            "result": fresh_result,
        },
        "wrong_context": wrong_context,
        "corrupted_target": corrupted_result,
        "corrupted_training": {
            "optimizer_updates": corrupted_updates_actual,
            "loss": corrupted_loss,
        },
        "persisted_target": persisted,
        "accounting": {
            "context_slots": bank.context_count,
            "controller_parameter_updates": 0,
            "old_source_replay_during_target": 0,
            "target_current_stream_replay": POSITION_COUNT * 2 * (target_updates_actual - 1),
            "target_updates": target_updates_actual,
            "fresh_target_updates": fresh_updates_actual,
        },
        "digests": {
            "controller": controller_digest,
            "source_model": source_model_digest,
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
    parser.add_argument("--seed", type=int, default=69811)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
