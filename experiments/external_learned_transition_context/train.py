"""Pressure test learned opaque context formation for external model memory."""

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
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 16
MODEL_HIDDEN_WIDTH = 48
CONTEXT_HIDDEN_WIDTH = 48
POSITION_COUNT = 6
CONTEXT_UPDATES = 500
BASE_UPDATES = 1200
TARGET_UPDATES = 125
TARGET_LOSS_THRESHOLD = 0.01
BASE_DELTAS = (-1, 1)
AUXILIARY_DELTAS = (-3, 3)
TARGET_DELTAS = (-2, 2)
TARGETS = ((0, 4), (4, 0), (1, 5))
AUXILIARY_TARGETS = ((0, 3), (3, 0))


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(seed: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, ExternalTransitionObservation],
]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )

    def observations(
        deltas: tuple[int, int],
        *,
        noise: float = 0.0,
        noise_seed: int = 0,
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
        state = torch.stack(states)
        intention = torch.stack(intentions)
        next_state = torch.stack(next_states)
        if noise:
            noise_generator = torch.Generator().manual_seed(noise_seed)
            state = state + noise * torch.randn(
                state.shape,
                generator=noise_generator,
            )
            intention = intention + noise * torch.randn(
                intention.shape,
                generator=noise_generator,
            )
            next_state = next_state + noise * torch.randn(
                next_state.shape,
                generator=noise_generator,
            )
        return ExternalTransitionObservation(
            state=state,
            intention=intention,
            next_state=next_state,
            confidence=torch.ones(state.shape[0]),
        )

    return (
        state_codes,
        intention_codes,
        {
            "base": observations(BASE_DELTAS),
            "auxiliary": observations(AUXILIARY_DELTAS),
            "target": observations(TARGET_DELTAS),
        },
    )


def _noisy_observation(
    deltas: tuple[int, int],
    *,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    noise: float,
    seed: int,
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
    state = torch.stack(states)
    intention = torch.stack(intentions)
    next_state = torch.stack(next_states)
    generator = torch.Generator().manual_seed(seed)
    return ExternalTransitionObservation(
        state=state + noise * torch.randn(state.shape, generator=generator),
        intention=intention + noise * torch.randn(
            intention.shape,
            generator=generator,
        ),
        next_state=next_state + noise * torch.randn(
            next_state.shape,
            generator=generator,
        ),
        confidence=torch.ones(state.shape[0]),
    )


def _train_context_encoder(
    encoder: ExternalTransitionContextEncoder,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    *,
    seed: int,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.003)
    final_loss = float("inf")
    for update in range(1, CONTEXT_UPDATES + 1):
        left = torch.stack(
            [
                encoder(
                    view.state.unsqueeze(0),
                    view.intention.unsqueeze(0),
                    view.next_state.unsqueeze(0),
                    view.confidence.unsqueeze(0),
                )[0]
                for index, deltas in enumerate((BASE_DELTAS, AUXILIARY_DELTAS))
                for view in (
                    _noisy_observation(
                        deltas,
                        state_codes=state_codes,
                        intention_codes=intention_codes,
                        noise=0.01,
                        seed=seed + update * 11 + index,
                    ),
                )
            ]
        )
        right = torch.stack(
            [
                encoder(
                    view.state.unsqueeze(0),
                    view.intention.unsqueeze(0),
                    view.next_state.unsqueeze(0),
                    view.confidence.unsqueeze(0),
                )[0]
                for index, deltas in enumerate((BASE_DELTAS, AUXILIARY_DELTAS))
                for view in (
                    _noisy_observation(
                        deltas,
                        state_codes=state_codes,
                        intention_codes=intention_codes,
                        noise=0.02,
                        seed=seed + update * 17 + index,
                    ),
                )
            ]
        )
        final_loss = float(
            encoder.contrastive_loss(left, right, temperature=0.1).detach()
        )
        optimizer.zero_grad()
        encoder.contrastive_loss(left, right, temperature=0.1).backward()
        optimizer.step()
    return final_loss, CONTEXT_UPDATES


def _train_slot(
    bank: ExternalTransitionModelBank,
    index: int,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
    updates: int,
    *,
    learning_rate: float = 0.01,
    stop_at_threshold: bool = True,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(bank.models[index].parameters(), lr=learning_rate)
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    final_loss = float("inf")
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
    for start, goal in targets:
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[goal].unsqueeze(0),
            intention_codes,
            horizon=horizon,
            transition_context=context.unsqueeze(0),
        )
        final = _execute_plan(result.intentions[0], intention_codes, start, deltas)
        predicted_final.append(final)
        successes.append(final == goal)
    return {
        "successes": successes,
        "mastery": sum(successes) / len(successes),
        "predicted_final_positions": predicted_final,
    }


def _new_bank() -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=MODEL_HIDDEN_WIDTH,
    )


def _nearest_prior(
    contexts: tuple[torch.Tensor, ...],
    target: torch.Tensor,
) -> int:
    scores = torch.stack([context @ target for context in contexts])
    return int(scores.argmax())


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    state_codes, intention_codes, observations = _fixture(seed)
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=CONTEXT_HIDDEN_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    encoder_loss, encoder_updates = _train_context_encoder(
        encoder,
        state_codes,
        intention_codes,
        seed=seed,
    )
    encoder.eval()
    with torch.no_grad():
        contexts = tuple(
            encoder.encode_observation(observations[name])
            for name in ("base", "auxiliary", "target")
        )
        noisy_contexts = tuple(
            encoder(
                view.state.unsqueeze(0),
                view.intention.unsqueeze(0),
                view.next_state.unsqueeze(0),
                view.confidence.unsqueeze(0),
            )[0]
            for index, (name, deltas) in enumerate(
                (
                    ("base", BASE_DELTAS),
                    ("auxiliary", AUXILIARY_DELTAS),
                    ("target", TARGET_DELTAS),
                )
            )
            for view in (
                _noisy_observation(
                    deltas,
                    state_codes=state_codes,
                    intention_codes=intention_codes,
                    noise=0.02,
                    seed=seed + 900 + index,
                ),
            )
        )
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
    base_index = bank.ensure_context(contexts[0])
    auxiliary_index = bank.ensure_context(contexts[1], initialize_from=base_index)
    base_loss, base_updates = _train_slot(
        bank,
        base_index,
        observations["base"],
        contexts[0],
        BASE_UPDATES,
        stop_at_threshold=False,
    )
    auxiliary_loss, auxiliary_updates = _train_slot(
        bank,
        auxiliary_index,
        observations["auxiliary"],
        contexts[1],
        BASE_UPDATES,
        stop_at_threshold=False,
    )
    base_digest = bank.models[base_index].digest()
    auxiliary_digest = bank.models[auxiliary_index].digest()
    target_prior = _nearest_prior(contexts[:2], contexts[2])
    target_index = bank.ensure_context(
        contexts[2],
        initialize_from=target_prior,
    )
    target_before = _evaluate(
        bank,
        state_codes,
        intention_codes,
        contexts[2],
        TARGET_DELTAS,
        horizon=2,
    )
    target_loss, target_updates = _train_slot(
        bank,
        target_index,
        observations["target"],
        contexts[2],
        TARGET_UPDATES,
    )
    target_after = _evaluate(
        bank,
        state_codes,
        intention_codes,
        contexts[2],
        TARGET_DELTAS,
        horizon=2,
    )
    base_after = _evaluate(
        bank,
        state_codes,
        intention_codes,
        contexts[0],
        BASE_DELTAS,
        horizon=4,
    )
    auxiliary_after = _evaluate(
        bank,
        state_codes,
        intention_codes,
        contexts[1],
        AUXILIARY_DELTAS,
        horizon=4,
        targets=AUXILIARY_TARGETS,
    )
    wrong_context = _evaluate(
        bank,
        state_codes,
        intention_codes,
        contexts[0],
        TARGET_DELTAS,
        horizon=2,
    )

    fresh = _new_bank()
    fresh_index = fresh.ensure_context(contexts[2])
    fresh_loss, fresh_updates = _train_slot(
        fresh,
        fresh_index,
        observations["target"],
        contexts[2],
        TARGET_UPDATES,
    )
    fresh_result = _evaluate(
        fresh,
        state_codes,
        intention_codes,
        contexts[2],
        TARGET_DELTAS,
        horizon=2,
    )
    corrupted = _new_bank()
    corrupted_index = corrupted.ensure_context(contexts[2])
    corrupted_observation = ExternalTransitionObservation(
        state=observations["target"].state,
        intention=observations["target"].intention,
        next_state=observations["target"].next_state.roll(1, dims=0),
        confidence=observations["target"].confidence,
    )
    corrupted_loss, corrupted_updates = _train_slot(
        corrupted,
        corrupted_index,
        corrupted_observation,
        contexts[2],
        TARGET_UPDATES,
    )
    corrupted_result = _evaluate(
        corrupted,
        state_codes,
        intention_codes,
        contexts[2],
        TARGET_DELTAS,
        horizon=2,
    )
    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    persisted = _evaluate(
        restored,
        state_codes,
        intention_codes,
        contexts[2],
        TARGET_DELTAS,
        horizon=2,
    )
    pairwise_cosines = torch.stack(contexts) @ torch.stack(contexts).T
    stability = torch.stack(
        [context @ noisy for context, noisy in zip(contexts, noisy_contexts, strict=True)]
    )
    gates = {
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_learns": encoder_loss < 0.05,
        "context_views_stable": bool(torch.all(stability > 0.95)),
        "context_regimes_separable": bool(
            torch.min(pairwise_cosines[~torch.eye(3, dtype=torch.bool)]) < 0.9
        ),
        "context_is_automatically_grown": target_index == 2,
        "base_model_learns": base_loss < TARGET_LOSS_THRESHOLD,
        "auxiliary_model_learns": auxiliary_loss < TARGET_LOSS_THRESHOLD,
        "target_model_adapts": float(target_after["mastery"]) >= 0.8,
        "target_learns_faster_than_fresh": target_updates < fresh_updates,
        "base_retained": float(base_after["mastery"]) >= 0.8,
        "auxiliary_retained": float(auxiliary_after["mastery"]) >= 0.8,
        "base_slot_byte_stable": base_digest == bank.models[base_index].digest(),
        "auxiliary_slot_byte_stable": auxiliary_digest == bank.models[auxiliary_index].digest(),
        "wrong_context_control": float(wrong_context["mastery"]) < 0.8,
        "corruption_control": float(corrupted_result["mastery"]) < 0.8,
        "fresh_control_is_matched": fresh_result["mastery"] < 0.8
        or target_after["mastery"] >= fresh_result["mastery"],
        "persistence_exact": persisted["successes"] == target_after["successes"],
    }
    report = {
        "schema": "neural-computer.external-learned-transition-context-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "model_hidden_width": MODEL_HIDDEN_WIDTH,
            "context_hidden_width": CONTEXT_HIDDEN_WIDTH,
            "base_deltas": list(BASE_DELTAS),
            "auxiliary_deltas": list(AUXILIARY_DELTAS),
            "target_deltas": list(TARGET_DELTAS),
            "context_updates": CONTEXT_UPDATES,
            "base_updates": BASE_UPDATES,
            "target_updates": TARGET_UPDATES,
            "regime_labels_used_by_learner": False,
            "policy": "none_external_learned_context_model_bank_search_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "context_encoder": {
            "optimizer_updates": encoder_updates,
            "loss": encoder_loss,
            "pairwise_cosines": pairwise_cosines.tolist(),
            "same_bundle_clean_noisy_cosines": stability.tolist(),
            "replayed_examples": 0,
        },
        "base": {
            "optimizer_updates": base_updates,
            "loss": base_loss,
            "retention": base_after,
            "replayed_examples": POSITION_COUNT * 2 * (base_updates - 1),
        },
        "auxiliary": {
            "optimizer_updates": auxiliary_updates,
            "loss": auxiliary_loss,
            "retention": auxiliary_after,
            "replayed_examples": POSITION_COUNT * 2 * (auxiliary_updates - 1),
        },
        "target": {
            "optimizer_updates": target_updates,
            "loss": target_loss,
            "prior_slot": target_prior,
            "before_adaptation": target_before,
            "after_adaptation": target_after,
            "replayed_old_examples": 0,
            "replayed_current_stream_examples": POSITION_COUNT * 2 * (target_updates - 1),
        },
        "fresh_target": {
            "optimizer_updates": fresh_updates,
            "loss": fresh_loss,
            "result": fresh_result,
        },
        "wrong_context": wrong_context,
        "corrupted_target": {
            "optimizer_updates": corrupted_updates,
            "loss": corrupted_loss,
            "result": corrupted_result,
        },
        "persisted_target": persisted,
        "accounting": {
            "context_slots": bank.context_count,
            "controller_parameter_updates": 0,
            "context_encoder_parameter_updates": encoder_updates,
            "old_base_replay_during_target": 0,
            "target_current_stream_replay": POSITION_COUNT * 2 * (target_updates - 1),
            "target_updates": target_updates,
            "fresh_target_updates": fresh_updates,
        },
        "digests": {
            "context_encoder": encoder.digest(),
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
    parser.add_argument("--seed", type=int, default=69911)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
