"""Causal audit of model-disagreement probing versus random intentions.

Two opaque factual models agree on the current observation and differ only
in the consequence of one available intention.  The active probe must choose
that intention from model predictions alone, then route the observed
consequence to the hidden model.  A random-intention control establishes the
floor.  The controller is frozen and each acquisition row is consumed once.
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
    ExternalModelBasedPlanner,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 1
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 2
EPISODES = 256
REGIME_NAMES = ("regime_a", "regime_b")


def _intentions(candidate_count: int) -> torch.Tensor:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    return torch.arange(candidate_count, dtype=torch.float32).unsqueeze(-1)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _new_bank() -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=2,
    )


def _acquire_bank(intentions: torch.Tensor) -> ExternalTransitionModelBank:
    bank = _new_bank()
    contexts = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    state = torch.zeros(intentions.shape[0] * 2, STATE_WIDTH)
    intention = intentions.repeat(2, 1)
    for regime_index, context in enumerate(contexts):
        slot = bank.ensure_context(context)
        next_state = torch.zeros_like(intention)
        if regime_index == 1:
            next_state[intention == intentions[-1]] = 1.0
        observation = ExternalTransitionObservation(state, intention, next_state)
        bank.adaptation_step(
            observation,
            bank.context_at(slot).unsqueeze(0).expand(state.shape[0], -1),
            None,
        )
    return bank


def _route(
    predictions: torch.Tensor,
    observed_next_state: torch.Tensor,
    generator: torch.Generator,
) -> int:
    errors = (predictions - observed_next_state).square().mean(dim=-1)
    winners = torch.where(errors == errors.min())[0]
    if winners.numel() == 1:
        return int(winners.item())
    choice = int(torch.randint(winners.numel(), (1,), generator=generator).item())
    return int(winners[choice].item())


def run(
    seed: int,
    report_out: Path,
    *,
    episodes: int = EPISODES,
    candidate_count: int = 2,
    outcome_noise_std: float = 0.0,
) -> dict[str, object]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if outcome_noise_std < 0.0:
        raise ValueError("outcome_noise_std cannot be negative")
    begun = time.perf_counter()
    torch.manual_seed(seed)
    intentions = _intentions(candidate_count)
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

    bank = _acquire_bank(intentions)
    bank_before = bank.digest()
    planner = ExternalModelBasedPlanner(bank, beam_width=1)
    state = torch.zeros(1, STATE_WIDTH)
    predictions = torch.stack(
        [
            bank.models[index](state, intentions[-1].unsqueeze(0)).squeeze(0)
            for index in range(bank.context_count)
        ]
    )
    probe = planner.select_disambiguating_intention(bank, state, intentions)
    hidden_regimes = [index % len(REGIME_NAMES) for index in range(episodes)]
    probe_correct = 0
    random_correct = 0
    random_probe_intentions: list[int] = []
    random_generator = torch.Generator().manual_seed(seed + 1009)
    for episode, hidden_index in enumerate(hidden_regimes):
        hidden_next_state = predictions[hidden_index].clone()
        if outcome_noise_std:
            hidden_next_state += outcome_noise_std * torch.randn(
                hidden_next_state.shape,
                generator=random_generator,
            )
        probe_route = _route(
            probe.predicted_next_states[probe.selected_intention_index],
            hidden_next_state,
            random_generator,
        )
        probe_correct += int(probe_route == hidden_index)

        random_intention_index = int(
            torch.randint(intentions.shape[0], (1,), generator=random_generator).item()
        )
        random_probe_intentions.append(random_intention_index)
        random_predictions = torch.stack(
            [
                bank.models[index](
                    state,
                    intentions[random_intention_index].unsqueeze(0),
                ).squeeze(0)
                for index in range(bank.context_count)
            ]
        )
        random_observed_next_state = random_predictions[hidden_index].clone()
        if outcome_noise_std:
            random_observed_next_state += outcome_noise_std * torch.randn(
                random_observed_next_state.shape,
                generator=random_generator,
            )
        random_route = _route(
            random_predictions,
            random_observed_next_state,
            random_generator,
        )
        random_correct += int(random_route == hidden_index)

    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    probe_accuracy = probe_correct / episodes
    random_control_accuracy = random_correct / episodes
    probe_quality = (
        probe_accuracy == 1.0
        if outcome_noise_std == 0.0
        else probe_accuracy >= 0.95
    )
    report = {
        "schema": "neural-computer.external-model-disambiguation-probe.v1",
        "seed": seed,
        "configuration": {
            "episodes": episodes,
            "candidate_count": candidate_count,
            "outcome_noise_std": outcome_noise_std,
            "regimes": REGIME_NAMES,
            "candidate_intentions": intentions.tolist(),
            "probe_policy": "max_factual_prediction_disagreement_v1",
            "random_control": "uniform_intention_random_tie_break_v1",
            "controller_frozen": True,
        },
        "metrics": {
            "probe_accuracy": probe_accuracy,
            "random_control_accuracy": random_control_accuracy,
            "probe_margin": probe_accuracy - random_control_accuracy,
            "selected_probe_intention": probe.selected_intention.tolist(),
            "selected_probe_intention_index": probe.selected_intention_index,
            "probe_disagreement_scores": probe.disagreement_scores.tolist(),
            "random_intention_histogram": {
                str(index): random_probe_intentions.count(index)
                for index in range(intentions.shape[0])
            },
        },
        "gates": {
            "probe_beats_random": probe_correct > random_correct,
            "probe_perfect": probe_correct == episodes,
            "probe_quality_gate": probe_quality,
            "disagreement_selects_informative_intention": (
                probe.selected_intention_index == candidate_count - 1
            ),
            "controller_unchanged": controller_before == _digest(controller),
            "bank_unchanged_during_queries": bank_before == bank.digest(),
            "persistence_exact": restored.digest() == bank.digest(),
        },
        "accounting": {
            "unique_verifier_lifetimes": candidate_count * 2,
            "unique_verifier_bits": candidate_count * 2,
            "external_model_observation_writes": candidate_count * 4,
            "model_acquisition_updates": 0,
            "controller_optimizer_updates": 0,
            "probe_optimizer_updates": 0,
            "raw_replayed_examples": 0,
            "search_compute": {
                "candidate_intentions": int(intentions.shape[0]),
                "candidate_models": bank.context_count,
            },
        },
        "promoted": all(
            (
                probe_quality,
                probe_correct > random_correct,
                probe.selected_intention_index == candidate_count - 1,
                controller_before == _digest(controller),
                bank_before == bank.digest(),
                restored.digest() == bank.digest(),
            )
        ),
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=83001)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--candidate-count", type=int, default=2)
    parser.add_argument("--outcome-noise-std", type=float, default=0.0)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.seed,
        args.report_out,
        episodes=args.episodes,
        candidate_count=args.candidate_count,
        outcome_noise_std=args.outcome_noise_std,
    )


if __name__ == "__main__":
    main()
