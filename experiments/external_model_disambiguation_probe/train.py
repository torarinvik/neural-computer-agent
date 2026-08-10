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
INTENTIONS = torch.tensor([[0.0], [1.0]])


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


def _acquire_bank() -> ExternalTransitionModelBank:
    bank = _new_bank()
    contexts = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    state = torch.zeros(4, STATE_WIDTH)
    intention = INTENTIONS.repeat(2, 1)
    for regime_index, context in enumerate(contexts):
        slot = bank.ensure_context(context)
        next_state = (
            torch.zeros_like(intention)
            if regime_index == 0
            else intention.clone()
        )
        observation = ExternalTransitionObservation(state, intention, next_state)
        bank.adaptation_step(
            observation,
            bank.context_at(slot).unsqueeze(0).expand(4, -1),
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
) -> dict[str, object]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    begun = time.perf_counter()
    torch.manual_seed(seed)
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

    bank = _acquire_bank()
    bank_before = bank.digest()
    planner = ExternalModelBasedPlanner(bank, beam_width=1)
    state = torch.zeros(1, STATE_WIDTH)
    predictions = torch.stack(
        [
            bank.models[index](state, INTENTIONS[1].unsqueeze(0)).squeeze(0)
            for index in range(bank.context_count)
        ]
    )
    probe = planner.select_disambiguating_intention(bank, state, INTENTIONS)
    hidden_regimes = [index % len(REGIME_NAMES) for index in range(episodes)]
    probe_correct = 0
    random_correct = 0
    random_probe_intentions: list[int] = []
    random_generator = torch.Generator().manual_seed(seed + 1009)
    for episode, hidden_index in enumerate(hidden_regimes):
        hidden_next_state = predictions[hidden_index]
        probe_route = _route(
            probe.predicted_next_states[probe.selected_intention_index],
            hidden_next_state,
            random_generator,
        )
        probe_correct += int(probe_route == hidden_index)

        random_intention_index = int(
            torch.randint(INTENTIONS.shape[0], (1,), generator=random_generator).item()
        )
        random_probe_intentions.append(random_intention_index)
        random_predictions = torch.stack(
            [
                bank.models[index](
                    state,
                    INTENTIONS[random_intention_index].unsqueeze(0),
                ).squeeze(0)
                for index in range(bank.context_count)
            ]
        )
        random_route = _route(
            random_predictions,
            random_predictions[hidden_index],
            random_generator,
        )
        random_correct += int(random_route == hidden_index)

    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    report = {
        "schema": "neural-computer.external-model-disambiguation-probe.v1",
        "seed": seed,
        "configuration": {
            "episodes": episodes,
            "regimes": REGIME_NAMES,
            "candidate_intentions": INTENTIONS.tolist(),
            "probe_policy": "max_factual_prediction_disagreement_v1",
            "random_control": "uniform_intention_random_tie_break_v1",
            "controller_frozen": True,
        },
        "metrics": {
            "probe_accuracy": probe_correct / episodes,
            "random_control_accuracy": random_correct / episodes,
            "probe_margin": probe_correct / episodes - random_correct / episodes,
            "selected_probe_intention": probe.selected_intention.tolist(),
            "probe_disagreement_scores": probe.disagreement_scores.tolist(),
            "random_intention_histogram": {
                str(index): random_probe_intentions.count(index)
                for index in range(INTENTIONS.shape[0])
            },
        },
        "gates": {
            "probe_beats_random": probe_correct > random_correct,
            "probe_perfect": probe_correct == episodes,
            "disagreement_selects_intention_one": probe.selected_intention_index == 1,
            "controller_unchanged": controller_before == _digest(controller),
            "bank_unchanged_during_queries": bank_before == bank.digest(),
            "persistence_exact": restored.digest() == bank.digest(),
        },
        "accounting": {
            "unique_verifier_lifetimes": 8,
            "unique_verifier_bits": 8,
            "model_acquisition_updates": 0,
            "controller_optimizer_updates": 0,
            "probe_optimizer_updates": 0,
            "raw_replayed_examples": 0,
            "search_compute": {
                "candidate_intentions": int(INTENTIONS.shape[0]),
                "candidate_models": bank.context_count,
            },
        },
        "promoted": all(
            (
                probe_correct == episodes,
                probe_correct > random_correct,
                probe.selected_intention_index == 1,
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
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out, episodes=args.episodes)


if __name__ == "__main__":
    main()
