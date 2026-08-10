"""End-to-end causal audit of active probing through the online router."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 1
CONTEXT_WIDTH = 2
EPISODES = 256
CANDIDATE_COUNT = 8
OUTCOME_NOISE_STD = 0.1
MATCH_TOLERANCE = 0.5
MATCH_MARGIN = 0.05


def _intentions(candidate_count: int) -> torch.Tensor:
    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")
    return torch.eye(candidate_count, dtype=torch.float32)


def _acquire_bank(intentions: torch.Tensor) -> ExternalTransitionModelBank:
    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        intentions.shape[1],
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-3,
        capacity=2,
    )
    contexts = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    state = torch.zeros(intentions.shape[0] * 2, STATE_WIDTH)
    intention = intentions.repeat(2, 1)
    for regime_index, context in enumerate(contexts):
        slot = bank.ensure_context(context)
        next_state = torch.zeros(state.shape[0], STATE_WIDTH)
        if regime_index == 1:
            informative_rows = (intention == intentions[-1]).all(dim=-1)
            next_state[informative_rows, 0] = 1.0
        bank.adaptation_step(
            ExternalTransitionObservation(state, intention, next_state),
            bank.context_at(slot).unsqueeze(0).expand(state.shape[0], -1),
            None,
        )
    return bank


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _route_error(
    predictions: torch.Tensor,
    observed_next_state: torch.Tensor,
) -> tuple[int | None, bool]:
    errors = (predictions - observed_next_state).square().mean(dim=-1)
    order = torch.argsort(errors)
    margin = float(errors[order[1]] - errors[order[0]])
    if float(errors[order[0]]) > MATCH_TOLERANCE or margin < MATCH_MARGIN:
        return None, False
    return int(order[0].item()), True


def run(
    seed: int,
    report_out: Path,
    *,
    episodes: int = EPISODES,
    candidate_count: int = CANDIDATE_COUNT,
    outcome_noise_std: float = OUTCOME_NOISE_STD,
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
        intention_width=intentions.shape[1],
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_before = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    bank = _acquire_bank(intentions)
    bank_before = bank.digest()
    router = ExternalOnlineTransitionContextRouter(
        bank,
        ExternalTransitionContextEncoder(
            STATE_WIDTH,
            intentions.shape[1],
            hidden_width=8,
            context_width=CONTEXT_WIDTH,
        ),
        match_tolerance=MATCH_TOLERANCE,
        match_margin=MATCH_MARGIN,
        admission_observations=1,
        max_contexts=2,
    )
    state = torch.zeros(1, STATE_WIDTH)
    ambiguous = ExternalTransitionObservation(
        state=state,
        intention=intentions[0].unsqueeze(0),
        next_state=torch.zeros(1, STATE_WIDTH),
    )
    generator = torch.Generator().manual_seed(seed + 9011)
    active_resolved = 0
    active_correct = 0
    active_informative = 0
    random_resolved = 0
    random_correct = 0
    random_intention_histogram = [0 for _ in range(candidate_count)]
    for episode in range(episodes):
        hidden_index = episode % 2
        probe = router.request_disambiguation_probe(ambiguous, intentions)
        active_informative += int(probe.selected_intention_index == candidate_count - 1)
        active_next_state = bank.models[hidden_index](
            state,
            probe.selected_intention.unsqueeze(0),
        ).squeeze(0)
        active_next_state = active_next_state + outcome_noise_std * torch.randn(
            active_next_state.shape,
            generator=generator,
        )
        result = router.observe(
            ExternalTransitionObservation(
                state=state,
                intention=probe.selected_intention.unsqueeze(0),
                next_state=active_next_state.unsqueeze(0),
            )
        )
        if result.status == "matched":
            active_resolved += 1
            active_correct += int(result.slot_index == hidden_index)

        random_index = int(
            torch.randint(candidate_count, (1,), generator=generator).item()
        )
        random_intention_histogram[random_index] += 1
        random_predictions = torch.stack(
            [
                bank.models[index](
                    state,
                    intentions[random_index].unsqueeze(0),
                ).squeeze(0)
                for index in range(bank.context_count)
            ]
        )
        random_next_state = random_predictions[hidden_index] + (
            outcome_noise_std
            * torch.randn(random_predictions[hidden_index].shape, generator=generator)
        )
        random_index_guess, resolved = _route_error(
            random_predictions,
            random_next_state,
        )
        random_resolved += int(resolved)
        random_correct += int(resolved and random_index_guess == hidden_index)

    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    active_resolution_rate = active_resolved / episodes
    active_accuracy = active_correct / episodes
    random_resolution_rate = random_resolved / episodes
    random_accuracy = random_correct / episodes
    report = {
        "schema": "neural-computer.external-online-disambiguation-probe.v1",
        "seed": seed,
        "configuration": {
            "episodes": episodes,
            "candidate_count": candidate_count,
            "outcome_noise_std": outcome_noise_std,
            "match_tolerance": MATCH_TOLERANCE,
            "match_margin": MATCH_MARGIN,
            "probe_policy": "router_request_model_disagreement_then_observe_v1",
            "regime_labels_used_by_router": False,
            "controller_frozen": True,
        },
        "metrics": {
            "active_resolution_rate": active_resolution_rate,
            "active_routing_accuracy": active_accuracy,
            "active_informative_selection_rate": active_informative / episodes,
            "random_resolution_rate": random_resolution_rate,
            "random_routing_accuracy": random_accuracy,
            "active_resolution_margin": active_resolution_rate
            - random_resolution_rate,
            "random_intention_histogram": {
                str(index): count
                for index, count in enumerate(random_intention_histogram)
            },
        },
        "gates": {
            "active_resolution_quality_gate": active_resolution_rate >= 0.95,
            "active_routing_quality_gate": active_accuracy >= 0.95,
            "active_selects_informative_every_episode": active_informative == episodes,
            "active_beats_random_resolution": active_resolved > random_resolved,
            "active_selects_informative_intention": active_informative == episodes,
            "controller_unchanged": controller_before == _digest(controller),
            "bank_unchanged": bank_before == bank.digest(),
            "persistence_exact": restored.bank.digest() == router.bank.digest(),
        },
        "accounting": {
            "unique_verifier_lifetimes": candidate_count * 2,
            "unique_verifier_bits": candidate_count * 2,
            "probe_requests": episodes,
            "router_observations": episodes,
            "external_model_observation_writes": candidate_count * 4,
            "controller_optimizer_updates": 0,
            "old_regime_replay": 0,
            "raw_replayed_examples": 0,
        },
        "promoted": all(
            (
                active_resolution_rate >= 0.95,
                active_accuracy >= 0.95,
                active_informative == episodes,
                active_resolved > random_resolved,
                controller_before == _digest(controller),
                bank_before == bank.digest(),
                restored.bank.digest() == router.bank.digest(),
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
    parser.add_argument("--seed", type=int, default=83201)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--candidate-count", type=int, default=CANDIDATE_COUNT)
    parser.add_argument("--outcome-noise-std", type=float, default=OUTCOME_NOISE_STD)
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
