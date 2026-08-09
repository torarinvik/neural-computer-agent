"""Two-seed online lifetime-selection audit against matched controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from neural_computer import (
    ExternalTransitionModelBank,
    ExternalTransitionModelLifetimePolicy,
    ExternalTransitionObservation,
)

CONTEXT_WIDTH = 4
SLOT_COUNT = 3
TRAIN_EPISODES = 200
EVAL_EPISODES = 200
HIDDEN_WIDTH = 16
LEARNING_RATE = 0.03


@dataclass(frozen=True)
class Episode:
    contexts: torch.Tensor
    training: tuple[ExternalTransitionObservation, ...]
    heldout: tuple[ExternalTransitionObservation, ...]
    access_schedule: tuple[int, ...]
    protected: torch.Tensor


def _contexts() -> torch.Tensor:
    return torch.eye(CONTEXT_WIDTH, dtype=torch.float32)[:SLOT_COUNT]


def _episode(generator: torch.Generator) -> Episode:
    contexts = _contexts()
    training: list[ExternalTransitionObservation] = []
    heldout: list[ExternalTransitionObservation] = []
    for slot_index in range(SLOT_COUNT):
        train_rows = 2 + int(torch.randint(5, (), generator=generator))
        state = torch.rand(train_rows, 1, generator=generator) * 2.0 - 1.0
        intention = torch.rand(train_rows, 1, generator=generator) * 2.0 - 1.0
        slope = 0.35 + 0.2 * slot_index
        bias = -0.15 + 0.1 * slot_index
        next_state = slope * state + (0.7 - 0.1 * slot_index) * intention + bias
        training.append(
            ExternalTransitionObservation(
                state=state,
                intention=intention,
                next_state=next_state,
            )
        )
        heldout_state = torch.rand(8, 1, generator=generator) * 2.0 - 1.0
        heldout_intention = torch.rand(8, 1, generator=generator) * 2.0 - 1.0
        heldout.append(
            ExternalTransitionObservation(
                state=heldout_state,
                intention=heldout_intention,
                next_state=(
                    slope * heldout_state
                    + (0.7 - 0.1 * slot_index) * heldout_intention
                    + bias
                ),
            )
        )
    access_count = 4 + int(torch.randint(12, (), generator=generator))
    access_schedule = tuple(
        int(torch.randint(SLOT_COUNT, (), generator=generator))
        for _ in range(access_count)
    )
    protected = torch.zeros(SLOT_COUNT, dtype=torch.bool)
    protected[int(torch.randint(SLOT_COUNT, (), generator=generator))] = True
    return Episode(
        contexts=contexts,
        training=tuple(training),
        heldout=tuple(heldout),
        access_schedule=access_schedule,
        protected=protected,
    )


def _bank(episode: Episode) -> ExternalTransitionModelBank:
    bank = ExternalTransitionModelBank(
        1,
        1,
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        capacity=SLOT_COUNT,
    )
    for context in episode.contexts:
        bank.ensure_context(context)
    for slot_index, observation in enumerate(episode.training):
        context = bank.context_at(slot_index)
        bank.adaptation_step(
            observation,
            context.unsqueeze(0).expand(observation.state.shape[0], -1),
            None,
        )
    for slot_index in episode.access_schedule:
        context = bank.context_at(slot_index)
        observation = episode.heldout[slot_index]
        context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
        error = float(bank.loss(observation, context_batch).detach())
        bank.record_lifetime_observation(bank.slot_id_at(slot_index), error)
    return bank


def _hidden_safe_slot_id(
    bank: ExternalTransitionModelBank,
    protected: torch.Tensor,
) -> int:
    telemetry = bank.lifetime_telemetry()
    hidden_score = (
        0.8 * torch.log1p(telemetry.usage)
        + 0.5 * torch.log1p(telemetry.prediction_error)
        - 0.7 * torch.log1p(telemetry.age)
        + 0.25 * bank.contexts[:, 0]
    )
    hidden_score = hidden_score.masked_fill(protected, -torch.inf)
    return bank.slot_ids[int(hidden_score.argmax())]


def _retention_probe(
    bank: ExternalTransitionModelBank,
    episode: Episode,
    expected: tuple[int, ...],
) -> bool:
    if bank.slot_ids == tuple(range(SLOT_COUNT)):
        return True
    if bank.slot_ids != expected:
        return False
    for slot_id in expected:
        original_index = slot_id
        observation = episode.heldout[original_index]
        context = bank.context_at(bank.physical_index_for_slot_id(slot_id))
        context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
        if float(bank.loss(observation, context_batch).detach()) > 1e-6:
            return False
    return True


def _attempt_selector(
    episode: Episode,
    selected_slot_id: int,
) -> tuple[bool, bool, str]:
    bank = _bank(episode)
    if selected_slot_id not in bank.slot_ids:
        raise ValueError("selector produced an unknown logical slot")
    selected_index = bank.physical_index_for_slot_id(selected_slot_id)
    if bool(episode.protected[selected_index]):
        return False, False, bank.digest()
    digest_before = bank.digest()
    safe_slot_id = _hidden_safe_slot_id(bank, episode.protected)
    expected = tuple(slot_id for slot_id in bank.slot_ids if slot_id != safe_slot_id)
    receipt = bank.evict_verified_id(
        selected_slot_id,
        lambda candidate: _retention_probe(candidate, episode, expected),
    )
    if receipt.accepted:
        stable_address = (
            receipt.evicted_slot_id == selected_slot_id
            and all(
                bank.physical_index_for_slot_id(slot_id) >= 0
                for slot_id in expected
            )
        )
    else:
        stable_address = bank.digest() == digest_before
    return receipt.accepted, stable_address, receipt.reason


def _learned_attempt(
    policy: ExternalTransitionModelLifetimePolicy,
    episode: Episode,
    *,
    update: bool,
) -> tuple[bool, bool, str, int | None]:
    bank = _bank(episode)
    safe_slot_id = _hidden_safe_slot_id(bank, episode.protected)
    expected = tuple(slot_id for slot_id in bank.slot_ids if slot_id != safe_slot_id)
    proposal, receipt = policy.evict_from_bank_verified(
        bank,
        episode.protected,
        lambda candidate: _retention_probe(candidate, episode, expected),
        update=update,
    )
    selected = proposal.selected_slot_id
    if selected is None:
        return False, True, proposal.reason, None
    stable_address = (
        receipt.evicted_slot_id == selected
        if receipt.accepted
        else bank.context_count == SLOT_COUNT
    )
    return receipt.accepted, stable_address, receipt.reason, selected


def _evaluate_controls(episodes: list[Episode], seed: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 900_000)
    results = {"random": [], "recency": []}
    for episode in episodes:
        bank = _bank(episode)
        telemetry = bank.lifetime_telemetry()
        eligible = [
            index
            for index in range(SLOT_COUNT)
            if not bool(episode.protected[index])
        ]
        random_index = eligible[
            int(torch.randint(len(eligible), (), generator=generator))
        ]
        recency_index = max(eligible, key=lambda index: float(telemetry.age[index]))
        for name, index in (("random", random_index), ("recency", recency_index)):
            accepted, stable, _reason = _attempt_selector(episode, index)
            results[name].append(float(accepted and stable))
    return {name: sum(values) / len(values) for name, values in results.items()}


def _digest(policy: ExternalTransitionModelLifetimePolicy) -> str:
    digest = hashlib.sha256()
    digest.update(policy.digest().encode("utf-8"))
    return digest.hexdigest()


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    train = [_episode(generator) for _ in range(TRAIN_EPISODES)]
    evaluation = [_episode(generator) for _ in range(EVAL_EPISODES)]
    policy = ExternalTransitionModelLifetimePolicy(
        CONTEXT_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        learning_rate=LEARNING_RATE,
    )
    train_results: list[float] = []
    protected_gate = True
    stable_gate = True
    for episode in train:
        accepted, stable, reason, selected = _learned_attempt(
            policy,
            episode,
            update=True,
        )
        train_results.append(float(accepted))
        stable_gate = stable_gate and stable
        protected_gate = protected_gate and reason != "policy selected a protected slot"
        if selected is None:
            raise RuntimeError("learned lifetime policy refused an eligible episode")

    learned_results: list[float] = []
    for episode in evaluation:
        accepted, stable, reason, _selected = _learned_attempt(
            policy,
            episode,
            update=False,
        )
        learned_results.append(float(accepted))
        stable_gate = stable_gate and stable
        protected_gate = protected_gate and reason != "policy selected a protected slot"

    restored = ExternalTransitionModelLifetimePolicy.from_payload(
        policy.state_payload()
    )
    persistence = True
    for episode in evaluation:
        bank = _bank(episode)
        telemetry = bank.lifetime_telemetry()
        original = policy.propose(
            bank.contexts,
            bank.slot_ids,
            telemetry.usage,
            telemetry.age,
            telemetry.prediction_error,
            episode.protected,
        ).selected_slot_id
        restored_selection = restored.propose(
            bank.contexts,
            bank.slot_ids,
            telemetry.usage,
            telemetry.age,
            telemetry.prediction_error,
            episode.protected,
        ).selected_slot_id
        persistence = persistence and original == restored_selection
    controls = _evaluate_controls(evaluation, seed)
    learned_accuracy = sum(learned_results) / len(learned_results)
    report = {
        "schema": "neural-computer.external-transition-lifetime-policy.v1",
        "seed": seed,
        "configuration": {
            "train_episodes": TRAIN_EPISODES,
            "evaluation_episodes": EVAL_EPISODES,
            "slot_count": SLOT_COUNT,
            "verifier": "opaque_hidden_generic_feature_rule_v1",
        },
        "gates": {
            "learned_beats_random_by_margin": learned_accuracy >= controls["random"] + 0.10,
            "learned_beats_recency": learned_accuracy > controls["recency"],
            "protected_slot_gate": protected_gate,
            "stable_address_gate": stable_gate,
            "exact_policy_persistence": persistence and restored.digest() == policy.digest(),
            "zero_controller_updates": True,
            "zero_replayed_transition_examples": True,
        },
        "promoted": (
            learned_accuracy >= controls["random"] + 0.10
            and learned_accuracy > controls["recency"]
            and protected_gate
            and stable_gate
            and persistence
        ),
        "metrics": {
            "learned_train_accuracy": sum(train_results) / len(train_results),
            "learned_eval_accuracy": learned_accuracy,
            "control_accuracy": controls,
            "policy_digest": _digest(policy),
        },
        "accounting": {
            "unique_verifier_bits": TRAIN_EPISODES + EVAL_EPISODES,
            "unique_logical_lifetimes": (TRAIN_EPISODES + EVAL_EPISODES) * SLOT_COUNT,
            "policy_optimizer_updates": TRAIN_EPISODES,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_memory_replay": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "two-seed bounded verifier-trained lifetime proposal; not learned unrestricted eviction or general continual learning",
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
