"""Two-seed matched-telemetry control for future memory relevance."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from neural_computer import ExternalTransitionModelBank
from neural_computer import ExternalTransitionModelLifetimePolicy
from neural_computer import ExternalTransitionObservation

CONTEXT_WIDTH = 5
SLOT_COUNT = 4
TRAIN_EPISODES = 400
EVAL_EPISODES = 200
HIDDEN_WIDTH = 16
LEARNING_RATE = 0.03


@dataclass(frozen=True)
class Episode:
    contexts: torch.Tensor
    heldout: ExternalTransitionObservation
    retained_slot_id: int


def _episode(generator: torch.Generator) -> Episode:
    contexts = torch.nn.functional.normalize(
        torch.randn(SLOT_COUNT, CONTEXT_WIDTH, generator=generator), dim=-1
    )
    retained_slot_id = 2 + int(torch.randint(2, (), generator=generator))
    state = torch.rand(8, 1, generator=generator) * 2.0 - 1.0
    intention = torch.rand(8, 1, generator=generator) * 2.0 - 1.0
    next_state = 0.4 * state + 0.7 * intention - 0.1
    return Episode(
        contexts=contexts,
        heldout=ExternalTransitionObservation(state, intention, next_state),
        retained_slot_id=retained_slot_id,
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
    training = ExternalTransitionObservation(
        episode.heldout.state[:4],
        episode.heldout.intention[:4],
        episode.heldout.next_state[:4],
    )
    for context in episode.contexts:
        index = bank.ensure_context(context)
        context_batch = bank.context_at(index).unsqueeze(0).expand(4, -1)
        bank.adaptation_step(training, context_batch, None)
    errors: list[float] = []
    for index in range(SLOT_COUNT):
        context = bank.context_at(index)
        context_batch = context.unsqueeze(0).expand(8, -1)
        errors.append(float(bank.loss(episode.heldout, context_batch).detach()))
    for _ in range(3):
        bank.record_lifetime_observations(bank.slot_ids, errors)
    return bank


def _probe(
    candidate: ExternalTransitionModelBank,
    expected: tuple[int, ...],
    episode: Episode,
) -> bool:
    if candidate.slot_ids == tuple(range(SLOT_COUNT)):
        return True
    if candidate.slot_ids != expected:
        return False
    for slot_id in expected:
        context = candidate.context_at(candidate.physical_index_for_slot_id(slot_id))
        context_batch = context.unsqueeze(0).expand(8, -1)
        if float(candidate.loss(episode.heldout, context_batch).detach()) > 1e-6:
            return False
    return True


def _attempt(
    policy: ExternalTransitionModelLifetimePolicy,
    episode: Episode,
    *,
    update: bool,
) -> bool:
    bank = _bank(episode)
    source = bank.slot_ids
    disposable = 5 - episode.retained_slot_id
    expected = tuple(slot_id for slot_id in source if slot_id != disposable)
    protected = torch.tensor([True, True, False, False], dtype=torch.bool)
    proposal, receipt = policy.evict_from_bank_verified(
        bank,
        protected,
        lambda candidate: candidate.slot_ids == source
        or _probe(candidate, expected, episode),
        update=update,
    )
    return bool(
        proposal.selected_slot_id == disposable
        and receipt is not None
        and receipt.accepted
    )


def _digest(policy: ExternalTransitionModelLifetimePolicy) -> str:
    return hashlib.sha256(policy.digest().encode("utf-8")).hexdigest()


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    policy = ExternalTransitionModelLifetimePolicy(
        CONTEXT_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        learning_rate=LEARNING_RATE,
    )
    training = [_episode(generator) for _ in range(TRAIN_EPISODES)]
    evaluation = [_episode(generator) for _ in range(EVAL_EPISODES)]
    train_results = [_attempt(policy, episode, update=True) for episode in training]
    eval_results = [_attempt(policy, episode, update=False) for episode in evaluation]
    restored = ExternalTransitionModelLifetimePolicy.from_payload(policy.state_payload())
    eval_accuracy = sum(eval_results) / len(eval_results)
    report = {
        "schema": "neural-computer.external-transition-future-relevance-ambiguity.v1",
        "seed": seed,
        "configuration": {
            "train_episodes": TRAIN_EPISODES,
            "evaluation_episodes": EVAL_EPISODES,
            "slot_count": SLOT_COUNT,
            "matched_telemetry": "usage_age_prediction_error_equalized_v1",
            "hidden_future_rule": "random_retained_candidate_v1",
        },
        "gates": {
            "no_signal_above_random": eval_accuracy <= 0.60,
            "exact_policy_persistence": restored.digest() == policy.digest(),
            "zero_controller_updates": True,
            "zero_replayed_transition_examples": True,
        },
        "promoted": False,
        "metrics": {
            "learned_train_accuracy": sum(train_results) / len(train_results),
            "learned_eval_accuracy": eval_accuracy,
            "random_expected_accuracy": 0.5,
            "policy_digest": _digest(policy),
        },
        "accounting": {
            "unique_verifier_bits": TRAIN_EPISODES,
            "unique_logical_lifetimes": TRAIN_EPISODES * SLOT_COUNT,
            "policy_optimizer_updates": TRAIN_EPISODES,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "old_memory_replay": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "decisively rejected as a general relevance mechanism; generic telemetry cannot reveal random future relevance",
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
