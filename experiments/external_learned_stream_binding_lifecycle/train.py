"""Outcome-trained, retention-safe anonymous binding lifecycle audit."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalOnlineStreamBindingMemory,
    ExternalStreamBindingLifecyclePolicy,
    ExternalStreamBindingLifecycleProposal,
    ExternalTransitionContextEncoder,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
STREAM_COUNT = 5
ROWS = 6
IDENTITY_UPDATES = 280
POLICY_UPDATES = 480


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _fixture(seed: int) -> list[ExternalTransitionObservation]:
    generator = torch.Generator().manual_seed(seed)
    observations: list[ExternalTransitionObservation] = []
    for stream in range(STREAM_COUNT):
        state = torch.randn(ROWS, STATE_WIDTH, generator=generator)
        state[:, 1] += stream * 5.0
        intention = torch.randn(ROWS, INTENTION_WIDTH, generator=generator)
        next_state = state + intention * torch.tensor([0.2 + stream, 1.0])
        observations.append(
            ExternalTransitionObservation(
                state,
                intention,
                next_state,
                torch.ones(ROWS),
            )
        )
    return observations


def _row(
    observation: ExternalTransitionObservation,
    index: int,
) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        observation.state[index : index + 1],
        observation.intention[index : index + 1],
        observation.next_state[index : index + 1],
        observation.confidence[index : index + 1]
        if observation.confidence is not None
        else None,
    )


def _train_identity(
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
) -> float:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    final_loss = float("inf")
    for update in range(IDENTITY_UPDATES):
        left: list[torch.Tensor] = []
        right: list[torch.Tensor] = []
        for stream, observation in enumerate(observations):
            left_index = (update + stream) % ROWS
            right_index = (update * 3 + stream + 1) % ROWS
            left.append(encoder.encode_observation(_row(observation, left_index)))
            right.append(encoder.encode_observation(_row(observation, right_index)))
        loss = encoder.contrastive_loss(torch.stack(left), torch.stack(right))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


def _new_memory(encoder: ExternalTransitionContextEncoder) -> ExternalOnlineStreamBindingMemory:
    return ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=4,
        max_streams=2,
        provisional_capacity=3,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        provisional_tolerance=0.7,
        match_margin=0.05,
    )


@dataclass(frozen=True)
class _Episode:
    memory: ExternalOnlineStreamBindingMemory
    track_stream: dict[int, int]
    provisional_stream: dict[int, int]
    good_stream: int
    evictable_stream: int
    safe_replacement: bool


def _episode(
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
    seed: int,
    *,
    safe_replacement: bool,
) -> _Episode:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(STREAM_COUNT, generator=generator).tolist()
    protected_stream, evictable_stream, good_stream, distractor_stream, contradiction_stream = order
    memory = _new_memory(encoder)
    track_stream: dict[int, int] = {}
    for stream, outcome in (
        (protected_stream, 1.0),
        (evictable_stream, 0.0),
    ):
        live_results = []
        for row_index in range(ROWS):
            result = memory.observe(
                _row(observations[stream], row_index),
                timestamp=float(row_index),
            )
            if result.track_id is None:
                raise RuntimeError("live stream failed to bind in lifecycle fixture")
            track_stream.setdefault(result.track_id, stream)
            live_results.append(result)
        for result in live_results:
            memory.observe_verifier_outcome(result, outcome)

    provisional_stream: dict[int, int] = {}
    for stream, outcome in (
        (good_stream, 1.0 if safe_replacement else 0.0),
        (distractor_stream, 0.0),
        (contradiction_stream, 0.0),
    ):
        provisional_results = []
        for row_index in range(ROWS):
            result = memory.observe(
                _row(observations[stream], row_index),
                timestamp=10.0 + row_index + stream * 0.1,
            )
            if result.provisional_id is None:
                raise RuntimeError("provisional stream failed to quarantine")
            provisional_stream.setdefault(result.provisional_id, stream)
            provisional_results.append(result)
        for result in provisional_results:
            memory.observe_verifier_outcome(result, outcome)
    if len(provisional_stream) != 3:
        raise RuntimeError("multiple provisional identities were not isolated")
    return _Episode(
        memory,
        track_stream,
        provisional_stream,
        good_stream,
        evictable_stream,
        safe_replacement,
    )


def _verifier_outcome(
    proposal: ExternalStreamBindingLifecycleProposal,
    episode: _Episode,
) -> float:
    if proposal.selected_provisional_id is None:
        return float(not episode.safe_replacement)
    provisional_stream = episode.provisional_stream[proposal.selected_provisional_id]
    track_stream = episode.track_stream[proposal.selected_track_id]
    return float(
        episode.safe_replacement
        and provisional_stream == episode.good_stream
        and track_stream == episode.evictable_stream
    )


def _train_policy(
    policy: ExternalStreamBindingLifecyclePolicy,
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
    *,
    seed: int,
    shuffle_outcomes: bool = False,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    accepted = 0
    for update in range(POLICY_UPDATES):
        episode = _episode(
            encoder,
            observations,
            seed + update,
            safe_replacement=update % 5 != 0,
        )
        proposal = policy.propose(episode.memory, sample=True, generator=generator)
        outcome = _verifier_outcome(proposal, episode)
        if shuffle_outcomes:
            outcome = float(torch.randint(0, 2, (), generator=generator))
        losses.append(policy.adaptation_step(proposal, outcome, optimizer=optimizer))
        accepted += int(outcome == 1.0)
    return {
        "optimizer_updates": POLICY_UPDATES,
        "mean_loss": sum(losses) / len(losses),
        "verifier_accepts": accepted,
        "replayed_examples": 0,
    }


def _evaluate(
    policy: ExternalStreamBindingLifecyclePolicy,
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
    *,
    seed: int,
    safe_replacement: bool,
    episodes: int = 24,
) -> dict[str, object]:
    correct = 0
    propensities: list[float] = []
    proposals: list[tuple[int | None, int | None]] = []
    for offset in range(episodes):
        episode = _episode(
            encoder,
            observations,
            seed + offset,
            safe_replacement=safe_replacement,
        )
        proposal = policy.propose(episode.memory, sample=False)
        propensities.append(proposal.selected_propensity)
        proposals.append(
            (proposal.selected_provisional_id, proposal.selected_track_id)
        )
        correct += int(_verifier_outcome(proposal, episode) == 1.0)
    return {
        "accuracy": correct / episodes,
        "correct": correct,
        "episodes": episodes,
        "propensity_min": min(propensities),
        "proposals": proposals,
    }


def run(seed: int = 2401) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    observations = _fixture(seed)
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=16,
        context_width=CONTEXT_WIDTH,
    )
    identity_loss = _train_identity(encoder, observations)
    encoder.eval()
    encoder_digest = encoder.digest()

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

    policy = ExternalStreamBindingLifecyclePolicy(
        CONTEXT_WIDTH,
        hidden_width=24,
        learning_rate=0.02,
        temperature=2.0,
    )
    policy_training = _train_policy(
        policy,
        encoder,
        observations,
        seed=seed + 1_000,
    )
    shuffled_policy = ExternalStreamBindingLifecyclePolicy(
        CONTEXT_WIDTH,
        hidden_width=24,
        learning_rate=0.02,
        temperature=2.0,
    )
    shuffled_training = _train_policy(
        shuffled_policy,
        encoder,
        observations,
        seed=seed + 2_000,
        shuffle_outcomes=True,
    )
    fresh_policy = ExternalStreamBindingLifecyclePolicy(
        CONTEXT_WIDTH,
        hidden_width=24,
        learning_rate=0.02,
        temperature=2.0,
    )
    safe = _evaluate(
        policy,
        encoder,
        observations,
        seed=seed + 3_000,
        safe_replacement=True,
    )
    fresh_safe = _evaluate(
        fresh_policy,
        encoder,
        observations,
        seed=seed + 3_000,
        safe_replacement=True,
    )
    contradiction = _evaluate(
        policy,
        encoder,
        observations,
        seed=seed + 4_000,
        safe_replacement=False,
    )
    shuffled = _evaluate(
        shuffled_policy,
        encoder,
        observations,
        seed=seed + 3_000,
        safe_replacement=True,
    )

    audit_episode = _episode(
        encoder,
        observations,
        seed + 5_000,
        safe_replacement=True,
    )
    audit_proposal = policy.propose(audit_episode.memory, sample=False)
    before = audit_episode.memory.digest()
    rejected = audit_episode.memory.replace_on_verifier_outcome(
        audit_proposal.selected_provisional_id or 0,
        audit_proposal.selected_track_id or 0,
        0.0,
    )
    rejected_atomic = (
        not rejected.accepted
        and rejected.reason
        in {"verifier_outcome_rejected", "unknown_provisional", "unknown_track"}
        and audit_episode.memory.digest() == before
    )
    policy_restored = ExternalStreamBindingLifecyclePolicy.from_payload(
        policy.state_payload()
    )
    gates = {
        "identity_loss_converged": identity_loss < 0.05,
        "multiple_provisional_identities": len(audit_episode.provisional_stream) == 3,
        "learned_safe_policy": safe["accuracy"] >= 0.75,
        "learned_beats_fresh": safe["accuracy"] > fresh_safe["accuracy"] + 0.2,
        "contradiction_prefers_hold": contradiction["accuracy"] >= 0.75,
        "outcome_shuffle_control_lower": shuffled["accuracy"] < safe["accuracy"],
        "propensity_logged": safe["propensity_min"] > 0.0,
        "atomic_rejection_safe": rejected_atomic,
        "policy_persistence_exact": policy_restored.digest() == policy.digest(),
        "binding_encoder_frozen": all(
            not parameter.requires_grad for parameter in audit_episode.memory.encoder.parameters()
        ),
        "controller_frozen_unchanged": (
            all(not parameter.requires_grad for parameter in controller.parameters())
            and controller_digest == _digest_module(controller)
        ),
    }
    return {
        "schema": "neural-computer.external-stream-binding-lifecycle-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "streams": STREAM_COUNT,
            "rows": ROWS,
            "identity_updates": IDENTITY_UPDATES,
            "policy_updates": POLICY_UPDATES,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "stream_keys_supplied_by_caller": False,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "identity": {
            "contrastive_loss": identity_loss,
            "optimizer_updates": IDENTITY_UPDATES,
            "encoder_digest": encoder_digest,
        },
        "policy_training": policy_training,
        "shuffled_policy_training": shuffled_training,
        "evaluation": {
            "learned_safe": safe,
            "fresh_safe": fresh_safe,
            "learned_contradiction": contradiction,
            "shuffled_outcome": shuffled,
        },
        "accounting": {
            "unique_verifier_bits": POLICY_UPDATES * 2,
            "unique_logical_lifetimes": POLICY_UPDATES * 5 * 2,
            "identity_optimizer_updates": IDENTITY_UPDATES,
            "policy_optimizer_updates": POLICY_UPDATES * 2,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "deployment_optimizer_updates": 0,
        },
        "claim_boundary": (
            "bounded outcome-trained anonymous replacement proposals with atomic "
            "retention verification; not learned verifier design, unrestricted "
            "growth, or general continual learning"
        ),
        "elapsed_seconds": time.perf_counter() - begun,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2401)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    report = run(args.seed)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
