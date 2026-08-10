"""Pressure test atomic learned binding/factual-memory growth."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    AmodalCognitiveController,
    ExternalLearnedMultiStreamTransitionContextRouter,
    ExternalMultiStreamTransitionContextRouter,
    ExternalOnlineStreamBindingMemory,
    ExternalOnlineTransitionContextRouter,
    ExternalStreamBindingLifecyclePolicy,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

SCHEMA = "neural-computer.external-learned-binding-factual-growth-pressure-test.v1"
STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
STREAM_COUNT = 4
ROWS = 8
IDENTITY_UPDATES = 240
POLICY_UPDATES = 320
FAMILY_CANDIDATES = (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
)


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
    del seed
    base = torch.linspace(-1.0, 1.0, ROWS)
    observations: list[ExternalTransitionObservation] = []
    for stream in range(STREAM_COUNT):
        state = torch.stack(
            (base + stream * 4.0, 0.5 * base.square() + stream * 3.0),
            dim=-1,
        )
        intention = (0.7 * base + 0.1 * stream).unsqueeze(-1)
        coefficient = 0.2 + 0.35 * stream
        next_state = state + intention * torch.tensor([coefficient, 1.0])
        if stream == 2:
            next_state = next_state + 2.0 * torch.sin(state * 2.7)
        if stream == 3:
            next_state = next_state - intention * torch.tensor([1.4, 0.7])
        observations.append(
            ExternalTransitionObservation(
                state,
                intention,
                next_state,
                torch.ones(ROWS),
            )
        )
    return observations


def _row(observation: ExternalTransitionObservation, index: int) -> ExternalTransitionObservation:
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
    seed: int,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    final_loss = float("inf")
    for update in range(IDENTITY_UPDATES):
        left = []
        right = []
        for stream, observation in enumerate(observations):
            left.append(encoder.encode_observation(_row(observation, (update + stream) % ROWS)))
            right.append(
                encoder.encode_observation(
                    _row(observation, (update * 3 + stream + 1) % ROWS)
                )
            )
        loss = encoder.contrastive_loss(torch.stack(left), torch.stack(right))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    encoder.eval()
    return final_loss, IDENTITY_UPDATES


def _new_binding(encoder: ExternalTransitionContextEncoder) -> ExternalOnlineStreamBindingMemory:
    return ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=6,
        max_streams=1,
        provisional_capacity=3,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        provisional_tolerance=0.7,
        match_margin=0.05,
    )


def _new_router(
    encoder: ExternalTransitionContextEncoder,
) -> ExternalLearnedMultiStreamTransitionContextRouter:
    bank_router = ExternalOnlineTransitionContextRouter(
        ExternalTransitionModelBank(
            STATE_WIDTH,
            INTENTION_WIDTH,
            CONTEXT_WIDTH,
            model_family="mixed_verified_v1",
            affine_ridge=1e-7,
            random_feature_width=16,
            random_feature_seed=17,
            capacity=1,
        ),
        encoder,
        admission_observations=6,
        max_contexts=1,
        defer_admission=True,
        continuation_tolerance=0.5,
        conflict_patience=2,
        candidate_model_families=FAMILY_CANDIDATES,
        provisional_evidence_policy="streaming_statistics",
    )
    return ExternalLearnedMultiStreamTransitionContextRouter(
        _new_binding(encoder),
        ExternalMultiStreamTransitionContextRouter(
            bank_router,
            stream_key_width=CONTEXT_WIDTH,
        ),
    )


def _train_policy(
    policy: ExternalStreamBindingLifecyclePolicy,
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
    seed: int,
    *,
    shuffle_outcomes: bool = False,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
    generator = torch.Generator().manual_seed(seed)
    accepted = 0
    verifier_bits = 0
    losses: list[float] = []
    for update in range(POLICY_UPDATES):
        order = torch.randperm(STREAM_COUNT, generator=generator).tolist()
        source, good, distractor, contradiction = order
        memory = _new_binding(encoder)
        track_for_stream: dict[int, int] = {}
        provisional_for_stream: dict[int, int] = {}
        for stream in (source,):
            for row_index in range(ROWS):
                result = memory.observe(
                    _row(observations[stream], row_index),
                    timestamp=float(row_index),
                )
                if result.track_id is None:
                    raise RuntimeError("policy training failed to bind source")
                track_for_stream.setdefault(stream, result.track_id)
                outcome = 1.0
                memory.observe_verifier_outcome(result, outcome)
                verifier_bits += 1
        for stream in (good, distractor, contradiction):
            delay_law = {good: 1.0, distractor: 2.5, contradiction: 4.0}[stream]
            for row_index in range(ROWS):
                result = memory.observe(
                    _row(observations[stream], row_index),
                    timestamp=100.0 + stream + row_index * delay_law,
                )
                if result.provisional_id is None:
                    raise RuntimeError("policy training failed to stage provisional")
                provisional_for_stream.setdefault(stream, result.provisional_id)
                if shuffle_outcomes:
                    outcome = float(torch.randint(0, 2, (), generator=generator))
                else:
                    outcome = 1.0 if stream == good else 0.0
                memory.observe_verifier_outcome(result, outcome)
                verifier_bits += 1
        proposal = policy.propose(memory, sample=True, generator=generator)
        selected_stream = (
            None
            if proposal.selected_provisional_id is None
            else next(
                (
                    stream
                    for stream, provisional_id in provisional_for_stream.items()
                    if provisional_id == proposal.selected_provisional_id
                ),
                None,
            )
        )
        if shuffle_outcomes:
            proposal_outcome = float(torch.randint(0, 2, (), generator=generator))
        else:
            proposal_outcome = float(
                selected_stream == good
                and proposal.selected_track_id == track_for_stream[source]
            )
        accepted += int(proposal_outcome == 1.0)
        verifier_bits += 1
        losses.append(policy.adaptation_step(proposal, proposal_outcome, optimizer=optimizer))
    return {
        "optimizer_updates": POLICY_UPDATES,
        "accepted_outcomes": accepted,
        "mean_loss": sum(losses) / len(losses),
        "unique_verifier_bits": verifier_bits,
        "replayed_examples": 0,
        "shuffle_outcomes": shuffle_outcomes,
    }


def _proposal_stream(
    proposal: object,
    provisional_stream: dict[int, int],
) -> int | None:
    provisional_id = getattr(proposal, "selected_provisional_id", None)
    return None if provisional_id is None else provisional_stream.get(provisional_id)


def _evaluate_policy(
    policy: ExternalStreamBindingLifecyclePolicy,
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
    seed: int,
    *,
    episodes: int = 24,
) -> dict[str, object]:
    """Evaluate proposal selection on fresh role permutations without updates."""

    generator = torch.Generator().manual_seed(seed)
    correct = 0
    verifier_bits = 0
    for _episode in range(episodes):
        source, good, distractor, contradiction = torch.randperm(
            STREAM_COUNT,
            generator=generator,
        ).tolist()
        memory = _new_binding(encoder)
        track_id = None
        provisional_stream: dict[int, int] = {}
        for row_index in range(ROWS):
            result = memory.observe(
                _row(observations[source], row_index),
                timestamp=float(row_index),
            )
            track_id = result.track_id
            memory.observe_verifier_outcome(result, 0.5)
            verifier_bits += 1
        delay_law = {good: 1.0, distractor: 2.5, contradiction: 4.0}
        for stream in (good, distractor, contradiction):
            for row_index in range(ROWS):
                result = memory.observe(
                    _row(observations[stream], row_index),
                    timestamp=100.0 + stream + row_index * delay_law[stream],
                )
                if result.provisional_id is None:
                    raise RuntimeError("policy evaluation failed to stage provisional")
                provisional_stream.setdefault(result.provisional_id, stream)
                # Keep the evaluation memory outcome-neutral.  The learned
                # policy must use generic delay telemetry, not a leaked
                # positive/negative label, while the fresh control sees the
                # same evidence.
                memory.observe_verifier_outcome(result, 0.5)
                verifier_bits += 1
        proposal = policy.propose(memory, sample=False)
        selected_stream = _proposal_stream(proposal, provisional_stream)
        correct += int(selected_stream == good and proposal.selected_track_id == track_id)
    return {
        "episodes": episodes,
        "correct": correct,
        "accuracy": correct / episodes,
        "verifier_bits": verifier_bits,
    }


def _prepare(
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
) -> dict[str, object]:
    router = _new_router(encoder)
    source_staged = None
    source_track_id = None
    for row_index in range(6):
        result = router.observe(_row(observations[0], row_index), timestamp=row_index * 1.4)
        source_track_id = result.binding.track_id
        router.observe_binding_outcome(result, 1.0)
        if result.routing is not None and result.routing.result.status == "staged":
            source_staged = result
            router.adaptation_step(result, None, replay_evidence=False)
    if source_staged is None or source_track_id is None:
        raise RuntimeError("source factual candidate was not staged")
    source_receipt = router.promote_staged_candidate(
        source_staged,
        _row(observations[0], 6),
        lambda candidate: candidate.context_count == 1,
        prediction_tolerance=100.0,
    )
    if not source_receipt.accepted:
        raise RuntimeError(f"source factual candidate failed: {source_receipt.reason}")

    target_results: dict[int, list[object]] = {1: [], 2: [], 3: []}
    target_order = (1, 3, 2)
    provisional_stream: dict[int, int] = {}
    target_timestamps = {1: 100.0, 2: 101.0, 3: 102.0}
    for row_index in range(6):
        for stream in target_order:
            result = router.observe(
                _row(observations[stream], row_index),
                timestamp=target_timestamps[stream],
            )
            target_timestamps[stream] += 0.8 + stream * 0.2
            if result.binding.provisional_id is None:
                raise RuntimeError("open-set target did not remain provisional")
            provisional_stream.setdefault(result.binding.provisional_id, stream)
            target_results[stream].append(result)
            router.observe_binding_outcome(result, 1.0 if stream == 1 else 0.0)
    return {
        "router": router,
        "source_track_id": source_track_id,
        "source_slot_id": source_receipt.slot_id,
        "target_results": target_results,
        "provisional_stream": provisional_stream,
        "bank_context_count_before_growth": router.router.bank.context_count,
        "binding_digest_before_growth": router.binding.digest(),
        "router_digest_before_growth": router.digest(),
    }


def run(seed: int) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    observations = _fixture(seed)
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=24,
        context_width=CONTEXT_WIDTH,
    )
    identity_loss, identity_updates = _train_identity(encoder, observations, seed + 11)
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
    policy_training = _train_policy(policy, encoder, observations, seed + 101)
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
        seed + 201,
        shuffle_outcomes=True,
    )
    fresh_policy = ExternalStreamBindingLifecyclePolicy(
        CONTEXT_WIDTH,
        hidden_width=24,
        learning_rate=0.02,
        temperature=2.0,
    )
    learned_control = _evaluate_policy(policy, encoder, observations, seed + 301)
    fresh_control = _evaluate_policy(fresh_policy, encoder, observations, seed + 301)
    shuffled_control = _evaluate_policy(
        shuffled_policy,
        encoder,
        observations,
        seed + 301,
    )

    prepared = _prepare(encoder, observations)
    router = prepared["router"]
    provisional_stream = prepared["provisional_stream"]
    proposal_a = policy.propose(router.binding, sample=False)
    fresh_a = fresh_policy.propose(router.binding, sample=False)
    shuffled_a = shuffled_policy.propose(router.binding, sample=False)
    learned_a_stream = _proposal_stream(proposal_a, provisional_stream)
    fresh_a_stream = _proposal_stream(fresh_a, provisional_stream)
    shuffled_a_stream = _proposal_stream(shuffled_a, provisional_stream)
    source_slot_id = prepared["source_slot_id"]
    source_model_digest = router.router.bank.models[0].digest()
    before_rejection_binding = router.binding.digest()
    before_rejection_router = router.digest()
    rejection = router.grow_with_factual_candidate(
        proposal_a,
        _row(observations[1], 6),
        0.0,
        prediction_tolerance=100.0,
    )
    scalar_rejection_atomic = (
        not rejection.accepted
        and rejection.reason == "verifier_outcome_rejected"
        and router.binding.digest() == before_rejection_binding
        and router.digest() == before_rejection_router
    )
    wrong_heldout = router.grow_with_factual_candidate(
        proposal_a,
        _row(observations[3], 6),
        1.0,
        prediction_tolerance=1.0,
    )
    wrong_heldout_atomic = (
        not wrong_heldout.accepted
        and router.binding.digest() == before_rejection_binding
        and router.digest() == before_rejection_router
    )
    growth_a = router.grow_with_factual_candidate(
        proposal_a,
        _row(observations[1], 6),
        1.0,
        prediction_tolerance=0.5,
    )
    if not growth_a.accepted:
        raise RuntimeError(f"first factual growth failed: {growth_a.reason}")

    target_b_results = prepared["target_results"][2]
    for result in target_b_results:
        router.observe_binding_outcome(result, 1.0)
    proposal_b = policy.propose(router.binding, sample=False)
    fresh_b = fresh_policy.propose(router.binding, sample=False)
    shuffled_b = shuffled_policy.propose(router.binding, sample=False)
    learned_b_stream = _proposal_stream(proposal_b, provisional_stream)
    fresh_b_stream = _proposal_stream(fresh_b, provisional_stream)
    shuffled_b_stream = _proposal_stream(shuffled_b, provisional_stream)
    growth_b = router.grow_with_factual_candidate(
        proposal_b,
        _row(observations[2], 6),
        1.0,
        prediction_tolerance=0.5,
    )
    if not growth_b.accepted:
        raise RuntimeError(f"second factual growth failed: {growth_b.reason}")

    source_retained = (
        router.router.bank.models[router.router.bank.physical_index_for_slot_id(source_slot_id)].digest()
        == source_model_digest
    )
    route_a = [
        router.observe(_row(observations[1], 6), timestamp=140.0 + index)
        for index in range(6)
    ]
    route_b = [
        router.observe(_row(observations[2], 6), timestamp=150.0 + index)
        for index in range(6)
    ]
    route_results = route_a + route_b
    routed_slot_ids = [
        result.routing.result.stable_slot_id
        for result in route_results
        if result.routing is not None and result.routing.result.stable_slot_id is not None
    ]
    delays = [
        router.binding.track_state(track_id)["mean_delay"]
        for track_id in router.binding.track_ids
    ]
    restored = ExternalLearnedMultiStreamTransitionContextRouter.from_payload(
        router.state_payload()
    )
    learned_correct_count = int(learned_a_stream == 1) + int(learned_b_stream == 2)
    fresh_correct_count = int(fresh_a_stream == 1) + int(fresh_b_stream == 2)
    shuffled_correct_count = int(shuffled_a_stream == 1) + int(shuffled_b_stream == 2)
    family_ids = [
        router.router.bank.model_family_at(index)
        for index in range(router.router.bank.context_count)
    ]
    gates = {
        "identity_loss_converged": identity_loss < 0.05,
        "learned_first_growth_proposal_correct": learned_a_stream == 1,
        "learned_second_growth_proposal_correct": learned_b_stream == 2,
        "learned_beats_fresh_control": (
            learned_control["accuracy"] > fresh_control["accuracy"]
        ),
        "learned_beats_shuffled_control": (
            learned_control["accuracy"] > shuffled_control["accuracy"]
        ),
        "scalar_rejection_atomic": scalar_rejection_atomic,
        "wrong_heldout_rejection_atomic": wrong_heldout_atomic,
        "first_growth_committed": growth_a.accepted,
        "second_growth_committed": growth_b.accepted,
        "binding_capacity_grew": router.binding.max_streams == 3,
        "factual_capacity_grew": (
            router.router.bank.capacity == 3
            and router.router.bank.context_count == 3
        ),
        "open_set_provisional_evidence_isolated": (
            prepared["bank_context_count_before_growth"] == 1
        ),
        "source_factual_slot_retained": source_retained,
        "new_slots_routed": (
            growth_a.slot_id in routed_slot_ids and growth_b.slot_id in routed_slot_ids
        ),
        "delays_learned": all(value is not None and float(value) > 0.0 for value in delays),
        "model_families_are_external": set(family_ids).issubset(set(FAMILY_CANDIDATES)),
        "joint_persistence_exact": restored.digest() == router.digest(),
        "controller_frozen_unchanged": (
            all(not parameter.requires_grad for parameter in controller.parameters())
            and controller_digest == _digest_module(controller)
        ),
        "binding_encoder_frozen": (
            all(not parameter.requires_grad for parameter in encoder.parameters())
            and encoder.digest() == encoder_digest
        ),
    }
    report = {
        "schema": SCHEMA,
        "seed": seed,
        "promoted": all(gates.values()),
        "identity": {
            "contrastive_loss": identity_loss,
            "optimizer_updates": identity_updates,
            "encoder_digest": encoder_digest,
        },
        "configuration": {
            "streams": STREAM_COUNT,
            "initial_live_streams": 1,
            "growth_transactions": 2,
            "initial_factual_capacity": 1,
            "final_factual_capacity": router.router.bank.capacity,
            "model_families": list(FAMILY_CANDIDATES),
            "replay_free_factual_candidates": True,
        },
        "policy_training": policy_training,
        "shuffled_policy_training": shuffled_training,
        "proposal_control": {
            "learned_correct_count": learned_correct_count,
            "fresh_correct_count": fresh_correct_count,
            "shuffled_correct_count": shuffled_correct_count,
            "learned_first_stream": learned_a_stream,
            "learned_second_stream": learned_b_stream,
            "fresh_first_stream": fresh_a_stream,
            "fresh_second_stream": fresh_b_stream,
            "shuffled_first_stream": shuffled_a_stream,
            "shuffled_second_stream": shuffled_b_stream,
            "learned_control": learned_control,
            "fresh_control": fresh_control,
            "shuffled_control": shuffled_control,
        },
        "growth": {
            "first": {
                "accepted": growth_a.accepted,
                "track_id": growth_a.track_id,
                "slot_id": growth_a.slot_id,
                "heldout_error": growth_a.heldout_error,
                "destination_stream_capacity": growth_a.destination_stream_capacity,
                "destination_factual_capacity": growth_a.destination_factual_capacity,
            },
            "second": {
                "accepted": growth_b.accepted,
                "track_id": growth_b.track_id,
                "slot_id": growth_b.slot_id,
                "heldout_error": growth_b.heldout_error,
                "destination_stream_capacity": growth_b.destination_stream_capacity,
                "destination_factual_capacity": growth_b.destination_factual_capacity,
            },
            "scalar_rejection_reason": rejection.reason,
            "wrong_heldout_reason": wrong_heldout.reason,
            "model_families": family_ids,
        },
        "retention": {
            "source_slot_id": source_slot_id,
            "source_model_digest_unchanged": source_retained,
            "routed_slot_ids": routed_slot_ids,
            "route_statuses": [
                None if result.routing is None else result.routing.result.status
                for result in route_results
            ],
            "route_stable_slot_ids": [
                None
                if result.routing is None
                else result.routing.result.stable_slot_id
                for result in route_results
            ],
            "route_prediction_errors": [
                None
                if result.routing is None
                else result.routing.result.prediction_error
                for result in route_results
            ],
            "mean_delays": delays,
        },
        "gates": gates,
        "accounting": {
            "identity_optimizer_updates": identity_updates,
            "policy_optimizer_updates": policy_training["optimizer_updates"],
            "shuffled_policy_optimizer_updates": shuffled_training["optimizer_updates"],
            "factual_optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "policy_verifier_bits": policy_training["unique_verifier_bits"],
            "shuffled_policy_verifier_bits": shuffled_training["unique_verifier_bits"],
            "deployment_transaction_verifier_bits": 4,
            "control_verifier_bits": (
                int(learned_control["verifier_bits"])
                + int(fresh_control["verifier_bits"])
                + int(shuffled_control["verifier_bits"])
            ),
            "unique_verifier_bits": (
                int(policy_training["unique_verifier_bits"])
                + int(shuffled_training["unique_verifier_bits"])
                + int(learned_control["verifier_bits"])
                + int(fresh_control["verifier_bits"])
                + int(shuffled_control["verifier_bits"])
                + 4
            ),
            "unique_logical_lifetimes": POLICY_UPDATES * STREAM_COUNT,
        },
        "claim_boundary": (
            "bounded learned anonymous binding and replay-free factual-memory growth "
            "under held-out retention; not unrestricted growth, learned verifier "
            "design, arbitrary new computation, or general continual learning"
        ),
        "elapsed_seconds": time.perf_counter() - begun,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2601)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    report = run(args.seed)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
