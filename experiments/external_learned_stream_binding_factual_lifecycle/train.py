"""Joint learned binding/factual-memory lifecycle pressure test."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
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

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
STREAM_COUNT = 5
IDENTITY_UPDATES = 280
POLICY_UPDATES = 480


def _fixture(seed: int) -> list[ExternalTransitionObservation]:
    generator = torch.Generator().manual_seed(seed)
    observations: list[ExternalTransitionObservation] = []
    for stream in range(STREAM_COUNT):
        state = torch.randn(6, STATE_WIDTH, generator=generator)
        state[:, 1] += stream * 5.0
        intention = torch.randn(6, INTENTION_WIDTH, generator=generator)
        next_state = state + intention * torch.tensor([0.2 + stream, 1.0])
        observations.append(
            ExternalTransitionObservation(
                state,
                intention,
                next_state,
                torch.ones(6),
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
        left = []
        right = []
        for stream, observation in enumerate(observations):
            left.append(encoder.encode_observation(_row(observation, (update + stream) % 6)))
            right.append(
                encoder.encode_observation(
                    _row(observation, (update * 3 + stream + 1) % 6)
                )
            )
        loss = encoder.contrastive_loss(torch.stack(left), torch.stack(right))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return final_loss


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _new_router(
    encoder: ExternalTransitionContextEncoder,
) -> ExternalLearnedMultiStreamTransitionContextRouter:
    bank_router = ExternalOnlineTransitionContextRouter(
        ExternalTransitionModelBank(
            STATE_WIDTH,
            INTENTION_WIDTH,
            CONTEXT_WIDTH,
            model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            capacity=2,
        ),
        encoder,
        admission_observations=2,
        max_contexts=2,
        defer_admission=True,
        continuation_tolerance=0.5,
        conflict_patience=2,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
        provisional_evidence_policy="streaming_statistics",
    )
    return ExternalLearnedMultiStreamTransitionContextRouter(
        ExternalOnlineStreamBindingMemory(
            encoder,
            window_capacity=4,
            max_streams=2,
            provisional_capacity=3,
            match_tolerance=0.55,
            new_track_tolerance=0.7,
            provisional_tolerance=0.7,
            match_margin=0.05,
        ),
        ExternalMultiStreamTransitionContextRouter(
            bank_router,
            stream_key_width=CONTEXT_WIDTH,
        ),
    )


def _new_binding(
    encoder: ExternalTransitionContextEncoder,
) -> ExternalOnlineStreamBindingMemory:
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


def _fit_policy(
    policy: ExternalStreamBindingLifecyclePolicy,
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
    seed: int,
) -> dict[str, object]:
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.02)
    generator = torch.Generator().manual_seed(seed)
    accepted = 0
    losses: list[float] = []
    for update in range(POLICY_UPDATES):
        order = torch.randperm(STREAM_COUNT, generator=generator).tolist()
        protected, evictable, good, distractor, contradiction = order
        memory = _new_binding(encoder)
        track_stream: dict[int, int] = {}
        live_results: dict[int, list[object]] = {}
        for stream, outcome in ((protected, 1.0), (evictable, 0.0)):
            live_results[stream] = []
            for row_index in range(6):
                result = memory.observe(_row(observations[stream], row_index))
                if result.track_id is None:
                    raise RuntimeError("policy-training live stream failed to bind")
                track_stream.setdefault(result.track_id, stream)
                live_results[stream].append(result)
            for result in live_results[stream]:
                memory.observe_verifier_outcome(result, outcome)
        provisional_stream: dict[int, int] = {}
        provisional_results: dict[int, list[object]] = {}
        for stream, outcome in (
            (good, 1.0 if update % 5 != 0 else 0.0),
            (distractor, 0.0),
            (contradiction, 0.0),
        ):
            provisional_results[stream] = []
            for row_index in range(6):
                result = memory.observe(_row(observations[stream], row_index))
                if result.provisional_id is None:
                    raise RuntimeError("policy-training provisional stream failed")
                provisional_stream.setdefault(result.provisional_id, stream)
                provisional_results[stream].append(result)
            for result in provisional_results[stream]:
                memory.observe_verifier_outcome(result, outcome)
        proposal = policy.propose(memory, sample=True, generator=generator)
        if proposal.selected_provisional_id is None:
            outcome = float(update % 5 == 0)
        else:
            outcome = float(
                update % 5 != 0
                and provisional_stream[proposal.selected_provisional_id] == good
                and track_stream[proposal.selected_track_id] == evictable
            )
        accepted += int(outcome == 1.0)
        losses.append(policy.adaptation_step(proposal, outcome, optimizer=optimizer))
    return {
        "optimizer_updates": POLICY_UPDATES,
        "unique_verifier_bits": POLICY_UPDATES,
        "accepted_outcomes": accepted,
        "mean_loss": sum(losses) / len(losses),
        "replayed_examples": 0,
    }


@dataclass(frozen=True)
class _Prepared:
    router: ExternalLearnedMultiStreamTransitionContextRouter
    protected_stream: int
    evictable_stream: int
    good_stream: int
    distractor_stream: int
    contradiction_stream: int
    track_stream: dict[int, int]
    provisional_stream: dict[int, int]
    live_stage_results: dict[int, object]


def _prepare(
    encoder: ExternalTransitionContextEncoder,
    observations: list,
    seed: int,
) -> _Prepared:
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(STREAM_COUNT, generator=generator).tolist()
    protected_stream, evictable_stream, good_stream, distractor_stream, contradiction_stream = order
    router = _new_router(encoder)
    track_stream: dict[int, int] = {}
    live_stage_results: dict[int, object] = {}
    for stream in (protected_stream, evictable_stream):
        live_results = []
        for row_index in range(2):
            result = router.observe(
                _row(observations[stream], row_index),
                timestamp=float(row_index),
            )
            if result.binding.track_id is None or result.routing is None:
                raise RuntimeError("live factual stream failed to bind")
            track_stream.setdefault(result.binding.track_id, stream)
            live_results.append(result)
            if result.routing.result.status == "staged":
                live_stage_results[stream] = result
                router.adaptation_step(result, None, replay_evidence=False)
        for result in live_results:
            router.observe_binding_outcome(result, 1.0 if stream == protected_stream else 0.0)
    for stream in (protected_stream, evictable_stream):
        receipt = router.promote_staged_candidate(
            live_stage_results[stream],
            _row(observations[stream], 2),
            lambda _candidate: True,
            prediction_tolerance=10_000.0,
        )
        if not receipt.accepted:
            raise RuntimeError(f"live factual stream failed promotion: {receipt.reason}")

    provisional_stream: dict[int, int] = {}
    provisional_results: dict[int, list[object]] = {}
    sequence = (
        (good_stream, 0),
        (distractor_stream, 0),
        (good_stream, 1),
        (contradiction_stream, 0),
        (distractor_stream, 1),
        (good_stream, 2),
        (contradiction_stream, 1),
        (distractor_stream, 2),
        (good_stream, 3),
        (contradiction_stream, 2),
        (distractor_stream, 3),
        (good_stream, 4),
        (contradiction_stream, 3),
        (distractor_stream, 4),
        (good_stream, 5),
        (contradiction_stream, 4),
        (distractor_stream, 5),
    )
    for step, (stream, row_index) in enumerate(sequence):
        result = router.observe(
            _row(observations[stream], row_index),
            timestamp=10.0 + step * 0.7,
        )
        if result.binding.provisional_id is None:
            raise RuntimeError("delayed provisional stream failed to quarantine")
        provisional_stream.setdefault(result.binding.provisional_id, stream)
        provisional_results.setdefault(result.binding.provisional_id, []).append(result)
    for provisional_id, results in provisional_results.items():
        stream = provisional_stream[provisional_id]
        for result in results:
            router.observe_binding_outcome(
                result,
                1.0 if stream == good_stream else 0.0,
            )
    if len(provisional_stream) != 3:
        raise RuntimeError("delayed contradictory provisional streams merged")
    return _Prepared(
        router,
        protected_stream,
        evictable_stream,
        good_stream,
        distractor_stream,
        contradiction_stream,
        track_stream,
        provisional_stream,
        live_stage_results,
    )


def _proposal_is_correct(proposal, prepared: _Prepared) -> bool:
    return (
        proposal.selected_provisional_id is not None
        and proposal.selected_track_id is not None
        and prepared.provisional_stream[proposal.selected_provisional_id]
        == prepared.good_stream
        and prepared.track_stream[proposal.selected_track_id] == prepared.evictable_stream
    )


def run(seed: int = 2501) -> dict[str, object]:
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
    policy = ExternalStreamBindingLifecyclePolicy(
        CONTEXT_WIDTH,
        hidden_width=24,
        learning_rate=0.02,
        temperature=2.0,
    )
    policy_training = _fit_policy(
        policy,
        encoder,
        observations,
        seed=seed + 1_000,
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

    prepared = _prepare(encoder, observations, seed + 2_000)
    proposal = policy.propose(prepared.router.binding, sample=False)
    fresh_router = ExternalLearnedMultiStreamTransitionContextRouter.from_payload(
        prepared.router.state_payload()
    )
    fresh_policy = ExternalStreamBindingLifecyclePolicy(
        CONTEXT_WIDTH,
        hidden_width=24,
        learning_rate=0.02,
        temperature=2.0,
    )
    fresh_proposal = fresh_policy.propose(fresh_router.binding, sample=False)
    prepared_payload = prepared.router.state_payload()
    before_binding = prepared.router.binding.digest()
    before_router = prepared.router.digest()
    rejected = prepared.router.replace_with_factual_candidate(
        proposal,
        _row(observations[prepared.good_stream], 1),
        0.0,
        prediction_tolerance=0.05,
    )
    scalar_rejection_safe = (
        not rejected.accepted
        and rejected.reason == "verifier_outcome_rejected"
        and prepared.router.binding.digest() == before_binding
        and prepared.router.digest() == before_router
    )
    wrong_heldout = prepared.router.replace_with_factual_candidate(
        proposal,
        _row(observations[prepared.distractor_stream], 1),
        1.0,
        prediction_tolerance=0.05,
    )
    wrong_heldout_safe = (
        not wrong_heldout.accepted
        and prepared.router.binding.digest() == before_binding
        and prepared.router.digest() == before_router
    )
    accepted = prepared.router.replace_with_factual_candidate(
        proposal,
        _row(observations[prepared.good_stream], 1),
        1.0,
        prediction_tolerance=0.5,
    )
    sibling_slot = prepared.router.router.bound_slot_id(
        prepared.router.binding.track_state(0)["stream_key"]
    )
    sibling_results = [
        prepared.router.observe(
            _row(observations[prepared.protected_stream], row_index),
            timestamp=30.0 + row_index * 0.7,
        )
        for row_index in (0, 0)
    ]
    sibling_result = sibling_results[-1]
    new_results = [
        prepared.router.observe(
            _row(observations[prepared.good_stream], row_index),
            timestamp=31.4 + row_index * 0.7,
        )
        for row_index in (1, 1)
    ]
    new_result = new_results[-1]
    bank_before_drift = prepared.router.bank.content_digest()
    # Keep identity stable while contradicting the newly retained factual
    # model.  A large perturbation would only test identity quarantine and
    # would not prove that the factual router protects its retained slot.
    drifted = _row(observations[prepared.good_stream], 1)
    drifted = type(drifted)(
        drifted.state,
        drifted.intention,
        drifted.next_state + 1.0,
        drifted.confidence,
    )
    drift_results = [
        prepared.router.observe(drifted, timestamp=31.4),
        prepared.router.observe(drifted, timestamp=32.1),
        prepared.router.observe(drifted, timestamp=32.8),
        prepared.router.observe(drifted, timestamp=33.5),
    ]
    bank_digest_after_drift = prepared.router.bank.content_digest()
    drift_isolated = (
        bank_digest_after_drift == bank_before_drift
        and any(
            result.binding.status == "matched"
            and result.routing is not None
            and result.routing.result.status == "conflict"
            for result in drift_results
        )
    )
    restored = ExternalLearnedMultiStreamTransitionContextRouter.from_payload(
        prepared.router.state_payload()
    )
    gates = {
        "identity_loss_converged": identity_loss < 0.05,
        "delayed_provisional_streams_isolated": len(prepared.provisional_stream) == 3,
        "learned_joint_proposal_correct": _proposal_is_correct(proposal, prepared),
        "fresh_control_not_identical": (
            _proposal_is_correct(fresh_proposal, prepared)
            < _proposal_is_correct(proposal, prepared)
        ),
        "scalar_rejection_atomic": scalar_rejection_safe,
        "wrong_heldout_rejection_atomic": wrong_heldout_safe,
        "joint_replacement_committed": accepted.accepted,
        "sibling_factual_slot_retained": (
            accepted.accepted
            and sibling_slot is not None
            and sibling_result.routing is not None
            and sibling_result.routing.result.stable_slot_id == sibling_slot
        ),
        "new_factual_slot_routed": (
            accepted.accepted
            and new_result.routing is not None
            and new_result.routing.result.stable_slot_id == accepted.slot_id
        ),
        "drift_does_not_mutate_factual_bank": drift_isolated,
        "joint_persistence_exact": restored.digest() == prepared.router.digest(),
        "controller_frozen_unchanged": (
            all(not parameter.requires_grad for parameter in controller.parameters())
            and controller_digest == _digest_module(controller)
        ),
        "binding_encoder_frozen": all(
            not parameter.requires_grad
            for parameter in prepared.router.binding.encoder.parameters()
        ),
    }
    return {
        "schema": "neural-computer.external-learned-stream-binding-factual-lifecycle-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "streams": STREAM_COUNT,
            "live_streams": 2,
            "provisional_streams": 3,
            "identity_updates": IDENTITY_UPDATES,
            "policy_updates": policy_training["optimizer_updates"],
            "factual_model_family": EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
            "factual_replay": 0,
            "controller_optimizer_updates": 0,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "identity": {
            "contrastive_loss": identity_loss,
            "encoder_digest": encoder_digest,
            "optimizer_updates": IDENTITY_UPDATES,
        },
        "policy_training": policy_training,
        "proposal": {
            "selected_provisional_id": proposal.selected_provisional_id,
            "selected_track_id": proposal.selected_track_id,
            "selected_propensity": proposal.selected_propensity,
            "correct": _proposal_is_correct(proposal, prepared),
            "fresh_correct": _proposal_is_correct(fresh_proposal, prepared),
        },
        "replacement": {
            "accepted": accepted.accepted,
            "heldout_error": accepted.heldout_error,
            "slot_id": accepted.slot_id,
            "retired_slot_id": accepted.retired_slot_id,
            "scalar_rejection_reason": rejected.reason,
            "wrong_heldout_reason": wrong_heldout.reason,
        },
        "drift": {
            "binding_statuses": [result.binding.status for result in drift_results],
            "statuses": [
                result.routing.result.status
                if result.routing is not None
                else "unbound"
                for result in drift_results
            ],
            "bank_digest_before": bank_before_drift,
            "bank_digest_after": bank_digest_after_drift,
            "bank_unchanged": drift_isolated,
        },
        "persistence": {
            "exact": restored.digest() == prepared.router.digest(),
        },
        "accounting": {
            "unique_verifier_bits": policy_training["optimizer_updates"] + 3,
            "lifecycle_verifier_bits": 3,
            "unique_logical_lifetimes": policy_training["optimizer_updates"] * STREAM_COUNT,
            "identity_optimizer_updates": IDENTITY_UPDATES,
            "policy_optimizer_updates": policy_training["optimizer_updates"],
            "factual_optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "deployment_optimizer_updates": 0,
        },
        "claim_boundary": (
            "bounded joint learned binding and factual replacement under a "
            "held-out gate; not general drift recovery, unrestricted growth, "
            "learned verifier design, or general continual learning"
        ),
        "prepared_state_digest": hashlib.sha256(
            repr(prepared_payload["sha256"]).encode("utf-8")
        ).hexdigest(),
        "elapsed_seconds": time.perf_counter() - begun,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2501)
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
