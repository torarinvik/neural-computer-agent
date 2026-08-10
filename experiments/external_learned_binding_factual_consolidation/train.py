"""Pressure test retention-safe factual consolidation after learned growth."""

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
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

SCHEMA = "neural-computer.external-learned-binding-factual-consolidation-pressure-test.v1"
STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
STREAM_COUNT = 3
ROWS = 8
IDENTITY_UPDATES = 240
FAMILY_CANDIDATES = (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
)


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


def _fixture() -> list[ExternalTransitionObservation]:
    base = torch.linspace(-1.0, 1.0, ROWS)
    observations: list[ExternalTransitionObservation] = []
    for stream in range(STREAM_COUNT):
        state = torch.stack(
            (base + stream * 4.0, 0.5 * base.square() + stream * 3.0),
            dim=-1,
        )
        intention = torch.sin(base * 1.7 + stream * 0.2).unsqueeze(-1)
        if stream < 2:
            next_state = state + intention * torch.tensor([0.2, 1.0])
        else:
            next_state = state + intention * torch.tensor([0.55, 1.0])
            next_state = next_state + 2.0 * torch.sin(state * 2.7)
        observations.append(
            ExternalTransitionObservation(
                state,
                intention,
                next_state,
                torch.ones(ROWS),
            )
        )
    return observations


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _train_identity(
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)
    final_loss = float("inf")
    for update in range(IDENTITY_UPDATES):
        left = []
        right = []
        for stream, observation in enumerate(observations):
            left.append(
                encoder.encode_observation(_row(observation, (update + stream) % ROWS))
            )
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


def _seed_redundant_router(
    encoder: ExternalTransitionContextEncoder,
    observations: list[ExternalTransitionObservation],
) -> ExternalLearnedMultiStreamTransitionContextRouter:
    binding = ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=6,
        max_streams=3,
        provisional_capacity=1,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        provisional_tolerance=0.7,
        match_margin=0.05,
    )
    keys = []
    for stream, observation in enumerate(observations):
        first = None
        for row_index in range(6):
            result = binding.observe(
                _row(observation, row_index),
                timestamp=float(stream * 100 + row_index),
            )
            if first is None:
                first = result
        if first is None or first.track_id is None or first.stream_key is None:
            raise RuntimeError("redundant consolidation stream did not bind")
        keys.append(first.stream_key)

    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family="mixed_verified_v1",
        affine_ridge=1e-7,
        random_feature_width=16,
        random_feature_seed=17,
        capacity=3,
    )
    source_index = bank.ensure_context(
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    duplicate_index = bank.ensure_context(
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    )
    distinct_index = bank.ensure_context(
        torch.tensor([0.0, 0.0, 1.0, 0.0]),
        model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    )
    for row_index in range(6):
        bank.models[source_index].observe(_row(observations[0], row_index))
        bank.models[distinct_index].observe(_row(observations[2], row_index))
    bank.models[duplicate_index].load_state_dict(
        bank.models[source_index].state_dict()
    )

    base_router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=6,
        max_contexts=3,
        defer_admission=True,
        continuation_tolerance=0.5,
        candidate_model_families=FAMILY_CANDIDATES,
        provisional_evidence_policy="streaming_statistics",
    )
    multistream = ExternalMultiStreamTransitionContextRouter(
        base_router,
        stream_key_width=CONTEXT_WIDTH,
    )
    for key, index in zip(
        keys,
        (source_index, duplicate_index, distinct_index),
        strict=True,
    ):
        multistream._child(key)
        multistream._bound_slot_ids[multistream._stream_id(key)] = bank.slot_id_at(index)
    return ExternalLearnedMultiStreamTransitionContextRouter(binding, multistream)


def run(seed: int) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    observations = _fixture()
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=24,
        context_width=CONTEXT_WIDTH,
    )
    identity_loss, identity_updates = _train_identity(encoder, observations)
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

    router = _seed_redundant_router(encoder, observations)
    first_track_id = 1
    second_track_id = 2

    source_key = router.binding.track_state(0)["stream_key"]
    equivalent_key = router.binding.track_state(first_track_id)["stream_key"]
    distinct_key = router.binding.track_state(second_track_id)["stream_key"]
    source_slot_id = router.router.bound_slot_id(source_key)
    equivalent_slot_id = router.router.bound_slot_id(equivalent_key)
    distinct_slot_id = router.router.bound_slot_id(distinct_key)
    if None in (source_slot_id, equivalent_slot_id, distinct_slot_id):
        raise RuntimeError("grown streams did not receive stable factual slots")
    slot_ids_before = router.router.bank.slot_ids
    binding_before = router.binding.digest()
    router_before = router.digest()
    physical_before = router.router.bank.physical_model_count

    distinct_rejection = router.consolidate_factual_slots_verified(
        source_key,
        distinct_key,
        [_row(observations[0], 6)],
        prediction_tolerance=0.5,
    )
    distinct_rejection_atomic = (
        not distinct_rejection.accepted
        and distinct_rejection.reason == "factual_consolidation_model_families_differ"
        and router.binding.digest() == binding_before
        and router.digest() == router_before
    )

    def retention_probe(
        candidate: ExternalLearnedMultiStreamTransitionContextRouter,
    ) -> bool:
        return (
            candidate.router.bank.slot_ids == slot_ids_before
            and candidate.router.bound_slot_id(source_key) == source_slot_id
            and candidate.router.bound_slot_id(equivalent_key) == equivalent_slot_id
            and candidate.router.bound_slot_id(distinct_key) == distinct_slot_id
        )

    mutating_before = router.digest()

    def mutating_probe(
        candidate: ExternalLearnedMultiStreamTransitionContextRouter,
    ) -> bool:
        candidate.binding.max_streams += 1
        return True

    mutating_rejection = router.consolidate_factual_slots_verified(
        source_key,
        equivalent_key,
        [_row(observations[0], 6)],
        prediction_tolerance=1e-5,
        retention_probe=mutating_probe,
    )
    mutating_rejection_atomic = (
        not mutating_rejection.accepted and router.digest() == mutating_before
    )

    consolidation = router.consolidate_factual_slots_verified(
        source_key,
        equivalent_key,
        [_row(observations[0], 6), _row(observations[1], 6)],
        prediction_tolerance=1e-5,
        retention_probe=retention_probe,
    )
    restored = ExternalLearnedMultiStreamTransitionContextRouter.from_payload(
        router.state_payload()
    )
    gates = {
        "identity_loss_converged": identity_loss < 0.05,
        "three_streams_bound": router.stream_count == 3,
        "distinct_rejection_atomic": distinct_rejection_atomic,
        "mutating_probe_rejection_atomic": mutating_rejection_atomic,
        "equivalent_consolidation_committed": consolidation.accepted,
        "slot_addresses_retained": router.router.bank.slot_ids == slot_ids_before,
        "physical_model_count_reduced": (
            physical_before == 3 and router.router.bank.physical_model_count == 2
        ),
        "binding_retained": router.binding.digest() == binding_before,
        "retention_probe_passed": consolidation.accepted,
        "exact_persistence": restored.digest() == router.digest(),
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
            "preexisting_external_slots": 3,
            "initial_factual_capacity": 3,
            "final_factual_capacity": router.router.bank.capacity,
            "candidate_model_families": list(FAMILY_CANDIDATES),
            "consolidation": "equivalent_physical_model_aliasing_v1",
            "replay_free": True,
        },
        "growth": {
            "track_ids": [0, first_track_id, second_track_id],
            "slot_ids": [source_slot_id, equivalent_slot_id, distinct_slot_id],
            "model_families": [
                router.router.bank.model_family_at(index)
                for index in range(router.router.bank.context_count)
            ],
        },
        "consolidation": {
            "accepted": consolidation.accepted,
            "first_track_id": consolidation.first_track_id,
            "second_track_id": consolidation.second_track_id,
            "first_slot_id": consolidation.first_slot_id,
            "second_slot_id": consolidation.second_slot_id,
            "physical_models_before": consolidation.physical_models_before,
            "physical_models_after": consolidation.physical_models_after,
            "max_heldout_difference": consolidation.max_heldout_difference,
            "distinct_rejection_reason": distinct_rejection.reason,
            "mutating_rejection_reason": mutating_rejection.reason,
        },
        "retention": {
            "slot_ids_before": list(slot_ids_before),
            "slot_ids_after": list(router.router.bank.slot_ids),
            "binding_digest_unchanged": router.binding.digest() == binding_before,
            "router_digest_before_consolidation": router_before,
            "router_digest_after_consolidation": router.digest(),
        },
        "gates": gates,
        "accounting": {
            "identity_optimizer_updates": identity_updates,
            "policy_optimizer_updates": 0,
            "factual_optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "growth_verifier_bits": 0,
            "consolidation_verifier_bits": 2,
            "unique_verifier_bits": 2,
            "unique_logical_lifetimes": STREAM_COUNT,
        },
        "claim_boundary": (
            "bounded retention-safe factual parameter sharing after learned "
            "binding; not learned consolidation policy, unrestricted "
            "memory growth, arbitrary new computation, or general continual learning"
        ),
        "elapsed_seconds": time.perf_counter() - begun,
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    report = run(args.seed)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
