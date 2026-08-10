"""Outcome-accounted learned anonymous stream-binding audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    AmodalCognitiveController,
    ExternalLearnedMultiStreamTransitionContextRouter,
    ExternalMultiStreamTransitionContextRouter,
    ExternalOnlineStreamBindingMemory,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
STREAM_COUNT = 3
ROWS = 6
IDENTITY_UPDATES = 240
OPEN_SET_STREAM_COUNT = 4


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
        # The identity is present in the learned event distribution, but no
        # semantic name or task ID is supplied to the deployed memory.
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
    observation: ExternalTransitionObservation, index: int
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


def _new_binding(
    encoder: ExternalTransitionContextEncoder,
) -> ExternalOnlineStreamBindingMemory:
    return ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=4,
        max_streams=STREAM_COUNT,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        match_margin=0.05,
    )


def _open_set_fixture(seed: int) -> list[ExternalTransitionObservation]:
    generator = torch.Generator().manual_seed(seed + 10_000)
    observations: list[ExternalTransitionObservation] = []
    for stream in range(OPEN_SET_STREAM_COUNT):
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


def _run_open_set(seed: int) -> dict[str, object]:
    observations = _open_set_fixture(seed)
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=16,
        context_width=CONTEXT_WIDTH,
    )
    identity_loss = _train_identity(encoder, observations)
    encoder.eval()
    binding = ExternalOnlineStreamBindingMemory(
        encoder,
        window_capacity=4,
        max_streams=3,
        provisional_capacity=2,
        match_tolerance=0.55,
        new_track_tolerance=0.7,
        provisional_tolerance=0.7,
        match_margin=0.05,
    )
    initial_statuses: list[str] = []
    for stream in range(3):
        initial_statuses.append(
            binding.observe(
                _row(observations[stream], 0),
                timestamp=float(stream),
            ).status
        )
        binding.observe(
            _row(observations[stream], 1),
            timestamp=float(stream) + 1.0,
        )
    live_before_open_set = binding.track_ids
    timestamps = (10.0, 11.5, 11.8, 14.2, 14.7, 18.0)
    provisional_results = [
        binding.observe(
            _row(observations[3], row_index),
            timestamp=timestamps[row_index],
        )
        for row_index in range(ROWS)
    ]
    for result in provisional_results:
        binding.observe_verifier_outcome(result, 1.0)
    provisional_id = provisional_results[0].provisional_id
    if provisional_id is None:
        raise RuntimeError("open-set stream did not enter provisional memory")
    rejected_retirement = binding.retire_verified_track(1, lambda _: False)
    failed_admission = binding.promote_provisional_track(
        provisional_id,
        lambda _: False,
    )
    retired = binding.retire_verified_track(1, lambda candidate: candidate.stream_count == 2)
    promoted = binding.promote_provisional_track(
        provisional_id,
        lambda candidate: candidate.stream_count == 3,
    )
    restored = ExternalOnlineStreamBindingMemory.from_payload(binding.state_payload())
    gates = {
        "open_identity_loss_converged": identity_loss < 0.05,
        "initial_live_tracks_complete": initial_statuses == ["new", "new", "new"],
        "open_arrival_quarantined": all(
            result.status == "provisional" for result in provisional_results
        ),
        "open_set_did_not_mutate_live_tracks": (
            live_before_open_set == (0, 1, 2)
            and binding.track_ids == (0, 2, 3)
        ),
        "retirement_rejection_safe": (
            not rejected_retirement.accepted
            and rejected_retirement.reason == "retention_probe_rejected"
        ),
        "admission_rejection_safe": (
            not failed_admission.accepted
            and failed_admission.reason == "live_capacity_full"
        ),
        "verified_retirement_and_admission": retired.accepted and promoted.accepted,
        "irregular_delay_estimated": all(
            binding.track_state(track_id)["mean_delay"] is not None
            for track_id in binding.track_ids
        ),
        "provisional_persistence_exact": restored.digest() == binding.digest(),
    }
    return {
        "gates": gates,
        "promoted": all(gates.values()),
        "identity_loss": identity_loss,
        "identity_optimizer_updates": IDENTITY_UPDATES,
        "initial_statuses": initial_statuses,
        "provisional_statuses": [result.status for result in provisional_results],
        "provisional_id": provisional_id,
        "delay_estimates": [
            binding.track_state(track_id)["mean_delay"]
            for track_id in binding.track_ids
        ],
        "retirement_rejection": rejected_retirement.reason,
        "failed_admission": failed_admission.reason,
        "retired_track": retired.track_id,
        "promoted_track": promoted.track_id,
        "live_track_ids": binding.track_ids,
        "provisional_ids_after_promotion": binding.provisional_ids,
        "replayed_examples": 0,
    }


def _new_router(
    encoder: ExternalTransitionContextEncoder,
) -> ExternalLearnedMultiStreamTransitionContextRouter:
    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        capacity=STREAM_COUNT,
    )
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=2,
        max_contexts=STREAM_COUNT,
        defer_admission=True,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    return ExternalLearnedMultiStreamTransitionContextRouter(
        _new_binding(encoder),
        ExternalMultiStreamTransitionContextRouter(single, stream_key_width=CONTEXT_WIDTH),
    )


def _run_sequence(
    router: ExternalLearnedMultiStreamTransitionContextRouter,
    observations: list[ExternalTransitionObservation],
    order: tuple[int, ...],
    *,
    skip: set[tuple[int, int]] | None = None,
) -> tuple[dict[int, int], list[tuple[int, object]], dict[str, int]]:
    first_track: dict[int, int] = {}
    results: list[tuple[int, object]] = []
    statuses: defaultdict[str, int] = defaultdict(int)
    skipped = skip or set()
    for row_index in range(ROWS):
        for stream in order:
            if (row_index, stream) in skipped:
                continue
            result = router.observe(
                _row(observations[stream], row_index),
                timestamp=row_index + stream * 0.1,
            )
            statuses[result.binding.status] += 1
            results.append((stream, result))
            if result.binding.track_id is not None:
                first_track.setdefault(stream, result.binding.track_id)
                router.observe_binding_outcome(result, 1.0)
    return first_track, results, dict(statuses)


def _binding_consistency(
    router: ExternalLearnedMultiStreamTransitionContextRouter,
    observations: list[ExternalTransitionObservation],
) -> tuple[float, dict[int, int], dict[str, int]]:
    assignments, results, statuses = _run_sequence(
        router,
        observations,
        (0, 1, 2),
    )
    counts = {stream: 0 for stream in range(STREAM_COUNT)}
    for stream, result in results:
        if result.binding.track_id is not None:
            counts[stream] += int(result.binding.track_id == assignments.get(stream))
    return min(counts.values()) / ROWS, assignments, statuses


def run(seed: int) -> dict[str, object]:
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

    router = _new_router(encoder)
    consistency, assignments, statuses = _binding_consistency(router, observations)
    stable = all(
        router.binding.track_state(track_id)["reliability"] > 0.5
        for track_id in router.binding.track_ids
    )
    delay_estimates = [
        router.binding.track_state(track_id)["mean_delay"]
        for track_id in router.binding.track_ids
    ]

    missing_router = _new_router(encoder)
    missing_assignments, _missing_results, missing_statuses = _run_sequence(
        missing_router,
        observations,
        (0, 1, 2),
        skip={(2, 1)},
    )
    missing_isolated = (
        len(missing_assignments) == STREAM_COUNT
        and missing_router.stream_count == STREAM_COUNT
        and missing_assignments == assignments
    )

    shuffled_router = _new_router(encoder)
    shuffled_assignments, _shuffled_results, shuffled_statuses = _run_sequence(
        shuffled_router,
        observations,
        (2, 0, 1),
    )
    shuffled_partition_preserved = len(set(shuffled_assignments.values())) == STREAM_COUNT

    fresh_encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=16,
        context_width=CONTEXT_WIDTH,
    )
    fresh_router = _new_router(fresh_encoder)
    fresh_consistency, fresh_assignments, fresh_statuses = _binding_consistency(
        fresh_router,
        observations,
    )
    open_set = _run_open_set(seed)

    restored = ExternalLearnedMultiStreamTransitionContextRouter.from_payload(
        router.state_payload()
    )
    persistence_exact = restored.digest() == router.digest()
    corrupted = router.state_payload()
    corrupted["binding"]["tracks"][0]["stream_key"][0] += 0.01
    try:
        ExternalLearnedMultiStreamTransitionContextRouter.from_payload(corrupted)
    except ValueError as error:
        checksum_rejected = "checksum" in str(error)
    else:
        checksum_rejected = False

    gates = {
        "identity_loss_converged": identity_loss < 0.05,
        "learned_tracks_complete": router.stream_count == STREAM_COUNT,
        "binding_encoder_frozen": all(
            not parameter.requires_grad
            for parameter in router.binding.encoder.parameters()
        ),
        "learned_assignment_consistent": consistency == 1.0,
        "learned_beats_fresh": consistency > fresh_consistency,
        "missing_stream_isolated": missing_isolated,
        "order_control_preserved_partition": shuffled_partition_preserved,
        "delay_estimated": all(value is not None and value >= 0 for value in delay_estimates),
        "reliability_updated": stable,
        "controller_frozen_unchanged": (
            all(not parameter.requires_grad for parameter in controller.parameters())
            and controller_digest == _digest_module(controller)
        ),
        "persistence_exact": persistence_exact,
        "checksum_rejected": checksum_rejected,
    }
    gates.update(open_set["gates"])
    return {
        "schema": "neural-computer.external-learned-stream-binding-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "streams": STREAM_COUNT,
            "rows": ROWS,
            "identity_updates": IDENTITY_UPDATES,
            "stream_keys_supplied_by_caller": False,
            "deployed_identity": "frozen_event_encoder_external_tracks_v1",
            "policy": "none_factual_router_after_binding_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "identity": {
            "optimizer_updates": IDENTITY_UPDATES,
            "contrastive_loss": identity_loss,
            "learned_consistency": consistency,
            "fresh_consistency": fresh_consistency,
            "learned_assignments": assignments,
            "fresh_assignments": fresh_assignments,
            "encoder_digest": encoder_digest,
        },
        "open_set": open_set,
        "transport": {
            "statuses": statuses,
            "missing_statuses": missing_statuses,
            "shuffled_statuses": shuffled_statuses,
            "delay_estimates": delay_estimates,
            "reliability": {
                track_id: router.binding.track_state(track_id)["reliability"]
                for track_id in router.binding.track_ids
            },
        },
        "controls": {
            "fresh_encoder_optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "missing_stream_row": [2, 1],
            "order_permutation": [2, 0, 1],
            "fresh_statuses": fresh_statuses,
        },
        "accounting": {
            "unique_verifier_bits": ROWS * STREAM_COUNT,
            "open_set_unique_verifier_bits": len(open_set["provisional_statuses"]),
            "total_unique_verifier_bits": (
                ROWS * STREAM_COUNT + len(open_set["provisional_statuses"])
            ),
            "unique_logical_lifetimes": STREAM_COUNT,
            "open_set_unique_logical_lifetimes": OPEN_SET_STREAM_COUNT,
            "total_unique_logical_lifetimes": STREAM_COUNT + OPEN_SET_STREAM_COUNT,
            "identity_optimizer_updates": IDENTITY_UPDATES,
            "open_set_identity_optimizer_updates": open_set["identity_optimizer_updates"],
            "external_memory_updates": ROWS * STREAM_COUNT,
            "provisional_memory_updates": ROWS,
            "deployment_optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
        },
        "persistence": {
            "exact": persistence_exact,
            "checksum_rejected": checksum_rejected,
        },
        "controller": {
            "frozen": all(not parameter.requires_grad for parameter in controller.parameters()),
            "digest": controller_digest,
        },
        "claim_boundary": (
            "bounded learned anonymous binding over one shared factual bank; "
            "not general continual learning or unrestricted memory growth"
        ),
        "elapsed_seconds": time.perf_counter() - begun,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2301)
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
