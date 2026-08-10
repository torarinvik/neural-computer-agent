"""Accounted interleaved multi-stream factual-memory pressure test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from neural_computer import (
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    AmodalCognitiveController,
    ExternalMultiStreamTransitionContextRouter,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
STREAM_KEY_WIDTH = 3
STREAM_COUNT = 3
ADMISSION_OBSERVATIONS = 2
PREDICTION_TOLERANCE = 0.2


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _stream_key(index: int) -> torch.Tensor:
    return torch.eye(STREAM_KEY_WIDTH, dtype=torch.float32)[index]


def _observation(stream_index: int, row_index: int) -> ExternalTransitionObservation:
    state = torch.tensor(
        [[0.1 + 0.13 * row_index, -0.4 + 0.09 * row_index]],
        dtype=torch.float32,
    )
    intention = torch.tensor([[0.2 + 0.17 * row_index]], dtype=torch.float32)
    next_state = torch.cat(
        (
            state[:, :1] + (0.18 + 0.11 * stream_index) * intention,
            state[:, 1:] + 0.07 * stream_index - (0.09 + 0.03 * stream_index) * intention,
        ),
        dim=1,
    )
    return ExternalTransitionObservation(
        state=state,
        intention=intention,
        next_state=next_state,
        confidence=torch.ones(1),
    ).validate(state_width=STATE_WIDTH, intention_width=INTENTION_WIDTH)


def _run(seed: int) -> dict[str, object]:
    torch.set_num_threads(1)
    torch.manual_seed(seed)

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

    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family=EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        # Context keys are normalized again on bank reads; the established
        # tolerance absorbs float32 round-off without merging these distinct
        # stream contexts.
        matching_tolerance=1e-4,
        capacity=STREAM_COUNT,
    )
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=8,
        context_width=CONTEXT_WIDTH,
    )
    single = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        admission_observations=ADMISSION_OBSERVATIONS,
        max_contexts=STREAM_COUNT,
        defer_admission=True,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    router = ExternalMultiStreamTransitionContextRouter(
        single,
        stream_key_width=STREAM_KEY_WIDTH,
    )

    staged_results: dict[int, object] = {}
    for stream_index in range(STREAM_COUNT):
        key = _stream_key(stream_index)
        router.observe(_observation(stream_index, 0), key)
        result = router.observe(_observation(stream_index, 1), key)
        if result.result.status != "staged":
            raise RuntimeError("stream did not stage an isolated candidate")
        staged_results[stream_index] = result

    candidate_digests_before = {
        stream_index: router.provisional_model_at(_stream_key(stream_index)).digest()
        for stream_index in range(STREAM_COUNT)
    }
    router.adaptation_step(
        staged_results[0],
        None,
        replay_evidence=False,
    )
    candidate_digests_after_first_update = {
        stream_index: router.provisional_model_at(_stream_key(stream_index)).digest()
        for stream_index in range(STREAM_COUNT)
    }
    untouched_candidates = all(
        candidate_digests_before[index] == candidate_digests_after_first_update[index]
        for index in range(1, STREAM_COUNT)
    )
    for stream_index in range(1, STREAM_COUNT):
        router.adaptation_step(
            staged_results[stream_index],
            None,
            replay_evidence=False,
        )

    promotions: list[dict[str, object]] = []
    for stream_index in range(STREAM_COUNT):
        key = _stream_key(stream_index)
        retained_digest = None if stream_index == 0 else bank.models[0].digest()
        receipt = router.promote_staged_candidate(
            key,
            _observation(stream_index, 2),
            lambda candidate, retained_digest=retained_digest: (
                retained_digest is None
                or candidate.models[0].digest() == retained_digest
            ),
            prediction_tolerance=PREDICTION_TOLERANCE,
        )
        if not receipt.accepted:
            raise RuntimeError(f"stream {stream_index} failed promotion: {receipt.reason}")
        promotions.append(
            {
                "stream_index": stream_index,
                "slot_id": receipt.slot_id,
                "heldout_error": receipt.heldout_error,
            }
        )

    route_statuses: list[str] = []
    route_slot_ids: list[int | None] = []
    for row_index in (0, 1):
        for stream_index in range(STREAM_COUNT):
            key = _stream_key(stream_index)
            router.observe(_observation(stream_index, row_index), key)
        for stream_index in range(STREAM_COUNT):
            key = _stream_key(stream_index)
            result = router.observe(_observation(stream_index, 1), key)
            route_statuses.append(result.result.status)
            route_slot_ids.append(result.result.stable_slot_id)

    payload = router.state_payload()
    restored = ExternalMultiStreamTransitionContextRouter.from_payload(payload)
    restored_route_statuses: list[str] = []
    restored_route_slot_ids: list[int | None] = []
    for stream_index in range(STREAM_COUNT):
        key = _stream_key(stream_index)
        restored.observe(_observation(stream_index, 0), key)
        result = restored.observe(_observation(stream_index, 1), key)
        restored_route_statuses.append(result.result.status)
        restored_route_slot_ids.append(result.result.stable_slot_id)

    corrupted = router.state_payload()
    corrupted["streams"][0]["stream_key"][0] += 0.01
    try:
        ExternalMultiStreamTransitionContextRouter.from_payload(corrupted)
    except ValueError as error:
        checksum_rejected = "checksum" in str(error)
    else:
        checksum_rejected = False

    return {
        "schema": "neural-computer.external-multi-stream-pressure-test.v1",
        "seed": seed,
        "stream_count": STREAM_COUNT,
        "shared_bank_context_count": router.bank.context_count,
        "promotions": promotions,
        "all_promoted": len(promotions) == STREAM_COUNT,
        "untouched_nonselected_candidates": untouched_candidates,
        "route_statuses": route_statuses,
        "route_slot_ids": route_slot_ids,
        "restored_route_statuses": restored_route_statuses,
        "restored_route_slot_ids": restored_route_slot_ids,
        "persistence_exact": router.digest() == restored.digest(),
        "checksum_rejected": checksum_rejected,
        "controller_frozen": all(
            not parameter.requires_grad for parameter in controller.parameters()
        ),
        "controller_digest": controller_digest,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "claim_boundary": (
            "bounded opaque stream binding over one shared factual bank; "
            "not learned identity formation or general continual learning"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()
    report = _run(args.seed)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
