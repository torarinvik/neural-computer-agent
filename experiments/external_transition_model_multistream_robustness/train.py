"""Accounted missing, contradiction, drift, and eviction pressure test."""

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
BANK_CAPACITY = 2
MATCH_TOLERANCE = 0.03
DRIFT_OFFSET = torch.tensor([[2.0, -2.0]])


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


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _drift_observation(stream_index: int, row_index: int) -> ExternalTransitionObservation:
    source = _observation(stream_index, row_index)
    return ExternalTransitionObservation(
        state=source.state,
        intention=source.intention,
        next_state=source.next_state + DRIFT_OFFSET,
        confidence=source.confidence,
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
        matching_tolerance=1e-4,
        capacity=BANK_CAPACITY,
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
        match_tolerance=MATCH_TOLERANCE,
        continuation_tolerance=MATCH_TOLERANCE,
        admission_observations=2,
        max_contexts=BANK_CAPACITY,
        defer_admission=True,
        conflict_patience=2,
        candidate_model_families=(EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,),
    )
    router = ExternalMultiStreamTransitionContextRouter(
        single,
        stream_key_width=STREAM_KEY_WIDTH,
    )

    staged: dict[int, object] = {}
    for stream_index in (0, 1):
        key = _stream_key(stream_index)
        router.observe(_observation(stream_index, 0), key)
        result = router.observe(_observation(stream_index, 1), key)
        if result.result.status != "staged":
            raise RuntimeError("initial stream failed to stage")
        staged[stream_index] = result
        router.adaptation_step(result, None, replay_evidence=False)

    initial_promotions: list[int | None] = []
    for stream_index in (0, 1):
        receipt = router.promote_staged_candidate(
            _stream_key(stream_index),
            _observation(stream_index, 2),
            lambda _candidate: True,
            prediction_tolerance=0.2,
        )
        if not receipt.accepted:
            raise RuntimeError(f"initial stream promotion failed: {receipt.reason}")
        initial_promotions.append(receipt.slot_id)

    retained_stream_digest = bank.models[0].digest()
    bank_digest_before_conflict = bank.content_digest()

    # Stream 0 continues while stream 1 is temporarily absent. Its pending
    # state must not borrow stream 0's complete window.
    router.observe(_observation(0, 0), _stream_key(0))
    stream_zero_second = router.observe(_observation(0, 1), _stream_key(0))
    stream_one_missing_pending = router.observe(
        _observation(1, 0), _stream_key(1)
    )
    missing_isolated = (
        stream_zero_second.result.status == "continuation"
        and stream_one_missing_pending.result.status == "pending"
        and router.pending_observations(_stream_key(1)) == 1
        and router.pending_observations(_stream_key(0)) == 0
    )
    stream_one_late = router.observe(_observation(1, 1), _stream_key(1))

    # A contradictory bundle must not overwrite stream 1's committed fact.
    contradiction_first = router.observe(
        _drift_observation(1, 0), _stream_key(1)
    )
    contradiction_second = router.observe(
        _drift_observation(1, 1), _stream_key(1)
    )
    contradiction_safe = (
        contradiction_first.result.status == "pending"
        and contradiction_second.result.status == "conflict"
        and bank.content_digest() == bank_digest_before_conflict
        and router.provisional_candidate_count == 0
    )

    eviction = router.evict_verified_id(
        initial_promotions[1],
        lambda candidate: (
            candidate.models[0].digest() == retained_stream_digest
            and candidate.context_count in {1, 2}
        ),
    )
    if not eviction.accepted:
        raise RuntimeError("drift-slot eviction failed")

    router.observe(_drift_observation(1, 0), _stream_key(1))
    drift_staged = router.observe(_drift_observation(1, 1), _stream_key(1))
    if drift_staged.result.status != "staged":
        raise RuntimeError("drift did not stage after verified eviction")
    router.adaptation_step(drift_staged, None, replay_evidence=False)
    drift_promotion = router.promote_staged_candidate(
        _stream_key(1),
        _drift_observation(1, 2),
        lambda candidate: candidate.models[0].digest() == retained_stream_digest,
        prediction_tolerance=0.2,
    )
    if not drift_promotion.accepted:
        raise RuntimeError(f"drift promotion failed: {drift_promotion.reason}")

    stream_zero_revisit = router.observe(_observation(0, 0), _stream_key(0))
    stream_zero_revisit = router.observe(_observation(0, 1), _stream_key(0))
    stream_one_revisit = router.observe(_drift_observation(1, 0), _stream_key(1))
    stream_one_revisit = router.observe(_drift_observation(1, 1), _stream_key(1))
    retention_after_drift = (
        stream_zero_revisit.result.stable_slot_id == initial_promotions[0]
        and stream_one_revisit.result.stable_slot_id == drift_promotion.slot_id
        and bank.context_count == BANK_CAPACITY
        and bank.models[0].digest() == retained_stream_digest
    )

    payload = router.state_payload()
    restored = ExternalMultiStreamTransitionContextRouter.from_payload(payload)
    restored_zero = restored.observe(_observation(0, 0), _stream_key(0))
    restored_zero = restored.observe(_observation(0, 1), _stream_key(0))
    restored_one = restored.observe(_drift_observation(1, 0), _stream_key(1))
    restored_one = restored.observe(_drift_observation(1, 1), _stream_key(1))

    corrupted = router.state_payload()
    corrupted["streams"][0]["bound_slot_id"] = 999
    try:
        ExternalMultiStreamTransitionContextRouter.from_payload(corrupted)
    except ValueError as error:
        checksum_rejected = "checksum" in str(error) or "unknown" in str(error)
    else:
        checksum_rejected = False

    return {
        "schema": "neural-computer.external-multi-stream-robustness.v1",
        "seed": seed,
        "bank_capacity": BANK_CAPACITY,
        "initial_promotions": initial_promotions,
        "evicted_slot_id": eviction.evicted_slot_id,
        "drift_slot_id": drift_promotion.slot_id,
        "missing_isolated": missing_isolated,
        "late_stream_status": stream_one_late.result.status,
        "contradiction_statuses": [
            contradiction_first.result.status,
            contradiction_second.result.status,
        ],
        "contradiction_safe": contradiction_safe,
        "retention_after_drift": retention_after_drift,
        "restored_route_slot_ids": [
            restored_zero.result.stable_slot_id,
            restored_one.result.stable_slot_id,
        ],
        "persistence_exact": router.digest() == restored.digest(),
        "checksum_rejected": checksum_rejected,
        "controller_frozen": all(
            not parameter.requires_grad for parameter in controller.parameters()
        ),
        "controller_digest": controller_digest,
        "optimizer_updates": 0,
        "replayed_examples": 0,
        "claim_boundary": (
            "bounded missing/contradictory/drifting stream handling over one "
            "shared factual bank; not learned identity formation or general "
            "continual learning"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2201)
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
