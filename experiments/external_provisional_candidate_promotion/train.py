"""Two-seed copy-on-write candidate promotion audit."""

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

STATE_WIDTH = 2
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
HIDDEN_WIDTH = 16
ADMISSION_OBSERVATIONS = 2
TRAIN_ROWS = 4
HELDOUT_ROWS = 2
UPDATES = 300


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _observation(
    seed: int,
) -> tuple[
    ExternalTransitionObservation,
    ExternalTransitionObservation,
    ExternalTransitionObservation,
]:
    generator = torch.Generator().manual_seed(seed)
    state = torch.randn(6, STATE_WIDTH, generator=generator)
    intention = torch.randn(6, INTENTION_WIDTH, generator=generator)
    source_next = state + intention.expand(-1, STATE_WIDTH)
    target_next = state + 2.0 * intention.expand(-1, STATE_WIDTH)
    return (
        ExternalTransitionObservation(
            state=state,
            intention=intention,
            next_state=source_next,
            confidence=torch.ones(state.shape[0]),
        ),
        ExternalTransitionObservation(
            state=state[:TRAIN_ROWS],
            intention=intention[:TRAIN_ROWS],
            next_state=target_next[:TRAIN_ROWS],
            confidence=torch.ones(TRAIN_ROWS),
        ),
        ExternalTransitionObservation(
            state=state[TRAIN_ROWS:],
            intention=intention[TRAIN_ROWS:],
            next_state=target_next[TRAIN_ROWS:],
            confidence=torch.ones(HELDOUT_ROWS),
        ),
    )


def _rows(observation: ExternalTransitionObservation) -> list[ExternalTransitionObservation]:
    return [
        ExternalTransitionObservation(
            state=observation.state[index : index + 1],
            intention=observation.intention[index : index + 1],
            next_state=observation.next_state[index : index + 1],
            confidence=observation.confidence[index : index + 1]
            if observation.confidence is not None
            else None,
        )
        for index in range(observation.state.shape[0])
    ]


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    source_observation, training, heldout = _observation(seed)
    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=1,
        intention_width=INTENTION_WIDTH,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        capacity=2,
    )
    source_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    source_index = bank.ensure_context(source_context)
    source_context_batch = source_context.unsqueeze(0).expand(
        source_observation.state.shape[0], -1
    )
    source_optimizer = torch.optim.Adam(
        bank.models[source_index].parameters(), lr=0.02
    )
    for _ in range(300):
        bank.adaptation_step(
            source_observation,
            source_context_batch,
            source_optimizer,
        )
    source_digest = bank.models[source_index].digest()
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=HIDDEN_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=1e-8,
        admission_observations=ADMISSION_OBSERVATIONS,
        max_contexts=2,
        defer_admission=True,
    )
    content_before_stage = bank.content_digest()
    staged_results = []
    for row in _rows(training):
        result = router.observe(row)
        staged_results.append(result.status)
        if result.status == "pending":
            continue
        assert result.status == "staged"
        optimizer = torch.optim.Adam(router.provisional_model.parameters(), lr=0.02)
        for _ in range(100):
            router.adaptation_step(result, optimizer)
    content_before_promotion = bank.content_digest()
    committed_count_before_promotion = bank.context_count
    heldout_context = router._provisional_context
    if heldout_context is None:
        raise RuntimeError("candidate context was not staged")
    heldout_batch = heldout_context.unsqueeze(0).expand(HELDOUT_ROWS, -1)

    def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
        return (
            candidate.context_count == 2
            and candidate.models[0].digest() == source_digest
        )

    receipt = router.promote_staged_candidate(
        heldout,
        retention_probe,
        prediction_tolerance=0.2,
    )
    if receipt.accepted:
        heldout_error = float(
            bank.loss(heldout, heldout_batch).detach()
        )
    else:
        heldout_error = receipt.heldout_error
    gates = {
        "controller_unchanged": controller_digest == _digest(controller),
        "candidate_staged_without_bank_write": (
            staged_results.count("staged") >= 2
            and all(status in {"pending", "staged"} for status in staged_results)
            and content_before_stage == content_before_promotion
            and committed_count_before_promotion == 1
        ),
        "promotion_accepted": receipt.accepted,
        "heldout_prediction_passes": heldout_error <= 0.2,
        "source_slot_byte_stable": bank.models[0].digest() == source_digest,
        "retention_probe_preserved_source": receipt.accepted,
        "persistence_payload_boundary": (
            router.state_payload()["bank"]["sha256"] == bank.digest()
        ),
    }
    report = {
        "schema": "neural-computer.external-provisional-candidate-promotion-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "admission_observations": ADMISSION_OBSERVATIONS,
            "training_rows": TRAIN_ROWS,
            "heldout_rows": HELDOUT_ROWS,
            "candidate_updates": staged_results.count("staged") * 100,
            "policy": "none_copy_on_write_external_model_candidate_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "candidate": {
            "staged_statuses": staged_results,
            "heldout_error": heldout_error,
            "receipt": {
                "accepted": receipt.accepted,
                "slot_index": receipt.slot_index,
                "reason": receipt.reason,
            },
        },
        "accounting": {
            "controller_parameter_updates": 0,
            "committed_bank_updates_before_promotion": 0,
            "provisional_candidate_updates": staged_results.count("staged") * 100,
            "old_slot_replay": 0,
        },
        "digests": {
            "controller": controller_digest,
            "source": source_digest,
            "bank_before_stage": content_before_stage,
            "bank_before_promotion": content_before_promotion,
            "bank_after_promotion": bank.content_digest(),
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=70611)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
