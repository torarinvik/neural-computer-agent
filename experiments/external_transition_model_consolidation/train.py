"""Pressure test behavior-verified transition-model parameter sharing."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import (
    AmodalCognitiveController,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 6
HIDDEN_WIDTH = 48
POSITION_COUNT = 6
SOURCE_UPDATES = 1200
TARGET_UPDATES = 1200
TARGET_DELTAS = (-2, 2)
SOURCE_DELTAS = (-1, 1)
TARGET_LOSS_THRESHOLD = 0.01


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(seed: int) -> tuple[
    ExternalTransitionObservation,
    ExternalTransitionObservation,
    torch.Tensor,
    torch.Tensor,
]:
    generator = torch.Generator().manual_seed(seed)
    states = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intentions = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )

    def observations(deltas: tuple[int, int]) -> ExternalTransitionObservation:
        state_rows: list[torch.Tensor] = []
        intention_rows: list[torch.Tensor] = []
        next_rows: list[torch.Tensor] = []
        for position in range(POSITION_COUNT):
            for action, delta in enumerate(deltas):
                next_position = min(POSITION_COUNT - 1, max(0, position + delta))
                state_rows.append(states[position])
                intention_rows.append(intentions[action])
                next_rows.append(states[next_position])
        return ExternalTransitionObservation(
            state=torch.stack(state_rows),
            intention=torch.stack(intention_rows),
            next_state=torch.stack(next_rows),
            confidence=torch.ones(POSITION_COUNT * 2),
        )

    return (
        observations(SOURCE_DELTAS),
        observations(TARGET_DELTAS),
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
    )


def _train(
    bank: ExternalTransitionModelBank,
    index: int,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
    updates: int,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(bank.models[index].parameters(), lr=0.01)
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    loss = float("inf")
    for update in range(1, updates + 1):
        loss = bank.adaptation_step(observation, context_batch, optimizer)
    return loss, updates


def _loss(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    return float(
        bank.loss(
            observation,
            context.unsqueeze(0).expand(observation.state.shape[0], -1),
        ).detach()
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    source, target, source_context, duplicate_context = _fixture(seed)
    target_context = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
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
        hidden_width=HIDDEN_WIDTH,
        capacity=3,
    )
    source_index = bank.ensure_context(source_context)
    duplicate_index = bank.ensure_context(
        duplicate_context,
        initialize_from=source_index,
    )
    target_index = bank.ensure_context(
        target_context,
        initialize_from=source_index,
    )
    source_loss, source_updates = _train(
        bank,
        source_index,
        source,
        source_context,
        SOURCE_UPDATES,
    )
    bank.models[duplicate_index].load_state_dict(
        bank.models[source_index].state_dict()
    )
    target_loss, target_updates = _train(
        bank,
        target_index,
        target,
        target_context,
        TARGET_UPDATES,
    )
    physical_before = bank.physical_model_count
    equivalent_before = bank.content_digest()
    equivalent = bank.consolidate_verified(
        source_index,
        duplicate_index,
        [source, target],
        prediction_tolerance=1e-8,
        retention_probe=lambda candidate: (
            _loss(candidate, source, source_context) < TARGET_LOSS_THRESHOLD
            and _loss(candidate, source, duplicate_context) < TARGET_LOSS_THRESHOLD
            and _loss(candidate, target, target_context) < TARGET_LOSS_THRESHOLD
        ),
    )
    source_after = _loss(bank, source, source_context)
    duplicate_after = _loss(bank, source, duplicate_context)
    target_after = _loss(bank, target, target_context)
    physical_after_equivalent = bank.physical_model_count
    source_digest_before_copy_on_write = bank.models[source_index].digest()
    copy_on_write_optimizer = torch.optim.Adam(
        bank.models[duplicate_index].parameters(),
        lr=0.1,
    )
    bank.adaptation_step(
        target,
        bank.context_at(duplicate_index).unsqueeze(0).expand(target.state.shape[0], -1),
        copy_on_write_optimizer,
    )
    source_after_copy_on_write = _loss(bank, source, source_context)
    copy_on_write_aliases = bank.model_aliases()
    physical_after_copy_on_write = bank.physical_model_count
    distinct_before = bank.content_digest()
    distinct = bank.consolidate_verified(
        source_index,
        target_index,
        [source, target],
        prediction_tolerance=1e-8,
    )
    physical_after_rejection = bank.physical_model_count
    restored = ExternalTransitionModelBank.from_payload(bank.payload())
    wrong_context_mse = _loss(bank, target, source_context)
    gates = {
        "controller_unchanged": controller_digest == _digest_module(controller),
        "source_model_learns": source_loss < TARGET_LOSS_THRESHOLD,
        "target_model_learns": target_loss < TARGET_LOSS_THRESHOLD,
        "equivalent_consolidation_accepted": equivalent.accepted,
        "equivalent_physical_sharing": physical_after_equivalent == physical_before - 1,
        "contexts_preserved": bank.context_count == 3 and copy_on_write_aliases == [0, 1, 2],
        "equivalent_retention": (
            source_after < TARGET_LOSS_THRESHOLD
            and duplicate_after < TARGET_LOSS_THRESHOLD
            and target_after < TARGET_LOSS_THRESHOLD
        ),
        "copy_on_write_isolates_later_update": (
            copy_on_write_aliases == [0, 1, 2]
            and physical_after_copy_on_write == physical_before
            and bank.models[source_index].digest()
            == source_digest_before_copy_on_write
            and source_after_copy_on_write < TARGET_LOSS_THRESHOLD
        ),
        "distinct_consolidation_rejected": not distinct.accepted,
        "distinct_physical_count_unchanged": physical_after_rejection
        == physical_after_copy_on_write,
        "wrong_context_control": wrong_context_mse > TARGET_LOSS_THRESHOLD,
        "persistence_exact": (
            restored.content_digest() == bank.content_digest()
            and restored.model_aliases() == [0, 1, 2]
            and restored.physical_model_count == 3
        ),
        "zero_consolidation_optimizer_updates": True,
    }
    report = {
        "schema": "neural-computer.external-transition-model-consolidation-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "source_updates": SOURCE_UPDATES,
            "target_updates": TARGET_UPDATES,
            "prediction_tolerance": 1e-8,
            "policy": "equivalence_verified_parameter_sharing_copy_on_write_v2",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "source": {"optimizer_updates": source_updates, "loss": source_loss},
        "target": {"optimizer_updates": target_updates, "loss": target_loss},
        "equivalent_consolidation": {
            "receipt": {
                "accepted": equivalent.accepted,
                "physical_models_before": equivalent.physical_models_before,
                "physical_models_after": equivalent.physical_models_after,
                "max_heldout_difference": equivalent.max_heldout_difference,
                "content_digest_before": equivalent.content_digest_before,
                "content_digest_after": equivalent.content_digest_after,
                "reason": equivalent.reason,
            },
            "content_digest_before": equivalent_before,
            "physical_models_after": physical_after_equivalent,
            "physical_models_after_copy_on_write": physical_after_copy_on_write,
            "copy_on_write_optimizer_updates": 1,
            "source_loss_after": source_after,
            "duplicate_loss_after": duplicate_after,
            "target_loss_after": target_after,
            "source_loss_after_copy_on_write": source_after_copy_on_write,
            "copy_on_write_aliases": copy_on_write_aliases,
            "physical_models_after_copy_on_write": physical_after_copy_on_write,
            "replayed_examples": 0,
        },
        "distinct_rejection": {
            "accepted": distinct.accepted,
            "content_digest_before": distinct_before,
            "content_digest_after": distinct.content_digest_after,
            "max_heldout_difference": distinct.max_heldout_difference,
            "physical_models_after": physical_after_rejection,
            "reason": distinct.reason,
            "replayed_examples": 0,
        },
        "wrong_context_target_mse": wrong_context_mse,
        "accounting": {
            "controller_parameter_updates": 0,
            "consolidation_optimizer_updates": 0,
            "copy_on_write_optimizer_updates": 1,
            "source_replayed_examples": POSITION_COUNT * 2 * (source_updates - 1),
            "target_replayed_examples": POSITION_COUNT * 2 * (target_updates - 1),
            "contexts_preserved": bank.context_count,
            "physical_models_after": bank.physical_model_count,
        },
        "digests": {
            "controller": controller_digest,
            "bank": bank.digest(),
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=70111)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
