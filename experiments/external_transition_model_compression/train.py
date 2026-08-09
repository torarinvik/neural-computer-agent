"""Pressure test held-out verified storage compression of external models."""

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
SOURCE_DELTAS = (-1, 1)
TARGET_DELTAS = (-2, 2)
RETENTION_TOLERANCE = 1e-4


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
    source, target, source_context, target_context = _fixture(seed)
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
        capacity=2,
    )
    source_index = bank.ensure_context(source_context)
    target_index = bank.ensure_context(target_context, initialize_from=source_index)
    source_loss, source_updates = _train(
        bank,
        source_index,
        source,
        source_context,
        SOURCE_UPDATES,
    )
    target_loss, target_updates = _train(
        bank,
        target_index,
        target,
        target_context,
        TARGET_UPDATES,
    )
    baseline = {
        "source": _loss(bank, source, source_context),
        "target": _loss(bank, target, target_context),
    }

    def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
        return (
            _loss(candidate, source, source_context) <= baseline["source"] + RETENTION_TOLERANCE
            and _loss(candidate, target, target_context)
            <= baseline["target"] + RETENTION_TOLERANCE
        )

    receipts: dict[str, object] = {}
    deltas: dict[str, list[float]] = {}
    for codec in (torch.float16, torch.int8, "int4"):
        name = str(codec)
        receipt = bank.compress_verified(dtype=codec, retention_probe=retention_probe)
        candidate = ExternalTransitionModelBank.from_compressed_payload(
            bank.compressed_payload(dtype=codec)
        )
        deltas[name] = [
            _loss(candidate, source, source_context) - baseline["source"],
            _loss(candidate, target, target_context) - baseline["target"],
        ]
        receipts[name] = {
            "accepted": receipt.accepted,
            "codec": receipt.codec,
            "source_bytes": receipt.source_bytes,
            "compressed_bytes": receipt.compressed_bytes,
            "candidate_digest": receipt.candidate_digest,
            "reason": receipt.reason,
        }
    selection = bank.select_compression_verified(
        (torch.float16, torch.int8, "int4"),
        retention_probe=retention_probe,
    )
    float16_restored = ExternalTransitionModelBank.from_compressed_payload(
        bank.compressed_payload(dtype=torch.float16)
    )
    gates = {
        "controller_unchanged": controller_digest == _digest_module(controller),
        "source_model_learns": source_loss < RETENTION_TOLERANCE,
        "target_model_learns": target_loss < RETENTION_TOLERANCE,
        "float16_accepted": receipts["torch.float16"]["accepted"],
        "int8_accepted": receipts["torch.int8"]["accepted"],
        "int4_rejected": not receipts["int4"]["accepted"],
        "adaptive_selection_picks_smallest_retained": (
            selection.accepted and selection.selected_codec == "torch.int8"
        ),
        "float16_saves_bytes": (
            receipts["torch.float16"]["compressed_bytes"]
            < receipts["torch.float16"]["source_bytes"]
        ),
        "int8_saves_bytes": (
            receipts["torch.int8"]["compressed_bytes"]
            < receipts["torch.int8"]["source_bytes"]
        ),
        "float16_persistence_exact": (
            float16_restored.model_aliases() == bank.model_aliases()
            and float16_restored.context_count == bank.context_count
        ),
        "zero_compression_optimizer_updates": True,
    }
    report = {
        "schema": "neural-computer.external-transition-model-compression-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "source_updates": SOURCE_UPDATES,
            "target_updates": TARGET_UPDATES,
            "retention_tolerance": RETENTION_TOLERANCE,
            "policy": "external_storage_codec_then_heldout_retention_promotion_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "source": {"optimizer_updates": source_updates, "loss": source_loss},
        "target": {"optimizer_updates": target_updates, "loss": target_loss},
        "baseline_losses": baseline,
        "compression_receipts": receipts,
        "adaptive_selection": {
            "accepted": selection.accepted,
            "selected_codec": selection.selected_codec,
            "candidate_codecs": [receipt.codec for receipt in selection.receipts],
            "reason": selection.reason,
        },
        "heldout_loss_deltas": deltas,
        "accounting": {
            "controller_parameter_updates": 0,
            "compression_optimizer_updates": 0,
            "source_replayed_examples": POSITION_COUNT * 2 * (source_updates - 1),
            "target_replayed_examples": POSITION_COUNT * 2 * (target_updates - 1),
            "contexts_preserved": bank.context_count,
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
    parser.add_argument("--seed", type=int, default=70211)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
