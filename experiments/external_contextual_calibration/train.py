"""Pressure test context-isolated online calibration without target replay."""

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
    ExternalContextualEvidenceCalibrator,
    ExternalTransitionEvidenceEvaluator,
)

STATE_WIDTH = 8
CONTEXT_WIDTH = 3
HIDDEN_WIDTH = 32
SOURCE_ROWS = 512
SOURCE_UPDATES = 500
TARGET_UPDATES = 256
SOURCE_NOISE = 0.08
TARGET_NOISE = 0.25
VERIFIER_THRESHOLD = 0.10


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _train_evaluator(seed: int) -> tuple[ExternalTransitionEvidenceEvaluator, float]:
    torch.manual_seed(seed)
    prediction = F.normalize(
        torch.randn(SOURCE_ROWS, STATE_WIDTH), dim=-1
    )
    positive = prediction + torch.randn_like(prediction) * SOURCE_NOISE
    negative = F.normalize(torch.randn_like(prediction), dim=-1)
    evaluator = ExternalTransitionEvidenceEvaluator(
        STATE_WIDTH,
        hidden_width=HIDDEN_WIDTH,
    )
    optimizer = torch.optim.Adam(evaluator.parameters(), lr=0.02)
    outcomes = torch.cat(
        (torch.ones(SOURCE_ROWS), torch.zeros(SOURCE_ROWS))
    )
    final_loss = float("inf")
    for _update in range(SOURCE_UPDATES):
        optimizer.zero_grad()
        loss = evaluator.loss(
            torch.cat((prediction, prediction)),
            torch.cat((positive, negative)),
            outcomes,
            torch.ones(SOURCE_ROWS * 2),
        )
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    return evaluator, final_loss


def _batch(seed: int, noise: float, count: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(seed)
    prediction = F.normalize(
        torch.randn(count, STATE_WIDTH, generator=generator), dim=-1
    )
    observed = prediction + torch.randn(
        prediction.shape,
        generator=generator,
    ) * noise
    outcomes = (
        (prediction - observed).square().mean(dim=-1) < VERIFIER_THRESHOLD
    ).to(torch.float32)
    return prediction, observed, outcomes


def _accuracy(
    calibrator: ExternalContextualEvidenceCalibrator,
    batch: tuple[torch.Tensor, ...],
    context: torch.Tensor,
) -> float:
    prediction, observed, outcomes = batch
    contexts = context.unsqueeze(0).expand(prediction.shape[0], -1)
    with torch.no_grad():
        probabilities = torch.sigmoid(
            calibrator(prediction, observed, torch.ones(prediction.shape[0]), contexts)
        )
    return float(((probabilities >= 0.5) == outcomes.bool()).float().mean())


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    evaluator, evaluator_loss = _train_evaluator(seed + 1000)
    evaluator_digest = evaluator.digest()

    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=2,
        intention_width=4,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_digest = _digest_module(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    calibrator = ExternalContextualEvidenceCalibrator(
        evaluator,
        CONTEXT_WIDTH,
        prior_strength=0.001,
    )
    source_context = torch.tensor([1.0, 0.0, 0.0])
    target_context = torch.tensor([0.0, 1.0, 0.0])
    source_index = calibrator.ensure_context(source_context)
    target_index = calibrator.ensure_context(target_context)
    source_slot_digest = calibrator.calibrators[source_index].digest()
    target_optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in calibrator.calibrators[target_index].parameters()
            if parameter.requires_grad
        ],
        lr=0.05,
    )

    source_holdout = _batch(seed + 2000, SOURCE_NOISE, SOURCE_ROWS)
    target_holdout = _batch(seed + 3000, TARGET_NOISE, SOURCE_ROWS)
    source_before = _accuracy(calibrator, source_holdout, source_context)
    target_before = _accuracy(calibrator, target_holdout, target_context)
    wrong_context_before = _accuracy(
        calibrator,
        target_holdout,
        source_context,
    )

    target_losses: list[float] = []
    for update in range(TARGET_UPDATES):
        prediction, observed, outcomes = _batch(
            seed + 5000 + update,
            TARGET_NOISE,
            1,
        )
        target_losses.append(
            calibrator.calibration_step(
                prediction,
                observed,
                outcomes,
                target_context.unsqueeze(0),
                target_optimizer,
                torch.ones(1),
            )
        )

    source_after = _accuracy(calibrator, source_holdout, source_context)
    target_after = _accuracy(calibrator, target_holdout, target_context)
    wrong_context_after = _accuracy(calibrator, target_holdout, source_context)
    restored = ExternalContextualEvidenceCalibrator.from_payload(
        calibrator.payload(),
        evaluator=evaluator,
    )
    persisted_target = _accuracy(restored, target_holdout, target_context)
    gates = {
        "target_improves": target_after >= target_before + 0.20,
        "source_retained": source_after >= source_before - 0.01,
        "source_slot_untouched": (
            calibrator.calibrators[source_index].digest() == source_slot_digest
        ),
        "base_evaluator_frozen": evaluator.digest() == evaluator_digest,
        "controller_frozen": controller_digest == _digest_module(controller),
        "context_is_required": wrong_context_after < target_after - 0.10,
        "external_state_persists": abs(persisted_target - target_after) < 1e-9,
    }
    report = {
        "schema": "neural-computer.external-contextual-calibration-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "source_noise": SOURCE_NOISE,
            "target_noise": TARGET_NOISE,
            "verifier_threshold": VERIFIER_THRESHOLD,
            "source_updates": SOURCE_UPDATES,
            "target_updates": TARGET_UPDATES,
            "policy": "append_only_context_isolated_scalar_calibration_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "source_before": source_before,
            "source_after": source_after,
            "target_before": target_before,
            "target_after": target_after,
            "target_gain": target_after - target_before,
            "wrong_context_before": wrong_context_before,
            "wrong_context_after": wrong_context_after,
            "persisted_target": persisted_target,
            "target_final_online_loss": target_losses[-1],
        },
        "evaluator": {
            "pretraining_updates": SOURCE_UPDATES,
            "pretraining_rows": SOURCE_ROWS * 2,
            "replayed_pretraining_rows": SOURCE_ROWS * 2 * (SOURCE_UPDATES - 1),
            "final_loss": evaluator_loss,
            "digest": evaluator_digest,
        },
        "accounting": {
            "target_unique_verifier_bits": TARGET_UPDATES,
            "target_optimizer_updates": TARGET_UPDATES,
            "target_replayed_examples": 0,
            "target_context_slots": calibrator.context_count,
            "controller_parameter_updates": 0,
        },
        "digests": {
            "controller": controller_digest,
            "evaluator": evaluator_digest,
            "contextual_calibrator": calibrator.digest(),
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=69801)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
