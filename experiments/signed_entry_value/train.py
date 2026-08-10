"""Audit signed external-entry polarity transfer with a frozen value model."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from neural_computer import ExternalSignedEntryValueModel

STATE_WIDTH = 4
ENTRY_WIDTH = 2
HIDDEN_WIDTH = 24
BATCH_SIZE = 64
SOURCE_UPDATES = 512
DIAGNOSTIC_BATCHES = 3


class SignedEntryVerifier:
    """Private scalar verifier for a polarity-dependent opaque value."""

    state_width = STATE_WIDTH
    entry_width = ENTRY_WIDTH

    @staticmethod
    def _salience(state: torch.Tensor) -> torch.Tensor:
        return 0.25 + 0.75 * torch.sigmoid(
            1.5 * state[:, 0] - 0.5 * state[:, 1]
        )

    def outcome(self, state: torch.Tensor, entry: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2 or state.shape[1] != self.state_width:
            raise ValueError("verifier state has the wrong shape")
        if entry.ndim != 2 or entry.shape[1] != self.entry_width:
            raise ValueError("verifier entry has the wrong shape")
        if state.shape[0] != entry.shape[0]:
            raise ValueError("verifier state and entry batches differ")
        polarity = entry[:, 0].clamp(-1.0, 1.0)
        return torch.sigmoid(4.0 * self._salience(state) * polarity)


class UnfactorizedValueModel(nn.Module):
    """Matched control that may learn arbitrary state-entry interactions."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(STATE_WIDTH + ENTRY_WIDTH, HIDDEN_WIDTH),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTH, 1),
        )

    def forward(self, state: torch.Tensor, entry: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((state, entry), dim=-1)).squeeze(-1)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _source_batch(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.randn(BATCH_SIZE, STATE_WIDTH, generator=generator)
    entry = torch.empty(BATCH_SIZE, ENTRY_WIDTH)
    entry[:, 0] = 0.2 + 0.8 * torch.rand(BATCH_SIZE, generator=generator)
    entry[:, 1] = torch.randn(BATCH_SIZE, generator=generator)
    return state, entry


def _target_batch(
    generator: torch.Generator,
    *,
    batch_size: int = BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.randn(batch_size, STATE_WIDTH, generator=generator)
    entry = torch.empty(batch_size, ENTRY_WIDTH)
    entry[:, 0] = -(0.2 + 0.8 * torch.rand(batch_size, generator=generator))
    entry[:, 1] = torch.randn(batch_size, generator=generator)
    return state, entry


def _mixed_batch(
    generator: torch.Generator,
    *,
    batch_size: int = BATCH_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.randn(batch_size, STATE_WIDTH, generator=generator)
    signs = torch.where(
        torch.arange(batch_size) % 2 == 0,
        torch.ones(batch_size),
        -torch.ones(batch_size),
    )
    entry = torch.empty(batch_size, ENTRY_WIDTH)
    entry[:, 0] = signs * (0.2 + 0.8 * torch.rand(batch_size, generator=generator))
    entry[:, 1] = torch.randn(batch_size, generator=generator)
    return state, entry


def _mean_error(
    model: nn.Module,
    verifier: SignedEntryVerifier,
    state: torch.Tensor,
    entry: torch.Tensor,
) -> float:
    with torch.no_grad():
        prediction = model(state, entry).sigmoid()
        target = verifier.outcome(state, entry)
    return float((prediction - target).square().mean())


def _intervention_error(
    model: nn.Module,
    verifier: SignedEntryVerifier,
    state: torch.Tensor,
    intervened_entry: torch.Tensor,
    original_entry: torch.Tensor,
) -> float:
    """Score an input intervention against the unchanged verifier outcome."""

    with torch.no_grad():
        prediction = model(state, intervened_entry).sigmoid()
        target = verifier.outcome(state, original_entry)
    return float((prediction - target).square().mean())


def _inference_latency_ms(
    model: nn.Module,
    state: torch.Tensor,
    entry: torch.Tensor,
    *,
    repeats: int = 32,
) -> float:
    """Measure prediction latency for one diagnostic batch."""

    model.eval()
    with torch.no_grad():
        for _ in range(4):
            model(state, entry)
        started_at = time.perf_counter()
        for _ in range(repeats):
            model(state, entry)
    return (time.perf_counter() - started_at) * 1_000.0 / repeats


def _train_source(
    model: nn.Module,
    verifier: SignedEntryVerifier,
    *,
    seed: int,
) -> int:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed + 1_000_001)
    model.train()
    for _ in range(SOURCE_UPDATES):
        state, entry = _source_batch(generator)
        outcome = verifier.outcome(state, entry)
        if isinstance(model, ExternalSignedEntryValueModel):
            loss = model.loss(state, entry, outcome)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(
                model(state, entry), outcome
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    return SOURCE_UPDATES


def run(seed: int, report_out: Path) -> dict[str, Any]:
    started_at = time.perf_counter()
    seed_everything(seed)
    verifier = SignedEntryVerifier()
    signed = ExternalSignedEntryValueModel(
        STATE_WIDTH,
        ENTRY_WIDTH,
        hidden_width=HIDDEN_WIDTH,
    )
    baseline = UnfactorizedValueModel()
    optimizer_updates = _train_source(signed, verifier, seed=seed)
    _train_source(baseline, verifier, seed=seed)

    signed_before_target = {
        name: value.detach().clone() for name, value in signed.state_dict().items()
    }
    baseline_before_target = {
        name: value.detach().clone() for name, value in baseline.state_dict().items()
    }
    source_generator = torch.Generator().manual_seed(seed + 2_000_001)
    target_generator = torch.Generator().manual_seed(seed + 3_000_001)
    source_state, source_entry = _source_batch(source_generator)
    target_state, target_entry = _target_batch(target_generator)
    zero_entry = torch.zeros_like(target_entry)
    mixed_state, mixed_entry = _mixed_batch(target_generator)
    shuffled_entry = mixed_entry[
        torch.randperm(mixed_entry.shape[0], generator=target_generator)
    ]

    signed_source_error = _mean_error(signed, verifier, source_state, source_entry)
    signed_target_error = _mean_error(signed, verifier, target_state, target_entry)
    baseline_source_error = _mean_error(baseline, verifier, source_state, source_entry)
    baseline_target_error = _mean_error(baseline, verifier, target_state, target_entry)
    signed_zero_error = _mean_error(signed, verifier, target_state, zero_entry)
    signed_mixed_error = _mean_error(signed, verifier, mixed_state, mixed_entry)
    signed_shuffled_error = _intervention_error(
        signed,
        verifier,
        mixed_state,
        shuffled_entry,
        mixed_entry,
    )
    mean_inference_latency_ms = _inference_latency_ms(
        signed,
        target_state,
        target_entry,
    )
    with torch.no_grad():
        signed_target_probability = signed(target_state, target_entry).sigmoid()
        signed_reversed_probability = signed(target_state, -target_entry).sigmoid()
        oddness_error = float(
            (signed(target_state, target_entry) + signed(target_state, -target_entry))
            .abs()
            .max()
        )
        zero_probability_error = float(
            (signed(target_state, zero_entry).sigmoid() - 0.5).abs().max()
        )
        target_reversal_error = float(
            (signed_reversed_probability - (1.0 - signed_target_probability)).abs().max()
        )
    restored = ExternalSignedEntryValueModel.from_payload(signed.state_payload())
    exact_persistence = restored.digest() == signed.digest() and torch.allclose(
        restored(target_state, target_entry), signed(target_state, target_entry)
    )
    signed_unchanged = all(
        torch.equal(value, signed_before_target[name])
        for name, value in signed.state_dict().items()
    )
    baseline_unchanged = all(
        torch.equal(value, baseline_before_target[name])
        for name, value in baseline.state_dict().items()
    )
    gates = {
        "source_mastery": signed_source_error < 0.01,
        "reversed_entry_transfer": signed_target_error < 0.01,
        "signed_beats_unfactorized_control": signed_target_error < baseline_target_error,
        "entry_shuffle_is_causal": signed_mixed_error + 0.01 < signed_shuffled_error,
        "zero_entry_is_neutral": zero_probability_error < 1e-6,
        "exact_odd_polarity": oddness_error < 1e-6,
        "reversed_probability_complement": target_reversal_error < 1e-6,
        "signed_model_unchanged_during_target_audit": signed_unchanged,
        "baseline_unchanged_during_target_audit": baseline_unchanged,
        "exact_persistence": exact_persistence,
        "zero_target_optimizer_updates": True,
        "zero_replayed_examples": True,
    }
    promoted = all(gates.values())
    report = {
        "experiment": "signed_entry_value",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "entry_width": ENTRY_WIDTH,
            "hidden_width": HIDDEN_WIDTH,
            "source_entry": "positive_first_coordinate_with_distractor",
            "target_entry": "negative_first_coordinate_with_distractor",
            "target_training": "none",
        },
        "metrics": {
            "signed_source_error": signed_source_error,
            "signed_target_error": signed_target_error,
            "baseline_source_error": baseline_source_error,
            "baseline_target_error": baseline_target_error,
            "signed_zero_entry_error": signed_zero_error,
            "signed_mixed_error": signed_mixed_error,
            "signed_shuffled_entry_error": signed_shuffled_error,
            "oddness_error": oddness_error,
            "zero_probability_error": zero_probability_error,
            "target_reversal_error": target_reversal_error,
            "mean_inference_latency_ms": mean_inference_latency_ms,
            "signed_source_retention_error": signed_source_error,
            "source_updates": optimizer_updates,
            "target_updates": 0,
        },
        "accounting": {
            "unique_verifier_bits": SOURCE_UPDATES * BATCH_SIZE,
            "unique_logical_lifetimes": SOURCE_UPDATES * BATCH_SIZE,
            "optimizer_updates": optimizer_updates * 2,
            "target_optimizer_updates": 0,
            "replayed_examples": 0,
            "controller_optimizer_updates": 0,
            "diagnostic_target_lifetimes": DIAGNOSTIC_BATCHES * BATCH_SIZE,
            "wall_time_seconds": time.perf_counter() - started_at,
            "stable_bits_to_threshold": None,
            "retention_on_mastered_primitives": "source_error_after_frozen_target_audit",
            "transfer_ratio_against_fresh_learner": None,
        },
        "gates": gates,
        "promoted": promoted,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed, args.report_out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
