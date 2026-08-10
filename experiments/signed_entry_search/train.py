"""Promoted audit of signed external entries inside factual beam search."""

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

from neural_computer import ExternalModelBasedPlanner, ExternalSignedEntryValueModel

STATE_WIDTH = 1
ENTRY_WIDTH = 1
HIDDEN_WIDTH = 16
BATCH_SIZE = 64
SOURCE_UPDATES = 512


class SignedEntryVerifier:
    """Private scalar verifier for a polarity-dependent external value."""

    @staticmethod
    def _salience(state: torch.Tensor) -> torch.Tensor:
        return 0.25 + 0.75 * torch.sigmoid(1.5 * state[:, 0])

    def outcome(self, state: torch.Tensor, entry: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(
            4.0
            * self._salience(state)
            * entry[:, 0].clamp(-1.0, 1.0)
        )


class AdditiveFactualModel(nn.Module):
    """Frozen opaque transition fact used only for search."""

    state_width = STATE_WIDTH
    intention_width = 1

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        return state + intention


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _source_batch(
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = torch.randn(BATCH_SIZE, STATE_WIDTH, generator=generator)
    entry = 0.2 + 0.8 * torch.rand(
        BATCH_SIZE,
        ENTRY_WIDTH,
        generator=generator,
    )
    return state, entry


def _train_source(
    model: ExternalSignedEntryValueModel,
    verifier: SignedEntryVerifier,
    *,
    seed: int,
) -> int:
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    generator = torch.Generator().manual_seed(seed + 1_000_001)
    for _ in range(SOURCE_UPDATES):
        state, entry = _source_batch(generator)
        outcome = verifier.outcome(state, entry)
        optimizer.zero_grad(set_to_none=True)
        model.loss(state, entry, outcome).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
    return SOURCE_UPDATES


def _mean_error(
    model: ExternalSignedEntryValueModel,
    verifier: SignedEntryVerifier,
    state: torch.Tensor,
    entry: torch.Tensor,
) -> float:
    with torch.no_grad():
        prediction = model(state, entry).sigmoid()
        target = verifier.outcome(state, entry)
    return float((prediction - target).square().mean())


def _search(
    planner: ExternalModelBasedPlanner,
    entries: torch.Tensor | None,
) -> torch.Tensor:
    result = planner.plan(
        torch.zeros(1, STATE_WIDTH),
        torch.zeros(1, STATE_WIDTH),
        torch.tensor([[-1.0], [1.0]]),
        candidate_entries=entries,
        entry_value_weight=1.0 if entries is not None else 0.0,
        horizon=1,
    )
    return result.intentions[0, 0].detach().clone()


def _search_latency_ms(
    planner: ExternalModelBasedPlanner,
    entries: torch.Tensor,
    *,
    repeats: int = 16,
) -> float:
    for _ in range(3):
        _search(planner, entries)
    started_at = time.perf_counter()
    for _ in range(repeats):
        _search(planner, entries)
    return (time.perf_counter() - started_at) * 1_000.0 / repeats


def run(seed: int, report_out: Path) -> dict[str, Any]:
    started_at = time.perf_counter()
    seed_everything(seed)
    verifier = SignedEntryVerifier()
    entry_model = ExternalSignedEntryValueModel(
        STATE_WIDTH,
        ENTRY_WIDTH,
        hidden_width=HIDDEN_WIDTH,
    )
    transition_model = AdditiveFactualModel()
    source_updates = _train_source(entry_model, verifier, seed=seed)
    entry_model_before_search = entry_model.digest()
    transition_model_before_search = {
        name: value.detach().clone()
        for name, value in transition_model.state_dict().items()
    }
    source_generator = torch.Generator().manual_seed(seed + 2_000_001)
    source_state, source_entry = _source_batch(source_generator)
    signed_source_error = _mean_error(
        entry_model,
        verifier,
        source_state,
        source_entry,
    )

    positive_regime = torch.tensor([[-1.0], [1.0]])
    reversed_regime = -positive_regime
    signed_planner = ExternalModelBasedPlanner(
        transition_model,
        beam_width=2,
        entry_value_model=entry_model,
    )
    baseline_planner = ExternalModelBasedPlanner(transition_model, beam_width=2)
    signed_positive = _search(signed_planner, positive_regime)
    signed_reversed = _search(signed_planner, reversed_regime)
    baseline_positive = _search(baseline_planner, None)
    baseline_reversed = _search(baseline_planner, None)
    expected_positive = torch.tensor([1.0])
    expected_reversed = torch.tensor([-1.0])
    signed_search_latency_ms = _search_latency_ms(
        signed_planner,
        positive_regime,
    )
    restored = ExternalSignedEntryValueModel.from_payload(entry_model.state_payload())
    exact_persistence = restored.digest() == entry_model.digest()
    entry_model_unchanged = entry_model.digest() == entry_model_before_search
    transition_model_unchanged = all(
        torch.equal(value, transition_model.state_dict()[name])
        for name, value in transition_model_before_search.items()
    )
    gates = {
        "source_mastery": signed_source_error < 0.01,
        "positive_regime_search": torch.equal(signed_positive, expected_positive),
        "reversed_regime_search": torch.equal(signed_reversed, expected_reversed),
        "entry_polarity_changes_search": not torch.equal(
            signed_positive,
            signed_reversed,
        ),
        "baseline_is_polarity_insensitive": torch.equal(
            baseline_positive,
            baseline_reversed,
        ),
        "signed_model_unchanged_during_search": entry_model_unchanged,
        "transition_model_unchanged_during_search": transition_model_unchanged,
        "exact_persistence": exact_persistence,
        "zero_target_optimizer_updates": True,
        "zero_replayed_examples": True,
    }
    report = {
        "experiment": "signed_entry_search",
        "seed": seed,
        "configuration": {
            "source_training": "positive_entries_only",
            "search": "factual_beam_search_with_external_entry_value_v1",
            "candidate_intentions": "opaque_minus_one_plus_one",
            "positive_regime_entries": positive_regime.tolist(),
            "reversed_regime_entries": reversed_regime.tolist(),
            "target_training": "none",
        },
        "metrics": {
            "signed_source_error": signed_source_error,
            "signed_search_latency_ms": signed_search_latency_ms,
            "signed_positive_intention": signed_positive.tolist(),
            "signed_reversed_intention": signed_reversed.tolist(),
            "baseline_positive_intention": baseline_positive.tolist(),
            "baseline_reversed_intention": baseline_reversed.tolist(),
            "source_updates": source_updates,
            "target_updates": 0,
        },
        "accounting": {
            "unique_verifier_bits": SOURCE_UPDATES * BATCH_SIZE,
            "unique_logical_lifetimes": SOURCE_UPDATES * BATCH_SIZE,
            "optimizer_updates": source_updates,
            "target_optimizer_updates": 0,
            "replayed_examples": 0,
            "controller_optimizer_updates": 0,
            "diagnostic_target_lifetimes": 2,
            "wall_time_seconds": time.perf_counter() - started_at,
            "stable_bits_to_threshold": None,
            "retention_on_mastered_primitives": "frozen_source_model_digest",
            "transfer_ratio_against_fresh_learner": None,
        },
        "gates": gates,
        "promoted": all(gates.values()),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=9201)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.seed, args.report_out)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
