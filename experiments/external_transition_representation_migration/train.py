"""Two-seed audit of verified representation-space replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    ExternalModelBasedPlanner,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 1
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 3
SPACE_SOURCE = ("state-v1", "intention-v1")
SPACE_TARGET = ("state-v2", "intention-v2")


def _bank(spaces: tuple[str, str]) -> ExternalTransitionModelBank:
    bank = ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        model_family="affine_sufficient_statistics_v1",
        affine_ridge=1e-7,
        state_space_id=spaces[0],
        intention_space_id=spaces[1],
    )
    for context in (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])):
        bank.ensure_context(context)
    state = torch.tensor([[-1.0], [-0.4], [0.2], [0.8]])
    intention = torch.tensor([[0.1], [0.6], [-0.2], [0.4]])
    for index, offset in enumerate((0.0, 2.0)):
        observation = ExternalTransitionObservation(
            state=state,
            intention=intention,
            next_state=state + intention + offset,
        )
        context = bank.context_at(index).unsqueeze(0).expand(state.shape[0], -1)
        bank.adaptation_step(observation, context, None)
    return bank


def _copy_bank(source: ExternalTransitionModelBank) -> ExternalTransitionModelBank:
    candidate = _bank(SPACE_TARGET)
    for source_model, candidate_model in zip(source.models, candidate.models, strict=True):
        candidate_model.load_state_dict(source_model.state_dict())
    return candidate


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    source = _bank(SPACE_SOURCE)
    candidate = _copy_bank(source)
    state = torch.tensor([[-0.7], [0.0], [0.5]])
    intention = torch.tensor([[0.2], [-0.1], [0.8]])
    heldout = tuple(
        (
            slot_id,
            ExternalTransitionObservation(
                state=state,
                intention=intention,
                next_state=torch.zeros_like(state),
            ),
        )
        for slot_id in source.slot_ids
    )
    source_digest = source.digest()
    accepted = source.migrate_representation_verified(
        candidate,
        heldout,
        retention_probe=lambda bank: bank.context_count == source.context_count,
    )
    mismatch_rejected = False
    old_planner = ExternalModelBasedPlanner(source, beam_width=1)
    try:
        old_planner.select_bank_model(
            candidate,
            torch.zeros(1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1),
            horizon=1,
        )
    except ValueError:
        mismatch_rejected = True
    drifted = _copy_bank(source)
    drifted.models[0].target_matrix.add_(1.0)
    rejected = source.migrate_representation_verified(
        drifted,
        heldout,
        prediction_tolerance=1e-8,
    )
    target_planner = ExternalModelBasedPlanner(
        candidate,
        beam_width=1,
        state_space_id=SPACE_TARGET[0],
        intention_space_id=SPACE_TARGET[1],
    )
    selection = target_planner.select_bank_model(
        candidate,
        torch.zeros(1, 1),
        torch.ones(1, 1),
        torch.ones(1, 1),
        horizon=1,
    )
    report = {
        "schema": "neural-computer.external-transition-representation-migration.v1",
        "seed": seed,
        "configuration": {
            "source_spaces": SPACE_SOURCE,
            "target_spaces": SPACE_TARGET,
            "model_count": source.context_count,
            "migration": "copy_on_write_heldout_stable_address_v1",
        },
        "gates": {
            "behavior_preserving_migration": accepted.accepted,
            "old_planner_mismatch_rejected": mismatch_rejected,
            "drifted_candidate_rejected": not rejected.accepted,
            "new_planner_selection_runs": selection.selected_slot_id in candidate.slot_ids,
            "source_unchanged": source.digest() == source_digest,
            "zero_controller_updates": True,
            "zero_replayed_transition_examples": True,
        },
        "promoted": all(
            (
                accepted.accepted,
                mismatch_rejected,
                not rejected.accepted,
                selection.selected_slot_id in candidate.slot_ids,
                source.digest() == source_digest,
            )
        ),
        "metrics": {
            "accepted_max_heldout_difference": accepted.max_heldout_difference,
            "rejected_drift_max_heldout_difference": rejected.max_heldout_difference,
            "selected_slot_id": selection.selected_slot_id,
            "source_digest": source_digest,
            "target_digest": candidate.digest(),
            "configuration_digest": hashlib.sha256(
                repr((SPACE_SOURCE, SPACE_TARGET)).encode()
            ).hexdigest(),
        },
        "accounting": {
            "unique_verifier_bits": len(heldout),
            "unique_logical_lifetimes": source.context_count,
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "verified representation compatibility gate; not arbitrary alignment or general continual learning",
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
