"""Exercise identity assignment from learned pixel-event histories.

Rendered RGB frames are decomposed into separately bound learned event tracks.
An external, versioned artifact scores action-conditioned dependence in those
event histories and hands only opaque slot evidence to the live gate. The
feeder supplies synthetic opaque action features so this remains a diagnostic
of the seam, not a claim that identity has been learned from a task holdout.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch

from neural_computer import (
    AmodalEvent,
    AmodalEventCollection,
    ExternalCausalIdentityArtifact,
    ExternalCausalIdentityAssignment,
)
from neural_computer.promotion import sha256_file

from .current_symbol_acquire import FRONTEND_SEED
from .live_identity_assignment_pixel import _machine
from .object_scene import render_markers, scene_slots
from .rendered_environment import RenderedBrainWorkshopEncoders

EXPERIMENT_ID = "brainworkshop-live-identity-assignment-learned-2026-08-16"
LEARNED_IDENTITY_SCHEMA = "neural-computer.live-identity-assignment-learned.v1"
DEVELOPMENT_SEED = 43
EVENT_WIDTH = 4
STATE_WIDTH = 12
INTENTION_WIDTH = 2
FRAME_SIZE = 36
STEPS = 8
IDENTITY_MARGIN = 0.15
POSITION_SEQUENCE = (0, 1, 2, 1, 0, 2, 1, 0)


def _positions(steps: int) -> tuple[int, ...]:
    return tuple(POSITION_SEQUENCE[index % len(POSITION_SEQUENCE)] for index in range(steps))


def _opaque_action_history(positions: tuple[int, ...]) -> torch.Tensor:
    transitions = [
        [1.0, 0.0] if right > left else [0.0, 1.0]
        for left, right in pairwise(positions)
    ]
    return torch.tensor([transitions], dtype=torch.float32)


def _pixel_events(
    encoders: RenderedBrainWorkshopEncoders,
    frame: torch.Tensor,
) -> AmodalEventCollection:
    payload = encoders.vision(torch.stack(scene_slots(frame)))
    return AmodalEventCollection.from_events(
        [AmodalEvent(payload=payload[index : index + 1]) for index in range(payload.shape[0])],
        width=encoders.event_width,
    )


def run_learned_identity(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    steps: int = STEPS,
) -> dict[str, Any]:
    if steps < 4:
        raise ValueError("learned identity diagnostic needs at least four frames")
    started = time.perf_counter()
    torch.manual_seed(int(seed))
    encoders = RenderedBrainWorkshopEncoders.seeded(
        EVENT_WIDTH,
        source_key_width=4,
        seed=FRONTEND_SEED,
    )
    frontend_digest_before = encoders.digest()
    repository = Path(__file__).parents[2]
    bank_path = repository / "artifacts/checkpoints" / "AgentBrain.bank"
    bank_digest_before = sha256_file(bank_path) if bank_path.is_file() else None

    positions = _positions(int(steps))
    action_history = _opaque_action_history(positions)
    artifact = ExternalCausalIdentityArtifact(minimum_history=4)
    goal_candidates = torch.zeros(2, STATE_WIDTH)
    goal_candidates[0, 0] = 1.0
    goal_candidates[1, 0] = -1.0
    learned_machine = _machine(
        identity_assignment=ExternalCausalIdentityAssignment(margin=IDENTITY_MARGIN),
        goal_state_candidates=goal_candidates,
    )
    passive_machine = _machine(
        identity_assignment=ExternalCausalIdentityAssignment(margin=IDENTITY_MARGIN),
        goal_state_candidates=goal_candidates,
    )
    constant_action_machine = _machine(
        identity_assignment=ExternalCausalIdentityAssignment(margin=IDENTITY_MARGIN),
        goal_state_candidates=goal_candidates,
    )

    payload_history: list[torch.Tensor] = []
    learned_evidence: list[list[float]] = []
    constant_action_evidence: list[list[float]] = []
    learned_emissions = 0
    passive_abstentions = 0
    constant_action_abstentions = 0
    selected_slots: list[int | None] = []
    for step, position in enumerate(positions):
        frame = render_markers((position, 7), size=FRAME_SIZE)
        events = _pixel_events(encoders, frame)
        payload_history.append(events.payload.squeeze(0).detach())
        event_history = torch.stack(payload_history).unsqueeze(0)
        if event_history.shape[1] >= artifact.minimum_history:
            evidence = artifact.evidence(event_history, action_history[:, :step])
            constant_evidence = artifact.evidence(
                event_history, torch.ones_like(action_history[:, :step])
            )
        else:
            evidence = torch.zeros(1, 2)
            constant_evidence = torch.zeros(1, 2)
        learned_evidence.append([float(value) for value in evidence[0]])
        constant_action_evidence.append(
            [float(value) for value in constant_evidence[0]]
        )
        learned_emissions += len(
            learned_machine.tick(
                events,
                (),
                now=float(step),
                elapsed=1.0,
                identity_evidence=evidence,
            )
        )
        passive_abstentions += int(
            not passive_machine.tick(
                events,
                (),
                now=float(step),
                elapsed=1.0,
                identity_evidence=torch.zeros(1, 2),
            )
        )
        constant_action_abstentions += int(
            not constant_action_machine.tick(
                events,
                (),
                now=float(step),
                elapsed=1.0,
                identity_evidence=constant_evidence,
            )
        )
        assignment = learned_machine.last_identity_assignment
        selected_slots.append(
            None
            if assignment is None or bool(assignment.abstained[0])
            else int(assignment.selected_slot[0].item())
        )

    frontend_digest_after = encoders.digest()
    bank_digest_after = sha256_file(bank_path) if bank_path.is_file() else None
    report = {
        "schema": LEARNED_IDENTITY_SCHEMA,
        "artifact": artifact.configuration(),
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "steps": steps,
        "positions_are_feeder_only": True,
        "selected_slots": selected_slots,
        "learned_evidence": learned_evidence,
        "constant_action_evidence": constant_action_evidence,
        "learned_emissions": learned_emissions,
        "passive_abstentions": passive_abstentions,
        "constant_action_abstentions": constant_action_abstentions,
        "frontend_digest_before": frontend_digest_before,
        "frontend_digest_after": frontend_digest_after,
        "frontend_unchanged": frontend_digest_before == frontend_digest_after,
        "bank_digest_before": bank_digest_before,
        "bank_digest_after": bank_digest_after,
        "bank_unchanged": bank_digest_before == bank_digest_after,
        "unique_verifier_bits": 0,
        "unique_logical_lifetimes": 0,
        "optimizer_updates": 0,
        "replayed_examples": steps * 3,
        "wall_seconds": time.perf_counter() - started,
        "claim_status": "learned_event_identity_interface_diagnostic",
        "claim_boundary": (
            "Frozen rendered pixels and bound learned event tracks feed an "
            "external action-conditioned evidence artifact. Opaque action "
            "features are synthetic feeder inputs; there is no identity "
            "holdout, verifier-bit, transfer, promotion, or bank-admission "
            "claim."
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "live_identity_assignment_learned.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    repository = Path(__file__).parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=repository
        / "session_records"
        / "brainworkshop_live_identity_assignment_learned_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--steps", type=int, default=STEPS)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_learned_identity(
                arguments.output, seed=arguments.seed, steps=arguments.steps
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
