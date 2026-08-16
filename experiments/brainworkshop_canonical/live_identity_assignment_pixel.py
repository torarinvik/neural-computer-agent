"""Exercise the identity seam with rendered pixels and a frozen frontend.

This is a transport/control diagnostic, not an identity-learning claim. The
scene is rendered as RGB pixels, each marker is encoded separately into a
learned event tensor, and the live machine sees only the resulting
``AmodalEventCollection``. A caller-owned evidence vector drives the external
identity gate; a tie must abstain without emitting a guessed action.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    ExternalCausalIdentityAssignment,
    ExternalModelBasedPlanner,
    PolicyFreeAmodalLiveMachine,
    PolicyFreeAmodalRuntime,
)
from neural_computer.promotion import sha256_file

from .current_symbol_acquire import FRONTEND_SEED
from .object_scene import PLACE_COUNT, render_markers, scene_slots
from .rendered_environment import RenderedBrainWorkshopEncoders

EXPERIMENT_ID = "brainworkshop-live-identity-assignment-pixel-2026-08-16"
PIXEL_IDENTITY_SCHEMA = "neural-computer.live-identity-assignment-pixel.v1"
DEVELOPMENT_SEED = 41
EVENT_WIDTH = 4
STATE_WIDTH = 12
INTENTION_WIDTH = 2
FRAME_SIZE = 36
STEPS = 8


class _AdditiveFactualModel(nn.Module):
    state_width = STATE_WIDTH
    intention_width = INTENTION_WIDTH

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, :INTENTION_WIDTH] += intention
        return result


class _Decision:
    def __init__(self, action: torch.Tensor, propensity: torch.Tensor) -> None:
        self.action = action
        self.propensity = propensity


class _OpaqueDecoder:
    intention_width = INTENTION_WIDTH

    def decide(self, intention, *, sample: bool = True):
        del sample
        return _Decision(
            intention.payload[:, :INTENTION_WIDTH].detach(),
            torch.ones(intention.payload.shape[0]),
        )


def _pixel_events(
    encoders: RenderedBrainWorkshopEncoders,
    frame: torch.Tensor,
) -> AmodalEventCollection:
    slots = scene_slots(frame)
    payload = encoders.vision(torch.stack(slots))
    events = [
        AmodalEvent(payload=payload[index : index + 1])
        for index in range(payload.shape[0])
    ]
    return AmodalEventCollection.from_events(events, width=encoders.event_width)


def _machine(
    *,
    identity_assignment: ExternalCausalIdentityAssignment | None,
    goal_state_candidates: torch.Tensor | None,
) -> PolicyFreeAmodalLiveMachine:
    controller = AmodalCognitiveController(
        width=EVENT_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=INTENTION_WIDTH,
        event_window_capacity=4,
    )
    runtime = AmodalControllerRuntime(controller)
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(_AdditiveFactualModel(), beam_width=4),
    )
    goal = torch.zeros(1, STATE_WIDTH)
    return PolicyFreeAmodalLiveMachine(
        policy_free,
        _OpaqueDecoder(),
        goal_state=goal,
        goal_state_candidates=goal_state_candidates,
        identity_assignment=identity_assignment,
        candidate_intentions=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
        ),
        output_key="pixel-opaque",
        sample=False,
    )


def run_pixel_identity(
    output_directory: Path,
    *,
    seed: int = DEVELOPMENT_SEED,
    steps: int = STEPS,
) -> dict[str, Any]:
    if steps < 2:
        raise ValueError("pixel identity diagnostic needs at least two frames")
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
    goal_candidates = torch.zeros(2, STATE_WIDTH)
    goal_candidates[0, 0] = 1.0
    goal_candidates[1, 0] = -1.0
    assignment_machine = _machine(
        identity_assignment=ExternalCausalIdentityAssignment(margin=0.2),
        goal_state_candidates=goal_candidates,
    )
    no_assignment_machine = _machine(
        identity_assignment=None,
        goal_state_candidates=None,
    )
    ambiguous_machine = _machine(
        identity_assignment=ExternalCausalIdentityAssignment(margin=0.2),
        goal_state_candidates=goal_candidates,
    )
    assignment_emissions = 0
    no_assignment_emissions = 0
    ambiguous_abstentions = 0
    event_counts: list[int] = []
    for step in range(int(steps)):
        frame = render_markers(
            ((step + 1) % PLACE_COUNT, (step + 4) % PLACE_COUNT),
            size=FRAME_SIZE,
        )
        events = _pixel_events(encoders, frame)
        event_counts.append(int(events.payload.shape[1]))
        assignment_emissions += len(
            assignment_machine.tick(
                events,
                (),
                now=float(step),
                elapsed=1.0,
                identity_evidence=torch.tensor([[3.0, 0.0]]),
            )
        )
        no_assignment_emissions += len(
            no_assignment_machine.tick(events, (), now=float(step), elapsed=1.0)
        )
        ambiguous_abstentions += int(
            not ambiguous_machine.tick(
                events,
                (),
                now=float(step),
                elapsed=1.0,
                identity_evidence=torch.tensor([[1.0, 1.0]]),
            )
        )
    frontend_digest_after = encoders.digest()
    bank_digest_after = sha256_file(bank_path) if bank_path.is_file() else None
    report = {
        "schema": PIXEL_IDENTITY_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "seed": seed,
        "steps": steps,
        "event_counts": event_counts,
        "assignment_emissions": assignment_emissions,
        "no_assignment_emissions": no_assignment_emissions,
        "ambiguous_abstentions": ambiguous_abstentions,
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
        "claim_status": "interface_pixel_diagnostic",
        "claim_boundary": (
            "Rendered-pixel transport and abstention only; caller evidence is "
            "not learned here, so no identity-learning, holdout, promotion, "
            "or curated-bank admission claim."
        ),
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "live_identity_assignment_pixel.json").write_text(
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
        / "brainworkshop_live_identity_assignment_pixel_2026-08-16",
    )
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--steps", type=int, default=STEPS)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_pixel_identity(arguments.output, seed=arguments.seed, steps=arguments.steps),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
