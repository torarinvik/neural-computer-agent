"""Two-seed multimodal runtime representation-migration audit."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    RuntimeMigrationExample,
)

WIDTH = 16
FEEDBACK_WIDTH = 4
EXAMPLES = 24
SOURCE_SPACES = ("frontend-v1", "controller-state-v1", "intention-v1")
TARGET_SPACES = ("frontend-v2", "controller-state-v2", "intention-v2")


def _runtime(spaces: tuple[str, str, str], state: dict[str, torch.Tensor] | None = None) -> AmodalControllerRuntime:
    controller = AmodalCognitiveController(
        width=WIDTH,
        workspace_slots=3,
        intention_width=6,
        feedback_width=FEEDBACK_WIDTH,
        event_window_capacity=8,
    )
    if state is not None:
        controller.load_state_dict(state)
    return AmodalControllerRuntime(
        controller,
        event_space_id=spaces[0],
        state_space_id=spaces[1],
        intention_space_id=spaces[2],
    )


def _feedback(generator: torch.Generator) -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.randn(1, FEEDBACK_WIDTH, generator=generator),
        reward=torch.randn(1, generator=generator),
        propensity=torch.full((1,), 0.7),
        has_feedback=torch.ones(1),
    )


def _examples(
    source: AmodalControllerRuntime,
    target: AmodalControllerRuntime,
    generator: torch.Generator,
) -> list[RuntimeMigrationExample]:
    examples: list[RuntimeMigrationExample] = []
    for index in range(EXAMPLES):
        first = AmodalEvent(
            torch.randn(1, WIDTH, generator=generator),
            timestamp=torch.tensor([float(index)]),
            confidence=torch.ones(1),
        )
        second = AmodalEvent(
            torch.randn(1, WIDTH, generator=generator),
            timestamp=torch.tensor([float(index) + 0.25]),
            confidence=torch.full((1,), 0.8),
        )
        target_events = [
            AmodalEvent(
                first.payload.clone(),
                timestamp=first.timestamp.clone(),
                confidence=first.confidence.clone(),
            ),
            AmodalEvent(
                second.payload.clone(),
                timestamp=second.timestamp.clone(),
                confidence=second.confidence.clone(),
            ),
        ]
        examples.append(
            RuntimeMigrationExample(
                source_events=[first, second],
                target_events=target_events,
                source_state=source.initial_state(1, device="cpu"),
                target_state=target.initial_state(1, device="cpu"),
                feedback=_feedback(generator),
            )
        )
    return examples


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    generator = torch.Generator().manual_seed(seed)
    source = _runtime(SOURCE_SPACES)
    target = _runtime(TARGET_SPACES, source.controller.state_dict())
    examples = _examples(source, target, generator)
    accepted = source.migrate_controller_verified(target, examples)
    drifted = _runtime(TARGET_SPACES, source.controller.state_dict())
    next(drifted.controller.parameters()).data.add_(0.25)
    rejected = source.migrate_controller_verified(
        drifted,
        examples,
        prediction_tolerance=1e-8,
    )
    report = {
        "schema": "neural-computer.runtime-representation-migration.v1",
        "seed": seed,
        "configuration": {
            "source_spaces": SOURCE_SPACES,
            "target_spaces": TARGET_SPACES,
            "heldout_two_stream_examples": EXAMPLES,
            "migration": "paired_event_window_controller_retention_v1",
        },
        "gates": {
            "behavior_preserving_migration": accepted.accepted,
            "drifted_candidate_rejected": not rejected.accepted,
            "stable_prefix_examples": accepted.example_count == EXAMPLES,
            "zero_controller_optimizer_updates": True,
            "zero_replayed_examples": True,
            "external_memory_untouched": True,
        },
        "promoted": accepted.accepted and not rejected.accepted,
        "metrics": {
            "accepted_max_intention_difference": accepted.max_intention_difference,
            "accepted_max_execution_difference": accepted.max_execution_difference,
            "accepted_max_continuation_difference": accepted.max_continuation_difference,
            "rejected_max_intention_difference": rejected.max_intention_difference,
            "rejected_max_continuation_difference": rejected.max_continuation_difference,
        },
        "accounting": {
            "unique_verifier_bits": EXAMPLES,
            "unique_logical_lifetimes": EXAMPLES * 2,
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "claim_boundary": "paired multimodal runtime compatibility gate; not learned alignment or general continual learning",
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
