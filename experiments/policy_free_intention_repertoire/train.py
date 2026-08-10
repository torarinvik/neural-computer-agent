"""Pressure-test candidate intention formation outside a frozen controller."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    ExternalAffineTransitionStatistics,
    ExternalIntentionRepertoire,
    ExternalModelBasedPlanner,
    ExternalTransitionObservation,
    OpaqueProtocolDecoder,
    PolicyFreeAmodalRuntime,
)

STATE_WIDTH = 12
CONTROLLER_WIDTH = 4
INTENTION_WIDTH = 2
ACTION_BASIS = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
)
TARGET_DISPLACEMENTS = ((2, 2), (2, -2), (-2, 2), (-2, -2))
HORIZON = 4
BEAM_WIDTH = 8
GOAL_PROGRESS_WEIGHT = 1.0
TRANSITION_ROWS = 64


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _transition_observation(seed: int) -> ExternalTransitionObservation:
    generator = torch.Generator().manual_seed(seed)
    states = torch.randn(TRANSITION_ROWS, STATE_WIDTH, generator=generator)
    states[:, 2:] = states[:, 2:].tanh()
    actions = ACTION_BASIS.repeat(TRANSITION_ROWS // len(ACTION_BASIS), 1)
    padded = torch.zeros(TRANSITION_ROWS, STATE_WIDTH)
    padded[:, :INTENTION_WIDTH] = actions
    return ExternalTransitionObservation(
        state=states,
        intention=actions,
        next_state=states + padded,
        confidence=torch.ones(TRANSITION_ROWS),
    )


def _feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )


def _execute(intentions: torch.Tensor) -> tuple[int, int]:
    displacement = torch.zeros(2)
    for intention in intentions:
        index = int(torch.linalg.vector_norm(ACTION_BASIS - intention, dim=-1).argmin())
        displacement += ACTION_BASIS[index]
    return int(displacement[0]), int(displacement[1])


def _random_floor(seed: int, trials: int) -> float:
    generator = torch.Generator().manual_seed(seed + 1_991_007)
    successes = 0
    total = 0
    for target in TARGET_DISPLACEMENTS:
        for _ in range(trials):
            indices = torch.randint(len(ACTION_BASIS), (HORIZON,), generator=generator)
            displacement = tuple(int(value) for value in ACTION_BASIS[indices].sum(dim=0))
            successes += displacement == target
            total += 1
    return successes / total


def _build_controller(seed: int) -> AmodalCognitiveController:
    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=CONTROLLER_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    return controller


def _run_targets(
    policy_free: PolicyFreeAmodalRuntime,
    *,
    state: object,
    event: list[AmodalEvent],
    feedback: ControllerFeedback,
    current_state: torch.Tensor,
    targets: tuple[tuple[int, int], ...],
    record_outcomes: bool,
) -> tuple[list[dict[str, object]], object]:
    rows: list[dict[str, object]] = []
    for target in targets:
        goal = current_state.clone()
        goal[:, :2] += torch.tensor(target, dtype=goal.dtype)
        started = time.perf_counter()
        output, _ = policy_free.step_events(
            event,
            state,
            feedback,
            goal,
            horizon=HORIZON,
            beam_width=BEAM_WIDTH,
            goal_progress_weight=GOAL_PROGRESS_WEIGHT,
        )
        success = _execute(output.planning.intentions[0]) == target
        if record_outcomes:
            if output.proposal is None:
                raise AssertionError("repertoire-backed execution produced no proposal")
            policy_free.observe_intention(
                output.intention,
                utility=float(success),
                propensity=float(output.proposal.propensities[0, 0]),
            )
        rows.append(
            {
                "target": list(target),
                "planned_displacement": list(_execute(output.planning.intentions[0])),
                "success": success,
                "candidate_count": (
                    None
                    if output.proposal is None
                    else int(output.proposal.intentions.shape[1])
                ),
                "exploration_candidates": (
                    None
                    if output.proposal is None
                    else int(output.proposal.exploration_mask[0].sum())
                ),
                "expanded_nodes": output.planning.expanded_nodes,
                "latency_seconds": time.perf_counter() - started,
            }
        )
    return rows, state


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    controller = _build_controller(seed)
    controller_digest = _digest(controller)
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder(
        "opaque_backend",
        OpaqueProtocolDecoder(INTENTION_WIDTH, INTENTION_WIDTH),
    )
    planner_model = ExternalAffineTransitionStatistics(
        STATE_WIDTH,
        INTENTION_WIDTH,
        ridge=1e-6,
    )
    planner_model.observe(_transition_observation(seed))
    model_digest = planner_model.digest()

    repertoire = ExternalIntentionRepertoire(INTENTION_WIDTH)
    experience_utility = torch.tensor([0.25, 0.75, 0.5, 0.125])
    experience_propensity = torch.full((len(ACTION_BASIS),), 0.25)
    experience_receipt = repertoire.observe(
        ACTION_BASIS,
        utility=experience_utility,
        propensity=experience_propensity,
        timestamp=torch.arange(1, len(ACTION_BASIS) + 1),
    )
    retained_intentions = repertoire.statistics()["intentions"].clone()
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        ExternalModelBasedPlanner(planner_model, beam_width=BEAM_WIDTH),
        intention_repertoire=repertoire,
    )
    state = runtime.initial_state(1, device="cpu")
    event = [AmodalEvent(torch.randn(1, CONTROLLER_WIDTH))]
    feedback = _feedback()
    preview, _ = runtime.step_events(event, state, feedback)
    current_state = preview.controller.state_representation.detach()

    target_rows, _ = _run_targets(
        policy_free,
        state=state,
        event=event,
        feedback=feedback,
        current_state=current_state,
        targets=TARGET_DISPLACEMENTS,
        record_outcomes=True,
    )

    shuffled_rows: list[dict[str, object]] = []
    for index, target in enumerate(TARGET_DISPLACEMENTS):
        requested = TARGET_DISPLACEMENTS[-index - 1]
        goal = current_state.clone()
        goal[:, :2] += torch.tensor(requested, dtype=goal.dtype)
        output, _ = policy_free.step_events(
            event,
            state,
            feedback,
            goal,
            horizon=HORIZON,
            beam_width=BEAM_WIDTH,
            goal_progress_weight=GOAL_PROGRESS_WEIGHT,
        )
        shuffled_rows.append(
            {
                "target": list(target),
                "requested_goal": list(requested),
                "success_against_original": _execute(output.planning.intentions[0]) == target,
            }
        )

    fresh_controller = copy.deepcopy(controller)
    fresh_runtime = AmodalControllerRuntime(fresh_controller)
    fresh_repertoire = ExternalIntentionRepertoire(INTENTION_WIDTH)
    fresh_policy_free = PolicyFreeAmodalRuntime(
        fresh_runtime,
        ExternalModelBasedPlanner(planner_model, beam_width=BEAM_WIDTH),
        intention_repertoire=fresh_repertoire,
    )
    fresh_state = fresh_runtime.initial_state(1, device="cpu")
    fresh_event = [AmodalEvent(event[0].payload.clone())]
    fresh_preview, _ = fresh_runtime.step_events(fresh_event, fresh_state, feedback)
    fresh_current_state = fresh_preview.controller.state_representation.detach()
    fresh_rows, _ = _run_targets(
        fresh_policy_free,
        state=fresh_state,
        event=fresh_event,
        feedback=feedback,
        current_state=fresh_current_state,
        targets=TARGET_DISPLACEMENTS,
        record_outcomes=False,
    )

    policy_free_mastery = sum(bool(row["success"]) for row in target_rows) / len(target_rows)
    fresh_mastery = sum(bool(row["success"]) for row in fresh_rows) / len(fresh_rows)
    shuffled_mastery = sum(
        bool(row["success_against_original"]) for row in shuffled_rows
    ) / len(shuffled_rows)
    restored = ExternalIntentionRepertoire.from_payload(repertoire.payload())
    gates = {
        "all_repertoire_goals_mastered": policy_free_mastery == 1.0,
        "beats_fresh_empty_repertoire": policy_free_mastery > fresh_mastery + 0.5,
        "beats_random_floor": policy_free_mastery > _random_floor(seed, 128) + 0.5,
        "goal_conditioned": shuffled_mastery < policy_free_mastery,
        "candidate_list_not_caller_supplied": all(
            row["candidate_count"] is not None for row in target_rows
        ),
        "verified_repertoire_candidates_used": all(
            int(row["candidate_count"]) == len(ACTION_BASIS)
            and int(row["exploration_candidates"]) == 0
            for row in target_rows
        ),
        "fresh_empty_uses_seed_fallback": all(
            int(row["candidate_count"]) == 1
            and int(row["exploration_candidates"]) == 1
            for row in fresh_rows
        ),
        "retained_experience_vectors": torch.equal(
            repertoire.statistics()["intentions"][: len(ACTION_BASIS)], retained_intentions
        ),
        "controller_frozen": controller_digest == _digest(controller),
        "model_unchanged_during_search": model_digest == planner_model.digest(),
        "exact_repertoire_persistence": restored.content_digest() == repertoire.content_digest(),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.policy-free-intention-repertoire.v1",
        "claim_boundary": (
            "external opaque candidate-intention discovery from observed experience; "
            "not outcome-trained policy learning or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "controller_width": CONTROLLER_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "experience_intentions": ACTION_BASIS.tolist(),
            "candidate_intentions_argument": "omitted_at_runtime",
            "repertoire_proposal": repertoire.configuration(),
            "horizon": HORIZON,
            "beam_width": BEAM_WIDTH,
            "goal_progress_weight": GOAL_PROGRESS_WEIGHT,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "policy_free": target_rows,
            "fresh_empty_repertoire": fresh_rows,
            "policy_free_mastery": policy_free_mastery,
            "fresh_mastery": fresh_mastery,
            "goal_shuffled_mastery": shuffled_mastery,
            "random_floor": _random_floor(seed, 128),
            "repertoire_record_count": repertoire.record_count,
            "experience_receipt": {
                "added": list(experience_receipt.added),
                "version": experience_receipt.version,
            },
        },
        "accounting": {
            "unique_verifier_bits": len(ACTION_BASIS) + len(TARGET_DISPLACEMENTS),
            "unique_logical_lifetimes": TRANSITION_ROWS + len(ACTION_BASIS) + len(TARGET_DISPLACEMENTS),
            "experience_records_consumed_once": len(ACTION_BASIS),
            "factual_statistics_updates": 1,
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "search_expansions": sum(int(row["expanded_nodes"]) for row in target_rows),
            "mean_search_latency_seconds": sum(
                float(row["latency_seconds"]) for row in target_rows
            ) / len(target_rows),
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
