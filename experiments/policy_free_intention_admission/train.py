"""Verify held-out, copy-on-write acquisition of new intention content."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.policy_free_intention_repertoire.train import (
    ACTION_BASIS,
    BEAM_WIDTH,
    CONTROLLER_WIDTH,
    GOAL_PROGRESS_WEIGHT,
    HORIZON,
    INTENTION_WIDTH,
    STATE_WIDTH,
    TRANSITION_ROWS,
    _build_controller,
    _feedback,
    _transition_observation,
)
from neural_computer import (
    AmodalControllerRuntime,
    AmodalEvent,
    ExternalAffineTransitionStatistics,
    ExternalIntentionRepertoire,
    ExternalModelBasedPlanner,
    OpaqueProtocolDecoder,
    PolicyFreeAmodalRuntime,
)

NEW_INTENTION = torch.tensor([0.5, 0.5])
REJECTED_INTENTION = torch.tensor([0.5, -0.5])
DIAGONAL_TARGET = (1.5, 1.5)


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _continuous_displacement(intentions: torch.Tensor) -> tuple[float, float]:
    displacement = intentions[:, :2].sum(dim=0)
    return float(displacement[0]), float(displacement[1])


def _goal(current_state: torch.Tensor) -> torch.Tensor:
    goal = current_state.clone()
    goal[:, :2] += torch.tensor(DIAGONAL_TARGET, dtype=goal.dtype)
    return goal


def _run(
    policy_free: PolicyFreeAmodalRuntime,
    *,
    state,
    event: list[AmodalEvent],
    feedback,
    goal: torch.Tensor,
) -> tuple[tuple[float, float], object, int]:
    output, next_state = policy_free.step_events(
        event,
        state,
        feedback,
        goal,
        horizon=HORIZON - 1,
        beam_width=BEAM_WIDTH,
        goal_progress_weight=GOAL_PROGRESS_WEIGHT,
    )
    return (
        _continuous_displacement(output.planning.intentions[0]),
        next_state,
        int(output.planning.expanded_nodes),
    )


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
    repertoire.observe(ACTION_BASIS)
    retained_before = repertoire.statistics()["intentions"].clone()
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
    goal = _goal(current_state)

    before_displacement, _, before_expansions = _run(
        policy_free,
        state=state,
        event=event,
        feedback=feedback,
        goal=goal,
    )
    before_success = torch.allclose(
        torch.tensor(before_displacement), torch.tensor(DIAGONAL_TARGET), atol=1e-6
    )

    heldout_generator = torch.Generator().manual_seed(seed + 19_001)
    heldout_state = torch.randn(16, STATE_WIDTH, generator=heldout_generator)
    heldout_padding = torch.zeros(16, STATE_WIDTH)
    heldout_padding[:, :INTENTION_WIDTH] = NEW_INTENTION
    heldout_next_state = heldout_state + heldout_padding
    heldout_prediction = planner_model(
        heldout_state,
        NEW_INTENTION.unsqueeze(0).expand(heldout_state.shape[0], -1),
    )
    heldout_error = float(
        (heldout_prediction - heldout_next_state).square().mean().detach()
    )

    def verify_new_intention(candidate: ExternalIntentionRepertoire) -> bool:
        retained = candidate.statistics()["intentions"][: len(ACTION_BASIS)]
        if not torch.equal(retained, retained_before):
            return False
        if heldout_error > 1e-4:
            return False
        candidate.observe(NEW_INTENTION, utility=1.0, propensity=1.0)
        return True

    admission = repertoire.admit_verified(NEW_INTENTION, verify_new_intention)
    retained_after_admission = repertoire.statistics()["intentions"][: len(ACTION_BASIS)]
    post_displacement, _, post_expansions = _run(
        policy_free,
        state=state,
        event=event,
        feedback=feedback,
        goal=goal,
    )
    post_success = torch.allclose(
        torch.tensor(post_displacement), torch.tensor(DIAGONAL_TARGET), atol=1e-6
    )

    rejected_digest = repertoire.content_digest()
    rejected_target_padding = torch.zeros(16, STATE_WIDTH)
    rejected_target_padding[:, :INTENTION_WIDTH] = torch.tensor([0.2, -0.2])
    rejected_target = heldout_state + rejected_target_padding
    rejected_error = float(
        (
            planner_model(
                heldout_state,
                REJECTED_INTENTION.unsqueeze(0).expand(heldout_state.shape[0], -1),
            )
            - rejected_target
        )
        .square()
        .mean()
        .detach()
    )
    rejected = repertoire.admit_verified(
        REJECTED_INTENTION,
        lambda _candidate: rejected_error <= 1e-4,
    )
    restored = ExternalIntentionRepertoire.from_payload(repertoire.payload())
    gates = {
        "pre_admission_goal_not_mastered": not before_success,
        "heldout_factual_probe_passed": heldout_error <= 1e-4,
        "new_intention_admitted": admission.accepted,
        "post_admission_goal_mastered": post_success,
        "retained_vectors_unchanged": torch.equal(
            retained_after_admission, retained_before
        ),
        "rejected_candidate_failed_heldout_probe": not rejected.accepted
        and rejected_error > 1e-4,
        "rejected_candidate_atomic": repertoire.content_digest() == rejected_digest,
        "controller_frozen": controller_digest == _digest(controller),
        "model_unchanged": model_digest == planner_model.digest(),
        "exact_repertoire_persistence": restored.content_digest() == repertoire.content_digest(),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.policy-free-intention-admission.v1",
        "claim_boundary": (
            "heldout verifier-gated acquisition of one opaque intention vector; "
            "not arbitrary intention synthesis or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "initial_repertoire_count": len(ACTION_BASIS),
            "new_intention": NEW_INTENTION.tolist(),
            "rejected_intention": REJECTED_INTENTION.tolist(),
            "goal": list(DIAGONAL_TARGET),
            "horizon": HORIZON - 1,
            "candidate_intentions_argument": "omitted_at_runtime",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "before_displacement": list(before_displacement),
            "after_displacement": list(post_displacement),
            "heldout_error": heldout_error,
            "rejected_error": rejected_error,
            "admission_entry_index": admission.entry_index,
            "rejected_record_count": repertoire.record_count,
            "before_expanded_nodes": before_expansions,
            "after_expanded_nodes": post_expansions,
        },
        "accounting": {
            "unique_verifier_bits": len(ACTION_BASIS) + 2,
            "unique_logical_lifetimes": TRANSITION_ROWS + len(ACTION_BASIS) + 2,
            "factual_statistics_updates": 1,
            "intention_admission_transactions": 2,
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85201)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
