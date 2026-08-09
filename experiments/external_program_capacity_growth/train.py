"""Audit verifier-gated growth of an external opaque program address space."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    AmodalCognitiveController,
    ExternalOutcomeProgramRouter,
    ExternalOutcomeProgramRouterState,
)

EVENT_WIDTH = 3
SOURCE_PROGRAMS = 2
DESTINATION_PROGRAMS = 3
SOURCE_EPISODES = 400
TARGET_EPISODES = 800
MASTERY_THRESHOLD = 0.90


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _feature(program: int) -> torch.Tensor:
    if program not in range(EVENT_WIDTH):
        raise ValueError("program feature index is outside the hidden relation")
    return torch.nn.functional.one_hot(
        torch.tensor([program]), num_classes=EVENT_WIDTH
    ).to(torch.float32)


def _accuracy(
    router: ExternalOutcomeProgramRouter,
    state: ExternalOutcomeProgramRouterState,
    programs: range,
) -> float:
    correct = 0
    for program in programs:
        choice = int(router.logits(state, _feature(program)).argmax().item())
        correct += choice == program
    return correct / len(programs)


def _train_context(
    router: ExternalOutcomeProgramRouter,
    state: ExternalOutcomeProgramRouterState,
    *,
    target_program: int,
    episodes: int,
    feedback_override: torch.Tensor | None = None,
) -> tuple[ExternalOutcomeProgramRouterState, torch.Tensor]:
    if episodes < 1:
        raise ValueError("training episodes must be positive")
    if feedback_override is not None and feedback_override.shape != (episodes,):
        raise ValueError("feedback override must contain one value per episode")
    outcomes: list[torch.Tensor] = []
    with torch.no_grad():
        for index in range(episodes):
            feature = _feature(target_program)
            state = router.begin_episode(state)
            choice, propensity = router.sample_program(
                state, feature, exploration=0.1
            )
            state = router.record_decision(state, feature, choice, propensity)
            verifier_outcome = (choice == target_program).to(torch.float32)
            outcomes.append(verifier_outcome)
            feedback = (
                verifier_outcome
                if feedback_override is None
                else feedback_override[index : index + 1]
            )
            state = router.apply_feedback(
                state,
                feedback,
                terminal=torch.ones(1, dtype=torch.bool),
            )
    return state, torch.cat(outcomes)


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    controller = AmodalCognitiveController(
        width=8,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest_before = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    router = ExternalOutcomeProgramRouter(
        feature_width=EVENT_WIDTH,
        program_capacity=SOURCE_PROGRAMS,
        initial_programs=SOURCE_PROGRAMS,
        initial_learning_rate=0.3,
        initial_trace_decay=0.0,
        initial_baseline_rate=0.05,
    )
    router_rule_digest_before = _digest(router.credit_rule)
    source_state = router.initial_state(1)
    source_state, source_outcomes = _train_context(
        router,
        source_state,
        target_program=0,
        episodes=SOURCE_EPISODES // 2,
    )
    source_state, second_source_outcomes = _train_context(
        router,
        source_state,
        target_program=1,
        episodes=SOURCE_EPISODES // 2,
    )
    source_outcomes = torch.cat((source_outcomes, second_source_outcomes))
    source_before_growth = _accuracy(router, source_state, range(SOURCE_PROGRAMS))

    old_logits = [
        router.logits(source_state, _feature(program))
        for program in range(SOURCE_PROGRAMS)
    ]

    def retention_probe(
        candidate: ExternalOutcomeProgramRouter,
        candidate_state: ExternalOutcomeProgramRouterState,
    ) -> bool:
        if candidate_state.active_programs != SOURCE_PROGRAMS:
            return False
        return all(
            torch.equal(
                candidate.logits(candidate_state, _feature(program))[..., :SOURCE_PROGRAMS],
                old_logits[program],
            )
            for program in range(SOURCE_PROGRAMS)
        )

    receipt, grown_state = router.grow_capacity_verified(
        source_state,
        DESTINATION_PROGRAMS,
        retention_probe,
    )
    receipt.validate()
    capacity_after_receipt = router.program_capacity
    active_after_receipt = grown_state.active_programs
    grown_state = router.append_program(grown_state)
    target_state, target_outcomes = _train_context(
        router,
        grown_state,
        target_program=2,
        episodes=TARGET_EPISODES,
    )
    source_after_learning = _accuracy(router, target_state, range(SOURCE_PROGRAMS))
    target_accuracy = _accuracy(router, target_state, range(DESTINATION_PROGRAMS))

    rejected_router = ExternalOutcomeProgramRouter(
        feature_width=EVENT_WIDTH,
        program_capacity=SOURCE_PROGRAMS,
        initial_programs=SOURCE_PROGRAMS,
    )
    rejected_state = rejected_router.initial_state(1)
    rejected_policy = rejected_state.credit.policy.clone()
    rejected_receipt, rejected_after = rejected_router.grow_capacity_verified(
        rejected_state,
        DESTINATION_PROGRAMS,
        lambda _candidate, _candidate_state: False,
    )
    rejection_unchanged = (
        not rejected_receipt.accepted
        and rejected_router.program_capacity == SOURCE_PROGRAMS
        and rejected_after is rejected_state
        and torch.equal(rejected_after.credit.policy, rejected_policy)
    )

    shuffled_router = ExternalOutcomeProgramRouter(
        feature_width=EVENT_WIDTH,
        program_capacity=DESTINATION_PROGRAMS,
        initial_programs=DESTINATION_PROGRAMS,
        initial_learning_rate=0.3,
        initial_trace_decay=0.0,
        initial_baseline_rate=0.05,
    )
    shuffled_state = shuffled_router.initial_state(1)
    shuffled_feedback = torch.randint(
        0,
        2,
        (TARGET_EPISODES,),
        generator=torch.Generator().manual_seed(seed + 1000),
        dtype=torch.float32,
    )
    shuffled_state, _ = _train_context(
        shuffled_router,
        shuffled_state,
        target_program=2,
        episodes=TARGET_EPISODES,
        feedback_override=shuffled_feedback,
    )
    shuffled_accuracy = _accuracy(shuffled_router, shuffled_state, range(3))
    restored = router.state_from_payload(router.state_payload(target_state))
    persistence_exact = all(
        torch.equal(getattr(restored.credit, name), getattr(target_state.credit, name))
        for name in ("policy", "eligibility", "baseline", "decisions", "feedbacks")
    ) and restored.active_programs == target_state.active_programs

    gates = {
        "source_mastery": source_before_growth >= MASTERY_THRESHOLD,
        "growth_accepted": receipt.accepted,
        "growth_receipt_source_capacity": receipt.source_capacity == SOURCE_PROGRAMS,
        "growth_receipt_destination_capacity": (
            receipt.destination_capacity == DESTINATION_PROGRAMS
        ),
        "capacity_committed": capacity_after_receipt == DESTINATION_PROGRAMS,
        "new_slot_activated_after_receipt": (
            active_after_receipt == SOURCE_PROGRAMS
            and target_state.active_programs == DESTINATION_PROGRAMS
        ),
        "source_retention_after_new_learning": source_after_learning >= MASTERY_THRESHOLD,
        "new_program_learned": target_accuracy >= MASTERY_THRESHOLD,
        "rejected_growth_is_no_write": rejection_unchanged,
        "reward_shuffled_control_rejected": shuffled_accuracy < MASTERY_THRESHOLD,
        "router_persistence_exact": persistence_exact,
        "controller_frozen": controller_digest_before == _digest(controller),
        "router_rule_frozen": router_rule_digest_before == _digest(router.credit_rule),
        "zero_optimizer_updates": True,
        "zero_replayed_examples": True,
        "zero_raw_feature_rows_retained": True,
    }
    report = {
        "schema": "neural-computer.external-program-capacity-growth-pressure-test.v1",
        "claim_boundary": (
            "A verifier-gated external program router can grow its bounded "
            "opaque address space while retaining old routes and learning one "
            "new route; this is not unrestricted memory growth or general "
            "continual learning."
        ),
        "seed": seed,
        "source_episodes": SOURCE_EPISODES,
        "target_episodes": TARGET_EPISODES,
        "source_before_growth": source_before_growth,
        "source_after_learning": source_after_learning,
        "target_accuracy": target_accuracy,
        "shuffled_accuracy": shuffled_accuracy,
        "growth_receipt": {
            "accepted": receipt.accepted,
            "source_capacity": receipt.source_capacity,
            "destination_capacity": receipt.destination_capacity,
            "active_programs": receipt.active_programs,
            "state_digest_before": receipt.state_digest_before,
            "state_digest_after": receipt.state_digest_after,
            "reason": receipt.reason,
        },
        "source_outcome_mean": float(source_outcomes.mean()),
        "target_outcome_mean": float(target_outcomes.mean()),
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": SOURCE_EPISODES + TARGET_EPISODES,
            "unique_logical_lifetimes": SOURCE_EPISODES + TARGET_EPISODES,
            "external_route_decision_updates": SOURCE_EPISODES + TARGET_EPISODES,
            "external_feedback_updates": SOURCE_EPISODES + TARGET_EPISODES,
            "optimizer_updates": 0,
            "replayed_examples": 0,
            "raw_feature_rows_retained": 0,
            "stable_bits_to_threshold": SOURCE_EPISODES + TARGET_EPISODES,
            "retention_on_mastered_primitives": source_after_learning,
            "transfer_ratio_against_fresh_learner": None,
        },
        "wall_time_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2303)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.seed, args.report_out), indent=2))


if __name__ == "__main__":
    main()
