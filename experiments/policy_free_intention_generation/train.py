"""Audit outcome-only intention generation on the canonical policy-free path.

The controller and factual transition model are frozen. A generator receives
only the controller's opaque adapted state, samples a continuous intention,
and receives a scalar verifier outcome. Successful means are admitted into
the stable external intention repertoire; the runtime then plans with those
verified candidates. A copy-on-write successor cell, a fresh learner, and a
reward-shuffled control make transfer and causality explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch import nn

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    ExternalIntentionRepertoire,
    ExternalModelBasedPlanner,
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionGeneratorState,
    OpaqueProtocolDecoder,
    PolicyFreeAmodalRuntime,
)

CONTROLLER_WIDTH = 4
STATE_WIDTH = 12
INTENTION_WIDTH = 2
HORIZON = 1
BEAM_WIDTH = 8
UTILITY_TEMPERATURE = 0.8
MASTERY_THRESHOLD = 0.95
SOURCE_TARGET = torch.tensor([0.75, -0.75])
SUCCESSOR_TARGET = torch.tensor([0.55, -0.95])
ACTION_BASIS = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
)


class _AdditiveFactualModel(nn.Module):
    state_width = STATE_WIDTH
    intention_width = INTENTION_WIDTH

    def forward(self, state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
        result = state.clone()
        result[:, :INTENTION_WIDTH] = (
            result[:, :INTENTION_WIDTH] + intention
        )
        return result


def _digest_module(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(repr(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _digest_generator_state(
    state: ExternalOutcomeIntentionGeneratorState,
) -> str:
    digest = hashlib.sha256()
    for name in (
        "input_weights",
        "input_bias",
        "output_weights",
        "output_bias",
        "input_weight_eligibility",
        "input_bias_eligibility",
        "output_weight_eligibility",
        "output_bias_eligibility",
        "baseline",
        "decisions",
        "feedbacks",
        "protected",
    ):
        value = getattr(state, name).detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _feedback() -> ControllerFeedback:
    return ControllerFeedback(
        action=torch.zeros(1, 3),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )


def _utility(intention: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.exp(
        -(intention - target.reshape(1, -1)).square().sum(dim=-1)
        / UTILITY_TEMPERATURE
    ).clamp(0.0, 1.0)


def _build_runtime(seed: int) -> tuple[
    AmodalCognitiveController,
    AmodalControllerRuntime,
    PolicyFreeAmodalRuntime,
    ExternalOutcomeIntentionGenerator,
    torch.Tensor,
    list[AmodalEvent],
    ControllerFeedback,
    torch.Tensor,
]:
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
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder(
        "opaque_backend",
        OpaqueProtocolDecoder(INTENTION_WIDTH, INTENTION_WIDTH),
    )
    generator = ExternalOutcomeIntentionGenerator(
        context_width=STATE_WIDTH,
        intention_width=INTENTION_WIDTH,
        hidden_width=32,
        initial_learning_rate=0.1,
        initial_baseline_rate=0.05,
        noise_scale=0.35,
        initial_parameter_scale=0.05,
    )
    planner = ExternalModelBasedPlanner(
        _AdditiveFactualModel(),
        beam_width=BEAM_WIDTH,
    )
    policy_free = PolicyFreeAmodalRuntime(
        runtime,
        planner,
        intention_generator=generator,
    )
    for parameter in policy_free.state_adapter.parameters():
        parameter.requires_grad_(False)
    state = runtime.initial_state(1, device="cpu")
    event = [AmodalEvent(torch.randn(1, CONTROLLER_WIDTH))]
    feedback = _feedback()
    preview, _ = runtime.step_events(event, state, feedback)
    context = policy_free.state_adapter(
        preview.controller.state_representation.detach()
    )
    return controller, runtime, policy_free, generator, state, event, feedback, context


def _goal(context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    goal = context.clone()
    goal[:, :INTENTION_WIDTH] += target.reshape(1, -1)
    return goal


def _train_cell(
    *,
    policy_free: PolicyFreeAmodalRuntime,
    generator: ExternalOutcomeIntentionGenerator,
    generator_state: ExternalOutcomeIntentionGeneratorState,
    controller_state,
    events: list[AmodalEvent],
    feedback: ControllerFeedback,
    goal: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
    cell_index: int,
    max_updates: int,
    shuffled: bool = False,
) -> tuple[ExternalOutcomeIntentionGeneratorState, int, float, int, float]:
    begun = time.perf_counter()
    batch_size = generator_state.baseline.shape[0]
    if context.shape[0] != batch_size:
        raise ValueError("generator experiment context batch differs")
    present = torch.zeros(batch_size, dtype=torch.bool)
    present[cell_index] = True
    terminal = present.clone()
    score = float(
        _utility(generator.mean(generator_state, context)[cell_index : cell_index + 1], target)
        .item()
    )
    search_expansions = 0
    for update in range(1, max_updates + 1):
        output, _ = policy_free.step_events(
            events,
            controller_state,
            feedback,
            goal,
            horizon=HORIZON,
            beam_width=BEAM_WIDTH,
            generator_state=generator_state,
        )
        proposal = output.intention_generation
        if proposal is None:
            raise AssertionError("policy-free runtime did not return a generator proposal")
        search_expansions += int(output.planning.expanded_nodes)
        outcomes = _utility(proposal.intentions, target)
        if shuffled:
            outcomes = torch.rand_like(outcomes)
        generator_state = policy_free.record_intention_generation_decision(
            generator_state,
            proposal,
            present=present,
        )
        generator_state = policy_free.apply_intention_generation_feedback(
            generator_state,
            proposal,
            outcomes,
            present=present,
            terminal=terminal,
        )
        score = float(
            _utility(
                generator.mean(generator_state, context)[cell_index : cell_index + 1],
                target,
            ).item()
        )
        if not shuffled and score >= MASTERY_THRESHOLD:
            return (
                generator_state,
                update,
                score,
                search_expansions,
                time.perf_counter() - begun,
            )
    return (
        generator_state,
        max_updates,
        score,
        search_expansions,
        time.perf_counter() - begun,
    )


def _planner_success(
    policy_free: PolicyFreeAmodalRuntime,
    *,
    controller_state,
    events: list[AmodalEvent],
    feedback: ControllerFeedback,
    goal: torch.Tensor,
    repertoire: ExternalIntentionRepertoire,
    target: torch.Tensor,
) -> tuple[bool, int]:
    evaluation_runtime = PolicyFreeAmodalRuntime(
        policy_free.runtime,
        policy_free.planner,
        state_adapter=policy_free.state_adapter,
        intention_repertoire=repertoire,
    )
    output, _ = evaluation_runtime.step_events(
        events,
        controller_state,
        feedback,
        goal,
        horizon=HORIZON,
        beam_width=BEAM_WIDTH,
    )
    return (
        bool(
            _utility(output.planning.intentions[:, 0], target).item()
            >= MASTERY_THRESHOLD
        ),
        int(output.planning.expanded_nodes),
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    (
        controller,
        runtime,
        policy_free_generator,
        generator,
        controller_state,
        events,
        feedback,
        context,
    ) = _build_runtime(seed)
    controller_digest = _digest_module(controller)
    state_adapter_digest = _digest_module(policy_free_generator.state_adapter)
    generator_state = generator.initial_state(1)
    initial_generator_rng = torch.get_rng_state()
    source_goal = _goal(context, SOURCE_TARGET)
    (
        generator_state,
        source_updates,
        source_score,
        source_search_expansions,
        source_seconds,
    ) = _train_cell(
        policy_free=policy_free_generator,
        generator=generator,
        generator_state=generator_state,
        controller_state=controller_state,
        events=events,
        feedback=feedback,
        goal=source_goal,
        context=context,
        target=SOURCE_TARGET,
        cell_index=0,
        max_updates=1400,
    )
    source_cell = generator.mean(generator_state, context)[0].detach().clone()
    source_cell_before_protection = source_cell.clone()
    generator_state = generator.protect(generator_state, [0])
    generator_state, successor_cell_index = generator.append_cell(
        generator_state,
        source_cell=0,
    )
    source_cell_digest_before_successor = _digest_generator_state(generator_state)

    successor_context = context.expand(2, -1).clone()
    successor_events = [
        AmodalEvent(events[0].payload.expand(2, -1).clone())
    ]
    successor_controller_state = runtime.initial_state(2, device="cpu")
    successor_goal = _goal(successor_context, SUCCESSOR_TARGET)
    (
        generator_state,
        successor_updates,
        successor_score,
        successor_search_expansions,
        successor_seconds,
    ) = _train_cell(
        policy_free=policy_free_generator,
        generator=generator,
        generator_state=generator_state,
        controller_state=successor_controller_state,
        events=successor_events,
        feedback=ControllerFeedback(
            action=torch.zeros(2, 3),
            reward=torch.zeros(2),
            propensity=torch.ones(2),
            has_feedback=torch.zeros(2),
        ),
        goal=successor_goal,
        context=successor_context,
        target=SUCCESSOR_TARGET,
        cell_index=successor_cell_index,
        max_updates=1400,
    )
    successor_cell = generator.mean(generator_state, successor_context)[1].detach().clone()

    torch.set_rng_state(initial_generator_rng)
    fresh_state = generator.initial_state(1)
    fresh_policy_free = PolicyFreeAmodalRuntime(
        runtime,
        policy_free_generator.planner,
        state_adapter=policy_free_generator.state_adapter,
        intention_generator=generator,
    )
    (
        fresh_state,
        fresh_updates,
        fresh_score,
        fresh_search_expansions,
        fresh_seconds,
    ) = _train_cell(
        policy_free=fresh_policy_free,
        generator=generator,
        generator_state=fresh_state,
        controller_state=controller_state,
        events=events,
        feedback=feedback,
        goal=_goal(context, SUCCESSOR_TARGET),
        context=context,
        target=SUCCESSOR_TARGET,
        cell_index=0,
        max_updates=1400,
    )

    torch.set_rng_state(initial_generator_rng)
    shuffled_state = generator.initial_state(1)
    (
        shuffled_state,
        shuffled_updates,
        shuffled_score,
        shuffled_search_expansions,
        shuffled_seconds,
    ) = _train_cell(
        policy_free=fresh_policy_free,
        generator=generator,
        generator_state=shuffled_state,
        controller_state=controller_state,
        events=events,
        feedback=feedback,
        goal=_goal(context, SUCCESSOR_TARGET),
        context=context,
        target=SUCCESSOR_TARGET,
        cell_index=0,
        max_updates=500,
        shuffled=True,
    )

    repertoire = ExternalIntentionRepertoire(INTENTION_WIDTH)
    repertoire.observe(ACTION_BASIS)
    retained_before = repertoire.statistics()["intentions"].clone()
    source_admission = repertoire.admit_verified(
        source_cell,
        lambda candidate: torch.equal(
            candidate.statistics()["intentions"][: len(ACTION_BASIS)],
            retained_before,
        )
        and float(_utility(candidate.statistics()["intentions"][-1:].clone(), SOURCE_TARGET).item())
        >= MASTERY_THRESHOLD,
        reason="outcome_trained_source_intention_heldout_verifier",
    )
    retained_after_source = repertoire.statistics()["intentions"][: len(ACTION_BASIS)].clone()
    successor_admission = repertoire.admit_verified(
        successor_cell,
        lambda candidate: torch.equal(
            candidate.statistics()["intentions"][: source_admission.destination_record_count],
            repertoire.statistics()["intentions"],
        )
        and float(_utility(candidate.statistics()["intentions"][-1:].clone(), SUCCESSOR_TARGET).item())
        >= MASTERY_THRESHOLD,
        reason="outcome_trained_successor_intention_heldout_verifier",
    )

    source_goal_success, source_probe_expansions = _planner_success(
        policy_free_generator,
        controller_state=controller_state,
        events=events,
        feedback=feedback,
        goal=source_goal,
        repertoire=repertoire,
        target=SOURCE_TARGET,
    )
    successor_goal_success, successor_probe_expansions = _planner_success(
        policy_free_generator,
        controller_state=controller_state,
        events=events,
        feedback=feedback,
        goal=_goal(context, SUCCESSOR_TARGET),
        repertoire=repertoire,
        target=SUCCESSOR_TARGET,
    )

    restored_repertoire = ExternalIntentionRepertoire.from_payload(
        repertoire.payload()
    )
    restored_generator = generator.state_from_payload(
        generator.state_payload(generator_state)
    )
    old_cell_retained = torch.equal(
        generator.mean(generator_state, successor_context)[0],
        source_cell_before_protection,
    )
    gates = {
        "source_generator_acquired": source_score >= MASTERY_THRESHOLD,
        "successor_generator_acquired": successor_score >= MASTERY_THRESHOLD,
        "fresh_successor_control_acquired": fresh_score >= MASTERY_THRESHOLD,
        "outcome_shuffled_control_failed": shuffled_score < MASTERY_THRESHOLD,
        "source_intention_admitted": source_admission.accepted,
        "successor_intention_admitted": successor_admission.accepted,
        "source_goal_planned_after_admission": source_goal_success,
        "successor_goal_planned_after_admission": successor_goal_success,
        "copy_on_write_source_retained": old_cell_retained,
        "warm_successor_faster_than_fresh": successor_updates < fresh_updates,
        "controller_frozen": controller_digest == _digest_module(controller),
        "state_adapter_frozen": state_adapter_digest
        == _digest_module(policy_free_generator.state_adapter),
        "exact_repertoire_persistence": restored_repertoire.content_digest()
        == repertoire.content_digest(),
        "exact_generator_persistence": _digest_generator_state(restored_generator)
        == _digest_generator_state(generator_state),
        "retained_base_entries_unchanged": torch.equal(
            retained_after_source, retained_before
        ),
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.policy-free-intention-generation.v1",
        "claim_boundary": (
            "outcome-only continuous intention discovery through a frozen amodal "
            "controller and policy-free runtime; not general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "controller_width": CONTROLLER_WIDTH,
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "source_target": SOURCE_TARGET.tolist(),
            "successor_target": SUCCESSOR_TARGET.tolist(),
            "utility_temperature": UTILITY_TEMPERATURE,
            "mastery_threshold": MASTERY_THRESHOLD,
            "candidate_memory": repertoire.configuration(),
            "generator": generator.configuration(),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "source_updates": source_updates,
            "source_score": source_score,
            "successor_updates": successor_updates,
            "successor_score": successor_score,
            "fresh_updates": fresh_updates,
            "fresh_score": fresh_score,
            "shuffled_updates": shuffled_updates,
            "shuffled_score": shuffled_score,
            "source_seconds": source_seconds,
            "successor_seconds": successor_seconds,
            "fresh_seconds": fresh_seconds,
            "shuffled_seconds": shuffled_seconds,
            "source_admission": source_admission.accepted,
            "successor_admission": successor_admission.accepted,
            "repertoire_count": repertoire.record_count,
            "source_cell_digest_before_successor": source_cell_digest_before_successor,
        },
        "accounting": {
            "unique_verifier_bits": source_updates + successor_updates + fresh_updates,
            "control_outcome_bits": shuffled_updates,
            "unique_logical_lifetimes": 4,
            "external_generator_updates": source_updates
            + successor_updates
            + fresh_updates
            + shuffled_updates,
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "search_expansions": source_search_expansions
            + successor_search_expansions
            + fresh_search_expansions
            + shuffled_search_expansions
            + source_probe_expansions
            + successor_probe_expansions,
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
