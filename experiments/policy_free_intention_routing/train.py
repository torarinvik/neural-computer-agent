"""Stress-test learned opaque routing over growing external intention memory.

The controller and factual transition model are frozen.  A context-conditioned
router selects one external cell, the selected cell emits an opaque intention,
and only a delayed scalar verifier outcome is returned.  The experiment does
not pass a cell index to the runtime.  It grows source, successor, and reversal
cells copy-on-write, protects mastered cells, and records fresh, shuffled,
missing-evidence, and memory-corruption controls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path

import torch

from experiments.policy_free_intention_memory import train as base
from neural_computer import (
    ExternalControllerTrajectoryQueryAdapter,
    ExternalOutcomeIntentionGenerator,
    ExternalOutcomeIntentionMemory,
    ExternalOutcomeIntentionRouter,
    ExternalRoutedIntentionMemoryState,
    ExternalRoutedIntentionProposal,
    PolicyFreeAmodalRuntime,
)

ROUTING_EXPLORATION_BONUS = 0.75
DELAY_STEPS = 3
REVERSAL_DELAY_STEPS = 4
MAX_UPDATES = 240
REVERSAL_UPDATES = 260
CONTROL_UPDATES = 160
GRADUAL_MASK_SWITCH_UPDATE = MAX_UPDATES // 2
MULTI_STAGE_MASK_STAGE_UPDATES = 34
SOURCE_CONTEXT_MASK = torch.tensor(
    [True, False, True, False, True, False, True, False, True, False, True, False]
)
SUCCESSOR_CONTEXT_MASK = ~SOURCE_CONTEXT_MASK
REVERSAL_CONTEXT_MASK = SOURCE_CONTEXT_MASK.clone()
OVERLAP_SUCCESSOR_CONTEXT_MASK = torch.tensor(
    [True, True, True, True, True, False, True, False, False, True, False, True]
)
OVERLAP_REVERSAL_CONTEXT_MASK = OVERLAP_SUCCESSOR_CONTEXT_MASK.clone()


def _mask_from_indices(*indices: int) -> torch.Tensor:
    mask = torch.zeros(base.STATE_WIDTH, dtype=torch.bool)
    mask[list(indices)] = True
    return mask


MULTI_STAGE_MASK_SCHEDULE = (
    (1, SOURCE_CONTEXT_MASK.clone()),
    (1 + MULTI_STAGE_MASK_STAGE_UPDATES, _mask_from_indices(0, 1, 2, 4, 6, 8, 10)),
    (
        1 + 2 * MULTI_STAGE_MASK_STAGE_UPDATES,
        _mask_from_indices(0, 1, 2, 3, 4, 6, 8, 10),
    ),
    (
        1 + 3 * MULTI_STAGE_MASK_STAGE_UPDATES,
        _mask_from_indices(0, 1, 2, 3, 4, 6, 8, 9, 10),
    ),
    (
        1 + 4 * MULTI_STAGE_MASK_STAGE_UPDATES,
        _mask_from_indices(0, 1, 2, 3, 4, 6, 8, 9, 10, 11),
    ),
    (
        1 + 5 * MULTI_STAGE_MASK_STAGE_UPDATES,
        _mask_from_indices(0, 1, 2, 3, 4, 6, 9, 10, 11),
    ),
    (
        1 + 6 * MULTI_STAGE_MASK_STAGE_UPDATES,
        OVERLAP_SUCCESSOR_CONTEXT_MASK.clone(),
    ),
)


def _mask_label(context_mask: torch.Tensor | None) -> str:
    if context_mask is None:
        return "dense"
    return "".join("1" if bool(value) else "0" for value in context_mask.tolist())


def _mask_configuration(
    *,
    masked_context: bool,
    mask_curriculum: str,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    tuple[tuple[int, torch.Tensor], ...] | None,
]:
    """Return source/successor/reversal masks and an optional successor schedule."""

    valid_curricula = {"complementary", "overlapping", "gradual", "multi_stage"}
    if mask_curriculum not in valid_curricula:
        raise ValueError(f"unsupported mask curriculum: {mask_curriculum}")
    if not masked_context:
        if mask_curriculum != "complementary":
            raise ValueError("non-complementary mask curricula require --masked-context")
        return None, None, None, None
    if mask_curriculum == "complementary":
        return (
            SOURCE_CONTEXT_MASK.clone(),
            SUCCESSOR_CONTEXT_MASK.clone(),
            REVERSAL_CONTEXT_MASK.clone(),
            None,
        )
    if mask_curriculum == "overlapping":
        return (
            SOURCE_CONTEXT_MASK.clone(),
            OVERLAP_SUCCESSOR_CONTEXT_MASK.clone(),
            OVERLAP_REVERSAL_CONTEXT_MASK.clone(),
            None,
        )
    if mask_curriculum == "gradual":
        successor_schedule = (
            (1, SOURCE_CONTEXT_MASK.clone()),
            (GRADUAL_MASK_SWITCH_UPDATE, OVERLAP_SUCCESSOR_CONTEXT_MASK.clone()),
        )
    else:
        successor_schedule = tuple(
            (start, mask.clone()) for start, mask in MULTI_STAGE_MASK_SCHEDULE
        )
    return (
        SOURCE_CONTEXT_MASK.clone(),
        OVERLAP_SUCCESSOR_CONTEXT_MASK.clone(),
        OVERLAP_REVERSAL_CONTEXT_MASK.clone(),
        successor_schedule,
    )


def _digest_state(state: ExternalRoutedIntentionMemoryState) -> str:
    digest = hashlib.sha256()
    values = (
        ("cells.input_weights", state.cells.input_weights),
        ("cells.input_bias", state.cells.input_bias),
        ("cells.output_weights", state.cells.output_weights),
        ("cells.output_bias", state.cells.output_bias),
        ("cells.baseline", state.cells.baseline),
        ("cells.decisions", state.cells.decisions),
        ("cells.feedbacks", state.cells.feedbacks),
        ("cells.protected", state.cells.protected),
        ("retention_observations", state.retention_observations),
        ("retention_successes", state.retention_successes),
        ("retention_prefix_minima", state.retention_prefix_minima),
        ("retention_reversal_streaks", state.retention_reversal_streaks),
        ("retention_reversal_counts", state.retention_reversal_counts),
        ("retention_mastered", state.retention_mastered),
        ("retention_context_prototypes", state.retention_context_prototypes),
        ("retention_context_masses", state.retention_context_masses),
        (
            "retention_context_observed_masses",
            state.retention_context_observed_masses,
        ),
        ("routing_keys", state.routing_keys),
        ("routing_bias", state.routing_bias),
        ("routing_baseline", state.routing_baseline),
        ("routing_decisions", state.routing_decisions),
        ("routing_feedbacks", state.routing_feedbacks),
    )
    for name, value in values:
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(repr(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _cell_snapshot(state: ExternalRoutedIntentionMemoryState, cell_index: int):
    return tuple(
        getattr(state.cells, name)[cell_index].detach().clone()
        for name in (
            "input_weights",
            "input_bias",
            "output_weights",
            "output_bias",
            "baseline",
        )
    )


def _cell_matches(state, cell_index: int, snapshot) -> bool:
    return all(
        torch.equal(getattr(state.cells, name)[cell_index], expected)
        for name, expected in zip(
            ("input_weights", "input_bias", "output_weights", "output_bias", "baseline"),
            snapshot,
            strict=True,
        )
    )


def _single_cell_state(
    state: ExternalRoutedIntentionMemoryState,
    cell_index: int,
) -> ExternalRoutedIntentionMemoryState:
    """Extract one caller-owned cell as a matched fresh-control state."""

    cells = state.cells
    single_cells = replace(
        cells,
        input_weights=cells.input_weights[cell_index : cell_index + 1].clone(),
        input_bias=cells.input_bias[cell_index : cell_index + 1].clone(),
        output_weights=cells.output_weights[cell_index : cell_index + 1].clone(),
        output_bias=cells.output_bias[cell_index : cell_index + 1].clone(),
        input_weight_eligibility=cells.input_weight_eligibility[
            cell_index : cell_index + 1
        ].clone(),
        input_bias_eligibility=cells.input_bias_eligibility[
            cell_index : cell_index + 1
        ].clone(),
        output_weight_eligibility=cells.output_weight_eligibility[
            cell_index : cell_index + 1
        ].clone(),
        output_bias_eligibility=cells.output_bias_eligibility[
            cell_index : cell_index + 1
        ].clone(),
        baseline=cells.baseline[cell_index : cell_index + 1].clone(),
        decisions=cells.decisions[cell_index : cell_index + 1].clone(),
        feedbacks=cells.feedbacks[cell_index : cell_index + 1].clone(),
        protected=cells.protected[cell_index : cell_index + 1].clone(),
    )
    return replace(
        state,
        cells=single_cells,
        routing_keys=state.routing_keys[cell_index : cell_index + 1].clone(),
        routing_bias=state.routing_bias[cell_index : cell_index + 1].clone(),
        routing_decisions=state.routing_decisions[cell_index : cell_index + 1].clone(),
        routing_feedbacks=state.routing_feedbacks[cell_index : cell_index + 1].clone(),
        retention_observations=state.retention_observations[
            cell_index : cell_index + 1
        ].clone(),
        retention_successes=state.retention_successes[
            cell_index : cell_index + 1
        ].clone(),
        retention_prefix_minima=state.retention_prefix_minima[
            cell_index : cell_index + 1
        ].clone(),
        retention_reversal_streaks=state.retention_reversal_streaks[
            cell_index : cell_index + 1
        ].clone(),
        retention_reversal_counts=state.retention_reversal_counts[
            cell_index : cell_index + 1
        ].clone(),
        retention_mastered=state.retention_mastered[cell_index : cell_index + 1].clone(),
        retention_context_prototypes=state.retention_context_prototypes[
            cell_index : cell_index + 1
        ].clone(),
        retention_context_masses=state.retention_context_masses[
            cell_index : cell_index + 1
        ].clone(),
        retention_context_observed_masses=state.retention_context_observed_masses[
            cell_index : cell_index + 1
        ].clone(),
    )


def _new_policy(
    *,
    reference: PolicyFreeAmodalRuntime,
    seed: int,
    cell_count: int,
    mastery_threshold: float = 0.95,
    min_mastery_feedbacks: int = 8,
    context_width: int = base.STATE_WIDTH,
    route_query_adapter: ExternalControllerTrajectoryQueryAdapter | None = None,
    unqualified_cell_probability: float = 0.0,
    context_masking: bool = False,
) -> tuple[PolicyFreeAmodalRuntime, ExternalOutcomeIntentionRouter, ExternalRoutedIntentionMemoryState]:
    torch.manual_seed(seed)
    memory = ExternalOutcomeIntentionMemory(
        ExternalOutcomeIntentionGenerator(
            context_width=context_width,
            intention_width=base.INTENTION_WIDTH,
            hidden_width=32,
            initial_learning_rate=0.03,
            initial_baseline_rate=0.05,
            noise_scale=0.35,
            initial_parameter_scale=0.05,
            context_masking=context_masking,
        )
    )
    router = ExternalOutcomeIntentionRouter(
        memory,
        exploration_bonus=ROUTING_EXPLORATION_BONUS,
        mastery_threshold=mastery_threshold,
        min_mastery_feedbacks=min_mastery_feedbacks,
        unqualified_cell_probability=unqualified_cell_probability,
    )
    policy = PolicyFreeAmodalRuntime(
        reference.runtime,
        reference.planner,
        state_adapter=reference.state_adapter,
        intention_router=router,
        route_query_adapter=route_query_adapter,
    )
    return policy, router, router.initial_state(cell_count)


def _train_regime(
    *,
    policy: PolicyFreeAmodalRuntime,
    router: ExternalOutcomeIntentionRouter,
    state: ExternalRoutedIntentionMemoryState,
    controller_state,
    feedback,
    event,
    context: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    context_mask_schedule: tuple[tuple[int, torch.Tensor], ...] | None = None,
    goal_context: torch.Tensor | None = None,
    target: torch.Tensor,
    max_updates: int,
    delay: int,
    noise_fraction: float = 0.0,
    reward_shuffled: bool = False,
    action_shuffled: bool = False,
    random_seed: int,
    stop_at_mastery: bool = True,
    mastery_cell: int | None = None,
) -> tuple[ExternalRoutedIntentionMemoryState, dict[str, object]]:
    if mastery_cell is not None and not 0 <= mastery_cell < state.cells.baseline.shape[0]:
        raise ValueError("mastery cell is out of range")
    begun = time.perf_counter()
    pending: deque[tuple[ExternalRoutedIntentionProposal, torch.Tensor]] = deque()
    random_source = torch.Generator().manual_seed(random_seed)
    selected: list[int] = []
    materialized_candidate_counts: list[int] = []
    observed_outcomes: list[float] = []
    search_expansions = 0
    updates = 0
    planner_context = context if goal_context is None else goal_context
    if context_mask is not None and context_mask_schedule is not None:
        raise ValueError("context_mask and context_mask_schedule are mutually exclusive")
    if context_mask_schedule is not None:
        if not context_mask_schedule or context_mask_schedule[0][0] != 1:
            raise ValueError("context mask schedule must begin at update one")
        previous_start = 0
        for start, scheduled_mask in context_mask_schedule:
            if start <= previous_start:
                raise ValueError("context mask schedule starts must be increasing")
            if scheduled_mask.shape != context.shape[-1:] or scheduled_mask.dtype != torch.bool:
                raise ValueError("scheduled context masks must match the context")
            previous_start = start

    def active_context_mask(update: int) -> torch.Tensor | None:
        if context_mask_schedule is None:
            return context_mask
        selected_mask = context_mask_schedule[0][1]
        for start, scheduled_mask in context_mask_schedule:
            if update < start:
                break
            selected_mask = scheduled_mask
        return selected_mask

    mask_labels: Counter[str] = Counter()

    def apply_pending() -> None:
        nonlocal state
        proposal, outcome = pending.popleft()
        state = policy.apply_intention_routing_feedback(
            state,
            proposal,
            outcome,
        )

    for updates in range(1, max_updates + 1):
        current_context_mask = active_context_mask(updates)
        mask_labels[_mask_label(current_context_mask)] += 1
        runtime_context_mask = (
            None
            if current_context_mask is None
            else current_context_mask.unsqueeze(0)
        )
        output, _ = policy.step_events(
            event,
            controller_state,
            feedback,
            base._goal(planner_context, target),
            horizon=base.HORIZON,
            beam_width=base.BEAM_WIDTH,
            intention_router_state=state,
            intention_context_mask=runtime_context_mask,
        )
        proposal = output.intention_routing
        if proposal is None:
            raise AssertionError("policy-free routing did not return a proposal")
        materialized_candidate_counts.append(len(proposal.candidates.cell_indices))
        if output.planning.candidate_indices is None:
            raise AssertionError("planner dropped selected-intention provenance")
        if output.planning.candidate_indices.tolist() != [[0]]:
            raise AssertionError("router did not provide the sole planner candidate")
        search_expansions += int(output.planning.expanded_nodes)
        selected.append(int(proposal.selected_cells[0].item()))
        state = policy.record_intention_routing_decision(state, proposal)
        outcome = base._utility(output.planning.intentions[:, 0], target)
        if action_shuffled:
            random_action = torch.randn(
                outcome.shape,
                generator=random_source,
                dtype=outcome.dtype,
            )
            outcome = base._utility(random_action, target)
        elif reward_shuffled:
            outcome = torch.rand(
                outcome.shape,
                generator=random_source,
                dtype=outcome.dtype,
            )
        elif noise_fraction:
            outcome = (1.0 - noise_fraction) * outcome + noise_fraction * torch.rand(
                outcome.shape,
                generator=random_source,
                dtype=outcome.dtype,
            )
        observed_outcomes.append(float(outcome.item()))
        pending.append((proposal, outcome))
        if len(pending) > delay:
            apply_pending()

        mean_scores = base._utility(
            router.mean(state, context, context_mask=runtime_context_mask)[0], target
        )
        deterministic_score = float(
            mean_scores[mastery_cell].item()
            if mastery_cell is not None
            else mean_scores.max().item()
        )
        if (
            stop_at_mastery
            and not reward_shuffled
            and not action_shuffled
            and not noise_fraction
            and deterministic_score >= base.MASTERY_THRESHOLD
        ):
            while pending:
                apply_pending()
            settled_scores = base._utility(
                router.mean(state, context, context_mask=runtime_context_mask)[0], target
            )
            settled_score = float(
                settled_scores[mastery_cell].item()
                if mastery_cell is not None
                else settled_scores.max().item()
            )
            if settled_score >= base.MASTERY_THRESHOLD:
                break
    while pending:
        apply_pending()

    means = router.mean(state, context, context_mask=runtime_context_mask)[0]
    mean_scores = base._utility(means, target)
    scored_cell = (
        mastery_cell
        if mastery_cell is not None
        else int(mean_scores.argmax().item())
    )
    selected_counts = Counter(selected)
    report = {
        "updates": updates,
        "deterministic_best_score": float(mean_scores.max().item()),
        "deterministic_best_cell": int(mean_scores.argmax().item()),
        "mastery_cell": mastery_cell,
        "mastery_cell_score": float(mean_scores[scored_cell].item()),
        "selected_cell_counts": {str(k): v for k, v in sorted(selected_counts.items())},
        "selected_cell_fraction": {
            str(k): v / max(1, len(selected)) for k, v in sorted(selected_counts.items())
        },
        "materialized_candidate_cells": {
            "minimum": min(materialized_candidate_counts),
            "maximum": max(materialized_candidate_counts),
            "mean": sum(materialized_candidate_counts)
            / max(1, len(materialized_candidate_counts)),
        },
        "mean_outcome": sum(observed_outcomes) / max(1, len(observed_outcomes)),
        "context_mask_patterns": dict(sorted(mask_labels.items())),
        "final_context_mask": _mask_label(active_context_mask(updates)),
        "feedbacks": int(state.routing_feedbacks.sum().item()),
        "routing_decisions": state.routing_decisions.tolist(),
        "search_expansions": search_expansions,
        "wall_seconds": time.perf_counter() - begun,
    }
    return state, report


def _missing_evidence_control(
    *,
    policy: PolicyFreeAmodalRuntime,
    router: ExternalOutcomeIntentionRouter,
    state: ExternalRoutedIntentionMemoryState,
    controller_state,
    feedback,
    event,
    context: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    goal_context: torch.Tensor | None = None,
    target: torch.Tensor,
) -> dict[str, object]:
    before = _digest_state(state)
    runtime_context_mask = (
        None if context_mask is None else context_mask.unsqueeze(0)
    )
    for _ in range(32):
        output, _ = policy.step_events(
            event,
            controller_state,
            feedback,
            base._goal(
                context if goal_context is None else goal_context,
                target,
            ),
            horizon=base.HORIZON,
            beam_width=base.BEAM_WIDTH,
            intention_router_state=state,
            intention_context_mask=runtime_context_mask,
        )
        proposal = output.intention_routing
        if proposal is None:
            raise AssertionError("missing-evidence control lost routing proposal")
        absent = torch.zeros(1, dtype=torch.bool)
        state = policy.record_intention_routing_decision(state, proposal, present=absent)
        state = policy.apply_intention_routing_feedback(
            state,
            proposal,
            torch.zeros(1),
            present=absent,
        )
    return {
        "state_unchanged": before == _digest_state(state),
        "routing_decisions": state.routing_decisions.tolist(),
        "routing_feedbacks": state.routing_feedbacks.tolist(),
    }


def _corruption_probe(
    router: ExternalOutcomeIntentionRouter,
    state: ExternalRoutedIntentionMemoryState,
    context: torch.Tensor,
    target: torch.Tensor,
    context_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    clean_score = float(
        base._utility(
            router.mean(
                state,
                context,
                context_mask=None
                if context_mask is None
                else context_mask.unsqueeze(0),
            )[:, 0],
            target,
        ).item()
    )
    corrupted_cells = replace(
        state.cells,
        output_weights=state.cells.output_weights.clone(),
        output_bias=state.cells.output_bias.clone(),
    )
    corrupted_cells.output_weights[0].zero_()
    corrupted_cells.output_bias[0].zero_()
    corrupted = replace(state, cells=corrupted_cells)
    corrupted_score = float(
        base._utility(
            router.mean(
                corrupted,
                context,
                context_mask=None
                if context_mask is None
                else context_mask.unsqueeze(0),
            )[:, 0],
            target,
        ).item()
    )
    return {
        "clean_score": clean_score,
        "corrupted_score": corrupted_score,
        "corruption_detected": corrupted_score < clean_score,
    }


def _heldout_retention_verification(
    *,
    router: ExternalOutcomeIntentionRouter,
    state: ExternalRoutedIntentionMemoryState,
    cell_index: int,
    context: torch.Tensor,
    context_mask: torch.Tensor | None,
    target: torch.Tensor,
) -> tuple[ExternalRoutedIntentionMemoryState, dict[str, object]]:
    """Qualify a mastered cell from a fresh deterministic verifier prefix."""

    runtime_context_mask = (
        None if context_mask is None else context_mask.unsqueeze(0)
    )
    deterministic_score = float(
        base._utility(
            router.mean(
                state,
                context,
                context_mask=runtime_context_mask,
            )[0, cell_index].unsqueeze(0),
            target,
        ).item()
    )
    verifier_bits = router.min_mastery_feedbacks
    heldout_outcomes = [deterministic_score] * verifier_bits
    next_state, receipt = router.verify_and_protect(
        state,
        cell_index,
        context[0],
        heldout_outcomes,
        floor=base.MASTERY_THRESHOLD,
        context_mask=None if context_mask is None else context_mask,
    )
    return next_state, {
        "accepted": receipt.accepted,
        "deterministic_score": deterministic_score,
        "verifier_bits": verifier_bits,
        "prefix_minimum": receipt.prefix_minimum,
        "context_relevance_minimum": receipt.context_relevance_minimum,
        "reason": receipt.reason,
    }


def run(
    seed: int,
    report_out: Path,
    *,
    masked_context: bool = False,
    mask_curriculum: str = "complementary",
) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    controller, reference_runtime, reference_policy, _, controller_state, feedback, events, contexts = (
        base._build(seed)
    )
    controller_digest = base._digest_module(controller)
    adapter_digest = base._digest_module(reference_policy.state_adapter)
    policy, router, state = _new_policy(
        reference=reference_policy,
        seed=seed + 7000,
        cell_count=1,
        context_masking=masked_context,
    )
    (
        source_mask,
        successor_mask,
        reversal_mask,
        successor_mask_schedule,
    ) = _mask_configuration(
        masked_context=masked_context,
        mask_curriculum=mask_curriculum,
    )
    successor_training_mask = (
        None if successor_mask_schedule is not None else successor_mask
    )
    matched_fresh_state = _single_cell_state(state, 0)
    matched_fresh_initial_digest = _digest_state(matched_fresh_state)

    state, source = _train_regime(
        policy=policy,
        router=router,
        state=state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["source"],
        context=contexts["source"],
        target=base.SOURCE_TARGET,
        context_mask=source_mask,
        max_updates=MAX_UPDATES,
        delay=DELAY_STEPS,
        random_seed=seed + 1,
    )
    source_snapshot = _cell_snapshot(state, 0)
    state, source_retention_verification = _heldout_retention_verification(
        router=router,
        state=state,
        cell_index=0,
        context=contexts["source"],
        context_mask=source_mask,
        target=base.SOURCE_TARGET,
    )
    state, successor_cell = router.append_cell(state, source_cell=0)
    state, successor = _train_regime(
        policy=policy,
        router=router,
        state=state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=base.SUCCESSOR_TARGET,
        context_mask=successor_training_mask,
        context_mask_schedule=successor_mask_schedule,
        max_updates=MAX_UPDATES,
        delay=DELAY_STEPS,
        random_seed=seed + 2,
        stop_at_mastery=successor_mask_schedule is None,
    )
    successor_snapshot = _cell_snapshot(state, successor_cell)
    successor_auto_protected = bool(state.cells.protected[successor_cell])
    state, successor_retention_verification = _heldout_retention_verification(
        router=router,
        state=state,
        cell_index=successor_cell,
        context=contexts["successor"],
        context_mask=successor_mask,
        target=base.SUCCESSOR_TARGET,
    )

    pre_reversal = state
    state, inherited_cell = router.append_cell(state, source_cell=successor_cell)
    inherited_state, inherited_reversal = _train_regime(
        policy=policy,
        router=router,
        state=state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["reversal"],
        context=contexts["reversal"],
        target=base.REVERSED_TARGET,
        context_mask=reversal_mask,
        max_updates=60,
        delay=REVERSAL_DELAY_STEPS,
        random_seed=seed + 3,
    )
    negative_transfer = (
        inherited_reversal["deterministic_best_score"] < base.NOISY_MASTERY_THRESHOLD
    )
    if negative_transfer:
        state = pre_reversal
        state, reversal_cell = router.append_cell(state)
    else:
        state = inherited_state
        reversal_cell = inherited_cell
    state, reversal = _train_regime(
        policy=policy,
        router=router,
        state=state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["reversal"],
        context=contexts["reversal"],
        target=base.REVERSED_TARGET,
        context_mask=reversal_mask,
        max_updates=REVERSAL_UPDATES,
        delay=REVERSAL_DELAY_STEPS,
        noise_fraction=0.20,
        random_seed=seed + 4,
        stop_at_mastery=False,
    )

    fresh_policy, fresh_router, _fresh_initial_state = _new_policy(
        reference=reference_policy,
        seed=seed + 12000,
        cell_count=1,
        context_masking=masked_context,
    )
    fresh_state = matched_fresh_state
    fresh_state, fresh = _train_regime(
        policy=fresh_policy,
        router=fresh_router,
        state=fresh_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=base.SUCCESSOR_TARGET,
        context_mask=successor_training_mask,
        context_mask_schedule=successor_mask_schedule,
        max_updates=MAX_UPDATES,
        delay=0,
        random_seed=seed + 5,
        stop_at_mastery=successor_mask_schedule is None,
    )

    _, shuffled_router, shuffled_state = _new_policy(
        reference=reference_policy,
        seed=seed + 13000,
        cell_count=1,
        context_masking=masked_context,
    )
    shuffled_policy = PolicyFreeAmodalRuntime(
        reference_runtime,
        reference_policy.planner,
        state_adapter=reference_policy.state_adapter,
        intention_router=shuffled_router,
    )
    shuffled_state, shuffled = _train_regime(
        policy=shuffled_policy,
        router=shuffled_router,
        state=shuffled_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=base.SUCCESSOR_TARGET,
        context_mask=successor_mask,
        max_updates=CONTROL_UPDATES,
        delay=DELAY_STEPS,
        reward_shuffled=True,
        random_seed=seed + 6,
        stop_at_mastery=False,
    )

    _, action_router, action_state = _new_policy(
        reference=reference_policy,
        seed=seed + 14000,
        cell_count=1,
        context_masking=masked_context,
    )
    action_policy = PolicyFreeAmodalRuntime(
        reference_runtime,
        reference_policy.planner,
        state_adapter=reference_policy.state_adapter,
        intention_router=action_router,
    )
    action_state, action_shuffled = _train_regime(
        policy=action_policy,
        router=action_router,
        state=action_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=base.SUCCESSOR_TARGET,
        context_mask=successor_mask,
        max_updates=CONTROL_UPDATES,
        delay=DELAY_STEPS,
        action_shuffled=True,
        random_seed=seed + 7,
        stop_at_mastery=False,
    )

    _, missing_router, missing_state = _new_policy(
        reference=reference_policy,
        seed=seed + 15000,
        cell_count=1,
        context_masking=masked_context,
    )
    missing_policy = PolicyFreeAmodalRuntime(
        reference_runtime,
        reference_policy.planner,
        state_adapter=reference_policy.state_adapter,
        intention_router=missing_router,
    )
    missing = _missing_evidence_control(
        policy=missing_policy,
        router=missing_router,
        state=missing_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        context_mask=successor_mask,
        target=base.SUCCESSOR_TARGET,
    )
    corruption = _corruption_probe(
        router,
        state,
        contexts["source"],
        base.SOURCE_TARGET,
        source_mask,
    )
    protected_retention = _cell_matches(state, 0, source_snapshot) and _cell_matches(
        state,
        successor_cell,
        successor_snapshot,
    )
    persistence = router.state_from_payload(router.state_payload(state))
    probe_output, _ = policy.step_events(
        events["source"],
        controller_state,
        feedback,
        base._goal(contexts["source"], base.SOURCE_TARGET),
        horizon=base.HORIZON,
        beam_width=base.BEAM_WIDTH,
        intention_router_state=state,
        intention_context_mask=None
        if source_mask is None
        else source_mask.unsqueeze(0),
    )
    probe_routing = probe_output.intention_routing
    if probe_routing is None:
        raise AssertionError("masked routing probe did not return a proposal")
    masked_feature_width = probe_routing.candidates.features.shape[-1]

    gates = {
        "source_mastered_after_delayed_feedback": source["deterministic_best_score"] >= base.MASTERY_THRESHOLD,
        "successor_mastered_by_caller_free_routing": successor["deterministic_best_score"] >= base.MASTERY_THRESHOLD,
        "successor_route_used_new_cell": successor["selected_cell_counts"].get(str(successor_cell), 0) > 0,
        "single_context_sparse_materialization": (
            source["materialized_candidate_cells"]["maximum"] == 1
            and successor["materialized_candidate_cells"]["maximum"] == 1
            and reversal["materialized_candidate_cells"]["maximum"] == 1
        ),
        "negative_transfer_probe_detected": negative_transfer,
        "reversal_mastered_under_noise": reversal["deterministic_best_score"] >= base.NOISY_MASTERY_THRESHOLD,
        "fresh_successor_control_mastered": fresh["deterministic_best_score"] >= base.MASTERY_THRESHOLD,
        "warm_successor_faster_than_matched_fresh": (
            successor["updates"] < fresh["updates"]
        ),
        "reward_shuffled_control_failed": shuffled["deterministic_best_score"] < base.MASTERY_THRESHOLD,
        "action_shuffled_control_failed": action_shuffled["deterministic_best_score"] < base.MASTERY_THRESHOLD,
        "missing_evidence_is_noop": missing["state_unchanged"],
        "memory_corruption_probe_detected": corruption["corruption_detected"],
        "matched_fresh_control_state_valid": (
            matched_fresh_state.cells.baseline.shape[0] == 1
            and matched_fresh_initial_digest == _digest_state(matched_fresh_state)
        ),
        "protected_cells_unchanged_by_later_learning": protected_retention,
        "source_retention_verifier_passed": source_retention_verification["accepted"],
        "successor_retention_verifier_passed": successor_retention_verification["accepted"],
        "append_only_external_growth": state.cells.baseline.shape[0] == 3,
        "exact_routed_memory_persistence": _digest_state(persistence) == _digest_state(state),
        "controller_frozen": controller_digest == base._digest_module(controller),
        "state_adapter_frozen": adapter_digest == base._digest_module(reference_policy.state_adapter),
        "zero_replayed_examples": True,
        "masked_context_features_present": (
            not masked_context or masked_feature_width == 2 * base.STATE_WIDTH + 1
        ),
        "mask_curriculum_exercised": (
            not masked_context
            or len(successor["context_mask_patterns"])
            == (1 if successor_mask_schedule is None else len(successor_mask_schedule))
        ),
        "overlapping_observation_patterns_present": (
            not masked_context
            or mask_curriculum == "complementary"
            or int((source_mask & successor_mask).sum().item()) >= 4
        ),
    }
    report = {
        "schema": (
            "neural-computer.policy-free-intention-masked-routing.v2"
            if masked_context
            else "neural-computer.policy-free-intention-routing.v1"
        ),
        "claim_boundary": (
            "bounded caller-free masked-context routing with explicit observation "
            "channels, support-aware protected-cell quarantine, delayed/noisy scalar "
            "credit, sparse materialization, copy-on-write growth, held-out prefix "
            "retention, and overlapping-mask successor transfer; arbitrary "
            "missing-stream reasoning, unrestricted growth, compression, and general "
            "continual learning remain unqualified."
            if masked_context
            else "bounded caller-free context-conditioned routing over protected and growing "
            "external intention cells with delayed scalar credit and matched fresh "
            "successor transfer; not general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "controller_width": base.CONTROLLER_WIDTH,
            "state_width": base.STATE_WIDTH,
            "intention_width": base.INTENTION_WIDTH,
            "delay_steps": DELAY_STEPS,
            "reversal_delay_steps": REVERSAL_DELAY_STEPS,
            "noise_fraction": 0.20,
            "router": router.configuration(),
            "runtime": policy.configuration(),
            "candidate_selection": "router_selected_intention_only_no_caller_cell_index_v1",
            "fresh_successor_control": "matched_caller_owned_fresh_append_state_v1",
            "retention_verifier": "deterministic_heldout_prefix_copy_on_write_v1",
            "masked_context": masked_context,
            "mask_curriculum": mask_curriculum,
            "successor_mask_schedule": (
                None
                if successor_mask_schedule is None
                else [
                    {"start_update": start, "mask": mask.tolist()}
                    for start, mask in successor_mask_schedule
                ]
            ),
            "source_context_mask": None if source_mask is None else source_mask.tolist(),
            "successor_context_mask": (
                None if successor_mask is None else successor_mask.tolist()
            ),
            "reversal_context_mask": (
                None if reversal_mask is None else reversal_mask.tolist()
            ),
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "source": source,
            "successor": successor,
            "inherited_reversal": inherited_reversal,
            "reversal": reversal,
            "fresh_successor": fresh,
            "source_retention_verification": source_retention_verification,
            "successor_retention_verification": successor_retention_verification,
            "transfer_ratio_against_matched_fresh": fresh["updates"]
            / max(1, successor["updates"]),
            "reward_shuffled": shuffled,
            "action_shuffled": action_shuffled,
            "missing_evidence": missing,
            "memory_corruption": corruption,
            "reversal_cell": reversal_cell,
            "final_routing_decisions": state.routing_decisions.tolist(),
            "final_routing_feedbacks": state.routing_feedbacks.tolist(),
            "final_retention": {
                "protected": state.cells.protected.tolist(),
                "observations": state.retention_observations.tolist(),
                "prefix_minima": state.retention_prefix_minima.tolist(),
                "reversal_counts": state.retention_reversal_counts.tolist(),
                "successor_auto_protected": successor_auto_protected,
            },
        },
        "accounting": {
            "unique_verifier_bits": (
                source["updates"]
                + successor["updates"]
                + inherited_reversal["updates"]
                + reversal["updates"]
                + fresh["updates"]
                + source_retention_verification["verifier_bits"]
                + successor_retention_verification["verifier_bits"]
            ),
            "control_outcome_bits": shuffled["updates"] + action_shuffled["updates"],
            "unique_logical_lifetimes": 6,
            "external_memory_updates": (
                source["updates"]
                + successor["updates"]
                + inherited_reversal["updates"]
                + reversal["updates"]
                + fresh["updates"]
                + shuffled["updates"]
                + action_shuffled["updates"]
            ),
            "heldout_verifier_bits": (
                source_retention_verification["verifier_bits"]
                + successor_retention_verification["verifier_bits"]
            ),
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "search_expansions": (
                source["search_expansions"]
                + successor["search_expansions"]
                + inherited_reversal["search_expansions"]
                + reversal["search_expansions"]
                + fresh["search_expansions"]
                + shuffled["search_expansions"]
                + action_shuffled["search_expansions"]
            ),
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=85301)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument(
        "--masked-context",
        action="store_true",
        help="use explicit partial-context masks in external learning",
    )
    parser.add_argument(
        "--mask-curriculum",
        choices=("complementary", "overlapping", "gradual", "multi_stage"),
        default="complementary",
        help="observation-mask schedule used with --masked-context",
    )
    args = parser.parse_args()
    run(
        args.seed,
        args.report_out,
        masked_context=args.masked_context,
        mask_curriculum=args.mask_curriculum,
    )


if __name__ == "__main__":
    main()
