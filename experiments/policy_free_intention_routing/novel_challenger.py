"""Test verifier-selected copy versus fresh external intention state.

The controller and state adapter are frozen. A mastered sequential memory is
given an unseen evidence combination and an unseen intention target. The
router creates isolated copied and fresh candidates, lets each receive a
bounded outcome-only probe, and commits only the higher-scoring branch. This
is the external-memory analogue of a CPU deciding whether a new file should
inherit a prior program or start from an empty file.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from experiments.policy_free_intention_memory import train as base
from experiments.policy_free_intention_routing import train as routed
from neural_computer import PolicyFreeAmodalRuntime

CHALLENGER_PROBE_UPDATES = 12
CHALLENGER_MIN_PROBE_MARGIN = 0.05
NOVEL_MAX_UPDATES = 240


@dataclass(frozen=True)
class ChallengerTask:
    """A verifier-only novel task used to test prior admission decisions."""

    task_id: str
    context_mask: torch.Tensor
    target: torch.Tensor
    expected_initialization: str
    probe_direction: str
    report_schema: str
    claim_boundary: str

    def validate(self) -> ChallengerTask:
        if not self.task_id:
            raise ValueError("challenger task id is required")
        if self.context_mask.shape != (base.STATE_WIDTH,):
            raise ValueError("challenger task mask has the wrong shape")
        if self.context_mask.dtype != torch.bool:
            raise ValueError("challenger task mask must be boolean")
        if self.target.shape != (base.INTENTION_WIDTH,):
            raise ValueError("challenger task target has the wrong shape")
        if self.expected_initialization not in {"transfer", "fresh"}:
            raise ValueError("challenger task initialization expectation is invalid")
        if self.probe_direction not in {"transfer", "fresh"}:
            raise ValueError("challenger task probe direction is invalid")
        if not self.report_schema or not self.claim_boundary:
            raise ValueError("challenger task report metadata is required")
        return self


HARMFUL_NOVEL_TASK = ChallengerTask(
    task_id="unseen_target_blind_copy_negative_transfer",
    context_mask=torch.tensor(
        [True, False, True, True, False, True, True, False, False, True, False, True]
    ),
    target=torch.tensor([-0.25, 0.35]),
    expected_initialization="fresh",
    probe_direction="fresh",
    report_schema="neural-computer.policy-free-intention-novel-challenger.v1",
    claim_boundary=(
        "bounded verifier-selected copy-or-fresh external intention admission "
        "for one unseen evidence combination and target after a known adaptive "
        "sequence; general distribution shift and unrestricted continual learning "
        "remain unqualified"
    ),
).validate()

# Backward-compatible names for small external probes that used the original
# single-task harness before task configurations became explicit.
NOVEL_CONTEXT_MASK = HARMFUL_NOVEL_TASK.context_mask
NOVEL_TARGET = HARMFUL_NOVEL_TASK.target


def _policy_with_router(
    reference: routed.PolicyFreeAmodalRuntime,
    router: routed.ExternalOutcomeIntentionRouter,
) -> routed.PolicyFreeAmodalRuntime:
    return PolicyFreeAmodalRuntime(
        reference.runtime,
        reference.planner,
        state_adapter=reference.state_adapter,
        intention_router=router,
    )


def _score(
    router: routed.ExternalOutcomeIntentionRouter,
    state: routed.ExternalRoutedIntentionMemoryState,
    cell_index: int,
    context: torch.Tensor,
    task: ChallengerTask,
) -> float:
    mean = router.mean(
        state,
        context,
        context_mask=task.context_mask.unsqueeze(0),
    )[0, cell_index].unsqueeze(0)
    return float(base._utility(mean, task.target).item())


def _cell_corruption_probe(
    router: routed.ExternalOutcomeIntentionRouter,
    state: routed.ExternalRoutedIntentionMemoryState,
    cell_index: int,
    context: torch.Tensor,
    task: ChallengerTask,
) -> dict[str, object]:
    clean_score = _score(router, state, cell_index, context, task)
    corrupted_cells = replace(
        state.cells,
        output_weights=state.cells.output_weights.clone(),
        output_bias=state.cells.output_bias.clone(),
        context_residual_weights=state.cells.context_residual_weights.clone(),
    )
    corrupted_cells.output_weights[cell_index].zero_()
    corrupted_cells.output_bias[cell_index].zero_()
    corrupted_cells.context_residual_weights[cell_index].zero_()
    corrupted_state = replace(state, cells=corrupted_cells)
    corrupted_mean = router.mean(
        corrupted_state,
        context,
        context_mask=task.context_mask.unsqueeze(0),
    )[0, cell_index].unsqueeze(0)
    corrupted_score = float(base._utility(corrupted_mean, task.target).item())
    return {
        "clean_score": clean_score,
        "corrupted_score": corrupted_score,
        "corruption_detected": corrupted_score < clean_score,
    }


def _run_causal_controls(
    seed: int,
    *,
    fresh: bool,
    task: ChallengerTask,
    reference_policy: routed.PolicyFreeAmodalRuntime,
    controller_state,
    feedback,
    event,
    context: torch.Tensor,
) -> dict[str, object]:
    def setup(offset: int):
        policy, router, state = routed._new_policy(
            reference=reference_policy,
            seed=seed + offset + (12000 if fresh else 7000),
            cell_count=1,
            context_masking=True,
            unqualified_cell_probability=0.75,
            context_mask_profile_scale=20.0,
            mask_stable_content=True,
            factorized_context_residual=True,
        )
        return policy, router, state

    reward_policy, reward_router, reward_state = setup(610)
    reward_initial_score = _score(reward_router, reward_state, 0, context, task)
    reward_state, reward_report = routed._train_regime(
        policy=reward_policy,
        router=reward_router,
        state=reward_state,
        controller_state=controller_state,
        feedback=feedback,
        event=event,
        context=context,
        target=task.target,
        context_mask=task.context_mask,
        max_updates=routed.CONTROL_UPDATES,
        delay=routed.DELAY_STEPS,
        reward_shuffled=True,
        random_seed=seed + 611,
        stop_at_mastery=False,
    )
    reward_report = dict(reward_report)
    reward_report["initial_score"] = reward_initial_score
    action_policy, action_router, action_state = setup(620)
    action_initial_score = _score(action_router, action_state, 0, context, task)
    action_state, action_report = routed._train_regime(
        policy=action_policy,
        router=action_router,
        state=action_state,
        controller_state=controller_state,
        feedback=feedback,
        event=event,
        context=context,
        target=task.target,
        context_mask=task.context_mask,
        max_updates=routed.CONTROL_UPDATES,
        delay=routed.DELAY_STEPS,
        action_shuffled=True,
        random_seed=seed + 621,
        stop_at_mastery=False,
    )
    action_report = dict(action_report)
    action_report["initial_score"] = action_initial_score
    missing_policy, missing_router, missing_state = setup(630)
    missing = routed._missing_evidence_control(
        policy=missing_policy,
        router=missing_router,
        state=missing_state,
        controller_state=controller_state,
        feedback=feedback,
        event=event,
        context=context,
        target=task.target,
        context_mask=task.context_mask,
    )
    return {
        "reward_shuffled": reward_report,
        "action_shuffled": action_report,
        "missing_evidence": missing,
    }


def _prepare(
    seed: int,
    *,
    fresh: bool,
) -> dict[str, object]:
    torch.set_num_threads(1)
    (
        controller,
        reference_runtime,
        reference_policy,
        _,
        controller_state,
        feedback,
        events,
        contexts,
    ) = base._build(seed)
    policy, router, state = routed._new_policy(
        reference=reference_policy,
        seed=seed + (12000 if fresh else 7000),
        cell_count=1,
        context_masking=True,
        unqualified_cell_probability=0.75,
        context_mask_profile_scale=20.0,
        mask_stable_content=True,
        factorized_context_residual=True,
    )
    source_mask, _, _, schedule = routed._mask_configuration(
        masked_context=True,
        mask_curriculum="adaptive_versioned_multi_stage",
    )
    state, source = routed._train_regime(
        policy=policy,
        router=router,
        state=state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["source"],
        context=contexts["source"],
        target=base.SOURCE_TARGET,
        context_mask=source_mask,
        max_updates=routed.MAX_UPDATES,
        delay=0 if fresh else routed.DELAY_STEPS,
        random_seed=seed + 1,
    )
    state, source_retention = routed._heldout_retention_verification(
        router=router,
        state=state,
        cell_index=0,
        context=contexts["source"],
        context_mask=source_mask,
        target=base.SOURCE_TARGET,
    )
    state, successor_cell = router.append_cell(state, source_cell=0)
    state, successor = routed._train_regime(
        policy=policy,
        router=router,
        state=state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=base.SUCCESSOR_TARGET,
        context_mask_schedule=schedule,
        max_updates=routed.MAX_UPDATES,
        delay=0 if fresh else routed.DELAY_STEPS,
        random_seed=seed + (5 if fresh else 2),
        stop_at_mastery=True,
        mastery_cell=successor_cell,
        fork_on_mask_change=True,
        adaptive_stage_mastery=True,
        adaptive_stage_min_updates=4,
    )
    successor_cell = int(successor["mastery_cell"])
    state, successor_retention = routed._heldout_retention_verification(
        router=router,
        state=state,
        cell_index=successor_cell,
        context=contexts["successor"],
        context_mask=routed.OVERLAP_SUCCESSOR_CONTEXT_MASK,
        target=base.SUCCESSOR_TARGET,
    )
    return {
        "controller": controller,
        "reference_runtime": reference_runtime,
        "reference_policy": reference_policy,
        "policy": policy,
        "router": router,
        "state": state,
        "controller_state": controller_state,
        "feedback": feedback,
        "events": events,
        "contexts": contexts,
        "successor_cell": successor_cell,
        "source": source,
        "successor": successor,
        "source_retention": source_retention,
        "successor_retention": successor_retention,
    }


def _run_challenger(
    seed: int,
    *,
    fresh: bool,
    task: ChallengerTask,
) -> dict[str, object]:
    prepared = _prepare(seed, fresh=fresh)
    router = prepared["router"]
    state = prepared["state"]
    reference_policy = prepared["reference_policy"]
    controller_state = prepared["controller_state"]
    feedback = prepared["feedback"]
    events = prepared["events"]
    contexts = prepared["contexts"]
    source_cell = prepared["successor_cell"]
    source_digest = router._state_digest(state)
    probe_records: dict[str, dict[str, object]] = {}

    def probe(
        transfer_router,
        transfer_state,
        transfer_cell,
        fresh_router,
        fresh_state,
        fresh_cell,
    ):
        torch.manual_seed(seed + (201 if fresh else 101))
        transfer_policy = _policy_with_router(reference_policy, transfer_router)
        transfer_state, transfer_report = routed._train_regime(
            policy=transfer_policy,
            router=transfer_router,
            state=transfer_state,
            controller_state=controller_state,
            feedback=feedback,
            event=events["successor"],
            context=contexts["successor"],
            target=task.target,
            context_mask=task.context_mask,
            max_updates=CHALLENGER_PROBE_UPDATES,
            delay=routed.DELAY_STEPS,
            random_seed=seed + (202 if fresh else 102),
            stop_at_mastery=False,
            mastery_cell=transfer_cell,
        )
        torch.manual_seed(seed + 301)
        fresh_policy = _policy_with_router(reference_policy, fresh_router)
        fresh_state, fresh_report = routed._train_regime(
            policy=fresh_policy,
            router=fresh_router,
            state=fresh_state,
            controller_state=controller_state,
            feedback=feedback,
            event=events["successor"],
            context=contexts["successor"],
            target=task.target,
            context_mask=task.context_mask,
            max_updates=CHALLENGER_PROBE_UPDATES,
            delay=routed.DELAY_STEPS,
            random_seed=seed + 302,
            stop_at_mastery=False,
            mastery_cell=fresh_cell,
        )
        transfer_score = _score(
            transfer_router,
            transfer_state,
            transfer_cell,
            contexts["successor"],
            task,
        )
        fresh_score = _score(
            fresh_router,
            fresh_state,
            fresh_cell,
            contexts["successor"],
            task,
        )
        probe_records.update(
            {
                "transfer": transfer_report,
                "fresh": fresh_report,
            }
        )
        return transfer_score, fresh_score, transfer_state, fresh_state

    torch.manual_seed(seed + 401)
    receipt, selected_router, selected_state, selected_cell = router.select_verified_transfer_prior(
        state,
        source_cell,
        probe,
        context_mask=task.context_mask,
        probe_updates=CHALLENGER_PROBE_UPDATES,
    )
    selected_policy = _policy_with_router(reference_policy, selected_router)
    torch.manual_seed(seed + 501)
    selected_state, continuation = routed._train_regime(
        policy=selected_policy,
        router=selected_router,
        state=selected_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
        target=task.target,
        context_mask=task.context_mask,
        max_updates=NOVEL_MAX_UPDATES,
        delay=routed.DELAY_STEPS,
        random_seed=seed + 502,
        stop_at_mastery=True,
        mastery_cell=selected_cell,
    )
    selected_state, retention = routed._heldout_retention_verification(
        router=selected_router,
        state=selected_state,
        cell_index=selected_cell,
        context=contexts["successor"],
        context_mask=task.context_mask,
        target=task.target,
    )
    controls = _run_causal_controls(
        seed,
        fresh=fresh,
        task=task,
        reference_policy=reference_policy,
        controller_state=controller_state,
        feedback=feedback,
        event=events["successor"],
        context=contexts["successor"],
    )
    corruption = _cell_corruption_probe(
        selected_router,
        selected_state,
        selected_cell,
        contexts["successor"],
        task,
    )
    reversal_state, reversal_cell = selected_router.append_cell(
        selected_state,
        source_cell=selected_cell,
        copy_route=False,
        context_mask=routed.OVERLAP_REVERSAL_CONTEXT_MASK,
    )
    reversal_policy = _policy_with_router(reference_policy, selected_router)
    reversal_state, reversal = routed._train_regime(
        policy=reversal_policy,
        router=selected_router,
        state=reversal_state,
        controller_state=controller_state,
        feedback=feedback,
        event=events["reversal"],
        context=contexts["reversal"],
        target=base.REVERSED_TARGET,
        context_mask=routed.OVERLAP_REVERSAL_CONTEXT_MASK,
        max_updates=routed.REVERSAL_UPDATES,
        delay=routed.REVERSAL_DELAY_STEPS,
        noise_fraction=0.20,
        random_seed=seed + 603,
        stop_at_mastery=False,
        mastery_cell=reversal_cell,
    )
    reversal_state, reversal_retention = routed._heldout_retention_verification(
        router=selected_router,
        state=reversal_state,
        cell_index=reversal_cell,
        context=contexts["reversal"],
        context_mask=routed.OVERLAP_REVERSAL_CONTEXT_MASK,
        target=base.REVERSED_TARGET,
    )
    _, novel_after_reversal_retention = routed._heldout_retention_verification(
        router=selected_router,
        state=reversal_state,
        cell_index=selected_cell,
        context=contexts["successor"],
        context_mask=task.context_mask,
        target=task.target,
    )
    source_unchanged = router._state_digest(state) == source_digest
    persistence = selected_router.state_from_payload(
        selected_router.state_payload(selected_state)
    )
    return {
        "fresh_control": fresh,
        "receipt": receipt,
        "probe": probe_records,
        "continuation": continuation,
        "retention": retention,
        "controls": controls,
        "corruption": corruption,
        "reversal": reversal,
        "reversal_retention": reversal_retention,
        "novel_after_reversal_retention": novel_after_reversal_retention,
        "source_unchanged": source_unchanged,
        "persistence_exact": (
            selected_router._state_digest(persistence)
            == selected_router._state_digest(selected_state)
        ),
        "controller_digest": base._digest_module(prepared["controller"]),
        "adapter_digest": base._digest_module(reference_policy.state_adapter),
        "selected_cell": selected_cell,
        "total_updates": CHALLENGER_PROBE_UPDATES + continuation["updates"],
        "unique_verifier_bits": (
            prepared["source"]["updates"]
            + prepared["successor"]["updates"]
            + prepared["source_retention"]["verifier_bits"]
            + prepared["successor_retention"]["verifier_bits"]
            + 2 * CHALLENGER_PROBE_UPDATES
            + continuation["updates"]
            + retention["verifier_bits"]
            + reversal["updates"]
            + reversal_retention["verifier_bits"]
            + novel_after_reversal_retention["verifier_bits"]
        ),
        "control_outcome_bits": (
            controls["reward_shuffled"]["updates"]
            + controls["action_shuffled"]["updates"]
        ),
        "known_successor": prepared["successor"],
        "known_source_retention": prepared["source_retention"],
        "known_successor_retention": prepared["successor_retention"],
        "selected_router": selected_router,
        "selected_state": selected_state,
    }


def run(
    seed: int,
    report_out: Path,
    *,
    task: ChallengerTask = HARMFUL_NOVEL_TASK,
) -> dict[str, object]:
    task.validate()
    begun = time.perf_counter()
    warm = _run_challenger(seed, fresh=False, task=task)
    fresh = _run_challenger(seed, fresh=True, task=task)
    controller_frozen = (
        warm["controller_digest"] == fresh["controller_digest"]
    )
    adapter_frozen = warm["adapter_digest"] == fresh["adapter_digest"]
    source_mask, successor_mask, reversal_mask, known_schedule = routed._mask_configuration(
        masked_context=True,
        mask_curriculum="adaptive_versioned_multi_stage",
    )
    known_masks = [source_mask, successor_mask, reversal_mask]
    if known_schedule is not None:
        known_masks.extend(mask for _, mask in known_schedule)
    task_mask_unseen = all(
        not torch.equal(task.context_mask, known_mask) for known_mask in known_masks
    )
    task_target_unseen = all(
        not torch.equal(task.target, known_target)
        for known_target in (
            base.SOURCE_TARGET,
            base.SUCCESSOR_TARGET,
            base.REVERSED_TARGET,
        )
    )

    def probe_margin(receipt) -> float:
        return float(
            receipt.transfer_probe_score - receipt.fresh_probe_score
            if task.probe_direction == "transfer"
            else receipt.fresh_probe_score - receipt.transfer_probe_score
        )

    gates = {
        "task_mask_is_unseen": task_mask_unseen,
        "task_target_is_unseen": task_target_unseen,
        "warm_known_sequence_mastered": warm["known_successor"]["stage_mastery_complete"],
        "fresh_known_sequence_mastered": fresh["known_successor"]["stage_mastery_complete"],
        "warm_source_retained": warm["known_source_retention"]["accepted"],
        "fresh_source_retained": fresh["known_source_retention"]["accepted"],
        "warm_known_successor_retained": warm["known_successor_retention"]["accepted"],
        "fresh_known_successor_retained": fresh["known_successor_retention"]["accepted"],
        "warm_candidate_selection_matches_task": (
            warm["receipt"].selected_initialization
            == task.expected_initialization
        ),
        "fresh_candidate_selection_matches_task": (
            fresh["receipt"].selected_initialization
            == task.expected_initialization
        ),
        "warm_probe_order_matches_task": (
            warm["receipt"].transfer_probe_score
            > warm["receipt"].fresh_probe_score
            if task.probe_direction == "transfer"
            else warm["receipt"].transfer_probe_score
            < warm["receipt"].fresh_probe_score
        ),
        "fresh_probe_order_matches_task": (
            fresh["receipt"].transfer_probe_score
            > fresh["receipt"].fresh_probe_score
            if task.probe_direction == "transfer"
            else fresh["receipt"].transfer_probe_score
            < fresh["receipt"].fresh_probe_score
        ),
        "warm_probe_margin_material": (
            probe_margin(warm["receipt"]) >= CHALLENGER_MIN_PROBE_MARGIN
        ),
        "fresh_probe_margin_material": (
            probe_margin(fresh["receipt"]) >= CHALLENGER_MIN_PROBE_MARGIN
        ),
        "warm_novel_mastered": warm["continuation"]["deterministic_best_score"] >= base.MASTERY_THRESHOLD,
        "fresh_novel_mastered": fresh["continuation"]["deterministic_best_score"] >= base.MASTERY_THRESHOLD,
        "warm_novel_retained": warm["retention"]["accepted"],
        "fresh_novel_retained": fresh["retention"]["accepted"],
        "warm_reward_shuffled_control_not_sample_efficient": (
            warm["controls"]["reward_shuffled"]["deterministic_best_score"]
            < base.MASTERY_THRESHOLD
            or warm["controls"]["reward_shuffled"]["updates"]
            > 4 * warm["continuation"]["updates"]
        ),
        "fresh_reward_shuffled_control_not_sample_efficient": (
            fresh["controls"]["reward_shuffled"]["deterministic_best_score"]
            < base.MASTERY_THRESHOLD
            or fresh["controls"]["reward_shuffled"]["updates"]
            > 4 * fresh["continuation"]["updates"]
        ),
        "warm_action_shuffled_control_not_sample_efficient": (
            warm["controls"]["action_shuffled"]["deterministic_best_score"]
            < base.MASTERY_THRESHOLD
            or warm["controls"]["action_shuffled"]["updates"]
            > 4 * warm["continuation"]["updates"]
        ),
        "fresh_action_shuffled_control_not_sample_efficient": (
            fresh["controls"]["action_shuffled"]["deterministic_best_score"]
            < base.MASTERY_THRESHOLD
            or fresh["controls"]["action_shuffled"]["updates"]
            > 4 * fresh["continuation"]["updates"]
        ),
        "warm_missing_evidence_is_noop": warm["controls"]["missing_evidence"][
            "state_unchanged"
        ],
        "fresh_missing_evidence_is_noop": fresh["controls"]["missing_evidence"][
            "state_unchanged"
        ],
        "warm_memory_corruption_detected": warm["corruption"][
            "corruption_detected"
        ],
        "fresh_memory_corruption_detected": fresh["corruption"][
            "corruption_detected"
        ],
        "warm_novel_survived_reversal": warm["novel_after_reversal_retention"][
            "accepted"
        ],
        "fresh_novel_survived_reversal": fresh[
            "novel_after_reversal_retention"
        ]["accepted"],
        "challenger_did_not_mutate_warm_source": warm["source_unchanged"],
        "challenger_did_not_mutate_fresh_source": fresh["source_unchanged"],
        "warm_persistence_exact": warm["persistence_exact"],
        "fresh_persistence_exact": fresh["persistence_exact"],
        "controller_frozen": controller_frozen,
        "state_adapter_frozen": adapter_frozen,
        "zero_replayed_examples": True,
    }
    report = {
        "schema": task.report_schema,
        "claim_boundary": task.claim_boundary,
        "seed": seed,
        "configuration": {
            "task_id": task.task_id,
            "novel_context_mask": task.context_mask.tolist(),
            "novel_target": task.target.tolist(),
            "expected_initialization": task.expected_initialization,
            "probe_direction": task.probe_direction,
            "minimum_probe_margin": CHALLENGER_MIN_PROBE_MARGIN,
            "probe_updates": CHALLENGER_PROBE_UPDATES,
            "max_continuation_updates": NOVEL_MAX_UPDATES,
            "known_curriculum": "adaptive_versioned_multi_stage",
            "known_stage_min_updates": 4,
            "candidate_selection": "outcome_only_isolated_copy_or_fresh_v1",
            "replayed_examples": 0,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "warm": {
                "selected_initialization": warm["receipt"].selected_initialization,
                "transfer_probe_score": warm["receipt"].transfer_probe_score,
                "fresh_probe_score": warm["receipt"].fresh_probe_score,
                "probe_margin": probe_margin(warm["receipt"]),
                "total_updates": warm["total_updates"],
                "continuation": warm["continuation"],
                "retention": warm["retention"],
                "controls": warm["controls"],
                "corruption": warm["corruption"],
                "reversal": warm["reversal"],
                "reversal_retention": warm["reversal_retention"],
                "novel_after_reversal_retention": warm[
                    "novel_after_reversal_retention"
                ],
            },
            "fresh": {
                "selected_initialization": fresh["receipt"].selected_initialization,
                "transfer_probe_score": fresh["receipt"].transfer_probe_score,
                "fresh_probe_score": fresh["receipt"].fresh_probe_score,
                "probe_margin": probe_margin(fresh["receipt"]),
                "total_updates": fresh["total_updates"],
                "continuation": fresh["continuation"],
                "retention": fresh["retention"],
                "controls": fresh["controls"],
                "corruption": fresh["corruption"],
                "reversal": fresh["reversal"],
                "reversal_retention": fresh["reversal_retention"],
                "novel_after_reversal_retention": fresh[
                    "novel_after_reversal_retention"
                ],
            },
            "warm_receipt": warm["receipt"].__dict__,
            "fresh_receipt": fresh["receipt"].__dict__,
            "warm_probe": warm["probe"],
            "fresh_probe": fresh["probe"],
        },
        "accounting": {
            "warm_unique_verifier_bits": warm["unique_verifier_bits"],
            "fresh_unique_verifier_bits": fresh["unique_verifier_bits"],
            "warm_control_outcome_bits": warm["control_outcome_bits"],
            "fresh_control_outcome_bits": fresh["control_outcome_bits"],
            "warm_replayed_examples": 0,
            "fresh_replayed_examples": 0,
            "optimizer_updates": 0,
            "controller_optimizer_updates": 0,
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
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
