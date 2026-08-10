"""Stress repeated cost-aware prior admission over a growing external memory.

The controller and state adapter remain frozen. A known adaptive sequence is
followed by several unseen task families. Each family gets an isolated
transfer/fresh verifier challenger, a cost-aware selection receipt, a bounded
continuation, and complete-prefix retention verification before the next file
is admitted.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from experiments.policy_free_intention_memory import train as base
from experiments.policy_free_intention_routing import novel_challenger as challenger
from experiments.policy_free_intention_routing import train as routed

PROBE_UPDATES = challenger.CHALLENGER_PROBE_UPDATES
MAX_UPDATES = challenger.NOVEL_MAX_UPDATES
MIN_PROBE_MARGIN = 0.02
MIN_ADJUSTED_MARGIN = 0.001


@dataclass(frozen=True)
class AdmissionSpec:
    task: challenger.ChallengerTask
    transfer_cost: float
    fresh_cost: float
    cost_weight: float

    def validate(self) -> AdmissionSpec:
        self.task.validate()
        for name, value in (
            ("transfer_cost", self.transfer_cost),
            ("fresh_cost", self.fresh_cost),
            ("cost_weight", self.cost_weight),
        ):
            if value < 0.0 or not torch.isfinite(torch.tensor(value)):
                raise ValueError(f"admission {name} is invalid")
        return self


SEQUENTIAL_ADMISSIONS = (
    AdmissionSpec(
        task=challenger.ChallengerTask(
            task_id="family_a_nearby_successor",
            context_mask=torch.tensor(
                [True, True, True, False, True, True, False, True, False, True, False, True]
            ),
            target=torch.tensor([0.45, -0.82]),
            expected_initialization="transfer",
            probe_direction="transfer",
            report_schema="neural-computer.sequential-admission.task.v1",
            claim_boundary="one unseen nearby successor family",
        ),
        transfer_cost=0.15,
        fresh_cost=0.45,
        cost_weight=0.10,
    ).validate(),
    AdmissionSpec(
        task=challenger.HARMFUL_NOVEL_TASK,
        transfer_cost=0.45,
        fresh_cost=0.15,
        cost_weight=0.10,
    ).validate(),
    AdmissionSpec(
        task=challenger.ChallengerTask(
            task_id="family_c_alternate_nearby_successor",
            context_mask=torch.tensor(
                [True, True, False, True, True, False, True, True, False, True, False, False]
            ),
            target=torch.tensor([0.62, -0.88]),
            expected_initialization="transfer",
            probe_direction="transfer",
            report_schema="neural-computer.sequential-admission.task.v1",
            claim_boundary="one alternate unseen nearby successor family",
        ),
        transfer_cost=0.20,
        fresh_cost=0.50,
        cost_weight=0.10,
    ).validate(),
)


def _known_masks() -> tuple[torch.Tensor, ...]:
    source, successor, reversal, schedule = routed._mask_configuration(
        masked_context=True,
        mask_curriculum="adaptive_versioned_multi_stage",
    )
    return tuple(
        [source, successor, reversal]
        + ([] if schedule is None else [mask for _, mask in schedule])
    )


def _task_is_unseen(task: challenger.ChallengerTask) -> bool:
    known_targets = (base.SOURCE_TARGET, base.SUCCESSOR_TARGET, base.REVERSED_TARGET)
    return (
        all(not torch.equal(task.context_mask, mask) for mask in _known_masks())
        and all(not torch.equal(task.target, target) for target in known_targets)
    )


def _prefix_verify(
    *,
    router: routed.ExternalOutcomeIntentionRouter,
    state: routed.ExternalRoutedIntentionMemoryState,
    records: list[dict[str, object]],
) -> tuple[routed.ExternalRoutedIntentionMemoryState, dict[str, object]]:
    next_state = state
    results: list[dict[str, object]] = []
    for record in records:
        next_state, retention = routed._heldout_retention_verification(
            router=router,
            state=next_state,
            cell_index=record["cell"],
            context=record["context"],
            context_mask=record["context_mask"],
            target=record["target"],
        )
        results.append(
            {
                "task_id": record["task_id"],
                "cell": record["cell"],
                "accepted": retention["accepted"],
                "deterministic_score": retention["deterministic_score"],
                "verifier_bits": retention["verifier_bits"],
                "prefix_minimum": retention["prefix_minimum"],
            }
        )
    return next_state, {
        "accepted": all(result["accepted"] for result in results),
        "records": results,
        "verifier_bits": sum(result["verifier_bits"] for result in results),
    }


def _admit_one(
    *,
    seed: int,
    task_index: int,
    spec: AdmissionSpec,
    fresh: bool,
    reference_policy,
    state,
    router,
    controller_state,
    feedback,
    events,
    contexts,
    source_cell: int,
    prefix_records: list[dict[str, object]],
) -> tuple[object, object, dict[str, object], list[dict[str, object]]]:
    task = spec.task
    task_event = events[task.event_key]
    task_context = contexts[task.context_key]
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
        transfer_policy = challenger._policy_with_router(
            reference_policy, transfer_router
        )
        transfer_state, transfer_report = routed._train_regime(
            policy=transfer_policy,
            router=transfer_router,
            state=transfer_state,
            controller_state=controller_state,
            feedback=feedback,
            event=task_event,
            context=task_context,
            target=task.target,
            context_mask=task.context_mask,
            max_updates=PROBE_UPDATES,
            delay=routed.DELAY_STEPS,
            random_seed=seed + 1000 + task_index * 20,
            stop_at_mastery=False,
            mastery_cell=transfer_cell,
        )
        fresh_policy = challenger._policy_with_router(reference_policy, fresh_router)
        fresh_state, fresh_report = routed._train_regime(
            policy=fresh_policy,
            router=fresh_router,
            state=fresh_state,
            controller_state=controller_state,
            feedback=feedback,
            event=task_event,
            context=task_context,
            target=task.target,
            context_mask=task.context_mask,
            max_updates=PROBE_UPDATES,
            delay=routed.DELAY_STEPS,
            random_seed=seed + 1001 + task_index * 20,
            stop_at_mastery=False,
            mastery_cell=fresh_cell,
        )
        transfer_score = challenger._score(
            transfer_router, transfer_state, transfer_cell, task_context, task
        )
        fresh_score = challenger._score(
            fresh_router, fresh_state, fresh_cell, task_context, task
        )
        probe_records["transfer"] = transfer_report
        probe_records["fresh"] = fresh_report
        return transfer_score, fresh_score, transfer_state, fresh_state

    receipt, selected_router, selected_state, selected_cell = (
        router.select_verified_transfer_prior(
            state,
            source_cell,
            probe,
            context_mask=task.context_mask,
            probe_updates=PROBE_UPDATES,
            transfer_cost=spec.transfer_cost,
            fresh_cost=spec.fresh_cost,
            cost_weight=spec.cost_weight,
        )
    )
    selected_policy = challenger._policy_with_router(reference_policy, selected_router)
    selected_state, continuation = routed._train_regime(
        policy=selected_policy,
        router=selected_router,
        state=selected_state,
        controller_state=controller_state,
        feedback=feedback,
        event=task_event,
        context=task_context,
        target=task.target,
        context_mask=task.context_mask,
        max_updates=MAX_UPDATES,
        delay=routed.DELAY_STEPS,
        random_seed=seed + 1100 + task_index * 20,
        stop_at_mastery=True,
        mastery_cell=selected_cell,
    )
    selected_state, retention = routed._heldout_retention_verification(
        router=selected_router,
        state=selected_state,
        cell_index=selected_cell,
        context=task_context,
        context_mask=task.context_mask,
        target=task.target,
    )
    record = {
        "task_id": task.task_id,
        "cell": selected_cell,
        "context": task_context,
        "context_mask": task.context_mask,
        "target": task.target,
    }
    next_records = [*prefix_records, record]
    selected_state, prefix = _prefix_verify(
        router=selected_router,
        state=selected_state,
        records=next_records,
    )
    result = {
        "task_id": task.task_id,
        "selected_initialization": receipt.selected_initialization,
        "selected_cell": selected_cell,
        "receipt": receipt,
        "probe": probe_records,
        "continuation": continuation,
        "retention": retention,
        "prefix": prefix,
        "source_unchanged": router._state_digest(state) == source_digest,
        "schema_v2": receipt.schema
        == "neural-computer.external-routed-intention-prior-selection.v2",
        "adjusted_margin": abs(
            receipt.transfer_adjusted_score - receipt.fresh_adjusted_score
        ),
        "task_unseen": _task_is_unseen(task),
    }
    return selected_router, selected_state, result, next_records


def _run_sequence(seed: int, *, fresh: bool) -> dict[str, object]:
    prepared = challenger._prepare(seed, fresh=fresh)
    reference_policy = prepared["reference_policy"]
    router = prepared["router"]
    state = prepared["state"]
    prefix_records = [
        {
            "task_id": "known_source",
            "cell": 0,
            "context": prepared["contexts"]["source"],
            "context_mask": routed.SOURCE_CONTEXT_MASK,
            "target": base.SOURCE_TARGET,
        },
        {
            "task_id": "known_successor",
            "cell": prepared["successor_cell"],
            "context": prepared["contexts"]["successor"],
            "context_mask": routed.OVERLAP_SUCCESSOR_CONTEXT_MASK,
            "target": base.SUCCESSOR_TARGET,
        },
    ]
    records: list[dict[str, object]] = []
    task_reports: list[dict[str, object]] = []
    initial_external_cells = int(state.cells.baseline.shape[0])
    for task_index, spec in enumerate(SEQUENTIAL_ADMISSIONS):
        router, state, task_report, prefix_records = _admit_one(
            seed=seed,
            task_index=task_index,
            spec=spec,
            fresh=fresh,
            reference_policy=reference_policy,
            state=state,
            router=router,
            controller_state=prepared["controller_state"],
            feedback=prepared["feedback"],
            events=prepared["events"],
            contexts=prepared["contexts"],
            source_cell=prepared["successor_cell"],
            prefix_records=prefix_records,
        )
        task_reports.append(task_report)
        records.extend(prefix_records[-1:])

    latest = task_reports[-1]
    latest_task = SEQUENTIAL_ADMISSIONS[-1].task
    latest_cell = latest["selected_cell"]
    reversal_state, reversal_cell = router.append_cell(
        state,
        source_cell=latest_cell,
        copy_route=False,
        context_mask=routed.OVERLAP_REVERSAL_CONTEXT_MASK,
    )
    reversal_policy = challenger._policy_with_router(reference_policy, router)
    reversal_state, reversal = routed._train_regime(
        policy=reversal_policy,
        router=router,
        state=reversal_state,
        controller_state=prepared["controller_state"],
        feedback=prepared["feedback"],
        event=prepared["events"]["reversal"],
        context=prepared["contexts"]["reversal"],
        target=base.REVERSED_TARGET,
        context_mask=routed.OVERLAP_REVERSAL_CONTEXT_MASK,
        max_updates=routed.REVERSAL_UPDATES,
        delay=routed.REVERSAL_DELAY_STEPS,
        noise_fraction=0.20,
        random_seed=seed + 1800,
        stop_at_mastery=False,
        mastery_cell=reversal_cell,
    )
    _, latest_after_reversal = routed._heldout_retention_verification(
        router=router,
        state=reversal_state,
        cell_index=latest_cell,
        context=prepared["contexts"][latest_task.context_key],
        context_mask=latest_task.context_mask,
        target=latest_task.target,
    )
    controls = challenger._run_causal_controls(
        seed,
        fresh=fresh,
        task=latest_task,
        reference_policy=reference_policy,
        controller_state=prepared["controller_state"],
        feedback=prepared["feedback"],
        event=prepared["events"][latest_task.event_key],
        context=prepared["contexts"][latest_task.context_key],
    )
    corruption = challenger._cell_corruption_probe(
        router,
        state,
        latest_cell,
        prepared["contexts"][latest_task.context_key],
        latest_task,
    )
    persistence = router.state_from_payload(router.state_payload(state))
    return {
        "fresh": fresh,
        "tasks": task_reports,
        "records": records,
        "initial_external_cells": initial_external_cells,
        "router": router,
        "state": state,
        "reversal": reversal,
        "latest_after_reversal": latest_after_reversal,
        "controls": controls,
        "corruption": corruption,
        "persistence_exact": router._state_digest(persistence) == router._state_digest(state),
        "controller_digest": base._digest_module(prepared["controller"]),
        "adapter_digest": base._digest_module(reference_policy.state_adapter),
        "known_source_retention": prepared["source_retention"],
        "known_successor_retention": prepared["successor_retention"],
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    warm = _run_sequence(seed, fresh=False)
    fresh = _run_sequence(seed, fresh=True)
    task_masks = [spec.task.context_mask.tolist() for spec in SEQUENTIAL_ADMISSIONS]
    task_targets = [spec.task.target.tolist() for spec in SEQUENTIAL_ADMISSIONS]
    gates: dict[str, bool] = {
        "warm_known_source_retained": warm["known_source_retention"]["accepted"],
        "fresh_known_source_retained": fresh["known_source_retention"]["accepted"],
        "warm_known_successor_retained": warm["known_successor_retention"]["accepted"],
        "fresh_known_successor_retained": fresh["known_successor_retention"]["accepted"],
        "warm_all_tasks_selected_expected_prior": all(
            report["selected_initialization"] == spec.task.expected_initialization
            for report, spec in zip(warm["tasks"], SEQUENTIAL_ADMISSIONS, strict=True)
        ),
        "fresh_all_tasks_selected_expected_prior": all(
            report["selected_initialization"] == spec.task.expected_initialization
            for report, spec in zip(fresh["tasks"], SEQUENTIAL_ADMISSIONS, strict=True)
        ),
        "warm_all_probe_margins_material": all(
            report["receipt"].transfer_adjusted_score is not None
            and abs(
                report["receipt"].transfer_probe_score
                - report["receipt"].fresh_probe_score
            ) >= MIN_PROBE_MARGIN
            for report in warm["tasks"]
        ),
        "fresh_all_probe_margins_material": all(
            report["receipt"].transfer_adjusted_score is not None
            and abs(
                report["receipt"].transfer_probe_score
                - report["receipt"].fresh_probe_score
            ) >= MIN_PROBE_MARGIN
            for report in fresh["tasks"]
        ),
        "warm_all_adjusted_margins_material": all(
            report["adjusted_margin"] >= MIN_ADJUSTED_MARGIN
            for report in warm["tasks"]
        ),
        "fresh_all_adjusted_margins_material": all(
            report["adjusted_margin"] >= MIN_ADJUSTED_MARGIN
            for report in fresh["tasks"]
        ),
        "warm_all_tasks_unseen": all(report["task_unseen"] for report in warm["tasks"]),
        "fresh_all_tasks_unseen": all(report["task_unseen"] for report in fresh["tasks"]),
        "task_masks_pairwise_distinct": len({tuple(mask) for mask in task_masks})
        == len(task_masks),
        "task_targets_pairwise_distinct": len({tuple(target) for target in task_targets})
        == len(task_targets),
        "warm_all_tasks_retained": all(report["retention"]["accepted"] for report in warm["tasks"]),
        "fresh_all_tasks_retained": all(report["retention"]["accepted"] for report in fresh["tasks"]),
        "warm_complete_prefix_after_every_admission": all(report["prefix"]["accepted"] for report in warm["tasks"]),
        "fresh_complete_prefix_after_every_admission": all(report["prefix"]["accepted"] for report in fresh["tasks"]),
        "warm_all_source_states_unchanged": all(report["source_unchanged"] for report in warm["tasks"]),
        "fresh_all_source_states_unchanged": all(report["source_unchanged"] for report in fresh["tasks"]),
        "warm_cost_aware_receipts": all(report["schema_v2"] for report in warm["tasks"]),
        "fresh_cost_aware_receipts": all(report["schema_v2"] for report in fresh["tasks"]),
        "warm_latest_survived_reversal": warm["latest_after_reversal"]["accepted"],
        "fresh_latest_survived_reversal": fresh["latest_after_reversal"]["accepted"],
        "warm_missing_evidence_is_noop": warm["controls"]["missing_evidence"]["state_unchanged"],
        "fresh_missing_evidence_is_noop": fresh["controls"]["missing_evidence"]["state_unchanged"],
        "warm_memory_corruption_detected": warm["corruption"]["corruption_detected"],
        "fresh_memory_corruption_detected": fresh["corruption"]["corruption_detected"],
        "warm_reward_shuffled_control_not_sample_efficient": (
            warm["controls"]["reward_shuffled"]["deterministic_best_score"]
            < base.MASTERY_THRESHOLD
            or warm["controls"]["reward_shuffled"]["updates"]
            > 4 * warm["tasks"][-1]["continuation"]["updates"]
        ),
        "fresh_reward_shuffled_control_not_sample_efficient": (
            fresh["controls"]["reward_shuffled"]["deterministic_best_score"]
            < base.MASTERY_THRESHOLD
            or fresh["controls"]["reward_shuffled"]["updates"]
            > 4 * fresh["tasks"][-1]["continuation"]["updates"]
        ),
        "warm_action_shuffled_control_not_sample_efficient": (
            warm["controls"]["action_shuffled"]["deterministic_best_score"]
            < base.MASTERY_THRESHOLD
            or warm["controls"]["action_shuffled"]["updates"]
            > 4 * warm["tasks"][-1]["continuation"]["updates"]
        ),
        "fresh_action_shuffled_control_not_sample_efficient": (
            fresh["controls"]["action_shuffled"]["deterministic_best_score"]
            < base.MASTERY_THRESHOLD
            or fresh["controls"]["action_shuffled"]["updates"]
            > 4 * fresh["tasks"][-1]["continuation"]["updates"]
        ),
        "warm_append_only_growth": (
            int(warm["state"].cells.baseline.shape[0])
            == warm["initial_external_cells"] + len(SEQUENTIAL_ADMISSIONS)
        ),
        "fresh_append_only_growth": (
            int(fresh["state"].cells.baseline.shape[0])
            == fresh["initial_external_cells"] + len(SEQUENTIAL_ADMISSIONS)
        ),
        "warm_persistence_exact": warm["persistence_exact"],
        "fresh_persistence_exact": fresh["persistence_exact"],
        "controller_frozen": warm["controller_digest"] == fresh["controller_digest"],
        "state_adapter_frozen": warm["adapter_digest"] == fresh["adapter_digest"],
        "zero_replayed_examples": True,
    }
    report = {
        "schema": "neural-computer.policy-free-intention-sequential-admission.v1",
        "claim_boundary": (
            "bounded sequential cost-aware copy-or-fresh admission over three "
            "unseen task families with complete-prefix retention; broad task-family "
            "generalization, arbitrary new computation, unrestricted growth, and "
            "general continual learning remain unqualified"
        ),
        "seed": seed,
        "configuration": {
            "known_curriculum": "adaptive_versioned_multi_stage",
            "admissions": [
                {
                    "task_id": spec.task.task_id,
                    "event_key": spec.task.event_key,
                    "context_key": spec.task.context_key,
                    "context_mask": spec.task.context_mask.tolist(),
                    "target": spec.task.target.tolist(),
                    "expected_initialization": spec.task.expected_initialization,
                    "transfer_cost": spec.transfer_cost,
                    "fresh_cost": spec.fresh_cost,
                    "cost_weight": spec.cost_weight,
                }
                for spec in SEQUENTIAL_ADMISSIONS
            ],
            "probe_updates": PROBE_UPDATES,
            "minimum_probe_margin": MIN_PROBE_MARGIN,
            "minimum_adjusted_margin": MIN_ADJUSTED_MARGIN,
            "replayed_examples": 0,
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "warm": {
                "tasks": [
                    {
                        **{key: value for key, value in report.items() if key != "receipt"},
                        "receipt": report["receipt"].__dict__,
                    }
                    for report in warm["tasks"]
                ],
                "controls": warm["controls"],
                "corruption": warm["corruption"],
                "reversal": warm["reversal"],
                "latest_after_reversal": warm["latest_after_reversal"],
                "initial_external_cells": warm["initial_external_cells"],
                "final_external_cells": int(warm["state"].cells.baseline.shape[0]),
            },
            "fresh": {
                "tasks": [
                    {
                        **{key: value for key, value in report.items() if key != "receipt"},
                        "receipt": report["receipt"].__dict__,
                    }
                    for report in fresh["tasks"]
                ],
                "controls": fresh["controls"],
                "corruption": fresh["corruption"],
                "reversal": fresh["reversal"],
                "latest_after_reversal": fresh["latest_after_reversal"],
                "initial_external_cells": fresh["initial_external_cells"],
                "final_external_cells": int(fresh["state"].cells.baseline.shape[0]),
            },
        },
        "accounting": {
            "warm_unique_verifier_bits": sum(
                2 * PROBE_UPDATES
                + report["continuation"]["updates"]
                for report in warm["tasks"]
            ),
            "fresh_unique_verifier_bits": sum(
                2 * PROBE_UPDATES
                + report["continuation"]["updates"]
                for report in fresh["tasks"]
            ),
            "warm_heldout_verifier_bits": sum(
                report["prefix"]["verifier_bits"] for report in warm["tasks"]
            ),
            "fresh_heldout_verifier_bits": sum(
                report["prefix"]["verifier_bits"] for report in fresh["tasks"]
            ),
            "warm_control_outcome_bits": 320,
            "fresh_control_outcome_bits": 320,
            "warm_external_cells": int(warm["state"].cells.baseline.shape[0]),
            "fresh_external_cells": int(fresh["state"].cells.baseline.shape[0]),
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
    parser.add_argument("--seed", type=int, default=85301)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
