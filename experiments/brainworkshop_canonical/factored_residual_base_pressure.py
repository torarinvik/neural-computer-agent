"""Pressure-test a replaceable frozen base plus external residual memory.

This audit isolates the representation seam exposed by the online-discovery
negative controls.  A replay-free affine base is trained on one rendered
regime, frozen, and then a random-feature residual is acquired in a separate
opaque factored slot for a novel regime.  The controller and decoder remain
frozen throughout.  The result is a bounded external-memory capability claim,
not general continual learning.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch

from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    ExternalAffineTransitionStatistics,
    ExternalControllerEventWindowStateAdapter,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalModelBasedPlanner,
    ExternalTransitionContextEncoder,
    PolicyFreeAmodalRuntime,
)

from .replay_free_transition_acquisition import (
    _controller_digest,
    _opaque_context,
    _rollout_bundle,
    _rollout_observations,
    _run_lifetime,
)
from .runner import CanonicalBrainWorkshopAgent

FACTORED_RESIDUAL_PRESSURE_SCHEMA = (
    "neural-computer.brainworkshop-factored-residual-base-pressure.v1"
)


@dataclass(frozen=True)
class FactoredResidualPressureResult:
    seed: int
    status: str
    target_staged: bool
    target_promoted: bool
    target_improved_over_frozen_base: bool
    source_retained: bool
    controller_unchanged: bool
    base_frozen: bool
    base_model_schema: str
    candidate_error: float
    frozen_base_error: float
    source_error_before: float
    source_error_after: float
    unique_verifier_bits: int
    logical_lifetimes: int
    transition_rows_consumed_once: int
    optimizer_updates: int
    replayed_examples: int

    def payload(self) -> dict[str, object]:
        return asdict(self)


def run_factored_residual_pressure(
    *,
    seed: int,
    steps: int = 6,
    source_training_lifetimes: int = 2,
    target_training_lifetimes: int = 2,
    residual_random_feature_width: int = 128,
    ridge: float = 1e-3,
) -> FactoredResidualPressureResult:
    """Run one source-retention and factored-residual acquisition pressure test."""

    if min(steps, source_training_lifetimes, target_training_lifetimes) < 1:
        raise ValueError("factored residual pressure budgets must be positive")
    if target_training_lifetimes < 2:
        raise ValueError("factored residual pressure needs a continuation lifetime")
    if residual_random_feature_width < 1 or ridge <= 0.0:
        raise ValueError("factored residual pressure model settings are invalid")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=8,
        event_width=4,
        intention_width=2,
        feedback_width=3,
        n_back=2,
        reader_kind="relation",
        seed=seed,
    )
    controller_before = _controller_digest(agent)
    for parameter in agent.parameters():
        parameter.requires_grad_(False)

    state_adapter = ExternalControllerEventWindowStateAdapter(
        agent.controller.width,
        state_width=agent.controller.width * 3,
    )
    base = ExternalAffineTransitionStatistics(
        state_adapter.state_width,
        agent.controller.intention_width,
        ridge=ridge,
    )
    model = ExternalFactoredTransitionModel(
        state_adapter.state_width,
        agent.controller.intention_width,
        agent.controller.width,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_random_feature_width=residual_random_feature_width,
        residual_ridge=ridge,
        base_model=base,
    )
    source_context = _opaque_context(agent, 6)
    if model.residual_bank is None:
        raise RuntimeError("factored pressure test requires a residual bank")
    model.residual_bank.ensure_context(source_context)

    policy_free = PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(model, beam_width=4),
        state_adapter=state_adapter,
    )
    candidate_intentions = torch.randn(
        6,
        agent.controller.intention_width,
        generator=torch.Generator().manual_seed(seed + 7000),
    )

    unique_verifier_bits = 0
    transition_rows = 0
    for lifetime in range(source_training_lifetimes):
        rollout, bits = _run_lifetime(
            agent,
            policy_free,
            model,
            source_context,
            n_back=2,
            steps=steps,
            seed=seed + lifetime,
            cue_symbol=6,
            candidate_intentions=candidate_intentions,
            learn=False,
        )
        unique_verifier_bits += bits
        for observation in _rollout_observations(rollout):
            base.observe(observation)
        transition_rows += steps
    model.freeze_base()

    source_holdout, source_bits = _run_lifetime(
        agent,
        policy_free,
        model,
        source_context,
        n_back=2,
        steps=steps,
        seed=seed + 10000,
        cue_symbol=6,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    unique_verifier_bits += source_bits
    planner = ExternalModelBasedPlanner(model, beam_width=1)
    source_error_before = planner.rollout_error(
        source_holdout,
        transition_context=source_context.unsqueeze(0),
    )

    context_encoder = ExternalTransitionContextEncoder(
        model.state_width,
        model.intention_width,
        hidden_width=max(16, model.state_width),
        context_width=model.context_width,
    )
    router = ExternalFactoredTransitionRouter(
        model,
        context_encoder,
        admission_observations=steps,
        max_contexts=2,
        residual_adaptation_updates=1,
    )

    target_rollouts = []
    for lifetime in range(target_training_lifetimes):
        rollout, bits = _run_lifetime(
            agent,
            policy_free,
            model,
            source_context,
            n_back=3,
            steps=steps,
            seed=seed + 2000 + lifetime,
            cue_symbol=7,
            candidate_intentions=candidate_intentions,
            learn=False,
        )
        target_rollouts.append(rollout)
        unique_verifier_bits += bits
        transition_rows += steps

    first_route = router.route_bundle(_rollout_observations(target_rollouts[0]))
    for rollout in target_rollouts[1:]:
        for observation in _rollout_observations(rollout):
            router.observe(observation)
    candidate = router._candidate_model
    candidate_context = router._candidate_context
    if candidate is None or candidate_context is None:
        return FactoredResidualPressureResult(
            seed=seed,
            status="target_not_staged",
            target_staged=False,
            target_promoted=False,
            target_improved_over_frozen_base=False,
            source_retained=False,
            controller_unchanged=controller_before == _controller_digest(agent),
            base_frozen=model.base_frozen,
            base_model_schema=str(getattr(model.base, "schema", "unknown")),
            candidate_error=float("inf"),
            frozen_base_error=float("inf"),
            source_error_before=source_error_before,
            source_error_after=float("inf"),
            unique_verifier_bits=unique_verifier_bits,
            logical_lifetimes=source_training_lifetimes
            + target_training_lifetimes
            + 2,
            transition_rows_consumed_once=transition_rows,
            optimizer_updates=0,
            replayed_examples=0,
        )

    candidate_policy = PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(candidate, beam_width=4),
        state_adapter=state_adapter,
    )
    target_holdout, target_bits = _run_lifetime(
        agent,
        candidate_policy,
        candidate,
        candidate_context,
        n_back=3,
        steps=steps,
        seed=seed + 12000,
        cue_symbol=7,
        candidate_intentions=candidate_intentions,
        learn=False,
    )
    unique_verifier_bits += target_bits
    candidate_planner = ExternalModelBasedPlanner(candidate, beam_width=1)
    candidate_error = candidate_planner.rollout_error(
        target_holdout,
        transition_context=candidate_context.unsqueeze(0),
    )
    frozen_base_error = ExternalModelBasedPlanner(base, beam_width=1).rollout_error(
        target_holdout,
    )
    source_error_after = candidate_planner.rollout_error(
        source_holdout,
        transition_context=source_context.unsqueeze(0),
    )

    def retention_probe(candidate_model: ExternalFactoredTransitionModel) -> bool:
        if candidate_model.base.digest() != model.base.digest():
            return False
        retained_source_error = ExternalModelBasedPlanner(
            candidate_model,
            beam_width=1,
        ).rollout_error(
            source_holdout,
            transition_context=source_context.unsqueeze(0),
        )
        candidate_target_error = ExternalModelBasedPlanner(
            candidate_model,
            beam_width=1,
        ).rollout_error(
            target_holdout,
            transition_context=candidate_context.unsqueeze(0),
        )
        return (
            abs(retained_source_error - source_error_before) <= 1e-7
            and candidate_target_error < frozen_base_error
        )

    promotion = router.promote_staged_candidate(
        _rollout_bundle(target_holdout),
        retention_probe,
        prediction_tolerance=1.0,
        heldout_rollout=target_holdout,
        rollout_error_tolerance=1.0,
    )
    restored_router = ExternalFactoredTransitionRouter.from_payload(
        router.state_payload()
    )
    persistence_ok = (
        restored_router.model.configuration()["base_model_schema"]
        == model.configuration()["base_model_schema"]
    )
    source_retained = abs(source_error_after - source_error_before) <= 1e-7
    target_improved = candidate_error < frozen_base_error
    controller_unchanged = controller_before == _controller_digest(agent)
    passed = (
        first_route.status == "staged"
        and promotion.accepted
        and persistence_ok
        and target_improved
        and source_retained
        and controller_unchanged
    )
    return FactoredResidualPressureResult(
        seed=seed,
        status=("factored_residual_base_pressure_passed" if passed else "rejected"),
        target_staged=first_route.status == "staged",
        target_promoted=promotion.accepted,
        target_improved_over_frozen_base=target_improved,
        source_retained=source_retained,
        controller_unchanged=controller_unchanged,
        base_frozen=router.model.base_frozen,
        base_model_schema=str(getattr(router.model.base, "schema", "unknown")),
        candidate_error=candidate_error,
        frozen_base_error=frozen_base_error,
        source_error_before=source_error_before,
        source_error_after=source_error_after,
        unique_verifier_bits=unique_verifier_bits,
        logical_lifetimes=source_training_lifetimes
        + target_training_lifetimes
        + 2,
        transition_rows_consumed_once=transition_rows,
        optimizer_updates=0,
        replayed_examples=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[91, 92, 93, 95, 99, 100, 101, 102, 103])
    parser.add_argument("--report-out", type=str)
    args = parser.parse_args()
    started = time.monotonic()
    results = [run_factored_residual_pressure(seed=seed) for seed in args.seeds]
    report = {
        "schema": FACTORED_RESIDUAL_PRESSURE_SCHEMA,
        "status": (
            "factored_residual_base_pressure_replicated"
            if all(result.status.endswith("passed") for result in results)
            else "factored_residual_base_pressure_failed"
        ),
        "results": [result.payload() for result in results],
        "aggregate": {
            "complete_passes": sum(
                result.status.endswith("passed") for result in results
            ),
            "total_runs": len(results),
            "target_staging_passes": sum(result.target_staged for result in results),
            "target_promotion_passes": sum(
                result.target_promoted for result in results
            ),
            "source_retention_passes": sum(
                result.source_retained for result in results
            ),
            "controller_unchanged_all_runs": all(
                result.controller_unchanged for result in results
            ),
            "unique_verifier_bits_total": sum(
                result.unique_verifier_bits for result in results
            ),
            "logical_lifetimes_total": sum(
                result.logical_lifetimes for result in results
            ),
            "transition_rows_consumed_once_total": sum(
                result.transition_rows_consumed_once for result in results
            ),
            "optimizer_updates_total": sum(
                result.optimizer_updates for result in results
            ),
            "replayed_examples_total": sum(
                result.replayed_examples for result in results
            ),
            "wall_time_seconds": round(time.monotonic() - started, 3),
        },
    }
    rendered = json.dumps(report, indent=2)
    if args.report_out:
        with open(args.report_out, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
