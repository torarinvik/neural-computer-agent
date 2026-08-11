"""Longer-horizon multi-regime pressure test for factored external memory.

The two-regime factored audit proves that one frozen replaceable base can
support one promoted residual.  This audit raises the pressure: several
nonstationary regimes are acquired sequentially, each candidate is compared
with a matched fresh residual challenger, all earlier recursive probes remain
protected, and the final memory is exercised in reverse order and through
partial evidence.  No controller or decoder weights are updated.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch

from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    ExternalAffineTransitionStatistics,
    ExternalControllerEventWindowStateAdapter,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalModelBasedPlanner,
    ExternalTransitionContextEncoder,
    ExternalTransitionRollout,
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

FACTORED_RESIDUAL_SEQUENCE_PRESSURE_SCHEMA = (
    "neural-computer.brainworkshop-factored-residual-sequence-pressure.v1"
)


@dataclass(frozen=True)
class RegimePressureResult:
    index: int
    n_back: int
    cue_symbol: int
    staged: bool
    promoted: bool
    candidate_errors: tuple[float, ...]
    fresh_errors: tuple[float, ...]
    fresh_improvement: bool
    prior_retention: bool
    selected_residual_ridge: float | None
    context_count_after: int

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequencePressureResult:
    seed: int
    status: str
    regimes: tuple[RegimePressureResult, ...]
    source_retained: bool
    reversal_passed: bool
    missing_evidence_passed: bool
    memory_corruption_rejected: bool
    controller_unchanged: bool
    base_frozen: bool
    final_context_count: int
    unique_verifier_bits: int
    logical_lifetimes: int
    transition_rows_consumed_once: int
    optimizer_updates: int
    replayed_examples: int
    reversal_errors: tuple[float, ...]

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _policy(
    agent: CanonicalBrainWorkshopAgent,
    model: ExternalFactoredTransitionModel,
    state_adapter: ExternalControllerEventWindowStateAdapter,
) -> PolicyFreeAmodalRuntime:
    return PolicyFreeAmodalRuntime(
        agent.runtime,
        ExternalModelBasedPlanner(model, beam_width=4),
        state_adapter=state_adapter,
    )


def _recursive_error(
    model: ExternalFactoredTransitionModel,
    rollout: ExternalTransitionRollout,
    context: torch.Tensor,
) -> float:
    return ExternalModelBasedPlanner(model, beam_width=1).rollout_error(
        rollout,
        transition_context=context.unsqueeze(0),
    )


def _fresh_model_for_context(
    payload: dict[str, object],
    context: torch.Tensor,
) -> ExternalFactoredTransitionModel:
    """Restore a pre-candidate model and add an empty matched target slot."""

    fresh = ExternalFactoredTransitionModel.from_payload(payload)
    if fresh.residual_bank is None:
        raise RuntimeError("sequence pressure requires a learned residual bank")
    fresh.residual_bank.ensure_context(context)
    return fresh


def run_factored_residual_sequence_pressure(
    *,
    seed: int,
    steps: int = 10,
    source_training_lifetimes: int = 3,
    target_training_lifetimes: int = 2,
    promotion_holdout_lifetimes: int = 2,
    regime_count: int = 3,
    residual_model_family: str = EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    residual_random_feature_width: int = 128,
    residual_ridge_candidates: Sequence[float] | None = None,
    routing_match_tolerance: float = 0.02,
    recursive_error_bound: float = 1.0,
    ridge: float = 1e-3,
) -> SequencePressureResult:
    """Acquire several opaque regimes while keeping all earlier probes safe."""

    if min(
        steps,
        source_training_lifetimes,
        target_training_lifetimes,
        promotion_holdout_lifetimes,
        regime_count,
    ) < 1:
        raise ValueError("factored sequence pressure budgets must be positive")
    if target_training_lifetimes < 2 or promotion_holdout_lifetimes < 2:
        raise ValueError("factored sequence pressure needs repeated evidence")
    if residual_random_feature_width < 1:
        raise ValueError("factored sequence residual width must be positive")
    if residual_model_family not in {
        EXTERNAL_TRANSITION_AFFINE_MODEL_FAMILY,
        EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    }:
        raise ValueError("factored sequence residual family is unsupported")
    ridge_candidates = (
        (float(ridge), 1e-2, 1e-1, 1.0, 10.0)
        if residual_ridge_candidates is None
        else tuple(float(value) for value in residual_ridge_candidates)
    )
    if not ridge_candidates or any(value <= 0.0 for value in ridge_candidates):
        raise ValueError("factored sequence ridge candidates must be positive")
    if len(set(ridge_candidates)) != len(ridge_candidates):
        raise ValueError("factored sequence ridge candidates must be unique")
    if routing_match_tolerance <= 0.0 or recursive_error_bound <= 0.0 or ridge <= 0.0:
        raise ValueError("factored sequence pressure thresholds must be positive")

    agent = CanonicalBrainWorkshopAgent(
        symbol_count=12,
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
        residual_model_family=residual_model_family,
        residual_random_feature_width=residual_random_feature_width,
        residual_ridge=ridge,
        base_model=base,
    )
    source_context = _opaque_context(agent, 6)
    if model.residual_bank is None:
        raise RuntimeError("factored sequence pressure requires a residual bank")
    model.residual_bank.ensure_context(source_context)
    candidate_intentions = torch.randn(
        6,
        agent.controller.intention_width,
        generator=torch.Generator().manual_seed(seed + 7000),
    )

    unique_verifier_bits = 0
    transition_rows = 0
    for lifetime in range(source_training_lifetimes):
        policy = _policy(agent, model, state_adapter)
        rollout, bits = _run_lifetime(
            agent,
            policy,
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

    source_policy = _policy(agent, model, state_adapter)
    source_holdout, source_bits = _run_lifetime(
        agent,
        source_policy,
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

    context_encoder = ExternalTransitionContextEncoder(
        model.state_width,
        model.intention_width,
        hidden_width=max(16, model.state_width),
        context_width=model.context_width,
    )
    router = ExternalFactoredTransitionRouter(
        model,
        context_encoder,
        match_tolerance=routing_match_tolerance,
        admission_observations=steps,
        max_contexts=regime_count,
        residual_adaptation_updates=1,
    )

    prior_probes: list[tuple[ExternalTransitionRollout, torch.Tensor]] = [
        (source_holdout, source_context.clone())
    ]
    prior_errors_before = [
        _recursive_error(router.model, rollout, context)
        for rollout, context in prior_probes
    ]
    regime_results: list[RegimePressureResult] = []
    regime_holdouts: list[tuple[int, int, torch.Tensor, ExternalTransitionRollout]] = []

    for regime_index in range(regime_count):
        n_back = regime_index + 3
        cue_symbol = regime_index + 7
        acquisition_policy = _policy(agent, router.model, state_adapter)
        training_rollouts: list[ExternalTransitionRollout] = []
        for lifetime in range(target_training_lifetimes):
            rollout, bits = _run_lifetime(
                agent,
                acquisition_policy,
                router.model,
                source_context,
                n_back=n_back,
                steps=steps,
                seed=seed + 2000 + regime_index * 100 + lifetime,
                cue_symbol=cue_symbol,
                candidate_intentions=candidate_intentions,
                learn=False,
            )
            training_rollouts.append(rollout)
            unique_verifier_bits += bits
            transition_rows += steps

        pre_candidate_payload = router.model.state_payload()
        route = router.route_bundle(_rollout_observations(training_rollouts[0]))
        if route.status == "staged":
            for rollout in training_rollouts[1:]:
                for observation in _rollout_observations(rollout):
                    router.observe(observation)

        candidate = router._candidate_model
        candidate_context = router._candidate_context
        if route.status != "staged" or candidate is None or candidate_context is None:
            regime_results.append(
                RegimePressureResult(
                    index=regime_index,
                    n_back=n_back,
                    cue_symbol=cue_symbol,
                    staged=False,
                    promoted=False,
                    candidate_errors=(),
                    fresh_errors=(),
                    fresh_improvement=False,
                    prior_retention=False,
                    selected_residual_ridge=None,
                    context_count_after=len(router.slot_ids),
                )
            )
            break

        candidate_policy = _policy(agent, candidate, state_adapter)
        holdouts: list[ExternalTransitionRollout] = []
        fresh_model = _fresh_model_for_context(
            pre_candidate_payload,
            candidate_context,
        )
        for holdout_index in range(promotion_holdout_lifetimes):
            holdout, bits = _run_lifetime(
                agent,
                candidate_policy,
                candidate,
                candidate_context,
                n_back=n_back,
                steps=steps,
                seed=seed + 12000 + regime_index * 100 + holdout_index,
                cue_symbol=cue_symbol,
                candidate_intentions=candidate_intentions,
                learn=False,
            )
            holdouts.append(holdout)
            unique_verifier_bits += bits

        # Ridge changes are analytic copies of the sufficient statistics. The
        # held-out verifier can therefore select a stable regularization for
        # this slot without replaying any transition row.
        candidate_ridges = tuple(dict.fromkeys((float(ridge), *ridge_candidates)))
        variant_records: list[
            tuple[
                ExternalFactoredTransitionModel,
                float,
                list[float],
                list[float],
                bool,
            ]
        ] = []
        for candidate_ridge in candidate_ridges:
            variant = (
                candidate
                if candidate_ridge == float(ridge)
                else candidate.reparameterized_residual_ridge(
                    candidate_context,
                    candidate_ridge,
                )
            )
            variant_errors = [
                _recursive_error(variant, holdout, candidate_context)
                for holdout in holdouts
            ]
            variant_fresh_errors = [
                _recursive_error(fresh_model, holdout, candidate_context)
                for holdout in holdouts
            ]
            variant_wins = sum(
                variant_error < fresh_error
                for variant_error, fresh_error in zip(
                    variant_errors,
                    variant_fresh_errors,
                    strict=True,
                )
            )
            variant_improvement = (
                bool(variant_errors)
                and all(
                    error <= recursive_error_bound for error in variant_errors
                )
                and variant_wins >= (len(variant_errors) + 1) // 2
                and sum(variant_errors) / len(variant_errors)
                < sum(variant_fresh_errors) / len(variant_fresh_errors)
            )
            variant_prior_errors = [
                _recursive_error(variant, rollout, context)
                for rollout, context in prior_probes
            ]
            variant_retention = all(
                abs(after - before) <= 1e-7
                for after, before in zip(
                    variant_prior_errors,
                    prior_errors_before,
                    strict=True,
                )
            )
            variant_records.append(
                (
                    variant,
                    candidate_ridge,
                    variant_errors,
                    variant_fresh_errors,
                    variant_improvement and variant_retention,
                )
            )
        viable_variants = [record for record in variant_records if record[-1]]
        selected_variant = min(
            viable_variants,
            key=lambda record: sum(record[2]) / len(record[2]),
            default=None,
        )
        if selected_variant is None:
            selected_variant = variant_records[0]
        candidate, selected_ridge, candidate_errors, fresh_errors, selected_valid = (
            selected_variant
        )
        router._candidate_model = candidate
        fresh_improvement = selected_valid

        retention_prior_errors = tuple(prior_errors_before)
        retention_target_context = candidate_context.clone()
        retention_holdouts = tuple(holdouts)
        retention_fresh_improvement = fresh_improvement

        def retention_probe(
            candidate_model: ExternalFactoredTransitionModel,
            *,
            expected_prior_errors: tuple[float, ...] = retention_prior_errors,
            target_context: torch.Tensor = retention_target_context,
            target_holdouts: tuple[ExternalTransitionRollout, ...] = retention_holdouts,
            target_improvement: bool = retention_fresh_improvement,
        ) -> bool:
            if candidate_model.base.digest() != router.model.base.digest():
                return False
            retained_errors = [
                _recursive_error(candidate_model, rollout, context)
                for rollout, context in prior_probes
            ]
            if any(
                abs(after - before) > 1e-7
                for after, before in zip(
                    retained_errors,
                    expected_prior_errors,
                    strict=True,
                )
            ):
                return False
            candidate_target_errors = [
                _recursive_error(candidate_model, holdout, target_context)
                for holdout in target_holdouts
            ]
            return (
                target_improvement
                and all(
                    error <= recursive_error_bound
                    for error in candidate_target_errors
                )
            )

        promotion = router.promote_staged_candidate(
            _rollout_bundle(holdouts[0]),
            retention_probe,
            prediction_tolerance=recursive_error_bound,
            heldout_rollout=holdouts[0],
            rollout_error_tolerance=recursive_error_bound,
        )
        prior_retention = all(
            abs(
                _recursive_error(router.model, rollout, context)
                - before
            )
            <= 1e-7
            for (rollout, context), before in zip(
                prior_probes,
                prior_errors_before,
                strict=True,
            )
        ) if promotion.accepted else False
        regime_results.append(
            RegimePressureResult(
                index=regime_index,
                n_back=n_back,
                cue_symbol=cue_symbol,
                staged=True,
                promoted=promotion.accepted,
                candidate_errors=tuple(candidate_errors),
                fresh_errors=tuple(fresh_errors),
                fresh_improvement=fresh_improvement,
                prior_retention=prior_retention,
                selected_residual_ridge=selected_ridge,
                context_count_after=len(router.slot_ids),
            )
        )
        if not promotion.accepted:
            break

        for holdout in holdouts:
            prior_probes.append((holdout, candidate_context.clone()))
        prior_errors_before = [
            _recursive_error(router.model, rollout, context)
            for rollout, context in prior_probes
        ]
        regime_holdouts.append((n_back, cue_symbol, candidate_context.clone(), holdouts[0]))

    source_retained = bool(regime_results) and all(
        result.prior_retention for result in regime_results
    )
    reversal_errors: list[float] = []
    reversal_passed = len(regime_results) == regime_count and all(
        result.promoted for result in regime_results
    )
    if reversal_passed:
        for reverse_index, (n_back, cue_symbol, context, _holdout) in enumerate(
            reversed(regime_holdouts)
        ):
            reverse_policy = _policy(agent, router.model, state_adapter)
            rollout, bits = _run_lifetime(
                agent,
                reverse_policy,
                router.model,
                context,
                n_back=n_back,
                steps=steps,
                seed=seed + 30000 + reverse_index,
                cue_symbol=cue_symbol,
                candidate_intentions=candidate_intentions,
                learn=False,
            )
            unique_verifier_bits += bits
            error = _recursive_error(router.model, rollout, context)
            reversal_errors.append(error)
        reversal_passed = bool(reversal_errors) and all(
            error <= recursive_error_bound for error in reversal_errors
        )

    missing_evidence_passed = False
    if reversal_passed and regime_holdouts:
        digest_before = router.digest()
        missing_evidence_passed = True
        for _n_back, _cue_symbol, context, holdout in regime_holdouts:
            observations = _rollout_observations(holdout)
            split = max(1, len(observations) // 2)
            result = router.route_partial_sequence(
                (
                    observations[:split],
                    observations[split:],
                ),
                min_match_fraction=1.0,
                match_tolerance=recursive_error_bound,
            )
            missing_evidence_passed = missing_evidence_passed and (
                result.status == "matched"
                and result.context is not None
                and torch.allclose(result.context, context)
            )
        missing_evidence_passed = missing_evidence_passed and router.digest() == digest_before

    memory_corruption_rejected = False
    if reversal_passed:
        corrupted = copy.deepcopy(router.state_payload())
        corrupted["model"]["sha256"] = "0" * 64
        try:
            ExternalFactoredTransitionRouter.from_payload(corrupted)
        except (TypeError, ValueError, KeyError):
            memory_corruption_rejected = True

    controller_unchanged = controller_before == _controller_digest(agent)
    complete = (
        len(regime_results) == regime_count
        and all(
            result.staged
            and result.promoted
            and result.fresh_improvement
            and result.prior_retention
            for result in regime_results
        )
        and source_retained
        and reversal_passed
        and missing_evidence_passed
        and memory_corruption_rejected
        and controller_unchanged
    )
    return SequencePressureResult(
        seed=seed,
        status=(
            "factored_residual_sequence_pressure_passed"
            if complete
            else "factored_residual_sequence_pressure_rejected"
        ),
        regimes=tuple(regime_results),
        source_retained=source_retained,
        reversal_passed=reversal_passed,
        missing_evidence_passed=missing_evidence_passed,
        memory_corruption_rejected=memory_corruption_rejected,
        controller_unchanged=controller_unchanged,
        base_frozen=router.model.base_frozen,
        final_context_count=len(router.slot_ids),
        unique_verifier_bits=unique_verifier_bits,
        logical_lifetimes=(
            source_training_lifetimes
            + 1
            + target_training_lifetimes * regime_count
            + promotion_holdout_lifetimes * regime_count
            + (regime_count if reversal_passed else 0)
        ),
        transition_rows_consumed_once=transition_rows,
        optimizer_updates=0,
        replayed_examples=0,
        reversal_errors=tuple(reversal_errors),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[91, 92, 93],
    )
    parser.add_argument("--report-out", type=str)
    args = parser.parse_args()
    started = time.monotonic()
    results = [
        run_factored_residual_sequence_pressure(seed=seed)
        for seed in args.seeds
    ]
    report = {
        "schema": FACTORED_RESIDUAL_SEQUENCE_PRESSURE_SCHEMA,
        "status": (
            "factored_residual_sequence_pressure_replicated"
            if all(result.status.endswith("passed") for result in results)
            else "factored_residual_sequence_pressure_failed"
        ),
        "results": [result.payload() for result in results],
        "aggregate": {
            "complete_passes": sum(
                result.status.endswith("passed") for result in results
            ),
            "total_runs": len(results),
            "regime_promotions": sum(
                sum(regime.promoted for regime in result.regimes)
                for result in results
            ),
            "source_retention_passes": sum(
                result.source_retained for result in results
            ),
            "reversal_passes": sum(result.reversal_passed for result in results),
            "missing_evidence_passes": sum(
                result.missing_evidence_passed for result in results
            ),
            "memory_corruption_rejection_passes": sum(
                result.memory_corruption_rejected for result in results
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
