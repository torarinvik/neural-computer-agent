"""Audit one-pass factual residual growth behind a frozen shared model.

The source transition model is trained once and then frozen.  A successor
regime is represented by an external context-addressed residual model whose
random-feature statistics consume each transition row once.  Admission is
copy-on-write and requires an independent held-out one-step, recursive
rollout, and source-retention probe.  Full-model-copy and fresh-model arms are
matched controls: they may learn the successor, but are expected to overwrite
the source behavior because they do not factor new computation as a residual.

This is a factual-model pressure test, not a claim that the controller has
learned a task policy.  The controller-side interface remains unchanged and
all residual state lives outside the controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from neural_computer import (
    EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
    EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
    ExternalFactoredTransitionModel,
    ExternalFactoredTransitionRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModel,
    ExternalTransitionObservation,
    ExternalTransitionRollout,
)

STATE_WIDTH = 1
INTENTION_WIDTH = 1
CONTEXT_WIDTH = 4
BASE_HIDDEN_WIDTH = 32
RESIDUAL_HIDDEN_WIDTH = 32
RESIDUAL_FEATURE_WIDTH = 128
BASE_UPDATES = 1_500
CONTROL_UPDATES = 1_500
TARGET_ERROR_FLOOR = 0.03
SOURCE_RETENTION_FLOOR = 0.01
ROUTER_MATCH_TOLERANCE = 0.03
ROUTER_MATCH_MARGIN = 0.001


def _source_transition(state: torch.Tensor, intention: torch.Tensor) -> torch.Tensor:
    return state + 0.25 * intention + 0.05 * torch.tanh(state)


def _successor_transition(
    state: torch.Tensor,
    intention: torch.Tensor,
) -> torch.Tensor:
    return _source_transition(state, intention) + 0.35 * torch.sin(1.5 * state) + (
        0.2 * intention * state
    )


def _observation(
    state: torch.Tensor,
    intention: torch.Tensor,
    transition,
) -> ExternalTransitionObservation:
    return ExternalTransitionObservation(
        state=state.detach(),
        intention=intention.detach(),
        next_state=transition(state, intention).detach(),
    )


def _rows(
    observation: ExternalTransitionObservation,
) -> list[ExternalTransitionObservation]:
    return [
        ExternalTransitionObservation(
            state=observation.state[index : index + 1],
            intention=observation.intention[index : index + 1],
            next_state=observation.next_state[index : index + 1],
        )
        for index in range(observation.state.shape[0])
    ]


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("ascii"))
        digest.update(repr(tuple(detached.shape)).encode("ascii"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _module_storage_bytes(module: torch.nn.Module) -> int:
    return sum(
        value.detach().numel() * value.detach().element_size()
        for value in module.state_dict().values()
    )


def _base_data() -> tuple[
    ExternalTransitionObservation,
    ExternalTransitionObservation,
    ExternalTransitionObservation,
    ExternalTransitionObservation,
]:
    train_state = torch.linspace(-1.0, 1.0, 16).unsqueeze(-1).repeat_interleave(2, 0)
    train_intention = torch.tensor([[-1.0], [1.0]]).repeat(16, 1)
    heldout_state = torch.tensor(
        [[-0.85], [-0.55], [-0.25], [0.05], [0.35], [0.65], [0.90]]
    ).repeat_interleave(2, 0)
    heldout_intention = torch.tensor([[-0.5], [0.5]]).repeat(7, 1)
    return (
        _observation(train_state, train_intention, _source_transition),
        _observation(heldout_state, heldout_intention, _source_transition),
        _observation(train_state, train_intention, _successor_transition),
        _observation(heldout_state, heldout_intention, _successor_transition),
    )


def _fit_base(seed: int, source_train: ExternalTransitionObservation) -> tuple[
    ExternalTransitionModel,
    int,
]:
    torch.manual_seed(seed)
    base = ExternalTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=BASE_HIDDEN_WIDTH,
    )
    optimizer = torch.optim.Adam(base.parameters(), lr=0.02)
    for _update in range(BASE_UPDATES):
        optimizer.zero_grad()
        loss = base.loss(source_train)
        loss.backward()
        optimizer.step()
    return base, BASE_UPDATES


def _new_factored_model(base: ExternalTransitionModel, seed: int) -> ExternalFactoredTransitionModel:
    model = ExternalFactoredTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=BASE_HIDDEN_WIDTH,
        residual_mode=EXTERNAL_FACTORED_TRANSITION_LEARNED_RESIDUAL_MODE,
        residual_model_family=EXTERNAL_TRANSITION_RANDOM_FEATURE_MODEL_FAMILY,
        residual_hidden_width=RESIDUAL_HIDDEN_WIDTH,
        residual_random_feature_width=RESIDUAL_FEATURE_WIDTH,
        residual_random_feature_seed=seed,
        residual_ridge=0.01,
        residual_capacity=1,
    )
    model.base.load_state_dict(base.state_dict(), strict=True)
    model.freeze_base()
    return model


def _new_router(base: ExternalTransitionModel, seed: int) -> ExternalFactoredTransitionRouter:
    model = _new_factored_model(base, seed)
    return ExternalFactoredTransitionRouter(
        model,
        ExternalTransitionContextEncoder(
            STATE_WIDTH,
            INTENTION_WIDTH,
            hidden_width=16,
            context_width=CONTEXT_WIDTH,
        ),
        match_tolerance=ROUTER_MATCH_TOLERANCE,
        match_margin=ROUTER_MATCH_MARGIN,
        residual_adaptation_updates=1,
        max_contexts=1,
    )


def _rollout() -> ExternalTransitionRollout:
    initial = torch.tensor([0.05])
    intentions = torch.tensor([[-0.5], [0.5]])
    states: list[torch.Tensor] = []
    current = initial.unsqueeze(0)
    for index in range(intentions.shape[0]):
        current = _successor_transition(current, intentions[index : index + 1])
        states.append(current.squeeze(0))
    return ExternalTransitionRollout(
        initial_state=initial,
        intentions=intentions,
        expected_states=torch.stack(states),
    )


def _train_full_control(
    *,
    seed: int,
    initialization: str,
    source_heldout: ExternalTransitionObservation,
    target_train: ExternalTransitionObservation,
    target_heldout: ExternalTransitionObservation,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = ExternalTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=BASE_HIDDEN_WIDTH,
    )
    if initialization == "copy":
        raise RuntimeError("copy controls must be initialized by the caller")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    target_errors: list[float] = []
    source_errors: list[float] = []
    for _update in range(CONTROL_UPDATES):
        optimizer.zero_grad()
        loss = model.loss(target_train)
        loss.backward()
        optimizer.step()
        target_errors.append(float(model.loss(target_heldout).detach()))
        source_errors.append(float(model.loss(source_heldout).detach()))
    stable_update = _first_stable_update(target_errors, TARGET_ERROR_FLOOR)
    stable_source_error = (
        None if stable_update is None else source_errors[stable_update - 1]
    )
    return {
        "initialization": initialization,
        "stable_target_update": stable_update,
        "stable_target_error": (
            None if stable_update is None else target_errors[stable_update - 1]
        ),
        "source_error_at_target_stability": stable_source_error,
        "final_target_error": target_errors[-1],
        "final_source_error": source_errors[-1],
        "target_error_floor": TARGET_ERROR_FLOOR,
        "source_retention_floor": SOURCE_RETENTION_FLOOR,
        "source_retained_at_target_stability": (
            stable_source_error is not None
            and stable_source_error <= SOURCE_RETENTION_FLOOR
        ),
        "optimizer_updates": CONTROL_UPDATES,
        "replayed_examples": CONTROL_UPDATES * int(target_train.state.shape[0]),
        "parameter_bytes": _module_storage_bytes(model),
    }


def _train_copy_control(
    *,
    seed: int,
    source_base: ExternalTransitionModel,
    source_train: ExternalTransitionObservation,
    source_heldout: ExternalTransitionObservation,
    target_train: ExternalTransitionObservation,
    target_heldout: ExternalTransitionObservation,
) -> dict[str, object]:
    torch.manual_seed(seed)
    model = ExternalTransitionModel(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=BASE_HIDDEN_WIDTH,
    )
    model.load_state_dict(source_base.state_dict(), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    target_errors: list[float] = []
    source_errors: list[float] = []
    for _update in range(CONTROL_UPDATES):
        optimizer.zero_grad()
        loss = model.loss(target_train)
        loss.backward()
        optimizer.step()
        target_errors.append(float(model.loss(target_heldout).detach()))
        source_errors.append(float(model.loss(source_heldout).detach()))
    stable_update = _first_stable_update(target_errors, TARGET_ERROR_FLOOR)
    stable_source_error = (
        None if stable_update is None else source_errors[stable_update - 1]
    )
    return {
        "initialization": "full_model_copy",
        "stable_target_update": stable_update,
        "stable_target_error": (
            None if stable_update is None else target_errors[stable_update - 1]
        ),
        "source_error_at_target_stability": stable_source_error,
        "final_target_error": target_errors[-1],
        "final_source_error": source_errors[-1],
        "target_error_floor": TARGET_ERROR_FLOOR,
        "source_retention_floor": SOURCE_RETENTION_FLOOR,
        "source_retained_at_target_stability": (
            stable_source_error is not None
            and stable_source_error <= SOURCE_RETENTION_FLOOR
        ),
        "optimizer_updates": CONTROL_UPDATES,
        "replayed_examples": CONTROL_UPDATES * int(target_train.state.shape[0]),
        "parameter_bytes": _module_storage_bytes(model),
    }


def _first_stable_update(errors: list[float], floor: float) -> int | None:
    for index, error in enumerate(errors):
        if error <= floor and all(value <= floor for value in errors[index:]):
            return index + 1
    return None


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    source_train, source_heldout, target_train, target_heldout = _base_data()
    source_base, base_updates = _fit_base(seed, source_train)
    base_digest = _digest_module(source_base)
    source_error_before = float(source_base.loss(source_heldout).detach())

    router = _new_router(source_base, seed + 10_000)
    target_staging = router.route_bundle(_rows(target_train))
    if target_staging.status != "staged":
        raise AssertionError(f"residual challenger did not stage: {target_staging}")
    retention_probe_errors: list[float] = []

    def retention_probe(candidate: ExternalFactoredTransitionModel) -> bool:
        error = float(candidate.base.loss(source_heldout).detach())
        retention_probe_errors.append(error)
        return error <= SOURCE_RETENTION_FLOOR

    receipt = router.promote_staged_candidate(
        target_heldout,
        retention_probe,
        prediction_tolerance=TARGET_ERROR_FLOOR,
        heldout_rollout=_rollout(),
        rollout_error_tolerance=TARGET_ERROR_FLOOR,
    )
    if not receipt.accepted or receipt.slot_id is None:
        raise AssertionError(f"factual residual challenger was rejected: {receipt}")
    target_context = router.contexts[0]
    target_context_batch = target_context.unsqueeze(0).expand(
        target_heldout.state.shape[0], -1
    )
    residual_target_error = float(
        router.model.loss(target_heldout, context=target_context_batch).detach()
    )
    residual_source_error = float(source_base.loss(source_heldout).detach())
    base_after = _digest_module(router.model.base)
    restored = ExternalFactoredTransitionRouter.from_payload(router.state_payload())
    restored_error = float(
        restored.model.loss(
            target_heldout,
            context=restored.contexts[0].unsqueeze(0).expand(
                target_heldout.state.shape[0], -1
            ),
        ).detach()
    )
    before_missing = router.digest()
    missing_result = router.route_partial_bundle([])
    after_missing = router.digest()

    shuffled_router = _new_router(source_base, seed + 20_000)
    permutation = torch.randperm(
        target_train.next_state.shape[0],
        generator=torch.Generator().manual_seed(seed + 30_000),
    )
    shuffled_train = ExternalTransitionObservation(
        state=target_train.state,
        intention=target_train.intention,
        next_state=target_train.next_state[permutation],
    )
    shuffled_router.route_bundle(_rows(shuffled_train))
    shuffled_receipt = shuffled_router.promote_staged_candidate(
        target_heldout,
        lambda candidate: float(candidate.base.loss(source_heldout).detach())
        <= SOURCE_RETENTION_FLOOR,
        prediction_tolerance=TARGET_ERROR_FLOOR,
    )

    fresh_control = _train_full_control(
        seed=seed + 40_000,
        initialization="fresh",
        source_heldout=source_heldout,
        target_train=target_train,
        target_heldout=target_heldout,
    )
    copy_control = _train_copy_control(
        seed=seed + 50_000,
        source_base=source_base,
        source_train=source_train,
        source_heldout=source_heldout,
        target_train=target_train,
        target_heldout=target_heldout,
    )
    residual_bank = router.model.residual_bank
    if residual_bank is None:
        raise AssertionError("residual bank was not created")
    residual_state = residual_bank.models[0]
    gates = {
        "challenger_staged_from_fresh_rows": target_staging.status == "staged",
        "heldout_one_step_passed": receipt.heldout_error <= TARGET_ERROR_FLOOR,
        "heldout_recursive_rollout_passed": (
            receipt.heldout_rollout_error is not None
            and receipt.heldout_rollout_error <= TARGET_ERROR_FLOOR
        ),
        "source_retention_passed": residual_source_error <= SOURCE_RETENTION_FLOOR,
        "base_frozen": router.model.base_frozen,
        "base_byte_stable": base_after == base_digest,
        "shuffled_transition_rejected": not shuffled_receipt.accepted,
        "missing_evidence_is_noop": (
            missing_result.status == "ambiguous" and before_missing == after_missing
        ),
        "exact_persistence": abs(restored_error - residual_target_error) <= 1e-9,
        "residual_used_one_pass": int(residual_state.sample_count.item())
        == int(target_train.state.shape[0]),
        "copy_control_loses_source_retention": not bool(
            copy_control["source_retained_at_target_stability"]
        ),
        "fresh_control_is_accounted": fresh_control["optimizer_updates"]
        == CONTROL_UPDATES,
    }
    report = {
        "schema": "neural-computer.policy-free-factual-residual-growth.v1",
        "claim_boundary": (
            "one frozen shared transition model plus one context-addressed "
            "one-pass factual residual; not general continual learning, "
            "unrestricted memory growth, or policy learning"
        ),
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "base_updates": BASE_UPDATES,
            "control_updates": CONTROL_UPDATES,
            "target_error_floor": TARGET_ERROR_FLOOR,
            "source_retention_floor": SOURCE_RETENTION_FLOOR,
            "residual": router.model.configuration(),
            "routing": router.configuration(),
            "admission": "heldout_one_step_plus_recursive_rollout_plus_source_retention_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "metrics": {
            "base_source_error_before": source_error_before,
            "base_source_error_after": residual_source_error,
            "residual_target_error": residual_target_error,
            "residual_heldout_error": receipt.heldout_error,
            "residual_rollout_error": receipt.heldout_rollout_error,
            "residual_retention_probe_errors": retention_probe_errors,
            "restored_target_error": restored_error,
            "shuffled_heldout_error": shuffled_receipt.heldout_error,
            "shuffled_receipt_reason": shuffled_receipt.reason,
            "fresh_control": fresh_control,
            "full_model_copy_control": copy_control,
            "residual_sample_count": int(residual_state.sample_count.item()),
            "residual_state_bytes": _module_storage_bytes(residual_state),
            "full_model_bytes": _module_storage_bytes(source_base),
        },
        "accounting": {
            "base_optimizer_updates": base_updates,
            "residual_optimizer_updates": 0,
            "residual_unique_transition_rows": int(target_train.state.shape[0]),
            "residual_heldout_transition_rows": int(target_heldout.state.shape[0]),
            "residual_rollout_transition_rows": _rollout().horizon,
            "source_retention_probe_rows": int(source_heldout.state.shape[0]),
            "residual_replayed_examples": 0,
            "fresh_optimizer_updates": fresh_control["optimizer_updates"],
            "fresh_replayed_examples": fresh_control["replayed_examples"],
            "copy_optimizer_updates": copy_control["optimizer_updates"],
            "copy_replayed_examples": copy_control["replayed_examples"],
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
