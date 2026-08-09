"""Pressure test alternating online transition-context identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import (
    AmodalCognitiveController,
    ExternalModelBasedPlanner,
    ExternalOnlineTransitionContextRouter,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 16
MODEL_HIDDEN_WIDTH = 48
CONTEXT_HIDDEN_WIDTH = 48
POSITION_COUNT = 6
CONTEXT_UPDATES = 500
BASE_UPDATES = 1200
TARGET_UPDATES = 125
TARGET_LOSS_THRESHOLD = 0.01
ROUTER_MATCH_TOLERANCE = 0.02
ROUTER_MATCH_MARGIN = 0.01
ADMISSION_OBSERVATIONS = POSITION_COUNT * 2
MAX_CONTEXTS = 3
BASE_DELTAS = (-1, 1)
AUXILIARY_DELTAS = (-3, 3)
TARGET_DELTAS = (-2, 2)
CAPACITY_DELTAS = (-4, 4)
TARGETS = ((0, 4), (4, 0), (1, 5))
AUXILIARY_TARGETS = ((0, 3), (3, 0))


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(seed: int) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, ExternalTransitionObservation],
]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = F.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = F.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )

    def observations(deltas: tuple[int, int]) -> ExternalTransitionObservation:
        states: list[torch.Tensor] = []
        intentions: list[torch.Tensor] = []
        next_states: list[torch.Tensor] = []
        for position in range(POSITION_COUNT):
            for action_index, delta in enumerate(deltas):
                next_position = min(
                    POSITION_COUNT - 1,
                    max(0, position + delta),
                )
                states.append(state_codes[position])
                intentions.append(intention_codes[action_index])
                next_states.append(state_codes[next_position])
        return ExternalTransitionObservation(
            state=torch.stack(states),
            intention=torch.stack(intentions),
            next_state=torch.stack(next_states),
            confidence=torch.ones(POSITION_COUNT * 2),
        )

    return (
        state_codes,
        intention_codes,
        {
            "base": observations(BASE_DELTAS),
            "auxiliary": observations(AUXILIARY_DELTAS),
            "target": observations(TARGET_DELTAS),
            "capacity": observations(CAPACITY_DELTAS),
        },
    )


def _train_context_encoder(
    encoder: ExternalTransitionContextEncoder,
    observations: dict[str, ExternalTransitionObservation],
    *,
    seed: int,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.003)
    final_loss = float("inf")
    for update in range(1, CONTEXT_UPDATES + 1):
        views: list[torch.Tensor] = []
        paired: list[torch.Tensor] = []
        for index, name in enumerate(("base", "auxiliary")):
            observation = observations[name]
            left = observation.state + 0.01 * torch.randn(
                observation.state.shape,
                generator=torch.Generator().manual_seed(seed + update * 11 + index),
            )
            right = observation.state + 0.02 * torch.randn(
                observation.state.shape,
                generator=torch.Generator().manual_seed(seed + update * 17 + index),
            )
            left_observation = ExternalTransitionObservation(
                state=left,
                intention=observation.intention,
                next_state=observation.next_state,
                confidence=observation.confidence,
            )
            right_observation = ExternalTransitionObservation(
                state=right,
                intention=observation.intention,
                next_state=observation.next_state,
                confidence=observation.confidence,
            )
            views.append(encoder.encode_observation(left_observation))
            paired.append(encoder.encode_observation(right_observation))
        left_context = torch.stack(views)
        right_context = torch.stack(paired)
        loss = encoder.contrastive_loss(left_context, right_context, temperature=0.1)
        final_loss = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return final_loss, CONTEXT_UPDATES


def _train_slot(
    bank: ExternalTransitionModelBank,
    index: int,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
    updates: int,
    *,
    stop_at_threshold: bool = False,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(bank.models[index].parameters(), lr=0.01)
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    final_loss = float("inf")
    for update in range(1, updates + 1):
        final_loss = bank.adaptation_step(observation, context_batch, optimizer)
        if stop_at_threshold and final_loss <= TARGET_LOSS_THRESHOLD:
            return final_loss, update
    return final_loss, updates


def _train_router_result(
    router: ExternalOnlineTransitionContextRouter,
    result,
    optimizers: dict[int, torch.optim.Optimizer],
) -> tuple[float, int, int]:
    if result.slot_index is None or result.observation is None:
        return 0.0, 0, 0
    optimizer = optimizers.get(result.slot_index)
    if optimizer is None:
        optimizer = torch.optim.Adam(
            router.bank.models[result.slot_index].parameters(),
            lr=0.01,
        )
        optimizers[result.slot_index] = optimizer
    loss = router.adaptation_step(result, optimizer)
    updates = 1
    current_rows = result.observation.state.shape[0]
    while (
        result.status == "admitted"
        and loss > TARGET_LOSS_THRESHOLD
        and updates < TARGET_UPDATES
    ):
        loss = router.adaptation_step(result, optimizer)
        updates += 1
    return loss, updates, current_rows


def _rows(observation: ExternalTransitionObservation) -> list[ExternalTransitionObservation]:
    return [
        ExternalTransitionObservation(
            state=observation.state[index : index + 1],
            intention=observation.intention[index : index + 1],
            next_state=observation.next_state[index : index + 1],
            confidence=(
                None
                if observation.confidence is None
                else observation.confidence[index : index + 1]
            ),
        )
        for index in range(observation.state.shape[0])
    ]


def _execute_plan(
    intentions: torch.Tensor,
    intention_codes: torch.Tensor,
    start: int,
    deltas: tuple[int, int],
) -> int:
    position = start
    for intention in intentions:
        action = int(
            torch.linalg.vector_norm(intention_codes - intention, dim=-1).argmin()
        )
        position = min(POSITION_COUNT - 1, max(0, position + deltas[action]))
    return position


def _evaluate(
    bank: ExternalTransitionModelBank,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    context: torch.Tensor,
    deltas: tuple[int, int],
    *,
    horizon: int,
    targets: tuple[tuple[int, int], ...] = TARGETS,
) -> dict[str, object]:
    planner = ExternalModelBasedPlanner(bank, beam_width=16)
    successes: list[bool] = []
    for start, goal in targets:
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[goal].unsqueeze(0),
            intention_codes,
            horizon=horizon,
            transition_context=context.unsqueeze(0),
        )
        final = _execute_plan(result.intentions[0], intention_codes, start, deltas)
        successes.append(final == goal)
    return {"successes": successes, "mastery": sum(successes) / len(successes)}


def _factual_error(
    bank: ExternalTransitionModelBank,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> float:
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    prediction = bank(observation.state, observation.intention, context_batch)
    return float((prediction - observation.next_state).square().mean().detach())


def _new_bank(capacity: int | None = None) -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=MODEL_HIDDEN_WIDTH,
        capacity=capacity,
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.manual_seed(seed)
    state_codes, intention_codes, observations = _fixture(seed)
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=CONTEXT_HIDDEN_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    context_loss, context_updates = _train_context_encoder(
        encoder,
        observations,
        seed=seed,
    )
    encoder.eval()
    with torch.no_grad():
        base_context = encoder.encode_observation(observations["base"])
        auxiliary_context = encoder.encode_observation(observations["auxiliary"])
    controller = AmodalCognitiveController(
        width=STATE_WIDTH,
        workspace_slots=2,
        intention_width=INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_digest = _digest_module(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    bank = _new_bank(capacity=MAX_CONTEXTS)
    base_index = bank.ensure_context(base_context)
    auxiliary_index = bank.ensure_context(
        auxiliary_context,
        initialize_from=base_index,
    )
    base_loss, base_updates = _train_slot(
        bank,
        base_index,
        observations["base"],
        base_context,
        BASE_UPDATES,
    )
    auxiliary_loss, auxiliary_updates = _train_slot(
        bank,
        auxiliary_index,
        observations["auxiliary"],
        auxiliary_context,
        BASE_UPDATES,
    )
    prior_digests = {
        "base": bank.models[base_index].digest(),
        "auxiliary": bank.models[auxiliary_index].digest(),
    }
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=ROUTER_MATCH_TOLERANCE,
        match_margin=ROUTER_MATCH_MARGIN,
        admission_observations=ADMISSION_OBSERVATIONS,
        max_contexts=MAX_CONTEXTS,
    )
    sequence = [
        ("base", observations["base"]),
        ("auxiliary", observations["auxiliary"]),
        ("base", observations["base"]),
        ("target", observations["target"]),
        ("auxiliary", observations["auxiliary"]),
        ("target", observations["target"]),
        ("base", observations["base"]),
        ("capacity", observations["capacity"]),
    ]
    optimizers: dict[int, torch.optim.Optimizer] = {}
    learnable_slots: set[int] = set()
    route_counts: Counter[str] = Counter()
    assignments: dict[str, set[int]] = defaultdict(set)
    target_updates = 0
    target_current_rows = 0
    target_admissions = 0
    target_matches_after_admission = 0
    capacity_events = 0
    unexpected_matches: list[str] = []
    trace: list[dict[str, object]] = []
    def consume(regime: str, observation: ExternalTransitionObservation) -> None:
        nonlocal target_updates, target_current_rows
        nonlocal target_admissions, target_matches_after_admission
        nonlocal capacity_events
        for row in _rows(observation):
            result = router.observe(row)
            route_counts[f"{regime}:{result.status}"] += 1
            if result.slot_index is not None:
                assignments[regime].add(result.slot_index)
            if regime == "target" and result.status == "admitted":
                target_admissions += 1
            if regime == "target" and target_admissions and result.status == "matched":
                target_matches_after_admission += 1
            if regime == "capacity" and result.status == "capacity":
                capacity_events += 1
            if regime == "capacity" and result.slot_index is not None:
                unexpected_matches.append(regime)
            if result.status == "admitted" and result.slot_index is not None:
                learnable_slots.add(result.slot_index)
            if result.slot_index in learnable_slots:
                loss, updates, current_rows = _train_router_result(
                    router,
                    result,
                    optimizers,
                )
            else:
                loss, updates, current_rows = 0.0, 0, 0
            if regime == "target":
                target_updates += updates
                target_current_rows += current_rows * updates
            trace.append(
                {
                    "regime_observed_for_diagnostic_only": regime,
                    "status": result.status,
                    "slot_index": result.slot_index,
                    "pending_observations": result.pending_observations,
                    "loss": loss,
                    "updates": updates,
                }
            )

    for regime, observation in sequence:
        consume(regime, observation)
    target_index = min(assignments["target"]) if assignments["target"] else None
    target_context = (
        None if target_index is None else router.bank.context_at(target_index)
    )
    pre_growth_content_digest = bank.content_digest()

    def retention_probe(candidate: ExternalTransitionModelBank) -> bool:
        if target_index is None:
            return False
        candidate_target_context = candidate.context_at(target_index)
        base_result = _evaluate(
            candidate,
            state_codes,
            intention_codes,
            base_context,
            BASE_DELTAS,
            horizon=4,
        )
        auxiliary_result = _evaluate(
            candidate,
            state_codes,
            intention_codes,
            auxiliary_context,
            AUXILIARY_DELTAS,
            horizon=4,
            targets=AUXILIARY_TARGETS,
        )
        target_result = _evaluate(
            candidate,
            state_codes,
            intention_codes,
            candidate_target_context,
            TARGET_DELTAS,
            horizon=2,
        )
        return (
            float(base_result["mastery"]) >= 0.8
            and float(auxiliary_result["mastery"]) >= 0.8
            and float(target_result["mastery"]) >= 0.8
        )

    growth_receipt = router.grow_verified(4, retention_probe)
    if growth_receipt.accepted:
        consume("capacity_growth", observations["capacity"])
    capacity_index = (
        min(assignments["capacity_growth"])
        if assignments["capacity_growth"]
        else None
    )
    capacity_context = (
        None if capacity_index is None else router.bank.context_at(capacity_index)
    )
    base_after = _evaluate(
        bank,
        state_codes,
        intention_codes,
        base_context,
        BASE_DELTAS,
        horizon=4,
    )
    auxiliary_after = _evaluate(
        bank,
        state_codes,
        intention_codes,
        auxiliary_context,
        AUXILIARY_DELTAS,
        horizon=4,
        targets=AUXILIARY_TARGETS,
    )
    target_after = (
        {"successes": [], "mastery": 0.0}
        if target_context is None
        else _evaluate(
            bank,
            state_codes,
            intention_codes,
            target_context,
            TARGET_DELTAS,
            horizon=2,
        )
    )
    capacity_after = (
        {"successes": [], "mastery": 0.0}
        if capacity_context is None
        else _evaluate(
            bank,
            state_codes,
            intention_codes,
            capacity_context,
            CAPACITY_DELTAS,
            horizon=2,
        )
    )
    fresh = _new_bank()
    fresh_index = fresh.ensure_context(target_context)
    fresh_loss, fresh_updates = _train_slot(
        fresh,
        fresh_index,
        observations["target"],
        target_context,
        TARGET_UPDATES,
        stop_at_threshold=True,
    )
    fresh_result = _evaluate(
        fresh,
        state_codes,
        intention_codes,
        target_context,
        TARGET_DELTAS,
        horizon=2,
    )
    wrong_context = _evaluate(
        bank,
        state_codes,
        intention_codes,
        base_context,
        TARGET_DELTAS,
        horizon=2,
    )
    wrong_context_mse = _factual_error(
        bank,
        observations["target"],
        base_context,
    )
    corrupted = _new_bank()
    corrupted_index = corrupted.ensure_context(target_context)
    corrupted_observation = ExternalTransitionObservation(
        state=observations["target"].state,
        intention=observations["target"].intention,
        next_state=observations["target"].next_state.roll(1, dims=0),
        confidence=observations["target"].confidence,
    )
    corrupted_loss, corrupted_updates = _train_slot(
        corrupted,
        corrupted_index,
        corrupted_observation,
        target_context,
        TARGET_UPDATES,
        stop_at_threshold=True,
    )
    corrupted_result = _evaluate(
        corrupted,
        state_codes,
        intention_codes,
        target_context,
        TARGET_DELTAS,
        horizon=2,
    )
    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    gates = {
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_learns": context_loss < 0.05,
        "base_model_learns": base_loss < TARGET_LOSS_THRESHOLD,
        "auxiliary_model_learns": auxiliary_loss < TARGET_LOSS_THRESHOLD,
        "alternating_base_routing": assignments["base"] == {base_index},
        "alternating_auxiliary_routing": assignments["auxiliary"] == {auxiliary_index},
        "target_admitted_without_label": target_admissions == 1 and target_index == 2,
        "target_reused_after_admission": target_matches_after_admission >= 1,
        "target_mastery": float(target_after["mastery"]) >= 0.8,
        "target_faster_than_fresh": target_updates < fresh_updates,
        "base_retained": float(base_after["mastery"]) >= 0.8,
        "auxiliary_retained": float(auxiliary_after["mastery"]) >= 0.8,
        "prior_slots_byte_stable": (
            prior_digests["base"] == bank.models[base_index].digest()
            and prior_digests["auxiliary"] == bank.models[auxiliary_index].digest()
        ),
        "capacity_refusal_no_write": (
            capacity_events >= 1
            and not assignments["capacity"]
            and not unexpected_matches
        ),
        "capacity_growth_verified": (
            growth_receipt.accepted
            and growth_receipt.destination_capacity == MAX_CONTEXTS + 1
            and growth_receipt.content_digest_before == pre_growth_content_digest
            and capacity_index == MAX_CONTEXTS
            and bank.capacity == MAX_CONTEXTS + 1
            and bank.context_count == MAX_CONTEXTS + 1
            and float(capacity_after["mastery"]) >= 0.8
        ),
        "wrong_context_control": wrong_context_mse > TARGET_LOSS_THRESHOLD,
        "corruption_control": float(corrupted_result["mastery"]) < 0.8,
        "persistence_exact": (
            restored.bank.digest() == router.bank.digest()
            and restored.context_encoder.digest() == router.context_encoder.digest()
            and restored.pending_observations == router.pending_observations
        ),
    }
    report = {
        "schema": "neural-computer.external-online-transition-context-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "context_updates": CONTEXT_UPDATES,
            "base_updates": BASE_UPDATES,
            "target_updates_limit": TARGET_UPDATES,
            "admission_observations": ADMISSION_OBSERVATIONS,
            "max_contexts": MAX_CONTEXTS,
            "match_tolerance": ROUTER_MATCH_TOLERANCE,
            "match_margin": ROUTER_MATCH_MARGIN,
            "regime_labels_used_by_router": False,
            "policy": "none_external_online_factual_route_then_context_admission_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "context_encoder": {
            "optimizer_updates": context_updates,
            "loss": context_loss,
        },
        "pretraining": {
            "base_optimizer_updates": base_updates,
            "auxiliary_optimizer_updates": auxiliary_updates,
            "base_loss": base_loss,
            "auxiliary_loss": auxiliary_loss,
        },
        "routing": {
            "counts": dict(route_counts),
            "assignments": {name: sorted(values) for name, values in assignments.items()},
            "target_admissions": target_admissions,
            "target_matches_after_admission": target_matches_after_admission,
            "capacity_events": capacity_events,
            "trace": trace,
        },
        "target": {
            "slot_index": target_index,
            "optimizer_updates": target_updates,
            "current_stream_rows_replayed": target_current_rows,
            "old_prior_rows_replayed": 0,
            "mastery": target_after,
        },
        "retention": {
            "base": base_after,
            "auxiliary": auxiliary_after,
            "capacity_growth_regime": capacity_after,
        },
        "capacity_growth": {
            "accepted": growth_receipt.accepted,
            "source_capacity": growth_receipt.source_capacity,
            "destination_capacity": growth_receipt.destination_capacity,
            "context_count": growth_receipt.context_count,
            "content_digest_before": growth_receipt.content_digest_before,
            "content_digest_after": growth_receipt.content_digest_after,
            "reason": growth_receipt.reason,
            "replayed_old_prior_rows": 0,
        },
        "fresh_target": {
            "optimizer_updates": fresh_updates,
            "loss": fresh_loss,
            "mastery": fresh_result,
        },
        "wrong_context": {
            "planner_diagnostic": wrong_context,
            "target_observation_mse": wrong_context_mse,
        },
        "corrupted_target": {
            "optimizer_updates": corrupted_updates,
            "loss": corrupted_loss,
            "mastery": corrupted_result,
        },
        "accounting": {
            "controller_parameter_updates": 0,
            "external_context_encoder_updates": context_updates,
            "external_prior_replay_during_target": 0,
            "target_current_stream_replay": target_current_rows,
            "unique_target_transition_lifetimes": POSITION_COUNT * 2,
            "context_slots_before_growth": MAX_CONTEXTS,
            "context_slots_after_capacity_growth": bank.context_count,
        },
        "digests": {
            "controller": controller_digest,
            "bank": bank.digest(),
            "context_encoder": encoder.digest(),
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=70011)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
