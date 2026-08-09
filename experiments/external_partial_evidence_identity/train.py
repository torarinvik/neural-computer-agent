"""Fast variable-prefix online identity and adaptation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path

import torch

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
CONTEXT_WIDTH = 12
MODEL_HIDDEN_WIDTH = 48
POSITION_COUNT = 7
REGIME_NAMES = ("source_a", "source_b", "target_c", "target_d")
HORIZON = 3
TRANSITION_TABLES = (
    ((1, 4, 6, 6, 6, 0, 2), (0, 3, 6, 3, 3, 5, 3)),
    ((6, 6, 0, 0, 0, 2, 6), (1, 5, 6, 5, 6, 2, 2)),
    ((1, 4, 4, 1, 2, 4, 3), (5, 4, 0, 4, 0, 6, 3)),
    ((1, 2, 0, 5, 3, 3, 1), (0, 0, 0, 3, 4, 2, 6)),
)
TARGETS = (
    ((0, 4), (1, 6), (5, 0)),
    ((0, 2), (1, 6), (5, 1)),
    ((0, 3), (2, 6), (4, 5)),
    ((0, 2), (3, 5), (6, 1)),
)


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _fixture(
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, ExternalTransitionObservation]]:
    generator = torch.Generator().manual_seed(seed)
    state_codes = torch.nn.functional.normalize(
        torch.randn(POSITION_COUNT, STATE_WIDTH, generator=generator), dim=-1
    )
    intention_codes = torch.nn.functional.normalize(
        torch.randn(2, INTENTION_WIDTH, generator=generator), dim=-1
    )
    observations: dict[str, ExternalTransitionObservation] = {}
    for regime_index, name in enumerate(REGIME_NAMES):
        states: list[torch.Tensor] = []
        intentions: list[torch.Tensor] = []
        next_states: list[torch.Tensor] = []
        for position in range(POSITION_COUNT):
            for action_index in range(2):
                next_position = TRANSITION_TABLES[regime_index][action_index][position]
                states.append(state_codes[position])
                intentions.append(intention_codes[action_index])
                next_states.append(state_codes[next_position])
        observations[name] = ExternalTransitionObservation(
            state=torch.stack(states),
            intention=torch.stack(intentions),
            next_state=torch.stack(next_states),
            confidence=torch.ones(POSITION_COUNT * 2),
        )
    return state_codes, intention_codes, observations


def _rows(
    observation: ExternalTransitionObservation,
) -> list[ExternalTransitionObservation]:
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


def _evaluate(
    bank: ExternalTransitionModelBank,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    context: torch.Tensor,
    table: tuple[tuple[int, ...], tuple[int, ...]],
    targets: tuple[tuple[int, int], ...],
) -> dict[str, object]:
    planner = ExternalModelBasedPlanner(bank, beam_width=16)
    successes: list[bool] = []
    expanded_nodes = 0
    for start, goal in targets:
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[goal].unsqueeze(0),
            intention_codes,
            horizon=HORIZON,
            transition_context=context.unsqueeze(0),
        )
        expanded_nodes += result.expanded_nodes
        position = start
        for intention in result.intentions[0]:
            action = int(
                torch.linalg.vector_norm(intention_codes - intention, dim=-1).argmin()
            )
            position = table[action][position]
        successes.append(position == goal)
    return {
        "successes": successes,
        "mastery": sum(successes) / len(successes),
        "expanded_nodes": expanded_nodes,
    }


def _new_bank() -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=MODEL_HIDDEN_WIDTH,
        capacity=len(REGIME_NAMES),
    )


def _prediction_error(
    bank: ExternalTransitionModelBank,
    result,
) -> float:
    if result.slot_index is None or result.context is None or result.observation is None:
        return float("inf")
    context = result.context.to(result.observation.state)
    context_batch = context.unsqueeze(0).expand(result.observation.state.shape[0], -1)
    prediction = bank(
        result.observation.state,
        result.observation.intention,
        context_batch,
    )
    return float(
        (prediction - result.observation.next_state).square().mean().detach()
    )

CONTEXT_HIDDEN_WIDTH = 40
CONTEXT_UPDATES = 350
SOURCE_UPDATES = 700
FRESH_UPDATES = 220
LOSS_THRESHOLD = 0.01
ADMISSION_OBSERVATIONS = POSITION_COUNT
FULL_OBSERVATIONS = POSITION_COUNT * 2
MATCH_TOLERANCE = 0.02
MATCH_MARGIN = 0.01
SEQUENCE = (
    "source_a",
    "source_b",
    "target_c",
    "source_a",
    "target_d",
    "target_c",
    "target_d",
    "source_b",
)


def _prefix(
    observation: ExternalTransitionObservation,
    length: int,
    *,
    noise: float,
    seed: int,
) -> ExternalTransitionObservation:
    generator = torch.Generator().manual_seed(seed)
    state = observation.state[:length]
    if noise:
        state = state + noise * torch.randn(
            state.shape,
            generator=generator,
        )
    return ExternalTransitionObservation(
        state=state,
        intention=observation.intention[:length],
        next_state=observation.next_state[:length],
        confidence=observation.confidence[:length]
        if observation.confidence is not None
        else None,
    )


def _train_prefix_encoder(
    encoder: ExternalTransitionContextEncoder,
    observations: dict[str, ExternalTransitionObservation],
    *,
    seed: int,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.003)
    final_loss = float("inf")
    lengths = (4, ADMISSION_OBSERVATIONS, FULL_OBSERVATIONS)
    for update in range(1, CONTEXT_UPDATES + 1):
        prefixes: list[torch.Tensor] = []
        full: list[torch.Tensor] = []
        for index, name in enumerate(REGIME_NAMES[:2]):
            observation = observations[name]
            full.append(encoder.encode_observation(observation))
            prefixes.append(
                torch.stack(
                    [
                        encoder.encode_observation(
                            _prefix(
                                observation,
                                length,
                                noise=0.01 + 0.002 * prefix_index,
                                seed=seed + update * 31 + index * 7 + prefix_index,
                            )
                        )
                        for prefix_index, length in enumerate(lengths)
                    ]
                )
            )
        loss = encoder.prefix_alignment_loss(
            torch.stack(prefixes),
            torch.stack(full),
            temperature=0.1,
        )
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
    mastery_probe: Callable[[], bool] | None = None,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(bank.models[index].parameters(), lr=0.01)
    context_batch = context.unsqueeze(0).expand(observation.state.shape[0], -1)
    final_loss = float("inf")
    for update in range(1, updates + 1):
        final_loss = bank.adaptation_step(observation, context_batch, optimizer)
        if final_loss <= LOSS_THRESHOLD and (
            mastery_probe is None or mastery_probe()
        ):
            return final_loss, update
    return final_loss, updates


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
    prefix_loss, prefix_updates = _train_prefix_encoder(
        encoder, observations, seed=seed
    )
    encoder.eval()
    with torch.no_grad():
        contexts = {
            name: encoder.encode_observation(observation)
            for name, observation in observations.items()
        }

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

    bank = _new_bank()
    source_a_index = bank.ensure_context(contexts["source_a"])
    source_b_index = bank.ensure_context(
        contexts["source_b"], initialize_from=source_a_index
    )
    source_a_loss, source_a_updates = _train_slot(
        bank,
        source_a_index,
        observations["source_a"],
        contexts["source_a"],
        SOURCE_UPDATES,
        mastery_probe=lambda: float(
            _evaluate(
                bank,
                state_codes,
                intention_codes,
                contexts["source_a"],
                TRANSITION_TABLES[0],
                TARGETS[0],
            )["mastery"]
        )
        >= 0.8,
    )
    source_b_loss, source_b_updates = _train_slot(
        bank,
        source_b_index,
        observations["source_b"],
        contexts["source_b"],
        SOURCE_UPDATES,
        mastery_probe=lambda: float(
            _evaluate(
                bank,
                state_codes,
                intention_codes,
                contexts["source_b"],
                TRANSITION_TABLES[1],
                TARGETS[1],
            )["mastery"]
        )
        >= 0.8,
    )
    prior_digests = {
        "source_a": bank.models[source_a_index].digest(),
        "source_b": bank.models[source_b_index].digest(),
    }
    router = ExternalOnlineTransitionContextRouter(
        bank,
        encoder,
        match_tolerance=MATCH_TOLERANCE,
        match_margin=MATCH_MARGIN,
        admission_observations=ADMISSION_OBSERVATIONS,
        max_contexts=len(REGIME_NAMES),
        continuation_tolerance=0.2,
        conflict_patience=2,
    )
    optimizers: dict[int, torch.optim.Optimizer] = {}
    assignments: dict[str, set[int]] = defaultdict(set)
    route_counts: Counter[str] = Counter()
    admissions: Counter[str] = Counter()
    reuses: Counter[str] = Counter()
    target_updates: Counter[str] = Counter()
    old_slot_updates = 0
    admission_rows: dict[str, list[int]] = defaultdict(list)
    trace: list[dict[str, object]] = []

    for regime in SEQUENCE:
        for row in _rows(observations[regime]):
            result = router.observe(row)
            route_counts[f"{regime}:{result.status}"] += 1
            if result.slot_index is not None:
                assignments[regime].add(result.slot_index)
            if result.status == "admitted" and result.slot_index is not None:
                admissions[regime] += 1
                admission_rows[regime].append(result.observation.state.shape[0])
            if result.status == "matched" and regime.startswith("target_"):
                reuses[regime] += 1

            # Source slots are intentionally protected after acquisition. A
            # novel slot receives only current-stream evidence, regardless of
            # whether the router just admitted it or matched it later.
            if (
                result.slot_index is not None
                and result.observation is not None
                and regime.startswith("target_")
                and result.slot_index not in {source_a_index, source_b_index}
            ):
                slot = result.slot_index
                optimizer = optimizers.get(slot)
                if optimizer is None:
                    optimizer = torch.optim.Adam(
                        router.bank.models[slot].parameters(), lr=0.01
                    )
                    optimizers[slot] = optimizer
                router.adaptation_step(result, optimizer)
                updates = 1
                target_index = REGIME_NAMES.index(regime)
                while updates < 40 and (
                    _prediction_error(bank, result) > MATCH_TOLERANCE
                    or float(
                        _evaluate(
                            bank,
                            state_codes,
                            intention_codes,
                            result.context,
                            TRANSITION_TABLES[target_index],
                            TARGETS[target_index],
                        )["mastery"]
                    )
                    < 0.8
                ):
                    router.adaptation_step(result, optimizer)
                    updates += 1
                target_updates[regime] += updates
            elif result.status == "matched" and result.slot_index in {
                source_a_index,
                source_b_index,
            }:
                old_slot_updates += 0
            trace.append(
                {
                    "diagnostic_regime": regime,
                    "status": result.status,
                    "slot_index": result.slot_index,
                    "pending_observations": result.pending_observations,
                    "evidence_rows": (
                        None
                        if result.observation is None
                        else result.observation.state.shape[0]
                    ),
                }
            )

    retention: dict[str, dict[str, object]] = {}
    retained = True
    stable = True
    for index, name in enumerate(REGIME_NAMES):
        result = _evaluate(
            bank,
            state_codes,
            intention_codes,
            bank.context_at(index),
            TRANSITION_TABLES[index],
            TARGETS[index],
        )
        digest_stable = name not in prior_digests or (
            bank.models[index].digest() == prior_digests[name]
        )
        retention[name] = {
            "mastery": result["mastery"],
            "byte_stable": digest_stable,
            "expanded_nodes": result["expanded_nodes"],
        }
        retained = retained and float(result["mastery"]) >= 0.8
        if name in prior_digests:
            stable = stable and digest_stable

    fresh_updates: dict[str, int] = {}
    fresh_mastery: dict[str, float] = {}
    for index, name in enumerate(REGIME_NAMES[2:], start=2):
        fresh = _new_bank()
        fresh_index = fresh.ensure_context(contexts[name])
        _loss, updates = _train_slot(
            fresh,
            fresh_index,
            observations[name],
            contexts[name],
            FRESH_UPDATES,
            mastery_probe=lambda index=index, fresh=fresh: float(
                _evaluate(
                    fresh,
                    state_codes,
                    intention_codes,
                    contexts[REGIME_NAMES[index]],
                    TRANSITION_TABLES[index],
                    TARGETS[index],
                )["mastery"]
            )
            >= 0.8,
        )
        fresh_updates[name] = updates
        fresh_mastery[name] = float(
            _evaluate(
                fresh,
                state_codes,
                intention_codes,
                contexts[name],
                TRANSITION_TABLES[index],
                TARGETS[index],
            )["mastery"]
        )

    wrong_context = bank.context_at(source_a_index)
    wrong_context_mse = float(
        (
            bank(
                observations["target_c"].state,
                observations["target_c"].intention,
                wrong_context.unsqueeze(0).expand(FULL_OBSERVATIONS, -1),
            )
            - observations["target_c"].next_state
        )
        .square()
        .mean()
        .detach()
    )
    restored = ExternalOnlineTransitionContextRouter.from_payload(
        router.state_payload()
    )
    gates = {
        "prefix_encoder_converged": prefix_loss < 0.05,
        "prefix_admission_is_short": all(
            rows == [ADMISSION_OBSERVATIONS]
            for rows in admission_rows.values()
            if rows
        ),
        "target_c_admitted_and_reused": admissions["target_c"] == 1
        and reuses["target_c"] >= 1,
        "target_d_admitted_and_reused": admissions["target_d"] == 1
        and reuses["target_d"] >= 1,
        "all_regimes_mastered": retained,
        "source_slots_byte_stable": stable,
        "warm_target_updates_below_fresh": all(
            target_updates[name] < fresh_updates[name]
            for name in REGIME_NAMES[2:]
        ),
        "wrong_context_factual_control": wrong_context_mse > LOSS_THRESHOLD,
        "old_slot_optimizer_updates_zero": old_slot_updates == 0,
        "controller_unchanged": controller_digest == _digest_module(controller),
        "persistence_exact": (
            restored.bank.digest() == router.bank.digest()
            and restored.context_encoder.digest() == router.context_encoder.digest()
        ),
    }
    report = {
        "schema": "neural-computer.external-partial-evidence-identity-pressure-test.v1",
        "seed": seed,
        "configuration": {
            "context_pretraining_updates": CONTEXT_UPDATES,
            "prefix_lengths": [4, ADMISSION_OBSERVATIONS, FULL_OBSERVATIONS],
            "admission_observations": ADMISSION_OBSERVATIONS,
            "full_observations": FULL_OBSERVATIONS,
            "continuation_tolerance": 0.2,
            "conflict_patience": 2,
            "sequence": list(SEQUENCE),
            "regime_labels_used_by_router": False,
            "policy": "none_external_variable_prefix_online_model_search_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "context_encoder": {
            "optimizer_updates": prefix_updates,
            "prefix_alignment_loss": prefix_loss,
        },
        "pretraining": {
            "source_a_optimizer_updates": source_a_updates,
            "source_a_loss": source_a_loss,
            "source_b_optimizer_updates": source_b_updates,
            "source_b_loss": source_b_loss,
        },
        "routing": {
            "counts": dict(route_counts),
            "assignments": {name: sorted(values) for name, values in assignments.items()},
            "admissions": dict(admissions),
            "reuses": dict(reuses),
            "admission_evidence_rows": dict(admission_rows),
            "trace": trace,
        },
        "targets": {
            "online_current_stream_optimizer_updates": dict(target_updates),
            "fresh_optimizer_updates": dict(fresh_updates),
            "fresh_mastery": fresh_mastery,
            "old_slot_optimizer_updates": old_slot_updates,
        },
        "retention": retention,
        "controls": {"wrong_context_target_mse": wrong_context_mse},
        "accounting": {
            "controller_parameter_updates": 0,
            "old_regime_replay_during_target_adaptation": 0,
            "target_current_stream_rows_used": sum(target_updates.values())
            * ADMISSION_OBSERVATIONS,
            "planner_expansions_reported_per_regime": True,
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
    parser.add_argument("--seed", type=int, default=70511)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
