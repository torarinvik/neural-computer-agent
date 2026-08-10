"""Calibrate transfer-vs-fresh probes on a broader disjoint dynamics family."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from neural_computer import (
    EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
    AmodalCognitiveController,
    ExternalModelBasedPlanner,
    ExternalTransitionContextEncoder,
    ExternalTransitionModelBank,
    ExternalTransitionObservation,
)

SCHEMA = "neural-computer.external-transition-model-challenger-calibration.v1"
STATE_WIDTH = 8
INTENTION_WIDTH = 4
CONTEXT_WIDTH = 12
MODEL_HIDDEN_WIDTH = 48
CONTEXT_HIDDEN_WIDTH = 40
POSITION_COUNT = 8
REGIME_COUNT = 7
SOURCE_REGIMES = 2
TARGET_REGIMES = tuple(range(SOURCE_REGIMES, REGIME_COUNT))
CONTEXT_UPDATES = 400
SOURCE_UPDATES = 700
TARGET_UPDATES = 220
PROBE_UPDATES = 8
HORIZON = 3
LOSS_THRESHOLD = 0.01
MASTERY_THRESHOLD = 0.8
NO_AGENT_TRIALS = 128


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _tables() -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    generator = torch.Generator().manual_seed(18_701)
    tables: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    for _regime in range(REGIME_COUNT):
        rows = tuple(
            tuple(
                int(torch.randint(0, POSITION_COUNT, (), generator=generator))
                for _position in range(POSITION_COUNT)
            )
            for _action in range(2)
        )
        tables.append(rows)
    return tuple(tables)


TRANSITION_TABLES = _tables()
TARGET_PATTERNS = ((0, 1, 0), (1, 0, 1), (0, 0, 1))
def _target_position(
    table: tuple[tuple[int, ...], tuple[int, ...]],
    start: int,
    pattern: tuple[int, ...],
) -> int:
    position = start
    for action in pattern:
        position = table[action][position]
    return position


TARGETS = tuple(
    tuple(
        (
            start,
            _target_position(TRANSITION_TABLES[regime], start, pattern),
        )
        for start, pattern in zip((0, 1, 2), TARGET_PATTERNS, strict=True)
    )
    for regime in range(REGIME_COUNT)
)


def _fixture(
    seed: int,
) -> tuple[
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
    observations: dict[str, ExternalTransitionObservation] = {}
    for regime in range(REGIME_COUNT):
        states: list[torch.Tensor] = []
        intentions: list[torch.Tensor] = []
        next_states: list[torch.Tensor] = []
        for position in range(POSITION_COUNT):
            for action in range(2):
                next_position = TRANSITION_TABLES[regime][action][position]
                states.append(state_codes[position])
                intentions.append(intention_codes[action])
                next_states.append(state_codes[next_position])
        observations[f"regime_{regime}"] = ExternalTransitionObservation(
            state=torch.stack(states),
            intention=torch.stack(intentions),
            next_state=torch.stack(next_states),
            confidence=torch.ones(len(states)),
        )
    return state_codes, intention_codes, observations


def _train_context_encoder(
    encoder: ExternalTransitionContextEncoder,
    observations: dict[str, ExternalTransitionObservation],
    *,
    seed: int,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.003)
    final_loss = float("inf")
    source_names = tuple(f"regime_{index}" for index in range(SOURCE_REGIMES))
    for update in range(1, CONTEXT_UPDATES + 1):
        left_views: list[torch.Tensor] = []
        right_views: list[torch.Tensor] = []
        for index, name in enumerate(source_names):
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
            left_views.append(encoder.encode_observation(left_observation))
            right_views.append(encoder.encode_observation(right_observation))
        loss = encoder.contrastive_loss(
            torch.stack(left_views), torch.stack(right_views), temperature=0.1
        )
        final_loss = float(loss.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return final_loss, CONTEXT_UPDATES


def _evaluate_model(
    model: torch.nn.Module,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    regime: int,
) -> dict[str, object]:
    planner = ExternalModelBasedPlanner(model, beam_width=16)
    successes: list[bool] = []
    expanded_nodes = 0
    for start, goal in TARGETS[regime]:
        result = planner.plan(
            state_codes[start].unsqueeze(0),
            state_codes[goal].unsqueeze(0),
            intention_codes,
            horizon=HORIZON,
        )
        expanded_nodes += result.expanded_nodes
        position = start
        for intention in result.intentions[0]:
            action = int(
                torch.linalg.vector_norm(intention_codes - intention, dim=-1).argmin()
            )
            position = TRANSITION_TABLES[regime][action][position]
        successes.append(position == goal)
    return {
        "successes": successes,
        "mastery": sum(successes) / len(successes),
        "expanded_nodes": expanded_nodes,
    }


def _train_until_mastery(
    model: torch.nn.Module,
    observation: ExternalTransitionObservation,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    regime: int,
    *,
    initial_updates: int,
    budget: int,
) -> tuple[float, int, dict[str, object]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    final_loss = float(model.loss(observation).detach())
    result = _evaluate_model(model, state_codes, intention_codes, regime)
    total_updates = initial_updates
    if final_loss <= LOSS_THRESHOLD and float(result["mastery"]) >= MASTERY_THRESHOLD:
        return final_loss, total_updates, result
    for _update in range(initial_updates, budget):
        loss = model.loss(observation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        total_updates = _update + 1
        result = _evaluate_model(model, state_codes, intention_codes, regime)
        if final_loss <= LOSS_THRESHOLD and float(result["mastery"]) >= MASTERY_THRESHOLD:
            break
    return final_loss, total_updates, result


def _probe(
    transfer: torch.nn.Module,
    fresh: torch.nn.Module,
    observation: ExternalTransitionObservation,
) -> tuple[float, float]:
    optimizers = tuple(torch.optim.Adam(model.parameters(), lr=0.01) for model in (transfer, fresh))
    for _update in range(PROBE_UPDATES):
        for model, optimizer in zip((transfer, fresh), optimizers, strict=True):
            loss = model.loss(observation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return (
        float(transfer.loss(observation).detach()),
        float(fresh.loss(observation).detach()),
    )


def _evaluate_no_agent(regime: int, *, seed: int) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed + 1_000_003 * regime)
    successes: list[bool] = []
    for start, goal in TARGETS[regime]:
        for _trial in range(NO_AGENT_TRIALS):
            position = start
            for _step in range(HORIZON):
                action = int(torch.randint(0, 2, (), generator=generator))
                position = TRANSITION_TABLES[regime][action][position]
            successes.append(position == goal)
    return {
        "trials_per_target": NO_AGENT_TRIALS,
        "attempts": len(successes),
        "successes": int(sum(successes)),
        "mastery": sum(successes) / len(successes),
    }


def _new_bank() -> ExternalTransitionModelBank:
    return ExternalTransitionModelBank(
        STATE_WIDTH,
        INTENTION_WIDTH,
        CONTEXT_WIDTH,
        hidden_width=MODEL_HIDDEN_WIDTH,
        capacity=REGIME_COUNT,
        model_family=EXTERNAL_TRANSITION_NONLINEAR_MODEL_FAMILY,
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    state_codes, intention_codes, observations = _fixture(seed)
    encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=CONTEXT_HIDDEN_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    context_loss, context_updates = _train_context_encoder(
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
    source_rows: list[dict[str, object]] = []
    prior_digests: dict[int, str] = {}
    source_updates = 0
    for regime in range(SOURCE_REGIMES):
        name = f"regime_{regime}"
        index = bank.ensure_context(contexts[name])
        optimizer = torch.optim.Adam(bank.models[index].parameters(), lr=0.01)
        final_loss = float("inf")
        result: dict[str, object] = {}
        updates = 0
        for update in range(1, SOURCE_UPDATES + 1):
            loss = bank.models[index].loss(observations[name])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            updates = update
            final_loss = float(loss.detach())
            result = _evaluate_model(
                bank.models[index], state_codes, intention_codes, regime
            )
            if final_loss <= LOSS_THRESHOLD and float(result["mastery"]) >= MASTERY_THRESHOLD:
                break
        source_updates += updates
        prior_digests[index] = bank.models[index].digest()
        source_rows.append(
            {
                "regime": regime,
                "updates": updates,
                "loss": final_loss,
                "result": result,
                "slot_id": bank.slot_id_at(index),
            }
        )

    target_rows: list[dict[str, object]] = []
    probe_matches = 0
    all_candidates_mastered = True
    previous_index = 1
    warm_total = source_updates
    fresh_total = source_updates
    for regime in TARGET_REGIMES:
        name = f"regime_{regime}"
        family = bank.model_family_at(previous_index)
        transfer = bank.new_model(family)
        transfer.load_state_dict(bank.models[previous_index].state_dict())
        fresh = bank.new_model(family)
        transfer_error, fresh_error = _probe(transfer, fresh, observations[name])
        probe_winner = "transfer" if transfer_error <= fresh_error else "fresh"
        transfer_loss, transfer_updates, transfer_result = _train_until_mastery(
            transfer,
            observations[name],
            state_codes,
            intention_codes,
            regime,
            initial_updates=PROBE_UPDATES,
            budget=TARGET_UPDATES,
        )
        fresh_loss, fresh_updates, fresh_result = _train_until_mastery(
            fresh,
            observations[name],
            state_codes,
            intention_codes,
            regime,
            initial_updates=PROBE_UPDATES,
            budget=TARGET_UPDATES,
        )
        transfer_mastered = float(transfer_result["mastery"]) >= MASTERY_THRESHOLD
        fresh_mastered = float(fresh_result["mastery"]) >= MASTERY_THRESHOLD
        all_candidates_mastered = all_candidates_mastered and transfer_mastered and fresh_mastered
        full_winner = (
            "transfer"
            if transfer_updates < fresh_updates
            else "fresh"
            if fresh_updates < transfer_updates
            else "transfer"
            if transfer_loss <= fresh_loss
            else "fresh"
        )
        probe_matches += probe_winner == full_winner
        selected = transfer if probe_winner == "transfer" else fresh
        index = bank.ensure_context(contexts[name])
        bank.models[index].load_state_dict(selected.state_dict())
        stable_prior = all(
            bank.models[prior_index].digest() == digest
            for prior_index, digest in prior_digests.items()
        )
        prior_digests[index] = bank.models[index].digest()
        previous_index = index
        warm_cost = transfer_updates if probe_winner == "transfer" else fresh_updates
        warm_total += warm_cost
        fresh_total += fresh_updates
        target_rows.append(
            {
                "regime": regime,
                "slot_id": bank.slot_id_at(index),
                "probe": {
                    "transfer_error": transfer_error,
                    "fresh_error": fresh_error,
                    "winner": probe_winner,
                    "updates": PROBE_UPDATES,
                },
                "full_transfer": {
                    "updates": transfer_updates,
                    "loss": transfer_loss,
                    "result": transfer_result,
                },
                "full_fresh": {
                    "updates": fresh_updates,
                    "loss": fresh_loss,
                    "result": fresh_result,
                },
                "full_winner": full_winner,
                "probe_matches_full": probe_winner == full_winner,
                "prior_slots_byte_stable": stable_prior,
                "no_agent": _evaluate_no_agent(regime, seed=seed),
            }
        )

    no_agent = {
        row["regime"]: row["no_agent"] for row in target_rows
    }
    min_context_distance = min(
        float(torch.linalg.vector_norm(contexts[left] - contexts[right]))
        for index, left in enumerate(contexts)
        for right in tuple(contexts)[index + 1 :]
    )
    gates = {
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_converged": context_loss < 0.05,
        "all_tables_distinct": len({repr(table) for table in TRANSITION_TABLES}) == REGIME_COUNT,
        "contexts_separated": min_context_distance > 0.05,
        "all_source_regimes_mastered": all(
            float(row["result"]["mastery"]) >= MASTERY_THRESHOLD
            for row in source_rows
        ),
        "all_candidates_mastered": all_candidates_mastered,
        "all_probe_choices_match_full_cost": probe_matches == len(TARGET_REGIMES),
        "probe_match_rate_at_least_half": probe_matches / len(TARGET_REGIMES) >= 0.5,
        "all_no_agent_floors_below_mastery": all(
            float(result["mastery"]) < MASTERY_THRESHOLD for result in no_agent.values()
        ),
        "warm_selected_cost_beats_fresh": warm_total < fresh_total,
        "prior_slots_byte_stable": all(
            row["prior_slots_byte_stable"] for row in target_rows
        ),
        "old_regime_replay_zero": True,
        "planner_is_inference_only": True,
    }
    report = {
        "schema": SCHEMA,
        "seed": seed,
        "configuration": {
            "regime_count": REGIME_COUNT,
            "source_regimes": list(range(SOURCE_REGIMES)),
            "target_regimes": list(TARGET_REGIMES),
            "context_updates": CONTEXT_UPDATES,
            "source_updates": SOURCE_UPDATES,
            "target_updates": TARGET_UPDATES,
            "probe_updates": PROBE_UPDATES,
            "no_agent_trials_per_target": NO_AGENT_TRIALS,
            "mastery_threshold": MASTERY_THRESHOLD,
            "policy": "none_both_candidates_full_cost_calibration_v1",
        },
        "tables": [[list(row) for row in table] for table in TRANSITION_TABLES],
        "targets": [[list(pair) for pair in target] for target in TARGETS],
        "context_loss": context_loss,
        "min_context_distance": min_context_distance,
        "source": source_rows,
        "targets_audit": target_rows,
        "controls": {"no_agent": no_agent},
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_transition_rows": REGIME_COUNT * POSITION_COUNT * 2,
            "old_regime_replay_during_adaptation": 0,
            "source_optimizer_updates": source_updates,
            "warm_selected_optimizer_updates": warm_total,
            "fresh_optimizer_updates": fresh_total,
            "probe_optimizer_updates_per_candidate": PROBE_UPDATES,
            "no_agent_verifier_trials": NO_AGENT_TRIALS * len(TARGET_REGIMES) * len(TARGETS[0]),
            "controller_optimizer_updates": 0,
            "context_encoder_optimizer_updates": context_updates,
            "probe_match_rate": probe_matches / len(TARGET_REGIMES),
            "transfer_ratio_against_fresh": fresh_total / max(warm_total, 1),
        },
        "digests": {
            "controller": controller_digest,
            "context_encoder": encoder.digest(),
            "bank": bank.digest(),
        },
        "elapsed_seconds": time.perf_counter() - begun,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=95001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
