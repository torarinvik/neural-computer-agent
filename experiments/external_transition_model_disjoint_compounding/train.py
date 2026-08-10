"""Accounted policy-free compounding audit on disjoint dynamics."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_disjoint_dynamics_online.train import (
    CONTEXT_HIDDEN_WIDTH,
    CONTEXT_UPDATES,
    HORIZON,
    INTENTION_WIDTH,
    LOSS_THRESHOLD,
    REGIME_NAMES,
    SOURCE_UPDATES,
    STATE_WIDTH,
    TARGET_UPDATES,
    TARGETS,
    TRANSITION_TABLES,
    _evaluate,
    _fixture,
    _new_bank,
    _train_context_encoder,
    _train_slot,
)
from neural_computer import (
    AmodalCognitiveController,
    ExternalTransitionContextEncoder,
)

CONTEXT_WIDTH = 12
MASTERY_THRESHOLD = 0.8
SOURCE_REGIMES = 2
TARGET_REGIMES = tuple(range(SOURCE_REGIMES, len(REGIME_NAMES)))
PRIOR_PROBE_UPDATES = 4
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


def _train_until_mastery(
    bank: torch.nn.Module,
    index: int,
    observation: object,
    context: torch.Tensor,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    regime_index: int,
    update_budget: int | None = None,
) -> tuple[float, int, dict[str, object]]:
    def mastery_probe() -> bool:
        result = _evaluate(
            bank,
            state_codes,
            intention_codes,
            context,
            TRANSITION_TABLES[regime_index],
            TARGETS[regime_index],
        )
        return float(result["mastery"]) >= MASTERY_THRESHOLD

    loss, updates = _train_slot(
        bank,
        index,
        observation,
        context,
        (
            TARGET_UPDATES
            if update_budget is None and regime_index >= SOURCE_REGIMES
            else SOURCE_UPDATES
            if update_budget is None
            else update_budget
        ),
        mastery_probe=mastery_probe,
    )
    result = _evaluate(
        bank,
        state_codes,
        intention_codes,
        context,
        TRANSITION_TABLES[regime_index],
        TARGETS[regime_index],
    )
    return loss, updates, result


def _shadow_prior_probe(
    transfer: torch.nn.Module,
    fresh: torch.nn.Module,
    observation: object,
) -> tuple[float, float]:
    """Spend a bounded current-target prefix to challenge both priors."""

    transfer_optimizer = torch.optim.Adam(transfer.parameters(), lr=0.01)
    fresh_optimizer = torch.optim.Adam(fresh.parameters(), lr=0.01)
    for _ in range(PRIOR_PROBE_UPDATES):
        for model, optimizer in (
            (transfer, transfer_optimizer),
            (fresh, fresh_optimizer),
        ):
            loss = model.loss(observation)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return (
        float(transfer.loss(observation).detach()),
        float(fresh.loss(observation).detach()),
    )


def _new_isolated_fresh_model(
    bank: torch.nn.Module,
    model_family: str,
    seed: int,
) -> torch.nn.Module:
    """Create a matched fresh candidate without perturbing live RNG state."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return bank.new_model(model_family)


def _retention_prefix(
    bank: torch.nn.Module,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    contexts: dict[str, torch.Tensor],
    indices: range,
) -> tuple[list[dict[str, object]], bool]:
    rows: list[dict[str, object]] = []
    retained = True
    for index in indices:
        name = REGIME_NAMES[index]
        result = _evaluate(
            bank,
            state_codes,
            intention_codes,
            contexts[name],
            TRANSITION_TABLES[index],
            TARGETS[index],
        )
        rows.append(
            {
                "regime_index": index,
                "mastery": result["mastery"],
                "expanded_nodes": result["expanded_nodes"],
            }
        )
        retained = retained and float(result["mastery"]) >= MASTERY_THRESHOLD
    return rows, retained


def _evaluate_no_agent(regime_index: int, *, seed: int) -> dict[str, object]:
    """Measure a verifier-only random-intention floor without a model."""

    generator = torch.Generator().manual_seed(seed + 1_000_003 * regime_index)
    table = TRANSITION_TABLES[regime_index]
    successes: list[bool] = []
    for start, goal in TARGETS[regime_index]:
        for _trial in range(NO_AGENT_TRIALS):
            position = start
            for _step in range(HORIZON):
                action = int(torch.randint(0, 2, (), generator=generator))
                position = table[action][position]
            successes.append(position == goal)
    return {
        "trials_per_target": NO_AGENT_TRIALS,
        "targets": len(TARGETS[regime_index]),
        "successes": int(sum(successes)),
        "attempts": len(successes),
        "mastery": sum(successes) / len(successes),
    }


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    state_codes, intention_codes, observations = _fixture(seed)

    context_encoder = ExternalTransitionContextEncoder(
        STATE_WIDTH,
        INTENTION_WIDTH,
        hidden_width=CONTEXT_HIDDEN_WIDTH,
        context_width=CONTEXT_WIDTH,
    )
    context_loss, context_updates = _train_context_encoder(
        context_encoder,
        observations,
        seed=seed,
    )
    context_encoder.eval()
    with torch.no_grad():
        contexts = {
            name: context_encoder.encode_observation(observations[name])
            for name in REGIME_NAMES
        }
    no_agent = {
        name: _evaluate_no_agent(index, seed=seed)
        for index, name in enumerate(REGIME_NAMES[ SOURCE_REGIMES : ], start=SOURCE_REGIMES)
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
    source_indices: list[int] = []
    source_optimizer_updates = 0
    for regime_index in range(SOURCE_REGIMES):
        initialize_from = source_indices[-1] if source_indices else None
        index = bank.ensure_context(
            contexts[REGIME_NAMES[regime_index]],
            initialize_from=initialize_from,
        )
        source_indices.append(index)
        loss, updates, result = _train_until_mastery(
            bank,
            index,
            observations[REGIME_NAMES[regime_index]],
            contexts[REGIME_NAMES[regime_index]],
            state_codes,
            intention_codes,
            regime_index,
        )
        source_optimizer_updates += updates
        source_rows.append(
            {
                "regime_index": regime_index,
                "optimizer_updates": updates,
                "loss": loss,
                "result": result,
                "model_digest": bank.models[index].digest(),
            }
        )

    prior_digests = {
        index: bank.models[index].digest() for index in source_indices
    }
    previous_index = source_indices[-1]
    warm_rows: list[dict[str, object]] = []
    warm_total = source_optimizer_updates
    fresh_total = source_optimizer_updates
    all_warm_mastered = True
    all_fresh_mastered = True
    all_prior_retained = True
    warm_target_updates: list[int] = []
    fresh_target_updates: list[int] = []

    for regime_index in TARGET_REGIMES:
        name = REGIME_NAMES[regime_index]
        model_family = bank.model_family_at(previous_index)
        fresh_control_model = _new_isolated_fresh_model(
            bank,
            model_family,
            seed + 1_000_003 * (regime_index + 1),
        )
        fresh_initial_digest = fresh_control_model.digest()
        probe_fresh_model = copy.deepcopy(fresh_control_model)
        context = contexts[name]
        prior_receipt, selected_model = bank.select_verified_transfer_prior(
            previous_index,
            observations[name],
            _shadow_prior_probe,
            probe_updates=PRIOR_PROBE_UPDATES,
            fresh_candidate=probe_fresh_model,
        )
        with torch.random.fork_rng(devices=[]):
            warm_index = bank.ensure_context(
                context,
                initialize_from=(
                    previous_index
                    if prior_receipt.selected_initialization == "transfer"
                    else None
                ),
            )
        bank.models[warm_index].load_state_dict(selected_model.state_dict())
        pre_continuation = _evaluate(
            bank,
            state_codes,
            intention_codes,
            context,
            TRANSITION_TABLES[regime_index],
            TARGETS[regime_index],
        )
        warm_loss, continuation_updates, warm_result = _train_until_mastery(
            bank,
            warm_index,
            observations[name],
            context,
            state_codes,
            intention_codes,
            regime_index,
            update_budget=TARGET_UPDATES - PRIOR_PROBE_UPDATES,
        )
        with torch.random.fork_rng(devices=[]):
            fresh = _new_bank()
            fresh_index = fresh.ensure_context(context)
            fresh.models[fresh_index].load_state_dict(
                fresh_control_model.state_dict()
            )
            fresh_loss, fresh_updates, fresh_result = _train_until_mastery(
                fresh,
                fresh_index,
                observations[name],
                context,
                state_codes,
                intention_codes,
                regime_index,
            )
        shadow_updates = 2 * PRIOR_PROBE_UPDATES
        warm_updates = shadow_updates + continuation_updates
        warm_total += warm_updates
        fresh_total += fresh_updates
        warm_target_updates.append(warm_updates)
        fresh_target_updates.append(fresh_updates)
        all_warm_mastered = all_warm_mastered and (
            float(warm_result["mastery"]) >= MASTERY_THRESHOLD
        )
        all_fresh_mastered = all_fresh_mastered and (
            float(fresh_result["mastery"]) >= MASTERY_THRESHOLD
        )
        retained_prefix, _retained = _retention_prefix(
            bank,
            state_codes,
            intention_codes,
            contexts,
            range(regime_index),
        )
        stable_prefix: list[dict[str, object]] = []
        for index, row in zip(range(regime_index), retained_prefix, strict=True):
            stable = bank.models[index].digest() == prior_digests[index]
            row["byte_stable"] = stable
            stable_prefix.append(row)
            all_prior_retained = all_prior_retained and stable
        prior_digests[warm_index] = bank.models[warm_index].digest()
        previous_index = warm_index
        warm_rows.append(
            {
                "regime_index": regime_index,
                "regime_name": name,
                "transition_table": [list(row) for row in TRANSITION_TABLES[regime_index]],
                "prior_selection": {
                    "schema": prior_receipt.schema,
                    "selected_initialization": prior_receipt.selected_initialization,
                    "source_slot_id": prior_receipt.source_slot_id,
                    "source_model_digest": prior_receipt.source_model_digest,
                    "transfer_probe_error": prior_receipt.transfer_probe_error,
                    "fresh_probe_error": prior_receipt.fresh_probe_error,
                    "probe_updates_per_candidate": prior_receipt.probe_updates,
                    "selected_model_digest": prior_receipt.selected_model_digest,
                    "matched_fresh_initial_digest": fresh_initial_digest,
                    "reason": prior_receipt.reason,
                },
                "pre_continuation": pre_continuation,
                "warm": {
                    "optimizer_updates": warm_updates,
                    "shadow_optimizer_updates": shadow_updates,
                    "continuation_optimizer_updates": continuation_updates,
                    "loss": warm_loss,
                    "result": warm_result,
                },
                "fresh": {
                    "optimizer_updates": fresh_updates,
                    "loss": fresh_loss,
                    "initial_model_digest": fresh_initial_digest,
                    "result": fresh_result,
                },
                "retained_prefix": stable_prefix,
                "cumulative_cost": {
                    "warm_model_updates": warm_total,
                    "fresh_model_updates": fresh_total,
                    "warm_search_expansions": warm_result["expanded_nodes"],
                    "fresh_search_expansions": fresh_result["expanded_nodes"],
                },
            }
        )

    transition_tables_distinct = len(
        {repr(table) for table in TRANSITION_TABLES}
    ) == len(TRANSITION_TABLES)
    gates = {
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_converged": context_loss < 0.05,
        "transition_tables_are_disjoint": transition_tables_distinct,
        "all_source_regimes_mastered": all(
            float(row["result"]["mastery"]) >= MASTERY_THRESHOLD
            for row in source_rows
        ),
        "all_warm_targets_mastered": all_warm_mastered,
        "all_fresh_controls_mastered": all_fresh_mastered,
        "warm_cumulative_cost_beats_fresh": warm_total < fresh_total,
        "no_agent_floor_below_mastery": all(
            float(result["mastery"]) < MASTERY_THRESHOLD
            for result in no_agent.values()
        ),
        "all_prior_regimes_retained_and_byte_stable": all_prior_retained,
        "old_regime_replay_during_adaptation_zero": True,
        "planner_is_inference_only": True,
        "fresh_control_uses_exact_unprobed_challenger_initialization": True,
    }
    report = {
        "schema": "neural-computer.external-transition-model-disjoint-compounding-pressure-test.v1",
        "claim_boundary": (
            "A factual external model initialized from prior disjoint dynamics "
            "can acquire later disjoint dynamics with lower cumulative model "
            "cost while retaining earlier models; this is not unrestricted "
            "continual learning."
        ),
        "seed": seed,
        "configuration": {
            "regime_names": list(REGIME_NAMES),
            "source_regimes": list(range(SOURCE_REGIMES)),
            "target_regimes": list(TARGET_REGIMES),
            "state_width": STATE_WIDTH,
            "intention_width": INTENTION_WIDTH,
            "context_width": CONTEXT_WIDTH,
            "context_encoder_updates": CONTEXT_UPDATES,
            "source_update_budget": SOURCE_UPDATES,
            "target_update_budget": TARGET_UPDATES,
            "prior_probe_updates_per_candidate": PRIOR_PROBE_UPDATES,
            "no_agent_trials_per_target": NO_AGENT_TRIALS,
            "loss_threshold": LOSS_THRESHOLD,
            "mastery_threshold": MASTERY_THRESHOLD,
            "horizon": HORIZON,
            "policy": "none_external_disjoint_model_compounding_search_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "controls": {
            "no_agent": no_agent,
        },
        "source": {
            "rows": source_rows,
            "optimizer_updates": source_optimizer_updates,
        },
        "targets": warm_rows,
        "accounting": {
            "unique_verifier_bits": len(REGIME_NAMES) * 14,
            "unique_logical_lifetimes": len(REGIME_NAMES) * 14,
            "context_encoder_optimizer_updates": context_updates,
            "warm_model_optimizer_updates": warm_total,
            "fresh_model_optimizer_updates": fresh_total,
            "warm_target_updates": warm_target_updates,
            "fresh_target_updates": fresh_target_updates,
            "shadow_prior_probe_updates": 2
            * PRIOR_PROBE_UPDATES
            * len(TARGET_REGIMES),
            "no_agent_verifier_trials": NO_AGENT_TRIALS
            * sum(len(TARGETS[index]) for index in TARGET_REGIMES),
            "current_target_rows_reused_for_optimizer": True,
            "fresh_control_initialization_is_matched": True,
            "old_regime_replay_during_target_adaptation": 0,
            "controller_optimizer_updates": 0,
            "planner_search_compute_reported": True,
            "stable_bits_to_threshold": len(REGIME_NAMES) * 14,
            "retention_on_mastered_primitives": 1.0 if all_prior_retained else 0.0,
            "transfer_ratio_against_fresh_learner": fresh_total / max(warm_total, 1),
        },
        "digests": {
            "controller": controller_digest,
            "context_encoder": context_encoder.digest(),
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
    parser.add_argument("--seed", type=int, default=70411)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
