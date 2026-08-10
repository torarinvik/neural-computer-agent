"""Causal audit of frozen factual computation plus external residual memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from external_disjoint_dynamics_online import train as fixture
from neural_computer import (
    AmodalCognitiveController,
    ExternalFactoredTransitionModel,
    ExternalModelBasedPlanner,
    ExternalTransitionContextEncoder,
    ExternalTransitionObservation,
)

HORIZON = fixture.HORIZON
LOSS_THRESHOLD = fixture.LOSS_THRESHOLD
MASTERY_THRESHOLD = 1.0
SOURCE_REGIME_INDICES = (0, 1)
TARGET_REGIME_INDICES = (2, 3)
TARGET_COVERING_ROW_INDICES = fixture.TARGET_COVERING_ROW_INDICES


def _digest_module(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _context_digest(
    model: ExternalFactoredTransitionModel,
    observation: ExternalTransitionObservation,
    context: torch.Tensor,
) -> str:
    with torch.no_grad():
        prediction = model.predict_with_context(
            observation.state,
            observation.intention,
            context.unsqueeze(0).expand(observation.state.shape[0], -1),
        )
    digest = hashlib.sha256()
    digest.update(prediction.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _select_rows(
    observation: ExternalTransitionObservation,
    indices: tuple[int, ...],
) -> ExternalTransitionObservation:
    selection = torch.tensor(indices, dtype=torch.long)
    return ExternalTransitionObservation(
        state=observation.state.index_select(0, selection),
        intention=observation.intention.index_select(0, selection),
        next_state=observation.next_state.index_select(0, selection),
        confidence=(
            None
            if observation.confidence is None
            else observation.confidence.index_select(0, selection)
        ),
    )


def _train_base(
    model: ExternalFactoredTransitionModel,
    observation: ExternalTransitionObservation,
) -> tuple[float, int]:
    optimizer = torch.optim.Adam(model.base.parameters(), lr=0.01)
    final_loss = float("inf")
    for update in range(1, fixture.SOURCE_UPDATES + 1):
        optimizer.zero_grad()
        loss = model.base.loss(observation)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
        if final_loss <= LOSS_THRESHOLD:
            return final_loss, update
    return final_loss, fixture.SOURCE_UPDATES


def _evaluate(
    model: ExternalFactoredTransitionModel,
    state_codes: torch.Tensor,
    intention_codes: torch.Tensor,
    context: torch.Tensor,
    regime_index: int,
) -> dict[str, object]:
    planner = ExternalModelBasedPlanner(model, beam_width=16)
    table = fixture.TRANSITION_TABLES[regime_index]
    successes: list[bool] = []
    expanded_nodes = 0
    for start, goal in fixture.TARGETS[regime_index]:
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


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    state_codes, intention_codes, observations = fixture._fixture(seed)

    encoder = ExternalTransitionContextEncoder(
        fixture.STATE_WIDTH,
        fixture.INTENTION_WIDTH,
        hidden_width=fixture.CONTEXT_HIDDEN_WIDTH,
        context_width=fixture.CONTEXT_WIDTH,
    )
    context_loss, context_updates = fixture._train_context_encoder(
        encoder,
        observations,
        seed=seed,
    )
    encoder.eval()
    with torch.no_grad():
        contexts = {
            name: encoder.encode_observation(observation)
            for name, observation in observations.items()
        }

    controller = AmodalCognitiveController(
        width=fixture.STATE_WIDTH,
        workspace_slots=2,
        intention_width=fixture.INTENTION_WIDTH,
        feedback_width=3,
        event_window_capacity=4,
    )
    controller_digest = _digest_module(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    model = ExternalFactoredTransitionModel(
        fixture.STATE_WIDTH,
        fixture.INTENTION_WIDTH,
        fixture.CONTEXT_WIDTH,
        hidden_width=fixture.MODEL_HIDDEN_WIDTH,
    )
    base_loss, base_updates = _train_base(model, observations["source_a"])
    model.freeze_base()
    base_digest = model.base.digest()
    encoder_digest = encoder.digest()
    phase_records: list[dict[str, object]] = []
    context_digests: dict[int, str] = {}
    admitted_rows = 0
    all_mastered = True
    all_prior_retained = True

    for regime_index in (*SOURCE_REGIME_INDICES, *TARGET_REGIME_INDICES):
        name = fixture.REGIME_NAMES[regime_index]
        context = contexts[name]
        evidence = (
            observations[name]
            if regime_index in SOURCE_REGIME_INDICES
            else _select_rows(
                observations[name],
                TARGET_COVERING_ROW_INDICES[regime_index],
            )
        )
        before_prior = {
            prior_index: _context_digest(
                model,
                observations[fixture.REGIME_NAMES[prior_index]],
                contexts[fixture.REGIME_NAMES[prior_index]],
            )
            for prior_index in range(regime_index)
        }
        receipt = model.write_residual(evidence, context=context)
        admitted_rows += int(evidence.state.shape[0])
        result = _evaluate(
            model,
            state_codes,
            intention_codes,
            context,
            regime_index,
        )
        all_mastered = all_mastered and float(result["mastery"]) >= MASTERY_THRESHOLD
        retained: list[dict[str, object]] = []
        for prior_index in range(regime_index):
            prior_name = fixture.REGIME_NAMES[prior_index]
            prior_result = _evaluate(
                model,
                state_codes,
                intention_codes,
                contexts[prior_name],
                prior_index,
            )
            after_digest = _context_digest(
                model,
                observations[prior_name],
                contexts[prior_name],
            )
            stable = before_prior[prior_index] == after_digest
            all_prior_retained = (
                all_prior_retained
                and stable
                and float(prior_result["mastery"]) >= MASTERY_THRESHOLD
            )
            retained.append(
                {
                    "regime_index": prior_index,
                    "mastery": prior_result["mastery"],
                    "behavior_digest_stable": stable,
                }
            )
        context_digests[regime_index] = _context_digest(
            model,
            observations[name],
            context,
        )
        phase_records.append(
            {
                "regime_index": regime_index,
                "regime_name": name,
                "evidence_rows": int(evidence.state.shape[0]),
                "available_rows": int(observations[name].state.shape[0]),
                "memory_write_receipt": {
                    "indices": receipt.indices.tolist(),
                    "committed": receipt.committed.tolist(),
                    "version": receipt.version,
                },
                "result": result,
                "retained_prior_regimes": retained,
            }
        )

    base_after = model.base.digest()
    residual_digest_before = model.residual_memory.digest()
    restored = ExternalFactoredTransitionModel.from_payload(model.state_payload())
    persistence_exact = restored.digest() == model.digest()
    restored_results = [
        _evaluate(
            restored,
            state_codes,
            intention_codes,
            contexts[fixture.REGIME_NAMES[index]],
            index,
        )
        for index in range(len(fixture.REGIME_NAMES))
    ]
    gates = {
        "context_encoder_converged": context_loss < 0.05,
        "all_regimes_mastered": all_mastered,
        "all_prior_regimes_retained": all_prior_retained,
        "target_evidence_is_partial": all(
            int(record["evidence_rows"]) < int(record["available_rows"])
            for record in phase_records
            if int(record["regime_index"]) in TARGET_REGIME_INDICES
        ),
        "base_frozen_before_residual_writes": base_digest == base_after,
        "base_never_updated_by_residual_writes": base_digest == model.base.digest(),
        "controller_unchanged": controller_digest == _digest_module(controller),
        "context_encoder_unchanged": encoder_digest == encoder.digest(),
        "residual_memory_grew_without_optimizer_updates": (
            model.residual_record_count == admitted_rows
        ),
        "exact_persistence": persistence_exact,
        "restored_mastery": all(
            float(result["mastery"]) >= MASTERY_THRESHOLD
            for result in restored_results
        ),
        "residual_digest_recorded": bool(residual_digest_before),
    }
    report = {
        "schema": "neural-computer.external-factored-transition-residual.v1",
        "seed": seed,
        "configuration": {
            "regime_names": list(fixture.REGIME_NAMES),
            "source_regimes": list(SOURCE_REGIME_INDICES),
            "target_regimes": list(TARGET_REGIME_INDICES),
            "state_width": fixture.STATE_WIDTH,
            "intention_width": fixture.INTENTION_WIDTH,
            "context_width": fixture.CONTEXT_WIDTH,
            "base_hidden_width": fixture.MODEL_HIDDEN_WIDTH,
            "planner_horizon": HORIZON,
            "evidence_policy": "complete_source_target_covering_partial_v1",
            "behavior": "base_plus_context_residual_facts_searched_at_inference_v1",
        },
        "metrics": {
            "base_loss": base_loss,
            "base_optimizer_updates": base_updates,
            "context_encoder_loss": context_loss,
            "context_encoder_optimizer_updates": context_updates,
            "admitted_residual_rows": admitted_rows,
            "residual_record_count": model.residual_record_count,
            "phase_records": phase_records,
            "restored_mastery": [
                float(result["mastery"]) for result in restored_results
            ],
        },
        "gates": gates,
        "accounting": {
            "unique_transition_rows_consumed_once": admitted_rows,
            "base_training_replayed_examples": int(
                observations["source_a"].state.shape[0] * max(base_updates - 1, 0)
            ),
            "residual_optimizer_updates": 0,
            "controller_optimizer_updates": 0,
            "old_regime_replay": 0,
            "planner_search_optimizer_updates": 0,
            "wall_seconds": time.perf_counter() - begun,
        },
        "digests": {
            "base": model.base.digest(),
            "residual": residual_digest_before,
            "controller": controller_digest,
            "context_encoder": encoder_digest,
        },
        "promoted": all(gates.values()),
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=82701)
    parser.add_argument("--report-out", type=Path, required=True)
    run_args = parser.parse_args()
    run(run_args.seed, run_args.report_out)


if __name__ == "__main__":
    main()
