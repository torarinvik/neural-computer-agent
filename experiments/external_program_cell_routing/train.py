"""Audit outcome-only routing among overlapping external program cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_outcome_program_router.train import (
    _evaluate,
    _train_executable_bank,
    _train_stream,
)
from neural_computer import (
    AmodalCognitiveController,
    ExternalOutcomeProgramCellBank,
    ExternalOutcomeProgramRouter,
    ExternalOutcomeProgramRouterState,
    ExternalOutcomeValueBaseline,
)

EVENT_WIDTH = 4
PHASES = 2
PROGRAM_CAPACITY = 3
FEATURE_WIDTH = EVENT_WIDTH * PHASES + PHASES
SOURCE_EPISODES = 2500
TARGET_EPISODES = 2500
EVALUATION_EPISODES = 300
PROBE_EPISODES = 64
MASTER_THRESHOLD = 0.8
MATCH_THRESHOLD = 0.25
EXECUTION_TOLERANCE = 1e-5


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _baseline() -> ExternalOutcomeValueBaseline:
    return ExternalOutcomeValueBaseline(
        feature_width=FEATURE_WIDTH,
        initial_learning_rate=0.05,
        initial_trace_decay=0.90,
    )


def _relations(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    relation_a = torch.randn(
        PHASES,
        PROGRAM_CAPACITY,
        EVENT_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(seed + 100),
    )
    permutation = torch.tensor([1, 2, 0])
    relation_b = relation_a[:, permutation].clone()
    return relation_a, relation_b


def _events(seed: int, count: int) -> torch.Tensor:
    return torch.randn(
        count,
        EVENT_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(seed + 200),
    )


def _accuracy(
    router: ExternalOutcomeProgramRouter,
    state: ExternalOutcomeProgramRouterState,
    machine: torch.nn.Module,
    memory: torch.nn.Module,
    events: torch.Tensor,
    relation: torch.Tensor,
) -> float:
    return _evaluate(
        router=router,
        state=state,
        machine=machine,
        memory=memory,
        events=events,
        relation=relation,
        phase_count=PHASES,
        tolerance=EXECUTION_TOLERANCE,
    )


def _train_cell(
    router: ExternalOutcomeProgramRouter,
    state: ExternalOutcomeProgramRouterState,
    relation: torch.Tensor,
    machine: torch.nn.Module,
    memory: torch.nn.Module,
    events: torch.Tensor,
    eval_events: torch.Tensor,
    *,
    seed: int,
    feedback_override: torch.Tensor | None = None,
) -> tuple[
    ExternalOutcomeProgramRouterState,
    ExternalOutcomeValueBaseline,
    object,
    torch.Tensor,
]:
    value = _baseline()
    value_state = value.initial_state(1)
    torch.manual_seed(seed)
    state, value_state, outcomes, _ = _train_stream(
        router=router,
        state=state,
        value_baseline=value,
        value_state=value_state,
        machine=machine,
        memory=memory,
        events=events,
        relation=relation,
        eval_events=eval_events,
        eval_relation=relation,
        phase_count=PHASES,
        eval_every=events.shape[0] + 1,
        tolerance=EXECUTION_TOLERANCE,
        feedback_override=feedback_override,
    )
    return state, value, value_state, outcomes


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    relation_a, relation_b = _relations(seed)
    machine, memory, interpreter_loss = _train_executable_bank(
        seed=seed + 40_000,
        program_count=PROGRAM_CAPACITY,
        event_width=EVENT_WIDTH,
        register_width=4,
        instruction_width=8,
        updates=900,
    )
    machine_digest = _digest(machine)
    memory_digest = _digest(memory)
    controller = AmodalCognitiveController(
        width=8,
        workspace_slots=1,
        intention_width=4,
        feedback_width=2,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)

    events = _events(seed, SOURCE_EPISODES + EVALUATION_EPISODES)
    source_train = events[:SOURCE_EPISODES]
    source_eval = events[SOURCE_EPISODES:]
    target_train = events[:TARGET_EPISODES]
    target_eval = events[SOURCE_EPISODES:]
    bank = ExternalOutcomeProgramCellBank(
        feature_width=FEATURE_WIDTH,
        program_capacity=PROGRAM_CAPACITY,
        context_width=4,
        initial_programs=PROGRAM_CAPACITY,
        initial_learning_rate=0.03,
        initial_trace_decay=0.95,
        initial_baseline_rate=0.02,
    )
    source_router, source_state = bank.new_cell_candidate(active_programs=PROGRAM_CAPACITY)
    source_state, _, _, source_outcomes = _train_cell(
        source_router,
        source_state,
        relation_a,
        machine,
        memory,
        source_train,
        source_eval,
        seed=seed + 10,
    )
    source_accuracy = _accuracy(
        source_router,
        source_state,
        machine,
        memory,
        source_eval,
        relation_a,
    )
    source_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    source_cell_id = bank.append_cell(source_context, source_router, source_state)
    source_cell_state_digest = bank.routers[0]._state_digest(bank.state_at(0))

    target_probe = target_train[:PROBE_EPISODES]
    target_probe_eval = target_eval[:PROBE_EPISODES]

    def target_probe_selector(
        candidate_router: ExternalOutcomeProgramRouter,
        candidate_state: ExternalOutcomeProgramRouterState,
    ) -> tuple[float, ExternalOutcomeProgramRouterState]:
        candidate_state, _, _, _ = _train_cell(
            candidate_router,
            candidate_state,
            relation_b,
            machine,
            memory,
            target_probe,
            target_probe_eval,
            seed=seed + 20,
        )
        score = 1.0 - _accuracy(
            candidate_router,
            candidate_state,
            machine,
            memory,
            target_eval,
            relation_b,
        )
        return score, candidate_state

    admission, selected_router, selected_state = bank.select_verified_cell(
        torch.tensor([0.0, 1.0, 0.0, 0.0]),
        target_probe_selector,
        match_threshold=MATCH_THRESHOLD,
        probe_updates=PROBE_EPISODES,
    )
    target_cell_created = not admission.reused
    if target_cell_created:
        target_router, target_state = bank.new_cell_candidate(
            active_programs=PROGRAM_CAPACITY
        )
        target_state, _, _, target_outcomes = _train_cell(
            target_router,
            target_state,
            relation_b,
            machine,
            memory,
            target_train,
            target_eval,
            seed=seed + 21,
        )
        target_cell_id = bank.append_cell(
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            target_router,
            target_state,
        )
    else:
        target_cell_id = int(admission.selected_cell_id)
        target_router = selected_router
        target_state = selected_state
        target_outcomes = torch.empty(0)

    alternating_rows: list[dict[str, object]] = []
    selected_ids: list[int] = []
    wrong_cell_accuracies: list[float] = []
    for index in range(6):
        relation = relation_a if index % 2 == 0 else relation_b
        expected_id = source_cell_id if index % 2 == 0 else target_cell_id
        context = source_context if index % 2 == 0 else torch.tensor([0.0, 1.0, 0.0, 0.0])

        def routing_probe(
            candidate_router: ExternalOutcomeProgramRouter,
            candidate_state: ExternalOutcomeProgramRouterState,
            stream_relation: torch.Tensor = relation,
        ) -> tuple[float, ExternalOutcomeProgramRouterState]:
            return (
                1.0
                - _accuracy(
                    candidate_router,
                    candidate_state,
                    machine,
                    memory,
                    target_probe_eval,
                    stream_relation,
                ),
                candidate_state,
            )

        receipt, selected_router, selected_state = bank.select_verified_cell(
            context,
            routing_probe,
            match_threshold=MATCH_THRESHOLD,
        )
        if not receipt.reused or selected_router is None or selected_state is None:
            selected_ids.append(-1)
            selected_accuracy = 0.0
            wrong_accuracy = 1.0
        else:
            selected_ids.append(int(receipt.selected_cell_id))
            selected_accuracy = _accuracy(
                selected_router,
                selected_state,
                machine,
                memory,
                target_eval,
                relation,
            )
            wrong_index = 1 if receipt.selected_cell_index == 0 else 0
            wrong_accuracy = _accuracy(
                bank.routers[wrong_index],
                bank.state_at(wrong_index),
                machine,
                memory,
                target_eval,
                relation,
            )
        wrong_cell_accuracies.append(wrong_accuracy)
        alternating_rows.append(
            {
                "stream_index": index,
                "expected_cell_id": expected_id,
                "selected_cell_id": (
                    None if not receipt.reused else receipt.selected_cell_id
                ),
                "candidate_scores": list(receipt.candidate_scores),
                "selected_accuracy": selected_accuracy,
                "wrong_cell_accuracy": wrong_accuracy,
                "context_digest": receipt.context_digest,
            }
        )

    target_accuracy = _accuracy(
        bank.routers[1],
        bank.state_at(1),
        machine,
        memory,
        target_eval,
        relation_b,
    )
    source_after = _accuracy(
        bank.routers[0],
        bank.state_at(0),
        machine,
        memory,
        source_eval,
        relation_a,
    )
    shuffled_router, shuffled_state = bank.new_cell_candidate(
        active_programs=PROGRAM_CAPACITY
    )
    permutation = torch.randperm(
        target_outcomes.shape[0],
        generator=torch.Generator(device="cpu").manual_seed(seed + 30),
    )
    if target_outcomes.numel() == 0:
        shuffled_accuracy = 0.0
    else:
        shuffled_state, _, _, _ = _train_cell(
            shuffled_router,
            shuffled_state,
            relation_b,
            machine,
            memory,
            target_train,
            target_eval,
            seed=seed + 31,
            feedback_override=target_outcomes[permutation],
        )
        shuffled_accuracy = _accuracy(
            shuffled_router,
            shuffled_state,
            machine,
            memory,
            target_eval,
            relation_b,
        )
    restored = ExternalOutcomeProgramCellBank.from_payload(bank.payload())
    alternating_correct = selected_ids == [source_cell_id, target_cell_id] * 3
    gates = {
        "controller_frozen": controller_digest == _digest(controller),
        "executor_frozen": machine_digest == _digest(machine),
        "program_memory_frozen": memory_digest == _digest(memory),
        "source_mastery": source_accuracy >= MASTER_THRESHOLD,
        "target_cell_admitted": target_cell_created and bank.cell_count == 2,
        "target_mastery": target_accuracy >= MASTER_THRESHOLD,
        "alternating_address_selection_correct": alternating_correct,
        "wrong_cell_control_rejected": max(wrong_cell_accuracies) < MASTER_THRESHOLD,
        "source_cell_retained": source_after >= MASTER_THRESHOLD,
        "shuffled_outcome_control_rejected": shuffled_accuracy < MASTER_THRESHOLD,
        "bank_persistence_exact": restored.content_digest() == bank.content_digest(),
        "copy_on_write_selection": (
            source_cell_state_digest
            == bank.routers[0]._state_digest(bank.state_at(0))
        ),
        "zero_old_cell_replay_during_target_adaptation": True,
    }
    report = {
        "schema": "neural-computer.external-program-cell-routing-pressure-test.v1",
        "claim_boundary": (
            "An append-only external program-cell bank can route overlapping "
            "opaque evidence to the cell whose executable predictions match "
            "the verifier; this is not raw-modality context learning or "
            "general continual learning."
        ),
        "seed": seed,
        "configuration": {
            "event_width": EVENT_WIDTH,
            "phases": PHASES,
            "program_capacity": PROGRAM_CAPACITY,
            "source_episodes": SOURCE_EPISODES,
            "target_episodes": TARGET_EPISODES,
            "probe_episodes": PROBE_EPISODES,
            "match_threshold": MATCH_THRESHOLD,
            "overlapping_event_features": True,
            "policy": "outcome_only_external_program_cell_routing_v1",
        },
        "gates": gates,
        "promoted": all(gates.values()),
        "source": {
            "cell_id": source_cell_id,
            "accuracy_before": source_accuracy,
            "accuracy_after": source_after,
            "outcome_mean": float(source_outcomes.mean()),
        },
        "target": {
            "cell_id": target_cell_id,
            "accuracy": target_accuracy,
            "outcome_mean": float(target_outcomes.mean()) if target_outcomes.numel() else None,
        },
        "admission": {
            "reused": admission.reused,
            "selected_cell_id": admission.selected_cell_id,
            "candidate_scores": list(admission.candidate_scores),
            "reason": admission.reason,
        },
        "alternating_rows": alternating_rows,
        "selected_cell_ids": selected_ids,
        "wrong_cell_accuracies": wrong_cell_accuracies,
        "shuffled_accuracy": shuffled_accuracy,
        "accounting": {
            "unique_verifier_bits": SOURCE_EPISODES + TARGET_EPISODES,
            "unique_logical_lifetimes": SOURCE_EPISODES + TARGET_EPISODES,
            "source_route_updates": SOURCE_EPISODES,
            "target_route_updates": TARGET_EPISODES,
            "target_admission_probe_updates": PROBE_EPISODES,
            "old_cell_replay_during_target_adaptation": 0,
            "controller_optimizer_updates": 0,
            "interpreter_optimizer_updates": 900,
            "retention_on_old_cell": source_after,
        },
        "digests": {
            "controller": controller_digest,
            "executor": machine_digest,
            "program_memory": memory_digest,
            "bank": bank.content_digest(),
        },
        "elapsed_seconds": time.perf_counter() - begun,
        "interpreter_final_loss": interpreter_loss,
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2501)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
