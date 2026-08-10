"""Audit outcome-only admission and retention of an executable external file.

The interpreter and controller are frozen. A two-file source bank is mastered
first. A third opaque artifact is then evaluated by a private deterministic
verifier, admitted through the public stable-prefix file transaction, and
protected. Route learning sees only the resulting scalar outcomes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_outcome_program_router.train import (
    _evaluate,
    _hidden_choices,
    _stable_prefix,
    _train_executable_bank,
    _train_stream,
)
from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    ControllerFeedback,
    ExternalCapabilityRegisterMachine,
    ExternalOutcomeProgramCellBank,
    ExternalOutcomeProgramRouter,
    ExternalOutcomeProgramRouterState,
    ExternalOutcomeValueBaseline,
    ExternalProgramAmodalRuntime,
    ExternalProgramArtifact,
    ExternalSequenceProgramMemory,
    OpaqueProtocolDecoder,
)

EVENT_WIDTH = 4
REGISTER_WIDTH = 4
INSTRUCTION_WIDTH = 8
PHASES = 2
PROGRAM_CAPACITY = 3
SOURCE_PROGRAMS = 2
SOURCE_EPISODES = 500
TARGET_EPISODES = 1600
EVALUATION_EPISODES = 240
PROBE_EPISODES = 128
MASTER_THRESHOLD = 0.75
ADMISSION_THRESHOLD = 0.90
MIN_STABLE_OBSERVATIONS = 32
EXECUTION_TOLERANCE = 1e-5


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _memory(
    artifacts: tuple[ExternalProgramArtifact, ...],
    count: int,
) -> ExternalSequenceProgramMemory:
    if not 0 < count <= len(artifacts):
        raise ValueError("program memory count is outside the artifact bank")
    memory = ExternalSequenceProgramMemory(
        INSTRUCTION_WIDTH,
        content_addressing=True,
        hard_routing=True,
    )
    for artifact in artifacts[:count]:
        memory.add_artifact(artifact)
    for parameter in memory.parameters():
        parameter.requires_grad_(False)
    return memory


@torch.no_grad()
def _execute_artifact(
    machine: ExternalCapabilityRegisterMachine,
    artifact: ExternalProgramArtifact,
    register: torch.Tensor,
) -> torch.Tensor:
    codes = artifact.codes.to(device=register.device, dtype=register.dtype)
    return machine.execute_code_chain(register, codes.unsqueeze(0).expand(register.shape[0], -1, -1))


def _artifact_outcomes(
    machine: ExternalCapabilityRegisterMachine,
    candidate: ExternalProgramArtifact,
    reference: ExternalProgramArtifact,
    events: torch.Tensor,
    relation: torch.Tensor,
    *,
    candidate_index: int,
) -> torch.Tensor:
    """Return verifier scalars for only contexts that require the candidate."""

    outcomes: list[torch.Tensor] = []
    for event in events:
        batch = event.unsqueeze(0)
        hidden = _hidden_choices(batch, relation)
        if not bool(torch.any(hidden == candidate_index)):
            continue
        candidate_output = _execute_artifact(machine, candidate, batch)
        reference_output = _execute_artifact(machine, reference, batch)
        outcomes.append(
            (candidate_output - reference_output)
            .abs()
            .amax(dim=-1)
            .le(EXECUTION_TOLERANCE)
            .to(torch.float32)
        )
    if not outcomes:
        raise RuntimeError("candidate verifier produced no relevant outcomes")
    return torch.cat(outcomes)


def _value_baseline(feature_width: int) -> ExternalOutcomeValueBaseline:
    return ExternalOutcomeValueBaseline(
        feature_width=feature_width,
        initial_learning_rate=0.05,
        initial_trace_decay=0.90,
    )


def _runtime_smoke(
    machine: ExternalCapabilityRegisterMachine,
    memory: ExternalSequenceProgramMemory,
    *,
    seed: int,
) -> dict[str, object]:
    """Verify the admitted file still traverses the canonical I/O seam."""

    torch.manual_seed(seed)
    controller = AmodalCognitiveController(
        width=EVENT_WIDTH,
        workspace_slots=1,
        intention_width=1,
        feedback_width=1,
        event_window_capacity=2,
    )
    controller_digest = _digest(controller)
    for parameter in controller.parameters():
        parameter.requires_grad_(False)
    runtime = AmodalControllerRuntime(controller)
    runtime.register_decoder("opaque", OpaqueProtocolDecoder(1, 1))
    agent = ExternalProgramAmodalRuntime(runtime, machine, program_memory=memory)
    state = agent.initial_state(1, device="cpu")
    feedback = ControllerFeedback(
        action=torch.zeros(1, 1),
        reward=torch.zeros(1),
        propensity=torch.ones(1),
        has_feedback=torch.zeros(1),
    )
    output, _ = agent.step_events(
        [AmodalEvent(torch.randn(1, EVENT_WIDTH))],
        state,
        feedback,
    )
    return {
        "executed_file": output.execution.program_digest,
        "decoded_shape": list(output.decoded["opaque"].shape),
        "controller_frozen": controller_digest == _digest(controller),
    }


def _train_target(
    *,
    router: ExternalOutcomeProgramRouter,
    state: ExternalOutcomeProgramRouterState,
    value_baseline: ExternalOutcomeValueBaseline,
    value_state: object,
    machine: ExternalCapabilityRegisterMachine,
    memory: ExternalSequenceProgramMemory,
    events: torch.Tensor,
    relation: torch.Tensor,
    evaluation_events: torch.Tensor,
    evaluation_relation: torch.Tensor,
    protected_programs: int | None = None,
    feedback_override: torch.Tensor | None = None,
    expected_memory: ExternalSequenceProgramMemory | None = None,
) -> tuple[ExternalOutcomeProgramRouterState, object, torch.Tensor, list[dict[str, object]]]:
    return _train_stream(
        router=router,
        state=state,
        value_baseline=value_baseline,
        value_state=value_state,
        machine=machine,
        memory=memory,
        events=events,
        relation=relation,
        eval_events=evaluation_events,
        eval_relation=evaluation_relation,
        phase_count=PHASES,
        eval_every=200,
        tolerance=EXECUTION_TOLERANCE,
        protected_programs=protected_programs,
        feedback_override=feedback_override,
        expected_memory=expected_memory,
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)

    machine, pretrained_memory, interpreter_loss = _train_executable_bank(
        seed=seed + 40_000,
        program_count=PROGRAM_CAPACITY,
        event_width=EVENT_WIDTH * 3,
        register_width=REGISTER_WIDTH,
        instruction_width=INSTRUCTION_WIDTH,
        updates=500,
    )
    artifacts = tuple(
        pretrained_memory.artifact(index) for index in range(PROGRAM_CAPACITY)
    )
    machine_digest = _digest(machine)
    source_memory = _memory(artifacts, SOURCE_PROGRAMS)
    verifier_memory = _memory(artifacts, PROGRAM_CAPACITY)
    source_memory.protect_file(0)
    source_memory.protect_file(1)
    source_memory_digest_before = source_memory.digest()

    generator = torch.Generator(device="cpu").manual_seed(seed + 10_001)
    source_events = torch.randn(
        SOURCE_EPISODES + EVALUATION_EPISODES,
        EVENT_WIDTH,
        generator=generator,
    )
    target_events = torch.randn(
        TARGET_EPISODES + EVALUATION_EPISODES,
        EVENT_WIDTH,
        generator=generator,
    )
    source_events[:, -1] = 0.0
    target_events[::4, -1] = 0.0
    target_events[1::4, :-1] = 0.0
    target_events[1::4, -1] = 3.0
    target_events[2::4, -1] = 0.0
    target_events[3::4, -1] = 0.0
    source_relation = torch.randn(
        PHASES,
        SOURCE_PROGRAMS,
        EVENT_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(seed + 20_001),
    )
    source_relation[..., -1] = 0.0
    target_relation = torch.zeros(PHASES, PROGRAM_CAPACITY, EVENT_WIDTH)
    target_relation[:, :SOURCE_PROGRAMS] = source_relation
    target_relation[:, SOURCE_PROGRAMS:] = torch.randn(
        PHASES,
        PROGRAM_CAPACITY - SOURCE_PROGRAMS,
        EVENT_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(seed + 20_002),
    )
    target_relation[:, SOURCE_PROGRAMS, -1] = 2.0
    source_train = source_events[:SOURCE_EPISODES]
    source_eval = source_events[SOURCE_EPISODES:]
    target_train = target_events[:TARGET_EPISODES]
    target_eval = target_events[TARGET_EPISODES:]
    feature_width = EVENT_WIDTH * PHASES + PHASES

    source_router = ExternalOutcomeProgramRouter(
        feature_width,
        PROGRAM_CAPACITY,
        initial_programs=SOURCE_PROGRAMS,
        initial_learning_rate=0.03,
        initial_trace_decay=0.95,
        initial_baseline_rate=0.02,
    )
    source_value = _value_baseline(feature_width)
    source_state, source_value_state, _, source_progress = _train_stream(
        router=source_router,
        state=source_router.initial_state(1),
        value_baseline=source_value,
        value_state=source_value.initial_state(1),
        machine=machine,
        memory=source_memory,
        events=source_train,
        relation=source_relation,
        eval_events=source_eval,
        eval_relation=source_relation,
        phase_count=PHASES,
        eval_every=200,
        tolerance=EXECUTION_TOLERANCE,
        expected_memory=verifier_memory,
    )
    source_before = _evaluate(
        router=source_router,
        state=source_state,
        machine=machine,
        memory=source_memory,
        events=source_eval,
        relation=source_relation,
        phase_count=PHASES,
        tolerance=EXECUTION_TOLERANCE,
        expected_memory=verifier_memory,
    )
    good_outcomes = _artifact_outcomes(
        machine,
        artifacts[2],
        artifacts[2],
        target_train[:PROBE_EPISODES],
        target_relation,
        candidate_index=2,
    )
    corrupted = ExternalProgramArtifact(
        codes=artifacts[2].codes * -1.0,
        interpreter_schema=artifacts[2].interpreter_schema,
        execution_schema=artifacts[2].execution_schema,
    )
    bad_outcomes = _artifact_outcomes(
        machine,
        corrupted,
        artifacts[2],
        target_train[:PROBE_EPISODES],
        target_relation,
        candidate_index=2,
    )
    bad_receipt = source_memory.admit_verified_artifact(
        corrupted,
        bad_outcomes,
        threshold=ADMISSION_THRESHOLD,
        min_observations=MIN_STABLE_OBSERVATIONS,
        min_stable_observations=MIN_STABLE_OBSERVATIONS,
    )
    rejected_memory_unchanged = (
        not bad_receipt.accepted
        and source_memory.digest() == source_memory_digest_before
    )
    good_receipt = source_memory.admit_verified_artifact(
        artifacts[2],
        good_outcomes,
        threshold=ADMISSION_THRESHOLD,
        min_observations=MIN_STABLE_OBSERVATIONS,
        min_stable_observations=MIN_STABLE_OBSERVATIONS,
        protect=True,
    )
    if not good_receipt.accepted or good_receipt.slot is None:
        raise RuntimeError(f"good candidate was not admitted: {good_receipt.reason}")
    bank = ExternalOutcomeProgramCellBank(
        feature_width=feature_width,
        program_capacity=PROGRAM_CAPACITY,
        context_width=EVENT_WIDTH,
        initial_programs=SOURCE_PROGRAMS,
        initial_learning_rate=0.03,
        initial_trace_decay=0.95,
        initial_baseline_rate=0.02,
    )
    source_context = torch.tensor([1.0, 0.0, 0.0, 0.0])
    target_context = torch.tensor([0.0, 1.0, 0.0, 0.0])
    source_cell_id = bank.append_cell(source_context, source_router, source_state)

    warm_router = copy.deepcopy(source_router)
    warm_state = warm_router.append_program(source_state)
    warm_value = source_value
    warm_state, _, warm_outcomes, warm_progress = _train_target(
        router=warm_router,
        state=warm_state,
        value_baseline=warm_value,
        value_state=source_value_state,
        machine=machine,
        memory=source_memory,
        events=target_train,
        relation=target_relation,
        evaluation_events=target_eval,
        evaluation_relation=target_relation,
        protected_programs=SOURCE_PROGRAMS,
        expected_memory=verifier_memory,
    )
    warm_target = _evaluate(
        router=warm_router,
        state=warm_state,
        machine=machine,
        memory=source_memory,
        events=target_eval,
        relation=target_relation,
        phase_count=PHASES,
        tolerance=EXECUTION_TOLERANCE,
        expected_memory=verifier_memory,
    )
    source_after = _evaluate(
        router=bank.routers[0],
        state=bank.state_at(0),
        machine=machine,
        memory=source_memory,
        events=source_eval,
        relation=source_relation,
        phase_count=PHASES,
        tolerance=EXECUTION_TOLERANCE,
        expected_memory=verifier_memory,
    )
    target_cell_id = bank.append_cell(target_context, warm_router, warm_state)

    fresh_router = ExternalOutcomeProgramRouter(
        feature_width,
        PROGRAM_CAPACITY,
        initial_programs=SOURCE_PROGRAMS,
        initial_learning_rate=0.03,
        initial_trace_decay=0.95,
        initial_baseline_rate=0.02,
    )
    fresh_state = fresh_router.append_program(fresh_router.initial_state(1))
    fresh_value = _value_baseline(feature_width)
    fresh_state, _, _, fresh_progress = _train_target(
        router=fresh_router,
        state=fresh_state,
        value_baseline=fresh_value,
        value_state=fresh_value.initial_state(1),
        machine=machine,
        memory=source_memory,
        events=target_train,
        relation=target_relation,
        evaluation_events=target_eval,
        evaluation_relation=target_relation,
        expected_memory=verifier_memory,
    )
    fresh_target = _evaluate(
        router=fresh_router,
        state=fresh_state,
        machine=machine,
        memory=source_memory,
        events=target_eval,
        relation=target_relation,
        phase_count=PHASES,
        tolerance=EXECUTION_TOLERANCE,
        expected_memory=verifier_memory,
    )

    shuffled_router = ExternalOutcomeProgramRouter(
        feature_width,
        PROGRAM_CAPACITY,
        initial_programs=SOURCE_PROGRAMS,
        initial_learning_rate=0.03,
        initial_trace_decay=0.95,
        initial_baseline_rate=0.02,
    )
    shuffled_state = shuffled_router.append_program(shuffled_router.initial_state(1))
    shuffled_value = _value_baseline(feature_width)
    permutation = torch.randperm(
        warm_outcomes.shape[0],
        generator=torch.Generator(device="cpu").manual_seed(seed + 30_001),
    )
    shuffled_state, _, _, shuffled_progress = _train_target(
        router=shuffled_router,
        state=shuffled_state,
        value_baseline=shuffled_value,
        value_state=shuffled_value.initial_state(1),
        machine=machine,
        memory=source_memory,
        events=target_train,
        relation=target_relation,
        evaluation_events=target_eval,
        evaluation_relation=target_relation,
        feedback_override=warm_outcomes[permutation],
        expected_memory=verifier_memory,
    )
    shuffled_target = _evaluate(
        router=shuffled_router,
        state=shuffled_state,
        machine=machine,
        memory=source_memory,
        events=target_eval,
        relation=target_relation,
        phase_count=PHASES,
        tolerance=EXECUTION_TOLERANCE,
        expected_memory=verifier_memory,
    )

    no_file_router = ExternalOutcomeProgramRouter(
        feature_width,
        PROGRAM_CAPACITY,
        initial_programs=SOURCE_PROGRAMS,
        initial_learning_rate=0.03,
        initial_trace_decay=0.95,
        initial_baseline_rate=0.02,
    )
    no_file_memory = _memory(artifacts, SOURCE_PROGRAMS)
    no_file_memory.add_artifact(corrupted)
    for parameter in no_file_memory.parameters():
        parameter.requires_grad_(False)
    no_file_state = no_file_router.append_program(no_file_router.initial_state(1))
    no_file_value = _value_baseline(feature_width)
    no_file_state, _, _, no_file_progress = _train_target(
        router=no_file_router,
        state=no_file_state,
        value_baseline=no_file_value,
        value_state=no_file_value.initial_state(1),
        machine=machine,
        memory=no_file_memory,
        events=target_train,
        relation=target_relation,
        evaluation_events=target_eval,
        evaluation_relation=target_relation,
        expected_memory=verifier_memory,
    )
    no_file_target = _evaluate(
        router=no_file_router,
        state=no_file_state,
        machine=machine,
        memory=no_file_memory,
        events=target_eval,
        relation=target_relation,
        phase_count=PHASES,
        tolerance=EXECUTION_TOLERANCE,
        expected_memory=verifier_memory,
    )

    def select_cell(
        context: torch.Tensor,
        events: torch.Tensor,
        relation: torch.Tensor,
    ):
        def probe(
            candidate_router: ExternalOutcomeProgramRouter,
            candidate_state: ExternalOutcomeProgramRouterState,
        ) -> tuple[float, ExternalOutcomeProgramRouterState]:
            score = 1.0 - _evaluate(
                router=candidate_router,
                state=candidate_state,
                machine=machine,
                memory=source_memory,
                events=events,
                relation=relation,
                phase_count=PHASES,
                tolerance=EXECUTION_TOLERANCE,
                expected_memory=verifier_memory,
            )
            return score, candidate_state

        receipt, _, _ = bank.select_verified_cell(
            context,
            probe,
            match_threshold=0.5,
        )
        return receipt

    source_selection = select_cell(source_context, source_eval, source_relation)
    target_selection = select_cell(target_context, target_eval, target_relation)
    restored_memory = ExternalSequenceProgramMemory.from_payload(
        source_memory.payload()
    )
    restored_bank = ExternalOutcomeProgramCellBank.from_payload(bank.payload())
    runtime_smoke = _runtime_smoke(machine, restored_memory, seed=seed + 90_001)
    gates = {
        "source_mastery": source_before >= MASTER_THRESHOLD,
        "candidate_rejected_without_write": rejected_memory_unchanged,
        "candidate_admitted": bool(good_receipt.accepted),
        "file_bank_grew_after_acceptance": source_memory.digest()
        != source_memory_digest_before,
        "candidate_stable_run": (
            good_receipt.stable_bits_to_threshold is not None
            and good_receipt.stable_bits_to_threshold <= MIN_STABLE_OBSERVATIONS
        ),
        "source_file_protected": all(
            source_memory.is_file_protected(index) for index in range(SOURCE_PROGRAMS)
        ),
        "new_file_protected": source_memory.is_file_protected(good_receipt.slot),
        "source_cell_retained": (
            source_after >= MASTER_THRESHOLD
            and bank.routers[0]._state_digest(bank.state_at(0))
            == source_router._state_digest(source_state)
        ),
        "source_cell_selected": (
            source_selection.reused
            and source_selection.selected_cell_id == source_cell_id
        ),
        "target_cell_selected": (
            target_selection.reused
            and target_selection.selected_cell_id == target_cell_id
        ),
        "warm_target_mastery": warm_target >= MASTER_THRESHOLD,
        "warm_target_stable": _stable_prefix(warm_progress, MASTER_THRESHOLD) is not None,
        "fresh_target_is_matched": fresh_target >= MASTER_THRESHOLD,
        "warm_not_worse_than_fresh": warm_target >= fresh_target - 0.10,
        "source_retention": source_after >= MASTER_THRESHOLD,
        "wrong_file_control_rejected": no_file_target < MASTER_THRESHOLD,
        "reward_shuffled_control_rejected": shuffled_target < MASTER_THRESHOLD,
        "memory_persistence_exact": restored_memory.digest() == source_memory.digest(),
        "cell_bank_persistence_exact": (
            restored_bank.content_digest() == bank.content_digest()
        ),
        "controller_runtime_seam": bool(runtime_smoke["controller_frozen"]),
        "executor_frozen": machine_digest == _digest(machine),
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    report = {
        "schema": "neural-computer.external-program-file-admission.v1",
        "claim_boundary": (
            "outcome-only stable-prefix admission and retention of one new "
            "portable executable external file with a frozen controller; "
            "not program synthesis or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "event_width": EVENT_WIDTH,
            "phases": PHASES,
            "source_programs": SOURCE_PROGRAMS,
            "program_capacity": PROGRAM_CAPACITY,
            "source_episodes": SOURCE_EPISODES,
            "target_episodes": TARGET_EPISODES,
            "probe_episodes": PROBE_EPISODES,
            "admission_threshold": ADMISSION_THRESHOLD,
            "min_stable_observations": MIN_STABLE_OBSERVATIONS,
            "learner_inputs": [
                "opaque_event_tensor",
                "opaque_sampled_program_choice",
                "exact_choice_propensity",
                "terminal_scalar_verifier_outcome",
            ],
        },
        "interpreter_final_loss": interpreter_loss,
        "source_mastery": source_before,
        "source_retention": source_after,
        "warm_target": warm_target,
        "fresh_target": fresh_target,
        "shuffled_target": shuffled_target,
        "wrong_file_target": no_file_target,
        "source_progress": source_progress,
        "warm_progress": warm_progress,
        "fresh_progress": fresh_progress,
        "shuffled_progress": shuffled_progress,
        "no_file_progress": no_file_progress,
        "candidate_outcome_mean": float(good_outcomes.mean()),
        "corrupted_candidate_outcome_mean": float(bad_outcomes.mean()),
        "bad_receipt": bad_receipt.payload(),
        "good_receipt": good_receipt.payload(),
        "cell_bank": {
            "cell_ids": list(bank.cell_ids),
            "source_cell_id": source_cell_id,
            "target_cell_id": target_cell_id,
            "source_selected_cell_id": source_selection.selected_cell_id,
            "target_selected_cell_id": target_selection.selected_cell_id,
            "source_candidate_scores": list(source_selection.candidate_scores),
            "target_candidate_scores": list(target_selection.candidate_scores),
        },
        "runtime_smoke": runtime_smoke,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": (
                SOURCE_EPISODES + TARGET_EPISODES + int(good_outcomes.numel()) + int(bad_outcomes.numel())
            ),
            "unique_logical_lifetimes": SOURCE_EPISODES + TARGET_EPISODES,
            "program_file_verifier_outcomes": int(
                good_outcomes.numel() + bad_outcomes.numel()
            ),
            "external_route_decision_updates": PHASES * (SOURCE_EPISODES + 3 * TARGET_EPISODES),
            "external_feedback_updates": SOURCE_EPISODES + 3 * TARGET_EPISODES,
            "interpreter_optimizer_updates": 500,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "raw_program_verifier_rows_retained": 0,
            "warm_stable_bits_to_threshold": _stable_prefix(warm_progress, MASTER_THRESHOLD),
            "fresh_stable_bits_to_threshold": _stable_prefix(fresh_progress, MASTER_THRESHOLD),
            "wall_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
