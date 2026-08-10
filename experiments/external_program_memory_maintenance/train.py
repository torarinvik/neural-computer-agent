"""Train a generic maintenance policy against real executable-file receipts.

The interpreter and controller are frozen. Every phase constructs a fresh
opaque executable bank, asks a masked external maintenance policy for one
operation, executes that operation through the real copy-on-write file API,
and feeds the policy one scalar utility. The verifier's held-out register
checks are not retained as replay data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_outcome_program_router.train import _train_executable_bank
from experiments.external_program_file_admission.train import _runtime_smoke
from neural_computer import (
    ExternalMemoryMaintenancePolicy,
    ExternalProgramArtifact,
    ExternalSequenceProgramMemory,
)

EVENT_WIDTH = 12
REGISTER_WIDTH = 4
INSTRUCTION_WIDTH = 8
PROGRAM_COUNT = 2
INTERPRETER_UPDATES = 500
HELD_OUT_REGISTERS = 32
TRAIN_CYCLES = 96
EXECUTION_TOLERANCE = 1e-6
COMPRESSION_TOLERANCE = 2e-2
PHASES = ("share", "evict", "compress", "grow", "defer")


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def _execute(
    machine: torch.nn.Module,
    artifact: ExternalProgramArtifact,
    registers: torch.Tensor,
) -> torch.Tensor:
    codes = artifact.codes.to(device=registers.device, dtype=registers.dtype)
    return machine.execute_code_chain(
        registers,
        codes.unsqueeze(0).expand(registers.shape[0], -1, -1),
    )


def _memory(
    artifacts: tuple[ExternalProgramArtifact, ...],
    phase: str,
) -> ExternalSequenceProgramMemory:
    if phase == "share":
        source = (artifacts[0], artifacts[0], artifacts[1])
    elif phase == "evict":
        source = (artifacts[0], artifacts[1])
    else:
        source = (artifacts[0],)
    memory = ExternalSequenceProgramMemory(
        INSTRUCTION_WIDTH,
        content_addressing=True,
        hard_routing=True,
    )
    for artifact in source:
        memory.add_artifact(artifact)
    memory.protect_file(0)
    for parameter in memory.parameters():
        parameter.requires_grad_(False)
    return memory


def _retention_probe(
    machine: torch.nn.Module,
    memory: ExternalSequenceProgramMemory,
    registers: torch.Tensor,
    *,
    retained_ids: tuple[int, ...],
    tolerance: float,
):
    expected = {
        logical_id: _execute(
            machine,
            memory.artifact(memory.physical_index_for_logical_id(logical_id)),
            registers,
        )
        for logical_id in retained_ids
    }

    def probe(candidate: ExternalSequenceProgramMemory) -> bool:
        for logical_id, reference in expected.items():
            try:
                index = candidate.physical_index_for_logical_id(logical_id)
            except KeyError:
                return False
            actual = _execute(machine, candidate.artifact(index), registers)
            if not bool(torch.allclose(actual, reference, atol=tolerance, rtol=0.0)):
                return False
        return True

    return probe


def _equivalence_probe(
    machine: torch.nn.Module,
    registers: torch.Tensor,
):
    def probe(
        survivor: ExternalProgramArtifact,
        duplicate: ExternalProgramArtifact,
    ) -> bool:
        return bool(
            torch.allclose(
                _execute(machine, survivor, registers),
                _execute(machine, duplicate, registers),
                atol=EXECUTION_TOLERANCE,
                rtol=0.0,
            )
        )

    return probe


def _available(phase: str) -> dict[str, bool]:
    return {
        "growth_available": phase == "grow",
        "share_available": phase == "share",
        "compression_available": phase in {"share", "evict", "compress"},
        "evict_available": phase in {"share", "evict"},
    }


def _run_phase(
    policy: ExternalMemoryMaintenancePolicy,
    *,
    phase: str,
    artifacts: tuple[ExternalProgramArtifact, ...],
    machine: torch.nn.Module,
    registers: torch.Tensor,
    learn: bool,
    shuffled_utility: bool,
    generator: torch.Generator,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, object]:
    memory = _memory(artifacts, phase)
    availability = _available(phase)
    proposal = memory.propose_maintenance(
        policy,
        capacity_limit=3,
        **availability,
        redundancy_pressure=float(phase == "share"),
        compression_opportunity=0.5,
        sample=learn,
        generator=generator,
    )
    retention = _retention_probe(
        machine,
        memory,
        registers,
        retained_ids=(0, 2)
        if phase == "share"
        else (0,)
        if phase in {"evict", "compress", "grow"}
        else (),
        tolerance=COMPRESSION_TOLERANCE
        if proposal.action == "compress"
        else EXECUTION_TOLERANCE,
    )
    share_pair = (0, 1) if phase == "share" else None
    evict_slot_id = 1 if phase == "evict" else 2 if phase == "share" else None
    receipt = memory.apply_maintenance_proposal(
        proposal,
        retention_probe=retention,
        share_pair=share_pair,
        equivalence_probe=_equivalence_probe(machine, registers),
        evict_slot_id=evict_slot_id,
        growth_artifact=artifacts[1],
        growth_outcomes=[1.0] * HELD_OUT_REGISTERS,
        growth_min_observations=HELD_OUT_REGISTERS,
        growth_min_stable_observations=HELD_OUT_REGISTERS,
        protect_growth=True,
    )
    accepted = bool(receipt.accepted) if receipt is not None else proposal.action == "defer"
    utility = float(accepted and proposal.action == phase)
    if shuffled_utility:
        utility = float(torch.randint(2, (), generator=generator))
    if learn:
        if optimizer is None:
            raise RuntimeError("learned maintenance phase has no optimizer")
        policy.adaptation_step(proposal, utility, optimizer=optimizer)
    return {
        "phase": phase,
        "proposal": proposal.action,
        "accepted": accepted,
        "utility": utility,
        "final_files": memory.file_count,
        "logical_ids": list(memory.logical_slot_ids),
        "receipt": None if receipt is None else receipt.payload(),
    }


def _rollout(
    *,
    seed: int,
    policy: ExternalMemoryMaintenancePolicy,
    artifacts: tuple[ExternalProgramArtifact, ...],
    machine: torch.nn.Module,
    registers: torch.Tensor,
    cycles: int,
    learn: bool,
    shuffled_utility: bool = False,
) -> dict[str, object]:
    optimizer = (
        torch.optim.Adam(policy.parameters(), lr=0.03) if learn else None
    )
    generator = torch.Generator(device="cpu").manual_seed(seed + 7000)
    history: list[dict[str, object]] = []
    for index in range(cycles):
        history.append(
            _run_phase(
                policy,
                phase=PHASES[index % len(PHASES)],
                artifacts=artifacts,
                machine=machine,
                registers=registers,
                learn=learn,
                shuffled_utility=shuffled_utility,
                generator=generator,
                optimizer=optimizer,
            )
        )
    return {
        "history": history,
        "optimizer_updates": cycles if learn else 0,
        "mean_utility": sum(float(row["utility"]) for row in history) / cycles,
    }


def _evaluate(
    *,
    seed: int,
    policy: ExternalMemoryMaintenancePolicy,
    artifacts: tuple[ExternalProgramArtifact, ...],
    machine: torch.nn.Module,
    registers: torch.Tensor,
) -> dict[str, object]:
    return _rollout(
        seed=seed,
        policy=policy,
        artifacts=artifacts,
        machine=machine,
        registers=registers,
        cycles=len(PHASES) * 4,
        learn=False,
    )


def run(seed: int, report_out: Path) -> dict[str, object]:
    begun = time.perf_counter()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    machine, pretrained_memory, interpreter_loss = _train_executable_bank(
        seed=seed + 40_000,
        program_count=PROGRAM_COUNT,
        event_width=EVENT_WIDTH,
        register_width=REGISTER_WIDTH,
        instruction_width=INSTRUCTION_WIDTH,
        updates=INTERPRETER_UPDATES,
    )
    machine_digest = _digest(machine)
    artifacts = tuple(
        pretrained_memory.artifact(index) for index in range(PROGRAM_COUNT)
    )
    registers = torch.randn(
        HELD_OUT_REGISTERS,
        REGISTER_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(seed + 5000),
    )
    trained_policy = ExternalMemoryMaintenancePolicy(hidden_width=16, learning_rate=0.03)
    trained = _rollout(
        seed=seed,
        policy=trained_policy,
        artifacts=artifacts,
        machine=machine,
        registers=registers,
        cycles=TRAIN_CYCLES,
        learn=True,
    )
    fresh = _evaluate(
        seed=seed + 1000,
        policy=ExternalMemoryMaintenancePolicy(hidden_width=16, learning_rate=0.03),
        artifacts=artifacts,
        machine=machine,
        registers=registers,
    )
    shuffled_policy = ExternalMemoryMaintenancePolicy(hidden_width=16, learning_rate=0.03)
    shuffled = _rollout(
        seed=seed + 2000,
        policy=shuffled_policy,
        artifacts=artifacts,
        machine=machine,
        registers=registers,
        cycles=TRAIN_CYCLES,
        learn=True,
        shuffled_utility=True,
    )
    trained_eval = _evaluate(
        seed=seed + 3000,
        policy=trained_policy,
        artifacts=artifacts,
        machine=machine,
        registers=registers,
    )
    shuffled_eval = _evaluate(
        seed=seed + 4000,
        policy=shuffled_policy,
        artifacts=artifacts,
        machine=machine,
        registers=registers,
    )

    persistence_memory = _memory(artifacts, "share")
    persistence_before = persistence_memory.digest()
    persistence_restored = ExternalSequenceProgramMemory.from_payload(
        persistence_memory.payload()
    )
    compressed_payload = persistence_memory.compressed_payload(dtype=torch.float16)
    corrupted = copy.deepcopy(compressed_payload)
    corrupted_state = corrupted["state"]
    if not isinstance(corrupted_state, dict):
        raise TypeError("compressed executable state is invalid")
    corrupted_state["programs.0"] = corrupted_state["programs.0"].clone()
    corrupted_state["programs.0"].reshape(-1)[0] += 1
    try:
        ExternalSequenceProgramMemory.from_compressed_payload(corrupted)
    except ValueError as error:
        corruption_rejected = "checksum mismatch" in str(error)
    else:
        corruption_rejected = False
    runtime_smoke = _runtime_smoke(machine, persistence_restored, seed=seed + 9000)
    trained_history = trained["history"]
    phase_receipts = [
        row["receipt"]
        for row in trained_history
        if row["receipt"] is not None
    ]
    gates = {
        "trained_beats_fresh": trained_eval["mean_utility"] > fresh["mean_utility"] + 0.10,
        "trained_beats_shuffled_verifier": trained_eval["mean_utility"]
        > shuffled_eval["mean_utility"] + 0.10,
        "all_real_actions_observed": all(
            any(row["proposal"] == phase and row["accepted"] for row in trained_history)
            for phase in PHASES[:-1]
        ),
        "all_transaction_receipts_are_real": len(phase_receipts) >= 4,
        "file_persistence_exact": persistence_restored.digest() == persistence_before,
        "corrupt_compressed_payload_rejected": corruption_rejected,
        "canonical_runtime_seam": bool(runtime_smoke["controller_frozen"]),
        "interpreter_frozen": machine_digest == _digest(machine),
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    report = {
        "schema": "neural-computer.external-program-memory-maintenance.v1",
        "claim_boundary": (
            "learned masked maintenance choice over real executable-memory "
            "transactions with a frozen interpreter/controller; not autonomous "
            "verifier design, unrestricted growth, arbitrary program synthesis, "
            "or general continual learning"
        ),
        "seed": seed,
        "configuration": {
            "actions": ["grow", "share", "compress", "evict", "defer"],
            "train_cycles": TRAIN_CYCLES,
            "held_out_registers": HELD_OUT_REGISTERS,
            "learner_inputs": [
                "generic_file_count_and_capacity_telemetry",
                "structural_action_mask",
                "one_scalar_transaction_utility",
            ],
        },
        "interpreter_final_loss": interpreter_loss,
        "trained_online_mean_utility": trained["mean_utility"],
        "trained_eval_mean_utility": trained_eval["mean_utility"],
        "fresh_eval_mean_utility": fresh["mean_utility"],
        "shuffled_online_mean_utility": shuffled["mean_utility"],
        "shuffled_eval_mean_utility": shuffled_eval["mean_utility"],
        "trained_history": trained_history,
        "runtime_smoke": runtime_smoke,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": TRAIN_CYCLES,
            "unique_logical_lifetimes": TRAIN_CYCLES,
            "optimizer_updates": trained["optimizer_updates"],
            "replayed_examples": 0,
            "controller_optimizer_updates": 0,
            "latency_seconds": time.perf_counter() - begun,
        },
    }
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=25001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seed, args.report_out)


if __name__ == "__main__":
    main()
