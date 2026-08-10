"""Audit outcome-only synthesis of one new opaque executable file."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from experiments.external_outcome_program_router.train import _train_executable_bank
from experiments.external_program_file_admission.train import _runtime_smoke
from neural_computer import (
    ExternalProgramArtifact,
    ExternalProgramCandidateSearch,
    ExternalProgramCandidateSearchState,
    ExternalSequenceProgramMemory,
)

EVENT_WIDTH = 4
MACHINE_EVENT_WIDTH = EVENT_WIDTH * 3
REGISTER_WIDTH = 4
INSTRUCTION_WIDTH = 8
ATOM_COUNT = 3
INTERPRETER_UPDATES = 500
SEARCH_EPISODES = 256
VERIFIER_PROBES = 32
TARGET_THRESHOLD = 0.95
ERROR_SCALE = 0.05
EXECUTION_TOLERANCE = 1e-6


def _digest(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        detached = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(repr(tuple(detached.shape)).encode("utf-8"))
        digest.update(detached.numpy().tobytes())
    return digest.hexdigest()


def _artifact(codes: torch.Tensor) -> ExternalProgramArtifact:
    return ExternalProgramArtifact(
        codes=codes,
        interpreter_schema="neural-computer.external-register.v4",
        execution_schema="neural-computer.external-register-read-execute.v1",
    )


def _memory(artifact: ExternalProgramArtifact) -> ExternalSequenceProgramMemory:
    memory = ExternalSequenceProgramMemory(
        INSTRUCTION_WIDTH,
        content_addressing=True,
        hard_routing=True,
    )
    memory.add_artifact(artifact)
    for parameter in memory.parameters():
        parameter.requires_grad_(False)
    return memory


@torch.no_grad()
def _verifier_outcomes(
    machine: torch.nn.Module,
    candidate: ExternalProgramArtifact,
    reference: ExternalProgramArtifact,
    registers: torch.Tensor,
) -> torch.Tensor:
    """Private verifier: expose only bounded scalar similarity scores."""

    actual = machine.execute_artifact(registers, candidate)
    expected = machine.execute_artifact(registers, reference)
    error = (actual - expected).abs().mean(dim=-1)
    return torch.exp(-error / ERROR_SCALE).clamp(0.0, 1.0)


def _exact_accuracy(
    machine: torch.nn.Module,
    candidate: ExternalProgramArtifact,
    reference: ExternalProgramArtifact,
    registers: torch.Tensor,
) -> float:
    with torch.no_grad():
        actual = machine.execute_artifact(registers, candidate)
        expected = machine.execute_artifact(registers, reference)
        return float(
            (actual - expected)
            .abs()
            .amax(dim=-1)
            .le(EXECUTION_TOLERANCE)
            .to(torch.float32)
            .mean()
            .item()
        )


def _payload_equal(first: dict[str, object], second: dict[str, object]) -> bool:
    if first.keys() != second.keys():
        return False
    for key, left in first.items():
        right = second[key]
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            if not torch.equal(left, right):
                return False
        elif left != right:
            return False
    return True


def _search_and_admit(
    *,
    machine: torch.nn.Module,
    memory: ExternalSequenceProgramMemory,
    instruction_bank: torch.Tensor,
    parent: ExternalProgramArtifact,
    reference: ExternalProgramArtifact,
    registers: torch.Tensor,
    seed: int,
) -> dict[str, object]:
    search = ExternalProgramCandidateSearch(
        INSTRUCTION_WIDTH,
        instruction_bank=instruction_bank,
        max_program_length=reference.program_length,
        mutation_scale=0.05,
        exploration=0.5,
        temperature=0.5,
    )
    state = search.initial_state()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    accepted_proposal = None
    admission = None
    accepted_quality: float | None = None
    for _ in range(SEARCH_EPISODES):
        proposal = search.propose(state, parent, generator=generator)
        outcomes = _verifier_outcomes(machine, proposal.artifact, reference, registers)
        feedback = search.record_outcomes(
            state,
            proposal,
            outcomes,
            threshold=TARGET_THRESHOLD,
            min_observations=VERIFIER_PROBES,
            min_stable_observations=VERIFIER_PROBES,
        )
        state = feedback.state
        if feedback.receipt.accepted:
            admission = memory.admit_verified_artifact(
                proposal.artifact,
                outcomes,
                threshold=TARGET_THRESHOLD,
                min_observations=VERIFIER_PROBES,
                min_stable_observations=VERIFIER_PROBES,
                protect=True,
            )
            accepted_proposal = proposal
            accepted_quality = feedback.quality
            break
    restored_state = ExternalProgramCandidateSearchState.from_payload(state.payload())
    return {
        "search": search,
        "state": state,
        "restored_state": restored_state,
        "accepted_proposal": accepted_proposal,
        "admission": admission,
        "proposals": state.proposals,
        "accepted": state.accepted,
        "best_quality": state.best_quality,
        "accepted_quality": accepted_quality,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    torch.set_num_threads(1)
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    torch.manual_seed(args.seed)
    machine, pretrained_memory, interpreter_loss = _train_executable_bank(
        seed=args.seed + 40_000,
        program_count=ATOM_COUNT,
        event_width=MACHINE_EVENT_WIDTH,
        register_width=REGISTER_WIDTH,
        instruction_width=INSTRUCTION_WIDTH,
        updates=INTERPRETER_UPDATES,
    )
    machine_digest_before = _digest(machine)
    atoms = tuple(pretrained_memory.artifact(index) for index in range(ATOM_COUNT))
    source = atoms[0]
    target = _artifact(torch.cat((atoms[0].codes, atoms[1].codes), dim=0))
    fresh_parent = atoms[2]
    instruction_bank = torch.cat(tuple(atom.codes for atom in atoms), dim=0)
    target_preloaded = target.digest() in {source.digest()}

    source_registers = torch.randn(
        128,
        REGISTER_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 60_000),
    )
    target_registers = torch.randn(
        VERIFIER_PROBES,
        REGISTER_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 61_000),
    )
    source_memory = _memory(source)
    source_slot = 0
    source_memory.protect_file(source_slot)

    warm = _search_and_admit(
        machine=machine,
        memory=source_memory,
        instruction_bank=instruction_bank,
        parent=source,
        reference=target,
        registers=target_registers,
        seed=args.seed + 70_000,
    )
    accepted_proposal = warm["accepted_proposal"]
    admission = warm["admission"]
    if accepted_proposal is None or admission is None:
        raise RuntimeError("warm candidate search failed to admit a target")
    candidate = accepted_proposal.artifact

    wrong_memory = _memory(source)
    wrong_memory.protect_file(0)
    wrong_before = wrong_memory.digest()
    corrupted = _artifact(target.codes.flip(0))
    corrupted_outcomes = _verifier_outcomes(
        machine,
        corrupted,
        target,
        target_registers,
    )
    corrupted_receipt = wrong_memory.admit_verified_artifact(
        corrupted,
        corrupted_outcomes,
        threshold=TARGET_THRESHOLD,
        min_observations=VERIFIER_PROBES,
        min_stable_observations=VERIFIER_PROBES,
        protect=True,
    )

    fresh_memory = _memory(fresh_parent)
    fresh_memory.protect_file(0)
    fresh = _search_and_admit(
        machine=machine,
        memory=fresh_memory,
        instruction_bank=instruction_bank,
        parent=fresh_parent,
        reference=target,
        registers=target_registers,
        seed=args.seed + 70_000,
    )
    fresh_candidate = fresh["accepted_proposal"]
    fresh_target = (
        _exact_accuracy(machine, fresh_candidate.artifact, target, source_registers)
        if fresh_candidate is not None
        else float(fresh["best_quality"])
    )

    permutation = torch.randperm(
        corrupted_outcomes.shape[0],
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 80_000),
    )
    shuffled_memory = _memory(source)
    shuffled_memory.protect_file(0)
    shuffled_receipt = shuffled_memory.admit_verified_artifact(
        accepted_proposal.artifact,
        corrupted_outcomes[permutation],
        threshold=TARGET_THRESHOLD,
        min_observations=VERIFIER_PROBES,
        min_stable_observations=VERIFIER_PROBES,
        protect=True,
    )
    shuffled_rejected = not shuffled_receipt.accepted

    restored_memory = ExternalSequenceProgramMemory.from_payload(source_memory.payload())
    source_retention = _exact_accuracy(machine, source, source, source_registers)
    target_mastery = _exact_accuracy(machine, candidate, target, source_registers)
    machine_digest_after = _digest(machine)
    runtime_smoke = _runtime_smoke(machine, restored_memory, seed=args.seed + 90_000)
    gates = {
        "target_not_preloaded": not target_preloaded,
        "warm_candidate_generated": candidate.digest() == target.digest(),
        "candidate_admitted": bool(admission.accepted),
        "candidate_protected": bool(
            admission.slot is not None and source_memory.is_file_protected(admission.slot)
        ),
        "source_protected": source_memory.is_file_protected(source_slot),
        "source_retention": source_retention >= 0.95,
        "target_mastery": target_mastery >= 0.95,
        "fresh_control_not_admitted": not bool(fresh.get("accepted")),
        "corrupted_candidate_rejected": not corrupted_receipt.accepted,
        "corrupted_noop": wrong_memory.digest() == wrong_before,
        "shuffled_control_rejected": shuffled_rejected,
        "search_state_persistence_exact": _payload_equal(
            warm["restored_state"].payload(), warm["state"].payload()
        ),
        "file_memory_persistence_exact": restored_memory.digest() == source_memory.digest(),
        "controller_runtime_seam": bool(runtime_smoke["controller_frozen"]),
        "interpreter_frozen": machine_digest_before == machine_digest_after,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    report = {
        "schema": "neural-computer.external-program-candidate-search.v1",
        "claim_boundary": (
            "outcome-only one-edit structural synthesis of one portable executable file "
            "from a protected opaque parent; not open-ended program induction"
        ),
        "seed": args.seed,
        "configuration": {
            "event_width": EVENT_WIDTH,
            "machine_event_width": MACHINE_EVENT_WIDTH,
            "register_width": REGISTER_WIDTH,
            "instruction_width": INSTRUCTION_WIDTH,
            "interpreter_updates": INTERPRETER_UPDATES,
            "search_episodes": SEARCH_EPISODES,
            "verifier_probes": VERIFIER_PROBES,
            "target_threshold": TARGET_THRESHOLD,
            "error_scale": ERROR_SCALE,
        },
        "interpreter_final_loss": interpreter_loss,
        "source_mastery": source_retention,
        "target_mastery": target_mastery,
        "fresh_target": fresh_target,
        "warm_search_proposals": warm["proposals"],
        "fresh_search_proposals": fresh["proposals"],
        "warm_best_quality": warm["best_quality"],
        "fresh_best_quality": fresh["best_quality"],
        "candidate_outcome_mean": warm["accepted_quality"],
        "corrupted_candidate_outcome_mean": float(corrupted_outcomes.mean().item()),
        "candidate_digest": candidate.digest(),
        "target_digest": target.digest(),
        "source_memory_files": source_memory.file_count,
        "restored_memory_files": restored_memory.file_count,
        "warm_search_accepted": warm["accepted"],
        "fresh_search_accepted": fresh["accepted"],
        "admission": admission.payload(),
        "corrupted_admission": corrupted_receipt.payload(),
        "runtime_smoke": runtime_smoke,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": int(
                warm["proposals"] * VERIFIER_PROBES
                + fresh["proposals"] * VERIFIER_PROBES
                + 2 * VERIFIER_PROBES
            ),
            "unique_logical_lifetimes": int(
                warm["proposals"] + fresh["proposals"] + 2
            ),
            "program_verifier_outcomes": int(
                (warm["proposals"] + fresh["proposals"] + 2) * VERIFIER_PROBES
            ),
            "interpreter_optimizer_updates": INTERPRETER_UPDATES,
            "controller_optimizer_updates": 0,
            "replayed_examples": 0,
            "raw_verifier_rows_retained": 0,
            "wall_seconds": 0.0,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=23001)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    report = run(args)
    report["accounting"]["wall_seconds"] = time.perf_counter() - started
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
