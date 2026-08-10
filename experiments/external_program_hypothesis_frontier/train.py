"""Promote a bounded multi-step opaque program hypothesis frontier.

The controller and interpreter are frozen.  A memory-side frontier composes
opaque executable files with generic edits, receives only deterministic scalar
verifier outcomes, and admits the verified target through the existing atomic
file transaction.  This is a causal architecture test for reusable external
computation, not a claim of open-ended program induction.
"""

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
    ExternalCapabilityRegisterMachine,
    ExternalProgramArtifact,
    ExternalProgramCandidateSearch,
    ExternalProgramHypothesisFrontier,
    ExternalProgramHypothesisFrontierState,
    ExternalSequenceProgramMemory,
)

MACHINE_EVENT_WIDTH = 12
REGISTER_WIDTH = 4
INSTRUCTION_WIDTH = 8
ATOM_COUNT = 3
INTERPRETER_UPDATES = 500
FRONTIER_EVALUATIONS = 256
VERIFIER_PROBES = 32
TARGET_THRESHOLD = 0.95
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
    machine: ExternalCapabilityRegisterMachine,
    candidate: ExternalProgramArtifact,
    reference: ExternalProgramArtifact,
    registers: torch.Tensor,
) -> torch.Tensor:
    """Return only bounded scalar correctness bits from a private verifier."""

    actual = machine.execute_artifact(registers, candidate)
    expected = machine.execute_artifact(registers, reference)
    return (
        (actual - expected)
        .abs()
        .amax(dim=-1)
        .le(EXECUTION_TOLERANCE)
        .to(torch.float32)
    )


@torch.no_grad()
def _exact_accuracy(
    machine: ExternalCapabilityRegisterMachine,
    candidate: ExternalProgramArtifact,
    reference: ExternalProgramArtifact,
    registers: torch.Tensor,
) -> float:
    return float(_verifier_outcomes(machine, candidate, reference, registers).mean().item())


def _frontier_search(
    *,
    machine: ExternalCapabilityRegisterMachine,
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
    frontier = ExternalProgramHypothesisFrontier(
        search,
        beam_width=32,
        max_depth=reference.program_length,
        minimum_quality=0.0,
        proposal_mode="exhaustive",
    )
    state = frontier.initial_state(parent)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    accepted_proposal = None
    accepted_feedback = None
    accepted_outcomes = None
    for _ in range(FRONTIER_EVALUATIONS):
        try:
            proposal = frontier.propose(state, generator=generator)
        except RuntimeError:
            break
        outcomes = _verifier_outcomes(machine, proposal.artifact, reference, registers)
        state, feedback = frontier.record_outcomes(
            state,
            proposal,
            outcomes,
            threshold=TARGET_THRESHOLD,
            min_observations=VERIFIER_PROBES,
            min_stable_observations=VERIFIER_PROBES,
        )
        if feedback.receipt.accepted:
            accepted_proposal = proposal
            accepted_feedback = feedback
            accepted_outcomes = outcomes
            break
    restored_state = ExternalProgramHypothesisFrontierState.from_payload(state.payload())
    return {
        "frontier": frontier,
        "state": state,
        "restored_state": restored_state,
        "accepted_proposal": accepted_proposal,
        "accepted_feedback": accepted_feedback,
        "accepted_outcomes": accepted_outcomes,
        "evaluations": state.evaluations,
        "accepted": state.search_state.accepted,
        "best_quality": state.best_quality,
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
    target = _artifact(torch.cat(tuple(atom.codes for atom in atoms), dim=0))
    instruction_bank = torch.cat(tuple(atom.codes for atom in atoms), dim=0)
    source_memory = _memory(source)
    source_memory.protect_file(0)

    verifier_registers = torch.randn(
        VERIFIER_PROBES,
        REGISTER_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 61_000),
    )
    held_out_registers = torch.randn(
        128,
        REGISTER_WIDTH,
        generator=torch.Generator(device="cpu").manual_seed(args.seed + 62_000),
    )
    warm = _frontier_search(
        machine=machine,
        instruction_bank=instruction_bank,
        parent=source,
        reference=target,
        registers=verifier_registers,
        seed=args.seed + 70_000,
    )
    accepted_proposal = warm["accepted_proposal"]
    accepted_feedback = warm["accepted_feedback"]
    accepted_outcomes = warm["accepted_outcomes"]
    if (
        accepted_proposal is None
        or accepted_feedback is None
        or accepted_outcomes is None
    ):
        raise RuntimeError("warm hypothesis frontier failed to find the target")
    candidate = accepted_proposal.artifact
    admission = source_memory.admit_verified_artifact(
        candidate,
        accepted_outcomes,
        threshold=TARGET_THRESHOLD,
        min_observations=VERIFIER_PROBES,
        min_stable_observations=VERIFIER_PROBES,
        protect=True,
    )
    if not admission.accepted or admission.slot is None:
        raise RuntimeError(f"verified frontier target was not admitted: {admission.reason}")

    random_parent = _artifact(
        torch.randn(
            source.codes.shape,
            generator=torch.Generator(device="cpu").manual_seed(args.seed + 63_000),
        )
    )
    fresh = _frontier_search(
        machine=machine,
        instruction_bank=instruction_bank,
        parent=random_parent,
        reference=target,
        registers=verifier_registers,
        seed=args.seed + 70_000,
    )
    fresh_proposal = fresh["accepted_proposal"]

    corrupted = _artifact(target.codes.flip(0))
    corrupted_outcomes = _verifier_outcomes(machine, corrupted, target, verifier_registers)
    wrong_memory = _memory(source)
    wrong_memory.protect_file(0)
    wrong_memory_digest_before = wrong_memory.digest()
    corrupted_receipt = wrong_memory.admit_verified_artifact(
        corrupted,
        corrupted_outcomes,
        threshold=TARGET_THRESHOLD,
        min_observations=VERIFIER_PROBES,
        min_stable_observations=VERIFIER_PROBES,
        protect=True,
    )

    restored_memory = ExternalSequenceProgramMemory.from_payload(source_memory.payload())
    source_retention = _exact_accuracy(machine, source, source, held_out_registers)
    target_mastery = _exact_accuracy(machine, candidate, target, held_out_registers)
    restored_frontier = ExternalProgramHypothesisFrontierState.from_payload(
        warm["state"].payload()
    )
    runtime_smoke = _runtime_smoke(machine, restored_memory, seed=args.seed + 90_001)
    machine_digest_after = _digest(machine)
    gates = {
        "target_not_preloaded": target.digest() not in {source.digest()},
        "warm_candidate_generated": candidate.digest() == target.digest(),
        "candidate_admitted": bool(admission.accepted),
        "candidate_protected": bool(source_memory.is_file_protected(admission.slot)),
        "source_protected": source_memory.is_file_protected(0),
        "frontier_root_retained": warm["state"].root_digest == source.digest(),
        "source_retention": source_retention >= 0.95,
        "target_mastery": target_mastery >= 0.95,
        "warm_more_efficient_than_fresh": (
            warm["evaluations"] < fresh["evaluations"]
            and fresh_proposal is not None
        ),
        "corrupted_candidate_rejected": not corrupted_receipt.accepted,
        "corrupted_noop": wrong_memory.digest() == wrong_memory_digest_before,
        "frontier_state_persistence_exact": restored_frontier.digest() == warm["state"].digest(),
        "file_memory_persistence_exact": restored_memory.digest() == source_memory.digest(),
        "controller_runtime_seam": bool(runtime_smoke["controller_frozen"]),
        "interpreter_frozen": machine_digest_before == machine_digest_after,
        "zero_replayed_examples": True,
        "zero_controller_optimizer_updates": True,
    }
    report = {
        "schema": "neural-computer.external-program-hypothesis-frontier.v1",
        "claim_boundary": (
            "bounded outcome-only breadth-first composition of one new opaque "
            "multi-step executable file from a useful parent, with protected-root "
            "retention; not open-ended program induction or general continual learning"
        ),
        "seed": args.seed,
        "configuration": {
            "register_width": REGISTER_WIDTH,
            "instruction_width": INSTRUCTION_WIDTH,
            "atom_count": ATOM_COUNT,
            "interpreter_updates": INTERPRETER_UPDATES,
            "frontier_evaluations": FRONTIER_EVALUATIONS,
            "frontier_beam_width": 32,
            "frontier_max_depth": target.program_length,
            "proposal_mode": "exhaustive",
            "verifier_probes": VERIFIER_PROBES,
            "target_threshold": TARGET_THRESHOLD,
            "learner_inputs": [
                "opaque_candidate_program_tensor",
                "opaque_parent_digest",
                "deterministic_scalar_verifier_outcome",
            ],
        },
        "interpreter_final_loss": interpreter_loss,
        "source_mastery": source_retention,
        "source_retention": source_retention,
        "target_mastery": target_mastery,
        "warm_frontier_evaluations": warm["evaluations"],
        "fresh_frontier_evaluations": fresh["evaluations"],
        "warm_frontier_best_quality": warm["best_quality"],
        "fresh_frontier_best_quality": fresh["best_quality"],
        "candidate_outcome_mean": accepted_feedback.quality,
        "corrupted_candidate_outcome_mean": float(corrupted_outcomes.mean().item()),
        "candidate_digest": candidate.digest(),
        "target_digest": target.digest(),
        "source_memory_files": source_memory.file_count,
        "restored_memory_files": restored_memory.file_count,
        "warm_frontier_accepted": warm["accepted"],
        "fresh_frontier_accepted": fresh["accepted"],
        "admission": admission.payload(),
        "corrupted_admission": corrupted_receipt.payload(),
        "runtime_smoke": runtime_smoke,
        "gates": gates,
        "promoted": all(gates.values()),
        "accounting": {
            "unique_verifier_bits": int(
                (warm["evaluations"] + fresh["evaluations"] + 1) * VERIFIER_PROBES
            ),
            "unique_logical_lifetimes": int(
                warm["evaluations"] + fresh["evaluations"] + 1
            ),
            "frontier_verifier_outcomes": int(
                (warm["evaluations"] + fresh["evaluations"] + 1) * VERIFIER_PROBES
            ),
            "held_out_mastery_probes": 128,
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
