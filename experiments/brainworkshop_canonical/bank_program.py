"""Task-agnostic temporal-program bank operations.

Admission, retrieval, and same-primitive composition are bank operations.
They do not mention Dual, Position, or n-back. Packed I/O stays an adapter.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from neural_computer import ExternalProgramAdmissionReceipt, ExternalTemporalProgramBank
from neural_computer.program import ExternalProgramArtifact
from neural_computer.temporal_program import (
    INTENTION_AND_EXECUTION_SCHEMA,
    INTENTION_AND_INTERPRETER_SCHEMA,
    INTENTION_INVERT_EXECUTION_SCHEMA,
    INTENTION_INVERT_INTERPRETER_SCHEMA,
    PROTOTYPE_MATCH_EXECUTION_SCHEMA,
    PROTOTYPE_MATCH_INTERPRETER_SCHEMA,
    RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
    RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
    TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
    TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    compose_recursive_temporal_program,
    recursive_temporal_primitive,
)

COMPOSITION_LINEAGE_SCHEMA = "neural-computer.temporal-composition-lineage.v1"

from .physical_program_bank import learned_event_context


def invert_artifact(parent: ExternalProgramArtifact) -> ExternalProgramArtifact:
    """Select-then-invert child. This is a combinator, not a new controller."""

    if parent.execution_schema == INTENTION_INVERT_EXECUTION_SCHEMA:
        raise ValueError("invert programs cannot invert again in this family")
    if parent.execution_schema not in {
        TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
        PROTOTYPE_MATCH_EXECUTION_SCHEMA,
    }:
        raise ValueError("only admitted delay or prototype files can invert")
    return ExternalProgramArtifact(
        codes=parent.snapshot(),
        interpreter_schema=INTENTION_INVERT_INTERPRETER_SCHEMA,
        execution_schema=INTENTION_INVERT_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        frontend_digest=parent.frontend_digest,
    )


def and_artifact(
    prototype: torch.Tensor,
    *,
    frontend_digest: str,
) -> ExternalProgramArtifact:
    """AND child: invert(delay) plus a bound prototype row."""

    if not frontend_digest:
        raise ValueError("and programs cannot be built without a frontend digest")
    codes = prototype.detach().cpu().reshape(1, -1)
    return ExternalProgramArtifact(
        codes=codes,
        interpreter_schema=INTENTION_AND_INTERPRETER_SCHEMA,
        execution_schema=INTENTION_AND_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        frontend_digest=frontend_digest,
    )


def prototype_match_artifact(
    event_width: int,
    *,
    prototype: torch.Tensor | None = None,
    frontend_digest: str | None = None,
) -> ExternalProgramArtifact:
    """One current-event template. This is not a relative-history file.

    Invent zeros may omit a frontend digest. Durable admission requires
    one: the template lives in event space of a specific encoder.
    """

    if event_width < 1:
        raise ValueError("prototype width must be positive")
    codes = (
        torch.zeros(1, event_width)
        if prototype is None
        else prototype.detach().cpu().reshape(1, event_width)
    )
    return ExternalProgramArtifact(
        codes=codes,
        interpreter_schema=PROTOTYPE_MATCH_INTERPRETER_SCHEMA,
        execution_schema=PROTOTYPE_MATCH_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        frontend_digest=frontend_digest,
    )


def temporal_address_artifact(machine) -> ExternalProgramArtifact:
    """Snapshot the live address file as one temporal-address row."""

    logits = machine.relative_address_logits.detach().cpu().unsqueeze(0)
    return ExternalProgramArtifact(
        codes=logits,
        interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
        execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
    )


def admit_temporal_program(
    bank_path: Path,
    artifact: ExternalProgramArtifact,
    context: torch.Tensor | Sequence[Sequence[float]],
    outcomes: Sequence[float],
    *,
    machine=None,
    threshold: float = 0.8,
    min_observations: int = 2,
    min_stable_observations: int = 1,
) -> ExternalProgramAdmissionReceipt:
    """Append a verified file, or attach experience to an identical file."""

    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    if machine is not None and not machine.accepts_controller_digest(
        bank.controller_digest
    ):
        raise ValueError("temporal program bank targets another frozen controller")
    if (
        artifact.execution_schema
        in {PROTOTYPE_MATCH_EXECUTION_SCHEMA, INTENTION_AND_EXECUTION_SCHEMA}
        and not artifact.frontend_digest
    ):
        raise ValueError(
            "prototype programs cannot be admitted without a frontend digest"
        )
    width = bank.context_width
    key = (
        context
        if isinstance(context, torch.Tensor)
        else learned_event_context(context, width=width)
    )
    before_slot0 = bank.artifact(0).digest() if bank.program_count else None
    before_count = bank.program_count
    receipt = bank.admit(
        artifact,
        key,
        outcomes,
        threshold=threshold,
        min_observations=min_observations,
        min_stable_observations=min_stable_observations,
    )
    if receipt.accepted:
        if before_slot0 is not None and bank.artifact(0).digest() != before_slot0:
            raise RuntimeError("admission mutated temporal slot 0")
        if bank.program_count < before_count:
            raise RuntimeError("admission removed a temporal program")
        bank.save_bank(bank_path)
    return receipt


def admit_composed_child(
    bank_path: Path,
    slots: Sequence[int],
    context: torch.Tensor | Sequence[Sequence[float]],
    outcomes: Sequence[float],
    *,
    machine=None,
    threshold: float = 0.8,
    min_observations: int = 1,
    min_stable_observations: int = 1,
) -> ExternalProgramAdmissionReceipt:
    """Admit a same-primitive compose as a child file."""

    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    child = compose_admitted_temporal(bank, slots)
    parent_digests = [bank.artifact(int(slot)).digest() for slot in slots]
    receipt = admit_temporal_program(
        bank_path,
        child,
        context,
        outcomes,
        machine=machine,
        threshold=threshold,
        min_observations=min_observations,
        min_stable_observations=min_stable_observations,
    )
    if receipt.accepted and receipt.slot is not None:
        restored = ExternalTemporalProgramBank.load_bank(bank_path)
        attach_composition_lineage(
            restored,
            int(receipt.slot),
            slots,
            parent_digests=parent_digests,
        )
        restored.save_bank(bank_path)
    return receipt


def admit_invert_child(
    bank_path: Path,
    slot: int,
    context: torch.Tensor | Sequence[Sequence[float]],
    outcomes: Sequence[float],
    *,
    machine=None,
    threshold: float = 0.8,
    min_observations: int = 1,
    min_stable_observations: int = 1,
) -> ExternalProgramAdmissionReceipt:
    """Admit invert(parent) as a child file with lineage."""

    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    child = invert_artifact(bank.artifact(int(slot)))
    parent_digest = bank.artifact(int(slot)).digest()
    receipt = admit_temporal_program(
        bank_path,
        child,
        context,
        outcomes,
        machine=machine,
        threshold=threshold,
        min_observations=min_observations,
        min_stable_observations=min_stable_observations,
    )
    if receipt.accepted and receipt.slot is not None:
        restored = ExternalTemporalProgramBank.load_bank(bank_path)
        attach_composition_lineage(
            restored,
            int(receipt.slot),
            (int(slot),),
            parent_digests=(parent_digest,),
        )
        restored.save_bank(bank_path)
    return receipt


def admit_and_child(
    bank_path: Path,
    delay_slot: int,
    prototype: torch.Tensor,
    context: torch.Tensor | Sequence[Sequence[float]],
    outcomes: Sequence[float],
    *,
    frontend_digest: str,
    machine=None,
    threshold: float = 0.8,
    min_observations: int = 1,
    min_stable_observations: int = 1,
) -> ExternalProgramAdmissionReceipt:
    """Admit invert(delay) and a prototype as one child file."""

    bank = ExternalTemporalProgramBank.load_bank(bank_path)
    child = and_artifact(prototype, frontend_digest=frontend_digest)
    parent_digest = bank.artifact(int(delay_slot)).digest()
    receipt = admit_temporal_program(
        bank_path,
        child,
        context,
        outcomes,
        machine=machine,
        threshold=threshold,
        min_observations=min_observations,
        min_stable_observations=min_stable_observations,
    )
    if receipt.accepted and receipt.slot is not None:
        restored = ExternalTemporalProgramBank.load_bank(bank_path)
        attach_composition_lineage(
            restored,
            int(receipt.slot),
            (int(delay_slot),),
            parent_digests=(parent_digest,),
        )
        restored.save_bank(bank_path)
    return receipt


def attach_composition_lineage(
    bank: ExternalTemporalProgramBank,
    slot: int,
    parent_slots: Sequence[int],
    *,
    parent_digests: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Persist parent slots and digests on the child's admission record."""

    child = bank.artifact(slot)
    record = {
        "schema": COMPOSITION_LINEAGE_SCHEMA,
        "slot": int(slot),
        "parent_slots": [int(item) for item in parent_slots],
        "parent_digests": (
            list(parent_digests)
            if parent_digests is not None
            else [bank.artifact(int(item)).digest() for item in parent_slots]
        ),
        "child_digest": child.digest(),
        "depth": len(parent_slots),
        "execution_schema": child.execution_schema,
        "inferred": False,
    }
    for admission in reversed(bank._admissions):
        if (
            admission.get("slot") == slot
            and admission.get("candidate_digest") == child.digest()
        ):
            admission["composition"] = record
            return record
    raise RuntimeError("no admission record for composed child")


def composition_lineage(
    bank: ExternalTemporalProgramBank, slot: int
) -> dict[str, Any] | None:
    """Return stored parent ids, or infer a same-primitive repeat."""

    artifact = bank.artifact(slot)
    for admission in reversed(bank._admissions):
        composition = admission.get("composition")
        if (
            isinstance(composition, dict)
            and admission.get("slot") == slot
            and composition.get("child_digest") == artifact.digest()
        ):
            return dict(composition)
    if (
        artifact.execution_schema != RECURSIVE_TEMPORAL_EXECUTION_SCHEMA
        or artifact.program_length < 2
    ):
        return None
    primitive = artifact.codes[0]
    matches = [
        index
        for index in range(bank.program_count)
        if bank.artifact(index).program_length == 1
        and bank.artifact(index).execution_schema == TEMPORAL_ADDRESS_EXECUTION_SCHEMA
        and torch.equal(bank.artifact(index).codes[0], primitive)
    ]
    if not matches:
        return None
    parent = matches[0]
    return {
        "schema": COMPOSITION_LINEAGE_SCHEMA,
        "slot": int(slot),
        "parent_slots": [parent] * artifact.program_length,
        "parent_digests": [bank.artifact(parent).digest()] * artifact.program_length,
        "child_digest": artifact.digest(),
        "depth": artifact.program_length,
        "execution_schema": artifact.execution_schema,
        "inferred": True,
    }


def compose_admitted_temporal(
    bank: ExternalTemporalProgramBank,
    slots: Sequence[int],
) -> ExternalProgramArtifact:
    """Compose admitted files if they share one primitive row.

    Different primitives fail closed: this family has no SEQUENCE interpreter
    for unequal address files. Depth is the number of parent slots.
    """

    if len(slots) < 2:
        raise ValueError("temporal composition needs at least two parent slots")
    parents = [bank.artifact(int(slot)) for slot in slots]
    primitive = parents[0].codes[0]
    for parent in parents:
        if parent.execution_schema != TEMPORAL_ADDRESS_EXECUTION_SCHEMA:
            raise ValueError(
                "only temporal-address primitives can compose in this family"
            )
        if parent.program_length != 1:
            raise ValueError("temporal composition parents must be one-row files")
        if not torch.equal(parent.codes[0], primitive):
            raise ValueError("unequal temporal primitives cannot compose in this family")
    return compose_recursive_temporal_program(
        recursive_temporal_primitive(parents[0]), len(parents)
    )


def bind_live_prototype(machine, encoders) -> ExternalProgramArtifact:
    """Snapshot the live template and bind it to the current frontend."""

    artifact = prototype_match_artifact(
        machine.event_width,
        prototype=machine.prototype.detach(),
        frontend_digest=encoders.digest(),
    )
    machine._frontend_digest = artifact.frontend_digest
    return artifact


def _delay_parent_for_and(
    bank: ExternalTemporalProgramBank, artifact: ExternalProgramArtifact
) -> ExternalProgramArtifact:
    """Resolve the invert/delay parent recorded on an AND child."""

    for slot in range(bank.program_count):
        lineage = composition_lineage(bank, slot)
        if (
            lineage is None
            or lineage.get("child_digest") != artifact.digest()
            or not lineage.get("parent_slots")
        ):
            continue
        parent = bank.artifact(int(lineage["parent_slots"][0]))
        if parent.execution_schema == INTENTION_INVERT_EXECUTION_SCHEMA:
            return parent
        return invert_artifact(parent)
    raise ValueError("and program has no delay parent in the bank")


def _install_invert_parent(machine, bank, artifact: ExternalProgramArtifact) -> None:
    """Load the wrapped parent, then the caller enables invert."""

    if artifact.instruction_width == bank.instruction_width:
        if artifact.program_length == 1:
            inner = ExternalProgramArtifact(
                codes=artifact.snapshot(),
                interpreter_schema=TEMPORAL_ADDRESS_INTERPRETER_SCHEMA,
                execution_schema=TEMPORAL_ADDRESS_EXECUTION_SCHEMA,
                output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
            )
            machine.load_admitted_program_artifact(
                inner, controller_digest=bank.controller_digest
            )
            if hasattr(machine, "composition_depth"):
                machine.composition_depth = 1
            return
        if not hasattr(machine, "load_recursive_program_artifact"):
            raise ValueError("machine cannot load recursive temporal programs")
        inner = ExternalProgramArtifact(
            codes=artifact.snapshot(),
            interpreter_schema=RECURSIVE_TEMPORAL_INTERPRETER_SCHEMA,
            execution_schema=RECURSIVE_TEMPORAL_EXECUTION_SCHEMA,
            output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        )
        machine.load_recursive_program_artifact(
            inner, controller_digest=machine.controller_digest()
        )
        return
    if not hasattr(machine, "load_prototype_artifact"):
        raise ValueError("machine cannot load prototype-match programs")
    inner = ExternalProgramArtifact(
        codes=artifact.snapshot(),
        interpreter_schema=PROTOTYPE_MATCH_INTERPRETER_SCHEMA,
        execution_schema=PROTOTYPE_MATCH_EXECUTION_SCHEMA,
        output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
        frontend_digest=artifact.frontend_digest,
    )
    machine.load_prototype_artifact(
        inner, controller_digest=machine.controller_digest()
    )


def require_prototype_frontend(artifact: ExternalProgramArtifact, encoders) -> None:
    """Fail closed if a bound prototype meets a different frontend."""

    if artifact.execution_schema not in {
        PROTOTYPE_MATCH_EXECUTION_SCHEMA,
        INTENTION_AND_EXECUTION_SCHEMA,
    }:
        return
    if artifact.frontend_digest is None:
        return
    live = encoders.digest()
    if artifact.frontend_digest != live:
        raise ValueError("prototype frontend digest does not match the live encoder")


def install_temporal_artifact(
    machine,
    bank: ExternalTemporalProgramBank,
    artifact: ExternalProgramArtifact,
    *,
    encoders=None,
) -> None:
    """Load a primitive or composed child onto the frozen machine."""

    if not machine.accepts_controller_digest(bank.controller_digest):
        raise ValueError("temporal program bank targets another frozen controller")
    if hasattr(machine, "_invert_intention"):
        machine._invert_intention = False
    if hasattr(machine, "_combine_and"):
        machine._combine_and = False
    if encoders is not None:
        require_prototype_frontend(artifact, encoders)
    if artifact.execution_schema == RECURSIVE_TEMPORAL_EXECUTION_SCHEMA:
        if not hasattr(machine, "load_recursive_program_artifact"):
            raise ValueError("machine cannot load recursive temporal programs")
        machine.load_recursive_program_artifact(
            artifact, controller_digest=machine.controller_digest()
        )
        return
    if artifact.execution_schema == PROTOTYPE_MATCH_EXECUTION_SCHEMA:
        if not hasattr(machine, "load_prototype_artifact"):
            raise ValueError("machine cannot load prototype-match programs")
        machine.load_prototype_artifact(
            artifact, controller_digest=machine.controller_digest()
        )
        return
    if artifact.execution_schema == INTENTION_INVERT_EXECUTION_SCHEMA:
        _install_invert_parent(machine, bank, artifact)
        if hasattr(machine, "_invert_intention"):
            machine._invert_intention = True
        return
    if artifact.execution_schema == INTENTION_AND_EXECUTION_SCHEMA:
        parent = _delay_parent_for_and(bank, artifact)
        _install_invert_parent(machine, bank, parent)
        if hasattr(machine, "load_prototype_artifact"):
            proto = ExternalProgramArtifact(
                codes=artifact.snapshot(),
                interpreter_schema=PROTOTYPE_MATCH_INTERPRETER_SCHEMA,
                execution_schema=PROTOTYPE_MATCH_EXECUTION_SCHEMA,
                output_schema=TEMPORAL_ADDRESS_OUTPUT_SCHEMA,
                frontend_digest=artifact.frontend_digest,
            )
            machine.load_prototype_artifact(
                proto, controller_digest=machine.controller_digest()
            )
        with torch.no_grad():
            machine.prototype.copy_(artifact.codes[0].to(machine.prototype))
        if hasattr(machine, "_invert_intention"):
            machine._invert_intention = True
        if hasattr(machine, "_combine_and"):
            machine._combine_and = True
        return
    machine.load_admitted_program_artifact(
        artifact, controller_digest=bank.controller_digest
    )
    if hasattr(machine, "composition_depth"):
        machine.composition_depth = max(1, int(artifact.program_length))


def retrieve_temporal_program(
    machine,
    bank: ExternalTemporalProgramBank,
    event_payloads: Sequence[Sequence[float]] | torch.Tensor,
    *,
    exploration: float = 0.0,
    sample: bool = False,
) -> Any:
    """Select by learned events and activate the admitted file."""

    if not machine.accepts_controller_digest(bank.controller_digest):
        raise ValueError("temporal program bank targets another frozen controller")
    context = learned_event_context(event_payloads, width=machine.event_width)
    selection = bank.select(context, exploration=exploration, sample=sample)
    install_temporal_artifact(machine, bank, selection.artifact)
    return selection
