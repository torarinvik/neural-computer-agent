"""Bank-driven program search.

Proposals come from admitted files and an allow-listed grammar. The
searcher does not receive task names, n-back, Dual/Position labels, or
correct programs. A lifetime runner is injected so gym and live share
the same search.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from neural_computer import ExternalTemporalProgramBank
from neural_computer.program import ExternalProgramArtifact
from neural_computer.temporal_program import INTENTION_INVERT_EXECUTION_SCHEMA

from .bank_program import (
    bind_live_prototype,
    compose_admitted_temporal,
    install_temporal_artifact,
    invert_artifact,
    prototype_match_artifact,
)

ProposalEvaluator = Callable[["ProgramProposal"], dict[str, Any]]


@dataclass(frozen=True)
class ProgramProposal:
    kind: str
    slots: tuple[int, ...]
    artifact: ExternalProgramArtifact | None
    reason: str | None = None
    # A fully specified template needs no acquire lifetime; the searcher
    # installs it and evaluates it directly.
    template: torch.Tensor | None = None
    template_label: tuple[int, ...] | None = None
    # A template answers a rule either as stated or flipped; both are one
    # hypothesis about which events matter, so both are offered.
    invert_intention: bool = False

    def label(self) -> str:
        joined = "+".join(str(slot) for slot in self.slots)
        base = f"{self.kind}:{joined}"
        if self.template_label is None:
            return base
        clusters = "+".join(str(index) for index in self.template_label)
        polarity = "~" if self.invert_intention else ""
        return f"{base}[{polarity}{clusters}]"


def propose_from_bank(
    bank: ExternalTemporalProgramBank,
    templates: tuple[tuple[tuple[int, ...], torch.Tensor], ...] = (),
) -> tuple[ProgramProposal, ...]:
    """Enumerate retrieve, compose, invert, and invent last.

    Retrieve is tried before compose. Unequal or non-address primitives are
    recorded as illegal and are not executable candidates. Invert wraps an
    admitted file and flips its intention. And combines that invert with
    a fresh prototype. Invent is always a fresh prototype-match file.
    """

    if bank.program_count < 1:
        raise LookupError("program search needs at least one admitted file")
    proposals: list[ProgramProposal] = []
    for slot in range(bank.program_count):
        proposals.append(
            ProgramProposal("retrieve", (slot,), bank.artifact(slot))
        )
    for first in range(bank.program_count):
        for second in range(bank.program_count):
            try:
                artifact = compose_admitted_temporal(bank, (first, second))
            except ValueError as error:
                proposals.append(
                    ProgramProposal(
                        "illegal_compose",
                        (first, second),
                        None,
                        reason=str(error),
                    )
                )
                continue
            proposals.append(ProgramProposal("compose", (first, second), artifact))
    for slot in range(bank.program_count):
        parent = bank.artifact(slot)
        try:
            proposals.append(
                ProgramProposal("invert", (slot,), invert_artifact(parent))
            )
        except ValueError as error:
            proposals.append(
                ProgramProposal(
                    "illegal_compose",
                    (slot,),
                    None,
                    reason=str(error),
                )
            )
    for slot in range(bank.program_count):
        parent = bank.artifact(slot)
        if parent.execution_schema == INTENTION_INVERT_EXECUTION_SCHEMA:
            delay = parent
        else:
            try:
                delay = invert_artifact(parent)
            except ValueError:
                continue
        proposals.append(ProgramProposal("and", (slot,), delay))
    proposals.append(
        ProgramProposal(
            "invent",
            (),
            prototype_match_artifact(bank.context_width),
        )
    )
    # Observed templates are appended, never inserted: an earlier winner stays
    # the winner, so every recorded campaign is unaffected. Within the tail,
    # simpler hypotheses come first because the templates arrive that way.
    for invert in (False, True):
        for subset, template in templates:
            proposals.append(
                ProgramProposal(
                    "invent",
                    (),
                    prototype_match_artifact(bank.context_width, prototype=template),
                    template=template,
                    template_label=subset,
                    invert_intention=invert,
                )
            )
    for slot in range(bank.program_count):
        parent = bank.artifact(slot)
        if parent.execution_schema == INTENTION_INVERT_EXECUTION_SCHEMA:
            delay = parent
        else:
            try:
                delay = invert_artifact(parent)
            except ValueError:
                continue
        for subset, template in templates:
            proposals.append(
                ProgramProposal(
                    "and",
                    (slot,),
                    delay,
                    template=template,
                    template_label=subset,
                )
            )
    return tuple(proposals)


def install_proposal(
    machine,
    bank: ExternalTemporalProgramBank,
    proposal: ProgramProposal,
) -> None:
    """Load one executable proposal onto a frozen machine."""

    if proposal.kind == "illegal_compose" or proposal.artifact is None:
        raise ValueError("illegal compose cannot be installed")
    if proposal.kind not in {"retrieve", "compose", "invent", "invert", "and"}:
        raise ValueError(f"unsupported proposal kind: {proposal.kind}")
    install_temporal_artifact(machine, bank, proposal.artifact)
    if proposal.kind in {"invert", "and"} and hasattr(machine, "_invert_intention"):
        machine._invert_intention = True
    if proposal.kind == "and" and hasattr(machine, "_combine_and"):
        machine._combine_and = True
        machine.sample = False
        with torch.no_grad():
            machine.prototype.zero_()
    if proposal.template is not None and hasattr(machine, "prototype"):
        with torch.no_grad():
            machine.prototype.copy_(proposal.template.to(machine.prototype))
    if proposal.invert_intention and hasattr(machine, "_invert_intention"):
        machine._invert_intention = True


def search_temporal_programs(
    bank: ExternalTemporalProgramBank,
    machine,
    evaluate: ProposalEvaluator,
    *,
    threshold: float = 0.8,
    minimum_bits: int = 8,
    acquire: ProposalEvaluator | None = None,
    encoders=None,
    templates: tuple[tuple[tuple[int, ...], torch.Tensor], ...] = (),
) -> dict[str, Any]:
    """Try existing files, then legal composes, then invent, until one gates.

    Rejected shallower files are not retried. Illegal pairs are counted
    and never executed. Invent may run an optional acquire lifetime
    before the frozen evaluation. That is still a closed grammar.
    """

    proposals = propose_from_bank(bank, templates)
    attempts: list[dict[str, Any]] = []
    winner: dict[str, Any] | None = None
    for proposal in proposals:
        if proposal.kind == "illegal_compose":
            attempts.append(
                {
                    "label": proposal.label(),
                    "kind": proposal.kind,
                    "slots": list(proposal.slots),
                    "executed": False,
                    "accepted": False,
                    "reason": proposal.reason,
                }
            )
            continue
        try:
            install_proposal(machine, bank, proposal)
        except ValueError as error:
            attempts.append(
                {
                    "label": proposal.label(),
                    "kind": proposal.kind,
                    "slots": list(proposal.slots),
                    "executed": False,
                    "accepted": False,
                    "reason": str(error),
                }
            )
            continue
        acquired = None
        bound = None
        if (
            proposal.kind in {"invent", "and"}
            and acquire is not None
            and proposal.template is None
        ):
            acquired = acquire(proposal)
            if proposal.kind == "invent" and encoders is not None:
                bound = bind_live_prototype(machine, encoders)
        report = evaluate(proposal)
        accuracy = float(report["accuracy"])
        bits = int(report["unique_verifier_bits"])
        accepted = accuracy >= threshold and bits >= minimum_bits
        row = {
            "label": proposal.label(),
            "kind": proposal.kind,
            "slots": list(proposal.slots),
            "executed": True,
            "accepted": accepted,
            "accuracy": accuracy,
            "unique_verifier_bits": bits,
        }
        if acquired is not None:
            row["acquire_accuracy"] = float(acquired.get("accuracy", 0.0))
            row["acquire_unique_verifier_bits"] = int(
                acquired.get("unique_verifier_bits", 0)
            )
        if bound is not None:
            row["frontend_digest"] = bound.frontend_digest
        elif (
            proposal.kind in {"and", "invent"}
            and encoders is not None
            and hasattr(machine, "prototype")
        ):
            row["frontend_digest"] = encoders.digest()
        attempts.append(row)
        if accepted:
            winner = row
            break
    return {
        "schema": "neural-computer.temporal-program-search.v1",
        "proposal_count": len(proposals),
        "executed": sum(1 for row in attempts if row["executed"]),
        "illegal_compose": sum(1 for row in attempts if row["kind"] == "illegal_compose"),
        "winner": winner,
        "templates_offered": len(templates),
        "attempts": attempts,
        "controller_digest": machine.controller_digest(),
        "bank_controller_digest": bank.controller_digest,
    }


def main() -> None:
    """Demo: inject a Dual 2-back gym. The searcher never sees that name."""

    import argparse
    import json
    from pathlib import Path

    from .controller_pretraining import (
        build_recursive_temporal_program_machine,
        load_temporal_controller_artifact,
    )
    from .rendered_dual_transfer_pilot import _encoders
    from .rendered_environment import RenderedBrainWorkshopConfig
    from .rendered_live import run_rendered_live_lifetime

    parser = argparse.ArgumentParser()
    repository = Path(__file__).parents[2]
    parser.add_argument(
        "--bank",
        type=Path,
        default=repository / "artifacts/checkpoints/AgentBrain.bank",
    )
    parser.add_argument(
        "--controller-artifact",
        type=Path,
        default=(
            repository
            / "artifacts/checkpoints/temporal_controller_previous_event_seed1001.pt"
        ),
    )
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument(
        "--match-rule",
        choices=("n_back", "current_symbol"),
        default="n_back",
    )
    parser.add_argument("--report-out", type=Path, default=None)
    arguments = parser.parse_args()
    bank = ExternalTemporalProgramBank.load_bank(arguments.bank)
    machine = build_recursive_temporal_program_machine(
        load_temporal_controller_artifact(arguments.controller_artifact),
        sample=False,
        max_sources=2,
        pack_source_actions=True,
    )
    encoders = _encoders(machine)
    config = (
        RenderedBrainWorkshopConfig(
            n_back=1,
            steps=arguments.steps,
            streams=("vision",),
            match_rule="current_symbol",
            target_symbol=0,
        )
        if arguments.match_rule == "current_symbol"
        else RenderedBrainWorkshopConfig(
            n_back=2, steps=arguments.steps, streams=("vision", "audio")
        )
    )

    def evaluate(proposal: ProgramProposal) -> dict[str, Any]:
        report = run_rendered_live_lifetime(
            machine,
            encoders,
            config,
            seed=77 + sum(proposal.slots),
            learn=False,
            sample=False,
        )
        return {
            "accuracy": report.eligible_accuracy,
            "unique_verifier_bits": report.unique_verifier_bits,
        }

    result = search_temporal_programs(bank, machine, evaluate)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.report_out is None:
        print(text, end="")
    else:
        arguments.report_out.parent.mkdir(parents=True, exist_ok=True)
        arguments.report_out.write_text(text)


if __name__ == "__main__":
    main()
