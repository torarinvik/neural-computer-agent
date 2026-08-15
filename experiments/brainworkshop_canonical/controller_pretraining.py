"""Pretrain and persist a reusable frozen temporal controller artifact."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch

from .rendered_environment import (
    RenderedBrainWorkshopConfig,
    RenderedBrainWorkshopEncoders,
)
from .rendered_live import (
    PretrainedControllerProgramMachine,
    RecursiveTemporalProgramMachine,
    SourcePreservingTemporalMachine,
    run_rendered_live_lifetime,
)

TEMPORAL_CONTROLLER_ARTIFACT_SCHEMA = (
    "neural-computer.pretrained-temporal-controller-artifact.v2"
)
TEMPORAL_CONTROLLER_REPORT_SCHEMA = (
    "neural-computer.pretrained-temporal-controller-report.v2"
)


@dataclass(frozen=True)
class TemporalControllerPretrainingReport:
    seed: int
    frontend_families: int
    lifetimes_per_family: int
    steps_per_lifetime: int
    unique_verifier_bits: int
    logical_lifetimes: int
    optimizer_updates: int
    replayed_examples: int
    wall_seconds: float
    heldout_frontends: int
    heldout_min_accuracy: float
    heldout_mean_accuracy: float
    inherited_address_probabilities: tuple[float, ...]
    controller_digest: str
    schema: str = TEMPORAL_CONTROLLER_REPORT_SCHEMA

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _controller_configuration(
    machine: SourcePreservingTemporalMachine,
) -> dict[str, int]:
    return {
        "event_width": machine.event_width,
        "source_key_width": machine.source_key_width,
        "max_history": machine.max_history,
        "max_sources": machine.max_sources,
        "action_count": machine.action_count,
        "intention_width": machine.intention_width,
    }


def _controller_state(
    machine: SourcePreservingTemporalMachine,
) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in machine.named_parameters()
        if name != "relative_address_logits"
    }


def pretrain_previous_event_controller(
    *,
    seed: int = 1_001,
    frontend_families: int = 160,
    lifetimes_per_family: int = 3,
    steps_per_lifetime: int = 24,
    heldout_frontends: int = 32,
    event_width: int = 16,
    source_key_width: int = 4,
    max_history: int = 4,
    intention_width: int = 16,
    hidden: int = 24,
    learning_rate: float = 3e-3,
) -> tuple[dict[str, object], TemporalControllerPretrainingReport]:
    """Learn previous-event comparison across independently projected pixels.

    Each frontend has fresh untrained projection and source-key weights. The
    shared controller must therefore retain a relation that transfers across
    opaque event coordinates rather than memorizing one visual encoding.
    """

    if min(
        frontend_families,
        lifetimes_per_family,
        steps_per_lifetime,
        heldout_frontends,
    ) < 1:
        raise ValueError("controller pretraining dimensions must be positive")
    torch.manual_seed(seed)
    machine = SourcePreservingTemporalMachine(
        event_width,
        source_key_width=source_key_width,
        max_history=max_history,
        max_sources=1,
        action_count=2,
        intention_width=intention_width,
        hidden=hidden,
        learning_rate=learning_rate,
        sample=True,
    )
    config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=steps_per_lifetime,
        streams=("vision",),
        symbol_count=8,
    )
    started = perf_counter()
    bits = 0
    lifetimes = 0
    for family in range(frontend_families):
        torch.manual_seed(seed + 10_000 + family)
        encoders = RenderedBrainWorkshopEncoders(
            event_width, source_key_width=source_key_width
        )
        for parameter in encoders.parameters():
            parameter.requires_grad_(False)
        for lifetime in range(lifetimes_per_family):
            trained = run_rendered_live_lifetime(
                machine,
                encoders,
                config,
                seed=family * 10 + lifetime,
                learn=True,
                sample=True,
            )
            bits += trained.unique_verifier_bits
            lifetimes += 1

    controller_state = _controller_state(machine)
    program_prior = machine.relative_address_logits.detach().cpu().clone()
    deployed = PretrainedControllerProgramMachine(
        event_width,
        source_key_width=source_key_width,
        max_history=max_history,
        max_sources=1,
        action_count=2,
        intention_width=intention_width,
        hidden=hidden,
        learning_rate=learning_rate,
        sample=False,
        controller_state=controller_state,
        program_prior=program_prior,
    )
    heldout_config = RenderedBrainWorkshopConfig(
        n_back=1,
        steps=200,
        streams=("vision",),
        symbol_count=2,
    )
    scores: list[float] = []
    for index in range(heldout_frontends):
        torch.manual_seed(seed + 100_000 + index)
        encoders = RenderedBrainWorkshopEncoders(
            event_width, source_key_width=source_key_width
        )
        evaluated = run_rendered_live_lifetime(
            deployed,
            encoders,
            heldout_config,
            seed=seed + 200_000 + index,
            learn=False,
            sample=False,
        )
        scores.append(evaluated.eligible_accuracy)
    elapsed = perf_counter() - started
    report = TemporalControllerPretrainingReport(
        seed=seed,
        frontend_families=frontend_families,
        lifetimes_per_family=lifetimes_per_family,
        steps_per_lifetime=steps_per_lifetime,
        unique_verifier_bits=bits,
        logical_lifetimes=lifetimes,
        optimizer_updates=machine.optimizer_updates,
        replayed_examples=0,
        wall_seconds=elapsed,
        heldout_frontends=heldout_frontends,
        heldout_min_accuracy=min(scores),
        heldout_mean_accuracy=sum(scores) / len(scores),
        inherited_address_probabilities=tuple(
            float(value) for value in program_prior.softmax(dim=0)
        ),
        controller_digest=deployed.controller_digest(),
    )
    payload: dict[str, object] = {
        "schema": TEMPORAL_CONTROLLER_ARTIFACT_SCHEMA,
        "configuration": _controller_configuration(machine),
        "hidden": hidden,
        "controller_state": controller_state,
        "program_prior": program_prior,
        "controller_digest": report.controller_digest,
        "pretraining_report": report.as_dict(),
    }
    return payload, report


def save_temporal_controller_artifact(
    payload: dict[str, object], path: Path
) -> None:
    if payload.get("schema") != TEMPORAL_CONTROLLER_ARTIFACT_SCHEMA:
        raise ValueError("unsupported temporal controller artifact")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_temporal_controller_artifact(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != TEMPORAL_CONTROLLER_ARTIFACT_SCHEMA
        or not isinstance(payload.get("configuration"), dict)
        or not isinstance(payload.get("controller_state"), dict)
        or not isinstance(payload.get("program_prior"), torch.Tensor)
        or not isinstance(payload.get("controller_digest"), str)
    ):
        raise ValueError("temporal controller artifact is malformed")
    return payload


def build_pretrained_controller_program_machine(
    payload: dict[str, object],
    *,
    learning_rate: float = 3e-3,
    sample: bool = True,
    inherit_program_prior: bool = False,
) -> PretrainedControllerProgramMachine:
    """Instantiate a frozen controller with a fresh or inherited task program."""

    configuration = payload.get("configuration")
    controller_state = payload.get("controller_state")
    program_prior = payload.get("program_prior")
    hidden = payload.get("hidden")
    if (
        payload.get("schema") != TEMPORAL_CONTROLLER_ARTIFACT_SCHEMA
        or not isinstance(configuration, dict)
        or set(configuration)
        != {
            "event_width",
            "source_key_width",
            "max_history",
            "max_sources",
            "action_count",
            "intention_width",
        }
        or any(not isinstance(value, int) for value in configuration.values())
        or not isinstance(hidden, int)
        or not isinstance(controller_state, dict)
        or any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in controller_state.items()
        )
        or not isinstance(program_prior, torch.Tensor)
    ):
        raise ValueError("temporal controller artifact cannot build a machine")
    machine = PretrainedControllerProgramMachine(
        configuration["event_width"],
        source_key_width=configuration["source_key_width"],
        max_history=configuration["max_history"],
        max_sources=configuration["max_sources"],
        action_count=configuration["action_count"],
        intention_width=configuration["intention_width"],
        hidden=hidden,
        learning_rate=learning_rate,
        sample=sample,
        controller_state=controller_state,
        program_prior=program_prior,
        initialize_program_from_prior=inherit_program_prior,
    )
    if machine.controller_digest() != payload.get("controller_digest"):
        raise ValueError("temporal controller artifact digest mismatch")
    return machine


def build_recursive_temporal_program_machine(
    payload: dict[str, object],
    *,
    learning_rate: float = 3e-3,
    sample: bool = False,
    max_history: int | None = None,
    max_sources: int | None = None,
    pack_source_actions: bool = False,
) -> RecursiveTemporalProgramMachine:
    """Reuse a verified controller artifact behind the recursive interpreter."""

    validated = build_pretrained_controller_program_machine(
        payload,
        learning_rate=learning_rate,
        sample=sample,
        inherit_program_prior=False,
    )
    controller_state = payload["controller_state"]
    program_prior = payload["program_prior"]
    hidden = payload["hidden"]
    if (
        not isinstance(controller_state, dict)
        or not isinstance(program_prior, torch.Tensor)
        or not isinstance(hidden, int)
    ):
        raise TypeError("controller artifact cannot build a recursive machine")
    history = validated.max_history if max_history is None else int(max_history)
    sources = validated.max_sources if max_sources is None else int(max_sources)
    if history < validated.max_history:
        raise ValueError("cannot shrink a frozen temporal history")
    if sources < validated.max_sources:
        raise ValueError("cannot shrink frozen source capacity")
    if isinstance(program_prior, torch.Tensor) and program_prior.numel() < history:
        program_prior = torch.cat(
            (
                program_prior,
                torch.zeros(history - program_prior.numel()),
            )
        )
    machine = RecursiveTemporalProgramMachine(
        validated.event_width,
        source_key_width=validated.source_key_width,
        max_history=history,
        max_sources=sources,
        action_count=validated.action_count,
        intention_width=validated.intention_width,
        hidden=hidden,
        learning_rate=learning_rate,
        sample=sample,
        pack_source_actions=pack_source_actions,
        identity_max_sources=validated.max_sources,
        identity_action_count=validated.action_count,
        controller_state=controller_state,
        program_prior=program_prior,
        initialize_program_from_prior=False,
    )
    resized = (
        max_history is not None
        or max_sources is not None
        or pack_source_actions
    )
    if not resized and machine.legacy_controller_digest() != payload.get(
        "controller_digest"
    ):
        raise ValueError("recursive machine changed the legacy controller weights")
    if resized:
        reference = build_recursive_temporal_program_machine(
            payload, learning_rate=learning_rate, sample=sample
        )
        if any(
            not torch.equal(machine.state_dict()[name], reference.state_dict()[name])
            for name, _value in reference.named_parameters()
            if name != "relative_address_logits" and name in machine.state_dict()
        ):
            raise ValueError("capacity growth changed frozen relation weights")
    return machine


def save_temporal_controller_report(
    report: TemporalControllerPretrainingReport, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")
