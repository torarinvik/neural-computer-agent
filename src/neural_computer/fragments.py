"""Composable external skill fragments for the amodal neural computer.

The durable unit in this module is deliberately smaller than a task policy.
An :class:`ExternalSkillFragmentArtifact` stores coefficients over one shared
opaque operator basis.  A memory-side router selects fragments from learned
queries, and the shared register interpreter executes the resulting ordered
chain.  The controller never sees a fragment index, task name, raw stream, or
device protocol.

This is an architecture boundary, not a claim that arbitrary programs have
been learned.  The shared basis and the route/composition policy are
replaceable external state; the controller remains fixed-size and the bank
may grow by appending fragment rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from .addressing import OpaqueCandidateGrowthRouter

EXTERNAL_SKILL_FRAGMENT_SCHEMA = "neural-computer.skill-fragment.v1"
EXTERNAL_SKILL_FRAGMENT_BANK_SCHEMA = "neural-computer.skill-fragment-bank.v1"
EXTERNAL_SKILL_FRAGMENT_ROUTE_SCHEMA = "neural-computer.skill-fragment-route.v1"
EXTERNAL_SKILL_FRAGMENT_COMPOSITION_SCHEMA = (
    "neural-computer.skill-fragment-composition.v2"
)
EXTERNAL_SKILL_FRAGMENT_TRACE_SCHEMA = "neural-computer.skill-fragment-trace.v1"
EXTERNAL_SKILL_FRAGMENT_RICH_TRACE_SCHEMA = (
    "neural-computer.skill-fragment-rich-trace.v2"
)
EXTERNAL_SKILL_FRAGMENT_GROWTH_SCHEMA = (
    "neural-computer.skill-fragment-growth-combiner.v2"
)
EXTERNAL_SKILL_FRAGMENT_SERIAL_SCHEMA = (
    "neural-computer.skill-fragment-serial-combiner.v1"
)
EXTERNAL_SKILL_FRAGMENT_OPERATOR_SCHEMA = (
    "neural-computer.skill-fragment-operator-combiner.v1"
)
PERSISTENT_EXTERNAL_SKILL_FRAGMENT_BANK_SCHEMA = (
    "neural-computer.persistent-skill-fragment-bank.v1"
)
PERSISTENT_EXTERNAL_SKILL_FRAGMENT_GROWTH_SCHEMA = (
    "neural-computer.persistent-skill-fragment-growth.v2"
)
PERSISTENT_EXTERNAL_SKILL_FRAGMENT_SERIAL_SCHEMA = (
    "neural-computer.persistent-skill-fragment-serial.v1"
)
PERSISTENT_EXTERNAL_SKILL_FRAGMENT_OPERATOR_SCHEMA = (
    "neural-computer.persistent-skill-fragment-operator.v1"
)


def _digest_tensor(digest: Any, value: torch.Tensor) -> None:
    detached = value.detach().cpu().contiguous()
    digest.update(str(detached.dtype).encode("utf-8"))
    digest.update(repr(tuple(detached.shape)).encode("utf-8"))
    digest.update(detached.numpy().tobytes())


def _validate_digest(value: str, *, name: str) -> None:
    if len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error


@dataclass(frozen=True)
class ExternalSkillFragmentArtifact:
    """Portable opaque coefficients for one reusable fragment.

    ``coefficients`` are not a task-specific parameter block.  They select a
    short sequence of operators from the bank's shared basis.  ``key`` is an
    opaque address learned from attempted scalar outcomes.  Parent digests are
    provenance only; they do not encode task or modality identity.
    """

    coefficients: torch.Tensor
    key: torch.Tensor
    parent_digests: tuple[str, ...] = ()
    schema: str = EXTERNAL_SKILL_FRAGMENT_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != EXTERNAL_SKILL_FRAGMENT_SCHEMA:
            raise ValueError("unsupported skill fragment schema")
        if (
            not isinstance(self.coefficients, torch.Tensor)
            or self.coefficients.ndim != 2
            or self.coefficients.shape[0] < 1
            or self.coefficients.shape[1] < 1
        ):
            raise ValueError("fragment coefficients must have shape [steps, basis]")
        if not isinstance(self.key, torch.Tensor) or self.key.ndim != 1:
            raise ValueError("fragment key must have shape [key_width]")
        if not bool(torch.isfinite(self.coefficients).all()) or not bool(
            torch.isfinite(self.key).all()
        ):
            raise ValueError("fragment tensors must be finite")
        if not isinstance(self.parent_digests, tuple):
            raise TypeError("fragment parent digests must be a tuple")
        for digest in self.parent_digests:
            _validate_digest(digest, name="fragment parent digest")

    @property
    def step_count(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def basis_count(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def key_width(self) -> int:
        return int(self.key.shape[0])

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_SKILL_FRAGMENT_SCHEMA,
            "step_count": self.step_count,
            "basis_count": self.basis_count,
            "key_width": self.key_width,
            "representation": "opaque_shared_basis_coefficients_v1",
            "parents": len(self.parent_digests),
        }

    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.schema.encode("utf-8"))
        _digest_tensor(digest, self.coefficients)
        _digest_tensor(digest, self.key)
        for parent in self.parent_digests:
            digest.update(parent.encode("ascii"))
        return digest.hexdigest()

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "coefficients": self.coefficients.detach().cpu().clone(),
            "key": self.key.detach().cpu().clone(),
            "parent_digests": list(self.parent_digests),
            "sha256": self.digest(),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExternalSkillFragmentArtifact:
        if not isinstance(payload, Mapping):
            raise TypeError("fragment payload must be a mapping")
        coefficients = payload.get("coefficients")
        key = payload.get("key")
        if not isinstance(coefficients, torch.Tensor) or not isinstance(
            key, torch.Tensor
        ):
            raise TypeError("fragment payload must contain tensor coefficients and key")
        configuration = payload.get("configuration")
        if not isinstance(configuration, Mapping):
            raise TypeError("fragment configuration is missing")
        artifact = cls(
            coefficients=coefficients,
            key=key,
            parent_digests=tuple(payload.get("parent_digests", ())),
            schema=str(payload.get("schema", "")),
        )
        if configuration.get("schema") != EXTERNAL_SKILL_FRAGMENT_SCHEMA:
            raise ValueError("fragment configuration schema mismatch")
        if int(configuration.get("step_count", -1)) != artifact.step_count:
            raise ValueError("fragment step-count metadata mismatch")
        if int(configuration.get("basis_count", -1)) != artifact.basis_count:
            raise ValueError("fragment basis-count metadata mismatch")
        if int(configuration.get("key_width", -1)) != artifact.key_width:
            raise ValueError("fragment key-width metadata mismatch")
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != artifact.digest():
            raise ValueError("fragment checksum mismatch")
        return artifact


@dataclass(frozen=True)
class ExternalSkillFragmentRoute:
    """Opaque route scores and physical positions for one query batch."""

    indices: torch.Tensor
    scores: torch.Tensor
    weights: torch.Tensor
    schema: str = EXTERNAL_SKILL_FRAGMENT_ROUTE_SCHEMA

    def validate(
        self, *, batch_size: int, fragment_count: int
    ) -> ExternalSkillFragmentRoute:
        if self.schema != EXTERNAL_SKILL_FRAGMENT_ROUTE_SCHEMA:
            raise ValueError("unsupported fragment route schema")
        if self.indices.ndim != 2 or self.scores.shape != self.indices.shape:
            raise ValueError("fragment route indices and scores must be [batch, top_k]")
        if self.weights.shape != self.indices.shape:
            raise ValueError("fragment route weights must align with indices")
        if self.indices.shape[0] != batch_size or self.indices.dtype != torch.int64:
            raise ValueError("fragment route batch or index dtype is invalid")
        if (
            fragment_count < 1
            or bool((self.indices < 0).any())
            or bool((self.indices >= fragment_count).any())
        ):
            raise ValueError("fragment route index is outside the bank")
        if not bool(torch.isfinite(self.scores).all()) or not bool(
            torch.isfinite(self.weights).all()
        ):
            raise ValueError("fragment route values must be finite")
        if not bool(
            torch.allclose(
                self.weights.sum(-1),
                torch.ones(batch_size, device=self.weights.device),
                atol=1e-5,
            )
        ):
            raise ValueError("fragment route weights must sum to one")
        return self


@dataclass(frozen=True)
class ExternalSkillFragmentComposition:
    """Padded executable chain produced by opaque fragment queries."""

    fragment_indices: torch.Tensor
    route_scores: torch.Tensor
    codes: torch.Tensor
    mask: torch.Tensor
    schema: str = EXTERNAL_SKILL_FRAGMENT_COMPOSITION_SCHEMA
    bank_fragment_count: int = 0
    fragment_step_counts: torch.Tensor | None = None

    def validate(
        self,
        *,
        batch_size: int,
        instruction_width: int,
        fragment_count: int,
    ) -> ExternalSkillFragmentComposition:
        if self.schema != EXTERNAL_SKILL_FRAGMENT_COMPOSITION_SCHEMA:
            raise ValueError("unsupported fragment composition schema")
        if self.bank_fragment_count < 1:
            raise ValueError("fragment composition must declare a nonempty bank")
        if fragment_count != self.bank_fragment_count:
            raise ValueError("fragment composition bank cardinality mismatch")
        if (
            self.fragment_indices.ndim != 2
            or self.route_scores.shape != self.fragment_indices.shape
        ):
            raise ValueError(
                "fragment composition route tensors must be [batch, steps]"
            )
        if (
            self.fragment_indices.shape[0] != batch_size
            or self.fragment_indices.dtype != torch.int64
        ):
            raise ValueError("fragment composition route shape is invalid")
        if (
            self.codes.ndim != 3
            or self.codes.shape[0] != batch_size
            or self.codes.shape[2] != instruction_width
        ):
            raise ValueError("fragment composition codes have the wrong shape")
        if self.mask.shape != self.codes.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("fragment composition mask has the wrong shape")
        if bool((self.fragment_indices < 0).any()) or bool(
            (self.fragment_indices >= fragment_count).any()
        ):
            raise ValueError("fragment composition index is outside the bank")
        if not bool(torch.isfinite(self.codes).all()) or not bool(
            torch.isfinite(self.route_scores).all()
        ):
            raise ValueError("fragment composition values must be finite")
        if self.fragment_step_counts is not None:
            if (
                self.fragment_step_counts.ndim != 2
                or self.fragment_step_counts.shape != self.fragment_indices.shape
                or self.fragment_step_counts.dtype != torch.int64
            ):
                raise ValueError("fragment step counts have the wrong shape")
            if bool((self.fragment_step_counts < 1).any()) or bool(
                (self.fragment_step_counts.sum(dim=1) != self.mask.sum(dim=1)).any()
            ):
                raise ValueError("fragment step counts do not match composition codes")
        return self


@dataclass(frozen=True)
class ExternalSkillFragmentExecutionTrace:
    """Padded evidence for one external serial execution.

    Version one carries only post-instruction states.  Version two may also
    carry the opaque materialized instruction code and state transition delta
    for every step.  These are learned external tensors, not fragment indices,
    raw events, task labels, or verifier metadata.  Keeping the richer form
    opt-in preserves the original state-only ABI while making compositional
    generalization possible for consumers that need operator identity.
    """

    states: torch.Tensor
    mask: torch.Tensor
    fragment_indices: torch.Tensor
    route_scores: torch.Tensor
    bank_fragment_count: int
    schema: str = EXTERNAL_SKILL_FRAGMENT_TRACE_SCHEMA
    instruction_codes: torch.Tensor | None = None
    transition_deltas: torch.Tensor | None = None
    fragment_step_counts: torch.Tensor | None = None

    def validate(
        self,
        *,
        batch_size: int,
        register_width: int,
    ) -> ExternalSkillFragmentExecutionTrace:
        if self.schema not in (
            EXTERNAL_SKILL_FRAGMENT_TRACE_SCHEMA,
            EXTERNAL_SKILL_FRAGMENT_RICH_TRACE_SCHEMA,
        ):
            raise ValueError("unsupported fragment execution trace schema")
        if (
            self.states.ndim != 3
            or self.states.shape[0] != batch_size
            or self.states.shape[1] < 1
            or self.states.shape[2] != register_width
        ):
            raise ValueError("fragment execution trace states have the wrong shape")
        if self.mask.shape != self.states.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("fragment execution trace mask has the wrong shape")
        if not bool(self.mask.any(dim=1).all()):
            raise ValueError("fragment execution trace cannot contain empty rows")
        if (
            self.fragment_indices.ndim != 2
            or self.route_scores.shape != self.fragment_indices.shape
        ):
            raise ValueError(
                "fragment execution trace route tensors must be [batch, steps]"
            )
        if (
            self.fragment_indices.shape[0] != batch_size
            or self.fragment_indices.dtype != torch.int64
        ):
            raise ValueError("fragment execution trace route shape is invalid")
        if (
            self.bank_fragment_count < 1
            or bool((self.fragment_indices < 0).any())
            or bool((self.fragment_indices >= self.bank_fragment_count).any())
        ):
            raise ValueError("fragment execution trace index is outside the bank")
        if not bool(torch.isfinite(self.states).all()) or not bool(
            torch.isfinite(self.route_scores).all()
        ):
            raise ValueError("fragment execution trace values must be finite")
        rich = self.schema == EXTERNAL_SKILL_FRAGMENT_RICH_TRACE_SCHEMA
        if rich != (
            self.instruction_codes is not None
            and self.transition_deltas is not None
            and self.fragment_step_counts is not None
        ):
            raise ValueError(
                "rich fragment traces must carry code, delta, and segment tensors"
            )
        if self.instruction_codes is not None:
            if (
                self.instruction_codes.ndim != 3
                or self.instruction_codes.shape[:2] != self.states.shape[:2]
            ):
                raise ValueError(
                    "fragment trace instruction codes have the wrong shape"
                )
            if not bool(torch.isfinite(self.instruction_codes).all()):
                raise ValueError("fragment trace instruction codes must be finite")
        if self.transition_deltas is not None:
            if self.transition_deltas.shape != self.states.shape:
                raise ValueError(
                    "fragment trace transition deltas have the wrong shape"
                )
            if not bool(torch.isfinite(self.transition_deltas).all()):
                raise ValueError("fragment trace transition deltas must be finite")
        if self.fragment_step_counts is not None:
            if (
                self.fragment_step_counts.ndim != 2
                or self.fragment_step_counts.shape[0] != batch_size
                or self.fragment_step_counts.shape[1] != self.fragment_indices.shape[1]
                or self.fragment_step_counts.dtype != torch.int64
            ):
                raise ValueError("fragment trace step counts have the wrong shape")
            if bool((self.fragment_step_counts < 1).any()):
                raise ValueError("fragment trace step counts must be positive")
            if bool(
                (self.fragment_step_counts.sum(dim=1) != self.mask.sum(dim=1)).any()
            ):
                raise ValueError("fragment trace step counts do not match its mask")
        return self

    @property
    def final_state(self) -> torch.Tensor:
        """Return the last valid post-instruction state for each row."""

        if self.states.ndim != 3 or self.mask.shape != self.states.shape[:2]:
            raise ValueError("fragment execution trace is not shape-valid")
        last = self.mask.sum(dim=1).to(dtype=torch.int64) - 1
        return self.states[
            torch.arange(self.states.shape[0], device=self.states.device), last
        ]

    def learner_view(self) -> ExternalSkillFragmentLearnerTrace:
        """Drop routing receipts before execution evidence reaches a learner."""

        self.validate(
            batch_size=self.states.shape[0],
            register_width=self.states.shape[2],
        )
        return ExternalSkillFragmentLearnerTrace(
            states=self.states,
            mask=self.mask,
            schema=self.schema,
            instruction_codes=self.instruction_codes,
            transition_deltas=self.transition_deltas,
            fragment_step_counts=self.fragment_step_counts,
        ).validate(
            batch_size=self.states.shape[0],
            register_width=self.states.shape[2],
        )


@dataclass(frozen=True)
class ExternalSkillFragmentLearnerTrace:
    """Strict learner view of an external execution.

    Routing receipts stay on the memory side of the boundary.  A combiner
    receives only learned execution evidence, masks, and structural segment
    lengths; it cannot inspect physical fragment indices, route scores, bank
    cardinality, or any other address metadata even if the transport trace
    carries those fields for diagnostics.
    """

    states: torch.Tensor
    mask: torch.Tensor
    schema: str = EXTERNAL_SKILL_FRAGMENT_TRACE_SCHEMA
    instruction_codes: torch.Tensor | None = None
    transition_deltas: torch.Tensor | None = None
    fragment_step_counts: torch.Tensor | None = None

    def validate(
        self,
        *,
        batch_size: int,
        register_width: int,
    ) -> ExternalSkillFragmentLearnerTrace:
        if self.schema not in (
            EXTERNAL_SKILL_FRAGMENT_TRACE_SCHEMA,
            EXTERNAL_SKILL_FRAGMENT_RICH_TRACE_SCHEMA,
        ):
            raise ValueError("unsupported fragment learner trace schema")
        if (
            self.states.ndim != 3
            or self.states.shape[0] != batch_size
            or self.states.shape[1] < 1
            or self.states.shape[2] != register_width
        ):
            raise ValueError("fragment learner trace states have the wrong shape")
        if self.mask.shape != self.states.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("fragment learner trace mask has the wrong shape")
        if not bool(self.mask.any(dim=1).all()):
            raise ValueError("fragment learner trace cannot contain empty rows")
        if not bool(torch.isfinite(self.states).all()):
            raise ValueError("fragment learner trace states must be finite")
        rich = self.schema == EXTERNAL_SKILL_FRAGMENT_RICH_TRACE_SCHEMA
        if rich != (
            self.instruction_codes is not None
            and self.transition_deltas is not None
            and self.fragment_step_counts is not None
        ):
            raise ValueError(
                "rich fragment learner traces must carry code, delta, and segment tensors"
            )
        if self.instruction_codes is not None:
            if (
                self.instruction_codes.ndim != 3
                or self.instruction_codes.shape[:2] != self.states.shape[:2]
            ):
                raise ValueError(
                    "fragment learner trace instruction codes have the wrong shape"
                )
            if not bool(torch.isfinite(self.instruction_codes).all()):
                raise ValueError(
                    "fragment learner trace instruction codes must be finite"
                )
        if self.transition_deltas is not None:
            if self.transition_deltas.shape != self.states.shape:
                raise ValueError(
                    "fragment learner trace transition deltas have the wrong shape"
                )
            if not bool(torch.isfinite(self.transition_deltas).all()):
                raise ValueError(
                    "fragment learner trace transition deltas must be finite"
                )
        if self.fragment_step_counts is not None and (
            self.fragment_step_counts.ndim != 2
            or self.fragment_step_counts.shape[0] != batch_size
            or self.fragment_step_counts.dtype != torch.int64
            or bool((self.fragment_step_counts < 1).any())
            or bool(
                (self.fragment_step_counts.sum(dim=1) != self.mask.sum(dim=1)).any()
            )
        ):
            raise ValueError(
                "fragment learner trace segment lengths have the wrong shape"
            )
        return self


class ExternalSkillFragmentCombiner(nn.Module):
    """Learned order-sensitive combiner over executed fragment states.

    This is deliberately outside the controller.  It receives only the
    post-instruction state trace and its transport mask; it cannot inspect raw
    events, feedback, fragment indices, or verifier metadata.  A growing bank
    can therefore train or replace this shared combiner independently while
    the controller remains frozen.
    """

    schema = "neural-computer.skill-fragment-combiner.v1"

    def __init__(
        self,
        register_width: int,
        output_width: int,
        *,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(register_width, output_width, hidden) < 1:
            raise ValueError("fragment combiner dimensions must be positive")
        self.register_width = int(register_width)
        self.output_width = int(output_width)
        self.hidden = int(hidden)
        self.cell = nn.GRUCell(self.register_width, self.hidden)
        self.output = nn.Linear(self.hidden, self.output_width)

    def configuration(self) -> dict[str, int | str]:
        return {
            "schema": self.schema,
            "register_width": self.register_width,
            "output_width": self.output_width,
            "hidden": self.hidden,
            "input": "post_instruction_state_trace_only_v1",
            "order": "causal_gru_cell_v1",
        }

    def forward(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
    ) -> torch.Tensor:
        if isinstance(trace, ExternalSkillFragmentExecutionTrace):
            trace = trace.learner_view()
        trace.validate(
            batch_size=trace.states.shape[0],
            register_width=self.register_width,
        )
        hidden = torch.zeros(
            trace.states.shape[0],
            self.hidden,
            device=trace.states.device,
            dtype=trace.states.dtype,
        )
        for state, present in zip(
            trace.states.transpose(0, 1),
            trace.mask.transpose(0, 1),
            strict=True,
        ):
            proposal = self.cell(state, hidden)
            hidden = torch.where(present.unsqueeze(-1), proposal, hidden)
        return self.output(hidden)


class ExternalSkillFragmentProgramCombiner(nn.Module):
    """Shared order-sensitive combiner over opaque instruction evidence.

    Unlike :class:`ExternalSkillFragmentCombiner`, this learner receives the
    post-instruction state, its transition delta, and the learned materialized
    instruction code for each step.  It still cannot inspect fragment indices,
    route keys, raw events, feedback, or verifier metadata.  The extra learned
    token stream preserves enough operator identity for one combiner to learn
    reusable composition rules instead of memorizing complete state traces.
    """

    schema = "neural-computer.skill-fragment-program-combiner.v1"
    requires_instruction_codes = True

    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        output_width: int,
        *,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(register_width, instruction_width, output_width, hidden) < 1:
            raise ValueError("program combiner dimensions must be positive")
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.output_width = int(output_width)
        self.hidden = int(hidden)
        self.input = nn.Sequential(
            nn.Linear(register_width * 2 + instruction_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.cell = nn.GRUCell(hidden, hidden)
        self.output = nn.Linear(hidden, output_width)

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": self.schema,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "output_width": self.output_width,
            "hidden": self.hidden,
            "input": "post_state_transition_delta_opaque_code_trace_v2",
            "uses_fragment_indices": False,
            "uses_verifier_metadata": False,
            "order": "causal_gru_cell_v1",
        }

    def forward(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
    ) -> torch.Tensor:
        if isinstance(trace, ExternalSkillFragmentExecutionTrace):
            trace = trace.learner_view()
        trace.validate(
            batch_size=trace.states.shape[0],
            register_width=self.register_width,
        )
        if trace.instruction_codes is None or trace.transition_deltas is None:
            raise ValueError("program combiner requires a rich fragment trace")
        if trace.instruction_codes.shape[2] != self.instruction_width:
            raise ValueError("program combiner instruction width does not match trace")
        hidden = torch.zeros(
            trace.states.shape[0],
            self.hidden,
            device=trace.states.device,
            dtype=trace.states.dtype,
        )
        for state, delta, code, present in zip(
            trace.states.transpose(0, 1),
            trace.transition_deltas.transpose(0, 1),
            trace.instruction_codes.transpose(0, 1),
            trace.mask.transpose(0, 1),
            strict=True,
        ):
            token = self.input(torch.cat((state, delta, code), dim=-1))
            proposal = self.cell(token, hidden)
            hidden = torch.where(present.unsqueeze(-1), proposal, hidden)
        return self.output(hidden)


class ExternalSkillFragmentSegmentCombiner(nn.Module):
    """Hierarchical combiner that preserves fragment boundaries.

    The inner recurrent cell reads instruction-code steps within one fragment;
    the outer recurrent cell reads one learned summary per fragment.  Segment
    lengths are transport metadata from the external bank, not semantic labels,
    and the learner still receives no fragment indices or verifier information.
    This prevents variable-length materialization from erasing the boundary
    between two reusable files before composition learning begins.
    """

    schema = "neural-computer.skill-fragment-segment-combiner.v1"
    requires_instruction_codes = True

    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        output_width: int,
        *,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        if min(register_width, instruction_width, output_width, hidden) < 1:
            raise ValueError("segment combiner dimensions must be positive")
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.output_width = int(output_width)
        self.hidden = int(hidden)
        self.step_input = nn.Sequential(
            nn.Linear(register_width * 2 + instruction_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.step_cell = nn.GRUCell(hidden, hidden)
        self.segment_input = nn.Sequential(
            nn.Linear(register_width + hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.segment_cell = nn.GRUCell(hidden, hidden)
        self.output = nn.Linear(hidden, output_width)

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": self.schema,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "output_width": self.output_width,
            "hidden": self.hidden,
            "input": "hierarchical_segment_trace_v2",
            "uses_fragment_indices": False,
            "uses_verifier_metadata": False,
            "inner_order": "causal_gru_cell_v1",
            "outer_order": "causal_gru_cell_v1",
        }

    def forward(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
    ) -> torch.Tensor:
        if isinstance(trace, ExternalSkillFragmentExecutionTrace):
            trace = trace.learner_view()
        trace.validate(
            batch_size=trace.states.shape[0],
            register_width=self.register_width,
        )
        if (
            trace.instruction_codes is None
            or trace.transition_deltas is None
            or trace.fragment_step_counts is None
        ):
            raise ValueError("segment combiner requires a rich fragment trace")
        if trace.instruction_codes.shape[2] != self.instruction_width:
            raise ValueError("segment combiner instruction width does not match trace")
        batch_size, max_steps, _ = trace.states.shape
        outer_hidden = torch.zeros(
            batch_size,
            self.hidden,
            device=trace.states.device,
            dtype=trace.states.dtype,
        )
        row_ids = torch.arange(batch_size, device=trace.states.device)
        for segment in range(trace.fragment_step_counts.shape[1]):
            starts = trace.fragment_step_counts[:, :segment].sum(dim=1)
            lengths = trace.fragment_step_counts[:, segment]
            active_segment = lengths > 0
            inner_hidden = torch.zeros_like(outer_hidden)
            for position in range(max_steps):
                active_step = (
                    active_segment
                    & (position >= starts)
                    & (position < starts + lengths)
                    & trace.mask[:, position]
                )
                token = self.step_input(
                    torch.cat(
                        (
                            trace.states[:, position],
                            trace.transition_deltas[:, position],
                            trace.instruction_codes[:, position],
                        ),
                        dim=-1,
                    )
                )
                proposal = self.step_cell(token, inner_hidden)
                inner_hidden = torch.where(
                    active_step.unsqueeze(-1), proposal, inner_hidden
                )
            final_positions = (starts + lengths - 1).clamp(0, max_steps - 1)
            final_states = trace.states[row_ids, final_positions]
            segment_token = self.segment_input(
                torch.cat((final_states, inner_hidden), dim=-1)
            )
            proposal = self.segment_cell(segment_token, outer_hidden)
            outer_hidden = torch.where(
                active_segment.unsqueeze(-1), proposal, outer_hidden
            )
        return self.output(outer_hidden)


class ExternalSkillFragmentOperatorCombiner(nn.Module):
    """Apply one reusable learned operator algebra across fragment segments.

    The combiner receives only rich learned execution evidence.  Each opaque
    segment is summarized, converted into a low-rank state transition, and
    applied to one persistent composition state.  The transition weights are
    shared across every segment and every composition depth; no depth slot,
    fragment index, route score, or verifier field enters the computation.
    This is an external replaceable codec, not a new controller branch.
    """

    schema = EXTERNAL_SKILL_FRAGMENT_OPERATOR_SCHEMA
    requires_instruction_codes = True

    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        output_width: int,
        *,
        hidden: int = 64,
        operator_rank: int = 8,
    ) -> None:
        super().__init__()
        if min(register_width, instruction_width, output_width, hidden, operator_rank) < 1:
            raise ValueError("operator combiner dimensions must be positive")
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.output_width = int(output_width)
        self.hidden = int(hidden)
        self.operator_rank = int(operator_rank)
        self.step_input = nn.Sequential(
            nn.Linear(register_width * 2 + instruction_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.step_cell = nn.GRUCell(hidden, hidden)
        self.segment_input = nn.Sequential(
            nn.Linear(register_width + hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.operator_input = nn.Sequential(
            nn.Linear(hidden + register_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.operator_left = nn.Linear(hidden, output_width * operator_rank)
        self.operator_right = nn.Linear(hidden, operator_rank * output_width)
        self.operator_bias = nn.Linear(hidden, output_width)
        self.operator_gate = nn.Linear(hidden, output_width)
        nn.init.constant_(self.operator_gate.bias, -2.0)
        self.readout = nn.Sequential(
            nn.Linear(output_width + register_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, output_width),
        )

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": self.schema,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "output_width": self.output_width,
            "hidden": self.hidden,
            "operator_rank": self.operator_rank,
            "input": "rich_segment_trace_only_v1",
            "composition": "shared_code_conditioned_low_rank_state_transition_v1",
            "uses_fragment_indices": False,
            "uses_verifier_metadata": False,
            "uses_depth_slots": False,
        }

    def forward(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
    ) -> torch.Tensor:
        if isinstance(trace, ExternalSkillFragmentExecutionTrace):
            trace = trace.learner_view()
        trace.validate(
            batch_size=trace.states.shape[0],
            register_width=self.register_width,
        )
        if (
            trace.instruction_codes is None
            or trace.transition_deltas is None
            or trace.fragment_step_counts is None
        ):
            raise ValueError("operator combiner requires a rich fragment trace")
        if trace.instruction_codes.shape[2] != self.instruction_width:
            raise ValueError("operator combiner instruction width does not match trace")
        batch_size, max_steps, _ = trace.states.shape
        segment_count = trace.fragment_step_counts.shape[1]
        outer_state = torch.zeros(
            batch_size,
            self.output_width,
            device=trace.states.device,
            dtype=trace.states.dtype,
        )
        last_state = torch.zeros(
            batch_size,
            self.register_width,
            device=trace.states.device,
            dtype=trace.states.dtype,
        )
        row_ids = torch.arange(batch_size, device=trace.states.device)
        for segment in range(segment_count):
            starts = trace.fragment_step_counts[:, :segment].sum(dim=1)
            lengths = trace.fragment_step_counts[:, segment]
            active_segment = lengths > 0
            inner_state = torch.zeros(
                batch_size,
                self.hidden,
                device=trace.states.device,
                dtype=trace.states.dtype,
            )
            for position in range(max_steps):
                active_step = (
                    active_segment
                    & (position >= starts)
                    & (position < starts + lengths)
                    & trace.mask[:, position]
                )
                token = self.step_input(
                    torch.cat(
                        (
                            trace.states[:, position],
                            trace.transition_deltas[:, position],
                            trace.instruction_codes[:, position],
                        ),
                        dim=-1,
                    )
                )
                proposal = self.step_cell(token, inner_state)
                inner_state = torch.where(
                    active_step.unsqueeze(-1), proposal, inner_state
                )
            final_positions = (starts + lengths - 1).clamp(0, max_steps - 1)
            final_segment_state = trace.states[row_ids, final_positions]
            segment_summary = self.segment_input(
                torch.cat((final_segment_state, inner_state), dim=-1)
            )
            operator_token = self.operator_input(
                torch.cat((segment_summary, final_segment_state), dim=-1)
            )
            left = torch.tanh(self.operator_left(operator_token)).reshape(
                batch_size, self.output_width, self.operator_rank
            )
            right = torch.tanh(self.operator_right(operator_token)).reshape(
                batch_size, self.operator_rank, self.output_width
            )
            projected = torch.einsum("bo,bro->br", outer_state, right)
            proposal = torch.einsum("br,bor->bo", projected, left)
            proposal = proposal + self.operator_bias(operator_token)
            gate = torch.sigmoid(self.operator_gate(operator_token))
            updated = outer_state + gate * torch.tanh(proposal)
            outer_state = torch.where(
                active_segment.unsqueeze(-1), updated, outer_state
            )
            last_state = torch.where(
                active_segment.unsqueeze(-1), final_segment_state, last_state
            )
        return self.readout(torch.cat((outer_state, last_state), dim=-1))

    def payload(self) -> dict[str, Any]:
        """Return a checksummed controller-independent external snapshot."""

        state = {
            "weights": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            }
        }
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": state,
            "sha256": self._snapshot_digest(state),
        }

    def _snapshot_digest(self, state: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self.configuration(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        weights = state.get("weights")
        if not isinstance(weights, Mapping):
            raise TypeError("operator combiner weights are malformed")
        for name, value in sorted(weights.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError("operator combiner weight is malformed")
            digest.update(name.encode("utf-8"))
            _digest_tensor(digest, value)
        return digest.hexdigest()

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> ExternalSkillFragmentOperatorCombiner:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported operator combiner payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("operator combiner payload is incomplete")
        combiner = cls(
            int(configuration.get("register_width", -1)),
            int(configuration.get("instruction_width", -1)),
            int(configuration.get("output_width", -1)),
            hidden=int(configuration.get("hidden", -1)),
            operator_rank=int(configuration.get("operator_rank", -1)),
        )
        weights = state.get("weights")
        if not isinstance(weights, Mapping) or set(weights) != set(combiner.state_dict()):
            raise ValueError("operator combiner state keys do not match configuration")
        combiner.load_state_dict(
            {name: value.to(device="cpu") for name, value in weights.items()},
            strict=True,
        )
        if dict(configuration) != combiner.configuration():
            raise ValueError("operator combiner configuration mismatch")
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != combiner._snapshot_digest(state):
            raise ValueError("operator combiner checksum mismatch")
        return combiner

    def save(self, path: Path) -> str:
        """Atomically persist the independent external operator memory."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": PERSISTENT_EXTERNAL_SKILL_FRAGMENT_OPERATOR_SCHEMA,
            "operator": self.payload(),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return str(payload["operator"]["sha256"])

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        map_location: torch.device | str = "cpu",
    ) -> ExternalSkillFragmentOperatorCombiner:
        """Reload operator memory only after schema and checksum validation."""

        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("format") != PERSISTENT_EXTERNAL_SKILL_FRAGMENT_OPERATOR_SCHEMA
        ):
            raise ValueError("unsupported persistent fragment operator format")
        operator_payload = payload.get("operator")
        if not isinstance(operator_payload, Mapping):
            raise TypeError("persistent fragment operator is missing its payload")
        return cls.from_payload(operator_payload)


class _ExternalSkillFragmentSerialStep(nn.Module):
    """One external state transition for one opaque fragment position."""

    def __init__(self, state_width: int, summary_width: int, hidden: int) -> None:
        super().__init__()
        self.state_width = int(state_width)
        self.summary_width = int(summary_width)
        self.hidden = int(hidden)
        self.input = nn.Sequential(
            nn.Linear(state_width + summary_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.proposal = nn.Linear(hidden, state_width)
        self.gate = nn.Linear(hidden, state_width)
        nn.init.constant_(self.gate.bias, -2.0)

    def zero_impact(self) -> None:
        """Make this transition an identity until a growth transaction trains it."""

        nn.init.zeros_(self.proposal.weight)
        nn.init.zeros_(self.proposal.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, state: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        hidden = self.input(torch.cat((state, summary), dim=-1))
        proposal = torch.tanh(self.proposal(hidden))
        gate = torch.sigmoid(self.gate(hidden))
        return state + gate * proposal


class ExternalSkillFragmentSerialCombiner(nn.Module):
    """Execute opaque fragment evidence through a protected serial state.

    The learner receives only the rich external execution trace. Each
    fragment boundary is summarized once, then one external step slot updates
    a persistent composition state. Slots are append-only and can be frozen as
    a prefix before later depths learn. This differs from the growth combiner:
    later slots consume the state produced by earlier slots instead of adding
    independent residuals only at the final readout.
    """

    schema = EXTERNAL_SKILL_FRAGMENT_SERIAL_SCHEMA
    requires_instruction_codes = True

    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        output_width: int,
        *,
        hidden: int = 64,
        step_sharing: str = "position",
    ) -> None:
        super().__init__()
        if min(register_width, instruction_width, output_width, hidden) < 1:
            raise ValueError("serial combiner dimensions must be positive")
        if step_sharing not in ("position", "shared"):
            raise ValueError("unsupported serial combiner step sharing")
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.output_width = int(output_width)
        self.hidden = int(hidden)
        self.step_sharing = step_sharing
        self.step_input = nn.Sequential(
            nn.Linear(register_width * 2 + instruction_width, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.step_cell = nn.GRUCell(hidden, hidden)
        self.segment_input = nn.Sequential(
            nn.Linear(register_width + hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.step_slots = nn.ModuleList()
        self._protected_step_count = 0
        self._base_protected = False

    @property
    def step_count(self) -> int:
        return len(self.step_slots)

    def append_step_slot(self) -> int:
        """Append one state transition, zero-impact after the first slot."""

        slot = _ExternalSkillFragmentSerialStep(
            self.output_width,
            self.hidden,
            self.hidden,
        )
        if self.step_count:
            slot.zero_impact()
        self.step_slots.append(slot)
        return self.step_count - 1

    def step_slot_parameters(self, index: int) -> tuple[nn.Parameter, ...]:
        if not 0 <= index < self.step_count:
            raise IndexError("serial combiner step slot is out of range")
        return tuple(self.step_slots[index].parameters())

    def protect_step_prefix(self, count: int | None = None) -> None:
        """Freeze an admitted prefix while leaving later slots trainable."""

        resolved = self.step_count if count is None else int(count)
        if not 0 <= resolved <= self.step_count:
            raise ValueError("serial combiner protection prefix is out of range")
        self._protected_step_count = resolved
        for slot in self.step_slots[:resolved]:
            for parameter in slot.parameters():
                parameter.requires_grad_(False)

    def protect_base(self) -> None:
        """Freeze the shared trace encoder for append-only growth."""

        self._base_protected = True
        for parameter in (
            *self.step_input.parameters(),
            *self.step_cell.parameters(),
            *self.segment_input.parameters(),
        ):
            parameter.requires_grad_(False)

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": self.schema,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "output_width": self.output_width,
            "hidden": self.hidden,
            "execution": "serial_segment_state_transition_v1",
            "growth": "append_only_protected_step_prefix_v1",
            "step_sharing": self.step_sharing,
            "step_slots": self.step_count,
            "protected_step_prefix": self._protected_step_count,
            "base_protected": self._base_protected,
            "uses_fragment_indices": False,
            "uses_verifier_metadata": False,
            "state_input": "previous_external_state_plus_segment_summary_v1",
            "prefix_state_snapshots": "forward_prefixes_v1",
            "causal_intervention": "leave_one_prefix_out_v1",
        }

    def _segment_summaries(
        self, trace: ExternalSkillFragmentLearnerTrace
    ) -> tuple[torch.Tensor, ...]:
        batch_size, max_steps, _ = trace.states.shape
        segment_count = trace.fragment_step_counts.shape[1]
        row_ids = torch.arange(batch_size, device=trace.states.device)
        summaries: list[torch.Tensor] = []
        for segment in range(segment_count):
            starts = trace.fragment_step_counts[:, :segment].sum(dim=1)
            lengths = trace.fragment_step_counts[:, segment]
            active_segment = lengths > 0
            inner_hidden = torch.zeros(
                batch_size,
                self.hidden,
                device=trace.states.device,
                dtype=trace.states.dtype,
            )
            for position in range(max_steps):
                active_step = (
                    active_segment
                    & (position >= starts)
                    & (position < starts + lengths)
                    & trace.mask[:, position]
                )
                token = self.step_input(
                    torch.cat(
                        (
                            trace.states[:, position],
                            trace.transition_deltas[:, position],
                            trace.instruction_codes[:, position],
                        ),
                        dim=-1,
                    )
                )
                proposal = self.step_cell(token, inner_hidden)
                inner_hidden = torch.where(
                    active_step.unsqueeze(-1), proposal, inner_hidden
                )
            final_positions = (starts + lengths - 1).clamp(0, max_steps - 1)
            final_states = trace.states[row_ids, final_positions]
            summary = self.segment_input(
                torch.cat((final_states, inner_hidden), dim=-1)
            )
            summaries.append(torch.where(
                active_segment.unsqueeze(-1),
                summary,
                torch.zeros_like(summary),
            ))
        return tuple(summaries)

    def _run_segment_summaries(
        self,
        summaries: tuple[torch.Tensor, ...],
        *,
        gate_logits: torch.Tensor | None = None,
        skip_index: int | None = None,
    ) -> torch.Tensor:
        """Return the external state after every executed segment."""

        if not summaries:
            raise ValueError("serial combiner requires at least one segment")
        required_slots = 1 if self.step_sharing == "shared" else len(summaries)
        if self.step_count < required_slots:
            raise ValueError("serial combiner has no slot for the requested depth")
        batch_size = summaries[0].shape[0]
        if gate_logits is not None:
            if gate_logits.shape != (batch_size, len(summaries)):
                raise ValueError(
                    "serial combiner gate logits must have shape [batch, segments]"
                )
            if not bool(torch.isfinite(gate_logits).all()):
                raise ValueError("serial combiner gate logits must be finite")
        if skip_index is not None and not 0 <= skip_index < len(summaries):
            raise ValueError("serial combiner skip index is out of range")
        state = torch.zeros(
            batch_size,
            self.output_width,
            device=summaries[0].device,
            dtype=summaries[0].dtype,
        )
        prefixes: list[torch.Tensor] = []
        if self.step_sharing == "shared":
            slot = self.step_slots[0]
            for index, summary in enumerate(summaries):
                proposal = slot(state, summary)
                if skip_index != index:
                    if gate_logits is None:
                        state = proposal
                    else:
                        gate = torch.sigmoid(gate_logits[:, index]).unsqueeze(-1)
                        state = state + gate * (proposal - state)
                prefixes.append(state)
        else:
            for index, summary in enumerate(summaries):
                proposal = self.step_slots[index](state, summary)
                if skip_index != index:
                    if gate_logits is None:
                        state = proposal
                    else:
                        gate = torch.sigmoid(gate_logits[:, index]).unsqueeze(-1)
                        state = state + gate * (proposal - state)
                prefixes.append(state)
        return torch.stack(prefixes, dim=1)

    def _validated_summaries(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
    ) -> tuple[torch.Tensor, ...]:
        if isinstance(trace, ExternalSkillFragmentExecutionTrace):
            trace = trace.learner_view()
        trace.validate(
            batch_size=trace.states.shape[0],
            register_width=self.register_width,
        )
        if (
            trace.instruction_codes is None
            or trace.transition_deltas is None
            or trace.fragment_step_counts is None
        ):
            raise ValueError("serial combiner requires a rich fragment trace")
        if trace.instruction_codes.shape[2] != self.instruction_width:
            raise ValueError("serial combiner instruction width does not match trace")
        return self._segment_summaries(trace)

    def forward_prefixes(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
    ) -> torch.Tensor:
        """Return one opaque external state snapshot per fragment boundary.

        The snapshots are a trainer-facing execution aid.  A verifier may
        score each snapshot on a fresh causal prefix task without exposing the
        prefix target, intervention, or verifier metadata to this module.
        ``forward`` remains the canonical final-state interface.
        """

        return self._run_segment_summaries(self._validated_summaries(trace))

    def forward_with_gates(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
        gate_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Execute the trace with an external, opaque transition-use policy."""

        return self._run_segment_summaries(
            self._validated_summaries(trace), gate_logits=gate_logits
        )[:, -1]

    def forward_leave_one_out(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
        gate_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return final states with each segment transition omitted in turn."""

        summaries = self._validated_summaries(trace)
        return torch.stack(
            tuple(
                self._run_segment_summaries(
                    summaries,
                    gate_logits=gate_logits,
                    skip_index=index,
                )[:, -1]
                for index in range(len(summaries))
            ),
            dim=1,
        )

    def forward(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
    ) -> torch.Tensor:
        return self.forward_prefixes(trace)[:, -1]

    def payload(self) -> dict[str, Any]:
        """Return a checksummed, controller-independent serial memory snapshot."""

        state = {
            "weights": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "protected_step_prefix": self._protected_step_count,
            "base_protected": self._base_protected,
        }
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": state,
            "sha256": self._snapshot_digest(state),
        }

    def _snapshot_digest(self, state: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self.configuration(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        weights = state.get("weights")
        if not isinstance(weights, Mapping):
            raise TypeError("serial combiner snapshot weights are malformed")
        for name, value in sorted(weights.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError("serial combiner snapshot weight is malformed")
            digest.update(name.encode("utf-8"))
            _digest_tensor(digest, value)
        for name in ("protected_step_prefix", "base_protected"):
            digest.update(name.encode("utf-8"))
            digest.update(
                json.dumps(
                    state.get(name), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
        return digest.hexdigest()

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> ExternalSkillFragmentSerialCombiner:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported serial combiner payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("serial combiner payload is incomplete")
        step_slots = int(configuration.get("step_slots", -1))
        combiner = cls(
            int(configuration.get("register_width", -1)),
            int(configuration.get("instruction_width", -1)),
            int(configuration.get("output_width", -1)),
            hidden=int(configuration.get("hidden", -1)),
            step_sharing=str(configuration.get("step_sharing", "position")),
        )
        for _ in range(step_slots):
            combiner.append_step_slot()
        weights = state.get("weights")
        if not isinstance(weights, Mapping):
            raise TypeError("serial combiner state weights are malformed")
        if set(weights) != set(combiner.state_dict()):
            raise ValueError("serial combiner state keys do not match configuration")
        combiner.load_state_dict(
            {name: value.to(device="cpu") for name, value in weights.items()},
            strict=True,
        )
        protected = int(state.get("protected_step_prefix", -1))
        if not 0 <= protected <= combiner.step_count:
            raise ValueError("serial combiner protection prefix is invalid")
        combiner._protected_step_count = protected
        combiner._base_protected = bool(state.get("base_protected", False))
        combiner.protect_step_prefix(protected)
        if combiner._base_protected:
            combiner.protect_base()
        if dict(configuration) != combiner.configuration():
            raise ValueError("serial combiner configuration mismatch")
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != combiner._snapshot_digest(state):
            raise ValueError("serial combiner checksum mismatch")
        return combiner

    def save(self, path: Path) -> str:
        """Atomically persist the external serial execution state."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": PERSISTENT_EXTERNAL_SKILL_FRAGMENT_SERIAL_SCHEMA,
            "serial": self.payload(),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return str(payload["serial"]["sha256"])

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        map_location: torch.device | str = "cpu",
    ) -> ExternalSkillFragmentSerialCombiner:
        """Reload serial execution memory only after schema validation."""

        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("format") != PERSISTENT_EXTERNAL_SKILL_FRAGMENT_SERIAL_SCHEMA
        ):
            raise ValueError("unsupported persistent serial combiner format")
        serial_payload = payload.get("serial")
        if not isinstance(serial_payload, Mapping):
            raise TypeError("persistent serial combiner is missing its payload")
        return cls.from_payload(serial_payload)


class ExternalSkillFragmentGrowthCombiner(nn.Module):
    """Append-only learner for continual composition-depth growth.

    The shared segment encoder and canonical output projection form a stable
    base.  Each newly mastered fragment depth receives one zero-initialized,
    trace-conditioned external residual slot.  A trainer can freeze the base
    and all earlier slots, then train only the newly appended slot on fresh
    outcomes.  Deeper programs reuse the protected prefix cumulatively, so
    an admitted composition prior remains useful instead of being discarded
    at the next structural depth.  This gives the frozen-memory protocol a
    real no-replay growth seam without creating a target- or modality-specific
    reasoning branch.
    """

    schema = EXTERNAL_SKILL_FRAGMENT_GROWTH_SCHEMA
    requires_instruction_codes = True

    def __init__(
        self,
        register_width: int,
        instruction_width: int,
        output_width: int,
        *,
        hidden: int = 64,
        slot_application: str = "cumulative_prefix",
    ) -> None:
        super().__init__()
        if min(register_width, instruction_width, output_width, hidden) < 1:
            raise ValueError("growth combiner dimensions must be positive")
        if slot_application not in ("cumulative_prefix", "exact_depth"):
            raise ValueError("unsupported growth slot application")
        self.register_width = int(register_width)
        self.instruction_width = int(instruction_width)
        self.output_width = int(output_width)
        self.hidden = int(hidden)
        self.slot_application = slot_application
        self.base = ExternalSkillFragmentSegmentCombiner(
            register_width,
            instruction_width,
            hidden,
            hidden=hidden,
        )
        self.base_output = nn.Linear(hidden, output_width)
        self.depth_slots = nn.ModuleList()
        self._protected_depth_count = 0
        self._base_protected = False

    @property
    def depth_count(self) -> int:
        return len(self.depth_slots)

    def append_depth_slot(self) -> int:
        """Append a zero-impact residual slot for the next fragment depth."""

        slot = ExternalSkillFragmentSegmentCombiner(
            self.register_width,
            self.instruction_width,
            self.output_width,
            hidden=self.hidden,
        )
        nn.init.zeros_(slot.output.weight)
        nn.init.zeros_(slot.output.bias)
        self.depth_slots.append(slot)
        return self.depth_count - 1

    def depth_slot_parameters(self, index: int) -> tuple[nn.Parameter, ...]:
        if not 0 <= index < self.depth_count:
            raise IndexError("growth combiner depth slot is out of range")
        return tuple(self.depth_slots[index].parameters())

    def protect_depth_prefix(self, count: int | None = None) -> None:
        """Freeze an admitted prefix of external depth slots."""

        resolved = self.depth_count if count is None else int(count)
        if not 0 <= resolved <= self.depth_count:
            raise ValueError("growth combiner protection prefix is out of range")
        self._protected_depth_count = resolved
        for slot in self.depth_slots[:resolved]:
            for parameter in slot.parameters():
                parameter.requires_grad_(False)

    def protect_base(self) -> None:
        """Freeze the shared representation while leaving future slots open."""

        self._base_protected = True
        for parameter in (*self.base.parameters(), *self.base_output.parameters()):
            parameter.requires_grad_(False)

    def configuration(self) -> dict[str, int | str | bool]:
        return {
            "schema": self.schema,
            "register_width": self.register_width,
            "instruction_width": self.instruction_width,
            "output_width": self.output_width,
            "hidden": self.hidden,
            "base": "shared_segment_encoder_plus_canonical_readout_v1",
            "growth": "append_only_trace_conditioned_depth_prefix_slots_v3",
            "slot_application": self.slot_application,
            "depth_slots": self.depth_count,
            "protected_depth_prefix": self._protected_depth_count,
            "base_protected": self._base_protected,
            "uses_fragment_indices": False,
            "uses_verifier_metadata": False,
            "depth_signal": "opaque_segment_count_only_v1",
        }

    def forward(
        self,
        trace: ExternalSkillFragmentExecutionTrace | ExternalSkillFragmentLearnerTrace,
    ) -> torch.Tensor:
        if isinstance(trace, ExternalSkillFragmentExecutionTrace):
            trace = trace.learner_view()
        trace.validate(
            batch_size=trace.states.shape[0],
            register_width=self.register_width,
        )
        if (
            trace.instruction_codes is None
            or trace.transition_deltas is None
            or trace.fragment_step_counts is None
        ):
            raise ValueError("growth combiner requires a rich fragment trace")
        if trace.instruction_codes.shape[2] != self.instruction_width:
            raise ValueError("growth combiner instruction width does not match trace")
        depths = (trace.fragment_step_counts > 0).sum(dim=1)
        if self.depth_count < int(depths.max().item()):
            raise ValueError("growth combiner has no slot for the requested depth")
        base_hidden = self.base(trace)
        output = self.base_output(base_hidden)
        for index, slot in enumerate(self.depth_slots):
            active = (
                depths >= index + 1
                if self.slot_application == "cumulative_prefix"
                else depths == index + 1
            )
            if bool(active.any()):
                residual = slot(trace)
                output = output + torch.where(
                    active.unsqueeze(-1), residual, torch.zeros_like(residual)
                )
        return output

    def payload(self) -> dict[str, Any]:
        """Return a checksummed, controller-independent memory snapshot."""

        state = {
            "weights": {
                name: value.detach().cpu().clone()
                for name, value in self.state_dict().items()
            },
            "protected_depth_prefix": self._protected_depth_count,
            "base_protected": self._base_protected,
        }
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": state,
            "sha256": self._snapshot_digest(state),
        }

    def _snapshot_digest(self, state: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self.configuration(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        weights = state.get("weights")
        if not isinstance(weights, Mapping):
            raise TypeError("growth combiner snapshot weights are malformed")
        for name, value in sorted(weights.items()):
            if not isinstance(value, torch.Tensor):
                raise TypeError("growth combiner snapshot weight is malformed")
            digest.update(name.encode("utf-8"))
            _digest_tensor(digest, value)
        for name in ("protected_depth_prefix", "base_protected"):
            digest.update(name.encode("utf-8"))
            digest.update(
                json.dumps(
                    state.get(name), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
        return digest.hexdigest()

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any]
    ) -> ExternalSkillFragmentGrowthCombiner:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported fragment growth payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("fragment growth payload is incomplete")
        depth_slots = int(configuration.get("depth_slots", -1))
        combiner = cls(
            int(configuration.get("register_width", -1)),
            int(configuration.get("instruction_width", -1)),
            int(configuration.get("output_width", -1)),
            hidden=int(configuration.get("hidden", -1)),
            slot_application=str(
                configuration.get("slot_application", "exact_depth")
            ),
        )
        for _ in range(depth_slots):
            combiner.append_depth_slot()
        weights = state.get("weights")
        if not isinstance(weights, Mapping):
            raise TypeError("fragment growth state weights are malformed")
        expected_names = set(combiner.state_dict())
        if set(weights) != expected_names:
            raise ValueError("fragment growth state keys do not match configuration")
        combiner.load_state_dict(
            {name: value.to(device="cpu") for name, value in weights.items()},
            strict=True,
        )
        protected = int(state.get("protected_depth_prefix", -1))
        if not 0 <= protected <= combiner.depth_count:
            raise ValueError("fragment growth protection prefix is invalid")
        combiner._protected_depth_count = protected
        combiner._base_protected = bool(state.get("base_protected", False))
        combiner.protect_depth_prefix(protected)
        if combiner._base_protected:
            combiner.protect_base()
        if dict(configuration) != combiner.configuration():
            raise ValueError("fragment growth configuration mismatch")
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != combiner._snapshot_digest(
            state
        ):
            raise ValueError("fragment growth checksum mismatch")
        return combiner

    def save(self, path: Path) -> str:
        """Atomically persist the external growth memory."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": PERSISTENT_EXTERNAL_SKILL_FRAGMENT_GROWTH_SCHEMA,
            "growth": self.payload(),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return str(payload["growth"]["sha256"])

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        map_location: torch.device | str = "cpu",
    ) -> ExternalSkillFragmentGrowthCombiner:
        """Reload growth memory only after schema and checksum validation."""

        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("format") != PERSISTENT_EXTERNAL_SKILL_FRAGMENT_GROWTH_SCHEMA
        ):
            raise ValueError("unsupported persistent fragment growth format")
        growth_payload = payload.get("growth")
        if not isinstance(growth_payload, Mapping):
            raise TypeError("persistent fragment growth is missing its payload")
        return cls.from_payload(growth_payload)


class ExternalSkillFragmentBank(nn.Module):
    """Growing bank of reusable fragments over one shared operator basis.

    The bank is intentionally separate from the controller.  Appending a
    fragment adds coefficient/key data only; the shared basis, router ABI,
    controller, and output interfaces retain their shapes.  Routing is
    content-addressed by default and can be refined by an outcome-trained,
    permutation-equivariant residual scorer after evidence enables it.
    """

    schema = EXTERNAL_SKILL_FRAGMENT_BANK_SCHEMA

    def __init__(
        self,
        instruction_width: int,
        basis_count: int,
        *,
        key_width: int | None = None,
        router_hidden: int = 64,
        max_fragment_steps: int = 16,
        code_norm: float = 1.0,
    ) -> None:
        super().__init__()
        if min(instruction_width, basis_count, router_hidden, max_fragment_steps) < 1:
            raise ValueError("fragment bank dimensions must be positive")
        if code_norm <= 0.0:
            raise ValueError("fragment code norm must be positive")
        self.instruction_width = int(instruction_width)
        self.basis_count = int(basis_count)
        self.key_width = int(instruction_width if key_width is None else key_width)
        self.router_hidden = int(router_hidden)
        self.max_fragment_steps = int(max_fragment_steps)
        self.code_norm = float(code_norm)
        self.shared_basis = nn.Parameter(
            torch.randn(self.basis_count, self.instruction_width) * 0.02
        )
        self.router = OpaqueCandidateGrowthRouter(self.key_width, hidden=router_hidden)
        self.coefficients = nn.ParameterList()
        self.keys = nn.ParameterList()
        self._parent_digests: list[tuple[str, ...]] = []
        self._protected: list[bool] = []
        self._logical_ids: list[int] = []
        self._next_logical_id = 0
        self._frozen_basis_rows = 0
        self._basis_freeze_hook: Any | None = None
        self.register_buffer(
            "learned_routing_enabled", torch.tensor(False, dtype=torch.bool)
        )

    def configuration(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "instruction_width": self.instruction_width,
            "basis_count": self.basis_count,
            "key_width": self.key_width,
            "router_hidden": self.router_hidden,
            "max_fragment_steps": self.max_fragment_steps,
            "code_norm": self.code_norm,
            "fragment_count": len(self.coefficients),
            "unit": "opaque_reusable_shared_basis_fragment_v1",
            "basis": "append_expandable_shared_operator_basis_v1",
            "materialization": "normalized_shared_basis_code_v1",
            "composition": "opaque_route_then_serial_fragment_chain_v1",
            "routing": "cosine_address_plus_outcome_trained_equivariant_residual_v1",
            "basis_growth": "append_only_protected_shared_basis_v1",
            "storage": "append_only_external_state_v1",
        }

    @property
    def fragment_count(self) -> int:
        return len(self.coefficients)

    @property
    def logical_ids(self) -> tuple[int, ...]:
        return tuple(self._logical_ids)

    def _validate_fragment(self, artifact: ExternalSkillFragmentArtifact) -> None:
        if (
            artifact.basis_count != self.basis_count
            or artifact.key_width != self.key_width
        ):
            raise ValueError("fragment dimensions do not match the bank")
        if artifact.step_count > self.max_fragment_steps:
            raise ValueError("fragment exceeds the bank's maximum step count")

    def add_fragment(
        self,
        coefficients: torch.Tensor,
        key: torch.Tensor,
        *,
        parent_digests: Sequence[str] = (),
    ) -> int:
        """Append one reusable fragment without changing shared dimensions."""

        artifact = ExternalSkillFragmentArtifact(
            coefficients=coefficients.detach().clone(),
            key=key.detach().clone(),
            parent_digests=tuple(parent_digests),
        )
        self._validate_fragment(artifact)
        self.coefficients.append(nn.Parameter(artifact.coefficients.clone()))
        self.keys.append(nn.Parameter(artifact.key.clone()))
        self._parent_digests.append(artifact.parent_digests)
        self._protected.append(False)
        self._logical_ids.append(self._next_logical_id)
        self._next_logical_id += 1
        return self.fragment_count - 1

    def add_artifact(self, artifact: ExternalSkillFragmentArtifact) -> int:
        if not isinstance(artifact, ExternalSkillFragmentArtifact):
            raise TypeError("fragment bank requires a skill fragment artifact")
        self._validate_fragment(artifact)
        return self.add_fragment(
            artifact.coefficients,
            artifact.key,
            parent_digests=artifact.parent_digests,
        )

    def artifact(self, index: int) -> ExternalSkillFragmentArtifact:
        if not 0 <= index < self.fragment_count:
            raise IndexError("fragment index is outside the bank")
        return ExternalSkillFragmentArtifact(
            coefficients=self.coefficients[index].detach(),
            key=self.keys[index].detach(),
            parent_digests=self._parent_digests[index],
        )

    def fragment_codes(self, index: int) -> torch.Tensor:
        """Materialize one fragment's shared-basis instruction sequence."""

        if not 0 <= index < self.fragment_count:
            raise IndexError("fragment index is outside the bank")
        raw = self.coefficients[index] @ self.shared_basis
        nonzero = raw.norm(dim=-1, keepdim=True) > 1e-8
        normalized = torch.nn.functional.normalize(raw, dim=-1, eps=1e-8)
        return torch.where(nonzero, normalized * self.code_norm, raw)

    def _all_keys(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.fragment_count:
            raise ValueError("fragment bank has no fragments")
        return torch.stack(
            tuple(key.to(device=device, dtype=dtype) for key in self.keys)
        )

    def _install_basis_freeze_hook(self) -> None:
        if self._basis_freeze_hook is not None:
            self._basis_freeze_hook.remove()
            self._basis_freeze_hook = None
        if self._frozen_basis_rows and self.shared_basis.requires_grad:
            frozen = self._frozen_basis_rows

            def mask_gradient(gradient: torch.Tensor) -> torch.Tensor:
                masked = gradient.clone()
                masked[:frozen] = 0.0
                return masked

            self._basis_freeze_hook = self.shared_basis.register_hook(mask_gradient)

    def grow_basis(self, additional_count: int = 1) -> tuple[int, ...]:
        """Append shared computation directions without resizing the controller.

        Existing coefficient rows receive zero padding, so every old fragment
        materializes the same code immediately after growth.  New basis rows
        can be trained by a later candidate while :meth:`freeze_basis_prefix`
        protects the directions already used by mastered fragments.
        """

        if additional_count < 1:
            raise ValueError("basis growth must append at least one direction")
        start = self.basis_count
        proposal = (
            torch.randn(
                additional_count,
                self.instruction_width,
                device=self.shared_basis.device,
                dtype=self.shared_basis.dtype,
            )
            * 0.02
        )
        old_requires_grad = self.shared_basis.requires_grad
        if self._basis_freeze_hook is not None:
            self._basis_freeze_hook.remove()
            self._basis_freeze_hook = None
        self.shared_basis = nn.Parameter(
            torch.cat((self.shared_basis.detach(), proposal), dim=0),
            requires_grad=old_requires_grad,
        )
        for index, coefficient in enumerate(tuple(self.coefficients)):
            padded = torch.nn.functional.pad(
                coefficient.detach(), (0, additional_count)
            )
            self.coefficients[index] = nn.Parameter(
                padded,
                requires_grad=coefficient.requires_grad,
            )
        self.basis_count += additional_count
        self._install_basis_freeze_hook()
        return tuple(range(start, self.basis_count))

    def freeze_basis_prefix(self, row_count: int) -> None:
        """Protect an acquired basis prefix from later external updates."""

        if not 0 <= row_count <= self.basis_count:
            raise ValueError("frozen basis prefix is outside the bank")
        self._frozen_basis_rows = int(row_count)
        self._install_basis_freeze_hook()

    def route_scores(self, query: torch.Tensor) -> torch.Tensor:
        """Return one opaque score per fragment row."""

        if query.ndim != 2 or query.shape[1] != self.key_width:
            raise ValueError(
                f"fragment query must have shape [batch, {self.key_width}]"
            )
        if not bool(torch.isfinite(query).all()):
            raise ValueError("fragment query must be finite")
        keys = self._all_keys(device=query.device, dtype=query.dtype)
        base = torch.nn.functional.cosine_similarity(
            query.unsqueeze(1), keys.unsqueeze(0), dim=-1
        )
        if not bool(self.learned_routing_enabled.item()):
            return base
        return base + self.router(query, keys)

    def route(
        self, query: torch.Tensor, *, top_k: int = 1
    ) -> ExternalSkillFragmentRoute:
        """Address fragments with a row-permutation-equivariant learned query."""

        if top_k < 1 or top_k > self.fragment_count:
            raise ValueError("fragment route top_k is outside the bank")
        scores = self.route_scores(query)
        values, indices = torch.topk(scores, top_k, dim=-1, largest=True, sorted=True)
        weights = torch.softmax(values, dim=-1)
        return ExternalSkillFragmentRoute(
            indices=indices.to(dtype=torch.int64),
            scores=values,
            weights=weights,
        ).validate(batch_size=query.shape[0], fragment_count=self.fragment_count)

    def enable_learned_routing(self) -> None:
        """Enable the outcome-trained residual after fresh evidence exists."""

        self.learned_routing_enabled.fill_(True)

    def outcome_ranking_loss(
        self,
        query: torch.Tensor,
        outcomes: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Train route refinement from paired attempted scalar outcomes only."""

        scores = self.route_scores(query)
        if outcomes.ndim == 1:
            outcomes = outcomes.unsqueeze(0)
        if outcomes.shape != scores.shape:
            raise ValueError("fragment outcomes must align with route scores")
        if not bool(torch.isfinite(outcomes).all()) or not bool(
            ((outcomes >= 0.0) & (outcomes <= 1.0)).all()
        ):
            raise ValueError("fragment outcomes must lie in [0, 1]")
        outcome_delta = outcomes.unsqueeze(2) - outcomes.unsqueeze(1)
        score_delta = scores.unsqueeze(2) - scores.unsqueeze(1)
        informative = outcome_delta > 0.0
        count = int(informative.sum().detach().cpu().item())
        if count == 0:
            return scores.sum() * 0.0, 0
        loss = torch.nn.functional.softplus(
            -outcome_delta[informative].detach() * score_delta[informative]
        ).mean()
        return loss, count

    def compose_indices(
        self, fragment_indices: torch.Tensor
    ) -> ExternalSkillFragmentComposition:
        """Materialize a serial chain from opaque row positions.

        Row positions are memory-side bookkeeping.  Deployed callers should
        normally obtain them through :meth:`compose_queries`, not provide a
        task-labelled slot.  Each row is independently padded so fragments of
        different lengths remain separately bindable.
        """

        if fragment_indices.ndim != 2 or fragment_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("fragment indices must have shape [batch, steps]")
        if (
            fragment_indices.shape[1] < 1
            or bool((fragment_indices < 0).any())
            or bool((fragment_indices >= self.fragment_count).any())
        ):
            raise ValueError("fragment index is outside the bank")
        rows: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        step_counts: list[torch.Tensor] = []
        for selected in fragment_indices.to(dtype=torch.int64):
            pieces = [self.fragment_codes(int(index)) for index in selected.tolist()]
            row = torch.cat(pieces, dim=0)
            rows.append(row)
            masks.append(torch.ones(row.shape[0], dtype=torch.bool, device=row.device))
            step_counts.append(
                torch.tensor(
                    [piece.shape[0] for piece in pieces],
                    dtype=torch.int64,
                    device=row.device,
                )
            )
        codes = pad_sequence(rows, batch_first=True)
        mask = pad_sequence(masks, batch_first=True, padding_value=False)
        route_scores = torch.zeros(
            fragment_indices.shape,
            dtype=codes.dtype,
            device=codes.device,
        )
        return ExternalSkillFragmentComposition(
            fragment_indices=fragment_indices.to(
                device=codes.device, dtype=torch.int64
            ),
            route_scores=route_scores,
            codes=codes,
            mask=mask,
            bank_fragment_count=self.fragment_count,
            fragment_step_counts=torch.stack(step_counts),
        ).validate(
            batch_size=fragment_indices.shape[0],
            instruction_width=self.instruction_width,
            fragment_count=self.fragment_count,
        )

    def compose_queries(
        self, queries: torch.Tensor
    ) -> ExternalSkillFragmentComposition:
        """Route one opaque query per serial composition step and execute later."""

        if queries.ndim != 3 or queries.shape[2] != self.key_width:
            raise ValueError(
                f"fragment queries must have shape [batch, steps, {self.key_width}]"
            )
        route_indices: list[torch.Tensor] = []
        route_scores: list[torch.Tensor] = []
        for step in range(queries.shape[1]):
            result = self.route(queries[:, step], top_k=1)
            route_indices.append(result.indices[:, 0])
            route_scores.append(result.scores[:, 0])
        indices = torch.stack(route_indices, dim=1)
        composition = self.compose_indices(indices)
        return ExternalSkillFragmentComposition(
            fragment_indices=composition.fragment_indices,
            route_scores=torch.stack(route_scores, dim=1),
            codes=composition.codes,
            mask=composition.mask,
            bank_fragment_count=self.fragment_count,
            fragment_step_counts=composition.fragment_step_counts,
        ).validate(
            batch_size=queries.shape[0],
            instruction_width=self.instruction_width,
            fragment_count=self.fragment_count,
        )

    def protect(self, index: int) -> None:
        if not 0 <= index < self.fragment_count:
            raise IndexError("fragment index is outside the bank")
        self._protected[index] = True

    def protection_mask(self) -> torch.Tensor:
        return torch.tensor(tuple(self._protected), dtype=torch.bool)

    def payload(self) -> dict[str, Any]:
        """Return a checksummed external-memory snapshot."""

        state = {
            "shared_basis": self.shared_basis.detach().cpu().clone(),
            "router": {
                name: value.detach().cpu().clone()
                for name, value in self.router.state_dict().items()
            },
            "fragments": [
                self.artifact(index).payload() for index in range(self.fragment_count)
            ],
            "protected": list(self._protected),
            "logical_ids": list(self._logical_ids),
            "next_logical_id": self._next_logical_id,
            "frozen_basis_rows": self._frozen_basis_rows,
            "learned_routing_enabled": bool(self.learned_routing_enabled.item()),
        }
        return {
            "schema": self.schema,
            "configuration": self.configuration(),
            "state": state,
            "sha256": self._snapshot_digest(state),
        }

    def _snapshot_digest(self, state: Mapping[str, Any]) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self.configuration(), sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        )
        shared_basis = state.get("shared_basis")
        router_state = state.get("router")
        fragments = state.get("fragments")
        if not isinstance(shared_basis, torch.Tensor) or not isinstance(
            router_state, Mapping
        ):
            raise TypeError("fragment bank snapshot state is malformed")
        if not isinstance(fragments, list):
            raise TypeError("fragment bank snapshot fragments are malformed")
        _digest_tensor(digest, shared_basis)
        for name, value in sorted(router_state.items()):
            digest.update(name.encode("utf-8"))
            _digest_tensor(digest, value)
        for fragment in fragments:
            digest.update(str(fragment["sha256"]).encode("ascii"))
        for name in (
            "protected",
            "logical_ids",
            "next_logical_id",
            "frozen_basis_rows",
            "learned_routing_enabled",
        ):
            digest.update(name.encode("utf-8"))
            digest.update(
                json.dumps(
                    state.get(name), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
        return digest.hexdigest()

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ExternalSkillFragmentBank:
        if not isinstance(payload, Mapping) or payload.get("schema") != cls.schema:
            raise ValueError("unsupported fragment bank payload")
        configuration = payload.get("configuration")
        state = payload.get("state")
        if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
            raise TypeError("fragment bank payload is incomplete")
        basis = state.get("shared_basis")
        router_state = state.get("router")
        fragments = state.get("fragments")
        if (
            not isinstance(basis, torch.Tensor)
            or not isinstance(router_state, Mapping)
            or not isinstance(fragments, list)
        ):
            raise TypeError("fragment bank state is malformed")
        bank = cls(
            int(configuration.get("instruction_width", -1)),
            int(configuration.get("basis_count", -1)),
            key_width=int(configuration.get("key_width", -1)),
            router_hidden=int(configuration.get("router_hidden", -1)),
            max_fragment_steps=int(configuration.get("max_fragment_steps", -1)),
            code_norm=float(configuration.get("code_norm", 1.0)),
        )
        if basis.shape != bank.shared_basis.shape:
            raise ValueError("fragment shared basis shape mismatch")
        bank.shared_basis.data.copy_(basis.to(bank.shared_basis))
        bank.router.load_state_dict(
            {name: value.to(device="cpu") for name, value in router_state.items()},
            strict=True,
        )
        for fragment_payload in fragments:
            bank.add_artifact(
                ExternalSkillFragmentArtifact.from_payload(fragment_payload)
            )
        protected = state.get("protected", [])
        logical_ids = state.get("logical_ids", [])
        if (
            len(protected) != bank.fragment_count
            or len(logical_ids) != bank.fragment_count
        ):
            raise ValueError("fragment bank lifecycle state does not align")
        bank._protected = [bool(value) for value in protected]
        bank._logical_ids = [int(value) for value in logical_ids]
        bank._next_logical_id = int(state.get("next_logical_id", bank.fragment_count))
        bank._frozen_basis_rows = int(state.get("frozen_basis_rows", 0))
        if not 0 <= bank._frozen_basis_rows <= bank.basis_count:
            raise ValueError("fragment bank frozen basis prefix is invalid")
        bank._install_basis_freeze_hook()
        bank.learned_routing_enabled.fill_(
            bool(state.get("learned_routing_enabled", False))
        )
        expected = payload.get("sha256")
        if not isinstance(expected, str) or expected != bank._snapshot_digest(state):
            raise ValueError("fragment bank checksum mismatch")
        return bank

    def save(self, path: Path) -> str:
        """Atomically persist the complete external bank and return its digest."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": PERSISTENT_EXTERNAL_SKILL_FRAGMENT_BANK_SCHEMA,
            "bank": self.payload(),
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        return str(payload["bank"]["sha256"])

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        map_location: torch.device | str = "cpu",
    ) -> ExternalSkillFragmentBank:
        """Reload a bank only after validating its schema and full digest."""

        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        if (
            not isinstance(payload, Mapping)
            or payload.get("format") != PERSISTENT_EXTERNAL_SKILL_FRAGMENT_BANK_SCHEMA
        ):
            raise ValueError("unsupported persistent fragment bank format")
        bank_payload = payload.get("bank")
        if not isinstance(bank_payload, Mapping):
            raise TypeError("persistent fragment bank is missing its bank payload")
        return cls.from_payload(bank_payload)


__all__ = [
    "EXTERNAL_SKILL_FRAGMENT_BANK_SCHEMA",
    "EXTERNAL_SKILL_FRAGMENT_COMPOSITION_SCHEMA",
    "EXTERNAL_SKILL_FRAGMENT_GROWTH_SCHEMA",
    "EXTERNAL_SKILL_FRAGMENT_OPERATOR_SCHEMA",
    "EXTERNAL_SKILL_FRAGMENT_RICH_TRACE_SCHEMA",
    "EXTERNAL_SKILL_FRAGMENT_ROUTE_SCHEMA",
    "EXTERNAL_SKILL_FRAGMENT_SCHEMA",
    "EXTERNAL_SKILL_FRAGMENT_SERIAL_SCHEMA",
    "PERSISTENT_EXTERNAL_SKILL_FRAGMENT_BANK_SCHEMA",
    "PERSISTENT_EXTERNAL_SKILL_FRAGMENT_GROWTH_SCHEMA",
    "PERSISTENT_EXTERNAL_SKILL_FRAGMENT_OPERATOR_SCHEMA",
    "PERSISTENT_EXTERNAL_SKILL_FRAGMENT_SERIAL_SCHEMA",
    "ExternalSkillFragmentArtifact",
    "ExternalSkillFragmentBank",
    "ExternalSkillFragmentCombiner",
    "ExternalSkillFragmentComposition",
    "ExternalSkillFragmentExecutionTrace",
    "ExternalSkillFragmentGrowthCombiner",
    "ExternalSkillFragmentLearnerTrace",
    "ExternalSkillFragmentProgramCombiner",
    "ExternalSkillFragmentRoute",
    "ExternalSkillFragmentSegmentCombiner",
    "ExternalSkillFragmentSerialCombiner",
]
