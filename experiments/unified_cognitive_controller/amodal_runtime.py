"""Behavior-preserving extracted runtime for the amodal migration rung."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .amodal_interface import (
    INTENTION_SCHEMA,
    NEURAL_IR_SCHEMA,
    AmodalEvent,
    AmodalEventCollection,
    IntentEvent,
)
from .environment import ACTIONS
from .model import (
    ControllerCoreOutput,
    ControllerOutput,
    ControllerState,
    UnifiedCognitiveController,
    VisionEventEncoder,
)

EXTRACTED_CHECKPOINT_FORMAT = "neural-computer.extracted-amodal-runtime.v1"
CANONICAL_INTENTION_MIGRATION = "neural-computer.action-residual-to-intention.v1"


class ActionIntentDecoder(nn.Module):
    """External decoder reproducing the inherited opaque action protocol.

    Migration-v1 intentions append ``ACTIONS`` compatibility coordinates to
    the inherited intention. This decoder is the only component that knows how
    those coordinates affect the old protocol.
    """

    def __init__(self, projection: nn.Linear, *, intention_width: int) -> None:
        super().__init__()
        if projection.in_features != intention_width:
            raise ValueError("decoder projection and intention width disagree")
        if projection.out_features != ACTIONS:
            raise ValueError("decoder projection and action count disagree")
        self.projection = projection
        self.intention_width = intention_width

    @property
    def payload_width(self) -> int:
        return self.intention_width + ACTIONS

    def forward(self, intention: IntentEvent | torch.Tensor) -> torch.Tensor:
        payload = intention.payload if isinstance(intention, IntentEvent) else intention
        if isinstance(intention, IntentEvent):
            intention.validate(width=self.payload_width)
        elif payload.ndim != 2 or payload.shape[1] != self.payload_width:
            raise ValueError(f"intention must have shape [batch, {self.payload_width}]")
        return (
            self.projection(payload[:, : self.intention_width])
            + payload[:, self.intention_width :]
        )


class OpaqueProtocolDecoder(nn.Module):
    """Independent backend from base intention to opaque protocol commands."""

    def __init__(
        self, intention_width: int, commands: int = ACTIONS, hidden: int = 0
    ) -> None:
        super().__init__()
        if intention_width < 1 or commands < 2 or hidden < 0:
            raise ValueError("protocol decoder dimensions are invalid")
        self.intention_width = intention_width
        self.commands = commands
        self.network = (
            nn.Sequential(
                nn.Linear(intention_width, hidden),
                nn.GELU(),
                nn.Linear(hidden, commands),
            )
            if hidden
            else nn.Linear(intention_width, commands)
        )

    def forward(self, intention: IntentEvent | torch.Tensor) -> torch.Tensor:
        payload = intention.payload if isinstance(intention, IntentEvent) else intention
        if payload.ndim != 2 or payload.shape[1] < self.intention_width:
            raise ValueError("intention payload is too narrow for protocol decoder")
        return self.network(payload[:, : self.intention_width])


class AmodalOutputBus(nn.Module):
    """Fan one intention out to a runtime-variable set of output backends."""

    def __init__(self, decoders: Mapping[str, nn.Module] | None = None) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict(dict(decoders or {}))

    def register_decoder(self, name: str, decoder: nn.Module) -> None:
        if not name or name in self.decoders:
            raise ValueError("decoder name must be nonempty and unique")
        self.decoders[name] = decoder

    def forward(self, intention: IntentEvent) -> dict[str, torch.Tensor]:
        return {name: decoder(intention) for name, decoder in self.decoders.items()}


class AmodalInputBus(nn.Module):
    """Combine a runtime-variable event set with generic learned attention."""

    def __init__(self, event_width: int, residual_hidden: int = 0) -> None:
        super().__init__()
        if event_width < 1 or residual_hidden < 0:
            raise ValueError("input-bus dimensions are invalid")
        self.event_width = event_width
        self.relevance = nn.Linear(event_width, 1)
        # Uniform attention is a behavior-preserving starting point. Learning
        # may later prefer useful events using only verified behavioral reward.
        nn.init.zeros_(self.relevance.weight)
        nn.init.zeros_(self.relevance.bias)
        self.residual = (
            nn.Sequential(
                nn.Linear(event_width * 2, residual_hidden),
                nn.GELU(),
                nn.Linear(residual_hidden, event_width),
            )
            if residual_hidden
            else None
        )
        if self.residual is not None:
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def forward(
        self, collection: AmodalEventCollection | Sequence[AmodalEvent]
    ) -> AmodalEvent:
        if not isinstance(collection, AmodalEventCollection):
            collection = AmodalEventCollection.from_events(collection)
        collection.validate(width=self.event_width)
        confidence = collection.confidence.to(collection.payload.dtype)
        scores = self.relevance(collection.payload).squeeze(-1)
        scores = scores + torch.log(confidence.clamp_min(1e-8))
        scores = scores.masked_fill(~collection.present, -torch.inf)
        weights = torch.softmax(scores, dim=1)
        payload = torch.einsum("be,bew->bw", weights, collection.payload)
        if self.residual is not None:
            centered = collection.payload - payload.unsqueeze(1)
            variance = torch.einsum("be,bew->bw", weights, centered.square())
            diversity = variance.mean(dim=-1, keepdim=True)
            # This exact-zero gate makes N=1 and identical duplicates permanent
            # compatibility invariants even after the generic set residual learns.
            diversity_gate = diversity / (diversity + 1e-4)
            payload = payload + diversity_gate * self.residual(
                torch.cat([payload, variance], dim=-1)
            )
        timestamp = (
            torch.einsum("be,be->b", weights, collection.timestamp)
            if collection.timestamp is not None
            else None
        )
        combined_confidence = torch.einsum("be,be->b", weights, confidence)
        return AmodalEvent(
            payload=payload,
            timestamp=timestamp,
            confidence=combined_confidence,
        ).validate(width=self.event_width)


class ExtractedAmodalRuntime(nn.Module):
    """External encoder, one controller core, and external decoder.

    The three children own disjoint parameters and serialize independently.
    This migration runtime accepts one event per step. Variable-N input is a
    later rung; variable-M output is provided by the external output bus.
    """

    def __init__(
        self,
        encoder: VisionEventEncoder,
        controller: UnifiedCognitiveController,
        decoder: ActionIntentDecoder,
    ) -> None:
        super().__init__()
        if controller.vision is not None or controller.actuator is not None:
            raise ValueError("extracted controller must not own input/output adapters")
        if encoder.network[-1].normalized_shape != (controller.width,):
            raise ValueError("encoder and controller event widths disagree")
        if decoder.intention_width != controller.intention_width:
            raise ValueError("controller and decoder intention widths disagree")
        self.encoder = encoder
        self.controller = controller
        self.decoder = decoder

    @classmethod
    def from_legacy(
        cls, model: UnifiedCognitiveController, *, copy_model: bool = True
    ) -> ExtractedAmodalRuntime:
        """Extract adapters from a legacy model without changing its weights."""
        controller = copy.deepcopy(model) if copy_model else model
        encoder = controller.vision
        actuator = controller.actuator
        if encoder is None or actuator is None:
            raise ValueError("model is already missing its legacy adapters")
        controller.vision = None
        controller.actuator = None
        decoder = ActionIntentDecoder(
            actuator, intention_width=controller.intention_width
        )
        return cls(encoder, controller, decoder)

    def encode(self, frame: torch.Tensor) -> AmodalEvent:
        return AmodalEvent(payload=self.encoder(frame)).validate(
            width=self.controller.width
        )

    @property
    def compatibility_suffix_active(self) -> bool:
        """Whether the controller can emit a nonzero migration suffix."""
        return bool(
            self.controller.action_adapter is not None
            and not self.controller.action_adapter_emits_intention
            and not self.controller.action_adapter_into_intention
        )

    def decode(self, intention: IntentEvent) -> torch.Tensor:
        return self.decoder(intention)

    def step_intention_event(
        self,
        event: AmodalEvent,
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[ControllerCoreOutput, ControllerState]:
        """Run the controller without selecting or formatting an output."""
        return self.controller.step_event(
            event,
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
            intention_basis=self.decoder.projection.weight,
        )

    def step_intention(
        self,
        frame: torch.Tensor,
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[ControllerCoreOutput, ControllerState]:
        return self.step_intention_event(
            self.encode(frame),
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
        )

    def step_event(
        self,
        event: AmodalEvent,
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[ControllerOutput, ControllerState]:
        core_output, next_state = self.step_intention_event(
            event,
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
        )
        return self._decode_core_output(core_output), next_state

    def step(
        self,
        frame: torch.Tensor,
        state: ControllerState,
        previous_action: torch.Tensor,
        previous_reward: torch.Tensor,
        has_feedback: torch.Tensor,
        retrieved_memory: torch.Tensor | None = None,
        *,
        disable_workspace: bool = False,
    ) -> tuple[ControllerOutput, ControllerState]:
        return self.step_event(
            self.encode(frame),
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
        )

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> ControllerState:
        return self.controller.initial_state(batch_size, device=device, dtype=dtype)

    def _decode_core_output(self, output: ControllerCoreOutput) -> ControllerOutput:
        return ControllerOutput(
            logits=self.decode(output.intent_event),
            intention=output.legacy_intention,
            memory_key=output.memory_key,
            memory_value=output.memory_value,
            memory_write_strength=output.memory_write_strength,
            workspace_read=output.workspace_read,
            skill_adapter_openings=output.skill_adapter_openings,
            skill_adapter_residual_norms=output.skill_adapter_residual_norms,
        )

    def component_state_dicts(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return independently loadable encoder/core/decoder weights."""
        return {
            "encoder_state_dict": self.encoder.state_dict(),
            "controller_state_dict": self.controller.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
        }

    def legacy_state_dict(self) -> dict[str, torch.Tensor]:
        """Reassemble the exact key layout expected by legacy checkpoints."""
        state = dict(self.controller.state_dict())
        state.update(
            {
                f"vision.{name}": value
                for name, value in self.encoder.state_dict().items()
            }
        )
        state.update(
            {
                f"actuator.{name.removeprefix('projection.')}": value
                for name, value in self.decoder.state_dict().items()
            }
        )
        return state

    def extracted_payload(
        self,
        *,
        model_configuration: Mapping[str, Any],
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "format": EXTRACTED_CHECKPOINT_FORMAT,
            "event_schema": NEURAL_IR_SCHEMA,
            "intention_schema": INTENTION_SCHEMA,
            "model_configuration": dict(model_configuration),
            "provenance": dict(provenance or {}),
            **self.component_state_dicts(),
        }


def runtime_from_legacy_payload(
    payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
) -> ExtractedAmodalRuntime:
    """Load an ordinary historical checkpoint through the extracted runtime."""
    configuration = payload.get("model_configuration")
    state = payload.get("state_dict")
    if not isinstance(configuration, Mapping) or not isinstance(state, Mapping):
        raise TypeError("legacy payload lacks model configuration or state dict")
    model = UnifiedCognitiveController(**dict(configuration)).to(device)
    model.load_state_dict(state)
    return ExtractedAmodalRuntime.from_legacy(model, copy_model=False)


def runtime_from_extracted_payload(
    payload: Mapping[str, Any], *, device: torch.device | str = "cpu"
) -> ExtractedAmodalRuntime:
    """Load independently serialized components into an extracted runtime."""
    if payload.get("format") != EXTRACTED_CHECKPOINT_FORMAT:
        raise ValueError("unsupported extracted checkpoint format")
    if payload.get("event_schema") != NEURAL_IR_SCHEMA:
        raise ValueError("unsupported extracted event schema")
    if payload.get("intention_schema") != INTENTION_SCHEMA:
        raise ValueError("unsupported extracted intention schema")
    configuration = payload.get("model_configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("extracted payload lacks model configuration")
    blank = UnifiedCognitiveController(**dict(configuration)).to(device)
    runtime = ExtractedAmodalRuntime.from_legacy(blank, copy_model=False)
    runtime.encoder.load_state_dict(payload["encoder_state_dict"])
    runtime.controller.load_state_dict(payload["controller_state_dict"])
    runtime.decoder.load_state_dict(payload["decoder_state_dict"])
    return runtime


def canonicalize_action_adapter_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fold a legacy two-logit residual into the base intention coordinates.

    The conversion is algebraic and uses no examples, labels, rewards, or
    optimizer updates. For actuator matrix ``W`` it computes the minimum-norm
    right inverse ``P = (W W^T)^-1 W`` and folds ``residual @ P`` into the last
    action-adapter layer. The external decoder then recovers the same residual
    through ``(residual @ P) @ W^T``.
    """
    configuration = payload.get("model_configuration")
    source_state = payload.get("state_dict")
    if not isinstance(configuration, Mapping) or not isinstance(source_state, Mapping):
        raise TypeError("checkpoint lacks model configuration or state dict")
    source_configuration = dict(configuration)
    if not int(source_configuration.get("action_adapter_width", 0)):
        raise ValueError("checkpoint has no action adapter to canonicalize")
    if source_configuration.get("action_adapter_emits_intention", False):
        raise ValueError("action adapter already emits an intention residual")
    if source_configuration.get("action_adapter_into_intention", False):
        raise ValueError(
            "runtime-projected action adapter must be normalized separately"
        )

    actuator_weight = source_state["actuator.weight"]
    old_weight = source_state["action_adapter.2.weight"]
    old_bias = source_state["action_adapter.2.bias"]
    if (
        actuator_weight.ndim != 2
        or old_weight.shape[0] != ACTIONS
        or old_bias.shape != (ACTIONS,)
    ):
        raise ValueError("checkpoint has an unsupported legacy adapter shape")
    calculation_dtype = (
        torch.float64
        if actuator_weight.dtype in (torch.float16, torch.float32)
        else actuator_weight.dtype
    )
    decoder = actuator_weight.to(calculation_dtype)
    gram = decoder @ decoder.T
    if int(torch.linalg.matrix_rank(gram)) != ACTIONS:
        raise ValueError("actuator has no full-row-rank intention right inverse")
    right_inverse = torch.linalg.solve(gram, decoder)

    state = {name: value.detach().clone() for name, value in source_state.items()}
    state["action_adapter.2.weight"] = (
        right_inverse.T @ old_weight.to(calculation_dtype)
    ).to(old_weight.dtype)
    state["action_adapter.2.bias"] = (
        old_bias.to(calculation_dtype) @ right_inverse
    ).to(old_bias.dtype)
    migrated_configuration = dict(source_configuration)
    migrated_configuration["action_adapter_emits_intention"] = True
    migrated_configuration["action_adapter_into_intention"] = False

    # Strict construction is part of conversion: a malformed state never
    # becomes a candidate checkpoint.
    candidate = UnifiedCognitiveController(**migrated_configuration)
    candidate.load_state_dict(state)
    return {
        "schema": payload.get("schema", "unified-cognitive-controller-v1"),
        "model_configuration": migrated_configuration,
        "state_dict": candidate.state_dict(),
        "migration": {
            "format": CANONICAL_INTENTION_MIGRATION,
            "source_action_adapter_output_width": ACTIONS,
            "target_action_adapter_output_width": candidate.intention_width,
            "uses_examples": False,
            "uses_optimizer_updates": False,
        },
    }


def canonicalize_action_adapter_checkpoint(
    source: Path, destination: Path, *, device: torch.device | str = "cpu"
) -> None:
    payload = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("legacy checkpoint must contain a mapping")
    canonical = canonicalize_action_adapter_payload(payload)
    canonical["migration"]["source_checkpoint"] = str(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(canonical, destination)


def convert_legacy_checkpoint(
    source: Path, destination: Path, *, device: torch.device | str = "cpu"
) -> None:
    """Convert one checkpoint without training or changing any tensor value."""
    payload = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(payload, Mapping):
        raise TypeError("legacy checkpoint must contain a mapping")
    runtime = runtime_from_legacy_payload(payload, device=device)
    extracted = runtime.extracted_payload(
        model_configuration=payload["model_configuration"],
        provenance={"legacy_checkpoint": str(source)},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(extracted, destination)
