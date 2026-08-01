"""Behavior-preserving extracted runtime for the amodal migration rung."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .amodal_interface import (
    INTENTION_SCHEMA,
    NEURAL_IR_SCHEMA,
    AmodalEvent,
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


class ExtractedAmodalRuntime(nn.Module):
    """External encoder, one controller core, and external decoder.

    The three children own disjoint parameters and serialize independently.
    This first migration runtime accepts one event per step; variable N/M buses
    are later rungs and are deliberately not claimed here.
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

    def decode(self, intention: IntentEvent) -> torch.Tensor:
        return self.decoder(intention)

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
        core_output, next_state = self.controller.step_event(
            event,
            state,
            previous_action,
            previous_reward,
            has_feedback,
            retrieved_memory,
            disable_workspace=disable_workspace,
            intention_basis=self.decoder.projection.weight,
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
