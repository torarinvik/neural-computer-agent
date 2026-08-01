"""Unified sensory controller with learned working and long-term memory."""

from .amodal_interface import AmodalEvent, IntentEvent
from .amodal_runtime import (
    ActionIntentDecoder,
    AmodalOutputBus,
    ExtractedAmodalRuntime,
    OpaqueProtocolDecoder,
)
from .environment import CognitiveLifetimeBatch, generate_lifetimes
from .model import ControllerState, UnifiedCognitiveController

__all__ = [
    "ActionIntentDecoder",
    "AmodalEvent",
    "AmodalOutputBus",
    "CognitiveLifetimeBatch",
    "ControllerState",
    "ExtractedAmodalRuntime",
    "IntentEvent",
    "OpaqueProtocolDecoder",
    "UnifiedCognitiveController",
    "generate_lifetimes",
]
