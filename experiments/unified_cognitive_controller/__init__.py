"""Unified sensory controller with learned working and long-term memory."""

from .environment import CognitiveLifetimeBatch, generate_lifetimes
from .model import ControllerState, UnifiedCognitiveController

__all__ = [
    "CognitiveLifetimeBatch",
    "ControllerState",
    "UnifiedCognitiveController",
    "generate_lifetimes",
]
