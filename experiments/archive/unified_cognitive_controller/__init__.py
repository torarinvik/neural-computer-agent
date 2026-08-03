"""Historical experiment namespace.

The production agent is :mod:`neural_computer`.  This namespace is retained so
old experiment modules remain runnable while their checkpoint and trainer
dependencies are migrated out of the experiment tree.
"""

from neural_computer import (
    AmodalCognitiveController,
    AmodalControllerRuntime,
    AmodalEvent,
    AmodalEventCollection,
    AmodalEventTimeline,
    AmodalEventWindow,
    AmodalEventWindowBuffer,
    AmodalEventWindowStatus,
    AmodalInputBus,
    AmodalOutputBus,
    AmodalRuntimeOutput,
    ControllerFeedback,
    ControllerOutput,
    ControllerState,
    IntentEvent,
    OpaqueProtocolDecoder,
)

__all__ = [
    "AmodalCognitiveController",
    "AmodalControllerRuntime",
    "AmodalEvent",
    "AmodalEventCollection",
    "AmodalEventTimeline",
    "AmodalEventWindow",
    "AmodalEventWindowBuffer",
    "AmodalEventWindowStatus",
    "AmodalInputBus",
    "AmodalOutputBus",
    "AmodalRuntimeOutput",
    "ControllerFeedback",
    "ControllerOutput",
    "ControllerState",
    "IntentEvent",
    "OpaqueProtocolDecoder",
]
