"""Production amodal neural-computer runtime.

Historical trainers and checkpoint readers live under ``experiments``.  New
agent code imports from this package instead.
"""

from .checkpoint import load_runtime_components, save_runtime
from .controller import AmodalCognitiveController, ControllerOutput, ControllerState
from .interface import (
    EVENT_SCHEMA,
    EVENT_WINDOW_SCHEMA,
    INTENTION_SCHEMA,
    MEMORY_SCHEMA,
    AmodalEvent,
    AmodalEventCollection,
    ControllerFeedback,
    EventTokenWindow,
    IntentEvent,
)
from .memory import (
    ContentAddressedMemory,
    MemoryQuery,
    MemoryRead,
    MemoryWriteReceipt,
)
from .policies import EventReliabilityPolicy, EventWaitPolicy
from .runtime import (
    AmodalControllerRuntime,
    AmodalEventTimeline,
    AmodalEventWindow,
    AmodalEventWindowBuffer,
    AmodalEventWindowStatus,
    AmodalInputBus,
    AmodalOutputBus,
    AmodalRuntimeOutput,
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
    "ContentAddressedMemory",
    "EVENT_SCHEMA",
    "EVENT_WINDOW_SCHEMA",
    "INTENTION_SCHEMA",
    "MEMORY_SCHEMA",
    "EventReliabilityPolicy",
    "EventTokenWindow",
    "EventWaitPolicy",
    "IntentEvent",
    "MemoryQuery",
    "MemoryRead",
    "MemoryWriteReceipt",
    "OpaqueProtocolDecoder",
    "load_runtime_components",
    "save_runtime",
]
