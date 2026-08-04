"""Production amodal neural-computer runtime.

Historical trainers and checkpoint readers live under ``experiments``.  New
agent code imports from this package instead.
"""

from .checkpoint import load_runtime_components, save_runtime
from .controller import (
    EXECUTION_STATES,
    AmodalCognitiveController,
    ControllerOutput,
    ControllerState,
)
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
    MEMORY_BACKEND_FORMAT,
    MEMORY_READ_MATCH_THRESHOLD,
    MEMORY_SNAPSHOT_FORMAT,
    ContentAddressedMemory,
    MemoryBackend,
    MemoryQuery,
    MemoryRead,
    MemoryWriteReceipt,
    PersistentContentAddressedMemory,
)
from .policies import EventReliabilityPolicy, EventWaitPolicy
from .runtime import (
    AmodalControllerRuntime,
    AmodalEventTimeline,
    AmodalEventWindow,
    AmodalEventWindowBuffer,
    AmodalEventWindowStatus,
    AmodalExecutionResult,
    AmodalInputBus,
    AmodalOutputBus,
    AmodalRuntimeOutput,
    OpaqueProtocolDecoder,
)

__all__ = [
    "EVENT_SCHEMA",
    "EVENT_WINDOW_SCHEMA",
    "EXECUTION_STATES",
    "INTENTION_SCHEMA",
    "MEMORY_BACKEND_FORMAT",
    "MEMORY_READ_MATCH_THRESHOLD",
    "MEMORY_SCHEMA",
    "MEMORY_SNAPSHOT_FORMAT",
    "AmodalCognitiveController",
    "AmodalControllerRuntime",
    "AmodalEvent",
    "AmodalEventCollection",
    "AmodalEventTimeline",
    "AmodalEventWindow",
    "AmodalEventWindowBuffer",
    "AmodalEventWindowStatus",
    "AmodalExecutionResult",
    "AmodalInputBus",
    "AmodalOutputBus",
    "AmodalRuntimeOutput",
    "ContentAddressedMemory",
    "ControllerFeedback",
    "ControllerOutput",
    "ControllerState",
    "EventReliabilityPolicy",
    "EventTokenWindow",
    "EventWaitPolicy",
    "IntentEvent",
    "MemoryBackend",
    "MemoryQuery",
    "MemoryRead",
    "MemoryWriteReceipt",
    "OpaqueProtocolDecoder",
    "PersistentContentAddressedMemory",
    "load_runtime_components",
    "save_runtime",
]
