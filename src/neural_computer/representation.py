"""Shared version identifiers for replaceable learned representations."""

from __future__ import annotations

REPRESENTATION_SPACE_SCHEMA = "neural-computer.external-representation-space.v1"
DEFAULT_EVENT_SPACE_ID = "opaque-event-v1"
DEFAULT_STATE_SPACE_ID = "opaque-state-v1"
DEFAULT_INTENTION_SPACE_ID = "opaque-intention-v1"
DEFAULT_CONTROLLER_STATE_SPACE_ID = "controller-state-v1"
DEFAULT_MEMORY_KEY_SPACE_ID = "opaque-memory-key-v1"
DEFAULT_MEMORY_VALUE_SPACE_ID = "opaque-memory-value-v1"
DEFAULT_ROUTE_QUERY_SPACE_ID = "opaque-route-query-v1"


def validate_representation_space_id(value: str, *, name: str) -> str:
    """Validate an opaque, caller-versioned learned-space identifier."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty representation-space ID")
    return value.strip()


__all__ = [
    "DEFAULT_CONTROLLER_STATE_SPACE_ID",
    "DEFAULT_EVENT_SPACE_ID",
    "DEFAULT_INTENTION_SPACE_ID",
    "DEFAULT_MEMORY_KEY_SPACE_ID",
    "DEFAULT_MEMORY_VALUE_SPACE_ID",
    "DEFAULT_ROUTE_QUERY_SPACE_ID",
    "DEFAULT_STATE_SPACE_ID",
    "REPRESENTATION_SPACE_SCHEMA",
    "validate_representation_space_id",
]
